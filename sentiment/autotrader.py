"""
AutoTrader — the autonomous DAILY decision (paper-first).

Deterministic and value-driven (so it runs unattended without depending on an LLM's
mood). Each decision cycle:

  1. Value-rank the configured universe (valuation.py — momentum ignored).
  2. SELL DISCIPLINE: exit any held name whose verdict has decayed to AVOID or
     HOLD/WAIT (it's no longer worth owning at this price).
  3. BUY: take the best WORTH-BUYING / FAIR names not already held, equal-weight,
     each routed through the hard RISK GATE (caps, R:R, cash, stop required).

Stop-loss + target are armed on every buy; intraday exits are handled by the
scheduler's 30-min monitor (paper) or by the broker's GTT (live). This module only
decides — the risk gate and broker enforce and execute.
"""
from __future__ import annotations

import os

import config
from .valuation import rank_by_value
from .risk_gate import Limits
from .trade_engine import TradeEngine, current_prices, DEFAULT_STOP_PCT, DEFAULT_TARGET_PCT
from .news_adapter import news_signal
from .market_filters import market_regime_ok, trend_ok, vol_weight, liquidity_cap
from .critic import review as critic_review

# turnover control
MIN_HOLD_DAYS = int(os.getenv("TRADITE_MIN_HOLD_DAYS", "10"))   # don't discretionary-sell a fresh buy
TOPUP_BELOW = float(os.getenv("TRADITE_TOPUP_BELOW", "0.90"))   # only scale-in while < 90% of target


def _days_held(pos: dict) -> int:
    from datetime import date
    try:
        y, m, d = (int(x) for x in str(pos.get("entry_date", "")).split("-"))
        return (date.today() - date(y, m, d)).days
    except Exception:
        return 999

# Default autonomous universe — liquid large-caps across sectors (override via env/config).
AUTOTRADER_UNIVERSE = getattr(config, "AUTOTRADER_UNIVERSE", [
    "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS",
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS",
    "ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS",
    "MARUTI.NS", "M&M.NS",
    "SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS",
    "RELIANCE.NS", "LT.NS", "NTPC.NS", "COALINDIA.NS",
])

START_CAPITAL = float(os.getenv("TRADITE_START_CAPITAL", "1000000"))
TARGET_NAMES = int(os.getenv("TRADITE_TARGET_NAMES", "8"))
# Adaptive exits — a TRAILING stop ratchets up with the high-water mark, locking in gains and
# letting winners run. Backtest (2y): trailing-8% returned +12.8% vs +5.2% fixed-8/20, lower
# drawdown, win-rate 29%→49%. The target is a far ceiling; the trailing stop is the real exit.
TRAIL_PCT = float(os.getenv("TRADITE_TRAIL_PCT", "8"))
TARGET_CEILING_PCT = float(os.getenv("TRADITE_TARGET_PCT", "50"))
TP1_PCT = float(os.getenv("TRADITE_PARTIAL_TP_PCT", "15"))   # book half at +15%, trail the rest
# Staggered entry — scale INTO a name over N daily decisions (1/N each) instead of all-in on
# day one; also tops up held names toward their target weight. Backtest (active): cut drawdown
# −14%→−11% and lifted return. Set TRADITE_STAGGER_TRANCHES=1 to buy full size at once.
STAGGER_TRANCHES = max(1, int(os.getenv("TRADITE_STAGGER_TRANCHES", "3")))


class AutoTrader:
    def __init__(self, universe=None, target_names=TARGET_NAMES,
                 stop_pct=DEFAULT_STOP_PCT, target_pct=TARGET_CEILING_PCT,
                 trail_pct=TRAIL_PCT, tp1_pct=TP1_PCT):
        self.universe = universe or AUTOTRADER_UNIVERSE
        self.target_names = target_names
        self.stop_pct = stop_pct
        self.target_pct = target_pct
        self.trail_pct = trail_pct
        self.tp1_pct = tp1_pct
        self.engine = TradeEngine(limits=Limits())
        self.broker = self.engine.broker
        if self.broker.capital == 0:
            self.broker.init(START_CAPITAL)

    def decide(self, live: bool = False) -> dict:
        ranked = rank_by_value(self.universe)            # best value first
        verdict = {v["ticker"].replace(".NS", ""): v for v in ranked}
        held = set(self.broker.positions)

        _news: dict[str, dict] = {}
        def news(sym):                                   # fetch once per name per cycle
            if sym not in _news:
                _news[sym] = news_signal(sym + ".NS")
            return _news[sym]

        # 1) SELL — event-driven (bearish NEWS) OR verdict decayed to AVOID/HOLD-WAIT.
        #    News adaptation: a regulatory/fraud/earnings-miss catalyst exits the name now,
        #    regardless of valuation, before the stop would.
        sells = []
        prices = current_prices(list(held)) if held else {}
        if held:
            for sym in list(held):
                v = verdict.get(sym)
                ns = news(sym)
                reason = None
                if ns.get("bearish"):
                    reason = "NEWS:" + ",".join(ns.get("bear_tags") or ["negative"])
                elif (v and v["verdict"].startswith(("AVOID", "HOLD"))
                      and _days_held(self.broker.positions.get(sym, {})) >= MIN_HOLD_DAYS):
                    reason = "VERDICT_DOWNGRADE"     # discretionary sell only after min-hold
                if reason and prices.get(sym):
                    pnl = self.broker.sell(sym, prices[sym], reason)
                    sells.append({"symbol": sym, "reason": reason,
                                  "price": prices[sym], "pnl": round(pnl, 2)})

        # 2) BUY — value names not held. ACTIVE by default: vol-targeted sizing + partial
        #    profit-taking + trailing stop (these trade the MOST and returned the most in
        #    backtest). News still vetoes a bearish name. The defensive cash-gates (market
        #    regime, per-name trend) are OFF by default — they idle the book — but honored
        #    if you enable them via env.
        orders, vetoed = [], []
        risk_on, regime_note = market_regime_ok()
        base_slot = self.broker.marked_nav() / self.target_names
        new_names = 0                                     # new distinct names opened this cycle
        if risk_on:
            for v in ranked:
                sym = v["ticker"].replace(".NS", "")
                if not v["verdict"].startswith(("WORTH BUYING", "FAIR")):
                    continue
                if not trend_ok(v["ticker"]):              # no-op unless TRADITE_USE_TREND=true
                    if sym not in self.broker.positions:
                        vetoed.append({"symbol": sym, "reason": "TREND:below-200DMA"})
                    continue
                ns = news(sym)
                if ns.get("bearish"):
                    if sym not in self.broker.positions:
                        vetoed.append({"symbol": sym, "net": ns.get("net"),
                                       "reason": "NEWS_VETO:" + ",".join(ns.get("bear_tags") or ["negative"])})
                    continue
                held_here = sym in self.broker.positions
                if not held_here and (len(self.broker.positions) + new_names) >= self.target_names:
                    continue                               # at the name cap; don't open a new one
                # STAGGERED SCALE-IN: fill toward a vol-targeted target over STAGGER_TRANCHES days,
                # and top up held names that are below their target weight.
                # vol-targeted size, capped by liquidity (≤% of avg daily traded value)
                target_rupees = min(base_slot * vol_weight(v["ticker"]), liquidity_cap(v["ticker"]))
                cur_val = (self.broker.positions[sym]["qty"] *
                           prices.get(sym, self.broker.positions[sym]["avg"])) if held_here else 0.0
                if cur_val >= target_rupees * TOPUP_BELOW:
                    continue                               # no-trade band: close enough to target
                add = min(target_rupees / STAGGER_TRANCHES, target_rupees - cur_val)
                if not held_here:
                    new_names += 1
                orders.append({"symbol": sym, "rupees": add,
                               "stop_pct": self.stop_pct, "target_pct": self.target_pct,
                               "trail_pct": self.trail_pct, "tp1_pct": self.tp1_pct,
                               "sector": v.get("sector", "Unknown")})

        # SELF-REFLECTION CRITIC — holistic audit of the whole plan before anything executes.
        crit = critic_review(orders, self.broker, {s["symbol"] for s in sells},
                             _news, self.engine.limits.min_cash_pct)
        orders = crit["approved"]
        for x in crit["vetoes"]:
            vetoed.append({"symbol": x["symbol"], "reason": "CRITIC:" + "; ".join(x["reasons"])})

        buys = self.engine.run_plan(orders, live=live) if orders else []
        news_summary = {s: {"net": d.get("net"), "tags": d.get("tags"),
                            "bearish": d.get("bearish"), "bullish": d.get("bullish")}
                        for s, d in _news.items() if d.get("ok")}
        return {"sells": sells, "buys": buys, "vetoed": vetoed, "regime": regime_note,
                "audit": crit["audit"], "news": news_summary,
                "summary": self.broker.summary(current_prices(list(self.broker.positions))
                                               if self.broker.positions else None)}
