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


class AutoTrader:
    def __init__(self, universe=None, target_names=TARGET_NAMES,
                 stop_pct=DEFAULT_STOP_PCT, target_pct=DEFAULT_TARGET_PCT):
        self.universe = universe or AUTOTRADER_UNIVERSE
        self.target_names = target_names
        self.stop_pct = stop_pct
        self.target_pct = target_pct
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
        if held:
            prices = current_prices(list(held))
            for sym in list(held):
                v = verdict.get(sym)
                ns = news(sym)
                reason = None
                if ns.get("bearish"):
                    reason = "NEWS:" + ",".join(ns.get("bear_tags") or ["negative"])
                elif v and v["verdict"].startswith(("AVOID", "HOLD")):
                    reason = "VERDICT_DOWNGRADE"
                if reason and prices.get(sym):
                    pnl = self.broker.sell(sym, prices[sym], reason)
                    sells.append({"symbol": sym, "reason": reason,
                                  "price": prices[sym], "pnl": round(pnl, 2)})

        # 2) BUY — value names not held, but VETO any with a bearish news catalyst
        #    (don't buy into bad news even if cheap).
        orders, vetoed = [], []
        for v in ranked:
            if len(self.broker.positions) + len(orders) >= self.target_names:
                break
            sym = v["ticker"].replace(".NS", "")
            if sym in self.broker.positions:
                continue
            if not v["verdict"].startswith(("WORTH BUYING", "FAIR")):
                continue
            ns = news(sym)
            if ns.get("bearish"):
                vetoed.append({"symbol": sym, "net": ns.get("net"),
                               "reason": "NEWS_VETO:" + ",".join(ns.get("bear_tags") or ["negative"])})
                continue
            rupees = self.broker.marked_nav() / self.target_names
            orders.append({"symbol": sym, "rupees": rupees,
                           "stop_pct": self.stop_pct, "target_pct": self.target_pct,
                           "sector": v.get("sector", "Unknown")})

        buys = self.engine.run_plan(orders, live=live) if orders else []
        news_summary = {s: {"net": d.get("net"), "tags": d.get("tags"),
                            "bearish": d.get("bearish"), "bullish": d.get("bullish")}
                        for s, d in _news.items() if d.get("ok")}
        return {"sells": sells, "buys": buys, "vetoed": vetoed, "news": news_summary,
                "summary": self.broker.summary(current_prices(list(self.broker.positions))
                                               if self.broker.positions else None)}
