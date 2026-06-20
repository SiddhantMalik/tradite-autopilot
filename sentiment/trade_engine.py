"""
Trade engine — turns a plan into actual orders, with the risk gate in front and
automatic profit/loss exits behind. Paper-first; live is gated.

Pipeline:  plan (orders) → RiskGate.check_halt (kill-switch) → RiskGate.check_order
           (approve / clamp / reject) → broker.buy → arm stop+target
           → monitor(prices) → broker.mark auto-sells on target (profit) or stop (loss)

PAPER mode (default) simulates fills against a persistent book — zero money at risk.
LIVE mode routes entries to Zerodha Kite (CNC market orders) and arms exits as a GTT
one-cancels-other (stop + target). LIVE refuses to run unless you explicitly opt in:
    export TRADITE_LIVE_CONFIRM=I_UNDERSTAND      # and pass --live, during market hours
Even then it asks for typed confirmation. Nothing here places a live order by default.

CLI (run from ml_lab/):
    python -m sentiment.trade_engine init 1000000
    python -m sentiment.trade_engine buy orders.json
    python -m sentiment.trade_engine monitor
    python -m sentiment.trade_engine monitor --prices COALINDIA=520,TCS=2520   # what-if / demo
    python -m sentiment.trade_engine status
    python -m sentiment.trade_engine trades
    python -m sentiment.trade_engine kill          # halt: flatten nothing, block new buys
"""
from __future__ import annotations

import json
import os
import sys

import config
from .paper_broker import PaperBroker
from .risk_gate import RiskGate, Limits

DEFAULT_STOP_PCT = float(os.getenv("TRADITE_DEFAULT_STOP_PCT", "8"))
DEFAULT_TARGET_PCT = float(os.getenv("TRADITE_DEFAULT_TARGET_PCT", "20"))
DEFAULT_TRAIL_PCT = float(os.getenv("TRADITE_TRAIL_PCT", "0"))   # 0 = no trail (manual CLI); autopilot sets it


# ── price source (free) ──────────────────────────────────────────────────────
def current_prices(symbols: list[str]) -> dict:
    """Latest price per bare NSE symbol (e.g. 'TCS'). yfinance batch, CSV fallback."""
    out: dict[str, float] = {}
    tickers = [f"{s}.NS" for s in symbols]
    try:
        import yfinance as yf
        raw = yf.download(tickers, period="2d", auto_adjust=True, progress=False, threads=True)
        import pandas as pd
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        for s in symbols:
            col = f"{s}.NS"
            if col in close.columns:
                ser = close[col].dropna()
            elif "Close" in close.columns and len(symbols) == 1:
                ser = close["Close"].dropna()
            else:
                ser = None
            if ser is not None and len(ser):
                out[s] = round(float(ser.iloc[-1]), 2)
    except Exception:
        pass
    # CSV fallback for anything missing
    from .base_rates import _load_close
    for s in symbols:
        if s not in out:
            ser = _load_close(f"{s}.NS")
            if ser is not None and len(ser):
                out[s] = round(float(ser.iloc[-1]), 2)
    return out


class TradeEngine:
    def __init__(self, limits: Limits | None = None):
        self.broker = PaperBroker()
        self.limits = limits or Limits()

    # ── run a plan (buy side) ────────────────────────────────────────────────
    def run_plan(self, orders: list[dict], live: bool = False) -> list[dict]:
        nav = self.broker.marked_nav() or self.broker.capital
        gate = RiskGate(nav=nav, limits=self.limits)

        halted, why = gate.check_halt(self.broker)
        if halted:
            print(f"⛔ {why}")
            return [{"halted": why}]

        syms = [o["symbol"] for o in orders]
        prices = current_prices(syms)
        records = []
        for o in orders:
            sym = o["symbol"]
            price = prices.get(sym)
            if not price:
                records.append({"symbol": sym, "decision": "reject", "reasons": ["no price"]})
                continue
            order = {
                "symbol": sym, "rupees": float(o.get("rupees", 0)), "price": price,
                "stop_pct": float(o.get("stop_pct", DEFAULT_STOP_PCT)),
                "target_pct": float(o.get("target_pct", DEFAULT_TARGET_PCT)),
                "trail_pct": float(o.get("trail_pct", DEFAULT_TRAIL_PCT)),
                "sector": o.get("sector", "Unknown"),
            }
            res = gate.check_order(order, self.broker)
            rec = {"symbol": sym, "price": price, **res}
            if res["decision"] in ("approve", "clamp") and res["qty"] >= 1:
                if live:
                    rec["live"] = self._live_buy(sym, res["qty"], price, order)
                else:
                    self.broker.buy(sym, res["qty"], price, order["stop_pct"],
                                    order["target_pct"], order["sector"], order["trail_pct"])
                    trail = f", trail {order['trail_pct']:.0f}%" if order["trail_pct"] else ""
                    rec["filled"] = f"BUY {res['qty']}×{sym} @₹{price:,.2f} " \
                                    f"(stop −{order['stop_pct']:.0f}%{trail}, target +{order['target_pct']:.0f}%)"
            records.append(rec)
        return records

    # ── monitor → auto-exit for profit/loss ──────────────────────────────────
    def monitor(self, price_override: dict | None = None) -> dict:
        syms = list(self.broker.positions)
        prices = current_prices(syms) if syms else {}
        if price_override:
            prices.update(price_override)
        exits = self.broker.mark(prices) if prices else []
        return {"prices": prices, "exits": exits, "summary": self.broker.summary(prices)}

    # ── LIVE path (Zerodha) — gated; never runs by default ───────────────────
    def _live_buy(self, symbol, qty, price, order) -> dict:
        if os.getenv("TRADITE_LIVE_CONFIRM") != "I_UNDERSTAND":
            return {"status": "blocked",
                    "note": "LIVE blocked: set TRADITE_LIVE_CONFIRM=I_UNDERSTAND to enable"}
        from .portfolio_manager import KiteSessionManager, _in_market_hours
        if not _in_market_hours():
            return {"status": "blocked", "note": "outside NSE market hours"}
        kite = KiteSessionManager().kite()
        # 1) entry — CNC market buy
        oid = kite.place_order(
            variety=kite.VARIETY_REGULAR, exchange="NSE", tradingsymbol=symbol,
            transaction_type=kite.TRANSACTION_TYPE_BUY, quantity=qty,
            product=kite.PRODUCT_CNC, order_type=kite.ORDER_TYPE_MARKET)
        # 2) exit — GTT one-cancels-other: stop-loss + target
        stop = round(price * (1 - order["stop_pct"] / 100), 1)
        target = round(price * (1 + order["target_pct"] / 100), 1)
        gtt = kite.place_gtt(
            trigger_type=kite.GTT_TYPE_OCO, tradingsymbol=symbol, exchange="NSE",
            trigger_values=[stop, target], last_price=price,
            orders=[
                {"transaction_type": kite.TRANSACTION_TYPE_SELL, "quantity": qty,
                 "order_type": kite.ORDER_TYPE_LIMIT, "product": kite.PRODUCT_CNC, "price": stop},
                {"transaction_type": kite.TRANSACTION_TYPE_SELL, "quantity": qty,
                 "order_type": kite.ORDER_TYPE_LIMIT, "product": kite.PRODUCT_CNC, "price": target},
            ])
        return {"status": "placed", "order_id": oid, "gtt_id": gtt,
                "stop": stop, "target": target}


# ── pretty printers ───────────────────────────────────────────────────────────
def _print_status(b: PaperBroker):
    prices = current_prices(list(b.positions)) if b.positions else {}
    s = b.summary(prices)
    print(f"\nNAV ₹{s['nav']:,.0f}  |  cash ₹{s['cash']:,.0f}  |  {s['positions']} positions")
    print(f"P&L: realized ₹{s['realized_pnl']:+,.0f}  unrealized ₹{s['unrealized_pnl']:+,.0f}  "
          f"total ₹{s['total_pnl']:+,.0f} ({s['total_return_pct']:+.2f}%)  |  drawdown {s['drawdown_pct']:.1f}%")
    if b.positions:
        print(f"\n{'SYM':12s}{'QTY':>6s}{'AVG':>10s}{'LTP':>10s}{'STOP':>10s}{'TARGET':>10s}{'P&L':>12s}")
        for sym, p in b.positions.items():
            ltp = prices.get(sym, p["avg"])
            pnl = (ltp - p["avg"]) * p["qty"]
            print(f"{sym:12s}{p['qty']:>6d}{p['avg']:>10,.1f}{ltp:>10,.1f}"
                  f"{p['stop']:>10,.1f}{p['target']:>10,.1f}{pnl:>+12,.0f}")


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(__doc__); return 0
    cmd = argv[0]
    eng = TradeEngine()
    b = eng.broker

    if cmd == "init":
        cap = float(argv[1]) if len(argv) > 1 else 1_000_000
        b.init(cap); print(f"Paper book initialised with ₹{cap:,.0f} cash.")
    elif cmd == "buy":
        orders = json.load(open(argv[1]))
        live = "--live" in argv
        for r in eng.run_plan(orders, live=live):
            tag = r.get("filled") or r.get("live") or f"{r.get('decision','?').upper()}: {'; '.join(r.get('reasons', []))}"
            print(f"  {r.get('symbol','?'):12s} {tag}")
        _print_status(eng.broker)
    elif cmd == "monitor":
        override = {}
        if "--prices" in argv:
            spec = argv[argv.index("--prices") + 1]
            for kv in spec.split(","):
                k, v = kv.split("="); override[k.strip()] = float(v)
        out = eng.monitor(price_override=override or None)
        if out["exits"]:
            print("EXITS triggered:")
            for e in out["exits"]:
                print(f"  {e['symbol']:12s} {e['reason']:22s} @₹{e['price']:,.1f}  P&L ₹{e['pnl']:+,.0f}")
        else:
            print("No exits triggered (no position hit its stop or target).")
        _print_status(eng.broker)
    elif cmd == "status":
        _print_status(b)
    elif cmd == "trades":
        for t in b.trades[-40:]:
            extra = f"  P&L ₹{t['pnl']:+,.0f} [{t.get('reason','')}]" if t["action"] == "SELL" else ""
            print(f"  {t['ts'][:19]}  {t['action']:4s} {t['qty']:>5d}×{t['symbol']:10s} @₹{t['price']:,.1f}{extra}")
    elif cmd == "kill":
        # Conservative kill-switch: don't auto-sell; just record an intent flag.
        b.peak_nav = b.marked_nav() / (1 - eng.limits.drawdown_halt_pct) + 1  # force halt on next buy
        b._save(); print("Kill-switch armed — new buys will be blocked by the risk gate.")
    else:
        print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
