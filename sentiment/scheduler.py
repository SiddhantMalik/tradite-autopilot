"""
Scheduler / autopilot — runs the paper system unattended.

Cadence (matches the swing/positional horizon + free delayed data):
  • MONITOR every 30 min during NSE hours (09:15–15:30 IST, Mon–Fri) → auto-exit on
    stop/target.
  • DECIDE once a day, post-close (default 15:45 IST) → value-rank, sell downgrades,
    buy the best worth-buying names (risk-gated).
  • REPORT at EOD → NAV/P&L snapshot to reports/ + append-only ledger.

Two ways to run (both paper by default):
  • Container worker (long-running):   python -m sentiment.scheduler run
  • Cron / scheduled jobs (one-shot):  python -m sentiment.scheduler monitor
                                       python -m sentiment.scheduler decide
                                       python -m sentiment.scheduler report

Live trading stays gated in trade_engine (TRADITE_LIVE_CONFIRM + --live); the autopilot
runs PAPER unless TRADITE_AUTOPILOT_LIVE=I_UNDERSTAND is set (not recommended yet).
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta

import config
from .trade_engine import TradeEngine, current_prices
from .autotrader import AutoTrader
from .portfolio_manager import _in_market_hours

MONITOR_EVERY_MIN = int(os.getenv("TRADITE_MONITOR_MIN", "30"))
DECIDE_AT_IST = os.getenv("TRADITE_DECIDE_AT", "15:45")     # HH:MM IST
REPORTS_DIR = config.ROOT / "reports"
_IST = timedelta(hours=5, minutes=30)
_LIVE = os.getenv("TRADITE_AUTOPILOT_LIVE") == "I_UNDERSTAND"
# DAILY decide by default (more active — Sid's preference). Set TRADITE_DECIDE_DAILY=false
# (with TRADITE_DECIDE_WEEKDAY) to switch to a lower-turnover weekly cadence instead.
DECIDE_DAILY = os.getenv("TRADITE_DECIDE_DAILY", "true").lower() == "true"
DECIDE_WEEKDAY = int(os.getenv("TRADITE_DECIDE_WEEKDAY", "0"))   # 0=Mon .. 4=Fri
# INTRADAY decisions: re-decide (new buys + news/verdict sells) every N minutes during market
# hours, so it acts through the day — not just once at close. 0 = once daily at DECIDE_AT.
DECIDE_EVERY_MIN = int(os.getenv("TRADITE_DECIDE_EVERY_MIN", "60"))


def _ist_now() -> datetime:
    return datetime.now(timezone.utc) + _IST


def _log(msg: str):
    print(f"[{_ist_now():%Y-%m-%d %H:%M} IST] {msg}", flush=True)


# ── actions ──────────────────────────────────────────────────────────────────
def do_monitor() -> dict:
    out = TradeEngine().monitor()
    for e in out["exits"]:
        _log(f"EXIT {e['symbol']} {e['reason']} @₹{e['price']:,.1f} P&L ₹{e['pnl']:+,.0f}")
    if not out["exits"]:
        s = out["summary"]
        _log(f"monitor: NAV ₹{s['nav']:,.0f}  total P&L ₹{s['total_pnl']:+,.0f} "
             f"({s['total_return_pct']:+.2f}%)  no exits")
    return out


def do_decide() -> dict:
    _log(f"DECIDE ({'LIVE' if _LIVE else 'paper'}) — value + news …")
    res = AutoTrader().decide(live=_LIVE)
    for s in res.get("sells", []):
        _log(f"SELL {s['symbol']} [{s['reason']}] @₹{s['price']:,.1f} P&L ₹{s['pnl']:+,.0f}")
    for v in res.get("vetoed", []):
        _log(f"VETO (skip buy) {v['symbol']} [{v['reason']}]")
    for b in res.get("buys", []):
        tag = b.get("filled") or f"{b.get('decision','?').upper()}: {'; '.join(b.get('reasons', []))}"
        _log(f"BUY {b.get('symbol','?')}: {tag}")
    if res.get("audit"):
        _log("CRITIC audit: " + " | ".join(res["audit"]))
    return res


def do_report(tag: str = "EOD") -> dict:
    REPORTS_DIR.mkdir(exist_ok=True)
    eng = TradeEngine(); b = eng.broker
    prices = current_prices(list(b.positions)) if b.positions else {}
    s = b.summary(prices)
    day = _ist_now().strftime("%Y-%m-%d")
    md = [f"# Tradite paper report — {day} ({tag})", "",
          f"- NAV: ₹{s['nav']:,.0f}  (capital ₹{s['capital']:,.0f}, return {s['total_return_pct']:+.2f}%)",
          f"- Cash: ₹{s['cash']:,.0f}  |  Positions: {s['positions']}",
          f"- P&L: realized ₹{s['realized_pnl']:+,.0f}, unrealized ₹{s['unrealized_pnl']:+,.0f}, "
          f"total ₹{s['total_pnl']:+,.0f}",
          f"- Drawdown: {s['drawdown_pct']:.1f}%", "", "## Positions", ""]
    for sym, p in b.positions.items():
        ltp = prices.get(sym, p["avg"]); pnl = (ltp - p["avg"]) * p["qty"]
        md.append(f"- {sym}: {p['qty']}×  avg ₹{p['avg']:,.1f}  ltp ₹{ltp:,.1f}  "
                  f"stop ₹{p['stop']:,.1f}  target ₹{p['target']:,.1f}  P&L ₹{pnl:+,.0f}")
    (REPORTS_DIR / f"paper_{day}.md").write_text("\n".join(md))
    with open(REPORTS_DIR / "ledger.jsonl", "a") as f:
        f.write(json.dumps({"ts": _ist_now().isoformat(), "tag": tag, **s}) + "\n")
    _log(f"report written → reports/paper_{day}.md  (NAV ₹{s['nav']:,.0f}, "
         f"P&L ₹{s['total_pnl']:+,.0f})")
    return s


# ── the loop (container worker) ──────────────────────────────────────────────
def run_loop():
    _log(f"autopilot started — monitor every {MONITOR_EVERY_MIN}m in market hours, "
         f"decide at {DECIDE_AT_IST} IST, mode={'LIVE' if _LIVE else 'PAPER'}")
    last_monitor = 0.0
    last_decide = time.time()      # cold-start already decided; wait one interval before the next
    decided_on = None
    reported_on = None

    # Cold start: on a fresh deploy the book is empty — initialize and make the first
    # allocation immediately so the dashboard shows a real ₹1cr portfolio right away,
    # instead of ₹0 until the next scheduled decision.
    try:
        if TradeEngine().broker.capital == 0 or not TradeEngine().broker.positions:
            _log("cold start: funding book + first allocation …")
            do_decide()
            do_report("cold-start")
            decided_on = _ist_now().date()
    except Exception as e:  # noqa: BLE001
        _log(f"cold-start error: {e}")

    while True:
        now = _ist_now()
        try:
            if _in_market_hours():
                if (time.time() - last_monitor) >= MONITOR_EVERY_MIN * 60:
                    do_monitor(); last_monitor = time.time()
                # INTRADAY: buy / sell through the day, not just at close
                if DECIDE_EVERY_MIN > 0 and (time.time() - last_decide) >= DECIDE_EVERY_MIN * 60:
                    do_decide(); last_decide = time.time()
            # once-daily fallback (only when intraday is off) + the EOD report
            if now.weekday() < 5 and now.strftime("%H:%M") >= DECIDE_AT_IST:
                is_decide_day = DECIDE_DAILY or now.weekday() == DECIDE_WEEKDAY
                if DECIDE_EVERY_MIN == 0 and is_decide_day and decided_on != now.date():
                    do_decide(); decided_on = now.date()
                if reported_on != now.date():
                    do_report("EOD"); reported_on = now.date()
        except Exception as e:  # never let the loop die
            _log(f"cycle error: {e}")
        time.sleep(60)


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    cmd = argv[0] if argv else "run"
    if cmd == "run":
        run_loop()
    elif cmd == "monitor":
        do_monitor()
    elif cmd == "decide":
        do_decide()
    elif cmd == "report":
        do_report(argv[1] if len(argv) > 1 else "manual")
    elif cmd == "once":          # cron-friendly: decide (if weekday) + monitor + report
        if _ist_now().weekday() < 5:
            do_decide()
        do_monitor(); do_report("once")
    else:
        print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
