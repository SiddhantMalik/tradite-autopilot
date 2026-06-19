"""
Tradite CLI — view, configure, and act on your portfolio.

Usage (run from ml_lab/):
    python -m sentiment.cli status              # holdings + live P&L + saved stop/target
    python -m sentiment.cli set ICICIBANK --stop 6 --target 18
    python -m sentiment.cli list                # show all saved stop/target overrides
    python -m sentiment.cli run --budget 100000 --stop 7 --target 15
    python -m sentiment.cli gtt ICICIBANK       # place OCO GTT on Kite
    python -m sentiment.cli gtt ICICIBANK --stop 6 --target 20 --dry-run
    python -m sentiment.cli gtts                # list active GTT orders
    python -m sentiment.cli gtts --cancel ID    # cancel a GTT by its numeric ID
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import config
from .portfolio_manager import KiteSessionManager, PortfolioAgent

# ── Config file (persists stop/target overrides per symbol) ──────────────────
CONFIG_FILE = config.DATA_DIR / "position_config.json"


def _load_cfg() -> dict[str, dict[str, float]]:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_cfg(cfg: dict[str, dict[str, float]]) -> None:
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


# ── Price helper (LTP → CSV fallback) ────────────────────────────────────────

def _get_price(kite, symbol: str, exchange: str = "NSE") -> float | None:
    """Try Kite LTP (paid plan), then fall back to last CSV close."""
    if kite is not None:
        try:
            data = kite.ltp(f"{exchange}:{symbol}")
            key = f"{exchange}:{symbol}"
            if key in data:
                return float(data[key]["last_price"])
        except Exception:
            pass

    import pandas as pd
    for suffix in (".NS", ""):
        csv = config.DATA_DIR / f"{symbol}{suffix}__yfinance.csv"
        if csv.exists():
            try:
                df = pd.read_csv(csv, index_col=0, parse_dates=True)
                col = next((c for c in ("Close", "close") if c in df.columns), None)
                if col:
                    return float(df[col].dropna().iloc[-1])
            except Exception:
                pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# status — view holdings with P&L and saved stop/target levels
# ══════════════════════════════════════════════════════════════════════════════

def cmd_status(args: argparse.Namespace) -> int:
    cfg = _load_cfg()
    session = KiteSessionManager()

    try:
        kite = session.kite()
    except RuntimeError as e:
        print(f"[Kite] {e}\nShowing config-only view (no live data).\n")
        kite = None

    agent = PortfolioAgent(session=session)
    _, state = agent.run()

    holdings = state.get("holdings", [])
    available = state.get("available")
    used = state.get("used_margin")

    # ── header ────────────────────────────────────────────────────────────────
    if available is not None:
        print(f"\nAvailable cash : ₹{available:>12,.0f}")
        print(f"Margin in use  : ₹{used:>12,.0f}\n")
    else:
        print("\n(no live Kite session — cash balances unavailable)\n")

    if not holdings and not cfg:
        print("No holdings and no saved stop/target config.")
        return 0

    # ── build combined rows (union of live holdings + saved config) ───────────
    symbols = {h["tradingsymbol"] for h in holdings} | set(cfg.keys())
    holding_map = {h["tradingsymbol"]: h for h in holdings}

    # column widths
    W = [16, 6, 10, 10, 10, 10, 8, 8, 10, 12]
    header = (
        f"{'SYMBOL':<{W[0]}} {'QTY':>{W[1]}} {'AVG':>{W[2]}} {'LTP':>{W[3]}} "
        f"{'P&L':>{W[4]}} {'P&L%':>{W[5]}} {'STOP%':>{W[6]}} {'TGT%':>{W[7]}} "
        f"{'STOP ₹':>{W[8]}} {'TARGET ₹':>{W[9]}}"
    )
    sep = "─" * len(header)
    print(header)
    print(sep)

    for sym in sorted(symbols):
        h = holding_map.get(sym)
        sym_cfg = cfg.get(sym, {})
        stop_pct = sym_cfg.get("stop_pct")
        tgt_pct = sym_cfg.get("target_pct")

        qty = h["quantity"] if h else 0
        avg = h["avg_price"] if h else 0.0
        ltp = h["ltp"] if h else (_get_price(kite, sym) or 0.0)
        pnl = (ltp - avg) * qty if qty and avg else 0.0
        pnl_pct = ((ltp / avg) - 1) * 100 if avg else 0.0

        stop_price = avg * (1 - stop_pct / 100) if avg and stop_pct else None
        tgt_price  = avg * (1 + tgt_pct  / 100) if avg and tgt_pct  else None

        pnl_str   = f"₹{pnl:>+9,.0f}"
        pnl_p_str = f"{pnl_pct:>+6.1f}%"
        stop_str  = f"{stop_pct:.1f}" if stop_pct else "—"
        tgt_str   = f"{tgt_pct:.1f}"  if tgt_pct  else "—"
        stop_r    = f"₹{stop_price:,.1f}" if stop_price else "—"
        tgt_r     = f"₹{tgt_price:,.1f}"  if tgt_price  else "—"

        print(
            f"{sym:<{W[0]}} {qty:>{W[1]}} {avg:>{W[2]},.1f} {ltp:>{W[3]},.1f} "
            f"{pnl_str:>{W[4]+1}} {pnl_p_str:>{W[5]+1}} "
            f"{stop_str:>{W[6]}} {tgt_str:>{W[7]}} "
            f"{stop_r:>{W[8]}} {tgt_r:>{W[9]}}"
        )

    print(sep)
    print("  STOP% / TGT% are saved overrides (set with: cli set SYMBOL --stop N --target N)")
    print("  Prices are live Kite LTP or last CSV close when Kite is not connected.\n")
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# set — save stop/target override for a symbol
# ══════════════════════════════════════════════════════════════════════════════

def cmd_set(args: argparse.Namespace) -> int:
    symbol = args.symbol.upper()
    cfg = _load_cfg()
    entry = cfg.get(symbol, {})

    if args.stop is not None:
        if not (0.5 <= args.stop <= 25):
            print("stop must be between 0.5% and 25%")
            return 1
        entry["stop_pct"] = args.stop

    if args.target is not None:
        if not (1.0 <= args.target <= 100):
            print("target must be between 1% and 100%")
            return 1
        entry["target_pct"] = args.target

    if not entry:
        print("Provide at least --stop or --target")
        return 1

    cfg[symbol] = entry
    _save_cfg(cfg)

    parts = []
    if "stop_pct" in entry:
        parts.append(f"stop={entry['stop_pct']}%")
    if "target_pct" in entry:
        parts.append(f"target={entry['target_pct']}%")
    print(f"✓ {symbol}: {', '.join(parts)} saved to {CONFIG_FILE.name}")
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# list — show all saved overrides
# ══════════════════════════════════════════════════════════════════════════════

def cmd_list(args: argparse.Namespace) -> int:
    cfg = _load_cfg()
    if not cfg:
        print("No overrides saved. Use: python -m sentiment.cli set SYMBOL --stop N --target N")
        return 0

    print(f"\n{'SYMBOL':<16} {'STOP %':>8} {'TARGET %':>10}")
    print("─" * 38)
    for sym, entry in sorted(cfg.items()):
        stop = f"{entry['stop_pct']:.1f}%" if "stop_pct" in entry else "—"
        tgt  = f"{entry['target_pct']:.1f}%" if "target_pct" in entry else "—"
        print(f"{sym:<16} {stop:>8} {tgt:>10}")
    print(f"\nConfig file: {CONFIG_FILE}\n")
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# run — execute the full multi-agent pipeline with optional constraints
# ══════════════════════════════════════════════════════════════════════════════

def cmd_run(args: argparse.Namespace) -> int:
    from .multi_agent import TradingOrchestrator

    tickers = args.tickers or None
    execute = args.execute
    live    = args.live
    dry_run = not live

    if execute and live:
        print("⚠️  LIVE mode: real orders will be placed on your Zerodha account.")
        confirm = input("Type 'yes' to proceed: ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            return 1

    universe_size = getattr(args, "universe", None)
    screen_to     = getattr(args, "screen", 15)

    orch = TradingOrchestrator(
        tickers=tickers,
        budget_inr=args.budget,
        execute=execute,
        dry_run=dry_run,
        stop_floor=args.stop,
        target_floor=args.target,
        universe_size=universe_size,
        screen_to=screen_to,
    )
    result = orch.run(verbose=True)

    import json as _json
    print("\n" + "═" * 60)
    print("FINAL TRADING DECISION")
    print("═" * 60)
    print(_json.dumps(result["decision"], indent=2))

    decision = result["decision"]
    if "positions" in decision:
        print(f"\n{'─'*60}")
        print(f"POSITION SIZING  (budget ₹{args.budget:,.0f})")
        print(f"{'─'*60}")
        for p in decision["positions"]:
            if p.get("direction") in ("long", "short"):
                amt = args.budget * p.get("allocation_pct", 0) / 100
                print(
                    f"  {p['direction'].upper():5s} {p['instrument']:20s} "
                    f"{p.get('allocation_pct', 0):3.0f}%  =  ₹{amt:>10,.0f}"
                    f"  stop={p.get('stop_loss_pct', '?')}%  target={p.get('target_pct', '?')}%"
                )

    if result.get("execution_records"):
        print(f"\n{'─'*60}")
        mode = "DRY-RUN" if dry_run else "LIVE"
        print(f"EXECUTION RECORDS  [{mode}]")
        print(f"{'─'*60}")
        for r in result["execution_records"]:
            status = r.get("status", "?")
            print(f"  [{status:12s}] {r.get('instrument', r.get('error', '?'))} — {r.get('note', '')}")

    return 0


# ══════════════════════════════════════════════════════════════════════════════
# gtt — place an OCO GTT (stop + target) for a held position
# ══════════════════════════════════════════════════════════════════════════════

def cmd_gtt(args: argparse.Namespace) -> int:
    symbol   = args.symbol.upper()
    exchange = args.exchange.upper()
    dry_run  = args.dry_run
    cfg      = _load_cfg()

    # Resolve stop/target: CLI arg > saved config > error
    sym_cfg  = cfg.get(symbol, {})
    stop_pct = args.stop   if args.stop   is not None else sym_cfg.get("stop_pct")
    tgt_pct  = args.target if args.target is not None else sym_cfg.get("target_pct")

    if stop_pct is None or tgt_pct is None:
        missing = []
        if stop_pct is None: missing.append("--stop")
        if tgt_pct  is None: missing.append("--target")
        print(
            f"Missing {' and '.join(missing)} for {symbol}.\n"
            f"Either pass them on the CLI or save them first:\n"
            f"  python -m sentiment.cli set {symbol} --stop 7 --target 15"
        )
        return 1

    # Connect to Kite
    session = KiteSessionManager()
    try:
        kite = session.kite()
    except RuntimeError as e:
        print(f"[Kite] {e}")
        return 1

    # Get the holding for this symbol
    try:
        holdings = kite.holdings()
    except Exception as e:
        print(f"Could not fetch holdings: {e}")
        return 1

    holding = next((h for h in holdings if h["tradingsymbol"] == symbol), None)
    if holding is None:
        print(f"No holding found for {symbol}. Use 'status' to see what you hold.")
        return 1

    qty = int(holding.get("quantity", 0))
    if qty <= 0:
        print(f"{symbol}: quantity is {qty} — nothing to protect.")
        return 1

    avg_price = float(holding.get("average_price", 0))
    if avg_price <= 0:
        print(f"{symbol}: average_price is {avg_price} — cannot calculate levels.")
        return 1

    # Get current price for the GTT's last_price field (required by Kite)
    ltp = _get_price(kite, symbol, exchange)
    if ltp is None:
        print(f"Could not determine current price for {symbol}. Cannot place GTT.")
        return 1

    # Calculate trigger prices and limit prices
    stop_trigger = round(avg_price * (1 - stop_pct / 100), 2)
    tgt_trigger  = round(avg_price * (1 + tgt_pct  / 100), 2)
    # Limit prices: stop sells slightly below trigger (slippage buffer),
    # target sells at trigger (limit, wants to fill at target)
    stop_limit = round(stop_trigger * 0.99, 2)   # 1% slippage buffer on stop
    tgt_limit  = round(tgt_trigger,         2)

    print(f"\nGTT OCO for {exchange}:{symbol}")
    print(f"  Holding  : {qty} shares @ avg ₹{avg_price:,.2f}")
    print(f"  LTP now  : ₹{ltp:,.2f}")
    print(f"  Stop     : {stop_pct}%  →  trigger ₹{stop_trigger:,.2f}  limit ₹{stop_limit:,.2f}")
    print(f"  Target   : {tgt_pct}%  →  trigger ₹{tgt_trigger:,.2f}  limit ₹{tgt_limit:,.2f}")

    if stop_trigger >= ltp:
        print(f"\n⚠️  Stop trigger ₹{stop_trigger:,.2f} is ABOVE current price ₹{ltp:,.2f}.")
        print("  This would trigger immediately. Reduce --stop or check your avg price.")
        return 1

    if tgt_trigger <= ltp:
        print(f"\n⚠️  Target trigger ₹{tgt_trigger:,.2f} is BELOW current price ₹{ltp:,.2f}.")
        print("  This would trigger immediately. Increase --target or check your avg price.")
        return 1

    if dry_run:
        print("\n[DRY-RUN] No GTT placed. Re-run without --dry-run to place it live.")
        return 0

    try:
        from kiteconnect import exceptions as kex
        gtt_id = kite.place_gtt(
            trigger_type=kite.GTT_TYPE_OCO,
            tradingsymbol=symbol,
            exchange=exchange,
            trigger_values=[stop_trigger, tgt_trigger],   # [lower, upper]
            last_price=ltp,
            orders=[
                {   # leg 0 → stop-loss
                    "transaction_type": kite.TRANSACTION_TYPE_SELL,
                    "quantity":         qty,
                    "product":          kite.PRODUCT_CNC,
                    "order_type":       kite.ORDER_TYPE_LIMIT,
                    "price":            stop_limit,
                },
                {   # leg 1 → target
                    "transaction_type": kite.TRANSACTION_TYPE_SELL,
                    "quantity":         qty,
                    "product":          kite.PRODUCT_CNC,
                    "order_type":       kite.ORDER_TYPE_LIMIT,
                    "price":            tgt_limit,
                },
            ],
        )
        print(f"\n✓ GTT placed — id={gtt_id}")
        print("  Use 'python -m sentiment.cli gtts' to monitor status.")

        # Persist GTT id back into the config for reference
        cfg_entry = cfg.get(symbol, {})
        cfg_entry["gtt_id"] = gtt_id
        cfg[symbol] = cfg_entry
        _save_cfg(cfg)

    except Exception as e:
        print(f"\n✗ GTT placement failed: {e}")
        return 1

    return 0


# ══════════════════════════════════════════════════════════════════════════════
# gtts — list (and optionally cancel) active GTT orders
# ══════════════════════════════════════════════════════════════════════════════

def cmd_gtts(args: argparse.Namespace) -> int:
    session = KiteSessionManager()
    try:
        kite = session.kite()
    except RuntimeError as e:
        print(f"[Kite] {e}")
        return 1

    # Cancel a specific GTT
    if args.cancel is not None:
        try:
            kite.delete_gtt(args.cancel)
            print(f"✓ GTT {args.cancel} cancelled.")
        except Exception as e:
            print(f"✗ Cancel failed: {e}")
            return 1
        return 0

    try:
        gtts = kite.get_gtts()
    except Exception as e:
        print(f"Could not fetch GTTs: {e}")
        return 1

    if not gtts:
        print("No GTT orders found.")
        return 0

    active   = [g for g in gtts if g.get("status") == "active"]
    inactive = [g for g in gtts if g.get("status") != "active"]

    def _print_gtt(g: dict[str, Any]) -> None:
        cond    = g.get("condition", {})
        sym     = cond.get("tradingsymbol", "?")
        exch    = cond.get("exchange", "NSE")
        tvs     = cond.get("trigger_values", [])
        lp      = cond.get("last_price", 0)
        status  = g.get("status", "?")
        gtt_id  = g.get("id", "?")
        expires = g.get("expires_at", "")[:10]
        typ     = g.get("type", "?")

        tv_str = " / ".join(f"₹{v:,.2f}" for v in tvs)
        print(f"  id={gtt_id:<10} {exch}:{sym:<16} {status:<12} "
              f"triggers=[{tv_str}]  ltp@place=₹{lp:,.2f}  expires={expires}")
        # Show leg details
        for i, order in enumerate(g.get("orders", [])):
            leg_label = "STOP" if i == 0 and typ == "two-leg" else ("TARGET" if i == 1 else f"leg{i}")
            print(f"    [{leg_label}] SELL {order.get('quantity')} @ ₹{order.get('price'):,.2f} LIMIT")

    print(f"\nActive GTTs ({len(active)}):")
    print("─" * 80)
    for g in active:
        _print_gtt(g)

    if inactive:
        print(f"\nInactive GTTs ({len(inactive)}) — last 7 days:")
        print("─" * 80)
        for g in inactive:
            _print_gtt(g)

    print(f"\nTo cancel: python -m sentiment.cli gtts --cancel <id>")
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m sentiment.cli",
        description="Tradite portfolio CLI — view, configure, and act on your positions.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── status ────────────────────────────────────────────────────────────────
    sub.add_parser("status", help="View holdings with live P&L and saved stop/target levels")

    # ── set ───────────────────────────────────────────────────────────────────
    s = sub.add_parser("set", help="Save stop/target override for a symbol")
    s.add_argument("symbol", help="e.g. ICICIBANK")
    s.add_argument("--stop",   type=float, metavar="PCT", help="Stop-loss %%  (e.g. 7)")
    s.add_argument("--target", type=float, metavar="PCT", help="Profit target %% (e.g. 15)")

    # ── list ──────────────────────────────────────────────────────────────────
    sub.add_parser("list", help="List all saved stop/target overrides")

    # ── run ───────────────────────────────────────────────────────────────────
    r = sub.add_parser("run", help="Run the full multi-agent pipeline")
    r.add_argument("--budget",  type=float, default=100_000, metavar="INR",
                   help="Total capital in ₹ (default 100000)")
    r.add_argument("--stop",    type=float, metavar="PCT",
                   help="Minimum stop-loss %% the LLM must apply (e.g. 7)")
    r.add_argument("--target",  type=float, metavar="PCT",
                   help="Minimum profit target %% the LLM must apply (e.g. 12)")
    r.add_argument("--execute", action="store_true",
                   help="Pass decisions to ExecutionAgent (dry-run by default)")
    r.add_argument("--live",    action="store_true",
                   help="Place real orders (requires --execute; DANGER)")
    r.add_argument("--universe", type=int, metavar="N",
                   help="Expand to Nifty N universe (e.g. 500). Screener picks best --screen stocks.")
    r.add_argument("--screen",   type=int, default=15, metavar="N",
                   help="How many stocks to pass to the pipeline after screening (default 15)")
    r.add_argument("tickers",   nargs="*",
                   help="Tickers to analyse (e.g. INFY.NS RELIANCE.NS); default = full universe")

    # ── gtt ───────────────────────────────────────────────────────────────────
    g = sub.add_parser("gtt", help="Place an OCO GTT (stop-loss + target) on Kite for a held position")
    g.add_argument("symbol",   help="e.g. ICICIBANK")
    g.add_argument("--stop",   type=float, metavar="PCT",
                   help="Stop-loss %% from avg price (overrides saved config)")
    g.add_argument("--target", type=float, metavar="PCT",
                   help="Profit target %% from avg price (overrides saved config)")
    g.add_argument("--exchange", default="NSE", help="Exchange (default NSE)")
    g.add_argument("--dry-run", action="store_true",
                   help="Print the GTT parameters without placing it")

    # ── gtts ──────────────────────────────────────────────────────────────────
    gt = sub.add_parser("gtts", help="List (or cancel) active GTT orders")
    gt.add_argument("--cancel", type=int, metavar="ID",
                    help="Cancel the GTT with this numeric ID")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "status": cmd_status,
        "set":    cmd_set,
        "list":   cmd_list,
        "run":    cmd_run,
        "gtt":    cmd_gtt,
        "gtts":   cmd_gtts,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
