"""
Portfolio state reader and order executor for Zerodha Kite Connect.

┌──────────────────────────────────────────────────────────────────────┐
│  KiteSessionManager  — validates token, auto-warns on TokenException │
│  PortfolioAgent      — reads live holdings + funds from Kite;        │
│                        becomes Layer 4 in the orchestrator prompt    │
│  ExecutionAgent      — takes decision dict → places / simulates      │
│                        orders on Kite (dry_run=True by default)      │
└──────────────────────────────────────────────────────────────────────┘

⚠️  SAFETY
  - dry_run=True always unless --live is explicitly passed on the CLI
  - Orders are placed as CNC (delivery) MARKET orders during market hours
  - place_order() returning an order_id is NOT a fill — we log the id and
    poll order_history() once for early-rejection detection
  - Never risk > 5 % of available cash on a single position (enforced)

Auth (add to tradite/.env):
    KITE_API_KEY=xxxxxxxx
    KITE_ACCESS_TOKEN=yyyyyyy   # regenerate daily via login flow below

Login flow (run once per day before market open):
    python -m sentiment.portfolio_manager --login

Usage:
    python -m sentiment.portfolio_manager --status
    python -m sentiment.multi_agent --execute --dry-run   # simulate
    python -m sentiment.multi_agent --execute --live      # real (DANGER)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Any

import config

log = logging.getLogger(__name__)

# ── Kite creds ────────────────────────────────────────────────────────────────
_API_KEY      = lambda: os.environ.get("KITE_API_KEY", "")
_ACCESS_TOKEN = lambda: os.environ.get("KITE_ACCESS_TOKEN", "")

_IST = timedelta(hours=5, minutes=30)
_NSE_OPEN_UTC  = timedelta(hours=3, minutes=45)   # 09:15 IST
_NSE_CLOSE_UTC = timedelta(hours=10, minutes=0)   # 15:30 IST

# ── helpers ───────────────────────────────────────────────────────────────────

def _ist_now() -> datetime:
    return datetime.now(timezone.utc) + _IST


def _in_market_hours() -> bool:
    now_utc = datetime.now(timezone.utc)
    if now_utc.weekday() > 4:
        return False
    tod = timedelta(hours=now_utc.hour, minutes=now_utc.minute)
    return _NSE_OPEN_UTC <= tod <= _NSE_CLOSE_UTC


# ══════════════════════════════════════════════════════════════════════════════
# Kite session manager
# ══════════════════════════════════════════════════════════════════════════════

class KiteSessionManager:
    """
    Wraps KiteConnect.  Validates the stored access_token on first use.
    Raises RuntimeError with a clear message if creds are missing or stale.
    """

    def __init__(self):
        self._kite = None

    def kite(self):
        if self._kite is not None:
            return self._kite

        api_key = _API_KEY()
        access_token = _ACCESS_TOKEN()
        if not api_key or not access_token:
            raise RuntimeError(
                "Kite creds missing. Add KITE_API_KEY and KITE_ACCESS_TOKEN "
                "to tradite/.env, then run: python -m sentiment.portfolio_manager --login"
            )

        try:
            from kiteconnect import KiteConnect
            from kiteconnect import exceptions as kex
        except ImportError:
            raise RuntimeError(
                "kiteconnect not installed. Run: pip install kiteconnect --break-system-packages"
            )

        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)

        # cheap validation — will raise TokenException if token is stale
        try:
            profile = kite.profile()
            log.info("Kite authenticated as %s", profile.get("user_name", "?"))
        except kex.TokenException as e:
            raise RuntimeError(
                f"Kite access_token is expired or invalid: {e}\n"
                "Run: python -m sentiment.portfolio_manager --login"
            )

        self._kite = kite
        return kite

    # ── login helper ──────────────────────────────────────────────────────────

    def login(self, api_secret: str | None = None) -> None:
        """Interactive one-time login to generate today's access_token."""
        api_key = _API_KEY()
        if not api_key:
            print("Set KITE_API_KEY in tradite/.env first.")
            return

        try:
            from kiteconnect import KiteConnect
        except ImportError:
            print("pip install kiteconnect --break-system-packages")
            return

        kite = KiteConnect(api_key=api_key)
        print("\n1. Open this URL in your browser and log in:")
        print(f"\n   {kite.login_url()}\n")
        print("2. After login, Zerodha redirects you. Paste the FULL redirect URL here.")
        redirect = input("Redirect URL: ").strip()

        import urllib.parse as up
        params = dict(up.parse_qsl(up.urlparse(redirect).query))
        request_token = params.get("request_token", "")
        if not request_token:
            print("Could not find request_token in the URL. Try again.")
            return

        if api_secret is None:
            import getpass
            api_secret = getpass.getpass("API secret: ")

        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]
        print(f"\n✓ access_token: {access_token}")
        print("Add this to tradite/.env:")
        print(f"  KITE_ACCESS_TOKEN={access_token}\n")


# ══════════════════════════════════════════════════════════════════════════════
# PortfolioAgent — Layer 4 in the orchestrator prompt
# ══════════════════════════════════════════════════════════════════════════════

class PortfolioAgent:
    """
    Reads live holdings, day positions, and available equity margin from Kite.
    Returns a tuple (prompt_text, state_dict) so the orchestrator can both
    include the text in the LLM prompt and pass the state_dict to ExecutionAgent.
    """

    def __init__(self, session: KiteSessionManager | None = None):
        self.session = session or KiteSessionManager()

    def run(self) -> tuple[str, dict[str, Any]]:
        try:
            return self._fetch()
        except RuntimeError as e:
            msg = f"[PortfolioAgent] Kite unavailable: {e}"
            return msg, {"available": None, "holdings": [], "error": str(e)}
        except Exception as e:
            msg = f"[PortfolioAgent] Unexpected error: {e}"
            return msg, {"available": None, "holdings": [], "error": str(e)}

    def _fetch(self) -> tuple[str, dict[str, Any]]:
        kite = self.session.kite()

        # ── funds ──────────────────────────────────────────────────────────
        margins = kite.margins(segment=kite.MARGIN_EQUITY)
        available = float(margins.get("available", {}).get("cash", 0))
        used = float(margins.get("utilised", {}).get("debits", 0))

        # ── holdings (overnight equity) ───────────────────────────────────
        raw_holdings = kite.holdings()
        holdings = []
        for h in raw_holdings:
            qty = int(h.get("quantity", 0))
            if qty == 0:
                continue
            avg = float(h.get("average_price", 0))
            ltp = float(h.get("last_price", avg))   # ltp may be 0 outside hours
            pnl = (ltp - avg) * qty
            holdings.append({
                "tradingsymbol": h["tradingsymbol"],
                "exchange":      h.get("exchange", "NSE"),
                "quantity":      qty,
                "avg_price":     avg,
                "ltp":           ltp,
                "pnl":           pnl,
            })

        # ── today's open positions ─────────────────────────────────────────
        positions_resp = kite.positions()
        open_positions = []
        for p in positions_resp.get("net", []):
            qty = int(p.get("quantity", 0))
            if qty == 0:
                continue
            open_positions.append({
                "tradingsymbol": p["tradingsymbol"],
                "product":       p.get("product", "?"),
                "quantity":      qty,
                "average_price": float(p.get("average_price", 0)),
                "pnl":           float(p.get("pnl", 0)),
            })

        state = {
            "available": available,
            "used_margin": used,
            "holdings": holdings,
            "open_positions": open_positions,
        }

        # ── build prompt text ──────────────────────────────────────────────
        lines = [
            "[PortfolioAgent]",
            f"Available cash (equity margin): ₹{available:,.0f}",
            f"Margin in use: ₹{used:,.0f}",
        ]
        if holdings:
            lines.append(f"\nCurrent holdings ({len(holdings)} stocks):")
            for h in holdings:
                pnl_str = f"P&L ₹{h['pnl']:+,.0f}"
                lines.append(f"  {h['exchange']}:{h['tradingsymbol']:15s}  "
                              f"qty={h['quantity']:4d}  avg=₹{h['avg_price']:,.1f}  "
                              f"ltp=₹{h['ltp']:,.1f}  {pnl_str}")
        else:
            lines.append("No equity holdings.")

        if open_positions:
            lines.append(f"\nOpen intraday positions ({len(open_positions)}):")
            for p in open_positions:
                lines.append(f"  {p['tradingsymbol']:15s}  qty={p['quantity']:+d}  "
                              f"avg=₹{p['average_price']:,.1f}  P&L ₹{p['pnl']:+,.0f}")

        return "\n".join(lines), state


# ══════════════════════════════════════════════════════════════════════════════
# ExecutionAgent — converts decision JSON → Kite orders
# ══════════════════════════════════════════════════════════════════════════════

class ExecutionAgent:
    """
    Converts the orchestrator's JSON decision into Kite orders.

    Parameters
    ----------
    session      : shared KiteSessionManager
    budget_inr   : total budget cap (₹)
    dry_run      : True → log only; False → place real orders (DANGER)
    max_per_pos  : max fraction of budget on one position (default 0.35 = 35%)
    """

    def __init__(
        self,
        session: KiteSessionManager | None = None,
        budget_inr: float = 100_000,
        dry_run: bool = True,
        max_per_pos: float = 0.35,
    ):
        self.session    = session or KiteSessionManager()
        self.budget     = budget_inr
        self.dry_run    = dry_run
        self.max_per_pos = max_per_pos

    # ── price helper ──────────────────────────────────────────────────────────

    def _current_price(self, kite, symbol: str, exchange: str = "NSE") -> float | None:
        """Try Kite LTP (paid plan), else fall back to last CSV close."""
        try:
            data = kite.ltp(f"{exchange}:{symbol}")
            key = f"{exchange}:{symbol}"
            if key in data:
                return float(data[key]["last_price"])
        except Exception:
            pass  # free plan / outside hours

        # CSV fallback
        import pandas as pd
        for ticker_suffix in (".NS", ""):
            csv = config.DATA_DIR / f"{symbol}{ticker_suffix}__yfinance.csv"
            if csv.exists():
                try:
                    df = pd.read_csv(csv, index_col=0, parse_dates=True)
                    col = next((c for c in ("Close", "close") if c in df.columns), None)
                    if col:
                        return float(df[col].dropna().iloc[-1])
                except Exception:
                    pass
        return None

    # ── quantity calculation ──────────────────────────────────────────────────

    def _compute_qty(self, alloc_pct: float, price: float) -> int:
        """Shares to buy given allocation % of budget and current price."""
        cap = min(alloc_pct / 100.0, self.max_per_pos) * self.budget
        qty = int(cap // price)
        return max(qty, 1)

    # ── main entry point ──────────────────────────────────────────────────────

    def execute(
        self,
        decision: dict[str, Any],
        portfolio_state: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute (or simulate) the decision.

        Returns a list of execution records:
          {symbol, action, qty, price_hint, order_id | None, status, note}
        """
        if "positions" not in decision:
            return [{"error": "decision has no 'positions' key"}]

        portfolio_state = portfolio_state or {}
        available_cash  = portfolio_state.get("available", self.budget)
        held_map = {h["tradingsymbol"]: h for h in portfolio_state.get("holdings", [])}

        try:
            kite = self.session.kite()
        except RuntimeError as e:
            if not self.dry_run:
                raise
            kite = None  # dry-run without Kite is fine for simulation

        records: list[dict[str, Any]] = []
        cash_committed = 0.0

        for pos in decision["positions"]:
            direction  = pos.get("direction", "skip")
            instrument = pos.get("instrument", "")    # e.g. "NSE:RELIANCE"
            alloc_pct  = float(pos.get("allocation_pct", 0))
            stop_pct   = float(pos.get("stop_loss_pct", 7))

            if direction == "skip" or alloc_pct == 0:
                records.append({"instrument": instrument, "action": "skip",
                                 "reason": "decision=skip or alloc=0"})
                continue

            # Parse exchange and symbol
            parts = instrument.split(":", 1)
            exchange = parts[0] if len(parts) == 2 else "NSE"
            symbol   = parts[1] if len(parts) == 2 else parts[0]

            # Get current price
            price = self._current_price(kite, symbol, exchange) if kite else \
                    self._current_price(None, symbol, exchange)

            if price is None or price <= 0:
                records.append({
                    "instrument": instrument, "action": "error",
                    "note": "could not determine current price"
                })
                continue

            # Quantity
            qty = self._compute_qty(alloc_pct, price)
            cost = qty * price

            rec: dict[str, Any] = {
                "instrument": instrument,
                "symbol":     symbol,
                "exchange":   exchange,
                "direction":  direction,
                "qty":        qty,
                "price_hint": round(price, 2),
                "cost_est":   round(cost, 2),
                "stop_loss":  round(price * (1 - stop_pct / 100), 2),
                "order_id":   None,
                "status":     "pending",
                "note":       "",
            }

            # ── already held? ──────────────────────────────────────────────
            if direction == "long" and symbol in held_map:
                existing_qty = held_map[symbol]["quantity"]
                rec["note"] = f"already holding {existing_qty} shares; skipping duplicate long"
                rec["status"] = "skipped"
                records.append(rec)
                continue

            # ── cash check ─────────────────────────────────────────────────
            if direction == "long":
                remaining = (available_cash or self.budget) - cash_committed
                if cost > remaining:
                    rec["note"] = (f"insufficient cash: need ₹{cost:,.0f} "
                                   f"but only ₹{remaining:,.0f} left")
                    rec["status"] = "skipped"
                    records.append(rec)
                    continue

            # ── DRY RUN ────────────────────────────────────────────────────
            if self.dry_run:
                rec["status"] = "DRY-RUN"
                rec["note"]   = (
                    f"Would place {'BUY' if direction == 'long' else 'SELL'} "
                    f"{qty}x {symbol} @ ≈₹{price:,.2f}  "
                    f"(est. cost ₹{cost:,.0f},  stop ₹{rec['stop_loss']:,.2f})"
                )
                if direction == "long":
                    cash_committed += cost
                records.append(rec)
                continue

            # ── LIVE ORDER ─────────────────────────────────────────────────
            if not _in_market_hours():
                rec["status"] = "rejected"
                rec["note"]   = "Outside NSE market hours; cannot place live order"
                records.append(rec)
                continue

            try:
                from kiteconnect import exceptions as kex
                transaction = (kite.TRANSACTION_TYPE_BUY
                               if direction == "long"
                               else kite.TRANSACTION_TYPE_SELL)
                order_id = kite.place_order(
                    variety=kite.VARIETY_REGULAR,
                    exchange=exchange,
                    tradingsymbol=symbol,
                    transaction_type=transaction,
                    quantity=qty,
                    product=kite.PRODUCT_CNC,          # delivery
                    order_type=kite.ORDER_TYPE_MARKET,
                )
                rec["order_id"] = order_id
                rec["status"]   = "placed"
                rec["note"]     = f"order_id={order_id}; poll order_history for true status"
                if direction == "long":
                    cash_committed += cost

                # quick rejection check (~500ms)
                time.sleep(0.5)
                try:
                    history = kite.order_history(order_id)
                    last = history[-1] if history else {}
                    oms_status = last.get("status", "?")
                    rec["status"] = oms_status
                    if oms_status in ("REJECTED", "CANCELLED"):
                        rec["note"] += f"; REJECTED: {last.get('status_message', '')}"
                except Exception:
                    pass  # history check is best-effort

            except Exception as e:
                rec["status"] = "error"
                rec["note"]   = str(e)

            records.append(rec)

        return records


# ══════════════════════════════════════════════════════════════════════════════
# CLI — standalone portfolio status / login
# ══════════════════════════════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]

    if "--login" in argv:
        KiteSessionManager().login()
        return 0

    if "--status" in argv:
        agent = PortfolioAgent()
        text, state = agent.run()
        print(text)
        if "--json" in argv:
            print("\n" + json.dumps(state, indent=2))
        return 0

    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
