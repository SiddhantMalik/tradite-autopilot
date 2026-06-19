"""
Hard pre-trade RISK GATE (PRD §17.3, formalised).

The LLM proposes; the gate disposes. Every order is checked by CODE — not a prompt
suggestion — before it can reach the broker. The gate can APPROVE, CLAMP (shrink to
the limit) or REJECT, and it owns the kill-switch (daily-loss + drawdown halts).

This is deliberately boring and deterministic: risk control must not depend on a
model's mood. Limits are configurable via env (TRADITE_RISK_*) or constructor.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _envf(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Limits:
    max_position_pct: float = _envf("TRADITE_RISK_MAX_POS_PCT", 0.15)    # ≤15% NAV per name
    max_sector_pct:   float = _envf("TRADITE_RISK_MAX_SECTOR_PCT", 0.30) # ≤30% NAV per sector
    min_cash_pct:     float = _envf("TRADITE_RISK_MIN_CASH_PCT", 0.05)   # keep ≥5% cash
    max_positions:    int   = int(_envf("TRADITE_RISK_MAX_POSITIONS", 10))
    require_stop:     bool  = os.getenv("TRADITE_RISK_REQUIRE_STOP", "true").lower() != "false"
    min_rr:           float = _envf("TRADITE_RISK_MIN_RR", 1.5)          # target:stop ≥ 1.5
    max_stop_pct:     float = _envf("TRADITE_RISK_MAX_STOP_PCT", 12.0)   # stop no wider than 12%
    daily_loss_halt_pct: float = _envf("TRADITE_RISK_DAILY_LOSS_HALT", 0.05)  # halt buys at −5% day
    drawdown_halt_pct:   float = _envf("TRADITE_RISK_DD_HALT", 0.15)          # kill-switch at −15% from peak


class RiskGate:
    def __init__(self, nav: float, limits: Limits | None = None):
        self.nav = float(nav)
        self.lim = limits or Limits()

    # ── kill-switch / halts (checked before any new buying) ──────────────────
    def check_halt(self, book) -> tuple[bool, str]:
        """Return (halted, reason). book exposes marked_nav(), peak_nav, day_pnl()."""
        peak = getattr(book, "peak_nav", None) or book.marked_nav()
        cur = book.marked_nav()
        if peak > 0 and (cur / peak - 1) <= -self.lim.drawdown_halt_pct:
            return True, (f"KILL-SWITCH: drawdown {(cur/peak-1)*100:.1f}% ≤ "
                          f"−{self.lim.drawdown_halt_pct*100:.0f}% from peak — no new buys")
        day = book.day_pnl()
        if self.nav > 0 and day / self.nav <= -self.lim.daily_loss_halt_pct:
            return True, (f"HALT: today's P&L {day/self.nav*100:.1f}% ≤ "
                          f"−{self.lim.daily_loss_halt_pct*100:.0f}% of NAV — no new buys today")
        return False, ""

    # ── per-order check ──────────────────────────────────────────────────────
    def check_order(self, order: dict, book) -> dict:
        """order: {symbol, rupees, price, stop_pct, target_pct, sector}.
        Returns {decision: approve|clamp|reject, rupees, qty, reasons[]}."""
        reasons: list[str] = []
        rupees = float(order.get("rupees", 0))
        price = float(order.get("price", 0))
        stop_pct = order.get("stop_pct")
        target_pct = order.get("target_pct")
        sector = order.get("sector") or "Unknown"
        sym = order.get("symbol", "?")

        if price <= 0 or rupees <= 0:
            return {"decision": "reject", "rupees": 0, "qty": 0,
                    "reasons": ["no price or zero allocation"]}

        # 1. stop-loss required + sane width
        if self.lim.require_stop and not stop_pct:
            return {"decision": "reject", "rupees": 0, "qty": 0,
                    "reasons": [f"{sym}: no stop-loss — required by risk policy"]}
        if stop_pct and stop_pct > self.lim.max_stop_pct:
            return {"decision": "reject", "rupees": 0, "qty": 0,
                    "reasons": [f"{sym}: stop {stop_pct}% wider than max {self.lim.max_stop_pct}%"]}

        # 2. reward:risk floor
        if stop_pct and target_pct:
            rr = target_pct / stop_pct
            if rr < self.lim.min_rr:
                return {"decision": "reject", "rupees": 0, "qty": 0,
                        "reasons": [f"{sym}: R:R {rr:.2f} < min {self.lim.min_rr}"]}

        # 3. max number of positions
        if sym not in book.positions and len(book.positions) >= self.lim.max_positions:
            return {"decision": "reject", "rupees": 0, "qty": 0,
                    "reasons": [f"{sym}: already at max {self.lim.max_positions} positions"]}

        # 4. single-name cap → clamp
        pos_cap = self.lim.max_position_pct * self.nav
        if rupees > pos_cap:
            reasons.append(f"clamped to single-name cap {self.lim.max_position_pct*100:.0f}% "
                           f"(₹{pos_cap:,.0f})")
            rupees = pos_cap

        # 5. sector cap → clamp
        sec_now = book.sector_exposure().get(sector, 0.0)
        sec_cap = self.lim.max_sector_pct * self.nav
        if sec_now + rupees > sec_cap:
            room = max(0.0, sec_cap - sec_now)
            reasons.append(f"clamped to sector cap {self.lim.max_sector_pct*100:.0f}% "
                           f"({sector}: ₹{room:,.0f} room)")
            rupees = room

        # 6. cash + min reserve → clamp
        reserve = self.lim.min_cash_pct * self.nav
        spendable = max(0.0, book.cash - reserve)
        if rupees > spendable:
            reasons.append(f"clamped to spendable cash (keep ₹{reserve:,.0f} reserve)")
            rupees = spendable

        qty = int(rupees // price)
        if qty < 1:
            return {"decision": "reject", "rupees": 0, "qty": 0,
                    "reasons": reasons + [f"{sym}: not enough room/cash for 1 share"]}

        return {"decision": "clamp" if reasons else "approve",
                "rupees": qty * price, "qty": qty, "reasons": reasons}
