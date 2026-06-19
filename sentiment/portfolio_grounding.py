"""
Portfolio & risk grounding for the sentiment LLM (PRD §8.3 / §11 / §17.3).

Companion to market_grounding.py. The sentiment LLM does NOT size positions — a
separate risk/sizing layer + a hard pre-trade clamp do, and the clamp can only
REDUCE exposure. This module makes the LLM *portfolio-aware*: given a live
PortfolioState (PIT in backtest), it emits grounding lines so the model tempers
conviction/horizon and flags concentration / drawdown / cap interactions, without
ever implying a quantity.

Two products:
  1. PORTFOLIO_PRINCIPLES — compact always-on block for the system prompt.
  2. portfolio_aware_lines(state, instrument, sector) — per-call state→action slices.

⚠️ Point-in-time: the METHODOLOGY here is timeless (leakage-safe). PortfolioState is
live/as-of — in backtests the caller must pass the portfolio snapshot at the item's
date, never the current book. Tax/cost LEVELS are 2025-26; re-verify at runtime.

Source of truth for what's injected; rationale + citations live in
knowledge/PORTFOLIO_GROUNDING.md. Keep them in sync.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------- #
# Always-on core (appended to SYSTEM_PROMPT)
# --------------------------------------------------------------------------- #
PORTFOLIO_PRINCIPLES = """\
PORTFOLIO & RISK AWARENESS — you ORIGINATE ideas; a separate risk/sizing layer + a hard pre-trade clamp decide size and can only REDUCE it. Therefore:
- Never imply a quantity, "add", "scale up", "buy more", or leverage. Output only direction/horizon/confidence; sizing happens downstream.
- Your confidence is NOT a calibrated probability — treat your own "high" as "more evidence than usual", not P=0.8. The risk layer discounts it; default modest.
- Respect portfolio state when given: if the book is in drawdown, near an exposure cap, at max positions, or already concentrated/correlated in this name or sector -> LOWER conviction, SHORTEN horizon, bias toward reducing not adding. If a hard limit is hit, prefer direction=neutral.
- Concentration guides (defaults): single name <=10-15% of NAV, single sector <=30%, ~5-15 names; correlated same-sector names act as ONE larger bet (hidden concentration), and correlations jump toward 1 in selloffs.
- Cost+tax floor: round-trip delivery friction ~0.25-0.3% and STCG is 20% (<=12m) vs LTCG 12.5% (>12m). If the expected net move is below ~0.5%, it is not actionable -> neutral. Sub-~Rs.25k delivery positions are uneconomic (fixed DP fee).
- Tax cliff: for an EXIT call on a profitable position held ~11-12 months, note the 12-month LTCG cliff (20% -> 12.5%)."""


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
class Position(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    instrument: str                      # e.g. "NSE:HDFCBANK"
    sector: str | None = None            # e.g. "Financials"
    weight_pct: float = 0.0              # fraction of NAV in [0,1]
    side: Literal["long", "short"] = "long"


class Limits(BaseModel):
    """Default risk limits drawn from PORTFOLIO_GROUNDING.md §2-§4."""
    model_config = ConfigDict(protected_namespaces=())
    single_name_cap: float = 0.15        # max fraction of NAV per name
    sector_cap: float = 0.30             # max fraction of NAV per sector
    max_positions: int = 12              # ~5-15 name book
    dd_soft: float = 0.05                # drawdown tiers (peak-to-current, +ve)
    dd_strong: float = 0.10
    dd_halt: float = 0.15
    daily_loss_limit: float = 0.02       # kill-switch level
    gross_cap: float = 1.0               # 100% invested; warn before it
    gross_warn: float = 0.85


DEFAULT_LIMITS = Limits()


class PortfolioState(BaseModel):
    """Live (point-in-time) snapshot of the operator's book.

    All *_pct fields are fractions of NAV in [0,1]. drawdown_pct is the positive
    peak-to-current drawdown (0.08 = 8% below peak). daily_pnl_pct may be negative.
    """
    model_config = ConfigDict(protected_namespaces=())
    nav: float | None = None
    cash_pct: float | None = None
    drawdown_pct: float = 0.0
    daily_pnl_pct: float | None = None
    gross_exposure_pct: float | None = None
    positions: list[Position] = Field(default_factory=list)
    limits: Limits = Field(default_factory=Limits)
    vol_regime_elevated: bool = False    # caller sets True if realized vol >~1.5x median

    # ---- derived helpers ----
    def sector_exposure(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for p in self.positions:
            if p.sector:
                out[p.sector] = round(out.get(p.sector, 0.0) + p.weight_pct, 4)
        return out

    def held(self, instrument: str) -> Position | None:
        for p in self.positions:
            if p.instrument.upper() == (instrument or "").upper():
                return p
        return None

    def names_in_sector(self, sector: str | None) -> int:
        if not sector:
            return 0
        return sum(1 for p in self.positions if (p.sector or "").lower() == sector.lower())


# --------------------------------------------------------------------------- #
# Per-call portfolio-aware grounding
# --------------------------------------------------------------------------- #
def portfolio_aware_lines(state: PortfolioState | None, instrument: str,
                          sector: str | None = None) -> list[str]:
    """Return grounding lines for the candidate item given live portfolio state.

    Lines are imperative cues the LLM should apply to direction/horizon/confidence.
    Returns [] when no state is supplied (the always-on PORTFOLIO_PRINCIPLES still
    applies via the system prompt).
    """
    if state is None:
        return []
    L = state.limits or DEFAULT_LIMITS
    lines: list[str] = []

    # --- Drawdown tiers (most important risk gate) ---
    dd = state.drawdown_pct or 0.0
    if dd >= L.dd_halt:
        lines.append(f"PORTFOLIO: book in DRAWDOWN HALT ({dd:.0%} >= {L.dd_halt:.0%}) -> block new entries; "
                     "emit only exit / stop-adjust reads. For a new long, prefer direction=neutral.")
    elif dd >= L.dd_strong:
        lines.append(f"PORTFOLIO: SIGNIFICANT drawdown ({dd:.0%}) -> highest-conviction single-name only; "
                     "no new F&O; reduce don't add; shorten horizon; cut confidence hard.")
    elif dd >= L.dd_soft:
        lines.append(f"PORTFOLIO: book in drawdown ({dd:.0%}) -> lower conviction on new longs, don't add positions, "
                     "shorten horizon ~50%, bias to reducing.")

    # --- Daily loss limit proximity (kill-switch zone) ---
    if state.daily_pnl_pct is not None and state.daily_pnl_pct <= -0.8 * L.daily_loss_limit:
        lines.append(f"PORTFOLIO: near daily loss limit (today {state.daily_pnl_pct:+.1%}, limit -{L.daily_loss_limit:.0%}) "
                     "-> no new long ideas today; only exits / stop adjustments.")

    # --- Already holding this name (add-on) ---
    pos = state.held(instrument)
    if pos is not None:
        headroom = max(0.0, L.single_name_cap - pos.weight_pct)
        if pos.weight_pct >= L.single_name_cap:
            lines.append(f"PORTFOLIO: already hold {pos.weight_pct:.0%} of {instrument} (>= {L.single_name_cap:.0%} cap) "
                         "-> name cap hit; a fresh long is an over-cap ADD -> prefer neutral.")
        else:
            lines.append(f"PORTFOLIO: already hold {pos.weight_pct:.0%} of {instrument} (cap {L.single_name_cap:.0%}, "
                         f"headroom {headroom:.0%}) -> this is an ADD-ON, not a fresh bet; size is constrained.")

    # --- Sector concentration ---
    if sector:
        sx = state.sector_exposure().get(sector, 0.0)
        n_same = state.names_in_sector(sector)
        if sx >= L.sector_cap:
            lines.append(f"PORTFOLIO: sector {sector} at {sx:.0%} (>= {L.sector_cap:.0%} cap) "
                         "-> sector cap hit; suppress regardless of sentiment -> prefer neutral.")
        elif sx >= 0.8 * L.sector_cap:
            lines.append(f"PORTFOLIO: sector {sector} at {sx:.0%} (cap {L.sector_cap:.0%}) -> little headroom; temper conviction.")
        if n_same >= 2:
            lines.append(f"PORTFOLIO: already hold {n_same} {sector} names -> hidden/correlated concentration "
                         "(treat as one larger bet; correlations rise toward 1 in selloffs) -> down-grade conviction one notch.")

    # --- Exposure ceiling / book full ---
    if state.gross_exposure_pct is not None and state.gross_exposure_pct >= L.gross_warn:
        lines.append(f"PORTFOLIO: near gross-exposure ceiling ({state.gross_exposure_pct:.0%} of {L.gross_cap:.0%}) "
                     "-> capital is ~zero-sum; a new idea needs an existing position trimmed first.")
    if len(state.positions) >= L.max_positions:
        lines.append(f"PORTFOLIO: at max position count ({len(state.positions)}/{L.max_positions}) "
                     "-> exit something before adding a new name.")

    # --- Volatility regime ---
    if state.vol_regime_elevated:
        lines.append("PORTFOLIO: elevated volatility regime -> prefer shorter horizon and smaller conviction; "
                     "do not read high vol as a buying opportunity unless the signal is very clear.")

    return lines


def summarize(state: PortfolioState | None) -> str | None:
    """One-line header of the book state for the prompt (compact)."""
    if state is None:
        return None
    parts = [f"positions={len(state.positions)}"]
    if state.drawdown_pct:
        parts.append(f"drawdown={state.drawdown_pct:.0%}")
    if state.gross_exposure_pct is not None:
        parts.append(f"gross={state.gross_exposure_pct:.0%}")
    if state.cash_pct is not None:
        parts.append(f"cash={state.cash_pct:.0%}")
    if state.daily_pnl_pct is not None:
        parts.append(f"today={state.daily_pnl_pct:+.1%}")
    sx = state.sector_exposure()
    if sx:
        top = sorted(sx.items(), key=lambda kv: kv[1], reverse=True)[:3]
        parts.append("sectors=" + ",".join(f"{k}:{v:.0%}" for k, v in top))
    return "; ".join(parts)


__all__ = ["PORTFOLIO_PRINCIPLES", "PortfolioState", "Position", "Limits",
           "DEFAULT_LIMITS", "portfolio_aware_lines", "summarize"]
