"""
Realistic Indian trading costs — the thing that turns a pretty backtest into a real P&L.

Every buy and sell pays statutory charges + slippage; every net short-term gain pays STCG.
Models NSE equity DELIVERY (CNC) on a discount broker (Zerodha-style, ₹0 delivery brokerage):

  per side : STT 0.10% + exchange ~0.003% + SEBI + stamp 0.015% (buy) + 18% GST + slippage
           ≈ 0.21% each way  →  ~0.42% round-trip before tax
  tax      : STCG 20% on net realised short-term gains (held < 12 months)

All tunable via env. This is deliberately a touch conservative — better to under-promise.
"""
from __future__ import annotations

import os

# one blended per-side rate (statutory + slippage). Override with TRADITE_COST_PER_SIDE.
COST_PER_SIDE = float(os.getenv("TRADITE_COST_PER_SIDE", "0.0021"))   # 0.21% each way
SLIPPAGE_EXTRA = float(os.getenv("TRADITE_SLIPPAGE", "0.0"))          # add more slippage if desired
STCG_RATE = float(os.getenv("TRADITE_STCG_RATE", "0.20"))            # short-term capital gains tax


def trade_cost(rupees: float) -> float:
    """Charges + slippage for one side (buy OR sell) of a delivery trade."""
    return abs(rupees) * (COST_PER_SIDE + SLIPPAGE_EXTRA)


def round_trip_pct() -> float:
    return 2 * (COST_PER_SIDE + SLIPPAGE_EXTRA)


def stcg(net_realised_gain: float) -> float:
    """Tax on net realised short-term gains (no tax on a net loss)."""
    return max(0.0, net_realised_gain) * STCG_RATE
