"""
Exact ZERODHA equity-DELIVERY (CNC) charges — itemised, not a blended guess.
Source: Zerodha brokerage-calculator components (2026).

Per leg, on trade value V (₹):
  brokerage        = ₹0            (Zerodha: free for equity delivery)
  STT              = 0.10% of V    (charged on BOTH buy and sell, delivery)
  exchange txn     = 0.00297% of V (NSE)
  SEBI charges     = ₹10 per crore = 0.0001% of V
  GST              = 18% of (brokerage + exchange txn + SEBI)
  stamp duty       = 0.015% of V   (BUY side only)
  DP charge        = ₹15.93 flat   (SELL side only: ₹13.5 + 18% GST, per scrip)
  slippage         = TRADITE_SLIPPAGE % (optional; NOT a Zerodha charge, default 0)

Tax: STCG 20% on net realised short-term gains (held < 12 months).
Round trip on a ₹1,00,000 delivery trade ≈ 0.23% + ₹15.93 ≈ ₹243 (vs the old flat ₹420 guess).
"""
from __future__ import annotations

import os

# Zerodha equity-delivery components (override any via env if rates change)
BROKERAGE_PCT = float(os.getenv("TRADITE_BROKERAGE_PCT", "0.0"))      # ₹0 delivery
STT_PCT = float(os.getenv("TRADITE_STT_PCT", "0.001"))               # 0.10% each side
EXCH_TXN_PCT = float(os.getenv("TRADITE_EXCH_TXN_PCT", "0.0000297"))  # NSE 0.00297%
SEBI_PCT = float(os.getenv("TRADITE_SEBI_PCT", "0.000001"))          # ₹10 / crore
STAMP_PCT = float(os.getenv("TRADITE_STAMP_PCT", "0.00015"))         # 0.015% buy only
GST_PCT = float(os.getenv("TRADITE_GST_PCT", "0.18"))               # 18%
DP_CHARGE = float(os.getenv("TRADITE_DP_CHARGE", "15.93"))           # ₹ flat, sell only
SLIPPAGE_PCT = float(os.getenv("TRADITE_SLIPPAGE", "0.0"))           # optional, both sides
STCG_RATE = float(os.getenv("TRADITE_STCG_RATE", "0.20"))


def _common(value: float) -> float:
    brk = BROKERAGE_PCT * value
    exch = EXCH_TXN_PCT * value
    sebi = SEBI_PCT * value
    gst = GST_PCT * (brk + exch + sebi)
    stt = STT_PCT * value
    slip = SLIPPAGE_PCT * value
    return brk + exch + sebi + gst + stt + slip


def buy_cost(value: float) -> float:
    return _common(value) + STAMP_PCT * value          # stamp duty on buy


def sell_cost(value: float) -> float:
    return _common(value) + DP_CHARGE                  # flat DP charge on sell


def trade_cost(value: float, side: str = "sell") -> float:
    """Total Zerodha-delivery charges for one leg. side = 'buy' | 'sell'."""
    value = abs(value)
    return buy_cost(value) if side == "buy" else sell_cost(value)


def round_trip_pct() -> float:
    """Percentage-based round-trip (excludes the flat ₹DP charge)."""
    return 2 * (BROKERAGE_PCT + EXCH_TXN_PCT + SEBI_PCT + STT_PCT + SLIPPAGE_PCT) \
        + GST_PCT * 2 * (BROKERAGE_PCT + EXCH_TXN_PCT + SEBI_PCT) + STAMP_PCT


def stcg(net_realised_gain: float) -> float:
    return max(0.0, net_realised_gain) * STCG_RATE
