"""
Pro-technique market filters (backtest-validated) — free, price-only.

  • REGIME    — only deploy when NIFTY is above its 200-day average (risk-on). In a
                downtrend the autopilot holds cash. Cut backtest max-drawdown ~13%→6%.
  • TREND     — only buy a name trading above its own 200-DMA (don't catch falling knives).
  • VOL-SIZE  — volatility-targeted sizing: bigger in calm names, smaller in jumpy ones,
                so each position contributes similar risk.

All toggle via env (TRADITE_USE_REGIME / _TREND / _VOLSIZE). Degrade to permissive on
missing data. Used by autotrader.decide().
"""
from __future__ import annotations

import os
from .base_rates import _load_close

# Defaults: the cash-GATING filters (regime, trend) are OFF — they cut the program's
# activity (~98 trades → 28) and return (+17% → +12%) by sitting in cash, so they only
# suit someone optimising for low drawdown. The activity-NEUTRAL winner (vol-sizing) stays ON.
# Turn the gates on with TRADITE_USE_REGIME=true / TRADITE_USE_TREND=true for a defensive book.
USE_REGIME = os.getenv("TRADITE_USE_REGIME", "false").lower() == "true"
USE_TREND = os.getenv("TRADITE_USE_TREND", "false").lower() == "true"
USE_VOLSIZE = os.getenv("TRADITE_USE_VOLSIZE", "true").lower() != "false"
REF_VOL = float(os.getenv("TRADITE_REF_VOL", "0.02"))     # ~2% daily vol = neutral size


def market_regime_ok() -> tuple[bool, str]:
    """(risk_on, note) — is the broad market (NIFTY) above its 200-day average?"""
    if not USE_REGIME:
        return True, "regime filter off"
    s = _load_close("^NSEI")
    if s is None or len(s) < 200:
        return True, "regime n/a"
    last, dma = float(s.iloc[-1]), float(s.tail(200).mean())
    ok = last >= dma
    return ok, f"NIFTY {last:,.0f} {'≥' if ok else '<'} 200DMA {dma:,.0f} → {'risk-ON' if ok else 'risk-OFF (hold cash)'}"


def trend_ok(ticker: str) -> bool:
    """Is the name above its own 200-DMA (uptrend)?"""
    if not USE_TREND:
        return True
    s = _load_close(ticker)
    if s is None or len(s) < 200:
        return True
    return float(s.iloc[-1]) >= float(s.tail(200).mean())


def vol_weight(ticker: str) -> float:
    """Inverse-volatility size multiplier in [0.6, 1.6] (1.0 = neutral)."""
    if not USE_VOLSIZE:
        return 1.0
    s = _load_close(ticker)
    if s is None or len(s) < 60:
        return 1.0
    v = float(s.pct_change().tail(60).std())
    if v <= 0:
        return 1.0
    return min(max(REF_VOL / v, 0.6), 1.6)


# ── liquidity / ADV cap ──────────────────────────────────────────────────────
MAX_ADV_PCT = float(os.getenv("TRADITE_MAX_ADV_PCT", "0.05"))   # ≤5% of avg daily traded value
_adv_cache: dict[str, float] = {}


def adv_value(ticker: str, days: int = 20) -> float:
    """Average daily TRADED VALUE (₹) over ~`days` sessions — for the liquidity cap.
    0.0 if unknown (then no cap is applied). Cached per process run."""
    if ticker in _adv_cache:
        return _adv_cache[ticker]
    val = 0.0
    try:
        import yfinance as yf
        h = yf.Ticker(ticker).history(period="2mo")
        if h is not None and not h.empty and "Volume" in h and "Close" in h:
            val = float((h["Close"] * h["Volume"]).tail(days).mean())
    except Exception:  # noqa: BLE001
        val = 0.0
    _adv_cache[ticker] = val
    return val


def liquidity_cap(ticker: str) -> float:
    """Max ₹ for a single position so we never try to take more than MAX_ADV_PCT of the
    stock's daily traded value (a fill we couldn't realistically get). inf if ADV unknown."""
    adv = adv_value(ticker)
    return adv * MAX_ADV_PCT if adv > 0 else float("inf")
