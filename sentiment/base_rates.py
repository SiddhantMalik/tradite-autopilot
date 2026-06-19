"""
Layer 8 — MEASURED historical base rates (the "past data" reality check).

The ML layer (Layer 6) reports a *fitted* mean-reversion probability (e.g.
"P(+5% in 20d) = 65%"). But this project's own research found price-only signals
carry little-to-no out-of-sample edge — a fitted probability can look confident
and still be noise. So before trusting it, we ask the data a blunt, non-parametric
question:

  "On THIS stock's own 8-year history, when the setup looked like it does today
   (RSI regime × position vs the 52-week range), what did the next 20 days
   ACTUALLY do — and was it any better than the stock's unconditional baseline?"

No model is fit; it's a conditional histogram of realised forward returns. That
makes it very hard to fool yourself. The agent is told: if this measured base rate
DISAGREES with the ML probability, trust the base rate and cut conviction.

Point-in-time / leakage: 'today' is described from the latest bar only. The
historical conditional stats use each past day's own realised forward window —
a descriptive base rate, not a peek at today's future.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ml_signals import _load_close

FWD = 20          # forward trading-day window
LOOKBACK_52W = 252


def _rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1.0 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _regime(rsi: float) -> str:
    if not np.isfinite(rsi):
        return "n/a"
    return "oversold" if rsi < 35 else "overbought" if rsi > 65 else "neutral"


def _position(price: float, hi52: float, lo52: float) -> str:
    if not (np.isfinite(price) and np.isfinite(hi52) and np.isfinite(lo52)) or hi52 <= 0:
        return "n/a"
    if price <= lo52 * 1.10:
        return "near-low"
    if price >= hi52 * 0.90:
        return "near-high"
    return "mid-range"


def _dist(x: pd.Series) -> dict:
    x = x.dropna()
    n = int(len(x))
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "mean": float(x.mean()) * 100,
        "median": float(x.median()) * 100,
        "p_gt0": float((x > 0).mean()) * 100,
        "p_gt5": float((x > 0.05).mean()) * 100,
    }


def base_rate(ticker: str) -> dict:
    """Measured forward-return base rate for this ticker's current setup."""
    s = _load_close(ticker)
    if s is None or len(s) < LOOKBACK_52W + FWD + 20:
        return {"ticker": ticker, "error": "insufficient history"}

    rsi = _rsi_series(s)
    hi = s.rolling(LOOKBACK_52W).max()
    lo = s.rolling(LOOKBACK_52W).min()
    fwd = s.shift(-FWD) / s - 1.0

    # today's setup (latest bar)
    price = float(s.iloc[-1])
    rsi_t = float(rsi.iloc[-1])
    hi_t, lo_t = float(hi.iloc[-1]), float(lo.iloc[-1])
    reg_t = _regime(rsi_t)
    pos_t = _position(price, hi_t, lo_t)
    mom_1m = (price / float(s.iloc[-21]) - 1.0) * 100 if len(s) >= 21 else float("nan")

    # historical days matching the SAME (regime, position) — exclude the tail
    # rows whose forward window runs past the data end.
    reg_hist = rsi.apply(_regime)
    pos_hist = pd.Series([_position(p, h, l) for p, h, l in zip(s, hi, lo)], index=s.index)
    valid = fwd.notna()
    mask = valid & (reg_hist == reg_t) & (pos_hist == pos_t)

    cond = _dist(fwd[mask])
    uncond = _dist(fwd[valid])

    verdict = "thin sample — low confidence"
    if cond.get("n", 0) >= 30 and uncond.get("n", 0) > 0:
        edge_mean = cond["mean"] - uncond["mean"]
        edge_p5 = cond["p_gt5"] - uncond["p_gt5"]
        if edge_mean > 0.75 and edge_p5 > 3:
            verdict = "EDGE+  (this setup historically beat the stock's baseline)"
        elif edge_mean < -0.75 or edge_p5 < -3:
            verdict = "EDGE-  (this setup historically UNDERperformed — discount bullish ML)"
        else:
            verdict = "NO EDGE  (indistinguishable from baseline — don't pay up for the ML signal)"

    return {
        "ticker": ticker, "price": price, "rsi": rsi_t, "regime": reg_t,
        "position": pos_t, "mom_1m": mom_1m, "cond": cond, "uncond": uncond,
        "verdict": verdict,
    }


class BaseRateAgent:
    """Layer 8 — measured forward-return base rates for each name's current setup."""

    def run(self, tickers: list[str]) -> str:
        blocks = []
        for t in tickers:
            r = base_rate(t)
            inst = "NSE:" + t.replace(".NS", "")
            if r.get("error"):
                blocks.append(f"{inst}: base rate n/a ({r['error']})")
                continue
            c, u = r["cond"], r["uncond"]
            if c.get("n", 0) == 0:
                blocks.append(
                    f"{inst} — setup: RSI {r['rsi']:.0f} ({r['regime']}), {r['position']}, "
                    f"1M {r['mom_1m']:+.0f}%\n  No prior days matched this exact setup "
                    f"(unconditional fwd-{FWD}d mean {u.get('mean', float('nan')):+.1f}%)."
                )
                continue
            blocks.append(
                f"{inst} — setup: RSI {r['rsi']:.0f} ({r['regime']}), {r['position']}, "
                f"1M momentum {r['mom_1m']:+.0f}%\n"
                f"  Measured fwd-{FWD}d after SIMILAR setups (n={c['n']}): "
                f"mean {c['mean']:+.1f}% | median {c['median']:+.1f}% | "
                f"P(>0)={c['p_gt0']:.0f}% | P(>+5%)={c['p_gt5']:.0f}%\n"
                f"  Unconditional fwd-{FWD}d (n={u['n']}): "
                f"mean {u['mean']:+.1f}% | P(>+5%)={u['p_gt5']:.0f}%\n"
                f"  VERDICT: {r['verdict']}"
            )
        head = (f"[BaseRateAgent]  measured from each name's own ~8y history "
                f"(non-parametric, point-in-time; the reality check on Layer 6's fitted ML probability)\n")
        return head + "\n\n".join(blocks)
