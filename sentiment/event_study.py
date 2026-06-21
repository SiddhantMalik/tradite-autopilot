"""
Event study — learn how stocks ACTUALLY behave after a shock, from price history.

News leaves a footprint: an abnormal (>2.5σ) daily move on heavy interest. We can't get a
free multi-year tagged news archive, so we use that price/volume footprint as the event proxy
and measure the market-adjusted cumulative abnormal return (CAR) path afterward — conditioned
on shock DIRECTION and the stock's POSITION in its 52-week range.

This replaces hand-coded "playbook" assumptions with measured numbers. Run it to (re)build
sentiment/event_model.json, which analyst.py loads. Re-run as more price history accrues:

    python -m sentiment.event_study
"""
from __future__ import annotations

import glob, os, json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import config
from .base_rates import _load_close, _position

MODEL_PATH = os.path.join(os.path.dirname(__file__), "event_model.json")
HORIZONS = (1, 5, 10, 20)
SHOCK_SIGMA = 2.5


def build_model(verbose: bool = True) -> dict:
    nif = _load_close("^NSEI")
    nret = nif.pct_change() if nif is not None else None
    rows = []
    files = glob.glob(str(config.DATA_DIR / "*__yfinance.csv"))
    for f in files:
        t = os.path.basename(f).replace("__yfinance.csv", "")
        if t.startswith("^"):
            continue
        s = _load_close(t)
        if s is None or len(s) < 400:
            continue
        r = s.pct_change()
        vol = r.rolling(60).std()
        ar = (r - nret.reindex(r.index)).values if nret is not None else r.values
        hi = s.rolling(252).max(); lo = s.rolling(252).min()
        rv = r.values; vv = vol.values
        for i in range(252, len(s) - (max(HORIZONS) + 1)):
            if vv[i] and abs(rv[i]) > SHOCK_SIGMA * vv[i]:
                d = "up" if rv[i] > 0 else "down"
                pos = _position(float(s.iloc[i]), float(hi.iloc[i]), float(lo.iloc[i]))
                cars = {str(h): float(np.nansum(ar[i + 1:i + 1 + h])) for h in HORIZONS}
                rows.append((d, pos, cars))

    def agg(subset):
        if len(subset) < 30:
            return None
        out = {"n": len(subset)}
        for h in HORIZONS:
            xs = np.array([c[str(h)] for c in subset])
            out[str(h)] = {"car": round(float(xs.mean()) * 100, 2),
                           "p_pos": round(float((xs > 0).mean()) * 100)}
        return out

    by_dir, by_dir_pos = {}, {}
    for d in ("up", "down"):
        sub = [c for (dd, _, c) in rows if dd == d]
        by_dir[d] = agg(sub)
        for p in ("near-low", "mid-range", "near-high"):
            sp = [c for (dd, pp, c) in rows if dd == d and pp == p]
            a = agg(sp)
            if a:
                by_dir_pos[f"{d}|{p}"] = a

    model = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
             "n_stocks": len(files), "n_events": len(rows),
             "shock_sigma": SHOCK_SIGMA, "horizons": list(HORIZONS),
             "by_dir": by_dir, "by_dir_pos": by_dir_pos}
    with open(MODEL_PATH, "w") as fh:
        json.dump(model, fh, indent=2)
    if verbose:
        print(f"event model: {len(rows)} events / {len(files)} stocks -> {MODEL_PATH}")
        for k, v in by_dir_pos.items():
            print(f"  {k:18s} n={v['n']:4d}  +20d CAR {v['20']['car']:+.2f}%  P>0 {v['20']['p_pos']}%")
    return model


if __name__ == "__main__":
    build_model()
