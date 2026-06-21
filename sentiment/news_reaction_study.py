"""
News -> reaction study (Marketaux-backed), with COMPOUNDING.

Question: does tagged news sentiment predict a stock's forward, market-adjusted return — and
does COMPOUNDING (several articles in a short window -> stronger net sentiment / higher
intensity) predict a bigger or more reliable move than a single article?

Method:
  1. news_history.csv -> per (symbol, calendar-day) aggregate: n_articles, mean sentiment.
     Marketaux entity sentiment is ~0-1 with 0.5 neutral; signed = (s-0.5)*2 in [-1,1].
  2. COMPOUND each event over a trailing WINDOW (default 3 calendar days): sum signed
     sentiment (compounded tone) and count articles (intensity).
  3. Align the event to the next trading day; measure market-adjusted CAR (vs ^NSEI) at
     +1/+5/+10/+20 trading days.
  4. Bucket by sentiment sign (single) and by compounded-tone × intensity (compounding);
     report mean CAR + hit-rate per bucket, and a simple correlation/IC.

Honesty: free-tier Marketaux is thin, so this is a first-pass measurement on a short, recent
window — it accumulates as news_history.csv grows. Run:  python -m sentiment.news_reaction_study
"""
from __future__ import annotations

import os, json, math
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from .base_rates import _load_close, _position, _rsi_series, _regime

HORIZONS = (1, 5, 10, 20)
CSV = os.path.join(os.path.dirname(__file__), "news_history.csv")
OUT = os.path.join(os.path.dirname(__file__), "news_reaction_model.json")


def _signed(s):
    """Marketaux 0-1 sentiment -> signed [-1,1]; NaN-safe."""
    try:
        return max(-1.0, min(1.0, (float(s) - 0.5) * 2))
    except Exception:  # noqa: BLE001
        return np.nan


def _fwd_car(sym: str, day: pd.Timestamp, nidx: pd.Series) -> dict | None:
    """Market-adjusted cumulative return at each horizon, from the first trading day >= day."""
    s = _load_close(sym)
    if s is None or len(s) < 60:
        return None
    s.index = pd.to_datetime(s.index)
    idx = s.index.searchsorted(day)           # first trading day on/after the news day
    if idx >= len(s) - (max(HORIZONS) + 1):
        return None
    p0 = float(s.iloc[idx])
    # setup context for conditioning
    rsi = _rsi_series(s); hi = s.rolling(252).max(); lo = s.rolling(252).min()
    try:
        pos = _position(p0, float(hi.iloc[idx]), float(lo.iloc[idx]))
        reg = _regime(float(rsi.iloc[idx]))
    except Exception:  # noqa: BLE001
        pos, reg = "mid-range", "neutral"
    out = {"pos": pos, "reg": reg}
    n0 = float(nidx.iloc[nidx.index.searchsorted(s.index[idx])]) if nidx is not None else None
    for h in HORIZONS:
        stock = float(s.iloc[idx + h]) / p0 - 1.0
        mkt = 0.0
        if nidx is not None:
            j = nidx.index.searchsorted(s.index[idx])
            if j + h < len(nidx):
                mkt = float(nidx.iloc[j + h]) / n0 - 1.0
        out[str(h)] = (stock - mkt) * 100.0
    return out


def build(window_days: int = 3, verbose: bool = True) -> dict:
    if not os.path.exists(CSV):
        raise SystemExit("no news_history.csv — run marketaux_news.fetch_history first")
    df = pd.read_csv(CSV)
    df["dt"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["day"] = df["dt"].dt.tz_convert("Asia/Kolkata").dt.normalize().dt.tz_localize(None)
    df["sgn"] = df["sentiment"].map(_signed)
    df = df.dropna(subset=["day"])

    nidx = _load_close("^NSEI")
    if nidx is not None:
        nidx.index = pd.to_datetime(nidx.index)

    # per (symbol, day) aggregate
    daily = (df.groupby(["symbol", "day"])
               .agg(n=("uuid", "nunique"), sgn=("sgn", "mean"))
               .reset_index())

    events = []
    for sym, g in daily.groupby("symbol"):
        g = g.sort_values("day").set_index("day")
        for day, row in g.iterrows():
            # COMPOUND over the trailing window
            lo = day - pd.Timedelta(days=window_days - 1)
            w = g.loc[lo:day]
            comp_tone = float(np.nansum(w["sgn"] * w["n"]))   # intensity-weighted compounded tone
            intensity = int(w["n"].sum())
            car = _fwd_car(sym, day, nidx)
            if car is None:
                continue
            events.append({"sym": sym, "day": str(day.date()), "n": int(row["n"]),
                           "sgn": float(row["sgn"]) if not math.isnan(row["sgn"]) else 0.0,
                           "comp_tone": comp_tone, "intensity": intensity,
                           "pos": car["pos"], **{f"car{h}": car[str(h)] for h in HORIZONS}})

    if not events:
        raise SystemExit("no events with forward prices — fetch older news or refresh prices")
    E = pd.DataFrame(events)

    def stat(sub):
        if len(sub) < 8:
            return None
        return {h: {"car": round(float(sub[f"car{h}"].mean()), 2),
                    "p_pos": round(float((sub[f"car{h}"] > 0).mean()) * 100),
                    "n": int(len(sub))} for h in HORIZONS}

    # ---- single-article tone buckets ----
    single = {
        "positive (sgn>+0.10)": stat(E[E["sgn"] > 0.10]),
        "neutral (|sgn|<=0.10)": stat(E[E["sgn"].abs() <= 0.10]),
        "negative (sgn<-0.10)": stat(E[E["sgn"] < -0.10]),
    }
    # ---- compounding buckets: compounded tone sign x intensity ----
    hi_int = E["intensity"] >= max(2, int(E["intensity"].median()))
    comp = {
        "compound +tone, HIGH intensity": stat(E[(E["comp_tone"] > 0.10) & hi_int]),
        "compound +tone, low intensity":  stat(E[(E["comp_tone"] > 0.10) & ~hi_int]),
        "compound -tone, HIGH intensity": stat(E[(E["comp_tone"] < -0.10) & hi_int]),
        "compound -tone, low intensity":  stat(E[(E["comp_tone"] < -0.10) & ~hi_int]),
    }
    # ---- information coefficient: corr(signal, fwd CAR) ----
    ic = {}
    for sig in ("sgn", "comp_tone", "intensity"):
        ic[sig] = {str(h): (round(float(E[sig].corr(E[f"car{h}"])), 3)
                            if E[sig].nunique() > 2 else None) for h in HORIZONS}

    model = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "source": "marketaux", "window_days": window_days,
        "n_events": int(len(E)), "n_symbols": int(E["sym"].nunique()),
        "date_span": [E["day"].min(), E["day"].max()],
        "single_tone": single, "compounding": comp, "information_coefficient": ic,
    }
    with open(OUT, "w") as fh:
        json.dump(model, fh, indent=2)

    if verbose:
        print(f"\nnews->reaction study: {len(E)} events, {E['sym'].nunique()} symbols, "
              f"{E['day'].min()}..{E['day'].max()}  (window {window_days}d)")
        def show(title, d):
            print(f"\n{title}")
            for k, v in d.items():
                if not v:
                    print(f"  {k:34s}  (thin)")
                    continue
                c20 = v[20]
                print(f"  {k:34s}  +20d CAR {c20['car']:+5.2f}%  P>0 {c20['p_pos']:3d}%  n={c20['n']}")
        show("SINGLE-DAY tone vs +20d market-adjusted return:", single)
        show("COMPOUNDING (tone x intensity) vs +20d:", comp)
        print("\nInformation coefficient  corr(signal, fwd CAR):")
        for sig, hs in ic.items():
            print(f"  {sig:11s} " + "  ".join(f"+{h}d {hs[str(h)]}" for h in HORIZONS))
        print(f"\nsaved -> {OUT}")
    return model


if __name__ == "__main__":
    build()
