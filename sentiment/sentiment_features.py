"""
Turn a stream of SentimentSignals into daily, per-instrument ML features that
JOIN with the price features in ../features.py — this is where "news trends" and
"user trends" become columns the model can learn from (PRD §19.5: sentiment as a
feature; meta-labeling).

POINT-IN-TIME: a `lag` (default 1 trading day) is applied so a feature at date t
uses only sentiment known strictly before t's bar — no same-bar lookahead. For a
true historical backtest you must feed signals from a point-in-time news archive
(an RSS feed only returns recent items). See RESEARCH_NOTES.md §19.6.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import SentimentSignal

FEATURE_COLS = [
    "sent_score", "news_sent", "social_sent", "sent_dispersion",
    "news_count", "social_count", "social_buzz", "sent_mom_5",
]


def signals_to_frame(signals: list[SentimentSignal]) -> pd.DataFrame:
    rows = []
    for s in signals:
        ts = pd.Timestamp(s.published_at)
        ts = ts.tz_convert(None) if ts.tzinfo else ts
        rows.append({"date": ts.normalize(), "sentiment": float(s.sentiment),
                     "confidence": float(s.confidence), "channel": s.channel})
    return pd.DataFrame(rows)


def _wmean(x):
    w = x["confidence"].clip(0.01, 1.0).values
    return float(np.average(x["sentiment"].values, weights=w))


def aggregate_daily(signals, price_index: pd.DatetimeIndex, lag: int = 1,
                    buzz_window: int = 20, persist: int = 3) -> pd.DataFrame:
    """Return a DataFrame indexed by price_index with FEATURE_COLS."""
    price_index = pd.DatetimeIndex(price_index)
    empty = pd.DataFrame(0.0, index=price_index, columns=FEATURE_COLS)
    if not signals:
        return empty

    df = signals_to_frame(signals)
    if df.empty:
        return empty

    g = df.groupby("date")
    daily = pd.DataFrame(index=sorted(df["date"].unique()))
    daily["sent_score"] = g.apply(_wmean)
    daily["sent_dispersion"] = g["sentiment"].std().fillna(0.0)
    daily["news_count"] = g.apply(lambda x: int((x["channel"] == "news").sum()))
    daily["social_count"] = g.apply(lambda x: int((x["channel"] == "social").sum()))
    daily["news_sent"] = df[df.channel == "news"].groupby("date")["sentiment"].mean()
    daily["social_sent"] = df[df.channel == "social"].groupby("date")["sentiment"].mean()

    out = daily.reindex(price_index)
    # sentiment levels persist a few days; counts are 0 when no items arrived
    smooth = ["sent_score", "news_sent", "social_sent", "sent_dispersion"]
    out[smooth] = out[smooth].ffill(limit=persist)
    out = out.fillna({c: 0.0 for c in FEATURE_COLS})

    total = out["news_count"] + out["social_count"]
    mu = total.rolling(buzz_window, min_periods=5).mean()
    sd = total.rolling(buzz_window, min_periods=5).std()
    out["social_buzz"] = ((total - mu) / (sd + 1e-9)).fillna(0.0)
    out["sent_mom_5"] = (out["sent_score"] - out["sent_score"].shift(5)).fillna(0.0)

    out = out[FEATURE_COLS].shift(lag).fillna(0.0)  # POINT-IN-TIME lag
    return out


def build_sentiment_features(ticker: str, price_index, backend: str = "heuristic",
                             lag: int = 1) -> pd.DataFrame:
    """Fetch live news+social, score them, and aggregate to daily features.
    Live feeds only cover recent dates, so historical rows will be ~0 — fine for a
    live signal, but for backtests feed a point-in-time archive into aggregate_daily()."""
    from .news_fetch import fetch_news, synthetic_news
    from .social_fetch import fetch_social
    from .llm_client import SentimentLLM

    news = fetch_news(ticker) or synthetic_news(ticker)
    social = fetch_social(ticker)
    llm = SentimentLLM(backend=backend)
    signals = llm.analyze_many(news + social)
    return aggregate_daily(signals, price_index, lag=lag)


def illustrative_sentiment_series(price_index, seed: int = 7) -> pd.DataFrame:
    """ILLUSTRATIVE ONLY — a plausible synthetic daily sentiment feature block over
    the full price history, so combine_demo can show the price+sentiment JOIN and
    that models ingest it. This is NOT real sentiment and implies NO edge."""
    price_index = pd.DatetimeIndex(price_index)
    rng = np.random.default_rng(seed)
    n = len(price_index)
    # mild autocorrelated sentiment + sparse news/social arrivals
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = 0.9 * s[i - 1] + rng.normal(0, 0.15)
    s = np.clip(s, -1, 1)
    news_count = rng.poisson(0.6, n).astype(float)
    social_count = rng.poisson(2.0, n).astype(float)
    out = pd.DataFrame(index=price_index)
    out["sent_score"] = s
    out["news_sent"] = np.clip(s + rng.normal(0, 0.1, n), -1, 1)
    out["social_sent"] = np.clip(s + rng.normal(0, 0.2, n), -1, 1)
    out["sent_dispersion"] = np.abs(rng.normal(0.2, 0.1, n))
    out["news_count"] = news_count
    out["social_count"] = social_count
    total = news_count + social_count
    out["social_buzz"] = ((total - total.mean()) / (total.std() + 1e-9))
    out["sent_mom_5"] = pd.Series(s, index=price_index).diff(5).fillna(0.0).values
    return out[FEATURE_COLS].shift(1).fillna(0.0)
