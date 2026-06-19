"""
End-to-end sentiment demo (run from ml_lab/):

    python -m sentiment.run_sentiment INFY.NS
    python -m sentiment.run_sentiment INFY.NS --backend anthropic   # needs API key

Pipeline: fetch news + social  ->  score each into the §8.1 contract  ->  print a
few signals  ->  aggregate into daily ML features (the block that joins price).
Works with zero API keys (heuristic backend + FinBERT/lexicon fallback).
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from .news_fetch import fetch_news, synthetic_news
from .social_fetch import fetch_social
from .llm_client import SentimentLLM
from .sentiment_features import aggregate_daily, FEATURE_COLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", nargs="?", default="INFY.NS")
    ap.add_argument("--backend", default="heuristic",
                    choices=["heuristic", "digitalocean", "openai", "anthropic", "mock"])
    args = ap.parse_args()

    print(f"\n# 1. Fetch news for {args.ticker}")
    news = fetch_news(args.ticker)
    if not news:
        print("  (live RSS empty/blocked -> synthetic headlines)")
        news = synthetic_news(args.ticker)
    print(f"  {len(news)} news items. e.g.: {news[0].title!r}")

    print(f"\n# 2. Fetch social ('user trends')")
    social = fetch_social(args.ticker)
    print(f"  {len(social)} social posts. e.g.: {social[0].title!r}")

    print(f"\n# 3. Score into the §8.1 contract (backend={args.backend})")
    llm = SentimentLLM(backend=args.backend)
    signals = llm.analyze_many(news + social)
    for s in signals[:3]:
        print(json.dumps(s.to_contract(), indent=2)[:500])

    print(f"\n# 4. Aggregate -> daily features that JOIN with price (../features.py)")
    idx = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=10)
    feats = aggregate_daily(signals, idx, lag=1)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(feats.tail(5)[FEATURE_COLS].round(3).to_string())

    avg = sum(s.sentiment for s in signals) / len(signals)
    print(f"\nSummary: {len(signals)} signals, mean sentiment {avg:+.3f}, "
          f"feature columns -> {FEATURE_COLS}")


if __name__ == "__main__":
    main()
