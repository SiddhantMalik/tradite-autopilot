"""
Tradite sentiment engine (PRD §8.1, §19).

The "news + social + LLM" half of the system. It reads news/filings/social,
scores them, and emits the §8.1 SentimentSignal contract — then aggregates those
signals into daily per-instrument ML features that JOIN with the price features
in ../features.py (see ../combine_demo.py).

Run the demos from the ml_lab/ directory as modules:
    python -m sentiment.run_sentiment INFY.NS

Everything has a zero-dependency, zero-API-key fallback so the pipeline always
runs; plug in FinBERT (transformers) and a real LLM when you want the quality.
"""
from .schema import NewsItem, SentimentSignal  # noqa: F401
