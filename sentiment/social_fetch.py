"""
Retail / social sentiment — the "user trends" half (PRD §8.1 social feeds).

StockTwits has a public per-symbol stream; Reddit needs app credentials (PRAW).
Both are best-effort and rate-limited, so each has a synthetic fallback. Social
maps to the SAME NewsItem shape with channel="social", then flows through the same
scorer/LLM and aggregates into buzz + sentiment features (sentiment_features.py).

Caveats worth remembering: retail social is noisy, easily astroturfed/pumped, and
US-skewed (StockTwits Indian coverage is thin). Treat social as a low-weight
secondary signal, never a sole trigger.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .schema import NewsItem


def _us_symbol(ticker: str) -> str:
    return ticker.split(".")[0]  # StockTwits uses bare symbols


def fetch_stocktwits(ticker: str, max_items: int = 30) -> list[NewsItem]:
    """Best-effort StockTwits stream. Returns [] on any failure (caller falls back)."""
    import json
    import urllib.request

    sym = _us_symbol(ticker)
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{sym}.json"
    instrument = ("NSE:" + sym) if ticker.endswith(".NS") else sym
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read().decode())
        items = []
        for m in data.get("messages", [])[:max_items]:
            created = m.get("created_at", "")
            try:
                ts = datetime.fromisoformat(created.replace("Z", "+00:00"))
            except Exception:  # noqa: BLE001
                ts = datetime.now(timezone.utc)
            items.append(NewsItem(
                instrument=instrument, title=m.get("body", "")[:200],
                body=m.get("body", ""), source="stocktwits",
                url="", published_at=ts, channel="social"))
        return items
    except Exception:  # noqa: BLE001
        return []


def fetch_reddit(ticker: str, subreddits=("IndianStockMarket", "stocks"), limit=30):
    """Requires PRAW + a Reddit app (client id/secret). Returns [] if unconfigured."""
    cid, csec = __import__("os").getenv("REDDIT_CLIENT_ID"), __import__("os").getenv("REDDIT_CLIENT_SECRET")
    if not (cid and csec):
        return []
    try:
        import praw  # pip install praw
        reddit = praw.Reddit(client_id=cid, client_secret=csec,
                            user_agent="tradite-sentiment/0.1")
        sym = _us_symbol(ticker)
        instrument = ("NSE:" + sym) if ticker.endswith(".NS") else sym
        items = []
        for sub in subreddits:
            for post in reddit.subreddit(sub).search(sym, limit=limit // len(subreddits)):
                items.append(NewsItem(
                    instrument=instrument, title=post.title, body=post.selftext[:1000],
                    source=f"reddit/{sub}", url="",
                    published_at=datetime.fromtimestamp(post.created_utc, tz=timezone.utc),
                    channel="social"))
        return items
    except Exception:  # noqa: BLE001
        return []


def synthetic_social(ticker: str, n: int = 12) -> list[NewsItem]:
    sym = _us_symbol(ticker)
    instrument = ("NSE:" + sym) if ticker.endswith(".NS") else sym
    msgs = [
        "loading up, this breaks out soon 🚀", "weak hands selling, I'm holding",
        "results looked solid ngl", "overvalued imo, trimming here",
        "support held, bullish", "dead money for months", "FII buying picking up",
        "management guidance was reassuring", "chart looks toppy, careful",
        "adding on dips", "buyback is a good signal", "meh, sideways chop",
    ]
    now = datetime.now(timezone.utc)
    return [NewsItem(instrument=instrument, title=m, body=m, source="synthetic-social",
                     url="", published_at=now, channel="social") for m in msgs[:n]]


def fetch_social(ticker: str) -> list[NewsItem]:
    """Try StockTwits then Reddit; synthesise if both are empty so demos run."""
    items = fetch_stocktwits(ticker) + fetch_reddit(ticker)
    return items or synthetic_social(ticker)
