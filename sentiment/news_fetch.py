"""
Free news ingestion via Google News RSS (no API key).

Returns point-in-time NewsItem objects. For honest backtesting you must persist
these to a point-in-time store keyed by published_at (PRD §9, §19.6) — an RSS
feed only gives you *recent* items, so live use is fine but historical backtests
need an archive you build up over time (or a paid news history provider).
"""
from __future__ import annotations

import urllib.parse
from datetime import datetime, timezone

from .schema import NewsItem

# Map a trading symbol to a human search query. Extend as your universe grows.
SYMBOL_QUERY = {
    "INFY.NS": "Infosys share",
    "RELIANCE.NS": "Reliance Industries share",
    "TCS.NS": "TCS Tata Consultancy share",
    "HDFCBANK.NS": "HDFC Bank share",
    "ICICIBANK.NS": "ICICI Bank share",
}


def _to_instrument(ticker: str) -> str:
    """RELIANCE.NS -> NSE:RELIANCE (the PRD instrument key)."""
    if ticker.endswith(".NS"):
        return "NSE:" + ticker[:-3]
    if ticker.endswith(".BO"):
        return "BSE:" + ticker[:-3]
    return ticker


def fetch_news(ticker: str, max_items: int = 25) -> list[NewsItem]:
    import feedparser

    query = SYMBOL_QUERY.get(ticker, f"{ticker.split('.')[0]} stock NSE")
    url = (
        "https://news.google.com/rss/search?q="
        + urllib.parse.quote(query)
        + "&hl=en-IN&gl=IN&ceid=IN:en"
    )
    feed = feedparser.parse(url)
    instrument = _to_instrument(ticker)

    items, seen = [], set()
    for e in feed.entries[:max_items]:
        title = getattr(e, "title", "").strip()
        if not title or title in seen:
            continue
        seen.add(title)
        if getattr(e, "published_parsed", None):
            published = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
        else:
            published = datetime.now(timezone.utc)
        items.append(
            NewsItem(
                instrument=instrument,
                title=title,
                body=getattr(e, "summary", ""),
                source=getattr(getattr(e, "source", None), "title", "Google News"),
                url=getattr(e, "link", ""),
                published_at=published,
                channel="news",
            )
        )
    return items


def synthetic_news(ticker: str, n: int = 8) -> list[NewsItem]:
    """Offline fallback so the demo always runs."""
    instrument = _to_instrument(ticker)
    samples = [
        ("Q4 profit beats estimates; board raises guidance", "news"),
        ("Brokerage upgrades stock to 'buy' on strong order book", "news"),
        ("Management flags margin pressure from wage hikes", "news"),
        ("Large deal win announced with European client", "news"),
        ("Regulator opens probe into accounting practices", "news"),
        ("Stock slips as IT spending outlook turns cautious", "news"),
        ("Dividend declared, record date set", "news"),
        ("Shares rally on better-than-feared revenue", "news"),
    ]
    now = datetime.now(timezone.utc)
    return [
        NewsItem(instrument=instrument, title=t, body=t, source="synthetic",
                 url="", published_at=now, channel=ch)
        for (t, ch) in samples[:n]
    ]
