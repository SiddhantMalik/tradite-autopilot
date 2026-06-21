"""
News adapter — makes the autopilot ADAPT to market news (event-driven), free / no API key.

Google News RSS headlines -> lexicon sentiment (FinBERTScorer fallback) + event-tag detection
(market_grounding.detect_event_tags). Produces a per-name signal the autotrader uses to:
  • VETO a buy when a name has a bearish catalyst (regulatory/fraud, earnings miss, guidance cut,
    promoter sell, pledging) or strongly negative net sentiment — even if it's cheap; and
  • SELL a holding that develops such a catalyst (an event-driven exit, on top of stop/target).

Disable with TRADITE_USE_NEWS=false (falls back to pure value). If a DO inference key is present
the heavier LLM path exists elsewhere; here we deliberately use the key-free heuristic so it runs
on the paper container with no secrets.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta

from .market_grounding import detect_event_tags

USE_NEWS = os.getenv("TRADITE_USE_NEWS", "true").lower() != "false"

BEARISH_TAGS = {"legal_regulatory", "earnings_miss", "guidance_cut", "promoter_sell", "pledging"}
BULLISH_TAGS = {"earnings_beat", "deal_win", "guidance_raise", "buyback", "bonus_issue"}

_scorer = None


def _get_scorer():
    global _scorer
    if _scorer is None:
        from .finbert_scorer import FinBERTScorer
        _scorer = FinBERTScorer(use_finbert=False)   # lexicon — no heavy model on the container
    return _scorer


def _gather_items(ticker: str, max_items: int) -> tuple[list[dict], str]:
    """Return (items, source). Each item: {title, body, published_at(aware dt), mx_sent|None}.
    Prefers Marketaux (entity-tagged, dated, sentiment); falls back to Google-News RSS."""
    from . import marketaux_news as mx
    if mx.available():
        rows = mx.fetch(ticker, limit=min(max_items, 3))  # free tier caps ~3/req
        if rows:
            return ([{"title": r["title"], "body": "", "published_at": r["published_at"],
                      "mx_sent": r.get("sentiment")} for r in rows], "marketaux")
    from .news_fetch import fetch_news
    items = fetch_news(ticker, max_items=max_items)
    out = []
    for it in items:
        pub = it.published_at
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        out.append({"title": it.title, "body": getattr(it, "body", ""),
                    "published_at": pub, "mx_sent": None})
    return out, "google_rss"


def news_signal(ticker: str, max_items: int = 8, days: int = 14) -> dict:
    """Recent-news signal for one ticker. Never raises — degrades to {ok:False}."""
    if not USE_NEWS:
        return {"ok": False, "disabled": True}
    try:
        items, source = _gather_items(ticker, max_items)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    sc = _get_scorer()
    tags: set[str] = set()
    tot, n, heads = 0.0, 0, []
    for it in items:
        pub = it["published_at"]
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        if pub < cutoff:
            continue
        text = f"{it['title']}. {it.get('body', '')}"
        s = sc.score_signed(text)
        # blend in Marketaux's own sentiment (0-1, 0.5 neutral) as a light second opinion
        mxs = it.get("mx_sent")
        if isinstance(mxs, (int, float)):
            s = 0.7 * s + 0.3 * max(-1.0, min(1.0, (float(mxs) - 0.5) * 2))
        tg = detect_event_tags(text)
        tags |= set(tg)
        tot += s
        n += 1
        heads.append({"title": it["title"][:120], "score": round(s, 2), "tags": tg,
                      "at": pub.strftime("%Y-%m-%d")})

    if n == 0:
        return {"ok": True, "n": 0, "net": 0.0, "tags": [], "bearish": False,
                "bullish": False, "top": [], "source": source}

    net = tot / n
    bear_tags = sorted(tags & BEARISH_TAGS)
    bull_tags = sorted(tags & BULLISH_TAGS)
    bearish = bool(bear_tags) or net <= -0.35
    bullish = bool(bull_tags) and net >= 0.10 and not bearish
    return {
        "ok": True, "n": n, "net": round(net, 3), "tags": sorted(tags),
        "bear_tags": bear_tags, "bull_tags": bull_tags,
        "bearish": bearish, "bullish": bullish, "top": heads[:3], "source": source,
    }
