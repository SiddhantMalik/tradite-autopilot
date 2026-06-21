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

# LEARNED (news_learn.py): ~80% of entity-tagged items are market/sector commentary that carries
# no idiosyncratic forward effect (positive & negative market tone gave the same ~+0.6%/20d).
# Only COMPANY-SPECIFIC news (title names the company) moves the single stock. So we weight the
# per-stock signal by scope: company 1.0, sector/other 0.35, pure index commentary 0.0.
_COMPANY_KW = {
    "RELIANCE": ["reliance", "jio", "ril", "ambani"], "HDFCBANK": ["hdfc"],
    "INFY": ["infosys", "infy"], "TCS": ["tcs", "tata consultancy"], "WIPRO": ["wipro"],
    "HCLTECH": ["hcl"], "TECHM": ["tech mahindra"], "TRENT": ["trent"],
    "VEDL": ["vedanta", "hindustan zinc"], "HINDUNILVR": ["hindustan unilever", "hul", "unilever"],
    "ICICIBANK": ["icici"], "LT": ["larsen", "l&t", "l and t"], "COALINDIA": ["coal india"],
    "ADANIENT": ["adani enterprises", "adani"], "MARUTI": ["maruti", "suzuki"],
    "SBIN": ["sbi", "state bank"], "AXISBANK": ["axis bank"], "ITC": ["itc"],
    "BHARTIARTL": ["bharti", "airtel"], "CIPLA": ["cipla"], "MAHABANK": ["maharashtra"],
    "KOTAKBANK": ["kotak"], "BAJFINANCE": ["bajaj fin"], "M&M": ["mahindra"],
    "SUNPHARMA": ["sun pharma"], "TITAN": ["titan"], "NESTLEIND": ["nestle"],
}
_MARKET_WORDS = ("sensex", "nifty", "dalal street", "d-street", "benchmark index",
                 "market today", "stock market", "share market", "gift nifty")


def _scope(symbol: str, title: str) -> str:
    """company | sector | market — how stock-specific this headline is."""
    t = (title or "").lower()
    sym = symbol.replace(".NS", "").upper()
    kws = _COMPANY_KW.get(sym, [sym.lower()])
    if any(k in t for k in kws):
        return "company"
    if any(w in t for w in _MARKET_WORDS):
        return "market"
    return "sector"


_SCOPE_W = {"company": 1.0, "sector": 0.35, "market": 0.0}

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


# in-memory TTL cache — Marketaux free tier is ~100 req/day, shared by every decide cycle.
# Without this the intraday scheduler (≈22 names × 13 cycles) would exhaust the quota in an hour.
_CACHE: dict[str, tuple[float, dict]] = {}
_TTL_S = float(os.getenv("TRADITE_NEWS_TTL_MIN", "180")) * 60


def news_signal(ticker: str, max_items: int = 8, days: int = 14) -> dict:
    """Recent-news signal for one ticker. Never raises — degrades to {ok:False}. TTL-cached."""
    if not USE_NEWS:
        return {"ok": False, "disabled": True}
    import time
    hit = _CACHE.get(ticker)
    if hit and (time.time() - hit[0]) < _TTL_S:
        return hit[1]
    try:
        items, source = _gather_items(ticker, max_items)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    sc = _get_scorer()
    all_tags: set[str] = set()
    co_tags: set[str] = set()              # company-specific event tags only (drive hard flags)
    co_scores, sec_scores = [], []         # idiosyncratic tone pools
    n_company = n_sector = n_market = 0
    wsum = tot = 0.0
    n, heads = 0, []
    for it in items:
        pub = it["published_at"]
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        if pub < cutoff:
            continue
        text = f"{it['title']}. {it.get('body', '')}"
        s = sc.score_signed(text)
        mxs = it.get("mx_sent")            # blend Marketaux's own score (0-1, 0.5 neutral), light
        if isinstance(mxs, (int, float)):
            s = 0.7 * s + 0.3 * max(-1.0, min(1.0, (float(mxs) - 0.5) * 2))
        tg = detect_event_tags(text)
        all_tags |= set(tg)
        scope = _scope(ticker, it["title"])
        w = _SCOPE_W[scope]
        tot += w * s
        wsum += w
        if scope == "company":
            n_company += 1; co_scores.append(s); co_tags |= set(tg)
        elif scope == "sector":
            n_sector += 1; sec_scores.append(s)
        else:
            n_market += 1
        n += 1
        heads.append({"title": it["title"][:120], "score": round(s, 2), "tags": tg,
                      "at": pub.strftime("%Y-%m-%d"), "scope": scope})

    if n == 0:
        empty = {"ok": True, "n": 0, "net": 0.0, "tags": [], "bearish": False, "bullish": False,
                 "top": [], "source": source, "n_company": 0, "n_sector": 0, "n_market": 0}
        _CACHE[ticker] = (time.time(), empty)
        return empty

    # scope-weighted net (company-dominant; pure market commentary contributes 0)
    net = (tot / wsum) if wsum > 0 else 0.0
    co_net = sum(co_scores) / len(co_scores) if co_scores else 0.0
    sec_net = sum(sec_scores) / len(sec_scores) if sec_scores else 0.0

    bear_tags = sorted(co_tags & BEARISH_TAGS)          # only COMPANY catalysts veto
    bull_tags = sorted(co_tags & BULLISH_TAGS)
    # flags fire on company-specific catalysts/tone; strong broad sector tone is a weaker trigger
    bearish = bool(bear_tags) or (n_company and co_net <= -0.35) or (n_sector >= 2 and sec_net <= -0.55)
    bullish = (bool(bull_tags) or (n_company and co_net >= 0.25)) and net >= 0.10 and not bearish
    # company-specific headlines first in the preview
    heads.sort(key=lambda h: {"company": 0, "sector": 1, "market": 2}[h["scope"]])
    result = {
        "ok": True, "n": n, "net": round(net, 3), "co_net": round(co_net, 3),
        "sec_net": round(sec_net, 3), "n_company": n_company, "n_sector": n_sector,
        "n_market": n_market, "tags": sorted(all_tags), "co_tags": sorted(co_tags),
        "bear_tags": bear_tags, "bull_tags": bull_tags,
        "bearish": bearish, "bullish": bullish, "top": heads[:3], "source": source,
    }
    _CACHE[ticker] = (time.time(), result)
    return result
