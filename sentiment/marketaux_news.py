"""
Marketaux news source — entity-tagged, timestamped Indian-equity headlines + sentiment.

Better than Google-News RSS for our purpose: each article is mapped to the exact ticker(s)
it's about and carries a sentiment score, with a real publish timestamp. Free tier returns a
few articles per request (~100 req/day) and needs a browser User-Agent (Cloudflare blocks the
default urllib UA with error 1010). Key via env TRADITE_MARKETAUX_KEY. Falls back silently
(returns []) so callers can use Google-News RSS instead.
"""
from __future__ import annotations

import os, json, urllib.request, urllib.error
from datetime import datetime, timezone

KEY = os.getenv("TRADITE_MARKETAUX_KEY", "")
_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
       "Accept": "application/json"}
_BASE = "https://api.marketaux.com/v1/news/all"


def available() -> bool:
    return bool(KEY)


def fetch(symbol: str, limit: int = 3, published_after: str | None = None) -> list[dict]:
    """Return [{symbol, title, published_at(datetime), sentiment(0-1|None), url}] for one ticker.
    Never raises — returns [] on any failure (caller falls back to RSS)."""
    if not KEY:
        return []
    sym = symbol if symbol.endswith((".NS", ".BO")) else symbol + ".NS"
    q = f"symbols={sym}&filter_entities=true&language=en&limit={limit}&api_token={KEY}"
    if published_after:
        q += f"&published_after={published_after}"
    try:
        req = urllib.request.Request(f"{_BASE}?{q}", headers=_UA)
        d = json.loads(urllib.request.urlopen(req, timeout=15).read().decode())
    except Exception:  # noqa: BLE001
        return []
    out = []
    for a in d.get("data", []):
        ents = [e for e in a.get("entities", []) if e.get("symbol") == sym] or a.get("entities", [])
        sent = ents[0].get("sentiment_score") if ents else None
        try:
            pub = datetime.fromisoformat(a.get("published_at", "").replace("Z", "+00:00"))
        except Exception:  # noqa: BLE001
            pub = datetime.now(timezone.utc)
        out.append({"symbol": sym, "title": a.get("title", ""), "published_at": pub,
                    "sentiment": sent, "url": a.get("url", "")})
    return out


def _raw(symbols: str, published_before: str | None, page: int = 1, limit: int = 3) -> dict:
    """Raw API call for a comma-joined symbol string. Returns the parsed JSON ({} on failure)."""
    q = (f"symbols={symbols}&filter_entities=true&language=en&limit={limit}&page={page}"
         f"&api_token={KEY}")
    if published_before:
        q += f"&published_before={published_before}"
    try:
        req = urllib.request.Request(f"{_BASE}?{q}", headers=_UA)
        return json.loads(urllib.request.urlopen(req, timeout=20).read().decode())
    except Exception:  # noqa: BLE001
        return {}


def fetch_history(symbols: list[str], max_requests: int = 40, out_csv: str | None = None,
                  start_before: str | None = None, verbose: bool = True) -> str:
    """Walk backward in time over a basket, one article-entity row per (article,symbol).

    Cursor pagination via published_before = oldest timestamp seen so far (page=N is unreliable
    on the free tier). Persists/append-dedupes to a CSV keyed by (uuid,symbol). Returns CSV path.
    Stays inside `max_requests` to respect the ~100 req/day free-tier quota.
    """
    import csv
    syms = [s if s.endswith((".NS", ".BO")) else s + ".NS" for s in symbols]
    symstr = ",".join(syms)
    want = set(syms)
    out_csv = out_csv or os.path.join(os.path.dirname(__file__), "news_history.csv")

    seen = set()
    rows_existing = 0
    if os.path.exists(out_csv):
        with open(out_csv, newline="") as fh:
            for r in csv.DictReader(fh):
                seen.add((r["uuid"], r["symbol"]))
                rows_existing += 1

    new_rows, cursor, req, added = [], start_before, 0, 0
    while req < max_requests:
        d = _raw(symstr, published_before=cursor)
        req += 1
        data = d.get("data") or []
        if not data:
            break
        oldest = None
        for a in data:
            uuid = a.get("uuid", "")
            pub = a.get("published_at", "")
            ts = pub.replace("Z", "+00:00")
            oldest = min(oldest, ts) if oldest else ts
            for e in a.get("entities", []):
                if e.get("symbol") in want:
                    key = (uuid, e["symbol"])
                    if key in seen:
                        continue
                    seen.add(key)
                    new_rows.append({"uuid": uuid, "symbol": e["symbol"],
                                     "published_at": pub,
                                     "sentiment": e.get("sentiment_score"),
                                     "title": (a.get("title", "") or "").replace("\n", " ")[:200]})
                    added += 1
        # step the cursor just before the oldest article on this page
        if not oldest:
            break
        try:
            cur_dt = datetime.fromisoformat(oldest) - __import__("datetime").timedelta(seconds=1)
            cursor = cur_dt.strftime("%Y-%m-%dT%H:%M:%S")
        except Exception:  # noqa: BLE001
            break

    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["uuid", "symbol", "published_at", "sentiment", "title"])
        if write_header:
            w.writeheader()
        for r in new_rows:
            w.writerow(r)
    if verbose:
        print(f"fetch_history: {req} requests, +{added} new rows "
              f"(total {rows_existing + added}) -> {out_csv}")
    return out_csv
