"""
Scheduled news poller with deduplication and market-hours awareness.

Solves three problems with the raw fetch_news() call:
  1. Deduplication  — seen URLs are persisted to disk; only NEW articles are
                      returned so the LLM is never charged for the same item twice.
  2. Market hours   — NSE trades 09:15–15:30 IST (03:45–10:00 UTC). Polling
                      outside those windows (weekends, nights) is skipped unless
                      you pass force=True. Pre-market window starts 30 min early.
  3. Rate limiting  — a configurable delay between per-ticker fetches so Google
                      doesn't soft-block the IP.

Usage — one-shot poll and score:
    python -m sentiment.news_poller                  # all tickers in config.UNIVERSE
    python -m sentiment.news_poller INFY.NS TCS.NS   # specific tickers
    python -m sentiment.news_poller --force           # ignore market hours

Usage — continuous loop (e.g. run in a tmux pane during market hours):
    python -m sentiment.news_poller --loop --interval 900   # poll every 15 min
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import config
from .news_fetch import fetch_news, SYMBOL_QUERY
from .llm_client import SentimentLLM
from .schema import NewsItem, SentimentSignal

# ── Persistence ────────────────────────────────────────────────────────────────
# Seen URLs are stored here so duplicates are skipped across runs.
SEEN_STORE = config.ROOT / "sentiment" / "_seen_urls.json"

# ── NSE market hours (UTC) ─────────────────────────────────────────────────────
# IST = UTC+5:30. NSE: 09:15–15:30 IST = 03:45–10:00 UTC.
# We add a 30-min pre-market buffer and 15-min post-market buffer.
_MARKET_OPEN_UTC  = timedelta(hours=3, minutes=15)   # 03:15 UTC = 08:45 IST
_MARKET_CLOSE_UTC = timedelta(hours=10, minutes=15)  # 10:15 UTC = 15:45 IST
_MARKET_DAYS = {0, 1, 2, 3, 4}  # Mon–Fri (weekday() values)

# ── Rate limiting ──────────────────────────────────────────────────────────────
# Seconds to sleep between fetching different tickers. Keep ≥2s to avoid
# hitting Google's informal rate limit.
INTER_TICKER_DELAY = 3.0  # seconds


def _load_seen() -> dict[str, set[str]]:
    """Load {ticker -> set-of-seen-urls} from disk."""
    if not SEEN_STORE.exists():
        return {}
    try:
        raw = json.loads(SEEN_STORE.read_text())
        return {k: set(v) for k, v in raw.items()}
    except Exception:
        return {}


def _save_seen(seen: dict[str, set[str]]) -> None:
    SEEN_STORE.parent.mkdir(parents=True, exist_ok=True)
    SEEN_STORE.write_text(json.dumps({k: list(v) for k, v in seen.items()}, indent=2))


def is_market_hours(now: datetime | None = None) -> bool:
    """True if the current UTC time falls within the NSE trading window."""
    now = now or datetime.now(timezone.utc)
    if now.weekday() not in _MARKET_DAYS:
        return False
    day_offset = timedelta(hours=now.hour, minutes=now.minute, seconds=now.second)
    return _MARKET_OPEN_UTC <= day_offset <= _MARKET_CLOSE_UTC


def poll_once(
    tickers: list[str] | None = None,
    force: bool = False,
    llm: SentimentLLM | None = None,
    verbose: bool = True,
) -> dict[str, list[SentimentSignal]]:
    """
    Fetch and score NEW articles for each ticker.

    Returns {ticker -> [SentimentSignal]} for articles not seen in previous runs.
    Skips entirely outside market hours unless force=True.
    """
    now = datetime.now(timezone.utc)

    if not force and not is_market_hours(now):
        ist = now + timedelta(hours=5, minutes=30)
        if verbose:
            print(f"[poller] Outside NSE market hours ({ist.strftime('%H:%M IST')}). "
                  "Pass --force to override.")
        return {}

    tickers = tickers or config.UNIVERSE
    seen = _load_seen()
    llm = llm or SentimentLLM()
    results: dict[str, list[SentimentSignal]] = {}

    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(INTER_TICKER_DELAY)

        ticker_seen = seen.setdefault(ticker, set())

        try:
            items = fetch_news(ticker, max_items=25)
        except Exception as e:
            if verbose:
                print(f"[{ticker}] fetch failed: {e}")
            continue

        new_items = [it for it in items if it.url not in ticker_seen]

        if verbose:
            ist = now + timedelta(hours=5, minutes=30)
            print(f"\n[{ticker}]  {len(items)} fetched, "
                  f"{len(new_items)} new  ({ist.strftime('%H:%M IST')})")

        if not new_items:
            continue

        signals = llm.analyze_many(new_items)

        # Mark all fetched URLs as seen (not just new ones) so old articles
        # don't re-appear on the next poll if they stay in the feed.
        ticker_seen.update(it.url for it in items)

        results[ticker] = signals

        if verbose:
            for item, sig in zip(new_items, signals):
                bar = ("▲ LONG " if sig.direction == "long"
                       else "▼ SHORT" if sig.direction == "short" else "─ NEUT ")
                routed = (sig.model_version.split("[")[1].rstrip("]")
                          if "[" in sig.model_version else "")
                via = f"  via {routed}" if routed else ""
                print(f"  {bar} [{sig.sentiment:+.3f}] conf={sig.confidence:.2f}{via}")
                print(f"  {item.title[:72]}")
                if sig.event_tags:
                    print(f"  tags  : {', '.join(sig.event_tags)}")
                print(f"  thesis: {sig.thesis}\n")

    _save_seen(seen)
    return results


def poll_loop(
    tickers: list[str] | None = None,
    interval: int = 900,   # seconds between polls (default 15 min)
    force: bool = False,
) -> None:
    """
    Continuous polling loop. Runs poll_once every `interval` seconds.
    Skips outside market hours (prints next-poll time and sleeps).
    Ctrl-C to stop.
    """
    llm = SentimentLLM()
    print(f"[poller] Starting loop — interval={interval}s, "
          f"tickers={tickers or config.UNIVERSE}")
    try:
        while True:
            poll_once(tickers=tickers, force=force, llm=llm)
            next_poll = datetime.now(timezone.utc) + timedelta(seconds=interval)
            ist = next_poll + timedelta(hours=5, minutes=30)
            print(f"[poller] Next poll at {ist.strftime('%H:%M:%S IST')} "
                  f"(in {interval}s) — Ctrl-C to stop")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[poller] Stopped.")


# ── CLI ────────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]

    loop     = "--loop"     in argv
    force    = "--force"    in argv
    interval = 900
    tickers  = []

    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--interval" and i + 1 < len(argv):
            interval = int(argv[i + 1]); i += 2; continue
        if not a.startswith("--"):
            tickers.append(a)
        i += 1

    tickers = tickers or None   # None → config.UNIVERSE

    if loop:
        poll_loop(tickers=tickers, interval=interval, force=force)
    else:
        poll_once(tickers=tickers, force=force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
