"""
Point-in-time RAG grounding for the sentiment LLM (PRD §19.2, §19.6).

LLM weights are frozen at a training cutoff, so we ground each judgement in
retrieved, point-in-time evidence:
  * the document text itself,
  * as-of numeric price context (last close + recent returns *known at publish time*),
  * a few PRIOR related headlines (strictly before this item's published_at).

The hard rule (§19.6 leakage guard): nothing dated at or after the item may enter
the grounding, and we never ask the model "what happens next" — only to judge the
sentiment of the evidence in front of it.
"""
from __future__ import annotations

import pandas as pd

from .schema import NewsItem
from . import market_grounding
from . import portfolio_grounding


def _asof_ts(published_at) -> pd.Timestamp:
    ts = pd.Timestamp(published_at)
    ts = ts.tz_convert(None) if ts.tzinfo else ts
    return ts.normalize()


def price_context_asof(price_df, published_at, lookbacks=(1, 5, 20)) -> dict:
    """Numeric context using only bars on/before the publish date."""
    if price_df is None or len(price_df) == 0 or "close" not in price_df:
        return {}
    asof = _asof_ts(published_at)
    hist = price_df.loc[price_df.index <= asof]
    if len(hist) < max(lookbacks) + 1:
        return {}
    close = hist["close"]
    ctx = {"last_close": round(float(close.iloc[-1]), 2)}
    for k in lookbacks:
        ctx[f"ret_{k}d"] = round(float(close.iloc[-1] / close.iloc[-1 - k] - 1), 4)
    return ctx


def related_headlines(corpus, item: NewsItem, max_n: int = 3) -> list[str]:
    """Prior headlines for the same instrument, strictly before this item."""
    if not corpus:
        return []
    asof = _asof_ts(item.published_at)
    prior = [c for c in corpus
             if c.instrument == item.instrument and _asof_ts(c.published_at) < asof
             and c.title != item.title]
    prior.sort(key=lambda c: c.published_at)
    return [c.title for c in prior[-max_n:]]


def build_grounding(item: NewsItem, price_df=None, corpus=None,
                    portfolio_state=None, sector: str | None = None) -> dict:
    """Assemble the point-in-time grounding payload for one news item.

    Adds market-grounding slices (general, leakage-safe market facts and base rates)
    retrieved from market_grounding.py based on the item's detected event tags, source
    channel, macro content, and as-of price extension — so the LLM grounds its judgement
    in evidence rather than priors.

    If a (point-in-time) portfolio_state is supplied, also injects portfolio-aware slices
    (drawdown, concentration, caps) from portfolio_grounding.py so the LLM tempers
    conviction/horizon to the book. The LLM still never sizes — sizing is downstream.
    """
    document = f"{item.title}. {item.body}".strip()
    price_ctx = price_context_asof(price_df, item.published_at)
    retrieved = market_grounding.retrieve(document, channel=item.channel, price_context=price_ctx)
    return {
        "instrument": item.instrument,
        "published_at": _asof_ts(item.published_at).date().isoformat(),
        "channel": item.channel,
        "document": document,
        "price_context_asof": price_ctx,
        "prior_headlines": related_headlines(corpus, item),
        "event_tags": retrieved["event_tags"],
        "market_grounding": retrieved["lines"],
        "portfolio_summary": portfolio_grounding.summarize(portfolio_state),
        "portfolio_grounding": portfolio_grounding.portfolio_aware_lines(
            portfolio_state, item.instrument, sector),
    }


def grounding_to_prompt(g: dict) -> str:
    """Render grounding as the user-message text for an LLM call."""
    lines = [
        f"Instrument: {g['instrument']}",
        f"As-of date: {g['published_at']}  (judge sentiment as known on this date only)",
        f"Document: {g['document']}",
    ]
    if g.get("price_context_asof"):
        lines.append(f"Price context as-of: {g['price_context_asof']}")
    if g.get("prior_headlines"):
        lines.append("Prior related headlines: " + " | ".join(g["prior_headlines"]))
    if g.get("event_tags"):
        lines.append("Detected event tags: " + ", ".join(g["event_tags"]))
    if g.get("market_grounding"):
        lines.append(
            "Relevant market grounding (general, point-in-time-safe priors — apply them, "
            "do not restate):\n  - " + "\n  - ".join(g["market_grounding"])
        )
    if g.get("portfolio_summary"):
        lines.append(f"Portfolio state (point-in-time): {g['portfolio_summary']}")
    if g.get("portfolio_grounding"):
        lines.append(
            "Portfolio-aware grounding (apply to conviction/horizon; you do NOT size):\n  - "
            + "\n  - ".join(g["portfolio_grounding"])
        )
    return "\n".join(lines)
