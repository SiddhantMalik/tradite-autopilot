"""
Market grounding — the factual knowledge layer for the sentiment LLM (PRD §8.1 / §19.2).

Two products, both derived from `knowledge/MARKET_GROUNDING.md` (the human master sheet):

  1. CORE_PRINCIPLES — a compact, always-on block injected into the system prompt on
     every call. The cardinal rules + confidence rubric that stop the model reasoning
     from generic priors.

  2. retrieve(...) — point-in-time-safe RAG retrieval. Given one news item's text,
     channel, and as-of price context, it returns ONLY the relevant grounding slices
     (matched on detected event tags, source channel, macro content, and price extension)
     so rag.py can inject them into that single call without bloating every prompt.

⚠️ Point-in-time discipline (PRD §19.6): everything here is GENERAL, timeless market
knowledge (mechanics + statistical base rates), never instrument-specific facts about
the future. No live macro levels (repo rate, crude price, FII stance) are embedded —
only structural relationships that are stable across time — so this is leakage-safe to
inject when scoring historical items.

`MARKET_GROUNDING.md` is the source of truth for rationale + citations; this module is the
source of truth for what actually gets injected. Keep them in sync when editing.
"""
from __future__ import annotations

import os
import re

KNOWLEDGE_PATH = os.path.join(os.path.dirname(__file__), "knowledge", "MARKET_GROUNDING.md")

# --------------------------------------------------------------------------- #
# 1. Always-on core (injected verbatim into SYSTEM_PROMPT)
# --------------------------------------------------------------------------- #
CORE_PRINCIPLES = """\
MARKET GROUNDING — ground every judgement in these facts about the Indian market (NSE/BSE); do NOT assume:
1. React to the SURPRISE vs. consensus, not the headline's tone or the absolute level. Expected, pre-announced, or re-reported news has ~0 incremental impact -> sentiment near 0, low confidence.
2. Default to weak. Most single items produce no durable, tradeable move. Start at confidence ~0.30 and raise only on strong evidence. Hard ceiling 0.85.
3. Sentiment alpha decays fast: ~1-2 trading days for liquid large-caps (horizon hours/days); a few days-2 weeks only for genuine earnings/estimate surprises (PEAD), M&A, buyback, or index changes (horizon weeks). Do not assign "weeks" without such a structural catalyst.
4. Net out market & sector beta. A move in line with index x beta is not stock-specific. Macro/flow news (RBI rates, crude, INR, FII/DII, global cues/GIFT Nifty) is SECTOR-level, not a single-stock signal.
5. Costs set a floor: round-trip delivery ~0.25-0.5% (statutory ~0.25% + a fixed DP fee that makes small trades dearer). If the expected net move is below ~0.5%, output direction=neutral / low confidence.
6. Source credibility scales confidence: exchange filing > company release/earnings call > tier-1 media > unverified media > social/rumor. Social-only on a small/illiquid name = manipulation risk -> very low confidence.
7. Use the price context given (ret_1d/5d/20d): if the stock already moved far in the news direction, much is priced in -> reversal risk, lower confidence/shorter horizon. News from a quiet base is cleaner.
8. Mechanical != informational: ex-dividend / bonus / split / rights ex-date price drops are mechanical, NOT bearish. Never short them.
9. Acquirer != target in M&A: the target rises (often +15-30%, floored by the SEBI open-offer price); the acquirer is usually flat-to-negative.
10. Respect structure: non-F&O scrips have daily price bands (2/5/10/20%) that cap/halt the move; F&O scrips have no static band, only a dynamic ~10% operating range relaxed intraday (so they can still move far). ASM/GSM/illiquid names are erratic (100% margins, trade-for-trade, weekly-only) -> lower confidence. Signals near F&O expiry and on RBI/Budget/Fed days are noisier.
CONFIDENCE RUBRIC: raise to 0.6-0.8 only when ALL hold — credible/official source AND genuine novelty or large surprise AND single clean driver AND liquid large-cap AND corroborated by >=2 independent sources. Lower to 0.1-0.3 for social/rumor, stale/expected news, illiquid/small-cap, mixed signals, or mostly-beta moves. Baseline 0.25-0.35. Never exceed 0.85; your own fluency is not evidence."""

# --------------------------------------------------------------------------- #
# 2. Event detection (extends the original 8 tags with researched event types)
# --------------------------------------------------------------------------- #
# Order matters only for readability; detection is membership-based.
EVENT_PATTERNS: dict[str, list[str]] = {
    "earnings_beat":    ["beat", "beats", "tops estimate", "better-than", "better than", "above estimate", "profit jump", "profit rises", "profit surges"],
    "earnings_miss":    ["miss", "misses", "below estimate", "shortfall", "profit falls", "profit drops", "profit declines", "disappoint"],
    "guidance_raise":   ["raises guidance", "raise guidance", "upbeat outlook", "raises outlook", "hikes guidance", "raises forecast", "upgrades guidance"],
    "guidance_cut":     ["cuts guidance", "lowers outlook", "profit warning", "warns", "lowers guidance", "cuts forecast", "weak outlook"],
    "rating_change":    ["upgrade", "downgrade", "rated", "target price", "initiates coverage", "raises target", "cuts target", "reiterates", "outperform", "underperform"],
    "deal_win":         ["deal win", "order win", "bags order", "wins deal", "wins contract", "secures order", "new order", "order worth", "contract worth", "bags contract"],
    "legal_regulatory": ["probe", "lawsuit", "regulator", "fraud", "sebi", "investigation", "raid", "penalty", "show cause", "ed summons", "tax demand", "ban", "bans", "banned"],
    "dividend_action":  ["dividend", "interim dividend", "final dividend", "special dividend", "record date"],
    "buyback":          ["buyback", "buy-back", "share repurchase", "repurchase", "tender offer"],
    "bonus_issue":      ["bonus issue", "bonus share", "bonus shares", "1:1 bonus", "issue of bonus"],
    "stock_split":      ["stock split", "share split", "sub-division", "subdivision", "split of shares", "face value split"],
    "index_change":     ["nifty inclusion", "index inclusion", "added to nifty", "included in nifty", "index exclusion", "removed from nifty", "dropped from nifty", "index rejig", "index reshuffle", "msci", "ftse"],
    "block_deal":       ["block deal", "bulk deal", "large trade", "stake acquired", "buys stake"],
    "promoter_sell":    ["promoter sells", "promoter sale", "offer for sale", "ofs", "stake sale", "promoter offloads", "promoter trims", "sells stake"],
    "pledging":         ["pledge", "pledged shares", "pledging", "shares pledged", "invoke pledge"],
    "mna":              ["acquire", "acquisition", "merger", "merges", "takeover", "open offer", "to buy", "buyout", "stake buy", "amalgamation"],
}

# Mechanical ex-date markers (suppress shorts on these — Cardinal Rule 8).
EX_DATE_MARKERS = ["ex-date", "ex date", "ex-dividend", "ex dividend", "ex-bonus", "ex-split", "ex-rights", "record date"]

# Macro / beta content -> not a single-stock signal (Cardinal Rule 4).
MACRO_MARKERS = ["crude", "brent", "oil price", "repo rate", "rbi", "interest rate", "rupee", "inr", "usd/inr", "fii", "fpi", "dii", "fed ", "fomc", "federal reserve", "union budget", "gift nifty", "inflation", "cpi", "monetary policy"]


def _kw_present(kw: str, text: str) -> bool:
    """Word-boundary match so 'ban' doesn't fire inside 'Bank', etc. Handles phrases
    and punctuation (e.g. 'buy-back', '1:1 bonus', 'usd/inr') via non-word lookarounds."""
    return re.search(r"(?<!\w)" + re.escape(kw) + r"(?!\w)", text) is not None


def detect_event_tags(text: str) -> list[str]:
    """Word-boundary event tagging over title+body. Returns a de-duplicated list."""
    t = (text or "").lower()
    return [tag for tag, kws in EVENT_PATTERNS.items() if any(_kw_present(k, t) for k in kws)]


def _rating_direction(text: str) -> str | None:
    t = text.lower()
    up = any(k in t for k in ["upgrade", "raises target", "initiates coverage", "outperform", "buy rating", "overweight"])
    down = any(k in t for k in ["downgrade", "cuts target", "underperform", "sell rating", "underweight", "reduce rating"])
    if up and not down:
        return "up"
    if down and not up:
        return "down"
    return None


# --------------------------------------------------------------------------- #
# 3. Per-tag grounding lines (the retrievable slices) — see §2 of the master sheet
# --------------------------------------------------------------------------- #
EVENT_GROUNDING: dict[str, str] = {
    "earnings_beat":    "earnings_beat -> default long, horizon weeks. PEAD: price drifts in the surprise direction for weeks (India-confirmed significance, NSE 2002-2017; the ~60-day window is the US benchmark). Needs a real surprise vs consensus; a beat that merely matches prior guidance/run-up gives ~0 drift. [High]",
    "earnings_miss":    "earnings_miss -> default short, horizon weeks. Misses are punished harder than beats are rewarded (-3 to -7% on day, then downward drift). Discount if one-off/tax-driven or already beaten down. [High]",
    "guidance_raise":   "guidance_raise -> default long, days-weeks. 'Beat AND raise' is the strongest bullish combo. India firms rarely give formal numeric guidance, so soft commentary = lower confidence. [Medium]",
    "guidance_cut":     "guidance_cut -> default short, days-weeks; can persist if it triggers analyst estimate cuts. Often a larger reaction than a raise. Discount if macro-driven and already known. [Medium]",
    "rating_change":    "rating_change -> upgrade: long, ~+3% day + weak ~30-day drift; downgrade: short, STRONGER and longer (~6-month) drift. Identify which side. First/initiation or big target change raises confidence; 'after the run' lowers it. [Medium-High]",
    "deal_win":         "deal_win -> default long, horizon days. Magnitude scales with deal value vs market cap; new client/geography is stronger. Routine/expected/repeat orders are muted. Evidence thin -> keep confidence modest. [Low]",
    "legal_regulatory": "legal_regulatory -> default short, weeks-months. -5% to -20%+ on disclosure, negative drift while a probe runs. SEBI interim order / ED / fraud allegation raises confidence; sector-wide rule or early show-cause lowers it. [Medium]",
    "dividend_action":  "dividend_action -> weak long, days; dividend INITIATION (+2.7-3.8%) or a surprise special dividend is stronger than a routine increase. WARNING: an ex-date price drop is mechanical, NOT bearish — never short it. [Medium]",
    "buyback":          "buyback -> default long, days-2 weeks. Tender offer (+2.1-2.8% day, India-confirmed) > open-market repurchase. Premium to market and undervaluation raise confidence. [High India]",
    "bonus_issue":      "bonus_issue -> weak long on announcement (+~1.8%); no fundamental change (liquidity/retail signalling). The ex-date halving is mechanical — never short it. [Low-Medium]",
    "stock_split":      "stock_split -> very weak long on announcement (+~0.8%); no fundamental change. The ex-date division is mechanical — never short it. [Low-Medium]",
    "index_change":     "index_change -> INCLUSION: long on announcement + pre-effective run-up, then ~4-7% REVERSAL within 60 days (time-box the trade, exit by effective date). EXCLUSION: short initially, then partial recovery over 60-240 days. [Medium India]",
    "block_deal":       "block_deal -> buy is mildly positive (1-3 days) but heavy pre-deal front-running often prices it in; a marquee buyer at a premium raises confidence. [Low-Medium]",
    "promoter_sell":    "promoter_sell / OFS -> default short, days-weeks; severity scales with discount to market and % of holding sold. A pre-planned OFS already known is muted. [Low-Medium]",
    "pledging":         "pledging -> use as a CONFIDENCE-REDUCER on longs, not a standalone short. Raises crash-tail risk (forced-sale cascade), correlates with weaker performance. [Medium India]",
    "mna":              "M&A -> TARGET/open-offer: long (+15-30%, floored by SEBI open-offer price; little drift as the arb spread closes). ACQUIRER: neutral-to-short, worse if stock-funded. Identify which company the item is about. [High]",
}

# Fallback line used when an event tag has been detected but the text could not be
# resolved to a clean sub-direction.
RATING_LINES = {
    "up":   "rating_change (UPGRADE) -> long; ~+3% on day then a weak ~30-day drift (short-lived). Strongest on first coverage or a large target hike; weak if the stock already ran. [Medium-High]",
    "down": "rating_change (DOWNGRADE) -> short; -4.7% on day and a STRONGER ~6-month drift than upgrades. Strongest as a first sell on a consensus-buy name or target cut >20%. [Medium-High]",
}


# --------------------------------------------------------------------------- #
# 4. Context-triggered grounding (channel, macro content, ex-date, price extension)
# --------------------------------------------------------------------------- #
CHANNEL_GROUNDING = {
    "social": "SOURCE=social/rumor -> lowest credibility; India small/illiquid names are manipulation-prone (Telegram/finfluencer pump-and-dump). Require corroboration by a filing/tier-1 source before confidence > 0.3; a big initial move is often the pump that then reverses. [High]",
    "filing": "SOURCE=exchange/company filing -> highest credibility and novelty, BUT check it is not a re-report of an already-public disclosure (institutions react in real time; repeated info ~ 0 alpha). [High]",
    "news":   None,  # default; no extra line
}

MACRO_LINE = ("MACRO/SECTOR content detected -> this is largely market/sector BETA, not a single-stock signal. "
              "Net out index x beta and score via the sector->driver map (rate cut: banks/NBFC/realty/autos +; crude up: OMC/aviation/tyres/paints -; INR weak: IT/pharma exporters +). "
              "On RBI/Budget/Fed/expiry windows, signals are noisier -> lower confidence. [High direction]")

EX_DATE_LINE = ("EX-DATE/record-date language detected -> any price drop is a MECHANICAL adjustment (dividend/bonus/split/rights), "
                "NOT a bearish event. Do not emit a short purely on the ex-date move. [High]")


def _price_extension_line(price_context: dict | None) -> str | None:
    """Emit a priced-in / reversal-risk note when the stock is already extended.

    Uses only as-of trailing returns (point-in-time safe). Sign-agnostic: the model
    combines it with the news direction it infers.
    """
    if not price_context:
        return None
    r20 = price_context.get("ret_20d")
    r5 = price_context.get("ret_5d")
    mag = max(abs(r20) if r20 is not None else 0.0, abs(r5) if r5 is not None else 0.0)
    if mag >= 0.25:
        return (f"PRICE CONTEXT: stock is strongly extended (ret_5d={r5}, ret_20d={r20}). "
                "If the news agrees with this move it is likely largely priced in -> exhaustion/reversal risk: cut confidence 25-40% and shorten horizon. "
                "If the item merely re-reports the move, expect reversal. [High]")
    if mag >= 0.10:
        return (f"PRICE CONTEXT: stock has already moved (ret_5d={r5}, ret_20d={r20}). "
                "If the news agrees, treat as partly priced in -> trim confidence ~15-25% and shorten horizon. [High]")
    return None


# --------------------------------------------------------------------------- #
# 5. Retrieval entry point
# --------------------------------------------------------------------------- #
def retrieve(text: str, channel: str = "news", price_context: dict | None = None,
             max_event_lines: int = 4) -> dict:
    """Return the grounding slices relevant to ONE news item.

    Args:
        text:          title + body of the item (point-in-time).
        channel:       "news" | "social" | "filing".
        price_context: as-of numeric context from rag.price_context_asof (last_close, ret_*d).
        max_event_lines: cap on event-specific lines (keeps prompts lean).

    Returns:
        {"event_tags": [...], "lines": [...]}  — lines are ready to render into the prompt.
    """
    tags = detect_event_tags(text)
    lines: list[str] = []

    # Event-specific grounding (rating_change resolves up/down when possible).
    for tag in tags[:max_event_lines]:
        if tag == "rating_change":
            d = _rating_direction(text)
            lines.append(RATING_LINES[d] if d else EVENT_GROUNDING["rating_change"])
        else:
            line = EVENT_GROUNDING.get(tag)
            if line:
                lines.append(line)

    # Ex-date mechanical guard (high-value, cheap to always check).
    low = (text or "").lower()
    if any(m in low for m in EX_DATE_MARKERS):
        lines.append(EX_DATE_LINE)

    # Macro/beta content.
    if any(m in low for m in MACRO_MARKERS):
        lines.append(MACRO_LINE)

    # Channel credibility.
    ch_line = CHANNEL_GROUNDING.get(channel)
    if ch_line:
        lines.append(ch_line)

    # Price extension / priced-in.
    pe = _price_extension_line(price_context)
    if pe:
        lines.append(pe)

    return {"event_tags": tags, "lines": lines}


def load_master_sheet() -> str:
    """Return the full human master sheet (for tools/audits; not injected per-call)."""
    try:
        with open(KNOWLEDGE_PATH, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


__all__ = ["CORE_PRINCIPLES", "detect_event_tags", "retrieve", "load_master_sheet",
           "EVENT_GROUNDING", "EVENT_PATTERNS", "KNOWLEDGE_PATH"]
