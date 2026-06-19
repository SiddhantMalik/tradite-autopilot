"""
Multi-agent trading analyst (PRD §8, §19).

Three specialist agents run in parallel threads, each building one knowledge layer.
Their outputs are fused into a single rich grounding prompt, then sent to the
DO Inference Router for the final portfolio decision.

    BookAgent        — reads books/ directory, extracts principles relevant to
                       the current market conditions (LLM-distilled)
    HistoricalAgent  — loads price CSVs from data_cache/, computes returns,
                       volatility, distance from 52w high/low, recovery rates
    NewsAgent        — runs the existing news-fetch + sentiment pipeline

                 ↓            ↓             ↓
         TradingOrchestrator fuses all three into one prompt
                 ↓
         DO Inference Router → final JSON trading decision

Usage (from ml_lab/):
    python -m sentiment.multi_agent                       # all tickers, ₹1L budget
    python -m sentiment.multi_agent --budget 200000       # ₹2L
    python -m sentiment.multi_agent INFY.NS RELIANCE.NS   # specific tickers
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import config
from .news_fetch import fetch_news, SYMBOL_QUERY
from .llm_client import SentimentLLM
from .market_grounding import CORE_PRINCIPLES
from .portfolio_grounding import PORTFOLIO_PRINCIPLES
from .portfolio_manager import PortfolioAgent, ExecutionAgent, KiteSessionManager

BOOKS_DIR = config.ROOT.parent / "books"

# ── helpers ───────────────────────────────────────────────────────────────────

def _openai_client():
    from openai import OpenAI
    return OpenAI(base_url=config.DO_BASE_URL, api_key=config.DO_KEY)


def _router_call(system: str, user: str, model: str | None = None,
                 max_tokens: int = 1200) -> str:
    client = _openai_client()
    resp = client.chat.completions.create(
        model=model or config.DO_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


# ══════════════════════════════════════════════════════════════════════════════
# Agent 1 — BookAgent
# Reads all markdown files in books/, asks the router to extract the principles
# most relevant to the current set of instruments and market conditions.
# ══════════════════════════════════════════════════════════════════════════════

class BookAgent:
    def run(self, instruments: list[str], market_context: str) -> str:
        """Return a distilled block of book knowledge relevant to today's setup."""
        books = list(BOOKS_DIR.glob("*.md")) + list(BOOKS_DIR.glob("*.txt"))
        if not books:
            return "[BookAgent] No books found in books/ directory."

        corpus = ""
        for b in books:
            try:
                corpus += f"\n\n### {b.stem}\n{b.read_text()[:4000]}"
            except Exception:
                pass

        system = (
            "You are a trading research analyst. You have been given excerpts from "
            "several trading books. Extract and synthesize ONLY the principles that "
            "are directly relevant to the instruments and market context provided. "
            "Be specific and concise — 300 words max. Format as bullet points."
        )
        user = (
            f"Instruments under consideration: {', '.join(instruments)}\n"
            f"Current market context: {market_context}\n\n"
            f"BOOK EXCERPTS:\n{corpus[:8000]}"
        )
        try:
            result = _router_call(system, user, max_tokens=600)
            return f"[BookAgent]\n{result}"
        except Exception as e:
            return f"[BookAgent] Error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# Agent 2 — HistoricalAgent
# Loads price CSVs, computes returns + volatility + 52w stats, then asks the
# router to interpret the patterns and flag historical analogues.
# ══════════════════════════════════════════════════════════════════════════════

class HistoricalAgent:
    def run(self, tickers: list[str]) -> str:
        try:
            import pandas as pd
        except ImportError:
            return "[HistoricalAgent] pandas not installed."

        stats_blocks = []
        for ticker in tickers:
            csv = config.DATA_DIR / f"{ticker}__yfinance.csv"
            if not csv.exists():
                # try to fetch fresh data
                try:
                    import yfinance as yf
                    df = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
                    if df.empty:
                        stats_blocks.append(f"{ticker}: no data")
                        continue
                    # yfinance ≥0.2.x returns MultiIndex columns — flatten before saving
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [col[0].lower() for col in df.columns]
                    else:
                        df.columns = [c.lower() for c in df.columns]
                    df.to_csv(csv)
                except Exception as e:
                    stats_blocks.append(f"{ticker}: fetch failed ({e})")
                    continue
            else:
                df = pd.read_csv(csv, index_col=0, parse_dates=True)
                # yfinance ≥0.2 may have saved a MultiIndex CSV (two header rows).
                # Detect that by checking if the second row looks like ticker names.
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0].lower() for col in df.columns]
                else:
                    # Single-level but values might be strings if saved with bad headers —
                    # re-read with header=[0,1] and flatten when the second row is non-numeric.
                    sample = df.iloc[0, 0] if not df.empty else None
                    try:
                        float(sample)
                    except (TypeError, ValueError):
                        # Second row was a "Ticker" row that pandas read as data; re-read properly
                        df2 = pd.read_csv(csv, header=[0, 1], index_col=0, parse_dates=True)
                        df2.columns = [col[0].lower() for col in df2.columns]
                        df = df2

            # CSVs may have "Close" (yfinance default) or "close" (our saved format)
            close_col = next((c for c in ("Close", "close") if c in df.columns), None)
            if df.empty or close_col is None:
                stats_blocks.append(f"{ticker}: empty data (cols: {list(df.columns)})")
                continue

            close = df[close_col].dropna()
            if len(close) < 20:
                stats_blocks.append(f"{ticker}: insufficient data")
                continue

            now_price  = float(close.iloc[-1].item() if hasattr(close.iloc[-1], 'item') else close.iloc[-1])
            w1_ret     = (float(close.iloc[-1]) / float(close.iloc[-5])  - 1) * 100 if len(close) >= 5  else None
            m1_ret     = (float(close.iloc[-1]) / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else None
            m3_ret     = (float(close.iloc[-1]) / float(close.iloc[-63]) - 1) * 100 if len(close) >= 63 else None
            y1_ret     = (float(close.iloc[-1]) / float(close.iloc[-252]) - 1) * 100 if len(close) >= 252 else None
            hi52       = float(close.tail(252).max())
            lo52       = float(close.tail(252).min())
            pct_from_hi = (now_price / hi52 - 1) * 100
            pct_from_lo = (now_price / lo52 - 1) * 100
            vol_30d    = float(close.pct_change().tail(30).std() * (252**0.5) * 100)

            # Historical recovery: after similar 1-month drops, how often did it recover?
            recovery_stat = ""
            if m1_ret is not None and m1_ret < -8:
                monthly_rets = close.pct_change(21).dropna() * 100
                similar = monthly_rets[monthly_rets < m1_ret * 0.8]
                if len(similar) >= 3:
                    # check if price was higher 20 days after each drop
                    recoveries = 0
                    for idx in similar.index:
                        pos = close.index.get_loc(idx)
                        if pos + 21 < len(close):
                            fwd = (close.iloc[pos + 21] / close.iloc[pos] - 1) * 100
                            if fwd > 0:
                                recoveries += 1
                    total = len(similar)
                    recovery_stat = (f"  Historical recovery (after similar >{abs(m1_ret):.0f}% 1M drop): "
                                     f"{recoveries}/{total} recovered in 20 days "
                                     f"({100*recoveries/total:.0f}% base rate)")

            lines = [
                f"{ticker}:",
                f"  Price ₹{now_price:,.1f}",
                f"  1W: {w1_ret:+.1f}%" if w1_ret is not None else "",
                f"  1M: {m1_ret:+.1f}%" if m1_ret is not None else "",
                f"  3M: {m3_ret:+.1f}%" if m3_ret is not None else "",
                f"  1Y: {y1_ret:+.1f}%" if y1_ret is not None else "",
                f"  52w high ₹{hi52:,.1f} ({pct_from_hi:+.1f}% from high)",
                f"  52w low  ₹{lo52:,.1f} ({pct_from_lo:+.1f}% from low)",
                f"  Ann. vol {vol_30d:.1f}%",
            ]
            if recovery_stat:
                lines.append(recovery_stat)
            stats_blocks.append("\n".join(l for l in lines if l))

        raw_stats = "\n\n".join(stats_blocks)

        system = (
            "You are a quantitative analyst. Given historical price statistics for "
            "several Indian NSE stocks, provide a brief interpretation of each: "
            "trend direction, momentum, proximity to key levels, and any notable "
            "historical patterns. Flag opportunities and risks. Max 300 words."
        )
        try:
            interpretation = _router_call(system, raw_stats, max_tokens=600)
            return f"[HistoricalAgent]\nRAW STATS:\n{raw_stats}\n\nINTERPRETATION:\n{interpretation}"
        except Exception as e:
            return f"[HistoricalAgent]\nRAW STATS:\n{raw_stats}\n\nInterpretation failed: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# Agent 3 — NewsAgent
# Runs the existing news-fetch + sentiment pipeline, returns a text summary.
# ══════════════════════════════════════════════════════════════════════════════

class NewsAgent:
    def run(self, tickers: list[str]) -> str:
        llm = SentimentLLM(cache=True)
        blocks = []
        for ticker in tickers:
            try:
                items = fetch_news(ticker, max_items=int(os.getenv("TRADITE_NEWS_MAX", "10")))
                if not items:
                    blocks.append(f"{ticker}: no news fetched")
                    continue
                sigs = llm.analyze_many(items)
                total_c = sum(s.confidence for s in sigs) or 1
                w_sent  = sum(s.sentiment * s.confidence for s in sigs) / total_c
                longs   = sum(1 for s in sigs if s.direction == "long")
                shorts  = sum(1 for s in sigs if s.direction == "short")

                lines = [f"{ticker}:  weighted_sentiment={w_sent:+.3f}  "
                         f"(▲{longs} ▼{shorts} ─{len(sigs)-longs-shorts})  "
                         f"[{len(sigs)} items]"]
                # Pass the LAW-CALIBRATED detail through to the decision layer — not
                # just the average. The decision model needs each signal's confidence,
                # horizon and event tags (which the cardinal laws shaped) to reason; an
                # average alone washes all of that out and yields vague output.
                top = sorted(sigs, key=lambda s: s.confidence, reverse=True)[:6]
                for s in top:
                    tags = ",".join(s.event_tags[:4]) or "-"
                    lines.append(
                        f"  [{s.direction:7s} sent={s.sentiment:+.2f} conf={s.confidence:.2f} "
                        f"{s.horizon:5s} tags={tags}] {s.thesis[:160]}"
                    )
                blocks.append("\n".join(lines))
            except Exception as e:
                blocks.append(f"{ticker}: news agent error — {e}")
            time.sleep(1)  # gentle rate limit between tickers

        return "[NewsAgent]\n" + "\n\n".join(blocks)


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator — runs all three agents in parallel, fuses output, calls router
# ══════════════════════════════════════════════════════════════════════════════

_DECISION_SYSTEM_BODY = """You are a senior portfolio manager at a quantitative hedge fund \
specialising in Indian equities (NSE). You have been given five layers of analysis:

1. BOOK KNOWLEDGE        — distilled principles from leading trading books
2. HISTORICAL DATA       — quantitative stats and pattern interpretation
3. LIVE NEWS             — sentiment signals from today's news
4. PORTFOLIO STATE       — current holdings, cash, and open positions
5. NEWS-AUGMENTED ANALOGUES — K most-similar past periods, each paired with the \
NEWS EVENT that drove the price pattern AND the actual forward outcome. Each example \
shows: [date] EVENT → PRICE SETUP → 20d OUTCOME. Also includes today's live news \
matched to the closest historical event, and sector cross-signal base rates. \
USE THESE AS CAUSAL FEW-SHOT EVIDENCE: if today's news resembles a historical event \
that preceded recovery, weight long; if it resembles a continuation pattern, skip or \
short. Cross-signals tell you which stocks move together — apply them to followers \
when a sector leader fires.

6. ML SIGNALS — Six quantitative signal families computed from 8 years of price \
history: (a) MARKET REGIME — bull/bear/sideways/volatile label with strategy hint; \
(b) TECHNICALS — RSI-14, MACD histogram cross, Bollinger %B per stock; \
(c) MEAN-REVERSION PROBABILITY — logistic regression P(+5% in 20d) vs base rate — \
a positive EDGE means this setup historically beats the base rate; \
(d) SECTOR RANK — percentile within NSE sector peers on 1M momentum; \
(e) CORRELATION RISK — flag ρ>0.7 pairs so you don't double-count bets; \
(f) VOLATILITY FORECAST — GARCH/EWMA 5d forward vol vs realized. \
USE THESE SIGNALS DIRECTLY: if P(recovery)=72% vs base 48% → overweight; \
if ρ>0.7 between two longs → size down one; if vol EXPANDING → widen stop; \
if BEAR regime → require stronger evidence before any long.

Your task: produce a concrete trading plan for Monday market open in JSON format.

The JSON must have exactly this structure:
{
  "market_view": "<one sentence on overall market bias>",
  "positions": [
    {
      "instrument": "NSE:XXX",
      "direction": "long" | "short" | "skip",
      "allocation_pct": <0-100>,
      "entry_price_hint": "<near current price, or 'market open'>",
      "stop_loss_pct": <1-10>,
      "target_pct": <2-25>,
      "rationale": "<2-3 sentences integrating all three layers>"
    }
  ],
  "risk_notes": "<key risks to watch>",
  "overall_confidence": <0.0-1.0>
}

Rules:
- Total allocation_pct across all long positions must not exceed 100.
- Skip instruments if the evidence is unclear or contradictory.
- Apply the 7-8% hard stop rule from book knowledge.
- Position size inversely proportional to volatility (higher vol = smaller size).
- Never ignore governance or regulatory risk signals from news layer.
- ANTI-VAGUENESS — every "rationale" and "market_view" MUST contain: (a) the number of the \
specific cardinal LAW it relies on (e.g. "Law 4: mostly sector beta"), (b) at least one \
CONCRETE NUMBER copied from the HISTORICAL or ML layer (price ₹, % move, RSI, P(recovery), \
vol), and (c) the dominant news event_tag. A rationale with no law number and no number is \
INVALID — rewrite it before returning. Do not output generic phrases like "cautiously \
optimistic", "monitor closely", or "mixed signals" without the three required specifics.
- CALIBRATION — overall_confidence and every implied conviction obey the confidence rubric: \
baseline ~0.30, HARD CEILING 0.85. Values above 0.85 are forbidden; your own fluency is not \
evidence. If the drivers are mostly index/sector beta or already priced in (large prior \
ret_5d/ret_20d in the news direction), set direction="skip" with a one-line reason.
- Output valid JSON only — no prose outside the JSON block.
"""

# ── The "laws" the user maintains (market_grounding.py / portfolio_grounding.py) ──
# These are the SAME cardinal rules + confidence rubric that govern per-headline
# sentiment scoring. They are injected into the FINAL decision prompt here so the
# trading plan actually obeys them. (Previously DECISION_SYSTEM never saw these, so
# editing the laws changed only the news sub-signals and the decision looked frozen.)
DECISION_LAWS = (
    "═══ NON-NEGOTIABLE MARKET & PORTFOLIO LAWS "
    "(apply to EVERY position; cite the rule number in each rationale) ═══\n"
    + CORE_PRINCIPLES + "\n\n" + PORTFOLIO_PRINCIPLES
)
DECISION_SYSTEM = DECISION_LAWS + "\n\n" + _DECISION_SYSTEM_BODY


# ══════════════════════════════════════════════════════════════════════════════
# Deliberate-then-rank decision (the "think like an expert" path)
#
# A one-shot "JSON only, no prose" call forces shallow answers — you're telling an
# analyst "don't show your work." Instead the strong model first REASONS through
# each name (bull/bear, valuation, catalyst, risk:reward), RANKS the buys, and only
# then emits the machine JSON. The reasoning text is surfaced to the user — that is
# the educated reasoning the JSON alone hid.
# ══════════════════════════════════════════════════════════════════════════════

_ANALYST_BODY = """\
You are a senior portfolio manager at a quantitative hedge fund specialising in Indian \
equities (NSE/BSE), deciding what to BUY right now. You are given seven layers of analysis:
  1 BOOK KNOWLEDGE  2 HISTORICAL PRICE  3 LIVE NEWS SENTIMENT  4 PORTFOLIO STATE
  5 HISTORICAL ANALOGUES  6 ML SIGNALS (regime, RSI/MACD/%B, mean-reversion P, sector rank, correlation, vol)
  7 FUNDAMENTALS & VALUATION (P/E vs peers, PEG, ROE, margins, debt, growth)
  8 MEASURED BASE RATES (non-parametric forward-return histogram for THIS setup, from 8y history)
  9 VALUATION VERDICT (fact-based 'worth buying at the CURRENT price?' scorecard — momentum ignored)

PRIMARY QUESTION — for every name, answer "is this worth buying at the CURRENT price?" anchored on
Layer 9 (valuation verdict) and Layer 7 (fundamentals). This is a VALUE/quality decision, not a
momentum one.

ANTI-MOMENTUM RULE (critical): do NOT recommend a name just because it is rising or near its
52-week high. A stock at/near its 52w high or with RSI>70 is EXTENDED — it needs a genuine
valuation reason (cheap vs peers, low PEG, earnings yield above the ~7% G-sec) to be a buy, not
just price strength. If Layer 9 says AVOID or HOLD/WAIT (expensive/chasing), do NOT make it a long,
no matter how strong its momentum or how positive its news. Prefer Layer 9 WORTH BUYING / FAIR
names that trade with a margin of safety (ideally near/below their buy-below price). State each
name's Layer 9 verdict and buy-below price in its rationale, and whether the current price offers a
margin of safety or you'd WAIT for a lower entry.

REALITY-CHECK RULE: Layer 6 gives a *fitted* ML probability; Layer 8 gives the *measured* base
rate for the same setup. If they DISAGREE, trust Layer 8 and CUT conviction — a fitted edge that
the raw history doesn't show is likely noise (price-only signals rarely survive OOS). Cite the
Layer 8 n and verdict (EDGE+/NO EDGE/EDGE-) in your rationale; never claim an ML edge that Layer 8
calls NO EDGE or EDGE-.

THINK STEP BY STEP. Write a section per candidate, headed "### NSE:XXX", containing:
  • Thesis — one sharp sentence on why this is or isn't a buy now.
  • Bull case / Bear case — the strongest honest version of each.
  • Valuation — cheap or rich? Cite P/E vs the shortlist median, PEG and growth (Layer 7). \
A high P/E is only justified by growth/quality; flag "expensive with no catalyst" as a SKIP.
  • Quality — ROE, margins, debt (Layer 7): is this a good business?
  • Catalyst & window — what makes it move, and over what horizon; name the cardinal LAW # \
that governs that horizon (sentiment decay vs structural catalyst).
  • Key risks — cite the LAW # that applies (beta/sector, already-priced-in, cost floor, liquidity/band).
  • Setup — entry, stop, target and the resulting RISK:REWARD ratio. If R:R < ~1.5:1, SKIP.

Then write "## RANKED BUYS NOW": order the names by conviction; state clearly what you would \
BUY, what to keep on watch, and what to SKIP, and WHY the top pick beats the runner-up. Be \
specific and quantitative — no "cautiously optimistic", "monitor closely" or "mixed signals" \
without numbers. Obey every LAW and the hard 0.85 confidence ceiling.

FINALLY, after the prose, output the machine plan as a single fenced ```json block:
{
  "market_view": "<one sentence, with a number>",
  "positions": [
    {"instrument":"NSE:XXX","direction":"long|short|skip","conviction":<0-0.85>,
     "allocation_pct":<0-100>,"entry_price_hint":"<near price or 'market open'>",
     "stop_loss_pct":<1-10>,"target_pct":<2-25>,"risk_reward":"<e.g. 2.1:1>",
     "rationale":"<MUST cite a LAW #, a number from Layer 2/6/7, and the news event_tag>"}
  ],
  "risk_notes":"<key risks>",
  "overall_confidence":<0-0.85>
}
Rules: total long allocation_pct <= 100; SKIP unclear/contradictory or expensive-no-catalyst names; \
7-8% hard stop; size inversely to volatility; never ignore governance/regulatory risk. JSON valid, fenced."""

ANALYST_SYSTEM = DECISION_LAWS + "\n\n" + _ANALYST_BODY

# Used only if the analyst's own JSON block fails to parse — distil the prose to JSON.
_FORMAT_SYSTEM = (
    "Extract the final trading plan from the analysis into ONE valid JSON object with keys: "
    "market_view, positions[], risk_notes, overall_confidence. Each positions[] item has: "
    "instrument, direction, conviction, allocation_pct, entry_price_hint, stop_loss_pct, "
    "target_pct, risk_reward, rationale. Output ONLY the JSON — no prose, no code fences."
)


def _router_call_resilient(system: str, user: str, models: list[str],
                           max_tokens: int = 4500) -> tuple[str, str]:
    """Call the decision models in order; return (text, model_used) from the first
    that returns non-empty content. Raises only if every model fails."""
    client = _openai_client()
    last_err: Exception | None = None
    for m in models:
        try:
            resp = client.chat.completions.create(
                model=m,
                messages=[{"role": "system", "content": system},
                          {"role": "user",   "content": user}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            txt = resp.choices[0].message.content or ""
            if txt.strip():
                return txt, (resp.model or m)
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(f"all decision models failed ({models}): {last_err}")


def _extract_decision_json(raw: str) -> dict | None:
    """Pull the decision JSON from analyst output — prefer the last ```json fence,
    else the last brace-balanced object."""
    import re
    fences = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    cand = fences[-1] if fences else None
    if cand is None and "{" in raw and "}" in raw:
        cand = raw[raw.find("{"): raw.rfind("}") + 1]
    if not cand:
        return None
    try:
        return json.loads(cand)
    except Exception:  # noqa: BLE001
        return None


def _analysis_text(raw: str) -> str:
    """The human-readable reasoning = everything before the JSON/code-fence tail."""
    idx = raw.find("```json")
    if idx == -1:
        idx = raw.find("```")
    if idx == -1:
        b = raw.rfind("\n{")
        idx = b if b > 0 else len(raw)
    return raw[:idx].strip()


def _clamp_confidence(decision: dict) -> dict:
    """Enforce the 0.85 cardinal-rule ceiling on the final plan in CODE (soft prompt
    rules get ignored). Applies to overall_confidence and each position's conviction."""
    if not isinstance(decision, dict):
        return decision
    oc = decision.get("overall_confidence")
    if isinstance(oc, (int, float)):
        decision["overall_confidence"] = min(0.85, float(oc))
    for p in decision.get("positions", []) or []:
        cv = p.get("conviction")
        if isinstance(cv, (int, float)):
            p["conviction"] = min(0.85, float(cv))
    return decision


class TradingOrchestrator:
    def __init__(
        self,
        tickers: list[str] | None = None,
        budget_inr: float = 100_000,
        execute: bool = False,
        dry_run: bool = True,
        stop_floor: float | None = None,
        target_floor: float | None = None,
        universe_size: int | None = None,
        screen_to: int = 15,
    ):
        self.budget        = budget_inr
        self.execute       = execute
        self.dry_run       = dry_run
        self.stop_floor    = stop_floor
        self.target_floor  = target_floor
        self.universe_size = universe_size   # if set, run screener instead of fixed list
        self.screen_to     = screen_to       # how many stocks to pass to pipeline after screening
        self._kite_session = KiteSessionManager()

        if tickers:
            # explicit list overrides everything
            self.tickers = tickers
        elif universe_size and universe_size > len(config.UNIVERSE):
            # screener will populate self.tickers at run() time
            self.tickers = None
        else:
            self.tickers = list(config.UNIVERSE)

    def run(self, verbose: bool = True, decide: bool = True) -> dict[str, Any]:
        # ── Optional screening pass ───────────────────────────────────────
        if self.tickers is None:
            from .screener import NiftyScreener
            n_mom = max(1, self.screen_to - 5)
            n_ov  = min(5, self.screen_to)
            screener = NiftyScreener(top_n=self.screen_to,
                                     momentum_n=n_mom, oversold_n=n_ov)
            self.tickers = screener.screen(verbose=verbose)
            if verbose:
                print(f"\n[Orchestrator] Screener selected {len(self.tickers)} stocks "
                      f"from Nifty 500 → passing to 5-agent pipeline\n")

        instruments = [
            ("NSE:" + t.replace(".NS", "")) for t in self.tickers
        ]
        market_context = (
            f"Instruments: {', '.join(instruments)}. "
            f"Date: {datetime.now(timezone.utc).strftime('%A %d %b %Y')}. "
            f"Next session: Monday NSE open. Budget: ₹{self.budget:,.0f}."
        )

        results: dict[str, str] = {}
        errors:  dict[str, str] = {}

        # ── Run all three agents in parallel threads ──────────────────────
        def run_agent(name: str, fn):
            try:
                results[name] = fn()
                if verbose:
                    print(f"  ✓ {name} done")
            except Exception as e:
                errors[name]  = str(e)
                results[name] = f"[{name}] ERROR: {e}"
                if verbose:
                    print(f"  ✗ {name} error: {e}")

        book_agent  = BookAgent()
        hist_agent  = HistoricalAgent()
        news_agent  = NewsAgent()
        port_agent  = PortfolioAgent(session=self._kite_session)

        from .fewshot import NewsAugmentedFewShotMiner
        fewshot_miner = NewsAugmentedFewShotMiner()

        from .ml_signals import MLSignalAgent
        ml_agent = MLSignalAgent()

        from .fundamentals import FundamentalsAgent
        fund_agent = FundamentalsAgent()

        from .base_rates import BaseRateAgent
        baserate_agent = BaseRateAgent()

        from .valuation import ValuationAgent
        valuation_agent = ValuationAgent()

        portfolio_state: dict[str, Any] = {}

        def run_portfolio_agent():
            text, state = port_agent.run()
            results["PortfolioAgent"] = text
            portfolio_state.update(state)
            if verbose:
                print("  ✓ PortfolioAgent done")

        threads = [
            threading.Thread(target=run_agent, args=(
                "BookAgent", lambda: book_agent.run(instruments, market_context))),
            threading.Thread(target=run_agent, args=(
                "HistoricalAgent", lambda: hist_agent.run(self.tickers))),
            threading.Thread(target=run_agent, args=(
                "NewsAgent", lambda: news_agent.run(self.tickers))),
            threading.Thread(target=run_portfolio_agent),
            threading.Thread(target=run_agent, args=(
                "FewShotAgent", lambda: fewshot_miner.mine(self.tickers))),
            threading.Thread(target=run_agent, args=(
                "MLSignalAgent", lambda: ml_agent.run(self.tickers))),
            threading.Thread(target=run_agent, args=(
                "FundamentalsAgent", lambda: fund_agent.run(self.tickers))),
            threading.Thread(target=run_agent, args=(
                "BaseRateAgent", lambda: baserate_agent.run(self.tickers))),
            threading.Thread(target=run_agent, args=(
                "ValuationAgent", lambda: valuation_agent.run(self.tickers))),
        ]

        if verbose:
            print("Spawning agents in parallel …")
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # ── Build constraint block from CLI overrides ─────────────────────
        constraint_lines = []
        if self.stop_floor is not None:
            constraint_lines.append(
                f"HARD CONSTRAINT — stop_loss_pct must be >= {self.stop_floor:.1f}% on every long position."
            )
        if self.target_floor is not None:
            constraint_lines.append(
                f"HARD CONSTRAINT — target_pct must be >= {self.target_floor:.1f}% on every long position."
            )
        constraint_block = (
            "\n═══ RISK CONSTRAINTS (CLI overrides — must be respected) ══════\n"
            + "\n".join(constraint_lines)
        ) if constraint_lines else ""

        # ── Fuse into one prompt ──────────────────────────────────────────
        fused_prompt = f"""
BUDGET: ₹{self.budget:,.0f}  |  INSTRUMENTS: {', '.join(instruments)}
DATE: {datetime.now(timezone.utc).strftime('%A %d %b %Y')}  |  NEXT SESSION: Monday NSE open
{constraint_block}
═══ LAYER 1: BOOK KNOWLEDGE ════════════════════════════════════
{results.get('BookAgent', 'unavailable')}

═══ LAYER 2: HISTORICAL PRICE ANALYSIS ════════════════════════
{results.get('HistoricalAgent', 'unavailable')}

═══ LAYER 3: LIVE NEWS SENTIMENT ══════════════════════════════
{results.get('NewsAgent', 'unavailable')}

═══ LAYER 4: CURRENT PORTFOLIO STATE ══════════════════════════
{results.get('PortfolioAgent', 'Kite not connected — assume no existing positions')}

═══ LAYER 5: HISTORICAL ANALOGUES & SECTOR CROSS-SIGNALS ══════
{results.get('FewShotAgent', 'unavailable')}

═══ LAYER 6: ML SIGNALS ═══════════════════════════════════════
{results.get('MLSignalAgent', 'unavailable')}

═══ LAYER 7: FUNDAMENTALS & VALUATION ═════════════════════════
{results.get('FundamentalsAgent', 'unavailable')}

═══ LAYER 8: MEASURED HISTORICAL BASE RATES ═══════════════════
{results.get('BaseRateAgent', 'unavailable')}

═══ LAYER 9: VALUATION VERDICT — WORTH BUYING AT THIS PRICE? ═══
{results.get('ValuationAgent', 'unavailable')}
"""

        # Allow building the full context without the (slow/costly) final LLM call —
        # useful to inspect/cache the fused prompt or run the decision separately.
        if not decide:
            return {
                "agent_outputs":     results,
                "portfolio_state":   portfolio_state,
                "fused_prompt":      fused_prompt,
                "decision_analysis": "",
                "decision":          {},
                "execution_records": [],
            }

        decision_models = [config.DO_DECISION_MODEL] + config.DO_DECISION_FALLBACKS
        if verbose:
            print(f"\nAll agents complete. Deliberating on the decision with "
                  f"{decision_models[0]} (fallbacks: {', '.join(decision_models[1:]) or 'none'}) …\n")

        # ── Final: deliberate (reason) → rank → decision JSON ────────────
        decision_analysis = ""
        decision: dict[str, Any] = {}
        try:
            raw, model_used = _router_call_resilient(
                ANALYST_SYSTEM, fused_prompt, decision_models, max_tokens=4500)
            decision_analysis = _analysis_text(raw)
            decision = _extract_decision_json(raw) or {}
            if not decision:
                # The analyst reasoned but didn't emit clean JSON — distil it.
                j, _ = _router_call_resilient(
                    _FORMAT_SYSTEM, decision_analysis or raw, decision_models, max_tokens=1200)
                decision = _extract_decision_json(j) or {
                    "error": "could not parse decision JSON", "raw": raw[:2000]}
            decision = _clamp_confidence(decision)
            decision.setdefault("_model_used", model_used)
        except Exception as e:  # noqa: BLE001
            decision = {"error": str(e)}

        # ── Optional execution ────────────────────────────────────────────
        execution_records: list[dict] = []
        if self.execute and "positions" in decision:
            mode = "DRY-RUN" if self.dry_run else "⚠️  LIVE"
            if verbose:
                print(f"\n[ExecutionAgent] Running in {mode} mode …")
            exec_agent = ExecutionAgent(
                session=self._kite_session,
                budget_inr=self.budget,
                dry_run=self.dry_run,
            )
            execution_records = exec_agent.execute(decision, portfolio_state)
            if verbose:
                for r in execution_records:
                    status = r.get("status", "?")
                    note   = r.get("note", "")
                    print(f"  [{status:12s}] {r.get('instrument', r.get('error', '?'))} — {note}")

        return {
            "agent_outputs":      results,
            "portfolio_state":    portfolio_state,
            "fused_prompt":       fused_prompt,
            "decision_analysis":  decision_analysis,
            "decision":           decision,
            "execution_records":  execution_records,
        }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]

    budget        = 100_000
    tickers       = []
    execute       = "--execute" in argv
    live          = "--live"    in argv
    dry_run       = not live
    universe_size = None
    screen_to     = 15

    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--budget" and i + 1 < len(argv):
            budget = float(argv[i + 1]); i += 2; continue
        if a == "--universe" and i + 1 < len(argv):
            universe_size = int(argv[i + 1]); i += 2; continue
        if a == "--screen" and i + 1 < len(argv):
            screen_to = int(argv[i + 1]); i += 2; continue
        if not a.startswith("--"):
            tickers.append(a)
        i += 1

    tickers = tickers or None

    # "What should I buy now?" → default to screening Nifty 500 then deep-diving the
    # top N, unless the user passed explicit tickers or their own --universe size.
    if tickers is None and universe_size is None:
        universe_size = 500
        print(f"[multi_agent] No tickers given → screening Nifty 500 → deep-dive top {screen_to}.")

    if execute and live:
        print("⚠️  LIVE mode: real orders will be placed on your Zerodha account.")
        confirm = input("Type 'yes' to proceed: ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            return 1

    orch = TradingOrchestrator(
        tickers=tickers, budget_inr=budget,
        execute=execute, dry_run=dry_run,
        universe_size=universe_size, screen_to=screen_to,
    )
    result = orch.run(verbose=True)

    # The educated reasoning — printed first so you can see HOW it decided,
    # not just the machine plan.
    if result.get("decision_analysis"):
        print("\n" + "═" * 60)
        print("ANALYST REASONING")
        print("═" * 60)
        print(result["decision_analysis"])

    print("\n" + "═" * 60)
    print("FINAL TRADING DECISION", end="")
    mu = result.get("decision", {}).get("_model_used")
    print(f"   [model: {mu}]" if mu else "")
    print("═" * 60)
    print(json.dumps(result["decision"], indent=2))

    # Show execution records if any
    if result.get("execution_records"):
        print(f"\n{'─'*60}")
        mode = "DRY-RUN" if dry_run else "LIVE"
        print(f"EXECUTION RECORDS  [{mode}]")
        print(f"{'─'*60}")
        for r in result["execution_records"]:
            if "error" in r:
                print(f"  ERROR: {r['error']}")
            else:
                print(f"  [{r.get('status','?'):12s}] {r.get('instrument','?'):22s} "
                      f"{r.get('direction',''):5s}  qty={r.get('qty','?')}  "
                      f"≈₹{r.get('price_hint','?')}")
                if r.get("note"):
                    print(f"                    {r['note']}")

    # Show per-position allocation in rupees
    decision = result["decision"]
    if "positions" in decision:
        print(f"\n{'─'*60}")
        print(f"POSITION SIZING  (budget ₹{budget:,.0f})")
        print(f"{'─'*60}")
        for p in decision["positions"]:
            if p.get("direction") in ("long", "short"):
                amt = budget * p.get("allocation_pct", 0) / 100
                print(f"  {p['direction'].upper():5s} {p['instrument']:18s} "
                      f"{p.get('allocation_pct', 0):3.0f}%  =  ₹{amt:,.0f}"
                      f"  stop={p.get('stop_loss_pct', '?')}%  "
                      f"target={p.get('target_pct', '?')}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
