"""
NewsAugmentedFewShotMiner — few-shot examples that pair NEWS context with
price reactions and forward outcomes.

Each example has the structure the LLM actually needs to reason causally:

  [date]  EVENT: "Infosys cuts FY23 guidance — BFSI client-spend slowdown"
          SETUP:  1Y=-22%  3M=-10%  vol=29%  near 52w-lo
          DAY-OF: -8.5%   |   20d LATER: +11.3% ✅
          LESSON: post-guidance-cut panic in Indian IT is typically overdone;
                  stock recovers within 3 weeks once initial selling exhausts

  [date]  EVENT: "RBI keeps repo rate at 6.5%; no rate cut signal"
          SETUP:  HDFCBANK -9% YoY  near 52w-lo  vol=22%
          DAY-OF: +2.3%   |   20d LATER: +11.5% ✅
          LESSON: rate-hold removes NIM compression fear; banks rally on relief

TODAY (NSE:INFY)
  LIVE NEWS:   "Infosys Q1 deal wins weak; management cautious on H2"
  SETUP:       1Y=-26%  3M=-11%  vol=33%  4% above 52w-lo
  CLOSEST ANALOGUE: [2022-10-14]  (guidance-cut panic → 90% recovery base rate)
  PREDICTION:  High probability of 20d recovery if selling continues to exhaust

Strategy:
  1. Find the K most price-similar historical dates (existing cosine logic).
  2. For dates < RECENT_DAYS: pull RSS items published within ±3 days and
     score them; take the highest-confidence headline as the "EVENT".
  3. For older dates: ask the DO router to narrate the likely news event from
     the price pattern + date  (it has training knowledge of Indian equity history).
  4. Fetch today's live news for each ticker via RSS.
  5. Ask the router to match today's news to the closest historical example and
     state the prediction.
  6. Stitch everything into one ready-to-inject prompt block.
"""
from __future__ import annotations

import json
import time
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── tunables ──────────────────────────────────────────────────────────────────
FWD_DAYS    = 20     # forward return window
K_NEAREST   = 3      # analogues per stock
MIN_ROWS    = 300    # min history to mine
RECENT_DAYS = 90     # within this many days → try live RSS for news context

# ── LLM router call ───────────────────────────────────────────────────────────

def _router_call(system: str, user: str, max_tokens: int = 500) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(base_url=config.DO_BASE_URL, api_key=config.DO_KEY)
        resp = client.chat.completions.create(
            model=config.DO_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"[LLM error: {e}]"


# ══════════════════════════════════════════════════════════════════════════════
# Price helpers (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

def _load_close(ticker: str, years: int = 8) -> pd.Series | None:
    csv = config.DATA_DIR / f"{ticker}__yfinance.csv"

    def _read(path: Path) -> pd.Series | None:
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            sample = df.iloc[0, 0]
            try:
                float(sample)
            except (TypeError, ValueError):
                df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
                df.columns = [c[0].lower() for c in df.columns]
            col = next((c for c in ("Close", "close") if c in df.columns), None)
            if col is None:
                return None
            s = df[col].dropna()
            if not isinstance(s.index, pd.DatetimeIndex):
                s.index = pd.to_datetime(s.index, errors="coerce")
                s = s[s.index.notna()]
            return s.sort_index()
        except Exception:
            return None

    if csv.exists():
        s = _read(csv)
        if s is not None and len(s) >= MIN_ROWS:
            return s

    try:
        import yfinance as yf
        df = yf.download(ticker, period=f"{years}y", auto_adjust=True, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        df.to_csv(csv)
        col = next((c for c in ("Close", "close") if c in df.columns), None)
        if col is None:
            return None
        s = df[col].dropna()
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index, errors="coerce")
            s = s[s.index.notna()]
        return s.sort_index()
    except Exception:
        return None


def _state_vec(close: pd.Series, i: int) -> np.ndarray | None:
    if i < 252:
        return None
    now = float(close.iloc[i])
    feats = [(now / float(close.iloc[i - w]) - 1) * 100 for w in (5, 21, 63, 252)]
    hi52  = float(close.iloc[i - 252: i + 1].max())
    lo52  = float(close.iloc[i - 252: i + 1].min())
    feats += [(now / hi52 - 1) * 100, (now / lo52 - 1) * 100]
    vol   = float(close.iloc[max(0, i - 30): i + 1].pct_change().std() * (252 ** 0.5) * 100)
    feats.append(vol)
    v = np.array(feats, dtype=float)
    return v if np.all(np.isfinite(v)) else None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def _find_analogues(close: pd.Series, k: int = K_NEAREST,
                    fwd: int = FWD_DAYS) -> list[dict[str, Any]]:
    n = len(close)
    cur = _state_vec(close, n - 1)
    if cur is None:
        return []
    candidates = []
    for i in range(252, n - fwd - 1):
        v = _state_vec(close, i)
        if v is not None:
            candidates.append((_cosine(cur, v), i))
    candidates.sort(reverse=True)
    results, seen_months = [], set()
    for sim, i in candidates:
        m = str(close.index[i].date())[:7]
        if m in seen_months:
            continue
        seen_months.add(m)
        fwd_ret = (float(close.iloc[i + fwd]) / float(close.iloc[i]) - 1) * 100
        v = _state_vec(close, i)
        results.append({
            "date": str(close.index[i].date()),
            "sim": round(sim, 3),
            "y1": round(float(v[3]), 1), "m3": round(float(v[2]), 1),
            "m1": round(float(v[1]), 1), "vol": round(float(v[6]), 1),
            "pct_hi": round(float(v[4]), 1), "pct_lo": round(float(v[5]), 1),
            "fwd_ret": round(fwd_ret, 1),
        })
        if len(results) >= k:
            break
    return results


# ══════════════════════════════════════════════════════════════════════════════
# News helpers
# ══════════════════════════════════════════════════════════════════════════════

def _import_fetch_news():
    """Import fetch_news whether we're called as a module or standalone."""
    try:
        from .news_fetch import fetch_news
        return fetch_news
    except ImportError:
        from news_fetch import fetch_news  # type: ignore[no-redef]
        return fetch_news


def _fetch_live_news(ticker: str, max_items: int = 8) -> list[str]:
    """Fetch current headlines from RSS.  Returns list of title strings."""
    try:
        fetch_news = _import_fetch_news()
        items = fetch_news(ticker, max_items=max_items)
        return [it.title for it in items]
    except Exception:
        return []


def _fetch_news_around_date(ticker: str, date_str: str,
                             window_days: int = 4) -> list[str]:
    """
    Try to pull RSS headlines that were published within window_days of date_str.
    Only feasible for dates within RECENT_DAYS; older dates return [].
    """
    try:
        target = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        cutoff = datetime.now(timezone.utc) - timedelta(days=RECENT_DAYS)
        if target < cutoff:
            return []   # too old for live RSS

        fetch_news = _import_fetch_news()
        items = fetch_news(ticker, max_items=30)
        lo = target - timedelta(days=window_days)
        hi = target + timedelta(days=window_days)
        nearby = [
            it.title for it in items
            if hasattr(it, "published_at") and lo <= it.published_at <= hi
        ]
        return nearby[:5]
    except Exception:
        return []


def _narrate_historical_event(ticker: str, analogue: dict[str, Any]) -> str:
    """
    Ask the router to narrate the most likely news event that drove this
    price pattern.  The LLM has training knowledge of Indian equity history
    through mid-2025 so can describe macro/corporate events for dated patterns.
    Returns a one-sentence event description.
    """
    sym  = ticker.replace(".NS", "")
    date = analogue["date"]
    system = (
        "You are a financial historian specialising in Indian equities (NSE). "
        "Given a stock, a date, and its price pattern at that date, describe in "
        "ONE concise sentence the most likely news or fundamental event that "
        "caused this price pattern. Be specific to the actual period if you know "
        "it (earnings, RBI decision, US macro, sector event, etc.). "
        "If unsure, describe the most plausible event type. "
        "Do NOT invent specific numbers. Output only the one-sentence description."
    )
    user = (
        f"Stock: {sym} (NSE India)\n"
        f"Date: {date}\n"
        f"Price pattern at that date:\n"
        f"  1Y return: {analogue['y1']:+.1f}%\n"
        f"  3M return: {analogue['m3']:+.1f}%\n"
        f"  1M return: {analogue['m1']:+.1f}%\n"
        f"  Annualised volatility: {analogue['vol']:.1f}%\n"
        f"  Distance from 52w high: {analogue['pct_hi']:+.1f}%\n"
        f"  Distance from 52w low:  {analogue['pct_lo']:+.1f}%\n"
        f"  What happened in the following {FWD_DAYS} days: "
        f"stock moved {analogue['fwd_ret']:+.1f}%\n\n"
        f"What was the most likely news/fundamental event driving this pattern?"
    )
    return _router_call(system, user, max_tokens=120).strip().strip('"')


def _match_news_to_analogues(ticker: str, live_headlines: list[str],
                              analogues_with_events: list[dict[str, Any]]) -> str:
    """
    Ask the router: given today's news and the historical examples (each with
    their event narrative + outcome), which analogue is most similar to today,
    and what does it predict for the next 20 days?
    Returns a concise prediction paragraph.
    """
    if not live_headlines or not analogues_with_events:
        return ""

    sym = ticker.replace(".NS", "")
    examples_text = "\n".join(
        f"  [{a['date']}] EVENT: {a.get('event', 'unknown')}\n"
        f"    Price setup: 1Y={a['y1']:+.1f}% 3M={a['m3']:+.1f}% vol={a['vol']:.1f}%\n"
        f"    Outcome {FWD_DAYS}d later: {a['fwd_ret']:+.1f}% "
        f"{'✅' if a['fwd_ret'] > 0 else '🔴'}"
        for a in analogues_with_events
    )
    news_text = "\n".join(f"  - {h}" for h in live_headlines[:6])

    system = (
        "You are a quantitative analyst. Given today's news headlines for a stock "
        "and a set of historical analogues (each with its event narrative and outcome), "
        "identify which analogue is most relevant to today's news and state a "
        "concrete 20-day prediction with reasoning. Be specific and brief (3-4 sentences). "
        "Do not repeat the headlines verbatim."
    )
    user = (
        f"Stock: {sym}\n\n"
        f"TODAY'S NEWS:\n{news_text}\n\n"
        f"HISTORICAL ANALOGUES:\n{examples_text}\n\n"
        "Which historical event is most similar to today's news? "
        f"What does it predict for the next {FWD_DAYS} days?"
    )
    return _router_call(system, user, max_tokens=200).strip()


def _sector_cross_signals(series_map: dict[str, pd.Series],
                           threshold_pct: float = 5.0,
                           window: int = 5,
                           min_events: int = 5) -> list[str]:
    aligned = pd.DataFrame(series_map).dropna()
    if aligned.empty or len(aligned) < 60:
        return []
    weekly = aligned.pct_change(window) * 100
    snippets: list[tuple[float, str]] = []
    tickers = list(series_map.keys())
    for leader in tickers:
        for follower in tickers:
            if leader == follower:
                continue
            mask = weekly[leader] <= -threshold_pct
            n = int(mask.sum())
            if n < min_events:
                continue
            events = weekly[follower][mask]
            rate = float((events <= -(threshold_pct / 2)).mean())
            if rate < 0.5:
                continue
            sl = leader.replace(".NS", "")
            sf = follower.replace(".NS", "")
            snippets.append((rate,
                f"When {sl} falls >{threshold_pct:.0f}% in a week → "
                f"{sf} also falls >{threshold_pct/2:.0f}%+ same week: "
                f"{rate*100:.0f}% of cases (n={n}, avg {float(events.mean()):+.1f}%)"))
    snippets.sort(reverse=True)
    return [s for _, s in snippets[:6]]


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

class NewsAugmentedFewShotMiner:
    """
    Builds few-shot examples that pair NEWS EVENT with PRICE REACTION and
    FORWARD OUTCOME — giving the decision LLM causal grounding, not just
    numeric pattern matching.
    """

    def __init__(self, fwd_days: int = FWD_DAYS, k: int = K_NEAREST):
        self.fwd_days = fwd_days
        self.k = k

    def mine(self, tickers: list[str]) -> str:
        blocks: list[str] = []
        series_map: dict[str, pd.Series] = {}

        for ticker in tickers:
            sym = ticker.replace(".NS", "")
            close = _load_close(ticker, years=8)
            if close is None or len(close) < MIN_ROWS:
                blocks.append(f"{sym}: insufficient history")
                continue
            series_map[ticker] = close

            # ── 1. Find price analogues ──────────────────────────────────
            analogues = _find_analogues(close, k=self.k, fwd=self.fwd_days)
            if not analogues:
                blocks.append(f"{sym}: no analogues found")
                continue

            # ── 2. Attach news context to each analogue ──────────────────
            for a in analogues:
                # try live RSS for recent dates first
                rss_headlines = _fetch_news_around_date(ticker, a["date"])
                if rss_headlines:
                    a["event"] = rss_headlines[0]   # best headline
                    a["event_source"] = "rss"
                else:
                    # ask LLM to narrate from price pattern + date
                    a["event"] = _narrate_historical_event(ticker, a)
                    a["event_source"] = "llm-inferred"
                time.sleep(0.2)   # gentle rate limit on LLM calls

            # ── 3. Fetch today's live news ───────────────────────────────
            live_headlines = _fetch_live_news(ticker, max_items=8)

            # ── 4. LLM: match today's news to closest historical example ─
            prediction = _match_news_to_analogues(ticker, live_headlines, analogues)

            # ── 5. Format block ──────────────────────────────────────────
            n = len(close)
            cur = _state_vec(close, n - 1)
            cur_line = (
                f"{sym} — current setup: "
                f"1Y={cur[3]:+.1f}%  3M={cur[2]:+.1f}%  1W={cur[0]:+.1f}%  "
                f"vol={cur[6]:.1f}%  {cur[4]:+.1f}% from 52w-hi"
            )

            analogue_lines = []
            for a in analogues:
                icon   = "✅" if a["fwd_ret"] > 0 else "🔴"
                src    = "*" if a["event_source"] == "llm-inferred" else ""
                analogue_lines.append(
                    f"  [{a['date']}]  sim={a['sim']:.2f}\n"
                    f"    EVENT{src}: {a['event']}\n"
                    f"    SETUP: 1Y={a['y1']:+.1f}%  3M={a['m3']:+.1f}%  vol={a['vol']:.1f}%\n"
                    f"    {self.fwd_days}d OUTCOME: {a['fwd_ret']:+.1f}%  {icon}"
                )

            today_news_lines = (
                ["  TODAY'S NEWS:"] + [f"    - {h}" for h in live_headlines[:5]]
                if live_headlines else ["  TODAY'S NEWS: (no live feed — weekend/off-hours)"]
            )

            prediction_lines = (
                [f"  PREDICTION: {prediction}"] if prediction else []
            )

            block = "\n".join(
                [cur_line, "  HISTORICAL ANALOGUES (news → price → outcome):"]
                + analogue_lines
                + today_news_lines
                + prediction_lines
            )
            blocks.append(block)

        # ── Sector cross-signals ─────────────────────────────────────────────
        cross = _sector_cross_signals(series_map)
        cross_block = ""
        if cross:
            cross_block = (
                "\nSECTOR CROSS-SIGNALS (when sector leader falls, follower follows):\n"
                + "\n".join(f"  {s}" for s in cross)
            )

        footer = (
            "\n* event marked with * is LLM-inferred from price pattern + date "
            "(training knowledge); unmarked events are from live RSS.\n"
            "INSTRUCTION: Use the EVENT→OUTCOME pairs as causal few-shot evidence. "
            "Today's news should be compared to the historical events to determine "
            "whether the current setup resembles a recovery or continuation. "
            "Cross-signals tell you which stocks to watch as leading indicators."
        )

        header = (
            f"[NewsAugmentedFewShotMiner]  "
            f"fwd={self.fwd_days}d  k={self.k}  "
            f"analogues carry news context\n"
        )
        return header + "\n\n".join(blocks) + cross_block + footer


# keep original price-only miner available for fast/offline use
class FewShotMiner:
    """Price-only analogue miner (no LLM calls, fast).  Use when offline."""

    def __init__(self, fwd_days: int = FWD_DAYS, k: int = K_NEAREST):
        self.fwd_days = fwd_days
        self.k = k

    def mine(self, tickers: list[str]) -> str:
        blocks, series_map = [], {}
        for ticker in tickers:
            close = _load_close(ticker, years=8)
            if close is None or len(close) < MIN_ROWS:
                blocks.append(f"{ticker}: insufficient history"); continue
            series_map[ticker] = close
            sym = ticker.replace(".NS", "")
            analogues = _find_analogues(close, k=self.k, fwd=self.fwd_days)
            if not analogues:
                blocks.append(f"{sym}: no analogues"); continue
            n, cur = len(close), _state_vec(close, len(close) - 1)
            cur_line = (
                f"{sym} — today: 1Y={cur[3]:+.1f}% 3M={cur[2]:+.1f}% "
                f"1W={cur[0]:+.1f}% vol={cur[6]:.1f}%"
            )
            rows = [
                f"  [{a['date']}] sim={a['sim']:.2f}  "
                f"1Y={a['y1']:+.1f}% 3M={a['m3']:+.1f}%  "
                f"→ {self.fwd_days}d: {a['fwd_ret']:+.1f}% "
                f"{'✅' if a['fwd_ret']>0 else '🔴'}"
                for a in analogues
            ]
            # recovery base rate
            all_a = _find_analogues(close, k=10, fwd=self.fwd_days)
            if all_a:
                rec = sum(1 for a in all_a if a["fwd_ret"] > 0)
                avg = sum(a["fwd_ret"] for a in all_a) / len(all_a)
                rows.append(f"  Base rate: {rec}/{len(all_a)} recovered  avg {avg:+.1f}%")
            blocks.append("\n".join([cur_line] + rows))

        cross = _sector_cross_signals(series_map)
        cross_block = ("\nSECTOR CROSS-SIGNALS:\n" + "\n".join(f"  {s}" for s in cross)) if cross else ""
        return "[FewShotMiner — price only]\n" + "\n\n".join(blocks) + cross_block
