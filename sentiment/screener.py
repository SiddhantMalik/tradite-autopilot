"""
NiftyScreener — reduces the full Nifty 500 universe to the top N candidates
for the multi-agent pipeline.

Flow:
  1. Fetch the Nifty 500 constituent list from NSE (cached locally).
  2. Batch-download 1-year price history via yfinance (one call for all tickers).
  3. Score every stock on risk-adjusted momentum + oversold-bounce signal.
  4. Return top N tickers split across two buckets:
       • Momentum  — trending up, risk-adjusted
       • Oversold  — near 52w low with recovering 1-week momentum

This is pure pandas/numpy — no LLM calls, runs in ~30-60 s for 500 stocks.

Usage (standalone):
    python -m sentiment.screener               # prints top 15
    python -m sentiment.screener --top 20      # prints top 20
"""
from __future__ import annotations

import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config

warnings.filterwarnings("ignore")

# ── tunables ──────────────────────────────────────────────────────────────────
NIFTY500_URL   = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
NIFTY500_CACHE = config.DATA_DIR / "nifty500_constituents.csv"
PRICE_CACHE    = config.DATA_DIR / "screener_prices.parquet"
CACHE_TTL_DAYS = 1        # refresh constituent list after N days
BATCH_SIZE     = 50       # tickers per yfinance batch call
DEFAULT_TOP    = 15       # how many stocks to hand to the pipeline
MOMENTUM_SLOTS = 10       # of the top N, how many are momentum picks
OVERSOLD_SLOTS = 5        # remainder are oversold/contrarian picks

# Weight vector for momentum score: [1W, 1M, 3M, 6M]
MOMENTUM_WEIGHTS = np.array([0.35, 0.30, 0.20, 0.15])


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — constituent list
# ══════════════════════════════════════════════════════════════════════════════

def _load_nifty500() -> list[str]:
    """
    Return list of tickers in 'SYMBOL.NS' format.
    Fetches from NSE if cache is stale or missing; falls back to Nifty 100
    hardcoded list if the URL is unreachable.
    """
    # refresh if cache is missing or older than TTL
    if NIFTY500_CACHE.exists():
        age = datetime.now() - datetime.fromtimestamp(NIFTY500_CACHE.stat().st_mtime)
        if age < timedelta(days=CACHE_TTL_DAYS):
            df = pd.read_csv(NIFTY500_CACHE)
            return [f"{s.strip()}.NS" for s in df["Symbol"].dropna()]

    try:
        # NSE requires a browser-like User-Agent
        import urllib.request
        req = urllib.request.Request(
            NIFTY500_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read().decode("utf-8", errors="replace")
        from io import StringIO
        df = pd.read_csv(StringIO(raw))
        df.to_csv(NIFTY500_CACHE, index=False)
        print(f"[Screener] Nifty 500 list refreshed — {len(df)} stocks")
        return [f"{s.strip()}.NS" for s in df["Symbol"].dropna()]
    except Exception as e:
        print(f"[Screener] Could not fetch Nifty 500 list ({e}). Using fallback Nifty 100.")
        return _nifty100_fallback()


def _nifty100_fallback() -> list[str]:
    """Hardcoded Nifty 100 as a fallback when NSE is unreachable."""
    symbols = [
        "ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK",
        "BAJAJ-AUTO","BAJFINANCE","BAJAJFINSV","BPCL","BHARTIARTL",
        "BRITANNIA","CIPLA","COALINDIA","DIVISLAB","DRREDDY",
        "EICHERMOT","GRASIM","HCLTECH","HDFCBANK","HDFCLIFE",
        "HEROMOTOCO","HINDALCO","HINDUNILVR","ICICIBANK","ITC",
        "INDUSINDBK","INFY","JSWSTEEL","KOTAKBANK","LT",
        "LTIM","M&M","MARUTI","NTPC","NESTLEIND",
        "ONGC","POWERGRID","RELIANCE","SBILIFE","SHRIRAMFIN",
        "SBIN","SUNPHARMA","TCS","TATACONSUM","TATAMOTORS",
        "TATASTEEL","TECHM","TITAN","ULTRACEMCO","WIPRO",
        # Extended Nifty Next 50
        "ABB","ADANIGREEN","ADANITRANS","AMBUJACEM","AUROPHARMA",
        "BANDHANBNK","BERGEPAINT","BIOCON","BOSCHLTD","CANBK",
        "CHOLAFIN","COLPAL","CONCOR","CUMMINSIND","DLF",
        "DMART","FEDERALBNK","GAIL","GODREJCP","HAVELLS",
        "ICICIGI","ICICIPRULI","IDFCFIRSTB","INDHOTEL","INDUSTOWER",
        "IRCTC","JSWENERGY","JUBLFOOD","LICHSGFIN","LODHA",
        "LUPIN","MARICO","MCDOWELL-N","MFSL","MOTHERSON",
        "MUTHOOTFIN","NAUKRI","NMDC","OFSS","PAGEIND",
        "PIIND","PNB","RECLTD","SAIL","SIEMENS",
        "TORNTPHARM","TRENT","TVSMOTOR","UBL","VBL",
    ]
    return [f"{s}.NS" for s in symbols]


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — batch price download
# ══════════════════════════════════════════════════════════════════════════════

def _batch_download(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    """
    Download close prices for all tickers in batches.
    Returns a DataFrame with tickers as columns, dates as index.
    Uses cached parquet when fresh enough.
    """
    import yfinance as yf

    if PRICE_CACHE.exists():
        age = datetime.now() - datetime.fromtimestamp(PRICE_CACHE.stat().st_mtime)
        if age < timedelta(hours=12):
            try:
                df = pd.read_parquet(PRICE_CACHE)
                # check all requested tickers are present
                missing = [t for t in tickers if t not in df.columns]
                if len(missing) == 0:
                    print(f"[Screener] Using cached prices ({len(df.columns)} tickers)")
                    return df
                print(f"[Screener] Cache missing {len(missing)} tickers — refreshing")
            except Exception:
                pass

    print(f"[Screener] Batch-downloading {len(tickers)} tickers …")
    frames = []
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i: i + BATCH_SIZE]
        try:
            raw = yf.download(
                batch,
                period=period,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw.empty:
                continue

            # yfinance returns MultiIndex (metric, ticker) when >1 ticker
            if isinstance(raw.columns, pd.MultiIndex):
                # pull just Close level
                close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw["close"]
            else:
                # single ticker — column is just "Close"
                col = "Close" if "Close" in raw.columns else "close"
                close = raw[[col]].rename(columns={col: batch[0]})

            frames.append(close)
        except Exception as e:
            print(f"  [Screener] Batch {i//BATCH_SIZE + 1} error: {e}")
        time.sleep(0.5)

    if not frames:
        return pd.DataFrame()

    prices = pd.concat(frames, axis=1)
    prices.sort_index(inplace=True)

    try:
        prices.to_parquet(PRICE_CACHE)
        print(f"[Screener] Saved price cache ({len(prices.columns)} tickers)")
    except Exception:
        pass

    return prices


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — scoring
# ══════════════════════════════════════════════════════════════════════════════

def _score(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute scoring metrics for every ticker with enough history.
    Returns a DataFrame indexed by ticker with columns:
        w1, m1, m3, m6 (returns %), hi52_pct, lo52_pct, vol, momentum, risk_adj, oversold
    """
    rows = []
    for ticker in prices.columns:
        s = prices[ticker].dropna()
        if len(s) < 60:          # need at least 3 months
            continue
        n   = len(s)
        now = float(s.iloc[-1])

        def ret(days: int) -> float | None:
            return (now / float(s.iloc[-days]) - 1) * 100 if n >= days else None

        w1  = ret(5)
        m1  = ret(21)
        m3  = ret(63)
        m6  = ret(126)

        if any(v is None for v in (w1, m1, m3, m6)):
            continue

        hi52       = float(s.tail(252).max())
        lo52       = float(s.tail(252).min())
        hi52_pct   = (now / hi52 - 1) * 100      # negative = below high
        lo52_pct   = (now / lo52 - 1) * 100       # positive = above low

        vol = float(s.pct_change().tail(30).std() * (252 ** 0.5) * 100)
        if vol == 0:
            continue

        # Weighted momentum across 4 horizons
        momentum = float(np.dot(MOMENTUM_WEIGHTS, [w1, m1, m3, m6]))

        # Risk-adjusted momentum (Sharpe-like)
        risk_adj = momentum / vol

        # Oversold score: stocks close to 52w low with positive 1W momentum
        # (potential bounce candidates)
        oversold = (w1 > 0) * (-hi52_pct)   # how far below 52w high, rewarded only if 1W positive

        rows.append({
            "ticker": ticker,
            "w1": round(w1, 2), "m1": round(m1, 2),
            "m3": round(m3, 2), "m6": round(m6, 2),
            "hi52_pct": round(hi52_pct, 2),
            "lo52_pct": round(lo52_pct, 2),
            "vol": round(vol, 1),
            "momentum": round(momentum, 3),
            "risk_adj": round(risk_adj, 4),
            "oversold": round(oversold, 2),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("ticker")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

class NiftyScreener:
    """
    Screens the Nifty 500 universe and returns the top N candidates for
    the multi-agent trading pipeline.

    Parameters
    ----------
    top_n       : total candidates to return (default 15)
    momentum_n  : how many of the top_n are momentum picks
    oversold_n  : how many of the top_n are oversold/contrarian picks
    """

    def __init__(
        self,
        top_n: int = DEFAULT_TOP,
        momentum_n: int = MOMENTUM_SLOTS,
        oversold_n: int = OVERSOLD_SLOTS,
    ):
        self.top_n       = top_n
        self.momentum_n  = momentum_n
        self.oversold_n  = oversold_n

    def screen(self, verbose: bool = True) -> list[str]:
        """
        Run the full screening pipeline.
        Returns a list of tickers (e.g. 'INFY.NS') in no particular order.
        """
        # 1. Get universe
        tickers = _load_nifty500()
        if verbose:
            print(f"[Screener] Universe: {len(tickers)} stocks")

        # 2. Price data
        prices = _batch_download(tickers)
        if prices.empty:
            if verbose:
                print("[Screener] No price data — returning config.UNIVERSE fallback")
            return list(config.UNIVERSE)

        # 3. Score
        scores = _score(prices)
        if scores.empty:
            return list(config.UNIVERSE)
        if verbose:
            print(f"[Screener] Scored {len(scores)} stocks with sufficient history")

        # 4. Pick momentum bucket (top risk_adj, exclude very high vol)
        mom_pool = scores[scores["vol"] < 60].copy()   # exclude >60% vol (micro-caps/junk)
        momentum_picks = (
            mom_pool.nlargest(self.momentum_n, "risk_adj").index.tolist()
        )

        # 5. Pick oversold bucket (high oversold score, not already in momentum)
        oversold_pool = scores[~scores.index.isin(momentum_picks)].copy()
        oversold_picks = (
            oversold_pool.nlargest(self.oversold_n, "oversold").index.tolist()
        )

        selected = list(dict.fromkeys(momentum_picks + oversold_picks))  # dedupe, preserve order

        if verbose:
            print(f"\n[Screener] ── Momentum picks ({len(momentum_picks)}) ──")
            for t in momentum_picks:
                r = scores.loc[t]
                print(f"  {t:20s}  1W={r.w1:+.1f}%  1M={r.m1:+.1f}%  3M={r.m3:+.1f}%  "
                      f"vol={r.vol:.0f}%  risk_adj={r.risk_adj:.3f}")
            print(f"\n[Screener] ── Oversold bounce picks ({len(oversold_picks)}) ──")
            for t in oversold_picks:
                r = scores.loc[t]
                print(f"  {t:20s}  1W={r.w1:+.1f}%  from_hi={r.hi52_pct:.1f}%  "
                      f"oversold_score={r.oversold:.1f}")

        return selected

    def scores_df(self) -> pd.DataFrame:
        """Return the full scored DataFrame for inspection."""
        tickers = _load_nifty500()
        prices  = _batch_download(tickers, verbose=False)
        return _score(prices)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    top  = DEFAULT_TOP
    i    = 0
    while i < len(argv):
        if argv[i] == "--top" and i + 1 < len(argv):
            top = int(argv[i + 1]); i += 2; continue
        i += 1

    screener = NiftyScreener(top_n=top,
                             momentum_n=max(1, top - 5),
                             oversold_n=min(5, top))
    picks = screener.screen(verbose=True)
    print(f"\nSelected {len(picks)} stocks for pipeline:")
    for t in picks:
        print(f"  {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
