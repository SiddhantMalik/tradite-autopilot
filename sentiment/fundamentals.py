"""
Fundamentals & valuation layer (Layer 7) for the multi-agent decision.

Free data via yfinance `Ticker.info` — valuation (P/E, fwd P/E, P/B, PEG),
quality (ROE, margins, debt, liquidity), growth (revenue/earnings) and income
(dividend yield, market cap). For Indian (.NS) names yfinance fundamentals are
partial, so EVERY field is guarded and missing values render as 'n/a' rather
than crashing the pipeline.

Why this layer exists: before it, the decision saw only price/momentum + news
headlines — it literally could not reason about whether a stock is cheap,
expensive, or a good business. An expert never buys on charts alone. The block
also adds a cheap relative read: each name's P/E vs the median P/E of the
shortlist, so the model can say "cheaper / in line / richer than peers".

Point-in-time note: `Ticker.info` reflects the LATEST fundamentals, so this
layer is for the LIVE "what should I buy now" decision only. Do NOT join it into
a historical backtest (it would leak future fundamentals). Cached per UTC day.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from statistics import median

import config

CACHE_DIR = config.DATA_DIR / "fundamentals"


# ── formatting helpers (all None-safe) ──────────────────────────────────────
def _first(info: dict, *keys):
    for k in keys:
        v = info.get(k)
        if v is not None:
            return v
    return None


def _pct(x) -> str:
    """For values stored as a fraction (ROE 0.31 -> '31%')."""
    return f"{x*100:.0f}%" if isinstance(x, (int, float)) else "n/a"


def _num(x, nd: int = 1) -> str:
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "n/a"


def _yield(x) -> str:
    """dividendYield is a fraction in some yfinance versions (0.026 = 2.6%) and an
    already-percent number in others (0.69 = 0.69%). Disambiguate by magnitude:
    a value < 0.2 can only be a fraction (0.2 as a fraction = 20% yield, implausible
    for these names); >= 0.2 is already a percent. Reject implausible >20% as bad data."""
    if not isinstance(x, (int, float)) or x <= 0:
        return "n/a"
    v = x * 100 if x < 0.2 else x
    return f"{v:.1f}%" if v <= 20 else "n/a"


def _crore(x) -> str:
    """Indian market-cap convention: ₹ crore (1 cr = 1e7)."""
    return f"₹{x/1e7:,.0f} cr" if isinstance(x, (int, float)) and x else "n/a"


# ── data fetch ──────────────────────────────────────────────────────────────
def fetch_fundamentals(ticker: str) -> dict:
    """Return a normalised fundamentals dict for one ticker (cached per UTC day)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    cache = CACHE_DIR / f"{ticker}_{day}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except Exception:  # noqa: BLE001
            pass

    out: dict = {"ticker": ticker}
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
    except Exception as e:  # noqa: BLE001
        out["error"] = f"yfinance unavailable: {e}"
        return out

    out.update({
        "name":           _first(info, "longName", "shortName") or ticker,
        "sector":         _first(info, "sector"),
        "industry":       _first(info, "industry"),
        "pe":             _first(info, "trailingPE"),
        "fwd_pe":         _first(info, "forwardPE"),
        "pb":             _first(info, "priceToBook"),
        "peg":            _first(info, "trailingPegRatio", "pegRatio"),
        "roe":            _first(info, "returnOnEquity"),
        "net_margin":     _first(info, "profitMargins"),
        "debt_to_equity": _first(info, "debtToEquity"),   # yfinance gives the % form
        "current_ratio":  _first(info, "currentRatio"),
        "rev_growth":     _first(info, "revenueGrowth"),
        "earn_growth":    _first(info, "earningsGrowth", "earningsQuarterlyGrowth"),
        "div_yield":      _first(info, "dividendYield"),
        "market_cap":     _first(info, "marketCap"),
        "price":          _first(info, "currentPrice", "regularMarketPrice", "previousClose"),
        "high_52w":       _first(info, "fiftyTwoWeekHigh"),
    })
    try:
        cache.write_text(json.dumps(out, default=str))
    except Exception:  # noqa: BLE001
        pass
    return out


# ── agent ───────────────────────────────────────────────────────────────────
class FundamentalsAgent:
    """Layer 7 — valuation & quality. Returns a text block for the fused prompt."""

    def run(self, tickers: list[str]) -> str:
        rows = [fetch_fundamentals(t) for t in tickers]
        pes = [r["pe"] for r in rows
               if isinstance(r.get("pe"), (int, float)) and r["pe"] > 0]
        med_pe = median(pes) if pes else None

        blocks: list[str] = []
        for r in rows:
            if r.get("error"):
                blocks.append(f"{r['ticker']}: fundamentals n/a ({r['error']})")
                continue
            inst = "NSE:" + r["ticker"].replace(".NS", "")
            sect = " / ".join(x for x in [r.get("sector"), r.get("industry")] if x) or "sector n/a"

            rel = ""
            if med_pe and isinstance(r.get("pe"), (int, float)) and r["pe"] > 0:
                ratio = r["pe"] / med_pe
                tag = ("CHEAPER than" if ratio < 0.9 else
                       "RICHER than"  if ratio > 1.1 else "in line with")
                rel = f"  | P/E {tag} shortlist median ({_num(med_pe)})"

            dd = ""
            if (isinstance(r.get("price"), (int, float))
                    and isinstance(r.get("high_52w"), (int, float)) and r["high_52w"]):
                dd = f"  | {(r['price']/r['high_52w']-1)*100:+.0f}% vs 52w high"

            blocks.append(
                f"{inst} ({r.get('name')}, {sect}):\n"
                f"  Valuation: P/E {_num(r.get('pe'))} (fwd {_num(r.get('fwd_pe'))}) | "
                f"P/B {_num(r.get('pb'))} | PEG {_num(r.get('peg'))}{rel}\n"
                f"  Quality:   ROE {_pct(r.get('roe'))} | net margin {_pct(r.get('net_margin'))} | "
                f"D/E {_num(r.get('debt_to_equity'))}% | current ratio {_num(r.get('current_ratio'))}\n"
                f"  Growth:    revenue {_pct(r.get('rev_growth'))} YoY | "
                f"earnings {_pct(r.get('earn_growth'))} YoY\n"
                f"  Income:    div yield {_yield(r.get('div_yield'))} | mkt cap {_crore(r.get('market_cap'))}{dd}"
            )

        head = (f"[FundamentalsAgent]  shortlist median P/E = "
                f"{_num(med_pe) if med_pe else 'n/a'}  "
                f"(use for relative-value reads; cite it in rationales)\n")
        return head + "\n\n".join(blocks)
