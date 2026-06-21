"""
Layer 9 — VALUATION VERDICT: "is this worth buying at the CURRENT price?"

This is the deliberate, fact-based counterweight to the momentum screener. The
screener ranks by recent return, so it surfaces stocks that have already run to
their highs; this module ignores momentum and asks the value question instead:

  Given today's price, is the stock CHEAP / FAIR / EXPENSIVE, is it a QUALITY
  business, and is the price EXTENDED (chasing a high) or at a reasonable base?

It produces a transparent, auditable scorecard (every point is shown with its
reason), a verdict (WORTH BUYING / FAIR / HOLD / AVOID), and a heuristic
"reasonable buy-below" price so you can see whether today's price already offers
a margin of safety or you should wait.

Honesty: valuation from free yfinance data is approximate and the buy-below price
is a PEG~1 / peer-anchored heuristic, NOT an intrinsic-value model. It is a
disciplined frame for judgement, not a price target. Quality gates that don't
apply to banks/financials (D/E, current ratio) are skipped for them.
"""
from __future__ import annotations

import os
import numpy as np

from .fundamentals import fetch_fundamentals
from .base_rates import _load_close, _rsi_series

GSEC_YIELD = 7.0  # approx Indian 10y G-sec; a fixed assumption, not a live feed.
MARKET_PE = float(os.getenv("TRADITE_MARKET_PE", "23"))  # ~Nifty median P/E — the cheapness anchor


def _num(x, nd=1):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) and np.isfinite(x) else "n/a"


def value_verdict(ticker: str, peer_pe_median: float | None = None) -> dict:
    f = fetch_fundamentals(ticker)
    if f.get("error"):
        return {"ticker": ticker, "error": f["error"]}
    s = _load_close(ticker)

    pe = f.get("pe"); fwd_pe = f.get("fwd_pe"); pb = f.get("pb"); peg = f.get("peg")
    roe = f.get("roe"); margin = f.get("net_margin"); dte = f.get("debt_to_equity")
    rev_g = f.get("rev_growth"); earn_g = f.get("earn_growth")
    sector = (f.get("sector") or "").lower()
    is_financial = any(k in sector for k in ("financial", "bank", "insurance"))

    price = f.get("price")
    if not isinstance(price, (int, float)) and s is not None and len(s):
        price = float(s.iloc[-1])
    if not isinstance(price, (int, float)):
        return {"ticker": ticker, "error": "no current price"}

    # price position vs 52w range, 200-DMA, RSI
    pct_from_hi = pct_from_lo = vs_dma = rsi = float("nan")
    if s is not None and len(s) >= 220:
        hi = float(s.tail(252).max()); lo = float(s.tail(252).min())
        dma200 = float(s.tail(200).mean())
        pct_from_hi = (price / hi - 1) * 100 if hi else float("nan")
        pct_from_lo = (price / lo - 1) * 100 if lo else float("nan")
        vs_dma = (price / dma200 - 1) * 100 if dma200 else float("nan")
        rsi = float(_rsi_series(s).iloc[-1])

    earn_yield = (100.0 / pe) if isinstance(pe, (int, float)) and pe > 0 else float("nan")

    card: list[tuple[str, float, str]] = []
    anchor = float(peer_pe_median) if peer_pe_median else MARKET_PE   # P/E baseline to judge cheapness
    val = qual = pos = 0.0

    def add(bucket: str, name: str, pts: float, detail: str):
        nonlocal val, qual, pos
        card.append((name, pts, detail))
        if bucket == "v": val += pts
        elif bucket == "q": qual += pts
        else: pos += pts

    # ---- VALUATION (price vs worth) — this GATES the verdict ----
    if isinstance(pe, (int, float)) and pe > 0:
        r = pe / anchor
        if r < 0.6:    add("v", "P/E far below market", 2, f"P/E {pe:.0f} vs mkt {anchor:.0f}")
        elif r < 0.85: add("v", "P/E below market", 1, f"P/E {pe:.0f} vs mkt {anchor:.0f}")
        elif r > 1.6:  add("v", "P/E far above market", -2, f"P/E {pe:.0f} vs mkt {anchor:.0f}")
        elif r > 1.2:  add("v", "P/E above market", -1, f"P/E {pe:.0f} vs mkt {anchor:.0f}")
    if isinstance(peg, (int, float)) and peg > 0:
        if peg < 1:     add("v", "PEG < 1 (cheap for growth)", 1.5, f"PEG {peg:.1f}")
        elif peg > 2.5: add("v", "PEG > 2.5 (dear vs growth)", -1.5, f"PEG {peg:.1f}")
        elif peg > 2:   add("v", "PEG > 2", -0.5, f"PEG {peg:.1f}")
    if np.isfinite(earn_yield):
        sp = earn_yield - GSEC_YIELD
        if sp > 2:     add("v", "earnings yield > G-sec+2%", 1, f"EY {earn_yield:.1f}%")
        elif sp < -3:  add("v", "earnings yield far below G-sec", -1.5, f"EY {earn_yield:.1f}%")
        elif sp < 0:   add("v", "earnings yield below G-sec", -0.5, f"EY {earn_yield:.1f}%")
    if isinstance(pb, (int, float)) and isinstance(roe, (int, float)):
        if pb < 3 and roe > 0.15:   add("v", "low P/B w/ solid ROE", 1, f"P/B {pb:.1f}")
        elif pb > 6 and roe < 0.20: add("v", "high P/B not backed by ROE", -1, f"P/B {pb:.1f}")

    # ---- QUALITY (capped so a great business can't look 'cheap') ----
    if isinstance(roe, (int, float)):
        if roe > 0.20:   add("q", "ROE > 20% (excellent)", 1.5, f"{roe*100:.0f}%")
        elif roe > 0.12: add("q", "ROE 12–20% (good)", 0.75, f"{roe*100:.0f}%")
        elif roe < 0.08: add("q", "ROE < 8% (weak)", -1, f"{roe*100:.0f}%")
    if isinstance(margin, (int, float)) and margin > 0.15:
        add("q", "net margin > 15%", 0.5, f"{margin*100:.0f}%")
    if not is_financial and isinstance(dte, (int, float)):
        if dte < 50:    add("q", "low debt (D/E < 50%)", 0.5, f"D/E {dte:.0f}%")
        elif dte > 150: add("q", "high debt (D/E > 150%)", -1, f"D/E {dte:.0f}%")
    if isinstance(earn_g, (int, float)):
        if earn_g > 0.12: add("q", "earnings growth > 12%", 0.5, f"{earn_g*100:.0f}%")
        elif earn_g < 0:  add("q", "earnings shrinking", -1, f"{earn_g*100:.0f}%")

    # ---- PRICE POSITION (don't chase highs) ----
    if np.isfinite(pct_from_hi):
        if pct_from_hi > -5:           add("p", "at/near 52w HIGH (extended)", -1.5, f"{pct_from_hi:+.0f}% vs 52wH")
        elif -40 < pct_from_hi < -12:  add("p", "reasonable pullback off high", 1, f"{pct_from_hi:+.0f}% vs 52wH")
    if np.isfinite(rsi) and rsi > 70:
        add("p", "RSI > 70 (overbought)", -0.5, f"RSI {rsi:.0f}")
    if np.isfinite(pct_from_hi) and pct_from_hi < -50 and isinstance(earn_g, (int, float)) and earn_g < 0:
        add("p", "down >50% & earnings falling (value trap)", -1.5, f"{pct_from_hi:+.0f}%")

    qual = min(qual, 2.5)                 # a great business is NOT a buy at any price
    score = round(val + qual + pos, 2)

    # VALUATION-GATED verdict: price is the gate, quality/position the tiebreaker
    expensive = (np.isfinite(earn_yield) and earn_yield < 2.5) or \
                (isinstance(pe, (int, float)) and pe > 0 and pe > 1.6 * anchor)
    extended = np.isfinite(pct_from_hi) and pct_from_hi > -3
    if expensive:
        verdict = ("AVOID — expensive at this price" if (extended or val <= -2)
                   else "HOLD / WAIT — quality but expensive (wait for a pullback)")
    elif val >= 2 and score >= 5 and not extended:
        verdict = "WORTH BUYING — cheap & quality at this price"
    elif val >= 0.5 and score >= 2.5:
        verdict = "FAIR — reasonably priced; accumulate on dips"
    elif val <= -1.5 or score <= -1:
        verdict = "AVOID — expensive / chasing / weak"
    else:
        verdict = "HOLD / NEUTRAL — fairly priced, no clear edge"

    # heuristic 'reasonable buy-below' = fair P/E / current P/E × price, where fair P/E is the
    # MEDIAN of three anchors: a quality-justified multiple (higher ROE earns a higher multiple),
    # the peer median, and a PEG~1 growth multiple. Using the median (not the min) avoids being
    # absurdly harsh on high-ROE compounders, which PEG~1 alone was.
    buy_below = float("nan")
    if isinstance(pe, (int, float)) and pe > 0:
        anchors = []
        if isinstance(roe, (int, float)):
            anchors.append(min(max(12 + 55 * roe, 10), 40))     # quality-justified P/E
        if peer_pe_median:
            anchors.append(float(peer_pe_median))
        if isinstance(earn_g, (int, float)) and earn_g > 0:
            anchors.append(min(max(earn_g * 100, 12), 35))      # PEG~1 anchor
        if anchors:
            buy_below = price * float(np.median(anchors)) / pe

    return {
        "ticker": ticker, "price": price, "verdict": verdict, "score": score,
        "pe": pe, "fwd_pe": fwd_pe, "pb": pb, "peg": peg, "roe": roe,
        "earn_yield": earn_yield, "pct_from_hi": pct_from_hi, "rsi": rsi,
        "is_financial": is_financial, "buy_below": buy_below, "scorecard": card,
        "name": f.get("name"), "sector": f.get("sector"),
    }


def _median_pe(verdicts: list[dict]) -> float | None:
    pes = [v["pe"] for v in verdicts if isinstance(v.get("pe"), (int, float)) and v["pe"] > 0]
    return float(np.median(pes)) if pes else None


def rank_by_value(tickers: list[str]) -> list[dict]:
    """Two-pass: compute peer median P/E, then score each name relative to it."""
    rough = [value_verdict(t) for t in tickers]
    med = _median_pe([r for r in rough if not r.get("error")])
    scored = [value_verdict(t, peer_pe_median=med) for t in tickers]
    scored = [v for v in scored if not v.get("error")]
    scored.sort(key=lambda v: v["score"], reverse=True)
    return scored


class ValuationAgent:
    """Layer 9 — per-name 'worth buying at the current price?' verdict, value-ranked."""

    def run(self, tickers: list[str]) -> str:
        ranked = rank_by_value(tickers)
        med = _median_pe(ranked)
        lines = [f"[ValuationAgent]  peer median P/E = {_num(med)}  "
                 f"(fact-based 'worth buying at CURRENT price?'; momentum is deliberately ignored)"]
        for v in ranked:
            inst = "NSE:" + v["ticker"].replace(".NS", "")
            bb = (f"  | reasonable buy-below ≈ ₹{v['buy_below']:,.0f}"
                  if np.isfinite(v.get("buy_below", float('nan'))) else "")
            drivers = "; ".join(f"{n} [{d}]" for n, p, d in
                                sorted(v["scorecard"], key=lambda x: x[1], reverse=True) if p > 0)
            negs = [f"{n} [{d}]" for n, p, d in v["scorecard"] if p < 0]
            lines.append(
                f"\n{inst} ({v.get('name')}) — ₹{v['price']:,.0f}  |  score {v['score']:+.1f}\n"
                f"  VERDICT: {v['verdict']}{bb}\n"
                f"  P/E {_num(v.get('pe'))} (fwd {_num(v.get('fwd_pe'))}) | PEG {_num(v.get('peg'))} | "
                f"P/B {_num(v.get('pb'))} | ROE {_num((v.get('roe') or 0)*100,0)}% | "
                f"EY {_num(v.get('earn_yield'))}% | {_num(v.get('pct_from_hi'),0)}% vs 52wH\n"
                f"  + {drivers if drivers else 'none'}\n"
                f"  - {'; '.join(negs) if negs else 'none'}"
            )
        return "\n".join(lines)
