"""
Deep analysis — interprets the facts the way an analyst would, modelling the chain:

    news event  →  PUBLIC (retail) reaction  →  INVESTOR (institutional) reaction
               →  expected PRICE PATH over time (empirical, from this stock's own history)
               →  a BUY / SELL TIMING plan for maximum gains

Honesty: nobody can predict exact future prices. So the "price path" here is the EMPIRICAL
conditional path — what this stock actually did, on average, at +1d/+1w/+2w/+1m after setups
like today's (RSI regime × position vs 52-week range). The reaction models encode well-evidenced
market mechanics (post-earnings drift, sentiment decay, beta-netting, mean-reversion) from the
project's market grounding — not a crystal ball.
"""
from __future__ import annotations

import json, os
import numpy as np
from .base_rates import _load_close, _rsi_series, _regime, _position

# measured post-shock behaviour from the event study (event_study.py); empty if not built yet
_EVENT_MODEL = {}
try:
    with open(os.path.join(os.path.dirname(__file__), "event_model.json")) as _fh:
        _EVENT_MODEL = json.load(_fh)
except Exception:  # noqa: BLE001
    _EVENT_MODEL = {}


# event tag -> (bias, horizon, how the move usually behaves)
NEWS_PLAYBOOK = {
    "earnings_beat":   ("bullish", "1–3 weeks", "post-earnings drift (PEAD) — real beats keep drifting up"),
    "earnings_miss":   ("bearish", "1–3 weeks", "downward drift; estimates get cut"),
    "guidance_raise":  ("bullish", "days–weeks", "re-rating as forecasts rise"),
    "guidance_cut":    ("bearish", "days–weeks", "de-rating; institutions trim"),
    "deal_win":        ("bullish", "days", "modest, usually fades unless order book is transformative"),
    "rating_change":   ("mixed",   "1–3 days", "broker upgrades/downgrades fade fast"),
    "legal_regulatory":("bearish", "weeks+",   "sharp drop, slow/uncertain recovery — don't catch it"),
    "mna":             ("bullish", "step-jump", "target gaps toward the offer price; acquirer flat-to-down"),
    "buyback":         ("bullish", "days–2 wks","mild support, stronger if at a premium"),
    "promoter_sell":   ("bearish", "days",      "supply + negative signalling"),
    "pledging":        ("bearish", "days",      "balance-sheet stress signal"),
    "block_deal":      ("mixed",   "days",      "depends who's buying/selling"),
    "index_change":    ("bullish", "to rebalance date", "passive funds must buy on inclusion"),
    "dividend_action": ("neutral", "mechanical", "ex-date drop is mechanical, not bearish"),
    "bonus_issue":     ("neutral", "mechanical", "no fundamental change"),
    "stock_split":     ("neutral", "mechanical", "no fundamental change"),
}


def forward_path(ticker: str, horizons=(1, 5, 10, 20)) -> dict:
    """Empirical cumulative forward return at each horizon, conditional on today's setup."""
    s = _load_close(ticker)
    if s is None or len(s) < 300:
        return {"ok": False}
    rsi = _rsi_series(s)
    hi = s.rolling(252).max(); lo = s.rolling(252).min()
    reg_today = _regime(float(rsi.iloc[-1]))
    pos_today = _position(float(s.iloc[-1]), float(hi.iloc[-1]), float(lo.iloc[-1]))
    reg_hist = rsi.apply(_regime)
    pos_hist = [_position(p, h, l) for p, h, l in zip(s, hi, lo)]
    pos_hist = np.array(pos_hist, dtype=object)
    base_mask = (reg_hist.values == reg_today) & (pos_hist == pos_today)
    out = []
    for h in horizons:
        fwd = s.shift(-h) / s - 1.0
        m = base_mask & fwd.notna().values
        x = fwd.values[m]
        if len(x) >= 25:
            out.append({"d": h, "mean": round(float(np.mean(x)) * 100, 2),
                        "p_pos": round(float(np.mean(x > 0)) * 100), "n": int(len(x))})
        else:  # thin sample → fall back to unconditional
            u = fwd.dropna().values
            out.append({"d": h, "mean": round(float(np.mean(u)) * 100, 2),
                        "p_pos": round(float(np.mean(u > 0)) * 100), "n": int(len(u)), "thin": True})
    return {"ok": True, "pos": pos_today,
            "setup": f"RSI {float(rsi.iloc[-1]):.0f} ({reg_today}), {pos_today}", "path": out}


def interpret(v: dict, ns: dict, path: dict) -> dict:
    """Turn facts into the reaction chain + a buy/sell timing plan."""
    tags = (ns or {}).get("tags", []) if ns and ns.get("ok") else []
    net = (ns or {}).get("net", 0.0) if ns and ns.get("ok") else 0.0
    pfh = v.get("pct_from_hi")
    plays = [{"tag": t, "bias": NEWS_PLAYBOOK[t][0], "window": NEWS_PLAYBOOK[t][1], "note": NEWS_PLAYBOOK[t][2]}
             for t in tags if t in NEWS_PLAYBOOK]

    # NEWS EFFECT
    if plays:
        news_effect = "; ".join(f"{p['tag']} → {p['bias']} ({p['window']}): {p['note']}" for p in plays)
    elif ns and ns.get("ok") and abs(net) >= 0.2:
        news_effect = f"No hard catalyst; general tone {'positive' if net>0 else 'negative'} (net {net:+.2f})."
    else:
        news_effect = "No material news — price action will be driven by valuation + flows, not a catalyst."

    # PUBLIC (retail) reaction — fast, emotional, fades in 1–2 days; overreacts
    moved = isinstance(pfh, (int, float)) and np.isfinite(pfh)
    primed = moved and pfh > -3
    if net <= -0.35 or any(p["bias"] == "bearish" for p in plays):
        public = "Retail likely sells the headline first (fear); knee-jerk move tends to overshoot and fade within 1–2 days."
    elif net >= 0.2 or any(p["bias"] == "bullish" for p in plays):
        public = ("Retail chases the good news intraday" + (" — but it's already near highs, so much is priced in (reversal risk)."
                  if primed else "; the pop usually fades in 1–2 days unless a real surprise."))
    else:
        public = "Muted retail interest; no crowd to fade."

    # INVESTOR (institutional) reaction — slow, fundamental, sets the multi-week trend
    verdict = v.get("verdict", "")
    if any(p["tag"] in ("earnings_beat", "guidance_raise") for p in plays):
        investor = "Institutions act on the SURPRISE, not the headline — genuine beats see steady accumulation over 1–3 weeks (PEAD)."
    elif any(p["tag"] in ("legal_regulatory", "earnings_miss", "guidance_cut") for p in plays):
        investor = "Institutions de-risk on governance/earnings damage — distribution can persist for weeks; recovery needs proof."
    elif verdict.startswith("WORTH BUYING") or verdict.startswith("FAIR"):
        investor = "Value buyers should accumulate while it's cheap vs the market; repricing toward fair value plays out over months."
    elif verdict.startswith("AVOID"):
        investor = "Smart money is unlikely to chase here — the multiple already prices in optimism; they wait for a pullback."
    else:
        investor = "No strong institutional pull either way at this price."

    # BUY / SELL TIMING for max gains
    bb = v.get("buy_below")
    price = v.get("price")
    has_bb = isinstance(bb, (int, float)) and np.isfinite(bb)
    if verdict.startswith("WORTH BUYING"):
        entry = f"BUY now / accumulate; current price already offers a margin of safety" + (f" (fair ≈ ₹{bb:,.0f})" if has_bb else "") + "."
    elif verdict.startswith("FAIR"):
        entry = (f"Start a position now, add on dips toward ₹{bb:,.0f}." if has_bb and bb < price
                 else "Start a partial position; reasonably priced.")
    elif "expensive" in verdict.lower() or verdict.startswith("HOLD / WAIT"):
        entry = (f"WAIT — only worth buying below ≈ ₹{bb:,.0f} ({(bb/price-1)*100:+.0f}%)." if has_bb else "WAIT for a pullback — expensive now.")
    elif verdict.startswith("AVOID"):
        entry = "Don't buy at this price."
    else:
        entry = "No edge to buy now; revisit on a dip."

    # hold window = horizon where the empirical path peaks
    hold, sell = "", ""
    if path.get("ok"):
        best = max(path["path"], key=lambda x: x["mean"])
        if best["mean"] > 0:
            hold = f"Historically this setup's gain peaks around +{best['d']} trading days (avg {best['mean']:+.1f}%, {best['p_pos']}% positive)."
            sell = (f"Take profits near the +{best['d']}-day mark or the fair-value ceiling"
                    + (f" (₹{v['buy_below']:,.0f} implied)" if has_bb else "")
                    + "; trail an 8% stop and exit immediately on a bearish catalyst.")
        else:
            hold = "This setup historically did NOT pay over the next month — wait for a better entry."
            sell = "If already holding: tighten the trailing stop; this setup tends to underperform."

    # MEASURED reaction from the event study (data, not assumptions)
    measured = ""
    direction = ("up" if (net > 0.1 or any(p["bias"] == "bullish" for p in plays))
                 else "down" if (net < -0.1 or any(p["bias"] == "bearish" for p in plays)) else "")
    pos = path.get("pos") if path.get("ok") else None
    if direction and _EVENT_MODEL.get("by_dir"):
        m = _EVENT_MODEL.get("by_dir_pos", {}).get(f"{direction}|{pos}") or _EVENT_MODEL["by_dir"].get(direction)
        if m and "20" in m:
            edge = "a faint historical edge" if abs(m["20"]["car"]) >= 1.5 else "essentially coin-flip (efficient)"
            measured = (f"Event study (n={m['n']:,}): after a {direction}-shock"
                        + (f" from a {pos} base" if pos else "")
                        + f", market-adjusted return averaged {m['20']['car']:+.1f}% over 20 days "
                        f"({m['20']['p_pos']}% positive) — {edge}.")

    return {"news_effect": news_effect, "public_reaction": public, "investor_reaction": investor,
            "measured_reaction": measured,
            "entry_timing": entry, "hold_window": hold, "exit_timing": sell, "plays": plays}
