"""
Self-reflection CRITIC — a second set of eyes that audits the proposed plan BEFORE any
order reaches the broker. The risk gate checks each order in isolation; the critic looks
at the WHOLE plan holistically and can veto, catching logical mistakes a per-order check
misses. Every decision gets a written audit (logged + returned).

Checks per proposed buy:
  • contradiction  — buying a name we are SELLING this same cycle
  • bad news       — buying a name flagged bearish by the news layer
  • no stop        — an order with no stop-loss
  • cash           — cumulative spend exceeds investable cash (after the min-cash reserve)
  • liquidity      — order larger than MAX_ADV_PCT of the stock's average daily traded value
"""
from __future__ import annotations

from .market_filters import liquidity_cap


def review(orders: list[dict], broker, sold_syms: set[str], news: dict, min_cash_pct: float) -> dict:
    """Return {approved: [...], vetoes: [...], audit: [str]}."""
    approved, vetoes, audit = [], [], []
    reserve = min_cash_pct * (broker.marked_nav() or broker.capital)
    investable = max(0.0, broker.cash - reserve)
    spend = 0.0
    for o in sorted(orders, key=lambda x: -x.get("rupees", 0)):   # fund biggest-conviction first
        sym = o.get("symbol", "?")
        rup = float(o.get("rupees", 0))
        reasons = []
        if sym in sold_syms:
            reasons.append("sold same cycle")
        if (news.get(sym) or {}).get("bearish"):
            reasons.append("bearish news")
        if not o.get("stop_pct"):
            reasons.append("no stop-loss")
        if spend + rup > investable + 1:
            reasons.append(f"exceeds investable cash (₹{investable - spend:,.0f} left)")
        cap = liquidity_cap(sym + ".NS")
        if rup > cap:
            reasons.append(f"illiquid: ₹{rup:,.0f} > ADV-cap ₹{cap:,.0f}")
        if reasons:
            vetoes.append({"symbol": sym, "reasons": reasons})
            audit.append(f"VETO {sym}: {'; '.join(reasons)}")
        else:
            approved.append(o)
            spend += rup
            audit.append(f"OK   {sym}: ₹{rup:,.0f}")
    return {"approved": approved, "vetoes": vetoes, "audit": audit}
