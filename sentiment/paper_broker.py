"""
Paper broker — a faithful simulated book so the full BUY → hold → SELL-for-profit
loop is testable with zero money at risk (Tradite is paper-first).

Holds cash + positions, each armed with a stop-loss and a target. `mark(prices)`
is the engine of the profit mechanism: it checks live prices and AUTO-SELLS any
position that hit its target (book the profit) or its stop (cut the loss),
realising P&L. Everything persists to data_cache/paper_book.json so the book
survives across runs.

The live broker (Zerodha Kite) mirrors this exact contract — entries via
place_order, exits via a GTT one-cancels-other (stop + target) — so switching
from paper to live changes only the backend, not the logic.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import config

BOOK_PATH = config.DATA_DIR / "paper_book.json"


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class PaperBroker:
    def __init__(self, path=BOOK_PATH):
        self.path = path
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────
    def _load(self):
        d = json.load(open(self.path)) if self.path.exists() else {}
        self.capital = d.get("capital", 0.0)
        self.cash = d.get("cash", 0.0)
        self.peak_nav = d.get("peak_nav", self.capital)
        self.day = d.get("day") or _today()
        self.day_start_nav = d.get("day_start_nav", self.capital)
        self.last_nav = d.get("last_nav", None)
        self.realized_pnl = d.get("realized_pnl", 0.0)
        self.positions = d.get("positions", {})
        self.trades = d.get("trades", [])

    def _save(self):
        json.dump({
            "capital": self.capital, "cash": self.cash, "peak_nav": self.peak_nav,
            "day": self.day, "day_start_nav": self.day_start_nav, "last_nav": self.last_nav,
            "realized_pnl": self.realized_pnl, "positions": self.positions, "trades": self.trades,
        }, open(self.path, "w"), indent=2, default=str)

    def init(self, capital: float):
        self.capital = self.cash = float(capital)
        self.peak_nav = self.day_start_nav = self.last_nav = float(capital)
        self.day = _today(); self.realized_pnl = 0.0
        self.positions = {}; self.trades = []
        self._save()

    # ── valuations ───────────────────────────────────────────────────────────
    def nav_cost(self) -> float:
        return self.cash + sum(p["qty"] * p["avg"] for p in self.positions.values())

    def marked_nav(self, prices: dict | None = None) -> float:
        if prices:
            return self.cash + sum(p["qty"] * prices.get(s, p["avg"])
                                   for s, p in self.positions.items())
        return self.last_nav if self.last_nav is not None else self.nav_cost()

    def day_pnl(self) -> float:
        return self.marked_nav() - self.day_start_nav

    def sector_exposure(self) -> dict:
        out: dict[str, float] = {}
        for p in self.positions.values():
            sec = p.get("sector", "Unknown")
            out[sec] = out.get(sec, 0.0) + p["qty"] * p["avg"]
        return out

    def _roll_day(self):
        if _today() != self.day:
            self.day = _today()
            self.day_start_nav = self.marked_nav()

    # ── orders ───────────────────────────────────────────────────────────────
    def buy(self, symbol, qty, price, stop_pct, target_pct, sector, trail_pct=0, tp1_pct=0):
        cost = qty * price
        self.cash -= cost
        if symbol in self.positions:                      # average up
            p = self.positions[symbol]; tot = p["qty"] + qty
            p["avg"] = (p["qty"] * p["avg"] + cost) / tot; p["qty"] = tot
            p["stop"] = round(p["avg"] * (1 - stop_pct / 100), 2)
            p["target"] = round(p["avg"] * (1 + target_pct / 100), 2)
            p["high"] = max(p.get("high", price), price)
        else:
            self.positions[symbol] = {
                "qty": qty, "avg": round(price, 2),
                "stop_pct": stop_pct, "target_pct": target_pct,
                "trail_pct": trail_pct, "tp1_pct": tp1_pct, "scaled": False,
                "stop": round(price * (1 - stop_pct / 100), 2),
                "target": round(price * (1 + target_pct / 100), 2),
                "high": round(price, 2),         # high-water mark for the trailing stop
                "sector": sector, "entry_date": _today(),
            }
        self.trades.append({"ts": _now(), "action": "BUY", "symbol": symbol,
                            "qty": qty, "price": round(price, 2), "cost": round(cost, 2)})
        self._roll_day(); self._save()

    def sell(self, symbol, price, reason) -> float:
        p = self.positions.pop(symbol)
        pnl = (price - p["avg"]) * p["qty"]
        self.cash += p["qty"] * price
        self.realized_pnl += pnl
        self.trades.append({"ts": _now(), "action": "SELL", "symbol": symbol,
                            "qty": p["qty"], "price": round(price, 2),
                            "pnl": round(pnl, 2), "reason": reason})
        return pnl

    def mark(self, prices: dict) -> list[dict]:
        """The profit mechanism: trail the stop up with the high-water mark, then
        auto-exit any position that hit its (trailed) stop or its target."""
        exits = []
        for sym in list(self.positions):
            px = prices.get(sym)
            if px is None:
                continue
            p = self.positions[sym]
            # ratchet the stop UP as price makes new highs (never down) — locks in gains
            tp = p.get("trail_pct", 0)
            if px > p.get("high", p["avg"]):
                p["high"] = round(px, 2)
            if tp:
                p["stop"] = max(p["stop"], round(p["high"] * (1 - tp / 100), 2))
            # partial profit-take: book HALF at +tp1, let the rest ride the trailing stop
            tp1 = p.get("tp1_pct", 0)
            if tp1 and not p.get("scaled") and p["qty"] >= 2 and px >= p["avg"] * (1 + tp1 / 100):
                half = p["qty"] // 2
                pnl = (px - p["avg"]) * half
                self.cash += half * px
                self.realized_pnl += pnl
                p["qty"] -= half
                p["scaled"] = True
                self.trades.append({"ts": _now(), "action": "SELL", "symbol": sym, "qty": half,
                                    "price": round(px, 2), "pnl": round(pnl, 2), "reason": "PARTIAL-TP"})
                exits.append({"symbol": sym, "reason": "PARTIAL profit (sold half)", "price": px, "pnl": pnl})
            if px <= p["stop"]:
                trailed = p["stop"] >= p["avg"]              # stop above cost ⇒ a profitable trail
                reason = "TRAIL (profit locked)" if trailed else "STOP-LOSS"
                pnl = self.sell(sym, p["stop"], "TRAIL" if trailed else "STOP")
                exits.append({"symbol": sym, "reason": reason, "price": p["stop"], "pnl": pnl})
            elif px >= p["target"]:
                pnl = self.sell(sym, p["target"], "TARGET")
                exits.append({"symbol": sym, "reason": "TARGET (profit booked)", "price": p["target"], "pnl": pnl})
        self.last_nav = self.marked_nav(prices)
        self.peak_nav = max(self.peak_nav, self.last_nav)
        self._roll_day(); self._save()
        return exits

    def summary(self, prices: dict | None = None) -> dict:
        nav = self.marked_nav(prices)
        unreal = sum((prices or {}).get(s, p["avg"]) * p["qty"] - p["avg"] * p["qty"]
                     for s, p in self.positions.items()) if prices else 0.0
        return {
            "capital": self.capital, "cash": round(self.cash, 2),
            "positions": len(self.positions), "nav": round(nav, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(unreal, 2),
            "total_pnl": round(self.realized_pnl + unreal, 2),
            "total_return_pct": round((nav / self.capital - 1) * 100, 2) if self.capital else 0.0,
            "peak_nav": round(self.peak_nav, 2),
            "drawdown_pct": round((nav / self.peak_nav - 1) * 100, 2) if self.peak_nav else 0.0,
        }
