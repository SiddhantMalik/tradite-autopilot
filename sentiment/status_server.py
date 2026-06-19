"""
Live status dashboard + autopilot host (one container).

A FastAPI app that (a) runs the autopilot scheduler loop in a background thread and
(b) serves a live dark dashboard showing NAV/P&L, current positions with live LTP,
the active stop/target EXIT orders resting "in the market", and recent trades.

Run:  uvicorn sentiment.status_server:app --host 0.0.0.0 --port 8080
On DigitalOcean App Platform this is a Service with public ingress → a URL you open
to watch the book live. Paper by default.
"""
from __future__ import annotations

import threading

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from .trade_engine import TradeEngine, current_prices
from . import scheduler

app = FastAPI(title="Tradite Autopilot")
_loop_started = False


@app.on_event("startup")
def _start_loop():
    global _loop_started
    if not _loop_started:
        threading.Thread(target=scheduler.run_loop, daemon=True).start()
        _loop_started = True


def _snapshot() -> dict:
    b = TradeEngine().broker
    prices = current_prices(list(b.positions)) if b.positions else {}
    s = b.summary(prices)
    positions, orders = [], []
    for sym, p in b.positions.items():
        ltp = prices.get(sym, p["avg"])
        pnl = (ltp - p["avg"]) * p["qty"]
        positions.append({
            "symbol": sym, "qty": p["qty"], "avg": round(p["avg"], 2), "ltp": round(ltp, 2),
            "stop": p["stop"], "target": p["target"], "sector": p.get("sector", ""),
            "pnl": round(pnl, 2), "pnl_pct": round((ltp / p["avg"] - 1) * 100, 2),
        })
        orders.append({"symbol": sym, "side": "SELL", "type": "STOP-LOSS",
                       "trigger": p["stop"], "qty": p["qty"], "status": "resting"})
        orders.append({"symbol": sym, "side": "SELL", "type": "TARGET",
                       "trigger": p["target"], "qty": p["qty"], "status": "resting"})
    trades = list(reversed(b.trades[-25:]))
    return {"summary": s, "positions": positions, "orders": orders, "trades": trades,
            "mode": "PAPER"}


@app.get("/api/status")
def api_status():
    return JSONResponse(_snapshot())


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def home():
    return _HTML


_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>Tradite Autopilot — live</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 :root{--bg:#0b0e14;--panel:#141925;--ln:#222a3a;--tx:#e6edf3;--mut:#8b98ad;--grn:#3fb950;--red:#f85149;--acc:#58a6ff}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--tx);font:14px/1.4 -apple-system,Segoe UI,Roboto,sans-serif}
 header{padding:16px 22px;border-bottom:1px solid var(--ln);display:flex;justify-content:space-between;align-items:center}
 h1{font-size:18px;margin:0} .mode{font-size:11px;color:#000;background:var(--acc);padding:2px 8px;border-radius:10px;font-weight:700}
 .wrap{padding:18px 22px;max-width:1100px;margin:0 auto}
 .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:18px}
 .card{background:var(--panel);border:1px solid var(--ln);border-radius:10px;padding:14px}
 .card .lab{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.5px}
 .card .val{font-size:22px;font-weight:700;margin-top:4px}
 h2{font-size:13px;color:var(--mut);text-transform:uppercase;letter-spacing:.5px;margin:22px 0 8px}
 table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--ln);border-radius:10px;overflow:hidden}
 th,td{padding:9px 12px;text-align:right;border-bottom:1px solid var(--ln);font-variant-numeric:tabular-nums}
 th{color:var(--mut);font-size:11px;text-transform:uppercase;text-align:right} th:first-child,td:first-child{text-align:left}
 tr:last-child td{border-bottom:none}
 .g{color:var(--grn)} .r{color:var(--red)} .mut{color:var(--mut)}
 .pill{font-size:11px;padding:1px 7px;border-radius:8px;border:1px solid var(--ln)}
 .foot{color:var(--mut);font-size:12px;margin-top:18px}
</style></head><body>
<header><h1>📈 Tradite Autopilot</h1><span class="mode" id="mode">PAPER</span></header>
<div class="wrap">
 <div class="cards" id="cards"></div>
 <h2>Positions</h2><div id="pos"></div>
 <h2>Active exit orders (resting in market)</h2><div id="ord"></div>
 <h2>Recent trades</h2><div id="trd"></div>
 <div class="foot" id="foot">loading…</div>
</div>
<script>
const inr=n=>'₹'+Math.round(n).toLocaleString('en-IN');
const cls=n=>n>0?'g':(n<0?'r':'mut');
async function load(){
 let d; try{ d=await (await fetch('/api/status')).json(); }catch(e){ document.getElementById('foot').textContent='offline'; return; }
 document.getElementById('mode').textContent=d.mode;
 const s=d.summary;
 document.getElementById('cards').innerHTML=[
  ['NAV',inr(s.nav)],['Cash',inr(s.cash)],['Positions',s.positions],
  ['Total P&L',`<span class="${cls(s.total_pnl)}">${inr(s.total_pnl)} (${s.total_return_pct}%)</span>`],
  ['Realized',`<span class="${cls(s.realized_pnl)}">${inr(s.realized_pnl)}</span>`],
  ['Drawdown',`${s.drawdown_pct}%`],
 ].map(([l,v])=>`<div class="card"><div class="lab">${l}</div><div class="val">${v}</div></div>`).join('');
 document.getElementById('pos').innerHTML=tbl(['Symbol','Sector','Qty','Avg','LTP','Stop','Target','P&L','%'],
  d.positions.map(p=>[p.symbol,`<span class="mut">${p.sector}</span>`,p.qty,inr(p.avg),inr(p.ltp),inr(p.stop),inr(p.target),
   `<span class="${cls(p.pnl)}">${inr(p.pnl)}</span>`,`<span class="${cls(p.pnl)}">${p.pnl_pct}%</span>`])) || empty();
 document.getElementById('ord').innerHTML=tbl(['Symbol','Side','Type','Trigger','Qty','Status'],
  d.orders.map(o=>[o.symbol,o.side,`<span class="pill">${o.type}</span>`,inr(o.trigger),o.qty,o.status])) || empty();
 document.getElementById('trd').innerHTML=tbl(['Time','Action','Symbol','Qty','Price','P&L'],
  d.trades.map(t=>[t.ts?t.ts.slice(0,16).replace('T',' '):'',
   `<span class="${t.action=='BUY'?'g':'r'}">${t.action}</span>`,t.symbol,t.qty,inr(t.price),
   t.pnl!=null?`<span class="${cls(t.pnl)}">${inr(t.pnl)}</span>`:'—'])) || empty();
 document.getElementById('foot').textContent='updated '+new Date().toLocaleTimeString()+' · auto-refresh 30s · paper (no live orders)';
}
function tbl(h,rows){ if(!rows.length) return '';
 return '<table><thead><tr>'+h.map(x=>`<th>${x}</th>`).join('')+'</tr></thead><tbody>'+
  rows.map(r=>'<tr>'+r.map(c=>`<td>${c}</td>`).join('')+'</tr>').join('')+'</tbody></table>';}
function empty(){return '<div class="card mut">none</div>';}
load(); setInterval(load,30000);
</script></body></html>"""
