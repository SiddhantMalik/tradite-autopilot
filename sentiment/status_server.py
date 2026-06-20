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
            "value": round(ltp * p["qty"], 2),
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


@app.get("/api/universe")
def api_universe():
    """Searchable stock list: [{s: symbol, n: company name}] from the Nifty 500 (with names)."""
    stocks = []
    try:
        import pandas as pd
        import sentiment.screener as S
        S._load_nifty500()                              # ensure the CSV cache exists
        df = pd.read_csv(S.NIFTY500_CACHE)
        symcol = next((c for c in df.columns if c.strip().lower() == "symbol"), None)
        namecol = next((c for c in df.columns if "name" in c.strip().lower()), None)
        for _, r in df.iterrows():
            sym = str(r.get(symcol, "")).strip()
            nm = str(r.get(namecol, "")).strip() if namecol else sym
            if sym and sym.lower() != "nan":
                stocks.append({"s": sym, "n": nm or sym})
    except Exception:  # noqa: BLE001
        stocks = []
    if not stocks:
        try:
            from .screener import _nifty100_fallback
            stocks = [{"s": t.replace(".NS", ""), "n": t.replace(".NS", "")}
                      for t in _nifty100_fallback()]
        except Exception:  # noqa: BLE001
            stocks = [{"s": x, "n": x} for x in ["TCS", "INFY", "RELIANCE", "HDFCBANK", "SBIN", "ITC"]]
    stocks.sort(key=lambda x: x["s"])
    return JSONResponse({"stocks": stocks})


@app.get("/api/analyze")
def api_analyze(symbol: str = ""):
    """On-demand analysis of ANY NSE symbol: valuation verdict + measured base rate + news."""
    sym = (symbol or "").strip().upper().replace("NSE:", "").replace(".NS", "")
    if not sym:
        return JSONResponse({"error": "no symbol"})
    tkr = sym + ".NS"
    try:
        from .valuation import value_verdict
        from .base_rates import base_rate
        from .news_adapter import news_signal
        v = value_verdict(tkr)
        if v.get("error"):
            return JSONResponse({"error": f"{sym}: {v['error']} (use the NSE symbol, e.g. TCS)"})
        sc = v.get("scorecard", [])
        br = base_rate(tkr)
        ns = news_signal(tkr)
        return JSONResponse({
            "symbol": sym, "name": v.get("name"), "sector": v.get("sector"),
            "price": v.get("price"), "verdict": v.get("verdict"), "score": v.get("score"),
            "buy_below": v.get("buy_below"), "pe": v.get("pe"), "fwd_pe": v.get("fwd_pe"),
            "pb": v.get("pb"), "roe": v.get("roe"), "earn_yield": v.get("earn_yield"),
            "pct_from_hi": v.get("pct_from_hi"), "rsi": v.get("rsi"),
            "drivers": [f"{n} [{d}]" for n, p, d in sc if p > 0],
            "drags": [f"{n} [{d}]" for n, p, d in sc if p < 0],
            "base_rate": ({"verdict": br.get("verdict"), "cond": br.get("cond"),
                           "uncond": br.get("uncond"),
                           "setup": f"RSI {br.get('rsi',0):.0f}, {br.get('position','')}"}
                          if not br.get("error") else {"error": br.get("error")}),
            "news": ({"net": ns.get("net"), "tags": ns.get("tags"), "bearish": ns.get("bearish"),
                      "bullish": ns.get("bullish"), "top": ns.get("top", [])}
                     if ns.get("ok") else {"n": 0}),
        })
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)})


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
 <div style="display:flex;gap:8px;margin-bottom:16px">
  <div style="position:relative;flex:1">
   <input id="q" autocomplete="off" placeholder="Search by company name or symbol — e.g. Infosys, TCS, Tata, HDFC"
     oninput="filterStocks()" onfocus="filterStocks()"
     onblur="setTimeout(function(){document.getElementById('qlist').style.display='none'},200)"
     style="width:100%;box-sizing:border-box;background:var(--panel);border:1px solid var(--ln);color:var(--tx);padding:10px 12px;border-radius:8px;font-size:14px">
   <div id="qlist" style="position:absolute;z-index:20;left:0;right:0;top:46px;background:var(--panel);border:1px solid var(--ln);border-radius:8px;max-height:300px;overflow:auto;display:none"></div>
  </div>
  <button onclick="analyze()" style="background:var(--acc);color:#04101f;border:0;padding:10px 18px;border-radius:8px;font-weight:700;cursor:pointer">Analyze</button>
 </div>
 <div id="ares"></div>
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
  ['NAV',inr(s.nav)],['Holdings value',inr(s.nav-s.cash)],['Cash',inr(s.cash)],['Positions',s.positions],
  ['Total P&L (net)',`<span class="${cls(s.total_pnl)}">${inr(s.total_pnl)} (${s.total_return_pct}%)</span>`],
  ['Costs paid',`<span class="r">−${inr(s.total_costs||0)}</span>`],
  ['Net after-tax',`<span class="${cls((s.net_after_tax||s.nav)-s.capital)}">${inr(s.net_after_tax||s.nav)} (${s.net_return_pct||s.total_return_pct}%)</span>`],
  ['Drawdown',`${s.drawdown_pct}%`],
 ].map(([l,v])=>`<div class="card"><div class="lab">${l}</div><div class="val">${v}</div></div>`).join('');
 document.getElementById('pos').innerHTML=tbl(['Symbol','Sector','Qty','Avg','LTP','Value','Stop','Target','P&L','%'],
  d.positions.map(p=>[p.symbol,`<span class="mut">${p.sector}</span>`,p.qty,inr(p.avg),inr(p.ltp),inr(p.value),inr(p.stop),inr(p.target),
   `<span class="${cls(p.pnl)}">${inr(p.pnl)}</span>`,`<span class="${cls(p.pnl)}">${p.pnl_pct}%</span>`])) || empty();
 document.getElementById('ord').innerHTML=tbl(['Symbol','Side','Type','Trigger','Qty','Status'],
  d.orders.map(o=>[o.symbol,o.side,`<span class="pill">${o.type}</span>`,inr(o.trigger),o.qty,o.status])) || empty();
 document.getElementById('trd').innerHTML=tbl(['Time','Action','Symbol','Qty','Price','P&L','Why'],
  d.trades.map(t=>[t.ts?t.ts.slice(0,16).replace('T',' '):'',
   `<span class="${t.action=='BUY'?'g':'r'}">${t.action}</span>`,t.symbol,t.qty,inr(t.price),
   t.pnl!=null?`<span class="${cls(t.pnl)}">${inr(t.pnl)}</span>`:'—',
   `<span class="mut">${t.reason||'—'}</span>`])) || empty();
 document.getElementById('foot').textContent='updated '+new Date().toLocaleTimeString()+' · auto-refresh 30s · paper (no live orders)';
}
function tbl(h,rows){ if(!rows.length) return '';
 return '<table><thead><tr>'+h.map(x=>`<th>${x}</th>`).join('')+'</tr></thead><tbody>'+
  rows.map(r=>'<tr>'+r.map(c=>`<td>${c}</td>`).join('')+'</tr>').join('')+'</tbody></table>';}
function empty(){return '<div class="card mut">none</div>';}
function f1(x){return (typeof x==='number'&&isFinite(x))?x.toFixed(1):'n/a';}
function pc(x){return (typeof x==='number')?Math.round(x*100)+'%':'n/a';}
async function analyze(){
 let sym=document.getElementById('q').value.trim(); if(!sym)return;
 const u=sym.toUpperCase();
 if(STOCKS.some(x=>x.s===u)){ sym=u; }
 else { const m=STOCKS.filter(x=>x.s.toLowerCase().includes(sym.toLowerCase())||x.n.toLowerCase().includes(sym.toLowerCase())); if(m.length) sym=m[0].s; }
 document.getElementById('q').value=sym;
 const el=document.getElementById('ares');
 el.innerHTML=`<div class="card mut">Analyzing ${sym.toUpperCase()} … (valuation + base rate + news, ~10-15s)</div>`;
 let d; try{ d=await (await fetch('/api/analyze?symbol='+encodeURIComponent(sym))).json(); }
 catch(e){ el.innerHTML='<div class="card r">request failed</div>'; return; }
 if(d.error){ el.innerHTML='<div class="card r">'+d.error+'</div>'; return; }
 const vc=d.verdict&&d.verdict.indexOf('WORTH')===0?'g':(d.verdict&&d.verdict.indexOf('AVOID')===0?'r':'');
 const br=d.base_rate||{}, ns=d.news||{};
 const brTxt = br.error?br.error:(br.verdict?`${br.verdict}${br.cond?` — fwd-20d mean ${f1(br.cond.mean)}% vs base ${f1((d.base_rate.uncond||{}).mean)}% (n=${br.cond.n})`:''}`:'n/a');
 const newsTxt = ns.bearish?'<span class="r">BEARISH</span>':(ns.bullish?'<span class="g">bullish</span>':'neutral');
 el.innerHTML=`<div class="card">
   <div style="font-size:16px;font-weight:700">${d.symbol} — ${d.name||''} <span class="mut" style="font-size:12px">${d.sector||''}</span></div>
   <div style="margin:6px 0;font-size:15px"><span class="${vc}" style="font-weight:700">${d.verdict||''}</span>
     <span class="mut">· score ${d.score}${isFinite(d.buy_below)?' · buy-below ₹'+Math.round(d.buy_below).toLocaleString('en-IN'):''}</span></div>
   <div class="mut" style="font-size:13px">LTP ₹${Math.round(d.price).toLocaleString('en-IN')} · P/E ${f1(d.pe)} (fwd ${f1(d.fwd_pe)}) · P/B ${f1(d.pb)} · ROE ${pc(d.roe)} · earnings-yield ${f1(d.earn_yield)}% · ${Math.round(d.pct_from_hi)}% vs 52w-high · RSI ${Math.round(d.rsi)}</div>
   <div style="margin-top:8px;font-size:13px"><span class="g">▲ for</span> ${(d.drivers||[]).join('; ')||'—'}</div>
   <div style="font-size:13px"><span class="r">▼ against</span> ${(d.drags||[]).join('; ')||'—'}</div>
   <div style="margin-top:8px;font-size:13px">Measured base rate: ${brTxt}</div>
   <div style="margin-top:4px;font-size:13px">News: ${newsTxt} ${ns.tags&&ns.tags.length?'['+ns.tags.join(', ')+']':''}</div>
   ${(ns.top||[]).map(h=>`<div class="mut" style="font-size:12px">• [${h.score>0?'+':''}${h.score}] ${h.title}</div>`).join('')}
 </div>`;
}
let STOCKS=[];
async function loadUniverse(){ try{ const d=await (await fetch('/api/universe')).json(); STOCKS=d.stocks||[]; }catch(e){} }
function esc(s){return (s||'').replace(/[<>'"]/g,'');}
function filterStocks(){
 const q=document.getElementById('q').value.trim().toLowerCase(); const box=document.getElementById('qlist');
 if(!q){box.style.display='none';return;}
 const m=STOCKS.filter(x=>x.s.toLowerCase().includes(q)||x.n.toLowerCase().includes(q)).slice(0,25);
 if(!m.length){box.style.display='none';return;}
 box.innerHTML=m.map(x=>`<div onmousedown="pick('${x.s}')" style="padding:8px 12px;cursor:pointer;border-bottom:1px solid var(--ln)"><b>${x.s}</b> <span class="mut" style="font-size:12px">${esc(x.n)}</span></div>`).join('');
 box.style.display='block';
}
function pick(s){ document.getElementById('q').value=s; document.getElementById('qlist').style.display='none'; analyze(); }
loadUniverse(); load(); setInterval(load,30000);
</script></body></html>"""
