"""
Learn how news ACTUALLY affects a stock — the enriched, honest version.

Core lesson from the data: ~80% of free-tier entity-tagged "news" is MARKET/SECTOR commentary
("Sensex jumps 790 pts") that Marketaux pins onto every large cap. That's beta, not company
information, and it drowns the idiosyncratic signal. So we:

  1. split each headline into COMPANY-SPECIFIC (title names the company) vs MARKET-COMMENTARY;
  2. re-score tone with our lexicon (more reliable than Marketaux's ~0.5-centred score) and
     detect event TYPE (earnings/M&A/regulatory/...);
  3. measure forward MARKET-ADJUSTED CAR (vs ^NSEI) at +5/+10/+20d for each cut;
  4. test COMPOUNDING (intensity = how many company-specific items cluster);
  5. do a time-ordered out-of-sample IC check (honest about the tiny sample).

Output -> sentiment/news_reaction_model.json (consumed by analyst.py). Run:
    python -m sentiment.news_learn
"""
from __future__ import annotations

import os, json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from .market_grounding import detect_event_tags
from .news_reaction_study import _fwd_car, HORIZONS
from .base_rates import _load_close

CSV = os.path.join(os.path.dirname(__file__), "news_history.csv")
OUT = os.path.join(os.path.dirname(__file__), "news_reaction_model.json")

# keyword(s) that mark a headline as genuinely about the company (not market commentary)
KW = {
    "RELIANCE": ["reliance", "jio", "ril", "ambani"], "HDFCBANK": ["hdfc"],
    "INFY": ["infosys", "infy"], "TCS": ["tcs", "tata consultancy"], "WIPRO": ["wipro"],
    "HCLTECH": ["hcl"], "TRENT": ["trent"], "VEDL": ["vedanta", "hindustan zinc"],
    "HINDUNILVR": ["hindustan unilever", "hul", "unilever"], "ICICIBANK": ["icici"],
    "LT": ["larsen", "l&t", "l and t"], "COALINDIA": ["coal india"], "ADANIENT": ["adani"],
    "MARUTI": ["maruti", "suzuki"], "SBIN": ["sbi", "state bank"], "AXISBANK": ["axis bank"],
    "BHARTIARTL": ["bharti", "airtel"], "ITC": ["itc"], "CIPLA": ["cipla"],
    "MAHABANK": ["maharashtra"], "BOSCHLTD": ["bosch"], "BANDHANBNK": ["bandhan"],
}


def _kw(sym: str) -> list[str]:
    return KW.get(sym.replace(".NS", ""), [sym.replace(".NS", "").lower()])


def build(verbose: bool = True) -> dict:
    df = pd.read_csv(CSV)
    df["dt"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["day"] = df["dt"].dt.tz_convert("Asia/Kolkata").dt.normalize().dt.tz_localize(None)
    df = df.dropna(subset=["day"])

    from .finbert_scorer import FinBERTScorer
    sc = FinBERTScorer(use_finbert=False)
    df["title"] = df["title"].fillna("")
    df["lex"] = df["title"].map(sc.score_signed)            # lexicon signed tone [-1,1]
    df["tags"] = df["title"].map(detect_event_tags)
    df["spec"] = df.apply(lambda r: any(k in r["title"].lower() for k in _kw(r["symbol"])), axis=1)

    nidx = _load_close("^NSEI")
    if nidx is not None:
        nidx.index = pd.to_datetime(nidx.index)

    # one row per (symbol, day): split company-specific vs market tone, intensity, tags
    rows = []
    for (sym, day), g in df.groupby(["symbol", "day"]):
        gs = g[g["spec"]]
        car = _fwd_car(sym, day, nidx)
        if car is None:
            continue
        tags = sorted({t for ts in gs["tags"] for t in ts})        # tags from company news only
        rows.append({
            "sym": sym, "day": str(day.date()),
            "spec_n": int(len(gs)), "mkt_n": int(len(g) - len(gs)),
            "spec_tone": float(gs["lex"].mean()) if len(gs) else 0.0,
            "mkt_tone": float(g[~g["spec"]]["lex"].mean()) if len(g) - len(gs) else 0.0,
            "tags": tags, "pos": car["pos"],
            **{f"car{h}": car[str(h)] for h in HORIZONS}})
    E = pd.DataFrame(rows)
    if E.empty:
        raise SystemExit("no events with forward prices")

    def stat(sub, h=20):
        if len(sub) < 6:
            return None
        x = sub[f"car{h}"]
        return {"car": round(float(x.mean()), 2), "p_pos": round(float((x > 0).mean()) * 100),
                "n": int(len(sub))}

    spec = E[E["spec_n"] > 0]
    mkt = E[(E["spec_n"] == 0) & (E["mkt_n"] > 0)]   # pure market-commentary days
    cuts = {
        "company-specific, POSITIVE tone": stat(spec[spec["spec_tone"] > 0.05]),
        "company-specific, NEGATIVE tone": stat(spec[spec["spec_tone"] < -0.05]),
        "company-specific, neutral":       stat(spec[spec["spec_tone"].abs() <= 0.05]),
        "MARKET-commentary, POSITIVE tone": stat(mkt[mkt["mkt_tone"] > 0.05]),
        "MARKET-commentary, NEGATIVE tone": stat(mkt[mkt["mkt_tone"] < -0.05]),
    }
    # by event type (company news only)
    by_tag = {}
    all_tags = sorted({t for ts in spec["tags"] for t in ts})
    for t in all_tags:
        s = stat(spec[spec["tags"].map(lambda x: t in x)])
        if s:
            by_tag[t] = s
    # compounding: company-specific intensity (1 item vs 2+ items same day)
    comp = {"company news, 1 item": stat(spec[spec["spec_n"] == 1]),
            "company news, 2+ items (compounded)": stat(spec[spec["spec_n"] >= 2])}

    # information coefficient: idiosyncratic (company) tone vs market tone
    ic = {}
    for label, col, sub in [("company_tone", "spec_tone", spec), ("market_tone", "mkt_tone", mkt)]:
        ic[label] = {str(h): (round(float(sub[col].corr(sub[f"car{h}"])), 3)
                              if len(sub) > 8 and sub[col].nunique() > 2 else None)
                     for h in HORIZONS}

    # honest out-of-sample: time-split company-specific (train early, test late), sign-of-tone rule
    spec_sorted = spec.sort_values("day")
    oos = {"note": "tiny sample — directional check only"}
    if len(spec_sorted) >= 20:
        cut = int(len(spec_sorted) * 0.6)
        tr, te = spec_sorted.iloc[:cut], spec_sorted.iloc[cut:]
        # rule learned on train: does sign(spec_tone) predict sign(car20)?
        tr_sign = np.sign(tr["spec_tone"]) ; tr_hit = (np.sign(tr["car20"]) == tr_sign)
        te_sign = np.sign(te["spec_tone"]) ; te_hit = (np.sign(te["car20"]) == te_sign)
        oos = {"train_dir_acc": round(float(tr_hit.mean()), 2), "n_train": int(len(tr)),
               "test_dir_acc": round(float(te_hit.mean()), 2), "n_test": int(len(te)),
               "test_ic20": (round(float(te["spec_tone"].corr(te["car20"])), 3)
                             if te["spec_tone"].nunique() > 2 else None)}

    model = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "source": "marketaux",
        "n_events": int(len(E)), "n_company_specific": int(len(spec)),
        "n_market_commentary": int(len(mkt)), "n_symbols": int(E["sym"].nunique()),
        "date_span": [E["day"].min(), E["day"].max()],
        "lesson": ("Most entity-tagged items are market commentary with ~no idiosyncratic effect; "
                   "trust COMPANY-SPECIFIC news (title names the company) for single-stock calls."),
        "cuts_20d": cuts, "by_event_type_20d": by_tag, "compounding_20d": comp,
        "information_coefficient": ic, "out_of_sample": oos,
    }
    with open(OUT, "w") as fh:
        json.dump(model, fh, indent=2)

    if verbose:
        print(f"\nNEWS-LEARN  {len(E)} events ({len(spec)} company-specific, {len(mkt)} market-only), "
              f"{E['sym'].nunique()} names, {E['day'].min()}..{E['day'].max()}")
        def show(title, d):
            print(f"\n{title}")
            for k, v in d.items():
                print(f"  {k:38s} " + (f"+20d CAR {v['car']:+5.2f}%  P>0 {v['p_pos']:3d}%  n={v['n']}"
                                       if v else "(too thin)"))
        show("TONE x scope (the key cut):", cuts)
        show("By event TYPE (company news only):", by_tag or {"(none with n>=6)": None})
        show("Compounding (company-news intensity):", comp)
        print("\nInformation coefficient corr(tone, fwd CAR):")
        for k, hs in ic.items():
            print(f"  {k:13s} " + "  ".join(f"+{h}d {hs[str(h)]}" for h in HORIZONS))
        print(f"\nOut-of-sample (company tone sign -> +20d direction): {oos}")
        print(f"saved -> {OUT}")
    return model


if __name__ == "__main__":
    build()
