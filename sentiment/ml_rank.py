"""
Gated cross-sectional ML ranker (Layer 10).

Honest framing, measured first (see ML_VERDICT.md): on a ~Nifty-100 large-cap
universe (2018-2026, monthly, net of cost), an ML ranker on price features had
ZERO out-of-sample rank-IC and LOST to 12-1 momentum AND to plain equal-weight —
while churning 3× the turnover. So this module ships with a GATE: the ML score is
allowed to influence selection ONLY if it actually clears a bar
(OOS IC t-stat > 2 AND net Sharpe > momentum). Otherwise it self-reports "no edge"
and the system ignores it. ML as a disciplined, gated tilt — never a blind input.

  python -m sentiment.ml_rank            # evaluate (head-to-head + gate) and cache verdict
  python -m sentiment.ml_rank scores     # today's cross-sectional ML scores (top/bottom)
"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np
import pandas as pd

import config

FEATS = ["mom_12_1", "mom_6_1", "mom_3", "rev_1", "vol_60", "dist_hi"]
GATE_PATH = config.DATA_DIR / "ml_gate.json"


def load_panel(min_frac: float = 0.6) -> pd.DataFrame:
    """Wide close panel. Prefers prebuilt mlpanel_*.csv, else per-ticker CSV cache."""
    parts = sorted(glob.glob(str(config.DATA_DIR / "mlpanel_*.csv")))
    if parts:
        px = pd.concat([pd.read_csv(p, index_col=0, parse_dates=True) for p in parts], axis=1)
    else:
        series = {}
        for f in glob.glob(str(config.DATA_DIR / "*__yfinance.csv")):
            sym = f.split("/")[-1].replace("__yfinance.csv", "")
            try:
                df = pd.read_csv(f, index_col=0, parse_dates=True)
                col = next((c for c in ("Close", "close") if c in df.columns), None)
                if col:
                    series[sym] = df[col].dropna()
            except Exception:  # noqa: BLE001
                pass
        px = pd.DataFrame(series)
    px = px.loc[:, ~px.columns.duplicated()].sort_index()
    return px.dropna(axis=1, thresh=int(len(px) * min_frac))


def _xs_z(s):
    r = s.rank()
    return (r - r.mean()) / (r.std() + 1e-9)


def build_table(px: pd.DataFrame) -> pd.DataFrame:
    rets = px.pct_change(fill_method=None)
    idx = px.index
    rows = []
    for t in range(252, len(idx) - 21, 21):
        valid = px.iloc[t].dropna().index
        if len(valid) < 30:
            continue
        f = pd.DataFrame(index=valid)
        f["mom_12_1"] = (px.iloc[t - 21] / px.iloc[t - 252] - 1).reindex(valid)
        f["mom_6_1"] = (px.iloc[t - 21] / px.iloc[t - 126] - 1).reindex(valid)
        f["mom_3"] = (px.iloc[t] / px.iloc[t - 63] - 1).reindex(valid)
        f["rev_1"] = -(px.iloc[t] / px.iloc[t - 21] - 1).reindex(valid)
        f["vol_60"] = rets.iloc[t - 60:t].std().reindex(valid)
        f["dist_hi"] = (px.iloc[t] / px.iloc[t - 252:t].max() - 1).reindex(valid)
        fwd = (px.iloc[t + 21] / px.iloc[t] - 1).reindex(valid)
        f = f.dropna()
        if len(f) < 30:
            continue
        fwd = fwd.reindex(f.index)
        fz = f.apply(_xs_z)
        fwd_dm = fwd - fwd.mean()
        for sym in f.index:
            rows.append((idx[t], sym, *fz.loc[sym].values, fwd.loc[sym], fwd_dm.loc[sym]))
    return pd.DataFrame(rows, columns=["date", "sym", *FEATS, "fwd", "fwd_dm"])


def evaluate(px: pd.DataFrame | None = None) -> dict:
    """Purged walk-forward → OOS IC + net-of-cost Sharpe vs momentum/EW + GATE."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from scipy.stats import spearmanr
    px = load_panel() if px is None else px
    P = build_table(px)
    dates = sorted(P["date"].unique())
    preds = {}
    for i in range(24, len(dates)):
        train = P[P["date"] <= dates[i - 2]]
        test = P[P["date"] == dates[i]]
        if len(train) < 800 or len(test) < 20:
            continue
        m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200,
                                          l2_regularization=1.0, min_samples_leaf=30, random_state=7)
        m.fit(train[FEATS], train["fwd_dm"])
        preds[dates[i]] = pd.Series(m.predict(test[FEATS]), index=test["sym"].values)
    pds = sorted(preds)
    ics = []
    for d in pds:
        sub = P[P["date"] == d].set_index("sym")
        ics.append(spearmanr(preds[d].reindex(sub.index), sub["fwd"]).correlation)
    ics = np.array([x for x in ics if x == x])
    ic_t = float(ics.mean() / (ics.std() / np.sqrt(len(ics)) + 1e-9))

    def book(kind):
        prev, mret, turn = set(), [], []
        for d in pds:
            sub = P[P["date"] == d].set_index("sym")
            sc = preds[d].reindex(sub.index) if kind == "ml" else \
                (sub["mom_12_1"] if kind == "mom" else pd.Series(1.0, index=sub.index))
            n = max(5, int(len(sub) * 0.2))
            picks = list(sub.index) if kind == "ew" else list(sc.sort_values(ascending=False).head(n).index)
            mret.append(sub.loc[picks, "fwd"].mean())
            turn.append(len(set(picks) ^ prev) / max(1, 2 * len(picks)))
            prev = set(picks)
        net = np.array(mret) - np.array(turn) * (2 * 30 / 10000)
        return float((net.mean() / (net.std() + 1e-9)) * np.sqrt(12)), float(np.mean(turn))

    ml_sh, ml_turn = book("ml")
    mom_sh, _ = book("mom")
    ew_sh, _ = book("ew")
    gate_pass = bool(ic_t > 2.0 and ml_sh > mom_sh + 0.1)
    out = {
        "n_months": int(len(ics)), "ic_mean": float(ics.mean()), "ic_t": ic_t,
        "ml_sharpe": ml_sh, "mom_sharpe": mom_sh, "ew_sharpe": ew_sh,
        "ml_turnover": ml_turn, "gate_pass": gate_pass,
        "n_stocks": int(px.shape[1]),
    }
    GATE_PATH.write_text(json.dumps(out, indent=2))
    return out


def today_scores(px: pd.DataFrame | None = None) -> pd.Series:
    """Train on ALL history, predict the latest cross-section (relative ML score)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    px = load_panel() if px is None else px
    P = build_table(px)
    m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200,
                                      l2_regularization=1.0, min_samples_leaf=30, random_state=7)
    m.fit(P[FEATS], P["fwd_dm"])
    # build today's feature row
    rets = px.pct_change(fill_method=None)
    t = len(px) - 1
    valid = px.iloc[t].dropna().index
    f = pd.DataFrame(index=valid)
    f["mom_12_1"] = (px.iloc[t - 21] / px.iloc[t - 252] - 1).reindex(valid)
    f["mom_6_1"] = (px.iloc[t - 21] / px.iloc[t - 126] - 1).reindex(valid)
    f["mom_3"] = (px.iloc[t] / px.iloc[t - 63] - 1).reindex(valid)
    f["rev_1"] = -(px.iloc[t] / px.iloc[t - 21] - 1).reindex(valid)
    f["vol_60"] = rets.iloc[t - 60:t].std().reindex(valid)
    f["dist_hi"] = (px.iloc[t] / px.iloc[t - 252:t].max() - 1).reindex(valid)
    f = f.dropna().apply(_xs_z)
    return pd.Series(m.predict(f), index=f.index).sort_values(ascending=False)


class MLRankAgent:
    """Layer 10 — gated ML rank. Emits scores ONLY if the cached gate passed."""

    def run(self, tickers: list[str]) -> str:
        try:
            gate = json.loads(GATE_PATH.read_text()) if GATE_PATH.exists() else evaluate()
        except Exception as e:  # noqa: BLE001
            return f"[MLRankAgent] unavailable: {e}"
        head = (f"[MLRankAgent]  OOS rank-IC t={gate['ic_t']:+.2f} | net Sharpe "
                f"ML {gate['ml_sharpe']:.2f} vs momentum {gate['mom_sharpe']:.2f} vs "
                f"equal-weight {gate['ew_sharpe']:.2f} | GATE: "
                f"{'PASS' if gate['gate_pass'] else 'FAILED'}")
        if not gate["gate_pass"]:
            return (head + "\n  ML has NO measured out-of-sample edge on this universe — "
                    "DO NOT use it for selection; defer to valuation (Layer 9) + momentum. "
                    "(This is the honest result; ML is wired but gated off.)")
        try:
            sc = today_scores()
            want = {t.replace(".NS", "") for t in tickers}
            sel = sc[[s.replace(".NS", "") in want or s in want for s in sc.index]]
            top = "; ".join(f"{s}={v:+.3f}" for s, v in sel.head(8).items())
            return head + "\n  GATE PASSED — ML cross-sectional scores (higher=better): " + top
        except Exception as e:  # noqa: BLE001
            return head + f"\n  (score computation failed: {e})"


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if argv and argv[0] == "scores":
        sc = today_scores()
        print("Top 10 ML cross-sectional scores (today):")
        for s, v in sc.head(10).items():
            print(f"  {s:14s} {v:+.4f}")
        print("Bottom 5:")
        for s, v in sc.tail(5).items():
            print(f"  {s:14s} {v:+.4f}")
        return 0
    g = evaluate()
    print(json.dumps(g, indent=2))
    print("\nGATE:", "PASS — ML may tilt selection" if g["gate_pass"]
          else "FAILED — ML has no OOS edge; system ignores it (correct, honest behaviour).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
