"""
MLSignalAgent — computes quantitative ML signals from price history and
injects them into the multi-agent trading prompt as Layer 6.

Six signal families, all computed from free yfinance data:

  1. REGIME          — market regime (bull_quiet / bull_volatile / bear / sideways)
                       derived from Nifty 50 trend + rolling vol
  2. TECHNICALS      — RSI-14, MACD (12/26/9), Bollinger Band %B per stock
  3. MEAN-REVERSION  — logistic regression P(+5% recovery in 20d) trained on
                       8 years of historical (setup → outcome) pairs per stock
  4. SECTOR RANK     — where each stock sits within its NSE sector on momentum
                       (uses Industry column from nifty500_constituents.csv)
  5. CORRELATION     — pairwise correlation matrix; flags over-concentrated picks
  6. GARCH           — forward volatility estimate via GARCH(1,1); falls back to
                       EWMA if the `arch` package is not installed

Usage (standalone):
    python -m sentiment.ml_signals ICICIBANK.NS HDFCBANK.NS RELIANCE.NS
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config

warnings.filterwarnings("ignore")

# ── constants ─────────────────────────────────────────────────────────────────
MARKET_TICKER  = "idx_NSEI"          # Nifty 50 index — used for regime
FWD_DAYS       = 20                  # forward return window for mean-reversion
RECOVERY_PCT   = 5.0                 # minimum gain to call a "recovery"
MIN_TRAIN_ROWS = 252                 # min history rows to train logistic model
NIFTY500_CSV   = config.DATA_DIR / "nifty500_constituents.csv"


# ══════════════════════════════════════════════════════════════════════════════
# Shared price loader
# ══════════════════════════════════════════════════════════════════════════════

def _load_close(ticker: str, years: int = 8) -> pd.Series | None:
    """Load close series from CSV cache or download fresh."""
    csv = config.DATA_DIR / f"{ticker}__yfinance.csv"

    def _read(path: Path) -> pd.Series | None:
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                sample = df.iloc[0, 0] if not df.empty else None
                try:
                    float(sample)
                except (TypeError, ValueError):
                    df2 = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
                    df2.columns = [c[0].lower() for c in df2.columns]
                    df = df2
            col = next((c for c in ("Close", "close") if c in df.columns), None)
            if col is None:
                return None
            s = df[col].dropna()
            if not isinstance(s.index, pd.DatetimeIndex):
                s.index = pd.to_datetime(s.index, errors="coerce")
                s = s[s.index.notna()]
            return s.sort_index()
        except Exception:
            return None

    if csv.exists():
        s = _read(csv)
        if s is not None and len(s) >= 60:
            return s

    try:
        import yfinance as yf
        df = yf.download(ticker, period=f"{years}y", auto_adjust=True, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        df.to_csv(csv)
        col = next((c for c in ("Close", "close") if c in df.columns), None)
        if col is None:
            return None
        s = df[col].dropna()
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index, errors="coerce")
            s = s[s.index.notna()]
        return s.sort_index()
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 1. REGIME
# ══════════════════════════════════════════════════════════════════════════════

def _compute_regime(market: pd.Series) -> dict[str, Any]:
    """
    Classify current market regime using Nifty 50:
      - Trend  : price vs 200-day SMA
      - Momentum: 1-month return
      - Vol    : 20d realised vol vs 60d average
    """
    if market is None or len(market) < 200:
        return {"regime": "unknown", "detail": "insufficient market history"}

    now      = float(market.iloc[-1])
    sma200   = float(market.tail(200).mean())
    sma50    = float(market.tail(50).mean())
    m1_ret   = (now / float(market.iloc[-21]) - 1) * 100 if len(market) >= 21 else 0.0
    vol20    = float(market.pct_change().tail(20).std() * (252 ** 0.5) * 100)
    vol60    = float(market.pct_change().tail(60).std() * (252 ** 0.5) * 100)
    vol_ratio = vol20 / vol60 if vol60 > 0 else 1.0

    above_200 = now > sma200
    above_50  = now > sma50
    high_vol  = vol_ratio > 1.25

    if above_200 and above_50 and not high_vol:
        regime = "BULL_QUIET"
        hint   = "Trend intact, vol subdued — momentum strategies work best"
    elif above_200 and high_vol:
        regime = "BULL_VOLATILE"
        hint   = "Trend up but choppy — reduce size, widen stops"
    elif not above_200 and m1_ret < -3:
        regime = "BEAR"
        hint   = "Below 200 SMA, downward momentum — prefer cash/shorts, tight stops on longs"
    elif not above_200 and abs(m1_ret) < 2:
        regime = "SIDEWAYS"
        hint   = "Range-bound below 200 SMA — mean-reversion plays better than momentum"
    else:
        regime = "RECOVERY"
        hint   = "Below 200 SMA but momentum turning — early-cycle longs, small size"

    return {
        "regime"   : regime,
        "nifty_vs_200sma": round((now / sma200 - 1) * 100, 2),
        "nifty_m1" : round(m1_ret, 2),
        "vol20"    : round(vol20, 1),
        "vol_ratio": round(vol_ratio, 2),
        "hint"     : hint,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. TECHNICALS — RSI, MACD, Bollinger Band %B
# ══════════════════════════════════════════════════════════════════════════════

def _rsi(s: pd.Series, period: int = 14) -> float:
    delta = s.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - 100 / (1 + rs)
    return float(rsi.iloc[-1]) if not rsi.empty else float("nan")


def _macd(s: pd.Series, fast: int = 12, slow: int = 26,
          signal: int = 9) -> dict[str, float]:
    ema_fast   = s.ewm(span=fast,   adjust=False).mean()
    ema_slow   = s.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return {
        "macd"      : round(float(macd_line.iloc[-1]), 4),
        "signal"    : round(float(signal_line.iloc[-1]), 4),
        "histogram" : round(float(histogram.iloc[-1]), 4),
        "cross"     : "bullish" if histogram.iloc[-1] > 0 and histogram.iloc[-2] <= 0
                      else "bearish" if histogram.iloc[-1] < 0 and histogram.iloc[-2] >= 0
                      else "neutral",
    }


def _bollinger(s: pd.Series, period: int = 20, std: float = 2.0) -> dict[str, float]:
    ma    = s.rolling(period).mean()
    band  = s.rolling(period).std()
    upper = ma + std * band
    lower = ma - std * band
    pct_b = (s - lower) / (upper - lower)   # 0 = at lower band, 1 = at upper band
    width = (upper - lower) / ma * 100       # band width as % of price
    return {
        "pct_b"   : round(float(pct_b.iloc[-1]), 3),
        "bb_width": round(float(width.iloc[-1]), 2),
        "signal"  : "oversold" if pct_b.iloc[-1] < 0.2
                    else "overbought" if pct_b.iloc[-1] > 0.8
                    else "neutral",
    }


def _technicals_block(ticker: str, close: pd.Series) -> str:
    sym  = ticker.replace(".NS", "")
    rsi  = _rsi(close)
    macd = _macd(close)
    bb   = _bollinger(close)

    rsi_label = (
        "OVERSOLD (<30)" if rsi < 30 else
        "OVERBOUGHT (>70)" if rsi > 70 else
        f"neutral ({rsi:.1f})"
    )
    return (
        f"{sym}: RSI={rsi:.1f} [{rsi_label}]  "
        f"MACD histogram={macd['histogram']:+.4f} [{macd['cross']}]  "
        f"BB %B={bb['pct_b']:.2f} [{bb['signal']}]  BB-width={bb['bb_width']:.1f}%"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. MEAN-REVERSION PROBABILITY
#    Logistic regression: (1W, 1M, 3M, vol, RSI) → P(+5% in 20d)
# ══════════════════════════════════════════════════════════════════════════════

def _build_feature_row(close: pd.Series, i: int) -> list[float] | None:
    """Build feature vector at index i."""
    if i < 252:
        return None
    now = float(close.iloc[i])
    try:
        w1  = (now / float(close.iloc[i - 5])   - 1) * 100
        m1  = (now / float(close.iloc[i - 21])  - 1) * 100
        m3  = (now / float(close.iloc[i - 63])  - 1) * 100
        y1  = (now / float(close.iloc[i - 252]) - 1) * 100
        vol = float(close.iloc[max(0, i-30): i+1].pct_change().std() * (252**0.5) * 100)
        rsi = _rsi(close.iloc[max(0, i-30): i+1])
        hi52 = float(close.iloc[i - 252: i + 1].max())
        pct_hi = (now / hi52 - 1) * 100
    except Exception:
        return None
    if not all(np.isfinite(v) for v in [w1, m1, m3, y1, vol, rsi, pct_hi]):
        return None
    return [w1, m1, m3, y1, vol, rsi, pct_hi]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _logistic_train(X: np.ndarray, y: np.ndarray,
                    lr: float = 0.01, epochs: int = 300,
                    l2: float = 0.01) -> np.ndarray:
    """
    Pure-numpy logistic regression via gradient descent.
    X: (n, p) standardised features; y: (n,) binary labels.
    Returns weight vector (p,).
    """
    n, p = X.shape
    w    = np.zeros(p)
    for _ in range(epochs):
        pred = _sigmoid(X @ w)
        grad = X.T @ (pred - y) / n + l2 * w
        w   -= lr * grad
    return w


def _mean_reversion_prob(close: pd.Series) -> dict[str, Any]:
    """
    Train a numpy logistic regression on historical (features → 20d recovery)
    pairs, then predict probability for the current state.
    No external ML library required.
    """
    if len(close) < MIN_TRAIN_ROWS + FWD_DAYS:
        return {"prob": None, "n_train": 0, "detail": "insufficient history"}

    n = len(close)
    X, y = [], []
    for i in range(MIN_TRAIN_ROWS, n - FWD_DAYS):
        feat = _build_feature_row(close, i)
        if feat is None:
            continue
        fwd_ret = (float(close.iloc[i + FWD_DAYS]) / float(close.iloc[i]) - 1) * 100
        X.append(feat)
        y.append(1 if fwd_ret >= RECOVERY_PCT else 0)

    if len(X) < 50:
        return {"prob": None, "n_train": len(X), "detail": "too few training samples"}

    X_arr  = np.array(X, dtype=float)
    y_arr  = np.array(y, dtype=float)

    # Standardise
    mu     = X_arr.mean(axis=0)
    sigma  = X_arr.std(axis=0) + 1e-8
    X_sc   = (X_arr - mu) / sigma

    w      = _logistic_train(X_sc, y_arr)

    # Current state
    cur_feat = _build_feature_row(close, n - 1)
    if cur_feat is None:
        return {"prob": None, "n_train": len(X), "detail": "cannot compute current features"}

    cur_sc  = (np.array(cur_feat) - mu) / sigma
    prob    = float(_sigmoid(cur_sc @ w))
    base_rate = float(y_arr.mean())

    return {
        "prob"      : round(prob, 3),
        "base_rate" : round(base_rate, 3),
        "n_train"   : len(X),
        "edge"      : round(prob - base_rate, 3),
        "detail"    : (
            f"P(+{RECOVERY_PCT:.0f}% in {FWD_DAYS}d) = {prob*100:.1f}%  "
            f"(base rate {base_rate*100:.1f}%  n={len(X)})"
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. SECTOR RELATIVE STRENGTH
# ══════════════════════════════════════════════════════════════════════════════

def _load_sector_map() -> dict[str, str]:
    """Return {ticker_without_NS: industry} from cached Nifty 500 CSV."""
    if not NIFTY500_CSV.exists():
        return {}
    try:
        df = pd.read_csv(NIFTY500_CSV)
        col = next((c for c in df.columns if "industry" in c.lower()), None)
        if col is None:
            return {}
        return {str(row["Symbol"]).strip(): str(row[col]).strip()
                for _, row in df.iterrows()}
    except Exception:
        return {}


def _sector_ranks(tickers: list[str],
                  prices_df: pd.DataFrame | None = None) -> dict[str, dict]:
    """
    For each ticker, find all Nifty 500 peers in the same sector,
    compute their 1M returns, and rank the ticker within that group.
    """
    sector_map = _load_sector_map()
    if not sector_map:
        return {}

    # Build sector → peer-returns map from screener parquet if available
    parquet = config.DATA_DIR / "screener_prices.parquet"
    if prices_df is None and parquet.exists():
        try:
            prices_df = pd.read_parquet(parquet)
        except Exception:
            prices_df = None

    if prices_df is None or prices_df.empty:
        return {}

    # Compute 1M return for every ticker in the parquet
    n = len(prices_df)
    if n < 21:
        return {}
    m1_returns: dict[str, float] = {}
    for col in prices_df.columns:
        s = prices_df[col].dropna()
        if len(s) >= 21:
            try:
                r = (float(s.iloc[-1]) / float(s.iloc[-21]) - 1) * 100
                m1_returns[col] = r
            except Exception:
                pass

    results = {}
    for ticker in tickers:
        sym = ticker.replace(".NS", "")
        industry = sector_map.get(sym)
        if not industry:
            results[ticker] = {"sector": "unknown", "rank": None, "n_peers": 0}
            continue

        # Find peers in same sector
        peers_ret = {
            t: m1_returns[t]
            for t, ret in m1_returns.items()
            if sector_map.get(t.replace(".NS", "")) == industry
        }
        if len(peers_ret) < 2:
            results[ticker] = {"sector": industry, "rank": None, "n_peers": len(peers_ret)}
            continue

        sorted_peers = sorted(peers_ret.items(), key=lambda x: x[1], reverse=True)
        rank_idx = next((i for i, (t, _) in enumerate(sorted_peers) if t == ticker), None)
        if rank_idx is None:
            results[ticker] = {"sector": industry, "rank": None, "n_peers": len(peers_ret)}
            continue

        rank       = rank_idx + 1
        n_peers    = len(sorted_peers)
        pct_rank   = round((1 - rank / n_peers) * 100, 1)  # 100 = top of sector
        sector_ret = sum(r for _, r in sorted_peers) / n_peers

        results[ticker] = {
            "sector"     : industry,
            "rank"       : rank,
            "n_peers"    : n_peers,
            "pct_rank"   : pct_rank,       # percentile within sector
            "sector_m1"  : round(sector_ret, 2),
            "stock_m1"   : round(peers_ret.get(ticker, float("nan")), 2),
            "vs_sector"  : round(peers_ret.get(ticker, 0) - sector_ret, 2),
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 5. CORRELATION MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def _correlation_block(tickers: list[str],
                       close_map: dict[str, pd.Series],
                       window: int = 60) -> str:
    """
    Compute rolling 60-day correlation matrix and flag high-correlation clusters.
    """
    if len(tickers) < 2:
        return "  (need ≥2 stocks for correlation)"

    # Align series on common dates
    aligned = pd.DataFrame(
        {t: close_map[t] for t in tickers if t in close_map}
    ).dropna()

    if len(aligned) < window:
        return "  (insufficient overlapping history for correlation)"

    ret = aligned.tail(window).pct_change().dropna()
    corr = ret.corr()

    lines = ["  Pairwise 60d correlation (|ρ| > 0.7 = high concentration risk):"]
    warned = False
    syms = [t.replace(".NS", "") for t in tickers if t in close_map]
    for i, ti in enumerate(syms):
        for j, tj in enumerate(syms):
            if j <= i:
                continue
            rho = float(corr.iloc[i, j]) if i < len(corr) and j < len(corr) else float("nan")
            flag = " ⚠️ HIGH" if abs(rho) > 0.7 else ""
            lines.append(f"    {ti} ↔ {tj}: ρ={rho:+.2f}{flag}")
            if abs(rho) > 0.7:
                warned = True

    if warned:
        lines.append("  → High-correlation pairs are NOT independent bets. Size down or drop one.")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 6. GARCH(1,1) VOLATILITY FORECAST
# ══════════════════════════════════════════════════════════════════════════════

def _garch_vol(close: pd.Series) -> dict[str, Any]:
    """
    Fit GARCH(1,1) and forecast 5-day ahead annualised vol.
    Falls back to EWMA if `arch` is not installed.
    """
    returns = close.pct_change().dropna() * 100   # in %
    if len(returns) < 100:
        return {"forecast": None, "method": "insufficient data"}

    try:
        from arch import arch_model
        am     = arch_model(returns, vol="Garch", p=1, q=1, dist="normal", rescale=True)
        res    = am.fit(disp="off", show_warning=False)
        fc     = res.forecast(horizon=5)
        var_5d = float(fc.variance.iloc[-1].mean())       # mean variance over 5 days
        vol_5d = (var_5d ** 0.5) * (252 ** 0.5)           # annualised
        return {
            "forecast": round(vol_5d, 1),
            "method"  : "GARCH(1,1)",
            "realized": round(float(returns.tail(20).std() * (252 ** 0.5)), 1),
        }
    except ImportError:
        pass
    except Exception:
        pass

    # EWMA fallback (λ = 0.94, RiskMetrics standard)
    lam    = 0.94
    ew_var = float(returns.ewm(alpha=1 - lam, adjust=False).var().iloc[-1])
    vol_ew = (ew_var ** 0.5) * (252 ** 0.5)
    return {
        "forecast": round(vol_ew, 1),
        "method"  : "EWMA(λ=0.94)",
        "realized": round(float(returns.tail(20).std() * (252 ** 0.5)), 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Public API — MLSignalAgent
# ══════════════════════════════════════════════════════════════════════════════

class MLSignalAgent:
    """
    Computes all ML signal families and returns a formatted string
    ready to be injected as Layer 6 in the trading prompt.
    """

    def run(self, tickers: list[str]) -> str:
        # Load price series for all tickers + market index
        close_map: dict[str, pd.Series] = {}
        for t in tickers:
            s = _load_close(t, years=8)
            if s is not None and len(s) >= 60:
                close_map[t] = s

        market = _load_close(MARKET_TICKER, years=5)

        sections: list[str] = []

        # ── 1. Regime ─────────────────────────────────────────────────────
        reg = _compute_regime(market)
        sections.append(
            f"MARKET REGIME: {reg['regime']}\n"
            f"  Nifty vs 200SMA: {reg.get('nifty_vs_200sma', '?'):+.1f}%  "
            f"1M return: {reg.get('nifty_m1', '?'):+.1f}%  "
            f"Vol ratio (20d/60d): {reg.get('vol_ratio', '?'):.2f}\n"
            f"  Strategy hint: {reg.get('hint', '')}"
        )

        # ── 2. Technicals ─────────────────────────────────────────────────
        tech_lines = []
        for t in tickers:
            if t not in close_map:
                tech_lines.append(f"  {t.replace('.NS','')}: no data")
                continue
            try:
                tech_lines.append("  " + _technicals_block(t, close_map[t]))
            except Exception as e:
                tech_lines.append(f"  {t.replace('.NS','')}: error ({e})")
        sections.append("TECHNICALS (RSI / MACD / Bollinger):\n" + "\n".join(tech_lines))

        # ── 3. Mean-reversion probability ─────────────────────────────────
        mr_lines = []
        for t in tickers:
            if t not in close_map:
                mr_lines.append(f"  {t.replace('.NS','')}: no data")
                continue
            try:
                mr = _mean_reversion_prob(close_map[t])
                if mr["prob"] is not None:
                    edge_label = (
                        "STRONG EDGE" if mr["edge"] > 0.15 else
                        "EDGE"        if mr["edge"] > 0.05 else
                        "WEAK"        if mr["edge"] > -0.05 else
                        "AGAINST"
                    )
                    mr_lines.append(
                        f"  {t.replace('.NS',''):12s}  {mr['detail']}  [{edge_label}]"
                    )
                else:
                    mr_lines.append(f"  {t.replace('.NS',''):12s}  {mr['detail']}")
            except Exception as e:
                mr_lines.append(f"  {t.replace('.NS','')}: error ({e})")
        sections.append(
            f"MEAN-REVERSION PROBABILITY (logistic regression, {FWD_DAYS}d horizon):\n"
            + "\n".join(mr_lines)
        )

        # ── 4. Sector relative strength ───────────────────────────────────
        sector_data = _sector_ranks(tickers)
        sec_lines = []
        for t in tickers:
            d = sector_data.get(t, {})
            if not d or d.get("rank") is None:
                sec_lines.append(f"  {t.replace('.NS',''):12s}  sector rank: unavailable")
                continue
            pct  = d["pct_rank"]
            flag = " 🔼 TOP QUARTILE" if pct >= 75 else " 🔽 BOTTOM QUARTILE" if pct <= 25 else ""
            sec_lines.append(
                f"  {t.replace('.NS',''):12s}  sector={d['sector'][:25]}  "
                f"rank={d['rank']}/{d['n_peers']}  "
                f"({pct:.0f}th pct)  "
                f"stock_1M={d['stock_m1']:+.1f}%  sector_avg={d['sector_m1']:+.1f}%  "
                f"vs_sector={d['vs_sector']:+.1f}%{flag}"
            )
        sections.append("SECTOR RELATIVE STRENGTH (1M, within Nifty 500 peers):\n"
                        + "\n".join(sec_lines))

        # ── 5. Correlation matrix ─────────────────────────────────────────
        corr_block = _correlation_block(tickers, close_map)
        sections.append("CORRELATION RISK:\n" + corr_block)

        # ── 6. GARCH volatility forecast ──────────────────────────────────
        garch_lines = []
        for t in tickers:
            if t not in close_map:
                garch_lines.append(f"  {t.replace('.NS','')}: no data")
                continue
            try:
                g = _garch_vol(close_map[t])
                if g["forecast"] is not None:
                    vs = g["forecast"] - g["realized"]
                    trend = "↑ EXPANDING" if vs > 3 else "↓ CONTRACTING" if vs < -3 else "→ stable"
                    garch_lines.append(
                        f"  {t.replace('.NS',''):12s}  "
                        f"realized={g['realized']:.1f}%  "
                        f"forecast={g['forecast']:.1f}% [{trend}]  "
                        f"method={g['method']}"
                    )
                else:
                    garch_lines.append(f"  {t.replace('.NS','')}: {g.get('method','?')}")
            except Exception as e:
                garch_lines.append(f"  {t.replace('.NS','')}: error ({e})")
        sections.append(
            "VOLATILITY FORECAST (GARCH/EWMA, 5-day annualised):\n"
            + "\n".join(garch_lines)
        )

        header = "[MLSignalAgent — 6 signal families]\n"
        return header + "\n\n".join(sections)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None):
    import sys
    argv = argv or sys.argv[1:]
    tickers = [a if a.endswith(".NS") else f"{a}.NS" for a in argv] or config.UNIVERSE
    agent = MLSignalAgent()
    print(agent.run(tickers))


if __name__ == "__main__":
    main()
