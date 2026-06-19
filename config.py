"""
Central configuration for the Tradite ML research harness.

Everything you'd want to tune lives here so experiments are reproducible.
Maps to Tradite PRD sections: data (§9), features/targets (§8.2),
walk-forward validation (§19.7), cost-aware backtest (§7.4 / §12).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Load tradite/.env automatically — project-scoped, no ~/.zshrc changes needed.
# Keys defined there take effect for every `python -m sentiment.*` invocation.
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT.parent / ".env", override=False)  # override=False: real env vars win
except ImportError:
    pass  # python-dotenv not installed; fall back to plain os.getenv
DATA_DIR = ROOT / "data_cache"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# Universe
# NSE tickers use the ".NS" suffix on Yahoo Finance; BSE uses ".BO".
# ^NSEI is the Nifty 50 index (used as the buy-and-hold benchmark).
# ----------------------------------------------------------------------
UNIVERSE = ["RELIANCE.NS", "INFY.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
BENCHMARK = "^NSEI"

# ----------------------------------------------------------------------
# Data window
# ----------------------------------------------------------------------
START = "2015-01-01"
END = None  # None -> today

# Data source: "yfinance" (free, real, needs network) or "synthetic"
# (a pure random walk used to PROVE the harness is honest — see data_loader.py).
DATA_SOURCE = os.getenv("TRADITE_DATA_SOURCE", "yfinance")

# ----------------------------------------------------------------------
# The prediction problem
# We predict the SIGN of the forward HORIZON-day return. Classification,
# not price regression — predicting the next price level is the classic
# trap that just reproduces a lagged random walk (see RESEARCH_NOTES.md).
# ----------------------------------------------------------------------
HORIZON = 5            # forward trading days
LABEL_THRESHOLD = 0.0  # forward return > threshold => label 1 ("up")

# ----------------------------------------------------------------------
# Walk-forward / purged cross-validation (PRD §19.7)
# ----------------------------------------------------------------------
N_SPLITS = 6           # number of expanding-window walk-forward folds
EMBARGO = HORIZON      # bars to drop between train and test (>= label horizon)

# ----------------------------------------------------------------------
# Cost model — round-trip is charged on every position change.
# Rough Indian-retail all-in (brokerage + STT + exchange + slippage).
# TUNE THIS to your real costs; edge that survives only at zero cost is fake.
# ----------------------------------------------------------------------
COST_PER_TRADE = 0.0015  # 15 bps, charged each time the position flips

# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    min_child_weight=5,
    objective="binary:logistic",
    eval_metric="logloss",
    n_jobs=4,
)

LSTM_PARAMS = dict(
    seq_len=30,     # lookback window of daily bars fed to the LSTM
    hidden=32,
    layers=1,
    dropout=0.2,
    epochs=15,
    lr=1e-3,
    batch=64,
)

SEED = 7

# ----------------------------------------------------------------------
# LLM sentiment backend (sentiment/llm_client.py)
# Primary real backend is DigitalOcean Inference Router (public preview) —
# a single OpenAI-compatible endpoint that routes each news item to the
# best-fit model based on the task (sentiment, earnings, risk events).
#
# Quick start:
#   export DIGITALOCEAN_INFERENCE_KEY=<your DO model access key>
#   export TRADITE_LLM_BACKEND=digitalocean
#
# First time: create the router (needs DIGITALOCEAN_TOKEN, not the inference key):
#   export DIGITALOCEAN_TOKEN=<your DO personal access token>
#   python -m sentiment.setup_router
# ----------------------------------------------------------------------
LLM_BACKEND = os.getenv("TRADITE_LLM_BACKEND", "heuristic")  # heuristic|digitalocean|openai|anthropic|mock

# DigitalOcean Inference Router (public preview)
# Base URL is identical to direct serverless inference; router is addressed
# via "router:<name>" in the model field — a zero-code drop-in replacement.
DO_BASE_URL = os.getenv("TRADITE_DO_BASE_URL", "https://inference.do-ai.run/v1/")

# Router name created by sentiment/setup_router.py.
# Set TRADITE_DO_ROUTER to a plain model ID (e.g. "openai-gpt-4o-mini") to
# skip routing and hit one model directly.
DO_ROUTER_NAME = os.getenv("TRADITE_DO_ROUTER", "tradite-news-router")
DO_MODEL = os.getenv("TRADITE_DO_MODEL", f"router:{DO_ROUTER_NAME}")

# The FINAL multi-agent decision is ONE call, so its token cost is negligible and
# model quality matters most — a small general model (e.g. what "router:general"
# picks, ~gemma-31B) produces vague, hedge-y plans. The decision uses a STRONG
# reasoning model; everything else (per-headline scoring, book/historical agents)
# stays on the cheap router.
#
# IMPORTANT (verified live on this account, Jun 2026): the Anthropic and
# OpenAI-COMMERCIAL models (anthropic-claude-*, openai-gpt-4o*) are LISTED by the
# API but return 403 "not available for your subscription tier" — they are NOT
# callable on the current DO plan. The DigitalOcean-hosted OPEN models work:
#   openai-gpt-oss-120b (117B, default) | llama3.3-70b-instruct | deepseek-3.2 (680B)
# If you upgrade the DO subscription to unlock Anthropic, set
#   TRADITE_DO_DECISION_MODEL=anthropic-claude-opus-4.8
# Check exactly what your key can CALL (not just list): `python -m sentiment.check_do`.
DO_DECISION_MODEL = os.getenv("TRADITE_DO_DECISION_MODEL", "openai-gpt-oss-120b")

# Tried in order if the primary decision model errors or isn't available on your
# tier. Ends on router:general (known-good on this account) so it never hard-fails.
DO_DECISION_FALLBACKS = [
    m.strip() for m in os.getenv(
        "TRADITE_DO_DECISION_FALLBACKS",
        "llama3.3-70b-instruct,deepseek-3.2,router:general",
    ).split(",") if m.strip()
]

# Accept either env name (DO docs use DIGITAL_OCEAN_MODEL_ACCESS_KEY in examples).
DO_KEY = os.getenv("DIGITALOCEAN_INFERENCE_KEY") or os.getenv("DIGITAL_OCEAN_MODEL_ACCESS_KEY")

# DO personal access token — only needed by setup_router.py to provision the router.
DO_TOKEN = os.getenv("DIGITALOCEAN_TOKEN")

# Model affinity: pin each ticker's batch to the same model for the whole run.
# Reduces KV-cache invalidation when processing many headlines per instrument.
# Set to False to let the router route every request independently.
DO_USE_AFFINITY = os.getenv("TRADITE_DO_USE_AFFINITY", "true").lower() != "false"

OPENAI_MODEL = os.getenv("TRADITE_OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.getenv("TRADITE_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 400
LLM_MAX_RETRIES = 2
LLM_USE_RAG = True
LLM_USE_CACHE = True
LLM_CACHE_DIR = ROOT / "sentiment" / "_llm_cache"

