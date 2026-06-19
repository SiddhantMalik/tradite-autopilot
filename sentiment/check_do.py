"""
DigitalOcean Inference Router connectivity check (run from ml_lab/):

    export DIGITALOCEAN_INFERENCE_KEY=<your DO model access key>
    python -m sentiment.check_do

Lists reachable models, then scores three sample headlines that cover the
three router tasks (general sentiment, earnings analysis, risk event) through
the Inference Router. The model_version field in each signal tells you which
model the router actually selected. Costs a few tokens.

If the router hasn't been created yet:
    export DIGITALOCEAN_TOKEN=<your DO personal access token>
    python -m sentiment.setup_router
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import config
from .llm_client import SentimentLLM
from .schema import NewsItem

# Three items designed to hit each router task
SAMPLE_ITEMS = [
    NewsItem(
        instrument="NSE:INFY",
        title="Infosys shares rally after brokerage upgrade on strong deal pipeline",
        body="",
        source="check_do",
        published_at=datetime.now(timezone.utc),
        # Expected router task: financial_news_sentiment
    ),
    NewsItem(
        instrument="NSE:RELIANCE",
        title="Reliance Q4 profit beats estimates by 12%; board raises FY guidance",
        body="Net profit Rs 21,243 crore vs estimate Rs 18,950 crore. Revenue growth 9% YoY.",
        source="check_do",
        published_at=datetime.now(timezone.utc),
        # Expected router task: earnings_analysis
    ),
    NewsItem(
        instrument="NSE:HDFCBANK",
        title="SEBI opens probe into HDFC Bank's derivative trading desk",
        body="Regulator investigating alleged front-running by three senior traders.",
        source="check_do",
        published_at=datetime.now(timezone.utc),
        # Expected router task: risk_event_detection
    ),
]


def main() -> int:
    if not config.DO_KEY:
        print(
            "DIGITALOCEAN_INFERENCE_KEY is not set.\n"
            "Create a model access key in the DigitalOcean control panel "
            "(Inference -> Model Access Keys), then:\n"
            "  export DIGITALOCEAN_INFERENCE_KEY=...\n"
            "  export TRADITE_LLM_BACKEND=digitalocean\n\n"
            "If you haven't created the router yet:\n"
            "  export DIGITALOCEAN_TOKEN=<your DO PAT>\n"
            "  python -m sentiment.setup_router"
        )
        return 1

    llm = SentimentLLM(backend="digitalocean", cache=False)
    print(f"Base URL : {config.DO_BASE_URL}")
    print(f"Model    : {config.DO_MODEL}  (router:{config.DO_ROUTER_NAME} by default)")
    print(f"Affinity : {'enabled' if config.DO_USE_AFFINITY else 'disabled'}\n")

    # ── 1. Connectivity: list reachable models ────────────────────────────
    try:
        models = llm.list_models()
    except Exception as e:  # noqa: BLE001
        print(f"Could not list models (check key/network): {e}")
        return 1
    print(f"Reachable models ({len(models)}):")
    for m in models[:40]:
        print("  -", m)
    print()

    # ── 2. Router: score three items that cover the three tasks ───────────
    router_is_active = config.DO_MODEL.startswith("router:")
    print(
        f"{'Router' if router_is_active else 'Direct model'} inference test "
        f"({len(SAMPLE_ITEMS)} items) …\n"
        + ("Note: 'routed via' shows which model the router selected.\n"
           if router_is_active else "")
    )

    for item in SAMPLE_ITEMS:
        try:
            sig = llm.analyze(item)
            routed = ""
            if router_is_active and "[" in sig.model_version:
                routed_model = sig.model_version.split("[")[1].rstrip("]")
                routed = f"  routed via: {routed_model}"
            print(f"  [{item.instrument}] {item.title[:70]}")
            print(f"    direction={sig.direction:8s}  sentiment={sig.sentiment:+.3f}"
                  f"  confidence={sig.confidence:.2f}{routed}")
        except Exception as e:  # noqa: BLE001
            print(f"  [{item.instrument}] FAILED: {e}")
            return 1

    print("\nFull signal for last item:")
    print(json.dumps(
        llm.analyze(SAMPLE_ITEMS[-1]).to_contract(), indent=2
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
