"""
Provision the Tradite Inference Router on DigitalOcean (run once).

Uses the DO REST API — NOT the model access key — so set DIGITALOCEAN_TOKEN
(a personal access token with write scope) before running:

    export DIGITALOCEAN_TOKEN=<your DO PAT>
    python -m sentiment.setup_router

The router is named "tradite-news-router" and has three tasks tuned for
financial news processing:

  1. financial_news_sentiment  — main task: score any news headline/body for
     directional bias and confidence. Policy: cost efficiency (many items/run).

  2. earnings_analysis         — earnings beats/misses, guidance changes, EPS
     surprises. Policy: optimal (DO-benchmarked best model for this task type).

  3. risk_event_detection      — regulatory probes, legal issues, credit events,
     management changes. Policy: speed (quick binary classification).

Fallback model: openai-gpt-oss-20b (cheap open-source, always available).

After provisioning, point the pipeline at the router:
    export TRADITE_LLM_BACKEND=digitalocean
    # TRADITE_DO_ROUTER defaults to "tradite-news-router" in config.py

To delete and re-create the router run with --recreate:
    python -m sentiment.setup_router --recreate
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

import config

ROUTER_NAME = config.DO_ROUTER_NAME  # "tradite-news-router"
DO_API = "https://api.digitalocean.com/v2"

ROUTER_PAYLOAD = {
    "name": ROUTER_NAME,
    "description": (
        "Routes financial news items to the best LLM for each task: "
        "general sentiment scoring, earnings analysis, and risk-event detection."
    ),
    "policies": [
        # ── Task 1: General news sentiment ──────────────────────────────
        {
            "custom_task": {
                "name": "financial_news_sentiment",
                "description": (
                    "Score financial news articles and headlines for stock market "
                    "sentiment, directional bias (long/short/neutral), and confidence. "
                    "Return structured JSON with sentiment score, direction, horizon, "
                    "event tags, and thesis."
                ),
            },
            "models": [
                "anthropic-claude-haiku-4.5",   # fast + finance-aware
                "openai-gpt-4o-mini",           # cheap fallback in pool
                "llama3.3-70b-instruct",        # open-source option
            ],
            "selection_policy": {"prefer": "cheapest"},
        },
        # ── Task 2: Earnings & guidance analysis ─────────────────────────
        {
            "custom_task": {
                "name": "earnings_analysis",
                "description": (
                    "Analyze earnings reports, EPS beats or misses, revenue surprises, "
                    "guidance raises or cuts, management commentary, and forward outlook "
                    "statements for listed Indian equities."
                ),
            },
            "models": [
                "anthropic-claude-haiku-4.5",
                "openai-gpt-4o-mini",
                "llama3.3-70b-instruct",
            ],
            "selection_policy": {"prefer": "optimal"},
        },
        # ── Task 3: Risk event detection ─────────────────────────────────
        {
            "custom_task": {
                "name": "risk_event_detection",
                "description": (
                    "Detect and classify risk events in financial news: regulatory probes, "
                    "SEBI investigations, legal disputes, credit downgrades, management "
                    "changes, fraud allegations, or force-majeure events for NSE/BSE stocks."
                ),
            },
            "models": [
                "openai-gpt-4o-mini",
                "llama3.3-70b-instruct",
                "anthropic-claude-haiku-4.5",
            ],
            "selection_policy": {"prefer": "fastest"},
        },
    ],
    # Fallback: if no task matches, use a cheap open-source model
    "fallback_models": ["openai-gpt-oss-20b", "llama3.3-70b-instruct"],
}


def _do_request(method: str, path: str, body: dict | None = None) -> dict:
    token = config.DO_TOKEN
    if not token:
        raise RuntimeError(
            "DIGITALOCEAN_TOKEN is not set.\n"
            "Create a personal access token at https://cloud.digitalocean.com/account/api/tokens\n"
            "  export DIGITALOCEAN_TOKEN=<token>"
        )
    url = f"{DO_API}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read()) if resp.length else {}
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode(errors="replace")
        raise RuntimeError(f"DO API {method} {url} -> {exc.code}: {err_body}") from exc


def _list_routers() -> list[dict]:
    resp = _do_request("GET", "/gen-ai/models/routers")
    return resp.get("routers") or resp.get("data") or []


def _delete_router(router_id: str) -> None:
    _do_request("DELETE", f"/gen-ai/models/routers/{router_id}")


def _create_router() -> dict:
    resp = _do_request("POST", "/gen-ai/models/routers", ROUTER_PAYLOAD)
    return resp.get("router") or resp


def main(recreate: bool = False) -> int:
    print(f"Checking for existing router '{ROUTER_NAME}' …")
    routers = _list_routers()
    existing = next((r for r in routers if r.get("name") == ROUTER_NAME), None)

    if existing:
        if not recreate:
            print(
                f"Router '{ROUTER_NAME}' already exists (id={existing.get('id')}).\n"
                "Nothing to do. Pass --recreate to delete and re-create it."
            )
            _print_usage()
            return 0
        print(f"--recreate: deleting existing router (id={existing['id']}) …")
        _delete_router(existing["id"])
        print("Deleted.")

    print(f"Creating router '{ROUTER_NAME}' with {len(ROUTER_PAYLOAD['policies'])} tasks …")
    router = _create_router()
    router_id = router.get("id") or router.get("router", {}).get("id", "<unknown>")
    print(f"Router created successfully  (id={router_id})\n")
    print(json.dumps(router, indent=2))
    print()
    _print_usage()
    return 0


def _print_usage() -> None:
    print(
        "─" * 60 + "\n"
        "To use the router in Tradite:\n"
        f"  export TRADITE_LLM_BACKEND=digitalocean\n"
        f"  # TRADITE_DO_ROUTER defaults to '{ROUTER_NAME}'\n"
        f"  # Model field sent to API:  router:{ROUTER_NAME}\n"
        "─" * 60
    )


if __name__ == "__main__":
    recreate = "--recreate" in sys.argv
    raise SystemExit(main(recreate=recreate))
