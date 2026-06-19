"""
Provider-agnostic sentiment LLM (PRD §8.1, §19.1 Level-1, §19.2, §19.4, §19.6).

One grounded call per news item -> a validated §8.1 SentimentSignal. Pipeline:
    cache lookup -> RAG grounding -> LLM call -> JSON/schema validation
                 -> retry+repair on failure -> heuristic fallback -> cache store

Backends:
  * "digitalocean" — DigitalOcean Inference Router (public preview). Routes each
        news item to the best model for the task (sentiment, earnings, risk events).
        OpenAI-compatible; same key as direct serverless inference.
        Base URL https://inference.do-ai.run/v1/, model "router:tradite-news-router".
        PRIMARY. Create the router first: python -m sentiment.setup_router
  * "openai" / "anthropic" — direct providers.
  * "heuristic"   — no key; FinBERT/lexicon score + rules. Always available.
  * "mock"        — no key; deterministic, for testing the full path offline.

Set up DigitalOcean Inference Router:
    export DIGITALOCEAN_TOKEN=<your DO personal access token>
    python -m sentiment.setup_router          # one-time router provisioning

    export DIGITALOCEAN_INFERENCE_KEY=<your DO model access key>
    export TRADITE_LLM_BACKEND=digitalocean
    pip install openai

Router affinity (X-Model-Affinity header):
    When processing a batch of headlines for the same ticker, the router pins
    all requests in the batch to the same model after the first routing decision.
    This avoids KV-cache invalidation and saves ~45-80% on cached input token costs
    for multi-turn or multi-item sessions. Controlled by DO_USE_AFFINITY in config.py.
"""
from __future__ import annotations

import json
import uuid

import config

from .schema import NewsItem, SentimentSignal
from .finbert_scorer import FinBERTScorer
from .rag import build_grounding, grounding_to_prompt
from .cache import ResponseCache
from .market_grounding import CORE_PRINCIPLES, detect_event_tags
from .portfolio_grounding import PORTFOLIO_PRINCIPLES, PortfolioState

MODEL_VERSION = "sentiment-llm-v1"

REPAIR_SUFFIX = (" Your previous response was not valid JSON. Return ONLY a single "
                 "valid JSON object — no prose, no markdown, no code fences.")

# Base instructions + schema. The market-grounding CORE_PRINCIPLES (cardinal rules +
# confidence rubric) are injected so the model reasons from market facts, not priors.
# §19.6 leakage guard stays in force, and the JSON-only instruction stays last.
SYSTEM_PROMPT_BASE = (
    "You are a financial news analyst for Indian equities (NSE/BSE). Read the news item about a "
    "single stock and return ONLY a JSON object with these fields: "
    "{instrument:string, sentiment:number in [-1,1], "
    "direction:'long'|'short'|'neutral', horizon:'hours'|'days'|'weeks', "
    "event_tags:string[], thesis:string (<=2 sentences, grounded in the rules/base rates below), "
    "confidence:number in [0,1], sources:string[]}. "
    "Judge ONLY the sentiment expressed in THIS document as of its as-of date. "
    "You must NOT use any knowledge of what happened to the stock afterwards."  # §19.6
)

SYSTEM_PROMPT = (SYSTEM_PROMPT_BASE + "\n\n" + CORE_PRINCIPLES + "\n\n" + PORTFOLIO_PRINCIPLES
                 + "\n\nOutput the JSON object only.")

# Event tagging + market grounding now live in market_grounding.py (detect_event_tags),
# so the LLM path and the heuristic fallback share one tag vocabulary and one set of facts.


class SentimentLLM:
    def __init__(self, backend=None, model=None, use_rag=None, cache=None,
                 max_retries=None, price_df=None, corpus=None, mock_fail_first=False):
        self.backend = backend or config.LLM_BACKEND
        self.model = model
        self.use_rag = config.LLM_USE_RAG if use_rag is None else use_rag
        self.max_retries = config.LLM_MAX_RETRIES if max_retries is None else max_retries
        self.price_df = price_df          # optional point-in-time price frame for grounding
        self.corpus = corpus              # optional list[NewsItem] for related-headline grounding
        self.mock_fail_first = mock_fail_first
        self._finbert = FinBERTScorer(use_finbert=(self.backend == "heuristic"))
        use_cache = config.LLM_USE_CACHE if cache is None else cache
        self.cache = ResponseCache(config.LLM_CACHE_DIR, enabled=use_cache)
        self._client = None
        # Per-instrument affinity ID for the Inference Router.
        # Pinned after the first request so the router reuses the same model
        # for the rest of the batch (avoids KV-cache invalidation mid-session).
        self._affinity_ids: dict[str, str] = {}

    # ---- public API -------------------------------------------------------
    def analyze(self, item: NewsItem, affinity_id: str | None = None,
                portfolio_state: PortfolioState | None = None,
                sector: str | None = None) -> SentimentSignal:
        """Score one news item. affinity_id pins the router to one model for the batch.

        portfolio_state (point-in-time) makes the judgement portfolio-aware; sector is the
        candidate's sector for concentration checks. Neither sizes the trade (downstream).
        """
        grounding = (build_grounding(item, self.price_df, self.corpus,
                                     portfolio_state=portfolio_state, sector=sector)
                     if self.use_rag else {"document": f"{item.title}. {item.body}",
                                           "instrument": item.instrument,
                                           "published_at": item.published_at.isoformat()})
        prompt = grounding_to_prompt(grounding) if self.use_rag else grounding["document"]
        # SYSTEM_PROMPT carries the market/portfolio "laws" (CORE_PRINCIPLES). Include it
        # in the cache key so editing the laws auto-invalidates stale cached signals —
        # otherwise the cache (keyed only on the user prompt) silently serves old output
        # and your law edits appear to do nothing.
        parts = [self.backend, self.model or "", SYSTEM_PROMPT, prompt]

        cached = self.cache.get(parts)
        if cached is not None:
            return SentimentSignal(**cached)

        if self.backend == "heuristic":
            sig = self._heuristic(item, grounding)
        else:
            sig = None
            for attempt in range(self.max_retries + 1):
                raw, routed_model = self._generate(prompt, item, attempt, affinity_id)
                sig = self._validate(raw, item, routed_model)
                if sig is not None:
                    break
            if sig is None:
                sig = self._heuristic(item, grounding)  # graceful degradation

        self.cache.set(parts, sig.to_contract())
        return sig

    def analyze_many(self, items: list[NewsItem],
                     portfolio_state: PortfolioState | None = None,
                     sectors: dict[str, str] | None = None) -> list[SentimentSignal]:
        """Score a batch, using per-instrument affinity for the router.

        portfolio_state (point-in-time) is shared across the batch; sectors maps
        instrument -> sector for concentration checks.
        """
        results = []
        for item in items:
            affinity_id = self._get_affinity_id(item.instrument)
            sec = sectors.get(item.instrument) if sectors else None
            results.append(self.analyze(item, affinity_id=affinity_id,
                                        portfolio_state=portfolio_state, sector=sec))
        return results

    def _get_affinity_id(self, instrument: str) -> str | None:
        """Return a stable affinity ID for this instrument (or None if affinity is off)."""
        if not (self.backend == "digitalocean" and config.DO_USE_AFFINITY):
            return None
        if instrument not in self._affinity_ids:
            # Short UUID so it's easy to read in logs
            self._affinity_ids[instrument] = f"tradite-{instrument}-{uuid.uuid4().hex[:8]}"
        return self._affinity_ids[instrument]

    def list_models(self) -> list[str]:
        """Connectivity check — lists models your key can reach (digitalocean/openai)."""
        client = self._openai_client()
        return [m.id for m in client.models.list().data]

    # ---- backend dispatch -------------------------------------------------
    def _generate(self, prompt: str, item: NewsItem, attempt: int,
                  affinity_id: str | None = None) -> tuple[str, str | None]:
        """Return (raw_text, routed_model_id). routed_model_id is None for non-DO backends."""
        repair = attempt > 0
        if self.backend in ("digitalocean", "openai"):
            return self._call_openai_compatible(prompt, repair, affinity_id)
        if self.backend == "anthropic":
            return self._call_anthropic(prompt, repair), None
        if self.backend == "mock":
            return self._mock(prompt, item, attempt), None
        raise ValueError(f"Unknown backend: {self.backend}")

    def _openai_client(self):
        if self._client is not None:
            return self._client
        from openai import OpenAI  # pip install openai
        if self.backend == "digitalocean":
            if not config.DO_KEY:
                raise RuntimeError(
                    "Set DIGITALOCEAN_INFERENCE_KEY (DO model access key).\n"
                    "Then provision the router: python -m sentiment.setup_router"
                )
            self._client = OpenAI(base_url=config.DO_BASE_URL, api_key=config.DO_KEY)
        else:  # openai
            self._client = OpenAI()  # uses OPENAI_API_KEY
        return self._client

    def _call_openai_compatible(self, prompt: str, repair: bool,
                                affinity_id: str | None = None) -> tuple[str, str | None]:
        """Call DO Inference Router (or direct model) and return (text, routed_model_id).

        The router model string is "router:<name>" — a zero-code drop-in replacement
        for any plain model ID. The response's .model field tells us which model was
        actually selected, which we embed in the SentimentSignal model_version.

        X-Model-Affinity pins the routing decision to one model for the whole batch
        once the first request has been routed, saving KV-cache recomputation costs.
        """
        client = self._openai_client()
        model = self.model or (config.DO_MODEL if self.backend == "digitalocean"
                               else config.OPENAI_MODEL)
        system = SYSTEM_PROMPT + (REPAIR_SUFFIX if repair else "")

        extra_headers: dict[str, str] = {}
        if affinity_id and self.backend == "digitalocean":
            extra_headers["X-Model-Affinity"] = affinity_id

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            extra_headers=extra_headers or None,
        )
        # resp.model reflects the actual model the router selected (e.g. "anthropic-claude-haiku-4.5")
        routed_model = resp.model if resp.model != model else None
        return resp.choices[0].message.content or "", routed_model

    def _call_anthropic(self, prompt: str, repair: bool) -> str:
        import anthropic  # pip install anthropic; set ANTHROPIC_API_KEY
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=self.model or config.ANTHROPIC_MODEL,
            max_tokens=config.LLM_MAX_TOKENS,
            system=SYSTEM_PROMPT + (REPAIR_SUFFIX if repair else ""),
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    def _mock(self, prompt: str, item: NewsItem, attempt: int) -> str:
        """Deterministic stand-in. Returns invalid JSON first if mock_fail_first,
        to exercise the retry+repair path. Wrapped in a code fence to also test
        brace-extraction in _validate."""
        if self.mock_fail_first and attempt == 0:
            return "Sure — here is the analysis: {invalid, not json :: }"
        score = self._finbert.score_signed(f"{item.title}. {item.body}")
        direction = "long" if score > 0.1 else "short" if score < -0.1 else "neutral"
        obj = {
            "instrument": item.instrument,
            "sentiment": round(float(score), 3),
            "direction": direction,
            "horizon": "days",
            "event_tags": detect_event_tags(f"{item.title} {item.body}"),
            "thesis": f"[mock] {item.title[:90]}",
            "confidence": round(min(0.85, abs(score) * 0.8 + 0.2), 3),
            "sources": [item.url or item.source],
        }
        return "```json\n" + json.dumps(obj) + "\n```"

    # ---- heuristic backend & validation ----------------------------------
    def _heuristic(self, item: NewsItem, grounding: dict | None = None) -> SentimentSignal:
        text = f"{item.title}. {item.body}".strip()
        score = self._finbert.score_signed(text)
        tags = detect_event_tags(text)
        direction = "long" if score > 0.1 else "short" if score < -0.1 else "neutral"
        ctx = (grounding or {}).get("price_context_asof") or {}
        ctx_note = f" (ctx {ctx})" if ctx else ""
        return SentimentSignal(
            instrument=item.instrument,
            sentiment=round(float(score), 3),
            direction=direction,
            horizon="days",
            event_tags=tags,
            thesis=f"{item.title} -> net {direction} read on {item.instrument}{ctx_note}.",
            confidence=round(min(0.85, abs(score) * 0.8 + 0.2), 3),
            sources=[item.url] if item.url else [item.source],
            published_at=item.published_at,
            model_version=MODEL_VERSION + "-heuristic",
            channel=item.channel,
        )

    def _validate(self, raw: str, item: NewsItem,
                  routed_model: str | None = None) -> SentimentSignal | None:
        """Parse + schema-validate. Return None on any failure (caller retries).

        routed_model is the actual model the DO Inference Router selected (from resp.model).
        We embed it in model_version so every SentimentSignal carries an audit trail of
        which model produced it — useful for comparing router decisions across instruments.
        """
        try:
            s = raw[raw.find("{"): raw.rfind("}") + 1]
            data = json.loads(s)
            data.setdefault("instrument", item.instrument)
            data.setdefault("published_at", item.published_at.isoformat())
            data.setdefault("direction", "neutral")
            if isinstance(data.get("event_tags"), str):
                data["event_tags"] = [data["event_tags"]]
            if isinstance(data.get("sources"), str):
                data["sources"] = [data["sources"]]
            sent = float(data.get("sentiment", 0.0))
            data.setdefault("confidence", round(min(0.85, abs(sent) * 0.8 + 0.2), 3))
            # Enforce the cardinal-rule confidence ceiling (0.85) in CODE. Soft prompt
            # rules get ignored — cached signals showed 0.90-0.95 despite the rubric —
            # so clamp authoritatively here rather than trusting the model.
            data["confidence"] = max(0.0, min(0.85, float(data["confidence"])))
            # Include routed model in version string for auditability
            if routed_model:
                data["model_version"] = f"{MODEL_VERSION}-{self.backend}[{routed_model}]"
            else:
                data["model_version"] = f"{MODEL_VERSION}-{self.backend}"
            data["channel"] = item.channel
            return SentimentSignal(**data)
        except Exception:  # noqa: BLE001
            return None
