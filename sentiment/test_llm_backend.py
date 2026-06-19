"""
No-key test of the full LLM backend path using the deterministic 'mock' backend
(run from ml_lab/):  python -m sentiment.test_llm_backend

Exercises every mechanism the real DigitalOcean/OpenAI/Anthropic backends use:
structured-output validation, retry+repair, RAG grounding, caching, and the
heuristic fallback. If this passes, swapping in a real key changes only the
network call, not the control flow.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pandas as pd

from .schema import NewsItem
from .llm_client import SentimentLLM
from .rag import build_grounding, grounding_to_prompt


def _item(title, body="", instrument="NSE:INFY"):
    return NewsItem(instrument=instrument, title=title, body=body, source="test",
                    url="", published_at=datetime(2026, 6, 12, tzinfo=timezone.utc))


def _price_df():
    idx = pd.bdate_range("2026-05-01", periods=30)
    close = pd.Series(range(100, 130), index=idx, dtype=float)
    return pd.DataFrame({"close": close})


def main():
    ok = 0

    # 1) valid structured output -> SentimentSignal
    llm = SentimentLLM(backend="mock", cache=False)
    sig = llm.analyze(_item("Infosys Q4 profit beats estimates; raises guidance"))
    assert sig.direction == "long" and sig.sentiment > 0, sig
    assert "earnings_beat" in sig.event_tags or "guidance_raise" in sig.event_tags
    assert sig.model_version.endswith("mock")
    print(f"1. structured-output parse OK  -> {sig.direction} {sig.sentiment:+.2f} {sig.event_tags}")
    ok += 1

    # 2) retry + repair: first attempt returns invalid JSON, retry succeeds
    llm2 = SentimentLLM(backend="mock", cache=False, mock_fail_first=True, max_retries=2)
    sig2 = llm2.analyze(_item("Brokerage upgrades stock to buy on strong order book"))
    assert sig2.model_version.endswith("mock"), "should have recovered via retry, not fallback"
    assert sig2.direction == "long"
    print(f"2. retry+repair OK            -> recovered on retry: {sig2.direction} {sig2.sentiment:+.2f}")
    ok += 1

    # 3) fallback: always-invalid (fail first, zero retries) -> heuristic
    llm3 = SentimentLLM(backend="mock", cache=False, mock_fail_first=True, max_retries=0)
    sig3 = llm3.analyze(_item("Regulator opens probe into accounting practices"))
    assert sig3.model_version.endswith("heuristic"), "should have fallen back"
    assert sig3.direction == "short"
    print(f"3. graceful fallback OK       -> heuristic: {sig3.direction} {sig3.sentiment:+.2f}")
    ok += 1

    # 4) RAG grounding: as-of price context + prior headlines flow into the prompt
    corpus = [_item("Earlier: company guides cautiously")]  # 2026-06-12 same day -> excluded
    older = NewsItem(instrument="NSE:INFY", title="Older: new large deal signed", source="t",
                     published_at=datetime(2026, 6, 1, tzinfo=timezone.utc))
    g = build_grounding(_item("Q4 results beat"), price_df=_price_df(), corpus=[older])
    prompt = grounding_to_prompt(g)
    assert g["price_context_asof"].get("last_close") == 129.0, g["price_context_asof"]
    assert "Older: new large deal signed" in prompt
    assert "what happens" not in prompt.lower()
    print(f"4. RAG grounding OK           -> price ctx {g['price_context_asof']}, "
          f"{len(g['prior_headlines'])} prior headline(s)")
    ok += 1

    # 5) cache: a fresh unique item is a miss then a hit (no recompute)
    llm5 = SentimentLLM(backend="mock", cache=True)
    it = _item(f"Unique headline {uuid.uuid4()}")
    llm5.analyze(it)
    hits_before = llm5.cache.hits
    llm5.analyze(it)
    assert llm5.cache.hits == hits_before + 1, llm5.cache.stats()
    print(f"5. caching OK                 -> {llm5.cache.stats()}")
    ok += 1

    print(f"\nALL {ok}/5 LLM-backend tests passed. Real DigitalOcean path differs "
          f"only in the network call (python -m sentiment.check_do to verify your key).")


if __name__ == "__main__":
    main()
