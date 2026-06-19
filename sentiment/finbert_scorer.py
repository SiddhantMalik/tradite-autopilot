"""
Fast calibrated sentiment score (PRD §18, §19.3).

FinBERT gives a quick, finance-tuned sentiment signal. It loads lazily; if
transformers/torch or the model weights aren't available, we fall back to a small
finance lexicon so the pipeline never hard-fails. Both return a SIGNED score in
[-1, +1] (negative = bearish, positive = bullish).

Note (from current research): FinBERT (110M params) is fast but insensitive to
numbers and degrades on complex sentences; fine-tuned finance LLMs now beat it by
~8-9% on sentiment. Use FinBERT for the cheap calibrated score and the LLM
(llm_client.py) for event tags + thesis. See RESEARCH_NOTES.md.
"""
from __future__ import annotations

import re

_FINBERT_MODEL = "ProsusAI/finbert"

# Minimal finance lexicon fallback (no dependency). Not a substitute for FinBERT —
# just keeps the demo alive offline.
_POS = {
    "beat", "beats", "upgrade", "upgrades", "raises", "raise", "surge", "surges",
    "rally", "rallies", "gain", "gains", "growth", "profit", "wins", "win",
    "strong", "outperform", "bullish", "record", "dividend", "buy", "jumps",
}
_NEG = {
    "miss", "misses", "downgrade", "downgrades", "cut", "cuts", "slips", "slump",
    "fall", "falls", "drop", "drops", "weak", "loss", "probe", "fraud", "lawsuit",
    "bearish", "warning", "pressure", "caution", "cautious", "sell", "plunge",
}


class FinBERTScorer:
    def __init__(self, use_finbert: bool = True):
        self.use_finbert = use_finbert
        self._pipe = None
        self._tried = False

    def _load(self):
        if self._tried:
            return self._pipe
        self._tried = True
        if not self.use_finbert:
            return None
        try:
            from transformers import pipeline
            self._pipe = pipeline("sentiment-analysis", model=_FINBERT_MODEL,
                                  truncation=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [finbert] falling back to lexicon ({type(e).__name__}: {e})")
            self._pipe = None
        return self._pipe

    def score_signed(self, text: str) -> float:
        """Signed sentiment in [-1, +1] for one text."""
        pipe = self._load()
        if pipe is not None:
            out = pipe(text[:512])[0]
            label, prob = out["label"].lower(), float(out["score"])
            if label.startswith("pos"):
                return prob
            if label.startswith("neg"):
                return -prob
            return 0.0
        return self._lexicon(text)

    @staticmethod
    def _lexicon(text: str) -> float:
        toks = re.findall(r"[a-z]+", text.lower())
        p = sum(t in _POS for t in toks)
        n = sum(t in _NEG for t in toks)
        if p + n == 0:
            return 0.0
        return (p - n) / (p + n)

    def score_many(self, texts: list[str]) -> list[float]:
        return [self.score_signed(t) for t in texts]
