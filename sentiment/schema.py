"""
The sentiment contract (PRD §8.1 / §19.4).

Structured output is the boundary between the LLM and the rest of the system:
nothing downstream ever consumes free-text. Every signal is validated against
SentimentSignal before it can become a feature or a gate. Reject/retry on
violation — never let unparsed model prose reach the strategy layer.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Direction = Literal["long", "short", "neutral"]
Horizon = Literal["hours", "days", "weeks"]
Channel = Literal["news", "social", "filing"]


class NewsItem(BaseModel):
    """A single point-in-time document mapped to an instrument."""
    model_config = ConfigDict(protected_namespaces=())

    instrument: str                      # e.g. "NSE:INFY"
    title: str
    body: str = ""
    source: str = ""
    url: str = ""
    published_at: datetime               # POINT-IN-TIME: never visible before this
    channel: Channel = "news"


class SentimentSignal(BaseModel):
    """Exactly the §8.1 output contract, plus provenance fields."""
    model_config = ConfigDict(protected_namespaces=())

    instrument: str
    sentiment: float = Field(ge=-1.0, le=1.0)      # -1 bearish .. +1 bullish
    direction: Direction
    horizon: Horizon = "days"
    event_tags: list[str] = Field(default_factory=list)
    thesis: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[str] = Field(default_factory=list)
    published_at: datetime
    model_version: str = "sentiment-llm-v1"
    channel: Channel = "news"

    def to_contract(self) -> dict:
        """JSON matching PRD §8.1 (datetimes as ISO strings)."""
        d = self.model_dump()
        d["published_at"] = self.published_at.isoformat()
        return d
