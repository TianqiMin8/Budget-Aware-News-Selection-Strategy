"""Define an auditable event-extraction interface without coupling the project to one LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class ExtractedEvent:
    """One event grounded in a selected source article."""

    article_id: str
    event_type: str
    description: str
    impact_direction: int
    confidence: float
    evidence_quote: str


@dataclass(frozen=True)
class ExtractionResult:
    """Events plus measurable provider usage for one extraction request."""

    events: tuple[ExtractedEvent, ...]
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    latency_seconds: float
    model_name: str


class EventExtractor(Protocol):
    """Provider adapters implement this contract and must retain source article IDs."""

    def extract(self, articles: Sequence[dict[str, str]]) -> ExtractionResult:
        """Extract events only from supplied articles; no provider is bundled by default."""
        ...


def validate_extraction(result: ExtractionResult, allowed_article_ids: set[str]) -> None:
    """Reject events that cite unsupplied articles or contain invalid direction/confidence."""
    for event in result.events:
        if event.article_id not in allowed_article_ids:
            raise ValueError(f"Event cites an article that was not supplied: {event.article_id}")
        if event.impact_direction not in (-1, 0, 1):
            raise ValueError(f"Invalid impact_direction for article {event.article_id}")
        if not 0 <= event.confidence <= 1:
            raise ValueError(f"Event confidence must be between 0 and 1 for {event.article_id}")
        if not event.evidence_quote.strip():
            raise ValueError(f"Event must include an evidence quote for article {event.article_id}")
