"""Select news with simple baselines and a budget-constrained novelty/coverage objective."""

from __future__ import annotations

import math
from collections.abc import Callable

import pandas as pd

from newsalpha.ingestion import article_token_count, parse_embedding

METHODS = ("recency", "relevance", "mmr", "novelty_coverage")


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("All article embedding vectors must have the same dimensions")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _require_relevance(frame: pd.DataFrame) -> list[float]:
    if "relevance_score" not in frame.columns or frame["relevance_score"].isna().any():
        raise ValueError("This selection method requires a numeric relevance_score for every article")
    return [float(score) for score in frame["relevance_score"]]


def _require_embeddings(frame: pd.DataFrame) -> list[tuple[float, ...]]:
    if "embedding" not in frame.columns:
        raise ValueError("This selection method requires an embedding column")
    vectors = [
        parse_embedding(value, str(article_id))
        for value, article_id in zip(frame["embedding"], frame["article_id"])
    ]
    if any(vector is None for vector in vectors):
        raise ValueError("This selection method requires an embedding for every candidate article")
    concrete_vectors = [vector for vector in vectors if vector is not None]
    if len({len(vector) for vector in concrete_vectors}) != 1:
        raise ValueError("All article embedding vectors must have the same dimensions")
    return concrete_vectors


def _normalize(scores: list[float]) -> list[float]:
    minimum, maximum = min(scores), max(scores)
    if maximum == minimum:
        return [1.0] * len(scores)
    return [(score - minimum) / (maximum - minimum) for score in scores]


def select_news(
    candidates: pd.DataFrame,
    method: str,
    token_budget: int,
    mmr_relevance_weight: float = 0.7,
    historical_articles: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Greedily select articles under the same token cap and return them in selection order."""
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {', '.join(METHODS)}")
    if token_budget <= 0:
        raise ValueError("token_budget must be a positive integer")
    if not 0 <= mmr_relevance_weight <= 1:
        raise ValueError("mmr_relevance_weight must be between 0 and 1")
    if candidates.empty:
        raise ValueError("Cannot select from an empty candidate set")

    frame = candidates.copy().reset_index(drop=True)
    frame["_tokens"] = frame.apply(article_token_count, axis=1)
    if (frame["_tokens"] > token_budget).all():
        raise ValueError("No candidate article fits within the token budget")

    scores: list[float] | None = None
    vectors: list[tuple[float, ...]] | None = None
    historical_vectors: list[tuple[float, ...]] = []
    if method == "relevance":
        scores = _normalize(_require_relevance(frame))
    elif method in ("mmr", "novelty_coverage"):
        scores = _normalize(_require_relevance(frame))
        vectors = _require_embeddings(frame)
        if method == "novelty_coverage" and historical_articles is not None and not historical_articles.empty:
            historical_vectors = _require_embeddings(historical_articles)
            if len({len(vector) for vector in [*vectors, *historical_vectors]}) != 1:
                raise ValueError("Candidate and historical embedding dimensions must match")

    timestamps = [pd.Timestamp(value).timestamp() for value in frame["published_at"]]
    timestamp_scores = _normalize(timestamps)

    remaining = set(range(len(frame)))
    chosen: list[int] = []
    total_tokens = 0
    priorities: dict[int, float] = {}

    while remaining:
        eligible = [
            index for index in remaining
            if total_tokens + int(frame.iloc[index]["_tokens"]) <= token_budget
        ]
        if not eligible:
            break

        def priority(index: int) -> float:
            row = frame.iloc[index]
            relevance = scores[index] if scores is not None else 0.0
            if method == "recency":
                value = pd.Timestamp(row["published_at"]).timestamp()
            elif method == "relevance":
                value = relevance
            elif method == "mmr":
                assert vectors is not None
                redundancy = max(
                    (_cosine(vectors[index], vectors[prior]) for prior in chosen),
                    default=0.0,
                )
                value = mmr_relevance_weight * relevance - (1 - mmr_relevance_weight) * redundancy
            else:
                assert vectors is not None
                historical_similarity = max(
                    (_cosine(vectors[index], prior) for prior in historical_vectors),
                    default=0.0,
                )
                novelty = 1 - historical_similarity
                coverage = 1 - max(
                    (_cosine(vectors[index], vectors[prior]) for prior in chosen),
                    default=0.0,
                )
                freshness = timestamp_scores[index]
                value = 0.25 * relevance + 0.25 * novelty + 0.25 * coverage + 0.25 * freshness
            priorities[index] = value
            return value

        best = max(
            eligible,
            key=lambda index: (
                priority(index),
                pd.Timestamp(frame.iloc[index]["published_at"]).timestamp(),
                str(frame.iloc[index]["article_id"]),
            ),
        )
        chosen.append(best)
        remaining.remove(best)
        total_tokens += int(frame.iloc[best]["_tokens"])

    result = frame.iloc[chosen].copy()
    result["selection_rank"] = range(1, len(result) + 1)
    result["selection_score"] = [priorities[index] for index in chosen]
    result["token_count_used"] = result["_tokens"].astype(int)
    return result.drop(columns=["_tokens"]).reset_index(drop=True)
