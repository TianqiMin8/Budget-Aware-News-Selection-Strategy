"""Load, validate, and as-of filter candidate news from a documented CSV dataset."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_NEWS_COLUMNS = {
    "article_id",
    "ticker",
    "published_at",
    "title",
    "text",
    "url",
}


def load_news(path: str | Path) -> pd.DataFrame:
    """Read news CSV and normalize timestamp, text, and optional numeric fields."""
    frame = pd.read_csv(path)
    missing = REQUIRED_NEWS_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"News CSV is missing required columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("News CSV contains no articles")

    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True, errors="coerce")
    invalid_timestamps = frame["published_at"].isna()
    if invalid_timestamps.any():
        rows = (frame.index[invalid_timestamps] + 2).tolist()
        raise ValueError(f"Invalid published_at timestamp on CSV row(s): {rows[:10]}")

    for column in ("article_id", "ticker", "title", "text", "url"):
        frame[column] = frame[column].fillna("").astype(str).str.strip()
        if (frame[column] == "").any():
            raise ValueError(f"News CSV contains an empty value in required column '{column}'")
    if frame["article_id"].duplicated().any():
        duplicate_ids = frame.loc[frame["article_id"].duplicated(), "article_id"].tolist()
        raise ValueError(f"article_id values must be unique; duplicates include: {duplicate_ids[:10]}")

    if "token_count" in frame.columns:
        frame["token_count"] = pd.to_numeric(frame["token_count"], errors="coerce")
        invalid_tokens = frame["token_count"].notna() & (
            (frame["token_count"] <= 0) | (frame["token_count"] % 1 != 0)
        )
        if invalid_tokens.any():
            raise ValueError("token_count values must be positive whole numbers")
    if "relevance_score" in frame.columns:
        frame["relevance_score"] = pd.to_numeric(frame["relevance_score"], errors="coerce")
        if frame["relevance_score"].isna().any():
            raise ValueError("relevance_score must be numeric and non-empty when the column is present")

    return frame


def filter_as_of(
    frame: pd.DataFrame,
    ticker: str,
    as_of: str | pd.Timestamp,
) -> pd.DataFrame:
    """Return one ticker's articles published no later than the UTC cutoff."""
    cutoff = pd.to_datetime(as_of, utc=True, errors="coerce")
    if pd.isna(cutoff):
        raise ValueError(f"Invalid as-of timestamp: {as_of}")
    selected = frame.loc[
        (frame["ticker"].str.casefold() == ticker.casefold())
        & (frame["published_at"] <= cutoff)
    ].copy()
    if selected.empty:
        raise ValueError(f"No articles for {ticker} were published by {cutoff.isoformat()}")
    return selected.sort_values("published_at", ascending=False).reset_index(drop=True)


def article_token_count(row: pd.Series) -> int:
    """Use a supplied count or a clearly approximate four-characters-per-token estimate."""
    value: Any = row.get("token_count")
    if pd.notna(value):
        count = int(value)
        if count <= 0:
            raise ValueError(f"token_count must be positive for article {row['article_id']}")
        return count
    content = f"{row['title']} {row['text']}".strip()
    return max(1, math.ceil(len(content) / 4))


def parse_embedding(value: Any, article_id: str) -> tuple[float, ...] | None:
    """Parse an optional JSON embedding and reject malformed or non-finite vectors."""
    if pd.isna(value) or value == "":
        return None
    try:
        vector = json.loads(value) if isinstance(value, str) else value
        result = tuple(float(component) for component in vector)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid embedding for article {article_id}: {error}") from error
    if not result or not all(math.isfinite(component) for component in result):
        raise ValueError(f"Embedding for article {article_id} must be a non-empty finite vector")
    return result
