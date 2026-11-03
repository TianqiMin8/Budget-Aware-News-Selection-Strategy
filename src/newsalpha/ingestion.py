"""Load, validate, and as-of filter candidate news from a documented CSV dataset."""

from __future__ import annotations

import json
import math
import re
import zipfile
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

ARCHIVED_NEWS_COLUMNS = {
    "ID",
    "Date",
    "Article_title",
    "Stock_symbol",
    "Url",
    "Article",
    "Lsa_summary",
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


def load_company_news_archive(path: str | Path, ticker: str) -> pd.DataFrame:
    """Load one company's archived CSV and normalize its columns to the project schema."""
    symbol = ticker.upper()
    if not re.fullmatch(r"[A-Z]+", symbol):
        raise ValueError(f"Invalid ticker symbol: {ticker!r}")
    member = f"financial_news_project/{symbol}_news.csv"
    with zipfile.ZipFile(path) as archive:
        try:
            with archive.open(member) as source:
                frame = pd.read_csv(source, usecols=ARCHIVED_NEWS_COLUMNS)
        except KeyError as error:
            raise ValueError(f"News archive does not contain data for ticker {symbol}") from error

    missing = ARCHIVED_NEWS_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Archived news is missing columns: {', '.join(sorted(missing))}")
    frame = frame.rename(
        columns={
            "ID": "article_id",
            "Date": "published_at",
            "Article_title": "title",
            "Stock_symbol": "ticker",
            "Url": "url",
            "Article": "text",
        }
    )
    frame["article_id"] = frame["article_id"].map(_normalize_article_id)
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True, errors="coerce")
    frame["ticker"] = frame["ticker"].fillna("").astype(str).str.strip()
    frame["title"] = frame["title"].fillna("").astype(str).str.strip()
    frame["text"] = frame["text"].fillna("").astype(str).str.strip()
    frame["url"] = frame["url"].fillna("").astype(str).str.strip()
    frame["text"] = frame["text"].where(frame["text"] != "", frame["Lsa_summary"].fillna(""))
    frame["text"] = frame["text"].where(frame["text"] != "", frame["title"])
    frame = frame.drop(columns=["Lsa_summary"])
    if frame["published_at"].isna().any():
        raise ValueError(f"Archived news contains invalid timestamps for ticker {symbol}")
    if frame["article_id"].eq("").any() or frame["article_id"].duplicated().any():
        raise ValueError(f"Archived news article IDs must be non-empty and unique for {symbol}")
    if frame[["title", "text", "url"]].eq("").any().any():
        raise ValueError(f"Archived news has an empty title, text, or URL for ticker {symbol}")
    if not frame["ticker"].str.casefold().eq(symbol.casefold()).all():
        raise ValueError(f"News archive member for {symbol} contains another ticker")
    return frame.sort_values("published_at").reset_index(drop=True)


def _normalize_article_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def load_stock_prices_archive(path: str | Path) -> pd.DataFrame:
    """Load the combined daily-price CSV from the project's stock-price archive."""
    member = "stock_prices/all_stocks_with_returns.csv"
    with zipfile.ZipFile(path) as archive:
        try:
            with archive.open(member) as source:
                frame = pd.read_csv(source)
        except KeyError as error:
            raise ValueError(f"Stock-price archive does not contain {member}") from error
    required = {
        "Date",
        "Stock_symbol",
        "Adj Close",
        "Volume",
        "Future_Return",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Archived stock prices are missing columns: {', '.join(sorted(missing))}")
    frame = frame.rename(columns={"Date": "date", "Stock_symbol": "ticker"})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    frame["ticker"] = frame["ticker"].fillna("").astype(str).str.strip().str.upper()
    if frame["date"].isna().any() or frame["ticker"].eq("").any():
        raise ValueError("Archived stock prices contain invalid dates or ticker symbols")
    return frame.sort_values(["ticker", "date"]).reset_index(drop=True)


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
