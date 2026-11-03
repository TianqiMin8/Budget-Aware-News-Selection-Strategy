"""End-to-end, offline-capable news-selection and return-prediction demo."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

from newsalpha.events import (
    ExtractionResult,
    MockEventExtractor,
    validate_extraction,
)
from newsalpha.evaluation import regression_metrics
from newsalpha.features import aggregate_event_features, build_model_frame
from newsalpha.ingestion import (
    article_token_count,
    load_company_news_archive,
    load_stock_prices_archive,
)
from newsalpha.model import RidgeReturnModel
from newsalpha.selection import METHODS, select_news

PRICE_FEATURE_COLUMNS = (
    "Return_1d",
    "Return_5d",
    "Volume_Ratio",
    "Volatility_5d",
)
EVENT_FEATURE_COLUMNS = (
    "event_count",
    "positive_event_count",
    "negative_event_count",
    "weighted_impact",
)
TRAIN_END = datetime(2023, 1, 1).date()
VALIDATION_END = datetime(2023, 7, 1).date()
MARKET_CUTOFF_TIME = time(16, 0)
EASTERN = ZoneInfo("America/New_York")

COMPANY_NAMES = {
    "AAPL": "Apple",
    "ABBV": "AbbVie",
    "AMD": "Advanced Micro Devices",
    "AMZN": "Amazon",
    "BA": "Boeing",
    "CAT": "Caterpillar",
    "DIS": "Disney",
    "GS": "Goldman Sachs",
    "MS": "Morgan Stanley",
    "MSFT": "Microsoft",
    "NKE": "Nike",
    "NVDA": "Nvidia",
    "TSLA": "Tesla",
    "V": "Visa",
}


@dataclass(frozen=True)
class DemoExperiment:
    ticker: str
    prediction_date: str
    selection_method: str
    selected_articles: pd.DataFrame
    extracted_events: pd.DataFrame
    baseline_metrics: dict[str, float | int]
    news_metrics: dict[str, float | int]
    baseline_prediction: float
    news_prediction: float
    actual_return: float
    mock_usage: ExtractionResult


def build_market_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Recreate the notebook's four price-volume features and discard split leakage."""
    required = {"date", "ticker", "Adj Close", "Volume", "Future_Return"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"Prices are missing columns: {', '.join(sorted(missing))}")
    frame = prices.sort_values(["ticker", "date"]).copy()
    grouped = frame.groupby("ticker", sort=False)
    frame["Return_1d"] = grouped["Adj Close"].pct_change(fill_method=None)
    frame["Return_5d"] = frame["Adj Close"] / grouped["Adj Close"].shift(5) - 1
    frame["Volume_MA5"] = grouped["Volume"].transform(
        lambda values: values.rolling(5, min_periods=5).mean()
    )
    frame["Volume_Ratio"] = frame["Volume"] / frame["Volume_MA5"]
    frame["Volatility_5d"] = frame.groupby("ticker", sort=False)["Return_1d"].transform(
        lambda values: values.rolling(5, min_periods=5).std()
    )
    frame["next_date"] = grouped["date"].shift(-1)
    frame["_split"] = np.select(
        [
            frame["date"] < TRAIN_END,
            (frame["date"] >= TRAIN_END) & (frame["date"] < VALIDATION_END),
            frame["date"] >= VALIDATION_END,
        ],
        ["train", "validation", "test"],
        default="outside",
    )
    frame["_next_split"] = np.select(
        [
            frame["next_date"] < TRAIN_END,
            (frame["next_date"] >= TRAIN_END) & (frame["next_date"] < VALIDATION_END),
            frame["next_date"] >= VALIDATION_END,
        ],
        ["train", "validation", "test"],
        default="outside",
    )
    usable = frame[
        list(PRICE_FEATURE_COLUMNS)
        + ["Future_Return", "next_date", "_split", "_next_split"]
    ].notna().all(axis=1)
    frame = frame.loc[usable & frame["_split"].eq(frame["_next_split"])].copy()
    frame = frame.rename(columns={"Future_Return": "target_return"})
    return frame[["ticker", "date", *PRICE_FEATURE_COLUMNS, "target_return"]].reset_index(
        drop=True
    )


def _assign_prediction_dates(news: pd.DataFrame, market_dates: pd.Series) -> pd.DataFrame:
    """Map each article to the first market close that occurs after publication."""
    dates = pd.Index(sorted(market_dates.unique()))
    if dates.empty:
        raise ValueError("No market dates are available for the selected ticker")
    cutoff_timestamps = pd.DatetimeIndex(
        [
            pd.Timestamp(datetime.combine(day, MARKET_CUTOFF_TIME, tzinfo=EASTERN)).tz_convert(
                "UTC"
            )
            for day in dates
        ]
    )
    published = pd.to_datetime(news["published_at"], utc=True, errors="coerce")
    positions = np.searchsorted(cutoff_timestamps.asi8, published.array.asi8, side="left")
    valid = positions < len(dates)
    mapped = news.loc[valid].copy()
    mapped["date"] = dates.to_numpy()[positions[valid]]
    return mapped.reset_index(drop=True)


def _prepare_selector_features(news: pd.DataFrame, ticker: str) -> pd.DataFrame:
    frame = news.copy()
    documents = [
        f"{title} {text[:2000]}"
        for title, text in zip(frame["title"].fillna(""), frame["text"].fillna(""))
    ]
    vectorizer = HashingVectorizer(
        n_features=64,
        alternate_sign=False,
        norm=None,
        stop_words="english",
    )
    vectors = vectorizer.transform(documents).toarray()
    frame["embedding"] = [json.dumps(vector.tolist()) for vector in vectors]

    company = COMPANY_NAMES.get(ticker.upper(), ticker)
    terms = [term.casefold() for term in re.findall(r"[A-Za-z0-9]+", company)]
    symbol_pattern = re.compile(rf"(?<![A-Z]){re.escape(ticker.upper())}(?![A-Z])")

    def relevance(row: pd.Series) -> float:
        title = str(row["title"])
        text = f"{title} {str(row['text'])[:2000]}"
        company_mentions = sum(text.casefold().count(term) for term in terms)
        symbol_mentions = len(symbol_pattern.findall(text.upper()))
        return 0.5 + min(company_mentions + symbol_mentions, 5) / 10

    frame["relevance_score"] = frame.apply(relevance, axis=1)
    return frame


def _select_articles_by_day(
    news: pd.DataFrame,
    method: str,
    token_budget: int,
) -> pd.DataFrame:
    if method not in METHODS:
        raise ValueError(f"Unknown selection method: {method}")
    if token_budget <= 0:
        raise ValueError("token_budget must be a positive integer")
    frame = _prepare_selector_features(news, str(news["ticker"].iloc[0]))
    selected_days: list[pd.DataFrame] = []
    for day, candidates in frame.groupby("date", sort=True):
        if not candidates["text"].notna().any():
            continue
        if not candidates.apply(article_token_count, axis=1).le(token_budget).any():
            continue
        historical = frame.loc[
            (frame["date"] < day)
            & (frame["date"] >= day - pd.Timedelta(days=30))
        ]
        selection = select_news(
            candidates,
            method,
            token_budget,
            historical_articles=historical,
        )
        if not selection.empty:
            selected_days.append(selection)
    if not selected_days:
        raise ValueError(
            "No articles fit the selected token budget; increase the budget or choose another date"
        )
    return pd.concat(selected_days, ignore_index=True)


def _event_frame(
    selected_articles: pd.DataFrame,
    extraction: ExtractionResult,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    allowed_ids = set(selected_articles["article_id"].astype(str))
    validate_extraction(extraction, allowed_ids)
    article_keys = selected_articles.set_index("article_id")[["ticker", "date"]]
    event_rows = [
        {
            "ticker": article_keys.loc[event.article_id, "ticker"],
            "date": article_keys.loc[event.article_id, "date"],
            "article_id": event.article_id,
            "impact_direction": event.impact_direction,
            "confidence": event.confidence,
        }
        for event in extraction.events
    ]
    events = pd.DataFrame(
        event_rows,
        columns=["ticker", "date", "article_id", "impact_direction", "confidence"],
    )
    return events, aggregate_event_features(events)


def run_ticker_demo(
    news_archive: str | Path,
    stock_archive: str | Path,
    ticker: str,
    prediction_date: str,
    selection_method: str = "novelty_coverage",
    token_budget: int = 1500,
) -> DemoExperiment:
    """Run the actual-data demo with mock extraction and identical temporal splits."""
    symbol = ticker.upper()
    prices = load_stock_prices_archive(stock_archive)
    ticker_prices = prices.loc[prices["ticker"] == symbol]
    market = build_market_features(ticker_prices)
    if market.empty:
        raise ValueError(f"No usable price rows found for ticker {symbol}")
    news = load_company_news_archive(news_archive, symbol)
    news = _assign_prediction_dates(news, ticker_prices["date"])
    if news.empty:
        raise ValueError(f"No archived news can be mapped to a market date for {symbol}")
    news = news.loc[news["date"] < VALIDATION_END].copy()
    if news.empty:
        raise ValueError(f"No train or validation news is available for {symbol}")

    selected = _select_articles_by_day(news, selection_method, token_budget)
    mock_extractor = MockEventExtractor()
    payload = [
        {
            "article_id": str(row.article_id),
            "title": str(row.title),
            "text": str(row.text),
        }
        for row in selected.itertuples(index=False)
    ]
    usage = mock_extractor.extract(payload)
    events, event_features = _event_frame(selected, usage)
    model_frame = build_model_frame(
        event_features,
        market[["ticker", "date", *PRICE_FEATURE_COLUMNS]],
        market[["ticker", "date", "target_return"]],
    )
    train = model_frame.loc[model_frame["date"] < TRAIN_END].copy()
    validation = model_frame.loc[
        (model_frame["date"] >= TRAIN_END) & (model_frame["date"] < VALIDATION_END)
    ].copy()
    if train.empty or validation.empty:
        raise ValueError("The selected ticker has insufficient train or validation rows")

    baseline = RidgeReturnModel(alpha=1.0).fit(train, PRICE_FEATURE_COLUMNS)
    news_columns = (*PRICE_FEATURE_COLUMNS, *EVENT_FEATURE_COLUMNS)
    news_model = RidgeReturnModel(alpha=1.0).fit(train, news_columns)
    baseline_validation = validation[["target_return"]].copy()
    baseline_validation["prediction"] = baseline.predict(validation)
    news_validation = validation[["target_return"]].copy()
    news_validation["prediction"] = news_model.predict(validation)

    target_date = pd.Timestamp(prediction_date).date()
    current = validation.loc[validation["date"] == target_date]
    if current.empty:
        raise ValueError(
            f"{target_date} is not a usable validation date for {symbol}; choose a date "
            "between 2023-01-01 and 2023-06-30"
        )
    current_news = selected.loc[
        (selected["ticker"] == symbol) & (selected["date"] == target_date)
    ].sort_values("selection_rank")
    if current_news.empty:
        raise ValueError(
            f"No articles fit the selection budget for {symbol} on {target_date}; "
            "choose another validation date or increase the budget"
        )
    current_events = events.loc[
        (events["ticker"] == symbol) & (events["date"] == target_date)
    ].copy()
    return DemoExperiment(
        ticker=symbol,
        prediction_date=target_date.isoformat(),
        selection_method=selection_method,
        selected_articles=current_news.reset_index(drop=True),
        extracted_events=current_events.reset_index(drop=True),
        baseline_metrics=regression_metrics(baseline_validation),
        news_metrics=regression_metrics(news_validation),
        baseline_prediction=float(baseline.predict(current).iloc[0]),
        news_prediction=float(news_model.predict(current).iloc[0]),
        actual_return=float(current["target_return"].iloc[0]),
        mock_usage=usage,
    )
