"""Aggregate sourced event outputs and market variables into ticker/date model features."""

from __future__ import annotations

import pandas as pd


EVENT_COLUMNS = {
    "ticker",
    "date",
    "article_id",
    "impact_direction",
    "confidence",
}


def aggregate_event_features(events: pd.DataFrame) -> pd.DataFrame:
    """Create one event-feature row per ticker/date, including an explicit empty-news default."""
    missing = EVENT_COLUMNS.difference(events.columns)
    if missing:
        raise ValueError(f"Events are missing required columns: {', '.join(sorted(missing))}")
    if events.empty:
        return pd.DataFrame(
            columns=[
                "ticker", "date", "event_count", "positive_event_count",
                "negative_event_count", "weighted_impact",
            ]
        )

    frame = events.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.date
    if not frame["impact_direction"].isin([-1, 0, 1]).all():
        raise ValueError("impact_direction values must be -1, 0, or 1")
    if not frame["confidence"].between(0, 1).all():
        raise ValueError("confidence values must be between 0 and 1")
    frame["_weighted_impact"] = frame["impact_direction"] * frame["confidence"]
    return (
        frame.groupby(["ticker", "date"], as_index=False)
        .agg(
            event_count=("article_id", "nunique"),
            positive_event_count=("impact_direction", lambda values: int((values == 1).sum())),
            negative_event_count=("impact_direction", lambda values: int((values == -1).sum())),
            weighted_impact=("_weighted_impact", "mean"),
        )
    )


def build_model_frame(
    event_features: pd.DataFrame,
    market_features: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """Join events, cutoff-safe market features, and realized targets by ticker/date."""
    keys = ["ticker", "date"]
    for label, frame in (
        ("event_features", event_features),
        ("market_features", market_features),
        ("targets", targets),
    ):
        missing = set(keys).difference(frame.columns)
        if missing:
            raise ValueError(f"{label} is missing join keys: {', '.join(sorted(missing))}")
        if frame.duplicated(keys).any():
            raise ValueError(f"{label} must have at most one row per ticker/date")

    merged = event_features.merge(market_features, on=keys, how="outer", validate="one_to_one")
    merged = merged.merge(targets, on=keys, how="inner", validate="one_to_one")
    event_numeric = [
        column for column in
        ("event_count", "positive_event_count", "negative_event_count", "weighted_impact")
        if column in merged.columns
    ]
    merged[event_numeric] = merged[event_numeric].fillna(0)
    return merged.sort_values(["date", "ticker"]).reset_index(drop=True)
