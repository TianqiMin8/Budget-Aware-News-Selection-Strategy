"""Chronological data splitting and date-level Rank IC and bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def chronological_split(
    frame: pd.DataFrame,
    date_column: str = "date",
    train_fraction: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split unique dates in order so observations from a date cannot cross the boundary."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if date_column not in frame.columns:
        raise ValueError(f"Missing date column: {date_column}")
    dates = pd.Index(frame[date_column].dropna().unique()).sort_values()
    if len(dates) < 2:
        raise ValueError("At least two distinct non-null dates are required")
    split_index = min(max(int(len(dates) * train_fraction), 1), len(dates) - 1)
    boundary = dates[split_index]
    return (
        frame.loc[frame[date_column] < boundary].copy(),
        frame.loc[frame[date_column] >= boundary].copy(),
    )


def daily_rank_ic(
    frame: pd.DataFrame,
    prediction_column: str = "prediction",
    target_column: str = "target_return",
    date_column: str = "date",
) -> pd.DataFrame:
    """Compute daily Spearman rank correlation; dates with fewer than two names are omitted."""
    required = {prediction_column, target_column, date_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Evaluation frame is missing columns: {', '.join(sorted(missing))}")
    records: list[dict[str, object]] = []
    for date, daily in frame.groupby(date_column):
        valid = daily[[prediction_column, target_column]].dropna()
        if len(valid) < 2:
            continue
        value = valid[prediction_column].corr(valid[target_column], method="spearman")
        if pd.notna(value):
            records.append({"date": date, "rank_ic": float(value), "n_stocks": len(valid)})
    return pd.DataFrame(records, columns=["date", "rank_ic", "n_stocks"])


def bootstrap_rank_ic(
    daily_ics: pd.DataFrame,
    confidence: float = 0.95,
    repetitions: int = 2000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Bootstrap daily IC values (resampling dates) and return mean and percentile interval."""
    if "rank_ic" not in daily_ics.columns:
        raise ValueError("daily_ics must contain a rank_ic column")
    values = daily_ics["rank_ic"].dropna().to_numpy(dtype=float)
    if len(values) == 0:
        raise ValueError("At least one daily Rank IC value is required")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    generator = np.random.default_rng(seed)
    samples = generator.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    tail = (1 - confidence) / 2
    lower, upper = np.quantile(samples, [tail, 1 - tail])
    return {
        "n_dates": int(len(values)),
        "mean_rank_ic": float(values.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence": confidence,
    }
