"""Fit the proposal's Ridge predictor on explicit numeric features and a return target."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd
from sklearn.linear_model import Ridge


@dataclass
class RidgeReturnModel:
    """Small wrapper that keeps feature order stable between fit and prediction."""

    alpha: float = 1.0
    feature_columns: tuple[str, ...] = ()
    estimator: Ridge | None = None

    def fit(
        self,
        training_frame: pd.DataFrame,
        feature_columns: Sequence[str],
        target_column: str = "target_return",
    ) -> "RidgeReturnModel":
        if not feature_columns:
            raise ValueError("At least one model feature is required")
        missing = set(feature_columns).union({target_column}).difference(training_frame.columns)
        if missing:
            raise ValueError(f"Training frame is missing columns: {', '.join(sorted(missing))}")
        if training_frame.empty:
            raise ValueError("Cannot fit Ridge on an empty training frame")
        if training_frame[list(feature_columns)].isna().any().any():
            raise ValueError("Training features contain missing values")
        if training_frame[target_column].isna().any():
            raise ValueError("Training target contains missing values")
        self.feature_columns = tuple(feature_columns)
        self.estimator = Ridge(alpha=self.alpha)
        self.estimator.fit(training_frame[list(self.feature_columns)], training_frame[target_column])
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        if self.estimator is None:
            raise RuntimeError("Model must be fit before prediction")
        missing = set(self.feature_columns).difference(frame.columns)
        if missing:
            raise ValueError(f"Prediction frame is missing columns: {', '.join(sorted(missing))}")
        if frame[list(self.feature_columns)].isna().any().any():
            raise ValueError("Prediction features contain missing values")
        return pd.Series(self.estimator.predict(frame[list(self.feature_columns)]), index=frame.index)
