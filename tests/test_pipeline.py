"""Tests for market feature construction and split-boundary handling."""

import unittest

import pandas as pd

from newsalpha.pipeline import build_market_features


class PipelineTests(unittest.TestCase):
    def test_feature_builder_excludes_targets_crossing_split_boundaries(self) -> None:
        dates = pd.bdate_range("2022-12-20", "2023-01-10")
        prices = pd.DataFrame(
            {
                "date": dates.date,
                "ticker": "ABC",
                "Adj Close": [100.0 + index for index in range(len(dates))],
                "Volume": [1000 + index * 10 for index in range(len(dates))],
                "Future_Return": [0.01] * len(dates),
            }
        )
        result = build_market_features(prices)
        training = result.loc[result["date"] < pd.Timestamp("2023-01-01").date()]
        validation = result.loc[
            (result["date"] >= pd.Timestamp("2023-01-01").date())
            & (result["date"] < pd.Timestamp("2023-07-01").date())
        ]
        self.assertFalse(training.empty)
        self.assertLess(training["date"].max(), pd.Timestamp("2022-12-30").date())
        self.assertLess(validation["date"].max(), pd.Timestamp("2023-01-10").date())
        self.assertFalse(result[["Return_1d", "Return_5d", "Volume_Ratio", "Volatility_5d"]].isna().any().any())


if __name__ == "__main__":
    unittest.main()
