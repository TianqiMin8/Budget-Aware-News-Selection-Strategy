"""Tests for date-aware splits and daily Rank IC calculations."""

import unittest

import pandas as pd

from newsalpha.evaluation import chronological_split, daily_rank_ic


class EvaluationTests(unittest.TestCase):
    def test_split_keeps_dates_in_order(self) -> None:
        frame = pd.DataFrame(
            {"date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]}
        )
        train, test = chronological_split(frame, train_fraction=0.5)
        self.assertEqual(set(train["date"]), {"2024-01-01"})
        self.assertEqual(set(test["date"]), {"2024-01-02"})

    def test_daily_rank_ic_is_spearman(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2024-01-01"] * 3,
                "prediction": [1.0, 2.0, 3.0],
                "target_return": [10.0, 20.0, 30.0],
            }
        )
        result = daily_rank_ic(frame)
        self.assertAlmostEqual(result.iloc[0]["rank_ic"], 1.0)


if __name__ == "__main__":
    unittest.main()
