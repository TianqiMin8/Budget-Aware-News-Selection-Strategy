"""Tests for scaled Ridge fitting and regression metrics."""

import unittest

import pandas as pd

from newsalpha.evaluation import regression_metrics
from newsalpha.model import RidgeReturnModel


class ModelTests(unittest.TestCase):
    def test_ridge_model_predicts_with_stable_feature_order(self) -> None:
        training = pd.DataFrame(
            {
                "first": [0.0, 1.0, 2.0, 3.0],
                "second": [3.0, 2.0, 1.0, 0.0],
                "target_return": [0.0, 0.01, 0.02, 0.03],
            }
        )
        model = RidgeReturnModel(alpha=0.1).fit(training, ["first", "second"])
        prediction = model.predict(training[["second", "first"]])
        self.assertEqual(len(prediction), len(training))
        self.assertTrue(prediction.notna().all())

    def test_regression_metrics_are_in_return_units(self) -> None:
        frame = pd.DataFrame(
            {"target_return": [0.01, -0.02], "prediction": [0.02, -0.01]}
        )
        metrics = regression_metrics(frame)
        self.assertAlmostEqual(metrics["mae"], 0.01)
        self.assertAlmostEqual(metrics["directional_accuracy"], 1.0)
        self.assertEqual(metrics["n_observations"], 2)

    def test_metrics_require_valid_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "(?i)at least one"):
            regression_metrics(
                pd.DataFrame({"target_return": [None], "prediction": [None]})
            )


if __name__ == "__main__":
    unittest.main()
