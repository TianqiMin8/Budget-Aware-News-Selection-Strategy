"""Tests for sourced event validation and event-feature aggregation."""

import unittest

import pandas as pd

from newsalpha.events import EventExtractor, ExtractedEvent, ExtractionResult, validate_extraction
from newsalpha.features import aggregate_event_features


class FeaturesTests(unittest.TestCase):
    def test_aggregate_event_features(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "ticker": "ABC",
                    "date": "2024-01-02",
                    "article_id": "a1",
                    "impact_direction": 1,
                    "confidence": 0.8,
                },
                {
                    "ticker": "ABC",
                    "date": "2024-01-02",
                    "article_id": "a2",
                    "impact_direction": -1,
                    "confidence": 0.5,
                },
            ]
        )
        result = aggregate_event_features(events).iloc[0]
        self.assertEqual(result["event_count"], 2)
        self.assertEqual(result["positive_event_count"], 1)
        self.assertAlmostEqual(result["weighted_impact"], 0.15)

    def test_event_must_cite_supplied_article(self) -> None:
        result = ExtractionResult(
            events=(ExtractedEvent("unknown", "earnings", "event", 1, 0.8, "quote"),),
            input_tokens=10,
            output_tokens=5,
            cost_usd=None,
            latency_seconds=0.2,
            model_name="frozen-model",
        )
        with self.assertRaisesRegex(ValueError, "not supplied"):
            validate_extraction(result, {"known"})


if __name__ == "__main__":
    unittest.main()
