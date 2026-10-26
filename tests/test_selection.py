"""Tests for token-budget enforcement and the implemented selection baselines."""

import unittest

import pandas as pd

from newsalpha.selection import select_news


class SelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = pd.DataFrame(
            [
                {
                    "article_id": "old",
                    "ticker": "ABC",
                    "published_at": "2024-01-01T10:00:00Z",
                    "title": "Old report",
                    "text": "old report text",
                    "url": "https://example.test/old",
                    "token_count": 6,
                    "relevance_score": 0.8,
                    "embedding": "[1.0, 0.0]",
                },
                {
                    "article_id": "new",
                    "ticker": "ABC",
                    "published_at": "2024-01-02T10:00:00Z",
                    "title": "New report",
                    "text": "new report text",
                    "url": "https://example.test/new",
                    "token_count": 5,
                    "relevance_score": 0.6,
                    "embedding": "[0.0, 1.0]",
                },
            ]
        )

    def test_recency_selection_respects_budget(self) -> None:
        selected = select_news(self.candidates, "recency", token_budget=6)
        self.assertEqual(selected["article_id"].tolist(), ["new"])
        self.assertLessEqual(int(selected["token_count_used"].sum()), 6)

    def test_relevance_selection_uses_highest_relevance(self) -> None:
        selected = select_news(self.candidates, "relevance", token_budget=11)
        self.assertEqual(selected.iloc[0]["article_id"], "old")

    def test_similarity_selector_requires_embeddings(self) -> None:
        candidates = self.candidates.drop(columns="embedding")
        with self.assertRaisesRegex(ValueError, "embedding column"):
            select_news(candidates, "mmr", token_budget=20)

    def test_invalid_budget_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            select_news(self.candidates, "recency", token_budget=0)


if __name__ == "__main__":
    unittest.main()
