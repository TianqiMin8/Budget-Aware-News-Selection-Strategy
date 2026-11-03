"""Tests for offline extraction and the OpenAI adapter contract."""

import unittest
from types import SimpleNamespace

from newsalpha.events import (
    MockEventExtractor,
    OpenAIEventExtractor,
)


class EventExtractorTests(unittest.TestCase):
    def test_mock_extractor_is_offline_and_auditable(self) -> None:
        result = MockEventExtractor().extract(
            [
                {
                    "article_id": "a1",
                    "title": "Company reports strong growth",
                    "text": "Revenue beat expectations.",
                }
            ]
        )
        self.assertEqual(result.model_name, "mock-keyword-v1")
        self.assertIsNone(result.cost_usd)
        self.assertEqual(result.events[0].article_id, "a1")
        self.assertEqual(result.events[0].impact_direction, 1)
        self.assertIn("growth", result.events[0].evidence_quote.casefold())

    def test_openai_adapter_uses_supplied_client_without_network(self) -> None:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"events":[{"article_id":"a1","event_type":"earnings",'
                            '"description":"Revenue beat expectations",'
                            '"impact_direction":1,"confidence":0.8,'
                            '"evidence_quote":"Revenue beat expectations"}]}'
                        )
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=12),
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: response,
                )
            )
        )
        result = OpenAIEventExtractor(client=client).extract(
            [{"article_id": "a1", "title": "Earnings", "text": "Revenue beat expectations"}]
        )
        self.assertEqual(result.input_tokens, 20)
        self.assertEqual(result.output_tokens, 12)
        self.assertEqual(result.events[0].impact_direction, 1)
        self.assertIsNone(result.cost_usd)

    def test_openai_adapter_rejects_unsourced_events(self) -> None:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"events":[{"article_id":"unknown","event_type":"earnings",'
                            '"description":"event","impact_direction":1,'
                            '"confidence":0.8,"evidence_quote":"quote"}]}'
                        )
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=12),
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: response)
            )
        )
        with self.assertRaisesRegex(ValueError, "not supplied"):
            OpenAIEventExtractor(client=client).extract(
                [{"article_id": "known", "title": "Title", "text": "Text"}]
            )


if __name__ == "__main__":
    unittest.main()
