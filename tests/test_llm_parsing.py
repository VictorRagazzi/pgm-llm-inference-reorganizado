import unittest

from pgm_llm_inference.strategies.llm.parsing import extract_last_json_object


class LlmJsonParsingTests(unittest.TestCase):
    def test_extracts_the_last_object_and_tolerates_comments(self) -> None:
        response = (
            'draft {"ignored": true}\n'
            'final {"url": "https://example.com", // explanation\n"value": 2}'
        )

        self.assertEqual(
            extract_last_json_object(response),
            {"url": "https://example.com", "value": 2},
        )

    def test_rejects_an_empty_response(self) -> None:
        with self.assertRaisesRegex(ValueError, "Empty LLM output"):
            extract_last_json_object("  ")
