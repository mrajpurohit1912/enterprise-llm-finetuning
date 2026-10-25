"""
tests/unit/test_sample_extractor.py
Unit tests for the EvaluationSampleExtractor composition service.
"""

import unittest
from src.domain.schemas.evaluation_schema import EvaluationConfig
from src.infrastructure.evaluation.sample_extractor import EvaluationSampleExtractor


class TestEvaluationSampleExtractor(unittest.TestCase):
    """Test suite for EvaluationSampleExtractor extraction priorities and heuristics."""

    def setUp(self) -> None:
        self.extractor = EvaluationSampleExtractor()

    def test_extract_with_explicit_columns(self) -> None:
        dataset = [
            {"custom_q": "What is net income?", "custom_a": "$500M"},
            {"custom_q": "What is revenue?", "custom_a": "$2000M"},
        ]
        config = EvaluationConfig(prompt_column="custom_q", target_column="custom_a")
        samples = self.extractor.extract_samples(dataset, config)

        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0], ("What is net income?", "$500M"))
        self.assertEqual(samples[1], ("What is revenue?", "$2000M"))

    def test_extract_with_chatml_messages(self) -> None:
        dataset = [
            {
                "messages": [
                    {"role": "system", "content": "You are a financial assistant."},
                    {"role": "user", "content": "Calculate EBITDA."},
                    {"role": "assistant", "content": "150.5"},
                ]
            }
        ]
        config = EvaluationConfig()
        samples = self.extractor.extract_samples(dataset, config)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0], ("Calculate EBITDA.", "150.5"))

    def test_extract_with_conversations(self) -> None:
        dataset = [
            {
                "conversations": [
                    {"from": "human", "value": "Compute margin."},
                    {"from": "gpt", "value": "0.25"},
                ]
            }
        ]
        config = EvaluationConfig()
        samples = self.extractor.extract_samples(dataset, config)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0], ("Compute margin.", "0.25"))

    def test_extract_with_ranked_candidate_pairs(self) -> None:
        dataset = [
            {"query": "What is EPS?", "answer": "2.45"},
            {"question": "What is ROE?", "answer": "18%"},
            {"instruction": "Extract gross profit", "output": "450M"},
        ]
        config = EvaluationConfig()
        samples = self.extractor.extract_samples(dataset, config)

        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0], ("What is EPS?", "2.45"))
        self.assertEqual(samples[1], ("What is ROE?", "18%"))
        self.assertEqual(samples[2], ("Extract gross profit", "450M"))

    def test_extract_respects_max_eval_samples(self) -> None:
        dataset = [{"query": f"Q{i}", "answer": f"A{i}"} for i in range(10)]
        config = EvaluationConfig(max_eval_samples=3)
        samples = self.extractor.extract_samples(dataset, config)

        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0], ("Q0", "A0"))
        self.assertEqual(samples[2], ("Q2", "A2"))

    def test_extract_empty_or_invalid_dataset(self) -> None:
        config = EvaluationConfig()
        self.assertEqual(self.extractor.extract_samples([], config), [])
        self.assertEqual(self.extractor.extract_samples([{"foo": "bar"}], config), [])


if __name__ == "__main__":
    unittest.main()
