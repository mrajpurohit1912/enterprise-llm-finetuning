"""
tests/unit/test_financial_evaluator.py
Unit tests for the FinancialEvaluator implementation.
"""

import unittest
from unittest.mock import MagicMock
import torch

from src.domain.schemas.evaluation_schema import EvaluationConfig, TaskType
from src.infrastructure.evaluation.financial_evaluator import FinancialEvaluator
from src.domain.interfaces.sample_extractor import ISampleExtractor


class DummyExtractor(ISampleExtractor):
    def __init__(self, samples):
        self._samples = samples

    def extract_samples(self, dataset, eval_config):
        return self._samples


class TestFinancialEvaluator(unittest.TestCase):
    """Test suite for FinancialEvaluator quantitative metrics computation."""

    def test_evaluate_with_mock_model(self) -> None:
        # Mock samples:
        # 1. 0.14629 vs 0.14464 (~1.14% error -> within 5%, not within 1%, sign matches)
        # 2. 100 vs 100 (exact match, within 1%, within 5%, sign matches)
        # 3. -50 vs 50 (sign mismatch, large error)
        # 4. "no number" vs 20 (unparseable)
        samples = [
            ("What is the ratio?", "0.14464"),
            ("What is total revenue?", "100"),
            ("What is net income?", "50"),
            ("What is EBITDA?", "20"),
        ]
        extractor = DummyExtractor(samples)
        evaluator = FinancialEvaluator(sample_extractor=extractor)

        # Mock model & tokenizer
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])

        # Return mock outputs for 4 samples:
        # Sample 1 -> "0.14629"
        # Sample 2 -> "100"
        # Sample 3 -> "-50"
        # Sample 4 -> "unknown text"
        mock_model.generate.return_value = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1], [1], [1], [1]]),
            "attention_mask": torch.tensor([[1], [1], [1], [1]]),
        }
        mock_tokenizer.batch_decode.return_value = [
            "0.14629",
            "100",
            "-50",
            "unknown text",
        ]

        config = EvaluationConfig(
            task_type=TaskType.TEXT_GENERATION,
            evaluator_type="financial",
            max_eval_samples=4,
            batch_size=2,
            min_tolerance_accuracy_threshold=0.50, # 2 out of 4 (50%) within 5%
        )

        mock_tokenizer.batch_decode.side_effect = [
            ["0.14629", "100"],
            ["-50", "unknown text"],
        ]

        result = evaluator.evaluate(
            model=mock_model,
            tokenizer=mock_tokenizer,
            eval_dataset=samples,
            eval_config=config,
            experiment_name="test_exp",
            model_id="test_model",
        )

        self.assertEqual(result.total_samples, 4)
        self.assertEqual(len(result.sample_details), 4)
        # 2 out of 4 within 5% error (0.14629 and 100) -> 50%
        self.assertEqual(result.metrics.tolerance_accuracy_5pct, 0.50)
        # 1 out of 4 within 1% error (100) -> 25%
        self.assertEqual(result.metrics.tolerance_accuracy_1pct, 0.25)
        # 3 out of 4 are parseable -> 75%
        self.assertEqual(result.metrics.parseable_number_rate, 0.75)
        # Quality gate should pass (50% >= 50%)
        self.assertTrue(result.passed_quality_gate)


if __name__ == "__main__":
    unittest.main()
