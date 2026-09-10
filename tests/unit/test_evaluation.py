"""
tests/unit/test_evaluation.py
Unit tests for evaluation domain schemas, DeterministicEvaluator, EvaluatorFactory, and EvaluateModelUseCase.
"""

import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import torch

from src.application.usecases.evaluate_model_usecase import EvaluateModelUseCase
from src.domain.exceptions import ConfigurationError, ModelEvaluationError
from src.domain.schemas.evaluation_schema import (
    EvaluationConfig,
    EvaluationResult,
    EvaluationSample,
    MetricScores,
)
from src.infrastructure.evaluation.deterministic_evaluator import DeterministicEvaluator
from src.infrastructure.factories.evaluator_factory import EvaluatorFactory


class TestEvaluationMetrics(unittest.TestCase):
    """Test metric computation mathematics in DeterministicEvaluator."""

    def test_text_normalization(self) -> None:
        raw = "  Hello, World! Here is a TEST: 123...  "
        expected = "hello world here is a test 123"
        self.assertEqual(DeterministicEvaluator.normalize_text(raw), expected)

    def test_exact_match_perfect_match(self) -> None:
        score = DeterministicEvaluator.compute_exact_match(
            prediction="The revenue was $5 million.",
            ground_truth="the revenue was $5 million!",
        )
        self.assertEqual(score, 1.0)

    def test_exact_match_mismatch(self) -> None:
        score = DeterministicEvaluator.compute_exact_match(
            prediction="Revenue was $5 million.",
            ground_truth="Revenue was $10 million.",
        )
        self.assertEqual(score, 0.0)

    def test_token_f1_score(self) -> None:
        # Exact match = 1.0
        self.assertEqual(
            DeterministicEvaluator.compute_token_f1("apple banana orange", "apple banana orange"),
            1.0,
        )
        # Partial match
        f1 = DeterministicEvaluator.compute_token_f1("apple banana", "apple banana orange")
        # precision = 2/2 = 1.0, recall = 2/3 = 0.6667, f1 = 2 * 1 * 0.6667 / 1.6667 = 0.8
        self.assertAlmostEqual(f1, 0.8, places=4)

        # Disjoint = 0.0
        self.assertEqual(
            DeterministicEvaluator.compute_token_f1("apple banana", "cat dog"),
            0.0,
        )

    def test_rouge_l_score(self) -> None:
        # Identical
        self.assertEqual(
            DeterministicEvaluator.compute_rouge_l("the cat sat on the mat", "the cat sat on the mat"),
            1.0,
        )
        # Subsequence overlap
        rouge_l = DeterministicEvaluator.compute_rouge_l("the cat mat", "the cat sat on the mat")
        self.assertGreater(rouge_l, 0.0)
        self.assertLessEqual(rouge_l, 1.0)

        # Empty
        self.assertEqual(DeterministicEvaluator.compute_rouge_l("", "something"), 0.0)


class TestEvaluatorFactory(unittest.TestCase):
    """Test dynamic evaluator resolution via EvaluatorFactory."""

    def test_get_deterministic_evaluator(self) -> None:
        evaluator = EvaluatorFactory.get_evaluator("deterministic")
        self.assertIsInstance(evaluator, DeterministicEvaluator)

    def test_get_evaluator_aliases(self) -> None:
        evaluator1 = EvaluatorFactory.get_evaluator("exact_match")
        evaluator2 = EvaluatorFactory.get_evaluator("f1_rouge")
        self.assertIsInstance(evaluator1, DeterministicEvaluator)
        self.assertIsInstance(evaluator2, DeterministicEvaluator)

    def test_unsupported_evaluator_raises_error(self) -> None:
        with self.assertRaises(ConfigurationError):
            EvaluatorFactory.get_evaluator("unknown_evaluator_strategy")


class TestDeterministicEvaluator(unittest.TestCase):
    """Test end-to-end evaluation execution with mock model and tokenizer."""

    def setUp(self) -> None:
        self.evaluator = DeterministicEvaluator()

    def test_evaluate_with_mock_model(self) -> None:
        # Mock model
        mock_model = MagicMock()
        mock_param = torch.nn.Parameter(torch.tensor([1.0]))
        mock_model.parameters.return_value = iter([mock_param])
        # Return input_ids (length 3) + generated ids (length 2)
        mock_model.generate.return_value = torch.tensor([[101, 102, 103, 201, 202]])

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[101, 102, 103]])}
        mock_tokenizer.decode.return_value = "Fine-tuned answer"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.apply_chat_template.return_value = "User: What is OfficeQA?"

        dataset = [
            {"question": "What is OfficeQA?", "answer": "Fine-tuned answer"},
            {"question": "Who is the CEO?", "answer": "Different answer"},
        ]

        config = EvaluationConfig(
            eval_split="test",
            max_eval_samples=2,
            min_f1_threshold=0.40,
        )

        result = self.evaluator.evaluate(
            model=mock_model,
            tokenizer=mock_tokenizer,
            eval_dataset=dataset,
            eval_config=config,
            experiment_name="test-exp",
            model_id="test-model",
        )

        self.assertEqual(result.total_samples, 2)
        self.assertEqual(result.experiment_name, "test-exp")
        self.assertIn("exact_match", result.metrics.model_dump())
        self.assertIn("f1_score", result.metrics.model_dump())
        self.assertTrue(result.passed_quality_gate)
        self.assertEqual(len(result.sample_details), 2)


class TestEvaluateModelUseCase(unittest.TestCase):
    """Test EvaluateModelUseCase report generation and error handling."""

    def test_use_case_generates_json_and_markdown_reports(self) -> None:
        mock_evaluator = MagicMock()
        sample = EvaluationSample(
            prompt="What is X?",
            ground_truth="X is 1",
            prediction="X is 1",
            exact_match=True,
            f1_score=1.0,
        )
        mock_evaluator.evaluate.return_value = EvaluationResult(
            experiment_name="officeqa-run",
            model_id="Qwen/Qwen2.5-0.5B",
            eval_split="test",
            total_samples=1,
            metrics=MetricScores(exact_match=1.0, f1_score=1.0, rouge_l=1.0),
            latency_per_sample_ms=25.0,
            throughput_tokens_per_sec=40.0,
            passed_quality_gate=True,
            sample_details=[sample],
        )

        use_case = EvaluateModelUseCase(evaluator=mock_evaluator)

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = use_case.execute(
                model=MagicMock(),
                tokenizer=MagicMock(),
                eval_dataset=[{"question": "What is X?", "answer": "X is 1"}],
                eval_config=EvaluationConfig(),
                experiment_name="officeqa-run",
                model_id="Qwen/Qwen2.5-0.5B",
                output_dir=tmp_dir,
            )

            self.assertEqual(result.metrics.f1_score, 1.0)

            # Verify JSON report exists and is valid
            json_file = Path(tmp_dir) / "evaluation_report.json"
            self.assertTrue(json_file.exists())
            with open(json_file, "r") as f:
                data = json.load(f)
                self.assertEqual(data["experiment_name"], "officeqa-run")
                self.assertEqual(data["metrics"]["f1_score"], 1.0)

            # Verify Markdown report exists
            md_file = Path(tmp_dir) / "evaluation_report.md"
            self.assertTrue(md_file.exists())
            with open(md_file, "r") as f:
                content = f.read()
                self.assertIn("Model Evaluation Report", content)
                self.assertIn("Exact Match (EM)", content)

    def test_use_case_wraps_unexpected_error_in_domain_error(self) -> None:
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.side_effect = RuntimeError("CUDA OOM during evaluation")

        use_case = EvaluateModelUseCase(evaluator=mock_evaluator)

        with self.assertRaises(ModelEvaluationError):
            use_case.execute(
                model=MagicMock(),
                tokenizer=MagicMock(),
                eval_dataset=[],
                eval_config=EvaluationConfig(),
                experiment_name="failing-run",
                model_id="test-model",
            )


if __name__ == "__main__":
    unittest.main()
