"""
src/domain/interfaces/evaluator.py
Abstract base interface for model evaluators following the Dependency Inversion Principle.
"""

from abc import ABC, abstractmethod
from typing import Any
from src.domain.schemas.evaluation_schema import EvaluationConfig, EvaluationResult


class EvaluatorBase(ABC):
    """
    Abstract Port defining the contract for evaluation engines.
    Concrete adapters (e.g. DeterministicEvaluator) in Infrastructure implement this.
    """

    @abstractmethod
    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        eval_dataset: Any,
        eval_config: EvaluationConfig,
        experiment_name: str,
        model_id: str,
    ) -> EvaluationResult:
        """
        Execute evaluation of a model against a dataset partition.

        Args:
            model: Foundation or fine-tuned model (PyTorch/PEFT).
            tokenizer: Model tokenizer for encoding inputs and decoding predictions.
            eval_dataset: Evaluation dataset partition containing questions and references.
            eval_config: Evaluation hyperparameters and thresholds.
            experiment_name: Active experiment identifier.
            model_id: Model or adapter name being evaluated.

        Returns:
            EvaluationResult: Strongly-typed evaluation metrics, latency, and sample details.
        """
        raise NotImplementedError
