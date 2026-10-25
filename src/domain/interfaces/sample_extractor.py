"""
src/domain/interfaces/sample_extractor.py
Abstract Port defining the contract for extracting evaluation prompt/reference pairs from datasets.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from src.domain.schemas.evaluation_schema import EvaluationConfig


class ISampleExtractor(ABC):
    """
    Port interface defining the contract for evaluation dataset sample extraction.
    Follows Single Responsibility Principle (SRP) and Dependency Inversion Principle (DIP).
    """

    @abstractmethod
    def extract_samples(
        self,
        dataset: Any,
        eval_config: EvaluationConfig,
    ) -> List[Tuple[str, str]]:
        """
        Dynamically extracts (prompt, reference_answer) pairs from diverse dataset structures.

        Args:
            dataset: Evaluation dataset partition.
            eval_config: Evaluation configuration containing column mappings and sample limits.

        Returns:
            List[Tuple[str, str]]: List of clean (prompt, reference_answer) string tuples.
        """
        raise NotImplementedError
