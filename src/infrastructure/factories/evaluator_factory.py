"""
src/infrastructure/factories/evaluator_factory.py
Factory for instantiating concrete EvaluatorBase adapters dynamically.
"""

import logging
from typing import Dict, Optional, Type

from src.domain.exceptions import ConfigurationError
from src.domain.interfaces.evaluator import EvaluatorBase
from src.infrastructure.evaluation.deterministic_evaluator import DeterministicEvaluator
from src.infrastructure.evaluation.financial_evaluator import FinancialEvaluator

logger = logging.getLogger(__name__)


class EvaluatorFactory:
    """
    Factory creating EvaluatorBase implementations based on string identifiers.
    Follows Open/Closed Principle (OCP) and supports enterprise auto-detection.
    """

    _registry: Dict[str, Type[EvaluatorBase]] = {
        "deterministic": DeterministicEvaluator,
        "exact_match": DeterministicEvaluator,
        "f1_rouge": DeterministicEvaluator,
        "financial": FinancialEvaluator,
        "financial_quantitative": FinancialEvaluator,
        "finqa": FinancialEvaluator,
        "quantitative": FinancialEvaluator,
    }

    @classmethod
    def register_evaluator(cls, evaluator_type: str, evaluator_cls: Type[EvaluatorBase]) -> None:
        """Dynamically register a new custom evaluation engine."""
        cls._registry[evaluator_type.lower()] = evaluator_cls

    @classmethod
    def get_evaluator(
        cls,
        evaluator_type: str = "deterministic",
        dataset_name: Optional[str] = None,
    ) -> EvaluatorBase:
        """
        Instantiate and return the appropriate EvaluatorBase adapter.

        Args:
            evaluator_type: Strategy name (e.g. 'financial', 'deterministic', 'auto').
            dataset_name: Optional dataset identifier for smart domain auto-detection.

        Returns:
            EvaluatorBase: Concrete evaluation adapter.

        Raises:
            ConfigurationError: If evaluator_type is not registered.
        """
        key = (evaluator_type or "deterministic").strip().lower()

        # Enterprise Auto-Detection: Resolve strategy based on dataset domain
        if key == "auto":
            d_name = (dataset_name or "").lower()
            if any(term in d_name for term in ["finqa", "finance", "financial", "sec", "accounting"]):
                logger.info("Auto-detected financial domain from dataset '%s' -> using FinancialEvaluator", dataset_name)
                return FinancialEvaluator()
            logger.info("Auto-detection defaulted to DeterministicEvaluator for dataset '%s'", dataset_name)
            return DeterministicEvaluator()

        evaluator_cls = cls._registry.get(key)
        if not evaluator_cls:
            raise ConfigurationError(
                f"Unsupported evaluator type '{evaluator_type}'. "
                f"Supported types: {list(cls._registry.keys())} or 'auto'"
            )
        return evaluator_cls()
