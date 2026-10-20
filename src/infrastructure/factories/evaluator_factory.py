"""
src/infrastructure/factories/evaluator_factory.py
Factory for instantiating concrete EvaluatorBase adapters dynamically.
"""

import logging
from typing import Dict, Type

from src.domain.exceptions import ConfigurationError
from src.domain.interfaces.evaluator import EvaluatorBase
from src.infrastructure.evaluation.deterministic_evaluator import DeterministicEvaluator

logger = logging.getLogger(__name__)


class EvaluatorFactory:
    """
    Factory creating EvaluatorBase implementations based on string identifiers.
    Follows Open/Closed Principle (OCP) for pluggable evaluation engines.
    """

    _registry: Dict[str, Type[EvaluatorBase]] = {
        "deterministic": DeterministicEvaluator,
        "exact_match": DeterministicEvaluator,
        "f1_rouge": DeterministicEvaluator,
    }

    @classmethod
    def register_evaluator(cls, evaluator_type: str, evaluator_cls: Type[EvaluatorBase]) -> None:
        """Dynamically register a new custom evaluation engine."""
        cls._registry[evaluator_type.lower()] = evaluator_cls

    @classmethod
    def get_evaluator(cls, evaluator_type: str = "deterministic") -> EvaluatorBase:
        """
        Instantiate and return the appropriate EvaluatorBase adapter.

        Args:
            evaluator_type: Strategy name (default: 'deterministic').

        Returns:
            EvaluatorBase: Concrete evaluation adapter.

        Raises:
            ConfigurationError: If evaluator_type is not registered.
        """
        key = evaluator_type.strip().lower()
        evaluator_cls = cls._registry.get(key)
        if not evaluator_cls:
            raise ConfigurationError(
                f"Unsupported evaluator type '{evaluator_type}'. "
                f"Supported types: {list(cls._registry.keys())}"
            )
        return evaluator_cls()
