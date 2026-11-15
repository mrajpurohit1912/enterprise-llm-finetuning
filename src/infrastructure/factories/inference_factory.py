"""
src/infrastructure/factories/inference_factory.py
Factory for creating and dynamically registering inference engines.
"""

from typing import Any, ClassVar

from src.domain.exceptions import ConfigurationError
from src.domain.interfaces.inference_engine import IInferenceEngine
from src.infrastructure.inference.vllm_engine import VLLMInferenceEngine


class InferenceEngineFactory:
    """Factory creating inference engine instances following Open-Closed Principle."""

    _registry: ClassVar[dict[str, type[IInferenceEngine]]] = {
        "vllm": VLLMInferenceEngine,
    }

    @classmethod
    def register_engine(cls, name: str, engine_cls: type[IInferenceEngine]) -> None:
        """Dynamically register a new inference backend."""
        cls._registry[name.lower()] = engine_cls

    @classmethod
    def get_engine(cls, engine_type: str = "vllm", **kwargs: Any) -> IInferenceEngine:
        """Instantiate the configured inference engine."""
        key = engine_type.strip().lower()
        engine_cls = cls._registry.get(key)
        if not engine_cls:
            raise ConfigurationError(
                f"Unsupported inference engine '{engine_type}'. "
                f"Valid supported types: {list(cls._registry.keys())}."
            )
        return engine_cls(**kwargs)


__all__ = ["InferenceEngineFactory"]