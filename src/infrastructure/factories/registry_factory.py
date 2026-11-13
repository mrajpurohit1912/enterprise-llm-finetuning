"""
src/infrastructure/factories/registry_factory.py
Factory creating IModelRegistry implementations dynamically based on configuration.
Follows Open/Closed Principle (OCP) for pluggable model registry backends.
"""

from typing import Any, ClassVar

from src.domain.exceptions import ConfigurationError
from src.domain.interfaces.model_registry import IModelRegistry
from src.infrastructure.registry.huggingface_registry import HuggingFaceHubRegistry
from src.infrastructure.registry.local_registry import LocalModelRegistry


class ModelRegistryFactory:
    """
    Factory creating IModelRegistry implementations based on string identifiers.
    """

    _registry: ClassVar[dict[str, type[IModelRegistry]]] = {
        "hf_hub": HuggingFaceHubRegistry,
        "huggingface": HuggingFaceHubRegistry,
        "local": LocalModelRegistry,
    }

    @classmethod
    def register_provider(
        cls,
        registry_type: str,
        registry_cls: type[IModelRegistry],
    ) -> None:
        """Dynamically register a new model registry backend."""
        cls._registry[registry_type.lower()] = registry_cls

    @classmethod
    def get_registry(
        cls,
        registry_type: str = "hf_hub",
        **kwargs: Any,
    ) -> IModelRegistry:
        """
        Instantiate and return the appropriate IModelRegistry adapter.

        Args:
            registry_type: Strategy identifier ('hf_hub', 'huggingface', 'local').
            **kwargs: Backend-specific arguments passed to adapter constructor.

        Returns:
            IModelRegistry: Configured registry implementation.

        Raises:
            ConfigurationError: If requested registry_type is not registered.
        """
        key = (registry_type or "hf_hub").strip().lower()
        registry_cls = cls._registry.get(key)
        if registry_cls is None:
            valid_types = list(cls._registry.keys())
            raise ConfigurationError(
                f"Unsupported model registry type '{registry_type}'. "
                f"Valid supported types: {valid_types}."
            )
        return registry_cls(**kwargs)
