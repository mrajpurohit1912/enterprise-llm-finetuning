"""
src/infrastructure/factories/compute_factory.py
Factory for instantiating concrete IComputeProvider adapters dynamically.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Type

from src.domain.exceptions import ConfigurationError
from src.domain.interfaces.compute_provider import IComputeProvider
from src.infrastructure.compute.kaggle.provider import KaggleComputeProvider

logger = logging.getLogger(__name__)


class ComputeProviderFactory:
    """
    Factory creating IComputeProvider implementations based on string identifiers.
    Follows Open/Closed Principle (OCP) for pluggable cloud GPU backends.
    """

    _registry: Dict[str, Type[IComputeProvider]] = {
        "kaggle": KaggleComputeProvider,
        "kaggle_dual_t4": KaggleComputeProvider,
    }

    @classmethod
    def register_provider(
        cls,
        provider_name: str,
        provider_cls: Type[IComputeProvider],
    ) -> None:
        """Dynamically register a new compute provider backend."""
        cls._registry[provider_name.lower()] = provider_cls

    @classmethod
    def get_provider(
        cls,
        provider_type: str = "kaggle",
        workspace_dir: Optional[Path] = None,
    ) -> IComputeProvider:
        """
        Instantiate and return the appropriate IComputeProvider adapter.

        Args:
            provider_type: Backend strategy name (default: 'kaggle').
            workspace_dir: Optional custom workspace staging directory.

        Returns:
            IComputeProvider: Concrete compute provider adapter.

        Raises:
            ConfigurationError: If provider_type is not registered.
        """
        key = provider_type.strip().lower()
        provider_cls = cls._registry.get(key)
        if not provider_cls:
            valid_providers = list(cls._registry.keys())
            raise ConfigurationError(
                f"Unknown compute provider: '{provider_type}'. Supported providers: {valid_providers}"
            )

        logger.info("Instantiating compute provider adapter: %s (%s)", key, provider_cls.__name__)
        if key.startswith("kaggle"):
            return KaggleComputeProvider(workspace_dir=workspace_dir)

        return provider_cls()
