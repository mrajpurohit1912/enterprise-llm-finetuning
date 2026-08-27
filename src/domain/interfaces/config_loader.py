"""
src/domain/interfaces/config_loader.py
Port interface for configuration loading.
"""

from abc import ABC, abstractmethod
from typing import Any


class ConfigLoaderBase(ABC):
    """Abstract port for configuration file parsers and loaders."""

    @abstractmethod
    def load_config(self) -> Any:
        """
        Parse, validate, and return the strongly-typed ExperimentConfig entity.

        Returns:
            Validated ExperimentConfig domain object.

        Raises:
            ConfigurationError: If configuration parsing or schema validation fails.
        """
        raise NotImplementedError("load_config method must be implemented by concrete loader.")
