"""
src/infrastructure/config/yaml_loader.py
YAML configuration loader adapter implementing ConfigLoaderBase.
"""

from pathlib import Path
from typing import Union
import yaml
from pydantic import ValidationError

from src.domain.exceptions import ConfigurationError
from src.domain.interfaces.config_loader import ConfigLoaderBase
from src.domain.schemas.config_schema import ExperimentConfig


class YamlConfigLoader(ConfigLoaderBase):
    """Parses and validates declarative YAML configuration files into ExperimentConfig."""

    def __init__(self, config_path: Union[str, Path]) -> None:
        self.config_path = Path(config_path)

    def load_config(self) -> ExperimentConfig:
        """
        Load and validate configuration from YAML file.

        Returns:
            Validated immutable ExperimentConfig aggregate root.

        Raises:
            ConfigurationError: If the file is missing, malformed, or fails schema validation.
        """
        if not self.config_path.is_file():
            raise ConfigurationError(
                f"Configuration file not found at path: '{self.config_path.resolve()}'"
            )

        try:
            with open(self.config_path, "r", encoding="utf-8") as file:
                raw_config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            raise ConfigurationError(
                f"Failed to parse YAML syntax in '{self.config_path}': {exc}"
            ) from exc
        except Exception as exc:
            raise ConfigurationError(
                f"Unexpected I/O error reading '{self.config_path}': {exc}"
            ) from exc

        if not isinstance(raw_config, dict):
            raise ConfigurationError(
                f"Configuration root in '{self.config_path}' must be a key-value mapping, got {type(raw_config).__name__}"
            )

        try:
            return ExperimentConfig.model_validate(raw_config)
        except ValidationError as exc:
            raise ConfigurationError(
                f"Configuration schema validation failed for '{self.config_path}':\n{exc}"
            ) from exc
