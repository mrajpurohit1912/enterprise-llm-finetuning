"""
src/infrastructure/config/__init__.py
Configuration adapters export.
"""

from src.infrastructure.config.yaml_loader import YamlConfigLoader

__all__ = ["YamlConfigLoader"]
