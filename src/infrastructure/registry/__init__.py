"""
src/infrastructure/registry/__init__.py
Model Registry Infrastructure Package.
"""

from src.infrastructure.registry.huggingface_registry import HuggingFaceHubRegistry
from src.infrastructure.registry.local_registry import LocalModelRegistry
from src.infrastructure.registry.model_card import ModelCardGenerator

__all__ = [
    "HuggingFaceHubRegistry",
    "LocalModelRegistry",
    "ModelCardGenerator",
]
