"""
src/infrastructure/__init__.py
Infrastructure Layer package.
"""

from src.infrastructure.compute import (
    KaggleApiClient,
    KaggleKernelBuilder,
    KaggleComputeProvider,
)
from src.infrastructure.config import YamlConfigLoader
from src.infrastructure.factories import (
    ComputeProviderFactory,
    DatasetLoaderFactory,
    EvaluatorFactory,
    ModelRegistryFactory,
    TrainerFactory,
)
from src.infrastructure.registry import (
    HuggingFaceHubRegistry,
    LocalModelRegistry,
    ModelCardGenerator,
)
from src.infrastructure.huggingface import (
    ChatTemplateFormatter,
    HuggingFaceDatasetLoader,
    HuggingFaceTokenizer,
)
from src.infrastructure.logging import (
    clear_logging_context,
    get_logging_context,
    set_logging_context,
    setup_logging,
)

__all__ = [
    "ChatTemplateFormatter",
    "ComputeProviderFactory",
    "DatasetLoaderFactory",
    "EvaluatorFactory",
    "HuggingFaceDatasetLoader",
    "HuggingFaceHubRegistry",
    "HuggingFaceTokenizer",
    "KaggleApiClient",
    "KaggleComputeProvider",
    "KaggleKernelBuilder",
    "LocalModelRegistry",
    "ModelCardGenerator",
    "ModelRegistryFactory",
    "TrainerFactory",
    "YamlConfigLoader",
    "setup_logging",
    "set_logging_context",
    "get_logging_context",
    "clear_logging_context",
]

