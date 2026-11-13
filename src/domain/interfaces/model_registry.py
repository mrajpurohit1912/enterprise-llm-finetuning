"""
src/domain/interfaces/model_registry.py
Abstract base interface for model registry backends following the Dependency Inversion Principle.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class IModelRegistry(ABC):
    """
    Abstract Port contract for model registry backends.
    Concrete adapters (e.g. HuggingFaceHubRegistry, MLflowRegistry) implement this.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Identifier for the registry backend (e.g. 'hf_hub', 'mlflow', 's3')."""

    @abstractmethod
    def register_model(
        self,
        artifact_path: Path,
        model_name: str,
        description: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        private: bool = True,
    ) -> str:
        """
        Publish and version a standalone fine-tuned model artifact to the registry.

        Args:
            artifact_path: Local filesystem directory containing model weights (safetensors) and tokenizer.
            model_name: Target model identifier or repository slug (e.g. 'org/model-name').
            description: Summary description or generated Model Card markdown content.
            tags: Categorical tags for indexing (e.g. ['finance', 'lora', 'llama-3']).
            metadata: Training hyperparameters, dataset lineage, and evaluation metrics.
            private: Whether the repository/model visibility should be restricted to private.

        Returns:
            str: Web URL or URI referencing the published model in the registry.

        Raises:
            ModelRegistryError: If authentication, file upload, or registration fails.
        """


# Backwards compatibility alias matching other interfaces (e.g., EvaluatorBase)
ModelRegistryBase = IModelRegistry

__all__ = ["IModelRegistry", "ModelRegistryBase"]
