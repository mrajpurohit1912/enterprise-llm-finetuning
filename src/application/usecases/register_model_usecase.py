"""
src/application/usecases/register_model_usecase.py
Atomic application use case orchestrating the publication of fine-tuned models to an enterprise model registry.
"""

import logging
from pathlib import Path
from typing import Any

from src.domain.exceptions import ModelRegistryError
from src.domain.interfaces.model_registry import IModelRegistry
from src.domain.schemas.config_schema import ExperimentConfig
from src.domain.schemas.evaluation_schema import EvaluationResult

logger = logging.getLogger(__name__)


class RegisterModelUseCase:
    """
    Orchestrates publishing a trained model and tokenizer to a model registry,
    bundling complete provenance lineage, hyperparameters, and evaluation scorecards.
    """

    def __init__(self, registry: IModelRegistry) -> None:
        if registry is None:
            raise ModelRegistryError("IModelRegistry implementation cannot be None.")
        self._registry = registry

    @property
    def registry(self) -> IModelRegistry:
        """Underlying registry provider adapter."""
        return self._registry

    def execute(
        self,
        artifact_path: Path,
        config: ExperimentConfig,
        evaluation_result: EvaluationResult | None = None,
        custom_metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Execute model registration with provenance metadata.

        Args:
            artifact_path: Base directory containing outputs (weights, checkpoints, merged_model).
            config: Validated experiment configuration.
            evaluation_result: Optional evaluation metrics and scorecard.
            custom_metadata: Optional additional metadata dict.

        Returns:
            str: Published model URI or repository URL.

        Raises:
            ModelRegistryError: If registration fails.
        """
        artifact_path = Path(artifact_path)
        logger.info(
            "Executing RegisterModelUseCase using backend '%s' for model '%s'...",
            self._registry.provider_name,
            config.registry.model_name,
        )

        # 1. Resolve source directory: prefer standalone merged model if enabled and present
        target_dir = artifact_path
        merged_dir = artifact_path / "merged_model"
        if config.artifact.save_merged_model and merged_dir.exists():
            target_dir = merged_dir
            logger.info(
                "Selected merged standalone model directory for registration: '%s'",
                target_dir,
            )
        else:
            logger.info("Selected adapter directory for registration: '%s'", target_dir)

        # 2. Compile enterprise lineage metadata
        metadata: dict[str, Any] = {
            "experiment_name": config.experiment.name,
            "base_model": config.llm_model.llm_model_id,
            "dataset": config.dataset.dataset_name,
            "hyperparameters": config.training_args.model_dump(),
        }
        if config.peft:
            metadata["peft"] = config.peft.model_dump()
        if config.quantization:
            metadata["quantization"] = config.quantization.model_dump()
        if evaluation_result is not None:
            metadata["evaluation"] = evaluation_result.model_dump()
        if custom_metadata:
            metadata.update(custom_metadata)

        # 3. Assemble tags
        tags = list(config.experiment.tags) if config.experiment.tags else []
        for default_tag in ("fine-tuned", "transformers", "lora"):
            if default_tag not in tags:
                tags.append(default_tag)

        # 4. Description
        description = (
            f"Enterprise fine-tuned model `{config.registry.model_name}` trained from `{config.llm_model.llm_model_id}` "
            f"on dataset `{config.dataset.dataset_name}` as part of experiment `{config.experiment.name}`."
        )

        # 5. Delegate to registry adapter
        try:
            published_uri = self._registry.register_model(
                artifact_path=target_dir,
                model_name=config.registry.model_name,
                description=description,
                tags=tags,
                metadata=metadata,
                private=True,
            )
            logger.info("Successfully registered model: %s", published_uri)
            return published_uri
        except Exception as exc:
            logger.exception("Model registration failed in RegisterModelUseCase.")
            if isinstance(exc, ModelRegistryError):
                raise
            raise ModelRegistryError(f"Failed to register model: {exc}") from exc
