"""
src/infrastructure/registry/huggingface_registry.py
Infrastructure adapter for publishing fine-tuned models to the Hugging Face Hub Model Registry.
"""

import logging
import os
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

from src.domain.exceptions import ModelRegistryError
from src.domain.interfaces.model_registry import IModelRegistry
from src.infrastructure.registry.model_card import ModelCardGenerator

logger = logging.getLogger(__name__)


class HuggingFaceHubRegistry(IModelRegistry):
    """
    Adapter publishing standalone fine-tuned models and tokenizers to Hugging Face Hub.
    Handles repo creation, Model Card generation, and folder synchronization.
    """

    def __init__(
        self,
        token: str | None = None,
        api: HfApi | None = None,
    ) -> None:
        self._token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        self._api = api or HfApi(token=self._token)

    @property
    def provider_name(self) -> str:
        """Identifier for the registry backend."""
        return "hf_hub"

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
        Publish model weights, tokenizer, and Model Card to Hugging Face Hub.

        Args:
            artifact_path: Local filesystem directory containing model artifacts.
            model_name: Target Hugging Face Hub repository identifier (e.g. 'org/model-name').
            description: Summary description or Model Card body.
            tags: Category tags for search indexing.
            metadata: Training hyperparameters and evaluation metrics.
            private: Whether the repository visibility should be restricted to private.

        Returns:
            str: Web URL of the published model repository.

        Raises:
            ModelRegistryError: If validation, repository creation, or upload fails.
        """
        artifact_path = Path(artifact_path)
        if not artifact_path.exists() or not artifact_path.is_dir():
            raise ModelRegistryError(
                f"Artifact directory '{artifact_path}' does not exist or is not a directory."
            )

        # Check if directory contains model weights or configs
        files = list(artifact_path.glob("*"))
        if not files:
            raise ModelRegistryError(
                f"Artifact directory '{artifact_path}' is empty. Nothing to register."
            )

        # Ensure authentication token is present
        token = self._token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            raise ModelRegistryError(
                "Hugging Face API token is required to publish models. "
                "Please configure HF_TOKEN or HUGGINGFACE_TOKEN in your environment or .env file."
            )

        try:
            logger.info(
                "Initializing Hugging Face Hub repository '%s' (private=%s)...",
                model_name,
                private,
            )

            # 1. Generate or update README.md (Model Card) if not already explicitly present
            readme_path = artifact_path / "README.md"
            if not readme_path.exists() or metadata:
                base_model = metadata.get("base_model") if metadata else None
                card_content = ModelCardGenerator.generate(
                    model_name=model_name,
                    base_model=base_model,
                    description=description,
                    tags=tags,
                    metadata=metadata,
                )
                readme_path.write_text(card_content, encoding="utf-8")
                logger.info("Generated Model Card at '%s'.", readme_path)

            # 2. Create the remote repository if it does not already exist
            repo_url = self._api.create_repo(
                repo_id=model_name,
                private=private,
                exist_ok=True,
                repo_type="model",
            )
            logger.info("Target Hugging Face repository resolved: %s", repo_url)

            # 3. Synchronize and upload folder contents
            commit_message = f"Enterprise pipeline publication: {model_name}"
            logger.info(
                "Uploading model artifacts from '%s' to '%s'...",
                artifact_path,
                model_name,
            )
            self._api.upload_folder(
                folder_path=str(artifact_path),
                repo_id=model_name,
                repo_type="model",
                commit_message=commit_message,
            )

            published_url = f"https://huggingface.co/{model_name}"
            logger.info(
                "Successfully published model to Hugging Face Hub: %s", published_url
            )
            return published_url

        except HfHubHTTPError as http_err:
            logger.exception("Hugging Face Hub API error during publication.")
            raise ModelRegistryError(
                f"Hugging Face Hub API failure: {http_err}"
            ) from http_err
        except Exception as exc:
            logger.exception(
                "Failed to publish model '%s' to Hugging Face Hub.", model_name
            )
            raise ModelRegistryError(
                f"Failed to publish model '{model_name}' to Hugging Face Hub: {exc}"
            ) from exc
