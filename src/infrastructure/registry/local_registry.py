"""
src/infrastructure/registry/local_registry.py
Infrastructure adapter for registering models to local enterprise disk storage or mounted shared volumes.
"""

import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.domain.exceptions import ModelRegistryError
from src.domain.interfaces.model_registry import IModelRegistry
from src.infrastructure.registry.model_card import ModelCardGenerator

logger = logging.getLogger(__name__)


class LocalModelRegistry(IModelRegistry):
    """
    Adapter storing versioned fine-tuned models on a local filesystem or mounted NFS/SAN.
    """

    def __init__(self, registry_dir: Path | None = None) -> None:
        self.registry_dir = Path(registry_dir or "outputs/model_registry")

    @property
    def provider_name(self) -> str:
        """Identifier for the local registry backend."""
        return "local"

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
        Copy model artifacts to versioned local storage and persist Model Card & metadata.

        Args:
            artifact_path: Source directory containing trained weights and tokenizer.
            model_name: Target identifier for the model.
            description: Summary description or Model Card markdown.
            tags: Category tags.
            metadata: Training hyperparameters and evaluation metrics.
            private: Ignored for local registry (permissions determined by filesystem).

        Returns:
            str: file:// URI of the registered model version.

        Raises:
            ModelRegistryError: If source files are missing or copying fails.
        """
        artifact_path = Path(artifact_path)
        if not artifact_path.exists() or not artifact_path.is_dir():
            raise ModelRegistryError(
                f"Artifact directory '{artifact_path}' does not exist or is not a directory."
            )

        files = list(artifact_path.glob("*"))
        if not files:
            raise ModelRegistryError(
                f"Artifact directory '{artifact_path}' is empty. Nothing to register."
            )

        try:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            # Sanitize model_name for path safely (e.g. 'org/model-name' -> 'org/model-name')
            target_version_dir = self.registry_dir / model_name / timestamp
            target_version_dir.mkdir(parents=True, exist_ok=True)

            logger.info(
                "Registering model '%s' locally to '%s'...",
                model_name,
                target_version_dir,
            )

            # Copy artifacts
            for item in artifact_path.iterdir():
                dest = target_version_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)

            # Write or update Model Card
            readme_path = target_version_dir / "README.md"
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

            # Persist raw metadata json for programmatic querying
            if metadata:
                metadata_path = target_version_dir / "registry_metadata.json"
                metadata_path.write_text(
                    json.dumps(metadata, indent=2, default=str), encoding="utf-8"
                )

            # Update 'latest' pointer directory
            latest_dir = self.registry_dir / model_name / "latest"
            if latest_dir.exists():
                if latest_dir.is_symlink() or latest_dir.is_file():
                    latest_dir.unlink()
                else:
                    shutil.rmtree(latest_dir)
            try:
                latest_dir.symlink_to(
                    target_version_dir.resolve(), target_is_directory=True
                )
            except OSError:
                # Fallback if symlinks not supported on filesystem
                shutil.copytree(target_version_dir, latest_dir, dirs_exist_ok=True)

            resolved_uri = f"file://{target_version_dir.resolve()}"
            logger.info("Successfully registered model locally: %s", resolved_uri)
            return resolved_uri

        except Exception as exc:
            logger.exception("Failed to register model locally.")
            raise ModelRegistryError(f"Local model registration failed: {exc}") from exc
