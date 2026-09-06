"""
src/infrastructure/huggingface/huggingface_dataset_loader.py
Hugging Face dataset loader adapter implementing DataLoaderBase.
"""

from typing import Any, Optional
from datasets import Dataset, DatasetDict, load_dataset

from src.domain.exceptions import DatasetIngestionError
from src.domain.interfaces.dataset_loader import DataLoaderBase


class HuggingFaceDatasetLoader(DataLoaderBase):
    """Loads datasets from the Hugging Face Hub or local filesystem via the datasets library."""

    def load_data(
        self,
        dataset_name: str,
        split: Optional[str] = None,
        subset_name: Optional[str] = None,
    ) -> Any:
        """
        Load dataset from Hugging Face Hub or local cache.

        Args:
            dataset_name: Hugging Face dataset identifier or local path.
            split: Specific split to load (e.g., 'train', 'test').
            subset_name: Optional dataset configuration or subset name.

        Returns:
            DatasetDict or Dataset instance.

        Raises:
            DatasetIngestionError: If the dataset cannot be downloaded, parsed, or split.
        """
        if not dataset_name or not dataset_name.strip():
            raise DatasetIngestionError("Dataset name cannot be empty or blank.")

        try:
            dataset = load_dataset(
                path=dataset_name.strip(),
                name=subset_name.strip() if subset_name else None,
                split=split,
            )
            return dataset
        except Exception as exc:
            raise DatasetIngestionError(
                f"Failed to load dataset '{dataset_name}' (subset: {subset_name}, split: {split}): {exc}"
            ) from exc
