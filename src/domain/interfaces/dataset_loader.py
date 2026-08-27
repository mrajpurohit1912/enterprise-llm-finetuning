"""
src/domain/interfaces/dataset_loader.py
Port interface for dataset ingestion.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class DataLoaderBase(ABC):
    """Abstract port for all dataset ingestion adapters (HuggingFace, S3, Local, Lakehouse)."""

    @abstractmethod
    def load_data(self, dataset_name: str, split: Optional[str] = None) -> Any:
        """
        Fetch and load a dataset into a standard dataset structure.

        Args:
            dataset_name: Dataset identifier, URI, or local file path.
            split: Optional specific split to load (e.g. 'train', 'test').

        Returns:
            Loaded dataset object (e.g., DatasetDict or Dataset).

        Raises:
            DatasetIngestionError: If loading fails or data source is unreachable.
        """
        raise NotImplementedError("load_data method must be implemented by concrete adapter.")
