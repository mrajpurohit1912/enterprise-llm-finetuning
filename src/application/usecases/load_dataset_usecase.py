"""
src/application/usecases/load_dataset_usecase.py
Use case for orchestrating dataset ingestion.
"""

from typing import Any
from src.domain.exceptions import DatasetIngestionError
from src.domain.interfaces.dataset_loader import DataLoaderBase
from src.domain.schemas.config_schema import DatasetConfig


class LoadDatasetUseCase:
    """Use case encapsulating dataset retrieval and validation from external or local sources."""

    def __init__(self, loader: DataLoaderBase) -> None:
        if loader is None:
            raise DatasetIngestionError("DataLoader cannot be None for LoadDatasetUseCase.")
        self._loader = loader

    def execute(self, dataset_config: DatasetConfig) -> Any:
        """
        Execute dataset loading based on configuration specification.

        Args:
            dataset_config: Validated DatasetConfig domain model.

        Returns:
            Dataset or DatasetDict object.

        Raises:
            DatasetIngestionError: If dataset loading fails.
        """
        if not dataset_config:
            raise DatasetIngestionError("DatasetConfig is required to execute LoadDatasetUseCase.")

        return self._loader.load_data(dataset_name=dataset_config.dataset_name)
