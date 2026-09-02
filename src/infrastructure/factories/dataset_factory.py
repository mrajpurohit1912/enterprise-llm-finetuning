"""
src/infrastructure/factories/dataset_factory.py
Factory for instantiating DataLoaderBase concrete adapters based on source type.
"""

from src.domain.exceptions import DatasetIngestionError
from src.domain.interfaces.dataset_loader import DataLoaderBase
from src.domain.schemas.config_schema import DatasetSourceType
from src.infrastructure.huggingface.huggingface_dataset_loader import HuggingFaceDatasetLoader


class DatasetLoaderFactory:
    """Factory creating appropriate dataset loader adapters based on configuration source type."""

    @staticmethod
    def get_loader(source_type: DatasetSourceType) -> DataLoaderBase:
        """
        Instantiate and return the concrete dataset loader matching the source type.

        Args:
            source_type: Configured dataset source (e.g. HUGGINGFACE, S3, LOCAL).

        Returns:
            Concrete DataLoaderBase adapter.

        Raises:
            DatasetIngestionError: If the source type is unsupported.
        """
        if source_type == DatasetSourceType.HUGGINGFACE:
            return HuggingFaceDatasetLoader()
        elif source_type in (DatasetSourceType.LOCAL, DatasetSourceType.S3, DatasetSourceType.KAGGLE, DatasetSourceType.LAKEHOUSE):
            raise DatasetIngestionError(
                f"Dataset source type '{source_type.value}' is planned but not yet implemented. Supported: ['huggingface']."
            )
        else:
            raise DatasetIngestionError(f"Unsupported dataset source type: '{source_type}'")
