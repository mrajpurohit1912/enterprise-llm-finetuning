"""
tests/unit/test_dataset_loader.py
Unit tests for DataLoaderBase implementations and factories.
"""

import unittest
from unittest.mock import MagicMock, patch

from src.domain.exceptions import DatasetIngestionError
from src.domain.schemas.config_schema import DatasetConfig, DatasetSourceType
from src.infrastructure.factories.dataset_factory import DatasetLoaderFactory
from src.infrastructure.huggingface.huggingface_dataset_loader import HuggingFaceDatasetLoader


class TestDatasetLoader(unittest.TestCase):
    """Tests dataset loading adapters and factory dispatch logic."""

    def test_factory_returns_huggingface_loader(self) -> None:
        """Factory should return HuggingFaceDatasetLoader for HUGGINGFACE source type."""
        loader = DatasetLoaderFactory.get_loader(DatasetSourceType.HUGGINGFACE)
        self.assertIsInstance(loader, HuggingFaceDatasetLoader)

    def test_factory_raises_for_unimplemented_sources(self) -> None:
        """Planned but unbuilt sources (S3, Lakehouse) should raise DatasetIngestionError."""
        with self.assertRaises(DatasetIngestionError) as ctx:
            DatasetLoaderFactory.get_loader(DatasetSourceType.S3)
        self.assertIn("planned but not yet implemented", str(ctx.exception))

    def test_empty_dataset_name_raises_error(self) -> None:
        """Blank or empty dataset name should raise DatasetIngestionError."""
        loader = HuggingFaceDatasetLoader()
        with self.assertRaises(DatasetIngestionError) as ctx:
            loader.load_data("")
        self.assertIn("cannot be empty", str(ctx.exception))

    @patch("src.infrastructure.huggingface.huggingface_dataset_loader.load_dataset")
    def test_load_data_success_mock(self, mock_load: MagicMock) -> None:
        """Loader should call load_dataset and return the result."""
        mock_dataset = MagicMock()
        mock_load.return_value = mock_dataset

        loader = HuggingFaceDatasetLoader()
        result = loader.load_data("databricks/officeqa", split="train")

        mock_load.assert_called_once_with("databricks/officeqa", split="train")
        self.assertEqual(result, mock_dataset)

    @patch("src.infrastructure.huggingface.huggingface_dataset_loader.load_dataset")
    def test_load_data_failure_wrapped_in_domain_error(self, mock_load: MagicMock) -> None:
        """Downstream exceptions must be wrapped into DatasetIngestionError."""
        mock_load.side_effect = ConnectionError("Network unreachable")

        loader = HuggingFaceDatasetLoader()
        with self.assertRaises(DatasetIngestionError) as ctx:
            loader.load_data("databricks/officeqa")
        self.assertIn("Network unreachable", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
