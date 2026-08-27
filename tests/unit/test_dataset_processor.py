"""
tests/unit/test_dataset_processor.py
Unit tests for DatasetProcessor application service.
"""

import unittest
from unittest.mock import MagicMock
from datasets import Dataset, DatasetDict

from src.application.services.dataset_processor import DatasetProcessor
from src.domain.exceptions import PreprocessingError
from src.domain.interfaces.prompt_formatter import PromptFormatterBase


class DummyFormatter(PromptFormatterBase):
    """Dummy formatter for unit testing processor."""

    def format_batch(self, batch):
        texts = [f"formatted: {q} -> {a}" for q, a in zip(batch["question"], batch["answer"])]
        return {"text": texts}


class TestDatasetProcessor(unittest.TestCase):
    """Tests DatasetProcessor mapping and validation over datasets."""

    def setUp(self) -> None:
        self.formatter = DummyFormatter()
        self.processor = DatasetProcessor(formatter=self.formatter)
        self.raw_data = {
            "question": ["Q1", "Q2"],
            "answer": ["A1", "A2"],
        }

    def test_null_formatter_raises_error(self) -> None:
        """Initializing with None formatter must raise PreprocessingError."""
        with self.assertRaises(PreprocessingError):
            DatasetProcessor(formatter=None)  # type: ignore

    def test_process_dataset_split(self) -> None:
        """Processing single Dataset split should add formatted 'text' column."""
        dataset = Dataset.from_dict(self.raw_data)
        processed = self.processor.process(dataset)

        self.assertIsInstance(processed, Dataset)
        self.assertIn("text", processed.column_names)
        self.assertEqual(processed[0]["text"], "formatted: Q1 -> A1")
        self.assertEqual(processed[1]["text"], "formatted: Q2 -> A2")

    def test_process_dataset_dict(self) -> None:
        """Processing DatasetDict must process all splits."""
        dataset_dict = DatasetDict({
            "train": Dataset.from_dict(self.raw_data),
            "test": Dataset.from_dict({"question": ["Q3"], "answer": ["A3"]}),
        })
        processed = self.processor.process(dataset_dict)

        self.assertIsInstance(processed, DatasetDict)
        self.assertIn("train", processed)
        self.assertIn("test", processed)
        self.assertEqual(processed["train"][0]["text"], "formatted: Q1 -> A1")
        self.assertEqual(processed["test"][0]["text"], "formatted: Q3 -> A3")

    def test_process_with_remove_columns(self) -> None:
        """Removing original columns should leave only 'text' column."""
        dataset = Dataset.from_dict(self.raw_data)
        processed = self.processor.process(dataset, remove_columns=True)

        self.assertEqual(processed.column_names, ["text"])

    def test_null_dataset_raises_error(self) -> None:
        """Processing None dataset must raise PreprocessingError."""
        with self.assertRaises(PreprocessingError):
            self.processor.process(None)  # type: ignore


if __name__ == "__main__":
    unittest.main()
