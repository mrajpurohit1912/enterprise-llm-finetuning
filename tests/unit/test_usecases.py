"""
tests/unit/test_usecases.py
Unit tests for application use cases (LoadDatasetUseCase, PreprocessDatasetUseCase, TrainPipelineUsecase).
"""

import unittest
from unittest.mock import MagicMock
from datasets import Dataset, DatasetDict

from src.application.services.dataset_processor import DatasetProcessor
from src.application.usecases.load_dataset_usecase import LoadDatasetUseCase
from src.application.usecases.preprocess_dataset_usecase import PreprocessDatasetUseCase
from src.application.usecases.train_pipeline import TrainPipelineUsecase
from src.domain.exceptions import DatasetIngestionError, PreprocessingError
from src.domain.schemas.config_schema import (
    DatasetConfig,
    DatasetSourceType,
    ExperimentConfig,
    ExperimentInfo,
    LLMModelConfig,
)


class TestUseCases(unittest.TestCase):
    """Tests application orchestration use cases in isolation using mocks."""

    def setUp(self) -> None:
        self.mock_dataset = DatasetDict({
            "train": Dataset.from_dict({"question": ["What is PEFT?"], "answer": ["Parameter-Efficient Fine-Tuning."]}),
        })

    def test_load_dataset_usecase_success(self) -> None:
        """LoadDatasetUseCase should delegate to loader and return dataset."""
        mock_loader = MagicMock()
        mock_loader.load_data.return_value = self.mock_dataset

        usecase = LoadDatasetUseCase(loader=mock_loader)
        config = DatasetConfig(source=DatasetSourceType.HUGGINGFACE, dataset_name="databricks/officeqa")
        result = usecase.execute(config)

        mock_loader.load_data.assert_called_once_with(dataset_name="databricks/officeqa", subset_name=None)
        self.assertEqual(result, self.mock_dataset)

    def test_load_dataset_usecase_null_loader_raises(self) -> None:
        """Initializing with None loader must raise error."""
        with self.assertRaises(DatasetIngestionError):
            LoadDatasetUseCase(loader=None)  # type: ignore

    def test_preprocess_dataset_usecase_success(self) -> None:
        """PreprocessDatasetUseCase should delegate to processor."""
        mock_processor = MagicMock()
        mock_processor.process.return_value = "processed_dataset"

        usecase = PreprocessDatasetUseCase(processor=mock_processor)
        result = usecase.execute(self.mock_dataset)

        mock_processor.process.assert_called_once_with(
            dataset=self.mock_dataset,
            remove_columns=False,
            num_proc=None,
        )
        self.assertEqual(result, "processed_dataset")

    def test_train_pipeline_orchestration(self) -> None:
        """TrainPipelineUsecase should orchestrate config, data loading, tokenizing, and preprocessing."""
        real_config = ExperimentConfig(
            experiment=ExperimentInfo(name="test-pipeline"),
            dataset=DatasetConfig(source=DatasetSourceType.HUGGINGFACE, dataset_name="test/dataset"),
            llm_model=LLMModelConfig(llm_model_id="test/model", trust_remote_code=False),
        )

        mock_config_loader = MagicMock()
        mock_config_loader.load_config.return_value = real_config

        mock_data_loader = MagicMock()
        mock_data_loader.load_data.return_value = self.mock_dataset

        mock_factory = MagicMock()
        mock_factory.get_loader.return_value = mock_data_loader

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "<user>What is PEFT?</user>"
        mock_tokenizer.chat_template = "mock_template"

        mock_tokenizer_loader = MagicMock()
        mock_tokenizer_loader.get_tokenizer.return_value = mock_tokenizer

        mock_llm_loader = MagicMock()
        mock_base_model = MagicMock()
        mock_peft_model = MagicMock()
        mock_llm_loader.load_model.return_value = mock_base_model
        mock_llm_loader.apply_peft.return_value = mock_peft_model

        pipeline = TrainPipelineUsecase(
            config_loader=mock_config_loader,
            dataset_loader_factory=mock_factory,
            tokenizer_loader=mock_tokenizer_loader,
            llm_model_loader=mock_llm_loader,
        )

        result = pipeline.run()

        mock_config_loader.load_config.assert_called_once()
        mock_factory.get_loader.assert_called_once_with(DatasetSourceType.HUGGINGFACE)
        mock_data_loader.load_data.assert_called_once_with(dataset_name="test/dataset", subset_name=None)
        mock_tokenizer_loader.get_tokenizer.assert_called_once_with(
            model_name="test/model", trust_remote_code=False
        )
        mock_llm_loader.load_model.assert_called_once()
        mock_llm_loader.apply_peft.assert_called_once()
        self.assertEqual(result, mock_peft_model)


if __name__ == "__main__":
    unittest.main()
