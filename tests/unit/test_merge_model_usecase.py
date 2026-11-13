"""
tests/unit/test_merge_model_usecase.py
Unit tests verifying MergeModelUseCase fusing, fallback directory resolution, and error handling.
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.application.usecases.merge_model_usecase import MergeModelUseCase
from src.domain.exceptions import ModelMergeError
from src.domain.interfaces.model_loader import ModelLoaderBase
from src.domain.interfaces.tokenizer import TokenizerBase
from src.domain.schemas.config_schema import (
    ArtifactConfig,
    DatasetConfig,
    DatasetSourceType,
    ExperimentConfig,
    ExperimentInfo,
    LLMModelConfig,
)


class TestMergeModelUseCase(unittest.TestCase):
    """Unit tests for atomic model merging use case."""

    def setUp(self) -> None:
        self.config = ExperimentConfig(
            experiment=ExperimentInfo(name="test-merge-exp"),
            dataset=DatasetConfig(source=DatasetSourceType.HUGGINGFACE, dataset_name="test/dataset"),
            llm_model=LLMModelConfig(llm_model_id="meta-llama/Llama-3.2-3B-Instruct"),
            artifact=ArtifactConfig(output_dir=Path("outputs/test-merge-exp"), save_merged_model=True),
        )
        self.mock_model_loader = MagicMock(spec=ModelLoaderBase)
        self.mock_tokenizer_loader = MagicMock(spec=TokenizerBase)

    @patch("pathlib.Path.exists")
    def test_merge_and_unload_model_success(self, mock_exists: MagicMock) -> None:
        mock_exists.return_value = True

        mock_base_model = MagicMock()
        mock_peft_model = MagicMock()
        mock_tokenizer = MagicMock()

        self.mock_model_loader.load_base_model.return_value = mock_base_model
        self.mock_model_loader.load_adapters.return_value = mock_peft_model
        self.mock_tokenizer_loader.get_tokenizer.return_value = mock_tokenizer

        usecase = MergeModelUseCase(
            config=self.config,
            model_loader=self.mock_model_loader,
            tokenizer_loader=self.mock_tokenizer_loader,
        )

        with patch("pathlib.Path.mkdir"):
            export_path = usecase.execute()

        self.mock_model_loader.load_base_model.assert_called_once_with(
            model_id="meta-llama/Llama-3.2-3B-Instruct",
            quantization_config=None,
            device_map="cpu",
        )
        self.mock_model_loader.load_adapters.assert_called_once()
        self.mock_model_loader.save_merged_model.assert_called_once()
        self.mock_tokenizer_loader.get_tokenizer.assert_called_once()
        mock_tokenizer.save_pretrained.assert_called_once()
        self.assertEqual(export_path, Path("outputs/test-merge-exp/merged_model"))

    @patch("pathlib.Path.exists")
    def test_merge_failure_raises_model_merge_error(self, mock_exists: MagicMock) -> None:
        mock_exists.return_value = True
        self.mock_model_loader.load_base_model.side_effect = RuntimeError("CPU OOM")

        usecase = MergeModelUseCase(
            config=self.config,
            model_loader=self.mock_model_loader,
        )

        with self.assertRaises(ModelMergeError) as ctx:
            usecase.execute()

        self.assertIn("Failed to merge model", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
