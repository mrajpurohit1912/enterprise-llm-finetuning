"""
tests/unit/test_register_model_usecase.py
Unit tests for RegisterModelUseCase in application layer.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.application.usecases.register_model_usecase import RegisterModelUseCase
from src.domain.exceptions import ModelRegistryError
from src.domain.schemas.config_schema import (
    ArtifactConfig,
    DatasetConfig,
    DatasetSourceType,
    ExperimentConfig,
    ExperimentInfo,
    LLMModelConfig,
    PEFTConfig,
    RegistryConfig,
    TrainingArgs,
)
from src.domain.schemas.evaluation_schema import (
    EvaluationResult,
    MetricScores,
    TaskType,
)


class TestRegisterModelUseCase(unittest.TestCase):
    """Unit tests for RegisterModelUseCase orchestrator."""

    def setUp(self) -> None:
        self.mock_registry = MagicMock()
        self.mock_registry.provider_name = "mock_registry"
        self.mock_registry.register_model.return_value = (
            "https://registry.example.com/models/test-model"
        )

        self.temp_dir = Path(tempfile.mkdtemp())
        self.artifact_dir = self.temp_dir / "outputs"
        self.artifact_dir.mkdir()
        (self.artifact_dir / "adapter_model.safetensors").write_text("weights")

        self.config = ExperimentConfig(
            experiment=ExperimentInfo(name="unit-test-exp", tags=["test", "finance"]),
            dataset=DatasetConfig(
                source=DatasetSourceType.HUGGINGFACE, dataset_name="test/dataset"
            ),
            llm_model=LLMModelConfig(llm_model_id="meta-llama/Llama-3.2-3B"),
            peft=PEFTConfig(r=16, lora_alpha=32),
            training_args=TrainingArgs(),
            registry=RegistryConfig(
                registry_type="hf_hub",
                model_name="org/test-finetuned-model",
                auto_register=True,
            ),
            artifact=ArtifactConfig(
                output_dir=str(self.artifact_dir),
                save_merged_model=True,
            ),
        )

    def test_null_registry_raises_error(self) -> None:
        """Instantiating use case without a registry adapter must raise ModelRegistryError."""
        with self.assertRaises(ModelRegistryError):
            RegisterModelUseCase(registry=None)  # type: ignore

    def test_register_adapter_directory(self) -> None:
        """When merged model is not present, adapter directory is registered."""
        usecase = RegisterModelUseCase(registry=self.mock_registry)
        uri = usecase.execute(artifact_path=self.artifact_dir, config=self.config)

        self.assertEqual(uri, "https://registry.example.com/models/test-model")
        self.mock_registry.register_model.assert_called_once()
        call_kwargs = self.mock_registry.register_model.call_args.kwargs
        self.assertEqual(call_kwargs["artifact_path"], self.artifact_dir)
        self.assertEqual(call_kwargs["model_name"], "org/test-finetuned-model")
        self.assertIn("finance", call_kwargs["tags"])
        self.assertIn("fine-tuned", call_kwargs["tags"])
        self.assertEqual(
            call_kwargs["metadata"]["base_model"], "meta-llama/Llama-3.2-3B"
        )

    def test_register_merged_directory_when_present(self) -> None:
        """When merged_model/ subdirectory exists, it is selected as the registration target."""
        merged_dir = self.artifact_dir / "merged_model"
        merged_dir.mkdir()
        (merged_dir / "model.safetensors").write_text("merged-weights")

        usecase = RegisterModelUseCase(registry=self.mock_registry)
        uri = usecase.execute(artifact_path=self.artifact_dir, config=self.config)

        self.assertEqual(uri, "https://registry.example.com/models/test-model")
        call_kwargs = self.mock_registry.register_model.call_args.kwargs
        self.assertEqual(call_kwargs["artifact_path"], merged_dir)

    def test_register_with_evaluation_result(self) -> None:
        """Evaluation results are serialized into metadata."""
        eval_result = EvaluationResult(
            experiment_name="unit-test-exp",
            model_id="test/model",
            eval_split="test",
            total_samples=10,
            latency_per_sample_ms=25.0,
            throughput_tokens_per_sec=40.0,
            task_type=TaskType.TEXT_GENERATION,
            metrics=MetricScores(f1_score=0.92, exact_match=0.85),
            passed_quality_gate=True,
        )

        usecase = RegisterModelUseCase(registry=self.mock_registry)
        usecase.execute(
            artifact_path=self.artifact_dir,
            config=self.config,
            evaluation_result=eval_result,
        )

        call_kwargs = self.mock_registry.register_model.call_args.kwargs
        self.assertIn("evaluation", call_kwargs["metadata"])
        self.assertTrue(call_kwargs["metadata"]["evaluation"]["passed_quality_gate"])

    def test_registry_failure_raises_model_registry_error(self) -> None:
        """Underlying registry adapter failures are raised as ModelRegistryError."""
        self.mock_registry.register_model.side_effect = RuntimeError("Network timeout")

        usecase = RegisterModelUseCase(registry=self.mock_registry)
        with self.assertRaises(ModelRegistryError) as ctx:
            usecase.execute(artifact_path=self.artifact_dir, config=self.config)
        self.assertIn("Failed to register model", str(ctx.exception))
