"""
tests/unit/test_config_schema.py
Unit tests for domain configuration schemas and Pydantic validation rules.
"""

import unittest
from pathlib import Path
from pydantic import ValidationError

from src.domain.schemas.config_schema import (
    ArtifactConfig,
    DatasetConfig,
    DatasetSourceType,
    ExperimentConfig,
    ExperimentInfo,
    HardwareConfig,
    LLMModelConfig,
    MonitoringConfig,
    PEFTConfig,
    QuantizationConfig,
    RegistryConfig,
    TelemetryConfig,
)


class TestConfigSchema(unittest.TestCase):
    """Tests Pydantic validation, default values, and immutability of domain configurations."""

    def setUp(self) -> None:
        self.valid_data = {
            "experiment": {
                "name": "test-run-001",
                "project": "enterprise-tests",
                "seed": 1234,
                "tags": ["test", "lora"],
            },
            "dataset": {
                "source": "huggingface",
                "dataset_name": "databricks/officeqa",
                "train_split": "train",
                "eval_split": "test",
            },
            "llm_model": {
                "llm_model_id": "Qwen/Qwen2.5-0.5B-Instruct",
                "trust_remote_code": False,
            },
            "quantization": {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_use_double_quant": True,
            },
            "peft": {
                "task_type": "CAUSAL_LM",
                "r": 32,
                "lora_alpha": 64,
                "lora_dropout": 0.1,
                "bias": "none",
                "target_modules": ["q_proj", "v_proj"],
            },
            "hardware": {
                "distributed_strategy": "single_gpu",
            },
            "monitoring": {
                "logging_steps": 5,
                "eval_steps": 25,
                "save_steps": 50,
                "save_total_limit": 2,
                "telemetry": {
                    "enable_wandb": False,
                    "wandb_project": "test-org",
                    "enable_prometheus": True,
                    "prometheus_port": 9090,
                    "enable_tensorboard": False,
                    "mlflow": False,
                },
            },
            "registry": {
                "registry_type": "local",
                "model_name": "test-model",
                "auto_register": False,
            },
            "artifact": {
                "output_dir": "./outputs/test-run",
                "save_merged_model": False,
                "export_format": "safetensors",
            },
        }

    def test_valid_experiment_config_instantiation(self) -> None:
        """Valid dictionary should parse successfully into ExperimentConfig."""
        config = ExperimentConfig.model_validate(self.valid_data)
        self.assertEqual(config.experiment.name, "test-run-001")
        self.assertEqual(config.dataset.source, DatasetSourceType.HUGGINGFACE)
        self.assertEqual(config.peft.r, 32)
        self.assertEqual(config.peft.lora_alpha, 64)
        self.assertEqual(config.monitoring.telemetry.prometheus_port, 9090)
        self.assertFalse(config.artifact.save_merged_model)

    def test_default_values_populated(self) -> None:
        """Minimal required fields should populate sensible enterprise defaults."""
        minimal_data = {
            "experiment": {"name": "minimal-run"},
            "dataset": {"dataset_name": "imdb"},
            "llm_model": {"llm_model_id": "meta-llama/Llama-3-8B-Instruct"},
        }
        config = ExperimentConfig.model_validate(minimal_data)
        self.assertEqual(config.experiment.seed, 42)
        self.assertEqual(config.peft.r, 16)
        self.assertEqual(config.peft.lora_alpha, 32)
        self.assertEqual(config.quantization.bnb_4bit_quant_type, "nf4")
        self.assertEqual(config.artifact.export_format, "safetensors")

    def test_immutability_enforced(self) -> None:
        """Config attributes should be frozen and prevent in-place mutation."""
        config = ExperimentConfig.model_validate(self.valid_data)
        with self.assertRaises(ValidationError):
            config.experiment.name = "mutated-name"  # type: ignore

    def test_validation_error_on_invalid_rank(self) -> None:
        """PEFT rank <= 0 should raise ValidationError."""
        invalid_data = dict(self.valid_data)
        invalid_data["peft"] = {"r": 0}  # Invalid rank (ge=1)
        with self.assertRaises(ValidationError):
            ExperimentConfig.model_validate(invalid_data)

    def test_validation_error_on_invalid_prometheus_port(self) -> None:
        """Prometheus port outside valid TCP range should fail."""
        invalid_data = dict(self.valid_data)
        invalid_data["monitoring"] = {
            "telemetry": {"prometheus_port": 80}  # < 1024
        }
        with self.assertRaises(ValidationError):
            ExperimentConfig.model_validate(invalid_data)

    def test_forbid_extra_fields(self) -> None:
        """Unknown fields should be rejected to prevent silent typo bugs."""
        invalid_data = dict(self.valid_data)
        invalid_data["unknown_top_level_field"] = "bad_value"
        with self.assertRaises(ValidationError):
            ExperimentConfig.model_validate(invalid_data)

    def test_dynamic_output_dir_interpolation(self) -> None:
        """Interpolate {experiment.name} and ${experiment.name} dynamically in output_dir."""
        data = dict(self.valid_data)
        data["artifact"] = {"output_dir": "./outputs/{experiment.name}"}
        config = ExperimentConfig.model_validate(data)
        self.assertEqual(config.artifact.output_dir, Path("./outputs/test-run-001"))

        # Test ${experiment.name} pattern
        data["artifact"] = {"output_dir": "./artifacts/${experiment.name}/checkpoints"}
        config2 = ExperimentConfig.model_validate(data)
        self.assertEqual(config2.artifact.output_dir, Path("./artifacts/test-run-001/checkpoints"))


if __name__ == "__main__":
    unittest.main()
