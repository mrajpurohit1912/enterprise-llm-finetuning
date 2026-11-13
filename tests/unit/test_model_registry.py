"""
tests/unit/test_model_registry.py
Unit tests for the Model Registry infrastructure layer:
HuggingFaceHubRegistry, LocalModelRegistry, ModelCardGenerator, and ModelRegistryFactory.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from huggingface_hub.utils import HfHubHTTPError

from src.domain.exceptions import ConfigurationError, ModelRegistryError
from src.infrastructure.factories.registry_factory import ModelRegistryFactory
from src.infrastructure.registry.huggingface_registry import HuggingFaceHubRegistry
from src.infrastructure.registry.local_registry import LocalModelRegistry
from src.infrastructure.registry.model_card import ModelCardGenerator


class TestModelCardGenerator(unittest.TestCase):
    """Unit tests for automated Model Card markdown generation."""

    def test_generate_basic_model_card(self) -> None:
        """Generates valid markdown with frontmatter and model header."""
        card = ModelCardGenerator.generate(
            model_name="org/test-model",
            base_model="meta-llama/Llama-3.2-3B",
            description="Enterprise fine-tuned model for finance.",
            tags=["finance", "qlora"],
        )
        self.assertIn("---", card)
        self.assertIn("base_model: meta-llama/Llama-3.2-3B", card)
        self.assertIn("- finance", card)
        self.assertIn("- qlora", card)
        self.assertIn("# org/test-model", card)
        self.assertIn("Enterprise fine-tuned model for finance.", card)
        self.assertIn("## Usage Example", card)

    def test_generate_with_evaluation_and_hyperparameters(self) -> None:
        """Card should format hyperparameter and evaluation score tables."""
        metadata = {
            "dataset": "ChanceFocus/flare-finqa",
            "experiment_name": "exp-001",
            "hyperparameters": {
                "learning_rate": 0.0002,
                "batch_size": 4,
                "lora_r": 16,
            },
            "evaluation": {
                "task_type": "financial",
                "passed_quality_gate": True,
                "metrics": {
                    "f1_score": 0.885,
                    "exact_match": 0.720,
                },
                "delta_metrics": {
                    "delta_f1_score": 0.125,
                },
            },
        }

        card = ModelCardGenerator.generate(
            model_name="org/fin-model",
            base_model="meta-llama/Llama-3.2-3B",
            metadata=metadata,
        )

        self.assertIn("| `learning_rate` | `0.0002` |", card)
        self.assertIn("| `f1_score` | `0.8850` |", card)
        self.assertIn("| `delta_f1_score` | `+0.1250` |", card)
        self.assertIn("Quality Gate:** `PASSED`", card)


class TestLocalModelRegistry(unittest.TestCase):
    """Unit tests for the filesystem/NFS local model registry adapter."""

    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp())
        self.artifact_dir = self.test_dir / "artifacts"
        self.artifact_dir.mkdir()
        (self.artifact_dir / "model.safetensors").write_text("dummy-weights")
        (self.artifact_dir / "tokenizer.json").write_text("{}")
        self.registry_dir = self.test_dir / "registry"
        self.registry = LocalModelRegistry(registry_dir=self.registry_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_provider_name(self) -> None:
        """Provider name should be 'local'."""
        self.assertEqual(self.registry.provider_name, "local")

    def test_register_model_success(self) -> None:
        """Registers files to versioned target directory and creates README.md."""
        uri = self.registry.register_model(
            artifact_path=self.artifact_dir,
            model_name="org/my-model",
            metadata={"dataset": "test-data"},
        )
        self.assertTrue(uri.startswith("file://"))

        # Verify artifacts copied
        target_path = Path(uri.replace("file://", ""))
        self.assertTrue((target_path / "model.safetensors").exists())
        self.assertTrue((target_path / "tokenizer.json").exists())
        self.assertTrue((target_path / "README.md").exists())
        self.assertTrue((target_path / "registry_metadata.json").exists())

        # Verify 'latest' directory exists
        latest_path = self.registry_dir / "org/my-model" / "latest"
        self.assertTrue(latest_path.exists())

    def test_register_model_nonexistent_dir_raises(self) -> None:
        """Non-existent artifact directory should raise ModelRegistryError."""
        with self.assertRaises(ModelRegistryError):
            self.registry.register_model(
                artifact_path=self.test_dir / "does_not_exist",
                model_name="test-model",
            )

    def test_register_model_empty_dir_raises(self) -> None:
        """Empty artifact directory should raise ModelRegistryError."""
        empty_dir = self.test_dir / "empty"
        empty_dir.mkdir()
        with self.assertRaises(ModelRegistryError):
            self.registry.register_model(
                artifact_path=empty_dir,
                model_name="test-model",
            )


class TestHuggingFaceHubRegistry(unittest.TestCase):
    """Unit tests for the Hugging Face Hub model registry adapter."""

    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp())
        self.artifact_dir = self.test_dir / "artifacts"
        self.artifact_dir.mkdir()
        (self.artifact_dir / "model.safetensors").write_text("dummy-weights")
        self.mock_api = MagicMock()

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_provider_name(self) -> None:
        """Provider name should be 'hf_hub'."""
        registry = HuggingFaceHubRegistry(token="dummy-token", api=self.mock_api)
        self.assertEqual(registry.provider_name, "hf_hub")

    def test_missing_token_raises_error(self) -> None:
        """Registering without any HF token should raise ModelRegistryError."""
        with patch.dict("os.environ", {}, clear=True):
            registry = HuggingFaceHubRegistry(token=None, api=self.mock_api)
            with self.assertRaises(ModelRegistryError) as ctx:
                registry.register_model(
                    artifact_path=self.artifact_dir,
                    model_name="user/test-model",
                )
            self.assertIn("Hugging Face API token is required", str(ctx.exception))

    def test_register_model_success(self) -> None:
        """Successful model registration invokes create_repo and upload_folder."""
        self.mock_api.create_repo.return_value = (
            "https://huggingface.co/user/test-model"
        )

        registry = HuggingFaceHubRegistry(token="hf_test_token", api=self.mock_api)
        url = registry.register_model(
            artifact_path=self.artifact_dir,
            model_name="user/test-model",
            description="Fine-tuned test model",
            private=True,
        )

        self.assertEqual(url, "https://huggingface.co/user/test-model")
        self.mock_api.create_repo.assert_called_once_with(
            repo_id="user/test-model",
            private=True,
            exist_ok=True,
            repo_type="model",
        )
        self.mock_api.upload_folder.assert_called_once()
        self.assertTrue((self.artifact_dir / "README.md").exists())

    def test_api_http_error_wraps_in_domain_error(self) -> None:
        """HfHubHTTPError should be wrapped in ModelRegistryError."""
        mock_response = MagicMock(status_code=401)
        self.mock_api.create_repo.side_effect = HfHubHTTPError(
            "Authentication 401 Unauthorized", response=mock_response
        )

        registry = HuggingFaceHubRegistry(token="hf_test_token", api=self.mock_api)
        with self.assertRaises(ModelRegistryError) as ctx:
            registry.register_model(
                artifact_path=self.artifact_dir,
                model_name="user/test-model",
            )
        self.assertIn("Hugging Face Hub API failure", str(ctx.exception))


class TestModelRegistryFactory(unittest.TestCase):
    """Unit tests for ModelRegistryFactory resolution and registration."""

    def test_get_registry_hf_hub(self) -> None:
        """Resolves HuggingFaceHubRegistry for 'hf_hub' and 'huggingface'."""
        with patch("src.infrastructure.registry.huggingface_registry.HfApi"):
            reg1 = ModelRegistryFactory.get_registry("hf_hub", token="test")
            self.assertIsInstance(reg1, HuggingFaceHubRegistry)

            reg2 = ModelRegistryFactory.get_registry("huggingface", token="test")
            self.assertIsInstance(reg2, HuggingFaceHubRegistry)

    def test_get_registry_local(self) -> None:
        """Resolves LocalModelRegistry for 'local'."""
        reg = ModelRegistryFactory.get_registry("local")
        self.assertIsInstance(reg, LocalModelRegistry)

    def test_get_registry_unsupported_raises(self) -> None:
        """Unsupported registry type raises ConfigurationError."""
        with self.assertRaises(ConfigurationError) as ctx:
            ModelRegistryFactory.get_registry("unknown_cloud")
        self.assertIn(
            "Unsupported model registry type 'unknown_cloud'", str(ctx.exception)
        )

    def test_register_provider_dynamic(self) -> None:
        """Supports dynamic registration of custom registry adapters."""
        mock_custom_cls = MagicMock()
        ModelRegistryFactory.register_provider("custom_s3", mock_custom_cls)
        instance = ModelRegistryFactory.get_registry("custom_s3")
        mock_custom_cls.assert_called_once()
        self.assertEqual(instance, mock_custom_cls.return_value)
