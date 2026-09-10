"""
tests/unit/test_yaml_loader.py
Unit tests for YamlConfigLoader adapter.
"""

import tempfile
import unittest
from pathlib import Path

from src.domain.exceptions import ConfigurationError
from src.infrastructure.config.yaml_loader import YamlConfigLoader


class TestYamlConfigLoader(unittest.TestCase):
    """Tests YamlConfigLoader error handling and validation."""

    def test_load_valid_yaml(self) -> None:
        """Valid YAML configuration should parse cleanly."""
        loader = YamlConfigLoader(config_path="src/finetuning_config.yaml")
        config = loader.load_config()
        self.assertTrue(config.experiment.name.startswith("qwen2.5-0.5b-officeqa"))
        self.assertEqual(config.dataset.dataset_name, "databricks/officeqa")
        self.assertEqual(config.llm_model.llm_model_id, "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertTrue(config.quantization.load_in_4bit)

    def test_file_not_found_raises_configuration_error(self) -> None:
        """Missing YAML file must raise domain ConfigurationError."""
        loader = YamlConfigLoader(config_path="non_existent_config.yaml")
        with self.assertRaises(ConfigurationError) as ctx:
            loader.load_config()
        self.assertIn("Configuration file not found", str(ctx.exception))

    def test_invalid_yaml_syntax_raises_configuration_error(self) -> None:
        """Malformed YAML syntax must raise ConfigurationError."""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
            tmp.write("experiment: [invalid yaml content : {broken")
            tmp_path = tmp.name

        try:
            loader = YamlConfigLoader(config_path=tmp_path)
            with self.assertRaises(ConfigurationError) as ctx:
                loader.load_config()
            self.assertIn("Failed to parse YAML syntax", str(ctx.exception))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_schema_validation_failure_raises_configuration_error(self) -> None:
        """Invalid fields failing Pydantic constraints must raise ConfigurationError."""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
            tmp.write("""
experiment:
  name: "bad-config"
dataset:
  source: "invalid_source_type"
  dataset_name: "test"
llm_model:
  llm_model_id: "test-model"
""")
            tmp_path = tmp.name

        try:
            loader = YamlConfigLoader(config_path=tmp_path)
            with self.assertRaises(ConfigurationError) as ctx:
                loader.load_config()
            self.assertIn("Configuration schema validation failed", str(ctx.exception))
        finally:
            Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
