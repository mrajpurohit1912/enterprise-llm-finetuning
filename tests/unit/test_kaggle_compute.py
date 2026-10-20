"""
tests/unit/test_kaggle_compute.py
Unit tests verifying Kaggle API client, kernel builder, provider, and factory using unittest.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.domain.exceptions import ConfigurationError, RemoteComputeError
from src.domain.schemas.compute_schema import RemoteJobStatus
from src.infrastructure.compute.kaggle.client import KaggleApiClient
from src.infrastructure.compute.kaggle.kernel_builder import KaggleKernelBuilder
from src.infrastructure.compute.kaggle.provider import KaggleComputeProvider
from src.infrastructure.factories.compute_factory import ComputeProviderFactory


class TestKaggleApiClient(unittest.TestCase):
    def test_credential_discovery_from_env(self):
        with patch.dict(os.environ, {"KAGGLE_USERNAME": "test_user", "KAGGLE_KEY": "test_key"}):
            client = KaggleApiClient()
            self.assertEqual(client.username, "test_user")

    def test_push_kernel_missing_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            client = KaggleApiClient(username="user", key="key")
            with self.assertRaises(RemoteComputeError) as ctx:
                client.push_kernel(Path(tmp_dir))
            self.assertIn("kernel-metadata.json not found", str(ctx.exception))

    @patch("subprocess.run")
    def test_push_kernel_success(self, mock_run):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            meta_file = tmp_path / "kernel-metadata.json"
            meta_file.write_text(json.dumps({"id": "test_user/test-slug"}), encoding="utf-8")

            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Kernel https://www.kaggle.com/code/test_user/test-slug successfully pushed",
                stderr="",
            )

            client = KaggleApiClient(username="test_user", key="test_key")
            remote_id = client.push_kernel(tmp_path)
            self.assertEqual(remote_id, "test_user/test-slug")

    @patch("subprocess.run")
    def test_get_status_mapping(self, mock_run):
        client = KaggleApiClient(username="test_user", key="test_key")

        mock_run.return_value = MagicMock(returncode=0, stdout='status: "complete"', stderr="")
        self.assertEqual(client.get_status("test_user/test-slug"), "complete")

        mock_run.return_value = MagicMock(returncode=0, stdout='status: "running"', stderr="")
        self.assertEqual(client.get_status("test_user/test-slug"), "running")

        mock_run.return_value = MagicMock(returncode=0, stdout='status: "queued"', stderr="")
        self.assertEqual(client.get_status("test_user/test-slug"), "queued")

        mock_run.return_value = MagicMock(returncode=0, stdout='status: "error"', stderr="")
        self.assertEqual(client.get_status("test_user/test-slug"), "error")

        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Kernel not found")
        self.assertEqual(client.get_status("test_user/test-slug"), "unknown")


class TestKaggleKernelBuilder(unittest.TestCase):
    def test_sanitize_slug(self):
        slug = KaggleKernelBuilder.sanitize_slug(
            "gemma-3-1b_flare-finqa_qlora-r16_20261002_155950_very_long_experiment_name"
        )
        self.assertLessEqual(len(slug), 45)
        self.assertNotIn("_", slug)
        self.assertTrue(slug.islower() or "-" in slug)

    def test_build_package_success(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            builder = KaggleKernelBuilder()
            config_path = tmp_path / "test_config.yaml"
            config_path.write_text("experiment:\n  name: 'unit_test'", encoding="utf-8")

            output_dir = tmp_path / "output_package"
            builder.build_package(
                experiment_name="unit_test_run",
                config_path=config_path,
                output_dir=output_dir,
                username="test_kaggle_user",
            )

            self.assertTrue((output_dir / "worker.py").exists())
            self.assertTrue((output_dir / "finetuning_config.yaml").exists())
            meta_file = output_dir / "kernel-metadata.json"
            self.assertTrue(meta_file.exists())

            metadata = json.loads(meta_file.read_text(encoding="utf-8"))
            self.assertEqual(metadata["id"], "test_kaggle_user/unit-test-run")
            self.assertTrue(metadata["enable_gpu"])
            self.assertTrue(metadata["enable_internet"])

    def test_build_package_missing_username(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            builder = KaggleKernelBuilder()
            config_path = tmp_path / "test_config.yaml"
            config_path.write_text("test", encoding="utf-8")

            with self.assertRaises(RemoteComputeError) as ctx:
                builder.build_package(
                    experiment_name="run",
                    config_path=config_path,
                    output_dir=tmp_path / "out",
                    username="",
                )
            self.assertIn("username is required", str(ctx.exception))


class TestKaggleComputeProvider(unittest.TestCase):
    def test_provider_name_and_status_mapping(self):
        mock_client = MagicMock()
        mock_builder = MagicMock()
        provider = KaggleComputeProvider(api_client=mock_client, kernel_builder=mock_builder)

        self.assertEqual(provider.provider_name, "kaggle_dual_t4")

        mock_client.get_status.return_value = "complete"
        self.assertEqual(provider.get_status("user/slug"), RemoteJobStatus.COMPLETED)

        mock_client.get_status.return_value = "running"
        self.assertEqual(provider.get_status("user/slug"), RemoteJobStatus.RUNNING)

        mock_client.get_status.return_value = "queued"
        self.assertEqual(provider.get_status("user/slug"), RemoteJobStatus.QUEUED)

        mock_client.get_status.return_value = "error"
        self.assertEqual(provider.get_status("user/slug"), RemoteJobStatus.FAILED)

        mock_client.get_status.return_value = "unknown"
        self.assertEqual(provider.get_status("user/slug"), RemoteJobStatus.UNKNOWN)

    def test_submit_job_flow(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            mock_client = MagicMock()
            mock_client.username = "mock_user"
            mock_client.push_kernel.return_value = "mock_user/test-experiment"

            mock_builder = MagicMock()
            provider = KaggleComputeProvider(
                api_client=mock_client,
                kernel_builder=mock_builder,
                workspace_dir=tmp_path,
            )

            dummy_config = tmp_path / "config.yaml"
            dummy_config.write_text("dummy", encoding="utf-8")

            remote_id = provider.submit_job("test_experiment", dummy_config)
            self.assertEqual(remote_id, "mock_user/test-experiment")
            self.assertTrue(mock_builder.build_package.called)
            self.assertTrue(mock_client.push_kernel.called)


class TestComputeProviderFactory(unittest.TestCase):
    def test_get_registered_provider(self):
        provider = ComputeProviderFactory.get_provider("kaggle")
        self.assertIsInstance(provider, KaggleComputeProvider)
        self.assertEqual(provider.provider_name, "kaggle_dual_t4")

    def test_get_unregistered_provider_raises(self):
        with self.assertRaises(ConfigurationError) as ctx:
            ComputeProviderFactory.get_provider("invalid_provider")
        self.assertIn("Unknown compute provider", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
