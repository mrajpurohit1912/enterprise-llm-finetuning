"""
tests/unit/test_remote_usecases.py
Unit tests for SubmitRemoteJobUseCase and SyncRemoteJobUseCase.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.application.usecases.submit_remote_job_usecase import SubmitRemoteJobUseCase
from src.application.usecases.sync_remote_job_usecase import SyncRemoteJobUseCase
from src.domain.exceptions import RemoteComputeError
from src.domain.schemas.compute_schema import RemoteJobStatus


class TestSubmitRemoteJobUseCase(unittest.TestCase):
    def test_init_with_none_provider_raises(self):
        with self.assertRaises(RemoteComputeError):
            SubmitRemoteJobUseCase(compute_provider=None)

    def test_submit_missing_config_raises(self):
        mock_provider = MagicMock()
        usecase = SubmitRemoteJobUseCase(compute_provider=mock_provider)
        with self.assertRaises(RemoteComputeError) as ctx:
            usecase.execute("test_exp", Path("non_existent_config.yaml"))
        self.assertIn("Configuration file not found", str(ctx.exception))

    def test_submit_success_saves_receipts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            receipts_dir = Path(tmp_dir) / "receipts"
            config_file = Path(tmp_dir) / "finetuning_config.yaml"
            config_file.write_text("dummy: config", encoding="utf-8")

            mock_provider = MagicMock()
            mock_provider.provider_name = "kaggle_dual_t4"
            mock_provider.submit_job.return_value = "user/test-experiment-slug"

            usecase = SubmitRemoteJobUseCase(
                compute_provider=mock_provider,
                receipts_dir=receipts_dir,
            )

            result = usecase.execute("test_exp", config_file)

            self.assertEqual(result.job_id, "test_exp")
            self.assertEqual(result.remote_job_id, "user/test-experiment-slug")
            self.assertEqual(result.status, RemoteJobStatus.QUEUED)

            # Check that receipts are written
            exp_receipt = receipts_dir / "test_exp.json"
            latest_receipt = receipts_dir / "latest.json"

            self.assertTrue(exp_receipt.exists())
            self.assertTrue(latest_receipt.exists())

            receipt_data = json.loads(exp_receipt.read_text(encoding="utf-8"))
            self.assertEqual(receipt_data["remote_job_id"], "user/test-experiment-slug")
            self.assertEqual(receipt_data["provider"], "kaggle_dual_t4")


class TestSyncRemoteJobUseCase(unittest.TestCase):
    def test_init_with_none_provider_raises(self):
        with self.assertRaises(RemoteComputeError):
            SyncRemoteJobUseCase(compute_provider=None)

    def test_sync_without_id_or_receipt_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_provider = MagicMock()
            usecase = SyncRemoteJobUseCase(
                compute_provider=mock_provider,
                receipts_dir=Path(tmp_dir) / "empty_receipts",
            )
            with self.assertRaises(RemoteComputeError) as ctx:
                usecase.execute()
            self.assertIn("No remote job ID was specified", str(ctx.exception))

    def test_sync_with_explicit_id_running(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_provider = MagicMock()
            mock_provider.get_status.return_value = RemoteJobStatus.RUNNING

            usecase = SyncRemoteJobUseCase(
                compute_provider=mock_provider,
                receipts_dir=Path(tmp_dir),
            )
            res = usecase.execute(remote_job_id="user/my-slug", auto_download=True)

            self.assertEqual(res.status, RemoteJobStatus.RUNNING)
            self.assertIsNone(res.artifact_path)
            self.assertFalse(mock_provider.download_artifact.called)

    def test_sync_auto_download_when_completed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            receipts_dir = Path(tmp_dir) / "receipts"
            receipts_dir.mkdir()
            outputs_dir = Path(tmp_dir) / "outputs"

            # Prepare a receipt
            receipt_file = receipts_dir / "latest.json"
            receipt_file.write_text(
                json.dumps({
                    "job_id": "test_exp_completed",
                    "remote_job_id": "user/test-completed",
                    "status": "running"
                }),
                encoding="utf-8"
            )

            mock_provider = MagicMock()
            mock_provider.get_status.return_value = RemoteJobStatus.COMPLETED

            usecase = SyncRemoteJobUseCase(
                compute_provider=mock_provider,
                receipts_dir=receipts_dir,
                outputs_dir=outputs_dir,
            )

            res = usecase.execute(auto_download=True)

            self.assertEqual(res.status, RemoteJobStatus.COMPLETED)
            self.assertEqual(res.remote_job_id, "user/test-completed")
            self.assertEqual(res.artifact_path, outputs_dir / "test_exp_completed")
            self.assertTrue(mock_provider.download_artifact.called)

            # Verify receipt updated to completed
            updated_receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
            self.assertEqual(updated_receipt["status"], "completed")
            self.assertIn("last_synced_at", updated_receipt)


if __name__ == "__main__":
    unittest.main()
