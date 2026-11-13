"""
src/application/usecases/sync_remote_job_usecase.py
Use case for polling remote compute execution status and downloading artifacts upon completion.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.domain.exceptions import RemoteComputeError
from src.domain.interfaces.compute_provider import IComputeProvider
from src.domain.schemas.compute_schema import RemoteJobResult, RemoteJobStatus

logger = logging.getLogger(__name__)


class SyncRemoteJobUseCase:
    """
    Synchronizes status of a remote GPU compute execution and downloads output artifacts.
    """

    def __init__(
        self,
        compute_provider: IComputeProvider,
        receipts_dir: Optional[Path] = None,
        outputs_dir: Optional[Path] = None,
    ) -> None:
        if compute_provider is None:
            raise RemoteComputeError("IComputeProvider implementation cannot be None.")
        self._provider = compute_provider
        self._receipts_dir = receipts_dir or Path("outputs/.remote_jobs")
        self._outputs_dir = outputs_dir or Path("outputs")

    def execute(
        self,
        remote_job_id: Optional[str] = None,
        experiment_name: Optional[str] = None,
        auto_download: bool = True,
    ) -> RemoteJobResult:
        """
        Queries remote job status, updates local receipt, and downloads artifacts if completed.

        Args:
            remote_job_id: Optional Kaggle job slug (e.g. 'username/kernel-slug').
            experiment_name: Optional local experiment identifier.
            auto_download: Whether to pull output artifacts if the job is completed.

        Returns:
            RemoteJobResult: Up-to-date execution status and output artifact location.
        """
        receipt_file = None
        job_id = experiment_name or "unknown"

        # 1. Resolve remote job ID from receipt if not explicitly supplied
        if not remote_job_id:
            if experiment_name and (self._receipts_dir / f"{experiment_name}.json").exists():
                receipt_file = self._receipts_dir / f"{experiment_name}.json"
            elif (self._receipts_dir / "latest.json").exists():
                receipt_file = self._receipts_dir / "latest.json"

            if receipt_file and receipt_file.exists():
                try:
                    with open(receipt_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    remote_job_id = data.get("remote_job_id")
                    job_id = data.get("job_id", job_id)
                except Exception as exc:
                    logger.warning("Could not read receipt from %s: %s", receipt_file, exc)

        if not remote_job_id:
            raise RemoteComputeError(
                "No remote job ID was specified and no recent job receipt exists in "
                f"'{self._receipts_dir}'. Please provide --remote-job-id."
            )

        # 2. Query status from remote compute provider
        logger.info("Querying remote status for '%s'...", remote_job_id)
        status = self._provider.get_status(remote_job_id)
        logger.info("Current remote status for '%s': %s", remote_job_id, status.value)

        # 3. Pull outputs if job has completed successfully
        artifact_path: Optional[Path] = None
        if status == RemoteJobStatus.COMPLETED and auto_download:
            dest_folder_name = job_id if job_id != "unknown" else remote_job_id.split("/")[-1]
            dest_dir = self._outputs_dir / dest_folder_name
            logger.info("Job completed! Downloading artifacts to %s...", dest_dir)
            self._provider.download_artifact(remote_job_id, dest_dir)
            artifact_path = dest_dir

        # 4. Update receipt file with latest status
        if receipt_file and receipt_file.exists():
            try:
                with open(receipt_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["status"] = status.value
                data["last_synced_at"] = datetime.now(timezone.utc).isoformat()
                if artifact_path:
                    data["artifact_path"] = str(artifact_path.resolve())

                with open(receipt_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception as exc:
                logger.warning("Failed to update receipt %s: %s", receipt_file, exc)

        return RemoteJobResult(
            job_id=job_id,
            remote_job_id=remote_job_id,
            status=status,
            artifact_path=artifact_path,
        )
