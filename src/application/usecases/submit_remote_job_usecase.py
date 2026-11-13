"""
src/application/usecases/submit_remote_job_usecase.py
Use case for packaging and dispatching training jobs to remote cloud GPU providers.
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


class SubmitRemoteJobUseCase:
    """
    Submits an experiment to a remote compute provider and persists a tracking receipt.
    """

    def __init__(
        self,
        compute_provider: IComputeProvider,
        receipts_dir: Optional[Path] = None,
    ) -> None:
        if compute_provider is None:
            raise RemoteComputeError("IComputeProvider implementation cannot be None.")
        self._provider = compute_provider
        self._receipts_dir = receipts_dir or Path("outputs/.remote_jobs")
        self._receipts_dir.mkdir(parents=True, exist_ok=True)

    def execute(self, experiment_name: str, config_path: Path) -> RemoteJobResult:
        """
        Executes package compilation, pushes kernel to remote backend, and saves receipt.

        Args:
            experiment_name: Unique identifier for the training experiment.
            config_path: Path to the experiment YAML configuration.

        Returns:
            RemoteJobResult: Summary containing local and remote job identifiers.
        """
        if not config_path.exists():
            raise RemoteComputeError(f"Configuration file not found: {config_path}")

        logger.info(
            "Submitting experiment '%s' to remote compute backend '%s'...",
            experiment_name,
            self._provider.provider_name,
        )

        remote_job_id = self._provider.submit_job(
            experiment_name=experiment_name,
            config_path=config_path,
        )

        # Persist receipt so the user can easily monitor without remembering the remote ID
        receipt_data = {
            "job_id": experiment_name,
            "remote_job_id": remote_job_id,
            "provider": self._provider.provider_name,
            "status": RemoteJobStatus.QUEUED.value,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "config_path": str(config_path.resolve()),
        }

        specific_receipt = self._receipts_dir / f"{experiment_name}.json"
        latest_receipt = self._receipts_dir / "latest.json"

        with open(specific_receipt, "w", encoding="utf-8") as f:
            json.dump(receipt_data, f, indent=2)

        with open(latest_receipt, "w", encoding="utf-8") as f:
            json.dump(receipt_data, f, indent=2)

        logger.info(
            "Tracking receipt saved to %s (and linked to 'latest.json').",
            specific_receipt,
        )

        return RemoteJobResult(
            job_id=experiment_name,
            remote_job_id=remote_job_id,
            status=RemoteJobStatus.QUEUED,
        )
