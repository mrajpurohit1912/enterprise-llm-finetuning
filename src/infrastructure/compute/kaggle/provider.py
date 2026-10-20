"""
src/infrastructure/compute/kaggle/provider.py
Kaggle Dual-T4 GPU compute provider implementing IComputeProvider.
"""

import logging
from pathlib import Path
from typing import Optional

from src.domain.interfaces.compute_provider import IComputeProvider
from src.domain.schemas.compute_schema import RemoteJobStatus
from src.infrastructure.compute.kaggle.client import KaggleApiClient
from src.infrastructure.compute.kaggle.kernel_builder import KaggleKernelBuilder

logger = logging.getLogger(__name__)


class KaggleComputeProvider(IComputeProvider):
    """Compute provider adapter dispatching jobs headlessly to Kaggle Cloud GPUs."""

    def __init__(
        self,
        api_client: Optional[KaggleApiClient] = None,
        kernel_builder: Optional[KaggleKernelBuilder] = None,
        workspace_dir: Optional[Path] = None,
    ) -> None:
        self._client = api_client or KaggleApiClient()
        self._builder = kernel_builder or KaggleKernelBuilder()
        self._workspace = workspace_dir or Path("outputs/.kaggle_workspace")
        self._workspace.mkdir(parents=True, exist_ok=True)

    @property
    def provider_name(self) -> str:
        """Identifier for the compute backend."""
        return "kaggle_dual_t4"

    def submit_job(self, experiment_name: str, config_path: Path) -> str:
        """
        Builds deployment bundle, pushes to Kaggle Cloud, and returns remote job ID.
        """
        username = self._client.username
        runner_dir = self._workspace / experiment_name

        logger.info(
            "Packaging and submitting remote job '%s' for user '%s'...",
            experiment_name,
            username,
        )

        self._builder.build_package(
            experiment_name=experiment_name,
            config_path=config_path,
            output_dir=runner_dir,
            username=username or "unknown",
        )

        remote_id = self._client.push_kernel(runner_dir)
        logger.info("Remote Kaggle job submitted successfully: %s", remote_id)
        return remote_id

    def get_status(self, remote_job_id: str) -> RemoteJobStatus:
        """
        Queries status of remote Kaggle execution and maps it to domain RemoteJobStatus.
        """
        status_str = self._client.get_status(remote_job_id)
        logger.debug("Remote Kaggle status for '%s': %s", remote_job_id, status_str)

        if status_str == "complete":
            return RemoteJobStatus.COMPLETED
        if status_str == "running":
            return RemoteJobStatus.RUNNING
        if status_str == "queued":
            return RemoteJobStatus.QUEUED
        if status_str == "error":
            return RemoteJobStatus.FAILED

        return RemoteJobStatus.UNKNOWN

    def download_artifact(self, remote_job_id: str, destination_path: Path) -> None:
        """
        Downloads completed artifacts from Kaggle Cloud into destination_path.
        """
        logger.info("Downloading remote artifacts for '%s' to %s...", remote_job_id, destination_path)
        self._client.download_output(remote_job_id, destination_path)
