from abc import ABC, abstractmethod
from pathlib import Path

from src.domain.schemas.compute_schema import RemoteJobStatus

class IComputerProvder(ABC):

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Identifier for the compute backend (e.g. 'kaggle_dual_t4').""" 
        pass

    @abstractmethod
    def submit_job(self, experiment_name: str, config_path: Path) -> str:
        """Submit a job to the compute backend and return a unique job identifier."""
        pass

    @abstractmethod
    def get_status(self, remote_job_id: str) -> RemoteJobStatus:
        """Retrieve the current status of a job given its unique identifier."""
        pass

    @abstractmethod
    def download_artifact(self, remote_job_id: str, destination_path: Path) -> None:
        """Download the artifact associated with a completed job to the specified destination path."""
        pass


# Backwards compatibility alias
IComputeProvider = IComputerProvder

__all__ = ["IComputerProvder", "IComputeProvider"]
