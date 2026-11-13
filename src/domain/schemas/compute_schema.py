from pydantic import BaseModel
from enum import Enum
from typing import Optional
from pathlib import Path

class RemoteJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    UNKNOWN = "unknown"                                                                                                                                                        
                                  
class RemoteJobResult(BaseModel):
    job_id: str
    remote_job_id: str
    status: RemoteJobStatus
    artifact_path: Optional[Path] = None
    error_message: Optional[str] = None

