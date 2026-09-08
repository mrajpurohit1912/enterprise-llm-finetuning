"""
src/domain/schemas/pipeline_schema.py
Domain schema entities representing execution results and status contracts for pipelines.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TrainPipelineResult:
    """
    Strongly-typed enterprise execution DTO returned by TrainPipelineUsecase.
    Contains operational status, performance metrics, output paths, and execution telemetry.
    """
    experiment_name: str
    status: str
    output_dir: str
    duration_seconds: float = 0.0
    train_loss: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None
    dataset_size: Optional[Dict[str, int]] = None
    model: Optional[Any] = field(default=None, repr=False)
    tokenizer: Optional[Any] = field(default=None, repr=False)
