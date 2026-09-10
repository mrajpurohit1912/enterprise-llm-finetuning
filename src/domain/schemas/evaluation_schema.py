"""
src/domain/schemas/evaluation_schema.py
Domain models and DTOs for model evaluation, metrics scoring, and quality gates.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class MetricScores(BaseModel):
    """Container for computed evaluation metric scores."""

    model_config = ConfigDict(frozen=True)

    exact_match: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Exact string match accuracy (0.0 to 1.0)",
    )
    f1_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Token-level harmonic mean of precision and recall (0.0 to 1.0)",
    )
    rouge_1: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="ROUGE-1 unigram overlap score",
    )
    rouge_2: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="ROUGE-2 bigram overlap score",
    )
    rouge_l: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="ROUGE-L longest common subsequence score",
    )
    custom_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Additional framework or domain-specific metric scores",
    )


class EvaluationSample(BaseModel):
    """Detailed record of a single evaluated question/answer pair."""

    prompt: str = Field(..., description="Input prompt presented to the model")
    ground_truth: str = Field(..., description="Expected reference answer")
    prediction: str = Field(..., description="Model generated response")
    exact_match: bool = Field(..., description="Whether prediction matches reference exactly")
    f1_score: float = Field(..., description="Token-level F1 score for this sample")


class EvaluationConfig(BaseModel):
    """Configuration governing model evaluation parameters and quality gate thresholds."""

    evaluator_type: str = Field(
        default="deterministic",
        description="Strategy for evaluation ('deterministic', 'llm_judge', etc.)",
    )
    eval_split: str = Field(
        default="test",
        description="Dataset split partition to evaluate against",
    )
    max_eval_samples: Optional[int] = Field(
        default=50,
        gt=0,
        description="Maximum number of holdout samples to evaluate (None for full split)",
    )
    batch_size: int = Field(
        default=4,
        gt=0,
        description="Inference batch size during evaluation",
    )
    max_new_tokens: int = Field(
        default=128,
        gt=0,
        description="Maximum tokens generated per prompt",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Generation temperature (low values favor deterministic factual answers)",
    )
    min_f1_threshold: Optional[float] = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Enterprise quality gate: minimum acceptable F1 score",
    )


class EvaluationResult(BaseModel):
    """Strongly-typed final report output from an evaluation run."""

    experiment_name: str = Field(..., description="Name of the active experiment")
    model_id: str = Field(..., description="Foundation model identifier or adapter path")
    eval_split: str = Field(..., description="Split evaluated (e.g. 'test')")
    total_samples: int = Field(..., ge=0, description="Total questions evaluated")
    metrics: MetricScores = Field(..., description="Aggregated metric scores")
    latency_per_sample_ms: float = Field(..., ge=0.0, description="Average generation latency per sample in ms")
    throughput_tokens_per_sec: float = Field(..., ge=0.0, description="Inference token generation speed")
    passed_quality_gate: bool = Field(default=True, description="Whether metrics met quality gate thresholds")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 evaluation timestamp",
    )
    sample_details: List[EvaluationSample] = Field(
        default_factory=list,
        description="Sample-level prompt, prediction, and ground truth details",
    )
