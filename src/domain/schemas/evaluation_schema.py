"""
src/domain/schemas/evaluation_schema.py
Domain models and DTOs for model evaluation, metrics scoring, and quality gates.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional,Literal
from pydantic import BaseModel, ConfigDict, Field
from enum import Enum

                                                                                                                                                                                          
class TaskType(str, Enum):
    TEXT_GENERATION = "text_generation"                                      
    CLASSIFICATION = "classification"                                      
    REGRESSION = "regression"                                      
    JSON_SCHEMA = "json_schema"                                      
                                           

class MetricScores(BaseModel):
    """Container for computed evaluation metric scores."""

    model_config = ConfigDict(frozen=True)

    #Text Generation Metrics
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


    # Classification Metrics
    accuracy: Optional[float] = Field(default=None,ge=0.0,le=1.0)
    precision: Optional[float] = Field(default=None,ge=0.0,le=1.0)
    recall: Optional[float] = Field(default=None,ge=0.0,le=1.0)

    # Structured Output / JSON Schema Metrics
    schema_validity_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)  

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
    task_type: TaskType = Field(default=TaskType.TEXT_GENERATION)
    evaluator_type: str = Field(default="deterministic")
    eval_split: str = Field(default="test")
    max_eval_samples: Optional[int] = Field(default=50, gt=0)
    batch_size: int = Field(default=4, gt=0)
    max_new_tokens: int = Field(default=128, gt=0)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
  
    # Dynamic Column Mapping (Allows null for auto-detection)
    prompt_column: Optional[str] = Field(default=None)
    target_column: Optional[str] = Field(default=None)
  
    # Comparative Baseline Benchmarking
    compare_with_base_model: bool = Field(default=False)
  
    # Quality Gate Thresholds
    min_f1_threshold: Optional[float] = Field(default=0.50)
    min_exact_match_threshold: Optional[float] = Field(default=None)
    fail_on_quality_gate_breach: bool = Field(default=False)


class EvaluationResult(BaseModel):
    """Strongly-typed final report output from an evaluation run."""

    experiment_name: str = Field(..., description="Name of the active experiment")
    model_id: str = Field(..., description="Foundation model identifier or adapter path")
    task_type:TaskType = Field(default=TaskType.TEXT_GENERATION,description="The domain task type being evaluated")
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
    base_model_metrics: Optional[MetricScores] = Field(
        default=None,
        description="Optional baseline model metrics for comparative evaluation",
    )
    delta_metrics: Optional[Dict[str, float]] = Field(
            default=None,
            description="Capability delta (Fine-Tuned - Base)",
        )
    # 2. Serving Telemetry Fields
    ttft_ms_p50: Optional[float] = Field(default=None, description="Time to first token median (ms)")
    ttft_ms_p95: Optional[float] = Field(default=None, description="Time to first token 95th percentile (ms)")
    tpot_ms_p50: Optional[float] = Field(default=None, description="Time per output token median (ms)")
    tpot_ms_p95: Optional[float] = Field(default=None, description="Time per output token 95th percentile (ms)")
    peak_memory_mb: Optional[float] = Field(default=None, description="Peak GPU VRAM allocated in MB")