# Phase 4: Contract
# Pydantic v2 Configuration Schemas

**Document Version:** 1.0.0  
**Status:** Approved  

---

## 1. Complete Configuration Domain Schema (`src/domain/schemas/config_schema.py`)

The platform uses **Pydantic v2** models to strictly validate declarative YAML configurations before any compute starts:

```python
"""
src/domain/schemas/config_schema.py
Strongly typed configuration models for enterprise LLM fine-tuning.
"""

from enum import Enum
from pathlib import Path
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


class DatasetSourceType(str, Enum):
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    S3 = "s3"
    KAGGLE = "kaggle"
    LAKEHOUSE = "lakehouse"


class ExperimentInfo(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str = Field(..., description="Unique identifier for the fine-tuning experiment run")
    project: str = Field(default="enterprise-llm-finetuning", description="Project namespace")
    seed: int = Field(default=42, description="Random seed for deterministic reproducibility")
    tags: List[str] = Field(default_factory=lambda: ["lora", "sft"])


class DatasetConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    source: DatasetSourceType = Field(default=DatasetSourceType.HUGGINGFACE)
    dataset_name: str = Field(..., description="Hugging Face hub path, S3 URI, or local file path")
    train_split: str = Field(default="train")
    eval_split: Optional[str] = Field(default="test")


class LLMModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    llm_model_id: str = Field(..., description="Foundation model ID (e.g. Qwen/Qwen2.5-0.5B-Instruct)")
    trust_remote_code: bool = Field(default=False)


class QuantizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    load_in_4bit: bool = Field(default=True)
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = Field(default="nf4")
    bnb_4bit_compute_dtype: Literal["bfloat16", "float16", "float32"] = Field(default="bfloat16")
    bnb_4bit_use_double_quant: bool = Field(default=True)


class PEFTConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    task_type: str = Field(default="CAUSAL_LM")
    r: int = Field(default=16, ge=1, le=256, description="LoRA attention dimension rank")
    lora_alpha: int = Field(default=32, description="LoRA scaling alpha factor")
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    bias: Literal["none", "all", "lora_only"] = Field(default="none")
    target_modules: List[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )


class TelemetryConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    enable_wandb: bool = Field(default=True)
    wandb_project: str = Field(default="llm-fine-tuning")
    enable_prometheus: bool = Field(default=True)
    prometheus_port: int = Field(default=8000)
    enable_tensorboard: bool = Field(default=True)


class MonitoringConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    logging_steps: int = Field(default=10, ge=1)
    eval_steps: int = Field(default=50, ge=1)
    save_steps: int = Field(default=100, ge=1)
    save_total_limit: int = Field(default=3, ge=1)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)


class RegistryConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    registry_type: Literal["hf_hub", "s3", "local", "mlflow"] = Field(default="hf_hub")
    model_name: str = Field(default="my-finetuned-model")
    auto_register: bool = Field(default=True)


class ArtifactConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    output_dir: Path = Field(default=Path("./outputs/run-v1"))
    save_merged_model: bool = Field(default=True)
    export_format: Literal["safetensors", "pytorch"] = Field(default="safetensors")


class ExperimentConfig(BaseModel):
    """Aggregate Root Configuration validating end-to-end experiment spec."""
    model_config = ConfigDict(frozen=True)
    experiment: ExperimentInfo
    dataset: DatasetConfig
    llm_model: LLMModelConfig
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    peft: PEFTConfig = Field(default_factory=PEFTConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
    artifact: ArtifactConfig = Field(default_factory=ArtifactConfig)
```
