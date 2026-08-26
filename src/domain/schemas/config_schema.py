from pydantic import BaseModel,Field
from typing import Literal
from enum import Enum
from pathlib import Path

class DatasetSourceType(str, Enum):
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    S3 = "s3"
    KAGGLE = "kaggle"
    LAKEHOUSE = "lakehouse"

class DatasetConfig(BaseModel):
    source:DatasetSourceType
    dataset_name:str
    train_split:Literal["train","test"] = "train"
    eval_split:Literal["train","test"] = "test"


class LLMModelConfig(BaseModel):
    llm_model_id:str = "Qwen/Qwen2.5-0.5B-Instruct"

class QuantizationConfig(BaseModel):
    load_in_4bit:bool = True
    bnb_4bit_quant_type:str = "nf4"
    bnb_4bit_compute_dtype:str = "bfloat16"
    bnb_4bit_use_double_quant:bool = True

class FsdpConfig(BaseModel):
    sharding_strategy:Literal["full_shard", "shard_grad_op", "no_shard"] = "full_shard"
    offload_params:bool = False
    

class HardwareConfig(BaseModel):
    distributed_strategy:Literal["single_gpu", "fsdp", "deepspeed_stage_2", "deepspeed_stage_3"] = "single_gpu"
    fsdp_config:FsdpConfig =  Field(default_factory=FsdpConfig)
    deepspeed_config_path:Path | None = None


class TelemetryConfig(BaseModel):
    enable_wandb:bool = True
    wandb_project:str = "my-org"          
    tensorboard:bool = True
    mlflow:bool = False

class MonitoringConfig(BaseModel):
    logging_steps:int = 10
    eval_steps:int = 50
    save_steps:int = 50
    save_total_limit:int = 3
    telemetry:TelemetryConfig = Field(default_factory=TelemetryConfig)

class RegistryConfig(BaseModel):
    registry_type:Literal["hf_hub","s3","local","mlflow"] = "hf_hub"
    model_name:str = "my-llm-finetuned-model"
    auto_register:bool = True

class ArtifactConfig(BaseModel):
    output_dir:Path = Path.cwd() / "output" / "lora_finetuning_output"
    save_merged_model:bool = True
    export_format:Literal["pytorch","safetensors"] = "safetensors"
    

class ExperimentConfig(BaseModel):
    experiment_name:str = "qwen2.5-0.5b-officeqa-v1"
    seed:int = 42
    tags: list[str] = ["qwen2.5", "officeqa", "lora", "sft"]  
    dataset:DatasetConfig = Field(default_factory=DatasetConfig)
    llm_model:LLMModelConfig = Field(default_factory=LLMModelConfig)
    quantization:QuantizationConfig = Field(default_factory=QuantizationConfig)
    hardware:HardwareConfig = Field(default_factory=HardwareConfig)
    monitoring:MonitoringConfig = Field(default_factory=MonitoringConfig)
    registry:RegistryConfig = Field(default_factory=RegistryConfig)
    artifact:ArtifactConfig = Field(default_factory=ArtifactConfig)
