# Phase 2: Model
# End-to-End Sequence Flows

**Document Version:** 1.0.0  
**Status:** Approved  

---

## 1. Complete End-to-End Fine-Tuning Sequence

The following sequence diagram models the exact runtime interaction between CLI Presentation, Application Use Cases, Domain Interfaces, and Infrastructure Adapters:

```mermaid
sequenceDiagram
    autonumber
    actor User as ML Engineer / CLI
    participant CLI as Presentation CLI
    participant ConfigLoader as YamlConfigLoader (Infra)
    participant Pipeline as TrainPipelineUsecase (App)
    participant DataFactory as DatasetLoaderFactory (Infra)
    participant DataLoader as HuggingFace/S3 Loader (Infra)
    participant Processor as DatasetProcessor (App)
    participant ModelLoader as HuggingFaceModelLoader (Infra)
    participant Telemetry as Prometheus / W&B (Infra)
    participant Trainer as SFTTrainerAdapter (Infra)
    participant Registry as ModelRegistryAdapter (Infra)

    User->>CLI: Run `python -m src.presentation.cli --config finetuning_config.yaml`
    CLI->>ConfigLoader: load_config("finetuning_config.yaml")
    ConfigLoader-->>CLI: Validated ExperimentConfig (Domain Entity)

    CLI->>Pipeline: run(experiment_config)

    %% Step 1: Ingest Data
    rect rgb(240, 248, 255)
        Note over Pipeline, DataLoader: Step 1: Dataset Ingestion
        Pipeline->>DataFactory: get_loader(config.dataset.source)
        DataFactory-->>Pipeline: DataLoader Instance
        Pipeline->>DataLoader: load_data(config.dataset.dataset_name)
        DataLoader-->>Pipeline: Raw DatasetDict
    end

    %% Step 2: Preprocess & Format
    rect rgb(245, 255, 250)
        Note over Pipeline, Processor: Step 2: Preprocessing & Chat Templating
        Pipeline->>Processor: process(raw_dataset, tokenizer, format_strategy)
        Processor-->>Pipeline: Formatted DatasetDict
    end

    %% Step 3: Load Model & LoRA
    rect rgb(255, 250, 240)
        Note over Pipeline, ModelLoader: Step 3: Quantized Model & LoRA Preparation
        Pipeline->>ModelLoader: load_quantized_model(config.llm_model, config.quantization)
        ModelLoader-->>Pipeline: Base Quantized Model
        Pipeline->>ModelLoader: apply_peft_adapters(base_model, config.peft)
        ModelLoader-->>Pipeline: Wrapped PeftModel
    end

    %% Step 4: Training & Monitoring
    rect rgb(255, 240, 245)
        Note over Pipeline, Telemetry: Step 4: Training & Live Telemetry
        Pipeline->>Telemetry: start_telemetry(config.monitoring)
        Telemetry-->>Pipeline: Telemetry Callbacks Attached (:8000 & W&B)
        Pipeline->>Trainer: train(model, dataset, training_args, callbacks)
        loop Every Logging Step
            Trainer->>Telemetry: emit_step_metrics(loss, lr, vram_mb)
        end
        Trainer-->>Pipeline: Training Complete & Checkpoint Saved
    end

    %% Step 5: Merge & Register
    rect rgb(240, 255, 240)
        Note over Pipeline, Registry: Step 5: Serialization & Registration
        Pipeline->>Registry: export_safetensors(model, config.artifact)
        Registry-->>Pipeline: Safetensors Artifact Verified
        Pipeline->>Registry: register_model(artifact_path, config.registry)
        Registry-->>Pipeline: Model Registered: version_v1
    end

    Pipeline-->>CLI: Pipeline Result (Success + Metrics)
    CLI-->>User: Display Summary Report
```

---

## 2. Production Inference Serving Sequence (vLLM)

```mermaid
sequenceDiagram
    autonumber
    actor Client as Enterprise API Client
    participant Gateway as FastAPI Router / Gateway
    participant Engine as AsyncvLLMEngine
    participant VRAM as GPU KV Cache / PagedAttention

    Client->>Gateway: POST /v1/chat/completions { prompt, stream: true }
    Gateway->>Gateway: Validate API Token & Request DTO
    Gateway->>Engine: generate(prompt, SamplingParams)
    Engine->>VRAM: Allocate continuous KV cache blocks
    
    loop Stream Generated Tokens
        VRAM-->>Engine: Next Token Output
        Engine-->>Gateway: Yield Token Chunk
        Gateway-->>Client: HTTP SSE Stream ("data: {'token': ...}")
    end

    Engine->>VRAM: Free KV Cache blocks
    Gateway-->>Client: Stream End ([DONE])
```
