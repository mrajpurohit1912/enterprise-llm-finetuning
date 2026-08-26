# Phase 3: Architect
# C4 System Architecture Model

**Document Version:** 1.0.0  
**Status:** Approved  

---

## 1. C4 Level 1: System Context Diagram

The System Context diagram illustrates the platform's relationship with external users, storage services, foundation model hubs, and observability backends:

```mermaid
flowchart TD
    User["ML Engineer / Data Scientist"] -->|Executes CLI / Triggers Workflows| Platform["Enterprise LLM Fine-Tuning Platform"]
    ClientApp["Enterprise Downstream Apps"] -->|Sends Real-time Chat / Extraction Requests| Platform

    Platform -->|Downloads Base Models & Tokenizers| HF["Hugging Face Hub"]
    Platform -->|Ingests Raw Training Data| Storage["Cloud Data Lake (AWS S3 / GCS / Local)"]
    Platform -->|Streams Live Step Metrics| Prom["Prometheus & Grafana Cloud"]
    Platform -->|Logs Experiments & Run Metadata| WandB["Weights & Biases / MLflow"]
    Platform -->|Pushes Versioned Safetensors Artifacts| Registry["Enterprise Model Registry"]
```

---

## 2. C4 Level 2: Container Diagram

The Container diagram decomposes the platform into executable boundaries, services, and storage systems:

```mermaid
flowchart TB
    subgraph PresentationContainer["1. Presentation & Ingress"]
        CLI["CLI Command Runner (Click / Typer)"]
        API["FastAPI Inference Microservice (:8000)"]
    end

    subgraph CorePlatformContainer["2. Core Fine-Tuning Engine (Python Package)"]
        ConfigEngine["Config Parsing & Validation Engine (Pydantic v2)"]
        PipelineOrchestrator["Pipeline Orchestrator (Use Cases)"]
        DataSubsystem["Data Ingestion & Tokenization Subsystem"]
        ModelSubsystem["Model Quantization & LoRA Engine"]
        TrainerSubsystem["TRL / PyTorch SFT Training Subsystem"]
    end

    subgraph ServingContainer["3. High-Performance Inference Engine"]
        vLLM["vLLM Async Serving Engine (PagedAttention)"]
    end

    subgraph ObservabilityContainer["4. Observability Stack"]
        DCGM["NVIDIA DCGM Exporter Container (:9400)"]
        PrometheusServer["Prometheus Server (:9090)"]
    end

    CLI --> ConfigEngine
    ConfigEngine --> PipelineOrchestrator
    PipelineOrchestrator --> DataSubsystem
    PipelineOrchestrator --> ModelSubsystem
    PipelineOrchestrator --> TrainerSubsystem

    API --> vLLM
    TrainerSubsystem --> PrometheusServer
    DCGM --> PrometheusServer
```

---

## 3. C4 Level 3: Component Diagram (Inside Core Engine)

```mermaid
flowchart LR
    subgraph PresentationLayer["Presentation Layer"]
        CLIController["CLI Entrypoint"]
    end

    subgraph ApplicationLayer["Application Layer"]
        TrainUC["TrainPipelineUsecase"]
        LoadDataUC["LoadDatasetUseCase"]
        PreprocessUC["PreprocessDatasetUseCase"]
        DatasetProcessor["DatasetProcessor (Service)"]
    end

    subgraph DomainLayer["Domain Layer (Contracts)"]
        DataLoaderBase["<<interface>> DataLoaderBase"]
        TokenizerBase["<<interface>> TokenizerBase"]
        PromptFormatterBase["<<interface>> PromptFormatterBase"]
        ConfigSchema["ExperimentConfig Schema"]
    end

    subgraph InfrastructureLayer["Infrastructure Layer"]
        HFDataLoader["HuggingFaceDatasetLoader"]
        S3DataLoader["S3DatasetLoader"]
        HFTokenizer["HuggingFaceTokenizer"]
        ChatFormatter["ChatTemplateFormatter"]
        YamlLoader["YamlConfigLoader"]
    end

    CLIController --> TrainUC
    TrainUC --> LoadDataUC
    TrainUC --> PreprocessUC
    PreprocessUC --> DatasetProcessor

    LoadDataUC --> DataLoaderBase
    PreprocessUC --> PromptFormatterBase
    DatasetProcessor --> PromptFormatterBase

    HFDataLoader -.->|implements| DataLoaderBase
    S3DataLoader -.->|implements| DataLoaderBase
    ChatFormatter -.->|implements| PromptFormatterBase
    HFTokenizer -.->|implements| TokenizerBase
    YamlLoader -.->|produces| ConfigSchema
```
