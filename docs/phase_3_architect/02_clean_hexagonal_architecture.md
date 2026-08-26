# Phase 3: Architect
# Clean / Hexagonal Architecture & Boundary Rules

**Document Version:** 1.0.0  
**Status:** Approved  

---

## 1. The Dependency Rule (The Golden Invariant)

The platform strictly enforces the **Clean Architecture Dependency Rule**:

> **Dependencies must point strictly inward.**  
> Outer layers may depend on inner layers, but inner layers must NEVER know about, import, or depend on outer layers.

```
       ┌─────────────────────────────────────────────────────────┐
       │ 1. PRESENTATION (CLI, FastAPI, REST Endpoints, DAGs)    │
       │   ┌─────────────────────────────────────────────────┐   │
       │   │ 2. INFRASTRUCTURE (HuggingFace, S3, vLLM, YAML) │   │
       │   │   ┌─────────────────────────────────────────┐   │   │
       │   │   │ 3. APPLICATION (Use Cases & Services)   │   │   │
       │   │   │   ┌─────────────────────────────────┐   │   │   │
       │   │   │   │ 4. DOMAIN (Schemas, Interfaces) │   │   │   │
       │   │   │   │    (Zero external dependencies) │   │   │   │
       │   │   │   └─────────────────────────────────┘   │   │   │
       │   │   └─────────────────────────────────────────┘   │   │
       │   └─────────────────────────────────────────────────┘   │
       └─────────────────────────────────────────────────────────┘
```

---

## 2. Layer Responsibilities & Package Boundaries

### Layer 1: `src/domain/` (The Core - Invariant)
* **What it contains:** Pure business entities, Pydantic configuration schemas, and abstract interfaces (Ports).
* **Rule:** May **NEVER** import from `application`, `infrastructure`, `presentation`, or third-party heavy ML libraries (`transformers`, `torch`, `boto3`).
* **Imports allowed:** Standard library (`abc`, `pathlib`, `typing`) and `pydantic`.

### Layer 2: `src/application/` (Business Orchestration)
* **What it contains:** Single-action Use Cases (`LoadDatasetUseCase`, `TrainModelUseCase`) and domain services (`DatasetProcessor`).
* **Rule:** Only imports from `domain/`. Coordinates interfaces without knowing which concrete infrastructure is running.

### Layer 3: `src/infrastructure/` (Technical Adapters & External I/O)
* **What it contains:** Concrete implementations of domain interfaces (`HuggingFaceDatasetLoader`, `YamlConfigLoader`, `PrometheusCallback`, `vLLMInferenceEngine`).
* **Rule:** Imports from `domain/` and third-party libraries (`transformers`, `peft`, `trl`, `boto3`, `yaml`).

### Layer 4: `src/presentation/` (Entrypoints & Triggers)
* **What it contains:** CLI handlers (Click/Argparse), FastAPI routes, Celery tasks.
* **Rule:** Acts as the **Composition Root**: instantiates infrastructure adapters, injects them into use cases, and triggers `.run()`.

---

## 3. Hexagonal Ports & Adapters Mapping

| Concept | Port (Domain Interface) | Adapter (Infrastructure Implementation) |
| :--- | :--- | :--- |
| **Dataset Ingestion** | `DataLoaderBase` | `HuggingFaceDatasetLoader`, `S3DatasetLoader`, `LocalCsvLoader` |
| **Tokenization** | `TokenizerBase` | `HuggingFaceTokenizer` |
| **Prompt Formatting** | `PromptFormatterBase` | `ChatTemplateFormatter`, `AlpacaTemplateFormatter` |
| **Config Parsing** | `ConfigLoaderBase` | `YamlConfigLoader`, `JsonConfigLoader` |
| **Telemetry / Monitoring** | `TelemetryCallbackBase` | `PrometheusLoggingCallback`, `WandbCallback` |
| **Inference Serving** | `InferenceEngineBase` | `vLLMAsyncEngine`, `HuggingFacePipelineEngine` |
| **Model Registry** | `ModelRegistryBase` | `HuggingFaceHubRegistry`, `MLflowRegistry`, `S3Registry` |
