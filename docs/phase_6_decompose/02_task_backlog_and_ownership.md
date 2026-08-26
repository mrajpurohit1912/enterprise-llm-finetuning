# Phase 6: Decompose
# Task Backlog, Developer Ownership & Acceptance Criteria

**Document Version:** 1.0.0  
**Status:** Approved  

---

## 1. Developer Ownership & Assignment Matrix

| Ticket ID | Task Title | Track | Owner | Est. Story Points |
| :--- | :--- | :--- | :--- | :---: |
| **CORE-101** | Implement Pydantic v2 Config Schemas | Track 1 | **Lead / Architect** | 3 |
| **CORE-102** | Implement Domain Interfaces & Protocols | Track 1 | **Lead / Architect** | 2 |
| **CORE-103** | Implement YamlConfigLoader & Unit Tests | Track 1 | **Developer A** | 3 |
| **DATA-201** | Implement HuggingFaceDatasetLoader Adapter | Track 2 | **Developer A** | 3 |
| **DATA-202** | Implement Dynamic ChatTemplateFormatter | Track 2 | **Developer A** | 5 |
| **DATA-203** | Implement DatasetProcessor Service | Track 2 | **Developer A** | 3 |
| **MODEL-301** | Implement 4-bit Quantization & ModelLoader | Track 3 | **Developer B** | 5 |
| **MODEL-302** | Implement PEFT LoRA Config & Parameter Counter | Track 3 | **Developer B** | 3 |
| **TRAIN-303** | Implement SFTTrainer Adapter & Pipeline Usecase | Track 3 | **Developer B** | 8 |
| **OBS-401** | Implement Prometheus VRAM/Loss Callback | Track 4 | **Developer C** | 5 |
| **OBS-402** | Implement Weights & Biases Telemetry Integration | Track 4 | **Developer C** | 3 |
| **REG-403** | Implement Safetensors Model Serializer & Registry | Track 4 | **Developer C** | 5 |
| **SERV-501** | Implement Async vLLM Inference Service & DTOs | Track 5 | **Developer D** | 8 |
| **CLI-502** | Implement Typer CLI Presentation Layer | Track 5 | **Developer D** | 3 |
| **OPS-503** | Configure GitHub Actions CI, Ruff & Mypy Gates | Track 5 | **Developer D** | 5 |

---

## 2. Detailed Task Specifications (Jira / Linear Ready)

### [DATA-202] Implement Dynamic ChatTemplateFormatter
* **Assignee:** Developer A
* **Description:** Build `ChatTemplateFormatter` implementing `PromptFormatterBase`. It must dynamically inspect the dataset columns (`messages`, `conversations`, or `question/answer`) and apply the model tokenizer's Jinja chat template.
* **Acceptance Criteria:**
  1. Implements `src/domain/interfaces/prompt_formatter.py`.
  2. Passes unit tests with `pytest tests/unit/test_formatter.py` ($\ge 90\%$ coverage).
  3. Supports both formatted dialogue pairs and raw text fields.

---

### [TRAIN-303] Implement SFTTrainer Adapter & Pipeline Usecase
* **Assignee:** Developer B
* **Description:** Complete `TrainPipelineUsecase` in `src/application/usecases/train_pipeline.py`. Connect dataset loading, preprocessing, model loading, LoRA configuration, and execution of `SFTTrainer`.
* **Acceptance Criteria:**
  1. Accepts `ExperimentConfig` and runs training for specified `max_steps` or `epochs`.
  2. Binds `PrometheusLoggingCallback` and `WandbCallback` dynamically based on config.
  3. Passes integration test with dummy mock model on CPU.

---

### [SERV-501] Implement Async vLLM Inference Service
* **Assignee:** Developer D
* **Description:** Build `vLLM` async serving microservice in `src/presentation/api/serve.py` exposing `/v1/chat/completions` with SSE token streaming.
* **Acceptance Criteria:**
  1. Conforms to OpenAI-compatible `ChatCompletionRequestDTO` and `ChatCompletionResponseDTO`.
  2. Implements PagedAttention with KV cache allocation.
  3. Provides `/health` endpoint returning GPU memory status.
