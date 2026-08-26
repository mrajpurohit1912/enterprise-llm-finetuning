# Phase 5: Decisions
# Architecture Decision Records (ADRs) Index

This directory contains the immutable, version-controlled **Architecture Decision Records (ADRs)** for the platform.

Each record uses the standard **MADR (Markdown Any Decision Record)** format, capturing:
* Context & Problem Statement
* Decision Drivers (Security, Latency, Memory, Maintainability)
* Considered Alternatives
* Decision Outcome & Rationale
* Pros and Cons (Trade-offs)

---

## ADR Index

| ADR ID | Title | Status | Date |
| :--- | :--- | :---: | :--- |
| [**ADR-0001**](./0001-adoption-of-clean-hexagonal-architecture.md) | Adoption of Clean / Hexagonal Architecture | Accepted | 2026-08-25 |
| [**ADR-0002**](./0002-standardize-model-export-on-safetensors.md) | Standardize Model Serialization strictly on Safetensors | Accepted | 2026-08-25 |
| [**ADR-0003**](./0003-pydantic-v2-and-yaml-for-configuration-management.md) | Use Pydantic v2 and YAML for Immutable Configuration Engine | Accepted | 2026-08-25 |
| [**ADR-0004**](./0004-qlora-and-bitsandbytes-for-vram-constrained-finetuning.md) | QLoRA (NF4) & BitsAndBytes for Resource-Constrained Fine-Tuning | Accepted | 2026-08-25 |
| [**ADR-0005**](./0005-asynchronous-vllm-engine-for-production-serving.md) | Asynchronous vLLM Engine with PagedAttention for Model Serving | Accepted | 2026-08-25 |
