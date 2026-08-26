# ADR-0002: Standardize Model Export strictly on Safetensors

* **Status:** Accepted
* **Deciders:** Lead ML Architect, Enterprise InfoSec Officer
* **Date:** 2026-08-25

## Context and Problem Statement
When fine-tuning LLMs, model weights and adapter weights must be serialized for downstream storage, evaluation, and serving. Historically, PyTorch used Python's `pickle` format (`.bin` / `.pt`), which permits arbitrary Python bytecode execution on deserialization (`torch.load()`), exposing enterprise infrastructure to critical Remote Code Execution (RCE) vulnerabilities.

## Decision Drivers
* **Security & Vulnerability Mitigation:** Eliminate RCE vulnerabilities from untrusted model checkpoints.
* **Loading Speed:** Maximize model loading speed into GPU VRAM via memory-mapping (`mmap`).
* **Serving Compatibility:** Direct compatibility with vLLM, Hugging Face TGI, and NVIDIA Triton.

## Considered Options
1. **PyTorch Native Pickles (`.pt` / `pytorch_model.bin`):** Legacy format; high security risk.
2. **Hugging Face `safetensors`:** Safe, memory-mapped tensor format with zero bytecode execution.
3. **ONNX (Open Neural Network Exchange):** Good for small models, complex to export for 7B+ decoder LLMs.

## Decision Outcome
Chosen Option: **`safetensors` format exclusively**.
* All model exports (`save_pretrained`, merged model exports) will enforce `export_format="safetensors"`.
* The platform will reject loading `.bin` / `.pkl` weights in production pipelines.

### Positive Consequences
* Zero risk of arbitrary code execution upon loading weights.
* Cold-start loading times in vLLM reduced by up to $5\times$ via zero-copy `mmap`.

### Negative Consequences
* Legacy internal pipelines expecting `.bin` files require minor loading updates.
