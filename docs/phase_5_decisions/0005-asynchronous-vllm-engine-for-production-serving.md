# ADR-0005: Asynchronous vLLM Engine with PagedAttention for Model Serving

* **Status:** Accepted
* **Deciders:** Lead ML Architect, Production Engineering Lead
* **Date:** 2026-08-25

## Context and Problem Statement
Standard Hugging Face `AutoModelForCausalLM.generate()` is designed for single-sequence interactive research. When exposed in a web API (e.g. standard FastAPI route), it runs synchronously, blocks the Python async event loop, wastes over 60-80% of GPU memory in KV-cache fragmentation, and collapses under concurrent traffic.

## Decision Drivers
* **Concurrent Throughput:** Serve hundreds of concurrent downstream requests with dynamic batching.
* **Memory Optimization:** Zero KV-cache waste via PagedAttention (operating system virtual memory paging model for attention keys/values).
* **Streaming & Latency:** Sub-millisecond token streaming with Server-Sent Events (SSE).

## Considered Options
1. **Synchronous Hugging Face `pipeline` in FastAPI:** Single request at a time; high latency, poor concurrency.
2. **Hugging Face Text Generation Inference (TGI):** Excellent throughput, but requires separate Rust/Docker orchestration.
3. **vLLM (AsyncLLMEngine):** Industry gold standard Python/C++ engine offering Continuous Batching and PagedAttention.

## Decision Outcome
Chosen Option: **vLLM with `AsyncLLMEngine`**.
* The platform's serving layer wraps `vLLM` to expose OpenAI-compatible REST API endpoints (`/v1/chat/completions`).
* GPU memory utilization is tuned to 90% for active KV-cache allocation.

### Positive Consequences
* $10\times - 24\times$ higher request throughput compared to Hugging Face Transformers baseline.
* Real-time token streaming with low Time-To-First-Token (TTFT $< 150\text{ms}$).

### Negative Consequences
* Requires CUDA compilation dependencies on deployment host.
