# ADR-0004: QLoRA (NF4) & BitsAndBytes for Resource-Constrained Fine-Tuning

* **Status:** Accepted
* **Deciders:** Lead ML Architect, Infrastructure FinOps Lead
* **Date:** 2026-08-25

## Context and Problem Statement
Full-parameter fine-tuning of 7B to 14B parameter models requires massive VRAM ($> 80\text{GB}-160\text{GB}$ across multiple A100/H100 GPUs) due to optimizer states, gradients, and activation overhead. To enable cost-effective fine-tuning on single commodity GPUs (24GB VRAM such as RTX 3090/4090 or A10G), we need a memory-efficient fine-tuning strategy.

## Decision Drivers
* **VRAM Budget:** Training 7B-14B models on single 24GB GPU nodes.
* **Accuracy Preservation:** Retain 99%+ of full 16-bit fine-tuning performance.
* **Optimizer Efficiency:** Prevent Out-Of-Memory (OOM) errors during peak gradient accumulation.

## Considered Options
1. **Full Parameter Fine-Tuning (16-bit BF16):** Best capacity, but cost-prohibitive for single-node development.
2. **Standard 16-bit LoRA:** Saves optimizer memory, but frozen base weights still consume $\approx 16\text{GB}$ VRAM for 7B.
3. **QLoRA (4-bit Normalized Float 4 + Double Quantization + Paged Optimizer):** Compresses frozen base weights to $\approx 4.5\text{GB}$ VRAM for 7B.

## Decision Outcome
Chosen Option: **QLoRA with `bitsandbytes` (NF4 + Double Quant + Paged AdamW 8-bit)**.
* Base weights quantized to 4-bit NormalFloat (`nf4`).
* LoRA adapters attached to all attention and MLP projection layers (`q, k, v, o, gate, up, down`) in 16-bit `bfloat16`.
* Activation checkpointing enabled to trade slight compute re-evaluation for massive activation memory savings.

### Positive Consequences
* Fine-tunes 7B-8B parameter models in $< 18\text{GB}$ VRAM with zero OOM crashes.
* Reduces cloud GPU compute costs by up to $70\%$.

### Negative Consequences
* QLoRA is roughly $20-30\%$ slower in forward/backward compute time compared to unquantized 16-bit LoRA due to on-the-fly dequantization.
