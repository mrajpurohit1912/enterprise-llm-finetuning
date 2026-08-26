# Phase 4: Contract
# DTOs & API Contracts Specification

**Document Version:** 1.0.0  
**Status:** Approved  

---

## 1. Inference Data Transfer Objects (DTOs)

The following DTOs define the request/response payloads for the FastAPI / vLLM serving microservice:

```python
"""
src/presentation/dtos.py
Data Transfer Objects (DTOs) for production inference and ingestion APIs.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the author: system | user | assistant")
    content: str = Field(..., description="Message text content")


class ChatCompletionRequestDTO(BaseModel):
    """OpenAI-compatible chat completion request schema."""
    model: str = Field(default="fine-tuned-model", description="Model alias")
    messages: List[ChatMessage] = Field(..., min_length=1)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    stream: bool = Field(default=False, description="Enable Server-Sent Events (SSE) streaming")


class ExtractionRequestDTO(BaseModel):
    """Domain-specific structured extraction request schema."""
    user_text: str = Field(..., min_length=3, description="Raw unstructured text")
    temperature: float = Field(default=0.0)


class ExtractionResponseDTO(BaseModel):
    """Domain-specific structured extraction response schema."""
    raw_response: str
    model_name: str
    latency_ms: float
```

---

## 2. OpenAPI REST Specification

### `POST /v1/chat/completions`
* **Summary:** Generate conversational completion via vLLM PagedAttention engine.
* **Content-Type:** `application/json` (or `text/event-stream` when `stream=true`).
* **Headers:** `Authorization: Bearer <API_KEY>`

```json
// Request Body
{
  "model": "qwen2.5-officeqa",
  "messages": [
    {"role": "user", "content": "How do I submit an expense report?"}
  ],
  "max_tokens": 150,
  "temperature": 0.1
}
```

```json
// Response Body (200 OK)
{
  "id": "chatcmpl-7f9a2b8e",
  "object": "chat.completion",
  "created": 1724590800,
  "model": "qwen2.5-officeqa",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "To submit an expense report, navigate to the Finance Portal..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 18,
    "completion_tokens": 42,
    "total_tokens": 60
  }
}
```

---

## 3. CLI Presentation Contract

The CLI command line interface is invoked via `typer` or `click`:

```bash
# Execute end-to-end fine-tuning pipeline
python -m src.presentation.cli train --config src/finetuning_config.yaml

# Run isolated dataset inspection/validation
python -m src.presentation.cli inspect-data --config src/finetuning_config.yaml --samples 5

# Launch async vLLM production server
python -m src.presentation.cli serve --model-path ./outputs/run-v1 --port 8000
```
