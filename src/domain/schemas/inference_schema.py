"""
src/domain/schemas/inference_schema.py
OpenAI-compatible request and response data contracts for LLM inference and serving.
"""

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Chat message representation containing role and text content."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request schema."""

    model: str = "meta-llama/Llama-3.2-3B-Instruct"
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=1)
    stream: bool = Field(default=False)
    stop: list[str] | None = None


class UsageInfo(BaseModel):
    """Token usage telemetry details."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponseChoice(BaseModel):
    """Individual generated completion choice for unary responses."""

    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length"] = "stop"


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response payload for non-streaming requests."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo


class ChatCompletionChunkDelta(BaseModel):
    """Delta payload representing incremental generated token content."""

    role: Literal["assistant"] | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    """Streaming chunk choice holding delta tokens."""

    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: Literal["stop", "length"] | None = None


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible Server-Sent Events (SSE) streaming chunk payload."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]
