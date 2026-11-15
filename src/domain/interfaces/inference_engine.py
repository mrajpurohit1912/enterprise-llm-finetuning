"""
src/domain/interfaces/inference_engine.py
Abstract base interface for LLM inference engines following the Dependency Inversion Principle.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from src.domain.schemas.inference_schema import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
)


class IInferenceEngine(ABC):
    """
    Abstract Port contract for inference backends.
    Infrastructure adapters (e.g. VLLMInferenceEngine, TransformersInferenceEngine) implement this.
    """

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Identifier for the inference backend (e.g. 'vllm', 'transformers_4bit')."""

    @abstractmethod
    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Execute non-streaming generation asynchronously.

        Args:
            request: Validated OpenAI-compatible chat completion request.

        Returns:
            ChatCompletionResponse: Full generated response with usage metrics.
        """

    @abstractmethod
    async def stream_generate(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        """
        Asynchronously yield token chunks for Server-Sent Events (SSE) streaming.

        Args:
            request: Validated OpenAI-compatible chat completion request with stream=True.

        Yields:
            ChatCompletionChunk: Incremental token delta chunks.
        """


InferenceEngineBase = IInferenceEngine

__all__ = ["IInferenceEngine", "InferenceEngineBase"]