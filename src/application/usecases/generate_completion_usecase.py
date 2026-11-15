"""
src/application/usecases/generate_completion_usecase.py
Application use case for non-streaming chat completions with performance telemetry.
"""

import time

import structlog

from src.domain.interfaces.inference_engine import IInferenceEngine
from src.domain.schemas.inference_schema import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)

logger = structlog.get_logger(__name__)


class GenerateCompletionUseCase:
    """Orchestrates unary chat completion requests and records generation telemetry."""

    def __init__(self, engine: IInferenceEngine) -> None:
        self.engine = engine

    async def execute(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        start_time = time.perf_counter()

        response = await self.engine.generate(request)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        tok_per_sec = (completion_tokens / (latency_ms / 1000.0)) if latency_ms > 0 else 0.0

        logger.info(
            "inference_completion_generated",
            engine=self.engine.engine_name,
            model=response.model,
            latency_ms=round(latency_ms, 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tokens_per_second=round(tok_per_sec, 2),
        )

        return response
