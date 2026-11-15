"""
src/application/usecases/stream_completion_usecase.py
Application use case for streaming chat completions over Server-Sent Events (SSE).
"""

from collections.abc import AsyncIterator

import structlog

from src.domain.interfaces.inference_engine import IInferenceEngine
from src.domain.schemas.inference_schema import ChatCompletionRequest

logger = structlog.get_logger(__name__)


class StreamCompletionUseCase:
    """Orchestrates streaming chat completion requests and yields SSE-formatted events."""

    def __init__(self, engine: IInferenceEngine) -> None:
        self.engine = engine

    async def execute(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        """
        Yields standard SSE formatted chunks:
        data: {"id": "...", ...}\n\n
        ...
        data: [DONE]\n\n
        """
        chunk_count = 0
        async for chunk in self.engine.stream_generate(request):
            chunk_count += 1
            yield f"data: {chunk.model_dump_json()}\n\n"

        logger.debug(
            "inference_stream_completed",
            engine=self.engine.engine_name,
            total_chunks=chunk_count,
        )

        # Standard OpenAI SSE stream termination signal
        yield "data: [DONE]\n\n"