"""
tests/unit/test_completion_usecases.py
Unit tests verifying GenerateCompletionUseCase and StreamCompletionUseCase.
"""

import json
import unittest
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

from src.application.usecases.generate_completion_usecase import (
    GenerateCompletionUseCase,
)
from src.application.usecases.stream_completion_usecase import StreamCompletionUseCase
from src.domain.interfaces.inference_engine import IInferenceEngine
from src.domain.schemas.inference_schema import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    UsageInfo,
)


class TestCompletionUseCases(unittest.IsolatedAsyncioTestCase):
    """Unit tests for inference application use cases."""

    async def test_generate_completion_usecase(self) -> None:
        mock_engine = MagicMock(spec=IInferenceEngine)
        mock_engine.engine_name = "mock_vllm"

        expected_response = ChatCompletionResponse(
            id="chatcmpl-test-123",
            created=1234567890,
            model="meta-llama/Llama-3.2-3B-Instruct",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Test answer"),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )

        mock_engine.generate = AsyncMock(return_value=expected_response)

        use_case = GenerateCompletionUseCase(engine=mock_engine)
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Test")])

        response = await use_case.execute(req)

        mock_engine.generate.assert_awaited_once_with(req)
        self.assertEqual(response.id, "chatcmpl-test-123")
        self.assertEqual(response.choices[0].message.content, "Test answer")
        self.assertEqual(response.usage.total_tokens, 30)

    async def test_stream_completion_usecase(self) -> None:
        mock_engine = MagicMock(spec=IInferenceEngine)
        mock_engine.engine_name = "mock_vllm"

        chunk1 = ChatCompletionChunk(
            id="chatcmpl-test-chunk-1",
            created=1234567890,
            model="meta-llama/Llama-3.2-3B-Instruct",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(content="Hello"),
                    finish_reason=None,
                )
            ],
        )

        chunk2 = ChatCompletionChunk(
            id="chatcmpl-test-chunk-1",
            created=1234567890,
            model="meta-llama/Llama-3.2-3B-Instruct",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(content=" world!"),
                    finish_reason="stop",
                )
            ],
        )

        async def mock_stream(request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
            yield chunk1
            yield chunk2

        mock_engine.stream_generate = mock_stream

        use_case = StreamCompletionUseCase(engine=mock_engine)
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hello")],
            stream=True,
        )

        sse_events: list[str] = []
        async for event in use_case.execute(req):
            sse_events.append(event)

        self.assertEqual(len(sse_events), 3)

        # Check first chunk event
        self.assertTrue(sse_events[0].startswith("data: "))
        self.assertTrue(sse_events[0].endswith("\n\n"))
        data1 = json.loads(sse_events[0][6:].strip())
        self.assertEqual(data1["choices"][0]["delta"]["content"], "Hello")

        # Check second chunk event
        self.assertTrue(sse_events[1].startswith("data: "))
        data2 = json.loads(sse_events[1][6:].strip())
        self.assertEqual(data2["choices"][0]["delta"]["content"], " world!")

        # Check terminal [DONE] event
        self.assertEqual(sse_events[2], "data: [DONE]\n\n")


if __name__ == "__main__":
    unittest.main()
