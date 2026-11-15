"""
tests/unit/test_inference_engine.py
Unit tests verifying InferenceEngineFactory and VLLMInferenceEngine with mocked vLLM backend.
"""

import unittest
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock, patch

from src.domain.exceptions import ConfigurationError, ModelInferenceError
from src.domain.interfaces.inference_engine import IInferenceEngine
from src.domain.schemas.inference_schema import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatMessage,
)
from src.infrastructure.factories.inference_factory import InferenceEngineFactory
from src.infrastructure.inference.vllm_engine import VLLMInferenceEngine


class DummyEngine(IInferenceEngine):
    @property
    def engine_name(self) -> str:
        return "dummy"

    async def generate(self, request: ChatCompletionRequest) -> Any:
        return None

    async def stream_generate(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        yield  # type: ignore


class TestInferenceEngineFactory(unittest.TestCase):
    """Unit tests for InferenceEngineFactory."""

    def test_factory_unsupported_engine_raises_configuration_error(self) -> None:
        with self.assertRaises(ConfigurationError) as ctx:
            InferenceEngineFactory.get_engine("unsupported_engine")
        self.assertIn("Unsupported inference engine", str(ctx.exception))

    def test_factory_dynamic_registration(self) -> None:
        InferenceEngineFactory.register_engine("dummy", DummyEngine)
        engine = InferenceEngineFactory.get_engine("DUMMY")
        self.assertIsInstance(engine, DummyEngine)
        self.assertEqual(engine.engine_name, "dummy")


class TestVLLMInferenceEngine(unittest.IsolatedAsyncioTestCase):
    """Unit tests for VLLMInferenceEngine using isolated async and mocks."""

    @patch("src.infrastructure.inference.vllm_engine.AutoTokenizer.from_pretrained")
    @patch("src.infrastructure.inference.vllm_engine.AsyncLLMEngine.from_engine_args")
    def setUp(self, mock_from_engine_args: MagicMock, mock_from_pretrained: MagicMock) -> None:
        self.mock_engine = MagicMock()
        mock_from_engine_args.return_value = self.mock_engine

        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.apply_chat_template.return_value = "<|user|>Hello<|assistant|>"
        mock_from_pretrained.return_value = self.mock_tokenizer

        self.engine = VLLMInferenceEngine(
            model_path="mock/llama-3.2-3b",
            quantization="bitsandbytes",
            max_model_len=2048,
        )

    def test_engine_name(self) -> None:
        self.assertEqual(self.engine.engine_name, "vllm")

    async def test_generate_success(self) -> None:
        """Verify unary non-streaming generation with mocked vLLM AsyncLLMEngine."""
        mock_output = MagicMock()
        mock_output.prompt_token_ids = [1, 2, 3]
        mock_choice = MagicMock()
        mock_choice.text = "Hello there! How can I assist you today?"
        mock_choice.finish_reason = "stop"
        mock_choice.token_ids = [4, 5, 6, 7]
        mock_output.outputs = [mock_choice]

        async def mock_async_generate(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            yield mock_output

        self.mock_engine.generate = mock_async_generate

        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hello")],
            temperature=0.7,
            max_tokens=100,
        )

        response = await self.engine.generate(req)

        self.assertTrue(response.id.startswith("chatcmpl-"))
        self.assertEqual(response.choices[0].message.content, "Hello there! How can I assist you today?")
        self.assertEqual(response.choices[0].finish_reason, "stop")
        self.assertEqual(response.usage.prompt_tokens, 3)
        self.assertEqual(response.usage.completion_tokens, 4)
        self.assertEqual(response.usage.total_tokens, 7)

    async def test_generate_empty_output_raises_error(self) -> None:
        """Verify ModelInferenceError is raised when engine produces no output."""
        async def mock_empty_generate(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            if False:
                yield None

        self.mock_engine.generate = mock_empty_generate

        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hello")],
        )

        with self.assertRaises(ModelInferenceError):
            await self.engine.generate(req)

    async def test_stream_generate_success(self) -> None:
        """Verify streaming generation yields delta chunks correctly."""
        output1 = MagicMock()
        choice1 = MagicMock(text="Hello", finish_reason=None)
        output1.outputs = [choice1]

        output2 = MagicMock()
        choice2 = MagicMock(text="Hello world!", finish_reason="stop")
        output2.outputs = [choice2]

        async def mock_stream(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            yield output1
            yield output2

        self.mock_engine.generate = mock_stream

        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hello")],
            stream=True,
        )

        chunks: list[ChatCompletionChunk] = []
        async for chunk in self.engine.stream_generate(req):
            chunks.append(chunk)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].choices[0].delta.content, "Hello")
        self.assertIsNone(chunks[0].choices[0].finish_reason)
        self.assertEqual(chunks[1].choices[0].delta.content, " world!")
        self.assertEqual(chunks[1].choices[0].finish_reason, "stop")


if __name__ == "__main__":
    unittest.main()
