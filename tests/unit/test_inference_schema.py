"""
tests/unit/test_inference_schema.py
Unit tests for the OpenAI-compatible inference schemas and IInferenceEngine port interface.
"""

import unittest

from pydantic import ValidationError

from src.domain.interfaces.inference_engine import IInferenceEngine, InferenceEngineBase
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


class TestInferenceSchema(unittest.TestCase):
    """Unit tests verifying OpenAI-compatible inference request and response DTOs."""

    def test_chat_message_valid(self) -> None:
        """ChatMessage accepts system, user, and assistant roles."""
        msg_user = ChatMessage(role="user", content="Hello!")
        msg_system = ChatMessage(role="system", content="You are a helpful assistant.")
        msg_assistant = ChatMessage(role="assistant", content="How can I assist you?")

        self.assertEqual(msg_user.role, "user")
        self.assertEqual(msg_system.role, "system")
        self.assertEqual(msg_assistant.role, "assistant")

    def test_chat_message_invalid_role_raises(self) -> None:
        """ChatMessage rejects unsupported roles."""
        with self.assertRaises(ValidationError):
            ChatMessage(role="moderator", content="Invalid")  # type: ignore

    def test_chat_completion_request_defaults(self) -> None:
        """ChatCompletionRequest provides production-ready defaults."""
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="What is EBITDA?")]
        )
        self.assertEqual(req.model, "meta-llama/Llama-3.2-3B-Instruct")
        self.assertEqual(len(req.messages), 1)
        self.assertEqual(req.temperature, 0.7)
        self.assertEqual(req.top_p, 0.9)
        self.assertEqual(req.max_tokens, 512)
        self.assertFalse(req.stream)
        self.assertIsNone(req.stop)

    def test_chat_completion_request_validation(self) -> None:
        """Temperature, top_p, and max_tokens bounds are strictly enforced."""
        with self.assertRaises(ValidationError):
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="Hi")],
                temperature=2.5,  # Exceeds 2.0 bound
            )

        with self.assertRaises(ValidationError):
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="Hi")],
                max_tokens=0,  # Below 1 bound
            )

    def test_chat_completion_response_contract(self) -> None:
        """ChatCompletionResponse matches the standard OpenAI response layout."""
        resp = ChatCompletionResponse(
            model="fine-tuned-llama",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="EBITDA is..."),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=10, completion_tokens=25, total_tokens=35),
        )

        self.assertTrue(resp.id.startswith("chatcmpl-"))
        self.assertEqual(resp.object, "chat.completion")
        self.assertIsInstance(resp.created, int)
        self.assertEqual(resp.choices[0].message.content, "EBITDA is...")
        self.assertEqual(resp.usage.total_tokens, 35)

    def test_chat_completion_chunk_streaming_contract(self) -> None:
        """ChatCompletionChunk matches Server-Sent Events incremental chunk format."""
        chunk = ChatCompletionChunk(
            id="chatcmpl-stream-001",
            created=1234567890,
            model="fine-tuned-llama",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(content="EBIT"),
                    finish_reason=None,
                )
            ],
        )

        self.assertEqual(chunk.object, "chat.completion.chunk")
        self.assertEqual(chunk.choices[0].delta.content, "EBIT")
        self.assertIsNone(chunk.choices[0].finish_reason)

    def test_inference_engine_port_abstract(self) -> None:
        """IInferenceEngine port cannot be instantiated directly without implementing abstract methods."""
        with self.assertRaises(TypeError):
            IInferenceEngine()  # type: ignore

        self.assertIs(InferenceEngineBase, IInferenceEngine)
