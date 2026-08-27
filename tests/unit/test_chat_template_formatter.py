"""
tests/unit/test_chat_template_formatter.py
Unit tests for ChatTemplateFormatter and prompt transformation logic.
"""

import unittest
from unittest.mock import MagicMock

from src.domain.exceptions import PreprocessingError
from src.infrastructure.huggingface.chat_template_formatter import ChatTemplateFormatter


class MockTokenizer:
    """Mock tokenizer simulating Hugging Face tokenizer apply_chat_template."""

    def __init__(self, has_template: bool = True) -> None:
        self.chat_template = "mock_template_string" if has_template else None

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
    ) -> str:
        parts = [f"<{msg['role']}>{msg['content']}</{msg['role']}>" for msg in messages]
        return "".join(parts)


class TestChatTemplateFormatter(unittest.TestCase):
    """Tests ChatTemplateFormatter multi-schema formatting."""

    def setUp(self) -> None:
        self.tokenizer = MockTokenizer(has_template=True)
        self.formatter = ChatTemplateFormatter(tokenizer=self.tokenizer)

    def test_null_tokenizer_raises_error(self) -> None:
        """Initializing with None tokenizer must raise PreprocessingError."""
        with self.assertRaises(PreprocessingError):
            ChatTemplateFormatter(tokenizer=None)  # type: ignore

    def test_format_question_answer_pairs(self) -> None:
        """Question & Answer column batches must format correctly into ChatML."""
        batch = {
            "question": ["What is LoRA?", "What is QLoRA?"],
            "answer": [
                "Low-Rank Adaptation freezes base weights.",
                "QLoRA quantizes weights to 4-bit NF4.",
            ],
        }
        result = self.formatter.format_batch(batch)
        self.assertIn("text", result)
        self.assertEqual(len(result["text"]), 2)
        self.assertEqual(
            result["text"][0],
            "<user>What is LoRA?</user><assistant>Low-Rank Adaptation freezes base weights.</assistant>",
        )

    def test_format_prompt_response_pairs(self) -> None:
        """Prompt & Response column batches must format correctly."""
        batch = {
            "prompt": ["Summarize this document."],
            "response": ["Here is the summary."],
        }
        result = self.formatter.format_batch(batch)
        self.assertIn("text", result)
        self.assertEqual(
            result["text"][0],
            "<user>Summarize this document.</user><assistant>Here is the summary.</assistant>",
        )

    def test_format_instruction_output_pairs(self) -> None:
        """Instruction & Output column batches must format correctly."""
        batch = {
            "instruction": ["Translate to French."],
            "output": ["Bonjour."],
        }
        result = self.formatter.format_batch(batch)
        self.assertIn("text", result)
        self.assertEqual(
            result["text"][0],
            "<user>Translate to French.</user><assistant>Bonjour.</assistant>",
        )

    def test_format_messages_list_openai_format(self) -> None:
        """Messages list with role/content dictionaries must format correctly."""
        batch = {
            "messages": [
                [
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
            ]
        }
        result = self.formatter.format_batch(batch)
        self.assertIn("text", result)
        self.assertEqual(
            result["text"][0],
            "<user>Hello!</user><assistant>Hi there!</assistant>",
        )

    def test_format_raw_text_column(self) -> None:
        """Pre-formatted 'text' column should pass through unchanged."""
        batch = {"text": ["<|im_start|>user\nHi<|im_end|>"]}
        result = self.formatter.format_batch(batch)
        self.assertEqual(result["text"], ["<|im_start|>user\nHi<|im_end|>"])

    def test_unsupported_columns_raises_error(self) -> None:
        """Arbitrary unrecognized columns must raise PreprocessingError."""
        batch = {"random_col_1": [123], "random_col_2": [456]}
        with self.assertRaises(PreprocessingError) as ctx:
            self.formatter.format_batch(batch)
        self.assertIn("Unsupported dataset batch columns", str(ctx.exception))

    def test_fallback_chatml_when_no_template_on_tokenizer(self) -> None:
        """Tokenizer without chat_template should fall back to ChatML standard."""
        tokenizer_no_template = MockTokenizer(has_template=False)
        formatter = ChatTemplateFormatter(tokenizer=tokenizer_no_template)
        batch = {
            "question": ["How are you?"],
            "answer": ["I am doing well."],
        }
        result = formatter.format_batch(batch)
        self.assertIn("<|im_start|>user\nHow are you?<|im_end|>", result["text"][0])
        self.assertIn("<|im_start|>assistant\nI am doing well.<|im_end|>", result["text"][0])


if __name__ == "__main__":
    unittest.main()
