"""
src/infrastructure/huggingface/chat_template_formatter.py
Applies Jinja2 chat templates to dataset batches for SFT training.
"""

from typing import Any, Dict, List, Optional
from transformers import PreTrainedTokenizerBase

from src.domain.exceptions import PreprocessingError
from src.domain.interfaces.prompt_formatter import PromptFormatterBase


class ChatTemplateFormatter(PromptFormatterBase):
    """
    Applies tokenizer Jinja chat templates across diverse dataset batch schemas.
    Supports OpenAI messages format, question/answer pairs, instruction/output pairs, and raw text.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        system_prompt: Optional[str] = None,
    ) -> None:
        if tokenizer is None:
            raise PreprocessingError("Tokenizer cannot be None when initializing ChatTemplateFormatter.")
        self._tokenizer = tokenizer
        self._system_prompt = system_prompt

    def format_batch(self, batch: Dict[str, List[Any]]) -> Dict[str, List[str]]:
        """
        Transform a dataset batch into standardized ChatML / chat-templated text strings.

        Args:
            batch: Dictionary mapping column names to lists of record values.

        Returns:
            Dictionary with a single key 'text' containing list of formatted string samples.

        Raises:
            PreprocessingError: If batch schema is unsupported or contains incompatible data.
        """
        if not batch:
            return {"text": []}

        # 1. Inspect batch columns and determine schema strategy
        columns = set(batch.keys())

        # Strategy 1: Pre-structured dialogue / messages list
        for msg_col in ("messages", "conversations", "dialogue"):
            if msg_col in columns:
                return self._format_messages_column(batch[msg_col])

        # Strategy 2: Question / Answer or Instruction / Output pairs
        pair_candidates = [
            ("question", "answer"),
            ("prompt", "response"),
            ("instruction", "output"),
            ("input", "output"),
            ("query", "response"),
        ]
        for user_key, assistant_key in pair_candidates:
            if user_key in columns and assistant_key in columns:
                return self._format_pair_columns(
                    user_inputs=batch[user_key],
                    assistant_responses=batch[assistant_key],
                    system_inputs=batch.get("system"),
                )

        # Strategy 3: Single 'text' or 'content' field already populated
        for text_col in ("text", "content"):
            if text_col in columns:
                return {"text": [str(item or "") for item in batch[text_col]]}

        raise PreprocessingError(
            f"Unsupported dataset batch columns: {list(columns)}. Expected columns matching "
            f"['messages'], ['question', 'answer'], ['instruction', 'output'], or ['text']."
        )

    def _format_pair_columns(
        self,
        user_inputs: List[Any],
        assistant_responses: List[Any],
        system_inputs: Optional[List[Any]] = None,
    ) -> Dict[str, List[str]]:
        """Formats parallel user and assistant column lists into chat messages."""
        texts: List[str] = []
        batch_size = len(user_inputs)

        for idx in range(batch_size):
            user_text = str(user_inputs[idx] or "").strip()
            assistant_text = str(assistant_responses[idx] or "").strip()

            messages: List[Dict[str, str]] = []

            # Add system prompt if available
            sys_text = (
                str(system_inputs[idx]).strip()
                if system_inputs and idx < len(system_inputs) and system_inputs[idx]
                else self._system_prompt
            )
            if sys_text:
                messages.append({"role": "system", "content": sys_text})

            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})

            formatted = self._apply_template_or_fallback(messages)
            texts.append(formatted)

        return {"text": texts}

    def _format_messages_column(self, messages_batch: List[Any]) -> Dict[str, List[str]]:
        """Formats a batch of pre-structured message lists."""
        texts: List[str] = []

        for row in messages_batch:
            if not isinstance(row, list):
                raise PreprocessingError(f"Expected list of messages per row, got {type(row).__name__}")

            normalized_messages: List[Dict[str, str]] = []
            for item in row:
                if isinstance(item, dict):
                    # Handle both OpenAI format ({'role': ..., 'content': ...})
                    # and ShareGPT format ({'from': ..., 'value': ...})
                    role = item.get("role") or ("user" if item.get("from") in ("human", "user") else "assistant")
                    content = item.get("content") or item.get("value") or ""
                    normalized_messages.append({"role": str(role), "content": str(content)})

            formatted = self._apply_template_or_fallback(normalized_messages)
            texts.append(formatted)

        return {"text": texts}

    def _apply_template_or_fallback(self, messages: List[Dict[str, str]]) -> str:
        """Applies tokenizer chat template, falling back to ChatML standard if unavailable."""
        try:
            if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
        except Exception:
            pass

        # Fallback to standard ChatML format
        formatted_lines: List[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted_lines.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        return "\n".join(formatted_lines)
