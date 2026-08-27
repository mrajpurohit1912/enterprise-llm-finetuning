"""
src/infrastructure/huggingface/huggingface_tokenizer.py
Hugging Face tokenizer adapter implementing TokenizerBase.
"""

from typing import Any
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.domain.exceptions import PreprocessingError
from src.domain.interfaces.tokenizer import TokenizerBase


class HuggingFaceTokenizer(TokenizerBase):
    """Loads and configures Hugging Face tokenizers."""

    def get_tokenizer(
        self,
        model_name: str,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> PreTrainedTokenizerBase:
        """
        Load tokenizer from Hugging Face hub or local cache.
        Ensures pad_token is assigned if missing.

        Args:
            model_name: Pretrained model identifier or directory.
            trust_remote_code: Allow custom code execution for community models.
            kwargs: Extra keyword arguments passed to AutoTokenizer.from_pretrained.

        Returns:
            Configured PreTrainedTokenizerBase instance.

        Raises:
            PreprocessingError: If tokenizer loading or initialization fails.
        """
        if not model_name or not model_name.strip():
            raise PreprocessingError("Tokenizer model name cannot be empty.")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name.strip(),
                trust_remote_code=trust_remote_code,
                **kwargs,
            )

            # Ensure pad_token is set (critical for batch training / collation)
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

            return tokenizer
        except Exception as exc:
            raise PreprocessingError(
                f"Failed to load tokenizer for model '{model_name}': {exc}"
            ) from exc
