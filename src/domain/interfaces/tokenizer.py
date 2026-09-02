"""
src/domain/interfaces/tokenizer.py
Port interface for tokenizer loading and management.
"""

from abc import ABC, abstractmethod
from typing import Any


class TokenizerBase(ABC):
    """Abstract port for tokenizer loading and vocabulary management."""

    @abstractmethod
    def get_tokenizer(self, model_name: str, **kwargs: Any) -> Any:
        """
        Load and configure target model tokenizer.

        Args:
            model_name: Foundation model identifier or local tokenizer path.
            kwargs: Additional tokenizer configuration arguments.

        Returns:
            Configured tokenizer instance.
        """
        raise NotImplementedError("get_tokenizer method must be implemented by concrete adapter.")
