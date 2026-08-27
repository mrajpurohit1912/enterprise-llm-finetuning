"""
src/domain/interfaces/prompt_formatter.py
Port interface for chat template formatting strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class PromptFormatterBase(ABC):
    """Abstract port for dataset batch prompt formatting strategies."""

    @abstractmethod
    def format_batch(self, batch: Dict[str, List[Any]]) -> Dict[str, List[str]]:
        """
        Format a batch of raw records into model-consumable formatted text strings.

        Args:
            batch: Dictionary of column lists (e.g., {'question': [...], 'answer': [...]},
                   {'prompt': [...], 'response': [...]}, or {'messages': [...]}).

        Returns:
            Dictionary containing formatted text strings (e.g., {'text': [...]}).

        Raises:
            PreprocessingError: If required fields are missing or cannot be formatted.
        """
        raise NotImplementedError("format_batch method must be implemented by concrete formatter.")
