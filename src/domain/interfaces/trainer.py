"""
src/domain/interfaces/trainer.py
Port interface for model fine-tuning trainers.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from src.domain.schemas.config_schema import TrainingArgs


class TrainerBase(ABC):
    """Abstract port for fine-tuning foundation models."""

    @abstractmethod
    def train(
        self,
        model: Any,
        train_dataset: Any,
        training_args: TrainingArgs,
        tokenizer: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        output_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Abstract method to execute training loop and persist trained model artifacts.

        Args:
            model: Pretrained or PEFT-wrapped model to fine-tune.
            train_dataset: Preprocessed dataset for training.
            training_args: TrainingArgs domain entity containing hyperparameters.
            tokenizer: Optional model tokenizer for padding and formatting.
            eval_dataset: Optional evaluation dataset split.
            output_dir: Target directory path for saving checkpoints.
            kwargs: Additional framework-specific training kwargs.

        Returns:
            Any: Result of the training process containing training metrics.
        """
        raise NotImplementedError("train method must be implemented by concrete trainer.")
