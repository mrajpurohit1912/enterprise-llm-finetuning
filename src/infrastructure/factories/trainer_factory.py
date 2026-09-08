"""
src/infrastructure/factories/trainer_factory.py
Factory for instantiating TrainerBase concrete adapters based on trainer_type.
"""

from typing import Union
from src.domain.exceptions import TrainerError
from src.domain.interfaces.trainer import TrainerBase
from src.infrastructure.trainer.huggingface_trainer import HuggingFaceTrainer
from src.infrastructure.trainer.trl_trainer import TrlTrainer


class TrainerFactory:
    """Factory creating appropriate model trainer adapters based on configuration."""

    @staticmethod
    def get_trainer(trainer_type: str) -> TrainerBase:
        """
        Instantiate and return the concrete trainer adapter matching the trainer type.

        Args:
            trainer_type: Configured trainer type identifier (e.g. 'huggingface_sft', 'huggingface').

        Returns:
            Concrete TrainerBase adapter.

        Raises:
            TrainerError: If the trainer type is unsupported or planned for a future release.
        """
        normalized_type = trainer_type.strip().lower()

        if normalized_type in ("huggingface_sft", "trl", "sft"):
            return TrlTrainer()
        elif normalized_type in ("huggingface", "hf", "transformers"):
            return HuggingFaceTrainer()
        elif normalized_type in ("accelerate", "lightning", "deepspeed"):
            raise TrainerError(
                f"Trainer '{trainer_type}' is planned for a future release. "
                "Currently supported: ['huggingface_sft', 'huggingface']."
            )
        else:
            raise TrainerError(
                f"Unsupported trainer type '{trainer_type}'. "
                "Supported: ['huggingface_sft', 'huggingface']."
            )
