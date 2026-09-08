"""
src/infrastructure/trainer/__init__.py
Trainer adapters package exports.
"""

from src.infrastructure.trainer.huggingface_trainer import HuggingFaceTrainer
from src.infrastructure.trainer.trl_trainer import TrlTrainer

__all__ = [
    "HuggingFaceTrainer",
    "TrlTrainer",
]
