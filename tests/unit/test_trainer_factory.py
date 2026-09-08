"""
tests/unit/test_trainer_factory.py
Unit tests verifying TrainerFactory instantiation, mapping, and error states.
"""

import unittest

from src.domain.exceptions import TrainerError
from src.infrastructure.factories.trainer_factory import TrainerFactory
from src.infrastructure.trainer.huggingface_trainer import HuggingFaceTrainer
from src.infrastructure.trainer.trl_trainer import TrlTrainer


class TestTrainerFactory(unittest.TestCase):
    """Unit tests for TrainerFactory."""

    def test_get_trainer_huggingface_sft(self) -> None:
        trainer = TrainerFactory.get_trainer("huggingface_sft")
        self.assertIsInstance(trainer, TrlTrainer)

    def test_get_trainer_trl_alias(self) -> None:
        trainer = TrainerFactory.get_trainer("trl")
        self.assertIsInstance(trainer, TrlTrainer)

    def test_get_trainer_huggingface(self) -> None:
        trainer = TrainerFactory.get_trainer("huggingface")
        self.assertIsInstance(trainer, HuggingFaceTrainer)

    def test_get_trainer_unsupported_type_raises_error(self) -> None:
        with self.assertRaises(TrainerError) as ctx:
            TrainerFactory.get_trainer("non_existent_trainer")
        self.assertIn("Unsupported trainer type", str(ctx.exception))

    def test_get_trainer_planned_type_raises_informative_error(self) -> None:
        with self.assertRaises(TrainerError) as ctx:
            TrainerFactory.get_trainer("accelerate")
        self.assertIn("planned for a future release", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
