"""
tests/unit/test_train_model_usecase.py
Unit tests verifying TrainModelUseCase in complete isolation using mock TrainerBase.
"""

import unittest
from unittest.mock import MagicMock

from src.application.usecases.train_model_usecase import TrainModelUseCase
from src.domain.exceptions import TrainerError
from src.domain.schemas.config_schema import TrainingArgs


class TestTrainModelUseCase(unittest.TestCase):
    """Unit tests for TrainModelUseCase."""

    def setUp(self) -> None:
        self.mock_trainer = MagicMock()
        self.use_case = TrainModelUseCase(trainer=self.mock_trainer)
        self.dummy_model = MagicMock()
        self.dummy_dataset = MagicMock()
        self.dummy_tokenizer = MagicMock()
        self.training_args = TrainingArgs(max_steps=10, per_device_train_batch_size=2)

    def test_init_raises_error_if_trainer_is_none(self) -> None:
        with self.assertRaises(TrainerError):
            TrainModelUseCase(trainer=None)

    def test_execute_delegates_to_trainer(self) -> None:
        expected_output = {"train_loss": 0.42, "train_runtime": 12.5}
        self.mock_trainer.train.return_value = expected_output

        result = self.use_case.execute(
            model=self.dummy_model,
            train_dataset=self.dummy_dataset,
            training_args=self.training_args,
            tokenizer=self.dummy_tokenizer,
            output_dir="./test_output",
        )

        self.mock_trainer.train.assert_called_once_with(
            model=self.dummy_model,
            train_dataset=self.dummy_dataset,
            training_args=self.training_args,
            tokenizer=self.dummy_tokenizer,
            eval_dataset=None,
            output_dir="./test_output",
        )
        self.assertEqual(result, expected_output)

    def test_execute_validates_required_inputs(self) -> None:
        with self.assertRaises(TrainerError):
            self.use_case.execute(
                model=None,
                train_dataset=self.dummy_dataset,
                training_args=self.training_args,
            )

        with self.assertRaises(TrainerError):
            self.use_case.execute(
                model=self.dummy_model,
                train_dataset=None,
                training_args=self.training_args,
            )

        with self.assertRaises(TrainerError):
            self.use_case.execute(
                model=self.dummy_model,
                train_dataset=self.dummy_dataset,
                training_args=None,
            )


if __name__ == "__main__":
    unittest.main()
