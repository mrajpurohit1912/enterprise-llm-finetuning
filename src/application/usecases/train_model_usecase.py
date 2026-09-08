"""
src/application/usecases/train_model_usecase.py
Use case orchestrating the model fine-tuning process using a TrainerBase port.
"""

import logging
from typing import Any, Optional

from src.domain.exceptions import TrainerError
from src.domain.interfaces.trainer import TrainerBase
from src.domain.schemas.config_schema import TrainingArgs

logger = logging.getLogger(__name__)


class TrainModelUseCase:
    """
    Atomic use case encapsulating model training execution and checkpoint persistence.
    Relies purely on the TrainerBase domain interface (Dependency Inversion Principle).
    """

    def __init__(self, trainer: TrainerBase) -> None:
        if trainer is None:
            raise TrainerError("TrainerBase implementation cannot be None for TrainModelUseCase.")
        self._trainer = trainer

    def execute(
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
        Execute training on the provided model and dataset using the configured trainer.

        Args:
            model: Foundation or PEFT-adapted model to train.
            train_dataset: Preprocessed dataset for fine-tuning.
            training_args: Validated TrainingArgs domain entity.
            tokenizer: Optional model tokenizer for padding and formatting.
            eval_dataset: Optional evaluation dataset split.
            output_dir: Destination path for saving checkpoints and artifacts.
            kwargs: Additional framework-specific training options.

        Returns:
            Any: TrainOutput or training result summary from the trainer.

        Raises:
            TrainerError: If inputs are invalid or training execution fails.
        """
        if model is None:
            raise TrainerError("Model cannot be None for training execution.")
        if train_dataset is None:
            raise TrainerError("Train dataset cannot be None for training execution.")
        if training_args is None:
            raise TrainerError("TrainingArgs cannot be None for training execution.")

        logger.info(
            "Executing TrainModelUseCase: max_steps=%d, batch_size=%d, learning_rate=%s",
            training_args.max_steps,
            training_args.per_device_train_batch_size,
            training_args.learning_rate,
        )

        return self._trainer.train(
            model=model,
            train_dataset=train_dataset,
            training_args=training_args,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            output_dir=output_dir,
            **kwargs,
        )
