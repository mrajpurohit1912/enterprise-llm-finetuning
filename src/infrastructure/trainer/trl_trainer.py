"""
src/infrastructure/trainer/trl_trainer.py
TRL SFTTrainer adapter implementing TrainerBase.
"""

import logging
from typing import Any, Optional
from trl import SFTConfig, SFTTrainer

from src.domain.interfaces.trainer import TrainerBase
from src.domain.schemas.config_schema import TrainingArgs

logger = logging.getLogger(__name__)


class TrlTrainer(TrainerBase):
    """Concrete Trainer adapter using Hugging Face TRL SFTTrainer."""

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
        Execute fine-tuning using Hugging Face TRL SFTTrainer.

        Args:
            model: Pretrained or PEFT-wrapped model to fine-tune.
            train_dataset: Preprocessed dataset for training.
            training_args: TrainingArgs domain entity containing hyperparameters.
            tokenizer: Optional model tokenizer for padding and formatting.
            eval_dataset: Optional evaluation dataset split.
            output_dir: Target directory path for saving checkpoints.
            kwargs: Additional arguments passed to SFTTrainer.

        Returns:
            Any: TrainOutput result containing training loss and runtime metrics.
        """
        save_path = output_dir or "./outputs"

        # 1. Translate domain TrainingArgs into framework-native SFTConfig
        sft_config = SFTConfig(
            output_dir=save_path,
            dataset_text_field=training_args.dataset_text_field,
            max_length=training_args.max_length,
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            learning_rate=training_args.learning_rate,
            logging_steps=training_args.logging_steps,
            max_steps=training_args.max_steps,
            bf16=training_args.bf16,
            fp16=training_args.fp16,
            optim=training_args.optim,
            save_strategy=training_args.save_strategy,
            save_steps=training_args.save_steps,
            report_to=training_args.report_to,
        )

        # 2. Defensively extract split if DatasetDict is provided
        if hasattr(train_dataset, "keys") and hasattr(train_dataset, "__getitem__"):
            if "train" in train_dataset:
                if eval_dataset is None:
                    eval_dataset = train_dataset.get("test") or train_dataset.get("validation") or train_dataset.get("eval")
                train_dataset = train_dataset["train"]

        callbacks = kwargs.pop("callbacks", None)

        # 3. Instantiate SFTTrainer
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            callbacks=callbacks,
            **kwargs,
        )

        # 4. Train model
        logger.info("Starting model training with TRL SFTTrainer (max_steps=%d)...", training_args.max_steps)
        try:
            train_result = trainer.train()

            # 5. Persist fine-tuned weights and tokenizer
            logger.info("Saving fine-tuned model and artifacts to '%s'...", save_path)
            trainer.save_model(save_path)
            if tokenizer is not None:
                tokenizer.save_pretrained(save_path)

            return train_result
        finally:
            from src.infrastructure.monitoring.wandb_tracker import WandbTracker
            WandbTracker.finish()