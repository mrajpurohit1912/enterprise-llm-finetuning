"""
src/application/usecases/train_pipeline.py
Orchestration pipeline use case connecting domain configurations, data ingestion, preprocessing, model preparation, and training.
"""

import logging
import time
from pathlib import Path
from typing import Any, Optional, Tuple

from src.application.services.dataset_processor import DatasetProcessor
from src.application.usecases.evaluate_model_usecase import EvaluateModelUseCase
from src.application.usecases.load_dataset_usecase import LoadDatasetUseCase
from src.application.usecases.preprocess_dataset_usecase import PreprocessDatasetUseCase
from src.application.usecases.train_model_usecase import TrainModelUseCase
from src.domain.exceptions import DomainError, PipelineExecutionError
from src.domain.interfaces.config_loader import ConfigLoaderBase
from src.domain.interfaces.model_loader import ModelLoaderBase
from src.domain.interfaces.tokenizer import TokenizerBase
from src.domain.schemas.config_schema import ExperimentConfig
from src.domain.schemas.pipeline_schema import TrainPipelineResult
from src.infrastructure.factories.dataset_factory import DatasetLoaderFactory
from src.infrastructure.huggingface.chat_template_formatter import ChatTemplateFormatter
from src.infrastructure.huggingface.huggingface_tokenizer import HuggingFaceTokenizer
from src.infrastructure.llm_model_loader.transformers_llm_model_loader import TransformerLlmModelLoader
from src.infrastructure.logging import clear_logging_context, set_logging_context

logger = logging.getLogger(__name__)


class TrainPipelineUsecase:
    """
    Enterprise orchestration use case executing fine-tuning pipeline stages.
    Coordinates configuration loading, dataset loading, chat-template formatting,
    quantized model loading, LoRA adapter wrapping, and model training.
    """

    def __init__(
        self,
        config_loader: ConfigLoaderBase,
        dataset_loader_factory: Optional[DatasetLoaderFactory] = None,
        tokenizer_loader: Optional[TokenizerBase] = None,
        llm_model_loader: Optional[ModelLoaderBase] = None,
        train_model_usecase: Optional[TrainModelUseCase] = None,
        evaluate_model_usecase: Optional[EvaluateModelUseCase] = None,
        callbacks: Optional[list] = None,
    ) -> None:
        self.config_loader = config_loader
        self.dataset_loader_factory = dataset_loader_factory or DatasetLoaderFactory()
        self.tokenizer_loader = tokenizer_loader or HuggingFaceTokenizer()
        self.llm_model_loader = llm_model_loader or TransformerLlmModelLoader()
        self.train_model_usecase = train_model_usecase
        self.evaluate_model_usecase = evaluate_model_usecase
        self.callbacks = callbacks or []
        self.config: Optional[ExperimentConfig] = None
        self.llm_base_model: Any = None
        self.peft_model: Any = None

    def run(self, eval_only: bool = False) -> TrainPipelineResult:
        """
        Execute end-to-end pipeline stages with lifecycle telemetry and error boundaries.

        Args:
            eval_only: If True, skips training and evaluates the existing model/adapter.

        Returns:
            TrainPipelineResult: Strongly-typed operational result DTO.

        Raises:
            PipelineExecutionError: If any pipeline stage encounters an unrecoverable failure.
        """
        pipeline_start_time = time.perf_counter()
        logger.info("Initializing enterprise LLM fine-tuning pipeline (eval_only: %s)...", eval_only)

        try:
            # 1. Load and Validate Configuration
            self.config = self.config_loader.load_config()
            set_logging_context(
                experiment_name=self.config.experiment.name,
                model_id=self.config.llm_model.llm_model_id,
                dataset_name=self.config.dataset.dataset_name,
            )
            logger.info("Loaded validated experiment configuration: '%s'", self.config.experiment.name)

            # 2. Ingest and Preprocess Data
            processed_dataset, tokenizer = self._prepare_data(self.config)

            # 3. Load Quantized Foundation Model and Attach LoRA Adapters
            adapter_dir = str(self.config.artifact.output_dir) if eval_only else None
            model = self._prepare_model(self.config, adapter_path=adapter_dir)

            # 4. Execute Model Training Stage (if not eval_only and TrainModelUseCase is configured)
            train_output = None
            if not eval_only and self.train_model_usecase is not None:
                train_output = self._execute_training(
                    config=self.config,
                    model=model,
                    train_dataset=processed_dataset,
                    tokenizer=tokenizer,
                )
            elif eval_only:
                logger.info("Executing in eval-only mode. Skipping training loop.")

            # 5. Execute Post-Training Model Evaluation Stage (if configured)
            eval_result = None
            if self.evaluate_model_usecase is not None and self.config.evaluation is not None:
                eval_split_name = self.config.evaluation.eval_split
                eval_dataset = None
                if hasattr(processed_dataset, "get"):
                    eval_dataset = processed_dataset.get(eval_split_name)
                    if eval_dataset is None and eval_split_name == "test":
                        eval_dataset = processed_dataset.get("validation") or processed_dataset.get("eval")
                elif hasattr(processed_dataset, "__getitem__") and hasattr(processed_dataset, "keys"):
                    eval_dataset = processed_dataset.get(eval_split_name)

                if eval_dataset is not None and len(eval_dataset) > 0:
                    logger.info("Executing evaluation stage on split '%s' (%d samples)...", eval_split_name, len(eval_dataset))
                    eval_result = self.evaluate_model_usecase.execute(
                        model=model,
                        tokenizer=tokenizer,
                        eval_dataset=eval_dataset,
                        eval_config=self.config.evaluation,
                        experiment_name=self.config.experiment.name,
                        model_id=self.config.llm_model.llm_model_id,
                        output_dir=str(self.config.artifact.output_dir),
                    )
                    # Log evaluation scorecard to Weights & Biases if session active
                    try:
                        import wandb
                        if wandb.run is not None:
                            wandb.log({
                                "eval/exact_match": eval_result.metrics.exact_match,
                                "eval/f1_score": eval_result.metrics.f1_score,
                                "eval/rouge_l": eval_result.metrics.rouge_l or 0.0,
                                "eval/latency_per_sample_ms": eval_result.latency_per_sample_ms,
                                "eval/throughput_tokens_per_sec": eval_result.throughput_tokens_per_sec,
                            })
                    except Exception:
                        pass
                else:
                    logger.warning("Evaluation split '%s' not found in processed dataset. Skipping evaluation.", eval_split_name)

            elapsed_seconds = time.perf_counter() - pipeline_start_time
            logger.info(
                "Fine-tuning pipeline completed successfully in %.2f seconds.",
                elapsed_seconds,
            )

            dataset_sizes = {}
            if hasattr(processed_dataset, "keys"):
                dataset_sizes = {split: len(processed_dataset[split]) for split in processed_dataset.keys()}
            elif hasattr(processed_dataset, "__len__"):
                dataset_sizes = {"train": len(processed_dataset)}

            # Extract metrics from training output if available
            train_loss = None
            metrics = None
            if train_output is not None:
                train_loss = getattr(train_output, "training_loss", None)
                metrics = getattr(train_output, "metrics", None)
                if isinstance(train_output, dict):
                    train_loss = train_output.get("training_loss", train_output.get("train_loss"))
                    metrics = train_output

            return TrainPipelineResult(
                experiment_name=self.config.experiment.name,
                status="SUCCESS",
                output_dir=str(self.config.artifact.output_dir),
                duration_seconds=round(elapsed_seconds, 2),
                train_loss=train_loss,
                metrics=metrics,
                dataset_size=dataset_sizes,
                model=model,
                tokenizer=tokenizer,
                evaluation=eval_result,
            )

        except DomainError:
            # Re-raise domain errors untouched to maintain layer integrity
            raise
        except Exception as exc:
            elapsed_seconds = time.perf_counter() - pipeline_start_time
            logger.error(
                "Pipeline execution failed after %.2f seconds: %s",
                elapsed_seconds,
                str(exc),
                exc_info=True,
            )
            raise PipelineExecutionError(f"Pipeline execution failed: {exc}") from exc
        finally:
            clear_logging_context()

    def _prepare_data(self, config: ExperimentConfig) -> Tuple[Any, Any]:
        """
        Execute dataset loading and chat-template formatting.

        Args:
            config: Validated experiment configuration.

        Returns:
            Tuple[Any, Any]: (processed_dataset, tokenizer)
        """
        logger.info("Ingesting dataset from source '%s': '%s'...", config.dataset.source, config.dataset.dataset_name)
        dataset_loader = self.dataset_loader_factory.get_loader(config.dataset.source)
        load_usecase = LoadDatasetUseCase(loader=dataset_loader)
        raw_dataset = load_usecase.execute(dataset_config=config.dataset)

        # Enterprise holdout split: if dataset lacks an eval split, auto-generate reproducible holdout partition
        eval_split_target = config.dataset.eval_split or (config.evaluation.eval_split if config.evaluation else "test")
        if eval_split_target:
            if hasattr(raw_dataset, "keys") and eval_split_target not in raw_dataset and "train" in raw_dataset:
                train_len = len(raw_dataset["train"]) if hasattr(raw_dataset["train"], "__len__") else 0
                if train_len > 1:
                    test_size = 0.15 if train_len >= 5 else 1
                    logger.info(
                        "Raw dataset only contains 'train' split. Creating reproducible holdout '%s' split (seed=%d)...",
                        eval_split_target,
                        config.experiment.seed,
                    )
                    splits = raw_dataset["train"].train_test_split(test_size=test_size, seed=config.experiment.seed)
                    if eval_split_target != "test":
                        splits[eval_split_target] = splits.pop("test")
                    raw_dataset = splits
            elif hasattr(raw_dataset, "train_test_split") and not hasattr(raw_dataset, "keys"):
                ds_len = len(raw_dataset) if hasattr(raw_dataset, "__len__") else 0
                if ds_len > 1:
                    test_size = 0.15 if ds_len >= 5 else 1
                    logger.info(
                        "Raw dataset is a single partition. Creating reproducible holdout '%s' split (seed=%d)...",
                        eval_split_target,
                        config.experiment.seed,
                    )
                    splits = raw_dataset.train_test_split(test_size=test_size, seed=config.experiment.seed)
                    if eval_split_target != "test":
                        splits[eval_split_target] = splits.pop("test")
                    raw_dataset = splits

        logger.info("Initializing tokenizer for model: '%s'...", config.llm_model.llm_model_id)
        tokenizer = self.tokenizer_loader.get_tokenizer(
            model_name=config.llm_model.llm_model_id,
            trust_remote_code=config.llm_model.trust_remote_code,
        )

        logger.info("Applying chat template formatting to dataset...")
        formatter = ChatTemplateFormatter(tokenizer=tokenizer)
        processor = DatasetProcessor(formatter=formatter)
        preprocess_usecase = PreprocessDatasetUseCase(processor=processor)
        processed_dataset = preprocess_usecase.execute(raw_dataset)

        return processed_dataset, tokenizer

    def _prepare_model(self, config: ExperimentConfig, adapter_path: Optional[str] = None) -> Any:
        """
        Execute foundation model loading, precision quantization, and PEFT adapter wrapping.

        Args:
            config: Validated experiment configuration.
            adapter_path: Optional path to existing trained adapter checkpoint.

        Returns:
            Any: Configured PEFT / base model ready for training or evaluation.
        """
        model_id = config.llm_model.llm_model_id
        is_4bit = bool(config.quantization and config.quantization.load_in_4bit)

        logger.info("Loading foundation model '%s' (4-bit: %s)...", model_id, is_4bit)
        self.llm_base_model = self.llm_model_loader.load_model(
            model_id=model_id,
            quantization_config=config.quantization,
            trust_remote_code=config.llm_model.trust_remote_code,
        )

        if adapter_path:
            p = Path(adapter_path)
            if (p / "adapter_model.safetensors").exists() or (p / "adapter_model.bin").exists():
                try:
                    from peft import PeftModel
                    logger.info("Loading existing trained LoRA adapter from '%s'...", adapter_path)
                    self.peft_model = PeftModel.from_pretrained(self.llm_base_model, str(adapter_path))
                    return self.peft_model
                except Exception as exc:
                    logger.warning("Failed to load adapter from '%s': %s. Re-initializing new adapter.", adapter_path, exc)

        if config.peft:
            logger.info("Attaching LoRA adapters (r=%d, alpha=%d)...", config.peft.r, config.peft.lora_alpha)
            self.peft_model = self.llm_model_loader.apply_peft(
                model=self.llm_base_model,
                peft_config=config.peft,
            )
            return self.peft_model

        self.peft_model = self.llm_base_model
        return self.peft_model

    def _execute_training(
        self,
        config: ExperimentConfig,
        model: Any,
        train_dataset: Any,
        tokenizer: Any,
    ) -> Any:
        """
        Execute model training using the injected TrainModelUseCase.

        Args:
            config: Validated experiment configuration.
            model: Foundation or adapted PEFT model.
            train_dataset: Formatted training dataset (Dataset or DatasetDict).
            tokenizer: Model tokenizer.

        Returns:
            Any: Framework training output metrics.
        """
        output_dir = str(config.artifact.output_dir)

        # Unpack split partitions if dataset is a DatasetDict
        train_split = train_dataset
        eval_split = None
        if hasattr(train_dataset, "keys") and hasattr(train_dataset, "__getitem__"):
            train_split = train_dataset.get("train", next(iter(train_dataset.values())))
            eval_split = train_dataset.get("test") or train_dataset.get("validation") or train_dataset.get("eval")
            logger.info("Extracted 'train' partition (%d examples) for training.", len(train_split))
            if eval_split is not None:
                logger.info("Extracted evaluation partition (%d examples) for evaluation.", len(eval_split))

        logger.info("Executing training stage via TrainModelUseCase (output_dir='%s')...", output_dir)
        return self.train_model_usecase.execute(
            model=model,
            train_dataset=train_split,
            eval_dataset=eval_split,
            training_args=config.training_args,
            tokenizer=tokenizer,
            output_dir=output_dir,
            callbacks=self.callbacks,
        )
