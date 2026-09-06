"""
src/application/usecases/train_pipeline.py
Orchestration pipeline use case connecting domain configurations, data ingestion, preprocessing, and model preparation.
"""

import logging
import time
from typing import Any, Optional, Tuple

from src.application.services.dataset_processor import DatasetProcessor
from src.application.usecases.load_dataset_usecase import LoadDatasetUseCase
from src.application.usecases.preprocess_dataset_usecase import PreprocessDatasetUseCase
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

logger = logging.getLogger(__name__)


class TrainPipelineUsecase:
    """
    Enterprise orchestration use case executing fine-tuning pipeline stages.
    Coordinates configuration loading, dataset loading, chat-template formatting,
    quantized model loading, and LoRA adapter wrapping.
    """

    def __init__(
        self,
        config_loader: ConfigLoaderBase,
        dataset_loader_factory: Optional[DatasetLoaderFactory] = None,
        tokenizer_loader: Optional[TokenizerBase] = None,
        llm_model_loader: Optional[ModelLoaderBase] = None,
    ) -> None:
        self.config_loader = config_loader
        self.dataset_loader_factory = dataset_loader_factory or DatasetLoaderFactory()
        self.tokenizer_loader = tokenizer_loader or HuggingFaceTokenizer()
        self.llm_model_loader = llm_model_loader or TransformerLlmModelLoader()
        self.config: Optional[ExperimentConfig] = None
        self.llm_base_model: Any = None
        self.peft_model: Any = None

    def run(self) -> TrainPipelineResult:
        """
        Execute end-to-end pipeline stages with lifecycle telemetry and error boundaries.

        Returns:
            TrainPipelineResult: Strongly-typed operational result DTO.

        Raises:
            PipelineExecutionError: If any pipeline stage encounters an unrecoverable failure.
        """
        pipeline_start_time = time.perf_counter()
        logger.info("Initializing enterprise LLM fine-tuning pipeline...")

        try:
            # 1. Load and Validate Configuration
            self.config = self.config_loader.load_config()
            logger.info("Loaded validated experiment configuration: '%s'", self.config.experiment.name)

            # 2. Ingest and Preprocess Data
            processed_dataset, tokenizer = self._prepare_data(self.config)

            # 3. Load Quantized Foundation Model and Attach LoRA Adapters
            model = self._prepare_model(self.config)

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

            return TrainPipelineResult(
                experiment_name=self.config.experiment.name,
                status="SUCCESS",
                output_dir=str(self.config.artifact.output_dir),
                duration_seconds=round(elapsed_seconds, 2),
                dataset_size=dataset_sizes,
                model=model,
                tokenizer=tokenizer,
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

    def _prepare_model(self, config: ExperimentConfig) -> Any:
        """
        Execute foundation model loading, precision quantization, and PEFT adapter wrapping.

        Args:
            config: Validated experiment configuration.

        Returns:
            Any: Configured PEFT / base model ready for training.
        """
        model_id = config.llm_model.llm_model_id
        is_4bit = bool(config.quantization and config.quantization.load_in_4bit)

        logger.info("Loading foundation model '%s' (4-bit: %s)...", model_id, is_4bit)
        self.llm_base_model = self.llm_model_loader.load_model(
            model_id=model_id,
            quantization_config=config.quantization,
            trust_remote_code=config.llm_model.trust_remote_code,
        )

        if config.peft:
            logger.info("Attaching LoRA adapters (r=%d, alpha=%d)...", config.peft.r, config.peft.lora_alpha)
            self.peft_model = self.llm_model_loader.apply_peft(
                model=self.llm_base_model,
                peft_config=config.peft,
            )
            return self.peft_model

        self.peft_model = self.llm_base_model
        return self.peft_model
