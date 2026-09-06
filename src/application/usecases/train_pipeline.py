"""
src/application/usecases/train_pipeline.py
Orchestration pipeline use case connecting domain configurations, data ingestion, and preprocessing.
"""

from typing import Any, Optional
from src.application.services.dataset_processor import DatasetProcessor
from src.application.usecases.load_dataset_usecase import LoadDatasetUseCase
from src.application.usecases.preprocess_dataset_usecase import PreprocessDatasetUseCase
from src.domain.interfaces.config_loader import ConfigLoaderBase
from src.domain.interfaces.model_loader import ModelLoaderBase
from src.domain.interfaces.tokenizer import TokenizerBase
from src.domain.schemas.config_schema import ExperimentConfig
from src.infrastructure.factories.dataset_factory import DatasetLoaderFactory
from src.infrastructure.huggingface.chat_template_formatter import ChatTemplateFormatter
from src.infrastructure.huggingface.huggingface_tokenizer import HuggingFaceTokenizer
from src.infrastructure.llm_model_loader.transformers_llm_model_loader import TransformerLlmModelLoader


class TrainPipelineUsecase:
    """
    Main orchestration use case executing fine-tuning pipeline stages.
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

    def run(self) -> Any:
        """
        Execute end-to-end pipeline stages.

        Returns:
            Configured PeftModel ready for SFTTrainer.
        """
        # 1. Load and Validate Configuration
        self.config = self.config_loader.load_config()
        print(f"[Pipeline] Loaded validated config for experiment: '{self.config.experiment.name}'")

        # 2. Dynamic Data Ingestion via Factory
        dataset_loader = self.dataset_loader_factory.get_loader(self.config.dataset.source)
        load_usecase = LoadDatasetUseCase(loader=dataset_loader)
        raw_dataset = load_usecase.execute(dataset_config=self.config.dataset)
        print(f"[Pipeline] Successfully loaded dataset: '{self.config.dataset.dataset_name}'")

        # 3. Initialize Model Tokenizer
        tokenizer = self.tokenizer_loader.get_tokenizer(
            model_name=self.config.llm_model.llm_model_id,
            trust_remote_code=self.config.llm_model.trust_remote_code,
        )
        print(f"[Pipeline] Initialized tokenizer for model: '{self.config.llm_model.llm_model_id}'")

        # 4. Initialize Preprocessing Strategy & Application Service
        formatter = ChatTemplateFormatter(tokenizer=tokenizer)
        processor = DatasetProcessor(formatter=formatter)
        preprocess_usecase = PreprocessDatasetUseCase(processor=processor)

        # 5. Execute Preprocessing Transformation
        processed_dataset = preprocess_usecase.execute(raw_dataset)
        print(f"[Pipeline] Successfully preprocessed and formatted dataset into ChatML standard.")

        # 6. Build Quantization Config & Load Foundation LLM Model
        model_id = self.config.llm_model.llm_model_id
        quant_config = None
        if self.config.quantization and self.config.quantization.load_in_4bit:
            import torch
            from transformers import BitsAndBytesConfig

            compute_dtype = getattr(torch, self.config.quantization.bnb_4bit_compute_dtype, torch.bfloat16)
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.quantization.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=self.config.quantization.bnb_4bit_use_double_quant,
            )

        print(f"[Pipeline] Loading LLM model: '{model_id}' (4-bit: {quant_config is not None})...")
        self.llm_base_model = self.llm_model_loader.load_model(
            model_id=model_id,
            quantization_config=quant_config,
            trust_remote_code=self.config.llm_model.trust_remote_code,
        )
        print(f"[Pipeline] Successfully loaded base model: {type(self.llm_base_model).__name__}")

        # 7. Apply PEFT / LoRA Adapters
        if self.config.peft:
            print(f"[Pipeline] Attaching LoRA adapters (r={self.config.peft.r}, alpha={self.config.peft.lora_alpha})...")
            self.peft_model = self.llm_model_loader.apply_peft(
                model=self.llm_base_model,
                peft_config=self.config.peft,
            )
            print("[Pipeline] Successfully configured PEFT model.")
        else:
            self.peft_model = self.llm_base_model

        return self.peft_model


