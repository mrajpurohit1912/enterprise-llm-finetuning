"""
src/application/usecases/train_pipeline.py
Orchestration pipeline use case connecting domain configurations, data ingestion, and preprocessing.
"""

from typing import Any, Optional
from src.application.services.dataset_processor import DatasetProcessor
from src.application.usecases.load_dataset_usecase import LoadDatasetUseCase
from src.application.usecases.preprocess_dataset_usecase import PreprocessDatasetUseCase
from src.domain.interfaces.config_loader import ConfigLoaderBase
from src.domain.interfaces.tokenizer import TokenizerBase
from src.domain.schemas.config_schema import ExperimentConfig
from src.infrastructure.factories.dataset_factory import DatasetLoaderFactory
from src.infrastructure.huggingface.chat_template_formatter import ChatTemplateFormatter
from src.infrastructure.huggingface.huggingface_tokenizer import HuggingFaceTokenizer


class TrainPipelineUsecase:
    """
    Main orchestration use case executing fine-tuning pipeline stages.
    Coordinates configuration loading, dataset loading, and chat-template formatting.
    """

    def __init__(
        self,
        config_loader: ConfigLoaderBase,
        dataset_loader_factory: Optional[DatasetLoaderFactory] = None,
        tokenizer_loader: Optional[TokenizerBase] = None,
    ) -> None:
        self.config_loader = config_loader
        self.dataset_loader_factory = dataset_loader_factory or DatasetLoaderFactory()
        self.tokenizer_loader = tokenizer_loader or HuggingFaceTokenizer()
        self.config: Optional[ExperimentConfig] = None

    def run(self) -> Any:
        """
        Execute end-to-end pipeline stages.

        Returns:
            Preprocessed dataset ready for SFTTrainer.
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

        return processed_dataset
