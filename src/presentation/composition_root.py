"""
src/presentation/composition_root.py
Enterprise Composition Root for assembling the dependency graph.
Wires domain factories, infrastructure adapters, and application use cases.
"""

from src.application.usecases.evaluate_model_usecase import EvaluateModelUseCase
from src.application.usecases.merge_model_usecase import MergeModelUseCase
from src.application.usecases.submit_remote_job_usecase import SubmitRemoteJobUseCase
from src.application.usecases.sync_remote_job_usecase import SyncRemoteJobUseCase
from src.application.usecases.train_model_usecase import TrainModelUseCase
from src.application.usecases.train_pipeline import TrainPipelineUsecase
from src.domain.interfaces.compute_provider import IComputeProvider
from src.domain.interfaces.config_loader import ConfigLoaderBase
from src.domain.schemas.config_schema import ExperimentConfig
from src.infrastructure.factories.compute_factory import ComputeProviderFactory
from src.infrastructure.factories.dataset_factory import DatasetLoaderFactory
from src.infrastructure.factories.evaluator_factory import EvaluatorFactory
from src.infrastructure.factories.trainer_factory import TrainerFactory
from src.infrastructure.huggingface.huggingface_tokenizer import HuggingFaceTokenizer
from src.infrastructure.llm_model_loader.transformers_llm_model_loader import (
    TransformerLlmModelLoader,
)
from src.infrastructure.monitoring.callback_factory import MonitoringCallbackFactory


def build_merge_usecase(config: ExperimentConfig) -> MergeModelUseCase:
    """Centralized assembly of MergeModelUseCase with model and tokenizer loaders."""
    llm_model_loader = TransformerLlmModelLoader()
    tokenizer_loader = HuggingFaceTokenizer()
    return MergeModelUseCase(
        config=config,
        model_loader=llm_model_loader,
        tokenizer_loader=tokenizer_loader,
    )


def build_train_pipeline(
    config: ExperimentConfig,
    config_loader: ConfigLoaderBase,
) -> TrainPipelineUsecase:
    """
    Central Composition Root assembling domain factories, loaders, and use cases
    for the primary training and evaluation pipeline.
    """
    dataset_factory = DatasetLoaderFactory()
    tokenizer_loader = HuggingFaceTokenizer()

    callbacks = MonitoringCallbackFactory.create_callbacks(
        monitoring_config=config.monitoring,
        experiment_name=config.experiment.name,
        hyperparameters=config.training_args.model_dump(),
    )

    trainer = TrainerFactory.get_trainer(config.training_args.trainer_type)
    train_model_usecase = TrainModelUseCase(trainer=trainer)

    evaluate_model_usecase = None
    if config.evaluation is not None:
        evaluator = EvaluatorFactory.get_evaluator(
            evaluator_type=config.evaluation.evaluator_type,
            dataset_name=config.dataset.dataset_name,
        )
        evaluate_model_usecase = EvaluateModelUseCase(evaluator=evaluator)

    merge_model_usecase = build_merge_usecase(config=config)

    return TrainPipelineUsecase(
        config_loader=config_loader,
        dataset_loader_factory=dataset_factory,
        tokenizer_loader=tokenizer_loader,
        train_model_usecase=train_model_usecase,
        evaluate_model_usecase=evaluate_model_usecase,
        merging_model_usecase=merge_model_usecase,
        callbacks=callbacks,
    )


def build_remote_usecases(
    provider_name: str,
) -> tuple[SubmitRemoteJobUseCase, SyncRemoteJobUseCase, IComputeProvider]:
    """Resolves compute backend provider and instantiates remote submission and synchronization use cases."""
    provider = ComputeProviderFactory.get_provider(provider_name)
    submit_usecase = SubmitRemoteJobUseCase(compute_provider=provider)
    sync_usecase = SyncRemoteJobUseCase(compute_provider=provider)
    return submit_usecase, sync_usecase, provider
