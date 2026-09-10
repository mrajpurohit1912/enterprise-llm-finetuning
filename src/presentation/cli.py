"""
src/presentation/cli.py
Command-line presentation interface for triggering fine-tuning pipelines.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from src.application.usecases.evaluate_model_usecase import EvaluateModelUseCase
from src.application.usecases.train_model_usecase import TrainModelUseCase
from src.application.usecases.train_pipeline import TrainPipelineUsecase
from src.domain.exceptions import DomainError
from src.infrastructure.config.yaml_loader import YamlConfigLoader
from src.infrastructure.factories.dataset_factory import DatasetLoaderFactory
from src.infrastructure.factories.evaluator_factory import EvaluatorFactory
from src.infrastructure.factories.trainer_factory import TrainerFactory
from src.infrastructure.huggingface.huggingface_tokenizer import HuggingFaceTokenizer
from src.infrastructure.logging import setup_logging
from src.infrastructure.monitoring.callback_factory import MonitoringCallbackFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enterprise LLM Fine-Tuning Pipeline CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="src/finetuning_config.yaml",
        help="Path to YAML fine-tuning experiment specification",
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=["development", "production"],
        default=os.getenv("ENVIRONMENT", "development"),
        help="Environment execution mode ('development' for colors, 'production' for JSON)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Logging verbosity level",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and run standalone evaluation on the existing saved model artifacts",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    setup_logging(env=args.env, log_level=args.log_level)
    config_path = Path(args.config)

    try:
        config_loader = YamlConfigLoader(config_path=config_path)
        config = config_loader.load_config()

        dataset_factory = DatasetLoaderFactory()
        tokenizer_loader = HuggingFaceTokenizer()

        # Composition Root: Resolve monitoring callbacks (Prometheus, W&B)
        callbacks = MonitoringCallbackFactory.create_callbacks(
            monitoring_config=config.monitoring,
            experiment_name=config.experiment.name,
            hyperparameters=config.training_args.model_dump(),
        )

        # Composition Root: Resolve trainer adapter and inject into atomic use case
        trainer = TrainerFactory.get_trainer(config.training_args.trainer_type)
        train_model_usecase = TrainModelUseCase(trainer=trainer)

        # Composition Root: Resolve evaluator adapter and inject into atomic use case
        evaluate_model_usecase = None
        if config.evaluation is not None:
            evaluator = EvaluatorFactory.get_evaluator(config.evaluation.evaluator_type)
            evaluate_model_usecase = EvaluateModelUseCase(evaluator=evaluator)

        pipeline = TrainPipelineUsecase(
            config_loader=config_loader,
            dataset_loader_factory=dataset_factory,
            tokenizer_loader=tokenizer_loader,
            train_model_usecase=train_model_usecase,
            evaluate_model_usecase=evaluate_model_usecase,
            callbacks=callbacks,
        )

        result = pipeline.run(eval_only=args.eval_only)

        print("\n=======================================================")
        print("          FINE-TUNING PIPELINE RUN SUMMARY             ")
        print("=======================================================")
        print(f"Experiment Name   : {result.experiment_name}")
        print(f"Execution Status  : {result.status}")
        print(f"Elapsed Time      : {result.duration_seconds}s")
        print(f"Artifacts Output  : {result.output_dir}")
        if result.train_loss is not None:
            print(f"Final Train Loss  : {result.train_loss:.4f}")
        if result.dataset_size:
            print(f"Dataset Partitions: {result.dataset_size}")
        if result.evaluation is not None:
            print(f"Evaluation F1     : {result.evaluation.metrics.f1_score:.4f}")
            print(f"Exact Match (EM)  : {result.evaluation.metrics.exact_match:.4f}")
            print(f"ROUGE-L           : {result.evaluation.metrics.rouge_l or 0.0:.4f}")
            gate = "PASSED" if result.evaluation.passed_quality_gate else "FAILED"
            print(f"Quality Gate      : {gate}")
        print("=======================================================\n")
    except DomainError as err:
        print(f"[Domain Error] {err}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"[Fatal Error] Unexpected failure: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
