"""
src/presentation/cli.py
Command-line presentation interface for triggering fine-tuning pipelines.
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from src.application.usecases.evaluate_model_usecase import EvaluateModelUseCase
from src.application.usecases.submit_remote_job_usecase import SubmitRemoteJobUseCase
from src.application.usecases.sync_remote_job_usecase import SyncRemoteJobUseCase
from src.application.usecases.train_model_usecase import TrainModelUseCase
from src.application.usecases.train_pipeline import TrainPipelineUsecase
from src.domain.exceptions import DomainError
from src.domain.schemas.compute_schema import RemoteJobStatus
from src.infrastructure.config.yaml_loader import YamlConfigLoader
from src.infrastructure.factories.compute_factory import ComputeProviderFactory
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
    # Remote Cloud Compute Arguments
    parser.add_argument(
        "--remote-submit",
        action="store_true",
        help="Package and dispatch training pipeline to remote cloud GPU (Kaggle Dual-T4)",
    )
    parser.add_argument(
        "--remote-status",
        action="store_true",
        help="Check the status of a remote compute job",
    )
    parser.add_argument(
        "--remote-pull",
        action="store_true",
        help="Download artifacts from a completed remote compute job",
    )
    parser.add_argument(
        "--remote-watch",
        action="store_true",
        help="Poll and stream remote job status until completion, then download output artifacts",
    )
    parser.add_argument(
        "--remote-job-id",
        type=str,
        default=None,
        help="Specific remote job ID (e.g. 'username/kernel-slug'). Defaults to latest job.",
    )
    parser.add_argument(
        "--remote-provider",
        type=str,
        default="kaggle",
        help="Remote compute backend provider ('kaggle')",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    setup_logging(env=args.env, log_level=args.log_level)
    config_path = Path(args.config)

    try:
        # -------------------------------------------------------------
        # Remote Cloud Compute Actions
        # -------------------------------------------------------------
        if args.remote_submit or args.remote_status or args.remote_pull or args.remote_watch:
            provider = ComputeProviderFactory.get_provider(args.remote_provider)

            if args.remote_submit:
                config_loader = YamlConfigLoader(config_path=config_path)
                config = config_loader.load_config()
                submit_usecase = SubmitRemoteJobUseCase(compute_provider=provider)
                res = submit_usecase.execute(
                    experiment_name=config.experiment.name,
                    config_path=config_path,
                )
                print("\n=======================================================")
                print("        REMOTE COMPUTE JOB SUBMISSION SUMMARY          ")
                print("=======================================================")
                print(f"Experiment Name   : {res.job_id}")
                print(f"Compute Backend   : {provider.provider_name}")
                print(f"Remote Job ID     : {res.remote_job_id}")
                print(f"Status            : {res.status.value.upper()}")
                print(f"Receipt Location  : outputs/.remote_jobs/{res.job_id}.json")
                print("-------------------------------------------------------")
                print("Run 'python -m src.main --remote-status' to check progress.")
                print("Run 'python -m src.main --remote-watch' to stream and auto-download.")
                print("=======================================================\n")
                return

            sync_usecase = SyncRemoteJobUseCase(compute_provider=provider)

            if args.remote_status:
                res = sync_usecase.execute(
                    remote_job_id=args.remote_job_id,
                    auto_download=False,
                )
                print("\n=======================================================")
                print("             REMOTE COMPUTE STATUS SUMMARY             ")
                print("=======================================================")
                print(f"Remote Job ID     : {res.remote_job_id}")
                print(f"Compute Backend   : {provider.provider_name}")
                print(f"Status            : {res.status.value.upper()}")
                print("=======================================================\n")
                return

            if args.remote_pull:
                res = sync_usecase.execute(
                    remote_job_id=args.remote_job_id,
                    auto_download=True,
                )
                print("\n=======================================================")
                print("            REMOTE ARTIFACT RETRIEVAL SUMMARY          ")
                print("=======================================================")
                print(f"Remote Job ID     : {res.remote_job_id}")
                print(f"Status            : {res.status.value.upper()}")
                print(f"Downloaded Path   : {res.artifact_path}")
                print("=======================================================\n")
                return

            if args.remote_watch:
                print("\n=======================================================")
                print("            WATCHING REMOTE COMPUTE EXECUTION          ")
                print("=======================================================")
                while True:
                    res = sync_usecase.execute(
                        remote_job_id=args.remote_job_id,
                        auto_download=True,
                    )
                    now_str = datetime.now().strftime("%H:%M:%S")
                    print(f"[{now_str}] Job '{res.remote_job_id}' Status: {res.status.value.upper()}")
                    if res.status in (RemoteJobStatus.COMPLETED, RemoteJobStatus.FAILED):
                        break
                    time.sleep(20)

                if res.status == RemoteJobStatus.COMPLETED:
                    print(f"\n✔ Job completed! Artifacts downloaded to: {res.artifact_path}")
                else:
                    print("\n✖ Job reported failure. Check remote Kaggle logs for details.")
                print("=======================================================\n")
                return

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
            print(f"Task Type         : {result.evaluation.task_type.value}")
            print(f"Evaluation F1     : {result.evaluation.metrics.f1_score:.4f}")
            print(f"Exact Match (EM)  : {result.evaluation.metrics.exact_match:.4f}")
            print(f"ROUGE-L           : {result.evaluation.metrics.rouge_l or 0.0:.4f}")
            if result.evaluation.delta_metrics:
                delta_f1 = result.evaluation.delta_metrics.get("delta_f1_score", 0.0)
                delta_em = result.evaluation.delta_metrics.get("delta_exact_match", 0.0)
                print(f"Delta Uplift (ΔF1): {delta_f1:+.4f}")
                print(f"Delta Uplift (ΔEM): {delta_em:+.4f}")
            if result.evaluation.tpot_ms_p50 is not None:
                print(f"TPOT (P50 Decode) : {result.evaluation.tpot_ms_p50:.2f} ms/token")
            if result.evaluation.ttft_ms_p50 is not None:
                print(f"TTFT (P50 Prefill): {result.evaluation.ttft_ms_p50:.2f} ms")
            if result.evaluation.peak_memory_mb is not None:
                print(f"Peak GPU VRAM     : {result.evaluation.peak_memory_mb:.2f} MB")
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
