"""
src/presentation/handlers.py
Command and action handlers for the CLI presentation interface.
Coordinates application use cases with presentation rendering.
"""

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

from src.domain.schemas.compute_schema import RemoteJobStatus
from src.domain.schemas.config_schema import ExperimentConfig
from src.domain.schemas.evaluation_schema import EvaluationResult 
from src.infrastructure.config.yaml_loader import YamlConfigLoader
from src.presentation.composition_root import (
    build_merge_usecase,
    build_remote_usecases,
    build_train_pipeline,
    build_register_usecase,
)
from src.presentation.presenter import ConsolePresenter


def is_remote_action(args: argparse.Namespace) -> bool:
    """Return True if any remote compute flag is specified."""
    return bool(
        args.remote_submit
        or args.remote_status
        or args.remote_pull
        or args.remote_watch
    )


def handle_remote_action(
    args: argparse.Namespace,
    config_path: Path,
    presenter: ConsolePresenter | None = None,
    poll_interval: int = 20,
) -> int:
    """
    Handle remote compute workflows: submit, status, pull, or watch.
    Returns 0 on success, 1 on job failure.
    """
    presenter = presenter or ConsolePresenter()
    submit_usecase, sync_usecase, provider = build_remote_usecases(args.remote_provider)

    if args.remote_submit:
        config_loader = YamlConfigLoader(config_path=config_path)
        config = config_loader.load_config()
        res = submit_usecase.execute(
            experiment_name=config.experiment.name,
            config_path=config_path,
        )
        presenter.render_remote_submission(res, provider.provider_name)
        return 0

    if args.remote_status:
        res = sync_usecase.execute(
            remote_job_id=args.remote_job_id,
            auto_download=False,
        )
        presenter.render_remote_status(res, provider.provider_name)
        return 0

    if args.remote_pull:
        res = sync_usecase.execute(
            remote_job_id=args.remote_job_id,
            auto_download=True,
        )
        presenter.render_remote_pull(res)
        return 0

    if args.remote_watch:
        presenter.render_remote_watch_header()
        while True:
            res = sync_usecase.execute(
                remote_job_id=args.remote_job_id,
                auto_download=True,
            )
            now_str = datetime.now(UTC).strftime("%H:%M:%S UTC")
            presenter.render_remote_watch_tick(
                now_str, res.remote_job_id, res.status.value
            )
            if res.status in (RemoteJobStatus.COMPLETED, RemoteJobStatus.FAILED):
                break
            time.sleep(poll_interval)

        presenter.render_remote_watch_completion(res)
        return 0 if res.status == RemoteJobStatus.COMPLETED else 1

    return 0


def handle_merge_action(
    config: ExperimentConfig,
    presenter: ConsolePresenter | None = None,
) -> int:
    """Handle standalone model merging and weights unloading."""
    merge_usecase = build_merge_usecase(config=config)
    merge_usecase.execute()
    return 0

def handle_model_registry_action(
    config: ExperimentConfig,
    presenter: ConsolePresenter | None = None,
) -> int:
    """Handle standalone model registration and publishing to model registry."""

    presenter = presenter or ConsolePresenter()
    registry_usecase = build_register_usecase(config=config)

    eval_result = None
    eval_file = Path(config.artifact.output_dir) / "evaluation_report.json"
    if eval_file.exists():
        eval_result = EvaluationResult.model_validate_json(eval_file.read_text(encoding="utf-8"))

    published_uri = registry_usecase.execute(
        artifact_path=Path(config.artifact.output_dir),
        config=config,
        evaluation_result=eval_result,
    )

    presenter.render_registry_summary(
        model_name=config.registry.model_name,
        published_uri=published_uri,
        provider_name=config.registry.registry_type,
    )

    return 0

def handle_pipeline_action(
    config: ExperimentConfig,
    config_loader: YamlConfigLoader,
    eval_only: bool,
    presenter: ConsolePresenter | None = None,
) -> int:
    """Handle local training and/or evaluation pipeline execution."""
    presenter = presenter or ConsolePresenter()
    pipeline = build_train_pipeline(config=config, config_loader=config_loader)
    result = pipeline.run(eval_only=eval_only)
    presenter.render_pipeline_summary(result)
    return 0
