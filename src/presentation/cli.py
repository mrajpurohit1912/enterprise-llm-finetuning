"""
src/presentation/cli.py
Command-line presentation interface for triggering fine-tuning pipelines.
Ultra-lean entrypoint delegating parsing, composition, and execution to modular components.
"""

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from src.domain.exceptions import DomainError
from src.infrastructure.config.yaml_loader import YamlConfigLoader
from src.infrastructure.logging import setup_logging
from src.presentation.handlers import (
    handle_merge_action,
    handle_pipeline_action,
    handle_remote_action,
    is_remote_action,
)
from src.presentation.presenter import ConsolePresenter


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments and flags."""
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
    parser.add_argument(
        "--merging-only",
        action="store_true",
        help="only for merging and saving the model",
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
    return parser.parse_args(args)


def run_cli(
    argv: Sequence[str] | None = None,
    presenter: ConsolePresenter | None = None,
) -> int:
    """
    Main CLI controller executing argument parsing, dispatching, and error handling.
    Returns exit code (0 for success, non-zero for errors).
    """
    presenter = presenter or ConsolePresenter()
    load_dotenv()
    args = parse_args(argv)
    setup_logging(env=args.env, log_level=args.log_level)
    config_path = Path(args.config)

    try:
        if is_remote_action(args):
            return handle_remote_action(args, config_path, presenter=presenter)

        config_loader = YamlConfigLoader(config_path=config_path)
        config = config_loader.load_config()

        if args.merging_only:
            return handle_merge_action(config, presenter=presenter)

        return handle_pipeline_action(
            config=config,
            config_loader=config_loader,
            eval_only=args.eval_only,
            presenter=presenter,
        )

    except KeyboardInterrupt:
        presenter.render_error("Aborted", "Execution cancelled by user.")
        return 130
    except DomainError as err:
        presenter.render_error("Domain Error", str(err))
        return 1
    except Exception as exc:  # noqa: BLE001
        presenter.render_error("Fatal Error", f"Unexpected failure: {exc}")
        return 1


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point wrapper invoked by console script and python -m."""
    exit_code = run_cli(argv)
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
