"""
src/presentation/__init__.py
Presentation Layer package.
"""

from typing import Any

__all__ = [
    "ConsolePresenter",
    "build_merge_usecase",
    "build_remote_usecases",
    "build_train_pipeline",
    "main",
    "parse_args",
    "run_cli",
]


def __getattr__(name: str) -> Any:
    if name in ("main", "parse_args", "run_cli"):
        from src.presentation.cli import main, parse_args, run_cli

        items = {"main": main, "parse_args": parse_args, "run_cli": run_cli}
        return items[name]

    if name == "ConsolePresenter":
        from src.presentation.presenter import ConsolePresenter

        return ConsolePresenter

    if name in ("build_merge_usecase", "build_remote_usecases", "build_train_pipeline"):
        from src.presentation.composition_root import (
            build_merge_usecase,
            build_remote_usecases,
            build_train_pipeline,
        )

        items = {
            "build_merge_usecase": build_merge_usecase,
            "build_remote_usecases": build_remote_usecases,
            "build_train_pipeline": build_train_pipeline,
        }
        return items[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
