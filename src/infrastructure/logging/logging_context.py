"""
src/infrastructure/logging/logging_context.py
Context-variable based metadata management for enterprise structured logging.
Thread-safe and async-safe context propagation across pipeline stages.
"""

import contextvars
from typing import Any, Dict

# ContextVar keeps variables scoped to the current execution thread / coroutine context.
_logging_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "llm_logging_context", default={}
)


def set_logging_context(**kwargs: Any) -> None:
    """
    Sets or updates contextual metadata for the current execution scope
    (e.g., experiment_name, model_id, dataset_name, step, device).
    """
    ctx = _logging_context.get().copy()
    ctx.update(kwargs)
    _logging_context.set(ctx)


def get_logging_context() -> Dict[str, Any]:
    """Retrieves the active logging context dictionary."""
    return _logging_context.get()


def clear_logging_context() -> None:
    """Clears all contextual metadata from the active scope."""
    _logging_context.set({})
