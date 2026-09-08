"""
src/infrastructure/logging/logging_config.py
Enterprise structured logging configuration using structlog and Python standard logging.
Provides unified JSON output for production and colorized output for local development,
with automatic secret masking and ML execution context injection.
"""

import logging
import os
import sys
from typing import Any, Dict

import structlog

from src.infrastructure.logging.logging_context import get_logging_context

# Sensitive keys to redact in log records
SENSITIVE_KEY_SUBSTRINGS = {
    "password",
    "secret",
    "token",
    "hf_token",
    "wandb_api_key",
    "aws_secret_access_key",
    "aws_access_key_id",
    "api_key",
    "auth_token",
    "jwt",
    "credentials",
    "private_key",
}


def secret_masker_processor(
    logger: Any, method_name: str, event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Masks sensitive values (tokens, credentials, API keys) to prevent accidental
    leakage in stdout, disk logs, or log collectors.
    """
    for key in list(event_dict.keys()):
        key_lower = key.lower()
        if any(sub in key_lower for sub in SENSITIVE_KEY_SUBSTRINGS):
            event_dict[key] = "********"
    return event_dict


def inject_context_processor(
    logger: Any, method_name: str, event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Injects thread-safe and async-safe context variables (e.g. experiment_name,
    model_id, dataset_name, step) into every emitted log event.
    """
    context = get_logging_context()
    for k, v in context.items():
        if k not in event_dict:
            event_dict[k] = v
    return event_dict


def setup_logging(env: str = "development", log_level: str = "INFO") -> None:
    """
    Configures Python standard logging and structlog to unify log formatting.
    
    Args:
        env: Environment name ('development' for colorized console output, 'production' for structured JSON).
        log_level: Minimum logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Core shared processors run across all log entries (standard logging + structlog)
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        inject_context_processor,
        secret_masker_processor,
    ]

    is_production = env.lower() == "production"

    if is_production:
        formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.dict_tracebacks,
                structlog.processors.JSONRenderer(),
            ],
        )
    else:
        formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.dev.ConsoleRenderer(colors=True),
            ],
        )

    # Configure stdout stream handler for root logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    # Remove existing handlers to avoid duplicate log lines
    for existing_h in root_logger.handlers[:]:
        root_logger.removeHandler(existing_h)
    root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)

    # Configure structlog to route through standard logging
    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Suppress verbose third-party HTTP & telemetry loggers
    noisy_loggers = [
        "httpx",
        "httpcore",
        "urllib3",
        "filelock",
        "datasets",
        "fsspec",
    ]
    for noisy in noisy_loggers:
        logging.getLogger(noisy).setLevel(logging.WARNING)
