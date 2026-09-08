"""
tests/unit/test_logging.py
Unit tests for enterprise logging configuration, secret masking, and context propagation.
"""

import io
import json
import logging
import unittest
from unittest.mock import patch

import structlog

from src.infrastructure.logging.logging_config import (
    inject_context_processor,
    secret_masker_processor,
    setup_logging,
)
from src.infrastructure.logging.logging_context import (
    clear_logging_context,
    get_logging_context,
    set_logging_context,
)


class TestLoggingContext(unittest.TestCase):
    def setUp(self) -> None:
        clear_logging_context()

    def tearDown(self) -> None:
        clear_logging_context()

    def test_set_get_and_clear_context(self) -> None:
        self.assertEqual(get_logging_context(), {})

        set_logging_context(
            experiment_name="qwen-finetune",
            model_id="Qwen/Qwen2.5-0.5B",
            global_step=42,
        )

        ctx = get_logging_context()
        self.assertEqual(ctx["experiment_name"], "qwen-finetune")
        self.assertEqual(ctx["model_id"], "Qwen/Qwen2.5-0.5B")
        self.assertEqual(ctx["global_step"], 42)

        # Update additional context
        set_logging_context(device="cuda:0")
        updated_ctx = get_logging_context()
        self.assertEqual(updated_ctx["device"], "cuda:0")
        self.assertEqual(updated_ctx["experiment_name"], "qwen-finetune")

        # Clear
        clear_logging_context()
        self.assertEqual(get_logging_context(), {})


class TestLoggingProcessors(unittest.TestCase):
    def setUp(self) -> None:
        clear_logging_context()

    def tearDown(self) -> None:
        clear_logging_context()

    def test_secret_masker_processor_redacts_sensitive_keys(self) -> None:
        event = {
            "event": "Authentication initiated",
            "hf_token": "hf_1234567890abcdef",
            "wandb_api_key": "abc123wandbkey999",
            "aws_secret_access_key": "topsecretkey",
            "password": "mypassword",
            "jwt": "header.payload.signature",
            "learning_rate": 0.0002,
            "experiment_name": "qwen2.5-0.5b",
        }

        processed = secret_masker_processor(None, "info", event)

        self.assertEqual(processed["hf_token"], "********")
        self.assertEqual(processed["wandb_api_key"], "********")
        self.assertEqual(processed["aws_secret_access_key"], "********")
        self.assertEqual(processed["password"], "********")
        self.assertEqual(processed["jwt"], "********")
        # Non-sensitive keys remain unchanged
        self.assertEqual(processed["learning_rate"], 0.0002)
        self.assertEqual(processed["experiment_name"], "qwen2.5-0.5b")
        self.assertEqual(processed["event"], "Authentication initiated")

    def test_inject_context_processor_merges_context(self) -> None:
        set_logging_context(
            experiment_name="test-run",
            dataset_name="databricks/officeqa",
        )

        event = {"event": "Batch finished", "loss": 0.45}
        processed = inject_context_processor(None, "info", event)

        self.assertEqual(processed["experiment_name"], "test-run")
        self.assertEqual(processed["dataset_name"], "databricks/officeqa")
        self.assertEqual(processed["loss"], 0.45)

    def test_inject_context_processor_does_not_overwrite_explicit_keys(self) -> None:
        set_logging_context(experiment_name="context-exp")

        event = {"event": "Step done", "experiment_name": "explicit-exp"}
        processed = inject_context_processor(None, "info", event)

        self.assertEqual(processed["experiment_name"], "explicit-exp")


class TestSetupLogging(unittest.TestCase):
    def setUp(self) -> None:
        clear_logging_context()

    def tearDown(self) -> None:
        clear_logging_context()

    def test_setup_logging_production_json(self) -> None:
        setup_logging(env="production", log_level="INFO")
        root = logging.getLogger()
        self.assertEqual(root.level, logging.INFO)
        self.assertTrue(len(root.handlers) >= 1)

    def test_setup_logging_development_console(self) -> None:
        setup_logging(env="development", log_level="DEBUG")
        root = logging.getLogger()
        self.assertEqual(root.level, logging.DEBUG)
        self.assertTrue(len(root.handlers) >= 1)


if __name__ == "__main__":
    unittest.main()
