"""
tests/unit/test_monitoring.py
Unit tests verifying Prometheus exporter, PrometheusCallback, WandbTracker, and MonitoringCallbackFactory.
"""

import unittest
from unittest.mock import MagicMock, patch

from transformers import TrainerControl, TrainerState, TrainingArguments

from src.domain.schemas.config_schema import MonitoringConfig, TelemetryConfig
from src.infrastructure.monitoring.callback_factory import MonitoringCallbackFactory
from src.infrastructure.monitoring.prometheus_callback import PrometheusCallback
from src.infrastructure.monitoring.prometheus_exporter import (
    PROM_EPOCH,
    PROM_LEARNING_RATE,
    PROM_TRAINING_LOSS,
    PROM_TRAINING_STEP,
    PrometheusMetricsServer,
)
from src.infrastructure.monitoring.wandb_tracker import WandbTracker


class TestMonitoring(unittest.TestCase):
    """Unit tests for enterprise monitoring components."""

    def test_prometheus_callback_on_log_updates_gauges(self) -> None:
        callback = PrometheusCallback()
        dummy_args = MagicMock(spec=TrainingArguments)
        dummy_state = MagicMock(spec=TrainerState)
        dummy_state.global_step = 42
        dummy_state.epoch = 2.5
        dummy_control = MagicMock(spec=TrainerControl)

        logs = {"loss": 0.314, "learning_rate": 0.0001}
        callback.on_log(args=dummy_args, state=dummy_state, control=dummy_control, logs=logs)

        self.assertAlmostEqual(PROM_TRAINING_LOSS._value.get(), 0.314, places=3)
        self.assertAlmostEqual(PROM_LEARNING_RATE._value.get(), 0.0001, places=5)
        self.assertEqual(PROM_TRAINING_STEP._value.get(), 42)
        self.assertAlmostEqual(PROM_EPOCH._value.get(), 2.5, places=1)

    def test_wandb_tracker_disabled_returns_false(self) -> None:
        initialized = WandbTracker.init(project="test", enabled=False)
        self.assertFalse(initialized)
        WandbTracker.finish()

    def test_monitoring_callback_factory_creates_prometheus_callback(self) -> None:
        monitoring_config = MonitoringConfig(
            telemetry=TelemetryConfig(
                enable_prometheus=True,
                prometheus_port=8000,
                enable_wandb=False,
            )
        )

        with patch.object(PrometheusMetricsServer, "start", return_value=True):
            callbacks = MonitoringCallbackFactory.create_callbacks(
                monitoring_config=monitoring_config,
                experiment_name="test-exp",
            )
            self.assertEqual(len(callbacks), 1)
            self.assertIsInstance(callbacks[0], PrometheusCallback)

    def test_monitoring_callback_factory_returns_empty_when_telemetry_disabled(self) -> None:
        monitoring_config = MonitoringConfig(
            telemetry=TelemetryConfig(
                enable_prometheus=False,
                enable_wandb=False,
            )
        )
        callbacks = MonitoringCallbackFactory.create_callbacks(
            monitoring_config=monitoring_config,
        )
        self.assertEqual(len(callbacks), 0)


if __name__ == "__main__":
    unittest.main()
