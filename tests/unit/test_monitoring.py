"""
tests/unit/test_monitoring.py
Unit tests verifying Prometheus exporter, PrometheusCallback, WandbTracker, and MonitoringCallbackFactory.
"""

import unittest
from unittest.mock import MagicMock, patch

from transformers import TrainerControl, TrainerState, TrainingArguments

from src.domain.schemas.config_schema import MonitoringConfig, TelemetryConfig
from src.domain.schemas.evaluation_schema import (
    EvaluationResult,
    EvaluationSample,
    MetricScores,
    TaskType,
)
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

    def test_wandb_tracker_log_evaluation_summary(self) -> None:
        eval_result = EvaluationResult(
            experiment_name="test-exp",
            model_id="test-model",
            task_type=TaskType.TEXT_GENERATION,
            eval_split="test",
            total_samples=2,
            metrics=MetricScores(
                exact_match=0.5,
                f1_score=0.5,
                tolerance_accuracy_5pct=0.5,
                tolerance_accuracy_1pct=0.5,
                sign_accuracy=1.0,
                parseable_number_rate=1.0,
                median_relative_error=0.02,
                mean_absolute_percentage_error=2.0,
            ),
            latency_per_sample_ms=100.0,
            throughput_tokens_per_sec=10.0,
            passed_quality_gate=True,
            sample_details=[
                EvaluationSample(
                    prompt="What is 1+1?",
                    ground_truth="2.0",
                    prediction="2.0",
                    exact_match=True,
                    f1_score=1.0,
                    parsed_ground_truth=2.0,
                    parsed_prediction=2.0,
                    relative_error=0.0,
                    within_tolerance=True,
                ),
                EvaluationSample(
                    prompt="What is 5*5?",
                    ground_truth="25.0",
                    prediction="35.0",
                    exact_match=False,
                    f1_score=0.0,
                    parsed_ground_truth=25.0,
                    parsed_prediction=35.0,
                    relative_error=0.40,
                    within_tolerance=False,
                ),
            ],
            ttft_ms_p50=50.0,
            tpot_ms_p50=20.0,
            peak_memory_mb=1024.0,
        )

        with patch("sys.modules", {**unittest.mock.MagicMock().__dict__}):
            pass

        with patch("wandb.init"), patch("wandb.log") as mock_log, patch("wandb.Table") as mock_table:
            WandbTracker._is_active = True
            mock_table_instance = MagicMock()
            mock_table.return_value = mock_table_instance

            WandbTracker.log_evaluation_summary(eval_result)

            self.assertTrue(mock_log.called)
            logged_payload = mock_log.call_args[0][0]
            self.assertEqual(logged_payload["eval/exact_match"], 0.5)
            self.assertEqual(logged_payload["eval/tolerance_acc_5pct"], 0.5)
            self.assertIn("eval/samples_breakdown_table", logged_payload)
            self.assertEqual(mock_table_instance.add_data.call_count, 2)
            WandbTracker._is_active = False



if __name__ == "__main__":
    unittest.main()
