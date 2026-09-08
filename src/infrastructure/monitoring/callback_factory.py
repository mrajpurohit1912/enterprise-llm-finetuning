"""
src/infrastructure/monitoring/callback_factory.py
Factory preparing and wiring active monitoring callbacks based on MonitoringConfig.
"""

import logging
from typing import Any, List, Optional
from transformers import TrainerCallback

from src.domain.schemas.config_schema import MonitoringConfig
from src.infrastructure.monitoring.prometheus_callback import PrometheusCallback
from src.infrastructure.monitoring.prometheus_exporter import PrometheusMetricsServer
from src.infrastructure.monitoring.wandb_tracker import WandbTracker

logger = logging.getLogger(__name__)


class MonitoringCallbackFactory:
    """Factory creating and initializing monitoring callbacks based on domain configuration."""

    @staticmethod
    def create_callbacks(
        monitoring_config: Optional[MonitoringConfig] = None,
        experiment_name: Optional[str] = None,
        hyperparameters: Optional[dict] = None,
    ) -> List[TrainerCallback]:
        """
        Instantiate active TrainerCallback instances and start telemetry daemons.

        Args:
            monitoring_config: Validated MonitoringConfig domain entity.
            experiment_name: Name of the active experiment.
            hyperparameters: Dict of experiment hyperparameters to log to trackers.

        Returns:
            List[TrainerCallback]: Assembled list of callbacks for the trainer.
        """
        callbacks: List[TrainerCallback] = []

        if monitoring_config is None or monitoring_config.telemetry is None:
            return callbacks

        telemetry = monitoring_config.telemetry

        # 1. Initialize Prometheus Exporter & Callback
        if telemetry.enable_prometheus:
            port = telemetry.prometheus_port
            logger.info("Enabling Prometheus metrics telemetry on port %d...", port)
            server_started = PrometheusMetricsServer.start(port=port)
            if server_started:
                callbacks.append(PrometheusCallback())
                logger.info("PrometheusCallback registered with trainer.")

        # 2. Initialize Weights & Biases Tracker
        if telemetry.enable_wandb:
            logger.info("Enabling Weights & Biases experiment tracking for '%s'...", telemetry.wandb_project)
            WandbTracker.init(
                project=telemetry.wandb_project,
                run_name=experiment_name,
                config=hyperparameters,
                enabled=True,
            )

        return callbacks
