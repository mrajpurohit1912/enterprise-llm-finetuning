"""
src/infrastructure/monitoring/prometheus_callback.py
Hugging Face Trainer callback exporting real-time training and hardware metrics to Prometheus.
"""

import logging
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from src.infrastructure.monitoring.prometheus_exporter import (
    PROM_EPOCH,
    PROM_GPU_VRAM_ALLOCATED_MB,
    PROM_GPU_VRAM_RESERVED_MB,
    PROM_LEARNING_RATE,
    PROM_TRAINING_LOSS,
    PROM_TRAINING_STEP,
)

logger = logging.getLogger(__name__)


class PrometheusCallback(TrainerCallback):
    """
    Hugging Face Trainer callback updating Prometheus metrics gauges during training steps.
    """

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict = None,
        **kwargs,
    ) -> None:
        """Invoked on each training log interval to record loss and hardware metrics."""
        if logs is None:
            return

        # 1. Update ML Training Metrics
        if "loss" in logs:
            try:
                PROM_TRAINING_LOSS.set(float(logs["loss"]))
            except (ValueError, TypeError):
                pass

        if "learning_rate" in logs:
            try:
                PROM_LEARNING_RATE.set(float(logs["learning_rate"]))
            except (ValueError, TypeError):
                pass

        if state.global_step is not None:
            PROM_TRAINING_STEP.set(state.global_step)

        if state.epoch is not None:
            PROM_EPOCH.set(float(state.epoch))

        # 2. Update GPU Hardware VRAM Metrics (if CUDA available)
        if torch.cuda.is_available():
            try:
                allocated_mb = torch.cuda.memory_allocated() / (1024**2)
                reserved_mb = torch.cuda.memory_reserved() / (1024**2)
                PROM_GPU_VRAM_ALLOCATED_MB.set(round(allocated_mb, 2))
                PROM_GPU_VRAM_RESERVED_MB.set(round(reserved_mb, 2))
            except Exception as exc:
                logger.debug("Failed to read CUDA memory telemetry: %s", exc)
