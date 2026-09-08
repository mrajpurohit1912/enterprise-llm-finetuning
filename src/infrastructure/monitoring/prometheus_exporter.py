"""
src/infrastructure/monitoring/prometheus_exporter.py
Prometheus metrics registry and HTTP exporter server.
"""

import logging
import socket
from typing import Optional
from prometheus_client import Gauge, start_http_server

logger = logging.getLogger(__name__)

# Enterprise Prometheus Gauges for tracking ML training & hardware metrics
PROM_TRAINING_LOSS = Gauge("llm_training_loss", "Current fine-tuning step loss")
PROM_LEARNING_RATE = Gauge("llm_learning_rate", "Current optimizer learning rate")
PROM_TRAINING_STEP = Gauge("llm_training_step", "Current global training step")
PROM_EPOCH = Gauge("llm_epoch", "Current fine-tuning epoch")
PROM_GPU_VRAM_ALLOCATED_MB = Gauge("llm_gpu_vram_allocated_mb", "GPU VRAM currently allocated in MB")
PROM_GPU_VRAM_RESERVED_MB = Gauge("llm_gpu_vram_reserved_mb", "GPU VRAM currently reserved in MB")


class PrometheusMetricsServer:
    """Manages the lifecycle of the Prometheus HTTP metrics scrape server."""

    _is_running: bool = False
    _active_port: Optional[int] = None

    @classmethod
    def is_port_in_use(cls, port: int) -> bool:
        """Check if target port is already bound by another process or existing instance."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            result = sock.connect_ex(("127.0.0.1", port))
            return result == 0

    @classmethod
    def start(cls, port: int = 8000) -> bool:
        """
        Start the Prometheus HTTP metrics exporter server in a daemon thread.

        Args:
            port: Port number for the metrics endpoint (default: 8000).

        Returns:
            bool: True if server is running or already started, False if port is unavailable.
        """
        if cls._is_running and cls._active_port == port:
            logger.info("Prometheus metrics server is already running on port %d.", port)
            return True

        if cls.is_port_in_use(port):
            logger.warning(
                "Port %d is already in use. Assuming external Prometheus exporter is active.",
                port,
            )
            cls._is_running = True
            cls._active_port = port
            return True

        try:
            start_http_server(port)
            cls._is_running = True
            cls._active_port = port
            logger.info("Prometheus metrics exporter started on http://0.0.0.0:%d/metrics", port)
            return True
        except Exception as exc:
            logger.error("Failed to start Prometheus metrics exporter on port %d: %s", port, exc)
            return False
