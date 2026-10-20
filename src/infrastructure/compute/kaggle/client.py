"""
src/infrastructure/compute/kaggle/client.py
Kaggle API wrapper client for headless remote kernel dispatch and monitoring.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from src.domain.exceptions import RemoteComputeError

logger = logging.getLogger(__name__)


class KaggleApiClient:
    """Wrapper client around the Kaggle CLI / REST API for remote executions."""

    def __init__(
        self,
        username: Optional[str] = None,
        key: Optional[str] = None,
    ) -> None:
        self._username = username or os.getenv("KAGGLE_USERNAME")
        self._key = key or os.getenv("KAGGLE_KEY")

        self._auto_discover_credentials()

        if self._username:
            os.environ["KAGGLE_USERNAME"] = self._username
        if self._key:
            os.environ["KAGGLE_KEY"] = self._key

    def _auto_discover_credentials(self) -> None:
        """Discover Kaggle credentials from ~/.kaggle/kaggle.json if not in env."""
        if not self._username or not self._key:
            kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
            if kaggle_json.exists():
                try:
                    with open(kaggle_json, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if not self._username and "username" in data:
                        self._username = data["username"]
                    if not self._key and "key" in data:
                        self._key = data["key"]
                except Exception as exc:
                    logger.warning("Failed to parse ~/.kaggle/kaggle.json: %s", exc)

    @property
    def username(self) -> Optional[str]:
        return self._username

    def _get_base_cmd(self) -> list[str]:
        """Resolves the executable kaggle command whether running in virtualenv or PATH."""
        # 1. Virtual environment bin check
        venv_kaggle = Path(sys.prefix) / "bin" / "kaggle"
        if venv_kaggle.exists() and os.access(venv_kaggle, os.X_OK):
            return [str(venv_kaggle)]

        # 2. PATH resolution
        kaggle_path = shutil.which("kaggle")
        if kaggle_path:
            return [kaggle_path]

        # 3. Fallback to python module execution
        return [sys.executable, "-m", "kaggle"]

    def push_kernel(self, kernel_dir: Path) -> str:
        """Pushes a kernel package to Kaggle Cloud. Returns normalized remote kernel ID."""
        meta_file = kernel_dir / "kernel-metadata.json"
        if not meta_file.exists():
            raise RemoteComputeError(f"kernel-metadata.json not found in {kernel_dir}")

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
            kernel_id = meta.get("id")

        if not kernel_id:
            raise RemoteComputeError("Missing 'id' in kernel-metadata.json")

        logger.info("Pushing kernel package '%s' to Kaggle Cloud...", kernel_id)
        cmd = self._get_base_cmd() + ["kernels", "push", "-p", str(kernel_dir)]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except Exception as exc:
            raise RemoteComputeError(f"Failed to execute Kaggle CLI push: {exc}") from exc

        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip()
            logger.error("Kaggle kernel push failed: %s", error_msg)
            raise RemoteComputeError(f"Kaggle kernel push failed: {error_msg}")

        logger.info("Successfully pushed kernel '%s': %s", kernel_id, result.stdout.strip())

        # Check if Kaggle normalized the slug in the output response URL
        match = re.search(r"kaggle\.com/code/([a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-]+)", result.stdout)
        if match:
            actual_id = match.group(1).strip()
            logger.info("Resolved remote Kaggle ID: %s", actual_id)
            return actual_id

        return kernel_id

    def get_status(self, kernel_id: str) -> str:
        """
        Queries status of remote Kaggle kernel.
        Returns: 'queued', 'running', 'complete', 'error', or 'unknown'.
        """
        cmd = self._get_base_cmd() + ["kernels", "status", kernel_id]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except Exception as exc:
            logger.warning("Failed to invoke kaggle status for '%s': %s", kernel_id, exc)
            return "unknown"

        if result.returncode != 0:
            logger.warning(
                "Kaggle CLI status returned error code %d for '%s': %s",
                result.returncode,
                kernel_id,
                result.stderr.strip(),
            )
            return "unknown"

        output = result.stdout.lower()
        if "complete" in output:
            return "complete"
        if "running" in output:
            return "running"
        if "queued" in output:
            return "queued"
        if "error" in output:
            return "error"

        return "unknown"

    def download_output(self, kernel_id: str, destination_dir: Path) -> Path:
        """Downloads remote kernel outputs into destination_dir."""
        destination_dir.mkdir(parents=True, exist_ok=True)
        cmd = self._get_base_cmd() + [
            "kernels",
            "output",
            kernel_id,
            "-p",
            str(destination_dir),
        ]

        logger.info("Downloading Kaggle kernel outputs for '%s' to %s...", kernel_id, destination_dir)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except Exception as exc:
            raise RemoteComputeError(f"Failed to execute Kaggle output download: {exc}") from exc

        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip()
            logger.error("Failed to download output for '%s': %s", kernel_id, error_msg)
            raise RemoteComputeError(f"Failed to download Kaggle output: {error_msg}")

        logger.info("Successfully retrieved output artifacts for '%s'.", kernel_id)
        return destination_dir
