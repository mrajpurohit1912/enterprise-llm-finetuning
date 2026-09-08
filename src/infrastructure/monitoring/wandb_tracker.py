"""
src/infrastructure/monitoring/wandb_tracker.py
Weights & Biases experiment tracking session manager.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class WandbTracker:
    """Manages the lifecycle of Weights & Biases experiment tracking sessions."""

    _is_active: bool = False

    @classmethod
    def init(
        cls,
        project: str,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ) -> bool:
        """
        Initialize a W&B experiment run if enabled.

        Args:
            project: W&B project name.
            run_name: Optional experiment run name.
            config: Experiment hyperparameters to log.
            enabled: Toggle to enable/disable W&B tracking.

        Returns:
            bool: True if W&B was initialized successfully, False otherwise.
        """
        if not enabled:
            return False

        try:
            import wandb

            wandb.init(
                project=project,
                name=run_name,
                config=config or {},
            )
            cls._is_active = True
            logger.info("Weights & Biases initialized for project '%s' (run: '%s').", project, run_name)
            return True
        except Exception as exc:
            logger.warning(
                "W&B initialization failed (proceeding without W&B): %s. "
                "Ensure WANDB_API_KEY is set or run 'wandb login'.",
                exc,
            )
            cls._is_active = False
            return False

    @classmethod
    def finish(cls) -> None:
        """Finish the active W&B run cleanly."""
        if cls._is_active:
            try:
                import wandb

                wandb.finish()
                logger.info("Weights & Biases run closed cleanly.")
            except Exception as exc:
                logger.debug("W&B finish encountered an error: %s", exc)
            finally:
                cls._is_active = False
