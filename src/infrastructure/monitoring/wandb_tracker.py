"""
src/infrastructure/monitoring/wandb_tracker.py
Weights & Biases experiment tracking session manager.
"""

import logging
from typing import Any, Dict, Optional

from src.domain.schemas.evaluation_schema import EvaluationResult

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
        try:
            import wandb

            if cls._is_active or wandb.run is not None:
                wandb.finish()
                logger.info("Weights & Biases run closed cleanly.")
        except Exception as exc:
            logger.debug("W&B finish encountered an error: %s", exc)
        finally:
            cls._is_active = False

    @classmethod
    def log_evaluation_summary(cls, eval_result: EvaluationResult) -> None:
        """
        Log comprehensive evaluation scorecard and interactive sample breakdown table to Weights & Biases.
        """
        try:
            import wandb

            if not cls._is_active and wandb.run is None:
                logger.debug("W&B is not active; skipping evaluation summary logging.")
                return

            summary_payload: Dict[str, Any] = {
                "eval/exact_match": eval_result.metrics.exact_match,
                "eval/f1_score": eval_result.metrics.f1_score,
                "eval/rouge_l": eval_result.metrics.rouge_l or 0.0,
                "eval/latency_per_sample_ms": eval_result.latency_per_sample_ms,
                "eval/throughput_tokens_per_sec": eval_result.throughput_tokens_per_sec,
                "eval/passed_quality_gate": eval_result.passed_quality_gate,
            }

            if eval_result.metrics.tolerance_accuracy_5pct is not None:
                summary_payload["eval/tolerance_acc_5pct"] = eval_result.metrics.tolerance_accuracy_5pct
            if eval_result.metrics.tolerance_accuracy_1pct is not None:
                summary_payload["eval/strict_acc_1pct"] = eval_result.metrics.tolerance_accuracy_1pct
            if eval_result.metrics.sign_accuracy is not None:
                summary_payload["eval/sign_accuracy"] = eval_result.metrics.sign_accuracy
            if eval_result.metrics.parseable_number_rate is not None:
                summary_payload["eval/parseable_rate"] = eval_result.metrics.parseable_number_rate
            if eval_result.metrics.median_relative_error is not None:
                summary_payload["eval/median_rel_error"] = eval_result.metrics.median_relative_error
            if eval_result.metrics.mean_absolute_percentage_error is not None:
                summary_payload["eval/mean_abs_pct_error"] = eval_result.metrics.mean_absolute_percentage_error
            if eval_result.ttft_ms_p50 is not None:
                summary_payload["eval/ttft_ms_p50"] = eval_result.ttft_ms_p50
            if eval_result.ttft_ms_p95 is not None:
                summary_payload["eval/ttft_ms_p95"] = eval_result.ttft_ms_p95
            if eval_result.tpot_ms_p50 is not None:
                summary_payload["eval/tpot_ms_p50"] = eval_result.tpot_ms_p50
            if eval_result.tpot_ms_p95 is not None:
                summary_payload["eval/tpot_ms_p95"] = eval_result.tpot_ms_p95
            if eval_result.peak_memory_mb is not None:
                summary_payload["eval/peak_vram_mb"] = eval_result.peak_memory_mb

            # Delta metrics (Uplift vs base baseline)
            if eval_result.delta_metrics:
                for k, v in eval_result.delta_metrics.items():
                    summary_payload[f"eval/{k}"] = v

            # Error distribution buckets if present in custom_metrics
            if eval_result.metrics.custom_metrics:
                for k, v in eval_result.metrics.custom_metrics.items():
                    summary_payload[f"eval/distribution/{k}"] = v

            # Build Interactive Sample-Level Failure Breakdown Table
            if eval_result.sample_details:
                columns = [
                    "sample_idx",
                    "status",
                    "ground_truth",
                    "prediction",
                    "parsed_ground_truth",
                    "parsed_prediction",
                    "rel_error_pct",
                    "exact_match",
                    "f1_score",
                    "prompt_snippet",
                ]
                table = wandb.Table(columns=columns)

                for idx, sample in enumerate(eval_result.sample_details, 1):
                    # Classify sample status for visual tagging
                    if sample.within_tolerance:
                        if sample.relative_error is not None and sample.relative_error <= 0.01:
                            status = "🟢 Within 1%"
                        else:
                            status = "🟡 1% - 5%"
                    elif sample.parsed_prediction is None:
                        status = "❌ Unparseable"
                    elif (sample.relative_error or 0.0) <= 0.20:
                        status = "🟠 5% - 20%"
                    else:
                        status = "🔴 Over 20%"

                    rel_err = round(sample.relative_error * 100, 2) if sample.relative_error is not None else None
                    prompt_snippet = sample.prompt[-300:] if len(sample.prompt) > 300 else sample.prompt

                    table.add_data(
                        idx,
                        status,
                        sample.ground_truth,
                        sample.prediction,
                        sample.parsed_ground_truth,
                        sample.parsed_prediction,
                        rel_err,
                        sample.exact_match,
                        round(sample.f1_score, 4),
                        prompt_snippet,
                    )

                summary_payload["eval/samples_breakdown_table"] = table

            wandb.log(summary_payload)
            logger.info("Evaluation scorecard and interactive sample table logged to Weights & Biases successfully.")
        except Exception as exc:
            logger.warning("Failed to log evaluation summary to W&B: %s", exc)
