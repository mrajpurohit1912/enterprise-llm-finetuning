"""
src/application/usecases/evaluate_model_usecase.py
Atomic application use case orchestrating model evaluation, scorecard persistence, and quality gates.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.domain.exceptions import ModelEvaluationError
from src.domain.interfaces.evaluator import EvaluatorBase
from src.domain.schemas.evaluation_schema import (
    EvaluationConfig,
    EvaluationResult,
    MetricScores,
)

logger = logging.getLogger(__name__)


class EvaluateModelUseCase:
    """
    Orchestrates the evaluation stage for fine-tuned or baseline LLMs.
    Computes comparative delta metrics and persists scorecards to disk.
    """

    def __init__(self, evaluator: EvaluatorBase) -> None:
        self.evaluator = evaluator

    def execute(
        self,
        model: Any,
        tokenizer: Any,
        eval_dataset: Any,
        eval_config: EvaluationConfig,
        experiment_name: str,
        model_id: str,
        output_dir: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Execute evaluation, compute comparative delta (if enabled), and persist scorecards.
        """
        logger.info(
            "Executing EvaluateModelUseCase for experiment '%s' (split: '%s', task: '%s')...",
            experiment_name,
            eval_config.eval_split,
            eval_config.task_type.value,
        )

        try:
            base_metrics: Optional[MetricScores] = None
            delta_metrics: Optional[Dict[str, float]] = None

            # ------------------------------------------------------------------
            # 1. Optional Comparative Baseline Evaluation (Unadapted Base Model)
            # ------------------------------------------------------------------
            if eval_config.compare_with_base_model:
                if hasattr(model, "disable_adapter"):
                    logger.info("Evaluating unadapted Base Model baseline (adapters disabled)...")
                    with model.disable_adapter():
                        base_result = self.evaluator.evaluate(
                            model=model,
                            tokenizer=tokenizer,
                            eval_dataset=eval_dataset,
                            eval_config=eval_config,
                            experiment_name=experiment_name,
                            model_id=f"{model_id} (Base Baseline)",
                        )
                    base_metrics = base_result.metrics
                    logger.info(
                        "Base Model Baseline: EM=%.4f, F1=%.4f",
                        base_metrics.exact_match,
                        base_metrics.f1_score,
                    )
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    logger.warning(
                        "Model does not support 'disable_adapter'. Skipping comparative base evaluation."
                    )

            # ------------------------------------------------------------------
            # 2. Evaluate Adapted / Fine-Tuned Model
            # ------------------------------------------------------------------
            logger.info("Evaluating Fine-Tuned Model checkpoint...")
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            result = self.evaluator.evaluate(
                model=model,
                tokenizer=tokenizer,
                eval_dataset=eval_dataset,
                eval_config=eval_config,
                experiment_name=experiment_name,
                model_id=model_id,
            )

            # ------------------------------------------------------------------
            # 3. Compute Comparative Delta Metrics (Δ Uplift)
            # ------------------------------------------------------------------
            if base_metrics is not None:
                delta_metrics = {
                    "delta_exact_match": round(
                        result.metrics.exact_match - base_metrics.exact_match, 4
                    ),
                    "delta_f1_score": round(
                        result.metrics.f1_score - base_metrics.f1_score, 4
                    ),
                    "delta_rouge_l": round(
                        (result.metrics.rouge_l or 0.0) - (base_metrics.rouge_l or 0.0), 4
                    ),
                }
                if (
                    result.metrics.tolerance_accuracy_5pct is not None
                    and base_metrics.tolerance_accuracy_5pct is not None
                ):
                    delta_metrics["delta_tolerance_accuracy_5pct"] = round(
                        result.metrics.tolerance_accuracy_5pct - base_metrics.tolerance_accuracy_5pct, 4
                    )
                if (
                    result.metrics.tolerance_accuracy_1pct is not None
                    and base_metrics.tolerance_accuracy_1pct is not None
                ):
                    delta_metrics["delta_tolerance_accuracy_1pct"] = round(
                        result.metrics.tolerance_accuracy_1pct - base_metrics.tolerance_accuracy_1pct, 4
                    )
                result.base_model_metrics = base_metrics
                result.delta_metrics = delta_metrics

            # ------------------------------------------------------------------
            # 4. Enterprise Quality Gate Assertion
            # ------------------------------------------------------------------
            if eval_config.fail_on_quality_gate_breach and not result.passed_quality_gate:
                threshold_info = (
                    f"Min Tolerance 5% Threshold={eval_config.min_tolerance_accuracy_threshold}"
                    if eval_config.min_tolerance_accuracy_threshold is not None
                    else f"Min F1 Threshold={eval_config.min_f1_threshold}"
                )
                err_msg = f"Quality gate breached! ({threshold_info})"
                logger.error(err_msg)
                raise ModelEvaluationError(err_msg)

            # ------------------------------------------------------------------
            # 5. Persist Scorecards (JSON & Markdown)
            # ------------------------------------------------------------------
            if output_dir:
                self._save_reports(result, Path(output_dir))

            return result

        except ModelEvaluationError:
            raise
        except Exception as exc:
            logger.error("EvaluateModelUseCase failed: %s", exc, exc_info=True)
            raise ModelEvaluationError(f"Model evaluation failed: {exc}") from exc

    def _save_reports(self, result: EvaluationResult, output_dir: Path) -> None:
        """Persist evaluation JSON and Markdown scorecard reports."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. JSON Report
        json_path = output_dir / "evaluation_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2)
        logger.info("Saved evaluation JSON report to '%s'", json_path)

        # 2. Markdown Summary Scorecard
        md_path = output_dir / "evaluation_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self._render_markdown_summary(result))
        logger.info("Saved evaluation Markdown scorecard to '%s'", md_path)

    @staticmethod
    def _render_markdown_summary(result: EvaluationResult) -> str:
        """Render comprehensive human-readable Markdown evaluation summary."""
        gate_badge = "✅ PASSED" if result.passed_quality_gate else "❌ FAILED"
        lines = [
            f"# Model Evaluation Report: {result.experiment_name}",
            "",
            f"- **Model Identifier**: `{result.model_id}`",
            f"- **Task Type**: `{result.task_type.value}`",
            f"- **Evaluated Split**: `{result.eval_split}`",
            f"- **Evaluated Samples**: `{result.total_samples}`",
            f"- **Quality Gate**: **{gate_badge}**",
            f"- **Timestamp**: `{result.timestamp}`",
            "",
            "## 1. Aggregate Fine-Tuned Metrics",
            "",
            "| Metric | Score |",
            "| :--- | :--- |",
            f"| **Exact Match (EM)** | `{result.metrics.exact_match:.4f}` |",
            f"| **Token F1-Score** | `{result.metrics.f1_score:.4f}` |",
            f"| **ROUGE-L** | `{result.metrics.rouge_l or 0.0:.4f}` |",
        ]

        # Financial quantitative metrics if present
        if result.metrics.tolerance_accuracy_5pct is not None:
            lines.extend([
                f"| **Operational Accuracy (<= 5% Error)** | `{result.metrics.tolerance_accuracy_5pct * 100:.2f}%` |",
                f"| **Strict Accuracy (<= 1% Error)** | `{(result.metrics.tolerance_accuracy_1pct or 0.0) * 100:.2f}%` |",
                f"| **Sign / Directional Accuracy** | `{(result.metrics.sign_accuracy or 0.0) * 100:.2f}%` |",
                f"| **Parseable Number Rate** | `{(result.metrics.parseable_number_rate or 0.0) * 100:.2f}%` |",
            ])
            if result.metrics.median_relative_error is not None:
                lines.append(f"| **Median Relative Error (MRE)** | `{result.metrics.median_relative_error * 100:.2f}%` |")
            if result.metrics.mean_absolute_percentage_error is not None:
                lines.append(f"| **Mean Absolute Pct Error (MAPE)** | `{result.metrics.mean_absolute_percentage_error * 100:.2f}%` |")

        lines.append("")

        # Error distribution buckets if present
        if result.metrics.custom_metrics and "error_bucket_under_1pct" in result.metrics.custom_metrics:
            cm = result.metrics.custom_metrics
            lines.extend([
                "### Error Distribution Breakdown",
                "",
                f"- 🟢 **Within 1% Error**: {int(cm.get('error_bucket_under_1pct', 0))} samples",
                f"- 🟡 **Within 1% - 5% Error**: {int(cm.get('error_bucket_1_to_5pct', 0))} samples",
                f"- 🟠 **Within 5% - 20% Error**: {int(cm.get('error_bucket_5_to_20pct', 0))} samples",
                f"- 🔴 **Over 20% Error**: {int(cm.get('error_bucket_over_20pct', 0))} samples",
                f"- ❌ **Unparseable Output**: {int(cm.get('error_bucket_unparseable', 0))} samples",
                "",
            ])

        # 2. Comparative Baseline Delta Table (if evaluated)
        if result.base_model_metrics and result.delta_metrics:
            def fmt_delta(val: float) -> str:
                return f"+{val:.4f}" if val >= 0 else f"{val:.4f}"

            delta_rows = [
                f"| **Exact Match** | `{result.base_model_metrics.exact_match:.4f}` | `{result.metrics.exact_match:.4f}` | `{fmt_delta(result.delta_metrics.get('delta_exact_match', 0.0))}` |",
                f"| **Token F1** | `{result.base_model_metrics.f1_score:.4f}` | `{result.metrics.f1_score:.4f}` | `{fmt_delta(result.delta_metrics.get('delta_f1_score', 0.0))}` |",
                f"| **ROUGE-L** | `{result.base_model_metrics.rouge_l or 0.0:.4f}` | `{result.metrics.rouge_l or 0.0:.4f}` | `{fmt_delta(result.delta_metrics.get('delta_rouge_l', 0.0))}` |",
            ]
            if "delta_tolerance_accuracy_5pct" in result.delta_metrics:
                b_tol = (result.base_model_metrics.tolerance_accuracy_5pct or 0.0) * 100
                f_tol = (result.metrics.tolerance_accuracy_5pct or 0.0) * 100
                d_tol = result.delta_metrics["delta_tolerance_accuracy_5pct"] * 100
                d_sign = "+" if d_tol >= 0 else ""
                delta_rows.append(f"| **Accuracy (<= 5% Error)** | `{b_tol:.2f}%` | `{f_tol:.2f}%` | `{d_sign}{d_tol:.2f}%` |")

            lines.extend([
                "## 2. Comparative Baseline Analysis (Base vs. Fine-Tuned)",
                "",
                "| Metric | Base Model (Zero-Shot) | Fine-Tuned Checkpoint | Delta (Δ Uplift) |",
                "| :--- | :---: | :---: | :---: |",
                *delta_rows,
                "",
            ])

        # 3. Production Serving & Hardware Telemetry Table
        ttft_p50 = f"{result.ttft_ms_p50:.2f} ms" if result.ttft_ms_p50 is not None else "N/A"
        ttft_p95 = f"{result.ttft_ms_p95:.2f} ms" if result.ttft_ms_p95 is not None else "N/A"
        tpot_p50 = f"{result.tpot_ms_p50:.2f} ms" if result.tpot_ms_p50 is not None else "N/A"
        tpot_p95 = f"{result.tpot_ms_p95:.2f} ms" if result.tpot_ms_p95 is not None else "N/A"
        peak_vram = (
            f"{result.peak_memory_mb:.2f} MB" if result.peak_memory_mb is not None else "N/A"
        )

        lines.extend([
            "## 3. Production Serving & Telemetry",
            "",
            "| Telemetry Parameter | Measured Value |",
            "| :--- | :--- |",
            f"| **TTFT (Median / P50)** | `{ttft_p50}` |",
            f"| **TTFT (Tail / P95)** | `{ttft_p95}` |",
            f"| **TPOT (Median / P50)** | `{tpot_p50}` |",
            f"| **TPOT (Tail / P95)** | `{tpot_p95}` |",
            f"| **Avg Latency per Sample** | `{result.latency_per_sample_ms:.2f} ms` |",
            f"| **Generation Throughput** | `{result.throughput_tokens_per_sec:.2f} tokens/sec` |",
            f"| **Peak GPU VRAM Allocated** | `{peak_vram}` |",
            "",
        ])

        # 4. Sample Predictions
        if result.sample_details:
            lines.extend([
                "## 4. Sample Predictions Inspection (First 3 Samples)",
                "",
            ])
            for i, sample in enumerate(result.sample_details[:3], 1):
                sample_lines = [
                    f"### Sample {i}",
                    f"- **Prompt**: {sample.prompt}",
                    f"- **Ground Truth**: {sample.ground_truth}",
                    f"- **Prediction**: {sample.prediction}",
                    f"- **Sample F1**: `{sample.f1_score:.4f}` (Exact Match: `{sample.exact_match}`)",
                ]
                if sample.parsed_prediction is not None and sample.parsed_ground_truth is not None:
                    tol_str = "✅ Within 5%" if sample.within_tolerance else "❌ Outside 5%"
                    err_pct = (sample.relative_error or 0.0) * 100.0
                    sample_lines.append(
                        f"- **Numerical Evaluation**: Parsed Pred=`{sample.parsed_prediction}`, "
                        f"Truth=`{sample.parsed_ground_truth}`, Rel Error=`{err_pct:.2f}%` ({tol_str})"
                    )
                sample_lines.append("")
                lines.extend(sample_lines)

        return "\n".join(lines)
