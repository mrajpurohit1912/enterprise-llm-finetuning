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
                else:
                    logger.warning(
                        "Model does not support 'disable_adapter'. Skipping comparative base evaluation."
                    )

            # ------------------------------------------------------------------
            # 2. Evaluate Adapted / Fine-Tuned Model
            # ------------------------------------------------------------------
            logger.info("Evaluating Fine-Tuned Model checkpoint...")
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
                result.base_model_metrics = base_metrics
                result.delta_metrics = delta_metrics

            # ------------------------------------------------------------------
            # 4. Enterprise Quality Gate Assertion
            # ------------------------------------------------------------------
            if eval_config.fail_on_quality_gate_breach and not result.passed_quality_gate:
                err_msg = (
                    f"Quality gate breached! F1={result.metrics.f1_score:.4f} "
                    f"(Min Threshold={eval_config.min_f1_threshold})"
                )
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
            "",
        ]

        # 2. Comparative Baseline Delta Table (if evaluated)
        if result.base_model_metrics and result.delta_metrics:
            def fmt_delta(val: float) -> str:
                return f"+{val:.4f}" if val >= 0 else f"{val:.4f}"

            lines.extend([
                "## 2. Comparative Baseline Analysis (Base vs. Fine-Tuned)",
                "",
                "| Metric | Base Model (Zero-Shot) | Fine-Tuned Checkpoint | Delta (Δ Uplift) |",
                "| :--- | :---: | :---: | :---: |",
                f"| **Exact Match** | `{result.base_model_metrics.exact_match:.4f}` | `{result.metrics.exact_match:.4f}` | `{fmt_delta(result.delta_metrics.get('delta_exact_match', 0.0))}` |",
                f"| **Token F1** | `{result.base_model_metrics.f1_score:.4f}` | `{result.metrics.f1_score:.4f}` | `{fmt_delta(result.delta_metrics.get('delta_f1_score', 0.0))}` |",
                f"| **ROUGE-L** | `{result.base_model_metrics.rouge_l or 0.0:.4f}` | `{result.metrics.rouge_l or 0.0:.4f}` | `{fmt_delta(result.delta_metrics.get('delta_rouge_l', 0.0))}` |",
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
                lines.extend([
                    f"### Sample {i}",
                    f"- **Prompt**: {sample.prompt}",
                    f"- **Ground Truth**: {sample.ground_truth}",
                    f"- **Prediction**: {sample.prediction}",
                    f"- **Sample F1**: `{sample.f1_score:.4f}` (Exact Match: `{sample.exact_match}`)",
                    "",
                ])

        return "\n".join(lines)
