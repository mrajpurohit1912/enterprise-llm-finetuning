"""
src/application/usecases/evaluate_model_usecase.py
Atomic application use case orchestrating model evaluation, scorecard persistence, and quality gates.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

from src.domain.exceptions import ModelEvaluationError
from src.domain.interfaces.evaluator import EvaluatorBase
from src.domain.schemas.evaluation_schema import EvaluationConfig, EvaluationResult

logger = logging.getLogger(__name__)


class EvaluateModelUseCase:
    """
    Orchestrates the evaluation stage for fine-tuned or baseline LLMs.
    Persists evaluation metrics and sample-level scorecards to disk.
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
        Execute evaluation and optionally persist scorecards to output_dir.

        Args:
            model: Foundation or adapted PEFT model.
            tokenizer: Model tokenizer.
            eval_dataset: Evaluation partition (Dataset or DatasetDict).
            eval_config: Evaluation configuration and thresholds.
            experiment_name: Name of the experiment.
            model_id: Model identifier.
            output_dir: Optional path to save evaluation artifacts.

        Returns:
            EvaluationResult: Strongly-typed evaluation result DTO.

        Raises:
            ModelEvaluationError: If evaluation execution fails.
        """
        logger.info(
            "Executing EvaluateModelUseCase for experiment '%s' (split: '%s')...",
            experiment_name,
            eval_config.eval_split,
        )

        try:
            result = self.evaluator.evaluate(
                model=model,
                tokenizer=tokenizer,
                eval_dataset=eval_dataset,
                eval_config=eval_config,
                experiment_name=experiment_name,
                model_id=model_id,
            )

            # Persist evaluation scorecard if output_dir is provided
            if output_dir:
                self._save_reports(result, Path(output_dir))

            return result

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
        """Render human-readable Markdown evaluation summary."""
        gate_badge = "✅ PASSED" if result.passed_quality_gate else "❌ FAILED"
        lines = [
            f"# Model Evaluation Report: {result.experiment_name}",
            "",
            f"- **Model ID**: `{result.model_id}`",
            f"- **Evaluated Split**: `{result.eval_split}`",
            f"- **Evaluated Samples**: `{result.total_samples}`",
            f"- **Quality Gate**: **{gate_badge}**",
            f"- **Timestamp**: `{result.timestamp}`",
            "",
            "## Aggregate Metrics",
            "",
            "| Metric | Score |",
            "| :--- | :--- |",
            f"| **Exact Match (EM)** | `{result.metrics.exact_match:.4f}` |",
            f"| **Token F1-Score** | `{result.metrics.f1_score:.4f}` |",
            f"| **ROUGE-L** | `{result.metrics.rouge_l or 0.0:.4f}` |",
            f"| **Avg Latency (ms/sample)** | `{result.latency_per_sample_ms:.2f} ms` |",
            f"| **Throughput (tokens/sec)** | `{result.throughput_tokens_per_sec:.2f} tok/s` |",
            "",
        ]

        # Add top sample comparisons
        if result.sample_details:
            lines.extend([
                "## Sample Predictions Inspection (First 3 Samples)",
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
