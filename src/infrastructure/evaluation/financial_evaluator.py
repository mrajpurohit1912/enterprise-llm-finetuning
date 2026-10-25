"""
src/infrastructure/evaluation/financial_evaluator.py
Financial Quantitative evaluation adapter implementing EvaluatorBase.
Composes ISampleExtractor and FinancialNumericalParser to score financial reasoning.
"""

import collections
import logging
import statistics
import string
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from src.domain.interfaces.evaluator import EvaluatorBase
from src.domain.interfaces.sample_extractor import ISampleExtractor
from src.domain.schemas.evaluation_schema import (
    EvaluationConfig,
    EvaluationResult,
    EvaluationSample,
    MetricScores,
)
from src.infrastructure.evaluation.numerical_parser import FinancialNumericalParser
from src.infrastructure.evaluation.sample_extractor import EvaluationSampleExtractor

logger = logging.getLogger(__name__)


class FinancialEvaluator(EvaluatorBase):
    """
    Enterprise-grade Financial Quantitative Evaluator.
    Measures operational accuracy within numerical tolerance (1%, 5%),
    directional sign correctness, financial entity parsing, and relative error distributions.
    """

    def __init__(
        self,
        sample_extractor: Optional[ISampleExtractor] = None,
        parser: Optional[FinancialNumericalParser] = None,
    ) -> None:
        """
        Initialize FinancialEvaluator via Dependency Injection.

        Args:
            sample_extractor: Composed ISampleExtractor for dataset parsing.
            parser: Financial numerical normalizer and comparison engine.
        """
        self._sample_extractor = sample_extractor or EvaluationSampleExtractor()
        self._parser = parser or FinancialNumericalParser()

    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        eval_dataset: Any,
        eval_config: EvaluationConfig,
        experiment_name: str,
        model_id: str,
    ) -> EvaluationResult:
        """
        Execute batched financial evaluation with quantitative tolerance scoring.
        """
        logger.info("Starting quantitative financial evaluation for model '%s'...", model_id)

        samples = self._sample_extractor.extract_samples(eval_dataset, eval_config=eval_config)
        if not samples:
            logger.warning("No evaluation samples found. Returning empty evaluation result.")
            return EvaluationResult(
                experiment_name=experiment_name,
                model_id=model_id,
                task_type=eval_config.task_type,
                eval_split=eval_config.eval_split,
                total_samples=0,
                metrics=MetricScores(
                    exact_match=0.0,
                    f1_score=0.0,
                    rouge_l=0.0,
                    tolerance_accuracy_5pct=0.0,
                    tolerance_accuracy_1pct=0.0,
                    sign_accuracy=0.0,
                    parseable_number_rate=0.0,
                ),
                latency_per_sample_ms=0.0,
                throughput_tokens_per_sec=0.0,
                passed_quality_gate=True,
                sample_details=[],
            )

        model.eval()
        device = next(model.parameters()).device

        # Track GPU VRAM
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # Configure Tokenizer for Left-Padded Batch Generation
        orig_padding_side = getattr(tokenizer, "padding_side", "right")
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 0

        batch_size = max(1, eval_config.batch_size)
        total_generated_tokens = 0
        evaluated_samples: List[EvaluationSample] = []

        # NLP metrics
        exact_matches: List[float] = []
        f1_scores: List[float] = []
        rouge_l_scores: List[float] = []

        # Quantitative Metrics Tracking
        parseable_count = 0
        within_5pct_count = 0
        within_1pct_count = 0
        within_custom_tol_count = 0
        sign_match_count = 0
        relative_errors: List[float] = []

        # Error distribution buckets
        bucket_under_1pct = 0
        bucket_1_to_5pct = 0
        bucket_5_to_20pct = 0
        bucket_over_20pct = 0
        bucket_unparseable = 0

        sample_latencies_ms: List[float] = []
        tpot_measurements_ms: List[float] = []
        ttft_measurements_ms: List[float] = []

        eval_start_time = time.perf_counter()
        target_tol = eval_config.numerical_tolerance if eval_config.numerical_tolerance is not None else 0.05

        try:
            with tqdm(total=len(samples), desc=f"Evaluating {model_id.split('/')[-1]}", unit="sample") as pbar:
                for i in range(0, len(samples), batch_size):
                    batch = samples[i : i + batch_size]
                    batch_prompts = [self._format_prompt(p, tokenizer) for p, _ in batch]
                    batch_truths = [gt for _, gt in batch]

                    # Tokenize batch with left padding and bounded sequence length
                    encoded = tokenizer(
                        batch_prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=1536,
                    )
                    inputs = {k: v.to(device) for k, v in encoded.items()}
                    input_length = inputs["input_ids"].shape[1]

                    # Measure batch generation time under inference_mode (no gradient or view tracking)
                    batch_start = time.perf_counter()
                    with torch.inference_mode():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=eval_config.max_new_tokens,
                            temperature=eval_config.temperature,
                            do_sample=eval_config.temperature > 0.0,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    batch_duration = time.perf_counter() - batch_start

                    # Decode newly generated tokens
                    newly_generated_tokens = outputs[:, input_length:]
                    decoded_predictions = tokenizer.batch_decode(
                        newly_generated_tokens, skip_special_tokens=True
                    )

                    batch_new_tokens_count = newly_generated_tokens.numel()
                    total_generated_tokens += batch_new_tokens_count

                    avg_sample_ms = (batch_duration / len(batch)) * 1000.0
                    sample_latencies_ms.extend([avg_sample_ms] * len(batch))

                    if batch_new_tokens_count > 0:
                        tpot_ms = (batch_duration / batch_new_tokens_count) * 1000.0
                        tpot_measurements_ms.append(tpot_ms)

                    ttft_estimate_ms = avg_sample_ms / max(1, newly_generated_tokens.shape[1])
                    ttft_measurements_ms.append(ttft_estimate_ms)

                    # Explicitly release CUDA tensor memory before metric post-processing
                    del encoded, inputs, outputs, newly_generated_tokens
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Score each sample quantitatively and textually
                    for idx, prediction_str in enumerate(decoded_predictions):
                        prediction = prediction_str.strip()
                        ground_truth = batch_truths[idx].strip()
                        raw_prompt = batch[idx][0]

                        # 1. Classical NLP overlap
                        em = self.compute_exact_match(prediction, ground_truth)
                        f1 = self.compute_token_f1(prediction, ground_truth)
                        rouge_l = self.compute_rouge_l(prediction, ground_truth)

                        exact_matches.append(em)
                        f1_scores.append(f1)
                        rouge_l_scores.append(rouge_l)

                        # 2. Financial Numerical Parsing
                        pred_num = self._parser.parse_float(prediction)
                        truth_num = self._parser.parse_float(ground_truth)

                        rel_err: Optional[float] = None
                        within_tol: Optional[bool] = None

                        if pred_num is not None:
                            parseable_count += 1

                        if pred_num is not None and truth_num is not None:
                            rel_err = self._parser.compute_relative_error(pred_num, truth_num)
                            relative_errors.append(rel_err)

                            # Tolerance checks
                            is_within_5pct = rel_err <= 0.05
                            is_within_1pct = rel_err <= 0.01
                            is_within_custom = rel_err <= target_tol

                            if is_within_5pct:
                                within_5pct_count += 1
                            if is_within_1pct:
                                within_1pct_count += 1
                            if is_within_custom:
                                within_custom_tol_count += 1

                            within_tol = is_within_custom

                            # Sign match check
                            if self._parser.is_sign_match(pred_num, truth_num):
                                sign_match_count += 1

                            # Distribution bucketing
                            if rel_err <= 0.01:
                                bucket_under_1pct += 1
                            elif rel_err <= 0.05:
                                bucket_1_to_5pct += 1
                            elif rel_err <= 0.20:
                                bucket_5_to_20pct += 1
                            else:
                                bucket_over_20pct += 1
                        else:
                            bucket_unparseable += 1

                        evaluated_samples.append(
                            EvaluationSample(
                                prompt=raw_prompt,
                                ground_truth=ground_truth,
                                prediction=prediction,
                                exact_match=(em == 1.0),
                                f1_score=f1,
                                parsed_ground_truth=truth_num,
                                parsed_prediction=pred_num,
                                relative_error=rel_err,
                                within_tolerance=within_tol,
                            )
                        )

                    pbar.update(len(batch))

        finally:
            tokenizer.padding_side = orig_padding_side

        total_elapsed = time.perf_counter() - eval_start_time
        total_eval_samples = len(samples)

        # Aggregate Metrics Calculations
        mean_em = sum(exact_matches) / total_eval_samples if total_eval_samples else 0.0
        mean_f1 = sum(f1_scores) / total_eval_samples if total_eval_samples else 0.0
        mean_rouge_l = sum(rouge_l_scores) / total_eval_samples if total_eval_samples else 0.0

        tol_acc_5pct = within_5pct_count / total_eval_samples if total_eval_samples else 0.0
        tol_acc_1pct = within_1pct_count / total_eval_samples if total_eval_samples else 0.0
        sign_acc = sign_match_count / max(1, len(relative_errors)) if relative_errors else 0.0
        parseable_rate = parseable_count / total_eval_samples if total_eval_samples else 0.0

        mre = statistics.median(relative_errors) if relative_errors else None
        mape = statistics.mean(relative_errors) if relative_errors else None

        # Quality Gate Check
        passed_quality_gate = True
        if eval_config.min_tolerance_accuracy_threshold is not None:
            passed_quality_gate = tol_acc_5pct >= eval_config.min_tolerance_accuracy_threshold
        elif eval_config.min_f1_threshold is not None:
            passed_quality_gate = mean_f1 >= eval_config.min_f1_threshold

        if eval_config.min_exact_match_threshold is not None:
            passed_quality_gate = passed_quality_gate and (
                mean_em >= eval_config.min_exact_match_threshold
            )

        # Latency and Telemetry
        avg_latency = (
            sum(sample_latencies_ms) / len(sample_latencies_ms)
            if sample_latencies_ms
            else 0.0
        )
        throughput = (
            total_generated_tokens / total_elapsed if total_elapsed > 0 else 0.0
        )

        p50_ttft = statistics.median(ttft_measurements_ms) if ttft_measurements_ms else None
        p95_ttft = (
            self._percentile(ttft_measurements_ms, 95) if ttft_measurements_ms else None
        )
        p50_tpot = statistics.median(tpot_measurements_ms) if tpot_measurements_ms else None
        p95_tpot = (
            self._percentile(tpot_measurements_ms, 95) if tpot_measurements_ms else None
        )

        peak_vram_mb: Optional[float] = None
        if torch.cuda.is_available() and device.type == "cuda":
            peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        custom_metrics: Dict[str, float] = {
            "error_bucket_under_1pct": float(bucket_under_1pct),
            "error_bucket_1_to_5pct": float(bucket_1_to_5pct),
            "error_bucket_5_to_20pct": float(bucket_5_to_20pct),
            "error_bucket_over_20pct": float(bucket_over_20pct),
            "error_bucket_unparseable": float(bucket_unparseable),
        }

        metric_scores = MetricScores(
            exact_match=round(mean_em, 4),
            f1_score=round(mean_f1, 4),
            rouge_l=round(mean_rouge_l, 4),
            tolerance_accuracy_5pct=round(tol_acc_5pct, 4),
            tolerance_accuracy_1pct=round(tol_acc_1pct, 4),
            sign_accuracy=round(sign_acc, 4),
            parseable_number_rate=round(parseable_rate, 4),
            median_relative_error=round(mre, 4) if mre is not None else None,
            mean_absolute_percentage_error=round(mape, 4) if mape is not None else None,
            custom_metrics=custom_metrics,
        )

        return EvaluationResult(
            experiment_name=experiment_name,
            model_id=model_id,
            task_type=eval_config.task_type,
            eval_split=eval_config.eval_split,
            total_samples=total_eval_samples,
            metrics=metric_scores,
            latency_per_sample_ms=round(avg_latency, 2),
            throughput_tokens_per_sec=round(throughput, 2),
            ttft_ms_p50=round(p50_ttft, 2) if p50_ttft is not None else None,
            ttft_ms_p95=round(p95_ttft, 2) if p95_ttft is not None else None,
            tpot_ms_p50=round(p50_tpot, 2) if p50_tpot is not None else None,
            tpot_ms_p95=round(p95_tpot, 2) if p95_tpot is not None else None,
            peak_memory_mb=round(peak_vram_mb, 2) if peak_vram_mb is not None else None,
            passed_quality_gate=passed_quality_gate,
            sample_details=evaluated_samples,
        )

    # --------------------------------------------------------------------------
    # Utility Methods
    # --------------------------------------------------------------------------

    @staticmethod
    def _percentile(data: List[float], p: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * (p / 100.0)
        f = int(k)
        c = f + 1
        if c < len(sorted_data):
            return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])
        return sorted_data[f]

    @staticmethod
    def _format_prompt(prompt: str, tokenizer: Any) -> str:
        """Wraps input string with chat template if supported by tokenizer."""
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            try:
                messages = [{"role": "user", "content": prompt}]
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        return prompt

    @staticmethod
    def normalize_text(text: str) -> str:
        """Lowercases, removes punctuation, articles, and extra whitespace."""
        text = text.lower()
        text = "".join(ch for ch in text if ch not in set(string.punctuation))
        tokens = [t for t in text.split() if t not in {"a", "an", "the"}]
        return " ".join(tokens)

    @classmethod
    def compute_exact_match(cls, prediction: str, ground_truth: str) -> float:
        return 1.0 if cls.normalize_text(prediction) == cls.normalize_text(ground_truth) else 0.0

    @classmethod
    def compute_token_f1(cls, prediction: str, ground_truth: str) -> float:
        pred_tokens = cls.normalize_text(prediction).split()
        truth_tokens = cls.normalize_text(ground_truth).split()
        if not pred_tokens or not truth_tokens:
            return 1.0 if pred_tokens == truth_tokens else 0.0
        common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        prec = 1.0 * num_same / len(pred_tokens)
        rec = 1.0 * num_same / len(truth_tokens)
        return (2 * prec * rec) / (prec + rec)

    @classmethod
    def compute_rouge_l(cls, prediction: str, ground_truth: str) -> float:
        pred_tokens = cls.normalize_text(prediction).split()
        truth_tokens = cls.normalize_text(ground_truth).split()
        m, n = len(truth_tokens), len(pred_tokens)
        if m == 0 or n == 0:
            return 0.0
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if truth_tokens[i - 1] == pred_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs = dp[m][n]
        if lcs == 0:
            return 0.0
        prec = lcs / n
        rec = lcs / m
        return (2 * prec * rec) / (prec + rec)
