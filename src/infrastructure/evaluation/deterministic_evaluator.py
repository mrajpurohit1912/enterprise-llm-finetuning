"""
src/infrastructure/evaluation/deterministic_evaluator.py
Concrete evaluation engine computing Exact Match, Token F1, ROUGE-L, and generation latency.
"""

import collections
import logging
import string
import time
from typing import Any, List, Optional, Tuple

import torch

from src.domain.interfaces.evaluator import EvaluatorBase
from src.domain.schemas.evaluation_schema import (
    EvaluationConfig,
    EvaluationResult,
    EvaluationSample,
    MetricScores,
)

logger = logging.getLogger(__name__)


class DeterministicEvaluator(EvaluatorBase):
    """
    Concrete evaluation engine using exact matching, token-level F1,
    and Longest Common Subsequence (ROUGE-L) to score model predictions.
    """

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
        Execute batched evaluation with serving telemetry and quality gates.
        """
        logger.info("Starting deterministic evaluation for model '%s'...", model_id)

        samples = self._extract_samples(eval_dataset, eval_config=eval_config)
        if not samples:
            logger.warning("No evaluation samples found. Returning empty evaluation result.")
            return EvaluationResult(
                experiment_name=experiment_name,
                model_id=model_id,
                task_type=eval_config.task_type,
                eval_split=eval_config.eval_split,
                total_samples=0,
                metrics=MetricScores(exact_match=0.0, f1_score=0.0, rouge_l=0.0),
                latency_per_sample_ms=0.0,
                throughput_tokens_per_sec=0.0,
                passed_quality_gate=True,
                sample_details=[],
            )

        model.eval()
        device = next(model.parameters()).device

        # Track GPU VRAM
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        # Configure Tokenizer for Left-Padded Batch Generation
        orig_padding_side = getattr(tokenizer, "padding_side", "right")
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 0

        batch_size = max(1, eval_config.batch_size)
        total_generated_tokens = 0
        evaluated_samples: List[EvaluationSample] = []
        exact_matches: List[float] = []
        f1_scores: List[float] = []
        rouge_l_scores: List[float] = []

        sample_latencies_ms: List[float] = []
        tpot_measurements_ms: List[float] = []
        ttft_measurements_ms: List[float] = []

        eval_start_time = time.perf_counter()

        try:
            for i in range(0, len(samples), batch_size):
                batch = samples[i : i + batch_size]
                batch_prompts = [self._format_prompt(p, tokenizer) for p, _ in batch]
                batch_truths = [gt for _, gt in batch]

                # Tokenize batch with left padding
                encoded = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                inputs = {k: v.to(device) for k, v in encoded.items()}
                input_length = inputs["input_ids"].shape[1]

                # Measure batch generation time
                batch_start = time.perf_counter()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=eval_config.max_new_tokens,
                        temperature=eval_config.temperature,
                        do_sample=eval_config.temperature > 0.0,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                batch_duration = time.perf_counter() - batch_start

                # Decode newly generated tokens (from input_length onwards)
                newly_generated_tokens = outputs[:, input_length:]
                decoded_predictions = tokenizer.batch_decode(
                    newly_generated_tokens, skip_special_tokens=True
                )

                # Batch telemetry
                batch_new_tokens_count = newly_generated_tokens.numel()
                total_generated_tokens += batch_new_tokens_count

                avg_sample_ms = (batch_duration / len(batch)) * 1000.0
                sample_latencies_ms.extend([avg_sample_ms] * len(batch))

                if batch_new_tokens_count > 0:
                    tpot_ms = (batch_duration / batch_new_tokens_count) * 1000.0
                    tpot_measurements_ms.append(tpot_ms)

                ttft_estimate_ms = avg_sample_ms / max(1, newly_generated_tokens.shape[1])
                ttft_measurements_ms.append(ttft_estimate_ms)

                # Score each sample in the batch
                for idx, prediction_str in enumerate(decoded_predictions):
                    prediction = prediction_str.strip()
                    ground_truth = batch_truths[idx].strip()
                    raw_prompt = batch[idx][0]

                    em = self.compute_exact_match(prediction, ground_truth)
                    f1 = self.compute_token_f1(prediction, ground_truth)
                    rouge_l = self.compute_rouge_l(prediction, ground_truth)

                    exact_matches.append(em)
                    f1_scores.append(f1)
                    rouge_l_scores.append(rouge_l)

                    evaluated_samples.append(
                        EvaluationSample(
                            prompt=raw_prompt,
                            ground_truth=ground_truth,
                            prediction=prediction,
                            exact_match=bool(em == 1.0),
                            f1_score=round(f1, 4),
                        )
                    )

        finally:
            tokenizer.padding_side = orig_padding_side

        # Aggregation & Telemetry Percentiles
        total_eval_time = time.perf_counter() - eval_start_time
        total_eval_samples = len(evaluated_samples)

        mean_em = sum(exact_matches) / total_eval_samples if total_eval_samples > 0 else 0.0
        mean_f1 = sum(f1_scores) / total_eval_samples if total_eval_samples > 0 else 0.0
        mean_rouge_l = sum(rouge_l_scores) / total_eval_samples if total_eval_samples > 0 else 0.0

        avg_latency_ms = (
            (total_eval_time / total_eval_samples) * 1000.0 if total_eval_samples > 0 else 0.0
        )
        throughput = (
            total_generated_tokens / total_eval_time if total_eval_time > 0 else 0.0
        )

        def calc_percentile(arr: List[float], p: float) -> Optional[float]:
            if not arr:
                return None
            sorted_arr = sorted(arr)
            idx = int(len(sorted_arr) * p)
            return round(sorted_arr[min(idx, len(sorted_arr) - 1)], 2)

        peak_vram_mb = (
            round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)
            if torch.cuda.is_available()
            else None
        )

        # Quality Gate Check
        passed_gate = True
        if eval_config.min_f1_threshold is not None:
            passed_gate = mean_f1 >= eval_config.min_f1_threshold
        if eval_config.min_exact_match_threshold is not None:
            passed_gate = passed_gate and (mean_em >= eval_config.min_exact_match_threshold)

        logger.info(
            "Evaluation complete: EM=%.4f, F1=%.4f, ROUGE-L=%.4f, Throughput=%.2f tok/s (Passed Gate: %s)",
            mean_em,
            mean_f1,
            mean_rouge_l,
            throughput,
            passed_gate,
        )

        return EvaluationResult(
            experiment_name=experiment_name,
            model_id=model_id,
            task_type=eval_config.task_type,
            eval_split=eval_config.eval_split,
            total_samples=total_eval_samples,
            metrics=MetricScores(
                exact_match=round(mean_em, 4),
                f1_score=round(mean_f1, 4),
                rouge_l=round(mean_rouge_l, 4),
            ),
            latency_per_sample_ms=round(avg_latency_ms, 2),
            throughput_tokens_per_sec=round(throughput, 2),
            ttft_ms_p50=calc_percentile(ttft_measurements_ms, 0.50),
            ttft_ms_p95=calc_percentile(ttft_measurements_ms, 0.95),
            tpot_ms_p50=calc_percentile(tpot_measurements_ms, 0.50),
            tpot_ms_p95=calc_percentile(tpot_measurements_ms, 0.95),
            peak_memory_mb=peak_vram_mb,
            passed_quality_gate=passed_gate,
            sample_details=evaluated_samples,
        )

    # --------------------------------------------------------------------------
    # Metric Calculation Utilities
    # --------------------------------------------------------------------------

    @staticmethod
    def normalize_text(text: str) -> str:
        """Lowercases text, removes punctuation, and normalizes whitespace."""
        text = text.lower()
        text = "".join(ch for ch in text if ch not in string.punctuation)
        return " ".join(text.split())

    @classmethod
    def compute_exact_match(cls, prediction: str, ground_truth: str) -> float:
        """Returns 1.0 if normalized strings match exactly, else 0.0."""
        return 1.0 if cls.normalize_text(prediction) == cls.normalize_text(ground_truth) else 0.0

    @classmethod
    def compute_token_f1(cls, prediction: str, ground_truth: str) -> float:
        """Computes token-level harmonic F1 score between prediction and ground truth."""
        pred_tokens = cls.normalize_text(prediction).split()
        truth_tokens = cls.normalize_text(ground_truth).split()

        if not pred_tokens or not truth_tokens:
            return 1.0 if pred_tokens == truth_tokens else 0.0

        common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(truth_tokens)
        return (2 * precision * recall) / (precision + recall)

    @classmethod
    def compute_rouge_l(cls, prediction: str, ground_truth: str) -> float:
        """Computes ROUGE-L F1 score based on Longest Common Subsequence (LCS)."""
        pred_tokens = cls.normalize_text(prediction).split()
        truth_tokens = cls.normalize_text(ground_truth).split()

        m, n = len(truth_tokens), len(pred_tokens)
        if m == 0 or n == 0:
            return 0.0

        # Dynamic programming for Longest Common Subsequence
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

    # --------------------------------------------------------------------------
    # Helper Prompt Extraction
    # --------------------------------------------------------------------------

    def _extract_samples(
        self,
        dataset: Any,
        eval_config: EvaluationConfig,
    ) -> List[Tuple[str, str]]:
        """
        Dynamically extracts (prompt, reference_answer) pairs from diverse dataset formats.

        Resolution Priority:
          1. Explicit column names from eval_config (prompt_column, target_column).
          2. ChatML / Conversation message sequences (messages, conversations).
          3. Standard industry candidate pairs (query/answer, instruction/output, etc.).
          4. Fuzzy keyword fallback search.
        """
        samples: List[Tuple[str, str]] = []
        max_samples = eval_config.max_eval_samples

        total_len = len(dataset) if hasattr(dataset, "__len__") else None
        limit = min(total_len, max_samples) if max_samples and total_len else max_samples

        if hasattr(dataset, "select") and limit:
            data_iter = dataset.select(range(limit))
        elif hasattr(dataset, "__getitem__") and total_len:
            data_iter = [dataset[i] for i in range(limit or total_len)]
        else:
            data_iter = dataset

        for item in data_iter:
            if not isinstance(item, dict):
                continue

            prompt, answer = None, None

            # Priority 1: Explicit YAML Mapping from EvaluationConfig
            if eval_config.prompt_column and eval_config.target_column:
                prompt = item.get(eval_config.prompt_column)
                answer = item.get(eval_config.target_column)

            # Priority 2: Structured Multi-turn / ChatML Messages
            if prompt is None and "messages" in item and isinstance(item["messages"], list):
                msgs = item["messages"]
                user_msgs = [m.get("content") for m in msgs if m.get("role") in ("user", "human")]
                assistant_msgs = [
                    m.get("content")
                    for m in msgs
                    if m.get("role") in ("assistant", "bot", "gpt")
                ]
                if user_msgs and assistant_msgs:
                    prompt = user_msgs[-1]
                    answer = assistant_msgs[-1]

            if prompt is None and "conversations" in item and isinstance(item["conversations"], list):
                msgs = item["conversations"]
                user_msgs = [m.get("value") for m in msgs if m.get("from") in ("human", "user")]
                assistant_msgs = [
                    m.get("value") for m in msgs if m.get("from") in ("gpt", "assistant")
                ]
                if user_msgs and assistant_msgs:
                    prompt = user_msgs[-1]
                    answer = assistant_msgs[-1]

            # Priority 3: Ranked Candidate Heuristic Pairs
            if prompt is None:
                candidate_pairs = [
                    ("query", "answer"),        
                    ("question", "answer"),     
                    ("instruction", "output"),  
                    ("prompt", "response"),    
                    ("input", "output"),       
                    ("context", "answer"),      
                    ("text", "label"),          
                    ("query", "response"),
                ]
                for p_col, a_col in candidate_pairs:
                    if p_col in item and a_col in item and item[p_col] and item[a_col]:
                        prompt = item[p_col]
                        answer = item[a_col]
                        break

            # Priority 4: Fuzzy Keyword Fallback Search
            if prompt is None:
                keys = list(item.keys())
                prompt_key = next(
                    (k for k in keys if any(w in k.lower() for w in ["query", "question", "prompt", "input"])),
                    None,
                )
                answer_key = next(
                    (k for k in keys if any(w in k.lower() for w in ["answer", "response", "output", "target", "label"])),
                    None,
                )
                if prompt_key and answer_key:
                    prompt = item[prompt_key]
                    answer = item[answer_key]

            # Validate and append
            if prompt is not None and answer is not None:
                clean_prompt = str(prompt).strip()
                clean_answer = str(answer).strip()
                if clean_prompt and clean_answer:
                    samples.append((clean_prompt, clean_answer))
                    if max_samples and len(samples) >= max_samples:
                        break

        if not samples and dataset:
            available_cols = (
                list(dataset[0].keys())
                if hasattr(dataset, "__getitem__") and len(dataset) > 0
                else []
            )
            logger.warning(
                "Dynamic extractor could not extract prompt/target pairs. Available columns: %s. "
                "Consider setting 'prompt_column' and 'target_column' in 'evaluation' configuration.",
                available_cols,
            )

        return samples

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
