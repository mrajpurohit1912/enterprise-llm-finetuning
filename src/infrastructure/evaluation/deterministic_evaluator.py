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
        Execute evaluation on the evaluation dataset split.
        """
        logger.info("Starting deterministic evaluation for model '%s'...", model_id)

        samples = self._extract_samples(eval_dataset, max_samples=eval_config.max_eval_samples)
        if not samples:
            logger.warning("No evaluation samples found. Returning empty evaluation result.")
            return EvaluationResult(
                experiment_name=experiment_name,
                model_id=model_id,
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

        total_generated_tokens = 0
        evaluated_samples: List[EvaluationSample] = []
        exact_matches: List[float] = []
        f1_scores: List[float] = []
        rouge_l_scores: List[float] = []

        inference_start = time.perf_counter()

        for idx, (prompt, ground_truth) in enumerate(samples):
            # Encode prompt
            formatted_prompt = self._format_prompt(prompt, tokenizer)
            encoded = tokenizer(formatted_prompt, return_tensors="pt")
            if hasattr(encoded, "to"):
                inputs = encoded.to(device)
            else:
                inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in encoded.items()}
            input_length = inputs["input_ids"].shape[1]

            # Generate prediction
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=eval_config.max_new_tokens,
                    temperature=eval_config.temperature,
                    do_sample=eval_config.temperature > 0.0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

            # Decode only newly generated tokens
            generated_tokens = outputs[0][input_length:]
            total_generated_tokens += len(generated_tokens)
            prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            # Calculate individual metrics
            em = self.compute_exact_match(prediction, ground_truth)
            f1 = self.compute_token_f1(prediction, ground_truth)
            rouge_l = self.compute_rouge_l(prediction, ground_truth)

            exact_matches.append(em)
            f1_scores.append(f1)
            rouge_l_scores.append(rouge_l)

            evaluated_samples.append(
                EvaluationSample(
                    prompt=prompt,
                    ground_truth=ground_truth,
                    prediction=prediction,
                    exact_match=bool(em == 1.0),
                    f1_score=round(f1, 4),
                )
            )

        total_inference_time = time.perf_counter() - inference_start
        total_eval_samples = len(evaluated_samples)

        mean_em = sum(exact_matches) / total_eval_samples if total_eval_samples > 0 else 0.0
        mean_f1 = sum(f1_scores) / total_eval_samples if total_eval_samples > 0 else 0.0
        mean_rouge_l = sum(rouge_l_scores) / total_eval_samples if total_eval_samples > 0 else 0.0

        latency_ms = (total_inference_time / total_eval_samples) * 1000.0 if total_eval_samples > 0 else 0.0
        throughput = total_generated_tokens / total_inference_time if total_inference_time > 0 else 0.0

        # Quality gate check
        passed_gate = True
        if eval_config.min_f1_threshold is not None:
            passed_gate = mean_f1 >= eval_config.min_f1_threshold

        logger.info(
            "Evaluation complete: EM=%.4f, F1=%.4f, ROUGE-L=%.4f (Passed Gate: %s)",
            mean_em,
            mean_f1,
            mean_rouge_l,
            passed_gate,
        )

        return EvaluationResult(
            experiment_name=experiment_name,
            model_id=model_id,
            eval_split=eval_config.eval_split,
            total_samples=total_eval_samples,
            metrics=MetricScores(
                exact_match=round(mean_em, 4),
                f1_score=round(mean_f1, 4),
                rouge_l=round(mean_rouge_l, 4),
            ),
            latency_per_sample_ms=round(latency_ms, 2),
            throughput_tokens_per_sec=round(throughput, 2),
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
        self, dataset: Any, max_samples: Optional[int] = None
    ) -> List[Tuple[str, str]]:
        """Extracts (prompt, reference_answer) pairs from dataset partitions."""
        samples: List[Tuple[str, str]] = []
        data_iter = dataset

        if hasattr(dataset, "__len__"):
            limit = min(len(dataset), max_samples) if max_samples else len(dataset)
            data_iter = [dataset[i] for i in range(limit)]

        for item in data_iter:
            prompt, answer = None, None

            # Case 1: ChatML messages list
            if "messages" in item and isinstance(item["messages"], list):
                msgs = item["messages"]
                user_msgs = [m["content"] for m in msgs if m.get("role") == "user"]
                assistant_msgs = [m["content"] for m in msgs if m.get("role") == "assistant"]
                if user_msgs and assistant_msgs:
                    prompt = user_msgs[-1]
                    answer = assistant_msgs[-1]

            # Case 2: question and answer / response
            elif "question" in item and ("answer" in item or "response" in item):
                prompt = item["question"]
                answer = item.get("answer") or item.get("response")

            # Case 3: instruction and output
            elif "instruction" in item and "output" in item:
                prompt = item["instruction"]
                answer = item["output"]

            if prompt and answer:
                samples.append((str(prompt).strip(), str(answer).strip()))
                if max_samples and len(samples) >= max_samples:
                    break

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
