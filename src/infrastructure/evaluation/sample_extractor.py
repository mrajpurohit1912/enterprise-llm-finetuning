"""
src/infrastructure/evaluation/sample_extractor.py
Concrete service extracting standardized (prompt, reference) pairs from diverse dataset structures.
Implements the ISampleExtractor domain port via Composition.
"""

import logging
from typing import Any, List, Optional, Tuple

from src.domain.interfaces.sample_extractor import ISampleExtractor
from src.domain.schemas.evaluation_schema import EvaluationConfig

logger = logging.getLogger(__name__)


class EvaluationSampleExtractor(ISampleExtractor):
    """
    Enterprise-grade sample extractor that dynamically extracts (prompt, reference_answer)
    pairs from diverse dataset formats.

    Resolution Priority:
      1. Explicit column names configured in eval_config (prompt_column, target_column).
      2. ChatML / Conversation message sequences (messages, conversations).
      3. Ranked industry candidate pairs (query/answer, question/answer, etc.).
      4. Fuzzy keyword fallback search.
    """

    def extract_samples(
        self,
        dataset: Any,
        eval_config: EvaluationConfig,
    ) -> List[Tuple[str, str]]:
        """
        Dynamically extracts (prompt, reference_answer) pairs from dataset partitions.

        Args:
            dataset: Partition of dataset (Hugging Face Dataset, list of dicts, or Iterable).
            eval_config: Evaluation configuration containing explicit mappings and limits.

        Returns:
            List[Tuple[str, str]]: Standardized list of (clean_prompt, clean_answer) tuples.
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
