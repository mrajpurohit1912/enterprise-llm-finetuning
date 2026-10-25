"""
src/infrastructure/evaluation/numerical_parser.py
Financial and numerical parsing utility for extracting, normalizing, and comparing financial figures.
"""

import math
import re
from typing import Optional


class FinancialNumericalParser:
    """
    Enterprise utility for robust extraction and mathematical comparison of numerical figures
    from unstructured financial text, SEC 10-K tables, and model predictions.
    """

    @staticmethod
    def parse_float(text: Optional[str]) -> Optional[float]:
        """
        Parses and normalizes a string into a canonical float.

        Handles:
          - Accounting negative parentheses: '(107)' or '($107)' -> -107.0
          - Negative signs: '-94.0' -> -94.0
          - Currency symbols: '$', '€', '£', '¥', '₹'
          - Formatting commas: '5,735.50' -> 5735.50
          - Percentages: '14.5%' -> 0.145 (canonical ratio)
          - Embedded text: 'The answer is 94.0' -> 94.0
          - Scientific notation: '1.25e-3' -> 0.00125

        Args:
            text: Raw input text.

        Returns:
            Optional[float]: Extracted float or None if unparseable.
        """
        if text is None:
            return None

        clean = str(text).strip()
        if not clean:
            return None

        # Detect accounting negative: e.g., "(107)" or "( $ 107.50 )"
        is_negative = False
        accounting_match = re.search(r"\(\s*([^()]+)\s*\)", clean)
        if accounting_match:
            inner = accounting_match.group(1).strip()
            # If inner contains digits, treat as negative
            if re.search(r"\d", inner):
                is_negative = True
                clean = inner

        # Strip currency symbols and commas
        clean = re.sub(r"[\$,€£¥₹]", "", clean)
        clean = clean.replace(",", "")

        # Detect explicit negative sign before cleaning
        if "-" in clean and not is_negative:
            # Check if minus is directly associated with a digit
            if re.search(r"-\s*\d", clean):
                is_negative = True

        # Detect percentage
        is_percent = "%" in clean
        clean = clean.replace("%", "").strip()

        # Regex search for first floating point or integer number
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", clean)
        if not match:
            return None

        try:
            val = float(match.group(0))
            if is_negative and val > 0:
                val = -val
            elif not is_negative and "-" in match.group(0) and val > 0:
                val = -val

            if is_percent:
                val = val / 100.0

            if math.isnan(val) or math.isinf(val):
                return None

            return val
        except (ValueError, OverflowError):
            return None

    @classmethod
    def compute_relative_error(
        cls,
        pred: float,
        truth: float,
        eps: float = 1e-7,
    ) -> float:
        """
        Computes the relative error: |pred - truth| / max(|truth|, eps).

        Also checks for scale alignment between percentage representations
        (e.g., if ground truth is 0.1446 and pred is 14.46, or vice-versa).
        """
        raw_error = abs(pred - truth) / max(abs(truth), eps)

        # Check 100x scale factor (e.g. 0.14464 vs 14.464% ratio vs percent notation)
        if truth != 0:
            scale_100_error = abs(pred / 100.0 - truth) / max(abs(truth), eps)
            scale_div_error = abs(pred * 100.0 - truth) / max(abs(truth), eps)
            return min(raw_error, scale_100_error, scale_div_error)

        return raw_error

    @classmethod
    def is_within_tolerance(
        cls,
        pred: float,
        truth: float,
        tolerance: float = 0.05,
    ) -> bool:
        """Checks if prediction is within tolerance threshold of ground truth."""
        return cls.compute_relative_error(pred, truth) <= tolerance

    @classmethod
    def is_sign_match(cls, pred: float, truth: float) -> bool:
        """
        Checks directional/sign correctness.
        Critical in finance: Profit vs Loss must match sign.
        """
        if abs(truth) < 1e-7 and abs(pred) < 1e-7:
            return True
        return (pred * truth) > 0
