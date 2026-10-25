"""
tests/unit/test_numerical_parser.py
Unit tests for FinancialNumericalParser.
"""

import unittest
from src.infrastructure.evaluation.numerical_parser import FinancialNumericalParser


class TestFinancialNumericalParser(unittest.TestCase):
    """Test suite for financial number extraction and relative error calculations."""

    def test_parse_simple_numbers(self) -> None:
        self.assertEqual(FinancialNumericalParser.parse_float("94.0"), 94.0)
        self.assertEqual(FinancialNumericalParser.parse_float("  194.5  "), 194.5)
        self.assertEqual(FinancialNumericalParser.parse_float("0"), 0.0)

    def test_parse_currencies_and_commas(self) -> None:
        self.assertEqual(FinancialNumericalParser.parse_float("$5,735.00"), 5735.0)
        self.assertEqual(FinancialNumericalParser.parse_float("€1,200.50"), 1200.50)
        self.assertEqual(FinancialNumericalParser.parse_float("£ 450"), 450.0)

    def test_parse_accounting_parentheses_negatives(self) -> None:
        self.assertEqual(FinancialNumericalParser.parse_float("(107)"), -107.0)
        self.assertEqual(FinancialNumericalParser.parse_float("($ 35.5)"), -35.5)
        self.assertEqual(FinancialNumericalParser.parse_float("-94.0"), -94.0)

    def test_parse_percentages(self) -> None:
        self.assertAlmostEqual(FinancialNumericalParser.parse_float("14.46%"), 0.1446, places=4)
        self.assertAlmostEqual(FinancialNumericalParser.parse_float("0.5%"), 0.005, places=4)

    def test_parse_embedded_in_text(self) -> None:
        self.assertEqual(FinancialNumericalParser.parse_float("The net change is $94.0 million"), 94.0)
        self.assertEqual(FinancialNumericalParser.parse_float("Answer: 0.14629"), 0.14629)

    def test_relative_error_and_tolerance(self) -> None:
        # User Kaggle Run Sample 2: 0.14629 vs 0.14464 (rel error ~1.14%)
        err = FinancialNumericalParser.compute_relative_error(0.14629, 0.14464)
        self.assertLess(err, 0.02)
        self.assertTrue(FinancialNumericalParser.is_within_tolerance(0.14629, 0.14464, tolerance=0.05))
        self.assertFalse(FinancialNumericalParser.is_within_tolerance(0.14629, 0.14464, tolerance=0.01))

    def test_percentage_ratio_scale_alignment(self) -> None:
        # When model outputs 14.46% (0.1446) and truth is 0.14464
        self.assertTrue(FinancialNumericalParser.is_within_tolerance(0.1446, 0.14464, tolerance=0.01))
        # When model outputs 14.46 (not divided by 100) and truth is 0.1446
        self.assertTrue(FinancialNumericalParser.is_within_tolerance(14.46, 0.1446, tolerance=0.01))

    def test_sign_matching(self) -> None:
        self.assertTrue(FinancialNumericalParser.is_sign_match(94.0, 194.0))
        self.assertTrue(FinancialNumericalParser.is_sign_match(-50.0, -10.0))
        self.assertFalse(FinancialNumericalParser.is_sign_match(-50.0, 50.0))
        self.assertFalse(FinancialNumericalParser.is_sign_match(50.0, -50.0))

    def test_invalid_and_empty(self) -> None:
        self.assertIsNone(FinancialNumericalParser.parse_float(""))
        self.assertIsNone(FinancialNumericalParser.parse_float("N/A"))
        self.assertIsNone(FinancialNumericalParser.parse_float("No answer possible"))


if __name__ == "__main__":
    unittest.main()
