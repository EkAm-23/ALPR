"""
test_mlops_metrics.py — Unit tests for calculate_mlops_scores() and
calculate_edit_distance() in backend/modules/mlops_metrics.py

No ML models are loaded. Pure Python logic only.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ALPR-app", "backend"))

from modules.mlops_metrics import calculate_edit_distance, calculate_mlops_scores


# ── calculate_edit_distance ──────────────────────────────────────────────────

class TestEditDistance:
    def test_identical_strings(self):
        assert calculate_edit_distance("MH01AB1234", "MH01AB1234") == 0

    def test_single_substitution(self):
        # One character differs: B → C
        assert calculate_edit_distance("MH01AB1234", "MH01AC1234") == 1

    def test_single_deletion(self):
        assert calculate_edit_distance("MH01AB1234", "MH01AB123") == 1

    def test_single_insertion(self):
        assert calculate_edit_distance("MH01AB123", "MH01AB1234") == 1

    def test_completely_different(self):
        d = calculate_edit_distance("AAAA", "BBBB")
        assert d == 4

    def test_empty_pred(self):
        assert calculate_edit_distance("", "MH01AB1234") == 10

    def test_both_empty(self):
        assert calculate_edit_distance("", "") == 0

    def test_symmetric(self):
        a = calculate_edit_distance("KL16J3636", "KL16J3637")
        b = calculate_edit_distance("KL16J3637", "KL16J3636")
        assert a == b


# ── calculate_mlops_scores ───────────────────────────────────────────────────

class TestMlopsScores:
    def test_exact_match(self):
        scores = calculate_mlops_scores("MH01AB1234", "MH01AB1234")
        assert scores["exact_match"] == 1.0
        assert scores["edit_distance"] == 0
        assert scores["character_accuracy"] == 1.0

    def test_no_ground_truth_returns_none(self):
        assert calculate_mlops_scores("MH01AB1234", None) is None
        assert calculate_mlops_scores("MH01AB1234", "") is None

    def test_case_insensitive_comparison(self):
        # Both should be normalised to uppercase internally
        scores = calculate_mlops_scores("mh01ab1234", "MH01AB1234")
        assert scores["exact_match"] == 1.0

    def test_space_stripped(self):
        scores = calculate_mlops_scores("MH 01 AB 1234", "MH01AB1234")
        assert scores["exact_match"] == 1.0

    def test_partial_match_accuracy(self):
        # One char wrong out of 10 → CER = 1/10 → char_acc = 0.9
        scores = calculate_mlops_scores("MH01AB1235", "MH01AB1234")
        assert scores["exact_match"] == 0.0
        assert scores["edit_distance"] == 1
        assert abs(scores["character_accuracy"] - 0.9) < 0.01

    def test_char_accuracy_non_negative(self):
        # Even a totally wrong prediction should not give negative accuracy
        scores = calculate_mlops_scores("AAAAAAAAAA", "MH01AB1234")
        assert scores["character_accuracy"] >= 0.0

    def test_returns_all_keys(self):
        scores = calculate_mlops_scores("KL16J3636", "KL16J3636")
        assert set(scores.keys()) == {"exact_match", "edit_distance", "character_accuracy"}
