"""Tests for metrics and judge calibration agreement."""
from __future__ import annotations

from eval import judge_calibration, metrics


def test_intent_metrics_known():
    m = metrics.intent_metrics(["complaint", "complaint", "compliment", "compliment"],
                               ["complaint", "compliment", "compliment", "compliment"])
    assert m["accuracy"] == 0.75


def test_escalation_cost_weighted_direction():
    # one false auto-handle (expensive), zero false escalate
    m = metrics.escalation_metrics(
        ["escalate", "auto_handle"], ["auto_handle", "auto_handle"],
        cost_false_auto=5.0, cost_false_escalate=1.0)
    assert m["false_auto_handle"] == 1
    assert m["cost_weighted_total"] == 5.0
    assert m["lean"] == "toward auto-handle"


def test_cohen_kappa_perfect_and_random():
    assert judge_calibration.cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    # completely opposite -> kappa = -1
    k = judge_calibration.cohen_kappa([1, 1, 0, 0], [0, 0, 1, 1])
    assert abs(k + 1.0) < 1e-9


def test_spearman_perfect():
    assert judge_calibration.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
