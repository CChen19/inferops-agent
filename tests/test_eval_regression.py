"""Unit tests for eval regression gate."""

from __future__ import annotations

from inferops.eval.regression import regression_gate


def _report(gap: float, composite: float) -> dict:
    return {
        "strategies": {
            "greedy_agent": [
                {
                    "workload_name": "chat_short",
                    "gap_pct": gap,
                    "composite": composite,
                }
            ]
        }
    }


def test_regression_gate_passes_within_thresholds():
    gate = regression_gate(
        current=_report(gap=6.0, composite=0.90),
        baseline=_report(gap=3.0, composite=0.92),
        max_outcome_regression_pct=5.0,
        min_composite_delta=-0.05,
    )

    assert gate.passed is True
    assert gate.failures == []


def test_regression_gate_fails_on_gap_regression():
    gate = regression_gate(
        current=_report(gap=12.0, composite=0.91),
        baseline=_report(gap=3.0, composite=0.92),
        max_outcome_regression_pct=5.0,
    )

    assert gate.passed is False
    assert "gap regressed" in gate.failures[0]


def test_regression_gate_fails_on_composite_drop():
    gate = regression_gate(
        current=_report(gap=4.0, composite=0.80),
        baseline=_report(gap=3.0, composite=0.92),
        min_composite_delta=-0.05,
    )

    assert gate.passed is False
    assert "composite dropped" in gate.failures[0]


def test_regression_gate_fails_when_strategy_missing():
    gate = regression_gate(current={"strategies": {}}, baseline=_report(1.0, 1.0))

    assert gate.passed is False
    assert "No current rows" in gate.failures[0]
