"""Unit tests for compare_experiments tool."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from inferops.tools.compare_experiments import CompareExperimentsInput, compare_experiments


def test_compare_throughput_b_wins(result, result_b):
    with patch("inferops.tools.compare_experiments.get_result_by_id", side_effect=[result, result_b]):
        out = compare_experiments(CompareExperimentsInput(
            experiment_id_a="test_default",
            experiment_id_b="test_big_batch",
            metric="throughput_rps",
            n_bootstrap=500,
        ))

    assert out.winner == "b"
    assert out.delta_pct > 0  # b is higher throughput
    assert out.value_a == pytest.approx(2.0, rel=0.1)
    assert out.value_b == pytest.approx(2.38, rel=0.1)
    assert "significant" in out.interpretation.lower() or "noise" in out.interpretation.lower()


def test_compare_latency_metric(result, result_b):
    with patch("inferops.tools.compare_experiments.get_result_by_id", side_effect=[result, result_b]):
        out = compare_experiments(CompareExperimentsInput(
            experiment_id_a="test_default",
            experiment_id_b="test_big_batch",
            metric="e2e_p50_ms",
            n_bootstrap=500,
        ))

    # result_b has lower E2E p50 (780 vs 900), so b wins for latency
    assert out.winner == "b"
    assert out.delta_pct < 0  # lower is better


def test_compare_missing_experiment(result):
    with patch("inferops.tools.compare_experiments.get_result_by_id", side_effect=[result, None]):
        with pytest.raises(ValueError, match="not found"):
            compare_experiments(CompareExperimentsInput(
                experiment_id_a="test_default",
                experiment_id_b="ghost",
            ))


def test_compare_ci_bounds_are_ordered(result, result_b):
    with patch("inferops.tools.compare_experiments.get_result_by_id", side_effect=[result, result_b]):
        out = compare_experiments(CompareExperimentsInput(
            experiment_id_a="test_default",
            experiment_id_b="test_big_batch",
            n_bootstrap=500,
        ))

    assert out.ci_low_pct <= out.ci_high_pct
