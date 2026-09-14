"""Unit tests for compare_experiments tool."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from inferops.tools.compare_experiments import (
    CompareExperimentsInput,
    compare_experiments,
)


def test_compare_throughput_b_wins_without_synthetic_ci(result, result_b):
    with patch("inferops.tools.compare_experiments.get_result_by_id", side_effect=[result, result_b]):
        out = compare_experiments(CompareExperimentsInput(
            experiment_id_a="test_default",
            experiment_id_b="test_big_batch",
            metric="throughput_rps",
            n_bootstrap=500,
        ))

    assert out.winner == "b"
    assert out.delta_pct > 0  # b is higher throughput
    assert out.value_a == pytest.approx(2.0)
    assert out.value_b == pytest.approx(2.38)
    assert out.ci_low_pct is None
    assert out.ci_high_pct is None
    assert out.significant is False
    assert out.ci_unavailable_reason is not None
    assert "aggregate" in out.ci_unavailable_reason.lower() or "samples" in out.ci_unavailable_reason.lower()
    low = out.interpretation.lower()
    assert "cannot be judged" in low or "unavailable" in low
    assert "not significant" not in low
    assert "may be noise" not in low
    assert "statistically significant" not in low


def test_compare_latency_with_raw_samples_keeps_bootstrap(result, result_b):
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
    assert out.ci_low_pct is not None and out.ci_high_pct is not None
    assert out.ci_low_pct <= out.ci_high_pct
    assert out.ci_unavailable_reason is None
    assert "this run's request samples" in out.interpretation.lower()
    assert "significant" in out.interpretation.lower() or "noise" in out.interpretation.lower()


def test_compare_latency_without_raw_samples_no_substitute(result, result_b):
    bare_a = result.model_copy(update={"raw_ttft_ms": None, "raw_e2e_ms": None})
    bare_b = result_b.model_copy(update={"raw_ttft_ms": None, "raw_e2e_ms": None})
    with patch(
        "inferops.tools.compare_experiments.get_result_by_id",
        side_effect=[bare_a, bare_b],
    ):
        out = compare_experiments(CompareExperimentsInput(
            experiment_id_a="test_default",
            experiment_id_b="test_big_batch",
            metric="e2e_p50_ms",
            n_bootstrap=500,
        ))

    assert out.ci_low_pct is None
    assert out.ci_high_pct is None
    assert out.significant is False
    assert out.ci_unavailable_reason is not None
    assert "raw_e2e_ms" in out.ci_unavailable_reason
    assert "synthesize" in out.ci_unavailable_reason.lower()
    low = out.interpretation.lower()
    assert "cannot be judged" in low or "unavailable" in low
    assert "not significant" not in low
    assert "may be noise" not in low
    # Point comparison still available from stored percentiles
    assert out.value_a == pytest.approx(900.0)
    assert out.value_b == pytest.approx(780.0)
    assert out.winner == "b"


def test_compare_missing_experiment(result):
    with patch("inferops.tools.compare_experiments.get_result_by_id", side_effect=[result, None]):
        with pytest.raises(ValueError, match="not found"):
            compare_experiments(CompareExperimentsInput(
                experiment_id_a="test_default",
                experiment_id_b="ghost",
            ))


def test_compare_latency_ci_bounds_are_ordered(result, result_b):
    with patch("inferops.tools.compare_experiments.get_result_by_id", side_effect=[result, result_b]):
        out = compare_experiments(CompareExperimentsInput(
            experiment_id_a="test_default",
            experiment_id_b="test_big_batch",
            metric="e2e_p50_ms",
            n_bootstrap=500,
        ))

    assert out.ci_low_pct is not None and out.ci_high_pct is not None
    assert out.ci_low_pct <= out.ci_high_pct


def test_compare_zero_baseline_raises_instead_of_silent_zero(result, result_b):
    """val_a == 0 must fail closed — never invent delta_pct = 0.0."""
    zero_a = result.model_copy(update={"throughput_rps": 0.0})
    with patch(
        "inferops.tools.compare_experiments.get_result_by_id",
        side_effect=[zero_a, result_b],
    ):
        with pytest.raises(ValueError, match="baseline denominator is 0"):
            compare_experiments(CompareExperimentsInput(
                experiment_id_a="test_default",
                experiment_id_b="test_big_batch",
                metric="throughput_rps",
                n_bootstrap=200,
            ))
