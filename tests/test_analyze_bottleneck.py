"""Unit tests for analyze_bottleneck tool — pure logic, no external calls."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from inferops.tools.analyze_bottleneck import AnalyzeBottleneckInput, analyze_bottleneck


def test_compute_bound(result):
    result.gpu_utilization_pct = 92.0
    result.throughput_rps = 17.0

    with patch("inferops.tools.analyze_bottleneck.get_result_by_id", return_value=result):
        out = analyze_bottleneck(AnalyzeBottleneckInput(experiment_id="test_default"))

    assert out.bottleneck == "compute-bound"
    assert out.confidence in ("high", "medium")
    assert any("92" in e for e in out.evidence)


def test_scheduling_bound(result):
    # High TTFT variance
    from inferops.schemas import LatencyPercentiles
    result.ttft = LatencyPercentiles(p50=40.0, p90=100.0, p95=150.0, p99=250.0)
    result.gpu_utilization_pct = 70.0

    with patch("inferops.tools.analyze_bottleneck.get_result_by_id", return_value=result):
        out = analyze_bottleneck(AnalyzeBottleneckInput(experiment_id="test_default"))

    assert out.bottleneck == "scheduling-bound"
    assert any("chunked" in r.lower() for r in out.recommendations)


def test_memory_bound(result):
    result.gpu_memory_used_gb = 5.7
    result.gpu_utilization_pct = 75.0

    with patch("inferops.tools.analyze_bottleneck.get_result_by_id", return_value=result):
        out = analyze_bottleneck(AnalyzeBottleneckInput(experiment_id="test_default"))

    assert out.bottleneck == "memory-bound"


def test_kv_bound(result):
    from inferops.schemas import LatencyPercentiles
    result.gpu_utilization_pct = 65.0
    result.e2e_latency = LatencyPercentiles(p50=2000.0, p90=3500.0, p95=4000.0, p99=5000.0)

    with patch("inferops.tools.analyze_bottleneck.get_result_by_id", return_value=result):
        out = analyze_bottleneck(AnalyzeBottleneckInput(experiment_id="test_default"))

    assert out.bottleneck == "kv-bound"


def test_experiment_not_found():
    with patch("inferops.tools.analyze_bottleneck.get_result_by_id", return_value=None):
        with pytest.raises(ValueError, match="not found"):
            analyze_bottleneck(AnalyzeBottleneckInput(experiment_id="ghost"))
