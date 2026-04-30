"""Unit tests for traffic load aggregation without a real HTTP server."""

from __future__ import annotations

import pytest

from inferops.schemas import WorkloadSpec
from inferops.tools import traffic
from inferops.tools.traffic import RequestMetrics, extract_percentiles, run_load


def test_extract_percentiles_empty_returns_zeroes():
    assert extract_percentiles([]) == {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}


def test_extract_percentiles_sorts_input():
    out = extract_percentiles([100.0, 10.0, 50.0, 90.0])

    assert out["p50"] == 90.0
    assert out["p90"] == 100.0
    assert out["p99"] == 100.0


@pytest.mark.asyncio
async def test_run_load_aggregates_successful_requests(monkeypatch):
    async def fake_send_one(client, base_url, prompt, max_tokens):
        if prompt == "fail":
            return RequestMetrics(False, 0.0, 5.0, 0, error="synthetic")
        if prompt == "warmup":
            return RequestMetrics(True, 1.0, 2.0, 1)
        idx = int(prompt.removeprefix("p"))
        return RequestMetrics(True, 10.0 + idx, 100.0 + idx, 2)

    monkeypatch.setattr(traffic, "_send_one", fake_send_one)
    workload = WorkloadSpec(
        name="unit",
        prompt_template="",
        num_requests=3,
        concurrency=2,
        input_len=1,
        output_len=4,
    )

    out = await run_load(
        "http://example.test",
        workload,
        prompts=["warmup", "p1", "fail", "p3"],
        warmup_requests=1,
    )

    assert out.total_requests == 3
    assert out.successful == 2
    assert out.throughput_rps > 0
    assert out.tokens_per_second > 0
    assert out.ttft_ms == [11.0, 13.0]
    assert out.e2e_ms == [101.0, 103.0]
