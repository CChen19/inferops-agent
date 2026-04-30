"""Shared pytest fixtures for inferops tool unit tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from inferops.schemas import (
    ExperimentConfig,
    ExperimentResult,
    InferenceEngine,
    LatencyPercentiles,
    ModelSize,
    SchedulerPolicy,
    WorkloadSpec,
)


@pytest.fixture
def workload() -> WorkloadSpec:
    return WorkloadSpec(
        name="chat_short",
        prompt_template="",
        num_requests=10,
        concurrency=4,
        input_len=64,
        output_len=64,
    )


@pytest.fixture
def config(workload) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="test_default",
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        model_size=ModelSize.HALF_B,
        engine=InferenceEngine.VLLM,
        max_num_seqs=64,
        max_num_batched_tokens=2048,
        max_model_len=1024,
        gpu_memory_utilization=0.80,
        enforce_eager=False,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        scheduler_policy=SchedulerPolicy.FCFS,
        workload=workload,
        tags={"variant": "default"},
    )


@pytest.fixture
def result(config) -> ExperimentResult:
    lp = LatencyPercentiles(p50=48.0, p90=60.0, p95=65.0, p99=70.0)
    e2e = LatencyPercentiles(p50=900.0, p90=950.0, p95=970.0, p99=1000.0)
    return ExperimentResult(
        experiment_id="test_default",
        config=config,
        total_requests=10,
        successful_requests=10,
        total_time_s=5.0,
        throughput_rps=2.0,
        tokens_per_second=128.0,
        ttft=lp,
        tpot=LatencyPercentiles(p50=6.0, p90=7.0, p95=7.5, p99=8.0),
        e2e_latency=e2e,
        gpu_memory_used_gb=3.7,
        gpu_utilization_pct=85.0,
        raw_ttft_ms=[40.0, 45.0, 48.0, 50.0, 55.0, 60.0, 62.0, 65.0, 68.0, 70.0],
        raw_e2e_ms=[850.0, 870.0, 900.0, 910.0, 920.0, 940.0, 960.0, 970.0, 990.0, 1000.0],
    )


@pytest.fixture
def result_b(config, workload) -> ExperimentResult:
    """A second result (big_batch variant) that is clearly better."""
    cfg_b = config.model_copy(update={
        "experiment_id": "test_big_batch",
        "max_num_batched_tokens": 4096,
        "tags": {"variant": "big_batch"},
    })
    lp = LatencyPercentiles(p50=52.0, p90=62.0, p95=64.0, p99=66.0)
    e2e = LatencyPercentiles(p50=780.0, p90=820.0, p95=840.0, p99=870.0)
    return ExperimentResult(
        experiment_id="test_big_batch",
        config=cfg_b,
        total_requests=10,
        successful_requests=10,
        total_time_s=4.2,
        throughput_rps=2.38,
        tokens_per_second=152.0,
        ttft=lp,
        tpot=LatencyPercentiles(p50=5.5, p90=6.5, p95=7.0, p99=7.5),
        e2e_latency=e2e,
        gpu_memory_used_gb=3.8,
        gpu_utilization_pct=88.0,
        raw_ttft_ms=[44.0, 48.0, 50.0, 52.0, 54.0, 58.0, 60.0, 62.0, 64.0, 66.0],
        raw_e2e_ms=[750.0, 770.0, 780.0, 790.0, 800.0, 820.0, 840.0, 850.0, 860.0, 870.0],
    )


@pytest.fixture
def tmp_db(tmp_path) -> Path:
    return tmp_path / "test_memory.db"


@pytest.fixture
def tmp_report(tmp_path) -> Path:
    return tmp_path / "test_report.md"
