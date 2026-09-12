"""Tool: analyze_bottleneck — classify the performance bottleneck of a completed experiment."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from inferops.memory.db import get_result_by_id
from inferops.observability import span

BottleneckType = Literal["compute-bound", "memory-bound", "scheduling-bound", "kv-bound", "unknown"]


class AnalyzeBottleneckInput(BaseModel):
    experiment_id: str = Field(description="ID of the completed experiment to analyze.")


class BottleneckAnalysis(BaseModel):
    experiment_id: str
    bottleneck: BottleneckType
    confidence: Literal["high", "medium", "low"]
    evidence: list[str] = Field(description="Observed signals that support the classification.")
    recommendations: list[str] = Field(description="Concrete next steps to address the bottleneck.")
    metrics_snapshot: dict[str, float]


def _snap(**kwargs) -> dict[str, float]:
    return {k: float(v) for k, v in kwargs.items() if v is not None}


def analyze_bottleneck(inp: AnalyzeBottleneckInput) -> BottleneckAnalysis:
    """
    Classify the primary performance bottleneck of a completed experiment.

    Reads the stored ExperimentResult and GPU metrics from the memory DB and
    applies rule-based heuristics to categorize the bottleneck as one of:

      compute-bound     GPU util high, throughput scales with batch size
      memory-bound      High VRAM usage, TTFT grows super-linearly with seq length
      scheduling-bound  High TTFT variance (p99/p50 > 3), many short sequences queued
      kv-bound          GPU util moderate but E2E latency high; KV cache thrashing
      unknown           Insufficient signal

    Missing / unsamped metrics are not invented as 0 for classification.
    """
    with span("tool.analyze_bottleneck", {"experiment_id": inp.experiment_id}):
        result = get_result_by_id(inp.experiment_id)

    if result is None:
        raise ValueError(f"Experiment '{inp.experiment_id}' not found in memory DB")

    gpu_util = result.gpu_utilization_pct  # None if not sampled
    gpu_mem = result.gpu_memory_used_gb
    ttft_p50 = result.ttft.p50
    ttft_p99 = result.ttft.p99
    e2e_p50 = result.e2e_latency.p50
    e2e_p99 = result.e2e_latency.p99
    rps = result.throughput_rps
    tok_s = result.tokens_per_second

    if ttft_p50 is None or ttft_p99 is None or e2e_p50 is None or e2e_p99 is None:
        return BottleneckAnalysis(
            experiment_id=inp.experiment_id,
            bottleneck="unknown",
            confidence="low",
            evidence=["Latency samples missing (n/a) — cannot classify bottleneck"],
            recommendations=[
                "Re-run with streaming client TTFT and a complete request ledger"
            ],
            metrics_snapshot=_snap(
                gpu_util_pct=gpu_util,
                gpu_mem_gb=gpu_mem,
                ttft_p50_ms=ttft_p50,
                ttft_p99_ms=ttft_p99,
                e2e_p50_ms=e2e_p50,
                e2e_p99_ms=e2e_p99,
                throughput_rps=rps,
                tokens_per_second=tok_s,
            ),
        )

    ttft_variance_ratio = ttft_p99 / ttft_p50 if ttft_p50 > 0 else 1.0
    e2e_variance_ratio = e2e_p99 / e2e_p50 if e2e_p50 > 0 else 1.0

    metrics = _snap(
        gpu_util_pct=gpu_util,
        gpu_mem_gb=gpu_mem,
        ttft_p50_ms=ttft_p50,
        ttft_p99_ms=ttft_p99,
        ttft_variance_ratio=round(ttft_variance_ratio, 2),
        e2e_p50_ms=e2e_p50,
        e2e_p99_ms=e2e_p99,
        e2e_variance_ratio=round(e2e_variance_ratio, 2),
        throughput_rps=rps,
        tokens_per_second=tok_s,
    )

    evidence: list[str] = []
    recommendations: list[str] = []
    bottleneck: BottleneckType = "unknown"
    confidence: Literal["high", "medium", "low"] = "low"

    rps_f = rps if rps is not None else 0.0
    tok_f = tok_s if tok_s is not None else 0.0

    # --- Scheduling-bound ---
    if ttft_variance_ratio > 3.0:
        bottleneck = "scheduling-bound"
        confidence = "high" if ttft_variance_ratio > 5.0 else "medium"
        evidence.append(f"TTFT p99/p50 ratio = {ttft_variance_ratio:.1f} (threshold: >3.0)")
        evidence.append(
            "High variance means some requests wait significantly longer for their first token"
        )
        recommendations.append(
            "Enable chunked_prefill to interleave long prefills with decode steps"
        )
        recommendations.append("Reduce max_num_batched_tokens to limit prefill chunk size")

    # --- KV-bound (requires sampled GPU util) ---
    elif gpu_util is not None and gpu_util < 80 and e2e_p50 > 1500 and e2e_variance_ratio > 1.5:
        bottleneck = "kv-bound"
        confidence = "medium"
        evidence.append(f"GPU util only {gpu_util:.0f}% but E2E p50 = {e2e_p50:.0f}ms (high)")
        evidence.append(
            f"E2E p99/p50 = {e2e_variance_ratio:.1f} — suggests KV cache thrashing or eviction"
        )
        recommendations.append("Reduce max_num_seqs to decrease KV cache pressure")
        recommendations.append("Enable prefix_caching if prompts share a common prefix")
        recommendations.append(
            "Lower gpu_memory_utilization slightly to leave headroom for KV pages"
        )

    # --- Compute-bound ---
    elif gpu_util is not None and gpu_util >= 88:
        bottleneck = "compute-bound"
        confidence = "high" if gpu_util >= 92 else "medium"
        evidence.append(f"GPU utilization = {gpu_util:.0f}% (threshold: ≥88%)")
        evidence.append(f"Throughput = {rps_f:.2f} rps, {tok_f:.0f} tok/s")
        recommendations.append(
            "Try increasing max_num_batched_tokens to improve arithmetic intensity"
        )
        recommendations.append(
            "This is close to hardware ceiling — further gains require hardware "
            "upgrade or quantization"
        )

    # --- Memory-bound ---
    elif gpu_mem is not None and gpu_mem > 5.5:
        bottleneck = "memory-bound"
        confidence = "medium"
        evidence.append(f"GPU memory used = {gpu_mem:.2f} GB (>5.5 GB on a 6 GB card)")
        evidence.append("Memory pressure likely causing KV cache page evictions")
        recommendations.append("Lower gpu_memory_utilization to 0.75 to reduce allocation")
        recommendations.append(
            "Reduce max_model_len if workload doesn't require long contexts"
        )
        recommendations.append("Reduce max_num_seqs to free KV cache slots")

    else:
        bottleneck = "unknown"
        confidence = "low"
        if gpu_util is None:
            evidence.append(
                "GPU util not sampled — cannot claim compute/memory classification"
            )
        else:
            mem_s = f"{gpu_mem:.2f}" if gpu_mem is not None else "n/a"
            evidence.append(f"GPU util = {gpu_util:.0f}%, mem = {mem_s} GB — no dominant signal")
        recommendations.append(
            "Run with higher concurrency or longer sequences to stress-test the system"
        )
        recommendations.append(
            "Compare against another variant to identify the limiting factor"
        )

    return BottleneckAnalysis(
        experiment_id=inp.experiment_id,
        bottleneck=bottleneck,
        confidence=confidence,
        evidence=evidence,
        recommendations=recommendations,
        metrics_snapshot=metrics,
    )
