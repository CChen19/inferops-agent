"""Tool: run_benchmark — run one vLLM experiment and persist the result."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from configs.search_space import make_configs
from inferops.bench_runner import BenchmarkError, run_experiment
from inferops.memory.db import save_result
from inferops.observability import span
from inferops.schemas import ExperimentResult
from workloads.definitions import ALL_WORKLOADS, get_prompts

# Safe parameter ranges for RTX 3060 Laptop (6 GB, WSL2)
_SAFE_RANGES: dict[str, tuple[Any, Any]] = {
    "gpu_memory_utilization": (0.50, 0.85),
    "max_num_seqs": (16, 256),
    "max_num_batched_tokens": (2048, 8192),
    "max_model_len": (512, 4096),
}
_ALLOWED_PATCH_KEYS = {
    *set(_SAFE_RANGES),
    "enable_chunked_prefill",
    "enable_prefix_caching",
    "enforce_eager",
}


class RunBenchmarkInput(BaseModel):
    """Input for run_benchmark."""
    experiment_id: str = Field(description="Unique name for this run, e.g. 'chunked_v2'")
    config_patch: dict[str, Any] = Field(
        default_factory=dict,
        description="vLLM parameter overrides applied on top of the default config. "
                    "Allowed keys: max_num_batched_tokens, max_num_seqs, max_model_len, "
                    "gpu_memory_utilization, enable_chunked_prefill, enable_prefix_caching.",
    )
    workload_name: str = Field(
        default="chat_short",
        description=(
            "Workload to benchmark. One of the workload names in "
            "workloads.definitions.ALL_WORKLOADS."
        ),
    )
    persist: bool = Field(
        default=True,
        description="Whether to save the result to the experiment memory DB.",
    )
    session_id: str | None = Field(
        default=None,
        description="Session prefix / id mapped to run_id and MLflow tags.",
    )


class RunBenchmarkOutput(BaseModel):
    """Output from run_benchmark. Missing metrics are None (never invent 0)."""
    experiment_id: str
    workload_name: str
    throughput_rps: float | None
    tokens_per_second: float | None
    ttft_p50_ms: float | None
    ttft_p99_ms: float | None
    e2e_p50_ms: float | None
    e2e_p99_ms: float | None
    error_rate: float | None = None
    gpu_util_pct: float | None
    gpu_mem_gb: float | None
    success_rate: str
    mlflow_run_id: str | None
    run_id: str = ""
    status: str = "insufficient_evidence"
    ledger_path: str | None = None
    error: str = ""


def run_benchmark(inp: RunBenchmarkInput) -> RunBenchmarkOutput:
    """
    Run one vLLM benchmark experiment.

    Starts a fresh vLLM subprocess with the given config, sends the specified
    workload, collects TTFT / E2E / throughput / GPU metrics, logs to MLflow,
    and optionally persists to the experiment memory DB.

    Raises ValueError if any config_patch value is outside the safe range for
    the RTX 3060 (6 GB) hardware.

    Contract: persisted results carry run_id / status / evidence. Missing
    critical evidence yields status=insufficient_evidence (never auto-valid).
    """
    # Validate patch ranges
    for key, val in inp.config_patch.items():
        if key not in _ALLOWED_PATCH_KEYS:
            raise ValueError(
                f"Unknown config_patch key: {key}. "
                f"Allowed keys: {', '.join(sorted(_ALLOWED_PATCH_KEYS))}"
            )
        if key in _SAFE_RANGES:
            lo, hi = _SAFE_RANGES[key]
            if not (lo <= val <= hi):
                raise ValueError(f"{key}={val} outside safe range [{lo}, {hi}] for RTX 3060")

    workload_map = {w.name: w for w in ALL_WORKLOADS}
    workload = workload_map.get(inp.workload_name)
    if workload is None:
        raise ValueError(
            f"Unknown workload: {inp.workload_name}. "
            f"Valid workloads: {', '.join(sorted(workload_map))}"
        )

    # Build config by patching the default
    tags = {"session_id": inp.session_id} if inp.session_id else {}
    base_cfg = make_configs(workload)[0]  # default variant as base
    patched = base_cfg.model_copy(
        update={
            **inp.config_patch,
            "experiment_id": inp.experiment_id,
            "tags": {**base_cfg.tags, **tags},
        }
    )
    if patched.max_num_batched_tokens < patched.max_model_len:
        raise ValueError(
            "max_num_batched_tokens must be >= max_model_len for vLLM "
            f"({patched.max_num_batched_tokens} < {patched.max_model_len})"
        )

    prompts = get_prompts(workload)

    with span(
        "tool.run_benchmark",
        {
            "experiment_id": inp.experiment_id,
            "workload": inp.workload_name,
            "session_id": inp.session_id or "",
        },
    ):
        try:
            result: ExperimentResult = run_experiment(
                patched,
                prompts,
                session_id=inp.session_id,
            )
        except BenchmarkError as exc:
            # P2-5: persist failed contract row so eval counts the attempt
            if exc.result is not None and inp.persist:
                save_result(exc.result)
            raise

    if inp.persist:
        save_result(result)

    status_value = (
        result.status.value if hasattr(result.status, "value") else str(result.status)
    )
    return RunBenchmarkOutput(
        experiment_id=result.experiment_id,
        workload_name=inp.workload_name,
        throughput_rps=result.throughput_rps,
        tokens_per_second=result.tokens_per_second,
        ttft_p50_ms=result.ttft.p50,
        ttft_p99_ms=result.ttft.p99,
        e2e_p50_ms=result.e2e_latency.p50,
        e2e_p99_ms=result.e2e_latency.p99,
        error_rate=result.error_rate,
        gpu_util_pct=result.gpu_utilization_pct,
        gpu_mem_gb=result.gpu_memory_used_gb,
        success_rate=f"{result.successful_requests}/{result.total_requests}",
        mlflow_run_id=result.mlflow_run_id,
        run_id=result.run_id,
        status=status_value,
        ledger_path=result.ledger_path,
    )
