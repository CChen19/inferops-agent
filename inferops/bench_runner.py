"""
Top-level benchmark orchestrator.

Flow per experiment:
  1. Start vLLM subprocess with ExperimentConfig params
  2. Wait for /health (with OOM / crash detection)
  3. Start GPU monitor
  4. Run async load (warmup + measure)
  5. Stop GPU monitor
  6. Stop vLLM subprocess
  7. Build ExperimentResult
  8. Log to MLflow
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import Callable

from rich.console import Console
from rich.table import Table

from inferops.observability import init_mlflow, log_experiment_result, mlflow_run
from inferops.schemas import (
    ExperimentConfig,
    ExperimentResult,
    LatencyPercentiles,
)
from inferops.tools.gpu_monitor import GPUMonitor
from inferops.tools.traffic import extract_percentiles, run_load
from inferops.tools.vllm_process import VLLMProcess

console = Console()

VLLM_HOST = os.getenv("VLLM_HOST", "127.0.0.1")
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))


class BenchmarkError(Exception):
    pass


class OOMError(BenchmarkError):
    pass


class StartupTimeoutError(BenchmarkError):
    pass


def _run_load_with_cleanup_workaround(
    cfg: ExperimentConfig,
    prompts: list[str],
    timeout_s: float = 400,
    poll_interval_s: float = 2,
):
    """
    Run traffic in a worker thread and publish the result before asyncio cleanup.

    In Chainlit/anyio environments, httpx cleanup inside asyncio.run() can hang
    after the load coroutine has returned. The result must be appended inside
    the coroutine, not around asyncio.run(...), so the caller can continue even
    if the worker thread gets stuck during event-loop shutdown.
    """
    import threading

    result: list = []
    errors: list[BaseException] = []

    async def _run_and_store() -> None:
        load = await run_load(
            base_url=f"http://{VLLM_HOST}:{VLLM_PORT}",
            workload=cfg.workload,
            prompts=prompts,
            close_client=False,
            stream_response=False,
        )
        result.append(load)

    def _worker() -> None:
        try:
            asyncio.run(_run_and_store())
        except BaseException as exc:  # noqa: BLE001
            if not result:
                errors.append(exc)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        thread.join(timeout=poll_interval_s)
        if result or errors:
            break

    if errors:
        raise errors[0]
    if not result:
        raise BenchmarkError("Traffic thread timed out without a result")
    return result[0]


def run_experiment(
    cfg: ExperimentConfig,
    prompts: list[str],
    mlflow_experiment: str = "inferops",
    on_progress: Callable[[str], None] | None = None,
) -> ExperimentResult:
    """
    Run one full experiment: start vLLM → benchmark → collect → stop.
    Raises OOMError or StartupTimeoutError on failure.
    """

    def log(msg: str) -> None:
        console.print(f"  [dim]{msg}[/dim]")
        if on_progress:
            on_progress(msg)

    init_mlflow(mlflow_experiment)

    # If vLLM is already running externally (e.g. via start_vllm.sh), skip
    # lifecycle management and send traffic directly. This avoids port conflicts
    # and removes the 15-second proc.stop() wait on every experiment.
    _external = False
    try:
        import httpx as _httpx
        _r = _httpx.get(f"http://{VLLM_HOST}:{VLLM_PORT}/health", timeout=2.0)
        _external = _r.status_code == 200
    except Exception:
        pass

    with mlflow_run(run_name=cfg.experiment_id, tags={**cfg.tags, "workload": cfg.workload.name}) as run:
        proc: VLLMProcess | None = None

        if _external:
            log(f"Using external vLLM at {VLLM_HOST}:{VLLM_PORT} (skipping lifecycle)")
        else:
            log(f"Starting vLLM ({cfg.model_name}) …")
            proc = VLLMProcess(cfg, host=VLLM_HOST, port=VLLM_PORT)
            proc.start()
            if proc.log_path:
                log(f"  vLLM log → {proc.log_path}")

            ready = proc.wait_ready_verbose(log)
            if not ready:
                if proc.oom_in_log():
                    raise OOMError(f"vLLM OOM during startup — config: {cfg.experiment_id}")
                if proc.is_crashed():
                    raise BenchmarkError(f"vLLM crashed (exit {proc.exit_code()}) — see {proc.log_path}")
                raise StartupTimeoutError(f"vLLM not ready after startup timeout — see {proc.log_path}")

        log("vLLM ready. Starting GPU monitor + load …")

        gpu = GPUMonitor(interval_s=0.5)
        gpu.start()

        try:
            load = _run_load_with_cleanup_workaround(cfg, prompts)
        finally:
            gpu_summary = gpu.stop()
            if proc is not None:
                log("Stopping vLLM …")
                proc.stop()

        # Build result
        ttft_p = extract_percentiles(load.ttft_ms)
        e2e_p = extract_percentiles(load.e2e_ms)
        # TPOT = (E2E - TTFT) / (output_tokens - 1) ≈ E2E percentiles for now
        tpot_p = {k: max(0.0, e2e_p[k] - ttft_p[k]) for k in ttft_p}

        result = ExperimentResult(
            experiment_id=cfg.experiment_id,
            config=cfg,
            total_requests=load.total_requests,
            successful_requests=load.successful,
            total_time_s=load.total_time_s,
            throughput_rps=load.throughput_rps,
            tokens_per_second=load.tokens_per_second,
            ttft=LatencyPercentiles(**ttft_p),
            tpot=LatencyPercentiles(**tpot_p),
            e2e_latency=LatencyPercentiles(**e2e_p),
            gpu_memory_used_gb=gpu_summary.max_mem_used_gb,
            gpu_utilization_pct=gpu_summary.avg_util_pct,
            raw_ttft_ms=load.ttft_ms,
            raw_e2e_ms=load.e2e_ms,
            mlflow_run_id=run.info.run_id,
        )

        log_experiment_result(result)
        log(f"Done — {result.throughput_rps:.1f} rps, TTFT p50={result.ttft.p50:.0f}ms")
        return result


def print_results_table(results: list[ExperimentResult]) -> None:
    t = Table(title="Benchmark Results", show_lines=True)
    t.add_column("Experiment", style="cyan", no_wrap=True)
    t.add_column("Workload")
    t.add_column("RPS", justify="right")
    t.add_column("Tok/s", justify="right")
    t.add_column("TTFT p50", justify="right")
    t.add_column("TTFT p99", justify="right")
    t.add_column("E2E p50", justify="right")
    t.add_column("E2E p99", justify="right")
    t.add_column("GPU util%", justify="right")
    t.add_column("GPU mem GB", justify="right")

    for r in results:
        t.add_row(
            r.experiment_id,
            r.config.workload.name,
            f"{r.throughput_rps:.2f}",
            f"{r.tokens_per_second:.0f}",
            f"{r.ttft.p50:.0f}ms",
            f"{r.ttft.p99:.0f}ms",
            f"{r.e2e_latency.p50:.0f}ms",
            f"{r.e2e_latency.p99:.0f}ms",
            f"{r.gpu_utilization_pct:.0f}%" if r.gpu_utilization_pct else "—",
            f"{r.gpu_memory_used_gb:.2f}" if r.gpu_memory_used_gb else "—",
        )

    console.print(t)
