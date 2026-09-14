"""
Top-level benchmark orchestrator.

Flow per experiment:
  1. Decide managed vs external vLLM lifecycle
  2. Managed: take the single-GPU lease (fail closed if another managed task
     holds it); refuse a healthy port occupant we did not spawn (never stop
     unknown services); otherwise start with the requested CLI
  3. Wait for /health, then bind readiness to NEW child PID (listener == child)
  4. Start GPU monitor + load
  5. Stop GPU monitor / managed process, release the lease
  6. Build ExperimentResult with contract evidence
  7. Log to MLflow

Stage C ownership: only the child registered via ``register_owned`` is ever
stopped by cancel/abort. ``service_mode=external`` never spawns or stops.

Week-1 item ②: healthy ≠ config applied. Never mark valid from health alone,
config file alone, or performance delta alone.

Managed `actual_config` records CLI-evidenced keys only. Promotion covers
that applyable subset; schema-only knobs (scheduler_policy,
tensor_parallel_size) stay on the request snapshot but do not block valid.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.table import Table

from inferops.metrics.aggregate import format_aggregate_report, recalculate_from_ledger
from inferops.metrics.ledger import persist_ledger
from inferops.observability import init_mlflow, log_experiment_result, mlflow_run
from inferops.schemas import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentValidityStatus,
    HardwareInfo,
    LatencyPercentiles,
    config_knobs,
    compute_workload_hash,
    derive_status,
    empty_latency,
    external_unverified_evidence,
    managed_start_evidence,
    resolve_git_sha,
)
from inferops.tools.gpu_monitor import GPUMonitor
from inferops.tools.managed_lifecycle import (
    ADOPT_STALE_ENV,
    GPUBusyError,
    GPULease,
    adopt_stale_managed_allowed,
    cancel_requested,
    is_recorded_managed_child,
    register_owned,
    release_owned,
)
from inferops.tools.traffic import extract_percentiles, run_load
from inferops.tools.vllm_process import (
    VLLMProcess,
    assert_listener_bound_to_child,
    cli_evidenced_knobs,
    knobs_match_requested,
    probe_live_instance,
)

console = Console()

VLLM_HOST = os.getenv("VLLM_HOST", "127.0.0.1")
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))


def _write_live_identity_probe(
    *,
    experiment_id: str,
    listener_pid: int,
    child_pid: int,
    start_token: str | None,
    instance_id: str,
) -> Path:
    """Write PID equality evidence while the managed child is still alive.

    Captured *before* load/teardown so Chris can assert listener==child without
    relying on post-hoc `ss` after `run_benchmark` has stopped the process.
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    path = log_dir / f"live_identity_{experiment_id}.json"
    payload = {
        "experiment_id": experiment_id,
        "listener_pid": listener_pid,
        "child_pid": child_pid,
        "pids_equal": listener_pid == child_pid,
        "start_token": start_token,
        "instance_id": instance_id,
        "host": VLLM_HOST,
        "port": VLLM_PORT,
        "captured": "while_managed_child_alive_before_teardown",
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def external_vllm_mode(service_mode: str | None = None) -> bool:
    """Return whether this run uses an external/shared server.

    A confirmed task's ``service_mode=external`` is authoritative. The
    environment variable remains supported for legacy scripts.
    """
    if service_mode is not None and service_mode.strip().lower() == "external":
        return True
    return os.getenv("INFEROPS_EXTERNAL_VLLM", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class BenchmarkError(Exception):
    """Benchmark failure. May carry a failed ExperimentResult (P2-5)."""

    def __init__(self, message: str, result: ExperimentResult | None = None):
        super().__init__(message)
        self.result = result


class OOMError(BenchmarkError):
    pass


class StartupTimeoutError(BenchmarkError):
    pass


class TaskCancelled(BenchmarkError):
    """User/graph cancelled the run. Hard control: executor must not recover it.

    Raised before spawning (cancel already requested) or after a mid-load
    stop of the owned child. ``result`` carries the failed row if one was
    logged so the attempt stays auditable.
    """


def _run_load_with_cleanup_workaround(
    cfg: ExperimentConfig,
    prompts: list[str],
    timeout_s: float = 400,
    poll_interval_s: float = 2,
    *,
    run_id: str | None = None,
    stream_response: bool = True,
    cache_enabled: bool | None = None,
):
    """
    Run traffic in a worker thread and publish the result before asyncio cleanup.

    In Chainlit/anyio environments, httpx cleanup inside asyncio.run() can hang
    after the load coroutine has returned. The result must be appended inside
    the coroutine, not around asyncio.run(...), so the caller can continue even
    if the worker thread gets stuck during event-loop shutdown.

    Client TTFT requires stream_response=True (default). Non-stream leaves
    ttft_ms=None rather than inventing E2E-as-TTFT.
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
            stream_response=stream_response,
            run_id=run_id,
            cache_enabled=cache_enabled,
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


def _ensure_managed_vllm(
    cfg: ExperimentConfig,
    requested_cli: dict,
    *,
    log: Callable[[str], None],
    on_progress: Callable[[str], None] | None,
) -> tuple[VLLMProcess, dict, object]:
    """Take the GPU lease, start managed vLLM, bind health to the new child identity.

    Stage C ownership rules (fail closed, never valid on refusal):
      * another live managed task holds the lease → ``BenchmarkError`` (gpu_busy)
      * a healthy listener we did not spawn → refuse, do **not** stop it
        (``unknown_occupant``); the only exception is a stale child recorded
        by a crashed InferOps run when ``INFEROPS_ADOPT_STALE_MANAGED=1``
      * any failure after spawn stops *our* child and releases the lease

    The child is registered as owned (``register_owned``) *as soon as it is
    spawned* — before the (often minutes-long) readiness wait — so a user
    Stop / ``cancel_owned_children`` can abort an in-flight model load. A
    cancel observed after spawn, after readiness, or right before returning
    fails closed with ``TaskCancelled`` (never ``valid``). On success the
    lease is released by ``release_owned`` in ``run_experiment``.

    Returns (proc, actual_config, evidence).
    """
    sess = cfg.tags.get("session_id") or cfg.tags.get("session_prefix")
    lease = GPULease(
        host=VLLM_HOST,
        port=VLLM_PORT,
        experiment_id=cfg.experiment_id,
        session_id=str(sess) if sess else None,
    )
    # Constructed only after the lease is ours (gpu_busy never builds a child);
    # construction is pure assignment, nothing spawns until start()/restart.
    proc: VLLMProcess | None = None

    def _teardown() -> None:
        """Stop the child we may have spawned (once), drop it from the owned
        registry, release the lease. Every step is idempotent, so this is safe
        whether or not the child was registered / already stopped by Stop."""
        try:
            if proc is not None:
                release_owned(proc)
        except Exception:
            pass
        finally:
            lease.release()

    def _abort(exc: Exception) -> None:
        """Always kill the child we may have spawned, release the lease, re-raise."""
        _teardown()
        raise exc

    def _abort_if_cancelled(stage: str) -> None:
        """User pressed Stop while we were starting: fail closed, never valid."""
        if not cancel_requested():
            return
        pid = proc.pid if proc is not None else None
        if on_progress:
            on_progress(f"status:cancelled:{stage}")
        _abort(
            TaskCancelled(
                f"Cancelled {cfg.experiment_id} during managed startup ({stage}); "
                f"spawned vLLM child pid={pid} stopped, GPU lease released."
            )
        )

    try:
        # Inside the guarded block so a Ctrl-C / SystemExit between acquire and
        # the first spawn cannot leak the lease.
        try:
            lease.acquire()
        except GPUBusyError as exc:
            log(str(exc))
            if on_progress:
                on_progress("status:failed:gpu_busy")
            raise BenchmarkError(str(exc)) from exc

        proc = VLLMProcess(cfg, host=VLLM_HOST, port=VLLM_PORT)
        probe = probe_live_instance(VLLM_HOST, VLLM_PORT)
        previous_identity = probe.identity if probe.healthy else None
        occupant_pid = previous_identity.pid if previous_identity is not None else None

        if not probe.healthy:
            reason = "fresh_start"
        else:
            adoptable = is_recorded_managed_child(
                lease.previous_record,
                listener_pid=occupant_pid,
                cmdline=probe.cmdline,
            )
            if adoptable and adopt_stale_managed_allowed():
                reason = "adopt_stale_managed_child"
            else:
                pid_s = str(occupant_pid) if occupant_pid is not None else "unknown"
                knob_s = (
                    "matching"
                    if knobs_match_requested(probe.observed_knobs, requested_cli)
                    else "different/unknown"
                )
                hint = (
                    f" It matches a stale InferOps-managed child recorded in "
                    f"{lease.path}; set {ADOPT_STALE_ENV}=1 to let InferOps stop "
                    f"and relaunch it."
                    if adoptable
                    else ""
                )
                msg = (
                    f"Port {VLLM_HOST}:{VLLM_PORT} is served by a process this task "
                    f"did not start (pid={pid_s}, knobs {knob_s}). Refusing managed "
                    f"start and NOT stopping it. Stop it yourself, or rerun with "
                    f"service_mode=external to benchmark the existing server "
                    f"as-is (insufficient_evidence).{hint}"
                )
                log(msg)
                if on_progress:
                    on_progress("status:failed:unknown_occupant")
                _abort(BenchmarkError(msg))

        if on_progress:
            on_progress(f"status:managed_lifecycle:{reason}")

        if probe.healthy:
            log(
                f"Adopting stale InferOps child pid={occupant_pid} ({reason}): stopping "
                f"it on {VLLM_HOST}:{VLLM_PORT} and relaunching with requested CLI"
            )
            if on_progress:
                on_progress("status:restarting_managed_vllm")
            stop_result = proc.restart_after_stop(
                previous_identity=previous_identity,
                stop_occupant=True,
            )
            if stop_result.still_listening:
                detail = (
                    f"listener_pid_after={stop_result.listener_pid_after}"
                    if stop_result.listener_pid_after is not None
                    else "listener PID unknown"
                )
                _abort(
                    BenchmarkError(
                        "Failed to stop prior occupant before managed restart "
                        f"({detail}) — refusing start; never valid"
                    )
                )
            if proc.pid is None:
                _abort(BenchmarkError("Managed restart did not spawn a child after stop"))
        else:
            log(f"Starting vLLM ({cfg.model_name}) …")
            if on_progress:
                on_progress("status:starting_managed_vllm")
            proc.start()
            if proc.pid is None:
                _abort(BenchmarkError("Managed start produced no child PID"))

        # Leave an ownership trail in the lock file while the child is alive.
        lease.record_child(
            child_pid=proc.pid,
            start_token=getattr(proc, "start_token", None),
            launch_cmd=list(getattr(proc, "launch_cmd", None) or []) or None,
        )
        # From here the child is owned: Stop / cancel_owned_children may stop
        # exactly this one — including during the readiness / model-load wait.
        register_owned(proc, lease, cfg.experiment_id)
        # A Stop that landed while we were spawning found nothing to stop yet.
        _abort_if_cancelled("after_spawn")

        if proc.log_path:
            log(f"  vLLM log → {proc.log_path}")

        # Deterministic startup-failure injection (GPU checklist) — BEFORE the
        # long readiness wait so oom/timeout/identity paths are fast and do not
        # depend on real /health success. Child is always stopped via _abort.
        sim_fail = os.getenv("INFEROPS_SIMULATE_STARTUP_FAILURE", "").strip().lower()
        if sim_fail == "oom":
            if on_progress:
                on_progress("status:failed:startup:simulated_oom")
            _abort(OOMError(f"INFEROPS_SIMULATE_STARTUP_FAILURE=oom — config: {cfg.experiment_id}"))
        if sim_fail == "timeout":
            if on_progress:
                on_progress("status:failed:startup:simulated_timeout")
            _abort(
                StartupTimeoutError(
                    "INFEROPS_SIMULATE_STARTUP_FAILURE=timeout — skipping readiness wait"
                )
            )
        if sim_fail == "identity":
            if on_progress:
                on_progress("status:failed:identity_bind:simulated")
            _abort(
                BenchmarkError(
                    "INFEROPS_SIMULATE_STARTUP_FAILURE=identity — "
                    "forced identity failure before readiness wait"
                )
            )

        ready = proc.wait_ready_verbose(log)
        # Stop pressed during the load window: the owned child was (or is now)
        # stopped. Report a cancel, not a mystery crash/timeout, and never valid.
        _abort_if_cancelled("during_startup_wait")
        if not ready:
            if on_progress:
                on_progress("status:failed:startup")
            if proc.oom_in_log():
                _abort(OOMError(f"vLLM OOM during startup — config: {cfg.experiment_id}"))
            if proc.is_crashed():
                _abort(
                    BenchmarkError(f"vLLM crashed (exit {proc.exit_code()}) — see {proc.log_path}")
                )
            _abort(
                StartupTimeoutError(f"vLLM not ready after startup timeout — see {proc.log_path}")
            )

        # Bind health to the NEW managed child — never trust a stale listener.
        try:
            bound_pid = assert_listener_bound_to_child(
                host=VLLM_HOST,
                port=VLLM_PORT,
                child_pid=proc.pid,
            )
        except RuntimeError as exc:
            if on_progress:
                on_progress("status:failed:identity_bind")
            _abort(BenchmarkError(str(exc)))

        final_identity = proc.identity()
        if final_identity.pid is None or not final_identity.start_token:
            _abort(
                BenchmarkError(
                    "Managed start did not yield a verifiable instance identity "
                    f"({final_identity.instance_id})"
                )
            )
        if final_identity.pid != bound_pid:
            _abort(
                BenchmarkError(f"Identity PID {final_identity.pid} != bound listener {bound_pid}")
            )

        # Emit / persist PID equality WHILE the managed child is still alive
        # (before load/teardown). Post-hoc `ss` after run_benchmark returns is useless.
        equality_msg = f"status:pid_equality:listener={bound_pid}:child={final_identity.pid}"
        log(equality_msg)  # also forwards to on_progress
        _write_live_identity_probe(
            experiment_id=cfg.experiment_id,
            listener_pid=bound_pid,
            child_pid=final_identity.pid,
            start_token=final_identity.start_token,
            instance_id=final_identity.instance_id,
        )
        # Optional hold so an operator can run `ss` in another shell before load.
        hold_s = float(os.getenv("INFEROPS_PID_PROBE_HOLD_S", "0") or "0")
        if hold_s > 0:
            log(f"INFEROPS_PID_PROBE_HOLD_S={hold_s}: holding before load (child still up)")
            time.sleep(hold_s)

        if previous_identity is not None and previous_identity.pid is not None:
            if final_identity.pid == previous_identity.pid:
                _abort(
                    BenchmarkError(
                        "Instance identity PID did not change after managed restart "
                        f"(still pid={final_identity.pid})"
                    )
                )
            log(
                f"Instance identity changed: "
                f"{previous_identity.instance_id} → {final_identity.instance_id}"
            )
            if on_progress:
                on_progress("status:identity_verified:changed")
        elif on_progress:
            on_progress(f"status:identity_verified:pid={bound_pid}")

        # Record only CLI-evidenced keys. Promotion checks complete coverage of
        # this applyable subset; schema-only request fields are not claimed.
        actual = proc.evidenced_actual_config()
        if not actual:
            # Fall back to schema helper from requested CLI snapshot.
            actual = dict(requested_cli)

        evidence = managed_start_evidence(
            process_pid=final_identity.pid,
            host=VLLM_HOST,
            port=VLLM_PORT,
            observed_params=actual,
            instance_id=final_identity.instance_id,
            start_token=final_identity.start_token,
            notes=(
                f"Managed vLLM start/restart ({reason}); listener PID bound to "
                f"child pid={bound_pid}; CLI knobs only in observed_params."
            ),
        )
        # Last gate before handing the child to the load phase.
        _abort_if_cancelled("after_ready")
        return proc, actual, evidence
    except BenchmarkError:
        raise
    except Exception as exc:
        _abort(BenchmarkError(f"managed lifecycle error: {exc}"))
        raise  # pragma: no cover
    except BaseException:
        # Ctrl-C / SystemExit while starting: never orphan the child we spawned.
        _teardown()
        raise


def run_experiment(
    cfg: ExperimentConfig,
    prompts: list[str],
    mlflow_experiment: str = "inferops",
    on_progress: Callable[[str], None] | None = None,
    session_id: str | None = None,
    service_mode: str | None = None,
) -> ExperimentResult:
    """
    Run one full experiment: ensure vLLM config applied → benchmark → collect.

    On startup / identity failure, builds a failed ExperimentResult (same run_id),
    logs it, attaches it to BenchmarkError, and re-raises so `run_benchmark` /
    executor can persist the attempt (P2-5).

    Contract:
      - Managed: restart when knobs differ / identity unknown; bind health to
        new child PID; actual_config = CLI-evidenced keys only.
      - External (INFEROPS_EXTERNAL_VLLM): insufficient_evidence, actual=null.
      - Zero successful requests: status=failed.
    """

    def log(msg: str) -> None:
        console.print(f"  [dim]{msg}[/dim]")
        if on_progress:
            on_progress(msg)

    if cancel_requested():
        # Fail closed before touching the GPU / port / lease. No attempt row:
        # nothing was started, so there is nothing to audit.
        if on_progress:
            on_progress("status:cancelled:before_start")
        raise TaskCancelled(f"Cancelled before starting {cfg.experiment_id}: no vLLM was spawned.")

    init_mlflow(mlflow_experiment)
    requested = config_knobs(cfg)
    requested_cli = cli_evidenced_knobs(cfg)
    code_sha = resolve_git_sha()
    run_id = uuid.uuid4().hex
    sess = session_id or cfg.tags.get("session_id") or cfg.tags.get("session_prefix")
    sess_str = str(sess) if sess else None

    hardware = HardwareInfo(
        model_name=cfg.model_name,
        engine=cfg.engine.value,
        vllm_version=os.getenv("VLLM_VERSION"),
        gpu_name=os.getenv("INFEROPS_GPU_NAME"),
        cuda_version=os.getenv("CUDA_VERSION"),
    )

    tags = {
        **{k: str(v) for k, v in cfg.tags.items()},
        "workload": cfg.workload.name,
        "run_id": run_id,
        "experiment_id": cfg.experiment_id,
        "schema_version": "1",
    }
    if sess_str:
        tags["session_id"] = sess_str
    if code_sha:
        tags["code_sha"] = code_sha

    def _failed_result(
        mlflow_run_id: str | None,
        reason: str,
        *,
        evidence=None,
        actual=None,
    ) -> ExperimentResult:
        # Missing metrics stay None / empty_latency — never fake 0 gains.
        return ExperimentResult(
            experiment_id=cfg.experiment_id,
            config=cfg,
            total_requests=0,
            successful_requests=0,
            total_time_s=0.0,
            throughput_rps=None,
            tokens_per_second=None,
            error_rate=None,
            ttft=empty_latency(),
            tpot=empty_latency(),
            e2e_latency=empty_latency(),
            gpu_memory_used_gb=None,
            gpu_utilization_pct=None,
            run_id=run_id,
            schema_version="1",
            code_sha=code_sha,
            session_id=sess_str,
            mlflow_run_id=mlflow_run_id,
            requested_config=requested,
            actual_config=actual,
            config_evidence=evidence,
            status=ExperimentValidityStatus.FAILED,
            workload_hash=compute_workload_hash(cfg.workload),
            hardware=hardware,
            notes=reason,
        )

    with mlflow_run(run_name=cfg.experiment_id, tags=tags) as run:
        proc: VLLMProcess | None = None
        evidence = None
        actual: dict | None = None
        status = ExperimentValidityStatus.INSUFFICIENT_EVIDENCE
        mlflow_id = run.info.run_id

        if external_vllm_mode(service_mode):
            log(f"External vLLM mode at {VLLM_HOST}:{VLLM_PORT} — not managing lifecycle")
            if on_progress:
                on_progress("status:insufficient_evidence:external_health_only")
            evidence = external_unverified_evidence(host=VLLM_HOST, port=VLLM_PORT)
            actual = None
            status = derive_status(
                evidence=evidence,
                actual_config=actual,
                requested_config=requested,
            )
        else:
            try:
                proc, actual, evidence = _ensure_managed_vllm(
                    cfg,
                    requested_cli,
                    log=log,
                    on_progress=on_progress,
                )
            except BenchmarkError as exc:
                # Child already stopped inside _ensure_managed_vllm.
                reason = str(exc)
                if on_progress:
                    on_progress(
                        "status:cancelled:startup"
                        if isinstance(exc, TaskCancelled)
                        else "status:failed:startup_or_identity"
                    )
                failed = _failed_result(mlflow_id, reason)
                try:
                    log_experiment_result(failed)
                except Exception:
                    pass
                exc.result = failed
                raise

            status = derive_status(
                evidence=evidence,
                actual_config=actual,
                requested_config=requested,
            )
            if status == ExperimentValidityStatus.INVALID and on_progress:
                on_progress("status:invalid:knob_mismatch")
            elif on_progress:
                on_progress(f"status:{status.value}:managed_process_start")

        log("vLLM ready. Starting GPU monitor + load …")

        gpu = GPUMonitor(interval_s=0.5)
        gpu_started = False
        gpu_summary = None
        load = None
        cancel_stage = "during_load"
        try:
            try:
                gpu.start()
                gpu_started = True
            except Exception as gpu_exc:
                log(f"GPU monitor unavailable ({gpu_exc}); util/mem will be n/a")
            if cancel_requested():
                # Stop landed after readiness but before traffic: do not run load.
                cancel_stage = "before_load"
                raise TaskCancelled(f"Cancelled {cfg.experiment_id} before load started")
            load = _run_load_with_cleanup_workaround(
                cfg,
                prompts,
                run_id=run_id,
                stream_response=True,
                cache_enabled=cfg.enable_prefix_caching,
            )
            if cancel_requested():
                # Stop landed while traffic ran; whatever came back is not a
                # trustworthy measurement of the requested config. Never valid.
                raise TaskCancelled(f"Cancelled {cfg.experiment_id} while load was running")
        except Exception as exc:
            if cancel_requested():
                # Owned child was stopped by cancel_owned_children (or the cancel
                # gates above fired); record the attempt as cancelled, not as a
                # mystery load failure.
                stage = cancel_stage
                if on_progress:
                    on_progress(f"status:cancelled:{stage}")
                failed = _failed_result(
                    mlflow_id,
                    f"cancelled by user ({stage.replace('_', ' ')}): {exc}",
                    evidence=evidence,
                    actual=actual,
                )
                log_experiment_result(failed)
                raise TaskCancelled(
                    f"Cancelled {cfg.experiment_id} {stage.replace('_', ' ')}", result=failed
                ) from exc
            if on_progress:
                on_progress("status:failed:load")
            failed = _failed_result(
                mlflow_id, f"load failed: {exc}", evidence=evidence, actual=actual
            )
            log_experiment_result(failed)
            raise BenchmarkError(str(exc), result=failed) from exc
        finally:
            if gpu_started:
                gpu_summary = gpu.stop()
            if proc is not None:
                log("Stopping vLLM …")
                # Stops only our registered child and releases the GPU lease.
                release_owned(proc)

        assert load is not None
        ledger = getattr(load, "ledger", None)

        # GPU only if sampled (samples > 0). Never invent 0 util as a measurement.
        gpu_util = None
        gpu_mem = None
        gpu_samples = getattr(gpu_summary, "samples", 0) if gpu_summary is not None else 0
        if gpu_summary is not None and gpu_samples > 0:
            gpu_util = gpu_summary.avg_util_pct
            gpu_mem = gpu_summary.max_mem_used_gb

        def _lp(stat) -> LatencyPercentiles:
            return LatencyPercentiles(
                p50=stat.p50,
                p90=stat.p90,
                p95=stat.p95,
                p99=stat.p99,
                sample_n=stat.sample_n,
                sample_scope=stat.sample_scope,
            )

        if ledger is None:
            # Legacy load without raw per-request facts — do NOT forge a ledger.
            # Persist nothing canonical; TPOT stays missing.
            ledger_path = None
            request_ledger: dict = {}
            raw_ttft = list(getattr(load, "ttft_ms", []) or [])
            raw_e2e = list(getattr(load, "e2e_ms", []) or [])
            ttft_p = extract_percentiles(raw_ttft)
            e2e_p = extract_percentiles(raw_e2e)
            n_ttft = len(raw_ttft)
            n_e2e = len(raw_e2e)
            successful = int(getattr(load, "successful", 0) or 0)
            status = derive_status(
                evidence=evidence,
                actual_config=actual,
                requested_config=requested,
                successful_requests=successful,
            )
            result = ExperimentResult(
                experiment_id=cfg.experiment_id,
                config=cfg,
                total_requests=int(getattr(load, "total_requests", 0) or 0),
                successful_requests=successful,
                total_time_s=float(getattr(load, "total_time_s", 0.0) or 0.0),
                throughput_rps=getattr(load, "throughput_rps", None),
                tokens_per_second=getattr(load, "tokens_per_second", None),
                error_rate=getattr(load, "error_rate", None),
                ttft=LatencyPercentiles(
                    **ttft_p, sample_n=n_ttft, sample_scope="legacy_raw_ttft_ms"
                ),
                tpot=empty_latency(),
                e2e_latency=LatencyPercentiles(
                    **e2e_p, sample_n=n_e2e, sample_scope="legacy_raw_e2e_ms"
                ),
                gpu_memory_used_gb=gpu_mem,
                gpu_utilization_pct=gpu_util,
                raw_ttft_ms=raw_ttft,
                raw_e2e_ms=raw_e2e,
                request_ledger=request_ledger,
                ledger_path=ledger_path,
                run_id=run_id,
                schema_version="1",
                code_sha=code_sha,
                session_id=sess_str,
                mlflow_run_id=mlflow_id,
                requested_config=requested,
                actual_config=actual,
                config_evidence=evidence,
                status=status,
                workload_hash=compute_workload_hash(cfg.workload),
                hardware=hardware,
                notes="No canonical request ledger (legacy load without per-request facts).",
            )
        else:
            ledger_path = Path("logs") / f"ledger_{run_id}.json"
            persist_ledger(ledger, ledger_path)
            agg = recalculate_from_ledger(
                ledger,
                gpu_utilization_pct=gpu_util,
                gpu_memory_used_gb=gpu_mem,
            )
            status = derive_status(
                evidence=evidence,
                actual_config=actual,
                requested_config=requested,
                successful_requests=agg.successful_requests,
            )
            result = ExperimentResult(
                experiment_id=cfg.experiment_id,
                config=cfg,
                total_requests=agg.total_requests,
                successful_requests=agg.successful_requests,
                total_time_s=agg.total_time_s if agg.total_time_s is not None else 0.0,
                throughput_rps=agg.throughput_rps,
                tokens_per_second=agg.tokens_per_second,
                error_rate=agg.error_rate,
                ttft=_lp(agg.ttft),
                tpot=_lp(agg.tpot),
                e2e_latency=_lp(agg.e2e),
                gpu_memory_used_gb=gpu_mem,
                gpu_utilization_pct=gpu_util,
                raw_ttft_ms=load.ttft_ms,
                raw_e2e_ms=load.e2e_ms,
                request_ledger=ledger.model_dump(mode="json"),
                ledger_path=str(ledger_path),
                run_id=run_id,
                schema_version="1",
                code_sha=code_sha,
                session_id=sess_str,
                mlflow_run_id=mlflow_id,
                requested_config=requested,
                actual_config=actual,
                config_evidence=evidence,
                status=status,
                workload_hash=compute_workload_hash(cfg.workload),
                hardware=hardware,
                notes=format_aggregate_report(agg).strip(),
            )

        log_experiment_result(result)
        rps_s = f"{result.throughput_rps:.1f}" if result.throughput_rps is not None else "n/a"
        ttft_s = f"{result.ttft.p50:.0f}" if result.ttft.p50 is not None else "n/a"
        log(
            f"Done — {rps_s} rps, TTFT p50={ttft_s}ms, "
            f"status={result.status.value}, run_id={result.run_id}, "
            f"ledger={ledger_path}"
        )
        if on_progress:
            on_progress(f"status:{result.status.value}:complete")
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
    t.add_column("err%", justify="right")
    t.add_column("GPU util%", justify="right")
    t.add_column("GPU mem GB", justify="right")

    def _f(v: float | None, fmt: str) -> str:
        return fmt.format(v) if v is not None else "—"

    for r in results:
        t.add_row(
            r.experiment_id,
            r.config.workload.name,
            _f(r.throughput_rps, "{:.2f}"),
            _f(r.tokens_per_second, "{:.0f}"),
            _f(r.ttft.p50, "{:.0f}ms"),
            _f(r.ttft.p99, "{:.0f}ms"),
            _f(r.e2e_latency.p50, "{:.0f}ms"),
            _f(r.e2e_latency.p99, "{:.0f}ms"),
            _f(r.error_rate, "{:.1%}") if r.error_rate is not None else "—",
            _f(r.gpu_utilization_pct, "{:.0f}%"),
            _f(r.gpu_memory_used_gb, "{:.2f}"),
        )

    console.print(t)
