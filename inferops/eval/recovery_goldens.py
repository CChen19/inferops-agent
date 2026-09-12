"""Week-3 P0-⑧ recovery goldens + deterministic CPU gate.

Consumes Tune ⑧ recovery only (``inferops.agent.recovery``,
``build_graph`` checkpoint/resume, retry/budget semantics). Does not
invent a second recovery schema, loosen Week-1 ``is_promotable`` /
``derive_status``, or rewrite Tune's state machine.

CPU / fixture only. GPU-not-run is not a pass. An empty or skipped
fixture set is a fail.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from inferops.agent.confirm_campaign import confirmation_slot_experiment_id
from inferops.agent.executor import (
    confirmation_run_arm_override,
    executor_node,
    tool_boundary_overrides,
)
from inferops.agent.graph import build_graph, graph_invoke_config, session_thread_id
from inferops.agent.recovery import (
    RECOVERY_FIELDS,
    current_attempt_latest,
)
from inferops.agent.reflector import reflector_node
from inferops.agent.state import initial_state
from inferops.bench_runner import BenchmarkError, OOMError
from inferops.eval.measurement_goldens import promotable_stub_result
from inferops.eval.real_graph import ScriptedBottleneckLLM
from inferops.metrics import DEFAULT_MIN_PAIRS, RepeatArm, RepeatPhase, is_confirmed_promotable
from inferops.metrics.ledger import RunConditions
from inferops.schemas import ExperimentValidityStatus, derive_status, is_promotable
from inferops.tools.run_benchmark import RunBenchmarkInput, RunBenchmarkOutput

GOLDEN_SCHEMA = "inferops.recovery_goldens.v1"
GPU_QUEUE_ENV = "INFEROPS_GPU_GOLDENS"
INVENTED_GPU_FIELDS = ("gpu_utilization_pct", "gpu_memory_used_gb", "cost_usd")
TUNE_TIP_SHA = "d1e5e8259601ec3eca69e5852cdb3774dfd9881d"
TUNE_MASTER_SHA = "45d2d4ed5253fa29ae98cd25826b894597c16288"
TUNE_PR = "https://github.com/CChen19/inferops-agent/pull/11"

DEFAULT_FIXTURE_DIR = Path("tests/fixtures/recovery_goldens")

REQUIRED_GOLDEN_IDS: tuple[str, ...] = (
    "propose_tool_error",
    "benchmark_no_result",
    "benchmark_error_failed_result",
    "confirmation_mid_fail",
    "prior_success_current_fail",
    "pre_tool_interrupt",
    "post_persist_pre_commit_interrupt",
    "idempotent_re_resume",
    "resume_equivalence",
)

CONDITIONS = RunConditions(
    workload_name="chat_short",
    num_requests=10,
    concurrency=4,
    input_len_target=64,
    output_len_target=64,
    distribution="uniform",
    arrival_rps=None,
    warmup_requests=0,
    stream_response=True,
    sampling_temperature=0.0,
    cache_enabled=False,
)


@dataclass
class GoldenCaseResult:
    golden_id: str
    ok: bool
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    terminal: dict[str, Any] | None = None


@dataclass
class RecoveryGateResult:
    passed: bool
    failures: list[str]
    warnings: list[str]
    cases: list[GoldenCaseResult]
    gpu_status: str  # "not_run" | "queued"

    def report(self) -> str:
        lines = [
            "### Recovery golden gate",
            "",
            f"- **passed**: `{self.passed}`",
            f"- **gpu_status**: `{self.gpu_status}` "
            "(GPU-not-run is not a pass by itself)",
            f"- **tune_tip**: `{TUNE_TIP_SHA[:7]}`",
            f"- **cases**: {len(self.cases)}",
            "",
        ]
        for case in self.cases:
            mark = "ok" if case.ok else "FAIL"
            lines.append(f"- `{case.golden_id}`: {mark}")
            for fail in case.failures:
                lines.append(f"  - {fail}")
        if self.failures:
            lines.append("")
            lines.append("Gate failures:")
            for fail in self.failures:
                lines.append(f"- {fail}")
        if self.warnings:
            lines.append("")
            for warn in self.warnings:
                lines.append(f"- warning: {warn}")
        lines.append("")
        return "\n".join(lines)


def fixture_dir(root: str | Path | None = None) -> Path:
    return Path(root) if root is not None else DEFAULT_FIXTURE_DIR


def load_catalog(root: str | Path | None = None) -> dict[str, Any]:
    path = fixture_dir(root) / "catalog.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != GOLDEN_SCHEMA:
        raise ValueError(
            f"golden catalog schema {data.get('schema')!r} is not {GOLDEN_SCHEMA!r}"
        )
    return data


def load_golden_specs(root: str | Path | None = None) -> list[dict[str, Any]]:
    directory = fixture_dir(root)
    specs: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        if path.name == "catalog.json":
            continue
        specs.append(json.loads(path.read_text(encoding="utf-8")))
    return specs


def gpu_goldens_queued() -> bool:
    return os.environ.get(GPU_QUEUE_ENV) == "1"


def _walk_numeric_gpu_claims(node: Any, *, path: str = "") -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{path}.{key}" if path else str(key)
            if key in INVENTED_GPU_FIELDS and value is not None:
                found.append((child, value))
            found.extend(_walk_numeric_gpu_claims(value, path=child))
    elif isinstance(node, list):
        for i, item in enumerate(node):
            found.extend(_walk_numeric_gpu_claims(item, path=f"{path}[{i}]"))
    return found


def _refuse_invented_gpu_numbers(spec: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    sampled = bool(spec.get("gpu_sampled"))
    queued = gpu_goldens_queued()
    if sampled and not queued:
        failures.append(
            f"{spec.get('id')}: gpu_sampled=true but {GPU_QUEUE_ENV} is unset; "
            "GPU-not-run ≠ pass"
        )
    if sampled and queued:
        return failures
    for path, value in _walk_numeric_gpu_claims(spec):
        failures.append(
            f"{spec.get('id')}: invented {path}={value!r} (gpu_sampled=false)"
        )
    return failures


def _contract(**overrides: Any) -> dict[str, Any]:
    base = {
        "run_id": "aa" * 16,
        "validity_status": "valid",
        "mlflow_run_id": "mlf",
        "has_config_evidence": True,
        "promotable": True,
        "failure_reason": "",
        "error_rate": 0.0,
    }
    base.update(overrides)
    return base


def _summary(
    *,
    eid: str = "cand",
    param: str | None = "max_num_batched_tokens",
    value: Any = 4096,
    vs: float | None = 10.0,
    validity: str = "valid",
    **overrides: Any,
) -> dict[str, Any]:
    row = {
        "experiment_id": eid,
        "param_changed": param,
        "value_changed": value,
        "throughput_rps": 2.2,
        "tokens_per_second": 140.0,
        "ttft_p50_ms": 50.0,
        "ttft_p99_ms": 70.0,
        "e2e_p50_ms": 800.0,
        "bottleneck": "compute-bound",
        "vs_baseline_pct": vs,
        **_contract(validity_status=validity),
    }
    row.update(overrides)
    return row


def _state_with_prior_success() -> dict[str, Any]:
    state = initial_state("chat_short", "sess_", max_experiments=6)
    baseline = _summary(
        eid="sess_baseline", param=None, value=None, vs=0.0, run_id="bb" * 16
    )
    prior = _summary(eid="sess_prior_ok", param="max_num_seqs", value=256, vs=12.0)
    state["baseline_summary"] = baseline
    state["best_summary"] = baseline
    state["experiment_summaries"] = [baseline, prior]
    state["tried_experiment_ids"] = [baseline["experiment_id"], prior["experiment_id"]]
    state["current_bottleneck"] = "compute-bound"
    state["experiments_remaining"] = 3
    state["hypotheses"] = [
        {
            "id": "h2",
            "param": "max_num_batched_tokens",
            "value": 4096,
            "rationale": "rps=2.0 compute-bound; raise batch tokens [source: vllm_scheduler]",
            "status": "pending",
            "experiment_id": None,
        }
    ]
    return state


def _pending_search_state() -> dict[str, Any]:
    state = initial_state("chat_short", "sess_", max_experiments=6)
    baseline = _summary(
        eid="sess_baseline",
        param=None,
        value=None,
        vs=0.0,
        run_id="base-search-rid",
    )
    state["baseline_summary"] = baseline
    state["best_summary"] = baseline
    state["experiment_summaries"] = [baseline]
    state["tried_experiment_ids"] = ["sess_baseline"]
    state["current_bottleneck"] = "compute-bound"
    state["experiments_remaining"] = 5
    state["hypotheses"] = [
        {
            "id": "h1",
            "param": "max_num_batched_tokens",
            "value": 4096,
            "rationale": "rps=2.0 compute-bound; raise batch tokens [source: vllm_scheduler]",
            "status": "pending",
            "experiment_id": None,
        }
    ]
    return state


def _merge(state: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    return {**state, **patch}


def _action_kind(action: Any) -> str:
    text = str(action or "")
    for kind in (
        "propose_config",
        "run_benchmark",
        "confirmation_campaign",
        "confirmation_slot",
        "skip_duplicate",
        "reflect",
    ):
        if kind in text:
            return kind
    return text.split("(", 1)[0] or "unknown"


def comparable_terminal(state: dict[str, Any]) -> dict[str, Any]:
    """Eval comparison record — projection of AgentState + ①/⑤/⑥ gates."""
    best = state.get("best_summary") or {}
    last = state.get("last_result")
    decision = state.get("confirmation_decision")
    last_rid = getattr(last, "run_id", None) if last is not None else None
    last_status = getattr(last, "status", None) if last is not None else None
    if hasattr(last_status, "value"):
        last_status = last_status.value
    traj_identity = []
    for step in state.get("trajectory") or []:
        traj_identity.append(
            {
                "node": step.get("node"),
                "action_kind": _action_kind(step.get("action")),
                "experiment_id": step.get("experiment_id"),
                "run_id": step.get("run_id"),
                "validity_status": step.get("validity_status"),
                "promoted_to_best": bool(
                    (step.get("result") or {}).get("promoted_to_best")
                ),
            }
        )
    reflect_steps = [
        s for s in (state.get("trajectory") or []) if s.get("node") == "reflector"
    ]
    cited = list((reflect_steps[-1] or {}).get("cited_run_ids") or []) if reflect_steps else []
    return {
        "stop_reason": state.get("stop_reason") or "",
        "should_stop": bool(state.get("should_stop")),
        "next_action": state.get("next_action") or "",
        "best_experiment_id": best.get("experiment_id"),
        "best_run_id": best.get("run_id"),
        "best_promotable": bool(best.get("promotable")),
        "best_confirmed_promotable": bool(best.get("confirmed_promotable")),
        "confirmed_gate": bool(is_confirmed_promotable(last, decision)),
        "experiments_remaining": int(state.get("experiments_remaining") or 0),
        "tried_experiment_ids": list(state.get("tried_experiment_ids") or []),
        "summary_run_ids": [
            str(s.get("run_id") or "")
            for s in (state.get("experiment_summaries") or [])
            if s.get("run_id")
        ],
        "last_result_run_id": last_rid,
        "last_result_validity_status": last_status,
        "confirmation_verdict": getattr(getattr(decision, "verdict", None), "value", None)
        if decision is not None
        else None,
        "confirmation_phase": getattr(getattr(decision, "phase", None), "value", None)
        if decision is not None
        else None,
        "confirmation_search_winner": bool(getattr(decision, "search_winner", False))
        if decision is not None
        else False,
        "confirmation_bound_run_ids": list(state.get("confirmation_bound_run_ids") or []),
        "reflect_cited_run_ids": cited,
        "trajectory_identity": traj_identity,
    }


def _assert_recovery_fields(event: dict[str, Any] | None, golden_id: str) -> list[str]:
    failures: list[str] = []
    if not event:
        return [f"{golden_id}: missing Tune last_recovery event"]
    for key in RECOVERY_FIELDS:
        if key not in event:
            failures.append(f"{golden_id}: Tune recovery missing field {key}")
    if event.get("this_attempt_failed") is not True:
        failures.append(f"{golden_id}: this_attempt_failed is not True")
    dumped = json.dumps(event, default=str)
    for banned in ("gpu_utilization", "gpu_memory", "cost_usd", "request_ledger"):
        if banned in dumped:
            failures.append(f"{golden_id}: recovery event smuggled {banned}")
    return failures


def _no_false_promote(state: dict[str, Any], golden_id: str) -> list[str]:
    failures: list[str] = []
    best = state.get("best_summary") or {}
    if best.get("confirmed_promotable") is True:
        failures.append(f"{golden_id}: best.confirmed_promotable=true (false promote)")
    last = state.get("last_result")
    decision = state.get("confirmation_decision")
    if is_confirmed_promotable(last, decision):
        failures.append(f"{golden_id}: is_confirmed_promotable True on recovery path")
    for step in state.get("trajectory") or []:
        if (step.get("result") or {}).get("promoted_to_best") is True:
            failures.append(f"{golden_id}: trajectory promoted_to_best")
    return failures


def _exec_then_reflect(state: dict[str, Any], exec_patch: dict[str, Any]) -> dict[str, Any]:
    merged = _merge(state, exec_patch)
    refl = reflector_node(merged)
    return _merge(merged, refl)


# ---------------------------------------------------------------------------
# Drivers — production nodes + Tune ⑧ entrypoints, tool-boundary stubs only
# ---------------------------------------------------------------------------

def _drive_propose_tool_error() -> dict[str, Any]:
    state = _state_with_prior_success()
    state["last_result"] = object()
    with patch("inferops.agent.executor.get_result_by_id", return_value=None), patch(
        "inferops.tools.propose_config.propose_config_patch",
        side_effect=RuntimeError("propose backend down"),
    ):
        exec_patch = executor_node(state)
    final = _exec_then_reflect(state, exec_patch)
    return {
        "state": final,
        "exec_patch": exec_patch,
        "budget_before": state["experiments_remaining"],
        "prior_run_id": state["experiment_summaries"][-1]["run_id"],
    }


def _drive_benchmark_no_result() -> dict[str, Any]:
    state = _pending_search_state()
    with patch("inferops.agent.executor.get_result_by_id", return_value=None), patch(
        "inferops.tools.propose_config.propose_config_patch"
    ), patch(
        "inferops.agent.executor.run_benchmark",
        side_effect=BenchmarkError("no contract row"),
    ):
        exec_patch = executor_node(state)
    final = _exec_then_reflect(state, exec_patch)
    return {
        "state": final,
        "exec_patch": exec_patch,
        "budget_before": state["experiments_remaining"],
    }


def _drive_benchmark_error_failed_result() -> dict[str, Any]:
    stub = promotable_stub_result()
    failed = stub.model_copy(
        update={
            "experiment_id": "sess_max_num_batched_tokens_4096",
            "status": ExperimentValidityStatus.FAILED,
            "notes": "vLLM OOM during startup",
            "successful_requests": 0,
            "error_rate": 1.0,
            "gpu_memory_used_gb": None,
            "gpu_utilization_pct": None,
        }
    )
    state = _pending_search_state()
    state["experiment_summaries"] = [
        state["baseline_summary"],
        _summary(eid="sess_prior_ok", param="max_num_seqs", value=256, vs=12.0),
    ]
    with patch("inferops.agent.executor.get_result_by_id", return_value=None), patch(
        "inferops.tools.propose_config.propose_config_patch"
    ), patch(
        "inferops.agent.executor.run_benchmark",
        side_effect=OOMError("vLLM OOM during startup", result=failed),
    ):
        exec_patch = executor_node(state)
    final = _exec_then_reflect(state, exec_patch)
    return {
        "state": final,
        "exec_patch": exec_patch,
        "failed": failed,
        "budget_before": state["experiments_remaining"],
    }


def _drive_confirmation_mid_fail() -> dict[str, Any]:
    from inferops.eval.measurement_goldens import rps_ledger

    stub = promotable_stub_result()
    state = _state_with_prior_success()
    state["hypotheses"][0]["param"] = "max_num_batched_tokens"
    state["hypotheses"][0]["value"] = 4096
    state["hypotheses"][0]["status"] = "pending"
    state["next_action"] = "remeasure"
    state["confirmation_target"] = {"param": "max_num_batched_tokens", "value": "4096"}
    state["last_result"] = stub
    state["repeat_ledgers"] = {
        "baseline": [],
        "candidate": [],
        "phase": RepeatPhase.SEARCH,
        "metric": "throughput_rps",
        "conditions": CONDITIONS,
    }
    completed: list[str] = []

    class _Arm:
        last_candidate_result = None

        def __call__(self, arm: RepeatArm, slot: Any) -> Any:
            if arm == RepeatArm.BASELINE:
                lg = rps_ledger(f"cf-b{slot.pair_index}", rps=2.0)
                completed.append(lg.run_id)
                return lg
            raise RuntimeError("candidate slot boom")

    with confirmation_run_arm_override(_Arm()):
        exec_patch = executor_node(state)
    final = _exec_then_reflect(state, exec_patch)
    return {
        "state": final,
        "exec_patch": exec_patch,
        "completed": completed,
        "budget_before": state["experiments_remaining"],
        "stub": stub,
    }


def _drive_prior_success_current_fail() -> dict[str, Any]:
    stub = promotable_stub_result()
    state = _state_with_prior_success()
    state["last_result"] = stub
    prior = state["experiment_summaries"][-1]
    with patch("inferops.agent.executor.get_result_by_id", return_value=None), patch(
        "inferops.tools.propose_config.propose_config_patch"
    ), patch(
        "inferops.agent.executor.run_benchmark",
        side_effect=RuntimeError("bench exploded"),
    ):
        exec_patch = executor_node(state)
    final = _exec_then_reflect(state, exec_patch)
    return {
        "state": final,
        "exec_patch": exec_patch,
        "prior": prior,
        "budget_before": state["experiments_remaining"],
    }


def _ledger_backed(template: Any, *, experiment_id: str, run_id: str, rps: float) -> Any:
    from inferops.eval.measurement_goldens import rps_ledger

    ledger = rps_ledger(run_id, rps=rps)
    return template.model_copy(
        update={
            "experiment_id": experiment_id,
            "run_id": run_id,
            "request_ledger": ledger.model_dump(mode="json"),
            "throughput_rps": rps,
            "error_rate": 0.0,
            "gpu_memory_used_gb": None,
            "gpu_utilization_pct": None,
        }
    )


def _graph_store(template: Any) -> tuple[dict[str, Any], Any, list[str]]:
    store: dict[str, Any] = {}
    calls: list[str] = []
    store["sess_baseline"] = _ledger_backed(
        template, experiment_id="sess_baseline", run_id="base-search-rid", rps=2.0
    )

    def run_benchmark_fn(inp: RunBenchmarkInput) -> RunBenchmarkOutput:
        calls.append(inp.experiment_id)
        rps = 2.0 if not inp.config_patch else 2.4
        rid = f"{inp.experiment_id}-rid"
        result = _ledger_backed(
            template, experiment_id=inp.experiment_id, run_id=rid, rps=rps
        )
        store[inp.experiment_id] = result
        return RunBenchmarkOutput(
            experiment_id=inp.experiment_id,
            workload_name=inp.workload_name,
            throughput_rps=rps,
            tokens_per_second=152.0,
            ttft_p50_ms=52.0,
            ttft_p99_ms=66.0,
            e2e_p50_ms=780.0,
            e2e_p99_ms=870.0,
            gpu_util_pct=None,
            gpu_mem_gb=None,
            success_rate="10/10",
            mlflow_run_id="mlflow-test-b",
            run_id=rid,
            status="valid",
        )

    return store, run_benchmark_fn, calls


def _graph_start_state() -> dict[str, Any]:
    state = initial_state("chat_short", "sess_", max_experiments=2)
    baseline = _summary(
        eid="sess_baseline",
        param=None,
        value=None,
        vs=0.0,
        run_id="base-search-rid",
    )
    state["baseline_summary"] = baseline
    state["best_summary"] = baseline
    state["experiment_summaries"] = [baseline]
    state["tried_experiment_ids"] = ["sess_baseline"]
    state["current_bottleneck"] = "compute-bound"
    state["experiments_remaining"] = 1
    return state


def _run_production_graph(
    *,
    interrupt_before: list[str] | None = None,
    crash_after_persist: bool = False,
    resume_times: int = 1,
) -> dict[str, Any]:
    from langgraph.checkpoint.memory import MemorySaver

    template = promotable_stub_result()
    store, bench, calls = _graph_store(template)
    if crash_after_persist:
        inner = bench

        def bench(inp: RunBenchmarkInput) -> Any:  # type: ignore[no-redef]
            out = inner(inp)
            if len(calls) == 1 and "confirm_" not in inp.experiment_id:
                raise KeyboardInterrupt("persist-then-interrupt")
            return out

    llm = ScriptedBottleneckLLM(default_bottleneck="compute-bound")
    checkpointer = MemorySaver()
    graph = build_graph(
        llm, checkpointer=checkpointer, interrupt_before=interrupt_before
    )
    config = graph_invoke_config("sess_")
    start = _graph_start_state()

    def _invoke(payload: Any) -> Any:
        with tool_boundary_overrides(
            run_benchmark_fn=bench,
            propose_config_fn=lambda _inp: None,
        ), patch(
            "inferops.agent.executor.get_result_by_id",
            side_effect=lambda e: store.get(e),
        ), patch(
            "inferops.agent.executor.analyze_bottleneck",
            return_value=MagicMock(bottleneck="compute-bound"),
        ), patch(
            "inferops.agent.executor.compare_experiments",
            return_value=MagicMock(delta_pct=20.0),
        ):
            return graph.invoke(payload, config)

    interrupted = False
    try:
        state = _invoke(start)
    except KeyboardInterrupt:
        interrupted = True
        state = graph.get_state(config).values
    if interrupt_before:
        snap = graph.get_state(config)
        if snap.next != ("executor",):
            raise AssertionError(f"expected interrupt before executor, got {snap.next}")
        if calls:
            raise AssertionError(f"tool ran during pre-tool interrupt: {calls}")
        state = _invoke(None)
        for _ in range(max(0, resume_times - 1)):
            state = _invoke(None)
    elif interrupted:
        state = _invoke(None)
        for _ in range(max(0, resume_times - 1)):
            state = _invoke(None)
    return {"state": state, "calls": calls, "config": config, "store": store, "graph": graph}


def _drive_pre_tool_interrupt() -> dict[str, Any]:
    return _run_production_graph(interrupt_before=["executor"])


def _drive_post_persist_pre_commit() -> dict[str, Any]:
    stub = promotable_stub_result()
    store: dict[str, Any] = {}
    calls: list[str] = []
    eid = "sess_max_num_batched_tokens_4096"
    result = stub.model_copy(
        update={
            "experiment_id": eid,
            "gpu_memory_used_gb": None,
            "gpu_utilization_pct": None,
        }
    )

    def _bench(inp: RunBenchmarkInput) -> Any:
        calls.append(inp.experiment_id)
        stored = result.model_copy(update={"experiment_id": inp.experiment_id})
        store[inp.experiment_id] = stored
        raise KeyboardInterrupt("persist-then-interrupt")

    state = _pending_search_state()
    with tool_boundary_overrides(
        run_benchmark_fn=_bench,
        propose_config_fn=lambda _inp: None,
    ), patch(
        "inferops.agent.executor.get_result_by_id",
        side_effect=lambda e: store.get(e),
    ), patch(
        "inferops.agent.executor.analyze_bottleneck",
        return_value=MagicMock(bottleneck="compute-bound"),
    ), patch(
        "inferops.agent.executor.compare_experiments",
        return_value=MagicMock(delta_pct=19.0),
    ):
        try:
            executor_node(state)
            raise AssertionError("expected KeyboardInterrupt after persist")
        except KeyboardInterrupt:
            pass
        resume = executor_node(state)
    final = _exec_then_reflect(state, resume)
    confirm = _drive_confirm_persist_resume()
    return {
        "state": final,
        "exec_patch": resume,
        "calls": calls,
        "store": store,
        "eid": eid,
        "budget_before": state["experiments_remaining"],
        "confirm_persist": confirm,
    }


def _drive_confirm_persist_resume() -> dict[str, Any]:
    """Tune ⑧ confirm-slot persist-then-crash: reuse rows, budget −1 once."""
    template = promotable_stub_result()
    store: dict[str, Any] = {}
    calls: list[str] = []
    crash_after = 2
    store["sess_baseline"] = _ledger_backed(
        template, experiment_id="sess_baseline", run_id="base-search-rid", rps=2.0
    )

    def bench(inp: RunBenchmarkInput) -> Any:
        calls.append(inp.experiment_id)
        rps = 2.0 if not inp.config_patch else 2.4
        rid = f"{inp.experiment_id}-rid"
        result = _ledger_backed(
            template, experiment_id=inp.experiment_id, run_id=rid, rps=rps
        )
        store[inp.experiment_id] = result
        if len(calls) == crash_after:
            raise KeyboardInterrupt("mid-confirm persist")
        return RunBenchmarkOutput(
            experiment_id=inp.experiment_id,
            workload_name=inp.workload_name,
            throughput_rps=rps,
            tokens_per_second=152.0,
            ttft_p50_ms=52.0,
            ttft_p99_ms=66.0,
            e2e_p50_ms=780.0,
            e2e_p99_ms=870.0,
            gpu_util_pct=None,
            gpu_mem_gb=None,
            success_rate="10/10",
            mlflow_run_id="mlflow-test-b",
            run_id=rid,
            status="valid",
        )

    state = _pending_search_state()
    state["next_action"] = "remeasure"
    state["remeasure_count"] = 1
    state["confirmation_target"] = {
        "param": "max_num_batched_tokens",
        "value": "4096",
    }
    state["hypotheses"][0]["status"] = "pending"
    state["experiments_remaining"] = 2
    pre_budget = state["experiments_remaining"]
    expected_slot = confirmation_slot_experiment_id(
        session_prefix="sess_",
        hypothesis=state["hypotheses"][0],
        arm=RepeatArm.CANDIDATE,
        pair_index=1,
        remasure_count=1,
    )

    with tool_boundary_overrides(
        run_benchmark_fn=bench,
        propose_config_fn=lambda _inp: None,
    ), patch(
        "inferops.agent.executor.get_result_by_id",
        side_effect=lambda e: store.get(e),
    ):
        try:
            executor_node(state)
            raise AssertionError("expected KeyboardInterrupt mid-confirm")
        except KeyboardInterrupt:
            pass
        first = list(calls)
        resume = executor_node(state)

    return {
        "calls": calls,
        "first": first,
        "resume": resume,
        "pre_budget": pre_budget,
        "expected_slot": expected_slot,
    }


def _drive_idempotent_re_resume() -> dict[str, Any]:
    first = _run_production_graph(interrupt_before=["executor"], resume_times=1)
    second = _run_production_graph(interrupt_before=["executor"], resume_times=2)
    return {
        "state": second["state"],
        "first": first,
        "second": second,
        "calls": second["calls"],
    }


def _drive_resume_equivalence() -> dict[str, Any]:
    plain = _run_production_graph()
    resumed = _run_production_graph(interrupt_before=["executor"])
    return {
        "state": resumed["state"],
        "uninterrupted": plain,
        "resumed": resumed,
    }


DRIVERS = {
    "propose_tool_error": _drive_propose_tool_error,
    "benchmark_no_result": _drive_benchmark_no_result,
    "benchmark_error_failed_result": _drive_benchmark_error_failed_result,
    "confirmation_mid_fail": _drive_confirmation_mid_fail,
    "prior_success_current_fail": _drive_prior_success_current_fail,
    "pre_tool_interrupt": _drive_pre_tool_interrupt,
    "post_persist_pre_commit_interrupt": _drive_post_persist_pre_commit,
    "idempotent_re_resume": _drive_idempotent_re_resume,
    "resume_equivalence": _drive_resume_equivalence,
}


def evaluate_golden(spec: dict[str, Any]) -> GoldenCaseResult:
    golden_id = str(spec.get("id") or "unknown")
    failures = _refuse_invented_gpu_numbers(spec)
    notes: list[str] = []
    if spec.get("skip") or spec.get("skipped"):
        failures.append(f"{golden_id}: skipped golden is not a pass")
    if spec.get("schema") != GOLDEN_SCHEMA:
        failures.append(f"{golden_id}: schema {spec.get('schema')!r} != {GOLDEN_SCHEMA!r}")
    if spec.get("synthetic") is not True:
        failures.append(f"{golden_id}: fixture must be synthetic")
    if spec.get("gpu_sampled") is True and not gpu_goldens_queued():
        failures.append(f"{golden_id}: GPU-not-run ≠ pass")
    if (spec.get("expect") or {}).get("confirmed_promotable") is True:
        failures.append(
            f"{golden_id}: this thin set must not claim confirmed_promotable=true"
        )

    driver_name = str(spec.get("driver") or golden_id)
    driver = DRIVERS.get(driver_name)
    if driver is None:
        failures.append(f"{golden_id}: unknown driver {driver_name!r}")
        return GoldenCaseResult(golden_id=golden_id, ok=False, failures=failures)

    try:
        payload = driver()
    except Exception as exc:  # noqa: BLE001 — gate records the failure
        return GoldenCaseResult(
            golden_id=golden_id,
            ok=False,
            failures=failures + [f"{golden_id}: driver raised {type(exc).__name__}: {exc}"],
        )

    state = payload["state"]
    terminal = comparable_terminal(state)
    expect = spec.get("expect") or {}
    failures.extend(_no_false_promote(state, golden_id))

    exec_patch = payload.get("exec_patch") or {}
    event = exec_patch.get("last_recovery") or state.get("last_recovery")
    traj = list(state.get("trajectory") or [])
    if event is None:
        for step in reversed(traj):
            if step.get("recovery"):
                event = step["recovery"]
                break

    if expect.get("require_recovery"):
        failures.extend(_assert_recovery_fields(event, golden_id))
        if event and expect.get("recovery_code") and event.get("code") != expect["recovery_code"]:
            failures.append(
                f"{golden_id}: recovery.code={event.get('code')!r}, "
                f"expected {expect['recovery_code']!r}"
            )
        if event and "result_persisted" in expect:
            if bool(event.get("result_persisted")) != bool(expect["result_persisted"]):
                failures.append(
                    f"{golden_id}: result_persisted={event.get('result_persisted')}, "
                    f"expected {expect['result_persisted']}"
                )
        if event and "budget_consumed" in expect:
            if bool(event.get("budget_consumed")) != bool(expect["budget_consumed"]):
                failures.append(
                    f"{golden_id}: budget_consumed={event.get('budget_consumed')}, "
                    f"expected {expect['budget_consumed']}"
                )

    if "next_action" in expect and state.get("next_action") != expect["next_action"]:
        # Graph stop paths use stop_reason; node paths use next_action.
        if expect.get("next_action_or_stop"):
            if state.get("next_action") != expect["next_action"] and not state.get(
                "should_stop"
            ):
                failures.append(
                    f"{golden_id}: next_action={state.get('next_action')!r} "
                    f"and should_stop={state.get('should_stop')}"
                )
        else:
            failures.append(
                f"{golden_id}: next_action={state.get('next_action')!r}, "
                f"expected {expect['next_action']!r}"
            )

    if expect.get("no_forged_summary"):
        if "experiment_summaries" in exec_patch:
            failures.append(f"{golden_id}: forged experiment_summaries on no-result path")
        if exec_patch.get("last_result") is not None:
            failures.append(f"{golden_id}: last_result invented without persist")

    if expect.get("keep_failed_result"):
        failed = payload.get("failed")
        summaries = exec_patch.get("experiment_summaries") or state.get("experiment_summaries")
        last_summary = (summaries or [])[-1] if summaries else None
        if last_summary is None or last_summary.get("validity_status") != "failed":
            failures.append(f"{golden_id}: failed contract row not kept")
        elif failed is not None and last_summary.get("run_id") != failed.run_id:
            failures.append(f"{golden_id}: failed run_id dropped")
        if failed is not None and is_promotable(failed):
            failures.append(f"{golden_id}: failed row became is_promotable")
        latest = current_attempt_latest(
            list(exec_patch.get("experiment_summaries") or state.get("experiment_summaries") or []),
            exec_patch.get("last_recovery"),
        )
        if latest is None:
            failures.append(f"{golden_id}: current_attempt_latest ignored persisted fail")

    if expect.get("no_stale_latest"):
        latest = current_attempt_latest(
            list(state.get("experiment_summaries") or []),
            exec_patch.get("last_recovery") or event,
        )
        if latest is not None:
            failures.append(
                f"{golden_id}: Reflect latest is stale prior success "
                f"{latest.get('experiment_id')}"
            )
        prior = payload.get("prior") or {}
        cited = terminal["reflect_cited_run_ids"]
        # Prior success may appear as best/baseline history, but current
        # confirmation bind must be empty and latest must not be that row.
        if state.get("confirmation_decision") is not None:
            failures.append(f"{golden_id}: stale confirmation_decision survived")
        if prior.get("run_id") and latest is not None and latest.get("run_id") == prior.get("run_id"):
            failures.append(f"{golden_id}: Reflect cited prior success as current")
        notes.append(f"cited_run_ids={cited}")

    if expect.get("partial_not_confirmed"):
        if exec_patch.get("confirmation_decision") is not None:
            failures.append(f"{golden_id}: partial campaign minted a ⑤ decision")
        stub = payload.get("stub")
        if stub is not None and is_confirmed_promotable(stub, None):
            failures.append(f"{golden_id}: search/partial became confirmed")
        completed = payload.get("completed") or []
        rec_ids = list((event or {}).get("cited_run_ids") or [])
        if completed and completed[0] not in rec_ids:
            failures.append(f"{golden_id}: completed confirm slot run_id not recorded")

    if expect.get("single_tool_call"):
        calls = [c for c in (payload.get("calls") or []) if "confirm_" not in c]
        if len(calls) != 1:
            failures.append(f"{golden_id}: expected one tool call, got {calls}")

    confirm_persist = payload.get("confirm_persist")
    if confirm_persist:
        first = list(confirm_persist.get("first") or [])
        calls = list(confirm_persist.get("calls") or [])
        resume = confirm_persist.get("resume") or {}
        expected_slots = DEFAULT_MIN_PAIRS * 2
        if len(first) != 2:
            failures.append(f"{golden_id}: confirm persist crash after {len(first)} slots")
        if any("_r1_" not in eid for eid in calls):
            failures.append(f"{golden_id}: confirm slots missing Tune remasure identity")
        expected_slot = confirm_persist.get("expected_slot")
        if expected_slot != "sess_confirm_max_num_batched_tokens_4096_r1_c1":
            failures.append(
                f"{golden_id}: Tune confirmation_slot_experiment_id drifted "
                f"{expected_slot!r}"
            )
        if calls[:2] != first:
            failures.append(f"{golden_id}: confirm persist-resume did not reuse first slots")
        if len(calls) != expected_slots or len(set(calls)) != expected_slots:
            failures.append(
                f"{golden_id}: confirm resume duplicate/missing slots {calls}"
            )
        if resume.get("experiments_remaining") != confirm_persist["pre_budget"] - 1:
            failures.append(
                f"{golden_id}: confirm campaign budget not charged once "
                f"({resume.get('experiments_remaining')} vs "
                f"{confirm_persist['pre_budget'] - 1})"
            )
        if resume.get("last_recovery") is not None:
            failures.append(f"{golden_id}: successful confirm resume left last_recovery")

    if expect.get("no_double_budget") and "budget_before" in payload:
        before = int(payload["budget_before"])
        after = int(state["experiments_remaining"])
        debit = 0 if expect.get("budget_consumed") is False else 1
        if after != before - debit:
            failures.append(
                f"{golden_id}: budget {before}→{after}, expected debit {debit}"
            )

    if golden_id == "propose_tool_error" and "experiments_remaining" in exec_patch:
        failures.append(f"{golden_id}: propose reject consumed budget")

    if golden_id == "idempotent_re_resume":
        t1 = comparable_terminal(payload["first"]["state"])
        t2 = comparable_terminal(payload["second"]["state"])
        for key in (
            "stop_reason",
            "best_experiment_id",
            "experiments_remaining",
            "tried_experiment_ids",
            "summary_run_ids",
        ):
            if t1[key] != t2[key]:
                failures.append(
                    f"{golden_id}: re-resume {key} {t1[key]!r} != {t2[key]!r}"
                )
        c1 = [c for c in payload["first"]["calls"] if "confirm_" not in c]
        c2 = [c for c in payload["second"]["calls"] if "confirm_" not in c]
        if c1 != c2 or len(c2) != 1:
            failures.append(f"{golden_id}: re-resume duplicate attempt {c1} vs {c2}")

    if golden_id == "resume_equivalence":
        tu = comparable_terminal(payload["uninterrupted"]["state"])
        tr = comparable_terminal(payload["resumed"]["state"])
        for key in (
            "stop_reason",
            "should_stop",
            "best_experiment_id",
            "best_confirmed_promotable",
            "confirmed_gate",
            "experiments_remaining",
            "tried_experiment_ids",
        ):
            if tu[key] != tr[key]:
                failures.append(
                    f"{golden_id}: U vs R {key} {tu[key]!r} != {tr[key]!r}"
                )
        u_keys = [
            (k["node"], k["action_kind"], k["experiment_id"])
            for k in tu["trajectory_identity"]
            if k["node"] == "executor"
        ]
        r_keys = [
            (k["node"], k["action_kind"], k["experiment_id"])
            for k in tr["trajectory_identity"]
            if k["node"] == "executor"
        ]
        if u_keys != r_keys:
            failures.append(f"{golden_id}: executor trajectory identity {u_keys} != {r_keys}")
        notes.append(f"U.stop={tu['stop_reason']!r} R.stop={tr['stop_reason']!r}")

    if expect.get("week1_gate_closed"):
        stub = promotable_stub_result()
        if not is_promotable(stub):
            failures.append(f"{golden_id}: loosened is_promotable on valid stub")
        if derive_status(
            evidence=stub.config_evidence,
            actual_config=stub.actual_config,
            requested_config=stub.requested_config,
            successful_requests=stub.successful_requests,
        ).value != "valid":
            failures.append(f"{golden_id}: derive_status loosened")

    notes.append(
        f"next_action={state.get('next_action')} "
        f"stop_reason={state.get('stop_reason') or '-'} "
        f"budget={state.get('experiments_remaining')}"
    )
    return GoldenCaseResult(
        golden_id=golden_id,
        ok=not failures,
        failures=failures,
        notes=notes,
        terminal=terminal,
    )


def recovery_golden_gate(
    root: str | Path | None = None,
) -> RecoveryGateResult:
    """Deterministic CI gate. Empty / GPU-not-run is not a pass."""
    failures: list[str] = []
    warnings: list[str] = []
    gpu_status = "queued" if gpu_goldens_queued() else "not_run"

    catalog = load_catalog(root)
    contract = catalog.get("tune_contract") or {}
    if contract.get("status") != "frozen":
        failures.append(
            "Blocked: waiting on Tune ⑧ interface freeze "
            f"(catalog.tune_contract.status={contract.get('status')!r})"
        )
    if contract.get("tip_sha") != TUNE_TIP_SHA:
        failures.append(
            f"tune tip_sha {contract.get('tip_sha')!r} != frozen {TUNE_TIP_SHA!r}"
        )
    if contract.get("master_sha") not in (None, TUNE_MASTER_SHA):
        failures.append(
            f"tune master_sha {contract.get('master_sha')!r} != frozen {TUNE_MASTER_SHA!r}"
        )
    if catalog.get("gpu_queued") and gpu_status != "queued":
        failures.append(
            "catalog.gpu_queued=true but GPU goldens were not queued; "
            "GPU-not-run ≠ pass"
        )
    if gpu_status == "not_run" and catalog.get("cpu_only") is not True:
        failures.append("CPU-only catalog required while GPU is not queued")

    required = tuple(catalog.get("required_ids") or REQUIRED_GOLDEN_IDS)
    specs = load_golden_specs(root)
    by_id = {str(spec.get("id")): spec for spec in specs}
    missing = [gid for gid in required if gid not in by_id]
    if missing:
        failures.append(f"required goldens missing: {missing}")
    if not specs:
        failures.append("GPU-not-run ≠ pass: no CPU goldens evaluated")
    skipped_ids = [
        gid
        for gid, spec in by_id.items()
        if spec.get("skip") or spec.get("skipped")
    ]
    if skipped_ids:
        failures.append(f"skipped fixture set is not a pass: {skipped_ids}")

    extra = sorted(set(by_id) - set(required) - set(REQUIRED_GOLDEN_IDS))
    if extra:
        warnings.append(f"extra golden ids (allowed): {extra}")

    cases = [evaluate_golden(spec) for spec in specs]
    for case in cases:
        failures.extend(case.failures)

    if any((spec.get("expect") or {}).get("confirmed_promotable") is True for spec in specs):
        failures.append(
            "this thin golden set must not claim confirmed_promotable=true; "
            "do not loosen ① / ⑤ gates"
        )

    return RecoveryGateResult(
        passed=not failures,
        failures=failures,
        warnings=warnings,
        cases=cases,
        gpu_status=gpu_status,
    )
