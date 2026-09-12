"""Week-3 P0-⑧: interrupt / tool-error recovery — fixture / CPU only.

GPU-not-run is not a pass. No invented GPU numbers. Week-1 gates stay
fail-closed. ④/⑤ schema and ⑦ goldens are not redefined here.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from inferops.agent.confirm_campaign import confirmation_slot_experiment_id
from inferops.agent.executor import (
    confirmation_run_arm_override,
    executor_node,
    tool_boundary_overrides,
)
from inferops.agent.graph import (
    build_graph,
    graph_invoke_config,
    session_thread_id,
)
from inferops.agent.recovery import (
    RECOVERY_FIELDS,
    current_attempt_latest,
    is_hard_control_exception,
    recovery_event,
)
from inferops.agent.reflect_constraints import MAX_REMEASURES
from inferops.metrics import (
    DEFAULT_MIN_PAIRS,
    RepeatArm,
    RepeatPhase,
    is_confirmed_promotable,
    verdict_from_ledgers,
)
from inferops.agent.reflector import reflector_node
from inferops.agent.state import initial_state
from inferops.bench_runner import BenchmarkError, OOMError
from inferops.schemas import ExperimentValidityStatus, is_promotable
from inferops.tools.run_benchmark import RunBenchmarkInput, RunBenchmarkOutput
from tests.test_constrained_reflect import (
    _ledger_backed_result,
    _pending_search_state,
    _run_search_remeasure_confirm,
    _state_with_candidate,
    _summary,
    _tool_boundary_store,
)
from tests.test_repeat_confirmation import CONDITIONS, _n_ledgers, make_rps_ledger


def _assert_recovery_contract(event: dict) -> None:
    for key in RECOVERY_FIELDS:
        assert key in event
    assert event["this_attempt_failed"] is True


def _merge(state, patch):
    return {**state, **patch}


def _assert_not_confirmed_promoted(patch: dict) -> None:
    best = patch.get("best_summary")
    if best:
        assert best.get("confirmed_promotable") is not True
    traj = patch.get("trajectory") or []
    if traj:
        assert traj[-1].get("result", {}).get("promoted_to_best") is not True


# ---------------------------------------------------------------------------
# Recovery contract + latest-selection
# ---------------------------------------------------------------------------

def test_recovery_event_has_required_fields_only_no_metrics_schema():
    event = recovery_event(
        experiment_id="sess_x",
        hypothesis={"id": "h1", "param": "max_num_seqs", "value": 64, "rationale": "r"},
        stage="run_benchmark",
        reason="boom",
        code="tool_exception",
        result_persisted=False,
        budget_consumed=True,
        retryable=False,
        next_action="rollback",
    )
    for key in RECOVERY_FIELDS:
        assert key in event
    assert event["attempt_id"] == "sess_x"
    assert event["tool"] == "run_benchmark"
    assert "throughput_rps" not in event
    assert "gpu_util_pct" not in event
    assert "request_ledger" not in event


def test_current_attempt_latest_ignores_stale_prior_success():
    prior = _summary(eid="sess_prior", vs=19.0, run_id="ok" * 16)
    recovery = recovery_event(
        experiment_id="sess_now",
        hypothesis={"id": "h2", "param": "max_num_seqs", "value": 64},
        stage="run_benchmark",
        reason="boom",
        code="tool_exception",
        result_persisted=False,
        budget_consumed=True,
        retryable=False,
        next_action="rollback",
    )
    assert current_attempt_latest([prior], recovery) is None
    persisted = _summary(eid="sess_now", validity="failed", promotable=False, vs=None)
    recovery["result_persisted"] = True
    recovery["experiment_id"] = "sess_now"
    assert current_attempt_latest([prior, persisted], recovery) is persisted


# ---------------------------------------------------------------------------
# Propose rejection
# ---------------------------------------------------------------------------

def test_propose_rejection_emits_recovery_and_does_not_promote():
    state = _state_with_candidate()
    prior = state["experiment_summaries"][-1]
    state["last_result"] = object()
    # Different (param, value) so this is a new attempt, not a duplicate skip.
    state["hypotheses"][0]["status"] = "pending"
    state["hypotheses"][0]["param"] = "max_num_seqs"
    state["hypotheses"][0]["value"] = 64
    state["hypotheses"][0]["experiment_id"] = None
    state["experiment_summaries"] = [state["baseline_summary"], prior]

    with patch("inferops.agent.executor.get_result_by_id", return_value=None), patch(
        "inferops.tools.propose_config.propose_config_patch",
        side_effect=ValueError("max_num_batched_tokens=99999 outside safe range"),
    ):
        exec_patch = executor_node(state)

    assert exec_patch["hypotheses"][0]["status"] == "failed"
    assert "experiments_remaining" not in exec_patch
    assert exec_patch["last_result"] is None
    assert exec_patch["confirmation_decision"] is None
    event = exec_patch["last_recovery"]
    _assert_recovery_contract(event)
    assert event["code"] == "propose_rejected"
    assert event["result_persisted"] is False
    assert event["budget_consumed"] is False
    assert "experiment_summaries" not in exec_patch
    assert exec_patch["trajectory"][-1]["result"]["promoted_to_best"] is False

    merged = _merge(state, exec_patch)
    assert current_attempt_latest(merged["experiment_summaries"], event) is None
    refl = reflector_node(merged)
    step = refl["trajectory"][-1]
    assert step["next_action"] == "rollback"
    assert step["constraint_checks"]["this_attempt_failed"] is True
    assert step["result"]["promoted_to_best"] is False
    assert refl["best_summary"]["experiment_id"] == "sess_baseline"
    _assert_not_confirmed_promoted(refl)


# ---------------------------------------------------------------------------
# Generic benchmark exception (no result) + stale prior success
# ---------------------------------------------------------------------------

def test_generic_benchmark_exception_does_not_forge_or_read_stale_success(result_b):
    state = _state_with_candidate(vs=19.0, run_id=result_b.run_id)
    state["last_result"] = result_b
    state["last_recovery"] = None
    state["hypotheses"][0]["status"] = "pending"
    state["hypotheses"][0]["param"] = "max_num_seqs"
    state["hypotheses"][0]["value"] = 64
    state["hypotheses"][0]["experiment_id"] = None
    prior = state["experiment_summaries"][-1]
    assert prior["validity_status"] == "valid"

    with patch("inferops.agent.executor.get_result_by_id", return_value=None), patch(
        "inferops.tools.propose_config.propose_config_patch"
    ), patch(
        "inferops.agent.executor.run_benchmark",
        side_effect=RuntimeError("bench exploded"),
    ):
        exec_patch = executor_node(state)

    assert exec_patch["hypotheses"][0]["status"] == "failed"
    assert exec_patch["experiments_remaining"] == state["experiments_remaining"] - 1
    assert exec_patch["last_result"] is None
    assert exec_patch["confirmation_decision"] is None
    assert exec_patch["repeat_ledgers"] is None
    assert "experiment_summaries" not in exec_patch
    event = exec_patch["last_recovery"]
    assert event["code"] == "tool_exception"
    assert event["result_persisted"] is False
    assert event.get("run_id") in (None, "")
    dumped = str(event)
    assert "throughput_rps" not in dumped
    assert "gpu_" not in dumped
    assert "request_ledger" not in dumped

    merged = _merge(state, exec_patch)
    assert merged["experiment_summaries"][-1] is prior
    assert current_attempt_latest(merged["experiment_summaries"], event) is None
    refl = reflector_node(merged)
    assert refl["next_action"] == "rollback"
    assert refl["trajectory"][-1]["constraint_checks"]["this_attempt_failed"] is True
    assert refl["best_summary"]["experiment_id"] == "sess_baseline"
    assert refl["trajectory"][-1]["result"]["promoted_to_best"] is False


def test_keyboardinterrupt_and_systemexit_are_not_swallowed_as_success():
    state = _pending_search_state()
    with patch("inferops.agent.executor.get_result_by_id", return_value=None), patch(
        "inferops.tools.propose_config.propose_config_patch"
    ), patch(
        "inferops.agent.executor.run_benchmark",
        side_effect=KeyboardInterrupt("user"),
    ):
        with pytest.raises(KeyboardInterrupt):
            executor_node(state)
    with patch("inferops.agent.executor.get_result_by_id", return_value=None), patch(
        "inferops.tools.propose_config.propose_config_patch"
    ), patch(
        "inferops.agent.executor.run_benchmark",
        side_effect=SystemExit(2),
    ):
        with pytest.raises(SystemExit):
            executor_node(state)
    assert is_hard_control_exception(KeyboardInterrupt()) is True
    assert is_hard_control_exception(SystemExit()) is True
    assert is_hard_control_exception(RuntimeError("x")) is False


# ---------------------------------------------------------------------------
# BenchmarkError (persisted failed contract row)
# ---------------------------------------------------------------------------

def test_benchmark_error_keeps_persisted_row_and_does_not_promote(result_b):
    failed = result_b.model_copy(
        update={
            "experiment_id": "sess_max_num_batched_tokens_4096",
            "status": ExperimentValidityStatus.FAILED,
            "notes": "vLLM OOM during startup",
            "successful_requests": 0,
            "error_rate": 1.0,
        }
    )
    assert is_promotable(failed) is False
    state = _pending_search_state()
    # Prior success is a different candidate — current hyp is still 4096.
    state["experiment_summaries"] = [
        state["baseline_summary"],
        _summary(eid="sess_prior_ok", param="max_num_seqs", value=256, vs=12.0),
    ]
    state["last_result"] = result_b

    with patch("inferops.agent.executor.get_result_by_id", return_value=None), patch(
        "inferops.tools.propose_config.propose_config_patch"
    ), patch(
        "inferops.agent.executor.run_benchmark",
        side_effect=OOMError("vLLM OOM during startup", result=failed),
    ):
        exec_patch = executor_node(state)

    summary = exec_patch["experiment_summaries"][-1]
    assert summary["validity_status"] == "failed"
    assert summary["run_id"] == failed.run_id
    assert summary["promotable"] is False
    assert exec_patch["last_result"] is failed
    event = exec_patch["last_recovery"]
    assert event["code"] == "benchmark_error"
    assert event["result_persisted"] is True
    assert event["experiment_id"] == summary["experiment_id"]
    assert current_attempt_latest(exec_patch["experiment_summaries"], event) is summary

    merged = _merge(state, exec_patch)
    refl = reflector_node(merged)
    assert refl["next_action"] == "rollback"
    assert refl["best_summary"]["experiment_id"] == "sess_baseline"
    assert refl["trajectory"][-1]["result"]["promoted_to_best"] is False
    assert is_confirmed_promotable(failed, None) is False


def test_benchmark_error_without_result_does_not_forge_summary():
    state = _pending_search_state()
    with patch("inferops.agent.executor.get_result_by_id", return_value=None), patch(
        "inferops.tools.propose_config.propose_config_patch"
    ), patch(
        "inferops.agent.executor.run_benchmark",
        side_effect=BenchmarkError("no contract row"),
    ):
        exec_patch = executor_node(state)
    assert "experiment_summaries" not in exec_patch
    assert exec_patch["last_result"] is None
    assert exec_patch["last_recovery"]["result_persisted"] is False


# ---------------------------------------------------------------------------
# Confirmation mid-slot fail + remasure cap
# ---------------------------------------------------------------------------

def test_confirmation_mid_slot_fail_records_ids_and_does_not_confirm(result_b):
    state = _state_with_candidate(run_id=result_b.run_id)
    state["last_result"] = result_b
    state["next_action"] = "remeasure"
    state["confirmation_target"] = {"param": "max_num_batched_tokens", "value": "4096"}
    state["hypotheses"][0]["status"] = "pending"
    state["repeat_ledgers"] = {
        "baseline": [],
        "candidate": [],
        "phase": RepeatPhase.SEARCH,
        "metric": "throughput_rps",
        "conditions": CONDITIONS,
    }
    completed = []

    class _Arm:
        last_candidate_result = None

        def __call__(self, arm: RepeatArm, slot):
            if arm == RepeatArm.BASELINE:
                lg = make_rps_ledger(f"cf-b{slot.pair_index}", rps=2.0, conditions=CONDITIONS)
                completed.append(lg.run_id)
                return lg
            raise RuntimeError("candidate slot boom")

    with confirmation_run_arm_override(_Arm()):
        exec_patch = executor_node(state)

    assert exec_patch["confirmation_decision"] is None
    assert exec_patch["last_result"] is None
    assert exec_patch["confirmation_blocked"] is True
    event = exec_patch["last_recovery"]
    assert event["stage"] == "confirmation_slot"
    assert event["code"] == "confirmation_slot_failed"
    assert event["cited_run_ids"]
    assert completed[0] in event["cited_run_ids"]
    assert exec_patch["trajectory"][-1]["result"]["promoted_to_best"] is False
    assert exec_patch["experiments_remaining"] == state["experiments_remaining"] - 1

    merged = _merge(state, exec_patch)
    refl = reflector_node(merged)
    assert refl["next_action"] == "remeasure"
    assert refl["best_summary"]["experiment_id"] == "sess_baseline"
    assert refl["trajectory"][-1]["result"]["promoted_to_best"] is False
    assert is_confirmed_promotable(result_b, None) is False


def test_confirmation_failures_are_capped_no_infinite_remeasure(result_b):
    state = _state_with_candidate()
    state["next_action"] = "remeasure"
    state["confirmation_target"] = {"param": "max_num_batched_tokens", "value": "4096"}
    state["hypotheses"][0]["status"] = "pending"
    state["repeat_ledgers"] = {"conditions": CONDITIONS, "metric": "throughput_rps"}
    state["remeasure_count"] = MAX_REMEASURES

    class _Arm:
        last_candidate_result = None

        def __call__(self, arm, slot):
            raise RuntimeError("always fail")

    with confirmation_run_arm_override(_Arm()):
        exec_patch = executor_node(state)
    assert exec_patch["last_recovery"]["retryable"] is False
    merged = _merge(state, exec_patch)
    refl = reflector_node(merged)
    assert refl["next_action"] in {"rollback", "stop"}
    assert refl["next_action"] != "remeasure"
    _assert_not_confirmed_promoted(refl)


# ---------------------------------------------------------------------------
# analyze / compare degrade — never silent 0 / promotion change
# ---------------------------------------------------------------------------

def test_analyze_and_compare_unavailable_are_recorded_not_silent_zero(result_b):
    state = _pending_search_state()
    result = result_b.model_copy(update={"experiment_id": "sess_max_num_batched_tokens_4096"})
    with patch("inferops.agent.executor.get_result_by_id", return_value=result), patch(
        "inferops.agent.executor.analyze_bottleneck",
        side_effect=RuntimeError("profile missing"),
    ), patch(
        "inferops.agent.executor.compare_experiments",
        side_effect=RuntimeError("bootstrap failed"),
    ):
        exec_patch = executor_node(state)

    tools = exec_patch["trajectory"][-1]["tools"]
    assert tools["analyze_bottleneck"]["status"] == "unavailable"
    assert tools["compare_experiments"]["status"] == "unavailable"
    vs = exec_patch["experiment_summaries"][-1]["vs_baseline_pct"]
    # Compare failed — use metric-derived delta, never a silent 0.
    assert vs is not None
    assert vs != 0
    baseline_rps = state["baseline_summary"]["throughput_rps"]
    expected = (result.throughput_rps - baseline_rps) / baseline_rps * 100
    assert vs == pytest.approx(round(expected, 2))
    assert exec_patch["best_summary"]["experiment_id"] == "sess_baseline"
    assert exec_patch["trajectory"][-1]["result"]["promoted_to_best"] is False


# ---------------------------------------------------------------------------
# Persist-then-interrupt-before-commit + reuse
# ---------------------------------------------------------------------------

def test_persist_then_interrupt_before_commit_reuses_result(result_b):
    """Crash after persist: resume reuses the row, no second benchmark."""
    store: dict[str, object] = {}
    calls: list[str] = []
    eid = "sess_max_num_batched_tokens_4096"
    result = result_b.model_copy(update={"experiment_id": eid})

    def _bench(inp: RunBenchmarkInput):
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
        with pytest.raises(KeyboardInterrupt):
            executor_node(state)
        assert calls == [eid]
        assert eid in store
        # Same incoming state — node never committed.
        resume = executor_node(state)

    assert calls == [eid]
    assert resume["experiments_remaining"] == state["experiments_remaining"] - 1
    assert resume["last_result"].experiment_id == eid
    assert resume["trajectory"][-1]["experiment_id"] == eid
    assert resume["best_summary"]["experiment_id"] == "sess_baseline"
    assert resume["trajectory"][-1]["result"]["promoted_to_best"] is False


# ---------------------------------------------------------------------------
# Production graph: interrupt_before tool + terminal equivalence
# ---------------------------------------------------------------------------

def _scripted_llm():
    from inferops.eval.real_graph import ScriptedBottleneckLLM

    return ScriptedBottleneckLLM(default_bottleneck="compute-bound")


def _graph_store(result_b):
    store: dict[str, object] = {}
    calls: list[str] = []
    store["sess_baseline"] = _ledger_backed_result(
        result_b, experiment_id="sess_baseline", run_id="base-search-rid", rps=2.0
    )

    def run_benchmark_fn(inp: RunBenchmarkInput):
        calls.append(inp.experiment_id)
        if not inp.config_patch:
            rps = 2.0
        else:
            rps = 2.4
        rid = f"{inp.experiment_id}-rid"
        result = _ledger_backed_result(
            result_b, experiment_id=inp.experiment_id, run_id=rid, rps=rps
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


def _graph_start_state(result_b):
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


def _run_production_graph(result_b, *, interrupt_before=None, crash_after_persist=False):
    from langgraph.checkpoint.memory import MemorySaver

    store, bench, calls = _graph_store(result_b)
    if crash_after_persist:
        inner = bench

        def bench(inp):  # noqa: F811
            out = inner(inp)
            if len(calls) == 1 and "confirm_" not in inp.experiment_id:
                raise KeyboardInterrupt("persist-then-interrupt")
            return out

    llm = _scripted_llm()
    checkpointer = MemorySaver()
    graph = build_graph(
        llm, checkpointer=checkpointer, interrupt_before=interrupt_before
    )
    config = graph_invoke_config("sess_")
    assert config["configurable"]["thread_id"] == session_thread_id("sess_")
    start = _graph_start_state(result_b)

    def _invoke(payload):
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
        assert snap.next == ("executor",)
        assert calls == []
        state = _invoke(None)
    elif interrupted:
        state = _invoke(None)
    return state, calls, config, store


def test_interrupt_before_tool_runs_benchmark_once_on_resume(result_b):
    final, calls, config, _store = _run_production_graph(
        result_b, interrupt_before=["executor"]
    )
    search_calls = [c for c in calls if "confirm_" not in c]
    assert len(search_calls) == 1
    assert config["configurable"]["thread_id"] == "sess"
    assert final["best_summary"].get("confirmed_promotable") is not True
    exec_steps = [s for s in final["trajectory"] if s.get("node") == "executor"]
    assert len(exec_steps) == 1


def test_persist_then_interrupt_graph_reuses_result_no_second_benchmark(result_b):
    final, calls, _config, store = _run_production_graph(
        result_b, crash_after_persist=True
    )
    search_calls = [c for c in calls if "confirm_" not in c]
    assert len(search_calls) == 1
    assert search_calls[0] in store
    exec_steps = [s for s in final["trajectory"] if s.get("node") == "executor"]
    assert len(exec_steps) == 1
    assert final["experiments_remaining"] == 0
    assert final["best_summary"].get("confirmed_promotable") is not True


def test_uninterrupted_vs_interrupted_resumed_terminal_equivalence(result_b):
    plain, plain_calls, _, _ = _run_production_graph(result_b)
    resumed, resumed_calls, _, _ = _run_production_graph(
        result_b, interrupt_before=["executor"]
    )
    assert [c for c in plain_calls if "confirm_" not in c] == [
        c for c in resumed_calls if "confirm_" not in c
    ]
    assert [s["experiment_id"] for s in plain["experiment_summaries"]] == [
        s["experiment_id"] for s in resumed["experiment_summaries"]
    ]
    assert plain["experiments_remaining"] == resumed["experiments_remaining"]
    assert (plain.get("best_summary") or {}).get("experiment_id") == (
        resumed.get("best_summary") or {}
    ).get("experiment_id")
    assert (plain.get("best_summary") or {}).get("confirmed_promotable") is not True
    assert (resumed.get("best_summary") or {}).get("confirmed_promotable") is not True
    plain_eids = [
        s.get("experiment_id")
        for s in plain["trajectory"]
        if s.get("node") == "executor"
    ]
    resumed_eids = [
        s.get("experiment_id")
        for s in resumed["trajectory"]
        if s.get("node") == "executor"
    ]
    assert plain_eids == resumed_eids


def test_no_confirmed_promotion_on_fail_or_resume_paths(result_b):
    """Any fail / resume path in this module stays fail-closed on confirm."""
    state = _state_with_candidate()
    state["hypotheses"][0]["status"] = "pending"
    with patch("inferops.agent.executor.get_result_by_id", return_value=None), patch(
        "inferops.tools.propose_config.propose_config_patch"
    ), patch(
        "inferops.agent.executor.run_benchmark",
        side_effect=RuntimeError("fail"),
    ):
        exec_patch = executor_node(state)
    refl = reflector_node(_merge(state, exec_patch))
    assert is_promotable(result_b) is True  # gates themselves unchanged
    assert is_confirmed_promotable(result_b, None) is False
    assert refl["best_summary"]["experiment_id"] == "sess_baseline"
    assert refl["trajectory"][-1]["result"]["promoted_to_best"] is False


# ---------------------------------------------------------------------------
# P1: confirm persist-resume + successful-confirm budget-once
# ---------------------------------------------------------------------------

def _remeasure_state(*, remaining: int = 2, remasure_count: int = 1):
    state = _pending_search_state()
    state["next_action"] = "remeasure"
    state["remeasure_count"] = remasure_count
    state["confirmation_target"] = {
        "param": "max_num_batched_tokens",
        "value": "4096",
    }
    state["hypotheses"][0]["status"] = "pending"
    state["experiments_remaining"] = remaining
    return state


def test_confirmation_slot_ids_include_remeasure_identity():
    hyp = {"param": "max_num_batched_tokens", "value": 4096}
    eid = confirmation_slot_experiment_id(
        session_prefix="sess_",
        hypothesis=hyp,
        arm=RepeatArm.CANDIDATE,
        pair_index=1,
        remasure_count=2,
    )
    assert eid == "sess_confirm_max_num_batched_tokens_4096_r2_c1"
    other = confirmation_slot_experiment_id(
        session_prefix="sess_",
        hypothesis=hyp,
        arm=RepeatArm.BASELINE,
        pair_index=0,
        remasure_count=1,
    )
    assert "_r1_b0" in other
    assert other != eid


def test_confirm_persist_then_crash_reuses_slots_budget_once(result_b):
    """Mid-confirm crash: resume reuses persisted slots, budget charged once."""
    store, inner = _tool_boundary_store(result_b, confirm_cand_rps=2.4)
    calls: list[str] = []
    crash_after = 2

    def bench(inp: RunBenchmarkInput):
        calls.append(inp.experiment_id)
        out = inner(inp)
        if len(calls) == crash_after:
            raise KeyboardInterrupt("mid-confirm persist")
        return out

    state = _remeasure_state(remaining=2, remasure_count=1)
    pre_budget = state["experiments_remaining"]

    with tool_boundary_overrides(
        run_benchmark_fn=bench,
        propose_config_fn=lambda _inp: None,
    ), patch(
        "inferops.agent.executor.get_result_by_id",
        side_effect=lambda e: store.get(e),
    ):
        with pytest.raises(KeyboardInterrupt):
            executor_node(state)
        first = list(calls)
        assert len(first) == crash_after
        assert all("_r1_" in eid for eid in first)
        for eid in first:
            assert eid in store
        # Same incoming state — campaign never committed.
        resume = executor_node(state)

    assert calls[:crash_after] == first
    expected_slots = DEFAULT_MIN_PAIRS * 2
    assert len(calls) == expected_slots
    assert len(set(calls)) == expected_slots
    assert all("_r1_" in eid for eid in calls)
    assert resume["experiments_remaining"] == pre_budget - 1
    assert resume["last_recovery"] is None
    assert resume["confirmation_decision"] is not None
    assert resume["trajectory"][-1]["result"]["promoted_to_best"] is False


def test_successful_confirm_promotes_and_consumes_budget_once(result_b):
    """Promote path decrements budget exactly once — never remaining-still-1."""
    search_exec, _search_refl, confirm_exec, final_refl, _after = (
        _run_search_remeasure_confirm(result_b, confirm_cand_rps=2.4)
    )
    pre_confirm = search_exec["experiments_remaining"]
    assert confirm_exec["experiments_remaining"] == pre_confirm - 1
    assert final_refl["best_summary"].get("confirmed_promotable") is True
    assert confirm_exec["experiments_remaining"] != pre_confirm


def test_successful_confirm_on_last_budget_slot_still_promotes(result_b):
    """Last budget slot: consume to 0, still promote (no leftover remaining=1)."""
    store, bench = _tool_boundary_store(result_b, confirm_cand_rps=2.4)
    state = _remeasure_state(remaining=1, remasure_count=1)

    with tool_boundary_overrides(
        run_benchmark_fn=bench,
        propose_config_fn=lambda _inp: None,
    ), patch(
        "inferops.agent.executor.get_result_by_id",
        side_effect=lambda e: store.get(e),
    ):
        confirm_exec = executor_node(state)

    assert confirm_exec["experiments_remaining"] == 0
    merged = _merge(state, confirm_exec)
    assert merged["experiments_remaining"] == 0
    refl = reflector_node(merged)
    assert refl["best_summary"].get("confirmed_promotable") is True
    assert refl["trajectory"][-1]["result"]["promoted_to_best"] is True
    assert refl["best_summary"].get("experiment_id") != "sess_baseline"


def _last_slot_confirmed_state(result_b, *, error_rate):
    """⑤-confirmed, Week-1-promotable candidate on the last budget slot."""
    cands = _n_ledgers("slo0c", 2.4, 3)
    bases = _n_ledgers("slo0b", 2.0, 3)
    decision = verdict_from_ledgers(bases, cands, phase=RepeatPhase.CONFIRMATION)
    last = result_b.model_copy(update={"run_id": cands[-1].run_id, "error_rate": 0.0})
    assert is_promotable(last) is True
    assert is_confirmed_promotable(last, decision) is True
    state = _state_with_candidate(run_id=last.run_id, error_rate=error_rate)
    state["experiments_remaining"] = 0
    state["last_result"] = last
    state["confirmation_decision"] = decision
    state["confirmation_bound_run_ids"] = [lg.run_id for lg in cands]
    state["confirmation_target"] = {
        "param": "max_num_batched_tokens",
        "value": "4096",
    }
    state["repeat_ledgers"] = {
        "baseline": bases,
        "candidate": cands,
        "phase": RepeatPhase.CONFIRMATION,
        "metric": "throughput_rps",
        "min_pairs": 3,
    }
    return state


def test_last_budget_slot_high_error_rate_does_not_promote(result_b):
    """remaining==0 + error_rate over SLO must not promote via would_promote."""
    state = _last_slot_confirmed_state(result_b, error_rate=0.40)
    refl = reflector_node(state)
    step = refl["trajectory"][-1]
    assert step["constraint_checks"]["slo_ok"] is False
    assert step["result"]["promoted_to_best"] is False
    assert refl["best_summary"]["experiment_id"] == "sess_baseline"
    assert refl["best_summary"].get("confirmed_promotable") is not True


def test_last_budget_slot_missing_error_rate_does_not_promote(result_b):
    """remaining==0 + missing error_rate must not promote (fail-closed SLO)."""
    state = _last_slot_confirmed_state(result_b, error_rate=None)
    refl = reflector_node(state)
    step = refl["trajectory"][-1]
    assert step["constraint_checks"]["slo_ok"] is False
    assert step["constraint_checks"]["slo"]["reason"] == "error_rate_missing_fail_closed"
    assert step["result"]["promoted_to_best"] is False
    assert refl["best_summary"]["experiment_id"] == "sess_baseline"
    assert refl["best_summary"].get("confirmed_promotable") is not True


# ---------------------------------------------------------------------------
# P2: generic propose error + GraphInterrupt not swallowed
# ---------------------------------------------------------------------------

def test_generic_propose_tool_error_emits_recovery_no_forge():
    state = _pending_search_state()
    with patch("inferops.agent.executor.get_result_by_id", return_value=None), patch(
        "inferops.tools.propose_config.propose_config_patch",
        side_effect=RuntimeError("propose backend down"),
    ):
        exec_patch = executor_node(state)

    event = exec_patch["last_recovery"]
    _assert_recovery_contract(event)
    assert event["code"] == "propose_tool_error"
    assert event["stage"] == "propose_config"
    assert event["result_persisted"] is False
    assert event["budget_consumed"] is False
    assert "experiments_remaining" not in exec_patch
    assert exec_patch["last_result"] is None
    assert "experiment_summaries" not in exec_patch
    refl = reflector_node(_merge(state, exec_patch))
    assert refl["next_action"] == "rollback"
    assert refl["best_summary"]["experiment_id"] == "sess_baseline"
    assert refl["trajectory"][-1]["result"]["promoted_to_best"] is False


def test_graphinterrupt_is_reraised_not_swallowed():
    from langgraph.errors import GraphInterrupt

    state = _pending_search_state()
    with patch("inferops.agent.executor.get_result_by_id", return_value=None), patch(
        "inferops.tools.propose_config.propose_config_patch"
    ), patch(
        "inferops.agent.executor.run_benchmark",
        side_effect=GraphInterrupt(),
    ):
        with pytest.raises(GraphInterrupt):
            executor_node(state)

    remasure = _remeasure_state()

    class _Arm:
        last_candidate_result = None

        def __call__(self, arm, slot):
            raise GraphInterrupt()

    with confirmation_run_arm_override(_Arm()):
        with pytest.raises(GraphInterrupt):
            executor_node(remasure)

    with patch("inferops.agent.executor.get_result_by_id", return_value=None), patch(
        "inferops.tools.propose_config.propose_config_patch",
        side_effect=GraphInterrupt(),
    ):
        with pytest.raises(GraphInterrupt):
            executor_node(state)

    assert is_hard_control_exception(GraphInterrupt()) is True
