"""CPU-only coverage for durable confirmed-task resume."""

from __future__ import annotations

from unittest.mock import patch

from langgraph.graph import END

from inferops import bench_runner
from inferops.agent.executor import _benchmark_input
from inferops.agent.graph import run_agent, session_thread_id
from inferops.agent.state import initial_state
from inferops.memory.db import delete_task, get_task, save_task, update_task_status
from inferops.schemas import ExperimentValidityStatus
from inferops.task import default_task_for_workload
from tests.test_config_application import _patch_common


def _baseline() -> dict:
    return {
        "experiment_id": "persist_baseline",
        "param_changed": None,
        "value_changed": None,
        "throughput_rps": 2.0,
        "tokens_per_second": 128.0,
        "ttft_p50_ms": 48.0,
        "ttft_p99_ms": 70.0,
        "e2e_p50_ms": 900.0,
        "bottleneck": "compute-bound",
        "vs_baseline_pct": 0.0,
        "run_id": "baseline-run",
        "validity_status": "valid",
        "mlflow_run_id": "m0",
        "has_config_evidence": True,
        "promotable": True,
        "failure_reason": "",
        "error_rate": 0.0,
    }


def test_confirm_persist_has_stable_task_and_thread_ids(tmp_path):
    db_path = tmp_path / "memory.db"
    task = default_task_for_workload("chat_short", 2)
    prefix = "persist_"
    thread_id = session_thread_id(prefix)

    first = save_task(
        task_id=task.task_id,
        session_prefix=prefix,
        thread_id=thread_id,
        confirmed_task=task.model_dump(mode="json"),
        db_path=db_path,
    )
    second = save_task(
        task_id=task.task_id,
        session_prefix="replacement_",
        thread_id="replacement",
        confirmed_task=task.model_dump(mode="json"),
        status="running",
        db_path=db_path,
    )

    assert first.task_id == second.task_id == task.task_id
    assert first.thread_id == second.thread_id == "persist"
    assert second.session_prefix == "persist_"
    assert get_task(task.task_id, db_path=db_path).confirmed_task["status"] == "confirmed"
    assert update_task_status(task.task_id, "completed", db_path=db_path).status == "completed"
    assert delete_task(task.task_id, db_path=db_path) is True
    assert get_task(task.task_id, db_path=db_path) is None


def test_new_checkpointer_instance_resumes_without_second_baseline(tmp_path):
    """Interrupt before executor, then rebuild graph/saver and resume once."""
    db_path = tmp_path / "memory.db"
    task = default_task_for_workload("chat_short", 2)
    baseline_calls: list[str] = []
    benchmark_calls: list[str] = []

    def prepare(workload_name, prefix, max_experiments=8, task=None):
        baseline_calls.append(prefix)
        state = initial_state(
            workload_name,
            prefix,
            max_experiments=max_experiments,
            optimization_task=task.model_dump(mode="json"),
        )
        baseline = _baseline()
        state.update(
            baseline_summary=baseline,
            best_summary=baseline,
            experiment_summaries=[baseline],
            tried_experiment_ids=[baseline["experiment_id"]],
            current_bottleneck="compute-bound",
            experiments_remaining=1,
        )
        return state

    def planner(state, llm=None):
        return {
            "hypotheses": [{
                "id": "h1",
                "param": "max_num_seqs",
                "value": 64,
                "rationale": "test",
                "status": "pending",
                "experiment_id": None,
            }]
        }

    def executor(state):
        benchmark_calls.append("benchmark")
        return {
            "experiments_remaining": state["experiments_remaining"] - 1,
            "tried_experiment_ids": [*state["tried_experiment_ids"], "persist_candidate"],
        }

    def reflector(state):
        return {"should_stop": True, "stop_reason": "budget_exhausted"}

    patches = {
        "prepare_initial_state": prepare,
        "planner_node": planner,
        "executor_node": executor,
        "reflector_node": reflector,
        "route_after_reflector": lambda _state: END,
        "_print_run_summary": lambda _state: None,
    }
    with patch.multiple("inferops.agent.graph", **patches):
        interrupted = run_agent(
            workload_name="chat_short",
            llm=object(),
            max_experiments=2,
            session_prefix="persist_",
            interrupt_before=["executor"],
            task=task,
            db_path=db_path,
        )
        assert interrupted["experiments_remaining"] == 1
        final = run_agent(
            workload_name=None,
            llm=object(),
            resume_task_id=task.task_id,
            db_path=db_path,
        )

    assert baseline_calls == ["persist_"]
    assert benchmark_calls == ["benchmark"]
    assert final["experiments_remaining"] == 0
    assert get_task(task.task_id, db_path=db_path).status == "completed"


def test_external_task_mode_skips_managed_start_without_env(monkeypatch, config):
    _patch_common(monkeypatch)
    monkeypatch.delenv("INFEROPS_EXTERNAL_VLLM", raising=False)
    monkeypatch.setattr(
        bench_runner,
        "_ensure_managed_vllm",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not manage vLLM")),
    )

    task = default_task_for_workload("chat_short", 2, service_mode="external")
    state = initial_state(
        "chat_short",
        "external_",
        max_experiments=2,
        optimization_task=task.model_dump(mode="json"),
    )
    inp = _benchmark_input(state, "external_candidate", {})
    assert inp.service_mode == "external"

    result = bench_runner.run_experiment(config, ["p"], service_mode=inp.service_mode)

    assert result.status == ExperimentValidityStatus.INSUFFICIENT_EVIDENCE
    assert result.config_evidence.kind == "external_unverified"
