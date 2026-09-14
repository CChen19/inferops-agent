"""A1 OptimizationTask contract: draft, validate, confirm, no silent replace."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from inferops.agent.graph import prepare_initial_state
from inferops.agent.intent import (
    Intent,
    extract_intent,
    interpret_user_request,
    merge_intents,
    task_from_intent,
)
from inferops.agent.reflect_constraints import check_slo, conclude_experiment
from inferops.agent.reflector import reflector_node
from inferops.agent.state import initial_state
from inferops.schemas import compute_workload_hash
from inferops.task import (
    DEFAULT_MODEL_NAME,
    TRAFFIC_ARRIVAL_RATE_NOTE,
    TaskStatus,
    build_optimization_task,
    confirm_task,
    default_task_for_workload,
    format_task_conditions_markdown,
    format_task_confirmation,
    require_confirmed,
    resolve_model_name,
    task_conditions,
)
from inferops.tools.final_report import FinalReportInput, write_final_report
from inferops.tools.run_benchmark import RunBenchmarkInput, run_benchmark
from tests.test_constrained_reflect import _state_with_candidate


def _llm(json_content: str) -> MagicMock:
    llm = MagicMock()
    resp = MagicMock()
    resp.content = json_content
    llm.invoke.return_value = resp
    return llm


def test_unknown_workload_is_not_silently_replaced():
    task = build_optimization_task(workload_name="nonexistent_workload", budget=6)
    assert task.status == TaskStatus.NEEDS_CLARIFICATION
    assert any("nonexistent_workload" in c for c in task.clarification_needed)
    assert task.workload.name == "chat_short"  # placeholder only; not executable


def test_unsupported_model_is_rejected_not_remapped():
    task = build_optimization_task(
        workload_name="chat_short",
        model_hint="Llama-3-70B",
        budget=4,
    )
    assert task.status == TaskStatus.REJECTED
    assert any("Llama-3-70B" in r for r in task.rejection_reasons)
    with pytest.raises(ValueError, match="rejected"):
        confirm_task(task)


def test_ambiguous_qwen_needs_clarification():
    task = build_optimization_task(workload_name="chat_short", model_hint="Qwen")
    assert task.status == TaskStatus.NEEDS_CLARIFICATION
    assert any("ambiguous" in c.lower() for c in task.clarification_needed)


def test_qwen_1_5b_alias_resolves():
    name, reason = resolve_model_name("Qwen 1.5B")
    assert reason == "matched"
    assert name == "Qwen/Qwen2.5-1.5B-Instruct"
    task = build_optimization_task(
        workload_name="chat_short",
        model_hint="Qwen2.5-1.5B",
        target_qps=10,
        max_ttft_ms=200,
        budget=6,
    )
    assert task.status == TaskStatus.READY
    assert task.model_name == "Qwen/Qwen2.5-1.5B-Instruct"
    assert task.target_qps == 10.0
    assert any(c.metric == "ttft_p99_ms" and c.value == 200 for c in task.constraints)


def test_missing_model_defaults_with_warning_not_silent():
    task = build_optimization_task(workload_name="chat_short")
    assert task.status == TaskStatus.READY
    assert task.model_name == DEFAULT_MODEL_NAME
    assert any(DEFAULT_MODEL_NAME in w for w in task.warnings)


def test_target_qps_is_not_copied_to_workload_rps():
    task = build_optimization_task(
        workload_name="chat_short",
        target_qps=10.0,
        budget=6,
    )
    assert task.target_qps == 10.0
    assert task.workload.rps is None
    cond = task_conditions(task)
    assert cond["target_qps_role"] == "measured_throughput_goal"
    assert cond["traffic_arrival_rate_supported"] is False
    assert cond["traffic_arrival_rate_note"] == TRAFFIC_ARRIVAL_RATE_NOTE


def test_offered_rps_is_recorded_but_flagged_unsupported():
    task = build_optimization_task(
        workload_name="chat_short",
        offered_rps=12.0,
        budget=6,
    )
    assert task.workload.rps == 12.0
    assert task.traffic_arrival_rate_supported is False
    assert any("arrival-rate" in w for w in task.warnings)


def test_unconfirmed_task_cannot_spend_budget():
    task = build_optimization_task(workload_name="chat_short", budget=4)
    with pytest.raises(ValueError, match="confirmed"):
        require_confirmed(task)
    with pytest.raises(ValueError, match="confirmed"):
        prepare_initial_state("chat_short", "sess_", task=task)


def test_unclear_task_cannot_be_confirmed():
    task = build_optimization_task(workload_name="not_a_workload")
    with pytest.raises(ValueError, match="clarification"):
        confirm_task(task)


def test_confirm_report_and_conditions_are_identical():
    task = confirm_task(
        build_optimization_task(
            workload_name="chat_short",
            model_hint="Qwen/Qwen2.5-1.5B-Instruct",
            target_qps=10,
            max_ttft_ms=180,
            budget=5,
            gpu_hint="RTX 3060",
        )
    )
    cond = task_conditions(task)
    confirm_md = format_task_confirmation(task)
    report_md = "\n".join(format_task_conditions_markdown(task))
    for blob in (confirm_md, report_md):
        assert cond["task_id"] in blob
        assert cond["model_name"] in blob
        assert cond["workload_hash"] in blob
        assert "ttft_p99_ms" in blob
        assert "measured throughput" in blob.lower() or "measured_throughput_goal" in blob
    assert cond["workload_hash"] == compute_workload_hash(task.workload)
    assert cond["experiment_budget"] == 5
    assert cond["service_mode"] == "managed"


def test_final_report_embeds_same_task_conditions(tmp_path):
    task = confirm_task(build_optimization_task(
        workload_name="chat_short",
        model_hint="Qwen2.5-0.5B",
        target_qps=8,
        budget=4,
    ))
    out = tmp_path / "report.md"
    write_final_report(
        FinalReportInput(
            workload_name="chat_short",
            session_prefix="sess_",
            experiment_summaries=[],
            output_path=str(out),
            optimization_task=task.model_dump(mode="json"),
        )
    )
    text = out.read_text()
    cond = task_conditions(task)
    assert "## Task Conditions" in text
    assert cond["task_id"] in text
    assert cond["model_name"] in text
    assert cond["workload_hash"] in text


def test_latency_violation_blocks_recommendation():
    task = default_task_for_workload("chat_short", budget=6, max_ttft_ms=80)
    state = _state_with_candidate(
        throughput_rps=9.0,
        ttft_p99_ms=200.0,
        vs=80.0,
        error_rate=0.0,
    )
    state["optimization_task"] = task.model_dump(mode="json")
    latest = state["experiment_summaries"][-1]
    slo = check_slo(latest, task.constraints)
    assert slo["ok"] is False
    assert any(v["metric"] == "ttft_p99_ms" for v in slo["violations"])

    conclusion = conclude_experiment(
        experiments_remaining=3,
        no_improvement_streak=0,
        current_bottleneck="compute-bound",
        latest=latest,
        baseline=state["baseline_summary"],
        best=state["best_summary"],
        summaries=state["experiment_summaries"],
        constraints=task.constraints,
        primary_metric="throughput_rps",
    )
    assert conclusion.promote is False
    assert conclusion.next_action == "rollback"

    patch = reflector_node(state)
    assert patch["best_summary"]["experiment_id"] == "sess_baseline"
    assert patch["next_action"] == "rollback"


def test_error_rate_slo_unchanged_without_task_constraints():
    assert check_slo({"error_rate": 0.01})["ok"] is True
    assert check_slo({"error_rate": 0.40})["ok"] is False
    assert check_slo({"error_rate": None})["reason"] == "error_rate_missing_fail_closed"


def test_run_benchmark_uses_task_model(result):
    with patch("inferops.tools.run_benchmark.run_experiment", return_value=result) as mock_run, \
         patch("inferops.tools.run_benchmark.save_result"), \
         patch("inferops.tools.run_benchmark.get_prompts", return_value=["p"] * 20):
        run_benchmark(
            RunBenchmarkInput(
                experiment_id="model_task",
                workload_name="chat_short",
                persist=False,
                model_name="Qwen/Qwen2.5-1.5B-Instruct",
            )
        )
    cfg = mock_run.call_args[0][0]
    assert cfg.model_name == "Qwen/Qwen2.5-1.5B-Instruct"


def test_run_benchmark_uses_explicit_workload_spec(result):
    task = build_optimization_task(
        workload_name="chat_short",
        concurrency=4,
        budget=3,
    )
    with patch("inferops.tools.run_benchmark.run_experiment", return_value=result) as mock_run, \
         patch("inferops.tools.run_benchmark.save_result"), \
         patch("inferops.tools.run_benchmark.get_prompts", return_value=["p"] * 20):
        run_benchmark(
            RunBenchmarkInput(
                experiment_id="wl_task",
                workload_name="chat_short",
                persist=False,
                workload=task.workload,
            )
        )
    cfg = mock_run.call_args[0][0]
    assert cfg.workload.concurrency == 4
    assert cfg.workload.name == "chat_short"


def test_interpret_followup_fills_missing_model():
    first = Intent(
        workload_name="chat_short",
        model_hint="",
        target_qps=10.0,
        gpu_hint="",
        budget=6,
        notes="",
        parse_ok=True,
    )
    llm = _llm(
        '{"workload_name": null, "model_hint": "Qwen2.5-1.5B", "target_qps": null}'
    )
    incoming = extract_intent("use the 1.5B model", llm)
    merged = merge_intents(first, incoming)
    task = task_from_intent(merged)
    assert merged.workload_name == "chat_short"
    assert merged.target_qps == 10.0
    assert task.model_name == "Qwen/Qwen2.5-1.5B-Instruct"
    assert task.status == TaskStatus.READY


def test_invalid_json_does_not_start_a_runnable_task():
    llm = _llm("sorry I cannot parse that")
    intent, task = interpret_user_request("anything", llm)
    assert intent.parse_ok is False
    assert task.status == TaskStatus.NEEDS_CLARIFICATION
    with pytest.raises(ValueError):
        confirm_task(task)


def test_initial_state_carries_optional_task():
    state = initial_state("chat_short", "sess_")
    assert state["optimization_task"] is None
    assert state["started_at_s"] is None
