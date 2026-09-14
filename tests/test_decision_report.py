"""A2 unified decision report: four outcomes + measured-config export."""

from __future__ import annotations

from inferops.decision import DecisionKind, build_decision, render_decision_markdown
from inferops.task import build_optimization_task, confirm_task, default_task_for_workload
from inferops.tools.final_report import FinalReportInput, write_final_report


def _row(
    eid: str,
    *,
    param=None,
    value=None,
    rps=15.0,
    ttft=70.0,
    vs=0.0,
    status="valid",
    promotable=True,
    error_rate=0.0,
    run_id="run",
    ledger="logs/ledger.json",
    actual=None,
    requested=None,
    **extra,
):
    row = {
        "experiment_id": eid,
        "param_changed": param,
        "value_changed": value,
        "throughput_rps": rps,
        "tokens_per_second": 1900.0,
        "ttft_p50_ms": 48.0,
        "ttft_p99_ms": ttft,
        "e2e_p50_ms": 900.0,
        "bottleneck": "compute-bound",
        "vs_baseline_pct": vs,
        "run_id": run_id,
        "validity_status": status,
        "mlflow_run_id": f"mlf-{eid}",
        "has_config_evidence": promotable,
        "promotable": promotable,
        "failure_reason": extra.pop("failure_reason", ""),
        "error_rate": error_rate,
        "ledger_path": ledger,
        "requested_config": requested or {
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "max_num_batched_tokens": 2048,
        },
        "actual_config": actual or {
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "max_num_batched_tokens": 2048,
        },
    }
    row.update(extra)
    return row


def test_confirmed_and_meets_goals():
    task = default_task_for_workload("chat_short", budget=4, target_qps=10, max_ttft_ms=100)
    baseline = _row("sess_baseline", rps=12.0, ttft=80.0)
    best = _row(
        "sess_chunked",
        param="enable_chunked_prefill",
        value=True,
        rps=16.0,
        ttft=60.0,
        vs=33.3,
        run_id="best-run",
        ledger="logs/ledger_best.json",
        actual={
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 2048,
        },
        requested={
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 2048,
        },
    )
    decision = build_decision(
        baseline_summary=baseline,
        best_summary=best,
        experiment_summaries=[baseline, best],
        optimization_task=task.model_dump(mode="json"),
        stop_reason="budget_exhausted",
    )
    assert decision.kind == DecisionKind.CONFIRMED_AND_MEETS_GOALS
    assert decision.adopt is True
    assert decision.meets_goals is True
    assert decision.adopted_config == best["actual_config"]
    assert decision.exported_matches_measured is True
    text = "\n".join(render_decision_markdown(decision))
    assert "best-run" in text
    assert "logs/ledger_best.json" in text
    assert "enable_chunked_prefill" in text


def test_improved_but_target_qps_unmet():
    task = default_task_for_workload("chat_short", budget=4, target_qps=20)
    baseline = _row("sess_baseline", rps=8.0)
    best = _row(
        "sess_big",
        param="max_num_batched_tokens",
        value=4096,
        rps=12.0,
        vs=50.0,
        actual={"max_num_batched_tokens": 4096},
        requested={"max_num_batched_tokens": 4096},
    )
    decision = build_decision(
        baseline_summary=baseline,
        best_summary=best,
        experiment_summaries=[baseline, best],
        optimization_task=task.model_dump(mode="json"),
    )
    assert decision.kind == DecisionKind.IMPROVED_BUT_UNMET_GOALS
    assert decision.adopt is False
    assert decision.meets_goals is False
    assert any(c.role == "goal" and c.ok is False for c in decision.goal_checks)


def test_no_reliable_improvement_keeps_baseline():
    baseline = _row("sess_baseline", rps=15.0)
    failed = _row(
        "sess_oom",
        param="max_num_seqs",
        value=256,
        status="failed",
        promotable=False,
        vs=None,
        failure_reason="OOM",
        actual=None,
    )
    decision = build_decision(
        baseline_summary=baseline,
        best_summary=baseline,
        experiment_summaries=[baseline, failed],
        stop_reason="no_reliable_improvement",
    )
    assert decision.kind == DecisionKind.NO_RELIABLE_IMPROVEMENT
    assert decision.keep_baseline is True
    assert decision.adopted_config == baseline["actual_config"]
    assert any("OOM" in line for line in decision.why_not_others)


def test_inconclusive_when_baseline_not_promotable():
    baseline = _row(
        "sess_baseline",
        status="insufficient_evidence",
        promotable=False,
        actual=None,
    )
    decision = build_decision(
        baseline_summary=baseline,
        best_summary=None,
        experiment_summaries=[baseline],
        stop_reason="",
    )
    assert decision.kind == DecisionKind.INCONCLUSIVE
    assert decision.adopted_config is None
    assert decision.exported_matches_measured is False


def test_unpromotable_high_score_is_not_an_improvement():
    baseline = _row("sess_baseline", rps=10.0)
    shiny = _row(
        "sess_fake",
        param="max_num_batched_tokens",
        value=4096,
        rps=99.0,
        vs=890.0,
        status="insufficient_evidence",
        promotable=False,
        actual=None,
    )
    decision = build_decision(
        baseline_summary=baseline,
        best_summary=shiny,
        experiment_summaries=[baseline, shiny],
    )
    assert decision.kind == DecisionKind.NO_RELIABLE_IMPROVEMENT
    assert decision.adopted_summary["experiment_id"] == "sess_baseline"


def test_exported_config_is_measured_not_requested():
    baseline = _row("sess_baseline")
    best = _row(
        "sess_cand",
        param="max_num_batched_tokens",
        value=4096,
        rps=18.0,
        vs=20.0,
        requested={"max_num_batched_tokens": 4096, "enable_prefix_caching": True},
        actual={"max_num_batched_tokens": 4096},
    )
    decision = build_decision(
        baseline_summary=baseline,
        best_summary=best,
        experiment_summaries=[baseline, best],
    )
    assert decision.adopted_config == {"max_num_batched_tokens": 4096}
    assert "enable_prefix_caching" not in decision.adopted_config
    text = "\n".join(render_decision_markdown(decision))
    assert "requested `True` → measured `<missing>`" in text


def test_write_final_report_uses_same_decision(tmp_path):
    task = confirm_task(build_optimization_task(
        workload_name="chat_short",
        target_qps=10,
        budget=4,
    ))
    baseline = _row("sess_baseline", rps=11.0)
    best = _row(
        "sess_win",
        param="enable_prefix_caching",
        value=True,
        rps=14.0,
        vs=27.0,
        run_id="win-run",
        ledger="logs/ledger_win.json",
        actual={"enable_prefix_caching": True, "model_name": "Qwen/Qwen2.5-0.5B-Instruct"},
    )
    out = tmp_path / "decision.md"
    result = write_final_report(
        FinalReportInput(
            workload_name="chat_short",
            session_prefix="sess_",
            experiment_summaries=[baseline, best],
            baseline_summary=baseline,
            best_summary=best,
            output_path=str(out),
            optimization_task=task.model_dump(mode="json"),
            stop_reason="budget_exhausted",
        )
    )
    text = out.read_text()
    assert result.decision_kind == DecisionKind.CONFIRMED_AND_MEETS_GOALS.value
    assert "## Decision" in text
    assert "confirmed_and_meets_goals" in text
    assert "win-run" in text
    assert "logs/ledger_win.json" in text
    assert "FP8" not in text
    assert task.model_name in text
