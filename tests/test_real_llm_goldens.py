"""Week-3 ⑦ closeout: real-LLM evidence is separate from fake/offline."""

from __future__ import annotations

from inferops.eval.real_graph import (
    MODE_REAL_GRAPH_LLM,
    MODE_REAL_GRAPH_OFFLINE,
    ScriptedBottleneckLLM,
    _llm_boundary_label,
)
from inferops.eval.real_llm_goldens import (
    LAYER,
    N_RUNS,
    assert_offline_not_labeled_live,
    blocked_campaign,
    judge_live_run,
    missing_credential,
    refuse_fake_labeled_live,
    run_real_llm_campaign,
    scripted_llm_is_not_live,
)


def _live_report(**row_updates: object) -> dict:
    row = {
        "stop_reason": "budget_exhausted",
        "n_experiments": 2,
        "trajectory_nodes": ["planner", "executor", "reflector"],
        "trajectory_score": 0.8,
        "composite": 0.5,
        "hypotheses": [{"param": "max_num_batched_tokens", "value": 4096, "status": "done"}],
        "benchmark_calls": [{"config_patch": {"max_num_batched_tokens": 4096}}],
    }
    row.update(row_updates)
    return {
        "mode": MODE_REAL_GRAPH_LLM,
        "llm_boundary": "live",
        "strategies": {"real_planner": [row]},
    }


def test_n_runs_requirement_is_at_least_three():
    assert N_RUNS >= 3


def test_missing_credential_is_explicit_blocker(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    reason = missing_credential("openrouter")
    assert reason is not None
    assert "OPENROUTER_API_KEY" in reason
    campaign = run_real_llm_campaign(n=3, backend="openrouter")
    assert campaign.layer == LAYER
    assert campaign.status == "blocked"
    assert campaign.passed is False
    assert campaign.pass_rate is None
    assert campaign.n_completed == 0
    assert campaign.n_accepted == 0
    assert campaign.runs == []
    assert "BLOCKED" in campaign.summary
    assert "not a pass" in campaign.summary.lower() or "NOT a real-LLM pass" in campaign.report_markdown()


def test_blocked_campaign_is_not_labeled_live():
    campaign = blocked_campaign()
    assert campaign.llm_boundary is None
    assert campaign.layer == "real_llm"
    md = campaign.report_markdown()
    assert "separate" in md.lower()
    assert "fake" in md.lower()


def test_scripted_llm_is_never_live():
    assert scripted_llm_is_not_live() == "fake_scripted"
    assert (
        refuse_fake_labeled_live(ScriptedBottleneckLLM(), MODE_REAL_GRAPH_LLM)
        is not None
    )


def test_injecting_scripted_llm_does_not_create_a_live_pass():
    campaign = run_real_llm_campaign(n=3, llm=ScriptedBottleneckLLM())
    assert campaign.passed is False
    assert campaign.status == "blocked"
    assert campaign.llm_boundary == "fake_scripted"
    assert "real LLM" in (campaign.blocker or "")


def test_offline_report_cannot_be_labeled_live():
    assert_offline_not_labeled_live(
        {"mode": MODE_REAL_GRAPH_OFFLINE, "llm_boundary": "fake_scripted"}
    )
    try:
        assert_offline_not_labeled_live(
            {"mode": MODE_REAL_GRAPH_OFFLINE, "llm_boundary": "live"}
        )
    except AssertionError as exc:
        assert "live" in str(exc)
    else:
        raise AssertionError("offline report labeled live was accepted")


def test_llm_boundary_keys_off_object_not_mode():
    fake = ScriptedBottleneckLLM()
    assert _llm_boundary_label(fake, MODE_REAL_GRAPH_LLM) == "fake_scripted"
    assert _llm_boundary_label(object(), MODE_REAL_GRAPH_LLM) == "injected"


def test_judge_live_run_accepts_first_class_stop_and_trajectory():
    verdict = judge_live_run(_live_report())
    assert verdict.accepted is True
    assert verdict.failures == []
    assert verdict.stop_reason == "budget_exhausted"


def test_judge_live_run_rejects_empty_zero_quality_and_wrong_stop():
    empty = {
        "mode": MODE_REAL_GRAPH_LLM,
        "llm_boundary": "live",
        "strategies": {"real_planner": []},
    }
    assert judge_live_run(empty).accepted is False
    assert any("empty" in f for f in judge_live_run(empty).failures)

    zero = judge_live_run(
        _live_report(
            stop_reason="",
            n_experiments=0,
            trajectory_nodes=[],
            trajectory_score=0.0,
            hypotheses=[],
        )
    )
    assert zero.accepted is False
    assert any("zero" in f or "missing stop" in f for f in zero.failures)

    wrong = judge_live_run(_live_report(stop_reason="eval_empty_plan"))
    assert wrong.accepted is False
    assert any("wrong-stop" in f for f in wrong.failures)


def test_live_boundary_failed_acceptance_does_not_pass_or_inflate_rate(monkeypatch):
    """P1-1: live boundary + failed acceptance cannot score 3/3."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-not-live")
    dead = {
        "mode": MODE_REAL_GRAPH_LLM,
        "llm_boundary": "live",
        "strategies": {
            "real_planner": [
                {
                    "stop_reason": "",
                    "n_experiments": 0,
                    "trajectory_nodes": [],
                    "trajectory_score": 0.0,
                    "composite": 0.0,
                    "hypotheses": [],
                    "benchmark_calls": [],
                }
            ]
        },
    }

    def _dead_eval(**_kwargs):
        return dead

    monkeypatch.setattr(
        "inferops.eval.real_llm_goldens.run_real_graph_eval", _dead_eval
    )
    campaign = run_real_llm_campaign(n=3)
    assert campaign.llm_boundary == "live"
    assert campaign.status == "ran"
    assert campaign.n_completed == 3
    assert campaign.n_accepted == 0
    assert all(run.status == "failed" for run in campaign.runs)
    assert all(run.accepted is False for run in campaign.runs)
    assert campaign.passed is False
    assert campaign.pass_rate == 0.0
    assert campaign.pass_rate != 1.0
