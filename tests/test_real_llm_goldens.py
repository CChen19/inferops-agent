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
    missing_credential,
    refuse_fake_labeled_live,
    run_real_llm_campaign,
    scripted_llm_is_not_live,
)


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
