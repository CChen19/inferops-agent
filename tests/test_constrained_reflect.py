"""Week-2 ⑥: constrained Reflect — fixture / offline trajectories.

No self-started GPU. Confirmation and promotion go through ⑤ APIs only.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from inferops.agent.confirm_campaign import decision_binds_to_result
from inferops.agent.executor import confirmation_run_arm_override, executor_node
from inferops.agent.reflect_constraints import (
    LLM_MUST_NOT_OWN,
    MAX_ERROR_RATE,
    MAX_REMEASURES,
    ReflectConclusion,
    check_slo,
    conclude_experiment,
    maybe_promote_best,
    optional_llm_explanation,
)
from inferops.agent.reflector import reflector_node, route_after_reflector
from inferops.agent.state import initial_state, is_promotable_summary, summary_from_result
from inferops.metrics import (
    ConfirmationDecision,
    ConfirmationVerdict,
    RepeatArm,
    RepeatPhase,
    is_confirmed_promotable,
    verdict_from_ledgers,
)
from inferops.schemas import is_promotable
from tests.test_repeat_confirmation import CONDITIONS, _n_ledgers, make_rps_ledger


def _contract(**overrides):
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


def _summary(*, eid="cand", param="max_num_batched_tokens", value=4096,
             vs=10.0, validity="valid", **overrides):
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


def _state_with_candidate(**summary_kw):
    state = initial_state("chat_short", "sess_", max_experiments=6)
    baseline = _summary(
        eid="sess_baseline",
        param=None,
        value=None,
        vs=0.0,
        run_id="bb" * 16,
    )
    cand = _summary(**summary_kw)
    state["baseline_summary"] = baseline
    state["best_summary"] = baseline
    state["experiment_summaries"] = [baseline, cand]
    state["tried_experiment_ids"] = [baseline["experiment_id"], cand["experiment_id"]]
    state["current_bottleneck"] = "compute-bound"
    state["experiments_remaining"] = 3
    state["hypotheses"] = [
        {
            "id": "h1",
            "param": cand.get("param_changed") or "max_num_batched_tokens",
            "value": cand.get("value_changed") if cand.get("value_changed") is not None else 4096,
            "rationale": "rps=2.0 compute-bound; raise batch tokens [source: vllm_scheduler]",
            "status": "success",
            "experiment_id": cand["experiment_id"],
        }
    ]
    return state


def _assert_reflect_step(step: dict):
    assert step["node"] == "reflector"
    assert step["action"] == "reflect"
    assert "hypothesis" in step
    assert "config_diff" in step and "summary" in step["config_diff"]
    assert isinstance(step["cited_run_ids"], list)
    checks = step["constraint_checks"]
    for key in (
        "validity_status",
        "slo_ok",
        "budget_remaining",
        "is_duplicate",
        "confirmed_promotable",
        "gate",
    ):
        assert key in checks
    assert step["next_action"] in {"continue", "remeasure", "rollback", "stop"}
    assert "stop_reason" in step


# ---------------------------------------------------------------------------
# Required cases
# ---------------------------------------------------------------------------

def test_duplicate_candidate_continue_trajectory():
    state = _state_with_candidate()
    state["last_skip_reason"] = "duplicate_candidate"
    state["last_skipped_hypothesis_id"] = "h1"
    state["hypotheses"][0]["status"] = "skipped"
    patch = reflector_node(state)
    step = patch["trajectory"][-1]
    _assert_reflect_step(step)
    assert step["constraint_checks"]["is_duplicate"] is True
    assert step["next_action"] == "continue"
    assert patch["should_stop"] is False
    assert patch["best_summary"]["experiment_id"] == "sess_baseline"
    assert step["hypothesis"]["id"] == "h1"


def test_oom_exec_fail_rollback_trajectory():
    state = _state_with_candidate(
        validity="failed",
        failure_reason="vLLM OOM during startup",
        promotable=False,
        vs=None,
        error_rate=1.0,
        run_id="cc" * 16,
    )
    patch = reflector_node(state)
    step = patch["trajectory"][-1]
    _assert_reflect_step(step)
    assert step["constraint_checks"]["exec_fail"] is True
    assert step["constraint_checks"]["validity_status"] == "failed"
    assert step["next_action"] == "rollback"
    assert patch["should_stop"] is False
    assert patch["best_summary"]["experiment_id"] == "sess_baseline"
    assert "cc" * 16 in step["cited_run_ids"]


def test_slo_breach_rollback_trajectory():
    state = _state_with_candidate(error_rate=0.40, vs=12.0)
    assert check_slo(state["experiment_summaries"][-1])["ok"] is False
    patch = reflector_node(state)
    step = patch["trajectory"][-1]
    _assert_reflect_step(step)
    assert step["constraint_checks"]["slo_ok"] is False
    assert step["constraint_checks"]["slo"]["error_rate"] == 0.40
    assert step["next_action"] == "rollback"
    assert patch["best_summary"]["experiment_id"] == "sess_baseline"


def test_budget_exhaust_stop_trajectory():
    state = _state_with_candidate()
    state["experiments_remaining"] = 0
    patch = reflector_node(state)
    step = patch["trajectory"][-1]
    _assert_reflect_step(step)
    assert patch["should_stop"] is True
    assert patch["next_action"] == "stop"
    assert patch["stop_reason"] == "budget_exhausted"
    assert step["stop_reason"] == "budget_exhausted"


def test_too_noisy_remeasure_trajectory():
    state = _state_with_candidate()
    # One pair < min_pairs=3 → ⑤ too_noisy (not invented).
    state["repeat_ledgers"] = {
        "baseline": _n_ledgers("tnb", 2.0, 1),
        "candidate": _n_ledgers("tnc", 2.4, 1),
        "phase": RepeatPhase.CONFIRMATION,
        "metric": "throughput_rps",
        "min_pairs": 3,
    }
    patch = reflector_node(state)
    step = patch["trajectory"][-1]
    _assert_reflect_step(step)
    assert step["constraint_checks"]["verdict"] == "too_noisy"
    assert step["next_action"] == "remeasure"
    assert patch["should_stop"] is False
    assert patch["hypotheses"][0]["status"] == "pending"
    assert patch["remeasure_count"] == 1
    assert route_after_reflector({**state, **patch}) == "executor"


def test_too_noisy_stops_after_remeasure_cap():
    state = _state_with_candidate()
    state["remeasure_count"] = MAX_REMEASURES
    state["repeat_ledgers"] = {
        "baseline": _n_ledgers("tnb2", 2.0, 1),
        "candidate": _n_ledgers("tnc2", 2.4, 1),
        "phase": RepeatPhase.CONFIRMATION,
        "min_pairs": 3,
    }
    patch = reflector_node(state)
    assert patch["next_action"] == "stop"
    assert patch["stop_reason"] == "too_noisy"
    assert patch["should_stop"] is True


def test_no_reliable_improvement_first_class_stop():
    state = _state_with_candidate(vs=0.5)
    state["repeat_ledgers"] = {
        "baseline": _n_ledgers("nrb", 2.0, 3),
        "candidate": _n_ledgers("nrc", 2.0, 3),
        "phase": RepeatPhase.CONFIRMATION,
        "metric": "throughput_rps",
    }
    patch = reflector_node(state)
    step = patch["trajectory"][-1]
    _assert_reflect_step(step)
    assert step["constraint_checks"]["verdict"] == "no_diff"
    assert patch["next_action"] == "stop"
    assert patch["stop_reason"] == "no_reliable_improvement"
    assert patch["should_stop"] is True


def test_streak_without_confirmation_is_also_no_reliable_improvement():
    state = _state_with_candidate(vs=1.0)
    state["no_improvement_streak"] = 2
    patch = reflector_node(state)
    assert patch["stop_reason"] == "no_reliable_improvement"
    assert patch["next_action"] == "stop"


# ---------------------------------------------------------------------------
# Promotion gate
# ---------------------------------------------------------------------------

def test_search_winner_remeasures_and_does_not_promote(result_b):
    cands = _n_ledgers("swc", 3.0, 1)
    bound = result_b.model_copy(update={"run_id": cands[0].run_id})
    state = _state_with_candidate(run_id=cands[0].run_id)
    state["last_result"] = bound
    state["repeat_ledgers"] = {
        "baseline": _n_ledgers("swb", 2.0, 1),
        "candidate": cands,
        "phase": RepeatPhase.SEARCH,
        "min_pairs": 1,
    }
    decision = verdict_from_ledgers(
        state["repeat_ledgers"]["baseline"],
        state["repeat_ledgers"]["candidate"],
        phase=RepeatPhase.SEARCH,
        min_pairs=1,
    )
    assert decision.search_winner is True
    assert is_confirmed_promotable(bound, decision) is False

    patch = reflector_node(state)
    step = patch["trajectory"][-1]
    assert step["next_action"] == "remeasure"
    assert step["constraint_checks"]["search_winner"] is True
    assert step["constraint_checks"]["confirmed_promotable"] is False
    assert patch["best_summary"]["experiment_id"] == "sess_baseline"
    assert step["result"]["promoted_to_best"] is False


def test_confirmed_improvement_promotes_only_via_is_confirmed_promotable(result_b):
    cands = _n_ledgers("okc", 2.4, 3)
    bound = result_b.model_copy(update={"run_id": cands[-1].run_id})
    state = _state_with_candidate(run_id=cands[-1].run_id)
    state["last_result"] = bound
    state["repeat_ledgers"] = {
        "baseline": _n_ledgers("okb", 2.0, 3),
        "candidate": cands,
        "phase": RepeatPhase.CONFIRMATION,
    }
    state["confirmation_bound_run_ids"] = [lg.run_id for lg in cands]
    state["confirmation_target"] = {
        "param": "max_num_batched_tokens",
        "value": "4096",
    }
    decision = verdict_from_ledgers(
        state["repeat_ledgers"]["baseline"],
        state["repeat_ledgers"]["candidate"],
        phase=RepeatPhase.CONFIRMATION,
    )
    assert decision.verdict == ConfirmationVerdict.CONFIRMED_IMPROVEMENT
    assert is_promotable(bound) is True
    assert is_confirmed_promotable(bound, decision) is True
    assert decision_binds_to_result(decision, bound, candidate=state["experiment_summaries"][-1])

    patch = reflector_node(state)
    step = patch["trajectory"][-1]
    _assert_reflect_step(step)
    assert patch["best_summary"]["experiment_id"] != "sess_baseline"
    assert patch["best_summary"].get("confirmed_promotable") is True
    assert step["constraint_checks"]["confirmed_promotable"] is True
    assert step["constraint_checks"]["gate_result"] is True
    assert step["result"]["promoted_to_best"] is True
    # Non-remeasure clears stale ⑤ state after the promote.
    assert patch["confirmation_decision"] is None
    assert patch["repeat_ledgers"] is None


def test_unevidenced_result_not_promoted_even_with_confirm(
    result_b, result_b_unevidenced
):
    cands = _n_ledgers("uec", 3.0, 3)
    bound = result_b.model_copy(update={"run_id": cands[-1].run_id})
    unevid = result_b_unevidenced.model_copy(update={"run_id": cands[-1].run_id})
    state = _state_with_candidate(run_id=cands[-1].run_id)
    state["last_result"] = unevid
    state["repeat_ledgers"] = {
        "baseline": _n_ledgers("ueb", 2.0, 3),
        "candidate": cands,
        "phase": RepeatPhase.CONFIRMATION,
    }
    decision = verdict_from_ledgers(
        state["repeat_ledgers"]["baseline"],
        state["repeat_ledgers"]["candidate"],
        phase=RepeatPhase.CONFIRMATION,
    )
    assert decision.verdict == ConfirmationVerdict.CONFIRMED_IMPROVEMENT
    assert is_confirmed_promotable(bound, decision) is True
    assert is_confirmed_promotable(unevid, decision) is False

    patch = reflector_node(state)
    assert patch["best_summary"]["experiment_id"] == "sess_baseline"
    assert patch["trajectory"][-1]["constraint_checks"]["confirmed_promotable"] is False
    assert patch["trajectory"][-1]["result"]["promoted_to_best"] is False


def test_forged_confirmation_cannot_promote(result_b):
    with pytest.raises(ValueError, match="forged ConfirmationDecision"):
        ConfirmationDecision(
            phase=RepeatPhase.CONFIRMATION,
            verdict=ConfirmationVerdict.CONFIRMED_IMPROVEMENT,
            numeric_signal="improvement",
            metric="throughput_rps",
        )
    current = _summary(eid="sess_baseline", param=None, value=None, vs=0.0)
    cand = _summary()
    best, promoted = maybe_promote_best(result_b, None, cand, current)
    assert promoted is False
    assert best["experiment_id"] == "sess_baseline"


def test_maybe_promote_best_is_the_only_best_write(result_b):
    """Search numbers + Week-1 promotable still fail without a ⑤ decision."""
    assert is_promotable(result_b) is True
    current = _summary(eid="keep", param=None, value=None, vs=0.0)
    cand = _summary(eid="hot", vs=50.0)
    best, promoted = maybe_promote_best(result_b, None, cand, current)
    assert promoted is False
    assert best is current


# ---------------------------------------------------------------------------
# Offline executor → reflector (duplicate + OOM) without GPU
# ---------------------------------------------------------------------------

def test_offline_executor_duplicate_then_reflect():
    state = initial_state("chat_short", "sess_", max_experiments=5)
    baseline = _summary(eid="sess_baseline", param=None, value=None, vs=0.0)
    prior = _summary(eid="sess_max_num_seqs_256", param="max_num_seqs", value=256, vs=1.0)
    state["baseline_summary"] = baseline
    state["best_summary"] = baseline
    state["experiment_summaries"] = [baseline, prior]
    state["experiments_remaining"] = 3
    state["hypotheses"] = [
        {
            "id": "h1",
            "param": "max_num_seqs",
            "value": 256,
            "rationale": "rps=2.0 more parallelism [source: vllm_scheduler]",
            "status": "pending",
            "experiment_id": None,
        }
    ]
    exec_patch = executor_node(state)
    assert exec_patch["last_skip_reason"] == "duplicate_candidate"
    assert exec_patch["hypotheses"][0]["status"] == "skipped"
    merged = {**state, **exec_patch}
    refl = reflector_node(merged)
    step = refl["trajectory"][-1]
    _assert_reflect_step(step)
    assert step["constraint_checks"]["is_duplicate"] is True
    assert step["next_action"] == "continue"


def test_offline_executor_oom_then_reflect_rollback(result_b):
    from inferops.bench_runner import OOMError
    from inferops.schemas import ExperimentValidityStatus

    failed = result_b.model_copy(
        update={
            "experiment_id": "sess_max_num_batched_tokens_4096",
            "status": ExperimentValidityStatus.FAILED,
            "notes": "vLLM OOM during startup",
            "successful_requests": 0,
            "error_rate": 1.0,
        }
    )
    state = _state_with_candidate()
    state["experiment_summaries"] = [state["baseline_summary"]]
    state["hypotheses"][0]["status"] = "pending"
    state["hypotheses"][0]["experiment_id"] = None
    from unittest.mock import patch

    with patch("inferops.agent.executor.get_result_by_id", return_value=None), \
         patch("inferops.tools.propose_config.propose_config_patch"), \
         patch(
             "inferops.agent.executor.run_benchmark",
             side_effect=OOMError("vLLM OOM during startup", result=failed),
         ):
        exec_patch = executor_node(state)
    assert exec_patch["experiment_summaries"][-1]["validity_status"] == "failed"
    merged = {**state, **exec_patch}
    refl = reflector_node(merged)
    step = refl["trajectory"][-1]
    _assert_reflect_step(step)
    assert step["next_action"] == "rollback"
    assert refl["best_summary"]["experiment_id"] == "sess_baseline"


# ---------------------------------------------------------------------------
# Deterministic vs LLM split
# ---------------------------------------------------------------------------

def test_conclusion_is_pure_and_llm_explanation_is_not_control_flow():
    c = conclude_experiment(
        experiments_remaining=2,
        no_improvement_streak=0,
        current_bottleneck="compute-bound",
        latest=_summary(error_rate=0.9),
        baseline=_summary(eid="b", param=None, value=None, vs=0.0),
        best=_summary(eid="b", param=None, value=None, vs=0.0),
        summaries=[],
    )
    assert isinstance(c, ReflectConclusion)
    assert c.next_action == "rollback"
    text = optional_llm_explanation(c)
    assert "rollback" in text
    # Forbidden fields stay owned by the dataclass, not the prose.
    assert "validity_status" in LLM_MUST_NOT_OWN
    assert c.constraint_checks["slo_ok"] is False


def test_slo_missing_error_rate_is_fail_closed():
    got = check_slo({"error_rate": None})
    assert got["ok"] is False
    assert got["error_rate"] is None
    assert got["reason"] == "error_rate_missing_fail_closed"
    assert MAX_ERROR_RATE == 0.05
    assert check_slo({})["ok"] is False


def test_fixture_trajectories_cover_required_cases():
    root = Path("tests/fixtures/reflect_trajectories")
    expected = {
        "duplicate_candidate.json": "continue",
        "oom_exec_fail.json": "rollback",
        "slo_breach.json": "rollback",
        "budget_exhaust.json": "stop",
        "too_noisy.json": "remeasure",
        "no_reliable_improvement.json": "stop",
    }
    for name, action in expected.items():
        step = json.loads((root / name).read_text())
        _assert_reflect_step(step)
        assert step["next_action"] == action


# ---------------------------------------------------------------------------
# P1 proofs
# ---------------------------------------------------------------------------

def test_confirmed_decision_for_a_does_not_promote_b(result_b):
    """Stale ⑤ confirm for candidate A must not promote a later candidate B."""
    ledgers_a = _n_ledgers("p1a", 2.4, 3)
    decision_a = verdict_from_ledgers(
        _n_ledgers("p1ab", 2.0, 3),
        ledgers_a,
        phase=RepeatPhase.CONFIRMATION,
    )
    assert decision_a.verdict == ConfirmationVerdict.CONFIRMED_IMPROVEMENT

    result_a = result_b.model_copy(update={"run_id": ledgers_a[-1].run_id})
    result_b2 = result_b.model_copy(update={"run_id": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"})
    assert result_a.run_id != result_b2.run_id
    assert is_confirmed_promotable(result_b2, decision_a) is True  # ⑤ does not bind
    assert decision_binds_to_result(decision_a, result_b2) is False

    cand_b = _summary(
        eid="sess_max_num_seqs_64",
        param="max_num_seqs",
        value=64,
        run_id=result_b2.run_id,
    )
    current = _summary(eid="sess_baseline", param=None, value=None, vs=0.0)
    best, promoted = maybe_promote_best(
        result_b2,
        decision_a,
        cand_b,
        current,
        bound_run_ids=[lg.run_id for lg in ledgers_a],
        bound_target={"param": "max_num_batched_tokens", "value": "4096"},
    )
    assert promoted is False
    assert best["experiment_id"] == "sess_baseline"

    state = _state_with_candidate(
        eid="sess_max_num_seqs_64",
        param="max_num_seqs",
        value=64,
        run_id=result_b2.run_id,
    )
    state["last_result"] = result_b2
    state["confirmation_decision"] = decision_a
    state["confirmation_bound_run_ids"] = [lg.run_id for lg in ledgers_a]
    state["confirmation_target"] = {"param": "max_num_batched_tokens", "value": "4096"}
    state["repeat_ledgers"] = {
        "baseline": _n_ledgers("p1ab2", 2.0, 3),
        "candidate": ledgers_a,
        "phase": RepeatPhase.CONFIRMATION,
    }
    patch = reflector_node(state)
    assert patch["best_summary"]["experiment_id"] == "sess_baseline"
    assert patch["trajectory"][-1]["result"]["promoted_to_best"] is False
    assert patch["confirmation_decision"] is None


def test_executor_clears_confirmation_when_candidate_changes(result_b):
    ledgers_a = _n_ledgers("clr", 2.4, 3)
    decision_a = verdict_from_ledgers(
        _n_ledgers("clrb", 2.0, 3),
        ledgers_a,
        phase=RepeatPhase.CONFIRMATION,
    )
    state = _state_with_candidate()
    state["confirmation_decision"] = decision_a
    state["confirmation_target"] = {"param": "max_num_batched_tokens", "value": "4096"}
    state["confirmation_bound_run_ids"] = [lg.run_id for lg in ledgers_a]
    state["next_action"] = "continue"
    state["experiment_summaries"] = [state["baseline_summary"]]
    state["hypotheses"] = [
        {
            "id": "h2",
            "param": "max_num_seqs",
            "value": 64,
            "rationale": "rps=2.0 try fewer seqs [source: paged_attention]",
            "status": "pending",
            "experiment_id": None,
        }
    ]
    with patch("inferops.agent.executor.get_result_by_id", return_value=result_b), \
         patch(
             "inferops.agent.executor.analyze_bottleneck",
             return_value=MagicMock(bottleneck="compute-bound"),
         ), \
         patch(
             "inferops.agent.executor.compare_experiments",
             return_value=MagicMock(delta_pct=1.0),
         ):
        out = executor_node(state)
    assert out["confirmation_decision"] is None
    assert out["repeat_ledgers"] is None
    assert out["confirmation_target"] is None


def _fixture_run_arm(prefix: str, base_rps: float, cand_rps: float, bound_run_id: str):
    def run_arm(arm: RepeatArm, slot):
        if arm == RepeatArm.BASELINE:
            return make_rps_ledger(
                f"{prefix}-b{slot.pair_index}",
                rps=base_rps,
                conditions=CONDITIONS,
            )
        rid = bound_run_id if slot.pair_index == 2 else f"{prefix}-c{slot.pair_index}"
        return make_rps_ledger(rid, rps=cand_rps, conditions=CONDITIONS)

    return run_arm


def test_search_winner_remeasure_campaign_promotes(result_b):
    """search winner → remasure → ⑤ confirmation campaign → bound promote."""
    search_cands = _n_ledgers("sw2c", 3.0, 1)
    bound_id = "campaign-cand-2"
    result = result_b.model_copy(update={"run_id": search_cands[0].run_id})
    state = _state_with_candidate(run_id=search_cands[0].run_id)
    state["last_result"] = result
    state["repeat_ledgers"] = {
        "baseline": _n_ledgers("sw2b", 2.0, 1),
        "candidate": search_cands,
        "phase": RepeatPhase.SEARCH,
        "min_pairs": 1,
        "conditions": CONDITIONS,
        "metric": "throughput_rps",
    }
    search_patch = reflector_node(state)
    assert search_patch["next_action"] == "remeasure"
    assert search_patch["best_summary"]["experiment_id"] == "sess_baseline"
    assert search_patch["hypotheses"][0]["status"] == "pending"

    merged = {**state, **search_patch}
    with confirmation_run_arm_override(_fixture_run_arm("prom", 2.0, 2.4, bound_id)):
        exec_patch = executor_node(merged)
    decision = exec_patch["confirmation_decision"]
    assert decision.phase == RepeatPhase.CONFIRMATION
    assert decision.verdict == ConfirmationVerdict.CONFIRMED_IMPROVEMENT
    assert bound_id in exec_patch["confirmation_bound_run_ids"]

    result_bound = result_b.model_copy(update={"run_id": bound_id})
    latest = dict(merged["experiment_summaries"][-1])
    latest["run_id"] = bound_id
    after = {
        **merged,
        **exec_patch,
        "last_result": result_bound,
        "experiment_summaries": merged["experiment_summaries"][:-1] + [latest],
    }
    assert is_confirmed_promotable(result_bound, decision) is True
    refl = reflector_node(after)
    assert refl["best_summary"].get("confirmed_promotable") is True
    assert refl["best_summary"]["experiment_id"] == latest["experiment_id"]
    assert refl["trajectory"][-1]["result"]["promoted_to_best"] is True


def test_search_winner_remeasure_unconfirmed_does_not_promote(result_b):
    search_cands = _n_ledgers("sw3c", 3.0, 1)
    bound_id = "flat-cand-2"
    result = result_b.model_copy(update={"run_id": search_cands[0].run_id})
    state = _state_with_candidate(run_id=search_cands[0].run_id)
    state["last_result"] = result
    state["repeat_ledgers"] = {
        "baseline": _n_ledgers("sw3b", 2.0, 1),
        "candidate": search_cands,
        "phase": RepeatPhase.SEARCH,
        "min_pairs": 1,
        "conditions": CONDITIONS,
        "metric": "throughput_rps",
    }
    search_patch = reflector_node(state)
    merged = {**state, **search_patch}
    with confirmation_run_arm_override(_fixture_run_arm("flat", 2.0, 2.0, bound_id)):
        exec_patch = executor_node(merged)
    decision = exec_patch["confirmation_decision"]
    assert decision.phase == RepeatPhase.CONFIRMATION
    assert decision.verdict != ConfirmationVerdict.CONFIRMED_IMPROVEMENT

    result_bound = result_b.model_copy(update={"run_id": bound_id})
    latest = dict(merged["experiment_summaries"][-1])
    latest["run_id"] = bound_id
    after = {
        **merged,
        **exec_patch,
        "last_result": result_bound,
        "experiment_summaries": merged["experiment_summaries"][:-1] + [latest],
    }
    assert is_confirmed_promotable(result_bound, decision) is False
    refl = reflector_node(after)
    assert refl["best_summary"]["experiment_id"] == "sess_baseline"
    assert refl["trajectory"][-1]["result"]["promoted_to_best"] is False


def test_invalid_and_insufficient_evidence_rollback():
    for status in ("invalid", "insufficient_evidence"):
        state = _state_with_candidate(validity=status, promotable=False, error_rate=0.0)
        patch = reflector_node(state)
        step = patch["trajectory"][-1]
        _assert_reflect_step(step)
        assert step["next_action"] == "rollback"
        assert step["constraint_checks"]["validity_rollback"] is True
        assert patch["best_summary"]["experiment_id"] == "sess_baseline"


def test_missing_error_rate_rollback_fail_closed():
    state = _state_with_candidate()
    state["experiment_summaries"][-1]["error_rate"] = None
    patch = reflector_node(state)
    step = patch["trajectory"][-1]
    _assert_reflect_step(step)
    assert step["constraint_checks"]["slo_ok"] is False
    assert step["constraint_checks"]["slo"]["reason"] == "error_rate_missing_fail_closed"
    assert step["next_action"] == "rollback"
    assert patch["best_summary"]["experiment_id"] == "sess_baseline"


def test_live_reflect_steps_match_fixture_contract():
    """Fixtures are the live Reflect contract (same keys / next_action)."""
    root = Path("tests/fixtures/reflect_trajectories")
    state = _state_with_candidate()
    state["last_skip_reason"] = "duplicate_candidate"
    live = reflector_node(state)["trajectory"][-1]
    fixture = json.loads((root / "duplicate_candidate.json").read_text())
    for key in fixture:
        assert key in live
    assert live["next_action"] == fixture["next_action"]


def test_summary_from_result_carries_error_rate(result_b):
    summary = summary_from_result(
        result_b,
        param_changed="max_num_batched_tokens",
        value_changed=4096,
        baseline_primary=2.0,
        primary_metric="throughput_rps",
    )
    assert "error_rate" in summary
    assert is_promotable_summary(summary) is True
