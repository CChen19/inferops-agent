"""Thin Tune adapter: drive ⑤ interleaved confirmation from Reflect remasure.

Consumes ``run_interleaved_repeats`` / ``evaluate_campaign`` / ``verdict_from_ledgers``.
Does not mint ``confirmed_improvement`` and does not invent a metrics schema.

Production remasure default: per-slot ``run_benchmark`` at the existing tool
boundary (④ ledger on each ``ExperimentResult``). CI / offline tests inject a
fixture ``run_arm`` or stub that tool edge — GPU-free, no invented numbers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from inferops.metrics import (
    DEFAULT_MIN_PAIRS,
    ConfirmationDecision,
    RepeatArm,
    RepeatCampaign,
    RepeatPhase,
    RepeatSlot,
    RunConditions,
    evaluate_campaign,
    ledger_from_result,
    run_interleaved_repeats,
    verdict_from_ledgers,
)


def candidate_fingerprint(param: Any, value: Any) -> dict[str, str]:
    return {"param": str(param), "value": str(value)}


def fingerprints_match(target: dict[str, Any] | None, param: Any, value: Any) -> bool:
    if not target:
        return False
    want = candidate_fingerprint(param, value)
    return target.get("param") == want["param"] and target.get("value") == want["value"]


def decision_candidate_run_ids(decision: ConfirmationDecision | None) -> list[str]:
    if decision is None:
        return []
    ids: list[str] = []
    for pair in decision.pairs:
        if pair.candidate_run_id and pair.candidate_run_id not in ids:
            ids.append(pair.candidate_run_id)
    return ids


def decision_binds_to_result(
    decision: ConfirmationDecision | None,
    result: Any,
    *,
    candidate: dict[str, Any] | None = None,
    bound_run_ids: list[str] | None = None,
    bound_target: dict[str, Any] | None = None,
) -> bool:
    """Tune bind: ⑤ decision may only apply to the candidate it was computed for."""
    if decision is None or result is None:
        return False
    rid = str(getattr(result, "run_id", "") or "")
    if not rid:
        return False
    pair_ids = set(decision_candidate_run_ids(decision))
    if rid not in pair_ids:
        return False
    if bound_run_ids is not None and rid not in set(bound_run_ids):
        return False
    if candidate is not None:
        cand_rid = str(candidate.get("run_id") or "")
        if cand_rid and cand_rid not in pair_ids:
            return False
        if bound_target:
            param = candidate.get("param_changed")
            value = candidate.get("value_changed")
            if param is not None and not fingerprints_match(bound_target, param, value):
                return False
    return True


def decision_applies_to_latest(
    decision: ConfirmationDecision | None,
    *,
    last_result: Any,
    latest: dict[str, Any] | None,
    bound_run_ids: list[str] | None,
    bound_target: dict[str, Any] | None,
) -> bool:
    """Stale-campaign filter. Promotion still requires ``decision_binds_to_result``."""
    if decision is None:
        return False
    if bound_target and latest and latest.get("param_changed") is not None:
        if not fingerprints_match(
            bound_target, latest.get("param_changed"), latest.get("value_changed")
        ):
            return False
    if last_result is not None:
        return decision_binds_to_result(
            decision,
            last_result,
            candidate=latest,
            bound_run_ids=bound_run_ids,
            bound_target=bound_target,
        )
    if bound_run_ids and latest:
        rid = str(latest.get("run_id") or "")
        if rid and rid not in set(bound_run_ids):
            return False
    return True


def search_verdict_from_results(
    baseline: Any,
    candidate: Any,
    *,
    metric: str = "throughput_rps",
) -> ConfirmationDecision | None:
    """⑤ search-phase verdict from two persisted ④-backed results.

    Returns ``None`` when either result lacks a request ledger. Callers must
    not invent ``search_winner`` from metric-only rows.
    """
    if baseline is None or candidate is None:
        return None
    base_ledger = ledger_from_result(baseline)
    cand_ledger = ledger_from_result(candidate)
    if base_ledger is None or cand_ledger is None:
        return None
    try:
        return verdict_from_ledgers(
            [base_ledger],
            [cand_ledger],
            metric=metric,
            phase=RepeatPhase.SEARCH,
            min_pairs=1,
        )
    except (TypeError, ValueError):
        return None


def search_winner_state_pack(
    *,
    baseline: Any,
    candidate: Any,
    hyp: dict[str, Any],
    primary_metric: str,
) -> dict[str, Any]:
    """State overlay after a genuine ⑤ search winner. Empty if not a winner."""
    decision = search_verdict_from_results(
        baseline, candidate, metric=primary_metric
    )
    if decision is None or not decision.search_winner:
        return {}
    base_ledger = ledger_from_result(baseline)
    cand_ledger = ledger_from_result(candidate)
    if base_ledger is None or cand_ledger is None:
        return {}
    return {
        "confirmation_decision": decision,
        "repeat_ledgers": {
            "baseline": [base_ledger],
            "candidate": [cand_ledger],
            "phase": RepeatPhase.SEARCH,
            "metric": primary_metric,
            "min_pairs": 1,
            "conditions": base_ledger.conditions,
        },
        "confirmation_target": candidate_fingerprint(hyp.get("param"), hyp.get("value")),
        "confirmation_bound_run_ids": [str(getattr(candidate, "run_id", "") or "")],
    }


def production_slot_run_arm(
    *,
    hypothesis: dict[str, Any],
    session_prefix: str,
    run_slot: Callable[[str, dict[str, Any]], Any],
) -> Callable[[RepeatArm, RepeatSlot], Any]:
    """Per-slot ``run_arm`` that calls the benchmark tool boundary.

    ``run_slot(experiment_id, config_patch)`` must return an
    ``ExperimentResult`` with a ④ ``request_ledger``. Baseline slots apply
    no candidate override; candidate slots apply ``{param: value}``.

    The last candidate result is stored as ``.last_candidate_result`` so the
    executor can bind ``last_result`` / ``experiment_summaries`` to a run_id
    that appears on the confirmation decision.
    """

    class _SlotRunner:
        last_candidate_result: Any = None

        def __call__(self, arm: RepeatArm, slot: RepeatSlot) -> Any:
            if arm == RepeatArm.BASELINE:
                config: dict[str, Any] = {}
                tag = "b"
            elif arm == RepeatArm.CANDIDATE:
                config = {hypothesis["param"]: hypothesis["value"]}
                tag = "c"
            else:
                raise ValueError(f"unknown confirmation arm {arm!r}")
            eid = (
                f"{session_prefix}confirm_{hypothesis['param']}_"
                f"{hypothesis['value']}_{tag}{slot.pair_index}"
            )
            result = run_slot(eid, config)
            ledger = ledger_from_result(result)
            if ledger is None:
                raise RuntimeError(
                    "confirmation slot produced no request ledger; "
                    "run_benchmark must persist ④ request_ledger on ExperimentResult"
                )
            if ledger.run_id != getattr(result, "run_id", None):
                raise RuntimeError(
                    "confirmation slot ledger run_id does not match ExperimentResult.run_id"
                )
            if arm == RepeatArm.CANDIDATE:
                self.last_candidate_result = result
            return ledger

    return _SlotRunner()


def run_confirmation_campaign(
    run_arm: Callable[[RepeatArm, RepeatSlot], Any],
    *,
    n_pairs: int = DEFAULT_MIN_PAIRS,
    metric: str = "throughput_rps",
    phase: RepeatPhase = RepeatPhase.CONFIRMATION,
    expected_conditions: RunConditions | None = None,
    start_arm: RepeatArm = RepeatArm.BASELINE,
) -> tuple[RepeatCampaign, ConfirmationDecision]:
    """One interleaved B/C campaign → official ⑤ decision.

    ``run_arm`` is the per-slot runner. Production wires this to
    ``run_benchmark`` at the tool boundary. Tests may inject a fixture
    ``run_arm`` that returns pre-authored ledgers so CI stays GPU-free.
    """
    campaign = run_interleaved_repeats(
        run_arm,
        n_pairs,
        phase=phase,
        expected_conditions=expected_conditions,
        start_arm=start_arm,
    )
    decision = evaluate_campaign(campaign, metric=metric)
    return campaign, decision


def campaign_to_repeat_ledgers(
    campaign: RepeatCampaign,
    *,
    metric: str,
) -> dict[str, Any]:
    return {
        "baseline": list(campaign.baseline_ledgers),
        "candidate": list(campaign.candidate_ledgers),
        "phase": campaign.phase,
        "metric": metric,
        "schedule": list(campaign.schedule),
        "conditions": campaign.conditions,
        "min_pairs": DEFAULT_MIN_PAIRS,
    }


def clear_confirmation_fields() -> dict[str, Any]:
    return {
        "confirmation_decision": None,
        "repeat_ledgers": None,
        "confirmation_target": None,
        "confirmation_bound_run_ids": None,
    }
