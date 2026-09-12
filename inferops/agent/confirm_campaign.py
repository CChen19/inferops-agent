"""Thin Tune adapter: drive ⑤ interleaved confirmation from Reflect remasure.

Consumes ``run_interleaved_repeats`` / ``evaluate_campaign`` / ``verdict_from_ledgers``.
Does not mint ``confirmed_improvement`` and does not invent a metrics schema.

CI / offline tests inject a ledger ``run_arm`` (no GPU). Production wiring may
call ``run_benchmark`` per interleave slot — that path is not exercised in CI
and must not invent GPU numbers.
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
    run_interleaved_repeats,
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


def run_confirmation_campaign(
    run_arm: Callable[[RepeatArm, RepeatSlot], Any],
    *,
    n_pairs: int = DEFAULT_MIN_PAIRS,
    metric: str = "throughput_rps",
    phase: RepeatPhase = RepeatPhase.CONFIRMATION,
    expected_conditions: RunConditions | None = None,
    start_arm: RepeatArm = RepeatArm.BASELINE,
) -> tuple[RepeatCampaign, ConfirmationDecision]:
    """One interleaved B/C campaign → official ⑤ decision. Fixture ``run_arm`` OK."""
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
