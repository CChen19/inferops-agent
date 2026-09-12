"""Week-2 P0-⑤: interleaved repeats + confirmation gate.

Fixture / CPU proof only. GPU-not-run is not a pass and invents no numbers.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inferops.metrics.aggregate import recalculate_from_ledger
from inferops.metrics.confirm import (
    DEFAULT_MIN_PAIRS,
    ConfirmationDecision,
    ConfirmationVerdict,
    NumericSignal,
    PairClass,
    RepeatArm,
    RepeatCampaign,
    RepeatPair,
    RepeatPhase,
    conditions_match,
    evaluate_campaign,
    format_confirmation_report,
    interleave_schedule,
    is_confirmed_promotable,
    primary_metric_value,
    require_same_conditions,
    run_interleaved_repeats,
    verdict_from_ledgers,
)
from inferops.metrics.ledger import (
    RequestLedger,
    RequestOutcome,
    RequestRecord,
    RunConditions,
    TerminationReason,
    TokenCountSource,
)
from inferops.schemas import (
    ExperimentValidityStatus,
    derive_status,
    is_promotable,
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


def _success_record(
    run_id: str,
    request_id: str,
    *,
    t0: float,
    e2e_s: float = 0.2,
    output_tokens: int | None = 16,
    token_count_source: TokenCountSource = TokenCountSource.USAGE,
    outcome: RequestOutcome = RequestOutcome.SUCCESS,
    termination: TerminationReason = TerminationReason.STOP,
) -> RequestRecord:
    t_first = t0 + 0.05 if outcome in (RequestOutcome.SUCCESS, RequestOutcome.TRUNCATE) else None
    t_end = t0 + e2e_s
    return RequestRecord(
        run_id=run_id,
        request_id=request_id,
        t_start_s=t0,
        t_first_token_s=t_first,
        t_end_s=t_end,
        output_tokens=output_tokens,
        input_tokens=8,
        token_count_source=token_count_source,
        outcome=outcome,
        termination_reason=termination,
    )


def make_rps_ledger(
    run_id: str,
    *,
    rps: float,
    n_success: int = 10,
    n_error: int = 0,
    error_outcome: RequestOutcome = RequestOutcome.TIMEOUT,
    tokens: int | None = 16,
    token_count_source: TokenCountSource = TokenCountSource.USAGE,
    conditions: RunConditions = CONDITIONS,
) -> RequestLedger:
    """Deterministic ledger with exact throughput_rps = n_success / window."""
    total = n_success + n_error
    window_s = total / rps if n_error == 0 else n_success / rps if rps else 1.0
    if n_error:
        # Keep successful / window = rps; errors sit inside the same window.
        window_s = n_success / rps
    ledger = RequestLedger(
        run_id=run_id,
        conditions=conditions,
        window_start_s=1000.0,
        window_end_s=1000.0 + window_s,
    )
    t0 = 1000.0
    for i in range(n_success):
        ledger.add(
            _success_record(
                run_id,
                f"req-{i:04d}",
                t0=t0 + i * 0.01,
                output_tokens=tokens,
                token_count_source=token_count_source,
            )
        )
    for i in range(n_error):
        term = {
            RequestOutcome.TIMEOUT: TerminationReason.TIMEOUT,
            RequestOutcome.CANCEL: TerminationReason.CANCEL,
            RequestOutcome.INCOMPLETE: TerminationReason.INCOMPLETE,
            RequestOutcome.FAIL: TerminationReason.ERROR,
        }[error_outcome]
        ledger.add(
            _success_record(
                run_id,
                f"err-{i:04d}",
                t0=t0 + 0.5 + i * 0.01,
                output_tokens=None,
                token_count_source=TokenCountSource.MISSING,
                outcome=error_outcome,
                termination=term,
            )
        )
    return ledger


def _n_ledgers(prefix: str, rps: float, n: int, **kwargs) -> list[RequestLedger]:
    return [make_rps_ledger(f"{prefix}_{i:02d}", rps=rps, **kwargs) for i in range(n)]


def _empty_ledgers(prefix: str, n: int) -> list[RequestLedger]:
    return [
        RequestLedger(run_id=f"{prefix}_{i:02d}", conditions=CONDITIONS)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

def test_interleave_schedule_is_independent_pairs():
    slots = interleave_schedule(3, phase=RepeatPhase.CONFIRMATION)
    assert [s.arm.value for s in slots] == [
        "baseline",
        "candidate",
        "baseline",
        "candidate",
        "baseline",
        "candidate",
    ]
    assert [s.pair_index for s in slots] == [0, 0, 1, 1, 2, 2]
    assert all(s.phase == RepeatPhase.CONFIRMATION for s in slots)


def test_run_interleaved_repeats_cpu_fixture_protocol():
    """CPU proof: fake runner, no GPU, numbers come only from ledgers."""

    def run_arm(arm: RepeatArm, slot):
        rps = 2.0 if arm == RepeatArm.BASELINE else 2.4
        return make_rps_ledger(f"{arm.value}-{slot.pair_index}", rps=rps)

    campaign = run_interleaved_repeats(
        run_arm,
        n_pairs=3,
        phase=RepeatPhase.CONFIRMATION,
        expected_conditions=CONDITIONS,
    )
    assert [s.arm for s in campaign.schedule][0] == RepeatArm.BASELINE
    assert len(campaign.baseline_ledgers) == 3
    assert len(campaign.candidate_ledgers) == 3
    assert conditions_match(campaign.conditions, CONDITIONS)
    decision = evaluate_campaign(campaign, metric="throughput_rps")
    assert decision.verdict == ConfirmationVerdict.CONFIRMED_IMPROVEMENT


def test_conditions_mismatch_refuses_comparison():
    other = CONDITIONS.model_copy(update={"concurrency": 16})
    base = _n_ledgers("b", 2.0, 3)
    cand = _n_ledgers("c", 2.4, 3, conditions=other)
    with pytest.raises(ValueError, match="RunConditions mismatch"):
        require_same_conditions(base + cand)
    with pytest.raises(ValueError, match="RunConditions mismatch"):
        verdict_from_ledgers(base, cand, phase=RepeatPhase.CONFIRMATION)


# ---------------------------------------------------------------------------
# Verdicts from ledger aggregates
# ---------------------------------------------------------------------------

def test_confirmation_consistent_gain_is_confirmed_improvement():
    decision = verdict_from_ledgers(
        _n_ledgers("b", 2.0, 3),
        _n_ledgers("c", 2.4, 3),
        metric="throughput_rps",
        phase=RepeatPhase.CONFIRMATION,
    )
    assert decision.numeric_signal == NumericSignal.IMPROVEMENT
    assert decision.verdict == ConfirmationVerdict.CONFIRMED_IMPROVEMENT
    assert decision.search_winner is False
    assert decision.usable_pairs == 3
    assert decision.median_rel_delta == pytest.approx(0.2)
    assert decision.median_improvement_pct == pytest.approx(20.0)
    assert all(p.classification == PairClass.BETTER for p in decision.pairs)


def test_search_phase_win_is_not_confirmed_and_does_not_promote(result_b):
    """Accidental / search-phase win must not auto-promote."""
    decision = verdict_from_ledgers(
        _n_ledgers("sb", 2.0, 1),
        _n_ledgers("sc", 3.0, 1),  # +50% on a single lucky pair
        metric="throughput_rps",
        phase=RepeatPhase.SEARCH,
        min_pairs=1,
    )
    assert decision.numeric_signal == NumericSignal.IMPROVEMENT
    assert decision.search_winner is True
    assert decision.verdict == ConfirmationVerdict.NO_DIFF
    assert decision.reason == "search_phase_unconfirmed"
    assert is_promotable(result_b) is True
    assert is_confirmed_promotable(result_b, decision) is False


def test_single_confirmation_pair_is_too_noisy_not_confirmed(result_b):
    """A one-shot confirmation repeat cannot satisfy min_pairs=3."""
    decision = verdict_from_ledgers(
        _n_ledgers("ob", 2.0, 1),
        _n_ledgers("oc", 3.0, 1),
        metric="throughput_rps",
        phase=RepeatPhase.CONFIRMATION,
        min_pairs=DEFAULT_MIN_PAIRS,
    )
    assert decision.verdict == ConfirmationVerdict.TOO_NOISY
    assert decision.reason == "insufficient_pairs"
    assert is_confirmed_promotable(result_b, decision) is False


def test_disagreeing_pairs_are_too_noisy(result_b):
    base = [
        make_rps_ledger("nb0", rps=2.0),
        make_rps_ledger("nb1", rps=2.0),
        make_rps_ledger("nb2", rps=2.0),
    ]
    cand = [
        make_rps_ledger("nc0", rps=3.0),  # +50%
        make_rps_ledger("nc1", rps=1.2),  # -40%
        make_rps_ledger("nc2", rps=2.6),  # +30%
    ]
    decision = verdict_from_ledgers(
        base, cand, metric="throughput_rps", phase=RepeatPhase.CONFIRMATION
    )
    assert decision.verdict == ConfirmationVerdict.TOO_NOISY
    assert decision.reason == "pair_disagreement"
    assert is_confirmed_promotable(result_b, decision) is False


def test_regression_verdict():
    decision = verdict_from_ledgers(
        _n_ledgers("rb", 2.0, 3),
        _n_ledgers("rc", 1.6, 3),
        metric="throughput_rps",
        phase=RepeatPhase.CONFIRMATION,
    )
    assert decision.numeric_signal == NumericSignal.REGRESSION
    assert decision.verdict == ConfirmationVerdict.REGRESSION
    assert decision.median_rel_delta == pytest.approx(-0.2)


def test_no_diff_within_threshold():
    decision = verdict_from_ledgers(
        _n_ledgers("db", 2.00, 3),
        _n_ledgers("dc", 2.04, 3),  # +2% < 5%
        metric="throughput_rps",
        phase=RepeatPhase.CONFIRMATION,
    )
    assert decision.verdict == ConfirmationVerdict.NO_DIFF
    assert decision.numeric_signal == NumericSignal.NO_DIFF
    assert decision.reason == "median_within_threshold"


def test_schema_rejects_confirmed_improvement_on_search_phase():
    with pytest.raises(ValidationError, match="illegal outside phase=confirmation"):
        ConfirmationDecision(
            phase=RepeatPhase.SEARCH,
            verdict=ConfirmationVerdict.CONFIRMED_IMPROVEMENT,
            numeric_signal=NumericSignal.IMPROVEMENT,
            metric="throughput_rps",
        )


def test_forged_confirmed_improvement_usable_pairs_zero_rejected():
    with pytest.raises(ValidationError, match="forged ConfirmationDecision"):
        ConfirmationDecision(
            phase=RepeatPhase.CONFIRMATION,
            verdict=ConfirmationVerdict.CONFIRMED_IMPROVEMENT,
            numeric_signal=NumericSignal.IMPROVEMENT,
            metric="throughput_rps",
            usable_pairs=0,
            pair_count=0,
        )


def test_forged_confirmed_improvement_without_improvement_signal_rejected():
    with pytest.raises(ValidationError, match="forged ConfirmationDecision"):
        ConfirmationDecision(
            phase=RepeatPhase.CONFIRMATION,
            verdict=ConfirmationVerdict.CONFIRMED_IMPROVEMENT,
            numeric_signal=NumericSignal.TOO_NOISY,
            metric="throughput_rps",
            usable_pairs=3,
            pair_count=3,
        )


def test_forged_confirmed_improvement_with_fake_pairs_rejected(result_b):
    """Even a fully populated hand-built decision cannot confirm."""
    pairs = [
        RepeatPair(
            pair_index=i,
            baseline_run_id=f"forge_b{i}",
            candidate_run_id=f"forge_c{i}",
            baseline_value=2.0,
            candidate_value=3.0,
            rel_delta=0.5,
            classification=PairClass.BETTER,
        )
        for i in range(3)
    ]
    with pytest.raises(ValidationError, match="forged ConfirmationDecision"):
        ConfirmationDecision(
            phase=RepeatPhase.CONFIRMATION,
            verdict=ConfirmationVerdict.CONFIRMED_IMPROVEMENT,
            numeric_signal=NumericSignal.IMPROVEMENT,
            metric="throughput_rps",
            min_pairs=3,
            usable_pairs=3,
            pair_count=3,
            median_rel_delta=0.5,
            pairs=pairs,
        )
    assert is_promotable(result_b) is True


def test_duplicate_run_id_pairs_rejected():
    same_b = make_rps_ledger("dup_b", rps=2.0)
    same_c = make_rps_ledger("dup_c", rps=2.4)
    with pytest.raises(ValueError, match="same RequestLedger object reused"):
        verdict_from_ledgers(
            [same_b, same_b, same_b],
            [same_c, same_c, same_c],
            phase=RepeatPhase.CONFIRMATION,
        )
    clones_b = [make_rps_ledger("dup_b", rps=2.0) for _ in range(3)]
    clones_c = [make_rps_ledger("dup_c", rps=2.4) for _ in range(3)]
    with pytest.raises(ValueError, match="duplicate run_id"):
        verdict_from_ledgers(clones_b, clones_c, phase=RepeatPhase.CONFIRMATION)


def test_evaluate_campaign_and_interleave_reject_duplicate_run_ids():
    with pytest.raises(ValidationError, match="same RequestLedger|duplicate run_id"):
        RepeatCampaign(
            phase=RepeatPhase.CONFIRMATION,
            conditions=CONDITIONS,
            baseline_ledgers=[make_rps_ledger("camp_b", rps=2.0)] * 3,
            candidate_ledgers=[make_rps_ledger("camp_c", rps=2.4)] * 3,
        )

    def run_arm(arm: RepeatArm, slot):
        return make_rps_ledger("always-the-same", rps=2.0)

    with pytest.raises(ValueError, match="duplicate run_id"):
        run_interleaved_repeats(
            run_arm, n_pairs=3, phase=RepeatPhase.CONFIRMATION
        )


def test_invalid_min_pairs_and_min_rel_delta_rejected():
    base = _n_ledgers("bound_b", 2.0, 3)
    cand = _n_ledgers("bound_c", 2.4, 3)
    with pytest.raises(ValueError, match="min_pairs must be > 0"):
        verdict_from_ledgers(base, cand, min_pairs=0)
    with pytest.raises(ValueError, match="min_pairs must be > 0"):
        verdict_from_ledgers(base, cand, min_pairs=-1)
    with pytest.raises(ValueError, match="min_rel_delta must be > 0"):
        verdict_from_ledgers(base, cand, min_rel_delta=0.0)
    with pytest.raises(ValueError, match="min_rel_delta must be > 0"):
        verdict_from_ledgers(base, cand, min_rel_delta=-0.05)
    with pytest.raises(ValidationError):
        ConfirmationDecision(
            phase=RepeatPhase.CONFIRMATION,
            verdict=ConfirmationVerdict.TOO_NOISY,
            numeric_signal=NumericSignal.TOO_NOISY,
            metric="throughput_rps",
            min_pairs=0,
        )
    with pytest.raises(ValidationError):
        ConfirmationDecision(
            phase=RepeatPhase.CONFIRMATION,
            verdict=ConfirmationVerdict.TOO_NOISY,
            numeric_signal=NumericSignal.TOO_NOISY,
            metric="throughput_rps",
            min_rel_delta=0.0,
        )


# ---------------------------------------------------------------------------
# Missing metrics / provenance cannot fake gains
# ---------------------------------------------------------------------------

def test_missing_token_provenance_cannot_drive_tok_s(result_b):
    """token_count_source=missing → tok-s is None → cannot confirm a token gain."""
    base = _n_ledgers("tb", 2.0, 3, tokens=16, token_count_source=TokenCountSource.USAGE)
    cand = _n_ledgers(
        "tc",
        2.0,
        3,
        tokens=999,  # populated but unusable
        token_count_source=TokenCountSource.MISSING,
    )
    # ④ still holds on the raw aggregates
    cand_agg = recalculate_from_ledger(cand[0])
    assert cand_agg.tokens_per_second is None
    assert cand_agg.tpot.p50 is None
    assert primary_metric_value(cand_agg, "tokens_per_second") is None

    decision = verdict_from_ledgers(
        base, cand, metric="tokens_per_second", phase=RepeatPhase.CONFIRMATION
    )
    assert all(p.classification == PairClass.MISSING for p in decision.pairs)
    assert all(p.candidate_value is None for p in decision.pairs)
    assert decision.verdict == ConfirmationVerdict.TOO_NOISY
    assert "missing_primary_metric" in decision.reason
    assert decision.median_improvement_pct is None
    assert is_confirmed_promotable(result_b, decision) is False


def test_incomplete_timeout_cancel_are_not_success():
    """Errors in the candidate cannot be packaged as a throughput win."""
    base = _n_ledgers("eb", 2.0, 3, n_success=10, n_error=0)
    # Same window-implied rps from successes only; extra incompletes don't add rps.
    cand = [
        make_rps_ledger(
            f"ec_{i:02d}",
            rps=2.0,
            n_success=10,
            n_error=5,
            error_outcome=outcome,
        )
        for i, outcome in enumerate(
            (
                RequestOutcome.INCOMPLETE,
                RequestOutcome.TIMEOUT,
                RequestOutcome.CANCEL,
            )
        )
    ]
    for ledger in cand:
        agg = recalculate_from_ledger(ledger)
        assert agg.successful_requests == 10
        assert agg.failed_requests == 5
        assert agg.throughput_rps == pytest.approx(2.0)

    decision = verdict_from_ledgers(
        base, cand, metric="throughput_rps", phase=RepeatPhase.CONFIRMATION
    )
    assert decision.verdict == ConfirmationVerdict.NO_DIFF
    assert is_confirmed_promotable(None, decision) is False


def test_missing_primary_stays_none_never_zero():
    empty = RequestLedger(run_id="empty_base_00", conditions=CONDITIONS)
    agg = recalculate_from_ledger(empty)
    assert agg.throughput_rps is None
    assert agg.tokens_per_second is None
    assert agg.ttft.p50 is None
    decision = verdict_from_ledgers(
        _empty_ledgers("empty_base", 3),
        _empty_ledgers("empty_cand", 3),
        metric="throughput_rps",
        phase=RepeatPhase.CONFIRMATION,
    )
    assert all(p.baseline_value is None and p.candidate_value is None for p in decision.pairs)
    assert decision.median_rel_delta is None
    assert decision.median_improvement_pct is None
    assert decision.verdict == ConfirmationVerdict.TOO_NOISY
    assert "missing_primary_metric" in decision.reason


def test_any_missing_primary_is_too_noisy_even_if_other_pairs_suffice(result_b):
    """P2: one missing primary cannot be dropped so the rest can confirm."""
    base = _n_ledgers("mixb", 2.0, 3) + _empty_ledgers("mixb_miss", 1)
    cand = _n_ledgers("mixc", 3.0, 3) + _empty_ledgers("mixc_miss", 1)
    decision = verdict_from_ledgers(
        base,
        cand,
        metric="throughput_rps",
        phase=RepeatPhase.CONFIRMATION,
        min_pairs=3,
    )
    assert decision.usable_pairs == 3
    assert decision.pair_count == 4
    assert decision.verdict == ConfirmationVerdict.TOO_NOISY
    assert "missing_primary_metric" in decision.reason
    assert is_confirmed_promotable(result_b, decision) is False


def test_gpu_not_run_is_not_a_pass_or_a_gain():
    """GPU-not-run stays None; confirmation does not invent 0% util as a win."""
    ledger = make_rps_ledger("gpu_none", rps=2.0)
    agg = recalculate_from_ledger(ledger)
    assert agg.gpu_utilization_pct is None
    assert agg.gpu_memory_used_gb is None
    assert agg.cost_usd is None
    # GPU is not a confirmable metric — asking for it is a caller error.
    with pytest.raises(ValueError, match="unsupported confirmation metric"):
        primary_metric_value(agg, "gpu_utilization_pct")


# ---------------------------------------------------------------------------
# Week-1 gate is untouched
# ---------------------------------------------------------------------------

def test_unevidenced_best_still_blocked_when_confirmation_looks_good(
    result_b, result_b_unevidenced
):
    decision = verdict_from_ledgers(
        _n_ledgers("ub", 2.0, 3),
        _n_ledgers("uc", 3.0, 3),
        metric="throughput_rps",
        phase=RepeatPhase.CONFIRMATION,
    )
    assert decision.verdict == ConfirmationVerdict.CONFIRMED_IMPROVEMENT
    assert is_promotable(result_b) is True
    assert is_confirmed_promotable(result_b, decision) is True
    assert is_promotable(result_b_unevidenced) is False
    assert is_confirmed_promotable(result_b_unevidenced, decision) is False


def test_week1_derive_status_and_promotable_unchanged(result_b, result_b_unevidenced):
    """⑤ must not loosen ① — same helpers, same answers."""
    assert is_promotable(result_b) is True
    assert is_promotable(result_b_unevidenced) is False
    assert (
        derive_status(
            evidence=result_b.config_evidence,
            actual_config=result_b.actual_config,
            requested_config=result_b.requested_config,
            successful_requests=result_b.successful_requests,
        )
        == ExperimentValidityStatus.VALID
    )
    unevidenced_status = derive_status(
        evidence=None,
        actual_config=None,
        requested_config=result_b_unevidenced.requested_config,
        successful_requests=99,
    )
    assert unevidenced_status == ExperimentValidityStatus.INSUFFICIENT_EVIDENCE


def test_zero_success_candidate_not_confirmed_promotable(result_b):
    failed = result_b.model_copy(update={"successful_requests": 0})
    assert is_promotable(failed) is False
    decision = verdict_from_ledgers(
        _n_ledgers("zb", 2.0, 3),
        _n_ledgers("zc", 3.0, 3),
        phase=RepeatPhase.CONFIRMATION,
    )
    assert is_confirmed_promotable(failed, decision) is False


def test_format_confirmation_report_marks_missing_as_na():
    decision = verdict_from_ledgers(
        _empty_ledgers("fb", 3),
        _empty_ledgers("fc", 3),
        phase=RepeatPhase.CONFIRMATION,
    )
    md = format_confirmation_report(decision)
    assert "n/a" in md
    assert "too_noisy" in md
    assert "phase=confirmation" in md
