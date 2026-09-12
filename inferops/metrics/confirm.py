"""Week-2 P0-⑤: interleaved repeats + confirmation verdicts.

Consumes the P0-④ ledger only (`RequestLedger` / `RunConditions` /
`recalculate_from_ledger`). Does not invent a second metrics schema.

Search-phase winners are tracked separately from confirmation. Accidental
search wins cannot become `confirmed_improvement` and cannot pass
`is_confirmed_promotable`. Week-1 `is_promotable` / `derive_status` are
unchanged and still required.
"""

from __future__ import annotations

from enum import Enum
from statistics import median
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, model_validator

from inferops.metrics.aggregate import AggregateMetrics, recalculate_from_ledger
from inferops.metrics.ledger import RequestLedger, RunConditions
from inferops.schemas import ExperimentResult, is_promotable

# ---------------------------------------------------------------------------
# Constants (stable for Tune / later Week-2 consumers)
# ---------------------------------------------------------------------------

DEFAULT_MIN_PAIRS: int = 3
DEFAULT_MIN_REL_DELTA: float = 0.05  # 5% relative; smaller deltas are no_diff

# Primary metrics read from AggregateMetrics (never invented as 0).
HIGHER_IS_BETTER: frozenset[str] = frozenset(
    {"throughput_rps", "tokens_per_second"}
)
LOWER_IS_BETTER: frozenset[str] = frozenset(
    {
        "error_rate",
        "ttft_p50_ms",
        "ttft_p99_ms",
        "tpot_p50_ms",
        "tpot_p99_ms",
        "e2e_p50_ms",
        "e2e_p99_ms",
    }
)
CONFIRMABLE_METRICS: frozenset[str] = HIGHER_IS_BETTER | LOWER_IS_BETTER

# Measurement fields that must match across interleaved repeats.
_CONDITION_COMPARE_FIELDS: tuple[str, ...] = (
    "workload_name",
    "num_requests",
    "concurrency",
    "input_len_target",
    "output_len_target",
    "distribution",
    "arrival_rps",
    "warmup_requests",
    "stream_response",
    "sampling_temperature",
    "cache_enabled",
)


class RepeatArm(str, Enum):
    BASELINE = "baseline"
    CANDIDATE = "candidate"


class RepeatPhase(str, Enum):
    SEARCH = "search"
    CONFIRMATION = "confirmation"


class ConfirmationVerdict(str, Enum):
    """Deterministic confirmation verdict. Search cannot emit confirmed_improvement."""

    NO_DIFF = "no_diff"
    REGRESSION = "regression"
    TOO_NOISY = "too_noisy"
    CONFIRMED_IMPROVEMENT = "confirmed_improvement"


class NumericSignal(str, Enum):
    """Direction from paired aggregates, before the confirmation-phase gate."""

    IMPROVEMENT = "improvement"
    REGRESSION = "regression"
    NO_DIFF = "no_diff"
    TOO_NOISY = "too_noisy"


class PairClass(str, Enum):
    BETTER = "better"
    WORSE = "worse"
    TIE = "tie"
    MISSING = "missing"


class RepeatSlot(BaseModel):
    """One independent run in an interleaved baseline/candidate schedule."""

    sequence_index: int
    pair_index: int
    arm: RepeatArm
    phase: RepeatPhase


class RepeatPair(BaseModel):
    """One independent (baseline, candidate) pair after recalculation."""

    pair_index: int
    baseline_run_id: str
    candidate_run_id: str
    baseline_value: float | None = None
    candidate_value: float | None = None
    rel_delta: float | None = None  # positive = candidate better; None if missing
    classification: PairClass


class RepeatCampaign(BaseModel):
    """Interleaved repeats under one RunConditions fingerprint."""

    phase: RepeatPhase
    conditions: RunConditions
    schedule: list[RepeatSlot] = Field(default_factory=list)
    baseline_ledgers: list[RequestLedger] = Field(default_factory=list)
    candidate_ledgers: list[RequestLedger] = Field(default_factory=list)


class ConfirmationDecision(BaseModel):
    """Tune/⑤-facing confirmation result. Rules live here, not in prose."""

    phase: RepeatPhase
    verdict: ConfirmationVerdict
    numeric_signal: NumericSignal
    search_winner: bool = False
    metric: str
    min_pairs: int = DEFAULT_MIN_PAIRS
    min_rel_delta: float = DEFAULT_MIN_REL_DELTA
    pair_count: int = 0
    usable_pairs: int = 0
    median_rel_delta: float | None = None
    median_improvement_pct: float | None = None  # None if not computable — never 0
    pairs: list[RepeatPair] = Field(default_factory=list)
    reason: str = ""

    @model_validator(mode="after")
    def _encode_phase_and_missing_rules(self) -> ConfirmationDecision:
        if (
            self.verdict == ConfirmationVerdict.CONFIRMED_IMPROVEMENT
            and self.phase != RepeatPhase.CONFIRMATION
        ):
            raise ValueError(
                "confirmed_improvement is illegal outside phase=confirmation"
            )
        if self.phase == RepeatPhase.SEARCH:
            self.search_winner = self.numeric_signal == NumericSignal.IMPROVEMENT
            if self.verdict == ConfirmationVerdict.CONFIRMED_IMPROVEMENT:
                raise ValueError("search phase cannot confirm improvement")
        else:
            self.search_winner = False
        if self.median_rel_delta is None:
            self.median_improvement_pct = None
        return self


def interleave_schedule(
    n_pairs: int,
    *,
    phase: RepeatPhase = RepeatPhase.CONFIRMATION,
    start_arm: RepeatArm = RepeatArm.BASELINE,
) -> list[RepeatSlot]:
    """Independent repeats, interleaved: B0 C0 B1 C1 … (or C0 B0 …)."""
    if n_pairs < 1:
        raise ValueError("n_pairs must be >= 1")
    other = (
        RepeatArm.CANDIDATE
        if start_arm == RepeatArm.BASELINE
        else RepeatArm.BASELINE
    )
    slots: list[RepeatSlot] = []
    seq = 0
    for pair_index in range(n_pairs):
        for arm in (start_arm, other):
            slots.append(
                RepeatSlot(
                    sequence_index=seq,
                    pair_index=pair_index,
                    arm=arm,
                    phase=phase,
                )
            )
            seq += 1
    return slots


def conditions_fingerprint(conditions: RunConditions) -> dict[str, Any]:
    """Stable measurement-context dict (no invented GPU/cost/perf)."""
    return {name: getattr(conditions, name) for name in _CONDITION_COMPARE_FIELDS}


def conditions_match(a: RunConditions, b: RunConditions) -> bool:
    return conditions_fingerprint(a) == conditions_fingerprint(b)


def require_same_conditions(ledgers: list[RequestLedger]) -> RunConditions:
    """All repeats in a campaign must share RunConditions. Mismatch is an error."""
    if not ledgers:
        raise ValueError("cannot compare an empty ledger list")
    first = ledgers[0].conditions
    for ledger in ledgers[1:]:
        if not conditions_match(first, ledger.conditions):
            raise ValueError(
                f"RunConditions mismatch between run_id={ledgers[0].run_id!r} "
                f"and run_id={ledger.run_id!r}; refuse comparison"
            )
    return first


def primary_metric_value(agg: AggregateMetrics, metric: str) -> float | None:
    """Read one confirmable metric. Missing stays None — never coerced to 0."""
    if metric not in CONFIRMABLE_METRICS:
        raise ValueError(f"unsupported confirmation metric: {metric!r}")
    if metric == "throughput_rps":
        return agg.throughput_rps
    if metric == "tokens_per_second":
        return agg.tokens_per_second
    if metric == "error_rate":
        return agg.error_rate
    lat_name, pct = _split_latency_metric(metric)
    stat = getattr(agg, lat_name)
    return getattr(stat, pct)


def _split_latency_metric(metric: str) -> tuple[str, str]:
    # ttft_p50_ms → ("ttft", "p50")
    body = metric.removesuffix("_ms")
    name, pct = body.rsplit("_", 1)
    return name if name != "e2e" else "e2e", pct


def relative_delta(
    baseline: float | None,
    candidate: float | None,
    *,
    metric: str,
) -> float | None:
    """Signed relative delta; positive means candidate is better.

    Missing or zero baseline → None (cannot invent a gain).
    """
    if baseline is None or candidate is None:
        return None
    if baseline == 0:
        return None
    raw = (candidate - baseline) / abs(baseline)
    if metric in LOWER_IS_BETTER:
        return -raw
    return raw


def classify_pair(delta: float | None, min_rel_delta: float) -> PairClass:
    if delta is None:
        return PairClass.MISSING
    if delta >= min_rel_delta:
        return PairClass.BETTER
    if delta <= -min_rel_delta:
        return PairClass.WORSE
    return PairClass.TIE


def _numeric_signal(
    classes: list[PairClass],
    deltas: list[float | None],
    *,
    min_pairs: int,
    min_rel_delta: float,
) -> tuple[NumericSignal, str]:
    usable = [c for c in classes if c != PairClass.MISSING]
    usable_deltas = [d for d in deltas if d is not None]
    if len(usable) < min_pairs:
        if any(c == PairClass.MISSING for c in classes):
            return (
                NumericSignal.TOO_NOISY,
                "missing_primary_metric — None cannot drive a gain",
            )
        return NumericSignal.TOO_NOISY, "insufficient_pairs"
    if any(c == PairClass.BETTER for c in usable) and any(
        c == PairClass.WORSE for c in usable
    ):
        return NumericSignal.TOO_NOISY, "pair_disagreement"
    n_better = sum(1 for c in usable if c == PairClass.BETTER)
    n_worse = sum(1 for c in usable if c == PairClass.WORSE)
    med = median(usable_deltas)
    if n_better == len(usable) and med >= min_rel_delta:
        return NumericSignal.IMPROVEMENT, "all_pairs_better"
    if n_worse == len(usable) and med <= -min_rel_delta:
        return NumericSignal.REGRESSION, "all_pairs_worse"
    return NumericSignal.NO_DIFF, "median_within_threshold"


def _verdict_for_phase(
    signal: NumericSignal, phase: RepeatPhase
) -> ConfirmationVerdict:
    if signal == NumericSignal.TOO_NOISY:
        return ConfirmationVerdict.TOO_NOISY
    if signal == NumericSignal.REGRESSION:
        return ConfirmationVerdict.REGRESSION
    if signal == NumericSignal.NO_DIFF:
        return ConfirmationVerdict.NO_DIFF
    if phase != RepeatPhase.CONFIRMATION:
        # Search-phase improvement is a search winner, not a confirmation.
        return ConfirmationVerdict.NO_DIFF
    return ConfirmationVerdict.CONFIRMED_IMPROVEMENT


def verdict_from_ledgers(
    baseline_ledgers: list[RequestLedger],
    candidate_ledgers: list[RequestLedger],
    *,
    metric: str = "throughput_rps",
    phase: RepeatPhase = RepeatPhase.CONFIRMATION,
    min_pairs: int = DEFAULT_MIN_PAIRS,
    min_rel_delta: float = DEFAULT_MIN_REL_DELTA,
) -> ConfirmationDecision:
    """Deterministic verdict from ④ aggregates. Does not invent missing values."""
    if metric not in CONFIRMABLE_METRICS:
        raise ValueError(f"unsupported confirmation metric: {metric!r}")
    if len(baseline_ledgers) != len(candidate_ledgers):
        raise ValueError(
            "baseline and candidate must have the same number of independent repeats"
        )
    if not baseline_ledgers:
        return ConfirmationDecision(
            phase=phase,
            verdict=ConfirmationVerdict.TOO_NOISY,
            numeric_signal=NumericSignal.TOO_NOISY,
            metric=metric,
            min_pairs=min_pairs,
            min_rel_delta=min_rel_delta,
            reason="insufficient_pairs",
        )

    require_same_conditions(list(baseline_ledgers) + list(candidate_ledgers))

    pairs: list[RepeatPair] = []
    for i, (base_ledger, cand_ledger) in enumerate(
        zip(baseline_ledgers, candidate_ledgers, strict=True)
    ):
        base_agg = recalculate_from_ledger(base_ledger)
        cand_agg = recalculate_from_ledger(cand_ledger)
        base_val = primary_metric_value(base_agg, metric)
        cand_val = primary_metric_value(cand_agg, metric)
        delta = relative_delta(base_val, cand_val, metric=metric)
        pairs.append(
            RepeatPair(
                pair_index=i,
                baseline_run_id=base_ledger.run_id,
                candidate_run_id=cand_ledger.run_id,
                baseline_value=base_val,
                candidate_value=cand_val,
                rel_delta=delta,
                classification=classify_pair(delta, min_rel_delta),
            )
        )

    classes = [p.classification for p in pairs]
    deltas = [p.rel_delta for p in pairs]
    signal, reason = _numeric_signal(
        classes, deltas, min_pairs=min_pairs, min_rel_delta=min_rel_delta
    )
    if signal == NumericSignal.IMPROVEMENT and phase != RepeatPhase.CONFIRMATION:
        reason = "search_phase_unconfirmed"
    usable_deltas = [d for d in deltas if d is not None]
    med = median(usable_deltas) if usable_deltas else None

    return ConfirmationDecision(
        phase=phase,
        verdict=_verdict_for_phase(signal, phase),
        numeric_signal=signal,
        metric=metric,
        min_pairs=min_pairs,
        min_rel_delta=min_rel_delta,
        pair_count=len(pairs),
        usable_pairs=sum(1 for c in classes if c != PairClass.MISSING),
        median_rel_delta=med,
        median_improvement_pct=(med * 100.0) if med is not None else None,
        pairs=pairs,
        reason=reason,
    )


def evaluate_campaign(
    campaign: RepeatCampaign,
    *,
    metric: str = "throughput_rps",
    min_pairs: int = DEFAULT_MIN_PAIRS,
    min_rel_delta: float = DEFAULT_MIN_REL_DELTA,
) -> ConfirmationDecision:
    require_same_conditions(campaign.baseline_ledgers + campaign.candidate_ledgers)
    if not conditions_match(
        campaign.conditions, campaign.baseline_ledgers[0].conditions
    ):
        raise ValueError("campaign.conditions does not match ledger RunConditions")
    return verdict_from_ledgers(
        campaign.baseline_ledgers,
        campaign.candidate_ledgers,
        metric=metric,
        phase=campaign.phase,
        min_pairs=min_pairs,
        min_rel_delta=min_rel_delta,
    )


def run_interleaved_repeats(
    run_arm: Callable[[RepeatArm, RepeatSlot], RequestLedger],
    n_pairs: int,
    *,
    phase: RepeatPhase,
    expected_conditions: RunConditions | None = None,
    start_arm: RepeatArm = RepeatArm.BASELINE,
) -> RepeatCampaign:
    """Execute interleaved independent repeats. Caller supplies the run hook.

    Fixture / CPU runners are the default proof path. A real GPU hook may be
    injected later — GPU-not-run is not a pass and must not invent numbers.
    """
    schedule = interleave_schedule(n_pairs, phase=phase, start_arm=start_arm)
    baseline: list[RequestLedger] = []
    candidate: list[RequestLedger] = []
    for slot in schedule:
        ledger = run_arm(slot.arm, slot)
        if expected_conditions is not None and not conditions_match(
            ledger.conditions, expected_conditions
        ):
            raise ValueError(
                f"run_id={ledger.run_id!r} RunConditions differ from expected"
            )
        if slot.arm == RepeatArm.BASELINE:
            baseline.append(ledger)
        else:
            candidate.append(ledger)
    conditions = require_same_conditions(baseline + candidate)
    return RepeatCampaign(
        phase=phase,
        conditions=conditions,
        schedule=schedule,
        baseline_ledgers=baseline,
        candidate_ledgers=candidate,
    )


def is_confirmed_promotable(
    result: ExperimentResult | None,
    decision: ConfirmationDecision | None,
) -> bool:
    """Promotion after ⑤: Week-1 gate AND confirmation-phase improvement.

    Search winners, noisy repeats, missing metrics, and unevidenced rows
    all stay False. Does not loosen `is_promotable`.
    """
    if result is None or decision is None:
        return False
    if not is_promotable(result):
        return False
    if decision.phase != RepeatPhase.CONFIRMATION:
        return False
    if decision.verdict != ConfirmationVerdict.CONFIRMED_IMPROVEMENT:
        return False
    if result.successful_requests <= 0:
        return False
    return True


def format_confirmation_report(decision: ConfirmationDecision) -> str:
    """Markdown for Tune/⑤ — same numbers as the decision object."""

    def fmt(v: float | None, digits: int = 4) -> str:
        if v is None:
            return "n/a"
        return f"{v:.{digits}f}"

    lines = [
        f"### Confirmation (`phase={decision.phase.value}`)",
        "",
        f"- **verdict**: `{decision.verdict.value}`",
        f"- **numeric_signal**: `{decision.numeric_signal.value}`",
        f"- **search_winner**: `{decision.search_winner}`",
        f"- **metric**: `{decision.metric}`",
        f"- **pairs**: {decision.usable_pairs}/{decision.pair_count} usable "
        f"(min_pairs={decision.min_pairs})",
        f"- **median_rel_delta**: {fmt(decision.median_rel_delta)}",
        f"- **median_improvement_pct**: {fmt(decision.median_improvement_pct, 2)}",
        f"- **reason**: `{decision.reason}`",
        "",
        "| pair | baseline | candidate | rel_delta | class |",
        "|---:|---:|---:|---:|---|",
    ]
    for p in decision.pairs:
        lines.append(
            f"| {p.pair_index} | {fmt(p.baseline_value)} | {fmt(p.candidate_value)} "
            f"| {fmt(p.rel_delta)} | `{p.classification.value}` |"
        )
    lines.append("")
    return "\n".join(lines)
