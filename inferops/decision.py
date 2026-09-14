"""Structured end-of-run decision shared by the UI and Markdown report.

Four outcomes only:
  1. confirmed_and_meets_goals
  2. improved_but_unmet_goals
  3. no_reliable_improvement  (keep baseline)
  4. inconclusive             (insufficient evidence / execution failed)

The exported configuration is the *measured* ``actual_config``. Requested
knobs that were not evidenced are never presented as deployable.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from inferops.agent.state import is_promotable_summary
from inferops.task import (
    OptimizationTask,
    compare_op,
    task_conditions,
    task_from_mapping,
)


class DecisionKind(str, Enum):
    CONFIRMED_AND_MEETS_GOALS = "confirmed_and_meets_goals"
    IMPROVED_BUT_UNMET_GOALS = "improved_but_unmet_goals"
    NO_RELIABLE_IMPROVEMENT = "no_reliable_improvement"
    INCONCLUSIVE = "inconclusive"


_HEADLINES = {
    DecisionKind.CONFIRMED_AND_MEETS_GOALS: (
        "Confirmed improvement that satisfies the task goals. Adopt the measured configuration."
    ),
    DecisionKind.IMPROVED_BUT_UNMET_GOALS: (
        "Reliable improvement over baseline, but the serving goals are still not met."
    ),
    DecisionKind.NO_RELIABLE_IMPROVEMENT: (
        "No reliable improvement. Keep the baseline configuration."
    ),
    DecisionKind.INCONCLUSIVE: (
        "Evidence is insufficient or execution failed. No configuration can be recommended."
    ),
}


class GoalCheck(BaseModel):
    metric: str
    op: str
    limit: float
    observed: float | None
    ok: bool
    role: str = "constraint"  # constraint | goal
    reason: str = ""


class EvidenceLink(BaseModel):
    label: str
    experiment_id: str
    run_id: str = ""
    ledger_path: str | None = None
    mlflow_run_id: str | None = None
    validity_status: str = ""


class DecisionReport(BaseModel):
    kind: DecisionKind
    headline: str
    stop_reason: str = ""
    adopt: bool = False
    keep_baseline: bool = False
    meets_goals: bool | None = None
    goal_checks: list[GoalCheck] = Field(default_factory=list)
    adopted_config: dict[str, Any] | None = None
    requested_config: dict[str, Any] | None = None
    baseline_config: dict[str, Any] | None = None
    config_diff: dict[str, Any] = Field(default_factory=dict)
    exported_matches_measured: bool = False
    adopted_summary: dict[str, Any] | None = None
    baseline_summary: dict[str, Any] | None = None
    why_not_others: list[str] = Field(default_factory=list)
    evidence: list[EvidenceLink] = Field(default_factory=list)
    task_conditions: dict[str, Any] | None = None


def _is_candidate(summary: dict[str, Any] | None) -> bool:
    return bool(summary) and summary.get("param_changed") not in (None, "")


def _measured_config(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not summary:
        return None
    actual = summary.get("actual_config")
    return dict(actual) if isinstance(actual, dict) and actual else None


def _requested_config(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not summary:
        return None
    requested = summary.get("requested_config")
    return dict(requested) if isinstance(requested, dict) and requested else None


def _config_diff(baseline: dict[str, Any] | None, adopted: dict[str, Any] | None) -> dict[str, Any]:
    if not baseline and not adopted:
        return {}
    keys = sorted(set(baseline or {}) | set(adopted or {}))
    changed = {}
    for key in keys:
        left = (baseline or {}).get(key, "<missing>")
        right = (adopted or {}).get(key, "<missing>")
        if left != right:
            changed[key] = {"from": left, "to": right}
    return changed


def _evidence(label: str, summary: dict[str, Any] | None) -> EvidenceLink | None:
    if not summary:
        return None
    return EvidenceLink(
        label=label,
        experiment_id=str(summary.get("experiment_id") or ""),
        run_id=str(summary.get("run_id") or ""),
        ledger_path=summary.get("ledger_path"),
        mlflow_run_id=summary.get("mlflow_run_id"),
        validity_status=str(summary.get("validity_status") or ""),
    )


def evaluate_goals(
    summary: dict[str, Any] | None,
    task: OptimizationTask | None,
) -> tuple[bool | None, list[GoalCheck]]:
    """Check task constraints + target_qps against a summary.

    Missing observed values fail closed. No task → ``meets`` is None.
    """
    if task is None:
        return None, []
    if not summary:
        return False, [
            GoalCheck(
                metric="summary",
                op="present",
                limit=1,
                observed=None,
                ok=False,
                role="constraint",
                reason="no_summary",
            )
        ]

    checks: list[GoalCheck] = []
    for constraint in task.constraints:
        observed = summary.get(constraint.metric)
        if observed is None:
            checks.append(
                GoalCheck(
                    metric=constraint.metric,
                    op=constraint.op.value,
                    limit=constraint.value,
                    observed=None,
                    ok=False,
                    role="constraint",
                    reason=f"{constraint.metric}_missing",
                )
            )
            continue
        ok = compare_op(constraint.op, float(observed), constraint.value)
        checks.append(
            GoalCheck(
                metric=constraint.metric,
                op=constraint.op.value,
                limit=constraint.value,
                observed=float(observed),
                ok=ok,
                role="constraint",
                reason="ok" if ok else f"{constraint.metric}_unmet",
            )
        )

    if task.target_qps is not None:
        rps = summary.get("throughput_rps")
        if rps is None:
            checks.append(
                GoalCheck(
                    metric="throughput_rps",
                    op=">=",
                    limit=task.target_qps,
                    observed=None,
                    ok=False,
                    role="goal",
                    reason="target_qps_missing_measurement",
                )
            )
        else:
            ok = float(rps) >= float(task.target_qps)
            checks.append(
                GoalCheck(
                    metric="throughput_rps",
                    op=">=",
                    limit=float(task.target_qps),
                    observed=float(rps),
                    ok=ok,
                    role="goal",
                    reason="ok" if ok else "target_qps_unmet",
                )
            )

    return all(c.ok for c in checks), checks


def _has_reliable_improvement(best: dict[str, Any] | None, baseline: dict[str, Any] | None) -> bool:
    if not is_promotable_summary(best) or not _is_candidate(best):
        return False
    if baseline and best.get("experiment_id") == baseline.get("experiment_id"):
        return False
    vs = best.get("vs_baseline_pct")
    if vs is None:
        return False
    return float(vs) > 0


def _why_not_others(
    *,
    kind: DecisionKind,
    baseline: dict[str, Any] | None,
    best: dict[str, Any] | None,
    summaries: list[dict[str, Any]],
) -> list[str]:
    lines: list[str] = []
    adopted_id = (best or {}).get("experiment_id") if kind in {
        DecisionKind.CONFIRMED_AND_MEETS_GOALS,
        DecisionKind.IMPROVED_BUT_UNMET_GOALS,
    } else (baseline or {}).get("experiment_id")

    if baseline and adopted_id and baseline.get("experiment_id") != adopted_id:
        lines.append(
            f"Baseline `{baseline.get('experiment_id')}` was not adopted "
            f"(primary vs adopted is the search baseline; "
            f"validity=`{baseline.get('validity_status')}`)."
        )
    elif baseline and kind == DecisionKind.NO_RELIABLE_IMPROVEMENT:
        lines.append(
            f"Keep baseline `{baseline.get('experiment_id')}` because no candidate "
            "cleared confirmation + Week-1 promotion."
        )

    for s in summaries:
        eid = s.get("experiment_id")
        if not eid or eid == adopted_id:
            continue
        if s.get("param_changed") in (None, ""):
            continue
        status = s.get("validity_status") or "insufficient_evidence"
        if status == "failed":
            lines.append(
                f"Reject `{eid}`: execution failed "
                f"({s.get('failure_reason') or 'no reason'})."
            )
        elif not is_promotable_summary(s):
            lines.append(
                f"Reject `{eid}`: status=`{status}` / not promotable "
                "(Week-1 promotion gate rejected the row)."
            )
        elif s.get("vs_baseline_pct") is None:
            lines.append(f"Reject `{eid}`: primary metric missing, no invented gain.")
        elif (s.get("vs_baseline_pct") or 0) <= 0:
            lines.append(
                f"Reject `{eid}`: no positive confirmed gain "
                f"({s.get('vs_baseline_pct')}% vs baseline)."
            )
        elif eid != adopted_id:
            lines.append(
                f"Not adopted `{eid}`: not selected as confirmed best "
                f"(validity=`{status}`, vs_baseline={s.get('vs_baseline_pct')})."
            )
    return lines


def build_decision(
    *,
    baseline_summary: dict[str, Any] | None,
    best_summary: dict[str, Any] | None,
    experiment_summaries: list[dict[str, Any]] | None = None,
    optimization_task: dict[str, Any] | OptimizationTask | None = None,
    stop_reason: str = "",
) -> DecisionReport:
    summaries = list(experiment_summaries or [])
    task = task_from_mapping(optimization_task)
    cond = task_conditions(task) if task is not None else None
    baseline_ok = is_promotable_summary(baseline_summary)
    improved = _has_reliable_improvement(best_summary, baseline_summary)
    adopted = best_summary if improved else (baseline_summary if baseline_ok else None)
    meets, checks = evaluate_goals(adopted, task)

    candidates = [s for s in summaries if _is_candidate(s)]
    any_failed = any(s.get("validity_status") == "failed" for s in candidates)
    any_unusable = any(
        s.get("validity_status") in {"failed", "insufficient_evidence", "invalid"}
        for s in candidates
    )

    if improved and meets is not False:
        kind = DecisionKind.CONFIRMED_AND_MEETS_GOALS
    elif improved and meets is False:
        kind = DecisionKind.IMPROVED_BUT_UNMET_GOALS
    elif baseline_ok:
        kind = DecisionKind.NO_RELIABLE_IMPROVEMENT
    else:
        kind = DecisionKind.INCONCLUSIVE
        if not summaries and not stop_reason:
            stop_reason = stop_reason or "no_experiments"
        elif any_failed and not baseline_ok:
            stop_reason = stop_reason or "execution_failed"
        elif any_unusable or not baseline_ok:
            stop_reason = stop_reason or "insufficient_evidence"

    measured = _measured_config(adopted) if kind != DecisionKind.INCONCLUSIVE else None
    requested = _requested_config(adopted)
    baseline_cfg = _measured_config(baseline_summary) or _requested_config(baseline_summary)
    exported_ok = bool(measured) and kind != DecisionKind.INCONCLUSIVE

    evidence: list[EvidenceLink] = []
    for label, row in (("adopted", adopted), ("baseline", baseline_summary), ("best", best_summary)):
        link = _evidence(label, row)
        if link and all(e.experiment_id != link.experiment_id or e.label != label for e in evidence):
            evidence.append(link)
    for s in summaries:
        link = _evidence("run", s)
        if link and all(e.experiment_id != link.experiment_id for e in evidence):
            evidence.append(link)

    return DecisionReport(
        kind=kind,
        headline=_HEADLINES[kind],
        stop_reason=stop_reason,
        adopt=kind == DecisionKind.CONFIRMED_AND_MEETS_GOALS,
        keep_baseline=kind == DecisionKind.NO_RELIABLE_IMPROVEMENT,
        meets_goals=meets,
        goal_checks=checks,
        adopted_config=measured,
        requested_config=requested,
        baseline_config=baseline_cfg,
        config_diff=_config_diff(baseline_cfg, measured) if measured else {},
        exported_matches_measured=exported_ok,
        adopted_summary=adopted,
        baseline_summary=baseline_summary,
        why_not_others=_why_not_others(
            kind=kind,
            baseline=baseline_summary,
            best=best_summary if improved else baseline_summary,
            summaries=summaries,
        ),
        evidence=evidence,
        task_conditions=cond,
    )


def render_decision_markdown(decision: DecisionReport) -> list[str]:
    """Markdown used by both the file report and the Chainlit UI."""
    lines = [
        "## Decision",
        "",
        f"**{decision.kind.value}** — {decision.headline}",
        "",
    ]
    if decision.stop_reason:
        lines += [f"**Stop reason:** `{decision.stop_reason}`", ""]

    if decision.goal_checks:
        lines += [
            "### Goal checks",
            "",
            "| role | metric | op | limit | observed | ok |",
            "|---|---|---|---|---|---|",
        ]
        for c in decision.goal_checks:
            observed = "n/a" if c.observed is None else f"{c.observed:g}"
            lines.append(
                f"| {c.role} | `{c.metric}` | {c.op} | {c.limit:g} | {observed} | "
                f"{'yes' if c.ok else 'no'} |"
            )
        lines.append("")

    if decision.adopted_config:
        lines += ["### Adopted configuration (measured `actual_config`)", ""]
        for key, val in decision.adopted_config.items():
            lines.append(f"- `{key}`: `{val}`")
        lines.append("")
        if decision.exported_matches_measured:
            lines.append(
                "Exported knobs are the measured `actual_config` for the adopted run."
            )
            lines.append("")
        if decision.requested_config and decision.requested_config != decision.adopted_config:
            lines += [
                "Requested knobs that differ from measured (not exported as deployable):",
                "",
            ]
            for key, val in decision.requested_config.items():
                measured = decision.adopted_config.get(key, "<missing>")
                if measured != val:
                    lines.append(f"- `{key}`: requested `{val}` → measured `{measured}`")
            lines.append("")
    elif decision.kind != DecisionKind.INCONCLUSIVE and decision.adopted_summary:
        lines += [
            "No measured `actual_config` is available, so no configuration is exported.",
            "",
        ]

    if decision.config_diff:
        lines += ["### Diff vs baseline", ""]
        for key, pair in decision.config_diff.items():
            lines.append(f"- `{key}`: `{pair.get('from')}` → `{pair.get('to')}`")
        lines.append("")

    if decision.why_not_others:
        lines += ["### Why not another result", ""]
        for item in decision.why_not_others:
            lines.append(f"- {item}")
        lines.append("")

    if decision.evidence:
        lines += ["### Evidence", ""]
        for ev in decision.evidence:
            ledger = ev.ledger_path or "n/a"
            lines.append(
                f"- {ev.label}: experiment `{ev.experiment_id}`  "
                f"run_id=`{ev.run_id}`  ledger=`{ledger}`  "
                f"status=`{ev.validity_status}`"
            )
        lines.append("")

    return lines
