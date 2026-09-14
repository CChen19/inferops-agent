"""Tool: write_final_report — generate the end-of-session Markdown report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from inferops.observability import span
from inferops.decision import DecisionKind, build_decision, render_decision_markdown
from inferops.task import (
    format_task_conditions_markdown,
    task_from_mapping,
)


def _fmt_metric(v: Any, spec: str) -> str:
    """Format a summary metric. Missing stays n/a — never 0.0."""
    if v is None:
        return "n/a"
    return format(float(v), spec)


def _fmt_vs_baseline(v: Any) -> str:
    """Render vs_baseline_pct at stored precision so the printed value is citable."""
    if v is None:
        return _fmt_metric(v, "+")
    return f"{_fmt_metric(v, '+')}%"


def _format_live_result_message(s: dict[str, Any]) -> str:
    """Per-trial result line. vs_baseline_pct prints at stored precision; None stays n/a."""
    improvement = s.get("vs_baseline_pct")
    if improvement is None:
        icon = "➡️"
    else:
        icon = "✅" if improvement > 0 else ("➡️" if improvement == 0 else "⬇️")
    return (
        f"{icon} **Result:** `{s['experiment_id']}`\n"
        f"  • throughput = **{s['throughput_rps']:.3f} RPS** "
        f"({_fmt_vs_baseline(improvement)} vs baseline)\n"
        f"  • TTFT p99 = {s['ttft_p99_ms']:.1f} ms\n"
        f"  • bottleneck = `{s['bottleneck']}`"
    )


def _format_all_experiments_table(summaries: list[dict[str, Any]]) -> list[str]:
    """All-experiments markdown table. Missing vs stays n/a — never +0.0%."""
    lines = [
        "| Experiment | param | value | rps | ttft_p99 | bottleneck | vs_baseline |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| `{s['experiment_id']}` | {s.get('param_changed') or 'baseline'} "
            f"| {s.get('value_changed', '')} "
            f"| {_fmt_metric(s.get('throughput_rps'), '.3f')} "
            f"| {_fmt_metric(s.get('ttft_p99_ms'), '.1f')}ms "
            f"| {s.get('bottleneck', 'unknown')} "
            f"| {_fmt_vs_baseline(s.get('vs_baseline_pct'))} |"
        )
    return lines


class FinalReportInput(BaseModel):
    workload_name: str = Field(description="Workload that was optimized")
    session_prefix: str = Field(description="Experiment ID prefix for this session")
    experiment_summaries: list[dict[str, Any]] = Field(
        description="All ExperimentSummary dicts from AgentState"
    )
    baseline_summary: dict[str, Any] | None = Field(
        default=None,
        description="Baseline ExperimentSummary",
    )
    best_summary: dict[str, Any] | None = Field(
        default=None,
        description="Best ExperimentSummary found",
    )
    citations: list[str] = Field(
        default_factory=list,
        description="List of '[source: X] quote' strings collected during the session",
    )
    output_path: str = Field(
        default="reports/agent_final_report.md",
        description="Where to write the Markdown file",
    )
    optimization_task: dict[str, Any] | None = Field(
        default=None,
        description="Confirmed OptimizationTask dump — same conditions as the confirm page",
    )
    stop_reason: str = Field(default="", description="Reflect stop reason")


class FinalReportOutput(BaseModel):
    output_path: str
    sections_written: int
    improvement_pct: float | None = None
    decision_kind: str | None = None


def write_final_report(inp: FinalReportInput) -> FinalReportOutput:
    """
    Generate a Markdown report summarising the optimization session.

    Includes: executive summary, experiment table, best config, citations,
    and a recommendation section. Writes to disk and returns the path.

    Deploy recommendations are withheld unless best_summary is contract-valid
    with critical config evidence.
    """
    with span("tool.write_final_report", {"workload": inp.workload_name}):
        lines: list[str] = []
        sections = 0

        # Header
        lines += [
            f"# InferOps Optimization Report — `{inp.workload_name}`",
            "",
            f"**Session prefix:** `{inp.session_prefix}`",
            f"**Experiments run:** {len(inp.experiment_summaries)}",
            "",
        ]

        task = task_from_mapping(inp.optimization_task)
        if task is not None:
            lines += format_task_conditions_markdown(task)
            sections += 1

        decision = build_decision(
            baseline_summary=inp.baseline_summary,
            best_summary=inp.best_summary,
            experiment_summaries=inp.experiment_summaries,
            optimization_task=inp.optimization_task,
            stop_reason=inp.stop_reason,
        )
        improvement: float | None = None
        if inp.best_summary and inp.best_summary.get("vs_baseline_pct") is not None:
            improvement = float(inp.best_summary["vs_baseline_pct"])

        lines += ["## Executive Summary", ""]
        if inp.baseline_summary and inp.best_summary:
            raw_imp = inp.best_summary.get("vs_baseline_pct")
            status = inp.best_summary.get("validity_status", "insufficient_evidence")
            lines += [
                f"Best observed change vs baseline: "
                f"**{_fmt_vs_baseline(raw_imp)}** "
                f"(validity=`{status}`). Decision below is authoritative.",
                "",
                f"- **Baseline:** `{inp.baseline_summary['experiment_id']}`  "
                f"rps={_fmt_metric(inp.baseline_summary.get('throughput_rps'), '.3f')}  "
                f"status=`{inp.baseline_summary.get('validity_status', 'unknown')}`",
                f"- **Best found:** `{inp.best_summary['experiment_id']}`  "
                f"rps={_fmt_metric(inp.best_summary.get('throughput_rps'), '.3f')}  "
                f"run_id=`{inp.best_summary.get('run_id', '')}`  "
                f"mlflow=`{inp.best_summary.get('mlflow_run_id') or ''}`",
                f"- **Bottleneck at best:** `{inp.best_summary.get('bottleneck', 'unknown')}`",
                "",
            ]
        elif inp.baseline_summary and not inp.best_summary:
            lines += [
                "No promotable (valid + evidenced) candidate was selected as best. "
                "High scores without actual-config evidence are not deployable.",
                "",
                f"- **Baseline:** `{inp.baseline_summary['experiment_id']}`  "
                f"rps={_fmt_metric(inp.baseline_summary.get('throughput_rps'), '.3f')}  "
                f"status=`{inp.baseline_summary.get('validity_status', 'unknown')}`",
                "",
            ]
        lines += render_decision_markdown(decision)
        sections += 1

        # Experiment table
        if inp.experiment_summaries:
            lines += [
                "## Experiment Log",
                "",
                "| # | experiment_id | run_id | mlflow | status | param | value | rps | ttft_p99 | vs baseline | failure_reason |",
                "|---|---|---|---|---|---|---|---|---|---|---|",
            ]
            for i, s in enumerate(inp.experiment_summaries, 1):
                reason = (s.get("failure_reason") or "").replace("|", "/")
                if len(reason) > 60:
                    reason = reason[:57] + "..."
                lines.append(
                    f"| {i} | `{s['experiment_id']}` "
                    f"| `{s.get('run_id', '')}` "
                    f"| `{s.get('mlflow_run_id') or ''}` "
                    f"| `{s.get('validity_status', 'insufficient_evidence')}` "
                    f"| {s.get('param_changed') or '—'} "
                    f"| {s.get('value_changed', '')} "
                    f"| {_fmt_metric(s.get('throughput_rps'), '.3f')} "
                    f"| {_fmt_metric(s.get('ttft_p99_ms'), '')} "
                    f"| {_fmt_vs_baseline(s.get('vs_baseline_pct'))} "
                    f"| {reason or '—'} |"
                )
            lines.append("")
            # Explicit failed-attempt section for report consumers
            failed = [
                s for s in inp.experiment_summaries
                if s.get("validity_status") == "failed"
            ]
            if failed:
                lines += ["### Failed attempts", ""]
                for s in failed:
                    lines.append(
                        f"- `{s['experiment_id']}`  run_id=`{s.get('run_id', '')}`  "
                        f"mlflow=`{s.get('mlflow_run_id') or ''}`  "
                        f"status=`failed`  "
                        f"reason: {s.get('failure_reason') or '(none)'}"
                    )
                lines.append("")
            sections += 1

        # Knowledge citations
        if inp.citations:
            lines += [
                "## Knowledge Citations",
                "",
            ]
            for c in inp.citations:
                lines.append(f"- {c}")
            lines.append("")
            sections += 1

        # Recommendation — same four outcomes as the Decision section, no generic FP8 tips.
        lines += ["## Recommendation", ""]
        if decision.kind == DecisionKind.CONFIRMED_AND_MEETS_GOALS and decision.adopted_summary:
            best = decision.adopted_summary
            lines.append(
                f"Adopt the measured configuration from **`{best['experiment_id']}`** "
                f"(run_id=`{best.get('run_id', '')}`, "
                f"ledger=`{best.get('ledger_path') or 'n/a'}`)."
            )
        elif decision.kind == DecisionKind.IMPROVED_BUT_UNMET_GOALS and decision.adopted_summary:
            best = decision.adopted_summary
            lines.append(
                f"**No deploy recommendation.** `{best['experiment_id']}` improved "
                "over baseline but the task goals are still not met."
            )
        elif decision.kind == DecisionKind.NO_RELIABLE_IMPROVEMENT:
            base = decision.baseline_summary or {}
            lines.append(
                f"**No deploy recommendation.** Keep baseline "
                f"`{base.get('experiment_id', 'n/a')}`. "
                "No candidate was both confirmed and promotable."
            )
        else:
            lines.append(
                "**No deploy recommendation.** Evidence is insufficient or execution "
                "failed. Config file alone, HTTP 200 alone, or a performance delta "
                "alone are never sufficient."
            )
        lines.append("")
        sections += 1

        # Write file
        out_path = Path(inp.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines), encoding="utf-8")

    return FinalReportOutput(
        output_path=str(out_path),
        sections_written=sections,
        improvement_pct=improvement,
        decision_kind=decision.kind.value,
    )
