"""Tool: write_final_report — generate the end-of-session Markdown report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from inferops.observability import span
from inferops.agent.state import is_promotable_summary


def _fmt_metric(v: Any, spec: str) -> str:
    """Format a summary metric. Missing stays n/a — never 0.0."""
    if v is None:
        return "n/a"
    return format(float(v), spec)


def _is_deployable_best(best: dict[str, Any] | None) -> bool:
    """Deploy recommendations use the SAME full gate as executor/eval/DB."""
    return is_promotable_summary(best)


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


class FinalReportOutput(BaseModel):
    output_path: str
    sections_written: int
    improvement_pct: float


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

        # Executive summary
        improvement = 0.0
        deployable = _is_deployable_best(inp.best_summary)
        if inp.baseline_summary and inp.best_summary:
            raw_imp = inp.best_summary.get("vs_baseline_pct")
            improvement = float(raw_imp) if raw_imp is not None else 0.0
            icon = "🟢" if raw_imp is not None and improvement > 5 else "🟡" if raw_imp is not None and improvement > 0 else "🔴"
            status = inp.best_summary.get("validity_status", "insufficient_evidence")
            lines += [
                "## Executive Summary",
                "",
                f"{icon} Best configuration achieved "
                f"**{f'{raw_imp:+.1f}%' if raw_imp is not None else 'n/a'}** "
                f"vs baseline on primary metric "
                f"(validity=`{status}`).",
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
            sections += 1
        elif inp.baseline_summary and not inp.best_summary:
            lines += [
                "## Executive Summary",
                "",
                "No promotable (valid + evidenced) candidate was selected as best. "
                "High scores without actual-config evidence are not deployable.",
                "",
                f"- **Baseline:** `{inp.baseline_summary['experiment_id']}`  "
                f"rps={_fmt_metric(inp.baseline_summary.get('throughput_rps'), '.3f')}  "
                f"status=`{inp.baseline_summary.get('validity_status', 'unknown')}`",
                "",
            ]
            sections += 1

        # Experiment table
        if inp.experiment_summaries:
            lines += [
                "## Experiment Log",
                "",
                "| # | experiment_id | run_id | mlflow | status | param | value | rps | vs baseline | failure_reason |",
                "|---|---|---|---|---|---|---|---|---|---|",
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
                    f"| {_fmt_metric(s.get('vs_baseline_pct'), '+.1f')}"
                    f"{'%' if s.get('vs_baseline_pct') is not None else ''} "
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

        # Recommendation
        lines += ["## Recommendation", ""]
        if deployable and inp.best_summary:
            best = inp.best_summary
            lines.append(
                f"Deploy experiment **`{best['experiment_id']}`** "
                f"(run_id=`{best.get('run_id', '')}`, "
                f"bottleneck: `{best.get('bottleneck', 'unknown')}`)."
            )
            lines.append("")
            lines.append("Suggested next steps:")
            bottleneck = best.get("bottleneck", "unknown")
            if bottleneck == "compute-bound":
                lines.append("- Consider increasing `max_num_batched_tokens` further or enabling FP8 quantisation.")
            elif bottleneck == "memory-bound":
                lines.append("- Reduce `max_num_seqs` or `max_model_len` to free KV cache headroom.")
            elif bottleneck in ("scheduling-bound", "kv-bound"):
                lines.append("- Enable `enable_prefix_caching` or `enable_chunked_prefill` if not already tried.")
            else:
                lines.append("- Run a wider search or try a different workload scenario.")
        elif inp.best_summary:
            best = inp.best_summary
            lines.append(
                f"**No deploy recommendation.** Candidate `{best['experiment_id']}` "
                f"has status=`{best.get('validity_status', 'insufficient_evidence')}` "
                "and/or lacks critical actual-config evidence. "
                "Config file alone, HTTP 200 alone, or performance change alone "
                "are never sufficient."
            )
        else:
            lines.append(
                "**No deploy recommendation.** No valid, evidenced best candidate "
                "was selected in this session."
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
    )
