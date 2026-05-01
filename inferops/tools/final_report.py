"""Tool: write_final_report — generate the end-of-session Markdown report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from inferops.observability import span


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
        if inp.baseline_summary and inp.best_summary:
            improvement = inp.best_summary.get("vs_baseline_pct", 0.0)
            icon = "🟢" if improvement > 5 else "🟡" if improvement > 0 else "🔴"
            lines += [
                "## Executive Summary",
                "",
                f"{icon} Best configuration achieved **{improvement:+.1f}%** "
                f"vs baseline on primary metric.",
                "",
                f"- **Baseline:** `{inp.baseline_summary['experiment_id']}`  "
                f"rps={inp.baseline_summary['throughput_rps']:.3f}",
                f"- **Best found:** `{inp.best_summary['experiment_id']}`  "
                f"rps={inp.best_summary['throughput_rps']:.3f}",
                f"- **Bottleneck at best:** `{inp.best_summary.get('bottleneck', 'unknown')}`",
                "",
            ]
            sections += 1

        # Experiment table
        if inp.experiment_summaries:
            lines += [
                "## Experiment Log",
                "",
                "| # | experiment_id | param | value | rps | ttft_p99 (ms) | bottleneck | vs baseline |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for i, s in enumerate(inp.experiment_summaries, 1):
                lines.append(
                    f"| {i} | `{s['experiment_id']}` "
                    f"| {s.get('param_changed') or '—'} "
                    f"| {s.get('value_changed', '')} "
                    f"| {s['throughput_rps']:.3f} "
                    f"| {s['ttft_p99_ms']:.1f} "
                    f"| {s.get('bottleneck', 'unknown')} "
                    f"| {s['vs_baseline_pct']:+.1f}% |"
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
        if inp.best_summary:
            best = inp.best_summary
            rec_lines = [
                "## Recommendation",
                "",
                f"Deploy experiment **`{best['experiment_id']}`** "
                f"(bottleneck: `{best.get('bottleneck', 'unknown')}`).",
                "",
                "Suggested next steps:",
            ]
            bottleneck = best.get("bottleneck", "unknown")
            if bottleneck == "compute-bound":
                rec_lines.append("- Consider increasing `max_num_batched_tokens` further or enabling FP8 quantisation.")
            elif bottleneck == "memory-bound":
                rec_lines.append("- Reduce `max_num_seqs` or `max_model_len` to free KV cache headroom.")
            elif bottleneck in ("scheduling-bound", "kv-bound"):
                rec_lines.append("- Enable `enable_prefix_caching` or `enable_chunked_prefill` if not already tried.")
            else:
                rec_lines.append("- Run a wider search or try a different workload scenario.")
            rec_lines.append("")
            lines += rec_lines
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
