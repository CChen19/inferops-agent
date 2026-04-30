"""Commit-level eval harness and Markdown dashboard generation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inferops.eval.baselines import BaselineRun, run_greedy_agent, run_random_agent
from inferops.eval.judge import judge_trajectory
from inferops.eval.metrics import (
    WorkloadScore,
    aggregate_scores,
    composite_score,
    compute_efficiency,
    compute_outcome,
)
from inferops.eval.runner import ALL_WORKLOAD_NAMES, load_ground_truth


def run_mock_eval(
    commit_sha: str,
    ground_truth_dir: str | Path,
    workloads: list[str] | None = None,
    budget: int = 6,
    seed: int = 42,
) -> dict[str, Any]:
    """Run deterministic eval using ground-truth rows and simulated baseline agents."""
    names = workloads or ALL_WORKLOAD_NAMES
    strategies: dict[str, list[dict[str, Any]]] = {
        "random_agent": [],
        "greedy_agent": [],
    }

    for wl_name in names:
        gt = load_ground_truth(wl_name, ground_truth_dir)
        random_run = run_random_agent(gt, budget=budget, seed=seed)
        greedy_run = run_greedy_agent(gt, budget=budget)
        for run in (random_run, greedy_run):
            strategies[run.agent_name].append(_score_baseline_run(gt, run))

    aggregates = {
        name: aggregate_scores([_row_to_workload_score(row) for row in rows])
        for name, rows in strategies.items()
    }

    return {
        "commit_sha": commit_sha,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "mock",
        "budget": budget,
        "strategies": strategies,
        "aggregates": aggregates,
    }


def write_eval_outputs(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = report["commit_sha"]
    json_path = out_dir / f"{sha}.json"
    md_path = out_dir / f"{sha}.md"
    import json

    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(render_markdown_report(report))
    return md_path, json_path


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# InferOps Eval Report: `{report['commit_sha']}`",
        "",
        f"- Mode: `{report.get('mode', 'unknown')}`",
        f"- Generated: `{report.get('generated_at', '')}`",
        f"- Budget: `{report.get('budget', '')}` experiments per strategy",
        "",
        "## Summary",
        "",
        "| Strategy | Mean gap % | Mean runs | Mean composite |",
        "|---|---:|---:|---:|",
    ]
    for name, agg in sorted(report.get("aggregates", {}).items()):
        lines.append(
            f"| {name} | {agg.get('mean_gap_pct', 0):+.2f} | "
            f"{agg.get('mean_n_experiments', 0):.1f} | {agg.get('mean_composite', 0):.4f} |"
        )

    lines += ["", "## Workloads", ""]
    for name, rows in sorted(report.get("strategies", {}).items()):
        lines += [
            f"### {name}",
            "",
            "| Workload | Metric | GT | Agent | Gap % | Runs | Traj | Composite |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| {row['workload_name']} | {row['primary_metric']} | "
                f"{row['ground_truth_value']:.3f} | {row['agent_value']:.3f} | "
                f"{row['gap_pct']:+.2f} | {row['n_experiments']} | "
                f"{row['trajectory_score']:.2f} | {row['composite']:.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


def _score_baseline_run(ground_truth: dict[str, Any], run: BaselineRun) -> dict[str, Any]:
    outcome = compute_outcome(ground_truth, run.best_result)
    efficiency = compute_efficiency(run.n_experiments, wall_clock_s=0.0)
    trajectory_score = judge_trajectory(run.trajectory).overall
    comp = composite_score(outcome, efficiency, trajectory_score=trajectory_score)
    return {
        "workload_name": outcome.workload_name,
        "primary_metric": outcome.primary_metric,
        "ground_truth_value": outcome.ground_truth_value,
        "agent_value": outcome.agent_value,
        "gap_pct": outcome.gap_pct,
        "n_experiments": efficiency.n_experiments,
        "trajectory_score": trajectory_score,
        "composite": comp,
        "best_experiment_id": run.best_result.get("experiment_id", ""),
    }


def _row_to_workload_score(row: dict[str, Any]) -> WorkloadScore:
    from inferops.eval.metrics import EfficiencyMetrics, OutcomeMetrics

    outcome = OutcomeMetrics(
        workload_name=row["workload_name"],
        primary_metric=row["primary_metric"],
        ground_truth_value=row["ground_truth_value"],
        agent_value=row["agent_value"],
        gap_pct=row["gap_pct"],
    )
    efficiency = EfficiencyMetrics(n_experiments=row["n_experiments"], wall_clock_s=0.0)
    return WorkloadScore(
        workload_name=row["workload_name"],
        outcome=outcome,
        efficiency=efficiency,
        trajectory_score=row["trajectory_score"],
        composite=row["composite"],
    )
