"""Commit-level eval harness and Markdown dashboard generation.

Modes:
  - mock              Preset strategy simulation (random/greedy over ground-truth).
                      Does NOT invoke production build_graph / planner_node.
  - session           Score a persisted agent session by experiment-id prefix.
  - real_graph_offline  Production LangGraph planner path with fake LLM + stubbed
                      benchmark tool edge (CI-safe).
  - real_graph_llm      Same graph path with a live LLM; fails loudly without creds.
"""

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
from inferops.eval.real_graph import (
    MODE_REAL_GRAPH_LLM,
    MODE_REAL_GRAPH_OFFLINE,
    run_real_graph_eval,
)
from inferops.eval.runner import ALL_WORKLOAD_NAMES, evaluate, load_ground_truth

__all__ = [
    "run_mock_eval",
    "run_session_eval",
    "run_real_graph_eval",
    "write_eval_outputs",
    "render_markdown_report",
    "MODE_REAL_GRAPH_OFFLINE",
    "MODE_REAL_GRAPH_LLM",
]


def run_mock_eval(
    commit_sha: str,
    ground_truth_dir: str | Path,
    workloads: list[str] | None = None,
    budget: int = 6,
    seed: int = 42,
) -> dict[str, Any]:
    """Preset strategy simulation over ground-truth rows (NOT production planner).

    Uses ``run_random_agent`` / ``run_greedy_agent`` only. Trajectory ``node``
    fields are baseline names — never planner / executor / reflector. Does not
    call ``build_graph`` or ``planner_node``.
    """
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
        "mode_label": "preset_strategy_simulation",
        "budget": budget,
        "strategies": strategies,
        "aggregates": aggregates,
    }


def run_session_eval(
    commit_sha: str,
    prefix: str,
    ground_truth_dir: str | Path,
    workloads: list[str] | None = None,
    wall_clock_s: float | None = None,
) -> dict[str, Any]:
    """Evaluate a real/manual agent session already persisted in experiment memory."""
    names = workloads or ALL_WORKLOAD_NAMES
    scores = evaluate(
        prefix=prefix,
        ground_truth_dir=ground_truth_dir,
        workload_names=names,
        wall_clock_s=wall_clock_s,
    )
    rows = [_score_to_row(score) for score in scores]
    return {
        "commit_sha": commit_sha,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "session",
        "prefix": prefix,
        "strategies": {"agent_session": rows},
        "aggregates": {"agent_session": aggregate_scores(scores)},
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
    mode = report.get("mode", "unknown")
    disclaimer = ""
    if mode == "mock":
        disclaimer = (
            "\n> **Mock eval only (preset strategy simulation).** "
            "random_agent / greedy_agent over ground-truth rows — "
            "**does not** invoke production `build_graph` / `planner_node`. "
            "Figures are **not** real measured performance claims.\n"
        )
    elif mode == MODE_REAL_GRAPH_OFFLINE:
        disclaimer = (
            "\n> **Real-graph offline eval.** Production LangGraph "
            "`planner → executor → reflector` with a **fake/scripted LLM** and "
            "**stubbed** `run_benchmark` tool edge. Not live OpenRouter / vLLM.\n"
        )
    elif mode == MODE_REAL_GRAPH_LLM:
        disclaimer = (
            "\n> **Real-graph live-LLM eval (separately labeled).** Production "
            "graph with a live ChatModel; tool/benchmark edge may still be stubbed. "
            "Missing API credentials must fail loudly (never silent pass).\n"
        )
    extra = ""
    if report.get("mode_label"):
        extra += f"\n- Mode label: `{report['mode_label']}`"
    if report.get("llm_boundary"):
        extra += f"\n- LLM boundary: `{report['llm_boundary']}`"
    if report.get("tool_boundary"):
        extra += f"\n- Tool boundary: `{report['tool_boundary']}`"
    if report.get("eval_db_path"):
        extra += f"\n- Eval DB: `{report['eval_db_path']}`"
    lines = [
        f"# InferOps Eval Report: `{report['commit_sha']}`",
        "",
        f"- Mode: `{mode}`",
        f"- Generated: `{report.get('generated_at', '')}`",
        f"- Budget: `{report.get('budget', '')}` experiments per strategy",
        extra,
        disclaimer,
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


def _score_to_row(score: WorkloadScore) -> dict[str, Any]:
    return {
        "workload_name": score.workload_name,
        "primary_metric": score.outcome.primary_metric,
        "ground_truth_value": score.outcome.ground_truth_value,
        "agent_value": score.outcome.agent_value,
        "gap_pct": score.outcome.gap_pct,
        "n_experiments": score.efficiency.n_experiments,
        "trajectory_score": score.trajectory_score or 0.0,
        "composite": score.composite,
        "best_experiment_id": "",
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
