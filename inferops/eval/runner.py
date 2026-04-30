"""eval_runner — compare any agent session's results against ground truth.

Workflow:
  1. Run the grid sweep to build ground truth:
       python scripts/run_grid_sweep.py --output-dir data/ground_truth

  2. Run an agent session (all its experiment_ids should share a common prefix,
     e.g. "agent_v1_"):
       python -m inferops.agent.optimizer  # Phase 4 — not yet implemented

  3. Evaluate:
       python -m inferops.eval.runner --prefix agent_v1_ --workloads all
       python -m inferops.eval.runner --prefix agent_v1_ --judge   # + LLM judge
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from inferops.eval.metrics import (
    WORKLOAD_PRIMARY_METRIC,
    WorkloadScore,
    aggregate_scores,
    composite_score,
    compute_efficiency,
    compute_outcome,
)
from inferops.memory.db import query_results

console = Console()

ALL_WORKLOAD_NAMES = list(WORKLOAD_PRIMARY_METRIC.keys())


# ---------------------------------------------------------------------------
# Ground truth I/O
# ---------------------------------------------------------------------------

def load_ground_truth(workload_name: str, ground_truth_dir: str | Path) -> dict[str, Any]:
    path = Path(ground_truth_dir) / f"{workload_name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Ground truth not found: {path}\n"
            f"  Run: python scripts/run_grid_sweep.py --workloads {workload_name}"
        )
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def _best_agent_result(
    prefix: str,
    workload_name: str,
    metric: str,
    direction: str,
) -> dict[str, Any] | None:
    """Return the agent's best result for a workload, or None if no runs found."""
    rows = query_results(workload_name=workload_name, sort_by=metric, top_k=500)
    agent_rows = [r for r in rows if r["experiment_id"].startswith(prefix)]
    if not agent_rows:
        return None
    if direction == "max":
        return max(agent_rows, key=lambda r: r.get(metric, 0.0))
    return min(agent_rows, key=lambda r: r.get(metric, float("inf")))


def evaluate(
    prefix: str,
    ground_truth_dir: str | Path,
    workload_names: list[str],
    wall_clock_s: float | None = None,
    llm=None,
    trajectory: list[dict[str, Any]] | None = None,
) -> list[WorkloadScore]:
    """
    Evaluate an agent session against ground truth.

    prefix:          Experiment ID prefix shared by all of the agent's runs.
    ground_truth_dir: Directory containing {workload_name}.json files.
    workload_names:  Which workloads to evaluate.
    wall_clock_s:    Total elapsed time for the session (optional; for efficiency).
    llm:             LangChain ChatModel for LLM-as-judge (optional).
    trajectory:      List of agent step dicts for LLM judging (optional).

    Returns a WorkloadScore per workload.
    """
    scores: list[WorkloadScore] = []

    for wl_name in workload_names:
        try:
            gt = load_ground_truth(wl_name, ground_truth_dir)
        except FileNotFoundError as exc:
            console.print(f"[yellow]Warning:[/] {exc}")
            continue

        metric, direction = WORKLOAD_PRIMARY_METRIC[wl_name]
        best = _best_agent_result(prefix, wl_name, metric, direction)
        if best is None:
            console.print(
                f"[yellow]No results for workload '{wl_name}' with prefix '{prefix}'.[/] "
                f"Run the agent first."
            )
            continue

        # Count experiments this session ran for this workload
        all_for_wl = query_results(workload_name=wl_name, sort_by=metric, top_k=1000)
        n_exp = sum(1 for r in all_for_wl if r["experiment_id"].startswith(prefix))

        outcome = compute_outcome(gt, best)
        efficiency = compute_efficiency(
            n_experiments=n_exp,
            wall_clock_s=wall_clock_s or 0.0,
        )

        traj_score: float | None = None
        if llm is not None and trajectory is not None:
            from inferops.eval.judge import judge_trajectory
            wl_steps = [s for s in trajectory if s.get("workload") == wl_name]
            traj_score = judge_trajectory(wl_steps, llm=llm).overall

        comp = composite_score(outcome, efficiency, trajectory_score=traj_score)
        scores.append(WorkloadScore(
            workload_name=wl_name,
            outcome=outcome,
            efficiency=efficiency,
            trajectory_score=traj_score,
            composite=comp,
        ))

    return scores


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_summary_table(scores: list[WorkloadScore]) -> None:
    t = Table(title="Eval Results", show_lines=True)
    t.add_column("Workload",       style="cyan", no_wrap=True)
    t.add_column("Metric")
    t.add_column("GT",             justify="right")
    t.add_column("Agent",          justify="right")
    t.add_column("Gap %",          justify="right")
    t.add_column("# Runs",         justify="right")
    t.add_column("Wall min",       justify="right")
    t.add_column("Traj",           justify="right")
    t.add_column("Composite",      justify="right", style="bold")

    for s in scores:
        gap_color = (
            "green" if s.outcome.gap_pct <= 5
            else "yellow" if s.outcome.gap_pct <= 15
            else "red"
        )
        t.add_row(
            s.workload_name,
            s.outcome.primary_metric,
            f"{s.outcome.ground_truth_value:.3f}",
            f"{s.outcome.agent_value:.3f}",
            f"[{gap_color}]{s.outcome.gap_pct:+.1f}%[/]",
            str(s.efficiency.n_experiments),
            f"{s.efficiency.wall_clock_s/60:.1f}" if s.efficiency.wall_clock_s else "—",
            f"{s.trajectory_score:.2f}" if s.trajectory_score is not None else "—",
            f"{s.composite:.3f}",
        )

    console.print(t)

    agg = aggregate_scores(scores)
    if agg:
        console.print(
            f"\n[bold]Aggregate[/]  "
            f"gap={agg['mean_gap_pct']:+.1f}%  "
            f"runs={agg['mean_n_experiments']:.1f}  "
            f"wall={agg['mean_wall_clock_min']:.1f}min  "
            f"composite={agg['mean_composite']:.3f}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an agent session against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--prefix", required=True,
        help="Experiment ID prefix for the agent session (e.g. 'agent_v1_')",
    )
    parser.add_argument(
        "--ground-truth", default="data/ground_truth",
        help="Directory containing ground truth JSON files (default: data/ground_truth)",
    )
    parser.add_argument(
        "--workloads", nargs="+", default=["all"],
        help="Workload names or 'all' (default: all 5)",
    )
    parser.add_argument(
        "--judge", action="store_true",
        help="Run LLM-as-judge on trajectory (requires ANTHROPIC_API_KEY)",
    )
    args = parser.parse_args()

    names = ALL_WORKLOAD_NAMES if args.workloads == ["all"] else args.workloads

    llm = None
    if args.judge:
        try:
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
            console.print("[dim]LLM judge: claude-sonnet-4-6[/]")
        except ImportError:
            console.print("[yellow]langchain-anthropic not installed — skipping LLM judge.[/]")

    scores = evaluate(args.prefix, args.ground_truth, names, llm=llm)

    if not scores:
        console.print("[red]No results found. Run the agent first, then re-evaluate.[/]")
        sys.exit(1)

    print_summary_table(scores)


if __name__ == "__main__":
    main()
