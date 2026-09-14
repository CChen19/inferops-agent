"""Commit-level eval harness and Markdown dashboard generation.

Modes:
  - mock              Fair-protocol simulation (default/random/online local search).
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

from inferops.eval.baselines import BaselineRun
from inferops.eval.judge import judge_trajectory
from inferops.eval.metrics import (
    WORKLOAD_PRIMARY_METRIC,
    WorkloadScore,
    OutcomeMetrics,
    aggregate_scores,
    composite_score,
    compute_efficiency,
    compute_outcome,
)
from inferops.eval.protocol import BudgetPolicy, HiddenResultFixture, is_valid_observation, primary_value
from inferops.eval.strategies import (
    StrategyRun,
    run_default_strategy,
    run_online_local_search,
    run_random_strategy,
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
    """Fair-protocol strategy simulation over ground-truth rows (NOT production planner).

    Runs ``default``, ``random``, and ``online_local_search`` via the shared
    ``HiddenResultFixture`` / ``BudgetPolicy`` / observe-after-pick contract.
    Does not call ``build_graph`` or ``planner_node``.
    """
    names = workloads or ALL_WORKLOAD_NAMES
    strategy_names = ("default", "random", "online_local_search")
    strategies: dict[str, list[dict[str, Any]]] = {name: [] for name in strategy_names}

    for wl_name in names:
        gt = load_ground_truth(wl_name, ground_truth_dir)
        fixture = _fixture_from_ground_truth(gt)
        policy = BudgetPolicy(total_slots=budget)
        wl = gt["workload_name"]
        runs = {
            "default": run_default_strategy(
                fixture, policy, workload_name=wl, gt_optimum=gt
            ),
            "random": run_random_strategy(
                fixture,
                BudgetPolicy(total_slots=budget),
                workload_name=wl,
                seed=seed,
                gt_optimum=gt,
            ),
            "online_local_search": run_online_local_search(
                fixture,
                BudgetPolicy(total_slots=budget),
                workload_name=wl,
                gt_optimum=gt,
            ),
        }
        for name, run in runs.items():
            strategies[name].append(_strategy_run_to_row(gt, run, budget=budget))

    aggregates = {name: _aggregate_protocol_rows(rows) for name, rows in strategies.items()}

    return {
        "commit_sha": commit_sha,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "mock",
        "mode_label": "fair_protocol_simulation",
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
            "\n> **Mock eval only (fair protocol simulation).** "
            "default / random / online_local_search over hidden ground-truth rows — "
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
    ]
    if mode == "mock":
        lines += [
            "| Strategy | Success | Mean gap % | Mean 1st valid | Wasted | Paid | Composite |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name, agg in sorted(report.get("aggregates", {}).items()):
            success_pct = agg.get("mean_success_in_budget", 0.0) * 100
            first_valid = agg.get("mean_first_valid_n")
            first_valid_s = f"{first_valid:.1f}" if first_valid is not None else "—"
            lines.append(
                f"| {name} | {success_pct:.0f}% | {agg.get('mean_gap_pct', 0):+.2f} | "
                f"{first_valid_s} | {agg.get('mean_wasted_trials', 0):.1f} | "
                f"{agg.get('mean_n_paid', 0):.1f} | {agg.get('mean_composite', 0):.4f} |"
            )
    else:
        lines += [
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
        if mode == "mock":
            lines += [
                f"### {name}",
                "",
                "| Workload | Success | 1st valid | Gain | Wasted | Paid | Gap % | Composite |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for row in rows:
                success = "yes" if row.get("success_in_budget") else "no"
                first_valid = row.get("first_valid_n")
                first_valid_s = str(first_valid) if first_valid is not None else "—"
                gain = row.get("confirmed_gain")
                gain_s = f"{gain:.3f}" if gain is not None else "—"
                gap = row.get("gap_pct")
                gap_s = f"{gap:+.2f}" if gap is not None else "—"
                lines.append(
                    f"| {row['workload_name']} | {success} | {first_valid_s} | {gain_s} | "
                    f"{row.get('wasted_trials', 0)} | {row.get('n_paid', 0)} | "
                    f"{gap_s} | {row.get('composite', 0):.4f} |"
                )
        else:
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


def _fixture_from_ground_truth(ground_truth: dict[str, Any]) -> HiddenResultFixture:
    """Build a fair-eval fixture; GT sweep rows lack SLO contract fields by default."""
    enriched = [
        {
            **row,
            "validity_status": row.get("validity_status", "valid"),
            "error_rate": row.get("error_rate", 0.01),
            "has_config_evidence": row.get("has_config_evidence", True),
            "bottleneck": row.get("bottleneck", "compute-bound"),
        }
        for row in ground_truth.get("experiments", [])
    ]
    return HiddenResultFixture.from_rows(enriched)


def _strategy_run_to_row(
    ground_truth: dict[str, Any],
    run: StrategyRun,
    *,
    budget: int,
) -> dict[str, Any]:
    """Map a protocol ``StrategyRun`` to a harness report row."""
    workload_name = run.workload_name
    metric, direction = WORKLOAD_PRIMARY_METRIC[workload_name]
    score = run.score
    gt_val = float(ground_truth["best_value"])

    agent_val = 0.0
    best = run.ledger.best_valid(metric, direction)
    if best is not None:
        _cfg, best_obs = best
        agent_val = primary_value(best_obs, metric)

    gap_pct = score["gap_pct"]
    if gap_pct is None:
        if direction == "max":
            gap_pct = (gt_val - agent_val) / gt_val * 100 if gt_val else 0.0
        else:
            gap_pct = (agent_val - gt_val) / gt_val * 100 if gt_val else 0.0
        gap_pct = round(gap_pct, 2)

    outcome = OutcomeMetrics(
        workload_name=workload_name,
        primary_metric=metric,
        ground_truth_value=gt_val,
        agent_value=agent_val,
        gap_pct=gap_pct,
    )
    efficiency = compute_efficiency(score["n_paid"], wall_clock_s=0.0)
    comp = composite_score(outcome, efficiency, budget_experiments=budget)

    return {
        "workload_name": workload_name,
        "primary_metric": metric,
        "ground_truth_value": gt_val,
        "agent_value": agent_val,
        "success_in_budget": score["success_in_budget"],
        "first_valid_n": score["first_valid_n"],
        "confirmed_gain": score["confirmed_gain"],
        "wasted_trials": score["wasted_trials"],
        "n_paid": score["n_paid"],
        "gap_pct": gap_pct,
        "composite": comp,
        "n_experiments": score["n_paid"],
    }


def _aggregate_protocol_rows(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    if not rows:
        return {}
    n = len(rows)
    first_valid_vals = [r["first_valid_n"] for r in rows if r.get("first_valid_n") is not None]
    return {
        "mean_success_in_budget": round(
            sum(1.0 if r.get("success_in_budget") else 0.0 for r in rows) / n, 4
        ),
        "mean_gap_pct": round(sum(r.get("gap_pct") or 0.0 for r in rows) / n, 2),
        "mean_first_valid_n": round(sum(first_valid_vals) / len(first_valid_vals), 1)
        if first_valid_vals
        else None,
        "mean_wasted_trials": round(sum(r.get("wasted_trials", 0) for r in rows) / n, 1),
        "mean_n_paid": round(sum(r.get("n_paid", 0) for r in rows) / n, 1),
        "mean_n_experiments": round(sum(r.get("n_paid", 0) for r in rows) / n, 1),
        "mean_wall_clock_min": 0.0,
        "mean_composite": round(sum(r.get("composite", 0.0) for r in rows) / n, 4),
    }


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
