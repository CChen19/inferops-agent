"""Offline fair strategy comparison over hidden ground-truth fixtures."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inferops.eval.harness import fixture_from_ground_truth
from inferops.eval.metrics import WORKLOAD_PRIMARY_METRIC
from inferops.eval.planner_strategy import run_fair_comparison
from inferops.eval.protocol import BudgetPolicy, primary_value
from inferops.eval.runner import ALL_WORKLOAD_NAMES, load_ground_truth

DISCLAIMER = (
    "offline replay of published GT rows; ScriptedBottleneckLLM; "
    "no live GPU; no live planner LLM; planner \"win\" here means the "
    "bottleneck heuristic beat other search policies under the same budget — "
    "not that a paid model found a better config."
)


def run_offline_fair_compare(
    commit_sha: str,
    ground_truth_dir: str | Path,
    workloads: list[str] | None = None,
    budget: int = 4,
    seed: int = 7,
) -> dict[str, Any]:
    names = workloads or ALL_WORKLOAD_NAMES
    strategy_names = (
        "default",
        "random",
        "online_local_search",
        "planner_rag",
        "planner_no_rag",
    )
    strategies: dict[str, list[dict[str, Any]]] = {name: [] for name in strategy_names}

    for wl_name in names:
        gt = load_ground_truth(wl_name, ground_truth_dir)
        fixture = fixture_from_ground_truth(gt)
        runs = run_fair_comparison(
            fixture,
            BudgetPolicy(total_slots=budget),
            workload_name=gt["workload_name"],
            seed=seed,
            gt_optimum=gt,
        )
        for name, run in runs.items():
            strategies[name].append(_strategy_run_to_row(gt, run, budget=budget))

    aggregates = {name: _aggregate_protocol_rows(rows) for name, rows in strategies.items()}

    return {
        "commit_sha": commit_sha,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "fair_compare",
        "mode_label": "offline_hidden_result_replay",
        "llm_boundary": "fake_scripted",
        "tool_boundary": "hidden_fixture_observe",
        "budget": budget,
        "disclaimer": DISCLAIMER,
        "strategies": strategies,
        "aggregates": aggregates,
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# InferOps Fair Compare Report: `{report['commit_sha']}`",
        "",
        f"- Mode: `{report.get('mode', '')}`",
        f"- Mode label: `{report.get('mode_label', '')}`",
        f"- LLM boundary: `{report.get('llm_boundary', '')}`",
        f"- Tool boundary: `{report.get('tool_boundary', '')}`",
        f"- Generated: `{report.get('generated_at', '')}`",
        f"- Budget: `{report.get('budget', '')}` experiments per strategy",
        "",
        f"> **Disclaimer:** {report.get('disclaimer', DISCLAIMER)}",
        "",
        "## Summary",
        "",
        "| Strategy | Success | Mean gap % | Mean 1st valid | Wasted | Paid |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, agg in sorted(report.get("aggregates", {}).items()):
        success_pct = agg.get("mean_success_in_budget", 0.0) * 100
        first_valid = agg.get("mean_first_valid_n")
        first_valid_s = f"{first_valid:.1f}" if first_valid is not None else "—"
        lines.append(
            f"| {name} | {success_pct:.0f}% | {agg.get('mean_gap_pct', 0):+.2f} | "
            f"{first_valid_s} | {agg.get('mean_wasted_trials', 0):.1f} | "
            f"{agg.get('mean_n_paid', 0):.1f} |"
        )

    lines += ["", "## Workloads", ""]
    for name, rows in sorted(report.get("strategies", {}).items()):
        lines += [
            f"### {name}",
            "",
            "| Workload | Success | 1st valid | Gain | Wasted | Paid | Gap % | Agent | GT |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
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
                f"{row.get('wasted_trials', 0)} | {row.get('n_paid', 0)} | {gap_s} | "
                f"{row.get('agent_value', 0.0):.3f} | {row.get('ground_truth_value', 0.0):.3f} |"
            )
        lines.append("")

    return "\n".join(lines)


def write_fair_compare_outputs(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = report["commit_sha"]
    json_path = out_dir / f"{sha}.json"
    md_path = out_dir / f"{sha}.md"
    import json

    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(render_markdown_report(report))
    return md_path, json_path


def _strategy_run_to_row(
    ground_truth: dict[str, Any],
    run: Any,
    *,
    budget: int,
) -> dict[str, Any]:
    workload_name = run.workload_name
    metric, direction = WORKLOAD_PRIMARY_METRIC[workload_name]
    score = run.score
    gt_val = float(ground_truth["best_value"])

    agent_val = 0.0
    best = run.ledger.best_valid(metric, direction)
    if best is not None:
        _cfg, best_obs = best
        agent_val = primary_value(best_obs, metric)

    gap_pct = score.get("gap_pct")
    if gap_pct is None:
        if direction == "max":
            gap_pct = (gt_val - agent_val) / gt_val * 100 if gt_val else 0.0
        else:
            gap_pct = (agent_val - gt_val) / gt_val * 100 if gt_val else 0.0
        gap_pct = round(gap_pct, 2)

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
        "n_experiments": score["n_paid"],
        "budget": budget,
        "picked_configs": [dict(record.config) for record in run.ledger.records],
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
    }
