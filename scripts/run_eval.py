#!/usr/bin/env python
"""Run commit-level eval and optional regression gate."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

from inferops.eval.harness import run_mock_eval, run_session_eval, write_eval_outputs
from inferops.eval.regression import load_eval_json, regression_gate
from inferops.eval.runner import ALL_WORKLOAD_NAMES

console = Console()


def _current_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 eval harness + regression gate")
    parser.add_argument("--commit-sha", default=None, help="Commit SHA for report naming")
    parser.add_argument(
        "--ground-truth",
        default="data/ground_truth",
        help="Ground-truth JSON directory",
    )
    parser.add_argument("--output-dir", default="eval_reports", help="Report output directory")
    parser.add_argument("--workloads", nargs="+", default=["all"], help="Workload names or all")
    parser.add_argument("--budget", type=int, default=6, help="Experiment budget per strategy")
    parser.add_argument("--seed", type=int, default=42, help="Random baseline seed")
    parser.add_argument(
        "--prefix",
        default=None,
        help="Experiment ID prefix for real/manual session eval",
    )
    parser.add_argument(
        "--wall-clock-s",
        type=float,
        default=None,
        help="Optional wall-clock seconds for real/manual session efficiency scoring",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run deterministic mock eval from ground-truth rows",
    )
    parser.add_argument(
        "--baseline-report",
        default=None,
        help="Previous eval JSON for regression gate",
    )
    parser.add_argument("--gate-strategy", default="greedy_agent", help="Strategy to gate")
    parser.add_argument("--max-outcome-regression-pct", type=float, default=5.0)
    parser.add_argument("--min-composite-delta", type=float, default=-0.05)
    args = parser.parse_args()

    if not args.mock and not args.prefix:
        console.print("[red]Provide --mock for CI-safe eval or --prefix for real session eval.[/]")
        sys.exit(2)

    sha = args.commit_sha or _current_sha()
    workloads = ALL_WORKLOAD_NAMES if args.workloads == ["all"] else args.workloads
    if args.mock:
        report = run_mock_eval(
            commit_sha=sha,
            ground_truth_dir=args.ground_truth,
            workloads=workloads,
            budget=args.budget,
            seed=args.seed,
        )
    else:
        report = run_session_eval(
            commit_sha=sha,
            prefix=args.prefix,
            ground_truth_dir=args.ground_truth,
            workloads=workloads,
            wall_clock_s=args.wall_clock_s,
        )
        if not report["strategies"]["agent_session"]:
            console.print("[red]No session results found for the requested prefix/workloads.[/]")
            sys.exit(1)
    md_path, json_path = write_eval_outputs(report, args.output_dir)
    console.print(f"[green]Eval report written:[/] {md_path}")
    console.print(f"[green]Eval JSON written:[/] {json_path}")

    if args.baseline_report:
        gate = regression_gate(
            current=report,
            baseline=load_eval_json(args.baseline_report),
            strategy=args.gate_strategy,
            max_outcome_regression_pct=args.max_outcome_regression_pct,
            min_composite_delta=args.min_composite_delta,
        )
        for warning in gate.warnings:
            console.print(f"[yellow]warning:[/] {warning}")
        if not gate.passed:
            console.print("[red]Regression gate failed:[/]")
            for failure in gate.failures:
                console.print(f"  - {failure}")
            sys.exit(1)
        console.print("[green]Regression gate passed.[/]")


if __name__ == "__main__":
    main()
