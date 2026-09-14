#!/usr/bin/env python
"""Run commit-level eval and optional regression gate.

Modes (mutually exclusive intent):
  --mock              Preset strategy simulation (random/greedy). NOT production planner.
  --real-graph        Production build_graph + planner with fake LLM + stubbed benchmark.
  --real-llm          Same graph with live LLM; fails loudly without API credentials.
  --prefix …         Score a persisted session from experiment memory.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

from inferops.eval.harness import (
    MODE_REAL_GRAPH_LLM,
    MODE_REAL_GRAPH_OFFLINE,
    run_mock_eval,
    run_real_graph_eval,
    run_session_eval,
    write_eval_outputs,
)
from inferops.eval.regression import load_eval_json, regression_gate
from inferops.eval.runner import ALL_WORKLOAD_NAMES

console = Console()

DEFAULT_MOCK_OUT = "eval_reports"
DEFAULT_REAL_GRAPH_OUT = "eval_reports/real_graph"
DEFAULT_REAL_LLM_OUT = "eval_reports/real_llm"


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
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Report output directory (defaults: eval_reports for --mock, "
            f"{DEFAULT_REAL_GRAPH_OUT} for --real-graph, "
            f"{DEFAULT_REAL_LLM_OUT} for --real-llm)"
        ),
    )
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
        help=(
            "Fair protocol simulation (default/random/online_local_search). "
            "Does NOT invoke production build_graph / planner_node."
        ),
    )
    parser.add_argument(
        "--real-graph",
        action="store_true",
        help=(
            "Offline real-graph eval: production planner→executor→reflector with "
            "fake/scripted LLM and stubbed run_benchmark (CI-safe). "
            "Writes forged rows only to a temp/dedicated eval DB "
            "(never inferops_memory.db); override with --eval-db."
        ),
    )
    parser.add_argument(
        "--real-llm",
        action="store_true",
        help=(
            "Live-LLM real-graph eval (separately labeled). Requires API credentials; "
            "missing creds fail loudly. Caps to ≤1 workload / small budget. "
            "Also uses a temp/dedicated eval DB by default (--eval-db to override)."
        ),
    )
    parser.add_argument(
        "--eval-db",
        default=None,
        help=(
            "SQLite path for --real-graph / --real-llm forged experiment rows. "
            "Default: a fresh temporary eval_memory.db under a temp dir "
            "(does not pollute inferops_memory.db)."
        ),
    )
    parser.add_argument(
        "--llm-backend",
        default="openrouter",
        choices=["openrouter", "deepseek", "claude"],
        help="LLM backend for --real-llm (default: openrouter)",
    )
    parser.add_argument(
        "--bottleneck",
        default="compute-bound",
        help="Initial bottleneck seed for --real-graph scripted planner",
    )
    parser.add_argument(
        "--baseline-report",
        default=None,
        help="Previous eval JSON for regression gate",
    )
    parser.add_argument(
        "--gate-strategy",
        default="online_local_search",
        help=(
            "Strategy to gate (mock default: online_local_search — honest local "
            "search without clairvoyant GT peeking)"
        ),
    )
    parser.add_argument("--max-outcome-regression-pct", type=float, default=5.0)
    parser.add_argument("--min-composite-delta", type=float, default=-0.05)
    args = parser.parse_args()

    mode_flags = sum(bool(x) for x in (args.mock, args.real_graph, args.real_llm, args.prefix))
    if mode_flags != 1:
        console.print(
            "[red]Provide exactly one of: --mock (preset simulation), "
            "--real-graph (offline production planner), --real-llm (live LLM), "
            "or --prefix (session eval).[/]"
        )
        sys.exit(2)

    sha = args.commit_sha or _current_sha()
    workloads = ALL_WORKLOAD_NAMES if args.workloads == ["all"] else args.workloads

    if args.mock:
        output_dir = args.output_dir or DEFAULT_MOCK_OUT
        report = run_mock_eval(
            commit_sha=sha,
            ground_truth_dir=args.ground_truth,
            workloads=workloads,
            budget=args.budget,
            seed=args.seed,
        )
    elif args.real_graph:
        output_dir = args.output_dir or DEFAULT_REAL_GRAPH_OUT
        report = run_real_graph_eval(
            commit_sha=sha,
            ground_truth_dir=args.ground_truth,
            workloads=workloads,
            budget=min(args.budget, 4),
            mode=MODE_REAL_GRAPH_OFFLINE,
            bottleneck=args.bottleneck,
            db_path=Path(args.eval_db) if args.eval_db else None,
        )
        console.print(f"[dim]Eval DB:[/] {report.get('eval_db_path', '')}")
    elif args.real_llm:
        output_dir = args.output_dir or DEFAULT_REAL_LLM_OUT
        try:
            report = run_real_graph_eval(
                commit_sha=sha,
                ground_truth_dir=args.ground_truth,
                workloads=workloads[:1],
                budget=min(args.budget, 3),
                mode=MODE_REAL_GRAPH_LLM,
                llm_backend=args.llm_backend,
                bottleneck=args.bottleneck,
                db_path=Path(args.eval_db) if args.eval_db else None,
            )
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/]")
            sys.exit(1)
        console.print(f"[dim]Eval DB:[/] {report.get('eval_db_path', '')}")
    else:
        output_dir = args.output_dir or DEFAULT_MOCK_OUT
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

    md_path, json_path = write_eval_outputs(report, output_dir)
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
