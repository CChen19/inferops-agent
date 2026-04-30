"""Compare Agent vs Default vs Random Search on 3 workloads.

For a fair comparison all three strategies share the same experiment budget K.

  Default     — 1 run (the baseline config; no search)
  Random      — K runs drawn uniformly at random from the grid, no LLM
  Agent       — K runs guided by the LangGraph planner/executor/reflector

Output: rich comparison table + optional JSON export.

Usage:
    python scripts/run_comparison.py \\
        --workloads chat_short long_context_qa high_concurrency_short_out \\
        --budget 6 --llm deepseek --session myrun1

    # With results already in DB (skip re-running experiments):
    python scripts/run_comparison.py --session myrun1 --results-only
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from inferops.agent.graph import make_llm, run_agent
from inferops.agent.state import WORKLOAD_PRIMARY_METRIC
from inferops.memory.db import init_db, query_results, save_result
from inferops.tools.run_benchmark import RunBenchmarkInput, run_benchmark

console = Console()

DEFAULT_WORKLOADS = ["chat_short", "long_context_qa", "high_concurrency_short_out"]

# Same grid axes as run_grid_sweep.py
_GRID_AXES = {
    "max_num_batched_tokens": [1024, 2048, 4096],
    "enable_chunked_prefill": [False, True],
    "enable_prefix_caching":  [False, True],
}


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def run_default(workload_name: str, prefix: str) -> dict:
    """Run only the default config (no search). Returns best result dict."""
    eid = f"{prefix}default_baseline"
    out = run_benchmark(RunBenchmarkInput(
        experiment_id=eid,
        config_patch={},
        workload_name=workload_name,
        persist=True,
    ))
    return out.model_dump()


def run_random_search(workload_name: str, budget: int, prefix: str, seed: int = 42) -> dict:
    """Run `budget` configs drawn uniformly at random from the grid. Returns best."""
    grid = list(itertools.product(*_GRID_AXES.values()))
    rng = random.Random(seed)
    rng.shuffle(grid)
    selected = grid[:budget]

    metric = WORKLOAD_PRIMARY_METRIC[workload_name]
    best: dict | None = None

    for i, combo in enumerate(selected):
        params = dict(zip(_GRID_AXES.keys(), combo))
        eid = f"{prefix}rand_{i}"
        try:
            out = run_benchmark(RunBenchmarkInput(
                experiment_id=eid,
                config_patch=params,
                workload_name=workload_name,
                persist=True,
            ))
            d = out.model_dump()
            if best is None or d.get(metric, 0) > best.get(metric, 0):
                best = d
        except Exception as exc:
            console.print(f"  [yellow]random search run {i} failed: {exc}[/yellow]")

    return best or {}


def run_agent_strategy(
    workload_name: str,
    budget: int,
    prefix: str,
    llm,
) -> dict:
    """Run the LangGraph agent. Returns best result dict."""
    final_state = run_agent(
        workload_name=workload_name,
        llm=llm,
        max_experiments=budget,
        session_prefix=prefix,
    )
    best = final_state.get("best_summary") or {}
    # Map ExperimentSummary keys → RunBenchmarkOutput keys
    metric = WORKLOAD_PRIMARY_METRIC[workload_name]
    return {
        metric:            best.get(metric, 0),
        "throughput_rps":  best.get("throughput_rps", 0),
        "tokens_per_second": best.get("tokens_per_second", 0),
        "ttft_p99_ms":     best.get("ttft_p99_ms", 0),
        "e2e_p50_ms":      best.get("e2e_p50_ms", 0),
        "experiment_id":   best.get("experiment_id", ""),
        "n_experiments":   len(final_state.get("tried_experiment_ids", [])),
    }


# ---------------------------------------------------------------------------
# Results-only mode (load from DB)
# ---------------------------------------------------------------------------

def _load_best_from_db(workload_name: str, prefix: str, metric: str) -> dict:
    rows = query_results(workload_name=workload_name, sort_by=metric, top_k=200)
    matching = [r for r in rows if r["experiment_id"].startswith(prefix)]
    if not matching:
        return {}
    return max(matching, key=lambda r: r.get(metric, 0))


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(results: list[dict]) -> None:
    """
    results: list of dicts with keys:
      workload, metric, default_val, random_val, random_n, agent_val, agent_n,
      random_vs_default_pct, agent_vs_default_pct
    """
    t = Table(title="Strategy Comparison", show_lines=True)
    t.add_column("Workload",          style="cyan", no_wrap=True)
    t.add_column("Metric")
    t.add_column("Default\n(1 run)",  justify="right")
    t.add_column("Random",            justify="right")
    t.add_column("Random vs Default", justify="right")
    t.add_column("Agent",             justify="right")
    t.add_column("Agent vs Default",  justify="right", style="bold")

    for r in results:
        rand_color = "green" if r["random_vs_default_pct"] > 5 else "yellow" if r["random_vs_default_pct"] > 0 else "red"
        agent_color = "green" if r["agent_vs_default_pct"] > 5 else "yellow" if r["agent_vs_default_pct"] > 0 else "red"
        t.add_row(
            r["workload"],
            r["metric"],
            f"{r['default_val']:.3f}",
            f"{r['random_val']:.3f} ({r['random_n']} runs)",
            f"[{rand_color}]{r['random_vs_default_pct']:+.1f}%[/]",
            f"{r['agent_val']:.3f} ({r['agent_n']} runs)",
            f"[{agent_color}]{r['agent_vs_default_pct']:+.1f}%[/]",
        )
    console.print(t)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Agent vs Default vs Random Search comparison")
    parser.add_argument("--workloads", nargs="+", default=DEFAULT_WORKLOADS,
                        help="Workloads to compare (default: 3 workloads)")
    parser.add_argument("--budget", type=int, default=6,
                        help="Experiment budget per strategy (default: 6)")
    parser.add_argument("--llm", default="deepseek", choices=["deepseek", "claude"],
                        help="LLM backend for agent (default: deepseek)")
    parser.add_argument("--session", default=None,
                        help="Session name prefix (auto-generated if omitted)")
    parser.add_argument("--results-only", action="store_true",
                        help="Skip running experiments; load existing results from DB")
    parser.add_argument("--output", default=None,
                        help="Write comparison JSON to this file")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for random search (default: 42)")
    args = parser.parse_args()

    init_db()
    session = args.session or f"cmp_{int(time.time())}"
    llm = make_llm(args.llm) if not args.results_only else None

    all_results = []

    for wl in args.workloads:
        metric = WORKLOAD_PRIMARY_METRIC[wl]
        pfx_default = f"{session}_{wl}_default_"
        pfx_random  = f"{session}_{wl}_random_"
        pfx_agent   = f"{session}_{wl}_agent_"

        console.rule(f"[bold]{wl}[/]  metric={metric}")

        if args.results_only:
            default_r  = _load_best_from_db(wl, pfx_default, metric)
            random_r   = _load_best_from_db(wl, pfx_random,  metric)
            agent_r    = _load_best_from_db(wl, pfx_agent,   metric)
            random_n   = len([r for r in query_results(workload_name=wl, sort_by=metric, top_k=200)
                               if r["experiment_id"].startswith(pfx_random)])
            agent_n    = len([r for r in query_results(workload_name=wl, sort_by=metric, top_k=200)
                               if r["experiment_id"].startswith(pfx_agent)])
        else:
            console.print("  [1/3] Default …")
            default_r = run_default(wl, pfx_default)

            console.print(f"  [2/3] Random search ({args.budget} runs) …")
            random_r  = run_random_search(wl, args.budget, pfx_random, seed=args.seed)
            random_n  = args.budget

            console.print(f"  [3/3] Agent ({args.budget} runs, llm={args.llm}) …")
            agent_r   = run_agent_strategy(wl, args.budget, pfx_agent, llm)
            agent_n   = agent_r.get("n_experiments", args.budget)

        default_val = float(default_r.get(metric, 0))
        random_val  = float(random_r.get(metric, 0))
        agent_val   = float(agent_r.get(metric, 0))

        def pct(candidate, base):
            return (candidate - base) / base * 100 if base else 0.0

        row = {
            "workload":              wl,
            "metric":                metric,
            "default_val":           default_val,
            "random_val":            random_val,
            "random_n":              random_n,
            "agent_val":             agent_val,
            "agent_n":               agent_n,
            "random_vs_default_pct": round(pct(random_val, default_val), 2),
            "agent_vs_default_pct":  round(pct(agent_val,  default_val), 2),
        }
        all_results.append(row)

    print_comparison_table(all_results)

    if args.output:
        Path(args.output).write_text(json.dumps(all_results, indent=2))
        console.print(f"\n[green]Results saved → {args.output}[/green]")


if __name__ == "__main__":
    main()
