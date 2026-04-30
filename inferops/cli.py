"""Typer CLI for inferops-agent.

The project also keeps the original scripts under ``scripts/`` for backwards
compatibility. This module exists so the ``inferops`` console script declared in
``pyproject.toml`` resolves to a real entry point after package installation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    help="Autonomous vLLM inference optimization tools.",
    no_args_is_help=True,
)


@app.command()
def agent(
    workload: str = typer.Option(..., help="Workload to optimize."),
    llm: str = typer.Option("deepseek", help="Planner backend: deepseek or claude."),
    budget: int = typer.Option(8, help="Max vLLM experiments including baseline."),
    prefix: Optional[str] = typer.Option(None, help="Experiment ID prefix for this session."),
    temperature: float = typer.Option(0.3, help="LLM temperature."),
) -> None:
    """Run the Plan-Execute-Reflect optimizer agent."""
    from inferops.agent.graph import make_llm, run_agent
    from inferops.agent.state import WORKLOAD_PRIMARY_METRIC

    valid = set(WORKLOAD_PRIMARY_METRIC)
    if workload not in valid:
        raise typer.BadParameter(f"Unknown workload '{workload}'. Valid: {', '.join(sorted(valid))}")

    model = make_llm(backend=llm, temperature=temperature)
    run_agent(
        workload_name=workload,
        llm=model,
        max_experiments=budget,
        session_prefix=prefix,
    )


@app.command("grid-sweep")
def grid_sweep(
    workloads: list[str] = typer.Option(["all"], help="Workload names, or 'all'."),
    output_dir: Path = typer.Option(Path("data/ground_truth"), help="Ground-truth output directory."),
    skip_existing: bool = typer.Option(False, help="Reuse existing results from experiment memory DB."),
    dry_run: bool = typer.Option(False, help="Print the sweep plan without running vLLM."),
) -> None:
    """Run the Phase 3 grid sweep and write ground-truth JSON files."""
    import scripts.run_grid_sweep as sweep

    sweep.init_db()
    all_wl_map = {w.name: w for w in sweep.ALL_WORKLOADS}
    selected = sweep.ALL_WORKLOADS if workloads == ["all"] else [all_wl_map[w] for w in workloads]
    for wl in selected:
        sweep.sweep_workload(wl, output_dir, skip_existing=skip_existing, dry_run=dry_run)


@app.command()
def eval(
    prefix: str = typer.Option(..., help="Experiment ID prefix for the agent session."),
    ground_truth: Path = typer.Option(Path("data/ground_truth"), help="Ground-truth JSON directory."),
    workloads: list[str] = typer.Option(["all"], help="Workload names, or 'all'."),
) -> None:
    """Evaluate an agent session against ground-truth grid-sweep results."""
    from inferops.eval.runner import ALL_WORKLOAD_NAMES, evaluate, print_summary_table

    names = ALL_WORKLOAD_NAMES if workloads == ["all"] else workloads
    scores = evaluate(prefix, ground_truth, names)
    if not scores:
        raise typer.Exit(code=1)
    print_summary_table(scores)


@app.command("memory")
def memory(
    workload: Optional[str] = typer.Option(None, help="Filter by workload."),
    sort_by: str = typer.Option("throughput_rps", help="Metric to sort by."),
    top_k: int = typer.Option(10, help="Number of rows to print."),
) -> None:
    """Print top experiment-memory rows."""
    from rich.console import Console
    from rich.table import Table

    from inferops.memory.db import query_results

    rows = query_results(workload_name=workload, sort_by=sort_by, top_k=top_k)
    table = Table(title="Experiment memory")
    table.add_column("experiment_id", style="cyan")
    table.add_column("workload")
    table.add_column("rps", justify="right")
    table.add_column("ttft_p99", justify="right")
    table.add_column("e2e_p50", justify="right")
    for row in rows:
        table.add_row(
            str(row["experiment_id"]),
            str(row["workload_name"]),
            f"{row['throughput_rps']:.3f}",
            f"{row['ttft_p99_ms']:.1f}",
            f"{row['e2e_p50_ms']:.1f}",
        )
    Console().print(table)
