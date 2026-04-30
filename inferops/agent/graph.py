"""LangGraph StateGraph for the inferops optimizer agent.

Graph topology:
  START → planner → executor → reflector ──┐
              ↑         ↑                   │ (conditional)
              └─────────┴───────────────────┘
                  or → END

Entry point: run_agent() — handles baseline, builds initial state, invokes graph.
"""

from __future__ import annotations

import os
import uuid
from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph
from rich.console import Console
from rich.table import Table

from inferops.agent.executor import executor_node
from inferops.agent.planner import planner_node
from inferops.agent.reflector import reflector_node, route_after_reflector
from inferops.agent.state import (
    WORKLOAD_PRIMARY_METRIC,
    AgentState,
    ExperimentSummary,
    initial_state,
)
from inferops.memory.db import get_result_by_id, init_db, save_result
from inferops.tools.analyze_bottleneck import AnalyzeBottleneckInput, analyze_bottleneck
from inferops.tools.run_benchmark import RunBenchmarkInput, run_benchmark

console = Console()


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def make_llm(backend: str = "deepseek", temperature: float = 0.3):
    """
    Create a LangChain ChatModel for the planner.

    backend: "deepseek" (requires DEEPSEEK_API_KEY) or "claude" (requires ANTHROPIC_API_KEY)
    """
    if backend == "deepseek":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="deepseek-chat",
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1",
            temperature=temperature,
            max_tokens=1024,
        )
    elif backend == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-sonnet-4-6",
            temperature=temperature,
            max_tokens=1024,
        )
    else:
        raise ValueError(f"Unknown LLM backend '{backend}'. Choose 'deepseek' or 'claude'.")


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph(llm) -> Any:
    """Compile the StateGraph with the given LLM bound into the planner node."""
    planner_with_llm = partial(planner_node, llm=llm)

    g = StateGraph(AgentState)
    g.add_node("planner",   planner_with_llm)
    g.add_node("executor",  executor_node)
    g.add_node("reflector", reflector_node)

    g.add_edge(START,      "planner")
    g.add_edge("planner",  "executor")
    g.add_edge("executor", "reflector")
    g.add_conditional_edges(
        "reflector",
        route_after_reflector,
        {"planner": "planner", "executor": "executor", "__end__": END},
    )

    return g.compile()


# ---------------------------------------------------------------------------
# Baseline helper
# ---------------------------------------------------------------------------

def _run_baseline(workload_name: str, session_prefix: str) -> tuple[ExperimentSummary, str]:
    """
    Run (or load) the default config as baseline. Returns (summary, bottleneck).
    """
    from configs.search_space import make_configs
    from workloads.definitions import ALL_WORKLOADS

    eid = f"{session_prefix}baseline"
    existing = get_result_by_id(eid)

    if existing is None:
        console.print(f"[bold]Running baseline experiment:[/] {eid} …")
        wl_map = {w.name: w for w in ALL_WORKLOADS}
        workload = wl_map[workload_name]
        base_cfg = make_configs(workload)[0].model_copy(update={"experiment_id": eid})
        from workloads.definitions import get_prompts
        prompts = get_prompts(workload)
        from inferops.bench_runner import run_experiment
        result = run_experiment(base_cfg, prompts)
        save_result(result)
    else:
        console.print(f"[dim]Baseline loaded from DB: {eid}[/dim]")
        result = existing

    primary_metric = WORKLOAD_PRIMARY_METRIC[workload_name]
    primary_val = getattr(result, primary_metric, result.throughput_rps)

    bottleneck = "unknown"
    try:
        ba = analyze_bottleneck(AnalyzeBottleneckInput(experiment_id=eid))
        bottleneck = ba.bottleneck
    except Exception:
        pass

    summary = ExperimentSummary(
        experiment_id=eid,
        param_changed=None,
        value_changed=None,
        throughput_rps=round(result.throughput_rps, 3),
        tokens_per_second=round(result.tokens_per_second, 1),
        ttft_p50_ms=round(result.ttft.p50, 1),
        ttft_p99_ms=round(result.ttft.p99, 1),
        e2e_p50_ms=round(result.e2e_latency.p50, 1),
        bottleneck=bottleneck,
        vs_baseline_pct=0.0,
    )
    return summary, bottleneck


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_agent(
    workload_name: str,
    llm,
    max_experiments: int = 8,
    session_prefix: str | None = None,
) -> AgentState:
    """
    Run the optimizer agent on a workload.

    Runs baseline first (or loads from DB if already done), then iterates
    planner → executor → reflector until budget is exhausted or convergence.

    Returns the final AgentState.
    """
    init_db()
    prefix = session_prefix or f"agent_{workload_name}_{uuid.uuid4().hex[:6]}_"
    console.rule(f"[bold cyan]Agent: {workload_name}[/]  prefix={prefix}")

    # 1. Baseline
    baseline_summary, baseline_bottleneck = _run_baseline(workload_name, prefix)

    # 2. Initial state
    state = initial_state(workload_name, prefix, max_experiments=max_experiments)
    state["baseline_summary"]    = baseline_summary
    state["best_summary"]        = baseline_summary
    state["experiment_summaries"] = [baseline_summary]
    state["tried_experiment_ids"] = [baseline_summary["experiment_id"]]
    state["current_bottleneck"]  = baseline_bottleneck
    # Baseline used one slot
    state["experiments_remaining"] = max_experiments - 1

    # 3. Run graph
    graph = build_graph(llm)
    final_state = graph.invoke(state)

    # 4. Print summary
    _print_run_summary(final_state)
    return final_state


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _print_run_summary(state: AgentState) -> None:
    primary = WORKLOAD_PRIMARY_METRIC[state["workload_name"]]
    best = state.get("best_summary")
    baseline = state.get("baseline_summary")

    console.rule("[bold]Agent run complete[/]")
    console.print(f"  Workload:     {state['workload_name']}")
    console.print(f"  Stop reason:  {state['stop_reason'] or 'not set'}")
    console.print(f"  Experiments:  {len(state['tried_experiment_ids'])} run")

    if baseline and best:
        console.print(
            f"  Baseline {primary}: {baseline[primary]:.3f}  →  "
            f"Best {primary}: {best[primary]:.3f}  "
            f"({best['vs_baseline_pct']:+.1f}%)"
        )

    if state["experiment_summaries"]:
        t = Table(show_lines=False, box=None, padding=(0, 1))
        t.add_column("experiment_id", style="dim")
        t.add_column("param", style="cyan")
        t.add_column("value")
        t.add_column(primary, justify="right")
        t.add_column("ttft_p99", justify="right")
        t.add_column("bottleneck")
        t.add_column("vs_baseline", justify="right")
        for s in state["experiment_summaries"]:
            t.add_row(
                s["experiment_id"].split("_")[-1] or "baseline",
                str(s.get("param_changed") or "—"),
                str(s.get("value_changed") or "—"),
                f"{s[primary]:.3f}",
                f"{s['ttft_p99_ms']}ms",
                s["bottleneck"],
                f"{s['vs_baseline_pct']:+.1f}%",
            )
        console.print(t)
