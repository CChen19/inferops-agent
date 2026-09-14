"""LangGraph StateGraph for the inferops optimizer agent.

Graph topology:
  START → planner → executor → reflector ──┐
              ↑         ↑                   │ (conditional)
              └─────────┴───────────────────┘
                  or → END

Reflect owns continue/remeasure/rollback/stop. Best promotion is only via
``is_confirmed_promotable`` (Week-1 ``is_promotable`` + ⑤ confirmation).

Entry point: run_agent() — handles baseline, builds initial state, invokes graph.
"""

from __future__ import annotations

import os
import sqlite3
import time
import uuid
from functools import partial
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from rich.console import Console
from rich.table import Table

from inferops.agent.executor import executor_node
from inferops.agent.planner import planner_node
from inferops.agent.recovery import reraise_hard_control
from inferops.agent.reflector import reflector_node, route_after_reflector
from inferops.agent.state import (
    WORKLOAD_PRIMARY_METRIC,
    AgentState,
    ExperimentSummary,
    initial_state,
    is_promotable_summary,
    primary_metric_of,
    summary_from_result,
)
from inferops.memory.db import (
    get_result_by_id,
    get_task,
    init_db,
    save_result,
    save_task,
    update_task_status,
)
from inferops.schemas import is_promotable
from inferops.task import (
    OptimizationTask,
    default_task_for_workload,
    require_confirmed,
    task_from_mapping,
)
from inferops.tools.analyze_bottleneck import AnalyzeBottleneckInput, analyze_bottleneck

console = Console()


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def make_llm(backend: str = "openrouter", temperature: float = 0.3):
    """
    Create a LangChain ChatModel for the planner.

    backend:
      "openrouter" (requires OPENROUTER_API_KEY) — default, uses deepseek/deepseek-chat
      "deepseek"   (requires DEEPSEEK_API_KEY)   — direct DeepSeek API
      "claude"     (requires ANTHROPIC_API_KEY)   — Anthropic direct
    """
    if backend == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat"),
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature,
            max_tokens=1024,
        )
    elif backend == "deepseek":
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
        raise ValueError(
            f"Unknown LLM backend '{backend}'. "
            "Choose 'openrouter', 'deepseek', or 'claude'."
        )


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def session_thread_id(session_prefix: str) -> str:
    """Stable LangGraph thread_id derived from the session prefix."""
    raw = (session_prefix or "").strip()
    return raw.rstrip("_") or raw or "inferops"


def graph_invoke_config(
    session_prefix: str,
    *,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """RunnableConfig with a stable thread/session identity for checkpoint/resume."""
    return {
        "configurable": {
            "thread_id": thread_id or session_thread_id(session_prefix),
        }
    }


def production_checkpointer(db_path: Path | str = Path("inferops_memory.db")) -> Any:
    """Create a disk-backed LangGraph saver sharing the experiment SQLite DB."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
        raise RuntimeError(
            "Production resume requires the 'langgraph-checkpoint-sqlite' package"
        ) from exc

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn)


def build_graph(
    llm,
    *,
    checkpointer: Any = None,
    interrupt_before: list[str] | None = None,
) -> Any:
    """Compile the StateGraph with the given LLM bound into the planner node.

    Production ``run_agent`` passes a SqliteSaver checkpointer and a stable
    ``thread_id``. Eval / unit assembly may omit the checkpointer so
    ``invoke(state)`` stays config-free.
    """
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

    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    if interrupt_before:
        compile_kwargs["interrupt_before"] = list(interrupt_before)
    return g.compile(**compile_kwargs)


# ---------------------------------------------------------------------------
# Baseline helper
# ---------------------------------------------------------------------------

def _run_baseline(
    workload_name: str,
    session_prefix: str,
    task: OptimizationTask | None = None,
) -> tuple[ExperimentSummary, str]:
    """
    Run (or load) the default config as baseline. Returns (summary, bottleneck).

    Legacy DB rows without contract evidence stay insufficient_evidence and are
    NOT auto-promoted to valid / best.
    """
    from configs.search_space import make_configs
    from workloads.definitions import ALL_WORKLOADS

    eid = f"{session_prefix}baseline"
    existing = get_result_by_id(eid)
    resolved_task = task
    workload = resolved_task.workload if resolved_task is not None else None
    model_name = resolved_task.model_name if resolved_task is not None else None

    if existing is None:
        console.print(f"[bold]Running baseline experiment:[/] {eid} …")
        if workload is None:
            wl_map = {w.name: w for w in ALL_WORKLOADS}
            workload = wl_map[workload_name]
        base_cfg = make_configs(workload, model_name=model_name)[0].model_copy(
            update={"experiment_id": eid}
        )
        from workloads.definitions import get_prompts
        prompts = get_prompts(workload)
        from inferops.bench_runner import BenchmarkError, run_experiment
        try:
            result = run_experiment(
                base_cfg,
                prompts,
                session_id=session_prefix,
                service_mode=(
                    resolved_task.service_mode.value if resolved_task is not None else None
                ),
            )
            save_result(result)
        except BenchmarkError as exc:
            if exc.result is not None:
                save_result(exc.result)
            raise
    else:
        console.print(f"[dim]Baseline loaded from DB: {eid}[/dim]")
        result = existing
        if not is_promotable(result):
            console.print(
                f"[yellow]Baseline {eid} is not promotable "
                f"(status={result.status.value}) — will not seed best_summary[/yellow]"
            )

    primary_metric = (
        resolved_task.primary_metric
        if resolved_task is not None
        else WORKLOAD_PRIMARY_METRIC[workload_name]
    )

    bottleneck = "unknown"
    try:
        ba = analyze_bottleneck(AnalyzeBottleneckInput(experiment_id=eid))
        bottleneck = ba.bottleneck
    except Exception as exc:
        reraise_hard_control(exc)

    summary = summary_from_result(
        result,
        param_changed=None,
        value_changed=None,
        baseline_primary=getattr(result, primary_metric, result.throughput_rps),
        primary_metric=primary_metric,
        bottleneck=bottleneck,
    )
    # Baseline id is session-scoped even when loading a reused row
    summary["experiment_id"] = eid
    summary["vs_baseline_pct"] = 0.0
    return summary, bottleneck


def prepare_initial_state(
    workload_name: str,
    session_prefix: str,
    max_experiments: int = 8,
    task: OptimizationTask | dict | None = None,
) -> AgentState:
    """
    Build an AgentState with the baseline experiment already run or loaded.

    Both the CLI and Chainlit UI need this setup before entering the graph:
    planner/executor logic expects baseline_summary to exist so that proposed
    changes can be compared against the default config.

    best_summary is only seeded from baseline when the baseline is promotable
    (valid + critical config evidence). Old / insufficient-evidence rows never
    auto-become best.

    An OptimizationTask, when provided, must already be confirmed. The CLI/eval
    path synthesizes a confirmed default task so conditions stay consistent.
    """
    resolved = task_from_mapping(task)
    if resolved is None:
        resolved = default_task_for_workload(workload_name, max_experiments)
    else:
        require_confirmed(resolved)
        workload_name = resolved.workload.name
        max_experiments = resolved.experiment_budget

    baseline_summary, baseline_bottleneck = _run_baseline(
        workload_name, session_prefix, task=resolved
    )

    state = initial_state(
        workload_name,
        session_prefix,
        max_experiments=max_experiments,
        optimization_task=resolved.model_dump(mode="json"),
    )
    state["baseline_summary"] = baseline_summary
    state["best_summary"] = (
        baseline_summary if is_promotable_summary(baseline_summary) else None
    )
    state["experiment_summaries"] = [baseline_summary]
    state["tried_experiment_ids"] = [baseline_summary["experiment_id"]]
    state["current_bottleneck"] = baseline_bottleneck
    state["experiments_remaining"] = max(0, max_experiments - 1)
    state["started_at_s"] = time.time()
    return state


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_agent(
    workload_name: str | None,
    llm,
    max_experiments: int = 8,
    session_prefix: str | None = None,
    interrupt_before: list[str] | None = None,
    task: OptimizationTask | dict | None = None,
    resume_task_id: str | None = None,
    db_path: Path | str = Path("inferops_memory.db"),
) -> AgentState:
    """
    Run the optimizer agent on a workload.

    Runs baseline first (or loads from DB if already done), then iterates
    planner → executor → reflector until budget is exhausted or convergence.

    Returns the final AgentState.
    """
    db_file = Path(db_path)
    init_db(db_file)

    stored = get_task(resume_task_id, db_path=db_file) if resume_task_id else None
    if resume_task_id and stored is None:
        raise ValueError(f"No persisted task found for task_id={resume_task_id!r}")

    if stored is not None:
        resolved_task = task_from_mapping(stored.confirmed_task)
        assert resolved_task is not None
        require_confirmed(resolved_task)
        prefix = stored.session_prefix
        thread_id = stored.thread_id
        workload_name = resolved_task.workload.name
        max_experiments = resolved_task.experiment_budget
    else:
        resolved_task = task_from_mapping(task)
        if resolved_task is None:
            if workload_name is None:
                raise ValueError("workload_name is required for a new task")
            resolved_task = default_task_for_workload(workload_name, max_experiments)
        require_confirmed(resolved_task)
        workload_name = resolved_task.workload.name
        max_experiments = resolved_task.experiment_budget
        prefix = session_prefix or f"agent_{workload_name}_{uuid.uuid4().hex[:6]}_"
        thread_id = session_thread_id(prefix)
        task_record = save_task(
            task_id=resolved_task.task_id,
            session_prefix=prefix,
            thread_id=thread_id,
            confirmed_task=resolved_task.model_dump(mode="json"),
            status="confirmed",
            db_path=db_file,
        )
        prefix = task_record.session_prefix
        thread_id = task_record.thread_id

    console.rule(f"[bold cyan]Agent: {workload_name}[/]  prefix={prefix}")

    # Build the graph before baseline so a persisted checkpoint can resume directly.
    graph = build_graph(
        llm,
        checkpointer=production_checkpointer(db_file),
        interrupt_before=interrupt_before,
    )
    config = graph_invoke_config(prefix, thread_id=thread_id)
    snapshot = graph.get_state(config) if hasattr(graph, "get_state") else None
    has_checkpoint = bool(snapshot and snapshot.values)
    state = None
    if not has_checkpoint:
        state = prepare_initial_state(
            workload_name,
            prefix,
            max_experiments=max_experiments,
            task=resolved_task,
        )

    update_task_status(resolved_task.task_id, "running", db_path=db_file)
    try:
        final_state = graph.invoke(state, config)
    except BaseException:
        update_task_status(resolved_task.task_id, "interrupted", db_path=db_file)
        raise

    post_run = graph.get_state(config) if hasattr(graph, "get_state") else None
    final_status = "running" if post_run and post_run.next else "completed"
    update_task_status(resolved_task.task_id, final_status, db_path=db_file)

    _print_run_summary(final_state)
    return final_state


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _print_run_summary(state: AgentState) -> None:
    primary = primary_metric_of(state)
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
