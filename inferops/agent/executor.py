"""Executor node — picks one pending hypothesis, runs the experiment, updates state.

Deduplication: if a (param, value) pair was already tried, marks the hypothesis
as "skipped" and moves on without spending an experiment slot.

Tool call chain per hypothesis:
  1. propose_config_patch  — validates param/value against safe ranges
  2. run_benchmark         — starts vLLM, runs load, persists result to DB
  3. analyze_bottleneck    — classifies bottleneck from the stored result
  4. compare_experiments   — bootstrap CI vs baseline
"""

from __future__ import annotations

from typing import Any

from rich.console import Console

from inferops.agent.state import (
    WORKLOAD_PRIMARY_METRIC,
    AgentState,
    ExperimentSummary,
    Hypothesis,
    is_duplicate,
    pending_hypotheses,
    summary_from_result,
)
from inferops.memory.db import get_result_by_id
from inferops.tools.analyze_bottleneck import AnalyzeBottleneckInput, analyze_bottleneck
from inferops.tools.compare_experiments import CompareExperimentsInput, compare_experiments
from inferops.tools.run_benchmark import RunBenchmarkInput, run_benchmark

console = Console()


def executor_node(state: AgentState) -> dict:
    """
    Pick the first pending hypothesis, run the experiment, and return a state patch.

    If the hypothesis is a duplicate, mark it "skipped" and return immediately
    without decrementing the budget.
    """
    pending = pending_hypotheses(state)
    if not pending:
        # Nothing to execute — signal planner to generate more
        return {}

    hyp = pending[0]
    primary_metric = WORKLOAD_PRIMARY_METRIC[state["workload_name"]]

    # --- Deduplication check ---
    if is_duplicate(state, hyp["param"], hyp["value"]):
        console.print(f"  [dim]executor: skip duplicate ({hyp['param']}={hyp['value']})[/dim]")
        updated_hyps = _set_status(state["hypotheses"], hyp["id"], "skipped", None)
        return {"hypotheses": updated_hyps}

    # --- Build experiment ID ---
    eid = f"{state['session_prefix']}{hyp['param']}_{hyp['value']}"

    # --- Check DB for existing result (in case of resume) ---
    existing = get_result_by_id(eid)
    if existing is not None:
        console.print(f"  [dim]executor: loaded from DB: {eid}[/dim]")
        result = existing
        bench_dict = _result_to_bench_dict(existing)
    else:
        # --- propose_config_patch: validate ranges ---
        try:
            from inferops.tools.propose_config import ProposeConfigInput, propose_config_patch
            base_eid = state["baseline_summary"]["experiment_id"] if state["baseline_summary"] else ""
            propose_config_patch(ProposeConfigInput(
                base_experiment_id=base_eid,
                param=hyp["param"],
                value=hyp["value"],
                rationale=hyp["rationale"],
                new_experiment_id=eid,
            ))
        except ValueError as exc:
            console.print(f"  [red]executor: propose rejected ({exc})[/red]")
            updated_hyps = _set_status(state["hypotheses"], hyp["id"], "failed", eid)
            return {"hypotheses": updated_hyps}

        # --- run_benchmark ---
        console.print(f"  executor: running {eid} ({hyp['param']}={hyp['value']}) …")
        try:
            bench_out = run_benchmark(RunBenchmarkInput(
                experiment_id=eid,
                config_patch={hyp["param"]: hyp["value"]},
                workload_name=state["workload_name"],
                persist=True,
            ))
        except Exception as exc:
            console.print(f"  [red]executor: benchmark failed ({exc})[/red]")
            updated_hyps = _set_status(state["hypotheses"], hyp["id"], "failed", eid)
            return {
                "hypotheses": updated_hyps,
                "tried_experiment_ids": state["tried_experiment_ids"] + [eid],
                "experiments_remaining": state["experiments_remaining"] - 1,
            }
        bench_dict = bench_out.model_dump()
        result = get_result_by_id(eid)

    # --- analyze_bottleneck ---
    bottleneck = "unknown"
    try:
        ba = analyze_bottleneck(AnalyzeBottleneckInput(experiment_id=eid))
        bottleneck = ba.bottleneck
    except Exception:
        pass

    # --- compare_experiments vs baseline ---
    vs_baseline_pct = 0.0
    if state["baseline_summary"] is not None:
        try:
            cmp = compare_experiments(CompareExperimentsInput(
                experiment_id_a=state["baseline_summary"]["experiment_id"],
                experiment_id_b=eid,
                metric=primary_metric,
                n_bootstrap=1000,
            ))
            # For "max" metrics: positive delta_pct = b is better
            vs_baseline_pct = cmp.delta_pct
        except Exception:
            pass

    # --- Build ExperimentSummary ---
    baseline_primary = (
        state["baseline_summary"][primary_metric]
        if state["baseline_summary"] else 0.0
    )
    summary = ExperimentSummary(
        experiment_id=eid,
        param_changed=hyp["param"],
        value_changed=hyp["value"],
        throughput_rps=bench_dict.get("throughput_rps", 0.0),
        tokens_per_second=bench_dict.get("tokens_per_second", 0.0),
        ttft_p50_ms=bench_dict.get("ttft_p50_ms", 0.0),
        ttft_p99_ms=bench_dict.get("ttft_p99_ms", 0.0),
        e2e_p50_ms=bench_dict.get("e2e_p50_ms", 0.0),
        bottleneck=bottleneck,
        vs_baseline_pct=round(vs_baseline_pct, 2),
    )

    # --- Update best ---
    current_primary = bench_dict.get(primary_metric, 0.0)
    best_primary = state["best_summary"][primary_metric] if state["best_summary"] else 0.0
    new_best = summary if current_primary > best_primary else state["best_summary"]

    # --- Mark hypothesis done ---
    status = "success" if vs_baseline_pct >= 0 else "failed"
    updated_hyps = _set_status(state["hypotheses"], hyp["id"], status, eid)

    # --- Trajectory ---
    traj_step = {
        "step": len(state["trajectory"]) + 1,
        "node": "executor",
        "workload": state["workload_name"],
        "action": f"run_benchmark({hyp['param']}={hyp['value']})",
        "experiment_id": eid,
        "reasoning": hyp["rationale"],
        "result": {
            primary_metric: current_primary,
            "ttft_p99_ms": summary["ttft_p99_ms"],
            "bottleneck": bottleneck,
            "vs_baseline_pct": vs_baseline_pct,
        },
    }

    console.print(
        f"  executor: done — {primary_metric}={current_primary:.3f}  "
        f"bottleneck={bottleneck}  vs_baseline={vs_baseline_pct:+.1f}%"
    )

    return {
        "hypotheses":            updated_hyps,
        "tried_experiment_ids":  state["tried_experiment_ids"] + [eid],
        "experiment_summaries":  state["experiment_summaries"] + [summary],
        "best_summary":          new_best,
        "current_bottleneck":    bottleneck,
        "experiments_remaining": state["experiments_remaining"] - 1,
        "trajectory":            state["trajectory"] + [traj_step],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_status(
    hypotheses: list[Hypothesis],
    hyp_id: str,
    status: str,
    experiment_id: str | None,
) -> list[Hypothesis]:
    return [
        {**h, "status": status, "experiment_id": experiment_id}
        if h["id"] == hyp_id else h
        for h in hypotheses
    ]


def _result_to_bench_dict(result) -> dict[str, Any]:
    return {
        "throughput_rps":   result.throughput_rps,
        "tokens_per_second": result.tokens_per_second,
        "ttft_p50_ms":      result.ttft.p50,
        "ttft_p99_ms":      result.ttft.p99,
        "e2e_p50_ms":       result.e2e_latency.p50,
    }
