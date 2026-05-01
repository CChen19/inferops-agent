"""InferOps Chainlit UI — natural language → vLLM optimization agent.

Usage:
    pip install ".[ui]"
    chainlit run app.py

Then open http://localhost:8000 and type something like:
    "I have Qwen2.5-1.5B on RTX 3060, chat scenario, target QPS=10"
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

import chainlit as cl

from inferops.agent.graph import build_graph, make_llm
from inferops.agent.intent import extract_intent
from inferops.agent.state import (
    WORKLOAD_DESCRIPTIONS,
    WORKLOAD_PRIMARY_METRIC,
    initial_state,
)
from inferops.memory.db import init_db

_WELCOME = """\
# InferOps — vLLM Optimization Agent

Describe your inference scenario and I'll find the best vLLM configuration automatically.

**Examples:**
- *"I have Qwen2.5-1.5B on RTX 3060, chat scenario, target QPS=10"*
- *"Long document QA workload, concurrency=4, need low TTFT"*
- *"High concurrency short outputs, 32 users, maximize throughput"*

Type your scenario to begin.
"""

_LLM_BACKEND = os.getenv("INFEROPS_LLM", "deepseek")


@cl.on_chat_start
async def on_start():
    init_db()
    await cl.Message(content=_WELCOME).send()


@cl.on_message
async def on_message(message: cl.Message):
    llm = make_llm(_LLM_BACKEND)

    # Step 1: Extract intent
    thinking = cl.Message(content="Analyzing your scenario…")
    await thinking.send()

    intent = extract_intent(message.content, llm)

    workload_desc = WORKLOAD_DESCRIPTIONS.get(intent.workload_name, "")
    primary_metric = WORKLOAD_PRIMARY_METRIC.get(intent.workload_name, "throughput_rps")

    summary_lines = [
        f"**Workload:** `{intent.workload_name}`",
        f"**Primary metric:** `{primary_metric}`",
        f"**Description:** {workload_desc}",
        f"**Budget:** {intent.budget} experiments",
    ]
    if intent.model_hint:
        summary_lines.insert(1, f"**Model:** `{intent.model_hint}`")
    if intent.target_qps:
        summary_lines.append(f"**Target QPS:** {intent.target_qps}")
    if intent.notes:
        summary_lines.append(f"**Notes:** {intent.notes}")

    await cl.Message(content="\n".join(summary_lines)).send()

    # Step 2: Run the agent with streaming step updates
    session_prefix = f"ui_{uuid.uuid4().hex[:8]}_"
    graph = build_graph(llm)
    state = initial_state(
        workload_name=intent.workload_name,
        session_prefix=session_prefix,
        max_experiments=intent.budget,
    )

    await cl.Message(content=f"Starting optimization loop (budget={intent.budget})…").send()

    t_start = time.time()
    step_count = 0

    async for event in graph.astream(state, stream_mode="updates"):
        for node_name, patch in event.items():
            step_count += 1
            await _handle_node_event(node_name, patch, step_count)

    elapsed = time.time() - t_start

    # Step 3: Final report
    final_state = await graph.ainvoke(state)  # get final state for report
    await _send_final_report(final_state, intent.workload_name, elapsed, session_prefix)


async def _handle_node_event(node_name: str, patch: dict[str, Any], step: int):
    """Stream a concise update for each node execution."""
    if node_name == "planner":
        hyps = patch.get("hypotheses", [])
        new_hyps = [h for h in hyps if h.get("status") == "pending"]
        if new_hyps:
            lines = [f"**Plan** (step {step}): generated {len(new_hyps)} hypothesis"]
            for h in new_hyps:
                lines.append(f"  • `{h['param']}={h['value']}` — {h['rationale'][:120]}…")
            await cl.Message(content="\n".join(lines)).send()

    elif node_name == "executor":
        summaries = patch.get("experiment_summaries", [])
        if summaries:
            s = summaries[-1]
            status_icon = "✅" if s.get("vs_baseline_pct", 0) >= 0 else "⚠️"
            await cl.Message(content=(
                f"{status_icon} **Experiment** `{s['experiment_id']}` — "
                f"rps={s['throughput_rps']:.3f}  "
                f"ttft_p99={s['ttft_p99_ms']:.1f}ms  "
                f"vs_baseline={s['vs_baseline_pct']:+.1f}%  "
                f"bottleneck={s['bottleneck']}"
            )).send()

    elif node_name == "reflector":
        if patch.get("should_stop"):
            reason = patch.get("stop_reason", "")
            await cl.Message(content=f"⏹ **Stopping:** {reason}").send()


async def _send_final_report(
    state: dict[str, Any],
    workload_name: str,
    elapsed_s: float,
    session_prefix: str,
):
    best = state.get("best_summary")
    baseline = state.get("baseline_summary")
    summaries = state.get("experiment_summaries", [])
    metric = WORKLOAD_PRIMARY_METRIC.get(workload_name, "throughput_rps")

    lines = [
        "---",
        f"## Optimization Report — `{workload_name}`",
        "",
        f"**Session:** `{session_prefix}`  |  "
        f"**Experiments run:** {len(summaries)}  |  "
        f"**Wall clock:** {elapsed_s/60:.1f} min",
        "",
    ]

    if baseline and best:
        improvement = best.get("vs_baseline_pct", 0)
        icon = "🟢" if improvement > 5 else "🟡" if improvement > 0 else "🔴"
        lines += [
            "### Result",
            "",
            f"| | Baseline | Best found | Improvement |",
            f"|---|---|---|---|",
            f"| `{metric}` | {baseline.get(metric, baseline.get('throughput_rps', 0)):.3f} "
            f"| {best.get(metric, best.get('throughput_rps', 0)):.3f} "
            f"| {icon} {improvement:+.1f}% |",
            f"| TTFT p99 | {baseline['ttft_p99_ms']:.1f}ms "
            f"| {best['ttft_p99_ms']:.1f}ms | — |",
            "",
            f"**Best config experiment:** `{best['experiment_id']}`",
            "",
        ]

    if summaries:
        lines += [
            "### All Experiments",
            "",
            "| Experiment | param | value | rps | ttft_p99 | bottleneck | vs_baseline |",
            "|---|---|---|---|---|---|---|",
        ]
        for s in summaries:
            lines.append(
                f"| `{s['experiment_id']}` | {s.get('param_changed') or 'baseline'} "
                f"| {s.get('value_changed', '')} "
                f"| {s['throughput_rps']:.3f} "
                f"| {s['ttft_p99_ms']:.1f}ms "
                f"| {s['bottleneck']} "
                f"| {s['vs_baseline_pct']:+.1f}% |"
            )

    stop_reason = state.get("stop_reason", "")
    if stop_reason:
        lines += ["", f"**Stop reason:** {stop_reason}"]

    await cl.Message(content="\n".join(lines)).send()
