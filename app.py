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
import asyncio
from typing import Any

from dotenv import load_dotenv
load_dotenv()

import chainlit as cl

from inferops.agent.graph import build_graph, make_llm, prepare_initial_state
from inferops.agent.intent import extract_intent
from inferops.agent.state import (
    WORKLOAD_DESCRIPTIONS,
    WORKLOAD_PRIMARY_METRIC,
)
from inferops.memory.db import init_db
from inferops.tools.final_report import FinalReportInput, write_final_report

_WELCOME = """\
# InferOps — vLLM Optimization Agent

Describe your inference scenario and I'll find the best vLLM configuration automatically.

**Examples:**
- *"I have Qwen2.5-1.5B on RTX 3060, chat scenario, target QPS=10"*
- *"Long document QA workload, concurrency=4, need low TTFT"*
- *"High concurrency short outputs, 32 users, maximize throughput"*

> **Note:** Running actual benchmarks requires vLLM to be running first:
> `bash scripts/start_vllm.sh`
> Planning and intent extraction work without it.

Type your scenario to begin.
"""

_LLM_BACKEND = os.getenv("INFEROPS_LLM", "openrouter")
_VLLM_HOST = os.getenv("VLLM_HOST", "127.0.0.1")
_VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))


async def _vllm_is_running() -> bool:
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"http://{_VLLM_HOST}:{_VLLM_PORT}/health", timeout=2.0)
            return r.status_code == 200
    except Exception:
        return False


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

    # Pre-flight: check vLLM is reachable before starting the graph
    if not await _vllm_is_running():
        await cl.Message(content=(
            "**vLLM is not running** — the planner above shows what the agent *would* do, "
            "but experiments require vLLM to be started first.\n\n"
            f"Start it in another terminal (port {_VLLM_PORT}):\n"
            "```bash\n"
            "bash scripts/start_vllm.sh 1.5B   # or 0.5B\n"
            "```\n"
            "Then send your message again."
        )).send()
        return

    # Step 2: Run the agent with streaming step updates
    session_prefix = f"ui_{uuid.uuid4().hex[:8]}_"
    t_start = time.time()

    try:
        await cl.Message(content="Running/loading baseline experiment…").send()
        state = await asyncio.to_thread(
            prepare_initial_state,
            intent.workload_name,
            session_prefix,
            intent.budget,
        )
        final_state: dict[str, Any] = state

        baseline = state["baseline_summary"]
        if baseline:
            await cl.Message(content=(
                f"**Baseline:** `{baseline['experiment_id']}`\n"
                f"  • throughput = **{baseline['throughput_rps']:.3f} RPS**\n"
                f"  • TTFT p99 = {baseline['ttft_p99_ms']:.1f} ms\n"
                f"  • bottleneck = `{baseline['bottleneck']}`"
            )).send()

        graph = build_graph(llm)
        await cl.Message(content=(
            f"Starting optimization loop "
            f"(remaining experiments={state['experiments_remaining']})…"
        )).send()

        async for mode, data in graph.astream(state, stream_mode=["updates", "values"]):
            if mode == "updates":
                for node_name, patch in data.items():
                    await _handle_node_event(node_name, patch)
            elif mode == "values":
                final_state = data
    except Exception as exc:
        err = str(exc)
        # Surface a helpful message for the most common failure: vLLM not running
        if any(k in err for k in ("Connection refused", "vllm", "VLLM", "timed out", "OOM")):
            await cl.Message(content=(
                "**Executor failed — vLLM server is not running.**\n\n"
                "Start vLLM first:\n```bash\nbash scripts/start_vllm.sh\n```\n"
                "Then retry your message. The planner output above was generated successfully."
            )).send()
        else:
            await cl.Message(content=f"**Agent error:** `{err[:300]}`").send()
        return

    elapsed = time.time() - t_start

    # Step 3: Final report
    await _send_final_report(final_state, intent.workload_name, elapsed, session_prefix)


async def _handle_node_event(node_name: str, patch: dict[str, Any] | None):
    """Stream a concise update for each node execution."""
    if patch is None:
        await cl.Message(content=f"⚠️ **{node_name}** returned no output (possible error).").send()
        return

    if node_name == "planner":
        hyps = patch.get("hypotheses", [])
        new_hyps = [h for h in hyps if h.get("status") == "pending"]
        if new_hyps:
            lines = [f"**Planner:** generated {len(new_hyps)} hypothesis to test"]
            for h in new_hyps:
                lines.append(f"  • `{h['param']} = {h['value']}`")
                lines.append(f"    _{h['rationale'][:200]}_")
            await cl.Message(content="\n".join(lines)).send()
            # Let user know the benchmark is about to start (it blocks for ~1 min)
            params = ", ".join(f"`{h['param']}={h['value']}`" for h in new_hyps)
            await cl.Message(
                content=f"⏳ **Executor:** running benchmark for {params} — please wait (~1 min per experiment)…"
            ).send()
        else:
            await cl.Message(content="**Planner:** no valid hypotheses generated (all rejected or already tried).").send()

    elif node_name == "executor":
        summaries = patch.get("experiment_summaries", [])
        if summaries:
            s = summaries[-1]
            improvement = s.get("vs_baseline_pct", 0)
            icon = "✅" if improvement > 0 else ("➡️" if improvement == 0 else "⬇️")
            await cl.Message(content=(
                f"{icon} **Result:** `{s['experiment_id']}`\n"
                f"  • throughput = **{s['throughput_rps']:.3f} RPS** ({improvement:+.1f}% vs baseline)\n"
                f"  • TTFT p99 = {s['ttft_p99_ms']:.1f} ms\n"
                f"  • bottleneck = `{s['bottleneck']}`"
            )).send()

    elif node_name == "reflector":
        if patch.get("should_stop"):
            reason = patch.get("stop_reason", "")
            await cl.Message(content=f"⏹ **Done.** {reason}").send()
        else:
            streak = patch.get("no_improvement_streak", 0)
            await cl.Message(content=(
                f"🔄 **Reflector:** continuing — no-improvement streak {streak}, "
                "trying next hypothesis…"
            )).send()


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
    report_path = ""

    try:
        out = write_final_report(FinalReportInput(
            workload_name=workload_name,
            session_prefix=session_prefix,
            experiment_summaries=summaries,
            baseline_summary=baseline,
            best_summary=best,
            citations=_collect_citations(state),
            output_path=f"reports/{session_prefix}final_report.md",
        ))
        report_path = out.output_path
    except Exception as exc:
        report_path = f"(failed to write report: {str(exc)[:160]})"

    lines = [
        "---",
        f"## Optimization Report — `{workload_name}`",
        "",
        f"**Session:** `{session_prefix}`  |  "
        f"**Experiments run:** {len(summaries)}  |  "
        f"**Wall clock:** {elapsed_s/60:.1f} min",
        "",
        f"**Report file:** `{report_path}`",
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


def _collect_citations(state: dict[str, Any]) -> list[str]:
    """Collect unique [source:] snippets from planner/executor rationale text."""
    citations: list[str] = []
    seen: set[str] = set()

    for hyp in state.get("hypotheses", []):
        rationale = hyp.get("rationale", "")
        if "[source:" not in rationale.lower():
            continue
        cleaned = " ".join(str(rationale).split())
        if cleaned not in seen:
            citations.append(cleaned)
            seen.add(cleaned)

    return citations
