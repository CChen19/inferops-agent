"""InferOps Chainlit UI — natural language → confirmed OptimizationTask → agent.

Usage:
    pip install ".[ui]"
    chainlit run app.py

Then open http://localhost:8000 and type something like:
    "I have Qwen2.5-1.5B on RTX 3060, chat scenario, target QPS=10, TTFT p99 <= 200ms"
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
from inferops.agent.intent import Intent, interpret_user_request
from inferops.memory.db import init_db
from inferops.task import (
    ServiceControlMode,
    TaskStatus,
    confirm_task,
    format_task_blockers,
    format_task_confirmation,
)
from inferops.tools.final_report import FinalReportInput, write_final_report

_WELCOME = """\
# InferOps — vLLM tuning assistant

Describe a serving goal. I will draft a task, wait for your confirmation, then
run a bounded set of experiments and tell you whether a change is worth adopting.

I will **not** silently swap your model or workload. Unsupported requests are
rejected or sent back for clarification before any GPU budget is spent.

**Examples:**
- *"I have Qwen2.5-1.5B on RTX 3060, chat scenario, target QPS=10, TTFT p99 under 200ms"*
- *"Long document QA, concurrency=4, keep TTFT p99 <= 400ms"*
- *"High concurrency short outputs, 32 users, maximize throughput"*

Target QPS is a **measured throughput goal**. Offered arrival-rate scheduling
is not implemented (load is concurrency-limited).

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


def _intent_from_session() -> Intent | None:
    raw = cl.user_session.get("pending_intent")
    if not raw:
        return None
    return Intent(**raw)


async def _ask_user_confirm() -> bool:
    """Ask the user to spend experiment budget. False = do not run."""
    actions = [
        cl.Action(name="confirm", payload={"value": "confirm"}, label="Confirm and run"),
        cl.Action(name="cancel", payload={"value": "cancel"}, label="Cancel"),
    ]
    try:
        res = await cl.AskActionMessage(
            content="Confirm this task to start the baseline and spend the experiment budget?",
            actions=actions,
            timeout=300,
        ).send()
    except TypeError:
        res = await cl.AskActionMessage(
            content="Confirm this task to start the baseline and spend the experiment budget?",
            actions=[
                cl.Action(name="confirm", value="confirm", label="Confirm and run"),
                cl.Action(name="cancel", value="cancel", label="Cancel"),
            ],
        ).send()
    if not res:
        return False
    value = res.get("value") if isinstance(res, dict) else getattr(res, "payload", {})
    if isinstance(value, dict):
        value = value.get("value")
    return str(value) == "confirm"


@cl.on_message
async def on_message(message: cl.Message):
    llm = make_llm(_LLM_BACKEND)

    thinking = cl.Message(content="Drafting an optimization task…")
    await thinking.send()

    previous = _intent_from_session()
    intent, task = interpret_user_request(message.content, llm, previous=previous)
    cl.user_session.set("pending_intent", intent.as_dict())

    if task.status in {TaskStatus.NEEDS_CLARIFICATION, TaskStatus.REJECTED}:
        await cl.Message(content=format_task_blockers(task)).send()
        return

    await cl.Message(content=format_task_confirmation(task)).send()
    if not await _ask_user_confirm():
        await cl.Message(
            content="Cancelled. No experiments were started and no GPU budget was spent."
        ).send()
        return

    task = confirm_task(task)
    cl.user_session.set("pending_intent", None)
    cl.user_session.set("confirmed_task", task.model_dump(mode="json"))

    if task.service_mode == ServiceControlMode.EXTERNAL and not await _vllm_is_running():
        await cl.Message(content=(
            "**External vLLM is not reachable.** This task is `external`, so "
            "InferOps will not start a managed server.\n\n"
            f"Start vLLM on port {_VLLM_PORT}, or resend the request with "
            "`managed` service mode.\n"
            "```bash\n"
            "bash scripts/start_vllm.sh 1.5B   # or 0.5B\n"
            "```"
        )).send()
        return

    # Step 2: Run the agent with streaming step updates
    session_prefix = f"ui_{uuid.uuid4().hex[:8]}_"
    t_start = time.time()

    try:
        await cl.Message(content="Running/loading baseline experiment…").send()
        state = await asyncio.to_thread(
            prepare_initial_state,
            task.workload.name,
            session_prefix,
            task.experiment_budget,
            task,
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
    await _send_final_report(final_state, task.workload.name, elapsed, session_prefix)


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
    report_path = ""
    task_dump = state.get("optimization_task")

    try:
        out = write_final_report(FinalReportInput(
            workload_name=workload_name,
            session_prefix=session_prefix,
            experiment_summaries=summaries,
            baseline_summary=baseline,
            best_summary=best,
            citations=_collect_citations(state),
            output_path=f"reports/{session_prefix}final_report.md",
            optimization_task=task_dump,
            stop_reason=str(state.get("stop_reason") or ""),
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
    from inferops.decision import build_decision, render_decision_markdown
    from inferops.task import format_task_conditions_markdown, task_from_mapping

    if task_dump:
        task_obj = task_from_mapping(task_dump)
        if task_obj is not None:
            lines += format_task_conditions_markdown(task_obj)

    decision = build_decision(
        baseline_summary=baseline,
        best_summary=best,
        experiment_summaries=summaries,
        optimization_task=task_dump,
        stop_reason=str(state.get("stop_reason") or ""),
    )
    lines += render_decision_markdown(decision)
    lines.append("")

    def _fmt(v: Any, spec: str) -> str:
        if v is None:
            return "n/a"
        return format(float(v), spec)

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
                f"| {_fmt(s.get('throughput_rps'), '.3f')} "
                f"| {_fmt(s.get('ttft_p99_ms'), '.1f')}ms "
                f"| {s.get('bottleneck', 'unknown')} "
                f"| {_fmt(s.get('vs_baseline_pct'), '+.1f')}"
                f"{'%' if s.get('vs_baseline_pct') is not None else ''} |"
            )

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
