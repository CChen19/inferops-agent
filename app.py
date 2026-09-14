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

from inferops.agent.graph import (
    build_graph,
    graph_invoke_config,
    make_llm,
    prepare_initial_state,
    production_checkpointer,
    run_agent,
    session_thread_id,
)
from inferops.agent.intent import Intent, interpret_user_request
from inferops.memory.db import get_task, init_db, save_task, update_task_status
from inferops.resume import (
    format_reflector_update,
    format_resume_help,
    is_resume_command,
    parse_resume_task_id,
)
from inferops.task import (
    ServiceControlMode,
    TaskStatus,
    confirm_task,
    format_task_blockers,
    format_task_confirmation,
)
from inferops.tools.final_report import (
    FinalReportInput,
    _format_all_experiments_table,
    _format_live_result_message,
    write_final_report,
)
from inferops.tools.managed_lifecycle import (
    cancel_owned_children,
    clear_cancel,
    request_cancel,
)

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

Type your scenario to begin, or `resume <task_id>` to continue a saved task.
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


@cl.on_stop
async def on_stop():
    """User pressed Stop: stop only the managed vLLM this process spawned.

    Sets the cancel flag first, so the running experiment fails closed at its
    next cancel gate (after spawn, after readiness, before/after load) and no
    further experiment starts. Then stops registered owned children — the
    child is registered as soon as it is spawned, so a Stop during the model
    load window aborts it — and releases the GPU lease. External / unknown
    services are never touched.
    """
    request_cancel()
    reports = await asyncio.to_thread(cancel_owned_children, "user pressed stop")
    task_dump = cl.user_session.get("confirmed_task") or {}
    task_id = task_dump.get("task_id")
    if task_id:
        update_task_status(task_id, "cancelled")
    if reports:
        lines = ["**Cancelled.** Stopped the managed vLLM InferOps started:"]
        for rep in reports:
            lines.append(
                f"  • pid={rep.get('pid')} experiment=`{rep.get('experiment_id')}` "
                f"(GPU lease released: {rep.get('lease_released')})"
            )
        lines.append(
            "That experiment is recorded as cancelled (never `valid`). "
            "No other process was stopped, and no further experiment will start."
        )
    else:
        lines = [
            "**Cancelled.** No InferOps-managed vLLM child was registered at this moment, "
            "so no process was stopped. If an experiment is in flight it will abort at its "
            "next cancel check (never `valid`) and stop its own child; no further "
            "experiment will start. An external vLLM you run yourself is never touched."
        ]
    if task_id:
        lines.append("")
        lines.append(format_resume_help(task_id))
    await cl.Message(content="\n".join(lines)).send()


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


async def _run_resumed_task(llm, resume_task_id: str) -> None:
    """Continue a persisted task. No new draft/confirm. Missing id → no GPU."""
    stored = get_task(resume_task_id)
    if stored is None:
        await cl.Message(
            content=(
                f"**Cannot resume.** No persisted task found for "
                f"`{resume_task_id}`.\n\n"
                f"{format_resume_help(resume_task_id)} "
                "No GPU budget was spent."
            )
        ).send()
        return

    cl.user_session.set("confirmed_task", stored.confirmed_task)
    await cl.Message(
        content=(
            f"Resuming task `{resume_task_id}` from the persisted checkpoint "
            f"(session `{stored.session_prefix}`). Skipping a new draft/confirm."
        )
    ).send()

    t_start = time.time()
    clear_cancel()
    try:
        final_state = await asyncio.to_thread(
            run_agent,
            None,
            llm,
            resume_task_id=resume_task_id,
        )
    except ValueError as exc:
        await cl.Message(
            content=f"**Cannot resume.** {exc}\n\nNo GPU budget was spent."
        ).send()
        return
    except Exception as exc:
        err = str(exc)
        if type(exc).__name__ == "TaskCancelled":
            update_task_status(resume_task_id, "cancelled")
            await cl.Message(content=f"**Run cancelled.** `{err[:300]}`").send()
            return
        update_task_status(resume_task_id, "failed")
        if "GPU busy" in err or "Refusing managed start" in err:
            await cl.Message(
                content=(
                    f"**Managed start refused — nothing was started or stopped.**\n\n`{err[:600]}`"
                )
            ).send()
            return
        await cl.Message(content=f"**Agent error:** `{err[:300]}`").send()
        return

    elapsed = time.time() - t_start
    workload = str(final_state.get("workload_name") or "")
    prefix = str(final_state.get("session_prefix") or stored.session_prefix)
    await _send_final_report(final_state, workload, elapsed, prefix)


@cl.on_message
async def on_message(message: cl.Message):
    llm = make_llm(_LLM_BACKEND)
    text = message.content or ""
    resume_id = parse_resume_task_id(text)
    if resume_id is not None:
        await _run_resumed_task(llm, resume_id)
        return
    if is_resume_command(text):
        await cl.Message(
            content=(
                "**Cannot resume.** Need a 12-character task id.\n\n"
                "Use `resume <task_id>` in chat or "
                "`inferops agent --resume-task <task_id>` in the CLI. "
                "No GPU budget was spent."
            )
        ).send()
        return

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

    session_prefix = f"ui_{uuid.uuid4().hex[:8]}_"
    thread_id = session_thread_id(session_prefix)
    save_task(
        task_id=task.task_id,
        session_prefix=session_prefix,
        thread_id=thread_id,
        confirmed_task=task.model_dump(mode="json"),
        status="confirmed",
    )
    await cl.Message(
        content=f"Task `{task.task_id}` saved. {format_resume_help(task.task_id)}"
    ).send()

    if task.service_mode == ServiceControlMode.EXTERNAL and not await _vllm_is_running():
        update_task_status(task.task_id, "blocked_external")
        await cl.Message(
            content=(
                "**External vLLM is not reachable.** This task is `external`, so "
                "InferOps will not start a managed server.\n\n"
                f"Start vLLM on port {_VLLM_PORT}, or resend the request with "
                "`managed` service mode.\n"
                "```bash\n"
                "bash scripts/start_vllm.sh 1.5B   # or 0.5B\n"
                "```"
            )
        ).send()
        return

    # Step 2: Run the agent with streaming step updates
    t_start = time.time()
    clear_cancel()

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
            await cl.Message(
                content=(
                    f"**Baseline:** `{baseline['experiment_id']}`\n"
                    f"  • throughput = **{baseline['throughput_rps']:.3f} RPS**\n"
                    f"  • TTFT p99 = {baseline['ttft_p99_ms']:.1f} ms\n"
                    f"  • bottleneck = `{baseline['bottleneck']}`"
                )
            ).send()

        with production_checkpointer() as checkpointer:
            graph = build_graph(llm, checkpointer=checkpointer)
            config = graph_invoke_config(session_prefix, thread_id=thread_id)
            update_task_status(task.task_id, "running")
            await cl.Message(
                content=(
                    f"Starting optimization loop "
                    f"(remaining experiments={state['experiments_remaining']})…"
                )
            ).send()

            events = graph.stream(
                state,
                config,
                stream_mode=["updates", "values"],
            )
            done = object()
            while True:
                item = await asyncio.to_thread(next, events, done)
                if item is done:
                    break
                mode, data = item
                if mode == "updates":
                    for node_name, patch in data.items():
                        await _handle_node_event(node_name, patch)
                elif mode == "values":
                    final_state = data
    except Exception as exc:
        err = str(exc)
        if type(exc).__name__ == "TaskCancelled":
            # on_stop already stopped owned children and reported them.
            update_task_status(task.task_id, "cancelled")
            await cl.Message(content=f"**Run cancelled.** `{err[:300]}`").send()
            return
        update_task_status(task.task_id, "failed")
        if "GPU busy" in err or "Refusing managed start" in err:
            await cl.Message(
                content=(
                    f"**Managed start refused — nothing was started or stopped.**\n\n`{err[:600]}`"
                )
            ).send()
            return
        # Surface a helpful message for the most common failure: vLLM not running
        if any(k in err for k in ("Connection refused", "vllm", "VLLM", "timed out", "OOM")):
            await cl.Message(
                content=(
                    "**Executor failed — vLLM server is not running.**\n\n"
                    "Start vLLM first:\n```bash\nbash scripts/start_vllm.sh\n```\n"
                    "Then retry your message. The planner output above was generated successfully."
                )
            ).send()
        else:
            await cl.Message(content=f"**Agent error:** `{err[:300]}`").send()
        return

    update_task_status(task.task_id, "completed")
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
            await cl.Message(
                content="**Planner:** no valid hypotheses generated (all rejected or already tried)."
            ).send()

    elif node_name == "executor":
        summaries = patch.get("experiment_summaries", [])
        recovery = patch.get("last_recovery") or {}
        if recovery.get("reason") and (
            "GPU busy" in recovery["reason"] or "Refusing managed start" in recovery["reason"]
        ):
            # Mutex / unknown-occupant refusal: say so plainly, never a silent skip.
            await cl.Message(
                content=(
                    f"⛔ **Executor refused to start `{recovery.get('experiment_id')}`** "
                    f"— nothing was started or stopped.\n\n`{recovery['reason'][:600]}`"
                )
            ).send()
            return
        if summaries:
            s = summaries[-1]
            if s.get("validity_status") == "failed" and s.get("throughput_rps") is None:
                await cl.Message(
                    content=(
                        f"❌ **Failed:** `{s['experiment_id']}` — "
                        f"{(s.get('failure_reason') or 'no reason recorded')[:400]}"
                    )
                ).send()
                return
            await cl.Message(content=_format_live_result_message(s)).send()

    elif node_name == "reflector":
        next_action = patch.get("next_action") or (
            "stop" if patch.get("should_stop") else "continue"
        )
        if patch.get("should_stop"):
            next_action = "stop"
        await cl.Message(
            content=format_reflector_update(
                next_action,
                stop_reason=str(patch.get("stop_reason") or ""),
                streak=int(patch.get("no_improvement_streak") or 0),
            )
        ).send()


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
        out = write_final_report(
            FinalReportInput(
                workload_name=workload_name,
                session_prefix=session_prefix,
                experiment_summaries=summaries,
                baseline_summary=baseline,
                best_summary=best,
                citations=_collect_citations(state),
                output_path=f"reports/{session_prefix}final_report.md",
                optimization_task=task_dump,
                stop_reason=str(state.get("stop_reason") or ""),
            )
        )
        report_path = out.output_path
    except Exception as exc:
        report_path = f"(failed to write report: {str(exc)[:160]})"

    lines = [
        "---",
        f"## Optimization Report — `{workload_name}`",
        "",
        f"**Session:** `{session_prefix}`  |  "
        f"**Experiments run:** {len(summaries)}  |  "
        f"**Wall clock:** {elapsed_s / 60:.1f} min",
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

    if summaries:
        lines += [
            "### All Experiments",
            "",
            *_format_all_experiments_table(summaries),
        ]

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
