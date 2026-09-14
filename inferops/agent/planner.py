"""Planner node — LLM-driven hypothesis generation with RAG grounding.

The planner:
  1. Queries the knowledge corpus for chunks relevant to the current bottleneck.
  2. Passes those chunks to the LLM as grounding context.
  3. Requires each hypothesis rationale to cite a numeric metric AND a [source:] tag.
  4. Rejects hypotheses without evidence or citations and retries once.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from inferops.agent.state import (
    AGENT_SEARCH_SPACE,
    WORKLOAD_DESCRIPTIONS,
    WORKLOAD_PRIMARY_METRIC,
    AgentState,
    Hypothesis,
    is_duplicate,
    model_name_of,
    task_of,
)
from inferops.citations import (
    DocumentRef,
    documents_from_context,
    sources_from_context,
    valid_structured_citations,
)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are an expert vLLM inference optimization engineer. Your task is to find the best \
serving configuration for a specific workload on an RTX 3060 Laptop (6 GB VRAM, WSL2).

You will be shown this-run experiment history, optional prior-session hints, \
relevant documentation excerpts, and asked to generate hypotheses for the next \
parameter change. Each hypothesis MUST:
  1. Cite an exact run_id, metric name, and metric value from this-run \
EXPERIMENT HISTORY only — never from PRIOR COMPATIBLE HISTORY.
  2. If knowledge chunks are provided, cite one of their sources and repeat it in \
the rationale as a [source: <document>] tag. If none are provided, do not invent one.
  3. Change exactly ONE parameter.
  4. Not repeat a (param, value) pair that has already been tried.
  5. Be consistent with the identified bottleneck type.

Respond with valid JSON only. No markdown, no prose outside the JSON.\
"""

_USER_TEMPLATE = """\
WORKLOAD: {workload_name}
DESCRIPTION: {workload_description}

PRIMARY METRIC (maximize): {primary_metric}
CURRENT BOTTLENECK: {current_bottleneck}
EXPERIMENTS REMAINING: {budget}

BASELINE:
  {baseline_line}

BEST SO FAR:
  {best_line}

EXPERIMENT HISTORY (most recent first):
{history_table}

{prior_history_section}ALREADY TRIED — do NOT repeat these (param, value) pairs:
{tried_pairs}

TUNABLE PARAMETERS (safe ranges for RTX 3060):
  max_num_batched_tokens  : {batched_values}  — tokens processed per scheduler step
  max_num_seqs            : {seqs_values}     — max concurrent sequences
  enable_chunked_prefill  : [true, false]     — interleave prefill with decode
  enable_prefix_caching   : [true, false]     — reuse KV for shared prompt prefixes

BOTTLENECK GUIDANCE:
  compute-bound     → try increasing max_num_batched_tokens (more GPU saturation)
  scheduling-bound  → try enable_chunked_prefill=true (reduce TTFT variance)
  memory-bound      → try reducing max_num_seqs (less KV pressure)
  kv-bound          → try enable_prefix_caching=true or reduce max_num_seqs

KNOWLEDGE CONTEXT (cite these in your rationale using [source: <source>]):
{knowledge_context}

Generate {n_hypotheses} hypothesis/hypotheses. Each rationale MUST include:
  - a specific metric value (e.g., "rps=15.0")
  - {source_rationale_requirement}
Each hypothesis MUST also include structured citations. The metric citation must use \
an exact run_id, metric field name, and value shown above. {document_requirement}

Respond with:
{{
  "analysis": "<one paragraph citing specific metric values and explaining the bottleneck>",
  "hypotheses": [
    {{
      "param": "...",
      "value": ...,
      "rationale": "{rationale_example}",
      "citations": {{
        "metric": {{"run_id": "...", "metric": "throughput_rps", "value": 0.0}}{document_example}
      }}
    }},
    ...
  ]
}}\
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_num(v: float | None, spec: str) -> str:
    if v is None:
        return "n/a"
    return format(v, spec)


def _fmt_vs_baseline(vs: float | None) -> str:
    """Render vs_baseline_pct at full precision so the printed value is citable as-is."""
    if vs is None:
        return "n/a"
    return f"{format(vs, '+')}%"


def _fmt_summary(s: dict | None) -> str:
    if s is None:
        return "(none yet)"
    vs_s = _fmt_vs_baseline(s.get("vs_baseline_pct"))
    return (
        f"experiment_id={s['experiment_id']}  run_id={s.get('run_id', '')}  "
        f"rps={_fmt_num(s.get('throughput_rps'), '')}  "
        f"ttft_p99={_fmt_num(s.get('ttft_p99_ms'), '')}ms  "
        f"e2e_p50={_fmt_num(s.get('e2e_p50_ms'), '')}ms  "
        f"bottleneck={s['bottleneck']}  "
        f"vs_baseline={vs_s}"
    )


def _build_history_table(summaries: list[dict]) -> str:
    if not summaries:
        return "  (no experiments yet)"
    lines = [
        "  run_id                 param_changed          value   throughput_rps  "
        "ttft_p99_ms  e2e_p50_ms  bottleneck      vs_baseline_pct"
    ]
    for s in reversed(summaries[-8:]):  # last 8, most recent first
        vs_s = _fmt_vs_baseline(s.get("vs_baseline_pct"))
        lines.append(
            f"  {str(s.get('run_id') or ''):<22} "
            f"{str(s.get('param_changed') or 'baseline'):<22} "
            f"{str(s.get('value_changed', '')):<7} "
            f"{_fmt_num(s.get('throughput_rps'), ''):<7} "
            f"{_fmt_num(s.get('ttft_p99_ms'), ''):<9} "
            f"{_fmt_num(s.get('e2e_p50_ms'), ''):<8} "
            f"{s['bottleneck']:<15} "
            f"{vs_s}"
        )
    return "\n".join(lines)


def _tried_pairs(summaries: list[dict], history_rows: list[dict] | None = None) -> str:
    pairs = [
        f"  {s['param_changed']}={s['value_changed']}" for s in summaries if s.get("param_changed")
    ]
    for row in history_rows or []:
        if row.get("param") is None:
            continue
        status = str(row.get("status") or "").lower()
        notes = str(row.get("notes") or "").lower()
        if status in {"failed", "invalid", "oom"} or "oom" in notes or "out of memory" in notes:
            pairs.append(
                f"  {row['param']}={row['value']}  (prior session failure — do not retry)"
            )
    return "\n".join(pairs) if pairs else "  (none)"


def _prior_history_section(rows: list[dict]) -> str:
    if not rows:
        return ""
    lines = [
        "PRIOR COMPATIBLE HISTORY (compatible = same model_name + workload_name "
        "+ hardware fingerprint [gpu_name, gpu_memory_total_gb, engine/vllm_version]. "
        "Unknown or mismatched hardware is excluded from ranking.):",
        "These run_ids are NOT this-run metric evidence; do not cite them as "
        "citations.metric.run_id. They must not skip a confirmation campaign.",
    ]
    for row in rows:
        lines.append(
            f"  run_id={row.get('run_id') or ''} session_id={row.get('session_id') or ''} "
            f"status={row.get('status') or ''} param={row.get('param')} value={row.get('value')} "
            f"throughput_rps={row.get('throughput_rps')} notes={row.get('notes') or ''} "
            f"claim_level={row.get('claim_level')}"
        )
    return "\n".join(lines) + "\n\n"


def _citation_summaries(state: AgentState) -> list[dict]:
    """Collect each summary visible in the prompt once, keyed by run_id."""
    candidates = [
        *state["experiment_summaries"],
        state.get("baseline_summary"),
        state.get("best_summary"),
    ]
    by_run_id: dict[str, dict] = {}
    for summary in candidates:
        if summary and summary.get("run_id"):
            by_run_id.setdefault(summary["run_id"], summary)
    return list(by_run_id.values())


def _validate_hypotheses(
    raw_hyps: list[dict],
    state: AgentState,
    available_sources: set[str] | None = None,
    available_documents: set[DocumentRef] | None = None,
) -> list[dict]:
    """Filter out invalid hypotheses (wrong param, out-of-range, already tried)."""
    available_sources = available_sources or set()
    available_documents = available_documents or set()
    citation_summaries = _citation_summaries(state)
    valid = []
    for h in raw_hyps:
        if not isinstance(h, dict):
            continue
        param = h.get("param", "")
        value = h.get("value")
        rationale = h.get("rationale", "")

        if not isinstance(rationale, str):
            continue
        if param not in AGENT_SEARCH_SPACE:
            continue
        allowed_vals = AGENT_SEARCH_SPACE[param]
        # Coerce bool params
        if isinstance(allowed_vals[0], bool):
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in ("true", "1", "yes"):
                    value = True
                elif lowered in ("false", "0", "no"):
                    value = False
                else:
                    continue
            else:
                value = bool(value)
        else:
            try:
                value = type(allowed_vals[0])(value)
            except (TypeError, ValueError):
                continue
        if value not in allowed_vals:
            continue
        if is_duplicate(state, param, value):
            continue
        # The prose still carries a human-readable number. Structured evidence
        # below is authoritative for run/metric/value/source existence.
        if not re.search(r"\d+(\.\d+)?", rationale):
            continue
        if not valid_structured_citations(
            h, citation_summaries, available_sources, available_documents
        ):
            continue
        h["param"] = param
        h["value"] = value
        valid.append(h)
    return valid


def _parse_llm_response(content: str) -> dict[str, Any]:
    """Extract JSON from LLM response; strip markdown fences if present."""
    content = content.strip()
    # Strip ```json ... ``` fences
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    return json.loads(content)


# ---------------------------------------------------------------------------
# RAG helpers
# ---------------------------------------------------------------------------


def _retrieve_knowledge(bottleneck: str, workload: str, top_k: int = 4) -> str:
    """
    Query the corpus for chunks relevant to the current bottleneck + workload.
    Falls back to an empty string if the index has not been built.
    """
    try:
        from inferops.tools.knowledge_retriever import (
            KnowledgeRetrieverInput,
            knowledge_retriever,
        )

        query = f"{bottleneck} optimization {workload} vLLM"
        result = knowledge_retriever(KnowledgeRetrieverInput(query=query, top_k=top_k))
        if result.index_incompatible:
            msg = result.message or (
                "knowledge index version is incompatible — rebuild with scripts/build_corpus.py"
            )
            return f"({msg})"
        if result.index_empty or not result.chunks:
            return "(knowledge index not built — run scripts/build_corpus.py)"
        lines = []
        for c in result.chunks:
            lines.append(
                f"[source: {c.source}] §{c.section}\n"
                f"chunk_id={c.chunk_id} version={c.version}\n"
                f"{c.text[:300]}…"
            )
        return "\n\n".join(lines)
    except Exception:
        return "(knowledge retrieval unavailable)"


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def planner_node(state: AgentState, llm) -> dict:
    """
    Generate 1–3 hypotheses using the LLM, with RAG-grounded evidence citation.
    Returns a state patch with updated hypotheses and messages.
    """
    # How many hypotheses to request (fewer when budget is tight)
    n = (
        1
        if state["experiments_remaining"] <= 2
        else (2 if state["experiments_remaining"] <= 4 else 3)
    )

    history_rows: list[dict] = []
    db_path = state.get("memory_db_path")
    model_name = model_name_of(state)
    if db_path and model_name:
        from inferops.memory.hardware import (
            collect_hardware_info,
            fingerprint_from_hardware,
        )
        from inferops.memory.history import query_compatible_history

        # Prefer the run-start fingerprint (captured with probe_nvidia=True).
        current_fp = fingerprint_from_hardware(state.get("hardware_fingerprint"))
        if current_fp is None:
            # Resume / legacy checkpoint without a stored fingerprint: production
            # may probe nvidia-smi once. Tests inject state or monkeypatch collect.
            task = task_of(state)
            engine = task.engine.value if task is not None else "vllm"
            current_fp = fingerprint_from_hardware(
                collect_hardware_info(
                    model_name=model_name,
                    engine=engine,
                    probe_nvidia=True,
                )
            )
        history_rows = query_compatible_history(
            model_name=model_name,
            workload_name=state["workload_name"],
            exclude_session_id=state["session_prefix"],
            db_path=db_path,
            current_fingerprint=current_fp,
        )
    state = {**state, "compatible_history": history_rows}

    knowledge_context = _retrieve_knowledge(
        bottleneck=state["current_bottleneck"],
        workload=state["workload_name"],
    )
    available_sources = sources_from_context(knowledge_context)
    available_documents = documents_from_context(knowledge_context)
    if available_sources:
        source_rationale_requirement = "a [source: <source>] tag from the knowledge context above"
        document_requirement = (
            "The document chunk_id, source, and version must exactly match one chunk shown "
            "in KNOWLEDGE CONTEXT."
        )
        document_example = (
            ',\n        "document": {"chunk_id": "<chunk>", "source": "<doc>", '
            '"version": "<version>"}'
        )
        rationale_example = "... metric=X.Y ... [source: <doc>] ..."
    else:
        source_rationale_requirement = (
            "no [source:] tag, because KNOWLEDGE CONTEXT has no retrieved sources"
        )
        document_requirement = (
            "Omit the document citation because KNOWLEDGE CONTEXT has no retrieved sources."
        )
        document_example = ""
        rationale_example = "... metric=X.Y ..."

    user_msg = _USER_TEMPLATE.format(
        workload_name=state["workload_name"],
        workload_description=WORKLOAD_DESCRIPTIONS.get(state["workload_name"], ""),
        primary_metric=WORKLOAD_PRIMARY_METRIC.get(state["workload_name"], "throughput_rps"),
        current_bottleneck=state["current_bottleneck"],
        budget=state["experiments_remaining"],
        baseline_line=_fmt_summary(state["baseline_summary"]),
        best_line=_fmt_summary(state["best_summary"]),
        history_table=_build_history_table(state["experiment_summaries"]),
        prior_history_section=_prior_history_section(history_rows),
        tried_pairs=_tried_pairs(state["experiment_summaries"], history_rows),
        batched_values=str(AGENT_SEARCH_SPACE["max_num_batched_tokens"]),
        seqs_values=str(AGENT_SEARCH_SPACE["max_num_seqs"]),
        knowledge_context=knowledge_context,
        source_rationale_requirement=source_rationale_requirement,
        document_requirement=document_requirement,
        document_example=document_example,
        rationale_example=rationale_example,
        n_hypotheses=n,
    )

    messages_in = [SystemMessage(content=_SYSTEM), HumanMessage(content=user_msg)]
    response = llm.invoke(messages_in)

    # Token tracking
    tokens_used = 0
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        tokens_used = response.usage_metadata.get("input_tokens", 0) + response.usage_metadata.get(
            "output_tokens", 0
        )

    # Parse and validate response. Invalid/forged evidence gets the same single
    # retry as malformed JSON.
    raw_hyps: list[dict] = []
    valid: list[dict] = []
    analysis = ""
    needs_retry = False
    try:
        data = _parse_llm_response(response.content)
        raw_hyps = data.get("hypotheses", [])
        if not isinstance(raw_hyps, list):
            raise ValueError("hypotheses must be a list")
        analysis = data.get("analysis", "")
        valid = _validate_hypotheses(
            raw_hyps, state, available_sources, available_documents
        )
        needs_retry = bool(raw_hyps) and len(valid) != len(raw_hyps)
    except (json.JSONDecodeError, ValueError):
        needs_retry = True

    if needs_retry:
        retry_msg = HumanMessage(
            content=(
                "Your response was invalid JSON or contained a hypothesis with unverifiable "
                "citations. Retry once with ONLY the JSON object. Use the structured citation "
                "shape requested above; cite only an exact run_id/metric/value from the shown "
                "summaries. Cite a document only when KNOWLEDGE CONTEXT provides sources."
            )
        )
        retry_response = llm.invoke(messages_in + [response, retry_msg])
        tokens_used += (
            (
                retry_response.usage_metadata.get("input_tokens", 0)
                + retry_response.usage_metadata.get("output_tokens", 0)
            )
            if hasattr(retry_response, "usage_metadata") and retry_response.usage_metadata
            else 0
        )
        try:
            data = _parse_llm_response(retry_response.content)
            raw_hyps = data.get("hypotheses", [])
            if not isinstance(raw_hyps, list):
                raise ValueError("hypotheses must be a list")
            analysis = data.get("analysis", "")
            response = retry_response
            valid = _validate_hypotheses(
                raw_hyps, state, available_sources, available_documents
            )
        except (json.JSONDecodeError, ValueError):
            raw_hyps = []
            valid = []

    # Convert to Hypothesis TypedDicts
    new_hypotheses: list[Hypothesis] = [
        Hypothesis(
            id=f"h{len(state['hypotheses']) + i + 1}",
            param=h["param"],
            value=h["value"],
            rationale=h["rationale"],
            status="pending",
            experiment_id=None,
        )
        for i, h in enumerate(valid)
    ]

    # Add planner step to trajectory
    trajectory_step = {
        "step": len(state["trajectory"]) + 1,
        "node": "planner",
        "workload": state["workload_name"],
        "action": f"generated {len(new_hypotheses)} hypothesis/hypotheses",
        "reasoning": analysis,
        "hypotheses": [{"param": h["param"], "value": h["value"]} for h in new_hypotheses],
        "tokens_used": tokens_used,
    }

    return {
        "hypotheses": state["hypotheses"] + new_hypotheses,
        "trajectory": state["trajectory"] + [trajectory_step],
        "messages": [AIMessage(content=response.content)],
        "compatible_history": history_rows,
    }
