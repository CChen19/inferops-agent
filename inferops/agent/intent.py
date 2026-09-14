"""Intent extraction: natural language → draft fields for OptimizationTask.

The LLM only drafts. ``build_optimization_task`` validates. Unknown workloads
are left as-is so validation can ask for clarification instead of silently
substituting ``chat_short``.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, fields

from inferops.task import OptimizationTask, build_optimization_task
from workloads.definitions import ALL_WORKLOADS

_WORKLOAD_NAMES = [w.name for w in ALL_WORKLOADS]

_INTENT_SYSTEM = """\
You are a vLLM configuration assistant. Extract the user's benchmarking intent and \
return a JSON object with these fields (all optional):

{
  "workload_name": "<one of: chat_short, long_context_qa, high_concurrency_short_out, long_generation, mixed_traffic, or null if unknown>",
  "model_hint": "<model name if mentioned, e.g. Qwen2.5-1.5B>",
  "target_qps": <float or null — measured throughput GOAL, not offered arrival rate>,
  "gpu_hint": "<GPU description if mentioned>",
  "budget": <int number of experiments, or null>,
  "max_ttft_ms": <float or null — hard TTFT p99 upper bound>,
  "max_e2e_ms": <float or null — hard E2E p50 upper bound>,
  "max_error_rate": <float or null>,
  "concurrency": <int or null>,
  "time_limit_s": <float or null>,
  "service_mode": "<managed|external or null>",
  "notes": "<any extra context>"
}

Rules:
- Map scenario descriptions to workload_name:
    chat / QA / short / conversational → chat_short
    long context / document / 1024-token → long_context_qa
    high concurrency / burst / many users → high_concurrency_short_out
    long generation / creative / 512-token output → long_generation
    mixed / varied traffic → mixed_traffic
- If the user names a workload that is not in the list, keep their string.
  Do NOT invent chat_short for an unknown named workload.
- If no scenario is mentioned, omit workload_name (null).
- target_qps is what they hope to MEASURE, not the offered send rate.
- Respond with ONLY the JSON object, no prose.\
"""


@dataclass
class Intent:
    workload_name: str | None
    model_hint: str
    target_qps: float | None
    gpu_hint: str
    budget: int | None
    notes: str
    max_ttft_ms: float | None = None
    max_e2e_ms: float | None = None
    max_error_rate: float | None = None
    concurrency: int | None = None
    time_limit_s: float | None = None
    service_mode: str | None = None
    parse_ok: bool = True
    user_message: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_intent(user_message: str, llm) -> Intent:
    """Parse a natural language message into a structured Intent draft."""
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=_INTENT_SYSTEM),
        HumanMessage(content=user_message),
    ]
    response = llm.invoke(messages)
    content = response.content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)

    try:
        data = json.loads(content)
        parse_ok = True
    except json.JSONDecodeError:
        data = {}
        parse_ok = False

    workload = data.get("workload_name")
    if workload == "":
        workload = None
    # Keep unknown names so validation can reject/clarify. Do not remap here.
    if workload is not None:
        workload = str(workload)

    return Intent(
        workload_name=workload,
        model_hint=str(data.get("model_hint") or ""),
        target_qps=_optional_float(data.get("target_qps")),
        gpu_hint=str(data.get("gpu_hint") or ""),
        budget=_optional_int(data.get("budget")),
        notes=str(data.get("notes") or ""),
        max_ttft_ms=_optional_float(data.get("max_ttft_ms")),
        max_e2e_ms=_optional_float(data.get("max_e2e_ms")),
        max_error_rate=_optional_float(data.get("max_error_rate")),
        concurrency=_optional_int(data.get("concurrency")),
        time_limit_s=_optional_float(data.get("time_limit_s")),
        service_mode=(str(data["service_mode"]) if data.get("service_mode") else None),
        parse_ok=parse_ok,
        user_message=user_message,
    )


def merge_intents(previous: Intent | None, incoming: Intent) -> Intent:
    """Follow-up messages fill empty fields; incoming non-empty values win."""
    if previous is None:
        return incoming
    merged = Intent(**asdict(previous))
    for f in fields(Intent):
        if f.name in {"parse_ok", "user_message"}:
            continue
        new_val = getattr(incoming, f.name)
        if new_val not in (None, ""):
            setattr(merged, f.name, new_val)
    merged.parse_ok = incoming.parse_ok or previous.parse_ok
    if incoming.user_message:
        merged.user_message = (
            f"{previous.user_message}\n{incoming.user_message}".strip()
            if previous.user_message
            else incoming.user_message
        )
    return merged


def task_from_intent(intent: Intent) -> OptimizationTask:
    return build_optimization_task(
        workload_name=intent.workload_name,
        model_hint=intent.model_hint or None,
        target_qps=intent.target_qps,
        gpu_hint=intent.gpu_hint or None,
        budget=intent.budget,
        max_ttft_ms=intent.max_ttft_ms,
        max_e2e_ms=intent.max_e2e_ms,
        max_error_rate=intent.max_error_rate,
        concurrency=intent.concurrency,
        time_limit_s=intent.time_limit_s,
        service_mode=intent.service_mode,
        user_message=intent.user_message,
        parse_ok=intent.parse_ok,
    )


def interpret_user_request(
    user_message: str,
    llm,
    previous: Intent | None = None,
) -> tuple[Intent, OptimizationTask]:
    """Draft + validate. Does not spend GPU budget."""
    extracted = extract_intent(user_message, llm)
    merged = merge_intents(previous, extracted)
    return merged, task_from_intent(merged)


# Kept so older imports do not break.
SUPPORTED_WORKLOADS = _WORKLOAD_NAMES
