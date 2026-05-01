"""Intent extraction: natural language → WorkloadSpec fields for the Chainlit UI."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from workloads.definitions import ALL_WORKLOADS

_WORKLOAD_NAMES = [w.name for w in ALL_WORKLOADS]

_INTENT_SYSTEM = """\
You are a vLLM configuration assistant. Extract the user's benchmarking intent and \
return a JSON object with these fields (all optional except workload_name):

{
  "workload_name": "<one of: chat_short, long_context_qa, high_concurrency_short_out, long_generation, mixed_traffic>",
  "model_hint": "<model name if mentioned, e.g. Qwen2.5-1.5B>",
  "target_qps": <float or null>,
  "gpu_hint": "<GPU description if mentioned>",
  "budget": <int number of experiments, default 6>,
  "notes": "<any extra context>"
}

Rules:
- Map scenario descriptions to workload_name:
    chat / QA / short / conversational → chat_short
    long context / document / 1024-token → long_context_qa
    high concurrency / burst / many users → high_concurrency_short_out
    long generation / creative / 512-token output → long_generation
    mixed / varied traffic → mixed_traffic
- If uncertain, default to chat_short.
- Respond with ONLY the JSON object, no prose.\
"""


@dataclass
class Intent:
    workload_name: str
    model_hint: str
    target_qps: float | None
    gpu_hint: str
    budget: int
    notes: str


def extract_intent(user_message: str, llm) -> Intent:
    """Parse a natural language message into a structured Intent."""
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
    except json.JSONDecodeError:
        data = {}

    workload = data.get("workload_name", "chat_short")
    if workload not in _WORKLOAD_NAMES:
        workload = "chat_short"

    return Intent(
        workload_name=workload,
        model_hint=data.get("model_hint", ""),
        target_qps=data.get("target_qps"),
        gpu_hint=data.get("gpu_hint", ""),
        budget=int(data.get("budget") or 6),
        notes=data.get("notes", ""),
    )
