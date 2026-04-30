"""LLM-as-judge: score an agent's decision trajectory on a fixed rubric."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Rubric
# ---------------------------------------------------------------------------

RUBRIC: dict[str, str] = {
    "evidence_based": (
        "Did the agent cite specific metric values (e.g. TTFT p99=120ms, throughput=14.5 rps) "
        "before each parameter decision? Score 0 if decisions appear arbitrary, "
        "1 if consistently grounded in measured evidence."
    ),
    "no_repeat": (
        "Did the agent avoid re-running configurations already present in experiment memory? "
        "Score 0 if it duplicated known configs, 1 if it always checked memory first."
    ),
    "replan": (
        "When the bottleneck classification changed (e.g. from scheduling-bound to "
        "compute-bound), did the agent explicitly adjust its optimization strategy? "
        "Score 0 if it ignored the shift, 1 if it replanned accordingly."
    ),
    "efficient": (
        "Did the agent converge to a good configuration without wasting runs on clearly "
        "suboptimal configs? Score 0 if exploration was random or exhaustive, "
        "1 if each run was targeted and informed by prior evidence."
    ),
}

# Weighted average weights (must sum to 1.0)
WEIGHTS: dict[str, float] = {
    "evidence_based": 0.30,
    "no_repeat":      0.25,
    "replan":         0.25,
    "efficient":      0.20,
}


FEW_SHOT_EXAMPLES: list[dict[str, Any]] = [
    {
        "name": "evidence-driven replanning",
        "trajectory": [
            {
                "step": 1,
                "action": "run_benchmark(max_num_batched_tokens=4096)",
                "reasoning": "baseline throughput_rps=14.9 and GPU util=91% suggest compute-bound.",
                "result": {
                    "throughput_rps": 17.2,
                    "ttft_p99_ms": 65,
                    "bottleneck": "compute-bound",
                },
            },
            {
                "step": 2,
                "action": "reflect",
                "reasoning": (
                    "bottleneck changed to scheduling-bound; "
                    "replan around TTFT p99=175ms."
                ),
            },
        ],
        "score": {
            "evidence_based": 1.0,
            "no_repeat": 1.0,
            "replan": 1.0,
            "efficient": 0.9,
        },
    },
    {
        "name": "random duplicate search",
        "trajectory": [
            {"step": 1, "action": "run_benchmark(random)", "reasoning": ""},
            {"step": 2, "action": "run_benchmark(random)", "reasoning": ""},
            {"step": 3, "action": "run_benchmark(random)", "reasoning": ""},
        ],
        "score": {
            "evidence_based": 0.0,
            "no_repeat": 0.4,
            "replan": 0.0,
            "efficient": 0.2,
        },
    },
]


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class JudgeScore:
    evidence_based: float   # 0–1
    no_repeat: float        # 0–1
    replan: float           # 0–1
    efficient: float        # 0–1
    overall: float          # weighted average
    reasoning: str


@dataclass
class ConsistencyResult:
    trials: int
    scores: list[float]
    max_delta: float
    consistent: bool


# ---------------------------------------------------------------------------
# Judge function
# ---------------------------------------------------------------------------

def judge_trajectory(
    trajectory: list[dict[str, Any]],
    llm=None,
) -> JudgeScore:
    """
    Score an agent trajectory using an LLM judge.

    trajectory: list of step dicts, each containing:
        step (int), action (str), reasoning (str), result (dict), workload (str)

    llm: a LangChain ChatModel instance. If None, falls back to a heuristic scorer.
         Recommended: ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
    """
    if llm is None or not trajectory:
        return _heuristic_judge(trajectory)

    from langchain_core.messages import HumanMessage, SystemMessage

    rubric_text = "\n".join(f"- {k} (weight={WEIGHTS[k]}): {v}" for k, v in RUBRIC.items())
    traj_text = _format_trajectory(trajectory)

    prompt = f"""\
You are evaluating the decision quality of an autonomous LLM inference optimization agent.

The agent's task: find the best vLLM serving configuration for a given workload by running
benchmarks, analyzing results, and iteratively proposing parameter changes.

RUBRIC (score each criterion 0.0–1.0):
{rubric_text}

FEW-SHOT CALIBRATION EXAMPLES:
{_format_few_shot_examples()}

AGENT TRAJECTORY:
{traj_text}

Respond with a JSON object only — no markdown fences, no extra text:
{{
  "evidence_based": <float 0-1>,
  "no_repeat": <float 0-1>,
  "replan": <float 0-1>,
  "efficient": <float 0-1>,
  "reasoning": "<one concise paragraph explaining the scores>"
}}"""

    messages = [
        SystemMessage(content="You are a rigorous ML systems evaluator. Be critical and precise."),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)
    raw = response.content.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(m.group()) if m else {}

    scores = {k: max(0.0, min(1.0, float(data.get(k, 0.5)))) for k in RUBRIC}
    overall = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)

    return JudgeScore(
        **scores,
        overall=round(overall, 4),
        reasoning=data.get("reasoning", ""),
    )


def judge_consistency(
    trajectory: list[dict[str, Any]],
    llm,
    trials: int = 3,
    tolerance: float = 0.05,
) -> ConsistencyResult:
    """Run the same judge call several times and report score stability."""
    if trials < 2:
        raise ValueError("trials must be >= 2")
    scores = [judge_trajectory(trajectory, llm=llm).overall for _ in range(trials)]
    max_delta = max(scores) - min(scores)
    return ConsistencyResult(
        trials=trials,
        scores=scores,
        max_delta=round(max_delta, 4),
        consistent=max_delta <= tolerance,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_trajectory(trajectory: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for step in trajectory:
        lines.append(f"Step {step.get('step', '?')}: {step.get('action', '(no action)')}")
        if step.get("reasoning"):
            lines.append(f"  Reasoning: {step['reasoning']}")
        result = step.get("result")
        if isinstance(result, dict):
            key_fields = ("throughput_rps", "ttft_p99_ms", "bottleneck", "winner", "gap_pct")
            summary = {k: result[k] for k in key_fields if k in result}
            if summary:
                lines.append(f"  Result: {summary}")
        lines.append("")
    return "\n".join(lines)


def _format_few_shot_examples() -> str:
    lines: list[str] = []
    for ex in FEW_SHOT_EXAMPLES:
        lines.append(f"Example: {ex['name']}")
        lines.append(_format_trajectory(ex["trajectory"]).rstrip())
        lines.append(f"Expected score: {json.dumps(ex['score'], sort_keys=True)}")
        lines.append("")
    return "\n".join(lines)


def _heuristic_judge(trajectory: list[dict[str, Any]]) -> JudgeScore:
    """
    Rule-based fallback when no LLM is configured.

    Checks three observable signals from the trajectory dict keys:
      - evidence_based: does each decision cite a numeric metric in "reasoning"?
      - no_repeat: are all action experiment_ids unique?
      - replan: does any step mention "replan" or "bottleneck changed" in reasoning?
      - efficient: did it take ≤6 steps to reach a decision?
    """
    if not trajectory:
        return JudgeScore(0.5, 0.5, 0.5, 0.5, 0.5, "Empty trajectory.")

    evidence_based = sum(
        1 for s in trajectory
        if re.search(r"\d+(\.\d+)?", str(s.get("reasoning", "")))
    ) / len(trajectory)

    seen_ids: set[str] = set()
    duplicates = 0
    for step in trajectory:
        eid = step.get("experiment_id") or step.get("action", "")
        if eid in seen_ids:
            duplicates += 1
        seen_ids.add(eid)
    no_repeat = max(0.0, 1.0 - duplicates / len(trajectory))

    replan_keywords = ("replan", "bottleneck changed", "switch strategy", "adjust")
    replan = float(any(
        any(kw in str(s.get("reasoning", "")).lower() for kw in replan_keywords)
        for s in trajectory
    ))

    efficient = max(0.0, 1.0 - max(0, len(trajectory) - 6) / 6)

    scores = {
        "evidence_based": round(evidence_based, 2),
        "no_repeat":      round(no_repeat, 2),
        "replan":         round(replan, 2),
        "efficient":      round(efficient, 2),
    }
    overall = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)

    return JudgeScore(
        **scores,
        overall=round(overall, 4),
        reasoning=(
            "Heuristic fallback — pass a LangChain ChatModel to judge_trajectory() "
            "for LLM scoring."
        ),
    )
