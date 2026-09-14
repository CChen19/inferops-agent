"""Thin optimization-task contract.

Natural language only drafts a task. Code validates. GPU budget is spent only
after the task is confirmed. Unsupported models / workloads are rejected or
sent back for clarification — never silently replaced.

Reuses ``WorkloadSpec`` and existing metric field names. Does not invent a
new metrics schema.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from inferops.eval.metrics import WORKLOAD_PRIMARY_METRIC
from inferops.resume import format_resume_help
from inferops.schemas import InferenceEngine, WorkloadSpec, compute_workload_hash
from workloads.definitions import ALL_WORKLOADS

# ---------------------------------------------------------------------------
# Catalog — what this single-machine assistant can actually run
# ---------------------------------------------------------------------------

SUPPORTED_MODELS: dict[str, dict[str, Any]] = {
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "size": "0.5B",
        "aliases": (
            "qwen/qwen2.5-0.5b-instruct",
            "qwen2.5-0.5b-instruct",
            "qwen2.5-0.5b",
            "qwen 0.5b",
            "qwen-0.5b",
            "0.5b",
        ),
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "size": "1.5B",
        "aliases": (
            "qwen/qwen2.5-1.5b-instruct",
            "qwen2.5-1.5b-instruct",
            "qwen2.5-1.5b",
            "qwen 1.5b",
            "qwen-1.5b",
            "1.5b",
        ),
    },
}

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_WORKLOAD_NAME = "chat_short"
DEFAULT_BUDGET = 6
MAX_BUDGET = 20
MIN_BUDGET = 1

# Constraint metrics must already exist on ExperimentSummary / ExperimentResult.
ALLOWED_CONSTRAINT_METRICS: frozenset[str] = frozenset(
    {
        "error_rate",
        "ttft_p50_ms",
        "ttft_p99_ms",
        "e2e_p50_ms",
        "throughput_rps",
        "tokens_per_second",
    }
)

DEFAULT_MAX_ERROR_RATE = 0.05

TRAFFIC_ARRIVAL_RATE_NOTE = (
    "Offered arrival-rate scheduling is not implemented. Load uses "
    "concurrency-limited asyncio.gather. target_qps is a measured-throughput "
    "goal, not an offered request-arrival rate."
)

_WORKLOAD_BY_NAME = {w.name: w for w in ALL_WORKLOADS}


class TaskStatus(str, Enum):
    READY = "ready"
    CONFIRMED = "confirmed"
    NEEDS_CLARIFICATION = "needs_clarification"
    REJECTED = "rejected"


class ServiceControlMode(str, Enum):
    MANAGED = "managed"
    EXTERNAL = "external"


class ConstraintOp(str, Enum):
    LE = "<="
    GE = ">="
    LT = "<"
    GT = ">"


class MetricConstraint(BaseModel):
    """One SLO / hard constraint on an existing metric field."""

    metric: str
    op: ConstraintOp = ConstraintOp.LE
    value: float
    source: Literal["user", "default"] = "default"


class OptimizationTask(BaseModel):
    """Confirmed-or-draft tuning task shared by UI, executor, and report."""

    task_id: str
    status: TaskStatus = TaskStatus.READY

    model_name: str
    gpu_hint: str | None = None
    engine: InferenceEngine = InferenceEngine.VLLM

    workload: WorkloadSpec
    primary_metric: str
    primary_direction: Literal["max", "min"] = "max"

    constraints: list[MetricConstraint] = Field(default_factory=list)
    target_qps: float | None = None

    experiment_budget: int = Field(ge=1)
    time_limit_s: float | None = None
    service_mode: ServiceControlMode = ServiceControlMode.MANAGED

    clarification_needed: list[str] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    user_message: str = ""

    @property
    def traffic_arrival_rate_supported(self) -> bool:
        return False

    def is_executable(self) -> bool:
        return self.status == TaskStatus.CONFIRMED and not self.rejection_reasons


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def normalize_model_hint(hint: str | None) -> str:
    return " ".join((hint or "").strip().lower().replace("_", "-").split())


def resolve_model_name(hint: str | None) -> tuple[str | None, str]:
    """Return (canonical_name | None, reason).

    reason is one of: matched, default, ambiguous, unsupported, empty.
    """
    raw = (hint or "").strip()
    if not raw:
        return None, "empty"

    key = normalize_model_hint(raw)
    if key in {"qwen", "qwen2.5", "qwen2"}:
        return None, "ambiguous"

    for canonical, meta in SUPPORTED_MODELS.items():
        aliases = {normalize_model_hint(canonical), *meta["aliases"]}
        if key in aliases:
            return canonical, "matched"
        size = str(meta["size"]).lower()
        if key.endswith(size) and "qwen" in key:
            return canonical, "matched"

    return None, "unsupported"


def resolve_workload_name(name: str | None) -> tuple[str | None, str]:
    """Return (workload_name | None, reason).

    reason: matched, default, unknown, empty.
    """
    raw = (name or "").strip()
    if not raw:
        return None, "empty"
    if raw in _WORKLOAD_BY_NAME:
        return raw, "matched"
    return raw, "unknown"


def workload_copy(name: str) -> WorkloadSpec:
    return _WORKLOAD_BY_NAME[name].model_copy()


def default_error_constraint(max_error_rate: float = DEFAULT_MAX_ERROR_RATE) -> MetricConstraint:
    return MetricConstraint(
        metric="error_rate",
        op=ConstraintOp.LE,
        value=max_error_rate,
        source="default",
    )


# ---------------------------------------------------------------------------
# Build / validate
# ---------------------------------------------------------------------------

def _explicit_positive_int(name: str, value: Any, *, lo: int, hi: int) -> int:
    """Require a real int in [lo, hi]. Reject bools and non-integer numbers."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{name} must be an explicit integer in [{lo}, {hi}] "
            f"(got {type(value).__name__}={value!r})"
        )
    if value < lo or value > hi:
        raise ValueError(f"{name}={value} is outside the supported range [{lo}, {hi}]")
    return value


def _max_model_len_ceiling() -> int:
    """Reuse run_benchmark safe max_model_len upper bound (not a new magic number)."""
    from inferops.tools.run_benchmark import _SAFE_RANGES

    return int(_SAFE_RANGES["max_model_len"][1])


def _default_engine_max_model_len() -> int:
    """ExperimentConfig.max_model_len default (same as make_configs)."""
    from inferops.schemas import ExperimentConfig

    return int(ExperimentConfig.model_fields["max_model_len"].default)


def _apply_workload_overrides(
    workload: WorkloadSpec,
    overrides: dict[str, Any],
    *,
    max_model_len: int | None = None,
) -> WorkloadSpec:
    update: dict[str, Any] = {}
    if overrides.get("concurrency") is not None:
        conc = int(overrides["concurrency"])
        if conc < 1 or conc > 64:
            raise ValueError(f"concurrency={conc} is outside the supported range [1, 64]")
        update["concurrency"] = conc
    if overrides.get("num_requests") is not None:
        n = int(overrides["num_requests"])
        if n < 1 or n > 500:
            raise ValueError(f"num_requests={n} is outside the supported range [1, 500]")
        update["num_requests"] = n
    # WorkloadSpec.ge=1; upper bound = run_benchmark safe max_model_len max (4096).
    len_lo = 1
    len_hi = _max_model_len_ceiling()
    if overrides.get("input_len") is not None:
        update["input_len"] = _explicit_positive_int(
            "input_len", overrides["input_len"], lo=len_lo, hi=len_hi
        )
    if overrides.get("output_len") is not None:
        update["output_len"] = _explicit_positive_int(
            "output_len", overrides["output_len"], lo=len_lo, hi=len_hi
        )
    # Arrival rate is recorded on WorkloadSpec but is NOT a scheduler today.
    if overrides.get("offered_rps") is not None:
        update["rps"] = float(overrides["offered_rps"])
    if not update:
        return workload
    merged = workload.model_copy(update=update)
    engine_cap = max_model_len if max_model_len is not None else _default_engine_max_model_len()
    total = merged.input_len + merged.output_len
    if total > engine_cap:
        raise ValueError(
            f"input_len+output_len={total} exceeds engine max_model_len={engine_cap}"
        )
    return merged


def build_optimization_task(
    *,
    workload_name: str | None,
    model_hint: str | None = None,
    target_qps: float | None = None,
    gpu_hint: str | None = None,
    budget: int | None = None,
    max_ttft_ms: float | None = None,
    max_e2e_ms: float | None = None,
    max_error_rate: float | None = None,
    concurrency: int | None = None,
    num_requests: int | None = None,
    input_len: int | None = None,
    output_len: int | None = None,
    offered_rps: float | None = None,
    time_limit_s: float | None = None,
    service_mode: str | None = None,
    user_message: str = "",
    parse_ok: bool = True,
    task_id: str | None = None,
) -> OptimizationTask:
    """Validate a draft. Never silently substitute an unsupported model/workload."""

    clarifications: list[str] = []
    rejections: list[str] = []
    warnings: list[str] = []

    if not parse_ok:
        clarifications.append(
            "Could not parse a structured request. Please name a supported "
            f"workload ({', '.join(sorted(_WORKLOAD_BY_NAME))}) and, if not "
            "using the default, a supported model (Qwen2.5 0.5B or 1.5B)."
        )

    wl_name, wl_reason = resolve_workload_name(workload_name)
    workload: WorkloadSpec | None = None
    if wl_reason == "matched" and wl_name is not None:
        workload = workload_copy(wl_name)
    elif wl_reason == "empty":
        workload = workload_copy(DEFAULT_WORKLOAD_NAME)
        warnings.append(
            f"No workload was specified; assuming `{DEFAULT_WORKLOAD_NAME}`. "
            "Confirm or name a different supported workload."
        )
    else:
        clarifications.append(
            f"Workload `{workload_name}` is not supported. Choose one of: "
            f"{', '.join(sorted(_WORKLOAD_BY_NAME))}."
        )
        workload = workload_copy(DEFAULT_WORKLOAD_NAME)

    try:
        workload = _apply_workload_overrides(
            workload,
            {
                "concurrency": concurrency,
                "num_requests": num_requests,
                "input_len": input_len,
                "output_len": output_len,
                "offered_rps": offered_rps,
            },
        )
    except ValueError as exc:
        clarifications.append(str(exc))

    model_name, model_reason = resolve_model_name(model_hint)
    if model_reason == "matched" and model_name is not None:
        resolved_model = model_name
    elif model_reason == "empty":
        resolved_model = DEFAULT_MODEL_NAME
        warnings.append(
            f"No model was specified; assuming `{DEFAULT_MODEL_NAME}`. "
            "Unsupported models will not be substituted automatically."
        )
    elif model_reason == "ambiguous":
        clarifications.append(
            f"Model hint `{model_hint}` is ambiguous. Specify "
            "`Qwen2.5-0.5B` or `Qwen2.5-1.5B`."
        )
        resolved_model = DEFAULT_MODEL_NAME
    else:
        rejections.append(
            f"Model `{model_hint}` is not supported on this single-machine "
            "assistant. Supported models: "
            + ", ".join(SUPPORTED_MODELS)
            + ". The run was not remapped to a different model."
        )
        resolved_model = DEFAULT_MODEL_NAME

    if budget is None:
        resolved_budget = DEFAULT_BUDGET
        warnings.append(f"Experiment budget defaulted to {DEFAULT_BUDGET}.")
    else:
        resolved_budget = int(budget)
        if resolved_budget < MIN_BUDGET:
            clarifications.append(
                f"Experiment budget must be at least {MIN_BUDGET}."
            )
            resolved_budget = DEFAULT_BUDGET
        elif resolved_budget > MAX_BUDGET:
            warnings.append(
                f"Budget {resolved_budget} exceeds the cap of {MAX_BUDGET}; "
                f"using {MAX_BUDGET}."
            )
            resolved_budget = MAX_BUDGET

    mode = ServiceControlMode.MANAGED
    if service_mode:
        raw_mode = service_mode.strip().lower()
        if raw_mode in {"managed", "external"}:
            mode = ServiceControlMode(raw_mode)
        else:
            clarifications.append(
                f"Unknown service_mode `{service_mode}`. Use managed or external."
            )

    constraints: list[MetricConstraint] = [
        default_error_constraint(
            float(max_error_rate) if max_error_rate is not None else DEFAULT_MAX_ERROR_RATE
        )
    ]
    if max_error_rate is not None:
        constraints[0].source = "user"
    if max_ttft_ms is not None:
        constraints.append(
            MetricConstraint(
                metric="ttft_p99_ms",
                op=ConstraintOp.LE,
                value=float(max_ttft_ms),
                source="user",
            )
        )
    if max_e2e_ms is not None:
        constraints.append(
            MetricConstraint(
                metric="e2e_p50_ms",
                op=ConstraintOp.LE,
                value=float(max_e2e_ms),
                source="user",
            )
        )

    if target_qps is not None and target_qps <= 0:
        clarifications.append("target_qps must be a positive measured-throughput goal.")
        target_qps = None

    if offered_rps is not None:
        warnings.append(TRAFFIC_ARRIVAL_RATE_NOTE)

    if time_limit_s is not None and time_limit_s <= 0:
        clarifications.append("time_limit_s must be positive if set.")
        time_limit_s = None

    primary_metric, primary_direction = WORKLOAD_PRIMARY_METRIC[workload.name]

    if rejections:
        status = TaskStatus.REJECTED
    elif clarifications:
        status = TaskStatus.NEEDS_CLARIFICATION
    else:
        status = TaskStatus.READY

    return OptimizationTask(
        task_id=task_id or uuid.uuid4().hex[:12],
        status=status,
        model_name=resolved_model,
        gpu_hint=(gpu_hint or None),
        workload=workload,
        primary_metric=primary_metric,
        primary_direction=primary_direction,  # type: ignore[arg-type]
        constraints=constraints,
        target_qps=target_qps,
        experiment_budget=resolved_budget,
        time_limit_s=time_limit_s,
        service_mode=mode,
        clarification_needed=clarifications,
        rejection_reasons=rejections,
        warnings=warnings,
        user_message=user_message,
    )


def confirm_task(task: OptimizationTask) -> OptimizationTask:
    """Mark a ready task confirmed. Rejected / unclear tasks cannot start."""
    if task.status == TaskStatus.REJECTED:
        raise ValueError(
            "Cannot confirm a rejected OptimizationTask: "
            + "; ".join(task.rejection_reasons)
        )
    if task.status == TaskStatus.NEEDS_CLARIFICATION:
        raise ValueError(
            "Cannot confirm a task that still needs clarification: "
            + "; ".join(task.clarification_needed)
        )
    return task.model_copy(update={"status": TaskStatus.CONFIRMED})


def require_confirmed(task: OptimizationTask) -> OptimizationTask:
    if not task.is_executable():
        raise ValueError(
            "OptimizationTask must be confirmed before spending experiment budget "
            f"(status={task.status.value})."
        )
    return task


def default_task_for_workload(
    workload_name: str,
    budget: int = 8,
    *,
    model_name: str | None = None,
    target_qps: float | None = None,
    max_ttft_ms: float | None = None,
    service_mode: str = "managed",
) -> OptimizationTask:
    """CLI / eval path: explicit supported workload, auto-confirmed."""
    task = build_optimization_task(
        workload_name=workload_name,
        model_hint=model_name,
        target_qps=target_qps,
        budget=budget,
        max_ttft_ms=max_ttft_ms,
        service_mode=service_mode,
        user_message=f"explicit:{workload_name}",
    )
    if task.status != TaskStatus.READY:
        reasons = task.rejection_reasons or task.clarification_needed
        raise ValueError("; ".join(reasons) or f"Invalid task for {workload_name}")
    return confirm_task(task)


# ---------------------------------------------------------------------------
# Shared view used by confirm page, executor, and report
# ---------------------------------------------------------------------------

def task_from_mapping(raw: dict[str, Any] | OptimizationTask | None) -> OptimizationTask | None:
    if raw is None:
        return None
    if isinstance(raw, OptimizationTask):
        return raw
    return OptimizationTask.model_validate(raw)


def task_conditions(task: OptimizationTask) -> dict[str, Any]:
    """Stable conditions that confirm UI, executor, and report must share."""
    return {
        "task_id": task.task_id,
        "status": task.status.value,
        "model_name": task.model_name,
        "gpu_hint": task.gpu_hint,
        "engine": task.engine.value,
        "workload_name": task.workload.name,
        "workload": {
            "name": task.workload.name,
            "num_requests": task.workload.num_requests,
            "concurrency": task.workload.concurrency,
            "input_len": task.workload.input_len,
            "output_len": task.workload.output_len,
            "distribution": task.workload.distribution,
            "rps": task.workload.rps,
        },
        "workload_hash": compute_workload_hash(task.workload),
        "primary_metric": task.primary_metric,
        "primary_direction": task.primary_direction,
        "constraints": [c.model_dump(mode="json") for c in task.constraints],
        "target_qps": task.target_qps,
        "target_qps_role": "measured_throughput_goal",
        "traffic_arrival_rate_rps": task.workload.rps,
        "traffic_arrival_rate_supported": False,
        "traffic_arrival_rate_note": TRAFFIC_ARRIVAL_RATE_NOTE,
        "experiment_budget": task.experiment_budget,
        "time_limit_s": task.time_limit_s,
        "service_mode": task.service_mode.value,
        "warnings": list(task.warnings),
    }


def format_task_confirmation(task: OptimizationTask) -> str:
    """Markdown confirmation page. Same facts as ``task_conditions``."""
    cond = task_conditions(task)
    wl = cond["workload"]
    constraint_lines = [
        f"- `{c['metric']} {c['op']} {c['value']}` ({c['source']})"
        for c in cond["constraints"]
    ]
    warning_lines = [f"- {w}" for w in cond["warnings"]] or ["- (none)"]
    target = cond["target_qps"]
    target_txt = (
        f"{target} (measured throughput goal — not an offered arrival rate)"
        if target is not None
        else "not set"
    )
    return "\n".join(
        [
            "## Confirm optimization task",
            "",
            "These conditions will be used by the executor and the final report. "
            "Nothing has been benchmarked yet.",
            "",
            f"- **Task id:** `{cond['task_id']}`",
            f"- {format_resume_help(cond['task_id'])}",
            f"- **Model:** `{cond['model_name']}`",
            f"- **GPU hint:** `{cond['gpu_hint'] or 'not specified'}`",
            f"- **Service mode:** `{cond['service_mode']}`",
            f"- **Workload:** `{wl['name']}`  "
            f"requests={wl['num_requests']}  concurrency={wl['concurrency']}  "
            f"in={wl['input_len']}  out={wl['output_len']}",
            f"- **Workload hash:** `{cond['workload_hash']}`",
            f"- **Primary objective:** `{cond['primary_direction']}` `{cond['primary_metric']}`",
            f"- **Target QPS:** {target_txt}",
            f"- **Offered arrival rate:** "
            f"{cond['traffic_arrival_rate_rps'] if cond['traffic_arrival_rate_rps'] is not None else 'not set'} "
            f"(supported={cond['traffic_arrival_rate_supported']})",
            f"- **Experiment budget:** {cond['experiment_budget']}",
            f"- **Time limit:** {cond['time_limit_s'] if cond['time_limit_s'] is not None else 'not set'}",
            "",
            "### Constraints",
            *constraint_lines,
            "",
            "### Warnings",
            *warning_lines,
            "",
            f"> {TRAFFIC_ARRIVAL_RATE_NOTE}",
        ]
    )


def format_task_blockers(task: OptimizationTask) -> str:
    lines = ["## Task cannot start", ""]
    if task.rejection_reasons:
        lines += ["**Rejected:**"]
        lines += [f"- {r}" for r in task.rejection_reasons]
        lines.append("")
    if task.clarification_needed:
        lines += ["**Need clarification before spending GPU budget:**"]
        lines += [f"- {c}" for c in task.clarification_needed]
        lines.append("")
    if task.warnings:
        lines += ["**Notes:**"]
        lines += [f"- {w}" for w in task.warnings]
    return "\n".join(lines).rstrip() + "\n"


def format_task_conditions_markdown(task: OptimizationTask) -> list[str]:
    """Report section — identical facts to the confirmation page."""
    cond = task_conditions(task)
    wl = cond["workload"]
    lines = [
        "## Task Conditions",
        "",
        f"- **Task id:** `{cond['task_id']}`",
        f"- **Model:** `{cond['model_name']}`",
        f"- **Service mode:** `{cond['service_mode']}`",
        f"- **Workload:** `{wl['name']}`  "
        f"requests={wl['num_requests']}  concurrency={wl['concurrency']}  "
        f"hash=`{cond['workload_hash']}`",
        f"- **Primary objective:** `{cond['primary_direction']}` `{cond['primary_metric']}`",
        f"- **Target QPS:** {cond['target_qps'] if cond['target_qps'] is not None else 'not set'} "
        f"({cond['target_qps_role']})",
        f"- **Offered arrival rate supported:** `{cond['traffic_arrival_rate_supported']}`",
        f"- **Experiment budget:** {cond['experiment_budget']}",
        f"- **Time limit:** {cond['time_limit_s'] if cond['time_limit_s'] is not None else 'not set'}",
        "",
        "Constraints:",
    ]
    for c in cond["constraints"]:
        lines.append(f"- `{c['metric']} {c['op']} {c['value']}` ({c['source']})")
    lines.append("")
    lines.append(f"_{TRAFFIC_ARRIVAL_RATE_NOTE}_")
    lines.append("")
    return lines


def compare_op(op: ConstraintOp | str, left: float, right: float) -> bool:
    token = op.value if isinstance(op, ConstraintOp) else op
    if token == "<=":
        return left <= right
    if token == ">=":
        return left >= right
    if token == "<":
        return left < right
    if token == ">":
        return left > right
    raise ValueError(f"Unsupported constraint op: {token}")
