"""Core Pydantic schemas shared across the inferops agent system."""

from __future__ import annotations

import hashlib
import json
import subprocess
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ModelSize(str, Enum):
    HALF_B = "0.5B"
    ONE_HALF_B = "1.5B"
    SEVEN_B = "7B"
    FOURTEEN_B = "14B"


class InferenceEngine(str, Enum):
    VLLM = "vllm"
    OLLAMA = "ollama"


class SchedulerPolicy(str, Enum):
    FCFS = "fcfs"
    PRIORITY = "priority"


class ExperimentValidityStatus(str, Enum):
    """Week-1 experiment contract validity statuses.

    Old / legacy rows MUST default to insufficient_evidence — never auto-promote
    to valid just because metrics look good.
    """

    VALID = "valid"
    INVALID = "invalid"
    FAILED = "failed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


# ---------------------------------------------------------------------------
# Contract constants
# ---------------------------------------------------------------------------

EXPERIMENT_SCHEMA_VERSION = "1"

# Tuneable knobs compared between requested and actual config.
CONFIG_KNOB_KEYS: tuple[str, ...] = (
    "model_name",
    "max_num_seqs",
    "max_num_batched_tokens",
    "max_model_len",
    "gpu_memory_utilization",
    "enforce_eager",
    "enable_chunked_prefill",
    "enable_prefix_caching",
    "scheduler_policy",
    "tensor_parallel_size",
)

# Keys that vllm_process._build_cmd actually passes on the CLI.
# Anything else (e.g. scheduler_policy, tensor_parallel_size) is NOT evidenced
# by a managed start until item ② can verify instance knobs.
MANAGED_CLI_EVIDENCED_KEYS: frozenset[str] = frozenset(
    {
        "model_name",
        "max_num_seqs",
        "max_num_batched_tokens",
        "max_model_len",
        "gpu_memory_utilization",
        "enforce_eager",
        "enable_chunked_prefill",
        "enable_prefix_caching",
    }
)

# Evidence kinds that are NEVER sufficient on their own (item ② / contract).
INSUFFICIENT_EVIDENCE_KINDS: frozenset[str] = frozenset(
    {
        "config_file_only",
        "http_ok_only",
        "health_check_only",
        "performance_delta_only",
        "external_unverified",
    }
)


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------

class WorkloadSpec(BaseModel):
    """Describes a synthetic or replay benchmark workload."""

    name: str
    prompt_template: str
    num_requests: int = Field(ge=1)
    concurrency: int = Field(default=1, ge=1)
    input_len: int = Field(default=128, ge=1, description="Approx tokens per prompt")
    output_len: int = Field(default=128, ge=1, description="Max new tokens")
    distribution: str = Field(default="uniform", description="poisson | uniform | fixed")
    rps: float | None = Field(default=None, description="Target requests-per-second (None = as fast as possible)")


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class ExperimentConfig(BaseModel):
    """Full specification of one optimization experiment run."""

    experiment_id: str
    model_name: str = Field(default="Qwen/Qwen2.5-0.5B-Instruct")
    model_size: ModelSize = ModelSize.HALF_B
    engine: InferenceEngine = InferenceEngine.VLLM

    # vLLM knobs under test
    max_num_seqs: int = Field(default=128, ge=1)
    max_num_batched_tokens: int = Field(default=2048, ge=128)
    max_model_len: int = Field(default=2048, ge=128)
    gpu_memory_utilization: float = Field(default=0.80, ge=0.1, le=1.0)
    enforce_eager: bool = False
    enable_chunked_prefill: bool = False
    enable_prefix_caching: bool = False
    scheduler_policy: SchedulerPolicy = SchedulerPolicy.FCFS
    tensor_parallel_size: int = Field(default=1, ge=1)

    # What to run against
    workload: WorkloadSpec

    # Free-form metadata for MLflow tagging
    tags: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Contract: evidence / hardware / identity
# ---------------------------------------------------------------------------

class ConfigEvidence(BaseModel):
    """Proof that requested config was (or was not) applied to a live instance.

    Rules encoded here (not just prose):
    - verified=False → never promotable
    - kinds in INSUFFICIENT_EVIDENCE_KINDS → never promotable alone
    - config file alone / HTTP 200 alone / perf delta alone → never sufficient
    """

    kind: str = Field(
        description=(
            "Evidence kind, e.g. managed_process_start, instance_identity, "
            "external_unverified, health_check_only, config_file_only, "
            "http_ok_only, performance_delta_only"
        )
    )
    verified: bool = False
    instance_id: str | None = None
    process_pid: int | None = None
    observed_params: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""

    def is_critical_evidence(self) -> bool:
        """True only when evidence is strong enough to support status=valid."""
        if not self.verified:
            return False
        if self.kind in INSUFFICIENT_EVIDENCE_KINDS:
            return False
        return True


class HardwareInfo(BaseModel):
    """Model / hardware / vLLM fingerprint for experiment reproducibility."""

    model_name: str = ""
    engine: str = "vllm"
    vllm_version: str | None = None
    gpu_name: str | None = None
    cuda_version: str | None = None


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

class LatencyPercentiles(BaseModel):
    """Latency percentiles. Missing samples → None (never invent 0.0).

    `sample_n` / `sample_scope` document which ledger rows entered the
    percentile (Week-2 P0-④). Legacy rows may omit them (default 0 / "").
    """

    p50: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None
    sample_n: int = 0
    sample_scope: str = ""


class ExperimentResult(BaseModel):
    """Collected metrics from a completed experiment run + Week-1 contract.

    Week-2 P0-④: aggregates are recalculable from `request_ledger` via
    `inferops.metrics.recalculate_from_ledger`. Missing metrics are None —
    never defaulted to 0 to invent gains.
    """

    experiment_id: str
    config: ExperimentConfig

    # Throughput (None = not computable; do not treat as 0 gain)
    total_requests: int
    successful_requests: int
    total_time_s: float
    throughput_rps: float | None = None  # successful / window_s
    tokens_per_second: float | None = None  # success output tokens / window_s
    error_rate: float | None = None  # failed / total_requests (measured)

    # Latency — sample_scope on each LatencyPercentiles states eligibility
    ttft: LatencyPercentiles  # client TTFT (ms); requires streaming
    tpot: LatencyPercentiles  # per-request TPOT; N/A when output_tokens < 2
    e2e_latency: LatencyPercentiles

    # Resource — only when sampled (GPU) / pricing exists (cost)
    gpu_memory_used_gb: float | None = None
    gpu_utilization_pct: float | None = None
    cost_usd: float | None = None

    # Raw per-request latency (compat for bootstrap CI; prefer request_ledger)
    raw_ttft_ms: list[float] = Field(default_factory=list)
    raw_e2e_ms: list[float] = Field(default_factory=list)
    # Per-request ledger dump (RequestLedger.model_dump) or legacy list of records.
    # Source of truth for recalculation. Prefer the full object (window + conditions).
    request_ledger: dict[str, Any] | list[dict[str, Any]] = Field(default_factory=dict)
    ledger_path: str | None = None

    # --- Week-1 experiment contract ---
    # Unique identity (distinct from human-readable experiment_id).
    # Empty / missing → filled by validator with a *stable* legacy id (not a
    # fresh UUID on every deserialize). New runs must pass an explicit run_id.
    run_id: str = ""
    schema_version: str = EXPERIMENT_SCHEMA_VERSION
    code_sha: str | None = None
    # Mapping: experiment_id ↔ session prefix ↔ MLflow
    session_id: str | None = None
    mlflow_run_id: str | None = None
    # Requested vs actual config (+ evidence). `config` remains the requested
    # ExperimentConfig for backward compatibility; knobs are also snapshotted.
    requested_config: dict[str, Any] = Field(default_factory=dict)
    actual_config: dict[str, Any] | None = None
    config_evidence: ConfigEvidence | None = None
    # Default is insufficient_evidence so legacy / incomplete rows never
    # auto-promote to valid.
    status: ExperimentValidityStatus = ExperimentValidityStatus.INSUFFICIENT_EVIDENCE
    workload_hash: str | None = None
    hardware: HardwareInfo | None = None

    notes: str = ""

    @model_validator(mode="after")
    def _fill_contract_defaults(self) -> ExperimentResult:
        if not self.requested_config:
            self.requested_config = config_knobs(self.config)
        if not self.workload_hash:
            self.workload_hash = compute_workload_hash(self.config.workload)
        if self.hardware is None:
            self.hardware = HardwareInfo(
                model_name=self.config.model_name,
                engine=self.config.engine.value,
            )
        if not self.run_id:
            # Stable across re-reads of the same legacy row (P2-6).
            self.run_id = stable_legacy_run_id(
                self.experiment_id, self.mlflow_run_id
            )
        return self


# ---------------------------------------------------------------------------
# Contract helpers (encode rules in code, not prose alone)
# ---------------------------------------------------------------------------

def config_knobs(cfg: ExperimentConfig) -> dict[str, Any]:
    """Stable snapshot of tuneable knobs from an ExperimentConfig."""
    out: dict[str, Any] = {}
    for key in CONFIG_KNOB_KEYS:
        val = getattr(cfg, key)
        out[key] = val.value if isinstance(val, Enum) else val
    return out


def managed_cli_actual_config(requested: dict[str, Any]) -> dict[str, Any]:
    """Subset of requested knobs that `_build_cmd` actually passes on the CLI.

    Non-CLI keys (scheduler_policy, tensor_parallel_size, …) are omitted so they
    cannot be falsely recorded as applied.
    """
    return {k: requested[k] for k in MANAGED_CLI_EVIDENCED_KEYS if k in requested}


def compute_workload_hash(workload: WorkloadSpec) -> str:
    """Content hash of workload shape (excludes prompt text body size noise)."""
    payload = {
        "name": workload.name,
        "num_requests": workload.num_requests,
        "concurrency": workload.concurrency,
        "input_len": workload.input_len,
        "output_len": workload.output_len,
        "distribution": workload.distribution,
        "rps": workload.rps,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def resolve_git_sha(short: bool = True) -> str | None:
    """Best-effort code SHA for the running checkout."""
    try:
        args = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"]
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def stable_legacy_run_id(experiment_id: str, mlflow_run_id: str | None = None) -> str:
    """Deterministic run_id for pre-contract rows (stable across re-reads)."""
    seed = f"legacy:{experiment_id}:{mlflow_run_id or ''}"
    return hashlib.sha256(seed.encode()).hexdigest()[:32]


def has_critical_config_evidence(evidence: ConfigEvidence | None) -> bool:
    return evidence is not None and evidence.is_critical_evidence()


def actual_covers_requested(
    requested_config: dict[str, Any] | None,
    actual_config: dict[str, Any] | None,
) -> bool:
    """True iff every requested key is present in actual AND values match.

    Missing keys are NOT matches (P1-2). Empty actual never covers non-empty
    requested.
    """
    if requested_config is None or actual_config is None:
        return False
    if not requested_config:
        return False
    for key, req_val in requested_config.items():
        if key not in actual_config:
            return False
        if actual_config[key] != req_val:
            return False
    return True


def actual_has_mismatch(
    requested_config: dict[str, Any] | None,
    actual_config: dict[str, Any] | None,
) -> bool:
    """True if any *shared* key disagrees (used to distinguish invalid vs insuff)."""
    if not requested_config or not actual_config:
        return False
    for key, req_val in requested_config.items():
        if key in actual_config and actual_config[key] != req_val:
            return True
    return False


def is_promotable(result: ExperimentResult) -> bool:
    """Single full promotion gate used everywhere (P1-3).

    Requires ALL of:
      - status == valid
      - critical config evidence present
      - actual_config covers every requested knob (no missing keys)
      - at least one successful request (all-failed workloads are not promotable)
    High scores alone never suffice. Empty/partial actual never promotes.
    """
    if result.status != ExperimentValidityStatus.VALID:
        return False
    if not has_critical_config_evidence(result.config_evidence):
        return False
    requested = result.requested_config or config_knobs(result.config)
    if not actual_covers_requested(requested, result.actual_config):
        return False
    if result.successful_requests <= 0:
        return False
    return True


def derive_status(
    *,
    failed: bool = False,
    evidence: ConfigEvidence | None = None,
    actual_config: dict[str, Any] | None = None,
    requested_config: dict[str, Any] | None = None,
    successful_requests: int | None = None,
) -> ExperimentValidityStatus:
    """Derive contract status from evidence + actual/requested config.

    Never returns valid unless evidence is critical AND actual covers every
    requested key. Missing keys → insufficient_evidence (not a match).
    Mismatched shared keys → invalid. Zero successes → failed.
    """
    if failed:
        return ExperimentValidityStatus.FAILED
    if successful_requests is not None and successful_requests <= 0:
        return ExperimentValidityStatus.FAILED
    if not has_critical_config_evidence(evidence):
        return ExperimentValidityStatus.INSUFFICIENT_EVIDENCE
    if actual_config is None:
        return ExperimentValidityStatus.INSUFFICIENT_EVIDENCE
    if requested_config:
        if actual_has_mismatch(requested_config, actual_config):
            return ExperimentValidityStatus.INVALID
        if not actual_covers_requested(requested_config, actual_config):
            # Partial actual (e.g. CLI-only managed keys) → not valid yet.
            return ExperimentValidityStatus.INSUFFICIENT_EVIDENCE
    else:
        # No requested snapshot → cannot claim valid application.
        return ExperimentValidityStatus.INSUFFICIENT_EVIDENCE
    return ExperimentValidityStatus.VALID


def managed_start_evidence(
    *,
    process_pid: int | None,
    host: str,
    port: int,
    observed_params: dict[str, Any],
    instance_id: str | None = None,
    start_token: str | None = None,
    notes: str | None = None,
) -> ConfigEvidence:
    """Evidence for CLI knobs actually passed when bench_runner started vLLM.

    `observed_params` must be the CLI-evidenced subset only — never the full
    requested dict (non-CLI knobs are unverified until item ② verifies them
    with complete coverage).
    """
    if instance_id is None:
        instance_id = (
            f"{host}:{port}:pid={process_pid}" if process_pid else f"{host}:{port}"
        )
        if start_token:
            instance_id = f"{instance_id}:gen={start_token}"
    return ConfigEvidence(
        kind="managed_process_start",
        verified=True,
        instance_id=instance_id,
        process_pid=process_pid,
        observed_params=dict(observed_params),
        notes=notes
        or (
            "vLLM process started by bench_runner; observed_params lists only "
            "knobs passed on the CLI. Non-CLI requested knobs are unverified."
        ),
    )


def instance_identity_evidence(
    *,
    process_pid: int | None,
    host: str,
    port: int,
    observed: dict[str, Any],
    instance_id: str,
    start_token: str | None = None,
    notes: str = "",
) -> ConfigEvidence:
    """Critical evidence when live instance identity + knobs were verified."""
    return ConfigEvidence(
        kind="instance_identity",
        verified=True,
        instance_id=instance_id,
        process_pid=process_pid,
        observed_params=dict(observed),
        notes=notes
        or (
            f"Instance identity verified (start_token={start_token or 'n/a'}); "
            "observed CLI knobs recorded. Complete requested coverage still required "
            "for status=valid / promotion."
        ),
    )


def external_unverified_evidence(*, host: str, port: int) -> ConfigEvidence:
    """External healthy server — healthy ≠ config applied."""
    return ConfigEvidence(
        kind="external_unverified",
        verified=False,
        instance_id=f"{host}:{port}",
        notes=(
            "External vLLM reported HTTP health OK; config application was not "
            "verified. Health/HTTP 200 alone is never sufficient evidence."
        ),
    )


def empty_latency() -> LatencyPercentiles:
    """Missing latency samples — all None, never fake zeros (P0-④)."""
    return LatencyPercentiles(
        p50=None,
        p90=None,
        p95=None,
        p99=None,
        sample_n=0,
        sample_scope="",
    )


# ---------------------------------------------------------------------------
# LangGraph agent state (legacy Pydantic mirror; runtime uses TypedDict)
# ---------------------------------------------------------------------------

class AgentState(BaseModel):
    """Shared mutable state threaded through the LangGraph agent."""

    # Current experiment being planned / executed
    current_config: ExperimentConfig | None = None
    pending_configs: list[ExperimentConfig] = Field(default_factory=list)
    completed_results: list[ExperimentResult] = Field(default_factory=list)

    # Iteration control
    iteration: int = 0
    max_iterations: int = 20
    should_stop: bool = False

    # Reasoning scratch-pad (populated by the LLM planner node)
    hypothesis: str = ""
    last_action: str = ""
    messages: list[dict[str, str]] = Field(default_factory=list)
