"""Core Pydantic schemas shared across the inferops agent system."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


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
    max_num_seqs: int = Field(default=256, ge=1)
    max_num_batched_tokens: int = Field(default=2048, ge=128)
    gpu_memory_utilization: float = Field(default=0.90, ge=0.1, le=1.0)
    enforce_eager: bool = False
    enable_chunked_prefill: bool = False
    scheduler_policy: SchedulerPolicy = SchedulerPolicy.FCFS
    tensor_parallel_size: int = Field(default=1, ge=1)

    # What to run against
    workload: WorkloadSpec

    # Free-form metadata for MLflow tagging
    tags: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

class LatencyPercentiles(BaseModel):
    p50: float
    p90: float
    p95: float
    p99: float


class ExperimentResult(BaseModel):
    """Collected metrics from a completed experiment run."""

    experiment_id: str
    config: ExperimentConfig

    # Throughput
    total_requests: int
    successful_requests: int
    total_time_s: float
    throughput_rps: float          # successful / total_time_s
    tokens_per_second: float       # output tokens / total_time_s

    # Latency
    ttft: LatencyPercentiles       # Time-to-first-token (ms)
    tpot: LatencyPercentiles       # Time-per-output-token (ms)
    e2e_latency: LatencyPercentiles  # End-to-end (ms)

    # Resource
    gpu_memory_used_gb: float | None = None
    gpu_utilization_pct: float | None = None

    # Agent bookkeeping
    mlflow_run_id: str | None = None
    notes: str = ""


# ---------------------------------------------------------------------------
# LangGraph agent state
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
