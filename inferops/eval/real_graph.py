"""Real LangGraph planner eval — production build_graph + planner_node.

Offline mode stubs ONLY:
  (1) the LLM invoke boundary (Fake / scripted ChatModel)
  (2) external execution at the executor tool edge (benchmark / propose)

Does NOT replace planner, executor, or reflector nodes with fake agents.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

from langchain_core.messages import AIMessage

from inferops.agent.executor import tool_boundary_overrides
from inferops.agent.graph import build_graph
from inferops.agent.reflector import reflector_node
from inferops.agent.state import (
    AGENT_SEARCH_SPACE,
    WORKLOAD_PRIMARY_METRIC,
    ExperimentSummary,
    initial_state,
    is_promotable_summary,
)
from inferops.eval.judge import judge_trajectory
from inferops.eval.metrics import (
    EfficiencyMetrics,
    OutcomeMetrics,
    WorkloadScore,
    aggregate_scores,
    composite_score,
    compute_efficiency,
    compute_outcome,
)
from inferops.eval.runner import ALL_WORKLOAD_NAMES, load_ground_truth
from inferops.memory.db import get_result_by_id, init_db, save_result
from inferops.schemas import (
    ConfigEvidence,
    ExperimentConfig,
    ExperimentResult,
    ExperimentValidityStatus,
    InferenceEngine,
    LatencyPercentiles,
    ModelSize,
    SchedulerPolicy,
    WorkloadSpec,
    config_knobs,
    is_promotable,
)
from inferops.tools.propose_config import ProposeConfigInput, ProposeConfigOutput, propose_config_patch
from inferops.tools.run_benchmark import RunBenchmarkInput, RunBenchmarkOutput

# Params the production planner / propose / run_benchmark allow.
_LEGAL_BENCHMARK_KEYS = frozenset(AGENT_SEARCH_SPACE.keys()) | {
    "max_model_len",
    "gpu_memory_utilization",
    "enforce_eager",
}

MODE_REAL_GRAPH_OFFLINE = "real_graph_offline"
MODE_REAL_GRAPH_LLM = "real_graph_llm"
STRATEGY_REAL_PLANNER = "real_planner"

# Dedicated eval SQLite — never the production default `inferops_memory.db`.
DEFAULT_EVAL_DB_DIRNAME = "inferops_real_graph_eval"


def default_eval_db_path() -> Path:
    """Temp SQLite path for offline/live real-graph eval (isolates forged rows)."""
    root = Path(tempfile.mkdtemp(prefix=f"{DEFAULT_EVAL_DB_DIRNAME}_"))
    return root / "eval_memory.db"


def resolve_eval_db_path(db_path: Path | str | None) -> Path:
    """Use an explicit path, else a fresh temporary eval DB."""
    if db_path is None:
        return default_eval_db_path()
    return Path(db_path)


@contextmanager
def _scoped_memory_db(db_path: Path | None) -> Iterator[None]:
    """Point executor (+ memory helpers used by stubs) at an isolated SQLite file."""
    if db_path is None:
        yield
        return
    import inferops.agent.executor as executor_mod
    import inferops.eval.real_graph as self_mod
    import inferops.memory.db as db_mod
    import inferops.tools.analyze_bottleneck as analyze_mod
    import inferops.tools.compare_experiments as compare_mod
    import inferops.tools.propose_config as propose_mod

    db_path = Path(db_path)
    init_db(db_path)

    def _get(eid: str, path: Path | None = None):
        return db_mod.get_result_by_id(eid, db_path=path or db_path)

    def _save(result: ExperimentResult, path: Path | None = None) -> None:
        db_mod.save_result(result, db_path=path or db_path)

    patches = [
        (executor_mod, "get_result_by_id"),
        (self_mod, "get_result_by_id"),
        (self_mod, "save_result"),
        (analyze_mod, "get_result_by_id"),
        (compare_mod, "get_result_by_id"),
        (propose_mod, "get_result_by_id"),
    ]
    previous: list[tuple[Any, str, Any]] = []
    for mod, name in patches:
        previous.append((mod, name, getattr(mod, name)))
        if name == "save_result":
            setattr(mod, name, _save)
        else:
            setattr(mod, name, _get)
    try:
        yield
    finally:
        for mod, name, orig in previous:
            setattr(mod, name, orig)


# ---------------------------------------------------------------------------
# Scripted / fake LLM (offline deterministic)
# ---------------------------------------------------------------------------

def _hyp_json(analysis: str, hyps: list[dict[str, Any]]) -> str:
    return json.dumps({"analysis": analysis, "hypotheses": hyps})


class ScriptedBottleneckLLM:
    """Deterministic LLM stand-in: hypothesis choice follows CURRENT BOTTLENECK.

    Only the invoke boundary is faked — production planner_node still runs.

    Emits a **queue** of distinct legal hypotheses per bottleneck so the
    production Reflect loop can terminate via budget / streak without any
    production Reflect heuristic changes. After the queue is exhausted,
    returns empty hypothesis lists (eval-scoped stop wiring then ends the run).
    """

    eval_llm_boundary = "fake_scripted"

    def __init__(
        self,
        *,
        default_bottleneck: str = "compute-bound",
        inject_illegal: bool = False,
    ) -> None:
        self.default_bottleneck = default_bottleneck
        self.inject_illegal = inject_illegal
        self.call_count = 0
        # Queues of one-hyp responses; popped per invoke so duplicates don't
        # replan forever under production Reflect (gain resets streak).
        self._queues: dict[str, list[str]] = {
            "compute-bound": [
                _hyp_json(
                    "rps=15.0 with compute-bound bottleneck; raise batch tokens.",
                    [{
                        "param": "max_num_batched_tokens",
                        "value": 4096,
                        "rationale": (
                            "rps=15.0 is below ceiling under compute-bound; "
                            "larger batches saturate GPU [source: vllm_scheduler]"
                        ),
                    }],
                ),
                _hyp_json(
                    "rps still compute-bound; try higher seq concurrency.",
                    [{
                        "param": "max_num_seqs",
                        "value": 256,
                        "rationale": (
                            "rps=15.0 with concurrency headroom; "
                            "more seqs [source: vllm_scheduler]"
                        ),
                    }],
                ),
            ],
            "scheduling-bound": [
                _hyp_json(
                    "TTFT p99=210ms under scheduling-bound; enable chunked prefill.",
                    [{
                        "param": "enable_chunked_prefill",
                        "value": True,
                        "rationale": (
                            "TTFT p99=210ms variance indicates scheduling-bound; "
                            "chunked prefill interleaves decode [source: chunked_prefill]"
                        ),
                    }],
                ),
                _hyp_json(
                    "still scheduling-bound; try prefix caching.",
                    [{
                        "param": "enable_prefix_caching",
                        "value": True,
                        "rationale": (
                            "TTFT p99=210ms; prefix reuse helps queueing "
                            "[source: prefix_caching]"
                        ),
                    }],
                ),
            ],
            "memory-bound": [
                _hyp_json(
                    "memory-bound; reduce concurrent sequences.",
                    [{
                        "param": "max_num_seqs",
                        "value": 64,
                        "rationale": (
                            "rps=12.0 with memory-bound pressure; "
                            "fewer seqs reduces KV [source: paged_attention]"
                        ),
                    }],
                ),
            ],
            "kv-bound": [
                _hyp_json(
                    "kv-bound; try prefix caching.",
                    [{
                        "param": "enable_prefix_caching",
                        "value": True,
                        "rationale": (
                            "e2e_p50=1200ms under kv-bound; "
                            "prefix caching reuses KV [source: prefix_caching]"
                        ),
                    }],
                ),
            ],
        }

    def invoke(self, messages: list[Any], **kwargs: Any) -> AIMessage:
        self.call_count += 1
        user_text = ""
        for m in messages:
            content = getattr(m, "content", "") or ""
            if "CURRENT BOTTLENECK:" in content:
                user_text = content
                break
        if not user_text and messages:
            user_text = str(getattr(messages[-1], "content", ""))

        bottleneck = self.default_bottleneck
        match = re.search(r"CURRENT BOTTLENECK:\s*(\S+)", user_text)
        if match:
            bottleneck = match.group(1).strip()

        if self.inject_illegal:
            content = json.dumps({
                "analysis": "rps=15.0; attempting an out-of-policy knob.",
                "hypotheses": [{
                    "param": "tensor_parallel_size",
                    "value": 8,
                    "rationale": "rps=15.0; illegal TP=8 [source: vllm_scheduler]",
                }],
            })
        else:
            queue = self._queues.get(bottleneck) or self._queues[self.default_bottleneck]
            if queue:
                content = queue.pop(0)
            else:
                content = _hyp_json(
                    f"no remaining distinct hypotheses for {bottleneck}",
                    [],
                )
        return AIMessage(content=content)


def _eval_scoped_reflector(state: dict) -> dict:
    """Eval-only wrapper: production Reflect unchanged; empty-plan → stop.

    When the scripted (or live) planner emits zero new hypotheses, production
    Reflect can replan forever if the latest experiment was a gain (streak
    reset). This wrapper is patched into ``build_graph`` only during eval.
    """
    traj = state.get("trajectory") or []
    if traj and traj[-1].get("node") == "planner":
        generated = traj[-1].get("hypotheses") or []
        if not generated:
            step = {
                "step": len(traj) + 1,
                "node": "reflector",
                "workload": state["workload_name"],
                "action": "reflect",
                "reasoning": "eval_scoped_stop: planner produced 0 hypotheses",
                "result": {"empty_plan": True, "eval_scoped_stop": True},
            }
            return {
                "should_stop": True,
                "stop_reason": "eval_empty_plan",
                "trajectory": traj + [step],
            }
    return reflector_node(state)


def mark_live_llm(llm: Any) -> Any:
    """Attach a trusted ``eval_llm_boundary='live'`` marker for eval labeling.

    Used only for real ``make_llm(...)`` results (or explicit live adapters).
    Unknown injected objects must NOT be labeled live.
    """
    try:
        object.__setattr__(llm, "eval_llm_boundary", "live")
        return llm
    except Exception:
        return _TrustedLiveLLM(llm)


class _TrustedLiveLLM:
    """Thin adapter that preserves invoke while carrying the live marker."""

    eval_llm_boundary = "live"

    def __init__(self, inner: Any) -> None:
        object.__setattr__(self, "_inner", inner)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.invoke(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _llm_boundary_label(llm: Any, mode: str = "") -> str:
    """Label from trusted markers / known fakes only — never mode alone.

    - ``ScriptedBottleneckLLM`` / ``eval_llm_boundary='fake_scripted'`` → fake_scripted
    - ``eval_llm_boundary='live'`` (set by ``mark_live_llm`` on real make_llm) → live
    - everything else unknown → injected
    """
    marked = getattr(llm, "eval_llm_boundary", None)
    if marked == "fake_scripted" or isinstance(llm, ScriptedBottleneckLLM):
        return "fake_scripted"
    if marked == "live":
        return "live"
    return "injected"


# ---------------------------------------------------------------------------
# Stub benchmark at tool boundary
# ---------------------------------------------------------------------------

@dataclass
class StubBenchmarkRecorder:
    """Records every run_benchmark call; never touches real vLLM."""

    calls: list[dict[str, Any]] = field(default_factory=list)
    # experiment_id → metric overrides / contract knobs
    outcome_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    default_gain: bool = True
    force_unevidenced: bool = False
    baseline_rps: float = 15.0

    def __call__(self, inp: RunBenchmarkInput) -> RunBenchmarkOutput:
        for key in inp.config_patch:
            if key not in _LEGAL_BENCHMARK_KEYS:
                raise AssertionError(
                    f"Illegal param reached stub benchmark: {key}={inp.config_patch[key]}"
                )
        self.calls.append({
            "experiment_id": inp.experiment_id,
            "config_patch": dict(inp.config_patch),
            "workload_name": inp.workload_name,
        })

        overrides = dict(self.outcome_overrides.get(inp.experiment_id, {}))
        # Infer metrics from patch when no explicit override
        rps = float(overrides.pop("throughput_rps", self._default_rps(inp.config_patch)))
        tokens = float(overrides.pop("tokens_per_second", rps * 128.0))
        ttft_p99 = float(overrides.pop("ttft_p99_ms", 66.0))
        e2e_p50 = float(overrides.pop("e2e_p50_ms", 880.0))
        unevidenced = bool(overrides.pop("unevidenced", self.force_unevidenced))
        bottleneck_hint = overrides.pop("bottleneck", None)

        result = _synthesize_result(
            experiment_id=inp.experiment_id,
            workload_name=inp.workload_name,
            config_patch=inp.config_patch,
            session_id=inp.session_id,
            throughput_rps=rps,
            tokens_per_second=tokens,
            ttft_p99_ms=ttft_p99,
            e2e_p50_ms=e2e_p50,
            unevidenced=unevidenced,
        )
        if inp.persist:
            save_result(result)

        # Stash bottleneck hint for analyze stub via notes / tags
        if bottleneck_hint:
            result = result.model_copy(update={"notes": f"bottleneck:{bottleneck_hint}"})
            if inp.persist:
                save_result(result)

        status_value = result.status.value
        return RunBenchmarkOutput(
            experiment_id=inp.experiment_id,
            workload_name=inp.workload_name,
            throughput_rps=result.throughput_rps,
            tokens_per_second=result.tokens_per_second,
            ttft_p50_ms=result.ttft.p50,
            ttft_p99_ms=result.ttft.p99,
            e2e_p50_ms=result.e2e_latency.p50,
            e2e_p99_ms=result.e2e_latency.p99,
            gpu_util_pct=result.gpu_utilization_pct,
            gpu_mem_gb=result.gpu_memory_used_gb,
            success_rate=f"{result.successful_requests}/{result.total_requests}",
            mlflow_run_id=result.mlflow_run_id,
            run_id=result.run_id,
            status=status_value,
        )

    def _default_rps(self, patch: dict[str, Any]) -> float:
        if not self.default_gain:
            return self.baseline_rps * 0.95  # no gain vs baseline
        if patch.get("max_num_batched_tokens") == 4096:
            return self.baseline_rps * 1.15
        if patch.get("enable_chunked_prefill") is True:
            return self.baseline_rps * 1.08
        if patch.get("enable_prefix_caching") is True:
            return self.baseline_rps * 1.05
        if patch.get("max_num_seqs") == 64:
            return self.baseline_rps * 1.02
        return self.baseline_rps * 1.01


def _guarding_propose(inp: ProposeConfigInput) -> ProposeConfigOutput:
    """Production propose_config_patch — rejects illegal params before benchmark."""
    return propose_config_patch(inp)


def _synthesize_result(
    *,
    experiment_id: str,
    workload_name: str,
    config_patch: dict[str, Any],
    session_id: str | None,
    throughput_rps: float,
    tokens_per_second: float,
    ttft_p99_ms: float,
    e2e_p50_ms: float,
    unevidenced: bool,
) -> ExperimentResult:
    workload = WorkloadSpec(
        name=workload_name,
        prompt_template="",
        num_requests=10,
        concurrency=4,
        input_len=64,
        output_len=64,
    )
    cfg_kwargs: dict[str, Any] = {
        "experiment_id": experiment_id,
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_size": ModelSize.HALF_B,
        "engine": InferenceEngine.VLLM,
        "max_num_seqs": 128,
        "max_num_batched_tokens": 2048,
        "max_model_len": 1024,
        "gpu_memory_utilization": 0.80,
        "enforce_eager": False,
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
        "scheduler_policy": SchedulerPolicy.FCFS,
        "workload": workload,
        "tags": {"eval": "real_graph_stub"},
    }
    cfg_kwargs.update(config_patch)
    cfg = ExperimentConfig(**cfg_kwargs)
    knobs = config_knobs(cfg)
    ttft = LatencyPercentiles(
        p50=max(20.0, ttft_p99_ms * 0.7),
        p90=ttft_p99_ms * 0.9,
        p95=ttft_p99_ms * 0.95,
        p99=ttft_p99_ms,
    )
    e2e = LatencyPercentiles(
        p50=e2e_p50_ms,
        p90=e2e_p50_ms * 1.05,
        p95=e2e_p50_ms * 1.08,
        p99=e2e_p50_ms * 1.12,
    )
    if unevidenced:
        return ExperimentResult(
            experiment_id=experiment_id,
            config=cfg,
            total_requests=10,
            successful_requests=10,
            total_time_s=10.0 / max(throughput_rps, 0.01),
            throughput_rps=throughput_rps,
            tokens_per_second=tokens_per_second,
            ttft=ttft,
            tpot=LatencyPercentiles(p50=6.0, p90=7.0, p95=7.5, p99=8.0),
            e2e_latency=e2e,
            gpu_memory_used_gb=3.7,
            gpu_utilization_pct=85.0,
            raw_ttft_ms=[ttft.p50] * 10,
            raw_e2e_ms=[e2e_p50_ms] * 10,
            run_id=uuid.uuid4().hex,
            session_id=session_id,
            requested_config=knobs,
            actual_config=None,
            config_evidence=None,
            status=ExperimentValidityStatus.INSUFFICIENT_EVIDENCE,
        )

    evidence = ConfigEvidence(
        kind="managed_process_start",
        verified=True,
        instance_id=f"stub:{experiment_id}",
        process_pid=1,
        observed_params=dict(knobs),
    )
    return ExperimentResult(
        experiment_id=experiment_id,
        config=cfg,
        total_requests=10,
        successful_requests=10,
        total_time_s=10.0 / max(throughput_rps, 0.01),
        throughput_rps=throughput_rps,
        tokens_per_second=tokens_per_second,
        ttft=ttft,
        tpot=LatencyPercentiles(p50=6.0, p90=7.0, p95=7.5, p99=8.0),
        e2e_latency=e2e,
        gpu_memory_used_gb=3.7,
        gpu_utilization_pct=85.0,
        raw_ttft_ms=[ttft.p50] * 10,
        raw_e2e_ms=[e2e_p50_ms] * 10,
        run_id=uuid.uuid4().hex,
        session_id=session_id,
        mlflow_run_id=f"stub-mlflow-{experiment_id[-8:]}",
        requested_config=knobs,
        actual_config=dict(knobs),
        config_evidence=evidence,
        status=ExperimentValidityStatus.VALID,
    )


def make_promotable_baseline_summary(
    workload_name: str,
    session_prefix: str,
    *,
    throughput_rps: float = 15.0,
    tokens_per_second: float = 1900.0,
    bottleneck: str = "compute-bound",
) -> ExperimentSummary:
    """Build a contract-valid baseline summary and persist matching DB row."""
    eid = f"{session_prefix}baseline"
    primary = WORKLOAD_PRIMARY_METRIC[workload_name]
    result = _synthesize_result(
        experiment_id=eid,
        workload_name=workload_name,
        config_patch={},
        session_id=session_prefix,
        throughput_rps=throughput_rps,
        tokens_per_second=tokens_per_second,
        ttft_p99_ms=70.0,
        e2e_p50_ms=1000.0,
        unevidenced=False,
    )
    save_result(result)
    assert is_promotable(result)
    from inferops.agent.state import summary_from_result

    summary = summary_from_result(
        result,
        param_changed=None,
        value_changed=None,
        baseline_primary=getattr(result, primary, result.throughput_rps),
        primary_metric=primary,
        bottleneck=bottleneck,
    )
    summary["experiment_id"] = eid
    summary["vs_baseline_pct"] = 0.0
    return summary


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def require_llm_credentials(backend: str = "openrouter") -> None:
    """Fail loudly when real-LLM mode lacks credentials (never silent pass)."""
    required = {
        "openrouter": "OPENROUTER_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
    }
    env_key = required.get(backend, "OPENROUTER_API_KEY")
    if not os.environ.get(env_key):
        raise RuntimeError(
            f"real_graph_llm mode requires {env_key} for backend={backend!r}. "
            "Refusing to silent-pass without credentials."
        )


def run_real_planner_on_workload(
    *,
    workload_name: str,
    llm: Any,
    budget: int = 3,
    bottleneck: str = "compute-bound",
    session_prefix: str | None = None,
    stub: StubBenchmarkRecorder | None = None,
    baseline_rps: float = 15.0,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Invoke production build_graph(llm) with stubbed tool edges only.

    Defaults to a temporary eval SQLite DB so forged rows never land in the
    production ``inferops_memory.db``.
    """
    resolved_db = resolve_eval_db_path(db_path)
    with _scoped_memory_db(resolved_db):
        prefix = session_prefix or f"rgeval_{workload_name}_{uuid.uuid4().hex[:6]}_"
        recorder = stub or StubBenchmarkRecorder(baseline_rps=baseline_rps)
        recorder.baseline_rps = baseline_rps

        baseline = make_promotable_baseline_summary(
            workload_name,
            prefix,
            throughput_rps=baseline_rps,
            bottleneck=bottleneck,
        )
        state = initial_state(workload_name, prefix, max_experiments=budget)
        state["baseline_summary"] = baseline
        state["best_summary"] = baseline if is_promotable_summary(baseline) else None
        state["experiment_summaries"] = [baseline]
        state["tried_experiment_ids"] = [baseline["experiment_id"]]
        state["current_bottleneck"] = bottleneck
        state["experiments_remaining"] = max(0, budget - 1)

        # Eval-scoped Reflect wrapper only — production reflector_node heuristics
        # stay untouched on master.
        with patch("inferops.agent.graph.reflector_node", _eval_scoped_reflector):
            graph = build_graph(llm)
            with tool_boundary_overrides(
                run_benchmark_fn=recorder,
                propose_config_fn=_guarding_propose,
            ):
                final_state = graph.invoke(state)

        return {
            "workload_name": workload_name,
            "session_prefix": prefix,
            "final_state": final_state,
            "trajectory": list(final_state.get("trajectory") or []),
            "stop_reason": final_state.get("stop_reason") or "",
            "best_summary": final_state.get("best_summary"),
            "baseline_summary": final_state.get("baseline_summary"),
            "hypotheses": list(final_state.get("hypotheses") or []),
            "benchmark_calls": list(recorder.calls),
            "llm_call_count": getattr(llm, "call_count", None),
            "eval_db_path": str(resolved_db),
        }


def _score_real_run(
    ground_truth: dict[str, Any],
    run: dict[str, Any],
) -> dict[str, Any]:
    wl = run["workload_name"]
    primary = WORKLOAD_PRIMARY_METRIC[wl]
    best = run.get("best_summary")
    # Eval best must be promotable (① gate) — fall back to zero agent value.
    if best is not None and is_promotable_summary(best):
        agent_row = {
            primary: best.get(primary, 0.0),
            "ttft_p99_ms": best.get("ttft_p99_ms", 0.0),
            "e2e_p50_ms": best.get("e2e_p50_ms", 0.0),
            "tokens_per_second": best.get("tokens_per_second", 0.0),
            "experiment_id": best.get("experiment_id", ""),
        }
    else:
        agent_row = {primary: 0.0, "experiment_id": ""}

    outcome = compute_outcome(ground_truth, agent_row)
    # Count non-baseline experiments from trajectory executor steps
    n_exp = sum(1 for step in run["trajectory"] if step.get("node") == "executor")
    efficiency = compute_efficiency(n_exp, wall_clock_s=0.0)
    trajectory_score = judge_trajectory(run["trajectory"]).overall
    comp = composite_score(outcome, efficiency, trajectory_score=trajectory_score)
    return {
        "workload_name": outcome.workload_name,
        "primary_metric": outcome.primary_metric,
        "ground_truth_value": outcome.ground_truth_value,
        "agent_value": outcome.agent_value,
        "gap_pct": outcome.gap_pct,
        "n_experiments": efficiency.n_experiments,
        "trajectory_score": trajectory_score,
        "composite": comp,
        "best_experiment_id": agent_row.get("experiment_id", ""),
        "stop_reason": run["stop_reason"],
        "trajectory_nodes": [s.get("node") for s in run["trajectory"]],
        "hypotheses": [
            {"param": h.get("param"), "value": h.get("value"), "status": h.get("status")}
            for h in run["hypotheses"]
        ],
        "benchmark_calls": run["benchmark_calls"],
    }


def run_real_graph_eval(
    commit_sha: str,
    ground_truth_dir: str | Path,
    workloads: list[str] | None = None,
    budget: int = 3,
    *,
    mode: str = MODE_REAL_GRAPH_OFFLINE,
    llm: Any | None = None,
    llm_backend: str = "openrouter",
    bottleneck: str = "compute-bound",
    inject_illegal: bool = False,
    default_gain: bool = True,
    force_unevidenced: bool = False,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Commit-level report using production graph + stubbed tool/LLM edges.

    Uses a temporary/dedicated eval DB by default (never ``inferops_memory.db``).
    """
    resolved_db = resolve_eval_db_path(db_path)

    if mode == MODE_REAL_GRAPH_LLM:
        require_llm_credentials(llm_backend)
        if llm is None:
            from inferops.agent.graph import make_llm
            llm = mark_live_llm(make_llm(backend=llm_backend, temperature=0.0))
        else:
            # Caller-supplied llm: only "live" if already trusted-marked.
            pass
        # Hard cap: ≤1 workload, ≤2 planner calls worth of budget
        names = (workloads or ALL_WORKLOAD_NAMES)[:1]
        budget = min(budget, 3)  # baseline + ≤2 experiments
    elif mode == MODE_REAL_GRAPH_OFFLINE:
        names = workloads or ALL_WORKLOAD_NAMES
        if llm is None:
            llm = ScriptedBottleneckLLM(
                default_bottleneck=bottleneck,
                inject_illegal=inject_illegal,
            )
    else:
        raise ValueError(f"Unknown real-graph mode: {mode!r}")

    stub = StubBenchmarkRecorder(
        default_gain=default_gain,
        force_unevidenced=force_unevidenced,
    )
    rows: list[dict[str, Any]] = []
    for wl_name in names:
        # Fresh stub call log per workload (report still lists per-row calls)
        wl_stub = StubBenchmarkRecorder(
            default_gain=default_gain,
            force_unevidenced=force_unevidenced,
            baseline_rps=stub.baseline_rps,
        )
        # Fresh scripted LLM per workload so queues don't share across WLs
        wl_llm = llm
        if mode == MODE_REAL_GRAPH_OFFLINE and isinstance(llm, ScriptedBottleneckLLM):
            wl_llm = ScriptedBottleneckLLM(
                default_bottleneck=bottleneck,
                inject_illegal=inject_illegal,
            )
        gt = load_ground_truth(wl_name, ground_truth_dir)
        run = run_real_planner_on_workload(
            workload_name=wl_name,
            llm=wl_llm,
            budget=budget,
            bottleneck=bottleneck,
            stub=wl_stub,
            db_path=resolved_db,
        )
        rows.append(_score_real_run(gt, run))

    scores = [
        WorkloadScore(
            workload_name=r["workload_name"],
            outcome=OutcomeMetrics(
                workload_name=r["workload_name"],
                primary_metric=r["primary_metric"],
                ground_truth_value=r["ground_truth_value"],
                agent_value=r["agent_value"],
                gap_pct=r["gap_pct"],
            ),
            efficiency=EfficiencyMetrics(n_experiments=r["n_experiments"], wall_clock_s=0.0),
            trajectory_score=r["trajectory_score"],
            composite=r["composite"],
        )
        for r in rows
    ]

    return {
        "commit_sha": commit_sha,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "budget": budget,
        "llm_boundary": _llm_boundary_label(llm, mode),
        "tool_boundary": "stubbed_benchmark",
        "eval_db_path": str(resolved_db),
        "strategies": {STRATEGY_REAL_PLANNER: rows},
        "aggregates": {STRATEGY_REAL_PLANNER: aggregate_scores(scores)},
    }
