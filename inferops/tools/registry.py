"""LangGraph tool registry — wraps all 8 inferops tools with @tool for agent use."""

from __future__ import annotations

from langchain_core.tools import tool

from inferops.tools.analyze_bottleneck import AnalyzeBottleneckInput, analyze_bottleneck
from inferops.tools.compare_experiments import CompareExperimentsInput, compare_experiments
from inferops.tools.experiment_memory import QueryMemoryInput, query_experiment_memory
from inferops.tools.profile_cpu import ProfileCpuInput, profile_with_pyspy
from inferops.tools.propose_config import ProposeConfigInput, propose_config_patch
from inferops.tools.read_gpu_metrics import ReadGpuMetricsInput, read_gpu_metrics
from inferops.tools.run_benchmark import RunBenchmarkInput, run_benchmark
from inferops.tools.write_report import WriteReportInput, write_report_section


# Each wrapper:
#  1. Has a clear docstring the LLM uses to decide when to call it
#  2. Accepts a single dict argument (JSON-serialisable, LLM-friendly)
#  3. Delegates to the typed Pydantic function for validation + OTel span

@tool
def tool_run_benchmark(config_patch: dict, workload_name: str = "chat_short", experiment_id: str = "", persist: bool = True) -> dict:
    """
    Run one vLLM benchmark experiment with optional config overrides.

    Use this to measure the effect of a specific parameter change. Always run
    the baseline (empty config_patch) first, then variants one at a time.

    Args:
        config_patch: Dict of vLLM knobs to override, e.g. {"max_num_batched_tokens": 4096}.
                      Allowed keys: max_num_batched_tokens, max_num_seqs, max_model_len,
                      gpu_memory_utilization, enable_chunked_prefill, enable_prefix_caching.
        workload_name: One of: chat_short, long_context_qa,
                       high_concurrency_short_out, long_generation, mixed_traffic.
        experiment_id: Unique name for this run. Auto-generated if empty.
        persist: Save result to experiment memory DB (default True).

    Returns dict with throughput_rps, ttft_p50_ms, e2e_p50_ms, gpu_util_pct, etc.
    """
    import uuid
    eid = experiment_id or f"agent_{uuid.uuid4().hex[:8]}"
    out = run_benchmark(RunBenchmarkInput(experiment_id=eid, config_patch=config_patch, workload_name=workload_name, persist=persist))
    return out.model_dump()


@tool
def tool_propose_config_patch(base_experiment_id: str, param: str, value: float, rationale: str, new_experiment_id: str) -> dict:
    """
    Validate a single-parameter config change and return the patch dict.

    Always propose changes one parameter at a time to isolate causality.
    Check query_experiment_memory first to avoid re-running known configs.

    Args:
        base_experiment_id: ID of the run to use as starting point.
        param: The vLLM parameter to change (e.g. "max_num_batched_tokens").
        value: New value. Will be validated against safe ranges for RTX 3060.
        rationale: One sentence explaining the expected improvement.
        new_experiment_id: Name for the resulting experiment.

    Returns patch dict and the old value for comparison.
    """
    out = propose_config_patch(ProposeConfigInput(base_experiment_id=base_experiment_id, param=param, value=value, rationale=rationale, new_experiment_id=new_experiment_id))
    return out.model_dump()


@tool
def tool_read_gpu_metrics(duration_s: float = 2.0, device_index: int = 0) -> dict:
    """
    Sample GPU utilization and VRAM usage over a short window.

    Use before benchmarks to check baseline GPU state, or after to confirm
    saturation level. Returns avg/max utilization % and memory in GB.

    Args:
        duration_s: Sampling window in seconds (0.5–30).
        device_index: GPU index (0 for single-GPU systems).
    """
    out = read_gpu_metrics(ReadGpuMetricsInput(duration_s=duration_s, device_index=device_index))
    return out.model_dump()


@tool
def tool_profile_with_pyspy(pid: int, duration_s: int = 5, top_n: int = 10) -> dict:
    """
    Run a short CPU profile on a live process using py-spy.

    Use when GPU utilization is low but CPU usage is high, to find Python-side
    bottlenecks (tokenisation, scheduler, dispatch overhead).

    Args:
        pid: PID of the process to profile (e.g. the vLLM server).
        duration_s: Profiling duration in seconds (1–60).
        top_n: Number of top hotspot functions to return.
    """
    out = profile_with_pyspy(ProfileCpuInput(pid=pid, duration_s=duration_s, top_n=top_n))
    return out.model_dump()


@tool
def tool_analyze_bottleneck(experiment_id: str) -> dict:
    """
    Classify the primary performance bottleneck of a completed experiment.

    Returns one of: compute-bound, memory-bound, scheduling-bound, kv-bound, unknown.
    Also returns evidence signals and concrete recommendations for the next step.

    Call this after every benchmark run before proposing the next config change.

    Args:
        experiment_id: ID of the completed experiment to analyze.
    """
    out = analyze_bottleneck(AnalyzeBottleneckInput(experiment_id=experiment_id))
    return out.model_dump()


@tool
def tool_compare_experiments(experiment_id_a: str, experiment_id_b: str, metric: str = "throughput_rps", n_bootstrap: int = 2000) -> dict:
    """
    Compare two experiments using bootstrap confidence intervals.

    Use after running a variant to determine if the improvement is real or noise.
    A result is statistically significant when the CI does not straddle zero.

    Args:
        experiment_id_a: Baseline experiment ID.
        experiment_id_b: Candidate experiment ID.
        metric: One of throughput_rps, tokens_per_second, ttft_p50_ms, ttft_p99_ms,
                e2e_p50_ms, e2e_p99_ms.
        n_bootstrap: Bootstrap iterations (200–10000, default 2000).
    """
    out = compare_experiments(CompareExperimentsInput(experiment_id_a=experiment_id_a, experiment_id_b=experiment_id_b, metric=metric, n_bootstrap=n_bootstrap))
    return out.model_dump()


@tool
def tool_query_experiment_memory(workload_name: str = "", sort_by: str = "throughput_rps", top_k: int = 5) -> dict:
    """
    Query the experiment history DB for past benchmark results.

    Always call this before proposing a new config — if the config has already
    been tried, use the stored result instead of re-running.

    Args:
        workload_name: Filter by workload, or "" for all workloads.
        sort_by: Rank by this metric (throughput_rps, ttft_p50_ms, e2e_p50_ms, etc.).
        top_k: Number of results to return (1–50).
    """
    wl = workload_name or None
    out = query_experiment_memory(QueryMemoryInput(workload_name=wl, sort_by=sort_by, top_k=top_k))
    return out.model_dump()


@tool
def tool_write_report_section(section_title: str, content: str, report_path: str = "reports/agent_report.md") -> dict:
    """
    Append a section to the agent's running Markdown report.

    Call this after analyzing each experiment iteration. The report accumulates
    incrementally — do not hold the full document in context.

    Args:
        section_title: H2 heading for this section (e.g. "Iteration 3 — chunked_v2 Results").
        content: Markdown body text summarising findings and decisions.
        report_path: Path to the report file (created if absent).
    """
    out = write_report_section(WriteReportInput(section_title=section_title, content=content, report_path=report_path))
    return out.model_dump()


# Exported list for binding to a LangGraph agent
ALL_TOOLS = [
    tool_run_benchmark,
    tool_propose_config_patch,
    tool_read_gpu_metrics,
    tool_profile_with_pyspy,
    tool_analyze_bottleneck,
    tool_compare_experiments,
    tool_query_experiment_memory,
    tool_write_report_section,
]
