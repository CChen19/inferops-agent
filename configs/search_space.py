"""
Parameter search space for Phase 1 baseline sweep.

Default + 3 variants per workload = 4 configs × 2 workloads = 8 runs total.

Variants chosen to have a measurable impact on a 3060 (6 GB) with Qwen2.5-0.5B:
  default   — safe baseline, vLLM stock settings
  chunked   — enable_chunked_prefill: reduces TTFT for long sequences
  prefix    — enable_prefix_caching: amortizes repeated prefix cost
  big_batch — larger max_num_batched_tokens: higher peak throughput, more GPU pressure
"""

from __future__ import annotations

from inferops.schemas import ExperimentConfig, ModelSize, SchedulerPolicy


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def make_configs(workload) -> list[ExperimentConfig]:  # workload: WorkloadSpec
    base = dict(
        model_name=MODEL,
        model_size=ModelSize.HALF_B,
        workload=workload,
        gpu_memory_utilization=0.85,
        max_num_seqs=256,
        enforce_eager=False,
        scheduler_policy=SchedulerPolicy.FCFS,
    )

    return [
        ExperimentConfig(
            experiment_id=f"default_{workload.name}",
            max_num_batched_tokens=2048,
            enable_chunked_prefill=False,
            tags={"variant": "default"},
            **base,
        ),
        ExperimentConfig(
            experiment_id=f"chunked_{workload.name}",
            max_num_batched_tokens=2048,
            enable_chunked_prefill=True,
            tags={"variant": "chunked_prefill"},
            **base,
        ),
        ExperimentConfig(
            experiment_id=f"prefix_cache_{workload.name}",
            max_num_batched_tokens=2048,
            enable_chunked_prefill=False,
            tags={"variant": "prefix_caching"},
            **base,
        ),
        ExperimentConfig(
            experiment_id=f"big_batch_{workload.name}",
            max_num_batched_tokens=4096,
            enable_chunked_prefill=False,
            tags={"variant": "big_batch"},
            **base,
        ),
    ]
