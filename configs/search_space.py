"""
Parameter search space for Phase 1 baseline sweep.

Default + 3 variants per workload = 4 configs × 2 workloads = 8 runs total.

Hardware constraints (RTX 3060 Laptop, 6 GB, WSL2):
  - PyTorch sees ~5 GB usable (Windows display takes ~1 GB, invisible to CUDA)
  - gpu_memory_utilization=0.80 → 4.8 GB budget → safe KV headroom
  - max_model_len=2048: prevents vLLM from reserving KV blocks for Qwen's 32k context
  - max_num_seqs=128: realistic for our concurrency levels (≤16)

Variants:
  default       — baseline
  chunked       — enable_chunked_prefill: reduces TTFT variance on long prefills
  prefix_cache  — enable_prefix_caching: amortizes shared prompt prefix cost
  big_batch     — max_num_batched_tokens=4096: higher batch → more GPU saturation
"""

from __future__ import annotations

from inferops.schemas import ExperimentConfig, ModelSize, SchedulerPolicy


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def make_configs(workload) -> list[ExperimentConfig]:  # workload: WorkloadSpec
    # long_context_qa needs larger max_model_len (1024 in + 256 out = 1280 min)
    max_model_len = 2048 if workload.name == "chat_short" else 2048

    base = dict(
        model_name=MODEL,
        model_size=ModelSize.HALF_B,
        workload=workload,
        gpu_memory_utilization=0.80,
        max_num_seqs=128,
        max_model_len=max_model_len,
        enforce_eager=False,
        scheduler_policy=SchedulerPolicy.FCFS,
    )

    return [
        ExperimentConfig(
            experiment_id=f"default_{workload.name}",
            max_num_batched_tokens=2048,
            enable_chunked_prefill=False,
            enable_prefix_caching=False,
            tags={"variant": "default"},
            **base,
        ),
        ExperimentConfig(
            experiment_id=f"chunked_{workload.name}",
            max_num_batched_tokens=2048,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            tags={"variant": "chunked"},
            **base,
        ),
        ExperimentConfig(
            experiment_id=f"prefix_cache_{workload.name}",
            max_num_batched_tokens=2048,
            enable_chunked_prefill=False,
            enable_prefix_caching=True,
            tags={"variant": "prefix_cache"},
            **base,
        ),
        ExperimentConfig(
            experiment_id=f"big_batch_{workload.name}",
            max_num_batched_tokens=4096,
            enable_chunked_prefill=False,
            enable_prefix_caching=False,
            tags={"variant": "big_batch"},
            **base,
        ),
    ]
