---
source: vLLM Documentation — Scheduler & Batching
section: Scheduler Internals
---

# vLLM Scheduler: Batching and Throughput

## Scheduler Policy

vLLM uses a **First-Come First-Served (FCFS)** scheduler by default. Requests are
processed in arrival order. Key parameters:

### `max_num_seqs`
Maximum concurrent sequences in a single forward pass. Higher values:
- Increase GPU utilisation (more parallelism)
- Increase KV memory pressure (more blocks allocated simultaneously)
- May increase TTFT if many sequences prefill simultaneously

Recommended range: 64–256. Diminishing returns above GPU memory saturation point.

### `max_num_batched_tokens`
Total token budget per scheduler iteration (prefill + decode combined). Larger values:
- Allow longer prompts to be processed in one pass
- Increase compute per iteration → higher throughput if GPU is compute-bound
- Must be ≥ max_model_len (vLLM hard constraint)

## Bottleneck Identification

### Compute-bound
Signs: GPU utilisation >85%, throughput scales with `max_num_batched_tokens`
Fix: Increase `max_num_batched_tokens` (3072 → 4096), disable unnecessary chunking

### Memory-bound
Signs: GPU memory near limit, OOM errors, evictions in logs
Fix: Reduce `max_num_seqs`, reduce `max_model_len`, lower `gpu_memory_utilization`

### Scheduling-bound
Signs: High TTFT variance, low GPU utilisation (<60%), many short requests queuing
Fix: Increase `max_num_seqs`, enable `enable_chunked_prefill` for fairness

### KV-bound
Signs: Moderate GPU util, high TTFT for long prompts specifically
Fix: Enable `enable_prefix_caching`, increase `gpu_memory_utilization` carefully

## Batch Composition Strategy

For mixed workloads (`mixed_traffic`): short and long requests compete for the token
budget. Chunked prefill + FCFS tends to favour short requests (they complete prefill
in one chunk). Consider that `max_num_batched_tokens=4096` allows more long requests
to share the budget without completely blocking short ones.
