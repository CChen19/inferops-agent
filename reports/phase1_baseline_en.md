# InferOps Phase 1 Baseline Report

Generated: 2026-04-28  
Model: Qwen/Qwen2.5-0.5B-Instruct  
Hardware: NVIDIA RTX 3060 Laptop 6GB, WSL2 Ubuntu, CUDA 12.8  
vLLM version: 0.18.0

---

## Experiment Setup

**Fixed parameters**

| Parameter | Value |
|---|---|
| gpu_memory_utilization | 0.80 |
| max_model_len | 2048 |
| max_num_seqs | 128 |
| dtype | auto (bfloat16) |
| enforce_eager | False |

**Variants (search space)**

| Variant | max_num_batched_tokens | enable_chunked_prefill | enable_prefix_caching |
|---|---|---|---|
| default | 2048 | False | False |
| chunked | 2048 | True | False |
| prefix_cache | 2048 | False | True |
| big_batch | 4096 | False | False |

**Workloads**

| Workload | Requests | Concurrency | Input tokens | Output tokens | Scenario |
|---|---|---|---|---|---|
| chat_short | 60 | 16 | ~15 | 128 | High-concurrency short chat |
| long_context_qa | 20 | 4 | ~1024 | 256 | Long-document QA |

---

## Results

### chat_short

| Variant | RPS | Tok/s | TTFT p50 | TTFT p99 | E2E p50 | E2E p99 | GPU util |
|---|---|---|---|---|---|---|---|
| default | 14.96 | 1929 | 48ms | 69ms | 1015ms | 1032ms | 86% |
| chunked | 16.68 | 2151 | 67ms ↑ | 79ms | 914ms | 951ms | 84% |
| prefix_cache | 16.84 | 2167 | 49ms | 175ms ↑↑ | 878ms | 1001ms | 81% |
| **big_batch** | **17.23** | **2222** | **52ms** | **65ms** | **883ms** | **892ms** | **86%** |

Success rate: 4 × 60/60 ✓

### long_context_qa

| Variant | RPS | Tok/s | TTFT p50 | TTFT p99 | E2E p50 | E2E p99 | GPU util |
|---|---|---|---|---|---|---|---|
| default | 2.43 | 625 | 39ms | 46ms | 1643ms | 1658ms | 92% |
| **chunked** | **2.90** | **745** | **38ms** | **43ms** | **1379ms** | **1392ms** | **89%** |
| **prefix_cache** | **2.90** | **745** | 39ms | 51ms | 1374ms | 1397ms | 92% |
| big_batch | 2.35 | 604 | 41ms | 52ms | 1703ms ↑ | 1711ms ↑ | 94% |

Success rate: 4 × 20/20 ✓

---

## Analysis

### 1. big_batch: wins on short sequences, hurts on long ones

On chat_short, big_batch leads with +15% RPS and the tightest tail latency. Larger token batches saturate GPU matrix ops more efficiently, and decode kernels run at higher utilization.

On long_context_qa, big_batch finishes last — RPS drops 3% below default (2.35 vs 2.43) and E2E latency is worst (1703ms). The reason: at 4096 tokens per step, long-sequence KV access overhead grows significantly. The GPU hits a memory bandwidth ceiling (94% utilization) while effective throughput falls.

**Takeaway: max_num_batched_tokens must be tuned to match sequence length — there is no universal optimum.**

### 2. chunked_prefill: hurts short sequences, helps long ones

- chat_short: TTFT p50 rises 48ms → 67ms (+40%). Short prefills are already fast; splitting them into chunks adds scheduler overhead with no benefit.
- long_context_qa: +19% RPS, E2E drops 1643ms → 1379ms (-16%). A 1024-token prefill chunked and interleaved with decode steps eliminates head-of-line blocking.

**Takeaway: chunked_prefill benefit scales with input length — disable it for short-sequence workloads.**

### 3. prefix_caching: only pays off when prefixes are actually shared

- chat_short: TTFT p99 spikes 69ms → 175ms (+153%). Prompts are all unique, so cache hit rate is ~0%, but every request still pays for hashing and cache lookup.
- long_context_qa: ties with chunked for first place (2.90 rps, E2E 1374ms). All 20 requests share the same ~5000-character background passage — the shared prefix is cached after the first request, reducing prefill cost for subsequent ones.

**Takeaway: prefix_caching is valuable when requests share a long common prefix (e.g., a fixed system prompt); turn it off for random-prompt workloads.**

### 4. The two workloads are fundamentally different

| Metric | chat_short | long_context_qa |
|---|---|---|
| TTFT | ~48ms (light prefill) | ~39ms (heavy prefill, but lower concurrency) |
| E2E | ~900ms | ~1400ms |
| E2E breakdown | 95% decode | decode-dominant, prefill costs more |
| GPU utilization | 81–86% | 89–94% |
| Bottleneck | decode throughput | decode + memory bandwidth |

Counterintuitively, long_context_qa has *lower* TTFT than chat_short (39ms vs 48ms). With only 4 concurrent requests vs 16, each batch has less competition — the full 1024-token prefill executes without fragmentation.

---

## Recommendations

| Scenario | Best variant | Key parameter |
|---|---|---|
| High-concurrency short chat | **big_batch** | max_num_batched_tokens=4096 |
| Long-document / long-context | **chunked** or **prefix_cache** | enable_chunked_prefill=True or enable_prefix_caching=True |
| Mixed workload | **chunked** | Acceptable on both short and long sequences |

---

## WSL2 Hardware Notes

- `nvidia-smi` reports ~5.9 GB GPU memory used, but vLLM actually holds ~3.7 GB; the remaining ~2.2 GB is the Windows Desktop Window Manager, invisible to PyTorch/CUDA
- `gpu_memory_utilization=0.80` is the safe ceiling for a 6 GB card in WSL2
- `max_model_len=2048` must be set explicitly — Qwen2.5's native context is 32768 tokens, and without this flag vLLM OOMs during KV cache block allocation
- vLLM logs `pin_memory=False` (WSL detected) — expected, no functional impact

---

## Next Steps (Phase 2)

- Wire the benchmark pipeline into a LangGraph agent (plan → run → analyze)
- Agent autonomously selects the next config based on previous results, closing the optimization loop
- Add multi-objective trade-off analysis (throughput vs tail latency) to find Pareto-optimal configs
