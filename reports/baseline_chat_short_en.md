# InferOps Baseline Report — chat_short

Generated: 2026-04-28  
Model: Qwen/Qwen2.5-0.5B-Instruct | Hardware: RTX 3060 Laptop 6GB, WSL2, CUDA 12.8  
Config: gpu_memory_utilization=0.80, max_model_len=2048, max_num_seqs=128  
Workload: 60 requests, concurrency=16, ~15-token prompts, 128 output tokens

## Results

| Variant | RPS | Tok/s | TTFT p50 | TTFT p99 | E2E p50 | E2E p99 | GPU util |
|---|---|---|---|---|---|---|---|
| default | 14.96 | 1929 | 48ms | 69ms | 1015ms | 1032ms | 86% |
| chunked_prefill | 16.68 | 2151 | **67ms** | 79ms | 914ms | 951ms | 84% |
| prefix_caching | 16.84 | 2167 | 49ms | **175ms** | 878ms | 1001ms | 81% |
| **big_batch** | **17.23** | **2222** | 52ms | **65ms** | **883ms** | **892ms** | 86% |

Success rate: 60/60 across all variants

## Analysis

**big_batch (max_num_batched_tokens=4096) wins overall:**
- Throughput +15% over default (17.23 vs 14.96 rps)
- Best E2E latency at both p50 and p99 (883ms / 892ms) — tightest tail
- TTFT penalty is only 4ms (52ms vs 48ms), well worth the trade-off
- Reason: larger token batches allow more efficient GPU matrix ops; decode kernels run at higher utilization

**chunked_prefill hurts short sequences:**
- TTFT p50 increases 48ms → 67ms (+40%)
- Expected behavior: chunked prefill interleaves prefill chunks with decode steps, which reduces TTFT variance for *long* prefills but adds scheduling overhead for short ones (~15 tokens)
- Throughput gain (+11%) comes from finer-grained scheduler batching

**prefix_caching introduces p99 spikes when there are no shared prefixes:**
- TTFT p99 spikes 69ms → 175ms (+153%)
- chat_short prompts are all unique — cache hit rate is 0%, but every request still pays the cost of hashing and cache lookup
- Prefix caching is the right choice when requests share a long common prefix (e.g., a fixed system prompt); the longer the shared prefix, the larger the benefit

**Why E2E ≈ 900ms:**
- Decode dominates: 128 tokens × ~7ms/token ≈ 896ms
- TTFT is only ~5% of E2E — this workload is decode-bound, not prefill-bound

## WSL2 Hardware Notes

- nvidia-smi reports ~5.97 GB GPU memory used, but vLLM actually holds only ~3.7 GB
  (the remaining ~2.2 GB is the Windows Desktop Window Manager, invisible to PyTorch/CUDA)
- gpu_memory_utilization=0.80 is the safe ceiling for this setup — no OOM observed
- max_model_len=2048 is critical: without it, vLLM defaults to Qwen's native 32768-token
  context, causing KV cache block allocation to OOM immediately on a 6 GB card

## Next Steps

- [ ] Run long_context_qa (1024-token inputs) — chunked_prefill expected to recover its advantage on long prefills
- [ ] Check whether big_batch remains optimal or hits memory pressure on long_context_qa
- [ ] Phase 2: wire this pipeline into a LangGraph agent for automated config search
