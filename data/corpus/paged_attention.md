---
source: PagedAttention (Kwon et al., 2023) — arXiv:2309.06180
section: Core Algorithm
---

# PagedAttention: Efficient Memory Management for LLM Serving

## Problem: KV Cache Memory Fragmentation

In transformer inference, each request generates a key-value (KV) cache that grows
token-by-token during decoding. Traditional systems pre-allocate a contiguous memory
block for the maximum possible sequence length, causing:

- **Internal fragmentation**: reserved but unused memory within a block
- **External fragmentation**: gaps between blocks that cannot be reused
- Typical GPU memory utilization: 20–40% wasted on fragmentation

## Solution: Paged Memory Management

PagedAttention divides KV cache into fixed-size **blocks** (analogous to OS virtual
memory pages). Each block holds K and V vectors for a fixed number of tokens
(block size, e.g., 16 tokens).

Key properties:
- Blocks are allocated **on demand**, not pre-allocated
- Non-contiguous blocks are chained via a **block table** per request
- Memory reclaimed immediately when a request finishes

## Impact on vLLM Parameters

### `gpu_memory_utilization`
Controls the fraction of GPU memory reserved for the KV cache pool. With PagedAttention,
higher values are safe because fragmentation is minimal. On RTX 3060 (6 GB):
- 0.80 leaves ~1.2 GB for model weights + overhead
- 0.85 is the practical ceiling before OOM under burst load

### `max_model_len`
Maximum sequence length (prefill + decode). Directly determines KV cache block count
needed per request. Reducing max_model_len from 32768 → 2048 on Qwen2.5 cuts
KV memory by ~16× per concurrent slot.

### `max_num_seqs`
Maximum concurrent sequences. Each occupies PagedAttention blocks. With block_size=16
and max_model_len=2048, each request needs at most 128 blocks. Total blocks available
= floor(gpu_memory_utilization × total_vram / block_size_bytes).

## Quantitative Results (from paper)

- Throughput improvement vs HuggingFace: **1.7–2.7×**
- Throughput improvement vs Orca: **2.7–3.8×**
- Memory waste reduced from 60–80% to under 4%

## Interaction with Chunked Prefill

PagedAttention blocks are filled during prefill. Chunked prefill splits long prefill
sequences across multiple scheduler iterations, reducing the latency spike caused by
filling many blocks at once. Enabling `enable_chunked_prefill=True` with PagedAttention
is most beneficial when prefill tokens >> decode tokens per batch.
