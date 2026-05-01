---
source: vLLM Documentation — Automatic Prefix Caching (APC)
section: KV Cache Optimisation
---

# Automatic Prefix Caching (APC) in vLLM

## Mechanism

Prefix caching (`enable_prefix_caching=True`) reuses KV cache blocks computed for
repeated prompt prefixes across requests. vLLM uses a hash of the token sequence to
identify cacheable blocks.

Example: A system prompt repeated across 100 chat requests is computed once;
subsequent requests skip prefill for those tokens and load the cached blocks.

## When It Helps

Prefix caching is effective when requests share a **common prefix**:
- System prompts (e.g., "You are a helpful assistant...")
- Few-shot examples in the prompt
- RAG retrieved context that is stable across a session
- Document QA where the same document is queried multiple times

**Not effective** for:
- Diverse user queries with unique prefixes
- Short prompts where caching overhead exceeds savings
- Single-request inference (no repeated prefixes)

## Interaction with `long_context_qa` Workload

The `long_context_qa` workload uses 1024-token prompts. If the first 512 tokens are a
shared system prompt, prefix caching can halve the prefill compute for all but the first
request. Observed improvement: +17% throughput (2.41 → 2.83 RPS) with
`enable_prefix_caching=True`.

## Interaction with Chunked Prefill

Prefix caching and chunked prefill are **compatible** and often complementary:
- Prefix caching skips prefill for cached blocks
- Chunked prefill handles the remaining non-cached tokens incrementally

## Memory Overhead

Cached blocks occupy KV cache space. vLLM evicts LRU blocks when the pool is full.
On a 6 GB GPU with `gpu_memory_utilization=0.80`, the cache pool holds approximately
400–600 blocks (block_size=16), supporting ~6400–9600 cached tokens.
