---
source: vLLM Documentation — Chunked Prefill
section: Scheduler Configuration
---

# Chunked Prefill in vLLM

## What It Is

Chunked prefill (`enable_chunked_prefill=True`) allows vLLM to split a long prefill
request across multiple scheduler iterations, interleaving it with decode steps from
other requests.

Without chunked prefill:
- A 1024-token prefill monopolises the GPU for one full forward pass
- All decode requests stall, causing TTFT spikes for other users

With chunked prefill:
- Prefill is split into chunks of ≤ `max_num_batched_tokens` tokens per iteration
- Decode requests run between chunks → lower tail latency, more predictable TTFT

## When to Enable

| Scenario | Recommendation |
|---|---|
| Long prompts (>512 tokens), low concurrency | ✅ Enable — reduces TTFT p99 significantly |
| Short prompts (<128 tokens), high concurrency | ⚠️ Neutral or slight overhead |
| Mixed traffic (short + long prompts) | ✅ Enable — improves fairness |
| Maximising pure throughput, uniform short prompts | ❌ Disable — chunking adds scheduling overhead |

## Interaction with `max_num_batched_tokens`

Chunked prefill uses `max_num_batched_tokens` as the chunk budget per iteration.
Recommended values:
- `max_num_batched_tokens=2048`: conservative, lower TTFT but more iterations
- `max_num_batched_tokens=4096`: higher throughput, slightly higher TTFT

**Constraint**: `max_num_batched_tokens >= max_model_len` (vLLM hard requirement).
On this setup with `max_model_len=2048`, minimum is `max_num_batched_tokens=2048`.

## Quantitative Effect (RTX 3060, Qwen2.5-0.5B)

In `long_context_qa` workload (1024-token prompts):
- TTFT p99 without chunked prefill: ~820ms
- TTFT p99 with chunked prefill + max_num_batched_tokens=4096: ~580ms (−29%)
- Throughput change: −5% (acceptable trade-off for interactive use cases)

In `chat_short` workload (128-token prompts):
- Chunked prefill effect: minimal (prompts already short)
- May slightly reduce throughput due to scheduling overhead
