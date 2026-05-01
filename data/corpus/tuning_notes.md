---
source: Tuning Notes — RTX 3060 Laptop (6 GB, WSL2)
section: Hardware-Specific Observations
---

# Qwen2.5-0.5B on RTX 3060 Laptop — Tuning Notes

## Hardware Constraints

- **GPU**: NVIDIA RTX 3060 Laptop, 6 GB GDDR6
- **VRAM available to PyTorch**: ~5.7 GB (Windows DWM consumes ~300 MB)
- **Environment**: WSL2, CUDA 12.x, vLLM 0.8+
- **Model**: Qwen/Qwen2.5-0.5B-Instruct (~1 GB weights in float16)
- **Remaining for KV cache**: ~4.5 GB at `gpu_memory_utilization=0.80`

## Validated Safe Parameter Ranges

| Parameter | Min | Max | Notes |
|---|---|---|---|
| `gpu_memory_utilization` | 0.50 | 0.85 | >0.85 causes OOM under burst |
| `max_num_seqs` | 16 | 256 | >256 causes scheduling stalls |
| `max_num_batched_tokens` | 2048 | 8192 | Must be ≥ max_model_len |
| `max_model_len` | 512 | 4096 | Qwen2.5 native 32768 → OOM |

## Key Findings from Grid Sweep (12 configs × 5 workloads)

### chat_short (128-token prompts, 128-token outputs)
- Best config: `max_num_batched_tokens=4096`, chunked=False, prefix=False
- Insight: compute-bound workload; larger batch tokens improve GPU utilisation
- Chunked prefill adds overhead with no benefit (prompts are short)

### long_context_qa (1024-token prompts, 256-token outputs)
- Best config: `max_num_batched_tokens=4096`, chunked=False, prefix=True
- Insight: KV-bound; prefix caching reduces repeated prefill for shared system prompts
- TTFT p99 improved 22% with prefix caching alone

### high_concurrency_short_out (64-token prompts, 32-token outputs, concurrency=32)
- Best config: `max_num_batched_tokens=4096`, chunked=False, prefix=False
- Insight: scheduling-bound; larger batch budget handles the burst queue

### long_generation (256-token prompts, 512-token outputs)
- Best config: `max_num_batched_tokens=2048`, chunked=True, prefix=False
- Insight: decode-heavy; chunked prefill frees GPU cycles for decode iterations

### mixed_traffic (50% short, 50% long prompts)
- Best config: `max_num_batched_tokens=4096`, chunked=False, prefix=False
- Insight: larger batch budget benefits both short and long; chunking not needed

## Observed OOM Conditions

- `gpu_memory_utilization=0.90` + `max_num_seqs=256`: OOM at startup
- `max_model_len=4096` + `max_num_seqs=256`: OOM during load
- Safe operating envelope: keep product of max_num_seqs × max_model_len ≤ 512,000 tokens
