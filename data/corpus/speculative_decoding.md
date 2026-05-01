---
source: Speculative Decoding (Leviathan et al., 2023) — arXiv:2211.17192
section: Algorithm Overview
---

# Speculative Decoding for LLM Inference

## Core Idea

Autoregressive decoding generates one token per forward pass — a sequential bottleneck.
Speculative decoding uses a small **draft model** to propose K tokens in parallel,
then verifies them all in a single forward pass of the **target model**.

- If all K tokens are accepted: K tokens generated at the cost of 1 target pass
- If token i is rejected: fall back to sampling from target at position i
- Expected tokens per target pass: γ = α/(1−αK+1) where α = acceptance rate

## Practical Impact

For decode-heavy workloads (long generation, low concurrency):
- Throughput improvement: **2–3×** when draft and target share vocabulary
- Best gains when target model is bottlenecked by decode (not prefill)
- Requires draft model that fits alongside target on the same GPU

## Applicability to This Setup (RTX 3060, 6 GB)

With Qwen2.5-0.5B as the target model:
- Draft model must be smaller (e.g., Qwen2.5-0.5B itself has no smaller sibling)
- Speculative decoding is **not directly applicable** without a draft model
- Alternative: `ngram` speculation (predicts tokens from the prompt itself)

## vLLM Support

vLLM supports speculative decoding via `speculative_model` and `num_speculative_tokens`
parameters. These are outside the current optimisation search space but worth
considering if a smaller draft model is available.

## Relevance to `long_generation` Workload

The `long_generation` workload (512-token outputs) is the most decode-heavy scenario.
Primary metric is `tokens_per_second`. Without speculative decoding, gains come from:
- Reducing scheduling overhead: `max_num_seqs` tuning
- KV cache efficiency: `gpu_memory_utilization`, `max_model_len`
- Chunked prefill: less useful here (short prompts, long decode)
