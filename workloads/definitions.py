"""
Workload prompt sets for benchmarking.

chat_short:      ~128-token prompts, high concurrency → tests scheduler throughput
long_context_qa: ~1024-token prompts, low concurrency → tests prefill efficiency + KV pressure
"""

from __future__ import annotations

from inferops.schemas import WorkloadSpec

# ---------------------------------------------------------------------------
# WorkloadSpec definitions
# ---------------------------------------------------------------------------

CHAT_SHORT = WorkloadSpec(
    name="chat_short",
    prompt_template="",       # prompts generated below
    num_requests=60,
    concurrency=16,
    input_len=128,
    output_len=128,
    distribution="uniform",
)

LONG_CONTEXT_QA = WorkloadSpec(
    name="long_context_qa",
    prompt_template="",
    num_requests=20,
    concurrency=4,
    input_len=1024,
    output_len=256,
    distribution="uniform",
)

# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

_SHORT_TEMPLATES = [
    "Explain the concept of {topic} in simple terms.",
    "What are the key differences between {topic} and {alt}?",
    "Give me a brief overview of {topic}.",
    "Summarize the main ideas behind {topic}.",
    "How does {topic} work at a high level?",
]

_TOPICS = [
    ("gradient descent", "stochastic gradient descent"),
    ("attention mechanism", "convolutional layers"),
    ("transformer architecture", "RNN"),
    ("batch normalization", "layer normalization"),
    ("reinforcement learning", "supervised learning"),
    ("quantization", "pruning"),
    ("KV cache", "flash attention"),
    ("speculative decoding", "beam search"),
    ("RLHF", "DPO"),
    ("tokenization", "embedding"),
]

_LONG_CONTEXT_BASE = """\
You are an expert in machine learning systems. Below is a detailed technical passage.

{passage}

---
Question: Based on the passage above, provide a comprehensive analysis of the key \
performance trade-offs described. Consider throughput, latency, memory efficiency, \
and scalability. What are the most important insights for practitioners building \
large-scale inference systems?

Answer:"""

_PASSAGE_CHUNK = (
    "Modern LLM inference systems face fundamental trade-offs between throughput and latency. "
    "Continuous batching allows the serving engine to add new requests to an in-flight batch "
    "without waiting for all sequences to finish, dramatically improving GPU utilization. "
    "However, long sequences compete for KV cache memory with short sequences, requiring "
    "careful memory management policies. PagedAttention addresses this by allocating KV cache "
    "in non-contiguous pages, similar to virtual memory in operating systems. "
    "Chunked prefill separates the prefill and decode phases, allowing the scheduler to "
    "interleave prefill chunks with decode steps, reducing head-of-line blocking. "
    "Prefix caching stores KV activations for repeated prompt prefixes, amortizing prefill "
    "cost across requests that share a common prefix such as a system prompt. "
)


def make_chat_short_prompts(n: int) -> list[str]:
    """Generate ~128-token chat prompts (varied topics, no exact repeats)."""
    prompts = []
    for i in range(n):
        tmpl = _SHORT_TEMPLATES[i % len(_SHORT_TEMPLATES)]
        topic, alt = _TOPICS[i % len(_TOPICS)]
        prompts.append(tmpl.format(topic=topic, alt=alt))
    return prompts


def make_long_context_prompts(n: int, target_tokens: int = 1024) -> list[str]:
    """Generate ~1024-token prompts by repeating the passage chunk."""
    # ~15 words per chunk, ~1.3 tokens/word → ~20 tokens per chunk
    # Need ~50 chunks for 1024 tokens
    repeats = max(1, target_tokens // 20)
    passage = (_PASSAGE_CHUNK * repeats)[:target_tokens * 5]  # rough char budget
    base = _LONG_CONTEXT_BASE.format(passage=passage)
    # Add index variation so prefix caching doesn't trivially cache everything
    return [f"{base}\n\n[Context ID: {i}]" for i in range(n)]


def get_prompts(workload: WorkloadSpec) -> list[str]:
    """Return prompt list (warmup + measure) for a given workload."""
    total = workload.num_requests + 10  # 10 extra for warmup
    if workload.name == "chat_short":
        return make_chat_short_prompts(total)
    elif workload.name == "long_context_qa":
        return make_long_context_prompts(total, target_tokens=workload.input_len)
    else:
        raise ValueError(f"Unknown workload: {workload.name}")
