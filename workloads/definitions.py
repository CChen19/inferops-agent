"""
Workload prompt sets for benchmarking.

chat_short:                ~128-token prompts, high concurrency → tests scheduler throughput
long_context_qa:           ~1024-token prompts, low concurrency → tests prefill efficiency + KV pressure
high_concurrency_short_out: ~64-token prompts, very high concurrency + short output → stress scheduler
long_generation:           ~256-token prompts, very long output (512 tok) → stress decode + KV cache
mixed_traffic:             50% short + 50% long prompts, medium concurrency → stress scheduler fairness
"""

from __future__ import annotations

from inferops.schemas import WorkloadSpec

# ---------------------------------------------------------------------------
# WorkloadSpec definitions
# ---------------------------------------------------------------------------

CHAT_SHORT = WorkloadSpec(
    name="chat_short",
    prompt_template="",
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

HIGH_CONCURRENCY_SHORT_OUT = WorkloadSpec(
    name="high_concurrency_short_out",
    prompt_template="",
    num_requests=120,
    concurrency=32,
    input_len=64,
    output_len=32,
    distribution="uniform",
)

LONG_GENERATION = WorkloadSpec(
    name="long_generation",
    prompt_template="",
    num_requests=10,
    concurrency=2,
    input_len=256,
    output_len=512,
    distribution="uniform",
)

MIXED_TRAFFIC = WorkloadSpec(
    name="mixed_traffic",
    prompt_template="",
    num_requests=40,
    concurrency=8,
    input_len=256,   # average; 50% short (~64 tok), 50% long (~512 tok)
    output_len=128,
    distribution="uniform",
)

ALL_WORKLOADS = [CHAT_SHORT, LONG_CONTEXT_QA, HIGH_CONCURRENCY_SHORT_OUT, LONG_GENERATION, MIXED_TRAFFIC]

# ---------------------------------------------------------------------------
# Prompt generation — chat_short
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


def make_chat_short_prompts(n: int) -> list[str]:
    """Generate ~128-token chat prompts (varied topics, no exact repeats)."""
    prompts = []
    for i in range(n):
        tmpl = _SHORT_TEMPLATES[i % len(_SHORT_TEMPLATES)]
        topic, alt = _TOPICS[i % len(_TOPICS)]
        prompts.append(tmpl.format(topic=topic, alt=alt))
    return prompts


# ---------------------------------------------------------------------------
# Prompt generation — long_context_qa
# ---------------------------------------------------------------------------

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


def make_long_context_prompts(n: int, target_tokens: int = 1024) -> list[str]:
    """Generate ~1024-token prompts by repeating the passage chunk."""
    repeats = max(1, target_tokens // 20)
    passage = (_PASSAGE_CHUNK * repeats)[:target_tokens * 5]
    base = _LONG_CONTEXT_BASE.format(passage=passage)
    return [f"{base}\n\n[Context ID: {i}]" for i in range(n)]


# ---------------------------------------------------------------------------
# Prompt generation — high_concurrency_short_out
# ---------------------------------------------------------------------------

_QUICK_TEMPLATES = [
    "What is {}?",
    "Define {}.",
    "Name three key facts about {}.",
    "How does {} work?",
    "Give one example of {}.",
]

_QUICK_TOPICS = [
    "gradient descent", "batch normalization", "attention mechanism",
    "learning rate", "dropout", "softmax", "cross-entropy loss",
    "backpropagation", "embedding layer", "residual connection",
    "weight decay", "momentum", "the Adam optimizer", "early stopping",
    "data augmentation", "transfer learning", "fine-tuning",
    "tokenization", "positional encoding", "layer normalization",
]


def make_high_concurrency_prompts(n: int) -> list[str]:
    """Generate very short prompts (~64 tokens) for high-concurrency stress testing."""
    prompts = []
    for i in range(n):
        tmpl = _QUICK_TEMPLATES[i % len(_QUICK_TEMPLATES)]
        topic = _QUICK_TOPICS[i % len(_QUICK_TOPICS)]
        prompts.append(tmpl.format(topic))
    return prompts


# ---------------------------------------------------------------------------
# Prompt generation — long_generation
# ---------------------------------------------------------------------------

_LONG_GEN_TEMPLATES = [
    "Write a comprehensive technical explanation of {topic}. Cover the underlying principles, "
    "key design decisions, common failure modes, and best practices for production use. "
    "Include concrete examples and quantitative comparisons where relevant.",
    "Provide an in-depth analysis of {topic} from first principles. Explain the mathematical "
    "foundations, algorithmic details, computational complexity, and trade-offs compared "
    "to alternative approaches. Discuss recent advances and open research questions.",
    "You are a senior ML systems engineer. Explain {topic} to a new team member. "
    "Start with the intuition, build up to the full technical picture, describe "
    "how to tune it in practice, and list the top 5 pitfalls to avoid.",
]

_LONG_GEN_TOPICS = [
    "the transformer attention mechanism and its modern variants",
    "vLLM's PagedAttention memory management system",
    "continuous batching in large language model inference",
    "KV cache optimization strategies for autoregressive decoding",
    "speculative decoding and its impact on inference latency",
    "the trade-offs between quantization methods for LLM inference",
    "flash attention and memory-efficient attention implementations",
    "tensor parallelism strategies for multi-GPU LLM serving",
    "chunked prefill and its effect on scheduling fairness",
    "prefix caching and shared prompt optimization techniques",
]


def make_long_generation_prompts(n: int) -> list[str]:
    """Generate prompts designed to elicit ~512-token responses."""
    prompts = []
    for i in range(n):
        tmpl = _LONG_GEN_TEMPLATES[i % len(_LONG_GEN_TEMPLATES)]
        topic = _LONG_GEN_TOPICS[i % len(_LONG_GEN_TOPICS)]
        prompts.append(tmpl.format(topic=topic))
    return prompts


# ---------------------------------------------------------------------------
# Prompt generation — mixed_traffic
# ---------------------------------------------------------------------------

def make_mixed_traffic_prompts(n: int) -> list[str]:
    """Alternate short (~64 tok) and long (~256 tok) prompts, 50/50 split."""
    prompts = []
    for i in range(n):
        if i % 2 == 0:
            tmpl = _QUICK_TEMPLATES[i % len(_QUICK_TEMPLATES)]
            topic = _QUICK_TOPICS[i % len(_QUICK_TOPICS)]
            prompts.append(tmpl.format(topic))
        else:
            tmpl = _SHORT_TEMPLATES[i % len(_SHORT_TEMPLATES)]
            topic, alt = _TOPICS[i % len(_TOPICS)]
            prompts.append(tmpl.format(topic=topic, alt=alt))
    return prompts


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def get_prompts(workload: WorkloadSpec) -> list[str]:
    """Return prompt list (warmup + measure) for a given workload."""
    total = workload.num_requests + 10  # 10 extra for warmup
    if workload.name == "chat_short":
        return make_chat_short_prompts(total)
    elif workload.name == "long_context_qa":
        return make_long_context_prompts(total, target_tokens=workload.input_len)
    elif workload.name == "high_concurrency_short_out":
        return make_high_concurrency_prompts(total)
    elif workload.name == "long_generation":
        return make_long_generation_prompts(total)
    elif workload.name == "mixed_traffic":
        return make_mixed_traffic_prompts(total)
    else:
        raise ValueError(f"Unknown workload: {workload.name}")
