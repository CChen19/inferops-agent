# inferops-agent

Autonomous LLM inference optimization agent — uses LangGraph to iteratively tune vLLM serving parameters (batch size, scheduler policy, KV-cache utilization) and maximize throughput/latency on a local RTX 3060.

## Hardware
- GPU: NVIDIA RTX 3060 Laptop (6 GB VRAM)
- Runtime: WSL2 Ubuntu, CUDA 12.8

## Stack
| Layer | Choice |
|---|---|
| Inference engine | vLLM 0.18 |
| Agent framework | LangGraph 1.x |
| Data validation | Pydantic v2 |
| Experiment tracking | MLflow (SQLite backend) |
| Observability | OpenTelemetry → console |
| Dependency mgmt | uv |

## Quick start

```bash
# 1. Create + activate venv
uv venv --python /home/chris/miniconda3/envs/vllm-dev/bin/python3.11 .venv
source .venv/bin/activate
uv pip install openai pynvml matplotlib numpy langgraph langchain langchain-community \
               pydantic mlflow opentelemetry-sdk httpx tenacity rich typer

# 2. Verify environment (GPU, CUDA, vLLM, LangGraph, MLflow)
bash scripts/verify_env.sh

# 3. Run full baseline sweep — starts/stops vLLM automatically per variant
python scripts/run_baseline.py --workload both

# 4. Run a specific workload or subset of variants
python scripts/run_baseline.py --workload chat_short --variants default,big_batch
```

> **Note:** The benchmark script launches vLLM using the `vllm-dev` conda env directly —
> no need to activate it manually.

## Phase 1 results (Qwen2.5-0.5B, RTX 3060 Laptop)

### chat_short — 60 req, concurrency=16, ~15-token inputs, 128-token outputs

| Variant | RPS | TTFT p50 | TTFT p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|
| default | 14.96 | 48ms | 69ms | 1015ms | 1032ms |
| chunked | 16.68 | 67ms ↑ | 79ms | 914ms | 951ms |
| prefix_cache | 16.84 | 49ms | 175ms ↑↑ | 878ms | 1001ms |
| **big_batch** | **17.23** | 52ms | **65ms** | **883ms** | **892ms** |

### long_context_qa — 20 req, concurrency=4, ~1024-token inputs, 256-token outputs

| Variant | RPS | TTFT p50 | TTFT p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|
| default | 2.43 | 39ms | 46ms | 1643ms | 1658ms |
| **chunked** | **2.90** | **38ms** | **43ms** | **1379ms** | **1392ms** |
| **prefix_cache** | **2.90** | 39ms | 51ms | 1374ms | 1397ms |
| big_batch | 2.35 | 41ms | 52ms | 1703ms ↑ | 1711ms ↑ |

**Key findings:**
- `big_batch` (max_num_batched_tokens=4096) wins on short sequences (+15% RPS) but is the worst on long sequences — no universal optimum
- `chunked_prefill` hurts short-sequence TTFT (+40%) but reduces long-sequence E2E by 16%
- `prefix_caching` benefits only when prompts share a common prefix; random prompts see p99 spikes

Full report: [reports/phase1_baseline_en.md](reports/phase1_baseline_en.md) | [中文版](reports/phase1_baseline.md)

## Project structure

```
inferops/
  schemas.py            — ExperimentConfig, ExperimentResult, WorkloadSpec, AgentState
  observability.py      — MLflow (SQLite) + OpenTelemetry bootstrap
  bench_runner.py       — orchestrator: start vLLM → load → collect metrics → MLflow
  agent/
    coin_flip.py        — LangGraph warm-up (2-node loop, conditional edges, interrupt)
  tools/
    vllm_process.py     — vLLM subprocess lifecycle, OOM detection, startup timeout
    traffic.py          — async SSE load generator; measures TTFT / E2E / throughput
    gpu_monitor.py      — background pynvml sampler (GPU util + VRAM)
  eval/                 — metric analysis (Phase 2)
  rag/                  — retrieval for config knowledge base (Phase 3)
  memory/               — long-term experiment memory (Phase 3)
configs/
  search_space.py       — default + chunked + prefix_cache + big_batch variants
workloads/
  definitions.py        — chat_short and long_context_qa prompt sets + WorkloadSpec
scripts/
  run_baseline.py       — entry point: runs N configs × M workloads, writes report
  start_vllm.sh         — manual vLLM server launcher (0.5B / 1.5B)
  verify_env.sh         — 5-point environment health check
reports/
  phase1_baseline.md        — full Phase 1 report (Chinese)
  phase1_baseline_en.md     — full Phase 1 report (English)
```

## WSL2 / RTX 3060 config notes

- `gpu_memory_utilization=0.80` — safe ceiling; Windows DWM consumes ~2.2 GB invisible to PyTorch
- `max_model_len=2048` — must be set explicitly; Qwen2.5's native 32768-token context causes OOM on 6 GB
- `max_num_seqs=128` — realistic ceiling for concurrency ≤ 16

## Roadmap

- **Phase 0** ✅ repo skeleton, vLLM + Qwen2.5-0.5B baseline, LangGraph mental model
- **Phase 1** ✅ benchmark pipeline, 8-run baseline sweep, bilingual report
- **Phase 2**: 3-node LangGraph agent (plan → run → analyze), automated config search
- **Phase 3**: multi-objective Pareto optimization, RAG config knowledge base, LangSmith tracing
