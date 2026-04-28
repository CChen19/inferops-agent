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
| Observability | OpenTelemetry → console (→ LangSmith later) |
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

# 3. Run baseline sweep (starts/stops vLLM automatically per variant)
python scripts/run_baseline.py --workload chat_short
# or run everything:
python scripts/run_baseline.py --workload both
```

> **Note:** vLLM must run in the `vllm-dev` conda env. The benchmark script handles
> this automatically by calling the conda env's Python directly.

## Running a single experiment manually

```bash
# Start vLLM in one terminal (conda activate vllm-dev first)
bash scripts/start_vllm.sh 0.5B   # or 1.5B

# Run a specific variant in another terminal (.venv activated)
python scripts/run_baseline.py --workload chat_short --variants big_batch
```

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
  rag/                  — retrieval for config knowledge base (Phase 2)
  memory/               — long-term experiment memory (Phase 2)
configs/
  search_space.py       — default + chunked_prefill + prefix_caching + big_batch variants
workloads/
  definitions.py        — chat_short and long_context_qa prompt sets
scripts/
  run_baseline.py       — entry point: runs N configs × M workloads, writes report
  start_vllm.sh         — manual vLLM server launcher (0.5B / 1.5B)
  verify_env.sh         — 5-point environment health check
reports/                — Markdown + JSON baseline reports (generated)
```

## Parameter search space

| Variant | max_num_batched_tokens | enable_chunked_prefill | enable_prefix_caching |
|---|---|---|---|
| default | 2048 | False | False |
| chunked | 2048 | True | False |
| prefix_cache | 2048 | False | True |
| big_batch | 4096 | False | False |

## Workloads

| Workload | Requests | Concurrency | Input tokens | Output tokens |
|---|---|---|---|---|
| chat_short | 60 | 16 | ~128 | 128 |
| long_context_qa | 20 | 4 | ~1024 | 256 |

## Roadmap

- **Phase 0** ✅ repo skeleton, vLLM + Qwen2.5-0.5B baseline, LangGraph mental model
- **Phase 1** ✅ walking skeleton: bench_runner, traffic generator, GPU monitor, 4-variant sweep
- **Phase 2**: 3-node LangGraph agent (plan → run → analyze), automated config search
- **Phase 3**: multi-objective Pareto optimization, RAG config knowledge base
- **Phase 4**: full autonomous loop, LangSmith tracing
