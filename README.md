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
| Observability | OpenTelemetry → console / Jaeger (OTLP) |
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
  observability.py      — MLflow (SQLite) + OpenTelemetry (console or OTLP/Jaeger)
  bench_runner.py       — orchestrator: start vLLM → load → collect metrics → MLflow
  agent/
    coin_flip.py        — LangGraph warm-up (2-node loop, conditional edges, interrupt)
  memory/
    db.py               — SQLite CRUD for experiment results (save, query, upsert)
  tools/
    vllm_process.py     — vLLM subprocess lifecycle, OOM detection, startup timeout
    traffic.py          — async SSE load generator; measures TTFT / E2E / throughput
    gpu_monitor.py      — background pynvml sampler (GPU util + VRAM)
    run_benchmark.py    — tool: run one vLLM experiment, persist to memory DB
    propose_config.py   — tool: validate + materialise a single-param config change
    read_gpu_metrics.py — tool: sample GPU util + VRAM over a short window
    profile_cpu.py      — tool: py-spy CPU hotspot profiler (top-N functions)
    analyze_bottleneck.py  — tool: rule-based bottleneck classifier (compute/memory/scheduling/kv)
    compare_experiments.py — tool: bootstrap CI comparison of two experiment results
    experiment_memory.py   — tool: query past results from SQLite memory
    write_report.py        — tool: append H2 section to a Markdown report file
    registry.py            — ALL_TOOLS list: @tool-decorated LangGraph wrappers
configs/
  search_space.py       — default + chunked + prefix_cache + big_batch variants
  eval/
    metrics.py          — OutcomeMetrics (gap%), EfficiencyMetrics, composite_score
    judge.py            — LLM-as-judge (4-criterion rubric; heuristic fallback)
    runner.py           — eval_runner CLI: load ground truth → query DB → summary table
workloads/
  definitions.py        — 5 golden workloads + prompt generators (chat_short,
                          long_context_qa, high_concurrency_short_out,
                          long_generation, mixed_traffic)
configs/
  search_space.py       — default + chunked + prefix_cache + big_batch variants
scripts/
  run_baseline.py       — Phase 1 entry point: runs N configs × M workloads
  run_grid_sweep.py     — Phase 3 sweep: 12 configs × 5 workloads → data/ground_truth/
  start_vllm.sh         — manual vLLM server launcher (0.5B / 1.5B)
  verify_env.sh         — 5-point environment health check
data/
  ground_truth/         — one JSON per workload after run_grid_sweep.py completes
tests/
  conftest.py           — shared pytest fixtures (result, tmp_db, tmp_report, …)
  test_*.py             — 50 unit tests, all tools mocked (no vLLM required)
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
- **Phase 2** ✅ 8 LangGraph tools, SQLite experiment memory, OTel spans, 37 unit tests
- **Phase 3** ✅ 5 golden workloads, grid sweep (60 experiments → ground truth), eval framework (outcome / efficiency / LLM-as-judge), 50 unit tests
- **Phase 4**: autonomous LangGraph agent loop (plan → run → analyze), Pareto multi-objective search
