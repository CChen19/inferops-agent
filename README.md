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
| Env loading | python-dotenv |

## Quick start

```bash
# 1. Create + activate venv
uv venv --python /home/chris/miniconda3/envs/vllm-dev/bin/python3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# 2. Verify environment (GPU, CUDA, vLLM, LangGraph, MLflow)
bash scripts/verify_env.sh

# 3. Run full baseline sweep — starts/stops vLLM automatically per variant
python scripts/run_baseline.py --workload both

# 4. Run a specific workload or subset of variants
python scripts/run_baseline.py --workload chat_short --variants default,big_batch
```

> **Note:** The benchmark script launches vLLM using the `vllm-dev` conda env directly —
> no need to activate it manually or install vLLM into `.venv`. To use a different
> vLLM environment, set `INFEROPS_VLLM_PYTHON=/path/to/python`.

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
  cli.py                — Typer console entry point (`inferops ...`)
  agent/
    state.py            — AgentState TypedDict, Hypothesis, ExperimentSummary, helpers
    planner.py          — Planner node: LLM generates 1-3 evidence-based hypotheses
    executor.py         — Executor node: propose_config → run_benchmark → analyze → compare
    reflector.py        — Reflector node: improvement check, streak control, routing
    graph.py            — LangGraph StateGraph assembly + run_agent() entry point
    coin_flip.py        — LangGraph warm-up example
  eval/
    baselines.py        — CI-safe random/greedy baseline agent simulators
    harness.py          — commit-level eval report generation
    metrics.py          — OutcomeMetrics (gap%), EfficiencyMetrics, composite_score
    judge.py            — LLM-as-judge rubric, few-shot calibration, consistency checks
    regression.py       — regression gate for commit-to-commit eval comparisons
    runner.py           — eval runner: load ground truth → query DB → summary table
  memory/
    db.py               — SQLite CRUD for experiment results (save, query, upsert)
  rag/
    chunker.py          — Markdown-aware text chunker (heading-split, overlap)
    embedder.py         — bge-base-zh-v1.5 sentence-transformers wrapper (lazy-loaded)
    store.py            — Chroma persistent vector store (build_index, query)
  tools/
    vllm_process.py     — vLLM subprocess lifecycle, OOM detection, startup timeout
    traffic.py          — async load generator; streaming TTFT mode + stable non-streaming benchmark mode
    gpu_monitor.py      — background pynvml sampler (GPU util + VRAM)
    run_benchmark.py    — tool: run one vLLM experiment, persist to memory DB
    propose_config.py   — tool: validate + materialise a single-param config change
    read_gpu_metrics.py — tool: sample GPU util + VRAM over a short window
    profile_cpu.py      — tool: py-spy CPU hotspot profiler (top-N functions)
    analyze_bottleneck.py  — tool: rule-based bottleneck classifier (compute/memory/scheduling/kv)
    compare_experiments.py — tool: bootstrap CI comparison of two experiment results
    experiment_memory.py   — tool: query past results from SQLite memory
    write_report.py        — tool: append H2 section to a Markdown report file
    knowledge_retriever.py — tool: semantic search over vLLM knowledge corpus (Chroma)
    final_report.py        — tool: generate end-of-session Markdown report with citations
    registry.py            — ALL_TOOLS list: 9 @tool-decorated LangGraph wrappers
configs/
  search_space.py       — default + chunked + prefix_cache + big_batch variants
workloads/
  definitions.py        — 5 golden workloads + prompt generators (chat_short,
                          long_context_qa, high_concurrency_short_out,
                          long_generation, mixed_traffic)
scripts/
  run_baseline.py       — Phase 1 entry point: runs N configs × M workloads
  run_grid_sweep.py     — Phase 3 sweep: 12 configs × 5 workloads → data/ground_truth/
  run_agent.py          — Phase 4: run agent on one workload (--llm deepseek|claude)
  run_comparison.py     — Phase 4: Agent vs Default vs Random on 3 workloads
  run_eval.py           — Phase 5: commit eval dashboard + regression gate
  start_vllm.sh         — manual vLLM server launcher (0.5B / 1.5B)
  verify_env.sh         — 5-point environment health check
data/
  ground_truth/         — one JSON per workload after run_grid_sweep.py completes
tests/
  conftest.py           — shared pytest fixtures (result, tmp_db, tmp_report, …)
  test_*.py             — 152 unit tests, all tools + agent nodes mocked (no vLLM required)
reports/
  phase1_baseline.md        — full Phase 1 report (Chinese)
  phase1_baseline_en.md     — full Phase 1 report (English)
```

## WSL2 / RTX 3060 config notes

- `gpu_memory_utilization=0.80` — normal 1.5B target; set `VLLM_GPU_MEM=0.65` if startup reports low free VRAM
- `max_model_len=2048` — must be set explicitly; Qwen2.5's native 32768-token context causes OOM on 6 GB
- `max_num_seqs=128` — realistic ceiling for concurrency ≤ 16

## Phase 5 eval harness

```bash
# CI-safe mock eval — deterministic, no vLLM needed
python scripts/run_eval.py --mock --commit-sha $(git rev-parse --short HEAD) \
  --ground-truth tests/fixtures/ground_truth --workloads chat_short long_generation \
  --budget 2 --seed 7

# Same, with regression gate vs a pinned baseline (exits 1 on regression):
python scripts/run_eval.py --mock --commit-sha $(git rev-parse --short HEAD) \
  --ground-truth tests/fixtures/ground_truth --workloads chat_short long_generation \
  --budget 2 --seed 7 \
  --baseline-report tests/fixtures/ci_baseline.json --gate-strategy greedy_agent

# Real/manual eval for an agent session already persisted in experiment memory:
python scripts/run_eval.py --prefix agent_chat_short_abc123_ --workloads chat_short \
  --ground-truth data/ground_truth
```

Reports are written to `eval_reports/<sha>.md` and `<sha>.json`.
The CI workflow (`.github/workflows/eval-mock.yml`) runs the mock eval + regression gate
on every push/PR using `tests/fixtures/ci_baseline.json` as the pinned baseline.

## Phase 6 RAG + Chainlit UI

```bash
# 1. Install UI extras (Chainlit + chromadb + sentence-transformers)
uv pip install -e ".[dev,ui]"

# 2. Build the knowledge index (downloads bge-base-zh-v1.5, runs once)
python scripts/build_corpus.py

# 3. Verify index was built
python scripts/build_corpus.py --dry-run

# 4. Terminal 1: launch vLLM from the vllm-dev conda environment.
#    On a 6 GB RTX 3060, use VLLM_GPU_MEM=0.65 if 1.5B startup reports
#    "free memory ... less than desired GPU memory utilization".
VLLM_GPU_MEM=0.65 bash scripts/start_vllm.sh 1.5B

# 5. Terminal 2: launch Chainlit from the project .venv.
#    For the default OpenRouter backend, put OPENROUTER_API_KEY in .env.
source .venv/bin/activate
INFEROPS_LLM=openrouter chainlit run app.py --port 8001
# open http://localhost:8001 and type e.g.:
#   "I have Qwen2.5-1.5B on RTX 3060, chat scenario, maximize throughput, budget=5"
```

The Planner now retrieves relevant chunks from the corpus before generating hypotheses.
Every hypothesis rationale must include a `[source: <document>]` citation — uncited
hypotheses are rejected before they reach the executor.

Chainlit runs or loads the baseline before entering the LangGraph loop, streams
planner / executor / reflector progress, and writes the final Markdown report to
`reports/<session_prefix>final_report.md`.

The Chainlit benchmark path intentionally uses non-streaming vLLM chat requests
to avoid a known `httpx` / `anyio` streaming cleanup deadlock in the UI process.
Throughput and end-to-end latency remain usable for optimization; TTFT is
currently approximated by request completion time on that path.

To add new corpus documents, drop Markdown files into `data/corpus/` and re-run
`scripts/build_corpus.py --reset`.

## Roadmap

- **Phase 0** ✅ repo skeleton, vLLM + Qwen2.5-0.5B baseline, LangGraph mental model
- **Phase 1** ✅ benchmark pipeline, 8-run baseline sweep, bilingual report
- **Phase 2** ✅ 8 LangGraph tools, SQLite experiment memory, OTel spans, 37 unit tests
- **Phase 3** ✅ 5 golden workloads, grid sweep (60 experiments → ground truth), eval framework (outcome / efficiency / LLM-as-judge), 50 unit tests
- **Phase 4** ✅ Plan-Execute-Reflect LangGraph agent (DeepSeek V3 / Claude Sonnet), config dedup, budget control, bottleneck-switch replan, 126 unit tests; `run_comparison.py` for Agent vs Default vs Random
- **Phase 5** ✅ Evaluation harness + regression gate: commit-level eval reports, random/greedy baselines, mock CI gate, 130 unit tests
- **Phase 6** ✅ RAG knowledge retrieval + Chainlit UI: 6-doc corpus (PagedAttention, chunked prefill, prefix caching, speculative decoding, vLLM scheduler, tuning notes), citation-grounded planner, Chroma vector store, intent extraction, Chainlit async chat UI, final report tool, stable non-streaming benchmark path, 152 unit tests
