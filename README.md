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
uv pip install -e ".[dev]"

# 2. Verify environment (GPU, CUDA, vLLM, LangGraph, MLflow)
bash scripts/verify_env.sh

# 3. Start the vLLM server (in a separate terminal, with conda activate vllm-dev)
bash scripts/start_vllm.sh 0.5B   # or 1.5B

# 4. Run the LangGraph warm-up
python -m inferops.agent.coin_flip
```

## Project structure

```
inferops/
  schemas.py         — ExperimentConfig, ExperimentResult, WorkloadSpec, AgentState
  observability.py   — MLflow + OpenTelemetry bootstrap
  agent/
    coin_flip.py     — LangGraph warm-up (2-node loop with conditional edges)
  tools/             — vLLM client, benchmark runner (Phase 1)
  eval/              — metric collection (Phase 1)
  rag/               — retrieval for config knowledge base (Phase 2)
  memory/            — long-term experiment memory (Phase 2)
configs/             — YAML experiment templates
workloads/           — synthetic prompt datasets
benchmarks/          — raw result storage
reports/             — aggregated analysis
```

## Roadmap

- **Phase 0** (current): repo skeleton, vLLM + Qwen2.5-0.5B baseline, LangGraph mental model
- **Phase 1**: baseline measurement tool, 3-node agent (plan → run → analyze), MLflow logging
- **Phase 2**: multi-objective Pareto optimization, RAG config knowledge base
- **Phase 3**: full autonomous experiment loop, LangSmith tracing
