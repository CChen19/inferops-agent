#!/usr/bin/env bash
# Starts a vLLM OpenAI-compatible server on the RTX 3060 (6 GB VRAM).
# Usage:
#   ./scripts/start_vllm.sh [0.5B|1.5B]
#
# Requires: conda activate vllm-dev  (has vLLM 0.18 + Python 3.11)
#
# RTX 3060 Laptop GPU = 6 GB — safe headroom:
#   0.5B: gpu_memory_utilization=0.85
#   1.5B: gpu_memory_utilization=0.90  (tight; disable KV-cache extras)

set -euo pipefail

MODEL_SIZE="${1:-0.5B}"

case "$MODEL_SIZE" in
  0.5B)
    MODEL="Qwen/Qwen2.5-0.5B-Instruct"
    GPU_MEM=0.85
    MAX_SEQS=256
    MAX_TOKENS=4096
    ;;
  1.5B)
    MODEL="Qwen/Qwen2.5-1.5B-Instruct"
    GPU_MEM=0.90
    MAX_SEQS=128
    MAX_TOKENS=2048
    ;;
  *)
    echo "Usage: $0 [0.5B|1.5B]" >&2
    exit 1
    ;;
esac

HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"

echo "[inferops] Starting vLLM ${MODEL_SIZE} on ${HOST}:${PORT}"
echo "  model:                 ${MODEL}"
echo "  gpu_memory_utilization: ${GPU_MEM}"
echo "  max_num_seqs:          ${MAX_SEQS}"
echo "  max_model_len:         ${MAX_TOKENS}"
echo ""

exec python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --gpu-memory-utilization "${GPU_MEM}" \
  --max-num-seqs "${MAX_SEQS}" \
  --max-model-len "${MAX_TOKENS}" \
  --dtype auto \
  --trust-remote-code \
  --served-model-name qwen
