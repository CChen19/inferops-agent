#!/usr/bin/env bash
# Verifies the full inferops dev environment:
#   1. NVIDIA GPU visible
#   2. CUDA accessible from Python (torch)
#   3. vLLM importable
#   4. LangGraph importable (from uv venv)
#   5. MLflow importable (from uv venv)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

ok()  { echo -e "\e[32m[OK]\e[0m  $*"; }
fail(){ echo -e "\e[31m[FAIL]\e[0m $*"; exit 1; }

echo "=== inferops environment check ==="
echo ""

# 1. GPU
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  && ok "NVIDIA GPU detected" || fail "nvidia-smi failed — is the NVIDIA driver loaded?"

# 2. CUDA from Python (requires INFEROPS_VLLM_PYTHON or VLLM_PYTHON)
VLLM_PYTHON_BIN="${INFEROPS_VLLM_PYTHON:-${VLLM_PYTHON:-}}"
[[ -n "$VLLM_PYTHON_BIN" ]] || fail "Set INFEROPS_VLLM_PYTHON to the vLLM Python interpreter (VLLM_PYTHON also accepted)"
[[ -x "$VLLM_PYTHON_BIN" ]] || fail "vLLM Python not executable: $VLLM_PYTHON_BIN"
"$VLLM_PYTHON_BIN" -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'  torch {torch.__version__}, CUDA {torch.version.cuda}, device: {torch.cuda.get_device_name(0)}')
" && ok "CUDA available via torch" || fail "torch.cuda not available in vLLM Python env"

# 3. vLLM
"$VLLM_PYTHON_BIN" -c "import vllm; print(f'  vllm {vllm.__version__}')" \
  && ok "vLLM importable" || fail "vLLM import failed in vLLM Python env"

# 4. LangGraph (uv venv)
UV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
[[ -x "$UV_PYTHON" ]] || fail "Project venv Python not executable: $UV_PYTHON"
"$UV_PYTHON" -c "
import importlib.metadata, langgraph
v = importlib.metadata.version('langgraph')
print(f'  langgraph {v}')
" && ok "LangGraph importable" || fail "LangGraph import failed — run: uv pip install langgraph"

# 5. MLflow (uv venv)
"$UV_PYTHON" -c "import mlflow; print(f'  mlflow {mlflow.__version__}')" \
  && ok "MLflow importable" || fail "MLflow import failed — run: uv pip install mlflow"

echo ""
echo "=== All checks passed ==="
