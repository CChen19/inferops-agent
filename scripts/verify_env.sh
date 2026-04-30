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

# 2. CUDA from Python (vllm-dev conda env)
CONDA_PYTHON="${INFEROPS_VLLM_PYTHON:-${VLLM_PYTHON:-/home/chris/miniconda3/envs/vllm-dev/bin/python}}"
[[ -x "$CONDA_PYTHON" ]] || fail "vLLM Python not executable: $CONDA_PYTHON"
"$CONDA_PYTHON" -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'  torch {torch.__version__}, CUDA {torch.version.cuda}, device: {torch.cuda.get_device_name(0)}')
" && ok "CUDA available via torch" || fail "torch.cuda not available in vllm-dev env"

# 3. vLLM
"$CONDA_PYTHON" -c "import vllm; print(f'  vllm {vllm.__version__}')" \
  && ok "vLLM importable" || fail "vLLM import failed in vllm-dev env"

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
