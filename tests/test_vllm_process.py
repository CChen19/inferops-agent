"""Unit tests for vLLM subprocess command construction."""

from __future__ import annotations

from inferops.tools.vllm_process import DEFAULT_VLLM_PYTHON, _build_cmd, get_vllm_python


def test_get_vllm_python_defaults_to_conda_path(monkeypatch):
    monkeypatch.delenv("INFEROPS_VLLM_PYTHON", raising=False)
    monkeypatch.delenv("VLLM_PYTHON", raising=False)

    assert get_vllm_python() == DEFAULT_VLLM_PYTHON


def test_get_vllm_python_uses_inferops_env(monkeypatch):
    monkeypatch.setenv("INFEROPS_VLLM_PYTHON", "/opt/vllm/bin/python")
    monkeypatch.setenv("VLLM_PYTHON", "/ignored/python")

    assert get_vllm_python() == "/opt/vllm/bin/python"


def test_build_cmd_uses_config_and_env_python(config, monkeypatch):
    monkeypatch.setenv("INFEROPS_VLLM_PYTHON", "/opt/vllm/bin/python")
    cfg = config.model_copy(update={
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
        "enforce_eager": True,
    })

    cmd = _build_cmd(cfg, "127.0.0.1", 9000)

    assert cmd[:3] == [
        "/opt/vllm/bin/python",
        "-m",
        "vllm.entrypoints.openai.api_server",
    ]
    assert "--port" in cmd
    assert "9000" in cmd
    assert "--enable-chunked-prefill" in cmd
    assert "--no-enable-chunked-prefill" not in cmd
    assert "--enable-prefix-caching" in cmd
    assert "--enforce-eager" in cmd
