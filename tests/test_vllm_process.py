"""Unit tests for vLLM subprocess command construction."""

from __future__ import annotations

from inferops.tools.vllm_process import (
    DEFAULT_VLLM_PYTHON,
    StopOccupantResult,
    VLLMProcess,
    _build_cmd,
    cli_evidenced_knobs,
    get_vllm_python,
    parse_vllm_cli_knobs,
)


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


def test_parse_vllm_cli_knobs_roundtrip(config, monkeypatch):
    monkeypatch.setenv("INFEROPS_VLLM_PYTHON", "/opt/vllm/bin/python")
    cfg = config.model_copy(update={
        "enable_chunked_prefill": False,
        "enable_prefix_caching": True,
        "enforce_eager": True,
    })
    cmd = _build_cmd(cfg, "127.0.0.1", 8000)
    parsed = parse_vllm_cli_knobs(cmd)
    evidenced = cli_evidenced_knobs(cfg)
    for key, val in evidenced.items():
        if key in parsed:
            assert parsed[key] == val
    assert parsed["enforce_eager"] is True
    assert parsed["enable_prefix_caching"] is True
    assert parsed["enable_chunked_prefill"] is False


def test_restart_after_stop_skips_start_when_still_listening(config, monkeypatch):
    proc = VLLMProcess(config, host="127.0.0.1", port=8000)
    monkeypatch.setattr(
        "inferops.tools.vllm_process.stop_port_occupant",
        lambda *a, **k: StopOccupantResult(
            previous_pid=1, stop_attempted=True, still_listening=True, listener_pid_after=1
        ),
    )
    started = {"n": 0}
    monkeypatch.setattr(proc, "start", lambda: started.__setitem__("n", started["n"] + 1))
    result = proc.restart_after_stop(previous_identity=None, stop_occupant=True)
    assert result.still_listening is True
    assert started["n"] == 0
    assert proc.pid is None
