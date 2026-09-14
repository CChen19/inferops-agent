"""Unit tests for vLLM subprocess command construction and readiness wait."""

from __future__ import annotations

import threading
import time

import httpx

from inferops.tools import managed_lifecycle as ml
from inferops.tools import vllm_process as vp
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
    cfg = config.model_copy(
        update={
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "enforce_eager": True,
        }
    )

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
    cfg = config.model_copy(
        update={
            "enable_chunked_prefill": False,
            "enable_prefix_caching": True,
            "enforce_eager": True,
        }
    )
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


# ---------------------------------------------------------------------------
# wait_ready_verbose: must exit promptly on Stop / cancel, never wait out the
# startup timeout polling a dead port. CPU-only: fake child + no real HTTP.
# ---------------------------------------------------------------------------


class _FakeChild:
    """Popen stand-in: alive until terminate()/kill(), never binds a port."""

    def __init__(self, pid: int = 4242, exit_code: int | None = None):
        self.pid = pid
        self._exit_code = exit_code
        self.terminated = False

    def poll(self):
        return self._exit_code

    def terminate(self):
        self.terminated = True
        self._exit_code = -15

    def kill(self):
        self.terminated = True
        self._exit_code = -9

    def wait(self, timeout=None):
        return self._exit_code


def _never_ready(monkeypatch, timeout_s: float = 6.0) -> None:
    """Health never answers; startup timeout shrunk so a regression fails fast."""
    monkeypatch.setattr(vp, "STARTUP_TIMEOUT_S", timeout_s)

    def _refused(*a, **k):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(vp.httpx, "get", _refused)


def _proc_with_fake_child(config, exit_code: int | None = None) -> VLLMProcess:
    proc = VLLMProcess(config, host="127.0.0.1", port=8000)
    proc._proc = _FakeChild(exit_code=exit_code)
    return proc


def test_wait_ready_returns_true_when_health_answers(config, monkeypatch):
    proc = _proc_with_fake_child(config)
    monkeypatch.setattr(vp, "STARTUP_TIMEOUT_S", 6)
    monkeypatch.setattr(vp.httpx, "get", lambda *a, **k: httpx.Response(200))
    assert proc.wait_ready() is True


def test_wait_ready_returns_false_immediately_when_never_started(config, monkeypatch):
    _never_ready(monkeypatch)
    proc = VLLMProcess(config, host="127.0.0.1", port=8000)
    t0 = time.monotonic()
    assert proc.wait_ready() is False
    assert time.monotonic() - t0 < 1.0


def test_wait_ready_returns_false_promptly_when_child_crashed(config, monkeypatch):
    _never_ready(monkeypatch)
    proc = _proc_with_fake_child(config, exit_code=1)
    t0 = time.monotonic()
    assert proc.wait_ready() is False
    assert time.monotonic() - t0 < 1.0
    assert proc.is_crashed() is True


def test_wait_ready_exits_promptly_on_cancel_flag(config, monkeypatch):
    """Cancel flag alone (no stop() call yet) must break the wait well before timeout."""
    _never_ready(monkeypatch, timeout_s=6.0)
    proc = _proc_with_fake_child(config)
    timer = threading.Timer(0.5, ml.request_cancel)
    timer.start()
    try:
        t0 = time.monotonic()
        ready = proc.wait_ready_verbose(None)
        elapsed = time.monotonic() - t0
    finally:
        timer.cancel()
    assert ready is False
    assert elapsed < 2.0, f"wait_ready waited {elapsed:.2f}s after cancel"


def test_wait_ready_exits_promptly_after_stop_clears_proc(config, monkeypatch):
    """stop() sets _proc=None; the loop must treat that as 'nothing to wait for'."""
    _never_ready(monkeypatch, timeout_s=6.0)
    proc = _proc_with_fake_child(config)
    child = proc._proc
    timer = threading.Timer(0.5, proc.stop)
    timer.start()
    try:
        t0 = time.monotonic()
        ready = proc.wait_ready_verbose(None)
        elapsed = time.monotonic() - t0
    finally:
        timer.cancel()
    assert ready is False
    assert child.terminated is True
    assert proc.pid is None
    assert elapsed < 2.0, f"wait_ready waited {elapsed:.2f}s after stop()"


def test_wait_ready_still_times_out_without_cancel(config, monkeypatch):
    """No cancel / no stop: the loop still honours the (shrunk) startup timeout."""
    _never_ready(monkeypatch, timeout_s=0.6)
    proc = _proc_with_fake_child(config)
    t0 = time.monotonic()
    assert proc.wait_ready() is False
    elapsed = time.monotonic() - t0
    assert 0.5 <= elapsed < 2.0


def test_wait_ready_health_get_uses_cancel_check_timeout(config, monkeypatch):
    """Hung /health must be bounded by CANCEL_CHECK_S, not a hardcoded 3s GET."""
    seen = {}

    def _ok(*_a, **k):
        seen["timeout"] = k.get("timeout")
        return httpx.Response(200)

    monkeypatch.setattr(vp.httpx, "get", _ok)
    proc = _proc_with_fake_child(config)
    assert proc.wait_ready() is True
    assert seen["timeout"] == vp.CANCEL_CHECK_S
    assert seen["timeout"] <= 0.25


def test_wait_ready_abort_during_hung_health_get_is_subsecond(config, monkeypatch):
    """Cancel during a hung health GET must not wait out the old 3s httpx timeout."""
    monkeypatch.setattr(vp, "STARTUP_TIMEOUT_S", 6.0)

    def _hung(*_a, **k):
        timeout = k.get("timeout", 3)
        time.sleep(float(timeout))
        raise httpx.TimeoutException("hung health")

    monkeypatch.setattr(vp.httpx, "get", _hung)
    proc = _proc_with_fake_child(config)
    timer = threading.Timer(0.05, ml.request_cancel)
    timer.start()
    try:
        t0 = time.monotonic()
        ready = proc.wait_ready_verbose(None)
        elapsed = time.monotonic() - t0
    finally:
        timer.cancel()
    assert ready is False
    assert elapsed < 1.0, f"wait_ready waited {elapsed:.2f}s on hung health GET"


def _racey_proc(config):
    """First _proc read is a live child; later reads are None (stop() on another thread)."""
    child = _FakeChild()
    reads = {"n": 0}

    class _RaceyProcVLLMProcess(VLLMProcess):
        def __getattribute__(self, name):
            if name == "_proc":
                reads["n"] += 1
                if reads["n"] == 1:
                    return child
                return None  # simulates stop() clearing _proc between former TOCTOU reads
            return super().__getattribute__(name)

    return _RaceyProcVLLMProcess(config, host="127.0.0.1", port=8000), reads


def test_wait_should_abort_binds_proc_once(config):
    """stop() can null _proc on another thread; must not re-read self._proc for poll()."""
    proc, reads = _racey_proc(config)
    assert proc._wait_should_abort() is False
    assert reads["n"] == 1


def test_is_crashed_binds_proc_once(config):
    """stop() can null _proc on another thread; must not re-read self._proc for poll()."""
    proc, reads = _racey_proc(config)
    assert proc.is_crashed() is False
    assert reads["n"] == 1


def test_exit_code_binds_proc_once(config):
    """stop() can null _proc on another thread; must not re-read self._proc for poll()."""
    proc, reads = _racey_proc(config)
    assert proc.exit_code() is None
    assert reads["n"] == 1


def test_pid_binds_proc_once(config):
    """stop() can null _proc on another thread; must not re-read self._proc for .pid."""
    proc, reads = _racey_proc(config)
    assert proc.pid == 4242
    assert reads["n"] == 1


def test_stop_binds_proc_once(config):
    """A racing second stop() can null _proc; must not re-read for poll/terminate/wait."""
    proc, reads = _racey_proc(config)
    proc.stop()  # must not AttributeError
    assert reads["n"] == 1
