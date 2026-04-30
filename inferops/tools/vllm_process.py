"""vLLM subprocess lifecycle: start, wait-ready, stop, OOM detection."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import httpx

from inferops.schemas import ExperimentConfig

DEFAULT_VLLM_PYTHON = "/home/chris/miniconda3/envs/vllm-dev/bin/python"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
STARTUP_TIMEOUT_S = 180  # CUDA graph compilation can be slow
HEALTH_POLL_S = 3


def get_vllm_python() -> str:
    """Return the Python executable used to launch the external vLLM server."""
    return (
        os.environ.get("INFEROPS_VLLM_PYTHON")
        or os.environ.get("VLLM_PYTHON")
        or DEFAULT_VLLM_PYTHON
    )


def _build_cmd(cfg: ExperimentConfig, host: str, port: int) -> list[str]:
    cmd = [
        get_vllm_python(), "-m", "vllm.entrypoints.openai.api_server",
        "--model", cfg.model_name,
        "--host", host,
        "--port", str(port),
        "--gpu-memory-utilization", str(cfg.gpu_memory_utilization),
        "--max-num-seqs", str(cfg.max_num_seqs),
        "--max-num-batched-tokens", str(cfg.max_num_batched_tokens),
        "--max-model-len", str(cfg.max_model_len),
        "--dtype", "auto",
        "--trust-remote-code",
        "--served-model-name", "qwen",
        "--kv-cache-metrics",  # expose kv cache utilization in /metrics
    ]
    if cfg.enforce_eager:
        cmd.append("--enforce-eager")
    if cfg.enable_chunked_prefill:
        cmd.append("--enable-chunked-prefill")
    else:
        cmd.append("--no-enable-chunked-prefill")
    if cfg.enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    return cmd


def _wait_for_ready(host: str, port: int, timeout_s: int) -> bool:
    url = f"http://{host}:{port}/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(HEALTH_POLL_S)
    return False


def _is_oom(proc: subprocess.Popen) -> bool:
    """Check if vLLM died due to CUDA OOM by scanning stderr."""
    if proc.stderr is None:
        return False
    # stderr is PIPE; read what's buffered without blocking
    try:
        out, _ = proc.communicate(timeout=2)
    except subprocess.TimeoutExpired:
        return False
    stderr_text = _.decode("utf-8", errors="replace") if _ else ""
    return "OutOfMemoryError" in stderr_text or "CUDA out of memory" in stderr_text


class VLLMProcess:
    """Context manager that owns the vLLM subprocess for one experiment."""

    def __init__(self, cfg: ExperimentConfig, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.cfg = cfg
        self.host = host
        self.port = port
        self._proc: subprocess.Popen | None = None
        self.log_path: Path | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        ts = int(time.time())
        self.log_path = log_dir / f"vllm_{self.cfg.experiment_id}_{ts}.log"

        cmd = _build_cmd(self.cfg, self.host, self.port)

        with open(self.log_path, "w") as logf:
            self._proc = subprocess.Popen(
                cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN),
            )

    def wait_ready(self) -> bool:
        return self.wait_ready_verbose(None)

    def wait_ready_verbose(self, log_fn) -> bool:
        """Block until /health returns 200, printing last log line while waiting."""
        deadline = time.time() + STARTUP_TIMEOUT_S
        url = f"{self.base_url}/health"
        last_reported_line = ""
        elapsed_ticks = 0

        while time.time() < deadline:
            if self._proc and self._proc.poll() is not None:
                return False  # crashed during startup
            try:
                r = httpx.get(url, timeout=3)
                if r.status_code == 200:
                    return True
            except Exception:
                pass

            # Every ~15s, tail the log so the user can see what vLLM is doing
            if log_fn and self.log_path and self.log_path.exists() and elapsed_ticks % 5 == 0:
                lines = self.log_path.read_text(errors="replace").splitlines()
                # Find last non-empty meaningful line
                for line in reversed(lines):
                    line = line.strip()
                    if line and line != last_reported_line and not line.startswith("{"):
                        log_fn(f"  [vLLM] {line[-120:]}")
                        last_reported_line = line
                        break

            elapsed_ticks += 1
            time.sleep(HEALTH_POLL_S)
        return False

    def is_crashed(self) -> bool:
        return self._proc is not None and self._proc.poll() is not None

    def exit_code(self) -> int | None:
        if self._proc is None:
            return None
        return self._proc.poll()

    def oom_in_log(self) -> bool:
        if self.log_path is None or not self.log_path.exists():
            return False
        text = self.log_path.read_text(errors="replace")
        return "OutOfMemoryError" in text or "CUDA out of memory" in text

    def stop(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        self._proc = None

    def __enter__(self) -> "VLLMProcess":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
