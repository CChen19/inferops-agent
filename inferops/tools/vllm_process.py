"""vLLM subprocess lifecycle: start, wait-ready, stop, OOM detection.

Also provides CLI-knob introspection and instance-identity helpers used by
bench_runner to prove config application (Week-1 item ②).

Health readiness is not enough: after start/restart the listener PID on the
port MUST equal the managed child PID, otherwise the result is never valid.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from inferops.schemas import (
    MANAGED_CLI_EVIDENCED_KEYS,
    ExperimentConfig,
    config_knobs,
    managed_cli_actual_config,
)
from inferops.tools.managed_lifecycle import cancel_requested

DEFAULT_VLLM_PYTHON = "/home/chris/miniconda3/envs/vllm-dev/bin/python"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
STARTUP_TIMEOUT_S = 180  # CUDA graph compilation can be slow
HEALTH_POLL_S = 3
# Between /health probes the wait loop sleeps in short slices so a Stop
# (cancel flag / stop()) is noticed within this many seconds, not HEALTH_POLL_S.
# Health GETs use the same budget so a hung /health cannot stall abort for 3s.
CANCEL_CHECK_S = 0.25

# Alias kept for tests / call sites; source of truth is schemas.
CLI_EVIDENCED_KNOB_KEYS: tuple[str, ...] = tuple(sorted(MANAGED_CLI_EVIDENCED_KEYS))


def get_vllm_python() -> str:
    """Return the Python executable used to launch the vLLM server."""
    return (
        os.environ.get("INFEROPS_VLLM_PYTHON")
        or os.environ.get("VLLM_PYTHON")
        or DEFAULT_VLLM_PYTHON
    )


def cli_evidenced_knobs(cfg: ExperimentConfig) -> dict[str, Any]:
    """Return only knobs that the managed launch command will actually pass."""
    return managed_cli_actual_config(config_knobs(cfg))


def _build_cmd(cfg: ExperimentConfig, host: str, port: int) -> list[str]:
    cmd = [
        get_vllm_python(),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        cfg.model_name,
        "--host",
        host,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(cfg.gpu_memory_utilization),
        "--max-num-seqs",
        str(cfg.max_num_seqs),
        "--max-num-batched-tokens",
        str(cfg.max_num_batched_tokens),
        "--max-model-len",
        str(cfg.max_model_len),
        "--dtype",
        "auto",
        "--trust-remote-code",
        "--served-model-name",
        "qwen",
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


def parse_vllm_cli_knobs(cmdline: list[str] | str) -> dict[str, Any]:
    """Best-effort parse of vLLM CLI argv into evidenced knob dict.

    Returns only keys found on the command line. Missing keys mean "unknown",
    not defaults — callers must not invent values for incomplete probes.
    """
    if isinstance(cmdline, str):
        parts = cmdline.split("\x00") if "\x00" in cmdline else cmdline.split()
    else:
        parts = list(cmdline)

    knobs: dict[str, Any] = {}

    def _flag_value(flag: str) -> str | None:
        if flag in parts:
            idx = parts.index(flag)
            if idx + 1 < len(parts):
                return parts[idx + 1]
        return None

    model = _flag_value("--model")
    if model is not None:
        knobs["model_name"] = model

    gmu = _flag_value("--gpu-memory-utilization")
    if gmu is not None:
        knobs["gpu_memory_utilization"] = float(gmu)

    mns = _flag_value("--max-num-seqs")
    if mns is not None:
        knobs["max_num_seqs"] = int(mns)

    mnbt = _flag_value("--max-num-batched-tokens")
    if mnbt is not None:
        knobs["max_num_batched_tokens"] = int(mnbt)

    mml = _flag_value("--max-model-len")
    if mml is not None:
        knobs["max_model_len"] = int(mml)

    if "--enforce-eager" in parts:
        knobs["enforce_eager"] = True

    if "--enable-chunked-prefill" in parts:
        knobs["enable_chunked_prefill"] = True
    elif "--no-enable-chunked-prefill" in parts:
        knobs["enable_chunked_prefill"] = False

    if "--enable-prefix-caching" in parts:
        knobs["enable_prefix_caching"] = True

    return knobs


def knobs_match_requested(
    observed: dict[str, Any] | None,
    requested_cli: dict[str, Any],
) -> bool:
    """True iff every requested CLI knob is present in observed and equal.

    For managed-style argv (core numeric flags present), absence of opt-in
    boolean flags is treated as False — matching `_build_cmd`. Incomplete
    probes must not invent values: if core flags are missing, match fails.
    """
    if not observed:
        return False
    effective = dict(observed)
    if "max_num_seqs" in observed and "gpu_memory_utilization" in observed:
        effective.setdefault("enforce_eager", False)
        effective.setdefault("enable_prefix_caching", False)
    for key, req_val in requested_cli.items():
        if key not in effective:
            return False
        if effective[key] != req_val:
            return False
    return True


@dataclass(frozen=True)
class InstanceIdentity:
    """Identity of a live vLLM instance (PID + generation/start token)."""

    host: str
    port: int
    pid: int | None = None
    start_token: str | None = None
    source: str = "unknown"  # managed_start | proc_probe | unknown

    @property
    def instance_id(self) -> str:
        pid_part = f"pid={self.pid}" if self.pid is not None else "pid=unknown"
        gen_part = f"gen={self.start_token}" if self.start_token else "gen=unknown"
        return f"{self.host}:{self.port}:{pid_part}:{gen_part}"

    def is_known(self) -> bool:
        return self.pid is not None and bool(self.start_token)


@dataclass
class LiveProbe:
    """Result of probing whatever is listening on host:port."""

    healthy: bool = False
    identity: InstanceIdentity | None = None
    observed_knobs: dict[str, Any] | None = None
    cmdline: list[str] | None = None


@dataclass
class StopOccupantResult:
    """Outcome of attempting to clear the port before a managed start."""

    previous_pid: int | None = None
    stop_attempted: bool = False
    still_listening: bool = False
    listener_pid_after: int | None = None


def probe_listener_pid(host: str, port: int) -> int | None:
    """Best-effort PID of the process listening on host:port (Linux)."""
    try:
        out = subprocess.check_output(
            ["ss", "-ltnp", f"sport = :{port}"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            if f":{port}" not in line:
                continue
            m = re.search(r"pid=(\d+)", line)
            if m:
                return int(m.group(1))
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            line = line.strip()
            if line.isdigit():
                return int(line)
    except Exception:
        pass
    return None


def read_process_cmdline(pid: int) -> list[str] | None:
    """Read argv for pid from /proc (Linux). Returns None if unavailable."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except Exception:
        return None
    if not raw:
        return None
    parts = [p.decode("utf-8", errors="replace") for p in raw.split(b"\x00") if p]
    return parts or None


def health_ok(host: str, port: int, timeout_s: float = 2.0) -> bool:
    try:
        r = httpx.get(f"http://{host}:{port}/health", timeout=timeout_s)
        return r.status_code == 200
    except Exception:
        return False


def probe_live_instance(host: str, port: int) -> LiveProbe:
    """Probe health + best-effort identity/knobs for whatever owns the port."""
    healthy = health_ok(host, port)
    if not healthy:
        return LiveProbe(healthy=False)

    pid = probe_listener_pid(host, port)
    cmdline = read_process_cmdline(pid) if pid is not None else None
    observed = parse_vllm_cli_knobs(cmdline) if cmdline else None
    identity = InstanceIdentity(
        host=host,
        port=port,
        pid=pid,
        start_token=None,  # cannot recover managed start_token from external probe
        source="proc_probe" if pid is not None else "unknown",
    )
    return LiveProbe(
        healthy=True,
        identity=identity,
        observed_knobs=observed if observed else None,
        cmdline=cmdline,
    )


def stop_port_occupant(
    host: str,
    port: int,
    timeout_s: float = 15.0,
    *,
    expected_pid: int | None = None,
) -> StopOccupantResult:
    """Terminate the process listening on host:port — only if it is ``expected_pid``.

    Stage C ownership rule: never kill a PID the caller has not explicitly
    identified. ``expected_pid=None`` or a listener that differs from it is a
    refusal (``stop_attempted=False``, ``still_listening=True``) — the
    occupant is left untouched.

    Returns whether something is still listening afterward. Callers MUST treat
    `still_listening=True` as stop failure and must not mark the run valid.

    Deterministic injection (GPU checklist): set INFEROPS_SIMULATE_STOP_FAILURE=1
    to report still_listening without relying on VRAM / permissions tricks.
    """
    if os.getenv("INFEROPS_SIMULATE_STOP_FAILURE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        pid = probe_listener_pid(host, port)
        return StopOccupantResult(
            previous_pid=pid,
            stop_attempted=True,
            still_listening=True,
            listener_pid_after=pid,
        )

    pid = probe_listener_pid(host, port)
    if pid is None:
        # No PID — if still healthy, treat as unknown occupant (stop failed).
        return StopOccupantResult(
            previous_pid=None,
            stop_attempted=False,
            still_listening=health_ok(host, port),
            listener_pid_after=None,
        )
    if expected_pid is None or pid != expected_pid:
        # Not the process we were told we own — refuse, leave it running.
        return StopOccupantResult(
            previous_pid=pid,
            stop_attempted=False,
            still_listening=True,
            listener_pid_after=pid,
        )

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        after = probe_listener_pid(host, port)
        return StopOccupantResult(
            previous_pid=pid,
            stop_attempted=True,
            still_listening=after is not None or health_ok(host, port),
            listener_pid_after=after,
        )
    except PermissionError:
        after = probe_listener_pid(host, port)
        return StopOccupantResult(
            previous_pid=pid,
            stop_attempted=True,
            still_listening=True,
            listener_pid_after=after if after is not None else pid,
        )

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.2)
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        time.sleep(0.2)

    after = probe_listener_pid(host, port)
    still = after is not None or health_ok(host, port)
    return StopOccupantResult(
        previous_pid=pid,
        stop_attempted=True,
        still_listening=still,
        listener_pid_after=after,
    )


def assert_listener_bound_to_child(
    *,
    host: str,
    port: int,
    child_pid: int | None,
) -> int:
    """Require the healthy listener PID to equal the managed child PID.

    Raises RuntimeError on unknown child, unknown listener, or mismatch.
    Returns the verified listener PID on success.
    """
    if child_pid is None:
        raise RuntimeError("managed child PID is unknown — cannot bind health to identity")
    listener_pid = probe_listener_pid(host, port)
    if listener_pid is None:
        raise RuntimeError("listener PID unknown after health ready — cannot prove new identity")
    if listener_pid != child_pid:
        raise RuntimeError(
            f"listener PID {listener_pid} != managed child PID {child_pid} "
            "(stale occupant or stop failure — never valid)"
        )
    return listener_pid


class VLLMProcess:
    """Context manager that owns the vLLM subprocess for one experiment."""

    def __init__(self, cfg: ExperimentConfig, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.cfg = cfg
        self.host = host
        self.port = port
        self._proc: subprocess.Popen | None = None
        self.log_path: Path | None = None
        self.start_token: str = uuid.uuid4().hex
        self.launch_cmd: list[str] = []
        self.pre_restart_identity: InstanceIdentity | None = None
        self.last_stop_result: StopOccupantResult | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def pid(self) -> int | None:
        proc = self._proc
        return proc.pid if proc is not None else None

    def identity(self) -> InstanceIdentity:
        return InstanceIdentity(
            host=self.host,
            port=self.port,
            pid=self.pid,
            start_token=self.start_token,
            source="managed_start",
        )

    def evidenced_actual_config(self) -> dict[str, Any]:
        """Knobs proven by the launch command we issued (not full requested dict)."""
        parsed = parse_vllm_cli_knobs(self.launch_cmd) if self.launch_cmd else {}
        if self.launch_cmd:
            if "enforce_eager" not in parsed:
                parsed["enforce_eager"] = False
            if "enable_prefix_caching" not in parsed:
                parsed["enable_prefix_caching"] = False
        return {k: parsed[k] for k in MANAGED_CLI_EVIDENCED_KEYS if k in parsed}

    def start(self) -> None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        ts = int(time.time())
        self.log_path = log_dir / f"vllm_{self.cfg.experiment_id}_{ts}.log"

        self.start_token = uuid.uuid4().hex
        self.launch_cmd = _build_cmd(self.cfg, self.host, self.port)

        with open(self.log_path, "w") as logf:
            self._proc = subprocess.Popen(
                self.launch_cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN),
            )

    def restart_after_stop(
        self,
        *,
        previous_identity: InstanceIdentity | None,
        stop_occupant: bool = True,
    ) -> StopOccupantResult:
        """Stop the identified occupant / prior proc, then start only if the port is clear.

        The occupant is stopped only when ``previous_identity.pid`` names it
        (``stop_port_occupant(expected_pid=...)``); an unidentified or
        different listener is refused and left running. If the port is still
        occupied (`still_listening=True`), this does **not** spawn a new
        child (avoids orphans / racing the stale listener). Caller must treat
        still_listening as failure — never valid.
        """
        self.pre_restart_identity = previous_identity
        if self._proc is not None:
            self.stop()
        stop_result = StopOccupantResult()
        if stop_occupant:
            expected = previous_identity.pid if previous_identity is not None else None
            stop_result = stop_port_occupant(self.host, self.port, expected_pid=expected)
        self.last_stop_result = stop_result
        if stop_result.still_listening:
            return stop_result
        self.start()
        return stop_result

    def wait_ready(self) -> bool:
        return self.wait_ready_verbose(None)

    def wait_ready_verbose(self, log_fn) -> bool:
        """Block until /health returns 200, printing last log line while waiting.

        Health alone does NOT prove identity — callers must call
        `assert_listener_bound_to_child` afterward.

        Returns False promptly (not after STARTUP_TIMEOUT_S) when the managed
        child has crashed, has been ``stop()``-ed (``_proc`` cleared), or the
        lifecycle cancel flag is set — there is no child worth waiting for.
        """
        deadline = time.time() + STARTUP_TIMEOUT_S
        url = f"{self.base_url}/health"
        last_reported_line = ""
        elapsed_ticks = 0

        while time.time() < deadline:
            if self._wait_should_abort():
                return False
            try:
                r = httpx.get(url, timeout=CANCEL_CHECK_S)
                if r.status_code == 200:
                    return True
            except Exception:
                pass

            if log_fn and self.log_path and self.log_path.exists() and elapsed_ticks % 5 == 0:
                lines = self.log_path.read_text(errors="replace").splitlines()
                for line in reversed(lines):
                    line = line.strip()
                    if line and line != last_reported_line and not line.startswith("{"):
                        log_fn(f"  [vLLM] {line[-120:]}")
                        last_reported_line = line
                        break

            elapsed_ticks += 1
            if self._sleep_until_next_poll(deadline):
                return False
        return False

    def _wait_should_abort(self) -> bool:
        """True when readiness polling is pointless: no child, dead child, or cancel."""
        proc = self._proc
        if proc is None:
            return True  # never started or stop()-ed — nothing to become ready
        if proc.poll() is not None:
            return True  # crashed during startup
        return cancel_requested()

    def _sleep_until_next_poll(self, deadline: float) -> bool:
        """Sleep HEALTH_POLL_S in CANCEL_CHECK_S slices; True if the wait should abort."""
        wake = min(time.time() + HEALTH_POLL_S, deadline)
        while True:
            remaining = wake - time.time()
            if remaining <= 0:
                return False
            time.sleep(min(CANCEL_CHECK_S, remaining))
            if self._wait_should_abort():
                return True

    def is_crashed(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is not None

    def exit_code(self) -> int | None:
        proc = self._proc
        if proc is None:
            return None
        return proc.poll()

    def oom_in_log(self) -> bool:
        if self.log_path is None or not self.log_path.exists():
            return False
        text = self.log_path.read_text(errors="replace")
        return "OutOfMemoryError" in text or "CUDA out of memory" in text

    def stop(self) -> None:
        """Terminate the managed child if still running (never leave orphans)."""
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        self._proc = None

    def __enter__(self) -> "VLLMProcess":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
