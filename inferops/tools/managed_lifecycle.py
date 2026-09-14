"""Single-GPU mutex + owned-process registry for managed vLLM (Stage C).

Three small, CPU-only primitives that bench_runner / graph / UI compose:

1. ``GPULease`` — an ``fcntl.flock`` on a well-known lock file. A second
   managed InferOps task fails closed with ``GPUBusyError`` (clear holder
   info, nothing started, nothing killed) instead of racing the port. The
   lock file carries a JSON owner record (owner pid, experiment, child pid,
   launch argv) so a blocked task can say *who* holds the GPU and a later
   task can recognise a stale InferOps child it is allowed to adopt.

2. Owned-child registry — ``register_owned`` / ``release_owned`` /
   ``cancel_owned_children``. Cancel/abort only ever stops processes that
   InferOps spawned in *this* process and registered here. It never
   resolves a PID from the port and kills it.

3. Cancel flag — ``request_cancel`` / ``cancel_requested``. Set by the UI
   stop button (or any caller); ``run_experiment`` refuses to spawn while
   it is set and turns a mid-load kill into an explicit cancellation.

Lock path: ``INFEROPS_GPU_LOCK_PATH`` or ``<tmpdir>/inferops/gpu.lock``.
Never the repo root. Tests point it at ``tmp_path`` (see conftest).
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

LOCK_PATH_ENV = "INFEROPS_GPU_LOCK_PATH"
ADOPT_STALE_ENV = "INFEROPS_ADOPT_STALE_MANAGED"


def gpu_lock_path() -> Path:
    """Well-known single-GPU lock file (env override for tests / multi-GPU)."""
    raw = os.getenv(LOCK_PATH_ENV, "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path(tempfile.gettempdir()) / "inferops" / "gpu.lock"


def adopt_stale_managed_allowed() -> bool:
    """Opt-in: let a new task stop+relaunch a *recorded* stale InferOps child."""
    return os.getenv(ADOPT_STALE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _pid_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclass
class LeaseRecord:
    """Owner record stored inside the lock file (readable while locked)."""

    owner_pid: int
    host: str
    port: int
    experiment_id: str | None = None
    session_id: str | None = None
    child_pid: int | None = None
    start_token: str | None = None
    launch_cmd: list[str] | None = None
    acquired_at: float = field(default_factory=time.time)
    released: bool = False
    released_at: float | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str | bytes | None) -> LeaseRecord | None:
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except Exception:
            return None
        if not isinstance(data, dict) or "owner_pid" not in data:
            return None
        known = set(cls.__dataclass_fields__)
        try:
            return cls(**{k: v for k, v in data.items() if k in known})
        except TypeError:
            return None

    def describe(self) -> str:
        who = f"InferOps pid={self.owner_pid}"
        if self.experiment_id:
            who += f" experiment={self.experiment_id}"
        if self.session_id:
            who += f" session={self.session_id}"
        if self.child_pid is not None:
            who += f" child_pid={self.child_pid}"
        return who


class GPUBusyError(Exception):
    """Another managed InferOps task holds the single-GPU lease."""

    def __init__(self, path: Path, holder: LeaseRecord | None):
        self.path = path
        self.holder = holder
        holder_s = holder.describe() if holder is not None else "unknown holder"
        super().__init__(
            f"GPU busy: managed vLLM lease at {path} is held by {holder_s}. "
            "Refusing to start a second managed vLLM (nothing was started or "
            "stopped). Wait for that task to finish or cancel it first."
        )


class GPULease:
    """Non-blocking exclusive flock on the single-GPU lock file.

    ``acquire()`` raises ``GPUBusyError`` when another *live* holder exists.
    On success the previous (stale) record is kept in ``previous_record`` so
    callers can tell a recorded InferOps orphan from an unknown service.
    """

    def __init__(
        self,
        path: Path | None = None,
        *,
        host: str,
        port: int,
        experiment_id: str | None = None,
        session_id: str | None = None,
    ):
        self.path = Path(path) if path is not None else gpu_lock_path()
        self.host = host
        self.port = port
        self.experiment_id = experiment_id
        self.session_id = session_id
        self._fd: int | None = None
        self.record: LeaseRecord | None = None
        self.previous_record: LeaseRecord | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    # -- lock file I/O ----------------------------------------------------

    def _read_record(self) -> LeaseRecord | None:
        assert self._fd is not None
        os.lseek(self._fd, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(self._fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
        return LeaseRecord.from_json(b"".join(chunks))

    def _write_record(self, record: LeaseRecord) -> None:
        assert self._fd is not None
        payload = (record.to_json() + "\n").encode("utf-8")
        os.lseek(self._fd, 0, os.SEEK_SET)
        os.ftruncate(self._fd, 0)
        os.write(self._fd, payload)
        try:
            os.fsync(self._fd)
        except OSError:
            pass
        self.record = record

    # -- lifecycle --------------------------------------------------------

    def acquire(self) -> GPULease:
        if self._fd is not None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._fd = fd
            try:
                holder = self._read_record()
            finally:
                self._fd = None
                os.close(fd)
            raise GPUBusyError(self.path, holder) from None
        except Exception:
            os.close(fd)
            raise
        self._fd = fd
        self.previous_record = self._read_record()
        self._write_record(
            LeaseRecord(
                owner_pid=os.getpid(),
                host=self.host,
                port=self.port,
                experiment_id=self.experiment_id,
                session_id=self.session_id,
            )
        )
        return self

    def record_child(
        self,
        *,
        child_pid: int | None,
        start_token: str | None,
        launch_cmd: list[str] | None,
    ) -> None:
        """Record the managed child we spawned (so a crash leaves a trail)."""
        if self._fd is None or self.record is None:
            return
        self.record.child_pid = child_pid
        self.record.start_token = start_token
        self.record.launch_cmd = list(launch_cmd) if launch_cmd else None
        self._write_record(self.record)

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            if self.record is not None:
                self.record.released = True
                self.record.released_at = time.time()
                self._write_record(self.record)
        except OSError:
            pass
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> GPULease:
        return self.acquire()

    def __exit__(self, *_: object) -> None:
        self.release()


def read_lease_record(path: Path | None = None) -> LeaseRecord | None:
    """Read the lock file record without taking the lock (status / messages)."""
    p = Path(path) if path is not None else gpu_lock_path()
    try:
        return LeaseRecord.from_json(p.read_bytes())
    except OSError:
        return None


def is_recorded_managed_child(
    record: LeaseRecord | None,
    *,
    listener_pid: int | None,
    cmdline: list[str] | None,
) -> bool:
    """True iff the port listener is the child a *crashed* InferOps run left.

    Requires: a stale (never released) record whose owner is dead, a PID
    match, and — when the record has argv — an exact argv match. Anything
    less is an unknown service and must not be stopped.
    """
    if record is None or listener_pid is None or record.child_pid is None:
        return False
    if record.released or _pid_alive(record.owner_pid):
        return False
    if record.child_pid != listener_pid:
        return False
    if record.launch_cmd:
        return cmdline is not None and list(cmdline) == list(record.launch_cmd)
    return True


# ---------------------------------------------------------------------------
# Owned-child registry + cancel flag (process-local)
# ---------------------------------------------------------------------------


@dataclass
class OwnedChild:
    proc: Any  # VLLMProcess-like: .pid, .stop()
    lease: GPULease | None
    experiment_id: str | None


_owned: dict[int, OwnedChild] = {}
_owned_lock = threading.Lock()
_cancel_event = threading.Event()


def register_owned(proc: Any, lease: GPULease | None, experiment_id: str | None) -> None:
    """Record a child *we spawned* so cancel/abort can stop exactly it."""
    with _owned_lock:
        _owned[id(proc)] = OwnedChild(proc=proc, lease=lease, experiment_id=experiment_id)


def owned_pids() -> list[int]:
    with _owned_lock:
        return [c.proc.pid for c in _owned.values() if getattr(c.proc, "pid", None)]


def release_owned(proc: Any) -> None:
    """Stop one owned child and release its lease (normal end of experiment)."""
    with _owned_lock:
        entry = _owned.pop(id(proc), None)
    try:
        proc.stop()
    finally:
        if entry is not None and entry.lease is not None:
            entry.lease.release()


def cancel_owned_children(reason: str = "cancelled") -> list[dict[str, Any]]:
    """Stop every registered owned child and release its lease.

    Returns one report per child. Never touches a PID that was not
    registered via ``register_owned`` — unknown / external services are
    left alone by construction.
    """
    with _owned_lock:
        entries = list(_owned.values())
        _owned.clear()
    reports: list[dict[str, Any]] = []
    for entry in entries:
        pid = getattr(entry.proc, "pid", None)
        report: dict[str, Any] = {
            "pid": pid,
            "experiment_id": entry.experiment_id,
            "reason": reason,
            "stopped": False,
            "lease_released": False,
        }
        try:
            entry.proc.stop()
            report["stopped"] = True
        except Exception as exc:  # noqa: BLE001 — keep going, report it
            report["error"] = str(exc)
        finally:
            if entry.lease is not None:
                try:
                    entry.lease.release()
                    report["lease_released"] = True
                except Exception as exc:  # noqa: BLE001
                    report["lease_error"] = str(exc)
        reports.append(report)
    return reports


def request_cancel() -> None:
    _cancel_event.set()


def clear_cancel() -> None:
    _cancel_event.clear()


def cancel_requested() -> bool:
    return _cancel_event.is_set()
