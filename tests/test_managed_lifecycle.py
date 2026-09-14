"""Stage C: single-GPU mutex, unknown-occupant refusal, cancel owned-only (CPU-only)."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from inferops import bench_runner
from inferops.agent.graph import abort_agent_run, run_agent
from inferops.agent.recovery import is_hard_control_exception, reraise_hard_control
from inferops.memory.db import get_task
from inferops.schemas import ExperimentValidityStatus
from inferops.tools import managed_lifecycle as ml
from inferops.tools import vllm_process as vp
from inferops.tools.vllm_process import InstanceIdentity, LiveProbe, StopOccupantResult
from tests.test_config_application import _dead_pid, _patch_common


class MockChild:
    """Stand-in for VLLMProcess: records stop(), never touches a real PID."""

    def __init__(self, pid: int, experiment_id: str = "mock"):
        self._pid: int | None = pid
        self.stop_calls = 0
        self.cfg = type("Cfg", (), {"experiment_id": experiment_id})()

    @property
    def pid(self):
        return self._pid

    def stop(self):
        self.stop_calls += 1
        self._pid = None


def _lock_is_free() -> bool:
    try:
        with ml.GPULease(host="127.0.0.1", port=8000):
            return True
    except ml.GPUBusyError:
        return False


# ---------------------------------------------------------------------------
# 1. Mutex
# ---------------------------------------------------------------------------


def test_default_lock_path_is_not_repo_root(monkeypatch):
    monkeypatch.delenv(ml.LOCK_PATH_ENV, raising=False)
    p = ml.gpu_lock_path()
    assert p.is_absolute()
    assert Path(tempfile.gettempdir()) in p.parents
    assert p.parent != Path.cwd()


def test_second_lease_in_same_process_is_refused_with_holder_info():
    first = ml.GPULease(host="127.0.0.1", port=8000, experiment_id="exp_a").acquire()
    first.record_child(child_pid=4321, start_token="tok", launch_cmd=["python", "-m", "vllm"])
    try:
        with pytest.raises(ml.GPUBusyError) as ei:
            ml.GPULease(host="127.0.0.1", port=8000, experiment_id="exp_b").acquire()
        assert ei.value.holder is not None
        assert ei.value.holder.owner_pid == os.getpid()
        assert ei.value.holder.child_pid == 4321
        assert "exp_a" in str(ei.value) and "child_pid=4321" in str(ei.value)
        assert "nothing was started or stopped" in str(ei.value)
    finally:
        first.release()
    assert _lock_is_free()
    rec = ml.read_lease_record()
    assert rec is not None and rec.released is True


def test_lease_blocks_across_processes(tmp_path):
    lock = tmp_path / "xproc.lock"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import fcntl,os,sys,time;"
                f"fd=os.open({str(lock)!r},os.O_RDWR|os.O_CREAT);"
                "fcntl.flock(fd,fcntl.LOCK_EX);"
                'os.write(fd,b\'{"owner_pid":%d,"host":"h","port":1}\'%os.getpid());'
                "print('locked',flush=True);time.sleep(30)"
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "locked"
        with pytest.raises(ml.GPUBusyError) as ei:
            ml.GPULease(lock, host="h", port=1).acquire()
        assert ei.value.holder is not None
        assert ei.value.holder.owner_pid == holder.pid
    finally:
        holder.kill()
        holder.wait()
    # Kernel dropped the flock with the holder → next task acquires.
    with ml.GPULease(lock, host="h", port=1):
        pass


def test_second_managed_task_blocked_first_child_not_killed(monkeypatch, config):
    """Task B fails closed while task A holds the GPU; A's child PID is untouched."""
    _patch_common(monkeypatch)
    task_a = ml.GPULease(host="127.0.0.1", port=8000, experiment_id="task_a").acquire()
    task_a.record_child(child_pid=777, start_token="a", launch_cmd=None)
    killed: list[int] = []
    monkeypatch.setattr(vp.os, "kill", lambda pid, sig: killed.append(pid))
    monkeypatch.setattr(
        bench_runner,
        "probe_live_instance",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not probe when busy")),
    )
    monkeypatch.setattr(
        bench_runner,
        "VLLMProcess",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not construct child")),
    )

    progress: list[str] = []
    try:
        with pytest.raises(bench_runner.BenchmarkError, match="GPU busy") as ei:
            bench_runner.run_experiment(config, ["p"], on_progress=progress.append)
    finally:
        task_a.release()

    assert killed == []
    assert any("failed:gpu_busy" in p for p in progress)
    assert "task_a" in str(ei.value) and "child_pid=777" in str(ei.value)
    assert ei.value.result is not None
    assert ei.value.result.status == ExperimentValidityStatus.FAILED
    assert ei.value.result.status != ExperimentValidityStatus.VALID
    assert "GPU busy" in (ei.value.result.notes or "")


def test_lease_released_after_successful_run(monkeypatch, config, tmp_path):
    _patch_common(monkeypatch)
    monkeypatch.chdir(tmp_path)  # live_identity_*.json goes to tmp, not repo logs/
    monkeypatch.setattr(bench_runner, "probe_live_instance", lambda *a, **k: LiveProbe())
    monkeypatch.setattr(bench_runner, "assert_listener_bound_to_child", lambda **k: 31)

    class Proc(MockChild):
        def __init__(self, cfg, host="127.0.0.1", port=8000):
            super().__init__(31, cfg.experiment_id)
            self.cfg, self.host, self.port = cfg, host, port
            self.log_path = None
            self.start_token = "tok"
            self.launch_cmd = []

        def identity(self):
            return InstanceIdentity(self.host, self.port, self._pid, self.start_token)

        def start(self):
            self.launch_cmd = vp._build_cmd(self.cfg, self.host, self.port)

        def wait_ready_verbose(self, log_fn):
            # Lease must be held and carry our child while we are alive.
            rec = ml.read_lease_record()
            assert rec is not None and rec.child_pid == 31 and rec.released is False
            assert not _lock_is_free()
            # Owned from spawn onward so Stop can abort the model-load wait.
            assert ml.owned_pids() == [31]
            return True

        def evidenced_actual_config(self):
            from inferops.tools.vllm_process import cli_evidenced_knobs

            return cli_evidenced_knobs(self.cfg)

        def oom_in_log(self):
            return False

        def is_crashed(self):
            return False

    monkeypatch.setattr(bench_runner, "VLLMProcess", Proc)
    result = bench_runner.run_experiment(config, ["p"])
    assert result.status == ExperimentValidityStatus.VALID
    assert ml.owned_pids() == []
    assert _lock_is_free()


# ---------------------------------------------------------------------------
# 2. Unknown occupant / ownership of stop
# ---------------------------------------------------------------------------


def test_stop_port_occupant_refuses_pid_it_was_not_told_it_owns(monkeypatch):
    monkeypatch.delenv("INFEROPS_SIMULATE_STOP_FAILURE", raising=False)
    monkeypatch.setattr(vp, "probe_listener_pid", lambda h, p: 5150)
    monkeypatch.setattr(vp, "health_ok", lambda h, p, timeout_s=2.0: True)
    killed: list[int] = []
    monkeypatch.setattr(vp.os, "kill", lambda pid, sig: killed.append(pid))

    for expected in (None, 999):
        res = vp.stop_port_occupant("127.0.0.1", 8000, expected_pid=expected)
        assert res.stop_attempted is False
        assert res.still_listening is True
        assert res.listener_pid_after == 5150
    assert killed == []


def test_stop_port_occupant_stops_only_the_expected_pid(monkeypatch):
    monkeypatch.delenv("INFEROPS_SIMULATE_STOP_FAILURE", raising=False)
    listener = {"pid": 5150}
    monkeypatch.setattr(vp, "probe_listener_pid", lambda h, p: listener["pid"])
    monkeypatch.setattr(vp, "health_ok", lambda h, p, timeout_s=2.0: listener["pid"] is not None)
    signals: list[tuple[int, int]] = []

    def fake_kill(pid, sig):
        signals.append((pid, sig))
        if sig == 0 and listener["pid"] is None:
            raise ProcessLookupError
        if sig != 0:
            listener["pid"] = None

    monkeypatch.setattr(vp.os, "kill", fake_kill)
    res = vp.stop_port_occupant("127.0.0.1", 8000, expected_pid=5150, timeout_s=1)
    assert res.stop_attempted is True
    assert res.still_listening is False
    assert {pid for pid, _ in signals} == {5150}


def test_restart_after_stop_passes_identified_pid(config, monkeypatch):
    proc = vp.VLLMProcess(config, host="127.0.0.1", port=8000)
    seen: dict = {}

    def fake_stop(host, port, timeout_s=15.0, *, expected_pid=None):
        seen["expected_pid"] = expected_pid
        return StopOccupantResult(previous_pid=expected_pid, still_listening=True)

    monkeypatch.setattr(vp, "stop_port_occupant", fake_stop)
    ident = InstanceIdentity(host="127.0.0.1", port=8000, pid=42)
    proc.restart_after_stop(previous_identity=ident, stop_occupant=True)
    assert seen["expected_pid"] == 42
    proc.restart_after_stop(previous_identity=None, stop_occupant=True)
    assert seen["expected_pid"] is None
    assert proc.pid is None


def test_is_recorded_managed_child_is_strict():
    dead = _dead_pid()
    cmd = ["python", "-m", "vllm.entrypoints.openai.api_server", "--port", "8000"]
    rec = ml.LeaseRecord(owner_pid=dead, host="h", port=8000, child_pid=11, launch_cmd=cmd)

    assert ml.is_recorded_managed_child(rec, listener_pid=11, cmdline=cmd) is True
    # argv drift → not ours
    assert ml.is_recorded_managed_child(rec, listener_pid=11, cmdline=cmd + ["--x"]) is False
    assert ml.is_recorded_managed_child(rec, listener_pid=11, cmdline=None) is False
    # different pid → not ours
    assert ml.is_recorded_managed_child(rec, listener_pid=12, cmdline=cmd) is False
    # owner still alive → its child, not an orphan
    alive = ml.LeaseRecord(owner_pid=os.getpid(), host="h", port=8000, child_pid=11)
    assert ml.is_recorded_managed_child(alive, listener_pid=11, cmdline=None) is False
    # cleanly released → whoever listens now is unknown
    rel = ml.LeaseRecord(owner_pid=dead, host="h", port=8000, child_pid=11, released=True)
    assert ml.is_recorded_managed_child(rel, listener_pid=11, cmdline=None) is False
    assert ml.is_recorded_managed_child(None, listener_pid=11, cmdline=None) is False


def test_external_mode_never_touches_lease_or_processes(monkeypatch, config):
    _patch_common(monkeypatch)
    other = ml.GPULease(host="127.0.0.1", port=8000, experiment_id="someone").acquire()
    try:
        monkeypatch.setattr(
            bench_runner,
            "VLLMProcess",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not start managed")),
        )
        result = bench_runner.run_experiment(config, ["p"], service_mode="external")
    finally:
        other.release()
    assert result.status == ExperimentValidityStatus.INSUFFICIENT_EVIDENCE
    assert result.config_evidence.kind == "external_unverified"


# ---------------------------------------------------------------------------
# 3. Cancel / abort — owned processes only
# ---------------------------------------------------------------------------


def test_cancel_owned_children_stops_owned_and_releases_lock_only(monkeypatch):
    killed: list[int] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append(pid))

    lease = ml.GPULease(host="127.0.0.1", port=8000, experiment_id="owned").acquire()
    owned = MockChild(pid=1001)
    lease.record_child(child_pid=1001, start_token="t", launch_cmd=None)
    ml.register_owned(owned, lease, "owned")
    foreign = MockChild(pid=2002)  # e.g. a user's own vLLM: never registered

    assert not _lock_is_free()
    reports = ml.cancel_owned_children("user pressed stop")

    assert [r["pid"] for r in reports] == [1001]
    assert reports[0]["stopped"] is True and reports[0]["lease_released"] is True
    assert owned.stop_calls == 1
    assert foreign.stop_calls == 0
    assert killed == []  # no PID was signalled directly
    assert ml.owned_pids() == []
    assert _lock_is_free()
    # Idempotent: nothing left to stop.
    assert ml.cancel_owned_children("again") == []


def test_task_cancelled_is_hard_control():
    exc = bench_runner.TaskCancelled("stop")
    assert is_hard_control_exception(exc)
    with pytest.raises(bench_runner.TaskCancelled):
        try:
            raise exc
        except Exception as caught:
            reraise_hard_control(caught)
    assert not is_hard_control_exception(bench_runner.BenchmarkError("x"))


def test_run_experiment_refuses_to_spawn_after_cancel(monkeypatch, config):
    _patch_common(monkeypatch)
    monkeypatch.setattr(
        bench_runner,
        "VLLMProcess",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not spawn")),
    )
    ml.request_cancel()
    progress: list[str] = []
    with pytest.raises(bench_runner.TaskCancelled, match="no vLLM was spawned"):
        bench_runner.run_experiment(config, ["p"], on_progress=progress.append)
    assert progress == ["status:cancelled:before_start"]
    assert _lock_is_free()


def test_mid_load_cancel_stops_owned_child_and_releases_lock(monkeypatch, config, tmp_path):
    """Stop button during load: owned child stopped once, lease freed, row = cancelled."""
    monkeypatch.chdir(tmp_path)  # live_identity_*.json goes to tmp, not repo logs/
    _patch_common(monkeypatch)
    monkeypatch.setattr(bench_runner, "probe_live_instance", lambda *a, **k: LiveProbe())
    monkeypatch.setattr(bench_runner, "assert_listener_bound_to_child", lambda **k: 55)
    procs: list = []

    class Proc(MockChild):
        def __init__(self, cfg, host="127.0.0.1", port=8000):
            super().__init__(55, cfg.experiment_id)
            self.cfg, self.host, self.port = cfg, host, port
            self.log_path = None
            self.start_token = "tok"
            self.launch_cmd = []
            procs.append(self)

        def identity(self):
            return InstanceIdentity(self.host, self.port, self._pid, self.start_token)

        def start(self):
            self.launch_cmd = vp._build_cmd(self.cfg, self.host, self.port)

        def wait_ready_verbose(self, log_fn):
            return True

        def evidenced_actual_config(self):
            from inferops.tools.vllm_process import cli_evidenced_knobs

            return cli_evidenced_knobs(self.cfg)

        def oom_in_log(self):
            return False

        def is_crashed(self):
            return False

    monkeypatch.setattr(bench_runner, "VLLMProcess", Proc)

    def load_then_user_stops(*a, **k):
        assert ml.owned_pids() == [55]
        ml.request_cancel()
        reports = ml.cancel_owned_children("user pressed stop")  # what app.on_stop does
        assert [r["pid"] for r in reports] == [55]
        raise ConnectionError("connection refused (server went away)")

    monkeypatch.setattr(bench_runner, "_run_load_with_cleanup_workaround", load_then_user_stops)

    progress: list[str] = []
    with pytest.raises(bench_runner.TaskCancelled, match="during load") as ei:
        bench_runner.run_experiment(config, ["p"], on_progress=progress.append)

    assert procs[0].stop_calls >= 1
    assert procs[0].pid is None
    assert any("cancelled:during_load" in p for p in progress)
    assert ei.value.result is not None
    assert ei.value.result.status == ExperimentValidityStatus.FAILED
    assert "cancelled by user" in (ei.value.result.notes or "")
    assert ml.owned_pids() == []
    assert _lock_is_free()


def _startup_proc_factory(pid: int, procs: list, *, on_wait_ready=None):
    """VLLMProcess stand-in whose wait_ready_verbose runs an injected hook."""

    class Proc(MockChild):
        def __init__(self, cfg, host="127.0.0.1", port=8000):
            super().__init__(pid, cfg.experiment_id)
            self.cfg, self.host, self.port = cfg, host, port
            self.log_path = None
            self.start_token = "tok"
            self.launch_cmd = []
            self.wait_ready_calls = 0
            procs.append(self)

        def identity(self):
            return InstanceIdentity(self.host, self.port, self._pid, self.start_token)

        def start(self):
            self.launch_cmd = vp._build_cmd(self.cfg, self.host, self.port)

        def wait_ready_verbose(self, log_fn):
            self.wait_ready_calls += 1
            if on_wait_ready is not None:
                return on_wait_ready(self)
            return True

        def evidenced_actual_config(self):
            from inferops.tools.vllm_process import cli_evidenced_knobs

            return cli_evidenced_knobs(self.cfg)

        def oom_in_log(self):
            return False

        def is_crashed(self):
            # A child Stop already terminated reads as crashed to the real class.
            return self._pid is None

    return Proc


def test_stop_during_wait_ready_aborts_load_never_valid(monkeypatch, config):
    """Stop pressed while vLLM is still loading the model (child spawned, not ready).

    Before this slice the child was registered only after readiness, so Stop
    found nothing, the load ran to completion and the row came back ``valid``.
    """
    _patch_common(monkeypatch)
    monkeypatch.setattr(bench_runner, "probe_live_instance", lambda *a, **k: LiveProbe())
    monkeypatch.setattr(
        bench_runner,
        "assert_listener_bound_to_child",
        lambda **k: (_ for _ in ()).throw(AssertionError("must not bind after cancel")),
    )
    monkeypatch.setattr(
        bench_runner,
        "_run_load_with_cleanup_workaround",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("load must not run")),
    )
    killed: list[int] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append(pid))
    procs: list = []
    stop_reports: list = []

    def user_stops_while_loading(proc):
        # Child is spawned and already owned; lease carries it.
        assert ml.owned_pids() == [77]
        rec = ml.read_lease_record()
        assert rec is not None and rec.child_pid == 77 and rec.released is False
        ml.request_cancel()
        stop_reports.extend(ml.cancel_owned_children("user pressed stop"))  # app.on_stop
        assert proc.pid is None  # our child was stopped by Stop itself
        return False  # /health never came up because the child is gone

    Proc = _startup_proc_factory(77, procs, on_wait_ready=user_stops_while_loading)
    monkeypatch.setattr(bench_runner, "VLLMProcess", Proc)

    progress: list[str] = []
    with pytest.raises(bench_runner.TaskCancelled, match="during managed startup") as ei:
        bench_runner.run_experiment(config, ["p"], on_progress=progress.append)

    assert [r["pid"] for r in stop_reports] == [77]
    assert stop_reports[0]["stopped"] is True and stop_reports[0]["lease_released"] is True
    assert procs[0].wait_ready_calls == 1
    assert procs[0].pid is None
    assert killed == []  # no unregistered PID was ever signalled
    assert "status:cancelled:during_startup_wait" in progress
    assert "status:cancelled:startup" in progress
    assert not any(p.startswith("status:failed:startup") for p in progress)
    assert ei.value.result is not None
    assert ei.value.result.status == ExperimentValidityStatus.FAILED
    assert ei.value.result.status != ExperimentValidityStatus.VALID
    assert ml.owned_pids() == []
    assert _lock_is_free()


def test_cancel_flag_during_wait_ready_without_stop_call_still_aborts(monkeypatch, config):
    """Only the flag is set (e.g. Stop raced the spawn): startup itself stops the child."""
    _patch_common(monkeypatch)
    monkeypatch.setattr(bench_runner, "probe_live_instance", lambda *a, **k: LiveProbe())
    monkeypatch.setattr(
        bench_runner,
        "_run_load_with_cleanup_workaround",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("load must not run")),
    )
    procs: list = []

    def flag_only(proc):
        ml.request_cancel()
        return True  # child happens to become ready anyway

    Proc = _startup_proc_factory(78, procs, on_wait_ready=flag_only)
    monkeypatch.setattr(bench_runner, "VLLMProcess", Proc)
    monkeypatch.setattr(bench_runner, "assert_listener_bound_to_child", lambda **k: 78)

    with pytest.raises(bench_runner.TaskCancelled) as ei:
        bench_runner.run_experiment(config, ["p"])
    assert procs[0].stop_calls == 1 and procs[0].pid is None
    assert ei.value.result is not None
    assert ei.value.result.status != ExperimentValidityStatus.VALID
    assert ml.owned_pids() == []
    assert _lock_is_free()


def test_cancel_after_ready_before_load_never_valid(monkeypatch, config, tmp_path):
    """Stop lands after readiness/identity bind but before traffic: load must not run."""
    _patch_common(monkeypatch)
    monkeypatch.chdir(tmp_path)  # live_identity_*.json goes to tmp, not repo logs/
    monkeypatch.setattr(bench_runner, "probe_live_instance", lambda *a, **k: LiveProbe())
    monkeypatch.setattr(bench_runner, "assert_listener_bound_to_child", lambda **k: 79)
    procs: list = []
    Proc = _startup_proc_factory(79, procs)
    monkeypatch.setattr(bench_runner, "VLLMProcess", Proc)

    load_calls: list = []

    def load_would_succeed(*a, **k):
        load_calls.append(1)
        raise AssertionError("load must not run after cancel")

    monkeypatch.setattr(bench_runner, "_run_load_with_cleanup_workaround", load_would_succeed)

    # Fire the cancel the moment startup hands the child over (before load).
    real_gpu_monitor = bench_runner.GPUMonitor

    class CancelOnMonitorStart(real_gpu_monitor):
        def start(self):
            ml.request_cancel()
            ml.cancel_owned_children("user pressed stop")
            raise RuntimeError("no GPU in CI")

    monkeypatch.setattr(bench_runner, "GPUMonitor", CancelOnMonitorStart)

    progress: list[str] = []
    with pytest.raises(bench_runner.TaskCancelled, match="before load") as ei:
        bench_runner.run_experiment(config, ["p"], on_progress=progress.append)

    assert load_calls == []
    assert "status:cancelled:before_load" in progress
    assert procs[0].pid is None
    assert ei.value.result is not None
    assert ei.value.result.status == ExperimentValidityStatus.FAILED
    assert "cancelled by user (before load)" in (ei.value.result.notes or "")
    assert ml.owned_pids() == []
    assert _lock_is_free()


def test_ctrl_c_between_lease_acquire_and_spawn_releases_lock(monkeypatch, config):
    """Signal after acquire() but before any spawn: lease must not leak."""
    _patch_common(monkeypatch)

    def probe_interrupted(*a, **k):
        assert not _lock_is_free()  # lease is held at this point
        raise KeyboardInterrupt

    monkeypatch.setattr(bench_runner, "probe_live_instance", probe_interrupted)
    procs: list = []
    Proc = _startup_proc_factory(80, procs)
    monkeypatch.setattr(bench_runner, "VLLMProcess", Proc)
    with pytest.raises(KeyboardInterrupt):
        bench_runner.run_experiment(config, ["p"])
    assert procs[0].stop_calls == 1  # idempotent stop on a never-started child
    assert ml.owned_pids() == []
    assert _lock_is_free()


def test_ctrl_c_during_startup_stops_spawned_child_and_releases_lock(monkeypatch, config):
    _patch_common(monkeypatch)
    monkeypatch.setattr(bench_runner, "probe_live_instance", lambda *a, **k: LiveProbe())
    procs: list = []

    class Proc(MockChild):
        def __init__(self, cfg, host="127.0.0.1", port=8000):
            super().__init__(66, cfg.experiment_id)
            self.cfg, self.host, self.port = cfg, host, port
            self.log_path = None
            self.start_token = "tok"
            self.launch_cmd = []
            procs.append(self)

        def start(self):
            self.launch_cmd = vp._build_cmd(self.cfg, self.host, self.port)

        def wait_ready_verbose(self, log_fn):
            raise KeyboardInterrupt  # user hits Ctrl-C while vLLM compiles

    monkeypatch.setattr(bench_runner, "VLLMProcess", Proc)
    with pytest.raises(KeyboardInterrupt):
        bench_runner.run_experiment(config, ["p"])
    assert procs[0].stop_calls == 1 and procs[0].pid is None
    assert _lock_is_free()


def test_run_agent_abort_stops_owned_child_releases_lock_marks_cancelled(tmp_path):
    """Graph-level abort (Ctrl-C mid-run): owned mock child stopped, foreign PID untouched."""
    from inferops.task import default_task_for_workload

    db_path = tmp_path / "memory.db"
    task = default_task_for_workload("chat_short", 2)
    owned = MockChild(pid=3003, experiment_id="agent_candidate")
    foreign = MockChild(pid=4004, experiment_id="not_ours")
    killed: list[int] = []

    baseline = {
        "experiment_id": "sess_baseline",
        "param_changed": None,
        "value_changed": None,
        "throughput_rps": 2.0,
        "tokens_per_second": 128.0,
        "ttft_p50_ms": 48.0,
        "ttft_p99_ms": 70.0,
        "e2e_p50_ms": 900.0,
        "bottleneck": "compute-bound",
        "vs_baseline_pct": 0.0,
        "run_id": "aa",
        "validity_status": "valid",
        "mlflow_run_id": "m0",
        "has_config_evidence": True,
        "promotable": True,
        "failure_reason": "",
    }

    class FakeGraph:
        def invoke(self, state, config=None):
            # Executor spawned a managed child and registered it as owned …
            lease = ml.GPULease(
                host="127.0.0.1", port=8000, experiment_id="agent_candidate"
            ).acquire()
            lease.record_child(child_pid=3003, start_token="t", launch_cmd=None)
            ml.register_owned(owned, lease, "agent_candidate")
            # … then the user hit Ctrl-C.
            raise KeyboardInterrupt

    with (
        patch("inferops.agent.graph._run_baseline", return_value=(baseline, "compute-bound")),
        patch("inferops.agent.graph.build_graph", return_value=FakeGraph()),
        patch("inferops.agent.graph._print_run_summary"),
        patch("os.kill", lambda pid, sig: killed.append(pid)),
    ):
        with pytest.raises(KeyboardInterrupt):
            run_agent(
                workload_name="chat_short",
                llm=object(),
                session_prefix="sess_",
                task=task,
                db_path=db_path,
            )

    assert owned.stop_calls == 1 and owned.pid is None
    assert foreign.stop_calls == 0 and foreign.pid == 4004
    assert killed == []
    assert ml.owned_pids() == []
    assert _lock_is_free()
    assert get_task(task.task_id, db_path=db_path).status == "cancelled"


def test_abort_agent_run_marks_interrupted_for_non_cancel_errors(tmp_path):
    from inferops.memory.db import save_task
    from inferops.task import default_task_for_workload

    db_path = tmp_path / "memory.db"
    task = default_task_for_workload("chat_short", 2)
    save_task(
        task_id=task.task_id,
        session_prefix="s_",
        thread_id="s",
        confirmed_task=task.model_dump(mode="json"),
        status="running",
        db_path=db_path,
    )
    assert abort_agent_run(task.task_id, RuntimeError("boom"), db_path=db_path) == []
    assert get_task(task.task_id, db_path=db_path).status == "interrupted"

    child = MockChild(pid=8)
    ml.register_owned(child, None, "x")
    reports = abort_agent_run(task.task_id, bench_runner.TaskCancelled("stop"), db_path=db_path)
    assert [r["pid"] for r in reports] == [8] and child.stop_calls == 1
    assert get_task(task.task_id, db_path=db_path).status == "cancelled"


def test_no_lock_file_left_in_repo_root():
    # Guard for the eval / golden paths: the mutex must live under tmp, not cwd.
    root = Path(__file__).resolve().parents[1]
    assert not list(root.glob("*.lock"))
    assert not (root / "gpu.lock").exists()
