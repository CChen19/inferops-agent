"""Week-1 item ②: config application — restart + identity bind (mocked)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from inferops import bench_runner
from inferops.schemas import (
    ExperimentValidityStatus,
    config_knobs,
    derive_status,
    is_promotable,
    managed_start_evidence,
)
from inferops.tools import vllm_process as vp
from inferops.tools.vllm_process import (
    CLI_EVIDENCED_KNOB_KEYS,
    InstanceIdentity,
    LiveProbe,
    StopOccupantResult,
    assert_listener_bound_to_child,
    cli_evidenced_knobs,
    knobs_match_requested,
    parse_vllm_cli_knobs,
)


class _FakeLoad:
    total_requests = 2
    successful = 2
    total_time_s = 1.0
    throughput_rps = 2.0
    tokens_per_second = 10.0
    ttft_ms = [10.0, 12.0]
    e2e_ms = [20.0, 22.0]


class _FakeGPU:
    def __init__(self, *a, **k):
        pass

    def start(self):
        return None

    def stop(self):
        return SimpleNamespace(max_mem_used_gb=1.0, avg_util_pct=50.0)


def _patch_common(monkeypatch):
    monkeypatch.setattr(bench_runner, "init_mlflow", lambda *a, **k: None)
    monkeypatch.setattr(bench_runner, "log_experiment_result", lambda *a, **k: None)
    monkeypatch.setattr(bench_runner, "GPUMonitor", _FakeGPU)
    monkeypatch.setattr(
        bench_runner,
        "_run_load_with_cleanup_workaround",
        lambda *a, **k: _FakeLoad(),
    )

    class _Run:
        info = SimpleNamespace(run_id="mlflow-fake")

    class _Ctx:
        def __enter__(self):
            return _Run()

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(bench_runner, "mlflow_run", lambda **k: _Ctx())
    monkeypatch.delenv("INFEROPS_EXTERNAL_VLLM", raising=False)


def test_parse_and_match_cli_knobs(config):
    cmd = vp._build_cmd(config, "127.0.0.1", 8000)
    parsed = parse_vllm_cli_knobs(cmd)
    requested = cli_evidenced_knobs(config)
    assert knobs_match_requested(parsed, requested)
    assert "scheduler_policy" not in parsed
    assert "tensor_parallel_size" not in parsed
    assert set(CLI_EVIDENCED_KNOB_KEYS) == set(requested)


def test_cli_evidenced_excludes_unsupported(config):
    full = config_knobs(config)
    cli = cli_evidenced_knobs(config)
    assert "scheduler_policy" in full
    assert "scheduler_policy" not in cli
    assert "tensor_parallel_size" not in cli


def test_complete_coverage_cli_only_is_insufficient_not_valid(config):
    """P0-①: CLI-only actual cannot be status=valid / promotable."""
    requested = config_knobs(config)
    actual = cli_evidenced_knobs(config)
    ev = managed_start_evidence(
        process_pid=9, host="127.0.0.1", port=8000, observed_params=actual
    )
    status = derive_status(
        evidence=ev, actual_config=actual, requested_config=requested, successful_requests=2
    )
    assert status == ExperimentValidityStatus.INSUFFICIENT_EVIDENCE
    # Build a result-like check via is_promotable would need full ExperimentResult;
    # status alone already blocks promotion.


def test_healthy_but_different_must_restart(monkeypatch, config):
    """Already-healthy server with different knobs → managed restart + bound PID."""
    _patch_common(monkeypatch)

    old_identity = InstanceIdentity(
        host="127.0.0.1", port=8000, pid=111, start_token=None, source="proc_probe"
    )
    probe = LiveProbe(
        healthy=True,
        identity=old_identity,
        observed_knobs={
            **cli_evidenced_knobs(config),
            "max_num_batched_tokens": 9999,
        },
    )
    monkeypatch.setattr(bench_runner, "probe_live_instance", lambda *a, **k: probe)
    monkeypatch.setattr(
        bench_runner,
        "assert_listener_bound_to_child",
        lambda **k: 222,
    )

    events: list[str] = []
    stopped_occupant = {"n": 0}

    class FakeProc:
        def __init__(self, cfg, host="127.0.0.1", port=8000):
            self.cfg = cfg
            self.host = host
            self.port = port
            self.log_path = None
            self.start_token = "token-new"
            self.launch_cmd = []
            self._pid = None
            self.pre_restart_identity = None
            self.last_stop_result = None

        @property
        def pid(self):
            return self._pid

        def identity(self):
            return InstanceIdentity(
                host=self.host,
                port=self.port,
                pid=self._pid,
                start_token=self.start_token,
                source="managed_start",
            )

        def restart_after_stop(self, *, previous_identity, stop_occupant=True):
            events.append("restart")
            self.pre_restart_identity = previous_identity
            if stop_occupant:
                stopped_occupant["n"] += 1
            self.last_stop_result = StopOccupantResult(
                previous_pid=111, stop_attempted=True, still_listening=False
            )
            self.start()
            return self.last_stop_result

        def start(self):
            events.append("start")
            self.start_token = "token-new"
            self.launch_cmd = vp._build_cmd(self.cfg, self.host, self.port)
            self._pid = 222

        def wait_ready_verbose(self, log_fn):
            return True

        def evidenced_actual_config(self):
            return cli_evidenced_knobs(self.cfg)

        def stop(self):
            events.append("stop")
            self._pid = None

        def oom_in_log(self):
            return False

        def is_crashed(self):
            return False

    monkeypatch.setattr(bench_runner, "VLLMProcess", FakeProc)

    progress: list[str] = []
    result = bench_runner.run_experiment(
        config, ["p"], on_progress=progress.append, session_id="sess_"
    )

    assert "restart" in events
    assert stopped_occupant["n"] == 1
    assert any("restarting_managed_vllm" in p for p in progress)
    assert any("identity_verified:changed" in p for p in progress)
    assert result.config_evidence is not None
    assert result.config_evidence.kind == "managed_process_start"
    assert result.config_evidence.verified is True
    assert result.config_evidence.process_pid == 222
    assert result.config_evidence.instance_id != old_identity.instance_id
    assert "pid=222" in (result.config_evidence.instance_id or "")
    assert result.actual_config is not None
    assert result.actual_config["max_num_batched_tokens"] == config.max_num_batched_tokens
    assert "scheduler_policy" not in result.actual_config
    # Complete-coverage: CLI-only actual → insufficient_evidence (not valid)
    assert result.status == ExperimentValidityStatus.INSUFFICIENT_EVIDENCE
    assert is_promotable(result) is False
    assert result.run_id


def test_external_health_only_insufficient_evidence(monkeypatch, config):
    _patch_common(monkeypatch)
    monkeypatch.setenv("INFEROPS_EXTERNAL_VLLM", "1")
    monkeypatch.setattr(
        bench_runner,
        "probe_live_instance",
        lambda *a, **k: LiveProbe(healthy=True, identity=None, observed_knobs=None),
    )
    monkeypatch.setattr(
        bench_runner,
        "VLLMProcess",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not start managed")),
    )

    progress: list[str] = []
    result = bench_runner.run_experiment(config, ["p"], on_progress=progress.append)

    assert result.status == ExperimentValidityStatus.INSUFFICIENT_EVIDENCE
    assert result.actual_config is None
    assert result.config_evidence is not None
    assert result.config_evidence.kind == "external_unverified"
    assert result.config_evidence.verified is False
    assert any("external_health_only" in p for p in progress)


def test_identity_change_required_after_restart(monkeypatch, config):
    _patch_common(monkeypatch)

    old = InstanceIdentity(host="127.0.0.1", port=8000, pid=4242, source="proc_probe")
    monkeypatch.setattr(
        bench_runner,
        "probe_live_instance",
        lambda *a, **k: LiveProbe(
            healthy=True,
            identity=old,
            observed_knobs={"max_num_batched_tokens": 1},
        ),
    )
    # Bind would succeed with same PID — identity-change check must still fail.
    monkeypatch.setattr(bench_runner, "assert_listener_bound_to_child", lambda **k: 4242)

    stopped = {"n": 0}

    class StaleProc:
        def __init__(self, cfg, host="127.0.0.1", port=8000):
            self.cfg = cfg
            self.host = host
            self.port = port
            self.log_path = None
            self.start_token = "same"
            self._pid = 4242

        @property
        def pid(self):
            return self._pid

        def identity(self):
            return InstanceIdentity(
                host=self.host,
                port=self.port,
                pid=self._pid,
                start_token=self.start_token,
                source="managed_start",
            )

        def restart_after_stop(self, **k):
            return StopOccupantResult(
                previous_pid=4242, stop_attempted=True, still_listening=False
            )

        def start(self):
            return None

        def wait_ready_verbose(self, log_fn):
            return True

        def stop(self):
            stopped["n"] += 1
            self._pid = None

        def oom_in_log(self):
            return False

        def is_crashed(self):
            return False

        def evidenced_actual_config(self):
            return cli_evidenced_knobs(self.cfg)

    monkeypatch.setattr(bench_runner, "VLLMProcess", StaleProc)

    with pytest.raises(bench_runner.BenchmarkError, match="PID did not change") as ei:
        bench_runner.run_experiment(config, ["p"])

    assert stopped["n"] >= 1  # orphan cleanup
    assert ei.value.result is not None
    assert ei.value.result.status == ExperimentValidityStatus.FAILED
    assert ei.value.result.run_id
    assert ei.value.result.actual_config is None


def test_stop_failure_refuses_start(monkeypatch, config):
    """If stop leaves a listener, do not spawn / never mark valid."""
    _patch_common(monkeypatch)
    old = InstanceIdentity(host="127.0.0.1", port=8000, pid=50, source="proc_probe")
    monkeypatch.setattr(
        bench_runner,
        "probe_live_instance",
        lambda *a, **k: LiveProbe(
            healthy=True, identity=old, observed_knobs={"max_num_seqs": 1}
        ),
    )

    started = {"n": 0}
    stopped = {"n": 0}

    class NoStopProc:
        def __init__(self, cfg, host="127.0.0.1", port=8000):
            self.cfg = cfg
            self.host = host
            self.port = port
            self.log_path = None
            self.start_token = "t"
            self._pid = None

        @property
        def pid(self):
            return self._pid

        def identity(self):
            return InstanceIdentity(
                host=self.host, port=self.port, pid=self._pid, start_token=self.start_token
            )

        def restart_after_stop(self, **k):
            # Simulates stop_port_occupant failure: still listening, no start.
            return StopOccupantResult(
                previous_pid=50,
                stop_attempted=True,
                still_listening=True,
                listener_pid_after=50,
            )

        def start(self):
            started["n"] += 1
            self._pid = 99

        def stop(self):
            stopped["n"] += 1
            self._pid = None

        def wait_ready_verbose(self, log_fn):
            return True

        def evidenced_actual_config(self):
            return cli_evidenced_knobs(self.cfg)

        def oom_in_log(self):
            return False

        def is_crashed(self):
            return False

    monkeypatch.setattr(bench_runner, "VLLMProcess", NoStopProc)

    with pytest.raises(bench_runner.BenchmarkError, match="Failed to stop prior occupant") as ei:
        bench_runner.run_experiment(config, ["p"])

    assert started["n"] == 0
    assert ei.value.result is not None
    assert ei.value.result.status == ExperimentValidityStatus.FAILED
    assert ei.value.result.status != ExperimentValidityStatus.VALID


def test_listener_pid_mismatch_fails_and_stops_child(monkeypatch, config):
    """Health OK on stale occupant (listener != child) → failed, child stopped."""
    _patch_common(monkeypatch)
    monkeypatch.setattr(
        bench_runner,
        "probe_live_instance",
        lambda *a, **k: LiveProbe(healthy=False),
    )

    def _bad_bind(**k):
        raise RuntimeError(
            "listener PID 111 != managed child PID 222 "
            "(stale occupant or stop failure — never valid)"
        )

    monkeypatch.setattr(bench_runner, "assert_listener_bound_to_child", _bad_bind)

    stopped = {"n": 0}

    class ChildProc:
        def __init__(self, cfg, host="127.0.0.1", port=8000):
            self.cfg = cfg
            self.host = host
            self.port = port
            self.log_path = None
            self.start_token = "tok"
            self._pid = None
            self.launch_cmd = []

        @property
        def pid(self):
            return self._pid

        def identity(self):
            return InstanceIdentity(
                host=self.host, port=self.port, pid=self._pid, start_token=self.start_token
            )

        def start(self):
            self._pid = 222
            self.launch_cmd = vp._build_cmd(self.cfg, self.host, self.port)

        def wait_ready_verbose(self, log_fn):
            return True

        def stop(self):
            stopped["n"] += 1
            self._pid = None

        def oom_in_log(self):
            return False

        def is_crashed(self):
            return False

        def evidenced_actual_config(self):
            return cli_evidenced_knobs(self.cfg)

    monkeypatch.setattr(bench_runner, "VLLMProcess", ChildProc)

    with pytest.raises(bench_runner.BenchmarkError, match="listener PID") as ei:
        bench_runner.run_experiment(config, ["p"])

    assert stopped["n"] >= 1
    assert ei.value.result is not None
    assert ei.value.result.status == ExperimentValidityStatus.FAILED


def test_stale_occupant_still_healthy_unknown_pid(monkeypatch, config):
    """Stop reports still listening with unknown PID → failed, no valid."""
    _patch_common(monkeypatch)
    monkeypatch.setattr(
        bench_runner,
        "probe_live_instance",
        lambda *a, **k: LiveProbe(
            healthy=True,
            identity=InstanceIdentity(host="127.0.0.1", port=8000, pid=7),
            observed_knobs=None,
        ),
    )

    class UnknownPidStop:
        def __init__(self, cfg, host="127.0.0.1", port=8000):
            self.cfg = cfg
            self.host = host
            self.port = port
            self.log_path = None
            self.start_token = "t"
            self._pid = None

        @property
        def pid(self):
            return self._pid

        def identity(self):
            return InstanceIdentity(
                host=self.host, port=self.port, pid=self._pid, start_token=self.start_token
            )

        def restart_after_stop(self, **k):
            return StopOccupantResult(
                previous_pid=7,
                stop_attempted=True,
                still_listening=True,
                listener_pid_after=None,
            )

        def start(self):
            raise AssertionError("must not start when still listening")

        def stop(self):
            return None

        def wait_ready_verbose(self, log_fn):
            return True

        def evidenced_actual_config(self):
            return {}

        def oom_in_log(self):
            return False

        def is_crashed(self):
            return False

    monkeypatch.setattr(bench_runner, "VLLMProcess", UnknownPidStop)

    with pytest.raises(bench_runner.BenchmarkError, match="Failed to stop") as ei:
        bench_runner.run_experiment(config, ["p"])
    assert ei.value.result.status == ExperimentValidityStatus.FAILED


def test_mismatch_yields_invalid(monkeypatch, config):
    """Critical evidence + shared key mismatch → invalid (never valid)."""
    _patch_common(monkeypatch)
    monkeypatch.setattr(
        bench_runner,
        "probe_live_instance",
        lambda *a, **k: LiveProbe(healthy=False),
    )
    monkeypatch.setattr(bench_runner, "assert_listener_bound_to_child", lambda **k: 7)

    class MismatchProc:
        def __init__(self, cfg, host="127.0.0.1", port=8000):
            self.cfg = cfg
            self.host = host
            self.port = port
            self.log_path = None
            self.start_token = "tok"
            self._pid = 7
            self.launch_cmd = []

        @property
        def pid(self):
            return self._pid

        def identity(self):
            return InstanceIdentity(
                host=self.host,
                port=self.port,
                pid=self._pid,
                start_token=self.start_token,
                source="managed_start",
            )

        def start(self):
            self.launch_cmd = vp._build_cmd(self.cfg, self.host, self.port)

        def wait_ready_verbose(self, log_fn):
            return True

        def evidenced_actual_config(self):
            knobs = cli_evidenced_knobs(self.cfg)
            knobs["max_num_batched_tokens"] = knobs["max_num_batched_tokens"] + 1
            return knobs

        def stop(self):
            return None

        def oom_in_log(self):
            return False

        def is_crashed(self):
            return False

    monkeypatch.setattr(bench_runner, "VLLMProcess", MismatchProc)

    progress: list[str] = []
    result = bench_runner.run_experiment(config, ["p"], on_progress=progress.append)

    assert result.status == ExperimentValidityStatus.INVALID
    assert result.config_evidence is not None
    assert result.config_evidence.is_critical_evidence()
    assert any("invalid:knob_mismatch" in p for p in progress)
    assert is_promotable(result) is False


def test_startup_oom_failed_row_keeps_run_id(monkeypatch, config):
    _patch_common(monkeypatch)
    monkeypatch.setattr(
        bench_runner,
        "probe_live_instance",
        lambda *a, **k: LiveProbe(healthy=False),
    )

    stopped = {"n": 0}

    class OOMProc:
        def __init__(self, cfg, host="127.0.0.1", port=8000):
            self.cfg = cfg
            self.host = host
            self.port = port
            self.log_path = "logs/fake.log"
            self.start_token = "t"
            self._pid = 9

        @property
        def pid(self):
            return self._pid

        def identity(self):
            return InstanceIdentity(
                host=self.host, port=self.port, pid=self._pid, start_token=self.start_token
            )

        def start(self):
            return None

        def wait_ready_verbose(self, log_fn):
            return False

        def oom_in_log(self):
            return True

        def is_crashed(self):
            return True

        def exit_code(self):
            return 1

        def stop(self):
            stopped["n"] += 1
            self._pid = None

        def evidenced_actual_config(self):
            return {}

    monkeypatch.setattr(bench_runner, "VLLMProcess", OOMProc)

    with pytest.raises(bench_runner.OOMError) as ei:
        bench_runner.run_experiment(config, ["p"])

    assert stopped["n"] >= 1
    failed = ei.value.result
    assert failed is not None
    assert failed.status == ExperimentValidityStatus.FAILED
    assert failed.run_id
    assert failed.requested_config
    assert failed.actual_config is None


def test_assert_listener_bound_helper():
    with pytest.raises(RuntimeError, match="child PID is unknown"):
        assert_listener_bound_to_child(host="127.0.0.1", port=8000, child_pid=None)


def test_managed_start_evidence_includes_generation():
    ev = managed_start_evidence(
        process_pid=5,
        host="127.0.0.1",
        port=8000,
        observed_params={"max_num_seqs": 64},
        start_token="abc",
    )
    assert "gen=abc" in (ev.instance_id or "")
    assert ev.is_critical_evidence()


def test_run_benchmark_persists_failed_row(monkeypatch, config, tmp_db):
    """Failed-row persistence is via run_benchmark (and executor), not raw run_experiment."""
    from inferops.tools import run_benchmark as rb
    from inferops.schemas import ExperimentResult, empty_latency

    failed = ExperimentResult(
        experiment_id="fail_persist",
        config=config,
        total_requests=0,
        successful_requests=0,
        total_time_s=0.0,
        throughput_rps=0.0,
        tokens_per_second=0.0,
        ttft=empty_latency(),
        tpot=empty_latency(),
        e2e_latency=empty_latency(),
        run_id="deadbeefdeadbeefdeadbeefdeadbeef",
        status=ExperimentValidityStatus.FAILED,
        notes="oom",
    )

    def _boom(*a, **k):
        raise bench_runner.OOMError("oom", result=failed)

    monkeypatch.setattr(rb, "run_experiment", _boom)
    monkeypatch.setattr(rb, "get_prompts", lambda w: ["p"])
    saved = {}

    def _save(result, db_path=None):
        saved["result"] = result

    monkeypatch.setattr(rb, "save_result", _save)

    with pytest.raises(bench_runner.OOMError):
        rb.run_benchmark(
            rb.RunBenchmarkInput(
                experiment_id="fail_persist",
                workload_name="chat_short",
                persist=True,
            )
        )
    assert saved["result"].run_id == failed.run_id
    assert saved["result"].status == ExperimentValidityStatus.FAILED


def test_live_identity_probe_written_while_child_alive(monkeypatch, config, tmp_path):
    """pid_equality progress + live_identity_*.json emitted before teardown."""
    _patch_common(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        bench_runner,
        "probe_live_instance",
        lambda *a, **k: LiveProbe(healthy=False),
    )
    monkeypatch.setattr(bench_runner, "assert_listener_bound_to_child", lambda **k: 77)

    class AliveProc:
        def __init__(self, cfg, host="127.0.0.1", port=8000):
            self.cfg = cfg
            self.host = host
            self.port = port
            self.log_path = None
            self.start_token = "tok"
            self._pid = 77
            self.launch_cmd = []

        @property
        def pid(self):
            return self._pid

        def identity(self):
            return InstanceIdentity(
                host=self.host, port=self.port, pid=self._pid, start_token=self.start_token
            )

        def start(self):
            self.launch_cmd = vp._build_cmd(self.cfg, self.host, self.port)

        def wait_ready_verbose(self, log_fn):
            return True

        def evidenced_actual_config(self):
            return cli_evidenced_knobs(self.cfg)

        def stop(self):
            self._pid = None

        def oom_in_log(self):
            return False

        def is_crashed(self):
            return False

    monkeypatch.setattr(bench_runner, "VLLMProcess", AliveProc)
    progress: list[str] = []
    result = bench_runner.run_experiment(config, ["p"], on_progress=progress.append)
    eq = [p for p in progress if p.startswith("status:pid_equality:")]
    assert eq == ["status:pid_equality:listener=77:child=77"]
    probe = tmp_path / "logs" / f"live_identity_{config.experiment_id}.json"
    assert probe.exists()
    import json
    data = json.loads(probe.read_text())
    assert data["pids_equal"] is True
    assert data["listener_pid"] == data["child_pid"] == 77
    assert result.status == ExperimentValidityStatus.INSUFFICIENT_EVIDENCE


def test_simulate_stop_failure_env(monkeypatch, config):
    _patch_common(monkeypatch)
    monkeypatch.setenv("INFEROPS_SIMULATE_STOP_FAILURE", "1")
    old = InstanceIdentity(host="127.0.0.1", port=8000, pid=50, source="proc_probe")
    monkeypatch.setattr(
        bench_runner,
        "probe_live_instance",
        lambda *a, **k: LiveProbe(
            healthy=True, identity=old, observed_knobs={"max_num_seqs": 1}
        ),
    )
    # Use real restart_after_stop path with mocked stop_port_occupant via env
    started = {"n": 0}

    class Proc:
        def __init__(self, cfg, host="127.0.0.1", port=8000):
            self.cfg = cfg
            self.host = host
            self.port = port
            self.log_path = None
            self.start_token = "t"
            self._pid = None
            self.last_stop_result = None

        @property
        def pid(self):
            return self._pid

        def identity(self):
            return InstanceIdentity(
                host=self.host, port=self.port, pid=self._pid, start_token=self.start_token
            )

        def restart_after_stop(self, **k):
            # Delegate to real helper semantics: env forces still_listening
            from inferops.tools.vllm_process import stop_port_occupant
            result = stop_port_occupant(self.host, self.port)
            self.last_stop_result = result
            if not result.still_listening:
                self.start()
            return result

        def start(self):
            started["n"] += 1
            self._pid = 99

        def stop(self):
            self._pid = None

        def wait_ready_verbose(self, log_fn):
            return True

        def evidenced_actual_config(self):
            return {}

        def oom_in_log(self):
            return False

        def is_crashed(self):
            return False

    monkeypatch.setattr(bench_runner, "VLLMProcess", Proc)
    with pytest.raises(bench_runner.BenchmarkError, match="Failed to stop") as ei:
        bench_runner.run_experiment(config, ["p"])
    assert started["n"] == 0
    assert ei.value.result.status == ExperimentValidityStatus.FAILED


def test_simulate_startup_identity_failure(monkeypatch, config):
    _patch_common(monkeypatch)
    monkeypatch.setenv("INFEROPS_SIMULATE_STARTUP_FAILURE", "identity")
    monkeypatch.setattr(
        bench_runner,
        "probe_live_instance",
        lambda *a, **k: LiveProbe(healthy=False),
    )
    stopped = {"n": 0}
    waited = {"n": 0}

    class Proc:
        def __init__(self, cfg, host="127.0.0.1", port=8000):
            self.cfg = cfg
            self.host = host
            self.port = port
            self.log_path = None
            self.start_token = "t"
            self._pid = 5

        @property
        def pid(self):
            return self._pid

        def identity(self):
            return InstanceIdentity(
                host=self.host, port=self.port, pid=self._pid, start_token=self.start_token
            )

        def start(self):
            return None

        def wait_ready_verbose(self, log_fn):
            waited["n"] += 1
            return True

        def stop(self):
            stopped["n"] += 1
            self._pid = None

        def evidenced_actual_config(self):
            return {}

        def oom_in_log(self):
            return False

        def is_crashed(self):
            return False

    monkeypatch.setattr(bench_runner, "VLLMProcess", Proc)
    with pytest.raises(bench_runner.BenchmarkError, match="SIMULATE_STARTUP_FAILURE=identity"):
        bench_runner.run_experiment(config, ["p"])
    assert stopped["n"] >= 1
    assert waited["n"] == 0  # must abort before readiness wait


@pytest.mark.parametrize(
    "mode,exc_type,match",
    [
        ("oom", bench_runner.OOMError, "SIMULATE_STARTUP_FAILURE=oom"),
        ("timeout", bench_runner.StartupTimeoutError, "SIMULATE_STARTUP_FAILURE=timeout"),
    ],
)
def test_simulate_startup_oom_and_timeout_before_wait(monkeypatch, config, mode, exc_type, match):
    """oom/timeout inject before wait_ready — fast, independent of real health."""
    _patch_common(monkeypatch)
    monkeypatch.setenv("INFEROPS_SIMULATE_STARTUP_FAILURE", mode)
    monkeypatch.setattr(
        bench_runner,
        "probe_live_instance",
        lambda *a, **k: LiveProbe(healthy=False),
    )
    waited = {"n": 0}
    stopped = {"n": 0}

    class Proc:
        def __init__(self, cfg, host="127.0.0.1", port=8000):
            self.cfg = cfg
            self.host = host
            self.port = port
            self.log_path = "logs/fake.log"
            self.start_token = "t"
            self._pid = 9

        @property
        def pid(self):
            return self._pid

        def identity(self):
            return InstanceIdentity(
                host=self.host, port=self.port, pid=self._pid, start_token=self.start_token
            )

        def start(self):
            return None

        def wait_ready_verbose(self, log_fn):
            waited["n"] += 1
            raise AssertionError("wait_ready must not run under simulate startup failure")

        def stop(self):
            stopped["n"] += 1
            self._pid = None

        def evidenced_actual_config(self):
            return {}

        def oom_in_log(self):
            return False

        def is_crashed(self):
            return False

    monkeypatch.setattr(bench_runner, "VLLMProcess", Proc)
    with pytest.raises(exc_type, match=match) as ei:
        bench_runner.run_experiment(config, ["p"])
    assert waited["n"] == 0
    assert stopped["n"] >= 1
    assert ei.value.result is not None
    assert ei.value.result.status == ExperimentValidityStatus.FAILED
