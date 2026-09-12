# Week-1 Item ② — Config Application Implementation Notes

**Audience:** Chris (local Windows RTX 3060 / WSL GPU evidence)  
**Acceptance checklist:** [`week1_config_application_acceptance.md`](./week1_config_application_acceptance.md)  
**Contract (complete-coverage):** [`week1_experiment_contract.md`](./week1_experiment_contract.md)

No GPU runs or invented metrics in this document — **commands + assertions only**.

## Behavior (code)

| Path | Behavior |
|---|---|
| **Managed (default)** | Healthy + knobs differ / identity unknown → **stop occupant**, then start with requested CLI. After `/health` OK, require **listener PID == managed child PID**. Evidence: `managed_process_start` (`verified=True`). `actual_config` = **CLI-evidenced keys only**. |
| **Complete coverage (P0-①)** | CLI-only `actual_config` does **not** cover non-CLI requested knobs (`scheduler_policy`, `tensor_parallel_size`, …) → **`status=insufficient_evidence`** (critical evidence may still be present). **`promotable=0` / not promotable**. Never claim `status=valid` from CLI-only actual. |
| **Stop / identity failure** | Stop fails, stale occupant still healthy, listener PID unknown, or `listener PID != child PID` → `status=failed`, **never valid**. Spawned child is always terminated (no orphans). |
| **External** | `INFEROPS_EXTERNAL_VLLM=1` → no restart; health OK → `insufficient_evidence`, `actual_config=null`. |
| **Failed-row persistence** | `run_experiment` attaches `BenchmarkError.result` with the **same `run_id`**. **SQLite persistence** happens in `run_benchmark` (and executor) when `persist=True` — not when calling `run_experiment` alone. |

### In-run PID capture (important)

After `run_benchmark` / `run_experiment` returns, the managed child is **already stopped**.
Post-hoc `ss` **cannot** verify `listener PID == process_pid`.

While the child is still alive (after identity bind, before teardown) the runner:

1. Emits progress: `status:pid_equality:listener=<PID>:child=<PID>`
2. Writes `logs/live_identity_<experiment_id>.json` with both PIDs
3. Optionally holds if `INFEROPS_PID_PROBE_HOLD_S=<seconds>` so you can run `ss` in another shell

## Pre-flight (copy-paste)

```bash
# Repo root (WSL recommended)
git fetch origin
git checkout cursor/w1-config-application-564d   # or merge this PR branch

export PATH="$HOME/.local/bin:$PATH"
unset INFEROPS_EXTERNAL_VLLM
unset INFEROPS_SIMULATE_STOP_FAILURE
unset INFEROPS_SIMULATE_STARTUP_FAILURE
export VLLM_HOST=127.0.0.1
export VLLM_PORT=8000
# export INFEROPS_VLLM_PYTHON=/path/to/vllm-dev/bin/python

pytest -q
pytest -q tests/test_config_application.py
```

## A. Managed restart + PID equality (GPU) — capture WHILE child is alive

```bash
# Terminal 1 — leave a healthy server with knobs you will override
VLLM_GPU_MEM=0.80 bash scripts/start_vllm.sh 0.5B

# Terminal 2 — record OLD occupant PID, then run with live PID probe
ss -ltnp 'sport = :8000'
# note users:(("python",pid=OLD,...))  → export OLD_PID=...

# Optional: hold 45s after identity bind so you can ss while child is up
export INFEROPS_PID_PROBE_HOLD_S=45

python - <<'PY'
import re
import subprocess
from pathlib import Path

from inferops.bench_runner import run_experiment
from inferops.memory.db import save_result
from inferops.schemas import config_knobs
from inferops.tools.vllm_process import cli_evidenced_knobs
from configs.search_space import make_configs
from workloads.definitions import ALL_WORKLOADS, get_prompts

wl = next(w for w in ALL_WORKLOADS if w.name == "chat_short")
cfg = make_configs(wl)[0].model_copy(update={
    "experiment_id": "w1_cfg_restart_probe",
    "max_num_batched_tokens": 4096,
    "tags": {"session_id": "w1cfg_"},
})
events: list[str] = []

def on_progress(msg: str) -> None:
    print(msg, flush=True)
    events.append(msg)
    # WHILE CHILD ALIVE: on pid_equality, probe listener immediately
    m = re.match(r"status:pid_equality:listener=(\d+):child=(\d+)$", msg)
    if m:
        listener_i, child_i = int(m.group(1)), int(m.group(2))
        assert listener_i == child_i, (listener_i, child_i)
        ss = subprocess.check_output(["ss", "-ltnp", "sport = :8000"], text=True)
        print("--- LIVE ss (child still up) ---", flush=True)
        print(ss, flush=True)
        assert f"pid={child_i}" in ss or f"pid={listener_i}" in ss, ss

result = run_experiment(cfg, get_prompts(wl), on_progress=on_progress, session_id="w1cfg_")
save_result(result)  # same persistence path as run_benchmark(persist=True)

eq = [e for e in events if e.startswith("status:pid_equality:")]
assert eq, "missing in-run pid_equality progress (child must be probed before teardown)"
print("status=", result.status.value, "promotable-related actual keys=", sorted((result.actual_config or {})))
assert result.status.value == "insufficient_evidence", result.status  # CLI-only actual
assert "scheduler_policy" not in (result.actual_config or {})
probe = Path("logs/live_identity_w1_cfg_restart_probe.json")
assert probe.exists(), probe
print(probe.read_text())
PY

# After return the child is stopped — do NOT use post-hoc ss for PID equality.
# Compare live probe file to SQLite evidence PID instead:
python - <<'PY'
import json
from pathlib import Path
import sqlite3
probe = json.loads(Path("logs/live_identity_w1_cfg_restart_probe.json").read_text())
assert probe["pids_equal"] is True
assert probe["listener_pid"] == probe["child_pid"]
conn = sqlite3.connect("inferops_memory.db")
row = conn.execute(
    "SELECT json_extract(result_json,'$.config_evidence.process_pid') "
    "FROM experiments WHERE experiment_id=? ORDER BY created_at DESC LIMIT 1",
    ("w1_cfg_restart_probe",),
).fetchone()
assert row and int(row[0]) == probe["child_pid"], (row, probe)
print("OK live listener==child==sqlite process_pid", probe["child_pid"])
PY

unset INFEROPS_PID_PROBE_HOLD_S
```

### Assertions (checkboxes)

- [ ] Console shows managed restart / stopping occupant (not “skipping lifecycle”)
- [ ] In-run line `status:pid_equality:listener=<N>:child=<N>` with **equal** N (captured before teardown)
- [ ] Optional: during `INFEROPS_PID_PROBE_HOLD_S`, live `ss` shows the same PID
- [ ] `logs/live_identity_w1_cfg_restart_probe.json` has `pids_equal: true` and matching `listener_pid` / `child_pid`
- [ ] SQLite `config_evidence.process_pid` == live probe `child_pid` (not a post-hoc `ss` after stop)
- [ ] Live child PID ≠ pre-restart `OLD_PID`
- [ ] `actual_config` has CLI keys only (no `scheduler_policy` / `tensor_parallel_size`)
- [ ] `status=insufficient_evidence` (CLI-only actual; complete-coverage) — **never `valid` from health / CLI-only**
- [ ] `config_evidence.kind=managed_process_start`, verified true
- [ ] `promotable=0` / not selected as best while non-CLI requested knobs remain uncovered

### SQLite checks

```bash
sqlite3 inferops_memory.db <<'SQL'
.headers on
.mode column
SELECT experiment_id, run_id, status, promotable,
       json_extract(result_json, '$.config_evidence.kind') AS ev_kind,
       json_extract(result_json, '$.config_evidence.verified') AS ev_verified,
       json_extract(result_json, '$.config_evidence.process_pid') AS ev_pid,
       json_extract(result_json, '$.actual_config.max_num_batched_tokens') AS actual_batched,
       json_extract(result_json, '$.actual_config.scheduler_policy') AS actual_sched
FROM experiments
WHERE experiment_id LIKE 'w1_cfg%'
ORDER BY created_at DESC
LIMIT 5;
SQL
```

Expect: `status=insufficient_evidence`; `promotable=0`; `actual_sched` **NULL**; `ev_pid` matches `logs/live_identity_*.json` `child_pid`.

### MLflow tags (resolve experiment by **name**, not id `"0"`)

Default tracking URI is `sqlite:///mlruns.db`; default experiment name is `inferops`.

```bash
python - <<'PY'
import mlflow
from mlflow.tracking import MlflowClient

# Resolve experiment id by name (do NOT hardcode "0")
name = "inferops"
client = MlflowClient()  # uses MLFLOW_TRACKING_URI if set
exp = client.get_experiment_by_name(name)
if exp is None:
    raise SystemExit(
        f"MLflow experiment {name!r} not found. "
        f"List: {[e.name for e in client.search_experiments()]}"
    )
print("experiment_name=", exp.name, "experiment_id=", exp.experiment_id)

runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    order_by=["attribute.start_time DESC"],
    max_results=5,
)
for r in runs:
    print(
        r.info.run_id,
        r.data.tags.get("run_id"),
        r.data.tags.get("status"),
        r.data.tags.get("experiment_id"),
    )
PY
```

`run_id` tag on the MLflow run must match the SQLite `run_id` column for the same attempt.

## B. Deterministic stop-failure / startup-failure injection

These use tiny env hooks (no VRAM starve, no sudo). Always `unset` afterward.

### B1. Stop failure (stale occupant path → `failed`, never valid)

Requires a healthy listener so managed restart is attempted:

```bash
# Terminal 1
bash scripts/start_vllm.sh 0.5B
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/health   # 200

# Terminal 2
export INFEROPS_SIMULATE_STOP_FAILURE=1
unset INFEROPS_SIMULATE_STARTUP_FAILURE
python - <<'PY'
from inferops.tools.run_benchmark import RunBenchmarkInput, run_benchmark
from inferops.bench_runner import BenchmarkError
try:
    run_benchmark(RunBenchmarkInput(
        experiment_id="w1_cfg_stop_fail",
        config_patch={"max_num_batched_tokens": 4096},
        workload_name="chat_short",
        persist=True,
        session_id="w1cfg_",
    ))
except BenchmarkError as exc:
    assert exc.result is not None
    assert exc.result.status.value == "failed"
    assert exc.result.run_id
    print("OK stop-failure", exc.result.run_id, exc.result.notes)
else:
    raise SystemExit("expected BenchmarkError on stop failure")
PY
unset INFEROPS_SIMULATE_STOP_FAILURE

# No new orphan managed child should remain from the refused start:
ss -ltnp 'sport = :8000' || true
# (Terminal 1 occupant may still be up — that is expected; InferOps refused to take over.)
```

- [ ] `status=failed` (not valid)
- [ ] Persisted row via `run_benchmark` has same `run_id` as `BenchmarkError.result` / MLflow tags
- [ ] No InferOps-spawned orphan left after the failed attempt

### B2. Startup failure — simulated OOM

```bash
# Prefer no healthy occupant so the managed fresh-start path is used:
# (stop Terminal 1 vLLM first if still running)

unset INFEROPS_SIMULATE_STOP_FAILURE
export INFEROPS_SIMULATE_STARTUP_FAILURE=oom
python - <<'PY'
from inferops.tools.run_benchmark import RunBenchmarkInput, run_benchmark
from inferops.bench_runner import OOMError
try:
    run_benchmark(RunBenchmarkInput(
        experiment_id="w1_cfg_oom_fail",
        workload_name="chat_short",
        persist=True,
        session_id="w1cfg_",
    ))
except OOMError as exc:
    assert exc.result is not None and exc.result.status.value == "failed"
    print("OK oom-failure", exc.result.run_id)
else:
    raise SystemExit("expected OOMError")
PY
unset INFEROPS_SIMULATE_STARTUP_FAILURE
```

### B3. Startup failure — simulated identity bind failure

```bash
unset INFEROPS_SIMULATE_STOP_FAILURE
export INFEROPS_SIMULATE_STARTUP_FAILURE=identity
python - <<'PY'
from inferops.tools.run_benchmark import RunBenchmarkInput, run_benchmark
from inferops.bench_runner import BenchmarkError
try:
    run_benchmark(RunBenchmarkInput(
        experiment_id="w1_cfg_identity_fail",
        workload_name="chat_short",
        persist=True,
        session_id="w1cfg_",
    ))
except BenchmarkError as exc:
    assert exc.result is not None and exc.result.status.value == "failed"
    print("OK identity-failure", exc.result.run_id, exc.result.notes)
else:
    raise SystemExit("expected BenchmarkError")
PY
unset INFEROPS_SIMULATE_STARTUP_FAILURE
```

- [ ] Each injection yields `status=failed`, same `run_id` on persisted row
- [ ] Never `valid` / never promotable

## C. External path

```bash
bash scripts/start_vllm.sh 0.5B
export INFEROPS_EXTERNAL_VLLM=1
python - <<'PY'
from inferops.tools.run_benchmark import RunBenchmarkInput, run_benchmark
out = run_benchmark(RunBenchmarkInput(
    experiment_id="w1_cfg_external",
    workload_name="chat_short",
    persist=True,
    session_id="w1cfg_",
))
print(out)
assert out.status == "insufficient_evidence"
PY
unset INFEROPS_EXTERNAL_VLLM
```

- [ ] `status=insufficient_evidence`, `actual_config` null, evidence `external_unverified`
- [ ] Not promotable (`promotable=0`)

## D. Failed-row persistence note

| Entry point | Failed row attached on `BenchmarkError.result`? | Written to SQLite? |
|---|---|---|
| `run_experiment(...)` | Yes (same `run_id`) | **No** (caller must persist) |
| `run_benchmark(..., persist=True)` | Yes | **Yes** (`save_result`) |
| Agent `executor_node` | Yes | Yes (via `run_benchmark`) + summary/trajectory |

Section A uses `run_experiment` + `save_result` so `on_progress` can probe PIDs live; B/C use `run_benchmark` for the normal persist path.

## E. CI proof (no GPU)

```text
pytest -q
pytest -q tests/test_config_application.py
```

Targeted cases: healthy-but-different restart, external health-only, identity change, stop failure, listener≠child, stale unknown PID, mismatch→invalid, OOM cleanup + run_id, simulate-stop / simulate-startup hooks, live identity probe file, `run_benchmark` persists failed row.
