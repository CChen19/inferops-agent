# Week-1 Item ② — Config Application Implementation Notes

**Audience:** Chris (local Windows RTX 3060 / WSL GPU evidence)  
**Acceptance checklist:** [`week1_config_application_acceptance.md`](./week1_config_application_acceptance.md)  
**Contract (complete-coverage):** [`week1_experiment_contract.md`](./week1_experiment_contract.md)

No GPU runs or invented metrics in this document — **commands + assertions only**.

## Behavior (code)

| Path | Behavior |
|---|---|
| **Managed (default)** | Healthy + knobs differ / identity unknown → **stop occupant**, then start with requested CLI. After `/health` OK, require **listener PID == managed child PID**. Evidence: `managed_process_start` (`verified=True`). `actual_config` = **CLI-evidenced keys only**. |
| **Complete coverage (P0-①)** | CLI-only `actual_config` does **not** cover non-CLI requested knobs (`scheduler_policy`, `tensor_parallel_size`, …) → `status=insufficient_evidence` (critical evidence may still be present). **Not promotable** until every requested key is evidenced. Never claim `status=valid` from CLI-only actual. |
| **Stop / identity failure** | Stop fails, stale occupant still healthy, listener PID unknown, or `listener PID != child PID` → `status=failed`, **never valid**. Spawned child is always terminated (no orphans). |
| **External** | `INFEROPS_EXTERNAL_VLLM=1` → no restart; health OK → `insufficient_evidence`, `actual_config=null`. |
| **Failed-row persistence** | `run_experiment` attaches `BenchmarkError.result` with the **same `run_id`**. **SQLite persistence** happens in `run_benchmark` (and executor) when `persist=True` — not when calling `run_experiment` alone. |

## Pre-flight (copy-paste)

```bash
# Repo root (WSL recommended)
git fetch origin
git checkout cursor/w1-config-application-564d   # or merge this PR branch

export PATH="$HOME/.local/bin:$PATH"
unset INFEROPS_EXTERNAL_VLLM
export VLLM_HOST=127.0.0.1
export VLLM_PORT=8000
# export INFEROPS_VLLM_PYTHON=/path/to/vllm-dev/bin/python

pytest -q
pytest -q tests/test_config_application.py
```

## A. Managed restart + PID equality (GPU)

```bash
# Terminal 1 — leave a healthy server with knobs you will override
VLLM_GPU_MEM=0.80 bash scripts/start_vllm.sh 0.5B

# Terminal 2 — record occupant PID, then run ONE managed experiment
ss -ltnp 'sport = :8000'
# note users:(("python",pid=OLD,...))  → OLD_PID=...

python - <<'PY'
from inferops.tools.run_benchmark import RunBenchmarkInput, run_benchmark
out = run_benchmark(RunBenchmarkInput(
    experiment_id="w1_cfg_restart_probe",
    config_patch={"max_num_batched_tokens": 4096},
    workload_name="chat_short",
    persist=True,
    session_id="w1cfg_",
))
print(out)
PY

ss -ltnp 'sport = :8000'
# NEW listener pid must differ from OLD_PID
```

### Assertions (checkboxes)

- [ ] Console shows managed restart / stopping occupant (not “skipping lifecycle”)
- [ ] `status:identity_verified:changed` (or `identity_verified:pid=…`)
- [ ] **PID equality:** after ready, `ss` listener PID == `config_evidence.process_pid` on the persisted row
- [ ] Listener PID ≠ pre-restart `OLD_PID`
- [ ] `actual_config` has CLI keys only (no `scheduler_policy` / `tensor_parallel_size`)
- [ ] `status` is **`insufficient_evidence`** (CLI-only actual; complete-coverage) **or** `invalid` on mismatch — **never `valid` from health alone**
- [ ] `config_evidence.kind=managed_process_start`, `verified=1` / true
- [ ] Not selected as best / `promotable=0` while non-CLI requested knobs remain uncovered

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

Expect: `actual_sched` **NULL**; `ev_pid` equals post-restart listener PID; `promotable` **0** for CLI-only coverage.

### MLflow tags

```bash
# If using file/sqlite tracking URI locally:
python - <<'PY'
import mlflow
from mlflow.tracking import MlflowClient
c = MlflowClient()
runs = c.search_runs(experiment_ids=["0"], order_by=["attribute.start_time DESC"], max_results=5)
for r in runs:
    print(r.info.run_id, r.data.tags.get("run_id"), r.data.tags.get("status"), r.data.tags.get("experiment_id"))
PY
```

`run_id` tag on the MLflow run must match the SQLite `run_id` column for the same attempt.

## B. Stop failure / stale occupant (never valid)

If a prior server cannot be killed (permissions / wrong user):

- [ ] Run fails with `status=failed` (not valid / not insufficient from “health OK”)
- [ ] No new orphan python/`vllm` child left running (`ps` / Task Manager)
- [ ] Persisted row (via `run_benchmark`) has same `run_id` as MLflow tag

## C. External path

```bash
bash scripts/start_vllm.sh 0.5B
export INFEROPS_EXTERNAL_VLLM=1
python - <<'PY'
from inferops.tools.run_benchmark import RunBenchmarkInput, run_benchmark
print(run_benchmark(RunBenchmarkInput(
    experiment_id="w1_cfg_external",
    workload_name="chat_short",
    persist=True,
    session_id="w1cfg_",
)))
PY
unset INFEROPS_EXTERNAL_VLLM
```

- [ ] `status=insufficient_evidence`, `actual_config` null, evidence `external_unverified`
- [ ] Not promotable

## D. Failed-row persistence note

| Entry point | Failed row attached on `BenchmarkError.result`? | Written to SQLite? |
|---|---|---|
| `run_experiment(...)` | Yes (same `run_id`) | **No** (caller must persist) |
| `run_benchmark(..., persist=True)` | Yes | **Yes** (`save_result`) |
| Agent `executor_node` | Yes | Yes (via `run_benchmark`) + summary/trajectory |

GPU checklist items that say “persisted row” assume **`run_benchmark` / agent**, not a bare `run_experiment` call.

## E. CI proof (no GPU)

```text
pytest -q
pytest -q tests/test_config_application.py
```

Targeted cases: healthy-but-different restart, external health-only, identity change, stop failure, listener≠child, stale unknown PID, mismatch→invalid, OOM cleanup + run_id, `run_benchmark` persists failed row.
