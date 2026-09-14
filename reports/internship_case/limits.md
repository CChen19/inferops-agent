# Limits: what is verified, and what is a known gap

Split into two lists on purpose. "Verified" means there is a real artifact or a
line of code behind it. "Known limits" are things the system does not do; they
are not bugs to be hidden in an interview, they are the scope boundary.

## Verified by real evidence

| Claim | Evidence |
|---|---|
| A real live run ended in `no_reliable_improvement` and kept the baseline | `live_3060_case_v2/meta.json`: `decision_kind: no_reliable_improvement`, `adopt: false`, `keep_baseline: true`, SHA `26dd06f` |
| Four valid trials, budget 10, 4 used | `live_3060_case_v2/case.md` experiment log; `tried_experiment_ids` has 4 entries |
| Baseline throughput 18.135 rps, TTFT p99 93.8 ms, error rate 0 | same report, goal checks and executive summary |
| Planner ran against a live LLM, not a mock | `meta.json`: `planner_model: deepseek/deepseek-chat`, `llm_boundary: live_openrouter` |
| The vLLM server was agent-managed | `meta.json`: `tool_boundary: managed_local_vllm`; service mode `managed` in task conditions |
| Requested-but-unapplied knobs are not exported as deployable | report lists `scheduler_policy: requested fcfs → measured <missing>` and `tensor_parallel_size: requested 1 → measured <missing>` |
| An SLO-violating trial is rejected even when the engine is healthy | `livefair_search_01`: `engine_validity: valid`, `error_rate: 0.0`, `ttft_p99_ms: 327.985`, `slo_ok: false`, `validity: invalid` |
| Search arm's best result carries no confirmed gain | `live_fair_compare.json`: `confirmed_gain: null`, `decision_kind: null`, `claim_level: observe_after_pick_protocol_score` |
| The search arm paid for all 10 slots | `budget_used: 10`, `n_paid: 10`, `wasted_trials: 7`, `first_valid_n: 2` |

## Verified in merged code and CPU tests (not new GPU measurements)

| Claim | Evidence |
|---|---|
| Production task resume survives a process restart | PR #28: production uses `SqliteSaver`, the SQLite `tasks` table preserves confirmed task/session/thread identity, and the CLI exposes `--resume-task`; `tests/test_task_persistence.py` rebuilds the saver and resumes without a second baseline |
| Eval recovery goldens remain isolated from production persistence | `inferops/eval/recovery_goldens.py` still constructs `MemorySaver` |
| Planner hypotheses require an existing structured metric citation | PR #29: `inferops/citations.py` checks exact `run_id`, allowed metric name, and numeric value against summaries visible to the planner; forged ids or values are rejected in CPU tests |
| Document citations are conditional on retrieval | PR #29: a retrieved source requires both the structured document source and matching rationale tag; with no retrieved sources, a document citation or source tag is rejected; forged sources are rejected |
| Managed GPU work is serialized and unknown occupants are not killed | PR #30: `GPULease` uses a non-blocking file lock; busy and unknown-occupant paths fail closed without starting or stopping another process |
| Cancellation is ownership-scoped | PR #30: the process-local registry stops registered owned children and releases their leases; foreign/unregistered processes are untouched in CPU tests |
| Stop during model load fails closed | PR #32: a managed child is registered as owned immediately at spawn, before readiness polling; Stop makes the in-flight experiment never valid, stops the owned child, and releases its lease in CPU tests |
| Readiness polling aborts after cancellation or Stop | PR #35: `wait_ready_verbose` aborts when `cancel_requested()` is true or `stop()` has cleared `_proc`, rather than polling a dead port until `STARTUP_TIMEOUT_S`; verified by deterministic CPU tests, not a live GPU timing |
| Readiness abort binds the child handle once | PR #37: `_wait_should_abort` binds `proc = self._proc` once so a racing `stop()` cannot AttributeError on `.poll()`; `test_wait_should_abort_binds_proc_once` in `tests/test_vllm_process.py`; CPU tests, not a live GPU timing |
| Identity and ledger leftovers stay untracked | PR #38: `.gitignore` includes `logs/*.json` so identity/ledger JSON under `logs/` stays untracked; repo hygiene, not a GPU measurement |
| Crash/exit helpers bind the child handle once | PR #39: `is_crashed()` and `exit_code()` bind `proc = self._proc` once so a racing `stop()` cannot AttributeError on `.poll()`; `test_is_crashed_binds_proc_once` and `test_exit_code_binds_proc_once` in `tests/test_vllm_process.py`; CPU tests, not a live GPU timing |
| Eval RAG stub must parse via production sources | PR #40: `tests/test_eval_planner_strategy.py` asserts `sources_from_context` on the RAG stub context; tests only, not a GPU measurement |
| Summaries persist the vs-baseline denominator | PR #42: `ExperimentSummary.baseline_primary` stores the denominator used for `vs_baseline_pct`; `summary_from_result` writes it, including `0.0` when the gain is left `None`; CPU tests, not a rewrite of the live case artifact |
| Experiment Log prints stored vs-baseline and TTFT p99 | PR #44: `write_final_report` prints `vs_baseline_pct` at stored precision (`+3.77%`, not `+3.8%`) and includes a `ttft_p99` column; CPU tests, not a new GPU report |
| `pid` and `stop()` bind the child handle once | PR #43: `pid` and `stop()` bind `proc = self._proc` once so a racing `stop()` cannot AttributeError; PR #52 extends `test_stop_binds_proc_once` to assert that the bound `FakeChild` was terminated, covering the existing production `stop()` behavior; CPU tests, not a live GPU timing |
| Production checkpointer closes the SQLite connection | PR #45: `production_checkpointer` is a context manager that closes the `SqliteSaver` connection; `test_production_checkpointer_closes_connection` in `tests/test_agent_graph.py`; `run_eval` session test uses `cwd=tmp_path` so the default db cannot leak into the worktree; eval goldens stay `MemorySaver` / disk-free |
| RAG eval test asserts parsed available sources | PR #46: RAG eval wraps `valid_structured_citations` and asserts `args[2] == {"test_doc"}`; ledger is `[baseline, trial]`; tests only |
| Readiness health GET is bounded by `CANCEL_CHECK_S` | PR #48: `wait_ready_verbose` uses `httpx.get(..., timeout=CANCEL_CHECK_S)` (0.25 s), not 3 s; Cancel/Stop does not interrupt an in-flight GET, `_wait_should_abort()` is checked between polls, and the timeout only caps abort-recognition delay; `health_ok()` remains a separate one-shot with `timeout_s=2.0`; CPU tests, not a live GPU timing |
| SQLite WAL/SHM sidecars stay untracked | PR #51: `.gitignore` includes `inferops_memory.db-wal` and `inferops_memory.db-shm`; repo hygiene, not a runtime or GPU claim |
| Compare fails closed on a zero baseline denominator | PR #49: `compare_experiments._delta_pct` raises `ValueError` if baseline `sa == 0` instead of inventing `0.0%`; executor records compare as `tool_unavailable` and leaves `vs_baseline_pct` as `None`; CPU tests |
| Executive Summary prints stored vs-baseline precision | PR #50: Best observed change uses `_fmt_vs_baseline` (e.g. `+3.77%`), same as the Experiment Log; missing stays `n/a`, never `+0.0%`; CPU tests, not a rewrite of the live case artifact |
| Remaining `vs_baseline_pct` printers preserve stored precision | PR #54: Chainlit live-result and all-experiments displays use tested helpers from `inferops.tools.final_report`; graph run summary, executor completion, and `scripts/run_agent.py` result print `+3.77%` rather than `+3.8%`; missing is `n/a` or `unavailable`, never invented `+0.0%`; CPU tests, not a new GPU report |
| Chat can resume a persisted task without redrafting it | PR #57: Chainlit-independent helpers in `inferops/resume.py` parse `resume <12-hex>`, `resume-task <12-hex>`, and bare 12-hex ids; valid commands call `run_agent(..., resume_task_id=...)`, while a missing id spends no GPU budget; `tests/test_resume.py` does not import `app.py` |
| Reflector actions are distinct in UI text | PR #57: `format_reflector_update` distinguishes `continue`, `remeasure`, `rollback`, and `stop`; CPU formatting tests, not a GPU run |
| Compatible history is scoped and hint-only | PR #58: `query_compatible_history` matches another session with the same model and workload (not GPU SKU) only when `memory_db_path` is present; results carry `claim_level=prior_session_hint` and cannot be this-run metric citations or enter experiment summaries, best selection, or promotion |
| Compatible history cannot bypass confirmation | PR #58: failed, invalid, and OOM parameter pairs count as duplicates, but a prior success does not short-circuit confirmation; CLI `run_agent` passes `db_path`; eval goldens remain `MemorySaver` and disk-free |
| Chainlit streaming can load compatible history | PR #60: `app.py` passes `db_path="inferops_memory.db"` to `prepare_initial_state`, matching the `run_agent` / `save_task` default; `tests/test_app_prepare_db_path.py` AST-parses rather than importing `app.py` and does not create or touch the database, while graph tests cover threading `db_path` into `memory_db_path` |
| Retrieval carries stable document identity fields | PR #59: Chroma query results include `chunk_id`, source, and metadata version `inferops-corpus-1`; rendering keeps `[source: {source}] §{section}` and adds `chunk_id=... version=...` on the next line; no checked-in Chroma index |
| Document citations bind to an exact retrieved chunk | PR #59: with retrieved sources, `valid_structured_citations` requires matching `(chunk_id, source, version)` and rejects forged or missing fields; empty RAG still omits document fields; existence-only, not semantic support |
| Stale-child adoption is not automatic | `INFEROPS_ADOPT_STALE_MANAGED` is opt-in and defaults off; adoption also requires the stale lease record, child PID, dead owner, and recorded argv checks to match |

These rows describe code paths, deterministic CPU tests, and repo hygiene
merged on master at `15d5920`. They are not additional RTX 3060 trials and add
no throughput, latency, or `confirmed_gain` result.

## Known limits

### Only eight knobs are applyable and provable

`MANAGED_CLI_EVIDENCED_KEYS` is exactly `model_name`, `max_num_seqs`,
`max_num_batched_tokens`, `max_model_len`, `gpu_memory_utilization`,
`enforce_eager`, `enable_chunked_prefill`, `enable_prefix_caching`. These are
the only knobs `_build_cmd` passes on the vLLM CLI, so they are the only ones
that can appear in a measured `actual_config`.

`scheduler_policy` and `tensor_parallel_size` exist on `ExperimentConfig` but
are never launched. The case report shows them as `<missing>`. Any claim about
tuning scheduling policy or tensor parallelism would be unsupported.

### Confirmation was never run on the +3.8% candidate

The best candidate in the case run (`live3060v2_max_num_seqs_64`, report
`vs_baseline_pct = 3.77`) sat below `IMPROVEMENT_THRESHOLD_PCT = 5.0`, so it was
counted as non-improving and the repeat/confirmation campaign was never
triggered. Consequences to state plainly:

- Nobody knows whether that ~3.8% is real. It is one measurement.
- The 5% threshold means genuine small gains are structurally unadoptable in
  this configuration. That is a tuning choice, not a validated optimum.
- Confirmation itself needs at least 3 usable interleaved pairs
  (`DEFAULT_MIN_PAIRS`) and a 5% median relative delta
  (`DEFAULT_MIN_REL_DELTA`), so on a 10-slot budget a confirmed adoption costs
  most of the budget.

### The live-case percentage basis is still not fully published

The live case report's `vs_baseline_pct` was computed against the
`baseline_primary` passed in at that step, which `live_3060_case_v2` does not
record. Recomputing from the two rounded rps values in the table does not
reproduce `3.77`. The number of record is `3.77%`; the exact denominator in
that artifact is **unknown** and should not be back-derived.

Merged code now persists `baseline_primary` on `ExperimentSummary` (PR #42).
That does not rewrite the published live artifact. Future reports can cite the
denominator; this case still cannot.

### Hardware scope: one 6 GB laptop GPU

Everything was measured on an RTX 3060 Laptop with 6 GB VRAM under WSL2, with
`Qwen/Qwen2.5-0.5B-Instruct` at `gpu_memory_utilization = 0.8` and
`max_model_len = 2048`. Results do not transfer to datacenter GPUs, larger
models, long-context workloads, or multi-GPU serving. VRAM headroom is small
enough that Windows/WSL display memory alone can change what fits.

### No offered arrival rate

Load is generated with concurrency-limited `asyncio.gather`. The case report
states `Offered arrival rate supported: False` and spells out that `target_qps`
is a measured-throughput goal, not a request-arrival rate. There is therefore no
open-loop load, no queueing, and no saturation curve. TTFT percentiles are
closed-loop numbers at a fixed concurrency (16 in this workload), which is not
the same thing as latency under a given QPS.

### Sequential runs, uncompensated thermal drift

Trials run one after another on a single laptop GPU with no cooldown control, no
clock/temperature logging, and no randomized trial ordering. Later trials may
run hotter and slower than earlier ones. The size of this effect is **unknown** —
it was not measured. This is another reason a single-shot 3–4% delta is not
treated as a result.

### SHA mismatch across the compare arms

In the fair compare, the planner arm was ingested from SHA
`26dd06fc5c8e084e1481e1dc0744a08991ad0229` while the search arm ran live at
`277167efa964b895f141dcd684b1709ea21136d9`. Different builds, so it is not a
same-build A/B, and the artifact leaves the cross-arm `gap_pct` as `null`.
Combined with the different `claim_level` values (production decision vs
observe-after-pick protocol score), no adoption-level winner can be declared.

### Other gaps worth naming before someone else does

- Each trial is a single 60-request, concurrency-16 run; no per-trial repeats
  and no confidence intervals outside the confirmation path.
- Per-trial `ttft_p99_ms` is published only for the baseline in the case run;
  for trials 2–4 it is **unknown** in the artifact.
- Production resume depends on the local SQLite database and its stored
  checkpoint/task rows. Eval recovery goldens intentionally use `MemorySaver`
  and do not establish cross-process persistence.
- `external` service mode cannot prove config application, so results there
  fall back to weaker evidence kinds.
- Ledgers stay in each run's own `logs/` directory and are not committed
  (`.gitignore` also ignores `logs/*.json`, PR #38), so full re-audit requires
  access to the original machine.
- `_fmt_vs_baseline` has separate copies in `planner.py` and
  `final_report.py`. They agree for reachable float/`None` inputs today, but
  could silently diverge later.
- CI tests the Chainlit-facing helpers in `final_report.py`, not the `app.py`
  call sites because Chainlit is absent from the dev environment; a future
  local re-round at an app call site could escape CI.
- `eval/runner.py` and `scripts/run_comparison.py` retain one-decimal formatting
  for `gap_pct` / `vs_default`; those are different metrics, not
  `vs_baseline_pct` regressions.
- Cancel/Stop cannot interrupt an in-flight readiness GET. Abort state is
  checked between polls; `CANCEL_CHECK_S` only caps how long that GET can delay
  recognition of the abort.
- A remote `VLLM_HOST` whose RTT exceeds `CANCEL_CHECK_S` (0.25 s) would never
  succeed `wait_ready`.
- `_run_resumed_task` catches every `ValueError`, so its "No GPU budget was
  spent" message can be false if a resumed run raises after doing GPU work.
  `resume nope` also follows the no-id error path instead of drafting a task.
- Resume uses non-streaming `run_agent`, so reflector updates do not appear
  during a resumed run.
- The PR #60 AST test is a source-level pin of the literal `db_path` keyword,
  not runtime proof that `prepare_initial_state` threads it into
  `memory_db_path`; graph tests cover that threading. CI still does not import
  `app.py` because Chainlit is absent. GPU SKU remains unstored and unmatched.
- `_recover_param_value` can mis-attribute a failed compatible-history row with
  two non-default knobs and no `_<knob>_` token in its id.
- Pre-PR #59 Chroma indexes have no version metadata and therefore fail closed
  for every document-citing hypothesis until rebuilt.
- Corpus body text containing a line-start `[source: x] §y` plus a following
  `chunk_id=... version=...` line can be parsed as an available ref. This is the
  pre-existing source-header exposure extended to the tuple.
