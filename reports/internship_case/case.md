# Case: one real run where the agent kept the baseline

Everything below is copied from one real local run. No number here was
re-derived, estimated, or filled in by hand. Values the source report does not
publish are written as `unknown`.

## Source of record

| Field | Value |
|---|---|
| Report | `reports/live_3060_case_v2/case.md` (worktree `feat-live-3060-case`) |
| Git SHA | `26dd06fc5c8e084e1481e1dc0744a08991ad0229` (`26dd06f`) |
| Generated at | `2026-09-14T06:19:57Z` |
| Session prefix | `live3060v2_` |
| LLM boundary | `live_openrouter`, planner model `deepseek/deepseek-chat` |
| Tool boundary | `managed_local_vllm` (agent starts and owns the vLLM child) |

## Task as confirmed

- Task id `959d5c6ce11e`, service mode `managed`.
- Model `Qwen/Qwen2.5-0.5B-Instruct` on an RTX 3060 Laptop (6 GB VRAM).
- Workload `chat_short`, requests = 60, concurrency = 16, workload hash
  `5794591456035486`.
- Objective: maximize `throughput_rps`. `target_qps` not set.
- Constraints: `error_rate <= 0.05` (default), `ttft_p99_ms <= 250.0` (set by
  the user).
- Experiment budget: **10**. Time limit: not set.
- Offered arrival rate is **not** supported by the load generator
  (`Offered arrival rate supported: False`). Load is concurrency-limited
  `asyncio.gather`, so `target_qps` would have been a measured-throughput goal,
  not a request-arrival rate.

## What actually ran: 4 trials out of a budget of 10

| # | experiment_id | param | value | rps | vs baseline (report) | validity | run_id | mlflow_run_id |
|---|---|---|---|---:|---:|---|---|---|
| 1 | `live3060v2_baseline` | — | — | 18.135 | +0.0% | valid | `73c79a479a1143fa83627382e57ed4b7` | `4d828a295f8742419003b5cb7b07503f` |
| 2 | `live3060v2_max_num_seqs_64` | `max_num_seqs` | 64 | 18.931 | +3.8% | valid | `440b75d07c9a4b3f92fffef3a1ba7ecb` | `2958d3055fbb4b149336f7e40e634370` |
| 3 | `live3060v2_enable_prefix_caching_True` | `enable_prefix_caching` | True | 18.817 | +3.3% | valid | `f15c64559a244961ae1405797761d88d` | `271217f2e4994f2485dfe87e12171d85` |
| 4 | `live3060v2_max_num_batched_tokens_2048` | `max_num_batched_tokens` | 2048 | 18.647 | +2.5% | valid | `d5945ed4831c426a8954499d3fd5c219` | `b319d515acdc4a6fb4e4b4262c78d7a1` |

All four trials were `valid` — nothing crashed, nothing OOMed, no trial broke an
SLO. `ttft_p99_ms` is published only for the baseline run (93.8 ms); for trials
2–4 the per-trial TTFT is `unknown` in this report.

The report's own `vs_baseline_pct` for trial 2 is `3.77` (rendered as `+3.8%` in
the log table). Recomputing from the two published rps values would give a
different figure, because `vs_baseline_pct` is computed against the
`baseline_primary` value passed in at that step, which this report does not
publish. Treat `3.77%` as the number of record and the exact denominator as
`unknown`.

## Decision

`no_reliable_improvement` — keep the baseline. `adopt = false`,
`keep_baseline = true`, stop reason `no_reliable_improvement`.

Goal checks against the adopted (baseline) run:

| role | metric | op | limit | observed | ok |
|---|---|---|---|---|---|
| constraint | `error_rate` | <= | 0.05 | 0 | yes |
| constraint | `ttft_p99_ms` | <= | 250 | 93.8 | yes |

Adopted configuration is the *measured* `actual_config` of
`live3060v2_baseline`: `max_num_seqs=128`, `max_num_batched_tokens=2048`,
`max_model_len=2048`, `gpu_memory_utilization=0.8`, `enforce_eager=False`,
`enable_prefix_caching=False`, `enable_chunked_prefill=False`.

Two requested knobs were **not** exported as deployable because the managed
launch command never passes them: `scheduler_policy` (requested `fcfs` →
measured `<missing>`) and `tensor_parallel_size` (requested `1` → measured
`<missing>`).

## Why the +3.8% trial was not adopted

This is the interesting part of the case, and it is a deliberate design choice
rather than a bug.

1. The reflect stage treats a trial as "improving" only at or above
   `IMPROVEMENT_THRESHOLD_PCT = 5.0` (`inferops/agent/reflect_constraints.py`).
   At `3.77%`, trial 2 counted toward the non-improving streak.
2. Trials 2, 3 and 4 all landed below that threshold, so the streak reached
   `MAX_STREAK = 3` and the run stopped with `no_reliable_improvement` after 4
   of 10 budgeted experiments.
3. Because no trial crossed the threshold, **the confirmation phase never ran**.
   Confirmation is a repeat/interleave campaign that needs at least
   `DEFAULT_MIN_PAIRS = 3` usable baseline/candidate pairs and a median relative
   delta of at least `DEFAULT_MIN_REL_DELTA = 0.05`; a verdict can only come
   from `verdict_from_ledgers` / `evaluate_campaign`.
4. `best_summary` is only advanced by `maybe_promote_best`, which requires a
   confirmation decision bound to that exact candidate. With no confirmation,
   `best_summary` stayed on the baseline row, and
   `build_decision` therefore reported `no_reliable_improvement`.

The report states this in its own words: *"Keep baseline `live3060v2_baseline`
because no candidate cleared confirmation + Week-1 promotion"*, and for trial 2:
*"not selected as confirmed best (validity=`valid`, vs_baseline=3.77)"*.

So the honest summary is: a single measurement showed roughly +3.8%, the system
refused to call that a win, and it emitted **no deploy recommendation**. A
one-shot 3.8% on a 60-request, concurrency-16 run on a thermally-unconstrained
laptop GPU is inside the range this project is not willing to claim without
repeated interleaved measurement.

## Evidence trail

Each row is auditable back to a per-request ledger (ledgers live in the run's
own `logs/` directory and are intentionally not copied into git):

- adopted / baseline / best: `live3060v2_baseline`,
  run_id `73c79a479a1143fa83627382e57ed4b7`,
  ledger `logs/ledger_73c79a479a1143fa83627382e57ed4b7.json`, status `valid`
- run: `live3060v2_max_num_seqs_64`, run_id `440b75d07c9a4b3f92fffef3a1ba7ecb`
- run: `live3060v2_enable_prefix_caching_True`, run_id `f15c64559a244961ae1405797761d88d`
- run: `live3060v2_max_num_batched_tokens_2048`, run_id `d5945ed4831c426a8954499d3fd5c219`
