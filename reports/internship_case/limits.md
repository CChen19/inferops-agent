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

### The percentage basis is not fully published

The report's `vs_baseline_pct` is computed against the `baseline_primary` passed
in at that step, which the published artifact does not record. Recomputing from
the two rounded rps values in the table does not reproduce `3.77`. The number of
record is `3.77%`; the exact denominator is **unknown** and should not be
back-derived.

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
- The LangGraph checkpointer is an in-process `MemorySaver`, so resume does not
  survive a process restart.
- `external` service mode cannot prove config application, so results there
  fall back to weaker evidence kinds.
- Ledgers stay in each run's own `logs/` directory and are not committed, so
  full re-audit requires access to the original machine.
