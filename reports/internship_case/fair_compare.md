# Fair compare: planner arm vs online_local_search arm

Source: `reports/live_fair_compare/live_fair_compare.{md,json}` in worktree
`feat-live-fair-compare`, generated `2026-09-14T06:59:11Z`.

## Read this first: the two arms do not make the same kind of claim

| | planner arm | online_local_search arm |
|---|---|---|
| claim_level | `production_decision_report` | `observe_after_pick_protocol_score` |
| What it means | a production decision, i.e. adopt or keep baseline | a protocol score under `observe_after_pick`, i.e. best value observed after the fact |
| LLM boundary | `live_openrouter` | `none` |
| Tool boundary | `managed_local_vllm` | `managed_local_vllm` |
| Source | ingested existing report | live managed benchmark |

The artifact carries this disclaimer verbatim: *"Search-path scores are
observe-after-pick protocol results, not deploy recommendations. No
confirmed_gain is claimed without Week-2 confirmation."*

**The two arms are therefore not comparable at adoption.** A protocol score is
the best number a sweep happened to observe; a production decision is what the
system is willing to hand to someone to deploy. Turning one into the other
requires confirmation, which was not run for the search arm.

## The arms ran on different git SHAs

- The planner arm was **ingested** from
  `feat-live-3060-case/reports/live_3060_case_v2/meta.json` at
  `source_sha = 26dd06fc5c8e084e1481e1dc0744a08991ad0229` (`26dd06f`).
- The search arm **ran live** at
  `commit_sha = 277167efa964b895f141dcd684b1709ea21136d9` (`277167e`).

These are not the same git SHA. The comparison is not a same-build A/B.

Shared conditions that *were* held equal: model
`Qwen/Qwen2.5-0.5B-Instruct`, workload `chat_short`, budget 10 per arm, SLO
`ttft_p99_ms <= 250.0` and `error_rate <= 0.05`, and the same search space
(`max_num_batched_tokens` ∈ {2048, 3072, 4096}, `max_num_seqs` ∈ {64, 128, 256},
`enable_chunked_prefill` ∈ {false, true}, `enable_prefix_caching` ∈
{false, true}).

## Budget actually spent

| Arm | Budget | Used | Best ID | Best rps | Best TTFT p99 | Decision |
|---|---:|---:|---|---:|---:|---|
| planner | 10 | **4** | `live3060v2_baseline` | 18.135 | 93.800 | `no_reliable_improvement` |
| online_local_search | 10 | **10** | `livefair_search_03` | **19.145** | 54.235 | none (`decision_kind: null`) |

The planner arm used 4 slots and its decision was to **keep the baseline at
18.135 rps** — no candidate was adopted. The search arm used all 10 slots and
its best *observed* result was **19.145 rps on `livefair_search_03`**
(run_id `ddd631375bfa4c668cc29634305df487`, config `max_num_batched_tokens=2048`,
`max_num_seqs=64`, `enable_chunked_prefill=false`, `enable_prefix_caching=true`).

**That 19.145 rps is not a deploy win.** It has no `confirmed_gain`: the
artifact records `confirmed_gain: null` and `decision_kind: null` for the search
arm. It was never repeated, never interleaved against a baseline, and never
passed through confirmation.

## The first search trial was invalid

`livefair_search_01` (`max_num_batched_tokens=2048`, `max_num_seqs=128`,
chunked prefill off, prefix caching off) measured 17.435 rps with
**`ttft_p99_ms = 327.985`, which breaks the 250 ms SLO** — so `slo_ok: false`
and `validity: invalid`, even though `engine_validity` was `valid` and
`error_rate` was 0.0. The engine was healthy and the config was genuinely
applied; the result simply violated the user's latency constraint and was
rejected on those grounds.

The arm's protocol score reflects that: `first_valid_n = 2`,
`valid_result_in_budget = true`, `wasted_trials = 7`, `n_paid = 10`,
`gap_pct = null`.

## All search observations

| # | Experiment ID | validity | rps | TTFT p99 |
|---:|---|---|---:|---:|
| 1 | `livefair_search_01` | invalid | 17.435 | 327.985 |
| 2 | `livefair_search_02` | valid | 18.873 | 61.943 |
| 3 | `livefair_search_03` | valid | 19.145 | 54.235 |
| 4 | `livefair_search_04` | valid | 19.109 | 54.637 |
| 5 | `livefair_search_05` | valid | 18.698 | 61.258 |
| 6 | `livefair_search_06` | valid | 18.804 | 62.701 |
| 7 | `livefair_search_07` | valid | 19.004 | 76.115 |
| 8 | `livefair_search_08` | valid | 19.094 | 59.181 |
| 9 | `livefair_search_09` | valid | 18.909 | 56.426 |
| 10 | `livefair_search_10` | valid | 18.919 | 48.081 |

## What may and may not be said about this compare

May be said:

- The planner arm reached a production decision (`keep baseline`, 18.135 rps)
  after 4 of 10 slots, and refused to adopt candidates it could not confirm.
- The search arm, paying all 10 slots, produced an observe-after-pick protocol
  score of 19.145 rps that is not a deploy recommendation.
- The search arm's first trial violated the TTFT SLO and was correctly marked
  invalid.
- The two arms ran on different git SHAs, so this is not a same-build A/B.

May **not** be said:

- Not "the agent beat search at adoption." The arms make different kinds of
  claim, so there is no adoption-level comparison to win.
- Not "search found +5–6% throughput." The artifact deliberately leaves
  `gap_pct` as `null`; cross-arm percentage gaps are unknown and must not be
  computed from these two numbers.
- Not "19.145 rps is the better config." It is the best observed value, not a
  confirmed gain.
- Not "the planner was more efficient because it used 4 slots instead of 10."
  It stopped early on a non-improving streak; a shorter run is not by itself
  evidence of better search.
