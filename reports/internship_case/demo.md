# Demo script (3–5 minutes)

**Interview freeze SHA:** `2731bed` (see [`interview.md`](interview.md)).
Demo artifacts and claims are pinned to that freeze. Do not ad-lib numbers from
later commits.

Goal of the demo: show that this is an inference-tuning agent whose interesting
property is *refusing to claim wins it cannot prove*. Do not try to show a
throughput victory — there isn't a confirmed one, and claiming one is the fastest
way to lose the room.

Total: ~4 minutes. No GPU run during the demo. Everything is pre-recorded
artifacts read from disk. Still **no live GPU** during the interview slot.

## 0. Setup before the call (not counted)

- Checkout or read artifacts at interview SHA `2731bed`.
- Have open: `reports/internship_case/interview.md`,
  `reports/internship_case/case.md`,
  `reports/live_fair_compare/live_fair_compare.md`, and `inferops/decision.py`.
- Do **not** plan a live vLLM launch. Startup can take up to 180 s
  (`STARTUP_TIMEOUT_S`) and a cold CUDA graph build will eat the whole slot.

## 1. The one-sentence framing (~20 s)

"It takes a natural-language serving goal for a local vLLM deployment, spends a
bounded experiment budget on real benchmark runs, and returns one of four
decisions — including 'keep your baseline', which is what it actually returned
on the run I'm about to show you."

## 2. Task confirmation (~40 s)

Show the Task Conditions block in `case.md`. Point at three things:

- The task is confirmed in code before any GPU budget is spent — task id
  `959d5c6ce11e`, budget 10, constraint `ttft_p99_ms <= 250` came from the user.
- The workload is pinned by hash (`5794591456035486`), so runs are comparable.
- The report itself states that offered arrival rate is not supported, so
  `target_qps` would be a measured-throughput goal. Say this out loud; it's the
  kind of limitation an interviewer will otherwise find for you.

## 3. The four trials and the decision (~90 s)

Show the experiment log: baseline at 18.135 rps, then `max_num_seqs=64` at
18.931 rps (report's `vs_baseline_pct = 3.77`), prefix caching at 18.817, and
`max_num_batched_tokens=2048` at 18.647. All four valid.

Then show the decision: `no_reliable_improvement`, keep the baseline, no deploy
recommendation.

Explain why, because this is the actual point of the demo: the improvement
threshold is 5%, so a 3.77% single-shot result counted as non-improving; three
such trials in a row hit the streak limit and stopped the run at 4 of 10 slots;
and because nothing crossed the threshold, the confirmation campaign — three
interleaved baseline/candidate pairs — never ran. `best_summary` is only
advanced by a confirmation decision bound to that exact candidate, so the
baseline stayed best.

Line to use: "It measured something that looked like a 3.8% gain and declined to
recommend it, because it had one measurement and its bar is repeated
interleaved measurement."

## 4. Evidence and the config it refuses to export (~45 s)

Show the adopted configuration section. Two points:

- Exported knobs are the *measured* `actual_config`, parsed back out of the
  launch command line of the vLLM child the agent owns — not the config that was
  requested.
- `scheduler_policy` and `tensor_parallel_size` were requested but show as
  `<missing>`, because the managed launch path never passes them. They are not
  exported as deployable. This is the anti-lying mechanism, shown concretely.

Optionally show one evidence line with its `run_id` and ledger path to make the
point that every number traces to a per-request ledger.

## 5. The fair compare, stated carefully (~45 s)

Open `live_fair_compare.md`. Say it in this order, and do not reorder:

1. "These two arms make different kinds of claim. The planner arm is a
   production decision; the search arm is an observe-after-pick protocol score."
2. "They also ran on different git SHAs — the planner data is ingested from
   `26dd06f`, the search arm ran at `277167e` — so it isn't a same-build A/B."
3. "The planner used 4 slots and decided to keep the baseline at 18.135 rps.
   The search arm paid all 10 slots and observed 19.145 rps at best. That is not
   a deploy win; there's no confirmed gain."
4. "Its first trial was invalid — TTFT p99 of 328 ms against a 250 ms
   constraint — and the harness rejected it even though the engine was healthy."

## 6. Close (~20 s)

"The part I'd defend in review isn't the throughput number, it's the promotion
gate: a result only becomes a recommendation if the config was provably applied
to a process I own, the SLOs hold, and a repeated interleaved campaign confirms
it. On this hardware, that means the honest answer is often 'keep your
baseline'."

## Things not to claim during this demo

- Do not say the agent beat the search baseline, or beat it at adoption.
- Do not describe 19.145 rps as a win, an improvement, or a recommended config.
- Do not present 3.77%/3.8% as a real speedup; it is one unconfirmed
  measurement.
- Do not quote a cross-arm percentage gap. It is `null` in the artifact.
- Do not imply the load generator drives an offered arrival rate or models
  queueing.
- Do not generalize the result beyond one 0.5B model on one 6 GB laptop GPU.
- Do not promise tensor-parallel or scheduler-policy tuning; those knobs are in
  the schema but are never launched.
