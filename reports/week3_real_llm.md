# Week-3 ⑦ real-LLM multi-run evidence

This report is **separate** from offline / fake-LLM eval.
Fake-scripted or `--real-graph` output must not be labeled live.

`pass_rate` is **accepted / n_requested** from a fail-closed
per-run judge (`judge_live_run`): live boundary, ordered
planner→executor→reflector, first-class stop
(`budget_exhausted` / `no_reliable_improvement`), and non-zero
quality. Empty rows, missing scores, or `eval_empty_plan` do
not count. A call that merely did not throw is not a pass.

- **layer**: `real_llm`
- **status**: `blocked`
- **passed**: `False`
- **backend**: `openrouter`
- **n_requested**: `3`
- **n_completed**: `0`
- **n_accepted**: `0`
- **pass_rate**: `None` (accepted / n_requested; never 'call didn't throw')
- **llm_boundary**: `None`
- **generated_at**: `2026-09-12T19:14:38.647000+00:00`

## Blocker

real-LLM campaign requires OPENROUTER_API_KEY for backend='openrouter'. No live key in this environment. Refusing to substitute fake/offline LLM.

This is **not** a real-LLM pass. Offline/fake-LLM evidence
lives under `eval_reports/real_graph/` and the unified
offline golden gate.

## Per-run outcomes

| run | status | accepted | llm_boundary | stop_reason | error |
|---:|---|---|---|---|---|
| — | blocked | no | — | — | no live run |

## Summary

BLOCKED — not a pass. Requested N=3 live runs; completed 0. Pass rate is undefined (None), not 100% and not 0/0.
