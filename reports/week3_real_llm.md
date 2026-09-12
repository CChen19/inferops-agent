# Week-3 ⑦ real-LLM multi-run evidence

This report is **separate** from offline / fake-LLM eval.
Fake-scripted or `--real-graph` output must not be labeled live.

- **layer**: `real_llm`
- **status**: `blocked`
- **passed**: `False`
- **backend**: `openrouter`
- **n_requested**: `3`
- **n_completed**: `0`
- **pass_rate**: `None`
- **llm_boundary**: `None`
- **generated_at**: `2026-09-12T19:05:29.632769+00:00`

## Blocker

real-LLM campaign requires OPENROUTER_API_KEY for backend='openrouter'. No live key in this environment. Refusing to substitute fake/offline LLM.

This is **not** a real-LLM pass. Offline/fake-LLM evidence
lives under `eval_reports/real_graph/` and the unified
offline golden gate.

## Per-run outcomes

| run | status | llm_boundary | stop_reason | error |
|---:|---|---|---|---|
| — | blocked | — | — | no live run |

## Summary

BLOCKED — not a pass. Requested N=3 live runs; completed 0. Pass rate is undefined (None), not 100% and not 0/0.
