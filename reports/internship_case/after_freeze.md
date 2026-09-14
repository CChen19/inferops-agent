# After freeze

Short log of work that landed **after** the interview freeze. The demo and interview
story stay pinned to freeze SHA `2731bed` — do not retarget them here.

**Interview / demo pin:** `2731bed` (master through merged PR #61).

## Merged after freeze

| Item | Merge SHA | Notes |
|---|---|---|
| PR [#63](https://github.com/CChen19/inferops-agent/pull/63) freeze docs | `61374af` | Documents the `2731bed` pin |
| PR [#62](https://github.com/CChen19/inferops-agent/pull/62) closeout | `5cb9e20` | Resume honesty, unique-knob history attribution, Chroma `CORPUS_VERSION`, hardware fingerprint for compatible history. **After freeze — not claimed in the interview SHA.** |
| PR [#65](https://github.com/CChen19/inferops-agent/pull/65) note | `287297f` | Notes PR #62 after freeze |
| PR [#64](https://github.com/CChen19/inferops-agent/pull/64) memory pre-experiment | `cec13ef` | See below |
| PR [#66](https://github.com/CChen19/inferops-agent/pull/66) note | `279765e` | Notes PR #64 after freeze |
| PR [#67](https://github.com/CChen19/inferops-agent/pull/67) history fingerprint scan | `8ebba43` | Page past incompatible history fingerprints (not newest-32 only) |
| PR [#68](https://github.com/CChen19/inferops-agent/pull/68) task engine fingerprint | `1e6d53f` | Fingerprint the confirmed task engine (not default-vLLM) |
| PR [#69](https://github.com/CChen19/inferops-agent/pull/69) planner fingerprint fallback | `619e5a1` | Pass task engine into planner fingerprint fallback |
| PR [#70](https://github.com/CChen19/inferops-agent/pull/70) interview SHA note | `a931790` | Pin PR #65 merge SHA in interview after-freeze table |
| PR [#71](https://github.com/CChen19/inferops-agent/pull/71) memory scenario note | `3a35714` | Honest t2048 OOM override note for irrelevant scenario too |
| PR [#72](https://github.com/CChen19/inferops-agent/pull/72) resume token chat | `7033cf7` | Non-id resume tokens stay ordinary chat |
| PR [#73](https://github.com/CChen19/inferops-agent/pull/73) after-freeze notes | `f871141` | Adds this after-freeze interviewer log |
| PR [#74](https://github.com/CChen19/inferops-agent/pull/74) resume missing-id copy | `5f88844` | Reuse `format_resume_failure` for bare resume / missing id |
| PR [#75](https://github.com/CChen19/inferops-agent/pull/75) incomplete hardware | `18ae66f` | Leave omitted experiment hardware unknown (no invented fingerprint) |
| PR [#76](https://github.com/CChen19/inferops-agent/pull/76) honest README | `4202778` | Keep-baseline product story (does not rewrite live numbers) |
| PR [#77](https://github.com/CChen19/inferops-agent/pull/77) README zh polish | `f88ab85` | Polish Chinese README tech wording; live claim boundaries unchanged |
| PR [#78](https://github.com/CChen19/inferops-agent/pull/78) English README homepage | `16754b5` | English README as GitHub homepage; Chinese → `README.zh.md` |
| PR [#79](https://github.com/CChen19/inferops-agent/pull/79) fail-closed missing primary | `50d2939` | Fail closed when primary metric is missing |
| PR [#80](https://github.com/CChen19/inferops-agent/pull/80) closeout hygiene | `f04bb41` | After-freeze notes, gitignore live dumps, `.env.example` |
| PR [#81](https://github.com/CChen19/inferops-agent/pull/81) portable vLLM Python | `a5b66e5` | Fail closed; no machine-local path fallback |
| PR [#82](https://github.com/CChen19/inferops-agent/pull/82) Chainlit theme | `da95b13` | First visual pass; landing-page follow-up is a later PR |
| PR [#83](https://github.com/CChen19/inferops-agent/pull/83) docs closeout | `dcf309b` | READMEs + after-freeze through #82; does not retarget freeze `2731bed` |
| PR [#84](https://github.com/CChen19/inferops-agent/pull/84) landing-page UI | `0af882e` | Developer-tool landing page and wide layout; honesty copy (no statistical-significance claim) |
| PR [#85](https://github.com/CChen19/inferops-agent/pull/85) stop synthetic CI | `1a14a6d` | No gauss/jitter synthesized bootstrap CI. Throughput CI is unavailable (single aggregates; cannot be read as tie / “not significant”). Latency CI only with real raw samples. No repeated-run CI. Also hardens `input_len` / `output_len` bounds. |
| PR [#86](https://github.com/CChen19/inferops-agent/pull/86) lasting config memory filter | `e069f22` | Production no longer blacklists generic `failed`. Cross-session hard filter: lasting identical full search config in a compatible env (model + `workload_hash` + hardware). Usable observation = valid + SLO + config evidence + finite primary. |

Current `origin/master` tip at time of this update: `e069f22` (through PR #86).

## PR #64 memory pre-experiment (`cec13ef`)

CPU A/B/C, scripted — **not** live GPU, **not** paid LLM.

- **B** (lasting exact-config filter): enough signal to keep.
- **C**: over-filters transients; do not keep.
- Decision: keep **B**; no RAG upgrade; no GPU rerun.
- Do **not** claim live waste reduction from memory.

## Live claims (unchanged)

These stay as in the freeze / fair-compare docs:

| Claim | Status |
|---|---|
| Planner adopted **18.135 rps** | Live, SHA `26dd06f` — keep baseline |
| Search observed **19.145 rps** | Live, SHA `277167e` — **unconfirmed** protocol score |
| `confirmed_gain` | `null` |
| Planner beat search / search beat planner at adoption | **Do not say** — exploratory compare only |
| Confirmation campaign on the live case | **Never ran** — best candidate 3.77% < 5% threshold |
