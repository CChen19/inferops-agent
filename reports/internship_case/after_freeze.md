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

## PR #64 memory pre-experiment (`cec13ef`)

CPU A/B/C, scripted — **not** live GPU, **not** paid LLM.

- **B** (lasting exact-config filter): enough signal to keep.
- **C**: over-filters transients; do not keep.
- Decision: keep **B**; no RAG upgrade; no GPU rerun.
- Do **not** claim live waste reduction from memory.

## Still open (not merged)

At time of writing — do not treat as landed:

- PR [#67](https://github.com/CChen19/inferops-agent/pull/67) — fingerprint scan of past incompatible history
- PR [#68](https://github.com/CChen19/inferops-agent/pull/68) — fingerprint the confirmed task engine
- PR [#69](https://github.com/CChen19/inferops-agent/pull/69) — pass task engine to planner fingerprint fallback

## Live claims (unchanged)

These stay as in the freeze / fair-compare docs:

| Claim | Status |
|---|---|
| Planner adopted **18.135 rps** | Live, SHA `26dd06f` — keep baseline |
| Search observed **19.145 rps** | Live, SHA `277167e` — **unconfirmed** protocol score |
| `confirmed_gain` | `null` |
| Planner beat search / search beat planner at adoption | **Do not say** — exploratory compare only |
| Confirmation campaign on the live case | **Never ran** — best candidate 3.77% < 5% threshold |
