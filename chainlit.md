# InferOps

**Local vLLM Serving Optimization & Experiment-Decision Assistant**

InferOps translates high-level serving requirements into principled, reproducible engine configurations. It explores candidate parameter assignments, benchmarks live workloads under concurrency limits, and evaluates whether proposed changes reliably improve performance before recommending adoption.

---

### Key Operational Guarantees

- **Confirm-Before-GPU:** Every proposed optimization task requires explicit user confirmation before any GPU budget is allocated or benchmark server is started.
- **Fail-Closed Lifecycle:** Managed child processes spawned by InferOps are tracked and cleanly stopped on cancellation. External or unmanaged servers are never touched.
- **Checkpointed & Resumable:** Session states are persisted to SQLite. Interrupted runs can be continued anytime using `resume <task_id>`.
- **Evidence-Based Evaluation:** Candidates are judged on measured throughput and tail latency SLOs (TTFT/TPOT) against the baseline before recommending adoption.

---

### Example Queries

Type a serving scenario into the composer below to begin:

- *"I have Qwen2.5-1.5B on RTX 3060, chat scenario, target QPS=10, TTFT p99 under 200ms"*
- *"Long document QA, concurrency=4, keep TTFT p99 <= 400ms"*
- *"High concurrency short outputs, 32 users, maximize throughput"*

*Note: Target QPS specifies a measured throughput goal under concurrency-limited load. To resume a previous run, use `resume <task_id>`.*
