# Day 2 XPENG JD Interview Review / 第二天 XPENG JD 面试复习总结

## 1. Day 2 Main Theme / 第二天主线

### Knowledge Points / 知识点

Day 2 的目标不是继续堆新概念，而是把 InferOps 项目包装成 XPENG JD 里想看到的能力：

- AI-native internal tooling
- Agent workflow infrastructure
- Agent reliability
- Evaluation framework
- Observability and debugging
- Prompt workflow visualization
- Electron desktop tooling migration
- MCP-compatible tool exposure

核心定位：

> InferOps is not just a chatbot demo. It is an agentic workflow infrastructure project that makes agent decisions reliable, observable, and evaluable.

中文理解：

> InferOps 不是简单聊天机器人，而是一个让 Agent workflow 变得可靠、可观察、可评估的工程系统。

### Related Questions / 相关题目

- Can you walk me through your agent project?
- How does your project align with this role?
- Why is your project relevant to AI infrastructure?

### Answer / 答案

中文：

InferOps 是一个面向本地 vLLM 推理服务的 autonomous optimization agent。它把自然语言 serving goal 转成闭环实验流程：先跑 baseline，然后通过 Plan -> Execute -> Reflect 循环提出安全配置改动、执行 benchmark、分析瓶颈，并决定继续、停止或重规划。这个项目的重点不是简单调用 LLM，而是构建一个可靠的 agent workflow，包括 structured state、validated tool execution、experiment memory、observability 和 eval harness。

English:

InferOps is an autonomous optimization agent for local vLLM inference. It turns a natural-language serving goal into a closed-loop experiment workflow: running a baseline, proposing safe configuration changes through a Plan -> Execute -> Reflect loop, benchmarking them, analyzing bottlenecks, and deciding whether to continue, stop, or replan. The focus is not just calling an LLM, but building a reliable agent workflow with structured state, validated tool execution, experiment memory, observability, and an evaluation harness.

## 2. JD Keyword Mapping / JD 关键词映射

### Knowledge Points / 知识点

XPENG 这类岗位里，JD 关键词通常不是孤立概念，而是围绕一个主题：

> Build reliable AI workflows and internal tools around LLMs.

关键词解释和项目映射：

| JD Keyword | Meaning | InferOps Mapping |
|---|---|---|
| LLM orchestration | 把 LLM 放进有步骤、有状态、有工具的 workflow | LangGraph Plan -> Execute -> Reflect |
| Multi-agent / workflow systems | 不同组件承担不同角色 | Planner, Executor, Reflector |
| Automated workflow systems | 自动化一串人工操作 | baseline -> benchmark -> bottleneck analysis -> report |
| Evaluation framework | 系统性评估 agent 是否可靠 | ground truth, baselines, composite score |
| LLM-as-judge metrics | 用 LLM 评估语义和推理质量 | trajectory quality judge |
| Structured validation | 对 LLM 输出做 schema 和安全校验 | Pydantic schemas, safe ranges, duplicate detection |
| Observability | 让 agent 的运行过程可追踪 | SQLite, MLflow, OpenTelemetry-style traces |
| Developer productivity | 减少工程师重复工作 | automated tuning, comparison, report generation |
| MCP-compatible connectors | 通过标准协议暴露 tools/resources | wrap InferOps tools as MCP tools |
| Process inspection / dashboards | 可视化 workflow、tool calls、metrics、logs | Chainlit now, Electron dashboard as extension |

### Related Questions / 相关题目

- What does LLM orchestration mean?
- How does your project map to this JD?
- Why is this more than a chatbot demo?

### Answer / 答案

中文：

这个项目和 XPENG JD 的匹配点在于，它不是一个 chatbot demo，而是一个 agentic workflow system。它用 LangGraph 做 LLM orchestration，用 structured state 管理 workflow，用 Pydantic 和规则做 action validation，用 SQLite、MLflow 和 tracing 做 observability，并用 eval harness 评估 outcome、efficiency 和 trajectory quality。这些能力都和 AI infrastructure、internal tooling、workflow inspection 和 agent reliability 相关。

English:

This project maps well to the XPENG role because it is not just a chatbot demo. It is an agentic workflow system. I use LangGraph for LLM orchestration, structured state to manage the workflow, Pydantic and rule-based checks for action validation, SQLite, MLflow, and tracing for observability, and an evaluation harness to measure outcome, efficiency, and trajectory quality. These are closely related to AI infrastructure, internal tooling, workflow inspection, and agent reliability.

## 3. Agent Reliability / Agent 可靠性

### Knowledge Points / 知识点

Agent reliability 指的是 agent 能重复、安全、可解释地做出有用决策，而不是偶尔跑通。

LLM agent 天然有一些风险：

- Hallucinated action: 提出不存在或非法参数
- Weak reasoning: 指标变差仍继续错误方向
- Duplicate exploration: 重复试同一个 config
- Overreacting to noise: 把 benchmark 波动当成真实提升
- Tool failure: benchmark timeout、OOM、metrics parse failed
- Bad stopping decision: 太早停止或无限继续

InferOps 的可靠性来自五层：

1. Structured state: AgentState 保存 goal、baseline、history、bottleneck、budget、best result
2. Validated actions: schema validation、allowlist、safe range、duplicate detection、budget check
3. Deterministic execution: executor 只负责 apply config、run benchmark、parse metrics、store result
4. Rule-based reflection: stop、continue、replan、budget control 由可测试规则控制
5. Evaluation + observability: 长期评估系统质量，单次失败可追踪

核心句：

> The LLM proposes; code validates; tools execute deterministically; rules control the loop; evaluation measures reliability.

### Related Questions / 相关题目

- What makes your agent reliable?
- How do you prevent unsafe or hallucinated actions?
- Why use a rule-based reflector?
- Why not let the LLM decide everything?

### Answer / 答案

中文：

我不会把 LLM 当成 source of truth。LLM 负责提出候选方向，但系统用 structured state、schema validation、safe ranges、duplicate detection 和 budget checks 来限制它。Executor 是 deterministic 的，只执行已经验证过的 action。Reflector 负责预算、停止条件和重规划，这些逻辑应该是 deterministic and testable。可靠性来自 LLM 周围的工程系统，而不是单纯依赖模型本身。

English:

I do not treat the LLM as the source of truth. The LLM proposes candidate directions, but the system constrains it with structured state, schema validation, safe ranges, duplicate detection, and budget checks. The executor is deterministic and only runs validated actions. The reflector controls budget, stopping conditions, and replanning, which should be deterministic and testable. Reliability comes from the engineering system around the LLM, not from relying on the model alone.

## 4. Benchmark Noise and Experiment Design / Benchmark 噪声与实验设计

### Knowledge Points / 知识点

性能实验有噪声。同一个配置跑两次，结果也可能不同。原因包括：

- GPU warm-up 状态不同
- 系统后台进程影响
- request arrival pattern 波动
- KV cache / memory 状态不同
- vLLM dynamic batching 和 scheduler 波动
- logging、I/O、timeout 等干扰

所以不能看到 throughput 提升 2% 就立刻认定优化成功。

实验设计原则：

- Same context: 固定 hardware、model、workload、request rate、benchmark duration
- Warm-up vs measurement: 尽量区分 warm-up 和正式测量
- One parameter at a time: 保证 causal attribution
- Threshold or rerun: 小幅提升不一定有意义，重要 config 可以 rerun
- Multi-objective metrics: throughput 必须和 latency、success rate、VRAM 一起看

关键指标：

- throughput
- TTFT
- E2E latency
- p95 / p99 latency
- GPU utilization
- VRAM usage
- success rate
- error rate

### Related Questions / 相关题目

- How do you handle benchmark noise?
- How do you know the improvement is real?
- Why tune one parameter at a time?
- What metrics matter for vLLM optimization?

### Answer / 答案

中文：

我把 benchmark result 当作实验观测，而不是绝对真相。为了降低噪声影响，我会固定 hardware、model、workload 和 benchmark settings，尽量区分 warm-up 和 measurement，并且一次只改一个参数来保证 causal attribution。对于很小的提升，我不会立刻认为它是真实改进，除非它超过阈值，或者在重复实验中得到支持。

English:

I treat benchmark results as experimental evidence, not absolute truth. To reduce noise, I keep the hardware, model, workload, and benchmark settings fixed, separate warm-up from measurement when possible, and change one parameter at a time for causal attribution. I do not treat small improvements as real unless they exceed a threshold or are supported by repeated evidence.

## 5. Evaluation Framework / 评估框架

### Knowledge Points / 知识点

Agent evaluation 不能只看最后结果，而应该分三层：

1. Outcome quality
   - best throughput
   - latency target satisfaction
   - gap to ground truth
   - workload-level score

2. Efficiency
   - number of experiments
   - wall clock time
   - GPU cost
   - comparison with random / greedy baselines

3. Trajectory quality
   - evidence-based decisions
   - no duplicate configs
   - safe parameter ranges
   - replan when bottleneck changes
   - efficient exploration

还要支持 regression evaluation：

- 每个 commit 生成 JSON / Markdown report
- CI-safe mock benchmark data
- 对比历史结果，发现性能和决策质量回退

### Related Questions / 相关题目

- How do you evaluate agent reliability?
- What does your eval harness measure?
- How do you compare the agent against baselines?
- How do you make evaluation CI-safe?

### Answer / 答案

中文：

我从 outcome、efficiency 和 trajectory quality 三个层面评估 agent reliability。Outcome 衡量最终结果距离 ground truth 多远，以及是否满足 latency 目标。Efficiency 衡量 agent 用了多少实验、多少时间，并和 random 或 greedy baseline 比较。Trajectory quality 衡量决策是否基于证据、是否避免重复、是否在瓶颈变化后重规划。我还会用 CI-safe mock benchmark data 做 regression evaluation，这样不依赖真实 GPU 也能检查 agent 行为是否退化。

English:

I evaluate agent reliability at three levels: outcome, efficiency, and trajectory quality. Outcome measures how close the final result is to ground truth and whether the latency target is satisfied. Efficiency measures how many experiments and how much time the agent used, compared with random or greedy baselines. Trajectory quality checks whether decisions are evidence-based, non-repetitive, and whether the agent replans when bottlenecks change. I also use CI-safe mock benchmark data for regression evaluation, so the agent behavior can be checked without requiring live GPU benchmarks.

## 6. LLM-as-Judge Boundaries / LLM-as-Judge 的边界

### Knowledge Points / 知识点

LLM-as-judge 是用一个 LLM 评估 agent 的输出或 trajectory。但它不能替代客观指标。

不适合用 LLM judge 的内容：

- throughput
- latency
- error rate
- number of experiments
- gap to ground truth

这些应该由代码 deterministic 地计算。

适合用 LLM judge 的内容：

- decision 是否 evidence-based
- planner 是否引用正确指标
- 是否忽略 latency / budget constraint
- 是否和 current bottleneck 对齐
- 是否重复尝试无效 action
- trajectory 是否高效

使用 LLM judge 时要有 rubric，并输出 structured JSON：

```json
{
  "evidence_use": 4,
  "constraint_awareness": 5,
  "action_validity": 5,
  "bottleneck_alignment": 4,
  "efficiency": 3,
  "overall": 4,
  "rationale": "The planner used GPU utilization and latency metrics, but did not discuss benchmark noise."
}
```

核心句：

> LLM judge should evaluate the reasoning around the metrics, not replace the metrics.

### Related Questions / 相关题目

- How does LLM-as-judge work in your project?
- Do you trust LLM-as-judge?
- Why not just use rules?
- What should be evaluated by code versus by LLM judge?

### Answer / 答案

中文：

我不会用 LLM-as-judge 评估 throughput、latency、error rate 这类客观指标，因为这些应该由代码直接计算。我使用 LLM-as-judge 来评估 trajectory quality，例如 planner 的决策是否基于 evidence、是否和 bottleneck 对齐、是否忽略了约束、是否重复。Judge 需要 structured rubric，并输出 JSON score 和 rationale。我把它当作 qualitative signal，而不是唯一真相。

English:

I do not use LLM-as-judge for objective metrics such as throughput, latency, or error rate, because those should be computed directly by code. I use LLM-as-judge for trajectory quality, such as whether the planner's decision is evidence-based, aligned with the bottleneck, aware of constraints, and non-repetitive. The judge should use a structured rubric and return JSON scores with rationale. I treat it as a qualitative signal, not the single source of truth.

## 7. Observability and Debugging / 可观测性与调试

### Knowledge Points / 知识点

Observability 回答：

> What happened?

Evaluation 回答：

> Was it good?

Agent observability 包括：

- workflow trace
- node execution status
- tool call inputs and outputs
- latency
- errors
- metrics
- logs
- intermediate state
- decision rationale
- token usage

Debug agent failure 可以分三层：

1. Workflow trace
   - planner failed?
   - executor failed?
   - reflector failed?
   - tool call failed?
   - schema validation error?

2. Experiment metrics
   - throughput
   - TTFT
   - E2E latency
   - p95 / p99 latency
   - GPU utilization
   - VRAM usage
   - success rate

3. Decision trajectory
   - planner cited what evidence?
   - action valid or unsafe?
   - repeated previous config?
   - bottleneck changed but strategy did not?
   - reflector should stop or replan?

### Related Questions / 相关题目

- What is observability in your project?
- How would you debug an agent failure?
- What is the difference between evaluation and observability?
- What traces or metrics would you collect?

### Answer / 答案

中文：

我会从三层 debug agent failure。第一层是 workflow trace，定位失败发生在 planner、executor、reflector 还是 tool call。第二层是 experiment metrics，看 throughput、TTFT、E2E latency、GPU utilization、VRAM usage 和 success rate。第三层是 decision trajectory，检查 planner 是否基于证据、是否重复、是否忽略约束或 bottleneck 变化。Observability 告诉我发生了什么，evaluation 告诉我这样做好不好。

English:

I would debug an agent failure from three layers. First, I would inspect the workflow trace to locate whether the failure happened in the planner, executor, reflector, or a tool call. Second, I would look at experiment metrics such as throughput, TTFT, E2E latency, GPU utilization, VRAM usage, and success rate. Third, I would inspect the decision trajectory to check whether the planner used evidence, repeated an action, or ignored constraints or bottleneck changes. Observability tells me what happened, while evaluation tells me whether it was good.

## 8. MCP Integration / MCP 集成

### Knowledge Points / 知识点

MCP 是 Model Context Protocol。它不是普通 REST API，而是让 AI clients 发现和调用外部 tools、resources 和 prompts 的标准协议。

不要说：

> I expose an endpoint and users call it.

更准确地说：

> I expose validated tools and resources through an MCP server, so MCP-compatible clients can discover and invoke them.

InferOps 可以暴露成 MCP tools：

- `run_baseline(model, workload, goal)`
- `run_optimization(goal, budget)`
- `run_benchmark(config)`
- `analyze_bottleneck(run_id)`
- `get_report(run_id)`
- `compare_runs(run_id_a, run_id_b)`

也可以暴露 resources：

- `inferops://runs/{run_id}`
- `inferops://reports/{run_id}`
- `inferops://traces/{run_id}`
- `inferops://metrics/{run_id}`

本地运行方式：

- stdio: MCP client 启动 server 子进程，通过 stdin/stdout 通信
- localhost HTTP: server 监听本地端口，适合多个 client 或桌面 app 连接

安全注意：

- bind to localhost
- origin validation
- authentication
- tool-level permission
- distinguish read-only tools from state-changing tools

### Related Questions / 相关题目

- What is MCP?
- How would you expose your agent through MCP?
- Is MCP just an HTTP endpoint?
- What tools or resources would you expose?

### Answer / 答案

中文：

如果把 InferOps 通过 MCP 提供给其他 AI client，我不会把整个 agent 暴露成一个模糊接口，而是把核心能力包装成 validated MCP tools，例如 run_baseline、run_optimization、get_report 和 compare_runs。实验记录、报告、trace 和 metrics 可以作为 MCP resources 暴露。对于本地桌面工具，可以通过 stdio 启动 MCP server；如果多个 client 需要连接，可以使用 localhost HTTP endpoint。同时要考虑 authentication、origin validation 和 tool-level permission。

English:

If I expose InferOps through MCP, I would not expose the whole agent as an unstructured interface. Instead, I would wrap its core capabilities as validated MCP tools, such as `run_baseline`, `run_optimization`, `get_report`, and `compare_runs`. Experiment records, reports, traces, and metrics can be exposed as MCP resources. For a local desktop tool, the MCP server could run over stdio; if multiple clients need to connect, it could expose a localhost HTTP endpoint. I would also consider authentication, origin validation, and tool-level permissions.

## 9. Chainlit vs Electron / Chainlit 和 Electron 的区别

### Knowledge Points / 知识点

Chainlit 和 Electron 不是同一类东西。

Chainlit:

- Python-native LLM / agent UI framework
- 适合快速做 chat-style prototype
- 可以展示 agent 输出、中间步骤、tool calls
- 更像 Streamlit / Gradio 在 LLM agent 场景下的工具

Electron:

- Desktop application framework
- Chromium + Node.js + native desktop shell
- 可以用 TypeScript / React / Vue 写 renderer UI
- 适合 production-grade desktop dashboard 和 local tooling

Electron 三层：

- Main process: process lifecycle、filesystem access、benchmark runtime、telemetry export
- Preload script: narrow typed API bridge
- Renderer process: workflow graph、dashboard、trace timeline、logs、report viewer

Electron 安全关键词：

- `contextIsolation: true`
- `nodeIntegration: false`
- typed IPC channel
- sandboxed renderer
- privileged operations stay in main process

### Related Questions / 相关题目

- What is the difference between Chainlit and Electron?
- Why did you use Chainlit?
- How would you migrate to Electron?
- What Electron security boundaries would you use?

### Answer / 答案

中文：

Chainlit 是一个 Python-native 的 LLM / agent UI 框架，适合快速验证 agent workflow 和聊天式交互。Electron 是桌面应用框架，更适合做生产级本地工具、dashboard 和 workflow inspection UI。我的项目使用 Chainlit 是因为重点在 agent workflow、evaluation 和 observability，而不是桌面 UI。如果迁移到 Electron，我会保留 Python agent runtime，把 workflow graph、trace timeline、metrics dashboard 和 report viewer 放到 TypeScript renderer 里。Main process 负责本地进程、文件系统、benchmark lifecycle 和 telemetry export，并通过 preload 暴露 narrow typed IPC API。

English:

Chainlit is a Python-native UI framework for quickly validating LLM and agent workflows through a chat-style interface. Electron is a desktop application framework that is better suited for production-grade local tools, dashboards, and workflow inspection UIs. I used Chainlit because the main focus was the agent workflow, evaluation, and observability, not desktop UI. If I migrated to Electron, I would keep the Python agent runtime and build workflow graphs, trace timelines, metrics dashboards, and report viewers in a TypeScript renderer. The main process would handle local processes, filesystem access, benchmark lifecycle, and telemetry export, exposing only a narrow typed IPC API through preload.

## 10. Prompt Workflow Visualization and Evaluation / Prompt Workflow 可视化与评估

### Knowledge Points / 知识点

Prompt workflow visualization 和 prompt workflow evaluation 不是同一件事。

Visualization:

> Shows what step the workflow is in and what happened.

Evaluation:

> Judges whether the steps and trajectory were good.

Workflow instrumentation 需要在每个 node 记录 structured events 或 spans：

```json
{
  "run_id": "run_123",
  "step_id": "step_004",
  "node": "planner",
  "input": "...",
  "output": "...",
  "latency_ms": 1200,
  "status": "success",
  "error": null,
  "timestamp": "..."
}
```

Visualization UI 可以展示：

- workflow graph
- step detail
- prompt input/output
- tool calls
- metrics
- logs
- trace timeline
- report viewer

Evaluation harness 可以评估：

- schema validity
- action safety
- duplicate detection
- outcome quality
- experiment efficiency
- reasoning quality
- bottleneck alignment
- regression against previous runs

### Related Questions / 相关题目

- What is prompt workflow evaluation?
- How would you build a workflow visualization dashboard?
- Is it just marking each step in code?
- How would you design an Electron-based process inspection tool?

### Answer / 答案

中文：

Prompt workflow evaluation 不只是把代码里的每一步标记出来。标记每个 node 并记录 input、output、tool call、latency、error 和 state 是 instrumentation 和 visualization，它回答发生了什么。真正的 evaluation 还要判断这些步骤是否可靠，例如 action 是否符合 schema、是否安全、是否重复、最终结果是否比 baseline 好、trajectory 是否基于 evidence。我的设计会把每次 agent run 记录成 structured events 和 OpenTelemetry spans，让 UI 能可视化 workflow，同时让 eval harness 对 outcome、efficiency 和 reasoning quality 打分。

English:

Prompt workflow evaluation is not just marking each step in the code. Marking each node and recording inputs, outputs, tool calls, latency, errors, and state is instrumentation and visualization; it tells us what happened. Real evaluation also judges whether those steps were reliable, such as whether the action followed the schema, whether it was safe, whether it was a duplicate, whether the final result improved over the baseline, and whether the trajectory was evidence-based. I would record each agent run as structured events and OpenTelemetry spans, so the UI can visualize the workflow while the evaluation harness scores outcome, efficiency, and reasoning quality.

## 11. Production Guardrails / 生产环境保护机制

### Knowledge Points / 知识点

生产环境不能让 agent 无限制地直接修改关键系统。

生产化需要 guarded automation：

- approval workflow
- canary testing
- rollback
- resource quota
- max experiment budget
- tool-level permission
- audit logs
- trace redaction
- secret handling
- prompt/tool/config/workload versioning
- cost control

核心观点：

> Productionizing an agent is about controlling blast radius.

也就是一次错误决策不能影响太大。

### Related Questions / 相关题目

- Why not fully automate production tuning?
- What guardrails would you add before production?
- How would you handle a bad agent recommendation?
- How do you control risk in an agent system?

### Answer / 答案

中文：

我不会让 agent 在生产环境中无限制地直接修改配置。生产环境的错误配置可能导致 latency 暴涨、GPU OOM、error rate 上升或成本增加。所以我会采用 guarded automation：agent 可以推荐改动、运行离线或 canary 实验、生成报告，但影响生产的操作需要 approval workflow、canary rollout、rollback、resource quota、tool-level permission 和 audit logs。同时我会做 trace redaction，并 version prompts、tools、configs、workloads 和 eval rubrics，保证可追踪和可复现。

English:

I would not let the agent directly modify production configurations without control. A bad production configuration can increase latency, cause GPU OOM, raise error rates, or increase cost. I would use guarded automation: the agent can recommend changes, run offline or canary experiments, and generate reports, but production-impacting actions should require approval workflows, canary rollout, rollback, resource quotas, tool-level permissions, and audit logs. I would also add trace redaction and version prompts, tools, configs, workloads, and evaluation rubrics to make runs traceable and reproducible.

## 12. Trade-Off Questions / 项目取舍问题

### Knowledge Points / 知识点

这些问题很容易被追问，回答要体现工程判断。

#### Why rule-based reflector?

中文：

因为预算控制、停止条件、重复检测和重规划条件应该 deterministic and testable。

English:

Because budget control, stopping conditions, duplicate detection, and replanning rules should be deterministic and testable.

#### Why one parameter at a time?

中文：

为了因果归因。一次改多个参数可能更快，但很难解释到底哪个参数造成了结果变化。

English:

For causal attribution. Changing multiple parameters at once may be faster, but it makes it hard to know which parameter caused the result change.

#### Why RAG?

中文：

因为 serving optimization 知识更新快。RAG 可以更新 planner 的知识来源，而不需要重写 orchestration loop。

English:

Because serving optimization knowledge changes quickly. RAG lets the planner use updated knowledge without rewriting the orchestration loop.

#### What are the project limitations?

中文：

当前 UI 是 Chainlit，不是 Electron；search space 是为了本地 GPU 实验而有意收窄；生产版本还需要 approval、canary、rollback、permission、audit log 和更完整的 trace export。

English:

The current UI is Chainlit rather than Electron, and the search space is intentionally constrained for local GPU experiments. A production version would need approval workflows, canary rollout, rollback, permissions, audit logs, and richer trace export.

#### What would you improve next?

中文：

我会增加 MCP-compatible tool wrapper、Electron dashboard、更完整的 OpenTelemetry trace export、online eval regression dashboard、confidence-aware planning，以及 multi-objective optimization。

English:

I would add an MCP-compatible tool wrapper, an Electron dashboard, richer OpenTelemetry trace export, an online evaluation regression dashboard, confidence-aware planning, and multi-objective optimization.

### Related Questions / 相关题目

- Why did you make this design choice?
- What would you improve next?
- What are the limitations of your project?
- How would you productionize it?

### Answer / 答案

中文：

这个项目的主要取舍是优先保证可靠性和可解释性，而不是追求最快搜索。比如我选择一次只改一个参数，是为了 causal attribution；选择 rule-based reflector，是为了让预算、停止和重规划逻辑可测试；使用 RAG，是为了让 planner 能吸收更新的 serving 知识。项目限制也很明确：当前 UI 是 Chainlit，不是 Electron；搜索空间较小；生产化还需要 approval、canary、rollback、permissions 和 audit logs。

English:

The main trade-off in this project is prioritizing reliability and interpretability over the fastest possible search. For example, I tune one parameter at a time for causal attribution, use a rule-based reflector so budget, stopping, and replanning logic are testable, and use RAG so the planner can absorb updated serving knowledge. The limitations are also clear: the current UI is Chainlit rather than Electron, the search space is relatively small, and productionization would require approval workflows, canary rollout, rollback, permissions, and audit logs.

## 13. Final Memorization Block / 最终背诵材料

### Core Positioning / 核心定位

中文：

我的项目核心不是做一个聊天机器人，而是让 agent workflow 变得可靠、可观察、可评估。LLM 负责提出方向，代码负责验证边界，工具负责确定性执行，observability 负责解释失败，evaluation 负责衡量长期质量。

English:

The core of my project is not building a chatbot, but making agent workflows reliable, observable, and evaluable. The LLM proposes directions, code validates boundaries, tools execute deterministically, observability explains failures, and evaluation measures long-term quality.

### JD Alignment / JD 对齐

中文：

InferOps 和 XPENG JD 的匹配点在于，它结合了 LLM orchestration、automated workflow execution、observability 和 evaluation。真正的挑战不是调用 LLM，而是构建一个有 structured state、validated actions、experiment feedback、tracing 和 regression-style evaluation 的可靠 agentic system。

English:

InferOps aligns with the XPENG role because it combines LLM orchestration, automated workflow execution, observability, and evaluation. The real challenge is not calling an LLM, but building a reliable agentic system with structured state, validated actions, experiment feedback, tracing, and regression-style evaluation.

### Electron Positioning / Electron 定位

中文：

我的项目没有直接使用 Electron，而是用 Chainlit 快速验证 agent workflow。如果迁移到 Electron，我会把 Python agent runtime 保留为 backend 或 local subprocess，main process 负责本地进程、文件系统、benchmark lifecycle 和 telemetry export，renderer 负责 workflow graph、metrics dashboard、trace timeline 和 report viewer，并通过 preload 暴露 narrow typed IPC API。

English:

My project does not directly use Electron. I used Chainlit to quickly validate the agent workflow. If I migrated it to Electron, I would keep the Python agent runtime as a backend or local subprocess. The main process would handle local processes, filesystem access, benchmark lifecycle, and telemetry export, while the renderer would show workflow graphs, metrics dashboards, trace timelines, and report viewers through a narrow typed IPC API exposed by preload.

