<div class="inferops-landing">
  <!-- Hero Section -->
  <section class="hero-section">
    <div class="hero-badge">
      <span class="badge-dot"></span>
      <span class="badge-text">vLLM 0.8+ Serving Telemetry &middot; Constrained Optimization</span>
    </div>
    <h1 class="hero-title">
      InferOps <span class="gradient-text">Serving Optimization</span>
    </h1>
    <p class="hero-description">
      Automated, evidence-based serving optimization for local vLLM deployments.
      InferOps explores parameter spaces under concurrency limits, enforces TTFT/TPOT SLOs,
      and recommends keeping the baseline configuration whenever evidence is noisy, inconclusive, or fails the confirmation gate (no_reliable_improvement).
    </p>
  </section>

  <!-- Flow Schematic Diagram Section -->
  <section class="diagram-section">
    <div class="diagram-header">
      <div class="diagram-title-group">
        <span class="diagram-icon">&#9881;</span>
        <span class="diagram-title">Experiment & Decision Pipeline</span>
      </div>
      <div class="diagram-legend">
        <span class="legend-item"><span class="legend-dot dot-cyan"></span> Workflow</span>
        <span class="legend-item"><span class="legend-dot dot-amber"></span> Safety Gate</span>
        <span class="legend-item"><span class="legend-dot dot-emerald"></span> Decision Gate</span>
      </div>
    </div>
    
    <div class="diagram-card">
      <svg class="pipeline-svg" viewBox="0 0 1020 360" fill="none" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <linearGradient id="cyanGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#0ea5e9" stop-opacity="0.25"/>
            <stop offset="100%" stop-color="#0284c7" stop-opacity="0.08"/>
          </linearGradient>
          <linearGradient id="amberGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#f59e0b" stop-opacity="0.28"/>
            <stop offset="100%" stop-color="#d97706" stop-opacity="0.08"/>
          </linearGradient>
          <linearGradient id="emeraldGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#10b981" stop-opacity="0.25"/>
            <stop offset="100%" stop-color="#059669" stop-opacity="0.08"/>
          </linearGradient>
          <linearGradient id="loopGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#6366f1" stop-opacity="0.22"/>
            <stop offset="100%" stop-color="#3b82f6" stop-opacity="0.06"/>
          </linearGradient>
          <marker id="arrowCyan" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
            <path d="M1 1L7 4L1 7" stroke="#38bdf8" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
          </marker>
          <marker id="arrowAmber" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
            <path d="M1 1L7 4L1 7" stroke="#fbbf24" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
          </marker>
          <marker id="arrowLoop" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
            <path d="M1 1L7 4L1 7" stroke="#818cf8" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
          </marker>
        </defs>

        <!-- Step 1: User Input -->
        <rect x="20" y="70" width="165" height="170" rx="12" fill="#0d1424" stroke="#1e293b" stroke-width="1.5"/>
        <rect x="32" y="82" width="28" height="28" rx="6" fill="#0ea5e9" fill-opacity="0.15"/>
        <text x="46" y="101" font-family="'JetBrains Mono', monospace" font-size="14" fill="#38bdf8" text-anchor="middle" font-weight="600">1</text>
        <text x="68" y="100" font-family="'Inter', sans-serif" font-size="13" font-weight="600" fill="#f8fafc">User Goal</text>
        <text x="32" y="128" font-family="'Inter', sans-serif" font-size="11" fill="#94a3b8">Natural language input:</text>
        <rect x="30" y="136" width="145" height="52" rx="6" fill="#070b14" stroke="#1e293b"/>
        <text x="38" y="152" font-family="'JetBrains Mono', monospace" font-size="9.5" fill="#38bdf8">&bull; Model & Hardware</text>
        <text x="38" y="167" font-family="'JetBrains Mono', monospace" font-size="9.5" fill="#38bdf8">&bull; Concurrency target</text>
        <text x="38" y="181" font-family="'JetBrains Mono', monospace" font-size="9.5" fill="#38bdf8">&bull; TTFT/TPOT SLOs</text>
        <text x="32" y="214" font-family="'JetBrains Mono', monospace" font-size="10" fill="#64748b">or resume &lt;task_id&gt;</text>

        <!-- Arrow 1 -> 2 -->
        <path d="M185 155 L220 155" stroke="#38bdf8" stroke-width="1.5" stroke-dasharray="3 3" marker-end="url(#arrowCyan)"/>

        <!-- Step 2: Intent & Task Parser -->
        <rect x="228" y="70" width="165" height="170" rx="12" fill="#0d1424" stroke="#1e293b" stroke-width="1.5"/>
        <rect x="240" y="82" width="28" height="28" rx="6" fill="#0ea5e9" fill-opacity="0.15"/>
        <text x="254" y="101" font-family="'JetBrains Mono', monospace" font-size="14" fill="#38bdf8" text-anchor="middle" font-weight="600">2</text>
        <text x="276" y="100" font-family="'Inter', sans-serif" font-size="13" font-weight="600" fill="#f8fafc">Task Parser</text>
        <text x="240" y="128" font-family="'Inter', sans-serif" font-size="11" fill="#94a3b8">Optimization draft:</text>
        <rect x="238" y="136" width="145" height="52" rx="6" fill="#070b14" stroke="#1e293b"/>
        <text x="246" y="152" font-family="'JetBrains Mono', monospace" font-size="9.5" fill="#cbd5e1">&bull; Search bounds</text>
        <text x="246" y="167" font-family="'JetBrains Mono', monospace" font-size="9.5" fill="#cbd5e1">&bull; Service mode</text>
        <text x="246" y="181" font-family="'JetBrains Mono', monospace" font-size="9.5" fill="#cbd5e1">&bull; Budget limit (N)</text>
        <text x="240" y="214" font-family="'Inter', sans-serif" font-size="10" fill="#64748b">Schema &amp; SLO checks</text>

        <!-- Arrow 2 -> 3 -->
        <path d="M393 155 L428 155" stroke="#fbbf24" stroke-width="1.5" marker-end="url(#arrowAmber)"/>

        <!-- Step 3: Safety Gate (Confirm-Before-GPU) -->
        <rect x="436" y="60" width="165" height="190" rx="12" fill="url(#amberGrad)" stroke="#f59e0b" stroke-width="1.5"/>
        <rect x="448" y="74" width="28" height="28" rx="6" fill="#f59e0b" fill-opacity="0.25"/>
        <text x="462" y="93" font-family="'JetBrains Mono', monospace" font-size="14" fill="#fbbf24" text-anchor="middle" font-weight="600">&#128274;</text>
        <text x="484" y="92" font-family="'Inter', sans-serif" font-size="13" font-weight="700" fill="#fbbf24">Safety Gate</text>
        <text x="448" y="118" font-family="'Inter', sans-serif" font-size="11" font-weight="600" fill="#fde68a">Confirm-Before-GPU</text>
        <rect x="446" y="126" width="145" height="66" rx="6" fill="#141108" stroke="#f59e0b" stroke-opacity="0.4"/>
        <text x="454" y="142" font-family="'Inter', sans-serif" font-size="9.5" fill="#fef3c7">&bull; No unconfirmed GPU</text>
        <text x="454" y="157" font-family="'Inter', sans-serif" font-size="9.5" fill="#fef3c7">&bull; Verified hardware</text>
        <text x="454" y="172" font-family="'Inter', sans-serif" font-size="9.5" fill="#fef3c7">&bull; Human authorization</text>
        <text x="454" y="186" font-family="'JetBrains Mono', monospace" font-size="9" fill="#f59e0b">[Confirm and run]</text>
        <text x="448" y="234" font-family="'Inter', sans-serif" font-size="10" fill="#fcd34d">Zero blind spend</text>

        <!-- Arrow 3 -> 4 -->
        <path d="M601 155 L636 155" stroke="#818cf8" stroke-width="1.5" marker-end="url(#arrowLoop)"/>

        <!-- Step 4: Optimization Loop Container -->
        <rect x="644" y="20" width="186" height="265" rx="14" fill="url(#loopGrad)" stroke="#6366f1" stroke-width="1.5" stroke-dasharray="4 2"/>
        <text x="658" y="44" font-family="'Inter', sans-serif" font-size="12" font-weight="700" fill="#a5b4fc">Optimization Loop</text>
        <text x="762" y="44" font-family="'JetBrains Mono', monospace" font-size="9.5" fill="#818cf8">budget N</text>

        <!-- 4a: Plan -->
        <rect x="654" y="54" width="166" height="58" rx="8" fill="#090d1a" stroke="#312e81"/>
        <text x="664" y="72" font-family="'Inter', sans-serif" font-size="11" font-weight="600" fill="#c7d2fe">1. Plan (Hypotheses)</text>
        <text x="664" y="88" font-family="'JetBrains Mono', monospace" font-size="9" fill="#94a3b8">max_num_batched_tokens</text>
        <text x="664" y="101" font-family="'JetBrains Mono', monospace" font-size="9" fill="#94a3b8">gpu_memory_utilization</text>

        <!-- Arrow Plan -> Exec -->
        <path d="M737 112 L737 122" stroke="#818cf8" stroke-width="1.5" marker-end="url(#arrowLoop)"/>

        <!-- 4b: Execute -->
        <rect x="654" y="126" width="166" height="58" rx="8" fill="#090d1a" stroke="#312e81"/>
        <text x="664" y="144" font-family="'Inter', sans-serif" font-size="11" font-weight="600" fill="#c7d2fe">2. Execute (Benchmark)</text>
        <text x="664" y="160" font-family="'JetBrains Mono', monospace" font-size="9" fill="#38bdf8">&bull; Managed vLLM lifecycle</text>
        <text x="664" y="173" font-family="'JetBrains Mono', monospace" font-size="9" fill="#38bdf8">&bull; Concurrency load run</text>

        <!-- Arrow Exec -> Reflect -->
        <path d="M737 184 L737 194" stroke="#818cf8" stroke-width="1.5" marker-end="url(#arrowLoop)"/>

        <!-- 4c: Reflect -->
        <rect x="654" y="198" width="166" height="58" rx="8" fill="#090d1a" stroke="#312e81"/>
        <text x="664" y="216" font-family="'Inter', sans-serif" font-size="11" font-weight="600" fill="#c7d2fe">3. Reflect (Guardrails)</text>
        <text x="664" y="232" font-family="'JetBrains Mono', monospace" font-size="9" fill="#94a3b8">&bull; TTFT &amp; TPOT SLO verify</text>
        <text x="664" y="245" font-family="'JetBrains Mono', monospace" font-size="9" fill="#94a3b8">&bull; Streak stop / budget</text>

        <!-- Arrow 4 -> 5 -->
        <path d="M830 155 L865 155" stroke="#34d399" stroke-width="1.5" marker-end="url(#arrowCyan)"/>

        <!-- Step 5: Decision Engine -->
        <rect x="873" y="60" width="130" height="190" rx="12" fill="url(#emeraldGrad)" stroke="#10b981" stroke-width="1.5"/>
        <rect x="883" y="74" width="28" height="28" rx="6" fill="#10b981" fill-opacity="0.25"/>
        <text x="897" y="93" font-family="'JetBrains Mono', monospace" font-size="14" fill="#34d399" text-anchor="middle" font-weight="600">&#9878;</text>
        <text x="918" y="92" font-family="'Inter', sans-serif" font-size="13" font-weight="700" fill="#34d399">Decision</text>
        <text x="883" y="118" font-family="'Inter', sans-serif" font-size="11" font-weight="600" fill="#a7f3d0">Evidence Gate</text>
        
        <rect x="881" y="126" width="114" height="42" rx="5" fill="#061a12" stroke="#10b981" stroke-opacity="0.4"/>
        <text x="887" y="141" font-family="'JetBrains Mono', monospace" font-size="9" font-weight="600" fill="#34d399">&#10003; Promote Best</text>
        <text x="887" y="156" font-family="'Inter', sans-serif" font-size="8.5" fill="#a7f3d0">Confirmed gain / SLO ok</text>

        <rect x="881" y="174" width="114" height="42" rx="5" fill="#111827" stroke="#374151"/>
        <text x="887" y="189" font-family="'JetBrains Mono', monospace" font-size="9" font-weight="600" fill="#9ca3af">&#8212; Keep Baseline</text>
        <text x="887" y="204" font-family="'Inter', sans-serif" font-size="8.5" fill="#6b7280">Inconclusive / no gain</text>

        <text x="883" y="238" font-family="'Inter', sans-serif" font-size="9.5" fill="#6ee7b7">Final Report + Citations</text>

        <!-- Bottom Timeline: SQLite State Persistence -->
        <rect x="20" y="302" width="983" height="44" rx="8" fill="#080c18" stroke="#1e293b"/>
        <rect x="30" y="312" width="24" height="24" rx="4" fill="#38bdf8" fill-opacity="0.1"/>
        <text x="42" y="328" font-family="'JetBrains Mono', monospace" font-size="11" fill="#38bdf8" text-anchor="middle">&#128190;</text>
        <text x="64" y="325" font-family="'Inter', sans-serif" font-size="11.5" font-weight="600" fill="#e2e8f0">SQLite Session Checkpoint (<code style="font-family:'JetBrains Mono'; color:#38bdf8; font-size:11px;">inferops_memory.db</code>)</text>
        <text x="64" y="338" font-family="'Inter', sans-serif" font-size="10" fill="#64748b">Session state written after steps &middot; Fully resumable via resume &lt;task_id&gt; &middot; Fail-closed process tracking</text>
      </svg>
    </div>
  </section>

  <!-- Core Operational Guarantees (Pillars Grid) -->
  <section class="pillars-section">
    <div class="pillars-grid">
      <div class="pillar-card card-amber">
        <div class="pillar-icon">&#128737;&#65039;</div>
        <h3 class="pillar-title">Confirm-Before-GPU (Safety Gate)</h3>
        <p class="pillar-desc">
          InferOps requires explicit user approval before allocating any GPU budget or spawning benchmark servers. No silent model swaps, no hidden GPU usage.
        </p>
      </div>

      <div class="pillar-card card-emerald">
        <div class="pillar-icon">&#9878;&#65039;</div>
        <h3 class="pillar-title">Evidence Gate (Keep-Baseline)</h3>
        <p class="pillar-desc">
          Candidate parameters must beat the baseline on the confirmation protocol without breaching TTFT or TPOT SLOs. If evidence is noisy, inconclusive, or fails confirmation (<code>no_reliable_improvement</code>), InferOps keeps the baseline.
        </p>
      </div>

      <div class="pillar-card card-blue">
        <div class="pillar-icon">&#128274;</div>
        <h3 class="pillar-title">Fail-Closed Lifecycle</h3>
        <p class="pillar-desc">
          Tracked child process groups and GPU mutex leases. Cancelling or stopping halts spawned servers immediately; unmanaged or external endpoints are never modified.
        </p>
      </div>

      <div class="pillar-card card-cyan">
        <div class="pillar-icon">&#128190;</div>
        <h3 class="pillar-title">Checkpointed &amp; Resumable</h3>
        <p class="pillar-desc">
          Session state is written to SQLite after steps; interrupted or paused runs can be resumed anytime with <code>resume &lt;task_id&gt;</code>.
        </p>
      </div>
    </div>
  </section>

  <!-- Starters / Scenarios Quick Action Strip -->
  <section class="scenarios-section">
    <div class="scenarios-header">
      <span class="scenarios-title">Example Serving Scenarios</span>
      <span class="scenarios-subtitle">Click a scenario to load it into the composer:</span>
    </div>
    <div class="scenarios-grid">
      <div class="scenario-card" onclick="window.inferopsFillPrompt && window.inferopsFillPrompt('I have Qwen2.5-1.5B on RTX 3060, chat scenario, target QPS=10, TTFT p99 under 200ms')">
        <div class="scenario-badge">Low-Latency Chat</div>
        <div class="scenario-text">"I have Qwen2.5-1.5B on RTX 3060, chat scenario, target QPS=10, TTFT p99 under 200ms"</div>
      </div>
      <div class="scenario-card" onclick="window.inferopsFillPrompt && window.inferopsFillPrompt('Long document QA, concurrency=4, keep TTFT p99 <= 400ms')">
        <div class="scenario-badge">Long-Context QA</div>
        <div class="scenario-text">"Long document QA, concurrency=4, keep TTFT p99 &lt;= 400ms"</div>
      </div>
      <div class="scenario-card" onclick="window.inferopsFillPrompt && window.inferopsFillPrompt('High concurrency short outputs, 32 users, maximize throughput')">
        <div class="scenario-badge">Max Concurrency</div>
        <div class="scenario-text">"High concurrency short outputs, 32 users, maximize throughput"</div>
      </div>
      <div class="scenario-card" onclick="window.inferopsFillPrompt && window.inferopsFillPrompt('resume ')">
        <div class="scenario-badge">Resume Saved Task</div>
        <div class="scenario-text">"resume &lt;task_id&gt;" &mdash; Continue a saved run from SQLite</div>
      </div>
    </div>
  </section>
</div>
