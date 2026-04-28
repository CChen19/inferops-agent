# InferOps 基线报告 — chat_short

生成时间：2026-04-28  
模型：Qwen/Qwen2.5-0.5B-Instruct | 硬件：RTX 3060 Laptop 6GB，WSL2，CUDA 12.8  
配置：gpu_memory_utilization=0.80，max_model_len=2048，max_num_seqs=128  
Workload：60 个请求，并发=16，~15 token 输入，128 token 输出

## 实验结果

| 变体 | RPS | Tok/s | TTFT p50 | TTFT p99 | E2E p50 | E2E p99 | GPU 利用率 |
|---|---|---|---|---|---|---|---|
| default | 14.96 | 1929 | 48ms | 69ms | 1015ms | 1032ms | 86% |
| chunked_prefill | 16.68 | 2151 | **67ms** | 79ms | 914ms | 951ms | 84% |
| prefix_caching | 16.84 | 2167 | 49ms | **175ms** | 878ms | 1001ms | 81% |
| **big_batch** | **17.23** | **2222** | 52ms | **65ms** | **883ms** | **892ms** | 86% |

成功率：全部 60/60

## 分析

**big_batch（max_num_batched_tokens=4096）综合最优：**
- 吞吐 +15%（17.23 vs 14.96 rps）
- E2E p50/p99 均最优（883ms / 892ms），尾延迟收束最好
- TTFT 仅多 4ms（52ms vs 48ms），可以接受
- 原因：更大的批次让 GPU 矩阵运算更饱和，decode kernel 效率更高

**chunked_prefill 在短序列上是负优化：**
- TTFT p50 从 48ms 涨到 67ms（+40%）
- chunked prefill 把 prefill 拆成小块和 decode 交替执行，对长序列能减少 TTFT 抖动，但对短序列（~15 tokens）只增加调度开销
- 吞吐小幅提升（+11%）是调度器粒度更细所致

**prefix_caching 在无共享前缀的场景下引入 p99 毛刺：**
- TTFT p99 从 69ms 飙到 175ms（+153%）
- chat_short 的 prompt 各不相同，缓存命中率为 0，哈希计算和缓存查询反而带来额外开销
- 适用场景：有固定系统 prompt 的对话服务，共享前缀越长收益越大

**E2E ≈ 900ms 的来源：**
- decode 128 tokens × ~7ms/token ≈ 896ms（此 workload 是 decode-bound）
- TTFT 只占 E2E 的 5%

## WSL2 硬件注意事项

- nvidia-smi 显示 GPU mem ~5.97 GB 已用，但 vLLM 实际只占约 3.7 GB
  （另外 ~2.2 GB 是 Windows 桌面合成器占用，PyTorch/CUDA 不可见）
- gpu_memory_utilization=0.80 在此环境安全，未触发 OOM
- max_model_len=2048 是关键参数：不设此值时 vLLM 默认支持 Qwen 的 32768 token 上下文，
  KV cache 块分配直接 OOM

## 下一步

- [ ] 跑 long_context_qa（1024 token 输入）——chunked_prefill 预计在长序列场景中反转劣势
- [ ] 确认 big_batch 在 long_context_qa 上是否仍有优势或触发显存压力
- [ ] Phase 2：把这条链路接入 LangGraph agent，实现自动参数搜索
