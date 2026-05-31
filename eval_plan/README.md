# 评测方案：LongBench (QMSum) & NarrativeQA & RepoQA

## 1. 环境概况

| 项目 | 规格 |
|------|------|
| GPU | 1× NVIDIA A800-SXM4-80GB |
| CPU RAM | ~1TB (可用 ~954GB) |
| 模型 | Qwen2.5-7B-Instruct |
| 框架 | SGLang (自定义 compressed_fa3 backend) |

## 2. 评测目标

对比两种配置下的模型生成效果：
- **Baseline（无压缩）**：使用 `fa3` backend，无 KV 压缩
- **压缩配置**：使用 `compressed_fa3` backend，`kv-compression-ratio=0.5`

评测指标：
- 生成质量（ROUGE-L / F1 / Pass Rate）
- 端到端延迟（每条样本的 wall-clock time）

## 3. 数据集与样本量设计

| 数据集 | 任务类型 | 上下文长度范围 | 采样数量 | 理由 |
|--------|----------|----------------|----------|------|
| LongBench-QMSum | 会议摘要 | 10k-30k tokens | 50 条 | QMSum 全集约 200 条，50 条兼顾统计意义和时间成本 |
| NarrativeQA | 长文档问答 | 30k-100k tokens | 50 条 | 长上下文代表性任务，50 条可覆盖不同长度分布 |
| RepoQA | 代码仓库理解 | 16k-100k tokens | 50 条 | 代码场景，50 条覆盖不同仓库规模 |

**总计 150 条**，预计每条 baseline 耗时 30s-3min（取决于上下文长度），压缩配置类似。
单配置总耗时约 2-4 小时，两配置合计 4-8 小时。

## 4. CPU Prefix Cache OOM 防护方案

### 4.1 问题分析

当前 `PrefixCPUCache` 的 `max_total_bytes` 默认为 `None`（无限制）。对于长上下文评测：
- Qwen2.5-7B：28 layers × 4 KV heads × 128 head_dim × bf16
- 每 token 的 KV 存储 = 28 × 2(K+V) × 4 × 128 × 2 bytes ≈ 57 KB
- 加上 Q 存储：28 × 4 × 128 × 2 ≈ 28.7 KB/token
- **总计约 86 KB/token**
- 100k token 的 entry ≈ **8.6 GB**

如果缓存 50 条不同前缀（最坏情况），需要 50 × 8.6GB = 430GB。虽然系统有 954GB 可用，但：
1. 评测过程中还有其他内存开销（Python、数据加载等）
2. 应留有安全余量

### 4.2 解决方案

**方案：设置 `max_total_bytes` 上限 + 降低 `max_entries`**

在启动 server 时，通过修改 backend 初始化代码或环境变量控制：

```python
# 在 compressed_flashattention_backend.py 中创建 PrefixCPUCache 时设置：
PrefixCPUCache(max_entries=8, max_total_bytes=200 * 1024**3)  # 最多 200GB
```

**具体实施**：由于评测是顺序执行（同一数据集内前缀相同），实际上：
- 同一数据集的样本共享相同前缀 → 只需缓存 1 个 entry
- 切换数据集时旧 entry 会被 LRU 淘汰
- 设置 `max_entries=8, max_total_bytes=200GB` 足够安全

**补充措施**：在评测脚本中加入内存监控，当可用内存低于阈值时主动触发 eviction。

### 4.3 代码修改

修改 `compressed_flashattention_backend.py` 中 PrefixCPUCache 的创建逻辑，增加环境变量控制：

```python
_max_cpu_cache_bytes = int(os.environ.get(
    "CPU_PREFIX_CACHE_MAX_BYTES",
    str(200 * 1024**3)  # 默认 200GB
))
_max_cpu_cache_entries = int(os.environ.get(
    "CPU_PREFIX_CACHE_MAX_ENTRIES",
    "8"
))
self.cpu_prefix_cache = PrefixCPUCache(
    max_entries=_max_cpu_cache_entries,
    max_total_bytes=_max_cpu_cache_bytes
)
```

## 5. 服务启动配置

### 5.1 Baseline（无压缩）

```bash
# script_baseline.sh
_USE_PINNED_MEMORY=1 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
CPU_PREFIX_CACHE_MAX_BYTES=$((200 * 1024 * 1024 * 1024)) \
CPU_PREFIX_CACHE_MAX_ENTRIES=8 \
python -m sglang.launch_server \
    --model-path ../../../autodl-tmp/Qwen2.5-7B-Instruct/ \
    --port 30000 \
    --tp 1 \
    --disable-cuda-graph \
    --mem-fraction-static 0.85 \
    --disable-radix-cache \
    --disable-context-len-check \
    --page-size 1 \
    --chunked-prefill-size -1 \
    --max-prefill-tokens 131072 \
    --context-length 131072 \
    --attention-backend fa3 \
    --kv-compression-ratio 0.0
```

### 5.2 压缩配置

```bash
# script_compressed.sh
_USE_PINNED_MEMORY=1 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
CPU_PREFIX_CACHE_MAX_BYTES=$((200 * 1024 * 1024 * 1024)) \
CPU_PREFIX_CACHE_MAX_ENTRIES=8 \
python -m sglang.launch_server \
    --model-path ../../../autodl-tmp/Qwen2.5-7B-Instruct/ \
    --port 30000 \
    --tp 1 \
    --disable-cuda-graph \
    --mem-fraction-static 0.55 \
    --disable-radix-cache \
    --disable-context-len-check \
    --page-size 1 \
    --chunked-prefill-size -1 \
    --max-prefill-tokens 131072 \
    --context-length 131072 \
    --attention-backend compressed_fa3 \
    --kv-compression-ratio 0.5
```

**注意**：
- TP 设为 1（当前只有 1 张 GPU）
- Baseline 的 `mem-fraction-static` 可以设高一些（0.85），因为无压缩时 KV 占用更大
- `context-length` 设为 128k，覆盖三个数据集的最大长度需求
- 压缩配置保持 0.55，因为压缩后 KV 占用减半，留更多空间给 prefill

## 6. 评测脚本设计

### 6.1 数据准备

使用 HuggingFace datasets 加载：
- `THUDM/LongBench` → qmsum 子集
- `narrativeqa` (deepmind/narrativeqa)
- `evalplus/repoqa` 或从官方 GitHub 获取

### 6.2 评测流程

```
对每个配置 (baseline / compressed):
    1. 启动 SGLang server
    2. 等待 server ready
    3. 对每个数据集:
        a. 加载数据，随机采样 50 条
        b. 对每条样本:
            - 构造 prompt（system + context + question）
            - 记录 start_time
            - 发送请求到 server（max_new_tokens=5120）
            - 记录 end_time
            - 保存: {sample_id, dataset, latency, output, reference}
    4. 关闭 server
    5. 计算评测指标
```

### 6.3 输出格式

每条记录保存为 JSONL：
```json
{
    "dataset": "qmsum",
    "sample_id": 0,
    "config": "baseline",
    "input_tokens": 15234,
    "output_tokens": 512,
    "latency_seconds": 45.2,
    "output": "模型生成的文本...",
    "reference": "参考答案...",
    "timestamp": "2026-05-14T12:00:00"
}
```

## 7. 评测指标

| 数据集 | 主要指标 | 计算方式 |
|--------|----------|----------|
| QMSum | ROUGE-L | rouge-score 库 |
| NarrativeQA | F1 (token-level) | 标准 token F1 |
| RepoQA | Pass Rate / Exact Match | 字符串匹配 + 语义相似度 |

## 8. 关键参数总结

| 参数 | Baseline | 压缩配置 |
|------|----------|----------|
| attention-backend | fa3 | compressed_fa3 |
| kv-compression-ratio | 0.0 | 0.5 |
| mem-fraction-static | 0.85 | 0.55 |
| max_new_tokens | 5120 | 5120 |
| context-length | 131072 | 131072 |
| CPU cache max bytes | 200GB | 200GB |
| CPU cache max entries | 8 | 8 |

## 9. 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| 单条超长输入 OOM (GPU) | 限制 context-length=131072，超长样本截断 |
| CPU 内存溢出 | max_total_bytes=200GB + max_entries=8 |
| 评测时间过长 | 50 条/数据集，合理平衡 |
| Server 崩溃 | 评测脚本加入重试逻辑和断点续传 |
| 结果不可复现 | 固定随机种子，记录完整配置 |
