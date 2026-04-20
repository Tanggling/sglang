# SnapKV 深度分析报告：Softmax 优化、CPU Prefix 策略与全量重算对比

## 实验环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA A800-SXM4-80GB |
| 模型参数 | Qwen2.5-7B-Instruct (4 kv_heads, 28 q_heads, 128 head_dim) |
| 数据类型 | BFloat16 |
| Window Size | 64 |
| Warmup / Repeat | 3 / 5 |

---

## 一、CPU Prefix 测试逻辑与 SGLang 实现的对比

### Benchmark 模拟的流程

```
Phase 1: CPU prefix KV → GPU (pageable, blocking .to(device))
         → 写入 GlobalKVPool (pool_k[slots] = k_gpu)
         → extend KV 写入 GlobalKVPool
         → 读回完整 KV (pool_k[slots].clone())
Phase 2: FlashAttention (SDPA 模拟)
Phase 3: SnapKV 压缩 → scatter write 到 real pool
```

### SGLang 实际实现的流程 (`_forward_extend_global_real_split`)

```
Phase 1: cpu_prefix_cache.load_layer_to_gpu() → CPU KV 传到 GPU
         → global_kv_pool.write_kv(prefix slots)
         → global_kv_pool.write_kv(extend slots)
         → global_kv_pool.read_kv(all slots) → k_full, v_full
Phase 2: flash_attn_varlen_func(q, k_all, v_all, ...)
Phase 3: _estimate_importance() → compressor.compress()
         → _write_compressed_to_real() → 写入 real KV pool
         → global_kv_pool.free(slots)
```

### 一致性分析

| 环节 | Benchmark | SGLang 实际 | 一致性 |
|------|-----------|------------|--------|
| CPU→GPU 传输 | `k_cpu.to(device, non_blocking=False)` | `cpu_prefix_cache.load_layer_to_gpu()` 内部也是 `.to(device)` | ✓ 一致（都是 pageable + blocking） |
| GlobalKVPool 写入 | `pool_k[slots] = k_gpu` | `global_kv_pool.write_kv(slots, k, v)` | ≈ 等价（封装差异） |
| GlobalKVPool 读回 | `pool_k[slots].clone()` | `global_kv_pool.read_kv(slots)` | ≈ 等价 |
| FlashAttention | `F.scaled_dot_product_attention` | `flash_attn_varlen_func` | ⚠ 不同（FA2/3 更高效） |
| 压缩算法 | 手写 SnapKV 逻辑 | `SnapKVStyleCompressor.compress()` | ≈ 等价（核心逻辑相同） |
| Q padding | 未模拟 | 当 extend_len < window_size 时从 CPU 加载 Q | ⚠ 缺失 |

**结论**：Benchmark 的 CPU prefix 流程与 SGLang 实现在数据流上一致，主要差异在 FA 实现（benchmark 用 SDPA 模拟，实际用 FlashAttention kernel）和 Q padding 逻辑。对于压缩算法本身的 profiling，结果是可信的。

---

## 二、SnapKV Softmax vs No-Softmax 对比

### 背景

SnapKV 原始论文使用 softmax 归一化注意力分数后再做 importance 估计。但实际上，对于 topk 选择来说，只需要相对大小排序，不需要归一化后的概率值。

**你的 SGLang 实现已经注释掉了 softmax**（`kv_compressor.py` 第 382 行）：
```python
# attention_scores = F.softmax(attn_weights_full, dim=-1, dtype=torch.float32).to(query.dtype)
attention_scores = attn_weights_full  # 直接使用原始 QK 分数
```

### 实验数据

| seq_len | ratio | with softmax (ms) | no softmax (ms) | 节省 (ms) | 节省比例 |
|---------|-------|-------------------|-----------------|----------|---------|
| 4,096 | 0.3 | 0.633 | 0.546 | 0.087 | 14% |
| 4,096 | 0.5 | 0.616 | 0.534 | 0.082 | 13% |
| 4,096 | 0.7 | 0.617 | 0.530 | 0.087 | 14% |
| 8,192 | 0.3 | 1.159 | 0.914 | 0.245 | 21% |
| 8,192 | 0.5 | 0.915 | 0.663 | 0.252 | 28% |
| 8,192 | 0.7 | 0.912 | 0.658 | 0.254 | 28% |
| 16,384 | 0.3 | 1.687 | 1.188 | 0.499 | 30% |
| 16,384 | 0.5 | 1.670 | 1.160 | 0.510 | 31% |
| 16,384 | 0.7 | 1.645 | 1.160 | 0.485 | 29% |
| 32,768 | 0.3 | 2.784 | 1.678 | 1.106 | 40% |
| 32,768 | 0.5 | 2.763 | 1.657 | 1.106 | 40% |
| 32,768 | 0.7 | 2.765 | 1.442 | 1.323 | 48% |
| 65,536 | 0.3 | 4.720 | 2.566 | 2.154 | 46% |
| 65,536 | 0.5 | 4.712 | 2.548 | 2.164 | 46% |
| 65,536 | 0.7 | 4.699 | 2.540 | 2.159 | 46% |

### 子操作对比（65K, ratio=0.5）

| 操作 | with softmax | no softmax | 差异 |
|------|-------------|------------|------|
| qk_matmul | 0.744ms | 0.740ms | 相同 |
| repeat_interleave | 0.977ms | 0.977ms | 相同 |
| **softmax** | **2.157ms** | **0.000ms** | **-2.157ms** |
| pool_sum | 0.185ms | 0.191ms | +0.006ms（bf16 sum 略慢于 fp32） |
| topk | 0.367ms | 0.367ms | 相同 |

### 分析

1. **softmax 节省随序列长度线性增长**：4K 节省 0.09ms → 65K 节省 2.16ms
2. **压缩比对 softmax 节省无影响**：softmax 的开销只取决于 seq_len，与 ratio 无关
3. **去掉 softmax 后的瓶颈转移**：
   - 65K no-softmax: repeat_interleave 占 38%（0.98ms），qk_matmul 占 29%（0.74ms）
   - 下一步优化应聚焦 repeat_interleave（GQA 展开）

### softmax 开销的根因

softmax 需要：
1. 将 bf16 张量 cast 到 fp32（`[28 heads, 64 window, N-64 prefix]`）
2. 计算 exp + sum + normalize
3. Cast 回 bf16

在 65K 时，中间 fp32 张量大小 = 28 × 64 × 65472 × 4B ≈ **460MB**，这个内存分配+计算是 2.16ms 的主要来源。

---

## 三、CPU Prefix Pipeline vs 全量从 CPU 重算

### 两种策略对比

**策略 A: CPU Prefix Pipeline（当前 SGLang 实现）**
```
1. 只传输 prefix 部分的 KV 从 CPU → GPU（prefix_len tokens）
2. extend 部分的 KV 已在 GPU（模型刚计算出来）
3. 组装完整 KV → FlashAttention → 压缩 → 写入 real pool
```

**策略 B: 全量从 CPU 重算**
```
1. 传输全部 KV + Q 从 CPU → GPU（seq_len tokens × 3 tensors）
2. 直接压缩（不需要 FA，因为 KV 已经是之前计算好的）
3. 写入 real pool
```

### 延迟对比

| seq_len | pfx ratio | CPU Prefix (ms) | Full Recompute (ms) | CPU Prefix 快 |
|---------|-----------|-----------------|--------------------|--------------| 
| 4,096 | 50% | 2.04 | 3.48 | 1.7x |
| 4,096 | 80% | 1.81 | 4.10 | 2.3x |
| 4,096 | 100% | 1.82 | 4.06 | 2.2x |
| 8,192 | 50% | 3.49 | 7.17 | 2.1x |
| 8,192 | 80% | 2.57 | 7.44 | 2.9x |
| 8,192 | 100% | 2.47 | 7.23 | 2.9x |
| 16,384 | 50% | 9.04 | 15.10 | 1.7x |
| 16,384 | 80% | 5.08 | 14.79 | 2.9x |
| 16,384 | 100% | 5.07 | 16.19 | 3.2x |
| 32,768 | 50% | 24.68 | 29.86 | 1.2x |
| 32,768 | 80% | 11.97 | 30.05 | 2.5x |
| 32,768 | 100% | 9.43 | 29.89 | 3.2x |

### 延迟分解（32K, pfx=80%）

| 阶段 | CPU Prefix | Full Recompute |
|------|-----------|----------------|
| CPU→GPU 传输 | 4.63ms (39%) | 27.99ms (93%) |
| FlashAttention | 4.80ms (40%) | — |
| 压缩算法 | 1.61ms (13%) | 1.65ms (6%) |
| scatter write | ~1ms (8%) | ~0.4ms (1%) |
| **总计** | **11.97ms** | **30.05ms** |

### 显存对比

| seq_len | CPU Prefix 峰值 | Full Recompute 峰值 | 差异 |
|---------|----------------|--------------------|----- |
| 4,096 | 102.8 MB | 108.9 MB | -6.1 MB |
| 8,192 | 208.4 MB | 209.2 MB | -0.7 MB |
| 16,384 | 406.8 MB | 409.3 MB | -2.5 MB |
| 32,768 | 804.0 MB | 809.6 MB | -5.6 MB |
| 65,536 | 1599.6 MB | 1610.1 MB | -10.5 MB |

### 分析

1. **CPU Prefix 策略的延迟优势来自减少传输量**：
   - CPU Prefix 只传 prefix KV（80% × seq_len × 2 tensors）
   - Full Recompute 传全部 KV + Q（seq_len × 3 tensors，约 3.75x 数据量）
   - 在 pageable memory 下，传输带宽仅 4-7 GB/s，传输量差异直接决定延迟

2. **FA 是 CPU Prefix 策略的额外开销**：
   - CPU Prefix 需要做 FA 来获得正确的 attention output（用于生成 token）
   - Full Recompute 跳过 FA（假设只需要压缩后的 KV，不需要 output）
   - 但 FA 是必须的——没有 FA 就没有正确的 prefill output

3. **显存几乎相同**：
   - 两种策略最终都需要在 GPU 上持有完整 KV 来做压缩
   - CPU Prefix 额外有 GlobalKVPool，但 Full Recompute 额外有 Q 在 GPU
   - 差异 <10MB，可忽略

4. **pfx_ratio 越高，CPU Prefix 优势越大**：
   - pfx=50% 时只快 1.2-2.1x（因为 extend 部分仍需 FA）
   - pfx=100% 时快 2.2-3.2x（几乎没有 FA 开销，只有传输+压缩）

---

## 四、综合结论与优化建议

### 当前系统状态

1. ✅ **softmax 已正确跳过**：你的 `SnapKVStyleCompressor` 已经使用原始 QK 分数，节省了 46% 的压缩算法时间
2. ✅ **CPU Prefix 策略有效**：相比全量重算快 2-3x，主要节省在 CPU→GPU 传输
3. ⚠ **CPU 传输仍是瓶颈**：使用 pageable memory，带宽仅 4-7 GB/s

### 优化优先级

| 优先级 | 优化方向 | 预期收益 | 难度 |
|--------|---------|---------|------|
| P0 | Pinned memory for CPU transfer | 传输速度 3-5x，总延迟降 30-40% | 低 |
| P1 | 消除 repeat_interleave（grouped matmul） | 压缩算法时间降 38% | 中 |
| P2 | Async CPU transfer + FA overlap | 隐藏传输延迟 | 中 |
| P3 | 减少 GlobalKVPool 读写（直接从 CPU 到 FA） | 减少 1 次 HBM 读写 | 高 |

### 关键数字（32K, ratio=0.5, pfx=80%）

```
当前总延迟:     11.97ms/layer
├── CPU transfer: 4.63ms (39%) ← P0 优化目标
├── FA:           4.80ms (40%) ← 不可压缩
├── 压缩算法:     1.61ms (13%) ← P1 优化目标
└── 其他:         0.93ms (8%)

优化后预期:      ~6ms/layer (pinned memory + grouped matmul)
```
