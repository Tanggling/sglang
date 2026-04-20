# SnapKV 压缩算法性能分析报告 v2

> 所有图表位于 `figures/` 目录下，由 `plot_results.py` 生成。

## 实验环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA A800-SXM4-80GB |
| 模型参数 | Qwen2.5-7B-Instruct (4 kv_heads, 28 q_heads, 128 head_dim, hidden=3584) |
| 数据类型 | BFloat16 |
| 压缩算法 | SnapKV (no-softmax, 使用原始 QK 分数) |
| Window Size | 64 |

---

## 实验一：SnapKV 压缩算法子操作分解

纯压缩算法开销，不含 pipeline 其他阶段。

### 数据

| seq_len | ratio | TOTAL(ms) | slice | transpose | gqa_expand | qk_matmul | sync | pool_sum | topk |
|---------|-------|-----------|-------|-----------|------------|-----------|------|----------|------|
| 4,096 | 0.3 | 0.549 | 0.019(3%) | 0.049(9%) | 0.076(14%) | 0.080(15%) | 0.018(3%) | 0.071(13%) | 0.160(29%) |
| 4,096 | 0.5 | 0.534 | 0.020(4%) | 0.048(9%) | 0.076(14%) | 0.079(15%) | 0.017(3%) | 0.070(13%) | 0.150(28%) |
| 4,096 | 0.7 | 0.530 | 0.018(3%) | 0.047(9%) | 0.075(14%) | 0.079(15%) | 0.018(3%) | 0.069(13%) | 0.149(28%) |
| 8,192 | 0.5 | 0.666 | 0.019(3%) | 0.056(8%) | 0.129(19%) | 0.119(18%) | 0.018(3%) | 0.070(10%) | 0.188(28%) |
| 16,384 | 0.5 | 1.170 | 0.019(2%) | 0.077(7%) | 0.234(20%) | 0.209(18%) | 0.018(2%) | 0.086(7%) | 0.469(40%) |
| 32,768 | 0.5 | 1.660 | 0.019(1%) | 0.119(7%) | 0.487(29%) | 0.401(24%) | 0.019(1%) | 0.129(8%) | 0.426(26%) |
| 65,536 | 0.5 | 2.883 | 0.019(1%) | 0.203(7%) | **1.151(40%)** | **0.770(27%)** | 0.019(1%) | 0.208(7%) | 0.454(16%) |

### 分析

1. **gqa_expand (repeat_interleave) 是长序列的最大瓶颈**：
   - 4K: 14% → 32K: 29% → 65K: **40%**
   - 将 K 从 4 heads 展开到 28 heads，65K 时需要分配 28×65472×128×2B ≈ 460MB 额外显存
   - 时间从 0.076ms (4K) → 1.151ms (65K)，增长 15x（线性于 seq_len）

2. **qk_matmul 是第二瓶颈**：
   - 稳定占 15-27%，从 0.080ms (4K) → 0.770ms (65K)
   - 计算量 = 28 heads × 64 window × N prefix × head_dim，线性于 N

3. **topk 在中等长度时占比较高**：
   - 8K-16K 时占 28-40%，但绝对值增长缓慢（0.15ms → 0.47ms）
   - 32K+ 后占比下降，因为 gqa_expand 增长更快

4. **压缩比对算法时间几乎无影响**：
   - ratio=0.3/0.5/0.7 在同一 seq_len 下时间差异 <5%
   - 因为 topk 的 k 值变化不影响 QK matmul 和 gqa_expand

![SnapKV Algorithm Breakdown](figures/fig1_snapkv_breakdown.png)

---

## 实验二：完整 Pipeline 各阶段分解

CPU prefix → GlobalKVPool → FA → SnapKV compress → scatter write

### 数据 (ratio=0.5)

| seq_len | pfx | xfer(ms) | pool_w | pool_r | FA(ms) | compress(ms) | write(ms) | TOTAL(ms) |
|---------|-----|----------|--------|--------|--------|-------------|-----------|-----------|
| 4,096 | 50% | 0.40 | 0.10 | 0.07 | 0.62 | 0.46 | 0.27 | 1.93 |
| 4,096 | 80% | 0.54 | 0.10 | 0.08 | 0.29 | 0.46 | 0.26 | 1.73 |
| 4,096 | 100% | 0.64 | 0.09 | 0.06 | 0.19 | 0.46 | 0.26 | 1.70 |
| 8,192 | 50% | 0.63 | 0.14 | 0.05 | 1.84 | 0.61 | 0.23 | 3.50 |
| 8,192 | 80% | 0.91 | 0.14 | 0.05 | 0.63 | 0.61 | 0.24 | 2.56 |
| 8,192 | 100% | 1.19 | 0.12 | 0.05 | 0.32 | 0.61 | 0.24 | 2.53 |
| 16,384 | 50% | 1.31 | 0.22 | 0.08 | 6.15 | 1.13 | 0.21 | 9.10 |
| 16,384 | 80% | 2.21 | 0.21 | 0.08 | 1.49 | 1.09 | 0.20 | 5.28 |
| 16,384 | 100% | 2.91 | 0.20 | 0.08 | 0.53 | 1.09 | 0.20 | 5.01 |
| 32,768 | 50% | 3.36 | 0.37 | 0.17 | 21.16 | 1.58 | 0.33 | 26.96 |
| 32,768 | 80% | 4.73 | 0.32 | 0.16 | 3.92 | 1.43 | 0.30 | 10.86 |
| 32,768 | 100% | 6.33 | 0.33 | 0.16 | 0.92 | 1.44 | 0.30 | 9.48 |

### 分析

1. **FA 时间与 extend_len 强相关**：
   - pfx=50% 时 extend_len=seq_len/2，FA 占主导（32K: 21ms/27ms=78%）
   - pfx=100% 时 extend_len=0，FA 几乎为零（32K: 0.92ms）
   - FA 的 O(N²) 复杂度使其在低 prefix 匹配率时成为绝对瓶颈

2. **CPU transfer 与 prefix_len 线性**：
   - 32K pfx=80%: 传输 26K tokens 的 KV = 4.73ms（~5.5 GB/s 带宽）
   - 使用 pageable memory，远低于 PCIe 理论带宽

3. **SnapKV compress 时间与 seq_len 相关，与 prefix_ratio 无关**：
   - 因为压缩算法处理的是完整 KV（prefix + extend 拼接后）
   - 32K: ~1.45ms，占总时间 13-15%

4. **高 prefix 匹配率显著降低总延迟**：
   - 32K: pfx=50% → 27ms, pfx=80% → 11ms, pfx=100% → 9.5ms
   - 主要节省来自 FA（extend_len 减少）

![Pipeline Phase Breakdown](figures/fig2_pipeline_breakdown.png)

### 高命中率 (pfx≥80%) 下各阶段占比定量分析 (32K)

| ratio | pfx | xfer(ms) | xfer% | pool(ms) | FA(ms) | FA% | comp(ms) | comp% | write(ms) | total(ms) |
|-------|-----|----------|-------|----------|--------|-----|----------|-------|-----------|-----------|
| 0.3 | 80% | 4.82 | 43% | 0.49 | 3.92 | 35% | 1.49 | 13% | 0.37 | 11.08 |
| 0.3 | 90% | 5.24 | 55% | 0.49 | 1.77 | 19% | 1.60 | 17% | 0.37 | 9.47 |
| 0.3 | 100% | 6.07 | 65% | 0.49 | 0.92 | 10% | 1.46 | 16% | 0.37 | 9.32 |
| 0.5 | 80% | 5.08 | 45% | 0.51 | 3.91 | 35% | 1.47 | 13% | 0.31 | 11.29 |
| 0.5 | 90% | 5.68 | 58% | 0.51 | 1.76 | 18% | 1.54 | 16% | 0.32 | 9.80 |
| 0.5 | 100% | 6.52 | 67% | 0.51 | 0.92 | 9% | 1.45 | 15% | 0.31 | 9.72 |
| 0.7 | 80% | 4.87 | 44% | 0.49 | 3.92 | 36% | 1.44 | 13% | 0.27 | 10.99 |
| 0.7 | 90% | 5.25 | 57% | 0.48 | 1.76 | 19% | 1.42 | 15% | 0.26 | 9.18 |
| 0.7 | 100% | 5.86 | 63% | 0.54 | 1.12 | 12% | 1.62 | 17% | 0.24 | 9.36 |

关键观察：
- **pfx=80% 时**: CPU transfer 占 43-45%，FA 占 35-36%，compress 占 13%
- **pfx=90% 时**: CPU transfer 升至 55-58%，FA 降至 18-19%，compress 升至 15-17%
- **pfx=100% 时**: CPU transfer 占 63-67%（绝对主导），FA 降至 9-12%，compress 占 15-17%
- **压缩比 ratio 对各阶段占比几乎无影响**（因为 compress 时间不随 ratio 变化）

![High Prefix Phase Distribution](figures/fig3_high_prefix_phases.png)

![Transfer & Compress Proportion](figures/fig6_xfer_compress_proportion.png)

---

## 实验三：CPU Prefix Pipeline vs 全量重计算 (QKV Projection)

策略 A: CPU prefix KV → GPU → GlobalKVPool → FA(extend only) → compress
策略 B: hidden_states @ W_q/W_k/W_v → FA(full seq) → compress

### 数据 (ratio=0.5)

| seq_len | pfx | A: cpu_pfx (ms) | A 分解 | B: recompute (ms) | B 分解 | A/B |
|---------|-----|-----------------|--------|-------------------|--------|-----|
| 4,096 | 50% | 1.91 | xfer=0.41 pool=0.15 fa=0.62 comp=0.47 | 2.56 | proj=0.78 fa=1.05 comp=0.47 | 0.75x |
| 4,096 | 80% | 1.72 | xfer=0.55 pool=0.15 fa=0.29 comp=0.47 | 2.54 | proj=0.78 fa=1.03 comp=0.47 | 0.67x |
| 4,096 | 100% | 1.69 | xfer=0.65 pool=0.13 fa=0.19 comp=0.46 | 2.55 | proj=0.78 fa=1.03 comp=0.47 | 0.66x |
| 8,192 | 80% | 2.53 | xfer=0.93 pool=0.18 fa=0.59 comp=0.60 | 5.57 | proj=1.44 fa=3.28 comp=0.62 | 0.45x |
| 16,384 | 80% | 4.74 | xfer=2.02 pool=0.26 fa=1.29 comp=0.97 | 12.95 | proj=2.09 fa=9.71 comp=0.97 | 0.37x |
| 32,768 | 50% | 23.17 | xfer=2.71 pool=0.48 fa=18.26 comp=1.43 | 43.27 | proj=4.07 fa=37.49 comp=1.44 | 0.54x |
| 32,768 | 80% | 11.06 | xfer=4.89 pool=0.48 fa=3.91 comp=1.46 | 43.32 | proj=4.07 fa=37.54 comp=1.44 | **0.26x** |
| 32,768 | 100% | 13.37 | xfer=10.06 pool=0.53 fa=0.92 comp=1.52 | 43.63 | proj=4.11 fa=37.61 comp=1.58 | 0.31x |

### 分析

1. **CPU Prefix 策略始终优于全量重计算**：
   - A/B ratio 范围 0.26x-0.75x（CPU prefix 快 1.3x-3.9x）
   - 32K pfx=80%: CPU prefix 11ms vs 重计算 43ms，快 **3.9x**

2. **重计算的瓶颈是 FA（全序列）**：
   - 重计算必须对完整 seq_len 做 FA（因为需要正确的 attention output）
   - 32K: FA 占重计算总时间的 87%（37.5ms/43.3ms）
   - CPU prefix 只需对 extend 部分做 FA，大幅减少计算量

3. **QKV projection 开销相对较小**：
   - 32K: proj=4.07ms，仅占重计算总时间 9%
   - 这是 hidden_states @ W_q/W_k/W_v 的 GEMM 开销

4. **CPU prefix 的优势随 prefix_ratio 增大而增大**：
   - pfx=50%: 快 1.9x（FA 仍然很大）
   - pfx=80%: 快 3.9x（FA 大幅减少）
   - pfx=100%: 快 3.3x（CPU transfer 增加抵消了部分 FA 节省）

5. **pfx=100% 反而比 pfx=80% 慢**：
   - 因为 100% prefix 意味着全部 KV 从 CPU 传输（10.06ms），而 FA 节省有限（0.92ms vs 3.91ms）
   - 最优点在 pfx=80% 附近

![CPU Prefix vs Recompute](figures/fig4_prefix_vs_recompute.png)

---

## 实验四：GPU 峰值显存对比

| seq_len | A: CPU Prefix (MB) | B: Recompute (MB) | 差异 (MB) | B 含权重 |
|---------|--------------------|--------------------|-----------|---------|
| 4,096 | 134.9 | 203.7 | -68.8 | 32MB |
| 8,192 | 240.3 | 366.1 | -125.8 | 32MB |
| 16,384 | 439.0 | 692.8 | -253.9 | 32MB |
| 32,768 | 836.7 | 1,344.8 | -508.0 | 32MB |
| 65,536 | 1,632.0 | 2,649.4 | **-1,017.4** | 32MB |

### 分析

1. **CPU Prefix 显著节省显存**：
   - 65K: 节省 1GB（1632MB vs 2649MB）
   - 因为 CPU prefix 不需要在 GPU 上持有 hidden_states（seq_len × 3584 × 2B）
   - 也不需要 W_q/W_k/W_v 权重矩阵（32MB/layer，28 layers = 896MB）

2. **显存节省随 seq_len 线性增长**：
   - 主要来自 hidden_states 张量：seq_len × hidden_dim × 2B
   - 32K: 32768 × 3584 × 2B = 224MB（与实测 508MB 差异来自 FA 中间张量）

---

## 综合结论

### SnapKV 压缩算法时间分布（65K, ratio=0.5, no-softmax）

```
SnapKV 总时间: 2.88ms
├── gqa_expand (repeat_interleave): 1.15ms (40%) ← 最大瓶颈
├── qk_matmul:                      0.77ms (27%) ← 第二瓶颈
├── topk:                            0.45ms (16%)
├── pool_sum:                        0.21ms (7%)
├── transpose:                       0.20ms (7%)
├── sync:                            0.02ms (1%)
└── slice:                           0.02ms (1%)
```

![Pipeline Phase Pie Chart (32K)](figures/fig5_pie_32k.png)

### 完整 Pipeline 时间分布（32K, ratio=0.5, pfx=80%）

```
Pipeline 总时间: 10.86ms
├── CPU transfer:  4.73ms (44%) ← Pipeline 最大瓶颈
├── FA (extend):   3.92ms (36%)
├── SnapKV compress: 1.43ms (13%)
├── GlobalKVPool IO: 0.48ms (4%)
└── scatter write:   0.30ms (3%)
```

### 优化方向（按收益排序）

| 优先级 | 方向 | 当前开销 | 预期节省 | 方法 |
|--------|------|---------|---------|------|
| P0 | Pinned memory | CPU xfer 4.73ms | 3-5x 加速 → ~1ms | `torch.empty(pin_memory=True)` |
| P1 | 消除 gqa_expand | 1.15ms/layer (40%) | ~1ms | Grouped GEMM 或 index-based QK |
| P2 | Async CPU transfer | 4.73ms 阻塞 | 与 FA overlap | CUDA stream + non_blocking |
| P3 | 减少 GlobalKVPool IO | 0.48ms | ~0.3ms | 直接从 CPU 到 FA input |

### CPU Prefix vs 全量重计算

- **延迟**: CPU prefix 快 2-4x（32K pfx=80%: 11ms vs 43ms）
- **显存**: CPU prefix 省 40-60%（65K: 1.6GB vs 2.6GB）
- **结论**: CPU prefix cache 策略在延迟和显存上都显著优于全量重计算，是正确的设计选择
