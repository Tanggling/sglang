# KV Compression Pipeline Benchmarks

针对 KV Cache 压缩流水线的性能分析工具集，用于量化各阶段时间开销、显存占用，并验证 IO 感知优化方向的理论分析。

## 快速开始

```bash
# 三阶段时间分解（核心）
python benchmark/kv_compression/bench_phase_breakdown.py \
    --seq-lens 1024,4096,8192 --methods key_norm,snapkv --ratios 0.5

# 压缩方法对比
python benchmark/kv_compression/bench_compression_methods.py \
    --seq-lens 1024,4096,8192,16384

# CPU→GPU 传输模式对比
python benchmark/kv_compression/bench_cpu_transfer.py

# 显存占用分析
python benchmark/kv_compression/bench_memory_profile.py \
    --seq-lens 4096,8192 --methods key_norm,snapkv

# SnapKV 子操作分解
python benchmark/kv_compression/bench_snapkv_bottlenecks.py \
    --seq-lens 2048,4096,8192,16384

# IO 复杂度理论分析
python benchmark/kv_compression/analysis/io_complexity_model.py \
    --model llama-3-8b --seq-len 8192 --ratio 0.5
```

## 文件结构

```
benchmark/kv_compression/
├── profiler/                          # 可复用的 profiling 工具
│   ├── phase_timer.py                 # 分阶段 CUDA event 计时器
│   ├── memory_tracker.py              # GPU 显存快照追踪
│   └── hbm_model.py                   # 理论 HBM 访问量模型
├── bench_phase_breakdown.py           # 三阶段时间分解
├── bench_compression_methods.py       # key_norm vs snapkv 对比
├── bench_cpu_transfer.py              # CPU→GPU 传输模式对比
├── bench_memory_profile.py            # 各阶段显存占用
├── bench_snapkv_bottlenecks.py        # SnapKV 子操作分解
└── analysis/
    └── io_complexity_model.py         # IO 复杂度 + 优化潜力分析
```

## 关键发现（预期）

1. **Phase 1 (CPU transfer)** 在 CPU prefix cache 命中时占主导
2. **Phase 3 (compression)** 中 SnapKV 的 `synchronize()` 和 `Q@K^T` 是瓶颈
3. **Pinned memory** 可显著加速 CPU→GPU 传输
4. **LSE-based importance** 可消除 Phase 3 的 K,V 重读（~30-50% 减少）
