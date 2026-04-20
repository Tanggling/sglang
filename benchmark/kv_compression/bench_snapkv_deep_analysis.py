"""SnapKV Deep Analysis: softmax vs no-softmax, CPU prefix strategies.

Experiments:
  1. SnapKV with softmax vs without softmax (raw QK scores)
  2. Multi seq_len x ratio x cpu_prefix_ratio grid
  3. CPU prefix + extend Q strategy vs full recompute from CPU

Usage:
    python benchmark/kv_compression/bench_snapkv_deep_analysis.py
"""
import argparse, math, sys, time
from typing import Dict, List
import torch
import torch.nn.functional as F
sys.path.insert(0, ".")
from benchmark.kv_compression.profiler.phase_timer import PhaseTimer


def _snapkv_core(k, v, q, num_to_keep, num_heads, num_kv_heads, timer, ws, use_softmax, label):
    """Shared SnapKV logic. use_softmax=True applies softmax, False uses raw scores."""
    seq_len = k.shape[0]
    head_dim = k.shape[2]
    kv_group = num_heads // num_kv_heads
    ws = min(ws, seq_len)

    with timer.phase(f"{label}"):
        with timer.phase(f"{label}.slice"):
            q_w = q[-ws:]
            k_pre = k[:-ws] if seq_len > ws else k[:0]

        if k_pre.shape[0] == 0:
            return torch.arange(seq_len, device=k.device).unsqueeze(0).expand(num_kv_heads, -1)

        with timer.phase(f"{label}.transpose"):
            q_t = q_w.transpose(0, 1).contiguous()
            k_t = k_pre.transpose(0, 1).contiguous()

        with timer.phase(f"{label}.repeat_interleave"):
            if kv_group > 1:
                k_t = k_t.repeat_interleave(kv_group, dim=0)

        with timer.phase(f"{label}.qk_matmul"):
            attn = torch.matmul(q_t, k_t.transpose(-2, -1)) / math.sqrt(head_dim)

        with timer.phase(f"{label}.sync"):
            torch.cuda.current_stream().synchronize()

        if use_softmax:
            with timer.phase(f"{label}.softmax"):
                attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        # else: use raw attn scores directly (no softmax)

        with timer.phase(f"{label}.pool_sum"):
            attn_sum = attn.sum(dim=1)
            if kv_group > 1:
                attn_sum = attn_sum.view(num_kv_heads, kv_group, -1).sum(dim=1)
            if attn_sum.shape[-1] > 5:
                attn_cache = F.max_pool1d(attn_sum.unsqueeze(0), kernel_size=5, padding=2, stride=1).squeeze(0)
            else:
                attn_cache = attn_sum

        with timer.phase(f"{label}.topk"):
            n_keep = max(1, num_to_keep - ws)
            _, indices = attn_cache.topk(n_keep, dim=-1)
            indices = torch.sort(indices, dim=-1).values
            win_idx = torch.arange(seq_len - ws, seq_len, device=k.device).unsqueeze(0).expand(num_kv_heads, -1)
            keep = torch.cat([indices, win_idx], dim=-1)

    return keep


def snapkv_with_softmax(k, v, q, num_to_keep, num_heads, num_kv_heads, timer, ws=64):
    return _snapkv_core(k, v, q, num_to_keep, num_heads, num_kv_heads, timer, ws, True, "snapkv_softmax")


def snapkv_no_softmax(k, v, q, num_to_keep, num_heads, num_kv_heads, timer, ws=64):
    return _snapkv_core(k, v, q, num_to_keep, num_heads, num_kv_heads, timer, ws, False, "snapkv_raw")

def pipeline_cpu_prefix(k_cpu, v_cpu, k_ext, v_ext, q_ext, pool_k, pool_v,
                        rpool_k, rpool_v, slots, rslots, prefix_len,
                        num_to_keep, num_heads, num_kv_heads, timer, ws=64):
    """SGLang-style: CPU prefix KV → GPU GlobalKVPool → FA → compress → real pool.
    Simulates the actual _forward_extend_global_real_split flow per layer."""
    head_dim = k_ext.shape[2] if k_ext.shape[0] > 0 else k_cpu.shape[2]
    extend_len = k_ext.shape[0]
    seq_len = prefix_len + extend_len
    device = pool_k.device

    with timer.phase("cpu_prefix"):
        # Phase 1a: CPU → GPU transfer (pageable, blocking — matches current impl)
        with timer.phase("cpu_prefix.transfer"):
            k_gpu = k_cpu.to(device, non_blocking=False)
            v_gpu = v_cpu.to(device, non_blocking=False)

        # Phase 1b: Write to GlobalKVPool
        with timer.phase("cpu_prefix.pool_write"):
            pool_k[slots[:prefix_len]] = k_gpu
            pool_v[slots[:prefix_len]] = v_gpu
            if extend_len > 0:
                pool_k[slots[prefix_len:seq_len]] = k_ext
                pool_v[slots[prefix_len:seq_len]] = v_ext

        # Phase 1c: Read back full KV
        with timer.phase("cpu_prefix.pool_read"):
            k_full = pool_k[slots[:seq_len]].clone()
            v_full = pool_v[slots[:seq_len]].clone()

        # Phase 2: FA (simulated with SDPA)
        with timer.phase("cpu_prefix.fa"):
            kv_group = num_heads // num_kv_heads
            q_t = q_ext.transpose(0, 1).unsqueeze(0)
            k_t = k_full.transpose(0, 1).unsqueeze(0)
            v_t = v_full.transpose(0, 1).unsqueeze(0)
            if kv_group > 1:
                k_t = k_t.repeat_interleave(kv_group, dim=1)
                v_t = v_t.repeat_interleave(kv_group, dim=1)
            out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
            del q_t, k_t, v_t, out

        # Phase 3: Compress (no-softmax, matching real impl)
        keep = _snapkv_core(k_full, v_full, q_ext, num_to_keep,
                            num_heads, num_kv_heads, timer, ws, False, "cpu_prefix.compress")

        # Phase 3b: scatter write to real pool
        with timer.phase("cpu_prefix.scatter_write"):
            nk = keep.shape[1]
            for h in range(num_kv_heads):
                rpool_k[rslots[:nk], h] = k_full[keep[h], h]
                rpool_v[rslots[:nk], h] = v_full[keep[h], h]

    return keep


def pipeline_full_recompute(k_cpu_full, v_cpu_full, q_cpu_full,
                            rpool_k, rpool_v, rslots, num_to_keep,
                            num_heads, num_kv_heads, timer, ws=64):
    """Alternative: transfer ALL KV from CPU, compress directly, no GlobalKVPool.
    Simulates: what if we just send full KV from CPU and compress without FA?"""
    device = rpool_k.device
    seq_len = k_cpu_full.shape[0]

    with timer.phase("full_recompute"):
        # Step 1: Transfer all KV from CPU
        with timer.phase("full_recompute.transfer"):
            k_gpu = k_cpu_full.to(device, non_blocking=False)
            v_gpu = v_cpu_full.to(device, non_blocking=False)

        # Step 2: Transfer Q from CPU
        with timer.phase("full_recompute.q_transfer"):
            q_gpu = q_cpu_full.to(device, non_blocking=False)

        # Step 3: Compress directly (no FA needed if we only want compressed KV)
        keep = _snapkv_core(k_gpu, v_gpu, q_gpu, num_to_keep,
                            num_heads, num_kv_heads, timer, ws, False, "full_recompute.compress")

        # Step 4: scatter write to real pool
        with timer.phase("full_recompute.scatter_write"):
            nk = keep.shape[1]
            for h in range(num_kv_heads):
                rpool_k[rslots[:nk], h] = k_gpu[keep[h], h]
                rpool_v[rslots[:nk], h] = v_gpu[keep[h], h]

    return keep

def avg_summaries(all_s):
    avg = {}
    for key in all_s[0]:
        avg[key] = sum(d[key] for d in all_s) / len(all_s)
    return avg


def run_exp1_softmax_comparison(args):
    """Exp1: SnapKV with softmax vs without softmax across seq_lens and ratios."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: SnapKV softmax vs no-softmax (pure algorithm)")
    print("=" * 80)
    dtype = torch.bfloat16
    device = "cuda"

    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    ratios = [float(x) for x in args.ratios.split(",")]

    for seq_len in seq_lens:
        k = torch.randn(seq_len, args.num_kv_heads, args.head_dim, dtype=dtype, device=device)
        v = torch.randn(seq_len, args.num_kv_heads, args.head_dim, dtype=dtype, device=device)
        q = torch.randn(seq_len, args.num_heads, args.head_dim, dtype=dtype, device=device)

        for ratio in ratios:
            ntk = max(1, int(seq_len * (1 - ratio)))
            for variant, fn in [("softmax", snapkv_with_softmax), ("raw(no-sm)", snapkv_no_softmax)]:
                all_s = []
                for i in range(args.warmup + args.repeat):
                    timer = PhaseTimer()
                    fn(k, v, q, ntk, args.num_heads, args.num_kv_heads, timer, args.window_size)
                    s = timer.summary()
                    if i >= args.warmup:
                        all_s.append(s)
                avg = avg_summaries(all_s)
                label = "snapkv_softmax" if "softmax" in variant else "snapkv_raw"
                total = avg.get(label, 0)
                sm_ms = avg.get(f"{label}.softmax", 0)
                qk_ms = avg.get(f"{label}.qk_matmul", 0)
                ri_ms = avg.get(f"{label}.repeat_interleave", 0)
                ps_ms = avg.get(f"{label}.pool_sum", 0)
                tk_ms = avg.get(f"{label}.topk", 0)
                print(f"  seq={seq_len:>6} ratio={ratio} {variant:<12} "
                      f"total={total:>7.3f}ms  qk={qk_ms:.3f} ri={ri_ms:.3f} "
                      f"sm={sm_ms:.3f} pool={ps_ms:.3f} topk={tk_ms:.3f}")
        print()


def run_exp2_pipeline_comparison(args):
    """Exp2: CPU prefix pipeline vs full recompute across configs."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: CPU prefix pipeline vs full-recompute-from-CPU")
    print("=" * 80)
    dtype = torch.bfloat16
    device = "cuda"

    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    ratios = [float(x) for x in args.ratios.split(",")]
    prefix_ratios = [float(x) for x in args.prefix_ratios.split(",")]

    for seq_len in seq_lens:
        for ratio in ratios:
            ntk = max(1, int(seq_len * (1 - ratio)))
            pool_sz = seq_len + 256

            for pfx_ratio in prefix_ratios:
                prefix_len = int(seq_len * pfx_ratio)
                extend_len = seq_len - prefix_len

                # Shared data
                k_cpu = torch.randn(prefix_len, args.num_kv_heads, args.head_dim, dtype=dtype)
                v_cpu = torch.randn(prefix_len, args.num_kv_heads, args.head_dim, dtype=dtype)
                k_ext = torch.randn(extend_len, args.num_kv_heads, args.head_dim, dtype=dtype, device=device)
                v_ext = torch.randn(extend_len, args.num_kv_heads, args.head_dim, dtype=dtype, device=device)
                q_ext = torch.randn(max(extend_len, args.window_size), args.num_heads, args.head_dim, dtype=dtype, device=device)

                pool_k = torch.zeros(pool_sz, args.num_kv_heads, args.head_dim, dtype=dtype, device=device)
                pool_v = torch.zeros(pool_sz, args.num_kv_heads, args.head_dim, dtype=dtype, device=device)
                rpool_k = torch.zeros(pool_sz, args.num_kv_heads, args.head_dim, dtype=dtype, device=device)
                rpool_v = torch.zeros(pool_sz, args.num_kv_heads, args.head_dim, dtype=dtype, device=device)
                slots = torch.arange(seq_len, device=device)
                rslots = torch.arange(ntk, device=device)

                # Full KV+Q on CPU for recompute path
                k_cpu_full = torch.randn(seq_len, args.num_kv_heads, args.head_dim, dtype=dtype)
                v_cpu_full = torch.randn(seq_len, args.num_kv_heads, args.head_dim, dtype=dtype)
                q_cpu_full = torch.randn(max(seq_len, args.window_size), args.num_heads, args.head_dim, dtype=dtype)

                # --- Run CPU prefix pipeline ---
                all_s1 = []
                for i in range(args.warmup + args.repeat):
                    timer = PhaseTimer()
                    pipeline_cpu_prefix(k_cpu, v_cpu, k_ext, v_ext, q_ext,
                                        pool_k, pool_v, rpool_k, rpool_v,
                                        slots, rslots, prefix_len, ntk,
                                        args.num_heads, args.num_kv_heads, timer, args.window_size)
                    s = timer.summary()
                    if i >= args.warmup:
                        all_s1.append(s)
                avg1 = avg_summaries(all_s1)

                # --- Run full recompute ---
                all_s2 = []
                for i in range(args.warmup + args.repeat):
                    timer = PhaseTimer()
                    pipeline_full_recompute(k_cpu_full, v_cpu_full, q_cpu_full,
                                            rpool_k, rpool_v, rslots, ntk,
                                            args.num_heads, args.num_kv_heads, timer, args.window_size)
                    s = timer.summary()
                    if i >= args.warmup:
                        all_s2.append(s)
                avg2 = avg_summaries(all_s2)

                t1 = avg1.get("cpu_prefix", 0)
                t1_xfer = avg1.get("cpu_prefix.transfer", 0)
                t1_fa = avg1.get("cpu_prefix.fa", 0)
                t1_comp = avg1.get("cpu_prefix.compress", 0)
                t2 = avg2.get("full_recompute", 0)
                t2_xfer = avg2.get("full_recompute.transfer", 0) + avg2.get("full_recompute.q_transfer", 0)
                t2_comp = avg2.get("full_recompute.compress", 0)

                speedup = t1 / t2 if t2 > 0 else 0
                print(f"  seq={seq_len:>6} ratio={ratio} pfx={pfx_ratio:.0%} | "
                      f"CPU_PREFIX: {t1:>7.2f}ms (xfer={t1_xfer:.2f} fa={t1_fa:.2f} comp={t1_comp:.2f}) | "
                      f"FULL_RECOMP: {t2:>7.2f}ms (xfer={t2_xfer:.2f} comp={t2_comp:.2f}) | "
                      f"ratio={speedup:.2f}x")

                del k_cpu, v_cpu, k_ext, v_ext, q_ext, pool_k, pool_v, rpool_k, rpool_v
                del k_cpu_full, v_cpu_full, q_cpu_full
                torch.cuda.empty_cache()
        print()


def run_exp3_memory_comparison(args):
    """Exp3: Peak memory comparison between the two strategies."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Peak GPU memory comparison")
    print("=" * 80)
    dtype = torch.bfloat16
    device = "cuda"

    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    for seq_len in seq_lens:
        ratio = 0.5
        pfx_ratio = 0.8
        ntk = max(1, int(seq_len * (1 - ratio)))
        prefix_len = int(seq_len * pfx_ratio)
        extend_len = seq_len - prefix_len
        pool_sz = seq_len + 256

        # CPU prefix pipeline memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1024**2

        k_cpu = torch.randn(prefix_len, args.num_kv_heads, args.head_dim, dtype=dtype)
        v_cpu = torch.randn(prefix_len, args.num_kv_heads, args.head_dim, dtype=dtype)
        k_ext = torch.randn(extend_len, args.num_kv_heads, args.head_dim, dtype=dtype, device=device)
        v_ext = torch.randn(extend_len, args.num_kv_heads, args.head_dim, dtype=dtype, device=device)
        q_ext = torch.randn(max(extend_len, args.window_size), args.num_heads, args.head_dim, dtype=dtype, device=device)
        pool_k = torch.zeros(pool_sz, args.num_kv_heads, args.head_dim, dtype=dtype, device=device)
        pool_v = torch.zeros(pool_sz, args.num_kv_heads, args.head_dim, dtype=dtype, device=device)
        rpool_k = torch.zeros(pool_sz, args.num_kv_heads, args.head_dim, dtype=dtype, device=device)
        rpool_v = torch.zeros(pool_sz, args.num_kv_heads, args.head_dim, dtype=dtype, device=device)
        slots = torch.arange(seq_len, device=device)
        rslots = torch.arange(ntk, device=device)

        timer = PhaseTimer()
        pipeline_cpu_prefix(k_cpu, v_cpu, k_ext, v_ext, q_ext,
                            pool_k, pool_v, rpool_k, rpool_v,
                            slots, rslots, prefix_len, ntk,
                            args.num_heads, args.num_kv_heads, timer, args.window_size)
        torch.cuda.synchronize()
        peak1 = torch.cuda.max_memory_allocated() / 1024**2

        del k_cpu, v_cpu, k_ext, v_ext, q_ext, pool_k, pool_v, rpool_k, rpool_v
        torch.cuda.empty_cache()

        # Full recompute memory
        torch.cuda.reset_peak_memory_stats()
        k_cpu_full = torch.randn(seq_len, args.num_kv_heads, args.head_dim, dtype=dtype)
        v_cpu_full = torch.randn(seq_len, args.num_kv_heads, args.head_dim, dtype=dtype)
        q_cpu_full = torch.randn(max(seq_len, args.window_size), args.num_heads, args.head_dim, dtype=dtype)
        rpool_k2 = torch.zeros(pool_sz, args.num_kv_heads, args.head_dim, dtype=dtype, device=device)
        rpool_v2 = torch.zeros(pool_sz, args.num_kv_heads, args.head_dim, dtype=dtype, device=device)
        rslots2 = torch.arange(ntk, device=device)

        timer2 = PhaseTimer()
        pipeline_full_recompute(k_cpu_full, v_cpu_full, q_cpu_full,
                                rpool_k2, rpool_v2, rslots2, ntk,
                                args.num_heads, args.num_kv_heads, timer2, args.window_size)
        torch.cuda.synchronize()
        peak2 = torch.cuda.max_memory_allocated() / 1024**2

        del k_cpu_full, v_cpu_full, q_cpu_full, rpool_k2, rpool_v2
        torch.cuda.empty_cache()

        saving = peak1 - peak2
        print(f"  seq={seq_len:>6} | CPU_PREFIX peak={peak1:>8.1f}MB | FULL_RECOMP peak={peak2:>8.1f}MB | diff={saving:>+8.1f}MB")


def main():
    parser = argparse.ArgumentParser(description="SnapKV deep analysis benchmark")
    parser.add_argument("--seq-lens", default="4096,8192,16384,32768,65536")
    parser.add_argument("--ratios", default="0.3,0.5,0.7")
    parser.add_argument("--prefix-ratios", default="0.5,0.8,1.0")
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=28)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--exp", default="all", help="all, 1, 2, 3")
    args = parser.parse_args()

    print(f"Config: kv_heads={args.num_kv_heads}, q_heads={args.num_heads}, "
          f"head_dim={args.head_dim}, window={args.window_size}")

    if args.exp in ("all", "1"):
        run_exp1_softmax_comparison(args)
    if args.exp in ("all", "2"):
        run_exp2_pipeline_comparison(args)
    if args.exp in ("all", "3"):
        run_exp3_memory_comparison(args)

if __name__ == "__main__":
    main()
