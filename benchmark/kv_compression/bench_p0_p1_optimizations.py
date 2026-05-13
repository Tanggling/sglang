"""
Benchmark P0 (Pinned Memory) and P1 (GQA Expand Elimination) Optimizations

P0: Compare pageable vs pinned memory for CPU→GPU KV cache transfer.
    Expected: 3-5x speedup with pinned memory.

P1: Compare repeat_interleave (current) vs grouped QK matmul (optimized).
    Current: K [kv_heads, prefix_len, head_dim] → repeat_interleave → [q_heads, prefix_len, head_dim]
             then matmul with Q [q_heads, window, head_dim]
    Optimized: Compute QK per kv_head group without expanding K.

Usage:
    python benchmark/kv_compression/bench_p0_p1_optimizations.py
    python benchmark/kv_compression/bench_p0_p1_optimizations.py --seq-lens 4096,8192,16384,32768,65536
"""

import argparse
import math
import torch
import torch.nn.functional as F
import time


# ═══════════════════════════════════════════════════════════════════════════════
# P0: Pinned Memory Benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def bench_p0_transfer(
    num_tokens: int,
    num_kv_heads: int = 4,
    head_dim: int = 128,
    dtype=torch.bfloat16,
    pinned: bool = False,
    warmup: int = 5,
    repeat: int = 20,
) -> float:
    """Benchmark CPU→GPU transfer. Returns average time in ms."""
    shape = (num_tokens, num_kv_heads, head_dim)

    if pinned:
        k_cpu = torch.empty(shape, dtype=dtype, pin_memory=True)
        v_cpu = torch.empty(shape, dtype=dtype, pin_memory=True)
        k_cpu.normal_()
        v_cpu.normal_()
    else:
        k_cpu = torch.randn(shape, dtype=dtype)
        v_cpu = torch.randn(shape, dtype=dtype)

    times = []
    for i in range(warmup + repeat):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        k_gpu = k_cpu.to("cuda", non_blocking=pinned)
        v_gpu = v_cpu.to("cuda", non_blocking=pinned)
        if pinned:
            torch.cuda.synchronize()

        end.record()
        torch.cuda.synchronize()

        if i >= warmup:
            times.append(start.elapsed_time(end))
        del k_gpu, v_gpu

    return sum(times) / len(times)


def run_p0_benchmark(seq_lens, num_kv_heads=4, head_dim=128):
    """Run P0 pinned memory benchmark."""
    print("\n" + "=" * 80)
    print("P0: Pinned Memory CPU→GPU Transfer Benchmark")
    print("=" * 80)
    print(f"Config: num_kv_heads={num_kv_heads}, head_dim={head_dim}, dtype=bf16")
    print(f"Data per token: K+V = 2 × {num_kv_heads} × {head_dim} × 2B = {2*num_kv_heads*head_dim*2}B")
    print()

    header = f"{'seq_len':>8} {'data_MB':>8} {'pageable(ms)':>14} {'pinned(ms)':>12} {'speedup':>8} {'page_BW':>10} {'pin_BW':>10}"
    print(header)
    print("─" * len(header))

    results = []
    for n_tok in seq_lens:
        data_mb = n_tok * 2 * num_kv_heads * head_dim * 2 / (1024**2)

        t_page = bench_p0_transfer(n_tok, num_kv_heads, head_dim, pinned=False)
        t_pin = bench_p0_transfer(n_tok, num_kv_heads, head_dim, pinned=True)

        speedup = t_page / t_pin if t_pin > 0 else 0
        bw_page = data_mb / t_page * 1000 if t_page > 0 else 0
        bw_pin = data_mb / t_pin * 1000 if t_pin > 0 else 0

        print(f"{n_tok:>8} {data_mb:>8.1f} {t_page:>12.3f}ms {t_pin:>10.3f}ms {speedup:>7.2f}x {bw_page:>8.1f}GB/s {bw_pin:>8.1f}GB/s")
        results.append({
            "seq_len": n_tok, "data_mb": data_mb,
            "pageable_ms": t_page, "pinned_ms": t_pin,
            "speedup": speedup, "bw_page": bw_page, "bw_pin": bw_pin,
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# P1: GQA Expand Elimination Benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def snapkv_original(q_window, k_prefix, num_kv_heads, kv_group_num, head_dim):
    """
    Original SnapKV: repeat_interleave K then full matmul.
    q_window: [num_q_heads, window_size, head_dim]
    k_prefix: [num_kv_heads, prefix_len, head_dim]

    Note: repeat_interleave(kv_group_num, dim=0) produces:
      [k0,k0,...(×group), k1,k1,...(×group), ...]
    And Q is arranged as:
      [q0,q1,...q_{group-1}, q_{group},...q_{2*group-1}, ...]
    where q_0..q_{group-1} all attend to k_0.
    So the grouping is: q[i*group : (i+1)*group] → k[i]
    """
    # Expand K from kv_heads to q_heads
    k_expanded = k_prefix.repeat_interleave(kv_group_num, dim=0)  # [q_heads, prefix_len, head_dim]
    # QK matmul
    attn = torch.matmul(q_window, k_expanded.transpose(-2, -1)) / math.sqrt(head_dim)
    # Sum over window dimension → [q_heads, prefix_len]
    attn_sum = attn.sum(dim=1)
    # Reduce back to kv_heads: sum over groups
    # q_heads are arranged as [group0_for_kv0, group1_for_kv0, ..., group0_for_kv1, ...]
    attn_sum = attn_sum.view(num_kv_heads, kv_group_num, -1).sum(dim=1)
    return attn_sum


def snapkv_grouped_matmul(q_window, k_prefix, num_kv_heads, kv_group_num, head_dim):
    """
    Optimized: Grouped matmul without expanding K.
    q_window: [num_q_heads, window_size, head_dim]
    k_prefix: [num_kv_heads, prefix_len, head_dim]

    Strategy: Reshape Q to [kv_heads, group_num * window, head_dim], then matmul with K [kv_heads, prefix_len, head_dim].
    This avoids allocating the expanded K tensor entirely.
    """
    num_q_heads = num_kv_heads * kv_group_num
    window_size = q_window.shape[1]
    prefix_len = k_prefix.shape[1]

    # Reshape Q: [q_heads, window, head_dim] → [kv_heads, group_num, window, head_dim]
    #           → [kv_heads, group_num * window, head_dim]
    q_grouped = q_window.view(num_kv_heads, kv_group_num, window_size, head_dim)
    q_grouped = q_grouped.reshape(num_kv_heads, kv_group_num * window_size, head_dim)

    # Matmul: [kv_heads, group_num*window, head_dim] × [kv_heads, head_dim, prefix_len]
    #       → [kv_heads, group_num*window, prefix_len]
    attn = torch.bmm(q_grouped, k_prefix.transpose(-2, -1)) / math.sqrt(head_dim)

    # Sum over the (group_num * window) dimension → [kv_heads, prefix_len]
    attn_sum = attn.sum(dim=1)
    return attn_sum


def snapkv_einsum(q_window, k_prefix, num_kv_heads, kv_group_num, head_dim):
    """
    Alternative: Use einsum to avoid expand.
    q_window: [num_q_heads, window_size, head_dim] → reshape to [kv_heads, group, window, head_dim]
    k_prefix: [num_kv_heads, prefix_len, head_dim]
    """
    window_size = q_window.shape[1]

    # Reshape Q to [kv_heads, group_num, window, head_dim]
    q_grouped = q_window.view(num_kv_heads, kv_group_num, window_size, head_dim)

    # einsum: (kv_heads, group, window, dim) × (kv_heads, prefix, dim) → (kv_heads, group, window, prefix)
    # Then sum over group and window → (kv_heads, prefix)
    attn = torch.einsum('hgwd,hpd->hgwp', q_grouped, k_prefix) / math.sqrt(head_dim)
    attn_sum = attn.sum(dim=(1, 2))  # sum over group and window
    return attn_sum


def bench_p1_method(method_fn, q_window, k_prefix, num_kv_heads, kv_group_num, head_dim, warmup=10, repeat=50):
    """Benchmark a single P1 method. Returns avg time in ms."""
    # Warmup
    for _ in range(warmup):
        _ = method_fn(q_window, k_prefix, num_kv_heads, kv_group_num, head_dim)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = method_fn(q_window, k_prefix, num_kv_heads, kv_group_num, head_dim)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return sum(times) / len(times)


def run_p1_benchmark(seq_lens, num_kv_heads=4, num_q_heads=28, head_dim=128, window_size=64):
    """Run P1 GQA expand elimination benchmark."""
    print("\n" + "=" * 80)
    print("P1: GQA Expand Elimination Benchmark (SnapKV QK Matmul)")
    print("=" * 80)
    kv_group_num = num_q_heads // num_kv_heads
    print(f"Config: num_kv_heads={num_kv_heads}, num_q_heads={num_q_heads}, "
          f"kv_group_num={kv_group_num}, head_dim={head_dim}, window_size={window_size}")
    print(f"Original: repeat_interleave K [{num_kv_heads}→{num_q_heads}] then matmul")
    print(f"Grouped:  reshape Q [{num_q_heads}→{num_kv_heads}×{kv_group_num}] then bmm (no K expand)")
    print(f"Einsum:   einsum-based grouped computation (no K expand)")
    print()

    header = f"{'prefix_len':>10} {'expand_MB':>10} {'original(ms)':>14} {'grouped(ms)':>13} {'einsum(ms)':>12} {'grp_speedup':>12} {'ein_speedup':>12} {'correct':>8}"
    print(header)
    print("─" * len(header))

    results = []
    for seq_len in seq_lens:
        prefix_len = seq_len - window_size
        if prefix_len <= 0:
            continue

        # Memory for expanded K: [q_heads, prefix_len, head_dim] × 2B
        expand_mb = num_q_heads * prefix_len * head_dim * 2 / (1024**2)

        q_window = torch.randn(num_q_heads, window_size, head_dim, device="cuda", dtype=torch.bfloat16)
        k_prefix = torch.randn(num_kv_heads, prefix_len, head_dim, device="cuda", dtype=torch.bfloat16)

        # Correctness check
        ref = snapkv_original(q_window, k_prefix, num_kv_heads, kv_group_num, head_dim)
        out_grp = snapkv_grouped_matmul(q_window, k_prefix, num_kv_heads, kv_group_num, head_dim)
        out_ein = snapkv_einsum(q_window, k_prefix, num_kv_heads, kv_group_num, head_dim)

        correct_grp = torch.allclose(ref.float(), out_grp.float(), rtol=1e-2, atol=1e-2)
        correct_ein = torch.allclose(ref.float(), out_ein.float(), rtol=1e-2, atol=1e-2)
        correct = "✓" if (correct_grp and correct_ein) else f"grp={'✓' if correct_grp else '✗'} ein={'✓' if correct_ein else '✗'}"

        # Benchmark
        t_orig = bench_p1_method(snapkv_original, q_window, k_prefix, num_kv_heads, kv_group_num, head_dim)
        t_grp = bench_p1_method(snapkv_grouped_matmul, q_window, k_prefix, num_kv_heads, kv_group_num, head_dim)
        t_ein = bench_p1_method(snapkv_einsum, q_window, k_prefix, num_kv_heads, kv_group_num, head_dim)

        sp_grp = t_orig / t_grp if t_grp > 0 else 0
        sp_ein = t_orig / t_ein if t_ein > 0 else 0

        print(f"{prefix_len:>10} {expand_mb:>8.1f}MB {t_orig:>12.3f}ms {t_grp:>11.3f}ms {t_ein:>10.3f}ms {sp_grp:>10.2f}x {sp_ein:>10.2f}x {correct:>8}")

        results.append({
            "prefix_len": prefix_len, "expand_mb": expand_mb,
            "original_ms": t_orig, "grouped_ms": t_grp, "einsum_ms": t_ein,
            "speedup_grp": sp_grp, "speedup_ein": sp_ein,
            "correct": correct,
        })

        del q_window, k_prefix
        torch.cuda.empty_cache()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# P1 Memory Savings Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def run_p1_memory_analysis(seq_lens, num_kv_heads=4, num_q_heads=28, head_dim=128, window_size=64):
    """Analyze memory savings from eliminating GQA expand."""
    print("\n" + "=" * 80)
    print("P1: Memory Savings from Eliminating GQA Expand")
    print("=" * 80)
    kv_group_num = num_q_heads // num_kv_heads
    print(f"Expanded K shape: [{num_q_heads}, prefix_len, {head_dim}] × bf16")
    print(f"Original K shape: [{num_kv_heads}, prefix_len, {head_dim}] × bf16")
    print(f"Savings ratio: {kv_group_num}x less memory for K in QK matmul")
    print()

    header = f"{'prefix_len':>10} {'expanded_K(MB)':>15} {'original_K(MB)':>15} {'saved(MB)':>10}"
    print(header)
    print("─" * len(header))

    for seq_len in seq_lens:
        prefix_len = seq_len - window_size
        if prefix_len <= 0:
            continue
        expanded = num_q_heads * prefix_len * head_dim * 2 / (1024**2)
        original = num_kv_heads * prefix_len * head_dim * 2 / (1024**2)
        saved = expanded - original
        print(f"{prefix_len:>10} {expanded:>13.1f}MB {original:>13.1f}MB {saved:>8.1f}MB")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Benchmark P0 and P1 optimizations")
    parser.add_argument("--seq-lens", default="4096,8192,16384,32768,65536",
                        help="Comma-separated sequence lengths")
    parser.add_argument("--num-kv-heads", type=int, default=4,
                        help="Number of KV heads (Qwen2.5-7B: 4)")
    parser.add_argument("--num-q-heads", type=int, default=28,
                        help="Number of Q heads (Qwen2.5-7B: 28)")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--p0-only", action="store_true")
    parser.add_argument("--p1-only", action="store_true")
    args = parser.parse_args()

    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    if not args.p1_only:
        run_p0_benchmark(seq_lens, num_kv_heads=args.num_kv_heads, head_dim=args.head_dim)

    if not args.p0_only:
        run_p1_benchmark(seq_lens, num_kv_heads=args.num_kv_heads,
                         num_q_heads=args.num_q_heads, head_dim=args.head_dim,
                         window_size=args.window_size)
        run_p1_memory_analysis(seq_lens, num_kv_heads=args.num_kv_heads,
                               num_q_heads=args.num_q_heads, head_dim=args.head_dim,
                               window_size=args.window_size)

    print("\n" + "=" * 80)
    print("Summary & Next Steps")
    print("=" * 80)
    print("""
P0 (Pinned Memory):
  - If speedup > 2x: Modify PrefixCPUCache to allocate with pin_memory=True
  - Implementation: Change torch.empty() → torch.empty(pin_memory=True) in prefix_cpu_cache.py
  - Risk: Pinned memory is limited; need to cap total pinned allocation

P1 (GQA Expand Elimination):
  - If grouped_matmul speedup > 1.3x: Replace repeat_interleave in SnapKVStyleCompressor
  - Implementation: Reshape Q instead of expanding K, use torch.bmm
  - Benefit: Also saves significant GPU memory (no expanded K allocation)
""")


if __name__ == "__main__":
    main()
