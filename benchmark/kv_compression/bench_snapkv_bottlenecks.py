"""Benchmark: SnapKV sub-operation decomposition.

Breaks down SnapKV's compress() into individual operations to identify
the dominant bottleneck.

Usage:
    python benchmark/kv_compression/bench_snapkv_bottlenecks.py \
        --seq-lens 2048,4096,8192,16384
"""

import argparse
import math
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from benchmark.kv_compression.profiler.phase_timer import PhaseTimer


def bench_snapkv_ops(
    seq_len: int,
    num_kv_heads: int = 8,
    num_heads: int = 32,
    head_dim: int = 128,
    compression_ratio: float = 0.5,
    window_size: int = 64,
    warmup: int = 5,
    repeat: int = 20,
):
    dtype = torch.bfloat16
    device = "cuda"
    kv_group = num_heads // num_kv_heads
    window_size = min(window_size, seq_len)
    num_to_keep = max(1, int(seq_len * (1 - compression_ratio)))

    k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    q = torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device=device)

    all_summaries = []
    for i in range(warmup + repeat):
        timer = PhaseTimer()

        with timer.phase("1_slice"):
            q_w = q[-window_size:]
            k_pre = k[:-window_size]

        prefix_len = k_pre.shape[0]
        if prefix_len == 0:
            continue

        with timer.phase("2_transpose_contiguous"):
            q_t = q_w.transpose(0, 1).contiguous()
            k_t = k_pre.transpose(0, 1).contiguous()

        with timer.phase("3_repeat_interleave"):
            if kv_group > 1:
                k_expanded = k_t.repeat_interleave(kv_group, dim=0)
            else:
                k_expanded = k_t

        with timer.phase("4_qk_matmul"):
            attn = torch.matmul(q_t, k_expanded.transpose(-2, -1)) / math.sqrt(head_dim)

        with timer.phase("5_synchronize"):
            torch.cuda.current_stream().synchronize()

        with timer.phase("6_softmax"):
            attn_soft = F.softmax(attn, dim=-1, dtype=torch.float32).to(dtype)

        with timer.phase("7_sum_pool"):
            attn_sum = attn_soft.sum(dim=1)
            if kv_group > 1:
                attn_sum = attn_sum.view(num_kv_heads, kv_group, -1).sum(dim=1)
            if attn_sum.shape[-1] > 5:
                attn_cache = F.max_pool1d(
                    attn_sum.unsqueeze(0), kernel_size=5, padding=2, stride=1
                ).squeeze(0)
            else:
                attn_cache = attn_sum

        with timer.phase("8_topk"):
            n_keep = max(1, num_to_keep - window_size)
            _, indices = attn_cache.topk(n_keep, dim=-1)
            indices = torch.sort(indices, dim=-1).values

        s = timer.summary()
        if i >= warmup:
            all_summaries.append(s)

        del q_t, k_t, k_expanded, attn, attn_soft, attn_sum, attn_cache

    if not all_summaries:
        return {}

    avg = {}
    for key in all_summaries[0]:
        avg[key] = sum(d[key] for d in all_summaries) / len(all_summaries)
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", default="2048,4096,8192,16384")
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    print(f"SnapKV Sub-operation Breakdown")
    print(f"Config: kv_heads={args.num_kv_heads}, q_heads={args.num_heads}, "
          f"head_dim={args.head_dim}, window={args.window_size}, ratio={args.ratio}")
    print()

    # Header
    ops = ["1_slice", "2_transpose_contiguous", "3_repeat_interleave",
           "4_qk_matmul", "5_synchronize", "6_softmax", "7_sum_pool", "8_topk"]

    print(f"{'seq_len':>8}", end="")
    for op in ops:
        label = op.split("_", 1)[1][:12]
        print(f" {label:>12}", end="")
    print(f" {'TOTAL':>10}")
    print("─" * (8 + 13 * len(ops) + 12))

    for seq_len in seq_lens:
        avg = bench_snapkv_ops(
            seq_len, args.num_kv_heads, args.num_heads, args.head_dim,
            args.ratio, args.window_size, args.warmup, args.repeat,
        )
        if not avg:
            print(f"{seq_len:>8}  (skipped, seq_len <= window_size)")
            continue

        total = sum(avg.values())
        print(f"{seq_len:>8}", end="")
        for op in ops:
            ms = avg.get(op, 0)
            pct = ms / total * 100 if total > 0 else 0
            print(f" {ms:>7.3f}({pct:>3.0f}%)", end="")
        print(f" {total:>10.3f}")

        # Highlight dominant op
        if avg:
            dominant = max(avg, key=avg.get)
            print(f"         ^ dominant: {dominant} = {avg[dominant]:.3f}ms "
                  f"({avg[dominant]/total*100:.0f}%)")
        print()


if __name__ == "__main__":
    main()
