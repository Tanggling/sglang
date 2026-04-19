"""Benchmark: Three-phase time breakdown of KV compression pipeline.

Simulates the three-phase pipeline with synthetic tensors to measure
per-phase timing without requiring a full model load.

Usage:
    python benchmark/kv_compression/bench_phase_breakdown.py \
        --seq-lens 1024,4096,8192 --methods key_norm,snapkv --ratios 0.5
"""

import argparse
import sys
import time
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from benchmark.kv_compression.profiler.phase_timer import PhaseTimer
from benchmark.kv_compression.profiler.memory_tracker import MemoryTracker


def make_synthetic_kv(
    seq_len: int, num_kv_heads: int, head_dim: int,
    dtype=torch.bfloat16, device="cuda",
):
    k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    return k, v


def make_synthetic_q(
    seq_len: int, num_heads: int, head_dim: int,
    dtype=torch.bfloat16, device="cuda",
):
    return torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device=device)


# ── Phase simulators ─────────────────────────────────────────────────────────

def simulate_phase1_assemble(
    k_cpu: torch.Tensor, v_cpu: torch.Tensor,
    k_extend: torch.Tensor, v_extend: torch.Tensor,
    pool_k: torch.Tensor, pool_v: torch.Tensor,
    slots: torch.Tensor,
    prefix_len: int,
    timer: PhaseTimer,
):
    """Phase 1: CPU transfer + GlobalKVPool write + read back."""
    with timer.phase("phase1"):
        # Sub-phase: CPU → GPU transfer
        with timer.phase("phase1.cpu_transfer"):
            k_gpu = k_cpu.to(pool_k.device, non_blocking=False)
            v_gpu = v_cpu.to(pool_v.device, non_blocking=False)

        # Sub-phase: Write to GlobalKVPool
        with timer.phase("phase1.pool_write"):
            pool_k[slots[:prefix_len]] = k_gpu
            pool_v[slots[:prefix_len]] = v_gpu
            extend_len = k_extend.shape[0]
            if extend_len > 0:
                pool_k[slots[prefix_len:prefix_len + extend_len]] = k_extend
                pool_v[slots[prefix_len:prefix_len + extend_len]] = v_extend

        # Sub-phase: Read back full KV for FA
        with timer.phase("phase1.pool_read"):
            k_full = pool_k[slots].clone()
            v_full = pool_v[slots].clone()

    return k_full, v_full


def simulate_phase2_flashattn(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    num_heads: int, num_kv_heads: int, head_dim: int,
    timer: PhaseTimer,
):
    """Phase 2: FlashAttention (simulated with scaled_dot_product_attention)."""
    with timer.phase("phase2"):
        seq_len_q = q.shape[0]
        seq_len_k = k.shape[0]
        kv_group = num_heads // num_kv_heads

        q_t = q.transpose(0, 1).unsqueeze(0)  # [1, num_heads, seq_q, head_dim]
        k_t = k.transpose(0, 1).unsqueeze(0)  # [1, num_kv_heads, seq_k, head_dim]
        v_t = v.transpose(0, 1).unsqueeze(0)

        if kv_group > 1:
            k_t = k_t.repeat_interleave(kv_group, dim=1)
            v_t = v_t.repeat_interleave(kv_group, dim=1)

        out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
    return out


def simulate_phase3_compress_key_norm(
    k_full: torch.Tensor, v_full: torch.Tensor,
    real_pool_k: torch.Tensor, real_pool_v: torch.Tensor,
    real_slots: torch.Tensor,
    num_to_keep: int,
    timer: PhaseTimer,
):
    """Phase 3 (key_norm): importance estimation + scatter write."""
    with timer.phase("phase3"):
        # Sub-phase: importance estimation
        with timer.phase("phase3.importance"):
            k_norm = k_full.norm(dim=-1).mean(dim=-1)  # [seq_len]
            v_norm = v_full.norm(dim=-1).mean(dim=-1)
            importance = (k_norm + v_norm) / 2
            _, sorted_idx = torch.sort(importance, descending=True)
            keep_idx = sorted_idx[:num_to_keep]
            keep_idx = torch.sort(keep_idx)[0]

        # Sub-phase: scatter write to real pool
        with timer.phase("phase3.scatter_write"):
            real_pool_k[real_slots[:num_to_keep]] = k_full[keep_idx]
            real_pool_v[real_slots[:num_to_keep]] = v_full[keep_idx]

    return keep_idx


def simulate_phase3_compress_snapkv(
    k_full: torch.Tensor, v_full: torch.Tensor,
    q: torch.Tensor,
    real_pool_k: torch.Tensor, real_pool_v: torch.Tensor,
    real_slots: torch.Tensor,
    num_to_keep: int,
    num_heads: int, num_kv_heads: int,
    timer: PhaseTimer,
    window_size: int = 64,
):
    """Phase 3 (snapkv): Q@K^T attention-based importance + scatter write."""
    import math
    seq_len = k_full.shape[0]
    head_dim = k_full.shape[2]
    kv_group = num_heads // num_kv_heads
    window_size = min(window_size, seq_len)

    with timer.phase("phase3"):
        with timer.phase("phase3.importance"):
            # SnapKV sub-operations
            with timer.phase("phase3.importance.slice"):
                q_window = q[-window_size:]
                k_prefix = k_full[:-window_size] if seq_len > window_size else k_full[:0]

            if k_prefix.shape[0] == 0:
                keep_idx = torch.arange(seq_len, device=k_full.device)
                keep_idx = keep_idx.unsqueeze(0).expand(num_kv_heads, -1)
            else:
                with timer.phase("phase3.importance.transpose"):
                    q_t = q_window.transpose(0, 1).contiguous()
                    k_t = k_prefix.transpose(0, 1).contiguous()

                with timer.phase("phase3.importance.repeat_interleave"):
                    if kv_group > 1:
                        k_t = k_t.repeat_interleave(kv_group, dim=0)

                with timer.phase("phase3.importance.qk_matmul"):
                    attn = torch.matmul(q_t, k_t.transpose(-2, -1)) / math.sqrt(head_dim)

                with timer.phase("phase3.importance.sync"):
                    torch.cuda.current_stream().synchronize()

                with timer.phase("phase3.importance.softmax"):
                    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)

                with timer.phase("phase3.importance.pool_sum"):
                    attn_sum = attn.sum(dim=1)  # [num_heads, prefix_len]
                    if kv_group > 1:
                        attn_sum = attn_sum.view(num_kv_heads, kv_group, -1).sum(dim=1)

                    if attn_sum.shape[-1] > 5:
                        attn_cache = F.max_pool1d(
                            attn_sum.unsqueeze(0), kernel_size=5, padding=2, stride=1
                        ).squeeze(0)
                    else:
                        attn_cache = attn_sum

                with timer.phase("phase3.importance.topk"):
                    n_prefix_keep = max(1, num_to_keep - window_size)
                    _, indices = attn_cache.topk(n_prefix_keep, dim=-1)
                    indices = torch.sort(indices, dim=-1).values
                    win_idx = torch.arange(seq_len - window_size, seq_len, device=k_full.device)
                    win_idx = win_idx.unsqueeze(0).expand(num_kv_heads, -1)
                    keep_idx = torch.cat([indices, win_idx], dim=-1)

        with timer.phase("phase3.scatter_write"):
            for h in range(num_kv_heads):
                real_pool_k[real_slots[:keep_idx.shape[1]], h] = k_full[keep_idx[h], h]
                real_pool_v[real_slots[:keep_idx.shape[1]], h] = v_full[keep_idx[h], h]

    return keep_idx


# ── Main benchmark runner ────────────────────────────────────────────────────

def run_one(
    seq_len: int,
    method: str,
    compression_ratio: float,
    cpu_prefix_ratio: float,
    num_kv_heads: int = 8,
    num_heads: int = 32,
    head_dim: int = 128,
    warmup: int = 3,
    repeat: int = 10,
    device: str = "cuda",
) -> Dict[str, float]:
    """Run one configuration and return average timing dict."""
    dtype = torch.bfloat16
    prefix_len = int(seq_len * cpu_prefix_ratio)
    extend_len = seq_len - prefix_len
    num_to_keep = max(1, int(seq_len * (1 - compression_ratio)))
    pool_size = seq_len + 256

    gpool_k = torch.zeros(pool_size, num_kv_heads, head_dim, dtype=dtype, device=device)
    gpool_v = torch.zeros(pool_size, num_kv_heads, head_dim, dtype=dtype, device=device)
    global_slots = torch.arange(seq_len, device=device)

    rpool_k = torch.zeros(pool_size, num_kv_heads, head_dim, dtype=dtype, device=device)
    rpool_v = torch.zeros(pool_size, num_kv_heads, head_dim, dtype=dtype, device=device)
    real_slots = torch.arange(num_to_keep, device=device)

    k_cpu = torch.randn(prefix_len, num_kv_heads, head_dim, dtype=dtype, device="cpu")
    v_cpu = torch.randn(prefix_len, num_kv_heads, head_dim, dtype=dtype, device="cpu")

    k_extend = torch.randn(extend_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    v_extend = torch.randn(extend_len, num_kv_heads, head_dim, dtype=dtype, device=device)

    q = make_synthetic_q(extend_len if extend_len > 0 else seq_len, num_heads, head_dim, dtype, device)

    all_summaries = []
    for i in range(warmup + repeat):
        timer = PhaseTimer()

        k_full, v_full = simulate_phase1_assemble(
            k_cpu, v_cpu, k_extend, v_extend,
            gpool_k, gpool_v, global_slots, prefix_len, timer,
        )

        simulate_phase2_flashattn(
            q, k_full, v_full, num_heads, num_kv_heads, head_dim, timer,
        )

        if method == "snapkv":
            simulate_phase3_compress_snapkv(
                k_full, v_full, q, rpool_k, rpool_v, real_slots,
                num_to_keep, num_heads, num_kv_heads, timer,
            )
        else:
            simulate_phase3_compress_key_norm(
                k_full, v_full, rpool_k, rpool_v, real_slots,
                num_to_keep, timer,
            )

        s = timer.summary()
        if i >= warmup:
            all_summaries.append(s)

    avg = {}
    for key in all_summaries[0]:
        avg[key] = sum(s[key] for s in all_summaries) / len(all_summaries)
    return avg


def print_result(
    seq_len: int, method: str, ratio: float,
    cpu_prefix_ratio: float, avg: Dict[str, float],
):
    total = avg.get("phase1", 0) + avg.get("phase2", 0) + avg.get("phase3", 0)
    if total == 0:
        total = 1e-9

    print(f"\n{'─'*65}")
    print(f"seq_len={seq_len}, method={method}, ratio={ratio}, cpu_prefix={cpu_prefix_ratio:.0%}")
    print(f"{'─'*65}")

    for key in sorted(avg.keys()):
        ms = avg[key]
        pct = ms / total * 100
        depth = key.count(".")
        indent = "  " * depth
        if depth == 0:
            print(f"  {indent}{key:<35} {ms:>8.3f} ms  ({pct:>5.1f}%)")
        else:
            print(f"  {indent}{key:<35} {ms:>8.3f} ms")

    print(f"  {'TOTAL':<35} {total:>8.3f} ms")


def main():
    parser = argparse.ArgumentParser(description="KV compression phase breakdown benchmark")
    parser.add_argument("--seq-lens", default="1024,4096,8192",
                        help="Comma-separated sequence lengths")
    parser.add_argument("--methods", default="key_norm,snapkv",
                        help="Comma-separated compression methods")
    parser.add_argument("--ratios", default="0.5",
                        help="Comma-separated compression ratios")
    parser.add_argument("--cpu-prefix-ratio", type=float, default=0.8,
                        help="Fraction of seq_len from CPU prefix cache")
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()

    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    methods = args.methods.split(",")
    ratios = [float(x) for x in args.ratios.split(",")]

    print(f"Config: kv_heads={args.num_kv_heads}, q_heads={args.num_heads}, "
          f"head_dim={args.head_dim}, warmup={args.warmup}, repeat={args.repeat}")

    for seq_len in seq_lens:
        for method in methods:
            for ratio in ratios:
                avg = run_one(
                    seq_len, method, ratio, args.cpu_prefix_ratio,
                    args.num_kv_heads, args.num_heads, args.head_dim,
                    args.warmup, args.repeat,
                )
                print_result(seq_len, method, ratio, args.cpu_prefix_ratio, avg)


if __name__ == "__main__":
    main()
