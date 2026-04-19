"""Benchmark: Compression method comparison (key_norm vs snapkv).

Isolates the compressor.compress() call to measure pure algorithm cost
without pipeline overhead.

Usage:
    python benchmark/kv_compression/bench_compression_methods.py \
        --seq-lens 1024,4096,8192,16384
"""

import argparse
import math
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from benchmark.kv_compression.profiler.phase_timer import PhaseTimer
from benchmark.kv_compression.profiler.memory_tracker import MemoryTracker


def bench_key_norm(k, v, num_to_keep, timer):
    with timer.phase("key_norm"):
        with timer.phase("key_norm.norm"):
            k_norm = k.norm(dim=-1).mean(dim=-1)
            v_norm = v.norm(dim=-1).mean(dim=-1)
            importance = (k_norm + v_norm) / 2
        with timer.phase("key_norm.sort"):
            _, sorted_idx = torch.sort(importance, descending=True)
            keep = sorted_idx[:num_to_keep]
            keep = torch.sort(keep)[0]
        with timer.phase("key_norm.gather"):
            ck = k[keep]
            cv = v[keep]
    return ck, cv, keep


def bench_snapkv(k, v, q, num_to_keep, num_heads, num_kv_heads, timer, window_size=64):
    seq_len = k.shape[0]
    head_dim = k.shape[2]
    kv_group = num_heads // num_kv_heads
    window_size = min(window_size, seq_len)

    with timer.phase("snapkv"):
        with timer.phase("snapkv.slice"):
            q_w = q[-window_size:]
            k_pre = k[:-window_size] if seq_len > window_size else k[:0]

        if k_pre.shape[0] == 0:
            return k, v, torch.arange(seq_len, device=k.device)

        with timer.phase("snapkv.transpose"):
            q_t = q_w.transpose(0, 1).contiguous()
            k_t = k_pre.transpose(0, 1).contiguous()

        with timer.phase("snapkv.repeat_interleave"):
            if kv_group > 1:
                k_t = k_t.repeat_interleave(kv_group, dim=0)

        with timer.phase("snapkv.qk_matmul"):
            attn = torch.matmul(q_t, k_t.transpose(-2, -1)) / math.sqrt(head_dim)

        with timer.phase("snapkv.synchronize"):
            torch.cuda.current_stream().synchronize()

        with timer.phase("snapkv.softmax"):
            attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)

        with timer.phase("snapkv.pool_sum"):
            attn_sum = attn.sum(dim=1)
            if kv_group > 1:
                attn_sum = attn_sum.view(num_kv_heads, kv_group, -1).sum(dim=1)
            if attn_sum.shape[-1] > 5:
                attn_cache = F.max_pool1d(
                    attn_sum.unsqueeze(0), kernel_size=5, padding=2, stride=1
                ).squeeze(0)
            else:
                attn_cache = attn_sum

        with timer.phase("snapkv.topk"):
            n_keep = max(1, num_to_keep - window_size)
            _, indices = attn_cache.topk(n_keep, dim=-1)
            indices = torch.sort(indices, dim=-1).values
            win_idx = torch.arange(seq_len - window_size, seq_len, device=k.device)
            win_idx = win_idx.unsqueeze(0).expand(num_kv_heads, -1)
            keep = torch.cat([indices, win_idx], dim=-1)

    return None, None, keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", default="1024,4096,8192,16384")
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    dtype = torch.bfloat16
    device = "cuda"

    print(f"{'seq_len':>8} {'method':>10} {'total_ms':>10} {'breakdown':>50}")
    print("─" * 85)

    for seq_len in seq_lens:
        num_to_keep = max(1, int(seq_len * (1 - args.ratio)))
        k = torch.randn(seq_len, args.num_kv_heads, args.head_dim, dtype=dtype, device=device)
        v = torch.randn(seq_len, args.num_kv_heads, args.head_dim, dtype=dtype, device=device)
        q = torch.randn(seq_len, args.num_heads, args.head_dim, dtype=dtype, device=device)

        for method in ["key_norm", "snapkv"]:
            all_s = []
            for i in range(args.warmup + args.repeat):
                timer = PhaseTimer()
                mem = MemoryTracker()
                mem.snapshot("before")
                if method == "key_norm":
                    bench_key_norm(k, v, num_to_keep, timer)
                else:
                    bench_snapkv(k, v, q, num_to_keep, args.num_heads, args.num_kv_heads, timer)
                mem.snapshot("after")
                s = timer.summary()
                if i >= args.warmup:
                    all_s.append(s)

            avg = {}
            for key in all_s[0]:
                avg[key] = sum(d[key] for d in all_s) / len(all_s)

            total = avg.get(method, 0)
            subs = {k: v for k, v in avg.items() if k != method and k.startswith(method)}
            sub_str = ", ".join(f"{k.split('.')[-1]}={v:.3f}" for k, v in sorted(subs.items()))
            print(f"{seq_len:>8} {method:>10} {total:>10.3f} {sub_str:>50}")

        print()


if __name__ == "__main__":
    main()
