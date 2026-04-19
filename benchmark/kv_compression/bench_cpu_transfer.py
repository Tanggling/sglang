"""Benchmark: CPU→GPU transfer modes for prefix cache.

Compares pageable vs pinned memory, blocking vs non-blocking transfers.

Usage:
    python benchmark/kv_compression/bench_cpu_transfer.py
"""

import argparse
import torch
import time


def bench_transfer(
    num_tokens: int,
    num_kv_heads: int,
    head_dim: int,
    dtype=torch.bfloat16,
    pinned: bool = False,
    non_blocking: bool = False,
    warmup: int = 5,
    repeat: int = 20,
) -> float:
    """Returns average transfer time in ms."""
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

        k_gpu = k_cpu.to("cuda", non_blocking=non_blocking)
        v_gpu = v_cpu.to("cuda", non_blocking=non_blocking)
        if non_blocking:
            torch.cuda.synchronize()

        end.record()
        torch.cuda.synchronize()

        if i >= warmup:
            times.append(start.elapsed_time(end))

        del k_gpu, v_gpu

    return sum(times) / len(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-sizes", default="1024,4096,8192,16384",
                        help="Comma-separated token counts")
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    sizes = [int(x) for x in args.token_sizes.split(",")]
    cell_bytes = 2 * args.num_kv_heads * args.head_dim * 2  # K+V, bf16

    modes = [
        ("pageable+blocking",     False, False),
        ("pinned+blocking",       True,  False),
        ("pageable+non_blocking", False, True),
        ("pinned+non_blocking",   True,  True),
    ]

    print(f"{'tokens':>8} {'data_MB':>8}", end="")
    for name, _, _ in modes:
        print(f" {name:>22}", end="")
    print(f" {'speedup(pin/page)':>18}")
    print("─" * 100)

    for n_tok in sizes:
        data_mb = n_tok * cell_bytes / 1024**2
        results = {}
        for name, pinned, nb in modes:
            ms = bench_transfer(
                n_tok, args.num_kv_heads, args.head_dim,
                pinned=pinned, non_blocking=nb,
                warmup=args.warmup, repeat=args.repeat,
            )
            results[name] = ms

        print(f"{n_tok:>8} {data_mb:>8.1f}", end="")
        for name, _, _ in modes:
            bw = data_mb / results[name] * 1000 if results[name] > 0 else 0
            print(f" {results[name]:>12.3f}ms {bw:>6.1f}GB/s", end="")

        baseline = results["pageable+blocking"]
        best_pinned = results["pinned+non_blocking"]
        speedup = baseline / best_pinned if best_pinned > 0 else 0
        print(f" {speedup:>14.2f}x")


if __name__ == "__main__":
    main()
