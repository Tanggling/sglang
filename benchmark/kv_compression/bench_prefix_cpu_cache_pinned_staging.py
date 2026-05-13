"""
Benchmark PrefixCPUCache pinned-memory bottleneck and a staging-buffer alternative.

This compares the current pattern used by PrefixCPUCache:
    GPU tensor -> freshly allocated pinned CPU tensor kept in the cache

against an optimized long-term storage pattern:
    GPU tensor -> reused pinned staging tensor -> pageable CPU tensor kept in the cache

Run:
    python benchmark/kv_compression/bench_prefix_cpu_cache_pinned_staging.py
"""

import argparse
import json
import os
import subprocess
import sys
import time

import torch


def tensor_mib(shape, dtype):
    numel = 1
    for dim in shape:
        numel *= dim
    return numel * torch.empty((), dtype=dtype).element_size() / 1024**2


def sync():
    torch.cuda.synchronize()


def run_case(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    dtype = getattr(torch, args.dtype)
    shape = (args.num_tokens, args.num_kv_heads, args.head_dim)
    total_tensors = args.num_layers * args.tensors_per_layer
    device = f"cuda:{args.device}"

    # Warm CUDA context and pinned allocator with a tiny allocation.
    torch.empty((1,), dtype=dtype, device=device)
    torch.empty((1,), dtype=dtype, device="cpu", pin_memory=True)
    sync()
    torch.cuda.empty_cache()
    sync()

    src = torch.randn(shape, dtype=dtype, device=device)
    sync()

    prealloc_ms = 0.0
    t0 = time.perf_counter()

    if args.mode == "current_pinned_keep":
        # Current PrefixCPUCache-like behavior: allocate pinned destination for
        # every K/V/Q tensor, copy into it, and keep it alive as the cache entry.
        cache_tensors = []
        for _ in range(total_tensors):
            dst = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
            dst.copy_(src, non_blocking=True)
            cache_tensors.append(dst)
        sync()

    elif args.mode == "reused_pinned_pool_keep":
        # Best-case pinned pool: all pinned cache tensors are preallocated before
        # the measured request path. This removes alloc from the request path but
        # still keeps long-lived pinned cache storage.
        p0 = time.perf_counter()
        cache_tensors = [
            torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
            for _ in range(total_tensors)
        ]
        sync()
        prealloc_ms = (time.perf_counter() - p0) * 1000

        t0 = time.perf_counter()
        for dst in cache_tensors:
            dst.copy_(src, non_blocking=True)
        sync()

    elif args.mode == "staging_to_pageable_keep":
        # Optimized long-term storage: allocate one reusable pinned staging
        # tensor, but keep the actual cache entry in normal pageable CPU memory.
        staging = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
        cache_tensors = []
        sync()
        t0 = time.perf_counter()
        for _ in range(total_tensors):
            dst = torch.empty(shape, dtype=dtype, device="cpu")
            staging.copy_(src, non_blocking=True)
            sync()
            dst.copy_(staging)
            cache_tensors.append(dst)

    elif args.mode == "pageable_direct_keep":
        # Baseline: direct GPU->pageable CPU copies. This avoids pinned alloc, but
        # CUDA may use an internal staging path and the transfer is less async.
        cache_tensors = []
        for _ in range(total_tensors):
            dst = src.detach().to("cpu", non_blocking=True)
            cache_tensors.append(dst)
        sync()

    else:
        raise ValueError(f"unknown mode: {args.mode}")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    total_mib = tensor_mib(shape, dtype) * total_tensors
    result = {
        "mode": args.mode,
        "elapsed_ms": elapsed_ms,
        "prealloc_ms": prealloc_ms,
        "total_mib": total_mib,
        "effective_gib_s": (total_mib / 1024) / (elapsed_ms / 1000),
    }
    print(json.dumps(result), flush=True)


def run_isolated(script_path, base_args, mode):
    cmd = [
        sys.executable,
        script_path,
        "--child-mode",
        mode,
        "--num-tokens",
        str(base_args.num_tokens),
        "--num-kv-heads",
        str(base_args.num_kv_heads),
        "--head-dim",
        str(base_args.head_dim),
        "--num-layers",
        str(base_args.num_layers),
        "--tensors-per-layer",
        str(base_args.tensors_per_layer),
        "--dtype",
        base_args.dtype,
        "--device",
        str(base_args.device),
    ]
    env = os.environ.copy()
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def print_results(args, results):
    dtype = getattr(torch, args.dtype)
    shape = (args.num_tokens, args.num_kv_heads, args.head_dim)
    total_tensors = args.num_layers * args.tensors_per_layer
    one_mib = tensor_mib(shape, dtype)
    total_mib = one_mib * total_tensors

    print("=" * 88)
    print("PrefixCPUCache pinned-memory staging benchmark")
    print("=" * 88)
    print(
        f"shape={shape}, dtype={args.dtype}, layers={args.num_layers}, "
        f"tensors/layer={args.tensors_per_layer}, total_tensors={total_tensors}"
    )
    print(f"one tensor={one_mib:.1f} MiB, total cache payload={total_mib:.1f} MiB")
    print()
    print(f"{'mode':<32} {'request_ms':>12} {'prealloc_ms':>12} {'GiB/s':>10} {'speedup':>10}")
    print("-" * 88)

    baseline_ms = results["current_pinned_keep"]["elapsed_ms"]
    for mode in [
        "current_pinned_keep",
        "reused_pinned_pool_keep",
        "staging_to_pageable_keep",
        "pageable_direct_keep",
    ]:
        r = results[mode]
        speedup = baseline_ms / r["elapsed_ms"]
        print(
            f"{mode:<32} {r['elapsed_ms']:>12.1f} "
            f"{r['prealloc_ms']:>12.1f} {r['effective_gib_s']:>10.1f} {speedup:>9.2f}x"
        )

    print()
    print("Interpretation:")
    print("- current_pinned_keep matches the current cache behavior most closely.")
    print("- reused_pinned_pool_keep shows the best case if pinned cache tensors are preallocated.")
    print("- staging_to_pageable_keep keeps only one pinned tensor long term, but pays an extra CPU memcpy.")
    print("- pageable_direct_keep avoids explicit pinned allocation, but gives up reliable async DMA behavior.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=28693)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--tensors-per-layer", type=int, default=3)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--child-mode", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.child_mode is not None:
        args.mode = args.child_mode
        run_case(args)
        return

    script_path = os.path.abspath(__file__)
    modes = [
        "current_pinned_keep",
        "reused_pinned_pool_keep",
        "staging_to_pageable_keep",
        "pageable_direct_keep",
    ]
    results = {}
    for mode in modes:
        print(f"Running {mode}...", flush=True)
        results[mode] = run_isolated(script_path, args, mode)
    print_results(args, results)


if __name__ == "__main__":
    main()
