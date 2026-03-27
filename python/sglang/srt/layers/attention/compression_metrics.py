"""
KV Cache Compression & CPU Prefix Cache Metrics

Collects and reports metrics for:
  1. Compression: original length, compressed length, ratio per request
  2. CPU Prefix Cache: hit/miss counts, match lengths, partial-match ratios
  3. Timing: prefill, CPU→GPU transfer, compression, decode per step

Usage:
    from sglang.srt.layers.attention.compression_metrics import get_metrics

    metrics = get_metrics()
    metrics.log_compression(req_id, original_len=1000, compressed_len=500)
    metrics.log_cpu_cache_hit(req_id, match_len=500, entry_len=1000)
    metrics.report()      # print summary
    metrics.reset()       # clear all stats
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

# ─── Singleton ────────────────────────────────────────────────────────────────

_global_metrics: Optional["CompressionMetrics"] = None


def get_metrics() -> "CompressionMetrics":
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = CompressionMetrics()
    return _global_metrics


# ─── Timer context manager ───────────────────────────────────────────────────


class CudaTimer:
    """Measure GPU-inclusive wall time using CUDA events."""

    def __init__(self, device: Optional[torch.device] = None):
        self._device = device
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record()
        else:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            self._end.record()
            torch.cuda.synchronize()
            self.elapsed_ms = self._start.elapsed_time(self._end)
        else:
            self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0


# ─── Per-request snapshot ─────────────────────────────────────────────────────


@dataclass
class RequestMetricSnapshot:
    """Metrics collected for a single request during one forward pass."""

    req_id: int = -1

    # Compression
    original_len: int = 0
    compressed_len: int = 0

    # CPU prefix cache
    cpu_cache_hit: bool = False
    cpu_match_len: int = 0  # actual matched prefix tokens
    cpu_entry_len: int = 0  # total tokens in the cached entry

    # Timing (ms)
    prefill_ms: float = 0.0  # FlashAttention extend forward
    cpu_transfer_ms: float = 0.0  # CPU→GPU KV transfer (all layers)
    compress_ms: float = 0.0  # importance estimation + write compressed
    decode_ms: float = 0.0  # single decode step


# ─── Aggregated counters ─────────────────────────────────────────────────────


@dataclass
class CompressionMetrics:
    """Aggregate metrics across requests."""

    # ── Compression stats ──
    total_requests: int = 0
    total_original_tokens: int = 0
    total_compressed_tokens: int = 0

    # ── CPU cache stats ──
    cpu_cache_hits: int = 0
    cpu_cache_misses: int = 0
    cpu_total_match_tokens: int = 0  # sum of match_len across hits
    cpu_total_entry_tokens: int = 0  # sum of entry_len across hits (for partial-match ratio)

    # ── Timing accumulators (ms) ──
    total_prefill_ms: float = 0.0
    total_cpu_transfer_ms: float = 0.0
    total_compress_ms: float = 0.0
    total_decode_ms: float = 0.0
    decode_steps: int = 0

    # ── Per-request history (last N for debugging) ──
    _history: List[RequestMetricSnapshot] = field(default_factory=list)
    _max_history: int = 128

    # ── Logging control ──
    _log_interval: int = 10  # log summary every N requests
    _enabled: bool = True

    # ─── Recording methods ────────────────────────────────────────────────

    def log_compression(
        self, req_id: int, original_len: int, compressed_len: int
    ) -> None:
        if not self._enabled:
            return
        self.total_requests += 1
        self.total_original_tokens += original_len
        self.total_compressed_tokens += compressed_len

        snap = self._get_or_create(req_id)
        snap.original_len = original_len
        snap.compressed_len = compressed_len

        if self.total_requests % self._log_interval == 0:
            self._log_summary()

    def log_cpu_cache_hit(
        self, req_id: int, match_len: int, entry_len: int
    ) -> None:
        if not self._enabled:
            return
        self.cpu_cache_hits += 1
        self.cpu_total_match_tokens += match_len
        self.cpu_total_entry_tokens += entry_len

        snap = self._get_or_create(req_id)
        snap.cpu_cache_hit = True
        snap.cpu_match_len = match_len
        snap.cpu_entry_len = entry_len

    def log_cpu_cache_miss(self, req_id: int) -> None:
        if not self._enabled:
            return
        self.cpu_cache_misses += 1

        snap = self._get_or_create(req_id)
        snap.cpu_cache_hit = False

    def log_prefill_time(self, req_id: int, ms: float) -> None:
        if not self._enabled:
            return
        self.total_prefill_ms += ms
        snap = self._get_or_create(req_id)
        snap.prefill_ms = ms

    def log_cpu_transfer_time(self, req_id: int, ms: float) -> None:
        if not self._enabled:
            return
        self.total_cpu_transfer_ms += ms
        snap = self._get_or_create(req_id)
        snap.cpu_transfer_ms += ms

    def log_compress_time(self, req_id: int, ms: float) -> None:
        if not self._enabled:
            return
        self.total_compress_ms += ms
        snap = self._get_or_create(req_id)
        snap.compress_ms += ms

    def log_decode_time(self, ms: float) -> None:
        if not self._enabled:
            return
        self.total_decode_ms += ms
        self.decode_steps += 1

    # ─── Reporting ────────────────────────────────────────────────────────

    def report(self) -> str:
        """Build a human-readable summary string and log it."""
        lines = ["=" * 60, "KV Compression Metrics Summary", "=" * 60]

        # Compression
        if self.total_requests > 0:
            avg_ratio = (
                self.total_compressed_tokens / self.total_original_tokens
                if self.total_original_tokens > 0
                else 0.0
            )
            lines.append(f"  Requests processed:    {self.total_requests}")
            lines.append(f"  Avg original length:   {self.total_original_tokens / self.total_requests:.0f}")
            lines.append(f"  Avg compressed length: {self.total_compressed_tokens / self.total_requests:.0f}")
            lines.append(f"  Avg compression ratio: {avg_ratio:.2%} (kept)")
            lines.append(f"  Total tokens saved:    {self.total_original_tokens - self.total_compressed_tokens}")

        # CPU cache
        total_lookups = self.cpu_cache_hits + self.cpu_cache_misses
        if total_lookups > 0:
            hit_rate = self.cpu_cache_hits / total_lookups
            lines.append("")
            lines.append(f"  CPU cache hits:   {self.cpu_cache_hits}/{total_lookups} ({hit_rate:.1%})")
            if self.cpu_cache_hits > 0:
                avg_match = self.cpu_total_match_tokens / self.cpu_cache_hits
                avg_entry = self.cpu_total_entry_tokens / self.cpu_cache_hits
                partial_ratio = self.cpu_total_match_tokens / self.cpu_total_entry_tokens if self.cpu_total_entry_tokens > 0 else 0
                lines.append(f"  Avg match length: {avg_match:.0f} / {avg_entry:.0f} entry tokens ({partial_ratio:.1%} utilization)")
                lines.append(f"  Avg prefill saved: {avg_match:.0f} tokens/hit")

        # Timing
        if self.total_prefill_ms > 0 or self.total_decode_ms > 0:
            lines.append("")
            lines.append("  Timing:")
            if self.total_requests > 0:
                lines.append(f"    Avg prefill:       {self.total_prefill_ms / self.total_requests:.1f} ms/req")
                lines.append(f"    Avg compression:   {self.total_compress_ms / self.total_requests:.1f} ms/req")
            if self.cpu_cache_hits > 0:
                lines.append(f"    Avg CPU transfer:  {self.total_cpu_transfer_ms / self.cpu_cache_hits:.1f} ms/hit")
            if self.decode_steps > 0:
                lines.append(f"    Avg decode step:   {self.total_decode_ms / self.decode_steps:.2f} ms/step")

        lines.append("=" * 60)
        text = "\n".join(lines)
        logger.info(text)
        return text

    def reset(self) -> None:
        self.total_requests = 0
        self.total_original_tokens = 0
        self.total_compressed_tokens = 0
        self.cpu_cache_hits = 0
        self.cpu_cache_misses = 0
        self.cpu_total_match_tokens = 0
        self.cpu_total_entry_tokens = 0
        self.total_prefill_ms = 0.0
        self.total_cpu_transfer_ms = 0.0
        self.total_compress_ms = 0.0
        self.total_decode_ms = 0.0
        self.decode_steps = 0
        self._history.clear()

    def get_last_snapshot(self, req_id: int) -> Optional[RequestMetricSnapshot]:
        for snap in reversed(self._history):
            if snap.req_id == req_id:
                return snap
        return None

    # ─── Internal ─────────────────────────────────────────────────────────

    def _get_or_create(self, req_id: int) -> RequestMetricSnapshot:
        for snap in reversed(self._history):
            if snap.req_id == req_id:
                return snap
        snap = RequestMetricSnapshot(req_id=req_id)
        self._history.append(snap)
        if len(self._history) > self._max_history:
            self._history.pop(0)
        return snap

    def _log_summary(self) -> None:
        """Auto-log summary at intervals."""
        if self.total_requests == 0:
            return
        avg_ratio = (
            self.total_compressed_tokens / self.total_original_tokens
            if self.total_original_tokens > 0
            else 0.0
        )
        total_lookups = self.cpu_cache_hits + self.cpu_cache_misses
        hit_rate = self.cpu_cache_hits / total_lookups if total_lookups > 0 else 0.0
        logger.info(
            f"[CompressionMetrics] reqs={self.total_requests} "
            f"avg_ratio={avg_ratio:.2%} "
            f"cpu_hit={hit_rate:.1%} "
            f"prefill={self.total_prefill_ms / self.total_requests:.1f}ms "
            f"compress={self.total_compress_ms / self.total_requests:.1f}ms"
        )
