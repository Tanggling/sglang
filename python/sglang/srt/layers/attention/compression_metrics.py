"""
KV Cache Compression & CPU Prefix Cache Metrics

Collects and reports metrics for:
  1. Compression: original length, compressed length, ratio per request
  2. CPU Prefix Cache: hit/miss counts, match lengths, partial-match ratios
  3. Timing: prefill, CPU→GPU transfer, compression, decode per step
  4. Per-request lifecycle: full prefill+decode timing for each request

Usage:
    from sglang.srt.layers.attention.compression_metrics import get_metrics

    metrics = get_metrics()
    metrics.log_compression(req_id, original_len=1000, compressed_len=500)
    metrics.log_cpu_cache_hit(req_id, match_len=500, entry_len=1000)
    metrics.log_decode_step([req_id_1, req_id_2], batch_ms=1.5)
    metrics.finalize_request(req_id)   # logs per-request summary
    metrics.report()      # print global summary
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


class LayerAccumTimer:
    """Non-blocking CUDA event timer that accumulates across transformer layers.

    Usage:
        timer = LayerAccumTimer()
        # In each layer:
        timer.mark_start()
        ...  # GPU work
        timer.mark_end()
        # After all layers:
        total_ms = timer.sync_and_total()

    Events are recorded without synchronization during forward passes.
    ``sync_and_total()`` synchronizes once at the end and sums up all
    (start, end) pairs to return the total elapsed milliseconds.
    """

    def __init__(self):
        self._pairs: List[tuple] = []  # list of (start_event, end_event)
        self._current_start = None
        self._use_cuda = torch.cuda.is_available()

    def mark_start(self):
        if self._use_cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._current_start = ev
        else:
            self._current_start = time.perf_counter()

    def mark_end(self):
        if self._current_start is None:
            return
        if self._use_cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._pairs.append((self._current_start, ev))
        else:
            elapsed = (time.perf_counter() - self._current_start) * 1000.0
            self._pairs.append(elapsed)
        self._current_start = None

    def sync_and_total(self) -> float:
        """Synchronize GPU and return total accumulated time in milliseconds."""
        if self._use_cuda:
            torch.cuda.synchronize()
            total = 0.0
            for start_ev, end_ev in self._pairs:
                total += start_ev.elapsed_time(end_ev)
            self._pairs.clear()
            return total
        else:
            total = sum(self._pairs)
            self._pairs.clear()
            return total


# ─── Per-request lifecycle tracker ───────────────────────────────────────────


@dataclass
class RequestMetrics:
    """Tracks the full lifecycle of a single request (prefill → decode → finish)."""

    req_id: int = -1

    # Compression info
    original_len: int = 0
    compressed_len: int = 0

    # CPU prefix cache
    cpu_cache_hit: bool = False
    cpu_match_len: int = 0
    cpu_entry_len: int = 0

    # Timing (ms) — all accumulated across layers
    prefill_ms: float = 0.0
    cpu_transfer_ms: float = 0.0
    compress_ms: float = 0.0
    compress_algo_ms: float = 0.0   # pure compression algorithm time
    kv_write_ms: float = 0.0        # KV write to real pool + GlobalKVPool free time
    finalize_ms: float = 0.0        # CPU prefix cache finalize (includes cuda sync)

    # Decode timing — accumulated across all decode steps
    decode_total_ms: float = 0.0
    decode_steps: int = 0


# ─── Aggregated counters ─────────────────────────────────────────────────────


@dataclass
class CompressionMetrics:
    """Aggregate metrics across requests, with per-request tracking."""

    # ── Compression stats ──
    total_requests: int = 0
    total_original_tokens: int = 0
    total_compressed_tokens: int = 0

    # ── CPU cache stats ──
    cpu_cache_hits: int = 0
    cpu_cache_misses: int = 0
    cpu_total_match_tokens: int = 0
    cpu_total_entry_tokens: int = 0

    # ── Timing accumulators (ms) — global ──
    total_prefill_ms: float = 0.0
    total_cpu_transfer_ms: float = 0.0
    total_compress_ms: float = 0.0
    total_compress_algo_ms: float = 0.0   # pure compression algorithm time
    total_kv_write_ms: float = 0.0        # KV write to real pool + GlobalKVPool free
    total_finalize_ms: float = 0.0        # CPU prefix cache finalize
    total_decode_ms: float = 0.0
    decode_steps: int = 0

    # ── Active per-request trackers (req_id → RequestMetrics) ──
    _active: Dict[int, RequestMetrics] = field(default_factory=dict)

    # ── Completed request history (last N for debugging) ──
    _history: List[RequestMetrics] = field(default_factory=list)
    _max_history: int = 256

    # ── Logging control ──
    _log_interval: int = 10
    _enabled: bool = True

    # ─── Internal: get or create active request ──────────────────────────

    def _get_active(self, req_id: int) -> RequestMetrics:
        if req_id not in self._active:
            self._active[req_id] = RequestMetrics(req_id=req_id)
        return self._active[req_id]

    # ─── Recording methods (prefill phase) ───────────────────────────────

    def log_compression(
        self, req_id: int, original_len: int, compressed_len: int
    ) -> None:
        if not self._enabled:
            return
        self.total_requests += 1
        self.total_original_tokens += original_len
        self.total_compressed_tokens += compressed_len

        rm = self._get_active(req_id)
        rm.original_len = original_len
        rm.compressed_len = compressed_len

    def log_cpu_cache_hit(
        self, req_id: int, match_len: int, entry_len: int
    ) -> None:
        if not self._enabled:
            return
        self.cpu_cache_hits += 1
        self.cpu_total_match_tokens += match_len
        self.cpu_total_entry_tokens += entry_len

        rm = self._get_active(req_id)
        rm.cpu_cache_hit = True
        rm.cpu_match_len = match_len
        rm.cpu_entry_len = entry_len

    def log_cpu_cache_miss(self, req_id: int) -> None:
        if not self._enabled:
            return
        self.cpu_cache_misses += 1
        self._get_active(req_id).cpu_cache_hit = False

    def log_prefill_time(self, req_id: int, ms: float) -> None:
        if not self._enabled:
            return
        self.total_prefill_ms += ms
        self._get_active(req_id).prefill_ms = ms

    def log_cpu_transfer_time(self, req_id: int, ms: float) -> None:
        if not self._enabled:
            return
        self.total_cpu_transfer_ms += ms
        self._get_active(req_id).cpu_transfer_ms += ms

    def log_compress_time(self, req_id: int, ms: float) -> None:
        if not self._enabled:
            return
        self.total_compress_ms += ms
        self._get_active(req_id).compress_ms += ms

    def log_compress_algo_time(self, req_id: int, ms: float) -> None:
        if not self._enabled:
            return
        self.total_compress_algo_ms += ms
        self._get_active(req_id).compress_algo_ms += ms

    def log_kv_write_time(self, req_id: int, ms: float) -> None:
        if not self._enabled:
            return
        self.total_kv_write_ms += ms
        self._get_active(req_id).kv_write_ms += ms

    def log_finalize_time(self, req_id: int, ms: float) -> None:
        if not self._enabled:
            return
        self.total_finalize_ms += ms
        self._get_active(req_id).finalize_ms += ms

    # ─── Recording methods (decode phase) ────────────────────────────────

    def log_decode_step(self, req_ids: List[int], batch_ms: float, add_step: bool = True) -> None:
        """Record one decode step for a batch of requests.

        The batch decode time is split equally among requests.
        Also updates global accumulators.
        """
        if not self._enabled:
            return
        self.total_decode_ms += batch_ms
        self.decode_steps += 1 if add_step else 0

        if not req_ids:
            return
        per_req_ms = batch_ms / len(req_ids)
        for rid in req_ids:
            rm = self._get_active(rid)
            rm.decode_total_ms += per_req_ms
            rm.decode_steps += 1 if add_step else 0

    # Backward-compatible: global-only decode time (no per-request)
    def log_decode_time(self, ms: float) -> None:
        if not self._enabled:
            return
        self.total_decode_ms += ms
        self.decode_steps += 1

    # ─── Per-request finalization ─────────────────────────────────────────

    def finalize_request(self, req_id: int) -> None:
        """Log per-request summary and move from active to history."""
        rm = self._active.pop(req_id, None)
        if rm is None:
            print(f"Warning: finalize_request called for unknown req_id {req_id}")
            return

        # Save to history
        self._history.append(rm)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        # Per-request log
        ratio = (
            rm.compressed_len / rm.original_len * 100
            if rm.original_len > 0 else 0.0
        )
        parts = [
            f"[KV Req#{rm.req_id}]",
            f"seq={rm.original_len}→{rm.compressed_len} ({ratio:.0f}%)",
        ]
        if rm.cpu_cache_hit:
            parts.append(f"cpu_hit={rm.cpu_match_len}tok")
        if rm.cpu_transfer_ms > 0:
            parts.append(f"transfer={rm.cpu_transfer_ms:.1f}ms")
        parts.append(f"prefill={rm.prefill_ms:.1f}ms")
        parts.append(f"compress={rm.compress_ms:.1f}ms")
        parts.append(f"compress_algo={rm.compress_algo_ms:.1f}ms")
        parts.append(f"kv_write={rm.kv_write_ms:.1f}ms")
        if rm.finalize_ms > 0:
            parts.append(f"finalize={rm.finalize_ms:.1f}ms")
        if rm.decode_steps > 0:
            avg_decode = rm.decode_total_ms / rm.decode_steps
            parts.append(
                f"decode={rm.decode_total_ms:.1f}ms "
                f"({rm.decode_steps}steps, {avg_decode:.2f}ms/tok)"
            )
        total = rm.prefill_ms + rm.compress_ms + rm.cpu_transfer_ms + rm.finalize_ms + rm.decode_total_ms
        parts.append(f"total={total:.1f}ms")
        logger.info(" | ".join(parts))

    # ─── Global summary reporting ─────────────────────────────────────────

    def log_global_summary(self) -> None:
        """Log one-line global average summary (called from backend)."""
        if not self._enabled or self.total_requests == 0:
            return
        avg_ratio = (
            self.total_compressed_tokens / self.total_original_tokens
            if self.total_original_tokens > 0 else 0.0
        )
        total_lookups = self.cpu_cache_hits + self.cpu_cache_misses
        hit_rate = self.cpu_cache_hits / total_lookups if total_lookups > 0 else 0.0

        _msg = (
            f"[KV Compress Global] reqs: {self.total_requests}, "
            f"avg ratio: {avg_ratio:.1%} kept, "
            f"cpu cache hit: {self.cpu_cache_hits}/{total_lookups} ({hit_rate:.0%})"
        )
        if self.cpu_cache_hits > 0:
            _msg += f", avg match: {self.cpu_total_match_tokens / self.cpu_cache_hits:.0f} tokens"
            _msg += f", avg transfer: {self.total_cpu_transfer_ms / self.cpu_cache_hits:.1f}ms"
        if self.total_requests > 0:
            _msg += f", avg prefill: {self.total_prefill_ms / self.total_requests:.1f}ms"
            _msg += f", avg compress: {self.total_compress_ms / self.total_requests:.1f}ms"
            _msg += f", avg compress_algo: {self.total_compress_algo_ms / self.total_requests:.1f}ms"
            _msg += f", avg kv_write: {self.total_kv_write_ms / self.total_requests:.1f}ms"
            if self.total_finalize_ms > 0:
                _msg += f", avg finalize: {self.total_finalize_ms / self.total_requests:.1f}ms"
        if self.decode_steps > 0:
            _msg += f", avg decode: {self.total_decode_ms / self.decode_steps:.2f}ms/tok"
        logger.info(_msg)

    def report(self) -> str:
        """Build a human-readable detailed summary string and log it."""
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
                lines.append(f"    Avg compress_algo: {self.total_compress_algo_ms / self.total_requests:.1f} ms/req")
                lines.append(f"    Avg kv_write:      {self.total_kv_write_ms / self.total_requests:.1f} ms/req")
                if self.total_finalize_ms > 0:
                    lines.append(f"    Avg finalize:      {self.total_finalize_ms / self.total_requests:.1f} ms/req")
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
        self.total_compress_algo_ms = 0.0
        self.total_kv_write_ms = 0.0
        self.total_finalize_ms = 0.0
        self.total_decode_ms = 0.0
        self.decode_steps = 0
        self._active.clear()
        self._history.clear()

    def get_last_snapshot(self, req_id: int) -> Optional[RequestMetrics]:
        for rm in reversed(self._history):
            if rm.req_id == req_id:
                return rm
        # Also check active
        return self._active.get(req_id)
