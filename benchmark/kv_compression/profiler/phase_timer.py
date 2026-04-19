"""Per-phase CUDA event timer with nested sub-phase support."""

import time
from contextlib import contextmanager
from typing import Dict, List, Optional

import torch


class PhaseTimer:
    """Hierarchical phase timer using CUDA events for GPU-accurate timing.

    Usage:
        timer = PhaseTimer()
        with timer.phase("phase1"):
            with timer.phase("phase1.cpu_transfer"):
                ...
            with timer.phase("phase1.global_pool_io"):
                ...
        with timer.phase("phase2"):
            ...
        print(timer.summary())
    """

    def __init__(self):
        self._records: List[tuple] = []  # (name, start_event, end_event)
        self._use_cuda = torch.cuda.is_available()
        self._synced = False

    @contextmanager
    def phase(self, name: str):
        if self._use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self._records.append((name, start, end))
                self._synced = False
        else:
            t0 = time.perf_counter()
            try:
                yield
            finally:
                elapsed = (time.perf_counter() - t0) * 1000.0
                self._records.append((name, elapsed))
                self._synced = False

    def sync(self):
        if self._use_cuda and not self._synced:
            torch.cuda.synchronize()
            self._synced = True

    def summary(self) -> Dict[str, float]:
        """Return {phase_name: elapsed_ms}. Syncs GPU if needed."""
        self.sync()
        result = {}
        for rec in self._records:
            name = rec[0]
            if self._use_cuda:
                _, start, end = rec
                ms = start.elapsed_time(end)
            else:
                ms = rec[1]
            result[name] = result.get(name, 0.0) + ms
        return result

    def reset(self):
        self._records.clear()
        self._synced = False

    def report(self, indent: int = 2) -> str:
        """Pretty-print hierarchical timing report."""
        s = self.summary()
        if not s:
            return "(no phases recorded)"

        lines = []
        top_total = sum(v for k, v in s.items() if "." not in k)

        for name, ms in s.items():
            depth = name.count(".")
            prefix = " " * indent * depth
            pct = f" ({ms / top_total * 100:.1f}%)" if top_total > 0 else ""
            lines.append(f"{prefix}{name}: {ms:.3f} ms{pct}")
        return "\n".join(lines)
