"""GPU memory snapshot tracker for profiling allocation patterns."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class MemSnapshot:
    label: str
    allocated_mb: float
    reserved_mb: float
    peak_allocated_mb: float
    # delta from previous snapshot
    delta_allocated_mb: float = 0.0


class MemoryTracker:
    """Track GPU memory at labeled checkpoints.

    Usage:
        tracker = MemoryTracker()
        tracker.snapshot("before_phase1")
        ...
        tracker.snapshot("after_phase1")
        print(tracker.report())
    """

    def __init__(self, device: Optional[int] = None):
        self._device = device
        self._snapshots: List[MemSnapshot] = []
        self._available = torch.cuda.is_available()
        if self._available:
            torch.cuda.reset_peak_memory_stats(self._device)

    def snapshot(self, label: str):
        if not self._available:
            return
        torch.cuda.synchronize(self._device)
        alloc = torch.cuda.memory_allocated(self._device) / 1024**2
        reserved = torch.cuda.memory_reserved(self._device) / 1024**2
        peak = torch.cuda.max_memory_allocated(self._device) / 1024**2

        prev_alloc = self._snapshots[-1].allocated_mb if self._snapshots else 0.0
        self._snapshots.append(MemSnapshot(
            label=label,
            allocated_mb=alloc,
            reserved_mb=reserved,
            peak_allocated_mb=peak,
            delta_allocated_mb=alloc - prev_alloc,
        ))

    def reset(self):
        self._snapshots.clear()
        if self._available:
            torch.cuda.reset_peak_memory_stats(self._device)

    def report(self) -> str:
        if not self._snapshots:
            return "(no snapshots)"
        lines = [f"{'Label':<30} {'Alloc(MB)':>10} {'Delta(MB)':>10} {'Peak(MB)':>10} {'Rsv(MB)':>10}"]
        lines.append("-" * 75)
        for s in self._snapshots:
            sign = "+" if s.delta_allocated_mb >= 0 else ""
            lines.append(
                f"{s.label:<30} {s.allocated_mb:>10.1f} "
                f"{sign}{s.delta_allocated_mb:>9.1f} "
                f"{s.peak_allocated_mb:>10.1f} {s.reserved_mb:>10.1f}"
            )
        return "\n".join(lines)

    @property
    def snapshots(self) -> List[MemSnapshot]:
        return list(self._snapshots)
