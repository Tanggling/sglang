"""
PrefixCPUCache: CPU-side KV cache for same-prefix reuse without radix cache.

Design Philosophy:
    When radix cache is disabled, repeated queries with the same prompt prefix
    would normally require a full prefill for each request. This module implements
    a CPU-side cache that:

    1. First request: run full prefill + compression. After prefill, save the
       FULL (pre-compression) KV for each layer to CPU RAM.

    2. Second+ requests with the same prefix: instead of running prefill,
       load full KV from CPU layer-by-layer into GPU KV pool, apply compression,
       and write compressed result to the KV pool. This reuses the expensive
       prefill computation.

    Trade-offs:
        + Avoids repeated prefill for common prefixes (large saving for long contexts)
        + CPU RAM is typically much larger than HBM → can store more prefixes
        - CPU→GPU transfer overhead per layer (amortized over many decode steps)
        - Compression must be deterministic across requests for same prefix
          (same keep_indices needed, ensured by using same q for compression)

    Note: This cache uses full (uncompressed) KV because:
        a) We need to run compression on the GPU side with the new request's query
        b) Different decode lengths → different query tokens → potentially different
           importance scores, so we recompute compression each time

    Key used: hash(tuple(prefix_token_ids))
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Environment variable switch: set USE_PINNED_MEMORY=0 to disable pinned memory (for A/B testing)
_USE_PINNED_MEMORY = os.environ.get("USE_PINNED_MEMORY", "1") != "0"

# ------------------------------------------------------------------ #
# Global registry – mirrors the get_global_server_args() pattern so  #
# that schedule_batch.prepare_for_extend can access the cache without #
# circular imports or dependency injection.                            #
# ------------------------------------------------------------------ #
_global_cpu_prefix_cache: Optional["PrefixCPUCache"] = None


def set_global_cpu_prefix_cache(cache: "PrefixCPUCache") -> None:
    """Register the singleton PrefixCPUCache used by this process."""
    global _global_cpu_prefix_cache
    _global_cpu_prefix_cache = cache


def get_global_cpu_prefix_cache() -> Optional["PrefixCPUCache"]:
    """Return the registered PrefixCPUCache, or None if not configured."""
    return _global_cpu_prefix_cache


@dataclass
class CPUKVEntry:
    """
    CPU-side storage for full (uncompressed) KV **and Q** data of a prefix.

    Fields:
        token_ids:  The prefix token IDs used as cache key (for collision detection)
        seq_len:    Number of prefix tokens
        kv_layers:  List of (k_cpu, v_cpu) per layer.
                    Each tensor shape: [seq_len, num_kv_heads, head_dim] on CPU
        q_layers:   List of q_cpu per layer.
                    Each tensor shape: [seq_len, num_heads, head_dim] on CPU.
                    Used to pad short queries during compression importance estimation.
        ref_count:  Number of active requests currently using this entry
    """

    token_ids: List[int]
    seq_len: int
    # kv_layers[layer_id] = (k_cpu_tensor, v_cpu_tensor)
    kv_layers: List[Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)
    # q_layers[layer_id] = q_cpu_tensor
    q_layers: List[torch.Tensor] = field(default_factory=list)
    ref_count: int = 0

    def mem_usage_bytes(self) -> int:
        """Approximate CPU RAM used by this entry."""
        total = 0
        for k_cpu, v_cpu in self.kv_layers:
            total += k_cpu.numel() * k_cpu.element_size()
            total += v_cpu.numel() * v_cpu.element_size()
        for q_cpu in self.q_layers:
            total += q_cpu.numel() * q_cpu.element_size()
        return total


class PrefixCPUCache:
    """
    CPU-side KV cache for same-prefix reuse (radix cache disabled mode).

    Usage:
        cache = PrefixCPUCache(max_entries=64)

        # During first request's prefill (call once per layer):
        cache.accumulate_layer_kqv(request_id, layer_id, k_gpu, v_gpu, q_gpu)

        # After all layers of first request's prefill complete:
        cache.finalize_entry(request_id, token_ids)

        # For second request with same prefix:
        entry = cache.lookup(token_ids)
        if entry is not None:
            # Load layer by layer, compress, write to real KV pool
            k_cpu, v_cpu = entry.kv_layers[layer_id]
            k_gpu = k_cpu.to(gpu_device)
            v_gpu = v_cpu.to(gpu_device)
            # ... apply compression ...
    """

    def __init__(self, max_entries: int = 64, max_total_bytes: Optional[int] = None):
        """
        Args:
            max_entries: Maximum number of prefix entries to cache.
            max_total_bytes: Maximum total CPU RAM to use (None = unlimited).
        """
        self.max_entries = max_entries
        self.max_total_bytes = max_total_bytes

        # Main cache: hash → CPUKVEntry
        self._cache: Dict[int, CPUKVEntry] = {}
        # LRU order (oldest first)
        self._lru_order: List[int] = []

        # Temporary accumulator for in-progress prefill
        # request_id → {layer_id: (k_cpu, v_cpu)}
        self._pending: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {}

        self._total_bytes = 0

    # ------------------------------------------------------------------ #
    # Building entries (during first prefill)                              #
    # ------------------------------------------------------------------ #

    def start_accumulating(self, request_id: int) -> None:
        """Begin accumulating KV data for a new prefix entry."""
        self._pending[request_id] = {}

    def accumulate_layer_kqv(
        self,
        request_id: int,
        layer_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
        q: torch.Tensor,
    ) -> None:
        """
        Save one layer's full K, V, and Q tensors to CPU during first prefill.

        Q is stored so that future CPU-hit requests with short queries can pad
        their Q with the stored Q to reach the compressor's window_size.

        Args:
            request_id: Unique identifier for the in-flight request.
            layer_id:   Transformer layer index.
            k:          Key tensor   [seq_len, num_kv_heads, head_dim] (GPU)
            v:          Value tensor [seq_len, num_kv_heads, v_head_dim] (GPU)
            q:          Query tensor [seq_len, num_heads, head_dim] (GPU)
        """
        if request_id not in self._pending:
            logger.warning(
                f"PrefixCPUCache: request_id={request_id} not started. "
                "Call start_accumulating() first."
            )
            return

        if _USE_PINNED_MEMORY:
            # Transfer to pinned CPU memory for faster future GPU transfers.
            # Pinned memory enables DMA-based CPU→GPU transfer (~22 GB/s vs ~11 GB/s pageable).
            k_cpu = torch.empty(k.shape, dtype=k.dtype, device="cpu", pin_memory=True)
            v_cpu = torch.empty(v.shape, dtype=v.dtype, device="cpu", pin_memory=True)
            q_cpu = torch.empty(q.shape, dtype=q.dtype, device="cpu", pin_memory=True)
            k_cpu.copy_(k, non_blocking=True)
            v_cpu.copy_(v, non_blocking=True)
            q_cpu.copy_(q, non_blocking=True)
        else:
            # Pageable memory (baseline): extra memcpy through staging buffer on transfer
            k_cpu = k.detach().to("cpu", non_blocking=True)
            v_cpu = v.detach().to("cpu", non_blocking=True)
            q_cpu = q.detach().to("cpu", non_blocking=True)
        self._pending[request_id][layer_id] = (k_cpu, v_cpu, q_cpu)

    def finalize_entry(self, request_id: int, token_ids) -> Optional[int]:
        """
        Finalize a pending entry and insert it into the cache.

        Call this after all layers of the first request's prefill have been
        accumulated.

        Args:
            request_id: The request ID used in accumulate_layer_kqv().
            token_ids:  Prefix token IDs (list or Tensor) used as cache key.

        Returns:
            The hash key if inserted successfully, None on failure.
        """
        if request_id not in self._pending:
            logger.warning(f"PrefixCPUCache: no pending data for request_id={request_id}")
            return None

        # Synchronize to ensure all CPU transfers are complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        pending_layers = self._pending.pop(request_id)
        if not pending_layers:
            return None

        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()

        h = self._hash_tokens(token_ids)

        # Build kv_layers and q_layers lists in sorted order
        num_layers = max(pending_layers.keys()) + 1
        kv_layers = []
        q_layers = []
        for layer_id in range(num_layers):
            if layer_id not in pending_layers:
                logger.error(f"PrefixCPUCache: missing layer {layer_id} for request {request_id}")
                return None
            k_cpu, v_cpu, q_cpu = pending_layers[layer_id]
            kv_layers.append((k_cpu, v_cpu))
            q_layers.append(q_cpu)

        entry = CPUKVEntry(
            token_ids=token_ids,
            seq_len=len(token_ids),
            kv_layers=kv_layers,
            q_layers=q_layers,
        )
        entry_bytes = entry.mem_usage_bytes()

        # Evict if over capacity
        self._evict_if_needed(entry_bytes)

        self._cache[h] = entry
        self._total_bytes += entry_bytes

        if h in self._lru_order:
            self._lru_order.remove(h)
        self._lru_order.append(h)

        logger.info(
            f"PrefixCPUCache: stored entry for {len(token_ids)} tokens, "
            f"{entry_bytes / 1024**2:.1f} MB, total={self._total_bytes / 1024**2:.1f} MB"
        )
        return h

    def cancel_accumulating(self, request_id: int) -> None:
        """Discard pending data for a request (e.g., on error)."""
        self._pending.pop(request_id, None)

    # ------------------------------------------------------------------ #
    # Lookup (for second+ requests)                                        #
    # ------------------------------------------------------------------ #

    def lookup(self, token_ids) -> Optional[CPUKVEntry]:
        """
        Look up a cached prefix entry by exact token IDs.

        Returns:
            CPUKVEntry if found and token_ids match (collision check), else None.
        """
        h = self._hash_tokens(token_ids)
        entry = self._cache.get(h, None)
        if entry is None:
            return None

        # Collision check: verify actual token IDs match
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()
        if entry.token_ids != token_ids:
            logger.warning("PrefixCPUCache: hash collision detected, ignoring entry")
            return None

        # Move to most-recently-used position
        if h in self._lru_order:
            self._lru_order.remove(h)
        self._lru_order.append(h)

        return entry

    def lookup_prefix(
        self, token_ids, min_match_len: int = 32
    ) -> Optional[Tuple[int, CPUKVEntry]]:
        """
        Find the cached entry that shares the longest common prefix with
        ``token_ids``.

        Unlike exact-prefix matching, this supports **partial reuse**: if a
        cached entry has 1000 tokens but only the first 500 match the current
        request, the 500-token common prefix is still returned so its KV can
        be loaded from CPU (saving 500 tokens of GPU prefill).

        Args:
            token_ids:     Full token IDs for the current request.
            min_match_len: Minimum number of matching tokens to consider a hit.
                           Very short matches save little compute and are not
                           worth the CPU→GPU transfer overhead.

        Returns:
            (match_len, entry) for the best matching entry, or None.
            ``match_len`` may be less than ``entry.seq_len`` (partial match).
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()

        best_len = 0
        best_entry = None
        best_hash = None

        for h, entry in self._cache.items():
            # Compute common prefix length between token_ids and entry.
            # Leave at least 1 token as extend so the model can produce logits.
            common = 0
            '''
                KEY CODE
            '''
            limit = min(len(token_ids) - 1, entry.seq_len)
            for i in range(limit):
                if token_ids[i] != entry.token_ids[i]:
                    break
                common = i + 1

            if common < min_match_len:
                continue
            if common <= best_len:
                continue

            best_len = common
            best_entry = entry
            best_hash = h

        if best_entry is None:
            return None

        # LRU update
        if best_hash in self._lru_order:
            self._lru_order.remove(best_hash)
        self._lru_order.append(best_hash)

        return best_len, best_entry

    def has(self, token_ids) -> bool:
        """Check if a prefix is cached (without LRU update)."""
        h = self._hash_tokens(token_ids)
        if h not in self._cache:
            return False
        # Quick token_ids verification
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()
        return self._cache[h].token_ids == token_ids

    def load_layer_to_gpu(
        self,
        entry: CPUKVEntry,
        layer_id: int,
        device: str,
        num_tokens: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transfer one layer's KV data from CPU to GPU.

        Args:
            entry:      CPUKVEntry returned by lookup()
            layer_id:   Transformer layer index
            device:     Target GPU device string (e.g., "cuda:0")
            num_tokens: If set, only transfer the first ``num_tokens`` tokens
                        (for partial prefix matches where match_len < entry.seq_len).

        Returns:
            k_gpu, v_gpu tensors on the specified device
        """
        k_cpu, v_cpu = entry.kv_layers[layer_id]
        if num_tokens is not None and num_tokens < k_cpu.shape[0]:
            k_cpu = k_cpu[:num_tokens]
            v_cpu = v_cpu[:num_tokens]
        # Use non_blocking=True since source tensors are in pinned memory
        k_gpu = k_cpu.to(device, non_blocking=True)
        v_gpu = v_cpu.to(device, non_blocking=True)
        return k_gpu, v_gpu

    def load_query_to_gpu(
        self,
        entry: CPUKVEntry,
        layer_id: int,
        device: str,
        start: int = 0,
        end: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Load stored Q from CPU to GPU for a given layer.

        Args:
            entry:    CPUKVEntry returned by lookup()
            layer_id: Transformer layer index
            device:   Target GPU device string
            start:    Start token index (inclusive)
            end:      End token index (exclusive), None = entry.seq_len

        Returns:
            q_gpu tensor [end-start, num_heads, head_dim] on device
        """
        q_cpu = entry.q_layers[layer_id]
        if end is None:
            end = q_cpu.shape[0]
        q_slice = q_cpu[start:end]
        return q_slice.to(device, non_blocking=False)

    # ------------------------------------------------------------------ #
    # Eviction                                                             #
    # ------------------------------------------------------------------ #

    def _evict_if_needed(self, incoming_bytes: int) -> None:
        """Evict LRU entries to make room for a new entry."""
        # Evict if over max_entries
        while len(self._cache) >= self.max_entries and self._lru_order:
            self._evict_lru()

        # Evict if over byte budget
        if self.max_total_bytes is not None:
            while (
                self._total_bytes + incoming_bytes > self.max_total_bytes
                and self._lru_order
            ):
                self._evict_lru()

    def _evict_lru(self) -> None:
        if not self._lru_order:
            return
        oldest_hash = self._lru_order.pop(0)
        entry = self._cache.pop(oldest_hash, None)
        if entry is not None:
            self._total_bytes -= entry.mem_usage_bytes()
            logger.debug(
                f"PrefixCPUCache: evicted entry with {entry.seq_len} tokens"
            )

    def evict(self, token_ids) -> bool:
        """Manually evict a specific prefix. Returns True if it was present."""
        h = self._hash_tokens(token_ids)
        entry = self._cache.pop(h, None)
        if entry is not None:
            self._total_bytes -= entry.mem_usage_bytes()
            if h in self._lru_order:
                self._lru_order.remove(h)
            return True
        return False

    # ------------------------------------------------------------------ #
    # Utilities                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _hash_tokens(token_ids) -> int:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()
        return hash(tuple(token_ids))

    def __len__(self) -> int:
        return len(self._cache)

    def mem_usage_bytes(self) -> int:
        return self._total_bytes

    def __repr__(self) -> str:
        return (
            f"PrefixCPUCache(entries={len(self._cache)}/{self.max_entries}, "
            f"mem={self._total_bytes / 1024**2:.1f} MB)"
        )
