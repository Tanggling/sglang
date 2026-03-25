"""
GlobalKVPool: Single-layer temporary KV buffer for the global/real split design.

Design Philosophy:
    In the standard compressed KV cache design, all layers share the same token
    allocation indices. The GlobalKVPool is a single-layer temporary buffer that
    holds the FULL (uncompressed) KV for the current layer being processed.

    After each layer's attention computation:
        1. Full k, v are written to GlobalKVPool
        2. Compression is applied → keep_indices determined
        3. Compressed k, v are written to the real KV pool (MHATokenToKVPool)
        4. GlobalKVPool slots are freed for reuse by the next layer

    Memory requirement:
        GlobalKVPool: max_batch_tokens × num_kv_heads × head_dim  (ONE layer only)
        RealKVPool:   compressed_tokens × num_kv_heads × head_dim × num_layers

    This is significantly more memory-efficient than pre-allocating full KV for all
    layers upfront.
"""

from typing import Optional

import torch


class GlobalKVPool:
    """
    Single-layer temporary KV buffer shared across transformer layers during prefill.

    This buffer is reused at each layer:
      - Layer L writes full k,v to this buffer
      - Compression determines keep_indices
      - Compressed data moves to real_kv_pool
      - Slots are freed for Layer L+1

    The allocator is a simple free-list with no paging (page_size=1), since
    the buffer is reset after every layer and fragmentation is not a concern.
    """

    def __init__(
        self,
        max_tokens: int,
        num_kv_heads: int,
        head_dim: int,
        v_head_dim: Optional[int],
        dtype: torch.dtype,
        device: str,
    ):
        self.max_tokens = max_tokens
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim if v_head_dim is not None else head_dim
        self.dtype = dtype
        self.device = device

        # Slot 0 is reserved as a dummy slot (consistent with MHATokenToKVPool convention)
        self.k_buffer = torch.zeros(
            (max_tokens + 1, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self.v_buffer = torch.zeros(
            (max_tokens + 1, num_kv_heads, self.v_head_dim),
            dtype=dtype,
            device=device,
        )

        # Free-list allocator: slot indices [1, max_tokens]
        self.free_slots = torch.arange(1, max_tokens + 1, dtype=torch.int64, device=device)

    def alloc(self, n: int) -> Optional[torch.Tensor]:
        """
        Allocate n contiguous slots from the free list.

        Returns:
            Tensor of slot indices [n], or None if not enough space.
        """
        if n > len(self.free_slots):
            return None
        slots = self.free_slots[:n].clone()
        self.free_slots = self.free_slots[n:]
        return slots

    def free(self, slots: torch.Tensor) -> None:
        """Return slots to the free list."""
        if slots.numel() == 0:
            return
        self.free_slots = torch.cat([self.free_slots, slots])

    def available_size(self) -> int:
        """Number of free slots remaining."""
        return len(self.free_slots)

    def write_kv(
        self,
        slots: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """
        Write key-value tensors to specified slots.

        Args:
            slots: Slot indices [seq_len]
            k: Key tensor [seq_len, num_kv_heads, head_dim]
            v: Value tensor [seq_len, num_kv_heads, v_head_dim]
        """
        self.k_buffer[slots] = k.to(self.dtype)
        self.v_buffer[slots] = v.to(self.dtype)

    def read_kv(self, slots: torch.Tensor):
        """
        Read key-value tensors from specified slots.

        Args:
            slots: Slot indices [n]

        Returns:
            k: [n, num_kv_heads, head_dim]
            v: [n, num_kv_heads, v_head_dim]
        """
        return self.k_buffer[slots], self.v_buffer[slots]

    def reset(self) -> None:
        """
        Reset the pool to full capacity.

        Call this if you want to discard all current allocations (e.g., on error recovery).
        """
        self.free_slots = torch.arange(
            1, self.max_tokens + 1, dtype=torch.int64, device=self.device
        )

    def mem_usage_bytes(self) -> int:
        """Return approximate GPU memory used by this pool in bytes."""
        k_bytes = self.k_buffer.numel() * self.k_buffer.element_size()
        v_bytes = self.v_buffer.numel() * self.v_buffer.element_size()
        return k_bytes + v_bytes

    def __repr__(self) -> str:
        used = self.max_tokens - self.available_size()
        return (
            f"GlobalKVPool(max_tokens={self.max_tokens}, "
            f"used={used}, free={self.available_size()}, "
            f"num_kv_heads={self.num_kv_heads}, head_dim={self.head_dim}, "
            f"dtype={self.dtype}, device={self.device})"
        )
