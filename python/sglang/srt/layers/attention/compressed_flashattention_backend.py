"""
Compressed FlashAttention Backend for SGLang

This backend extends FlashAttentionBackend to support KV cache compression
during the prefill (extend) phase.

Key Features:
1. Uses FlashAttention for efficient attention computation
2. Computes full attention output first (no quality loss)
3. Estimates token importance using LSE or key norms
4. Applies compression algorithm to select important KV entries
5. Releases unused KV cache slots after attention computation
6. Updates req_to_token mapping for compressed sequences
7. Works with radix cache disabled mode

Importance Estimation Methods:
- "lse": Use FlashAttention's softmax LSE (fast, approximate)
- "key_norm": Use key vector norms (fast, no attention needed)
- "snapkv": Compute full attention scores (slow, accurate)

CPU Prefix Cache Mode (Feature):
    When `cpu_prefix_cache` is provided, the backend:
    - On first request: saves full KV per layer to CPU after prefill
    - On second+ request with same prefix: loads from CPU, compresses, skips prefill
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Literal
import logging

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend, flash_attn_varlen_func
from sglang.srt.layers.attention.compression_metrics import (
    get_metrics,
    CudaTimer,
    LayerAccumTimer,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.kv_compressor import CompressionConfig
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.mem_cache.prefix_cpu_cache import PrefixCPUCache
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator


ImportanceMethod = Literal["lse", "key_norm", "snapkv"]
CompressionScheme = Literal["standard", "single_layer_zero_out", "global_real_split"]


class CompressedFlashAttentionBackend(FlashAttentionBackend):
    """
    FlashAttention backend with KV cache compression support.

    This backend first computes full attention output, then compresses KV cache
    after each attention layer's computation during the prefill phase, immediately
    releasing unused memory. The returned output is from full attention computation.

    Importance Estimation Methods:
        - "lse": Use FlashAttention's softmax LSE (fast, memory efficient)
        - "key_norm": Use key vector norms (fastest, no extra computation)
        - "snapkv": Compute full attention scores (accurate but slow)

    Compression Schemes:
        - "standard": Standard compression with memory release and mapping update
        - "single_layer_zero_out": Only compress first layer, zero out non-kept tokens
          without releasing memory or updating mappings
        - "global_real_split": Use separate global (full) and real (compressed) KV pools.
          Allocation uses compressed length. Supports CPU prefix cache (Feature 2).

    Usage:
        from sglang.srt.layers.attention.kv_compressor import CompressionConfig

        compression_config = CompressionConfig(
            enabled=True,
            compression_ratio=0.5,
            compression_method="importance",
            window_size=64,
        )

        self.attn_backend = CompressedFlashAttentionBackend(
            runner,
            compression_config=compression_config,
            importance_method="key_norm",  # or "lse" or "snapkv"
            compression_scheme="standard",  # or "single_layer_zero_out" or "global_real_split"
        )
    """

    def __init__(
        self,
        runner,
        compression_config: Optional["CompressionConfig"] = None,
        fa_impl_ver: int = 3,
        importance_method: ImportanceMethod = "snapkv",
        compression_scheme: CompressionScheme = "standard",
        # Feature 2: CPU prefix cache (if None, auto-created)
        cpu_prefix_cache: Optional["PrefixCPUCache"] = None,
    ):
        super().__init__(runner, fa_impl_ver=fa_impl_ver)

        from sglang.srt.layers.attention.kv_compressor import (
            CompressionConfig,
            create_compressor,
        )

        from sglang.srt.server_args import ServerArgs, get_global_server_args
        server_args = get_global_server_args()
        kv_compression_ratio = server_args.kv_compression_ratio
        self.compression_config = compression_config or CompressionConfig(compression_ratio=kv_compression_ratio)
        self.compressor = create_compressor(self.compression_config)
        self.importance_method = importance_method
        self.compression_scheme = compression_scheme

        # Store reference to token_to_kv_pool_allocator from runner
        self.token_to_kv_pool_allocator = runner.token_to_kv_pool_allocator

        # Store references to k_buffer and v_buffer for each layer
        # These are used to move KV cache data after compression
        self.k_buffer = runner.token_to_kv_pool.k_buffer
        self.v_buffer = runner.token_to_kv_pool.v_buffer

        # ------------------------------------------------------------------ #
        # GlobalKVPool for global/real split design
        # ------------------------------------------------------------------ #
        self.global_kv_pool = getattr(runner, 'global_kv_pool', None)
        if self.global_kv_pool is not None:
            logger.info(
                f"[KV Compress] Using GlobalKVPool with max_tokens={self.global_kv_pool.max_tokens}"
            )

        # ------------------------------------------------------------------ #
        # Feature 2: CPU Prefix Cache (default: enabled)
        # ------------------------------------------------------------------ #
        if cpu_prefix_cache is not None:
            self.cpu_prefix_cache = cpu_prefix_cache
        else:
            from sglang.srt.mem_cache.prefix_cpu_cache import (
                PrefixCPUCache,
                get_global_cpu_prefix_cache,
                set_global_cpu_prefix_cache,
            )
            existing = get_global_cpu_prefix_cache()
            if existing is not None:
                self.cpu_prefix_cache = existing
            else:
                self.cpu_prefix_cache = PrefixCPUCache(max_entries=64)
                set_global_cpu_prefix_cache(self.cpu_prefix_cache)
                logger.info("[KV Compress] Created global PrefixCPUCache (max_entries=64)")
        self.save_prefix_to_cpu: bool = True

        self._compression_stats = {
            "total_compressed": 0,
            "total_freed": 0,
            "layer_stats": {},
        }

        self._verification_enabled = False  # Set to True to enable KV cache verification (debugging)

    # ================================================================== #
    # Main forward_extend entry point
    # ================================================================== #

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for extend mode with KV cache compression.

        Dispatches to either:
          - global_real_split flow (Feature 1): global buffer → compress → real buffer
          - standard flow: write to pool, then compress in-place

        In both cases the FULL attention output is returned (no quality loss from compression).
        """
        if save_kv_cache:
            # print(f"q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")
            return self._forward_extend_global_real_split(
                q, k, v, layer, forward_batch, **kwargs
            )

        # ---- Legacy standard flow ----
        output = super().forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

        layer_id = layer.layer_id
        total_tokens = q.shape[0]
        should_compress = self.compressor.should_compress(layer_id, total_tokens)

        if should_compress and save_kv_cache:
            self._compress_kv_cache_after_attention(
                q, k, v, layer, forward_batch, **kwargs
            )
        return output

    # ================================================================== #
    # Feature 1: Global/Real split forward path
    # ================================================================== #

    def _forward_extend_global_real_split(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        **kwargs,
    ) -> torch.Tensor:
        """
        Global/Real split prefill with GlobalKVPool support.

        Flow per layer:
          1. Assemble full KV in GlobalKVPool (CPU prefix + model extend)
          2. FlashAttention with full KV → correct output (no quality loss)
          3. Compress: select important tokens from full KV
          4. Write compressed KV to real pool slots
          5. Free GlobalKVPool slots (reused by next layer)

        Memory savings:
          GlobalKVPool: 1 layer × max_batch_tokens (shared across layers)
          Real pool:    num_layers × compressed_tokens (permanent for decode)

        Layers 1+ reuse ``real_slots`` determined at layer 0.
        """
        layer_id = layer.layer_id
        num_heads = layer.tp_q_head_num
        num_kv_heads = layer.tp_k_head_num
        head_dim = layer.head_dim
        num_layers = len(self.k_buffer)
        device = k.device

        metadata = self.forward_metadata

        do_cpu = self.cpu_prefix_cache is not None

        # Get GlobalKVPool from runner
        global_kv_pool = self.global_kv_pool
        if global_kv_pool is None:
            # Fallback to legacy behavior if GlobalKVPool is not available
            return self._forward_extend_global_real_split_legacy(
                q, k, v, layer, forward_batch, **kwargs
            )

        # ── Layer-0 initialisation ─────────────────────────────────────────
        if layer_id == 0:
            forward_batch._gr_real_slots: Dict[int, torch.Tensor] = {}
            forward_batch._gr_compressed_lens: List[int] = []
            forward_batch._gr_slots_to_free: List[torch.Tensor] = []
            forward_batch._gr_cpu_prefix_lens: Dict[int, int] = {}
            # Timers that accumulate across layers (sync only at last layer)
            forward_batch._timer_prefill = LayerAccumTimer()
            forward_batch._timer_compress = LayerAccumTimer()
            forward_batch._timer_compress_algo = LayerAccumTimer()
            forward_batch._timer_kv_write = LayerAccumTimer()
            forward_batch._timer_transfer = LayerAccumTimer()

            forward_batch._cpu_miss_set: set = set()
            _metrics = get_metrics()
            if do_cpu:
                reqs = getattr(forward_batch, "reqs", None)
                if reqs is not None:
                    req_pool_indices_now = forward_batch.req_pool_indices
                    batch_size_now = metadata.cu_seqlens_q.shape[0] - 1
                    for _si in range(batch_size_now):
                        _req = reqs[_si]
                        _req_id = req_pool_indices_now[_si].item()
                        _match_len = getattr(_req, "cpu_match_len", 0)
                        if _match_len > 0:
                            forward_batch._gr_cpu_prefix_lens[_si] = _match_len
                            _entry = _req.cpu_prefix_entry
                            _metrics.log_cpu_cache_hit(
                                _req_id, match_len=_match_len,
                                entry_len=_entry.seq_len,
                            )
                        else:
                            self.cpu_prefix_cache.start_accumulating(_req_id)
                            forward_batch._cpu_miss_set.add(_si)
                            _metrics.log_cpu_cache_miss(_req_id)

        cu_seqlens_q = metadata.cu_seqlens_q
        batch_size = cu_seqlens_q.shape[0] - 1
        req_pool_indices = forward_batch.req_pool_indices

        # Get compressed total lengths from schedule_batch
        compressed_total_lens_cpu = getattr(forward_batch, "compressed_total_lens_cpu", None)

        # Build cu_seqlens for compressed slots (to index into out_cache_loc)
        cu_seqlens_compressed = [0]
        if compressed_total_lens_cpu is not None:
            for _cl in compressed_total_lens_cpu:
                cu_seqlens_compressed.append(cu_seqlens_compressed[-1] + _cl)

        real_cache_loc = forward_batch.out_cache_loc  # Pre-allocated compressed total length
        cpu_prefix_lens = getattr(forward_batch, "_gr_cpu_prefix_lens", {})
        cpu_miss_set = getattr(forward_batch, "_cpu_miss_set", set())
        reqs = getattr(forward_batch, "reqs", None)
        _metrics = get_metrics()

        q_view = q.view(-1, num_heads, head_dim)
        all_compressed_lens: List[int] = getattr(forward_batch, "_gr_compressed_lens", [])

        # Compressor window size for Q padding
        _window_size = getattr(self.compression_config, 'window_size', 64)

        # ── PHASE 1: Assemble full KV in GlobalKVPool ──────────────────────
        all_k_for_attn = []
        all_v_for_attn = []
        all_cu_seqlens_k = [0]
        max_seqlen_k = 0

        # Per-sequence data for phase 3 (compression)
        seq_data = []  # list of (k_full, v_full, full_seq_len, global_slots, real_slots_seq, ...)

        for seq_idx in range(batch_size):
            q_start = cu_seqlens_q[seq_idx].item()
            q_end = cu_seqlens_q[seq_idx + 1].item()
            extend_len = q_end - q_start

            req_id = req_pool_indices[seq_idx].item()
            _prefix_len = cpu_prefix_lens.get(seq_idx, 0)

            # Get real_slots for this sequence based on compressed_total_lens
            if compressed_total_lens_cpu is not None:
                comp_start = cu_seqlens_compressed[seq_idx]
                comp_end = cu_seqlens_compressed[seq_idx + 1]
                real_slots_seq = real_cache_loc[comp_start:comp_end]
            else:
                real_slots_seq = None

            full_seq_len = _prefix_len + extend_len

            if full_seq_len == 0:
                if layer_id == 0:
                    # Use None (not 0) so process_batch_result_prefill skips
                    # this sequence.  Using 0 would reset kv_committed_len for
                    # decode requests in mixed batches, leaking all their slots.
                    all_compressed_lens.append(None)
                all_cu_seqlens_k.append(all_cu_seqlens_k[-1])
                seq_data.append(None)
                continue

            # ── Allocate GlobalKVPool slots for full KV ────────────────────
            global_slots = global_kv_pool.alloc(full_seq_len)
            if global_slots is None:
                raise RuntimeError(
                    f"GlobalKVPool out of memory. Requested {full_seq_len} tokens, "
                    f"available {global_kv_pool.available_size()}"
                )

            # ── Load CPU prefix KV into GlobalKVPool ──────────────────────
            if _prefix_len > 0:
                forward_batch._timer_transfer.mark_start()
                _req = reqs[seq_idx]
                _entry = _req.cpu_prefix_entry
                _k_doc, _v_doc = self.cpu_prefix_cache.load_layer_to_gpu(
                    _entry, layer_id, str(device), num_tokens=_prefix_len
                )
                global_kv_pool.write_kv(global_slots[:_prefix_len], _k_doc, _v_doc)
                forward_batch._timer_transfer.mark_end()

            # ── Write extend KV to GlobalKVPool ───────────────────────────
            if extend_len > 0:
                k_extend = k[q_start:q_end]
                v_extend = v[q_start:q_end]
                global_kv_pool.write_kv(
                    global_slots[_prefix_len:_prefix_len + extend_len],
                    k_extend, v_extend
                )

            # ── Read full KV for FA and later compression ─────────────────
            k_full, v_full = global_kv_pool.read_kv(global_slots)
            all_k_for_attn.append(k_full)
            all_v_for_attn.append(v_full)
            all_cu_seqlens_k.append(all_cu_seqlens_k[-1] + full_seq_len)
            max_seqlen_k = max(max_seqlen_k, full_seq_len)

            # Save full KQV to CPU for miss sequences
            if do_cpu and seq_idx in cpu_miss_set:
                q_seq_for_save = q_view[q_start:q_end]
                self.cpu_prefix_cache.accumulate_layer_kqv(
                    req_id, layer_id, k_full, v_full, q_seq_for_save
                )

            seq_data.append((k_full, v_full, full_seq_len, global_slots, real_slots_seq))

        # ── PHASE 2: FlashAttention with full KV ──────────────────────────
        _timer_prefill = forward_batch._timer_prefill
        _timer_prefill.mark_start()
        if len(all_k_for_attn) > 0:
            k_all = torch.cat(all_k_for_attn, dim=0)
            v_all = torch.cat(all_v_for_attn, dim=0)
            cu_seqlens_k_tensor = torch.tensor(
                all_cu_seqlens_k, dtype=torch.int32, device=device
            )

            output = flash_attn_varlen_func(
                q=q_view,
                k=k_all,
                v=v_all,
                cu_seqlens_q=cu_seqlens_q.to(torch.int32),
                cu_seqlens_k=cu_seqlens_k_tensor,
                max_seqlen_q=metadata.max_seq_len_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=layer.scaling,
                causal=True,
            )
            # Reshape to [total_q, num_heads * v_head_dim] to match parent's return format
            output = output.view(-1, num_heads * layer.v_head_dim)
        else:
            output = q.new_zeros(q.shape[0], num_heads * layer.v_head_dim)
        _timer_prefill.mark_end()

        # Free intermediate FA tensors early
        del all_k_for_attn, all_v_for_attn

        # ── PHASE 3: Per-sequence compression and write to real pool ──────
        _timer_compress = forward_batch._timer_compress
        _timer_compress.mark_start()
        for seq_idx in range(batch_size):
            if seq_data[seq_idx] is None:
                continue

            k_full, v_full, full_seq_len, global_slots, real_slots_seq = seq_data[seq_idx]
            q_start = cu_seqlens_q[seq_idx].item()
            q_end = cu_seqlens_q[seq_idx + 1].item()
            extend_len = q_end - q_start
            req_id = req_pool_indices[seq_idx].item()
            _prefix_len = cpu_prefix_lens.get(seq_idx, 0)

            # ── Build Q for compression importance estimation ─────────────
            if _prefix_len > 0 and extend_len < _window_size:
                _req = reqs[seq_idx]
                _entry = _req.cpu_prefix_entry
                _need_from_cpu = min(_window_size - extend_len, _prefix_len)
                _q_pad = self.cpu_prefix_cache.load_query_to_gpu(
                    _entry, layer_id, str(device),
                    start=_prefix_len - _need_from_cpu,
                    end=_prefix_len,
                )
                if extend_len > 0:
                    q_for_compress = torch.cat([_q_pad, q_view[q_start:q_end]], dim=0)
                else:
                    q_for_compress = _q_pad
            else:
                q_for_compress = q_view[q_start:q_end]

            # ── Compress: select important tokens from the full sequence ──
            forward_batch._timer_compress_algo.mark_start()
            _, _, keep_indices = self._estimate_importance(
                method=self.importance_method,
                q=q_for_compress,
                k=k_full,
                v=v_full,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                scaling=layer.scaling,
            )
            forward_batch._timer_compress_algo.mark_end()

            is_per_head = keep_indices is not None and keep_indices.dim() == 2
            if keep_indices is None:
                num_to_keep = full_seq_len
            elif is_per_head:
                num_to_keep = keep_indices.shape[1]
            else:
                num_to_keep = keep_indices.shape[0]

            # ── Determine real_slots at layer 0 ──────────────────────────
            if layer_id == 0:
                if real_slots_seq is not None:
                    real_slots = real_slots_seq[:num_to_keep]
                    # Free excess pre-allocated slots that won't be used
                    excess = real_slots_seq[num_to_keep:]
                    if excess.numel() > 0:
                        forward_batch._gr_slots_to_free.append(excess)
                else:
                    # Fallback: should not happen when compressed allocation is used
                    real_slots = global_slots[:num_to_keep]

                forward_batch.req_to_token_pool.write(
                    (req_id, slice(0, num_to_keep)), real_slots
                )

                forward_batch._gr_real_slots[seq_idx] = real_slots
                all_compressed_lens.append(num_to_keep)

                _metrics.log_compression(
                    req_id, original_len=full_seq_len, compressed_len=num_to_keep
                )
            else:
                real_slots = forward_batch._gr_real_slots[seq_idx]
                num_to_keep = real_slots.shape[0]

            # ── Write compressed KV to real pool slots ───────────────────
            forward_batch._timer_kv_write.mark_start()
            self._write_compressed_to_real(
                layer_id=layer_id,
                k_seq=k_full,
                v_seq=v_full,
                keep_indices=keep_indices,
                is_per_head=is_per_head,
                num_to_keep=num_to_keep,
                real_slots=real_slots,
                num_kv_heads=num_kv_heads,
            )

            # ── Free GlobalKVPool slots immediately (reuse for next layer) ─
            global_kv_pool.free(global_slots)
            forward_batch._timer_kv_write.mark_end()

        _timer_compress.mark_end()

        # ── After last layer: free excess slots + finalize CPU save ───────
        if layer_id == num_layers - 1:
            # Free excess pre-allocated real slots that weren't used
            slots_to_free = getattr(forward_batch, "_gr_slots_to_free", [])
            if slots_to_free:
                self.token_to_kv_pool_allocator.free(torch.cat(slots_to_free))

            if do_cpu and cpu_miss_set:
                _input_ids_list = forward_batch.input_ids.cpu().tolist()
                for _si in cpu_miss_set:
                    _qs = cu_seqlens_q[_si].item()
                    _qe = cu_seqlens_q[_si + 1].item()
                    _toks = _input_ids_list[_qs:_qe]
                    _req_id = req_pool_indices[_si].item()
                    self.cpu_prefix_cache.finalize_entry(_req_id, _toks)

        forward_batch._gr_compressed_lens = all_compressed_lens
        forward_batch.kv_compressed_lens = [
            (n + 1 if n is not None else None) for n in all_compressed_lens
        ]

        # ── Sync timers and log metrics at last layer ─────────────────────
        if layer_id == num_layers - 1:
            _prefill_total_ms = forward_batch._timer_prefill.sync_and_total()
            _compress_total_ms = forward_batch._timer_compress.sync_and_total()
            _transfer_total_ms = forward_batch._timer_transfer.sync_and_total()
            _compress_algo_total_ms = forward_batch._timer_compress_algo.sync_and_total()
            _kv_write_total_ms = forward_batch._timer_kv_write.sync_and_total()

            _cm = get_metrics()
            for _si in range(batch_size):
                _req_id = req_pool_indices[_si].item()
                _cm.log_prefill_time(_req_id, _prefill_total_ms / max(batch_size, 1))
                _cm.log_compress_time(_req_id, _compress_total_ms / max(batch_size, 1))
                _cm.log_compress_algo_time(_req_id, _compress_algo_total_ms / max(batch_size, 1))
                if _si in cpu_prefix_lens and cpu_prefix_lens[_si] > 0:
                    _cm.log_cpu_transfer_time(_req_id, _transfer_total_ms / max(len(cpu_prefix_lens), 1))
                _cm.log_kv_write_time(_req_id, _kv_write_total_ms / max(batch_size, 1))

            _cm.log_global_summary()

        return output

    def _forward_extend_global_real_split_legacy(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        **kwargs,
    ) -> torch.Tensor:
        """
        Legacy Global/Real split prefill (fallback when GlobalKVPool is not available).

        This is the original implementation that allocates full extend length slots
        and then frees excess slots after compression.
        """
        layer_id = layer.layer_id
        num_heads = layer.tp_q_head_num
        num_kv_heads = layer.tp_k_head_num
        head_dim = layer.head_dim
        num_layers = len(self.k_buffer)
        device = k.device

        metadata = self.forward_metadata

        do_cpu = self.cpu_prefix_cache is not None

        # ── Layer-0 initialisation ─────────────────────────────────────────
        if layer_id == 0:
            forward_batch._gr_real_slots: Dict[int, torch.Tensor] = {}
            forward_batch._gr_compressed_lens: List[int] = []
            forward_batch._gr_slots_to_free: List[torch.Tensor] = []
            forward_batch._gr_cpu_prefix_slots: Dict[int, torch.Tensor] = {}

            forward_batch._cpu_miss_set: set = set()
            _metrics = get_metrics()
            if do_cpu:
                reqs = getattr(forward_batch, "reqs", None)
                if reqs is not None:
                    req_pool_indices_now = forward_batch.req_pool_indices
                    batch_size_now = metadata.cu_seqlens_q.shape[0] - 1
                    for _si in range(batch_size_now):
                        _req = reqs[_si]
                        _req_id = req_pool_indices_now[_si].item()
                        _match_len = getattr(_req, "cpu_match_len", 0)
                        if _match_len > 0:
                            forward_batch._gr_cpu_prefix_slots[_si] = _req.prefix_indices
                            _entry = _req.cpu_prefix_entry
                            _metrics.log_cpu_cache_hit(
                                _req_id, match_len=_match_len,
                                entry_len=_entry.seq_len,
                            )
                        else:
                            self.cpu_prefix_cache.start_accumulating(_req_id)
                            forward_batch._cpu_miss_set.add(_si)
                            _metrics.log_cpu_cache_miss(_req_id)

        cu_seqlens_q = metadata.cu_seqlens_q
        batch_size = cu_seqlens_q.shape[0] - 1
        req_pool_indices = forward_batch.req_pool_indices
        full_cache_loc = forward_batch.out_cache_loc
        cpu_prefix_slots = getattr(forward_batch, "_gr_cpu_prefix_slots", {})
        cpu_miss_set = getattr(forward_batch, "_cpu_miss_set", set())
        reqs = getattr(forward_batch, "reqs", None)
        _metrics = get_metrics()

        # ── Load CPU KV into prefix pool slots BEFORE FlashAttention ──────
        if reqs is not None and cpu_prefix_slots:
            with CudaTimer() as _t_transfer:
                for _si, _prefix_slots in cpu_prefix_slots.items():
                    _req = reqs[_si]
                    _entry = _req.cpu_prefix_entry
                    _match_len = _prefix_slots.shape[0]
                    _k_doc, _v_doc = self.cpu_prefix_cache.load_layer_to_gpu(
                        _entry, layer_id, str(device), num_tokens=_match_len
                    )
                    self.k_buffer[layer_id][_prefix_slots] = _k_doc
                    self.v_buffer[layer_id][_prefix_slots] = _v_doc
            for _si in cpu_prefix_slots:
                _req_id = req_pool_indices[_si].item()
                _metrics.log_cpu_transfer_time(_req_id, _t_transfer.elapsed_ms)

        # ── FlashAttention forward pass ───────────────────────────────────
        with CudaTimer() as _t_prefill:
            output = super().forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
            )
        if layer_id == 0:
            for _si in range(batch_size):
                _req_id = req_pool_indices[_si].item()
                _metrics.log_prefill_time(_req_id, _t_prefill.elapsed_ms)

        q_view = q.view(-1, num_heads, head_dim)
        all_compressed_lens: List[int] = getattr(forward_batch, "_gr_compressed_lens", [])

        # Compressor window size for Q padding
        _window_size = getattr(self.compression_config, 'window_size', 64)

        # ── Per-sequence compression and KV pool write ─────────────────────
        for seq_idx in range(batch_size):
            q_start = cu_seqlens_q[seq_idx].item()
            q_end = cu_seqlens_q[seq_idx + 1].item()
            extend_len = q_end - q_start

            req_id = req_pool_indices[seq_idx].item()
            _prefix_slots = cpu_prefix_slots.get(seq_idx, None)
            extend_cache_loc = full_cache_loc[q_start:q_end]

            # ── Determine full K/V ─────────────────────────────────────────
            if _prefix_slots is not None:
                k_doc = self.k_buffer[layer_id][_prefix_slots]
                v_doc = self.v_buffer[layer_id][_prefix_slots]
                if extend_len > 0:
                    k_query = self.k_buffer[layer_id][extend_cache_loc]
                    v_query = self.v_buffer[layer_id][extend_cache_loc]
                    k_full = torch.cat([k_doc, k_query], dim=0)
                    v_full = torch.cat([v_doc, v_query], dim=0)
                else:
                    k_full = k_doc
                    v_full = v_doc
                full_seq_len = _prefix_slots.shape[0] + extend_len
            else:
                k_full = self.k_buffer[layer_id][extend_cache_loc]
                v_full = self.v_buffer[layer_id][extend_cache_loc]
                full_seq_len = extend_len

            if full_seq_len == 0:
                if layer_id == 0:
                    # Use None (not 0) so process_batch_result_prefill skips
                    # this sequence.  Using 0 would reset kv_committed_len for
                    # decode requests in mixed batches, leaking all their slots.
                    all_compressed_lens.append(None)
                continue

            # Save full KQV to CPU for miss sequences
            if do_cpu and seq_idx in cpu_miss_set:
                q_seq_for_save = q_view[q_start:q_end]
                self.cpu_prefix_cache.accumulate_layer_kqv(
                    req_id, layer_id, k_full, v_full, q_seq_for_save
                )

            # ── Build Q for compression importance estimation ──────────────
            if _prefix_slots is not None and extend_len < _window_size:
                _req = reqs[seq_idx]
                _entry = _req.cpu_prefix_entry
                _match_len = _prefix_slots.shape[0]
                _need_from_cpu = min(_window_size - extend_len, _match_len)
                _q_pad = self.cpu_prefix_cache.load_query_to_gpu(
                    _entry, layer_id, str(device),
                    start=_match_len - _need_from_cpu,
                    end=_match_len,
                )
                if extend_len > 0:
                    q_for_compress = torch.cat([_q_pad, q_view[q_start:q_end]], dim=0)
                else:
                    q_for_compress = _q_pad
            else:
                q_for_compress = q_view[q_start:q_end]

            # ── Compress: select important tokens from the full sequence ───
            _, _, keep_indices = self._estimate_importance(
                method=self.importance_method,
                q=q_for_compress,
                k=k_full,
                v=v_full,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                scaling=layer.scaling,
            )

            is_per_head = keep_indices is not None and keep_indices.dim() == 2
            if keep_indices is None:
                num_to_keep = full_seq_len
            elif is_per_head:
                num_to_keep = keep_indices.shape[1]
            else:
                num_to_keep = keep_indices.shape[0]

            # ── Determine real_slots at layer 0 ────────────────────────────
            if layer_id == 0:
                if _prefix_slots is not None:
                    all_available = torch.cat([_prefix_slots, extend_cache_loc]) \
                        if extend_len > 0 else _prefix_slots
                else:
                    all_available = extend_cache_loc

                real_slots = all_available[:num_to_keep]
                excess_slots = all_available[num_to_keep:]
                if excess_slots.numel() > 0:
                    forward_batch._gr_slots_to_free.append(excess_slots)

                forward_batch.req_to_token_pool.write(
                    (req_id, slice(0, num_to_keep)), real_slots
                )

                forward_batch._gr_real_slots[seq_idx] = real_slots
                all_compressed_lens.append(num_to_keep)

                _metrics.log_compression(
                    req_id, original_len=full_seq_len, compressed_len=num_to_keep
                )
            else:
                real_slots = forward_batch._gr_real_slots[seq_idx]
                num_to_keep = real_slots.shape[0]

            # ── Write compressed KV in-place to real_slots ────────────────
            self._write_compressed_to_real(
                layer_id=layer_id,
                k_seq=k_full,
                v_seq=v_full,
                keep_indices=keep_indices,
                is_per_head=is_per_head,
                num_to_keep=num_to_keep,
                real_slots=real_slots,
                num_kv_heads=num_kv_heads,
            )

        # ── After last layer: free excess slots + finalize CPU save ──────
        if layer_id == num_layers - 1:
            slots_to_free = getattr(forward_batch, "_gr_slots_to_free", [])
            if slots_to_free:
                self.token_to_kv_pool_allocator.free(torch.cat(slots_to_free))

            if do_cpu and cpu_miss_set:
                _input_ids_list = forward_batch.input_ids.cpu().tolist()
                for _si in cpu_miss_set:
                    _qs = cu_seqlens_q[_si].item()
                    _qe = cu_seqlens_q[_si + 1].item()
                    _toks = _input_ids_list[_qs:_qe]
                    _req_id = req_pool_indices[_si].item()
                    self.cpu_prefix_cache.finalize_entry(_req_id, _toks)

        forward_batch._gr_compressed_lens = all_compressed_lens
        forward_batch.kv_compressed_lens = [
            (n + 1 if n is not None else None) for n in all_compressed_lens
        ]

        return output

    def _write_compressed_to_real(
        self,
        layer_id: int,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        keep_indices: Optional[torch.Tensor],
        is_per_head: bool,
        num_to_keep: int,
        real_slots: torch.Tensor,
        num_kv_heads: int,
    ) -> None:
        """
        Write compressed KV data from in-flight tensors to the real KV pool buffers.

        For per-head compression each head selects different tokens, so we write
        head-by-head. For global compression we write all heads at once.

        Args:
            layer_id:    Transformer layer index.
            k_seq:       Full key tensor [seq_len, num_kv_heads, head_dim]
            v_seq:       Full value tensor [seq_len, num_kv_heads, v_head_dim]
            keep_indices: Per-head [num_kv_heads, num_to_keep] or global [num_to_keep]
            is_per_head: Whether keep_indices is 2-D (per-head)
            num_to_keep: Number of tokens kept (same for all heads)
            real_slots:  Destination slot indices in real_kv_pool [num_to_keep]
            num_kv_heads: Number of KV heads
        """
        k_buf = self.k_buffer[layer_id]  # [pool_size, num_kv_heads, head_dim]
        v_buf = self.v_buffer[layer_id]

        if is_per_head:
            for head_idx in range(num_kv_heads):
                src_local = keep_indices[head_idx]  # [num_to_keep]
                k_buf[real_slots, head_idx] = k_seq[src_local, head_idx]
                v_buf[real_slots, head_idx] = v_seq[src_local, head_idx]
        elif keep_indices is not None:
            k_buf[real_slots] = k_seq[keep_indices]   # [num_to_keep, num_kv_heads, head_dim]
            v_buf[real_slots] = v_seq[keep_indices]
        else:
            # No compression: keep all tokens (identity mapping)
            k_buf[real_slots] = k_seq
            v_buf[real_slots] = v_seq

    # ================================================================== #
    # Feature 2: CPU Prefix Cache – Prefill from CPU
    # ================================================================== #

    def prefill_from_cpu_cache(
        self,
        token_ids,
        forward_batch: "ForwardBatch",
        seq_idx: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scaling: float,
        q_for_compression: Optional[torch.Tensor] = None,
    ) -> bool:
        """
        Skip GPU prefill by loading KV from CPU cache and compressing.

        This method is called instead of forward_extend when a matching prefix
        is found in the CPU cache. It processes all layers sequentially.

        Flow for each layer:
          1. Load full k, v from CPU → GPU
          2. (Optional) Compress using provided query (q_for_compression)
          3. Allocate real_kv_pool slots (compressed size)
          4. Write compressed KV to real_kv_pool
          5. Update req_to_token mapping

        Args:
            token_ids:          Prefix token IDs for cache lookup.
            forward_batch:      Forward batch metadata.
            seq_idx:            Index of the sequence in the current batch.
            num_heads:          Number of query heads (for GQA expansion).
            num_kv_heads:       Number of KV heads.
            head_dim:           Head dimension.
            scaling:            Attention scale factor.
            q_for_compression:  Query tensor [seq_len, num_heads, head_dim] on GPU.
                                If None, keeps all tokens (no compression).

        Returns:
            True if successfully loaded from CPU cache, False if no cache hit.
        """
        if self.cpu_prefix_cache is None:
            return False

        entry = self.cpu_prefix_cache.lookup(token_ids)
        if entry is None:
            return False

        req_pool_idx = forward_batch.req_pool_indices[seq_idx].item()
        device = self.k_buffer[0].device
        num_layers = len(self.k_buffer)

        if len(entry.kv_layers) != num_layers:
            import logging
            logging.getLogger(__name__).warning(
                f"CPU cache entry has {len(entry.kv_layers)} layers, "
                f"model has {num_layers}. Skipping."
            )
            return False

        real_slots = None  # allocated at layer 0

        for layer_id in range(num_layers):
            # Step 1: CPU → GPU transfer
            k_cpu, v_cpu = entry.kv_layers[layer_id]
            k_gpu = k_cpu.to(device)
            v_gpu = v_cpu.to(device)
            # k_gpu: [seq_len, num_kv_heads, head_dim]

            seq_len = k_gpu.shape[0]

            # Step 2: Compress (if query provided)
            if q_for_compression is not None and self.compressor.should_compress(layer_id, seq_len):
                _, _, keep_indices = self._estimate_importance(
                    method=self.importance_method,
                    q=q_for_compression,
                    k=k_gpu,
                    v=v_gpu,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    scaling=scaling,
                )
                is_per_head = keep_indices is not None and keep_indices.dim() == 2
                num_to_keep = (
                    keep_indices.shape[1]
                    if is_per_head
                    else (keep_indices.shape[0] if keep_indices is not None else seq_len)
                )
            else:
                keep_indices = None
                is_per_head = False
                num_to_keep = seq_len

            # Step 3 (layer 0 only): Allocate real_kv_pool slots
            if layer_id == 0:
                real_slots = self.token_to_kv_pool_allocator.alloc(num_to_keep)
                if real_slots is None:
                    raise RuntimeError(
                        f"[CPUPrefixCache] RealKVPool OOM: need {num_to_keep} slots"
                    )
                # Update req_to_token mapping
                forward_batch.req_to_token_pool.write(
                    (req_pool_idx, slice(0, num_to_keep)),
                    real_slots,
                )

            # Step 4: Write compressed KV to real_kv_pool
            self._write_compressed_to_real(
                layer_id=layer_id,
                k_seq=k_gpu,
                v_seq=v_gpu,
                keep_indices=keep_indices,
                is_per_head=is_per_head,
                num_to_keep=num_to_keep,
                real_slots=real_slots,
                num_kv_heads=num_kv_heads,
            )

        return True

    def finalize_cpu_save(
        self,
        request_id: int,
        token_ids,
    ) -> None:
        """
        Finalize saving the accumulated KV layers to the CPU prefix cache.

        Call this after the first request's prefill completes (all layers processed).

        Args:
            request_id: The req_pool_idx or unique request identifier.
            token_ids:  Prefix token IDs to use as cache key.
        """
        if self.cpu_prefix_cache is None or not self.save_prefix_to_cpu:
            return

        pending = self._cpu_accumulator.pop(request_id, None)
        if not pending:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        num_layers = len(self.k_buffer)
        kv_layers = []
        for layer_id in range(num_layers):
            if layer_id not in pending:
                return  # Incomplete, skip
            kv_layers.append(pending[layer_id])

        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()

        h = self.cpu_prefix_cache._hash_tokens(token_ids)

        from sglang.srt.mem_cache.prefix_cpu_cache import CPUKVEntry
        entry = CPUKVEntry(
            token_ids=token_ids,
            seq_len=len(token_ids),
            kv_layers=kv_layers,
        )
        self.cpu_prefix_cache._evict_if_needed(entry.mem_usage_bytes())
        self.cpu_prefix_cache._cache[h] = entry
        self.cpu_prefix_cache._total_bytes += entry.mem_usage_bytes()
        if h not in self.cpu_prefix_cache._lru_order:
            self.cpu_prefix_cache._lru_order.append(h)

    # ================================================================== #
    # Legacy standard compression path (unchanged)
    # ================================================================== #

    def _compress_kv_cache_after_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        **kwargs,
    ):
        """
        Compress KV cache after attention computation.

        This method processes each sequence in the batch separately:
        1. Estimates importance for each sequence's tokens
        2. Determines which tokens to keep per sequence
        3. Updates req_to_token mapping for each request
        4. Frees unused cache slots

        Note:
        - q has shape [total_tokens, num_heads * head_dim] (flattened)
        - k, v have shape [total_tokens, num_kv_heads, head_dim]
        """
        layer_id = layer.layer_id
        num_heads = layer.tp_q_head_num
        num_kv_heads = layer.tp_k_head_num
        head_dim = layer.head_dim

        # Reshape q from [total_tokens, num_heads * head_dim] to [total_tokens, num_heads, head_dim]
        q = q.view(-1, num_heads, head_dim)

        metadata = self.forward_metadata
        cu_seqlens_q = metadata.cu_seqlens_q

        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        batch_size = cu_seqlens_q.shape[0] - 1
        req_pool_indices = forward_batch.req_pool_indices

        all_keep_local_indices = []
        all_num_to_keep = []
        all_seq_cache_locs = []

        for seq_idx in range(batch_size):
            start_idx = cu_seqlens_q[seq_idx].item()
            end_idx = cu_seqlens_q[seq_idx + 1].item()
            seq_len = end_idx - start_idx

            if seq_len == 0:
                all_keep_local_indices.append(None)
                all_num_to_keep.append(0)
                all_seq_cache_locs.append(None)
                continue

            q_seq = q[start_idx:end_idx]
            k_seq = k[start_idx:end_idx]
            v_seq = v[start_idx:end_idx]

            compressed_k_seq, compressed_v_seq, keep_indices = self._estimate_importance(
                method=self.importance_method,
                q=q_seq,
                k=k_seq,
                v=v_seq,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                scaling=layer.scaling,
            )

            seq_cache_loc = cache_loc[start_idx:end_idx]

            all_keep_local_indices.append(keep_indices)
            # For per-head compression, keep_indices shape is [num_kv_heads, num_tokens_to_keep]
            if keep_indices is not None and keep_indices.dim() == 2:
                num_to_keep = keep_indices.shape[1]
                all_num_to_keep.append(num_to_keep)
            else:
                all_num_to_keep.append(keep_indices.shape[0] if keep_indices is not None else 0)
            all_seq_cache_locs.append(seq_cache_loc)

        # Print compression info (same format as SnapKV)
        total_original = cu_seqlens_q[-1].item()
        total_kept = sum(n for n in all_num_to_keep if n > 0)
        # print(f"[SnapKV] Compression: {total_original} -> {total_kept} tokens (freed {total_original - total_kept})")

        # Print all KV heads' keep indices for first sequence (sorted from small to large)
        if batch_size > 0 and all_keep_local_indices[0] is not None and all_keep_local_indices[0].dim() == 2:
            first_seq_indices = all_keep_local_indices[0]  # [num_kv_heads, num_tokens_to_keep]
            num_kv_heads = first_seq_indices.shape[0]
            for i in range(num_kv_heads):
                head_indices = sorted(first_seq_indices[i].tolist())
                # print(f"[SnapKV] {i} KV head keep indices ({len(head_indices)} tokens): {head_indices}")

        # Choose compression scheme
        if self.compression_scheme == "single_layer_zero_out":
            # Single-layer compression: only compress first layer, zero out non-kept tokens
            compressed_lens = self._single_layer_compression_zero_out(
                forward_batch=forward_batch,
                req_pool_indices=req_pool_indices,
                cu_seqlens_q=cu_seqlens_q,
                all_keep_local_indices=all_keep_local_indices,
                all_num_to_keep=all_num_to_keep,
                all_seq_cache_locs=all_seq_cache_locs,
                batch_size=batch_size,
                layer_id=layer_id,
            )
        else:
            # Standard compression: reorganize KV cache and update mapping
            compressed_lens = self._reorganize_kv_cache_and_update_mapping(
                forward_batch=forward_batch,
                req_pool_indices=req_pool_indices,
                cu_seqlens_q=cu_seqlens_q,
                all_keep_local_indices=all_keep_local_indices,
                all_num_to_keep=all_num_to_keep,
                all_seq_cache_locs=all_seq_cache_locs,
                batch_size=batch_size,
                layer_id=layer_id,
            )

        forward_batch.kv_compressed_lens = compressed_lens

        total_original = cu_seqlens_q[-1].item()
        total_kept = sum(n for n in all_num_to_keep if n > 0)
        total_freed = total_original - total_kept

        if total_freed > 0:
            self._update_compression_stats(layer_id, total_original, total_kept)

    def _reorganize_kv_cache_and_update_mapping(
        self,
        forward_batch: "ForwardBatch",
        req_pool_indices: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        all_keep_local_indices: list,
        all_num_to_keep: list,
        all_seq_cache_locs: list,
        batch_size: int,
        layer_id: int = None,
    ) -> list:
        """
        Reorganize KV cache by keeping only important tokens for each sequence.

        This method:
        1. Moves KV cache data from kept positions to the beginning of seq_cache_loc
        2. Updates req_to_token mapping for each request
        3. Frees unused cache slots
        4. Returns the compressed lengths for each sequence

        For per-head compression:
        - keep_local_indices shape: [num_kv_heads, num_tokens_to_keep]
        - Each head has its own compression indices
        - KV cache data is moved per-head

        For each sequence and each head, we move:
        - k_buffer[layer_id][head_idx][kept_cache_locs] -> k_buffer[layer_id][head_idx][seq_cache_loc[:num_to_keep]]
        - v_buffer[layer_id][head_idx][kept_cache_locs] -> v_buffer[layer_id][head_idx][seq_cache_loc[:num_to_keep]]
        """
        all_indices_to_free = []
        compressed_lens = []

        for seq_idx in range(batch_size):
            keep_local_indices = all_keep_local_indices[seq_idx]
            num_to_keep = all_num_to_keep[seq_idx]
            seq_cache_loc = all_seq_cache_locs[seq_idx]

            if keep_local_indices is None or seq_cache_loc is None:
                compressed_lens.append(0)
                continue

            seq_len = seq_cache_loc.shape[0]
            if num_to_keep >= seq_len:
                compressed_lens.append(seq_len)
                continue

            req_pool_idx = req_pool_indices[seq_idx].item()

            # Check if per-head compression (keep_local_indices is 2D)
            is_per_head = keep_local_indices.dim() == 2

            if is_per_head:
                # Per-head compression: each head has its own indices
                num_kv_heads = keep_local_indices.shape[0]

                # Move KV cache data per-head
                if layer_id is not None:
                    self._move_kv_cache_data_per_head(
                        layer_id=layer_id,
                        keep_local_indices=keep_local_indices,
                        seq_cache_loc=seq_cache_loc,
                        num_to_keep=num_to_keep,
                        seq_idx=seq_idx,
                    )

                # Update mapping to point to the new contiguous positions
                if layer_id == 0:
                    forward_batch.req_to_token_pool.write(
                        (req_pool_idx, slice(0, num_to_keep)),
                        seq_cache_loc[:num_to_keep],
                    )

                # After moving KV cache data to seq_cache_loc[:num_to_keep],
                # the positions to free are seq_cache_loc[num_to_keep:]
                # because all heads share the same seq_cache_loc
                freed_cache_locs = seq_cache_loc[num_to_keep:]
            else:
                # Original single-index compression
                kept_cache_locs = seq_cache_loc[keep_local_indices]

                # Move KV cache data from kept positions to the beginning of seq_cache_loc
                if layer_id is not None:
                    self._move_kv_cache_data(
                        layer_id=layer_id,
                        src_loc=kept_cache_locs,
                        dst_loc=seq_cache_loc[:num_to_keep],
                    )

                # After moving, free the positions after num_to_keep
                freed_cache_locs = seq_cache_loc[num_to_keep:]

                # Update mapping to point to the new contiguous positions
                if layer_id == 0:
                    forward_batch.req_to_token_pool.write(
                        (req_pool_idx, slice(0, num_to_keep)),
                        seq_cache_loc[:num_to_keep],
                    )

            compressed_lens.append(num_to_keep)
            if freed_cache_locs.numel() > 0:
                all_indices_to_free.append(freed_cache_locs)

        if all_indices_to_free and layer_id == 0:
            all_freed = torch.cat(all_indices_to_free)
            self.token_to_kv_pool_allocator.free(all_freed)

        return compressed_lens

    def _move_kv_cache_data_per_head(
        self,
        layer_id: int,
        keep_local_indices: torch.Tensor,
        seq_cache_loc: torch.Tensor,
        num_to_keep: int,
        seq_idx: int = 0,
    ) -> None:
        """
        Move KV cache data per-head for per-head compression.

        Args:
            layer_id: The layer ID
            keep_local_indices: Indices to keep per head [num_kv_heads, num_tokens_to_keep]
            seq_cache_loc: Cache locations for this sequence [seq_len]
            num_to_keep: Number of tokens to keep per head
            seq_idx: seq_idx in the batch
        """
        k_buffer = self.k_buffer[layer_id]
        v_buffer = self.v_buffer[layer_id]
        num_kv_heads = keep_local_indices.shape[0]

        # Initialize verification data structure for this layer and sequence
        if self._verification_enabled:
            if layer_id not in self._verification_data:
                self._verification_data[layer_id] = {}
            if seq_idx not in self._verification_data[layer_id]:
                self._verification_data[layer_id][seq_idx] = {}

        # Track verification results for this layer/seq
        all_passed = True

        # For each head, move its KV cache data
        for head_idx in range(num_kv_heads):
            head_keep_indices = keep_local_indices[head_idx]
            src_locs = seq_cache_loc[head_keep_indices]
            dst_locs = seq_cache_loc[:num_to_keep]

            # === VERIFICATION: Save original values before moving ===
            original_k_values = k_buffer[src_locs, head_idx].clone()
            original_v_values = v_buffer[src_locs, head_idx].clone()

            # Store for decode phase verification
            if self._verification_enabled:
                self._verification_data[layer_id][seq_idx][head_idx] = {
                    "k": original_k_values,
                    "v": original_v_values,
                    "src_locs": src_locs.clone(),
                    "dst_locs": dst_locs.clone(),
                    "num_to_keep": num_to_keep,
                }

            # Move key data for this head
            k_buffer[dst_locs, head_idx] = original_k_values
            # Move value data for this head
            v_buffer[dst_locs, head_idx] = original_v_values

            # === VERIFICATION: Check if moved values match original ===
            moved_k_values = k_buffer[dst_locs, head_idx]
            moved_v_values = v_buffer[dst_locs, head_idx]

            k_match = torch.allclose(original_k_values, moved_k_values, rtol=1e-5, atol=1e-5)
            v_match = torch.allclose(original_v_values, moved_v_values, rtol=1e-5, atol=1e-5)

            if not (k_match and v_match):
                all_passed = False

        # Print summary for this layer/seq
        status = "PASSED" if all_passed else "FAILED"
        # print(f"[PREFILL] Layer {layer_id}, Seq {seq_idx}: {status} (num_to_keep={num_to_keep}, heads={num_kv_heads})")

        # Print per-head token selection for debugging
        if seq_idx == 0:  # Only print for first sequence to avoid too much output
            # print(f"[PER-HEAD COMPRESSION] Layer {layer_id}:")
            for head_idx in range(min(2, num_kv_heads)):  # Only print first 2 heads
                head_keep_indices = keep_local_indices[head_idx]
                sorted_indices = sorted(head_keep_indices.tolist())
                # print(f"  Head {head_idx} keeps tokens: {sorted_indices[:min(10, len(sorted_indices))]}")

    def _move_kv_cache_data(
        self,
        layer_id: int,
        src_loc: torch.Tensor,
        dst_loc: torch.Tensor,
    ) -> None:
        """
        Move KV cache data from src_loc to dst_loc for a specific layer.

        Args:
            layer_id: The layer ID
            src_loc: Source token locations [num_tokens]
            dst_loc: Destination token locations [num_tokens]
        """
        # Get the KV buffer for this layer
        k_buffer = self.k_buffer[layer_id]
        v_buffer = self.v_buffer[layer_id]

        # Move key data
        k_buffer[dst_loc] = k_buffer[src_loc]

        # Move value data
        v_buffer[dst_loc] = v_buffer[src_loc]

    def _single_layer_compression_zero_out(
        self,
        forward_batch: "ForwardBatch",
        req_pool_indices: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        all_keep_local_indices: list,
        all_num_to_keep: list,
        all_seq_cache_locs: list,
        batch_size: int,
        layer_id: int = None,
    ) -> list:
        """
        Single-layer compression: only compress the first layer.

        This method:
        1. For layer_id == 0: zero out the KV cache for non-kept tokens
        2. Does NOT free any cache slots
        3. Does NOT update req_to_token mapping
        4. Does NOT modify sequence lengths
        5. Returns None (no compression information)

        Args:
            forward_batch: Forward batch information
            req_pool_indices: Request pool indices
            cu_seqlens_q: Cumulative sequence lengths for queries
            all_keep_local_indices: Indices to keep for each sequence
            all_num_to_keep: Number of tokens to keep for each sequence
            all_seq_cache_locs: Cache locations for each sequence
            batch_size: Batch size
            layer_id: Layer ID

        Returns:
            None (no compression information)
        """
        # Only process layer 0
        if layer_id != 0:
            return None

        # Process each sequence in the batch
        for seq_idx in range(batch_size):
            keep_local_indices = all_keep_local_indices[seq_idx]
            num_to_keep = all_num_to_keep[seq_idx]
            seq_cache_loc = all_seq_cache_locs[seq_idx]

            if keep_local_indices is None or seq_cache_loc is None:
                continue

            seq_len = seq_cache_loc.shape[0]
            if num_to_keep >= seq_len:
                continue

            # Check if per-head compression
            is_per_head = keep_local_indices.dim() == 2

            if is_per_head:
                # Per-head compression: zero out non-kept tokens for each head
                num_kv_heads = keep_local_indices.shape[0]
                k_buffer = self.k_buffer[layer_id]
                v_buffer = self.v_buffer[layer_id]

                for head_idx in range(num_kv_heads):
                    head_keep_indices = keep_local_indices[head_idx]

                    # Create a mask for tokens to zero out
                    zero_out_mask = torch.ones(seq_len, dtype=torch.bool, device=seq_cache_loc.device)
                    zero_out_mask[head_keep_indices] = False

                    # Get cache locations to zero out
                    zero_out_cache_locs = seq_cache_loc[zero_out_mask]

                    # Zero out the KV cache for this head
                    k_buffer[zero_out_cache_locs, head_idx] = 0
                    v_buffer[zero_out_cache_locs, head_idx] = 0

                print(f"[SINGLE_LAYER_COMPRESSION] Layer {layer_id}, Seq {seq_idx}: zeroed out {seq_len - num_to_keep} tokens per head (per-head compression)")
            else:
                # Single-index compression: zero out non-kept tokens
                # Create a mask for tokens to zero out
                zero_out_mask = torch.ones(seq_len, dtype=torch.bool, device=seq_cache_loc.device)
                zero_out_mask[keep_local_indices] = False

                # Get cache locations to zero out
                zero_out_cache_locs = seq_cache_loc[zero_out_mask]

                # Zero out the KV cache
                k_buffer = self.k_buffer[layer_id]
                v_buffer = self.v_buffer[layer_id]
                k_buffer[zero_out_cache_locs] = 0
                v_buffer[zero_out_cache_locs] = 0

                print(f"[SINGLE_LAYER_COMPRESSION] Layer {layer_id}, Seq {seq_idx}: zeroed out {len(zero_out_cache_locs)} tokens (single-index compression)")

        return None

    def _estimate_importance(
        self,
        method: ImportanceMethod,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scaling: float,
    ):
        """
        Estimate token importance using selected method and compress KV cache.

        Args:
            method: Importance estimation method
            q: Query tensor [seq_len, num_heads, head_dim] for a single sequence
            k: Key tensor [seq_len, num_kv_heads, head_dim] for a single sequence
            v: Value tensor [seq_len, num_kv_heads, head_dim] for a single sequence
            num_heads: Number of query heads
            num_kv_heads: Number of key/value heads
            head_dim: Head dimension
            scaling: Softmax scale

        Returns:
            compressed_k: Compressed key tensor
            compressed_v: Compressed value tensor
            keep_indices: Indices of retained tokens
        """
        compressed_k, compressed_v, keep_indices = self.compressor.compress(
            k, v, query=q
        )

        return compressed_k, compressed_v, keep_indices

    def _update_compression_stats(
        self,
        layer_id: int,
        original_len: int,
        compressed_len: int,
    ):
        """Update compression statistics."""
        self._compression_stats["total_compressed"] += 1
        self._compression_stats["total_freed"] += original_len - compressed_len

        if layer_id not in self._compression_stats["layer_stats"]:
            self._compression_stats["layer_stats"][layer_id] = {
                "count": 0,
                "total_original": 0,
                "total_compressed": 0,
            }

        stats = self._compression_stats["layer_stats"][layer_id]
        stats["count"] += 1
        stats["total_original"] += original_len
        stats["total_compressed"] += compressed_len

    def get_compression_stats(self) -> dict:
        """Get compression statistics."""
        return self._compression_stats.copy()

    def reset_compression_stats(self):
        """Reset compression statistics."""
        self._compression_stats = {
            "total_compressed": 0,
            "total_freed": 0,
            "layer_stats": {},
        }

    def verify_decode_kv_cache(
        self,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
    ):
        """
        Verify that KV cache values in decode phase match the original compressed values.

        This method is called at the beginning of each decode forward pass to verify
        that the KV cache values accessed through page_table match the values that
        were saved during compression.

        Args:
            layer: The attention layer
            forward_batch: The forward batch containing metadata
        """
        if not self._verification_enabled:
            return

        layer_id = layer.layer_id
        if layer_id not in self._verification_data:
            return

        metadata = self.forward_metadata
        cache_seqlens = metadata.cache_seqlens_int32

        # Get KV buffer for this layer
        k_buffer = self.k_buffer[layer_id]
        v_buffer = self.v_buffer[layer_id]

        all_passed = True

        # Verify each sequence
        for seq_idx, seq_data in self._verification_data[layer_id].items():
            seq_len = cache_seqlens[seq_idx].item()
            req_pool_idx = forward_batch.req_pool_indices[seq_idx].item()

            # Read the mapping from req_to_token_pool
            cache_locs = forward_batch.req_to_token_pool.req_to_token[req_pool_idx, :seq_len]

            # Verify each head
            for head_idx, head_data in seq_data.items():
                original_k = head_data["k"]
                original_v = head_data["v"]
                expected_num_to_keep = head_data["num_to_keep"]

                # The first num_to_keep cache_locs should be used
                actual_cache_locs = cache_locs[:expected_num_to_keep]

                # Read KV values from these cache locations
                actual_k = k_buffer[actual_cache_locs, head_idx]
                actual_v = v_buffer[actual_cache_locs, head_idx]

                # Compare with original values
                k_match = torch.allclose(original_k, actual_k, rtol=1e-5, atol=1e-5)
                v_match = torch.allclose(original_v, actual_v, rtol=1e-5, atol=1e-5)

                if not (k_match and v_match):
                    all_passed = False

        # Print summary for this layer
        status = "PASSED" if all_passed else "FAILED"
        print(f"[DECODE] Layer {layer_id}: {status}")

        # Clear verification data after last layer's decode to avoid memory overhead
        # Only clear after all layers have been verified
        max_layer_id = max(self._verification_data.keys()) if self._verification_data else 0
        if layer_id == max_layer_id:
            self._verification_data = {}

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for decode mode with KV cache verification.

        This method:
        1. Verifies that KV cache values match the original compressed values
        2. Calls parent's forward_decode for actual computation

        In global_real_split mode, decode writes new token KV directly to
        real_kv_pool (no global buffer needed for single tokens).
        """
        layer_id = layer.layer_id
        with CudaTimer() as _t_decode:
            result = super().forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        # Only record at layer 0 to avoid over-counting
        if layer_id == 0:
            req_ids = forward_batch.req_pool_indices.tolist()
            get_metrics().log_decode_step(req_ids, _t_decode.elapsed_ms)
        return result

    # ================================================================== #
    # Scheduler Integration Helper
    # ================================================================== #

    def estimate_memory_needed(self, seq_len: int) -> int:
        """
        Estimate the number of real KV pool slots needed for a request.

        In global_real_split mode, allocation is based on compressed length.
        In standard mode, the full seq_len is used (freed after compression).

        Args:
            seq_len: Full sequence length of the request.

        Returns:
            Estimated number of real KV pool slots needed.
        """
        if not self.use_global_real_split:
            return seq_len

        # Estimate compressed size using compression ratio
        config = self.compression_config
        num_to_keep = max(
            config.min_tokens_to_keep,
            int(seq_len * (1 - config.compression_ratio)),
        )
        # Add window_size tokens that are always kept
        num_to_keep = min(seq_len, num_to_keep + config.window_size)
        return num_to_keep

    def can_accept_request(self, seq_len: int) -> bool:
        """
        Check if there's enough memory to accept a request of seq_len tokens.

        Uses compressed length for the KV pool check.
        """
        compressed_len = self.estimate_memory_needed(seq_len)

        if self.token_to_kv_pool_allocator.available_size() < compressed_len:
            return False

        return True


def create_compressed_backend(
    runner,
    compression_config: Optional["CompressionConfig"] = None,
    importance_method: ImportanceMethod = "key_norm",
    **kwargs,
) -> CompressedFlashAttentionBackend:
    """
    Factory function to create a compressed attention backend.

    Args:
        runner: ModelRunner instance
        compression_config: Configuration for KV cache compression
        importance_method: Method for estimating token importance
            - "key_norm": Use key norms (fastest, recommended)
            - "lse": Use FlashAttention LSE (fast, approximate)
            - "snapkv": Compute full attention (slow, accurate)
        **kwargs: Additional arguments for CompressedFlashAttentionBackend

    Returns:
        CompressedFlashAttentionBackend instance
    """
    return CompressedFlashAttentionBackend(
        runner,
        compression_config=compression_config,
        importance_method=importance_method,
        **kwargs,
    )
