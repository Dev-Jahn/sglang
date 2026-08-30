"""Inference-only Qwen4-Exp (text + VL) on the Qwen3.5 backbone."""

import json
import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from typing import Any, Iterable, Optional, Set, Tuple

import msgspec
import sympy
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn

from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul
from sglang.srt.configs.qwen4_exp import (
    Qwen4ExpConfig,
    Qwen4ExpTextConfig,
    resolve_ple_storage,
)
from sglang.srt.distributed import get_tp_group, tensor_model_parallel_all_reduce
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather,
    attn_tp_all_reduce,
    dp_gather_replicate,
    dp_scatter,
    get_attention_dp_size,
    get_dp_global_num_tokens,
    get_global_dp_buffer,
    get_local_dp_buffer,
    is_allocation_symmetric,
    is_dp_attention_enabled,
)
from sglang.srt.layers.hyperconnection import (
    GatedResidual,
    HyperConnectionConfig,
)
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe import get_moe_a2a_backend, should_use_dp_reduce_scatterv
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import (
    CudaGraphReplayInput,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_req_to_token_pool,
)
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
    _is_stream_capturing,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_5 import (
    Qwen3_5AttentionDecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5GatedDeltaNet,
    Qwen3_5LinearDecoderLayer,
)
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_sm120_supported, is_sm121, logger
from sglang.srt.utils.numa_utils import allocate_interleaved_pinned_table

# Decode/verify-sized batches only: at prefill sizes both chains are compute
# bound and serializing them on one stream is faster than contending.
_QSA_INDEXER_OVERLAP_TOKEN_THRESHOLD = 1024
_PLE_EMBEDDING_ATTRIBUTES = (
    "quant_config",
    "enable_tp",
    "use_attn_tp_group",
    "tp_size",
    "num_embeddings",
    "org_vocab_size",
    "padding_size",
    "num_added_embeddings",
    "use_presharded_weights",
    "org_vocab_size_padded",
    "num_embeddings_padded",
    "shard_indices",
    "embedding_dim",
    "num_embeddings_per_partition",
    "num_org_embeddings_per_partition",
    "num_added_embeddings_per_partition",
)


def _should_interleave_ple_table() -> bool:
    return is_sm120_supported() and not is_sm121()


def _set_cuda_device(device: int) -> None:
    torch.cuda.set_device(device)


def _get_ple_forward_mode(forward_batch: ForwardBatch) -> ForwardMode:
    if forward_batch._original_forward_mode is not None:
        return forward_batch._original_forward_mode
    return forward_batch.forward_mode


def _get_processed_token_count(
    forward_batch: ForwardBatch, physical_tokens: int
) -> int:
    processed_tokens = forward_batch.num_token_non_padded_cpu
    if processed_tokens is None and forward_batch.extend_seq_lens_cpu is not None:
        processed_tokens = sum(forward_batch.extend_seq_lens_cpu)
    if processed_tokens is None:
        return physical_tokens
    processed_tokens = int(processed_tokens)
    if not 0 <= processed_tokens <= physical_tokens:
        raise RuntimeError(
            f"invalid PLE token counts: {processed_tokens=}, {physical_tokens=}"
        )
    return processed_tokens


class _PLEBatch(msgspec.Struct, frozen=True):
    mode: ForwardMode
    use_decode_fast_path: bool
    physical_tokens: int
    processed_tokens: int
    lengths: torch.Tensor
    row_width: int
    req_indices: torch.Tensor
    token_offsets: torch.Tensor
    valid_tokens: torch.Tensor
    state_indices: torch.Tensor
    ngram_context: Optional[torch.Tensor]
    ngram_eos_token_id: Optional[int]


def _prepare_ple_batch(
    input_ids: torch.Tensor,
    forward_batch: ForwardBatch,
    *,
    ngram_size: Optional[int],
    ngram_eos_token_id: Optional[int],
    replay: Optional[CudaGraphReplayInput] = None,
) -> Optional[_PLEBatch]:
    """Prepare the token layout and the shared N-gram history once per forward."""

    if forward_batch.tbo_parent_token_range is not None:
        raise NotImplementedError("Qwen4 PLE is not compatible with two-batch overlap")
    spec_algorithm = forward_batch.spec_algorithm
    if spec_algorithm is not None and spec_algorithm.is_ngram():
        raise NotImplementedError("Qwen4 PLE does not support NGRAM speculation")
    if (
        forward_batch.spec_info is not None
        and getattr(forward_batch.spec_info, "topk", 1) != 1
    ):
        raise NotImplementedError("Qwen4 PLE speculative decoding supports only topk=1")

    mode = (
        replay.forward_mode
        if replay is not None
        else _get_ple_forward_mode(forward_batch)
    )
    get_req_to_token_pool().ple_window_cache = None
    if mode.is_idle():
        return None
    use_decode_fast_path = (
        envs.SGLANG_ENABLE_QWEN4_PLE_FUSION.get() and mode.is_decode()
    )

    if input_ids.dim() > 1:
        input_ids = input_ids.reshape(-1)
    physical_tokens = input_ids.shape[0]
    processed_tokens = (
        int(replay.padded_num_tokens)
        if replay is not None
        else _get_processed_token_count(forward_batch, physical_tokens)
    )
    if replay is not None and processed_tokens != physical_tokens:
        raise RuntimeError(
            "PLE graph replay input does not match its padded token extent: "
            f"{processed_tokens=} {physical_tokens=}"
        )
    tokens = input_ids[:processed_tokens]
    positions = torch.arange(processed_tokens, device=tokens.device, dtype=torch.long)

    if mode.is_target_verify():
        assert forward_batch.spec_info is not None
        row_width = int(forward_batch.spec_info.draft_token_num)
        if row_width <= 0 or processed_tokens % row_width != 0:
            raise RuntimeError(
                "target verify rows must contain complete draft strides: "
                f"{processed_tokens=} {row_width=}"
            )
        sequence_count = processed_tokens // row_width
        # Eager verify can carry a valid length smaller than its fixed row
        # stride.  Ignore the synthetic one-token lengths created when DP
        # attention temporarily rewrites verify as EXTEND.
        lengths = (
            forward_batch.extend_seq_lens[:sequence_count].long()
            if replay is None
            and forward_batch.forward_mode.is_target_verify()
            and forward_batch.extend_seq_lens is not None
            else torch.full(
                (sequence_count,),
                row_width,
                dtype=torch.long,
                device=tokens.device,
            )
        )
        if lengths.shape[0] != sequence_count:
            raise RuntimeError(
                "target verify length metadata does not match its fixed rows: "
                f"{lengths.shape[0]=} {sequence_count=}"
            )
        req_indices = torch.div(positions, row_width, rounding_mode="floor")
        token_offsets = positions - req_indices * row_width
    elif mode.is_decode():
        lengths = torch.ones(processed_tokens, dtype=torch.long, device=tokens.device)
        row_width = 1
        req_indices = positions
        token_offsets = torch.zeros_like(positions)
    else:
        if forward_batch.extend_seq_lens is None:
            raise RuntimeError(f"PLE requires sequence lengths in {mode!r}")
        lengths = forward_batch.extend_seq_lens.long()
        extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
        row_width = (
            max(extend_seq_lens_cpu, default=0)
            if extend_seq_lens_cpu is not None
            else processed_tokens
        )
        query_start_loc = torch.cat(
            [lengths.new_zeros(1), torch.cumsum(lengths, dim=0)]
        )
        sequence_count = lengths.shape[0]
        req_indices = torch.searchsorted(query_start_loc, positions, right=True) - 1
        if processed_tokens:
            req_indices = req_indices.clamp(min=0, max=sequence_count - 1)
        token_offsets = positions - query_start_loc.index_select(0, req_indices)

    sequence_count = lengths.shape[0]
    out_cache_loc = (
        replay.out_cache_loc if replay is not None else forward_batch.out_cache_loc
    )
    if use_decode_fast_path:
        # Decode has one real token position per row.  Retain explicit tensors for
        # the common PLE batch contract without launching an index-select solely
        # to prove that every token offset (zero) is below every length (one).
        valid_tokens = torch.ones(
            processed_tokens, dtype=torch.bool, device=tokens.device
        )
        if out_cache_loc is not None:
            valid_tokens = valid_tokens & out_cache_loc[:processed_tokens].ne(0)
    else:
        valid_tokens = token_offsets < lengths.index_select(0, req_indices)
        if mode.is_target_verify() and out_cache_loc is not None:
            valid_tokens = valid_tokens & out_cache_loc[:processed_tokens].ne(0)

    req_pool_indices = (
        replay.req_pool_indices
        if replay is not None
        else forward_batch.req_pool_indices
    )
    state_indices = (
        get_req_to_token_pool()
        .get_mamba_indices(req_pool_indices[:sequence_count])
        .long()
    )

    # CUDA graph padding uses request slot 0, which may belong to a real request.
    # Map padded sequences to the state pools' reserved dummy slot instead.
    if use_decode_fast_path:
        if out_cache_loc is not None:
            state_indices = torch.where(
                out_cache_loc[:sequence_count].ne(0),
                state_indices,
                torch.zeros_like(state_indices),
            )
    else:
        valid = lengths.ne(0)
        if out_cache_loc is not None and mode.is_decode():
            valid = valid & out_cache_loc[:sequence_count].ne(0)
        elif out_cache_loc is not None and mode.is_target_verify():
            valid = valid & out_cache_loc[:processed_tokens].reshape(
                sequence_count, row_width
            ).ne(0).any(dim=1)
        state_indices = torch.where(
            valid, state_indices, torch.zeros_like(state_indices)
        )

    ngram_context = None
    if ngram_size is not None:
        assert ngram_eos_token_id is not None
        if use_decode_fast_path:
            # For decode, rows and tokens are one-to-one and valid_tokens is all
            # true.  The view is exactly the tensor the fill/where/index-put chain
            # would materialize.
            padded = tokens.unsqueeze(1)
        else:
            padded = tokens.new_full((sequence_count, row_width), ngram_eos_token_id)
            if processed_tokens:
                padded[req_indices, token_offsets] = torch.where(
                    valid_tokens,
                    tokens,
                    tokens.new_full((), ngram_eos_token_id),
                )
        history = get_req_to_token_pool().get_ngram_context(state_indices)
        if history.shape[1] != ngram_size - 1:
            raise RuntimeError(
                "Qwen4 PLE N-gram cache has the wrong context width: "
                f"{history.shape[1]=} {ngram_size=}"
            )
        ngram_context = torch.cat([history, padded], dim=1)

    return _PLEBatch(
        mode=mode,
        use_decode_fast_path=use_decode_fast_path,
        physical_tokens=physical_tokens,
        processed_tokens=processed_tokens,
        lengths=lengths,
        row_width=row_width,
        req_indices=req_indices,
        token_offsets=token_offsets,
        valid_tokens=valid_tokens,
        state_indices=state_indices,
        ngram_context=ngram_context,
        ngram_eos_token_id=ngram_eos_token_id,
    )


def _commit_ple_batch(batch: Optional[_PLEBatch], forward_batch: ForwardBatch) -> None:
    """Commit the shared N-gram history after every PLE layer consumed it."""

    if batch is None or batch.ngram_context is None or not batch.processed_tokens:
        return

    pool = get_req_to_token_pool()
    context = batch.ngram_context
    context_len = context.shape[1] - batch.row_width
    if batch.mode.is_target_verify():
        step_contexts = context.unfold(1, context_len, 1)[:, 1:]
        valid_steps = batch.valid_tokens.reshape(
            batch.lengths.shape[0], batch.row_width
        )
        pool.set_ngram_intermediate_context(
            torch.where(
                valid_steps.unsqueeze(-1),
                step_contexts,
                torch.full_like(step_contexts, batch.ngram_eos_token_id),
            )
        )
        return

    if batch.use_decode_fast_path:
        # Decode advances every two-token history by exactly one column.  Slicing
        # preserves the int64 values while avoiding arange + gather launches.
        next_context = context[:, batch.row_width :]
        pool.set_ngram_context(batch.state_indices, next_context)
        track = _ple_track_targets(forward_batch, batch)
        if track is not None:
            track_indices, _ = track
            pool.set_ngram_context(track_indices, next_context)
        return

    context_cols = torch.arange(context_len, device=context.device, dtype=torch.long)
    next_context = context.gather(
        1, batch.lengths.unsqueeze(1) + context_cols.unsqueeze(0)
    )
    pool.set_ngram_context(batch.state_indices, next_context)

    track = _ple_track_targets(forward_batch, batch)
    if track is not None:
        track_indices, track_offsets = track
        pool.set_ngram_context(
            track_indices,
            context.gather(1, track_offsets.unsqueeze(1) + context_cols.unsqueeze(0)),
        )


def _ple_track_targets(
    forward_batch: ForwardBatch, batch: _PLEBatch
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Destination slots and gather offsets for the extra-buffer track snapshot.

    With extra_buffer it is the ping-pong track slot — not the request's working
    slot — that `cache_{un,}finished_req` hand to the radix tree, so a side state
    that only writes the working slot gets cached with whatever the track slot's
    previous owner left behind.

    Both PLE side states are laid out `[incoming_state | this chunk's tokens]`, so
    the boundary value is the state's own gather at a smaller offset; callers differ
    only in tensor rank, hence returning offsets rather than doing the gather.
    Masked-off rows route to reserved slot 0 instead of being compacted, so there is
    no host sync and no data-dependent shape and this stays CUDA-graph capturable.

    None when tracking is inactive or its metadata is absent.
    """
    track_indices = forward_batch.mamba_track_indices
    track_mask = forward_batch.mamba_track_mask
    if track_indices is None or track_mask is None:
        return None

    rows = batch.lengths.shape[0]
    track_indices = track_indices[:rows]
    dst = torch.where(track_mask[:rows], track_indices, torch.zeros_like(track_indices))

    if batch.mode.is_decode():
        # One token per step, so the boundary offset is the current one. Decode never
        # carries mamba_track_seqlens, so this path must not consult it.
        return dst, batch.lengths

    aligned = forward_batch.mamba_track_aligned_lens()
    if aligned is None:
        return None

    return dst, aligned[:rows].clamp(min=0).minimum(batch.lengths)


def _pad_token_rows(x: torch.Tensor, total_tokens: int) -> torch.Tensor:
    if x.shape[0] == total_tokens:
        return x
    out = x.new_zeros((total_tokens, *x.shape[1:]))
    out[: x.shape[0]] = x
    return out


def _use_attn_tp_ngram() -> bool:
    return is_dp_attention_enabled() and envs.SGLANG_USE_ATTN_TP_NGRAM.get()


class Qwen4ExpPLEGroupedNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        group_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        if group_size is not None and hidden_size % group_size != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by group_size ({group_size})"
            )
        self.eps = eps
        self.group_size = group_size
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        # The JIT kernel requires group_size to be a multiple of 512; this is
        # init-static, so resolve it once here (device/dtype stay per-call).
        effective_group_size = group_size if group_size is not None else hidden_size
        self._jit_group_size = (
            effective_group_size if effective_group_size % 512 == 0 else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self._jit_group_size is not None
            and x.is_cuda
            and x.dtype in (torch.bfloat16, torch.float16)
        ):
            from sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm import (
                grouped_gemma_rmsnorm,
            )

            return grouped_gemma_rmsnorm(x, self.weight, self._jit_group_size, self.eps)
        compute_dtype = x.dtype
        x_float = x.float()
        if self.group_size is None:
            variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        else:
            group_shape = x_float.shape[:-1] + (-1, self.group_size)
            variance = x_float.reshape(group_shape).pow(2).mean(dim=-1, keepdim=True)
            variance = variance.expand(group_shape).reshape_as(x_float)
        x_norm = x_float * torch.rsqrt(variance + self.eps)
        weight = self.weight.float() + 1.0
        return (x_norm * weight).to(compute_dtype)


class Qwen4ExpNGramEmbedding(nn.Module):
    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        embedding_dim: int,
        ple_layer_index: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.ngram_embed_dim = int(embedding_dim)
        self.ngram_size = int(config.ngram_size)
        self.heads_per_ngram = int(config.heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.ple_layer_index = int(ple_layer_index)
        self.unigram_vocab_size = int(config.vocab_size)
        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {self.ngram_size}")
        if self.heads_per_ngram <= 0:
            raise ValueError(f"heads_per_ngram must be > 0, got {self.heads_per_ngram}")
        if self.ngram_embed_dim % self.ngram_heads != 0:
            raise ValueError(
                "ple_embed_dim must be divisible by total ngram heads: "
                f"{self.ngram_embed_dim} % {self.ngram_heads} != 0"
            )
        self.ngram_vocab_size_base = int(config.ngram_vocab_size_base)
        if self.ngram_vocab_size_base <= 0:
            raise ValueError("ngram_vocab_size_base must be > 0")
        self.make_ngram_vocab_size_divisible_by = int(
            config.make_ngram_vocab_size_divisible_by
        )
        self.head_dim_per_ngram = self.ngram_embed_dim // self.ngram_heads
        self.eos_token_id = int(config.eos_token_id)
        self.enable_ple_fusion = envs.SGLANG_ENABLE_QWEN4_PLE_FUSION.get()
        ple_storage = resolve_ple_storage(config, default="gpu")
        if ple_storage == "disk":
            from sglang.srt.models.qwen4_ple_disk import ROW_BYTES

            if self.head_dim_per_ngram != ROW_BYTES:
                raise ValueError(
                    "--ple-storage disk requires a PLE row width of "
                    f"{ROW_BYTES}, got {self.head_dim_per_ngram} from ple_embed_dim"
                )

        self.register_buffer(
            "layer_multipliers",
            self._build_layer_multipliers(self.ngram_size),
            persistent=True,
        )
        head_vocab_sizes, head_offsets, total_vocab_size = (
            self._build_head_vocab_and_offsets()
        )
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.tensor(head_vocab_sizes, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "ngram_heads_offsets",
            torch.tensor(head_offsets, dtype=torch.long),
            persistent=True,
        )
        padded_vocab_size = (
            (total_vocab_size + self.make_ngram_vocab_size_divisible_by - 1)
            // self.make_ngram_vocab_size_divisible_by
        ) * self.make_ngram_vocab_size_divisible_by
        self.use_attn_tp_ngram = _use_attn_tp_ngram()
        self.gather_dp_tokens = (
            is_dp_attention_enabled()
            and get_attention_dp_size() > 1
            and not self.use_attn_tp_ngram
        )
        allocation_context = (
            torch.device("meta") if ple_storage == "disk" else nullcontext()
        )
        with allocation_context:
            self.ngram_embedding = VocabParallelEmbedding(
                padded_vocab_size,
                self.head_dim_per_ngram,
                params_dtype=(
                    torch.float8_e4m3fn
                    if (quant_config is not None and quant_config.get_name() == "fp8")
                    or getattr(config, "ple_embedding_dtype", None) == "float8_e4m3fn"
                    else torch.bfloat16
                ),
                output_dtype=torch.bfloat16,
                use_attn_tp_group=self.use_attn_tp_ngram,
            )
        self.ngram_embedding.register_buffer(
            "weight_scale", torch.ones(1, dtype=torch.bfloat16), persistent=True
        )

    def _build_layer_multipliers(self, size: int) -> torch.Tensor:
        from sglang.srt.models.qwen4_ple_hash import build_layer_multipliers

        values = build_layer_multipliers(
            size,
            vocab_size=self.unigram_vocab_size,
            seed=int(getattr(self.config, "seed", 1234)),
            ple_layer_index=self.ple_layer_index,
        )
        return torch.from_numpy(values)

    @staticmethod
    def _find_nth_prime_after(start: int, n: int) -> int:
        prime = int(start)
        for _ in range(n):
            prime = int(sympy.nextprime(prime))
        return prime

    def _build_head_vocab_and_offsets(self):
        sizes = []
        offsets = []
        total = 0
        for head_idx in range(self.ngram_heads):
            global_head_idx = self.ple_layer_index * self.ngram_heads + head_idx
            size = self._find_nth_prime_after(
                self.ngram_vocab_size_base - 1, global_head_idx + 1
            )
            sizes.append(size)
            offsets.append(total)
            total += size
        return sizes, offsets, total

    def _embed_ngram_ids(
        self,
        ngram_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        physical_tokens: int,
    ) -> torch.Tensor:
        lookup_ids, semantic_tokens = self._prepare_embedding_lookup(
            ngram_ids, forward_batch, physical_tokens
        )
        embeddings = self.ngram_embedding(lookup_ids)
        embeddings = embeddings * self.ngram_embedding.weight_scale
        return self._finish_embedding_lookup(
            embeddings, semantic_tokens, forward_batch, physical_tokens
        )

    def _prepare_embedding_lookup(
        self,
        ngram_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        physical_tokens: int,
    ) -> Tuple[torch.Tensor, int]:
        semantic_tokens = ngram_ids.shape[0]
        if not self.gather_dp_tokens:
            return ngram_ids, semantic_tokens

        padded_ngram_ids = _pad_token_rows(ngram_ids, physical_tokens)
        global_tokens = forward_batch.global_dp_buffer_len
        if global_tokens is None:
            raise RuntimeError(
                "global-TP Qwen4 N-gram lookup under DP attention requires a "
                "DP token layout; set SGLANG_USE_ATTN_TP_NGRAM=1 to shard the "
                "table within each attention-TP group"
            )

        global_ngram_ids = ngram_ids.new_empty((global_tokens, *ngram_ids.shape[1:]))
        dp_gather_replicate(
            global_ngram_ids, padded_ngram_ids.contiguous(), forward_batch
        )
        return global_ngram_ids, semantic_tokens

    def _finish_embedding_lookup(
        self,
        embeddings: torch.Tensor,
        semantic_tokens: int,
        forward_batch: ForwardBatch,
        physical_tokens: int,
    ) -> torch.Tensor:
        if not self.gather_dp_tokens:
            return embeddings
        local_embeddings = embeddings.new_empty(
            (physical_tokens, *embeddings.shape[1:])
        )
        dp_scatter(local_embeddings, embeddings.contiguous(), forward_batch)
        return local_embeddings[:semantic_tokens]

    def _hash_contexts(
        self, contexts: torch.Tensor, *, decode_sized: bool = False
    ) -> torch.Tensor:
        contexts = contexts.to(torch.long)
        if self.enable_ple_fusion and decode_sized:
            from sglang.kernels.ops.qwen4_ple import (
                can_fuse_qwen4_ngram_hash,
                fused_qwen4_ngram_hash,
            )

            if can_fuse_qwen4_ngram_hash(
                contexts,
                self.layer_multipliers,
                self.ngram_heads_vocab_sizes,
                self.ngram_heads_offsets,
            ):
                return fused_qwen4_ngram_hash(
                    contexts,
                    self.layer_multipliers,
                    self.ngram_heads_vocab_sizes,
                    self.ngram_heads_offsets,
                    self.eos_token_id,
                )

        pool = get_req_to_token_pool()
        cached = pool.ple_window_cache
        if cached is not None and cached[1] is contexts and cached[2] is not None:
            shifted_tokens = cached[2]
            assert len(shifted_tokens) == self.ngram_size
        else:
            shifted_tokens = [contexts]
            for shift in range(1, self.ngram_size):
                shifted_tokens.append(self._shift_right_ignore_eos(contexts, shift))
            if cached is not None and cached[1] is contexts:
                pool.ple_window_cache = (cached[0], contexts, shifted_tokens)

        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            ngram_idx = ngram - 2
            start_idx = ngram_idx * self.heads_per_ngram
            end_idx = start_idx + self.heads_per_ngram
            mix = shifted_tokens[0] * self.layer_multipliers[0]
            for pos in range(1, ngram):
                mix = torch.bitwise_xor(
                    mix, shifted_tokens[pos] * self.layer_multipliers[pos]
                )
            head_vocab_sizes = self.ngram_heads_vocab_sizes[start_idx:end_idx]
            head_offsets = self.ngram_heads_offsets[start_idx:end_idx]
            ngram_ids = torch.remainder(
                mix[:, -1:].unsqueeze(-1), head_vocab_sizes.view(1, 1, -1)
            )
            ngram_ids = ngram_ids + head_offsets.view(1, 1, -1)
            blocks.append(ngram_ids[:, 0])
        return torch.cat(blocks, dim=-1)

    def _shift_right_ignore_eos(self, tensor: torch.Tensor, n: int) -> torch.Tensor:
        if n == 0:
            return tensor
        batch_size, seq_len = tensor.shape
        idx = torch.arange(seq_len, device=tensor.device, dtype=torch.long)
        eos_mask = tensor == self.eos_token_id
        eos_pos = torch.where(eos_mask, idx, -1)
        prev_eos_inclusive = torch.cummax(eos_pos, dim=1).values
        prev_eos = torch.cat(
            [eos_pos.new_full((batch_size, 1), -1), prev_eos_inclusive[:, :-1]],
            dim=1,
        )
        segment_start = prev_eos + 1
        pos_in_segment = idx.unsqueeze(0) - segment_start
        src_idx = idx - n
        gather_idx = torch.clamp(src_idx, min=0).unsqueeze(0).expand(batch_size, -1)
        shifted = tensor.gather(dim=1, index=gather_idx)
        valid_mask = (pos_in_segment >= n) & (src_idx.unsqueeze(0) >= 0)
        return torch.where(valid_mask, shifted, tensor.new_full((), self.eos_token_id))

    def forward_idle(self, forward_batch: ForwardBatch) -> None:
        if not self.gather_dp_tokens:
            return
        input_ids = forward_batch.input_ids.reshape(-1)
        dummy_ids = input_ids.new_zeros((input_ids.shape[0], self.ngram_heads))
        self._embed_ngram_ids(dummy_ids, forward_batch, input_ids.shape[0])

    def forward(
        self,
        batch: _PLEBatch,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        ngram_ids = self.compute_ngram_ids(batch)
        embeddings = self._embed_ngram_ids(
            ngram_ids, forward_batch, batch.physical_tokens
        )
        return embeddings.flatten(start_dim=-2)

    def compute_ngram_ids(self, batch: _PLEBatch) -> torch.Tensor:
        assert batch.ngram_context is not None
        pool = get_req_to_token_pool()
        cached = pool.ple_window_cache
        if cached is not None and cached[0] is batch:
            contexts = cached[1]
        else:
            if batch.use_decode_fast_path:
                contexts = batch.ngram_context
            else:
                contexts = batch.ngram_context.unfold(1, self.ngram_size, 1)[
                    batch.req_indices, batch.token_offsets
                ]
            contexts = contexts.to(torch.long)
            pool.ple_window_cache = (batch, contexts, None)
        ngram_ids = self._hash_contexts(
            contexts,
            decode_sized=batch.mode.is_decode() or batch.mode.is_target_verify(),
        )
        return torch.where(
            batch.valid_tokens.unsqueeze(-1), ngram_ids, ngram_ids.new_full((), -1)
        )


@triton.jit
def _gather_ple_embedding_from_pinned_kernel(
    weight_ptr,
    ids_ptr,
    output_ptr,
    embedding_dim,
    tp_vocab_start,
    tp_vocab_end,
    is_fp8: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_id = tl.program_id(0)
    global_idx = tl.load(ids_ptr + row_id)
    in_range = (global_idx >= tp_vocab_start) & (global_idx < tp_vocab_end)
    local_idx = tl.where(in_range, global_idx - tp_vocab_start, 0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < embedding_dim
    if is_fp8:
        weight_ptr = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
    else:
        weight_ptr = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.bfloat16))
    values = tl.load(
        weight_ptr + local_idx * embedding_dim + offsets,
        mask=mask & in_range,
        other=0.0,
    ).to(tl.bfloat16)
    tl.store(
        output_ptr + row_id * embedding_dim + offsets,
        tl.where(in_range, values, 0.0),
        mask=mask,
    )


class Qwen4ExpPinnedHostEmbedding(VocabParallelEmbedding):
    """PLE table read directly from pinned host memory.

    The table stays in its checkpoint storage dtype (fp8 with a per-tensor
    weight_scale for fp8 checkpoints, bf16 otherwise); gathers emit bf16.
    """

    def __init__(self, embedding: VocabParallelEmbedding) -> None:
        nn.Module.__init__(self)
        if not isinstance(embedding.quant_method, UnquantizedEmbeddingMethod):
            raise NotImplementedError(
                "PLE embedding offload requires an unquantized embedding table"
            )
        if embedding.weight.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            raise TypeError(
                "PLE embedding offload requires bfloat16 or fp8 weights, got "
                f"{embedding.weight.dtype}"
            )
        if embedding.num_added_embeddings:
            raise NotImplementedError(
                "PLE embedding offload does not support added vocabulary rows"
            )
        for name in _PLE_EMBEDDING_ATTRIBUTES:
            setattr(self, name, getattr(embedding, name))
        # The unquantized CUDA post-load hook is a no-op. Exclude this CPU-only
        # table so the generic loader does not stage it back to GPU unnecessarily.
        self.quant_method = None

        source_weight = embedding.weight
        # The GPU reads this table over PCIe, so it does not need to sit on the
        # NUMA node local to the GPU; spreading it keeps a multi-GiB shard from
        # exhausting a single node (see allocate_interleaved_pinned_table).
        pinned_data, self._pinned_buffer = allocate_interleaved_pinned_table(
            tuple(source_weight.shape),
            source_weight.dtype,
            interleave=_should_interleave_ple_table(),
        )
        cpu_weight = nn.Parameter(pinned_data, requires_grad=False)
        for name, value in vars(source_weight).items():
            setattr(cpu_weight, name, value)
        cpu_weight.weight_loader = self.weight_loader
        # The tensor is a bare view over an mmap this object owns; keep the
        # mapping reachable from the parameter itself so that a holder of the
        # weight cannot outlive the memory it points at.
        cpu_weight.pinned_host_mapping = self._pinned_buffer
        self.register_parameter("weight", cpu_weight)
        # The scale is tiny; keep it with the model instead of offloading it
        # with the table.
        self.register_buffer("weight_scale", embedding.weight_scale, persistent=True)
        del embedding.weight
        self._block_d = triton.next_power_of_2(self.embedding_dim)

    def allocate_output(
        self, shape: Tuple[int, ...], device: torch.device
    ) -> torch.Tensor:
        allocation_context = nullcontext()
        if self.tp_size > 1:
            allocation_context = use_symmetric_memory(
                get_tp_group(), disabled=not is_allocation_symmetric()
            )
        with allocation_context, torch.inference_mode(False):
            # The gather kernel emits bf16 rows regardless of the table dtype.
            return torch.empty(shape, dtype=torch.bfloat16, device=device)

    def gather(
        self, input_ids: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        expected_shape = (*input_ids.shape, self.embedding_dim)
        if out is None:
            output = self.allocate_output(expected_shape, input_ids.device)
        else:
            if tuple(out.shape) != expected_shape:
                raise ValueError(
                    f"invalid PLE prefetch output shape: {tuple(out.shape)} != "
                    f"{expected_shape}"
                )
            if out.dtype != torch.bfloat16 or out.device != input_ids.device:
                raise ValueError(
                    "PLE prefetch output must be bfloat16 on the id device"
                )
            output = out

        flat_ids = input_ids.reshape(-1).long()
        if flat_ids.numel():
            _gather_ple_embedding_from_pinned_kernel[(flat_ids.numel(),)](
                self.weight.data_ptr(),
                flat_ids,
                output,
                embedding_dim=self.embedding_dim,
                tp_vocab_start=self.shard_indices.org_vocab_start_index,
                tp_vocab_end=self.shard_indices.org_vocab_end_index,
                is_fp8=self.weight.dtype == torch.float8_e4m3fn,
                BLOCK_D=self._block_d,
            )
        return output

    def reduce(self, output: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1 and not get_attn_tp_context().input_scattered:
            if self.use_attn_tp_group:
                return attn_tp_all_reduce(output)
            return tensor_model_parallel_all_reduce(output)
        return output

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.reduce(self.gather(input_ids))


class Qwen4ExpDiskEmbedding(VocabParallelEmbedding):
    """TP-local exact FP8 PLE image with asynchronous host-driven fetches."""

    def __init__(
        self,
        embedding: VocabParallelEmbedding,
        config,
        valid_vocab_size: int,
        module_prefix: str = "ple",
    ) -> None:
        nn.Module.__init__(self)
        if not isinstance(embedding.quant_method, UnquantizedEmbeddingMethod):
            raise NotImplementedError(
                "PLE disk storage requires an unquantized embedding table"
            )
        if embedding.weight.dtype != torch.float8_e4m3fn:
            raise TypeError(
                "PLE disk storage requires float8_e4m3fn rows; set "
                'text_config.ple_embedding_dtype="float8_e4m3fn"'
            )
        if embedding.num_added_embeddings:
            raise NotImplementedError("PLE disk storage does not support added rows")
        for name in _PLE_EMBEDDING_ATTRIBUTES:
            setattr(self, name, getattr(embedding, name))
        self.quant_method = None
        self.weight_dtype = embedding.weight.dtype
        self._cuda_device = (
            embedding.weight.device.index
            if embedding.weight.device.index is not None
            else torch.cuda.current_device()
        )
        self.register_buffer("weight_scale", embedding.weight_scale, persistent=True)
        del embedding.weight

        from sglang.srt.models.qwen4_ple_disk import PLEImageBuilder, config_digest

        parallel = get_parallel()
        rank = parallel.attn_tp_rank if self.use_attn_tp_group else parallel.tp_rank
        disk_dir = getattr(config, "ple_disk_dir", None)
        if not disk_dir:
            raise ValueError("Qwen4 PLE disk storage requires ple_disk_dir")
        self._builder_args = {
            "root": disk_dir,
            "config_sha256": config_digest(config),
            "rank": rank,
            "tp_size": self.tp_size,
            "vocab_start": self.shard_indices.org_vocab_start_index,
            "vocab_end": min(
                self.shard_indices.org_vocab_end_index, int(valid_vocab_size)
            ),
            "module_prefix": module_prefix,
            "image_count": max(1, len(getattr(config, "ple_layer_ids", ()))),
        }
        self._image_builder = PLEImageBuilder(**self._builder_args)
        self._rank = rank
        self._module_prefix = module_prefix
        cache_budget_divisor = self.tp_size * max(
            1, len(getattr(config, "ple_layer_ids", ()))
        )
        self._hot_cache_gb = (
            float(getattr(config, "ple_disk_hot_cache_gb", 8.0)) / cache_budget_divisor
        )
        self._hot_frequency_file = getattr(config, "ple_disk_hot_frequency_file", None)
        self._dynamic_cache_gb = (
            float(getattr(config, "ple_disk_dynamic_cache_gb", 0.0))
            / cache_budget_divisor
        )
        self._prefill_buffer_tokens = int(
            getattr(config, "ple_disk_prefill_buffer_tokens", 0)
        )
        self._prefill_read_pages = int(
            getattr(config, "ple_disk_prefill_read_pages", 128)
        )
        self._ngram_heads = (int(getattr(config, "ngram_size", 3)) - 1) * int(
            getattr(config, "heads_per_ngram", 8)
        )
        from sglang.srt.models.qwen4_ple_disk import resolve_max_read_pages

        self._max_read_pages = resolve_max_read_pages(
            getattr(config, "ple_disk_max_read_pages", None)
        )
        if self._prefill_read_pages > self._max_read_pages:
            logger.warning(
                "Qwen4 PLE disk prefill read pages %d exceed max read pages %d; "
                "clamping the prefill value",
                self._prefill_read_pages,
                self._max_read_pages,
            )
            self._prefill_read_pages = self._max_read_pages
        self._fetcher_kwargs = {
            "hot_frequency_file": self._hot_frequency_file,
            "hot_cache_gb": self._hot_cache_gb,
            "dynamic_cache_gb": self._dynamic_cache_gb,
            "prefill_buffer_tokens": self._prefill_buffer_tokens,
            "prefill_read_pages": self._prefill_read_pages,
            "max_pages": self._max_read_pages,
            "ngram_heads": self._ngram_heads,
        }
        self._image = None
        self._fetcher = None
        self._weight_reload_pending = False
        self._executor = self._new_executor()
        self._future: Optional[Future] = None
        self._prefill_submit_future: Optional[Future] = None
        self._completion_event = None
        self._transfer_buffers = {}
        self._transfer_buffer_retain_rows = 65536
        self._active_transfer_device = None
        self._prefill_host_ids = None
        self._graph_generation = 0
        self._active_graph_generation = None
        self._stats_log_interval = int(
            getattr(config, "ple_disk_stats_log_interval", 0)
        )
        self._stats = {
            "steps": 0,
            "rows_requested": 0,
            "static_hits": 0,
            "dynamic_hits": 0,
            "prefill_hits": 0,
            "cold_pages": 0,
            "coalesced_pages": 0,
            "host_wait_us": 0.0,
            "ids_ready_wait_us": 0.0,
            "storage_fetch_us": 0.0,
            "hit_only_steps": 0,
            "miss_steps": 0,
            "hit_only_host_wait_us": 0.0,
            "miss_host_wait_us": 0.0,
            "hit_only_ids_ready_wait_us": 0.0,
            "miss_ids_ready_wait_us": 0.0,
            "hit_only_storage_fetch_us": 0.0,
            "miss_storage_fetch_us": 0.0,
            "wait_histogram_bounds_us": [
                50,
                100,
                250,
                500,
                1000,
                2500,
                5000,
                10000,
                25000,
            ],
            "hit_only_wait_hist": [0] * 10,
            "miss_wait_hist": [0] * 10,
        }

    def _new_executor(self) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"ple-disk-rank{self._rank}",
            initializer=_set_cuda_device,
            initargs=(self._cuda_device,),
        )

    def _open_fetcher(self) -> None:
        from sglang.srt.models.qwen4_ple_disk import DiskRowFetcher

        if self._image is None:
            raise RuntimeError("PLE disk image is unavailable")
        self._fetcher = DiskRowFetcher(self._image, **self._fetcher_kwargs)

    def add_checkpoint_shard(
        self, name: str, loaded_weight: torch.Tensor, row_start: int, row_end: int
    ) -> None:
        self._begin_weight_reload()
        self._image_builder.add_shard(name, loaded_weight, row_start, row_end)

    def finalize_image(self) -> None:
        if self._weight_reload_pending:
            self._weight_reload_pending = False
            return
        if self._image_builder is None:
            return
        image, reused, stats = self._image_builder.finalize(
            float(self.weight_scale.item())
        )
        self._image_builder = None
        self._image = image
        logger.info(
            "Qwen4 PLE disk image rank=%d path=%s reused=%s "
            "conversion=%.3fs throughput=%.3f GiB/s",
            self._rank,
            image.path,
            reused,
            stats["conversion_seconds"],
            stats["conversion_gib_per_s"],
        )
        if not self._hot_frequency_file and self._hot_cache_gb:
            logger.warning(
                "Qwen4 PLE disk hot-cache budget is %.2f GiB but no frequency "
                "file was supplied; starting with an empty exact hot set",
                self._hot_cache_gb,
            )
        try:
            self._open_fetcher()
        except BaseException:
            self.close()
            raise
        hot_bytes = self._fetcher.hot.rows.numel()
        metadata_bytes = (
            self._fetcher.hot.bitmap.nbytes + self._fetcher.hot.rank_prefix.nbytes
        )
        staging_bytes = self._fetcher.reader.staging.numel()
        dynamic_bytes = self._fetcher.dynamic.rows.numel()
        prefill_bytes = sum(
            slot["rows"].numel() for slot in self._fetcher._prefill_slots
        )
        logger.info(
            "Qwen4 PLE disk rank=%d RAM: hot_rows=%.3f GiB, "
            "dynamic_rows=%.3f GiB, prefill_rows=%.3f MiB, "
            "bitmap+rank=%.3f MiB, staging=%.3f MiB",
            self._rank,
            hot_bytes / (1 << 30),
            dynamic_bytes / (1 << 30),
            prefill_bytes / (1 << 20),
            metadata_bytes / (1 << 20),
            staging_bytes / (1 << 20),
        )

    def allocate_output(
        self, shape: Tuple[int, ...], device: torch.device
    ) -> torch.Tensor:
        allocation_context = nullcontext()
        if self.tp_size > 1:
            allocation_context = use_symmetric_memory(
                get_tp_group(), disabled=not is_allocation_symmetric()
            )
        with allocation_context, torch.inference_mode(False):
            return torch.empty(shape, dtype=torch.bfloat16, device=device)

    def _fetch_to_device(
        self,
        host_ids: torch.Tensor,
        ids_ready: torch.cuda.Event,
        raw_host: torch.Tensor,
        raw_device: torch.Tensor,
        output: torch.Tensor,
        stream: torch.cuda.Stream,
        admit_dynamic: bool,
    ) -> tuple[torch.cuda.Event, Any, float, float]:
        ids_wait_started = time.perf_counter_ns()
        ids_ready.synchronize()
        ids_ready_wait_us = (time.perf_counter_ns() - ids_wait_started) / 1000.0
        fetch_started = time.perf_counter_ns()
        self._fetcher.fetch(host_ids.numpy(), out=raw_host, admit_dynamic=admit_dynamic)
        storage_fetch_us = (time.perf_counter_ns() - fetch_started) / 1000.0
        fetch_stats = self._fetcher.last_fetch_stats
        with torch.cuda.stream(stream):
            raw_device.copy_(raw_host, non_blocking=True)
            output.copy_(raw_device.view(torch.float8_e4m3fn).to(torch.bfloat16))
            complete = torch.cuda.Event()
            complete.record(stream)
        return complete, fetch_stats, ids_ready_wait_us, storage_fetch_us

    def _get_transfer_buffers(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = input_ids.device
        required = input_ids.numel()
        cached = self._transfer_buffers.get(key)
        if cached is None or cached[0] < required:
            host_ids = torch.empty(
                required, dtype=torch.long, device="cpu", pin_memory=True
            )
            raw_host = torch.empty(
                (required, self.embedding_dim),
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            )
            raw_device = torch.empty(
                (required, self.embedding_dim),
                dtype=torch.uint8,
                device=input_ids.device,
            )
            cached = required, host_ids, raw_host, raw_device
            self._transfer_buffers[key] = cached
        _, host_ids, raw_host, raw_device = cached
        shape = tuple(input_ids.shape)
        raw_shape = (*shape, self.embedding_dim)
        return (
            host_ids[:required].view(shape),
            raw_host[:required].view(raw_shape),
            raw_device[:required].view(raw_shape),
        )

    def _launch_fetch(
        self,
        input_ids: torch.Tensor,
        output: torch.Tensor,
        *,
        admit_dynamic: bool = True,
    ) -> None:
        if self._prefill_submit_future is not None:
            self._prefill_submit_future.result()
            self._prefill_submit_future = None
        host_ids, raw_host, raw_device = self._get_transfer_buffers(input_ids)
        stream = torch.cuda.current_stream(input_ids.device)
        host_ids.copy_(input_ids, non_blocking=True)
        ids_ready = torch.cuda.Event()
        ids_ready.record(stream)
        self._future = self._executor.submit(
            self._fetch_to_device,
            host_ids,
            ids_ready,
            raw_host,
            raw_device,
            output,
            stream,
            admit_dynamic,
        )
        self._active_transfer_device = input_ids.device

    def queue_prefill(
        self,
        input_ids: Optional[torch.Tensor],
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        if (
            input_ids is None
            or not input_ids.numel()
            or not self._prefill_buffer_tokens
        ):
            return
        if self._prefill_submit_future is not None:
            if not self._prefill_submit_future.done():
                return
            self._prefill_submit_future.result()
        required = input_ids.numel()
        host_ids = self._prefill_host_ids
        if host_ids is None or host_ids.numel() < required:
            host_ids = torch.empty(
                required, dtype=torch.long, device="cpu", pin_memory=True
            )
            self._prefill_host_ids = host_ids
        host_ids = host_ids[:required].view(input_ids.shape)
        if stream is None:
            stream = torch.cuda.current_stream(input_ids.device)
        # The copy runs on the side stream. Keep the source allocation alive
        # until that stream has consumed it.
        input_ids.record_stream(stream)
        ready = torch.cuda.Event()
        with torch.cuda.stream(stream):
            host_ids.copy_(input_ids, non_blocking=True)
            ready.record(stream)

        def submit() -> bool:
            ready.synchronize()
            return self._fetcher.submit_prefill(host_ids.numpy())

        self._prefill_submit_future = self._executor.submit(submit)

    def gather(
        self,
        input_ids: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        *,
        admit_dynamic: bool = True,
    ) -> torch.Tensor:
        if self._fetcher is None:
            raise RuntimeError("PLE disk image was not finalized after weight loading")
        if self._future is not None:
            raise RuntimeError("previous PLE disk fetch was not consumed")
        expected_shape = (*input_ids.shape, self.embedding_dim)
        output = (
            out
            if out is not None
            else self.allocate_output(expected_shape, input_ids.device)
        )
        if tuple(output.shape) != expected_shape:
            raise ValueError(
                f"invalid PLE prefetch output shape: {tuple(output.shape)} != "
                f"{expected_shape}"
            )
        if output.dtype != torch.bfloat16 or output.device != input_ids.device:
            raise ValueError("PLE prefetch output must be bfloat16 on the id device")
        if get_is_capture_mode():
            return output
        self._launch_fetch(input_ids, output, admit_dynamic=admit_dynamic)
        return output

    def stage_graph_step(self, input_ids: torch.Tensor, out: torch.Tensor) -> int:
        if self._future is not None:
            raise RuntimeError("previous PLE disk graph generation was not consumed")
        expected_shape = (*input_ids.shape, self.embedding_dim)
        if tuple(out.shape) != expected_shape:
            raise ValueError(
                f"invalid PLE graph output shape: {tuple(out.shape)} != "
                f"{expected_shape}"
            )
        self._graph_generation += 1
        self._active_graph_generation = self._graph_generation
        self._launch_fetch(input_ids, out)
        return self._graph_generation

    def wait_for_graph_step(self, generation: int) -> None:
        if generation != self._active_graph_generation:
            raise RuntimeError(
                "PLE disk graph generation mismatch: "
                f"expected {self._active_graph_generation}, got {generation}"
            )
        try:
            self.wait_for_prefetch()
        finally:
            self._active_graph_generation = None

    def _wait_histogram_bin(self, wait_us: float) -> int:
        bounds = self._stats["wait_histogram_bounds_us"]
        for index, bound in enumerate(bounds):
            if wait_us < bound:
                return index
        return len(bounds)

    def stats_snapshot(self) -> dict:
        return {
            key: list(value) if isinstance(value, list) else value
            for key, value in self._stats.items()
        }

    def wait_for_prefetch(self) -> None:
        future = self._future
        if future is None:
            return
        active_device = self._active_transfer_device
        wait_started = time.perf_counter_ns()
        try:
            (
                self._completion_event,
                fetch_stats,
                ids_ready_wait_us,
                storage_fetch_us,
            ) = future.result()
        finally:
            self._future = None
            self._active_transfer_device = None
        wait_us = (time.perf_counter_ns() - wait_started) / 1000.0
        torch.cuda.current_stream().wait_event(self._completion_event)
        stats = self._stats
        stats["steps"] += 1
        stats["rows_requested"] += fetch_stats.rows_requested
        stats["static_hits"] += fetch_stats.static_hits
        stats["dynamic_hits"] += fetch_stats.dynamic_hits
        stats["prefill_hits"] += fetch_stats.prefill_hits
        stats["cold_pages"] += fetch_stats.cold_pages
        stats["coalesced_pages"] += fetch_stats.coalesced_pages
        stats["host_wait_us"] += wait_us
        stats["ids_ready_wait_us"] += ids_ready_wait_us
        stats["storage_fetch_us"] += storage_fetch_us
        hit_only = fetch_stats.cold_pages == 0
        kind = "hit_only" if hit_only else "miss"
        stats[f"{kind}_steps"] += 1
        stats[f"{kind}_host_wait_us"] += wait_us
        stats[f"{kind}_ids_ready_wait_us"] += ids_ready_wait_us
        stats[f"{kind}_storage_fetch_us"] += storage_fetch_us
        stats[f"{kind}_wait_hist"][self._wait_histogram_bin(wait_us)] += 1
        if self._stats_log_interval and stats["steps"] % self._stats_log_interval == 0:
            logger.info(
                "Qwen4 PLE disk stats rank=%d module=%s %s",
                self._rank,
                self._module_prefix,
                json.dumps(self.stats_snapshot(), separators=(",", ":")),
            )
        cached = self._transfer_buffers.get(active_device)
        if cached is not None and cached[0] > self._transfer_buffer_retain_rows:
            self._completion_event.synchronize()
            self._transfer_buffers.pop(active_device, None)

    def reduce(self, output: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1 and not get_attn_tp_context().input_scattered:
            if self.use_attn_tp_group:
                return attn_tp_all_reduce(output)
            return tensor_model_parallel_all_reduce(output)
        return output

    def close(self) -> None:
        future = getattr(self, "_prefill_submit_future", None)
        if future is not None:
            try:
                future.result()
            except BaseException:
                logger.warning(
                    "PLE prefill submission failed during shutdown", exc_info=True
                )
            self._prefill_submit_future = None
        future = getattr(self, "_future", None)
        if future is not None:
            try:
                future.result()
            except BaseException:
                logger.warning("PLE disk fetch failed during shutdown", exc_info=True)
            self._future = None
        fetcher = getattr(self, "_fetcher", None)
        if fetcher is not None:
            fetcher.close()
            self._fetcher = None
        builder = getattr(self, "_image_builder", None)
        if builder is not None:
            builder.close()
            self._image_builder = None
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=True)
            self._executor = None
        self._transfer_buffers.clear()
        self._active_transfer_device = None
        self._prefill_host_ids = None
        self._active_graph_generation = None

    def reset_graph_step(self) -> None:
        future = getattr(self, "_future", None)
        try:
            if future is not None:
                try:
                    result = future.result()
                    completion = result[0]
                    completion.synchronize()
                except BaseException:
                    logger.warning("PLE disk graph fetch failed", exc_info=True)
        finally:
            self._future = None
            self._active_graph_generation = None
            self._active_transfer_device = None

    def prepare_weight_reload(self) -> None:
        self._weight_reload_pending = True

    def _begin_weight_reload(self) -> None:
        if not self._weight_reload_pending:
            return
        from sglang.srt.models.qwen4_ple_disk import PLEImageBuilder

        self.close()
        self._image = None
        self._image_builder = PLEImageBuilder(**self._builder_args)
        self._executor = self._new_executor()
        self._weight_reload_pending = False

    @property
    def prefill_buffer_tokens(self) -> int:
        return self._prefill_buffer_tokens

    def resume_storage(self) -> None:
        if self._fetcher is not None:
            return
        if self._executor is None:
            self._executor = self._new_executor()
        self._open_fetcher()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        output = self.gather(input_ids)
        self.wait_for_prefetch()
        return self.reduce(output)


class Qwen4ExpPLELayer(nn.Module):
    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        layer_id: Optional[int] = None,
        ple_layer_index: int = 0,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.ple_embed_dim = config.ple_embed_dim
        self.conv_kernel_size = config.ple_conv_kernel_size
        self.hc_count = config.hc_count
        self.hc_hidden_size = self.hidden_size * self.hc_count
        self.ple_embedding = Qwen4ExpNGramEmbedding(
            config,
            self.ple_embed_dim,
            ple_layer_index=ple_layer_index,
            quant_config=quant_config,
        )
        ple_storage = resolve_ple_storage(config, default="gpu")
        if ple_storage == "pinned":
            self.ple_embedding.ngram_embedding = Qwen4ExpPinnedHostEmbedding(
                self.ple_embedding.ngram_embedding
            )
        elif ple_storage == "disk":
            self.ple_embedding.ngram_embedding = Qwen4ExpDiskEmbedding(
                self.ple_embedding.ngram_embedding,
                config,
                int(self.ple_embedding.ngram_heads_vocab_sizes.sum().item()),
                module_prefix=prefix,
            )
        self.short_conv_dilation = self.ple_embedding.ngram_size
        self.short_conv_state_len = (
            self.conv_kernel_size - 1
        ) * self.short_conv_dilation
        self.conv_channels = self.hc_hidden_size
        self.key_proj = ReplicatedLinear(
            self.ple_embed_dim,
            self.conv_channels,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.key_proj",
        )
        self.value_proj = ReplicatedLinear(
            self.ple_embed_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.value_proj",
        )
        norm_hidden = self.hc_hidden_size
        norm_group = self.hidden_size
        self.norm_key = Qwen4ExpPLEGroupedNorm(
            norm_hidden,
            eps=config.rms_norm_eps,
            group_size=norm_group,
        )
        self.norm_query = Qwen4ExpPLEGroupedNorm(
            norm_hidden,
            eps=config.rms_norm_eps,
            group_size=norm_group,
        )
        self.norm_conv = Qwen4ExpPLEGroupedNorm(
            norm_hidden,
            eps=config.rms_norm_eps,
            group_size=norm_group,
        )
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_channels,
            out_channels=self.conv_channels,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_channels,
            padding=(self.conv_kernel_size - 1) * self.short_conv_dilation,
            dilation=self.short_conv_dilation,
            bias=False,
        )
        nn.init.zeros_(self.conv1d.weight)
        self._prefetch_stream = (
            torch.cuda.Stream() if ple_storage in ("pinned", "disk") else None
        )
        self._graph_prefetch_buffer = None
        self._graph_prefetch_buffers = {}
        self._eager_prefetch_buffer = None
        self._prefetch_state = None
        self._graph_replay_generation = None
        self._graph_replay_stage_expected = False
        self._graph_replay_lookup_tokens = None
        self._graph_replay_prefetch_buffer = None
        self._future_lookup_contexts = None
        self._validate_graph_staging = envs.SGLANG_PLE_DISK_VALIDATE_GRAPH_STAGING.get()
        self._graph_lookup_validation_interval = (
            envs.SGLANG_PLE_DISK_GRAPH_LOOKUP_VALIDATION_INTERVAL.get()
        )
        self._graph_embedding_snapshot_buffers = {}
        self._pending_graph_embedding_validation = None
        self._completed_graph_embedding_validation = None
        self._graph_lookup_id_buffers = {}
        self._graph_lookup_validation_due = set()
        self._pending_graph_lookup_validation = None
        self._completed_graph_lookup_validation = None
        self._graph_replay_steps = 0

    def _apply_ple_norm(self, norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
        y = norm(x.flatten(-2, -1))
        return y.unflatten(-1, (self.hc_count, self.hidden_size))

    def _short_conv(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        batch: _PLEBatch,
    ) -> torch.Tensor:
        if x.shape[0] == 0:
            return x
        pool = get_req_to_token_pool()
        conv_state = pool.short_conv_layer_cache(self.layer_id)

        if batch.use_decode_fast_path:
            # Preserve the native depthwise-convolution and SiLU implementations;
            # only remove decode identities around them.  With row_width=1 the
            # padded/transpose path is exactly x.unsqueeze(-1), and every state
            # boundary is the contiguous one-column shift below.
            from sglang.kernels.ops.qwen4_ple import (
                can_fuse_qwen4_short_conv_state,
                fused_qwen4_short_conv_state,
            )

            fused_state = can_fuse_qwen4_short_conv_state(
                conv_state, batch.state_indices, x
            )
            if fused_state:
                conv_input = fused_qwen4_short_conv_state(
                    conv_state, batch.state_indices, x
                )
            else:
                state = conv_state.index_select(0, batch.state_indices).to(
                    dtype=x.dtype
                )
                conv_input = torch.cat([state, x.unsqueeze(-1)], dim=-1)
            conv_output = F.conv1d(
                conv_input,
                self.conv1d.weight.to(dtype=x.dtype),
                bias=None,
                dilation=self.short_conv_dilation,
                groups=self.conv_channels,
            ).squeeze(-1)
            next_state = conv_input[:, :, batch.row_width :]
            if not fused_state:
                conv_state[batch.state_indices] = next_state.to(dtype=conv_state.dtype)

            track = _ple_track_targets(forward_batch, batch)
            if track is not None:
                track_indices, _ = track
                conv_state[track_indices] = next_state.to(dtype=conv_state.dtype)
            return F.silu(conv_output)

        state = conv_state.index_select(0, batch.state_indices).to(dtype=x.dtype)
        padded_seq = x.new_zeros(
            (batch.lengths.shape[0], batch.row_width, self.conv_channels)
        )
        padded_seq[batch.req_indices, batch.token_offsets] = x
        conv_input = torch.cat([state, padded_seq.transpose(1, 2)], dim=-1)
        conv_output = F.conv1d(
            conv_input,
            self.conv1d.weight.to(dtype=x.dtype),
            bias=None,
            dilation=self.short_conv_dilation,
            groups=self.conv_channels,
        ).transpose(1, 2)

        if batch.mode.is_target_verify():
            intermediate_cache = pool.short_conv_layer_intermediate_cache(self.layer_id)
            if intermediate_cache is not None:
                if self.short_conv_state_len:
                    intermediate_state = (
                        conv_input.unfold(2, self.short_conv_state_len, 1)[
                            :, :, 1 : batch.row_width + 1
                        ]
                        .permute(0, 2, 1, 3)
                        .contiguous()
                    )
                else:
                    intermediate_state = x.new_empty(
                        (
                            batch.lengths.shape[0],
                            batch.row_width,
                            self.conv_channels,
                            0,
                        )
                    )
                valid_steps = batch.valid_tokens.reshape(
                    batch.lengths.shape[0], batch.row_width, 1, 1
                )
                intermediate_state = torch.where(
                    valid_steps,
                    intermediate_state,
                    torch.zeros_like(intermediate_state),
                )
                intermediate_cache[: batch.lengths.shape[0], : batch.row_width].copy_(
                    intermediate_state.to(dtype=intermediate_cache.dtype)
                )
        else:
            state_cols = torch.arange(
                self.short_conv_state_len, device=x.device, dtype=torch.long
            )

            def _gather_at(offsets: torch.Tensor) -> torch.Tensor:
                return conv_input.gather(
                    2,
                    (offsets.unsqueeze(1) + state_cols.unsqueeze(0))
                    .unsqueeze(1)
                    .expand(-1, self.conv_channels, -1),
                )

            next_state = _gather_at(batch.lengths)
            conv_state[batch.state_indices] = next_state.to(dtype=conv_state.dtype)

            # Same boundary mamba uses, into the slot the radix tree reads.
            track = _ple_track_targets(forward_batch, batch)
            if track is not None:
                track_indices, track_offsets = track
                conv_state[track_indices] = _gather_at(track_offsets).to(
                    dtype=conv_state.dtype
                )

        return F.silu(conv_output[batch.req_indices, batch.token_offsets])

    def forward_idle(self, forward_batch: ForwardBatch) -> None:
        if self._prefetch_state is not None:
            self._consume_prefetched_embeddings(forward_batch)
        else:
            self.ple_embedding.forward_idle(forward_batch)

    def _allocate_prefetch_buffer(
        self, lookup_tokens: int, device: torch.device
    ) -> torch.Tensor:
        offloaded_embedding = self.ple_embedding.ngram_embedding
        output = offloaded_embedding.allocate_output(
            (lookup_tokens, self.ple_embed_dim), device
        )
        if isinstance(offloaded_embedding, Qwen4ExpDiskEmbedding):
            output.zero_()
        return output

    def prepare_cuda_graph_prefetch_buffer(
        self, lookup_tokens: int, device: torch.device
    ) -> None:
        """Allocate the shared graph output before any capture begins."""
        if (
            not is_sm120_supported()
            or is_sm121()
            or self._prefetch_stream is None
            or lookup_tokens <= 0
        ):
            return
        self.reset_cuda_graph_capture_buffers()
        buffer = self._graph_prefetch_buffer
        if buffer is not None:
            if buffer.shape[0] < lookup_tokens:
                raise RuntimeError(
                    "PLE graph prefetch buffer is already referenced by captured "
                    f"graphs ({buffer.shape[0]} tokens < {lookup_tokens})"
                )
            return
        self._graph_prefetch_buffer = self._allocate_prefetch_buffer(
            lookup_tokens, device
        )

    def reset_cuda_graph_capture_buffers(self) -> None:
        self._graph_prefetch_buffers.clear()
        self._graph_lookup_id_buffers.clear()
        self._graph_embedding_snapshot_buffers.clear()
        self._graph_lookup_validation_due.clear()

    @staticmethod
    def _is_capturing() -> bool:
        """Whether this call is being recorded into a CUDA graph.

        ``get_is_capture_mode`` is a runner-set flag and not every graph runner
        in the tree sets it (the non-breakable prefill runner and the vision
        runners capture without it). Missing a capture here is not a slow path
        but a use-after-free: the shared eager buffer's address gets baked into
        the graph, and the next eager forward that needs more tokens replaces
        that buffer, leaving the graph writing into a freed block. Ask the
        driver as well.
        """
        if get_is_capture_mode():
            return True
        return torch.cuda.is_available() and _is_stream_capturing(
            torch.cuda.current_stream()
        )

    def _get_prefetch_buffer(
        self, lookup_tokens: int, lookup_ids: torch.Tensor
    ) -> torch.Tensor:
        if self._is_capturing():
            return self._select_graph_prefetch_buffer(
                lookup_tokens, device=lookup_ids.device, allocate=True
            )

        buffer = self._eager_prefetch_buffer
        if buffer is None or buffer.shape[0] < lookup_tokens:
            buffer = self._allocate_prefetch_buffer(lookup_tokens, lookup_ids.device)
            self._eager_prefetch_buffer = buffer
        return buffer[:lookup_tokens]

    def _select_graph_prefetch_buffer(
        self,
        lookup_tokens: int,
        *,
        device: Optional[torch.device] = None,
        allocate: bool = False,
    ) -> torch.Tensor:
        if (
            is_sm120_supported()
            and not is_sm121()
            and self._graph_prefetch_buffer is not None
        ):
            if not allocate and lookup_tokens not in self._graph_lookup_id_buffers:
                raise RuntimeError(
                    "PLE graph replay has no captured staging buffer for "
                    f"{lookup_tokens} lookup tokens"
                )
            buffer = self._graph_prefetch_buffer
            if buffer.shape[0] < lookup_tokens:
                raise RuntimeError(
                    "PLE graph prefetch buffer is too small "
                    f"({buffer.shape[0]} tokens available, {lookup_tokens} needed)"
                )
            return buffer[:lookup_tokens]

        buffer = self._graph_prefetch_buffers.get(lookup_tokens)
        if buffer is None and allocate:
            if device is None:
                raise ValueError("a device is required for graph buffer allocation")
            buffer = self._allocate_prefetch_buffer(lookup_tokens, device)
            self._graph_prefetch_buffers[lookup_tokens] = buffer
        if buffer is None:
            raise RuntimeError(
                "PLE graph replay has no captured staging buffer for "
                f"{lookup_tokens} lookup tokens"
            )
        return buffer

    def start_prefetch(
        self,
        batch: Optional[_PLEBatch],
        forward_batch: ForwardBatch,
    ) -> None:
        """Gather PLE rows via UVA while the preceding decoder layer runs."""
        if self._prefetch_stream is None:
            return
        if self._prefetch_state is not None:
            raise RuntimeError("PLE prefetch state was not consumed before reuse")
        if batch is None:
            if not self.ple_embedding.gather_dp_tokens:
                return
            physical_tokens = forward_batch.input_ids.numel()
            semantic_tokens = physical_tokens
        else:
            physical_tokens = batch.physical_tokens
            semantic_tokens = batch.processed_tokens
        offloaded_embedding = self.ple_embedding.ngram_embedding
        if (
            isinstance(offloaded_embedding, Qwen4ExpDiskEmbedding)
            and not self._is_capturing()
        ):
            self.validate_cuda_graph_replay()
        capturing_disk = self._is_capturing() and isinstance(
            offloaded_embedding, Qwen4ExpDiskEmbedding
        )
        if batch is None:
            ngram_ids = forward_batch.input_ids.new_zeros(
                (physical_tokens, self.ple_embedding.ngram_heads)
            )
        else:
            ngram_ids = self.ple_embedding.compute_ngram_ids(batch)
        lookup_ids, semantic_tokens = self.ple_embedding._prepare_embedding_lookup(
            ngram_ids, forward_batch, physical_tokens
        )
        lookup_tokens = lookup_ids.shape[0]
        if lookup_tokens == 0:
            self._future_lookup_contexts = None
            return
        future_lookup_ids = None
        future_contexts = self._future_lookup_contexts
        self._future_lookup_contexts = None
        if future_contexts is not None and future_contexts.numel():
            future_lookup_ids = self.ple_embedding._hash_contexts(future_contexts)
        prefetched = self._get_prefetch_buffer(lookup_tokens, lookup_ids)
        output_view = prefetched.view(lookup_tokens, self.ple_embedding.ngram_heads, -1)

        if capturing_disk:
            self._graph_lookup_id_buffers[lookup_tokens] = lookup_ids
            self._graph_lookup_validation_due.add(lookup_tokens)
            offloaded_embedding.gather(lookup_ids, out=output_view)
            self._prefetch_state = (
                prefetched,
                semantic_tokens,
                physical_tokens,
                future_lookup_ids,
            )
            return

        stream = self._prefetch_stream
        stream.wait_stream(torch.cuda.current_stream())
        lookup_ids.record_stream(stream)
        if not self._is_capturing():
            # The eager buffer is written by the side stream but allocated and
            # freed on the current one, so the allocator would otherwise hand
            # its block straight back when a larger buffer replaces it. Graph
            # buffers are excluded: they are held for the life of the process
            # and record_stream is not defined for graph-pool blocks.
            prefetched.record_stream(stream)
        with torch.cuda.stream(stream):
            if isinstance(offloaded_embedding, Qwen4ExpDiskEmbedding):
                offloaded_embedding.gather(
                    lookup_ids,
                    out=output_view,
                    admit_dynamic=batch is not None
                    and (batch.mode.is_decode() or batch.mode.is_target_verify()),
                )
            else:
                offloaded_embedding.gather(lookup_ids, out=output_view)
        self._prefetch_state = (
            prefetched,
            semantic_tokens,
            physical_tokens,
            future_lookup_ids,
        )

    def set_disk_prefill_contexts(self, contexts: Optional[torch.Tensor]) -> None:
        self._future_lookup_contexts = contexts

    def _graph_lookup_validation_required(self, lookup_tokens: int) -> bool:
        self._graph_replay_steps += 1
        interval = self._graph_lookup_validation_interval
        required = interval <= 1 or lookup_tokens in self._graph_lookup_validation_due
        if interval > 1:
            required = required or self._graph_replay_steps % interval == 0
        if required:
            self._graph_lookup_validation_due.discard(lookup_tokens)
        return required

    def prepare_cuda_graph_replay(
        self, batch: Optional[_PLEBatch], lookup_ids: Optional[torch.Tensor]
    ) -> None:
        offloaded_embedding = self.ple_embedding.ngram_embedding
        if not isinstance(offloaded_embedding, Qwen4ExpDiskEmbedding):
            return
        if (
            self._graph_replay_generation is not None
            or self._graph_replay_stage_expected
        ):
            raise RuntimeError("PLE graph replay was prepared twice without a wait")
        # Report the preceding replay after the scheduler has formed this batch.
        # A host read directly after replay changes loaded request interleaving.
        self.validate_cuda_graph_replay()
        if (
            self._pending_graph_embedding_validation is not None
            or self._pending_graph_lookup_validation is not None
        ):
            raise RuntimeError("PLE graph replay validation did not finish")

        if batch is None or lookup_ids is None:
            return
        lookup_tokens = lookup_ids.shape[0]
        if lookup_tokens == 0:
            return
        validate_lookup_ids = self._graph_lookup_validation_required(lookup_tokens)
        if validate_lookup_ids:
            self._pending_graph_lookup_validation = (
                lookup_tokens,
                lookup_ids.detach().clone(),
            )
        prefetched = self._select_graph_prefetch_buffer(lookup_tokens)
        output_view = prefetched.view(lookup_tokens, self.ple_embedding.ngram_heads, -1)
        stream = self._prefetch_stream
        stream.wait_stream(torch.cuda.current_stream())
        lookup_ids.record_stream(stream)
        self._graph_replay_stage_expected = True
        with torch.cuda.stream(stream):
            self._graph_replay_generation = offloaded_embedding.stage_graph_step(
                lookup_ids, output_view
            )
        self._graph_replay_lookup_tokens = lookup_tokens
        self._graph_replay_prefetch_buffer = prefetched

    def finish_cuda_graph_replay(self) -> None:
        if self._pending_graph_embedding_validation is not None:
            if self._completed_graph_embedding_validation is not None:
                raise RuntimeError("PLE graph staging validation was not reported")
            previous_tokens, expected_embeddings = (
                self._pending_graph_embedding_validation
            )
            actual_embeddings = self._graph_embedding_snapshot_buffers[previous_tokens]
            self._completed_graph_embedding_validation = (
                previous_tokens,
                expected_embeddings,
                actual_embeddings.ne(expected_embeddings),
            )
            self._pending_graph_embedding_validation = None
        if self._pending_graph_lookup_validation is not None:
            if self._completed_graph_lookup_validation is not None:
                raise RuntimeError("PLE graph lookup validation was not reported")
            previous_tokens, expected_ids = self._pending_graph_lookup_validation
            actual_ids = self._graph_lookup_id_buffers[previous_tokens]
            self._completed_graph_lookup_validation = (
                previous_tokens,
                expected_ids,
                actual_ids.ne(expected_ids),
            )
            self._pending_graph_lookup_validation = None

    def validate_cuda_graph_replay(self) -> None:
        if self._completed_graph_embedding_validation is not None:
            previous_tokens, expected_embeddings, mismatch = (
                self._completed_graph_embedding_validation
            )
            actual_embeddings = self._graph_embedding_snapshot_buffers[previous_tokens]
            if mismatch.any().item():
                index = mismatch.nonzero()[0].tolist()
                raise RuntimeError(
                    "PLE graph staging differs at consumption at "
                    f"{index}: {actual_embeddings[tuple(index)].item()} != "
                    f"{expected_embeddings[tuple(index)].item()}"
                )
            self._completed_graph_embedding_validation = None
        if self._completed_graph_lookup_validation is not None:
            previous_tokens, expected_ids, mismatch = (
                self._completed_graph_lookup_validation
            )
            actual_ids = self._graph_lookup_id_buffers[previous_tokens]
            if mismatch.any().item():
                index = mismatch.nonzero()[0].tolist()
                raise RuntimeError(
                    "PLE graph lookup IDs differ from disk staging at "
                    f"{index}: {actual_ids[tuple(index)].item()} != "
                    f"{expected_ids[tuple(index)].item()}"
                )
            self._completed_graph_lookup_validation = None

    def wait_cuda_graph_replay(self) -> None:
        offloaded_embedding = self.ple_embedding.ngram_embedding
        if not isinstance(offloaded_embedding, Qwen4ExpDiskEmbedding):
            return
        try:
            if self._graph_replay_generation is None:
                if self._graph_replay_stage_expected:
                    raise RuntimeError(
                        "PLE disk graph replay staging produced no generation"
                    )
                return
            offloaded_embedding.wait_for_graph_step(self._graph_replay_generation)
            torch.cuda.current_stream().wait_stream(self._prefetch_stream)
            if self._validate_graph_staging:
                lookup_tokens = self._graph_replay_lookup_tokens
                self._pending_graph_embedding_validation = (
                    lookup_tokens,
                    self._graph_replay_prefetch_buffer.detach().clone(),
                )
        finally:
            self._graph_replay_generation = None
            self._graph_replay_stage_expected = False
            self._graph_replay_lookup_tokens = None
            self._graph_replay_prefetch_buffer = None

    def reset_cuda_graph_replay(self) -> None:
        offloaded_embedding = self.ple_embedding.ngram_embedding
        try:
            if isinstance(offloaded_embedding, Qwen4ExpDiskEmbedding):
                offloaded_embedding.reset_graph_step()
        finally:
            self._graph_replay_generation = None
            self._graph_replay_stage_expected = False
            self._graph_replay_lookup_tokens = None
            self._graph_replay_prefetch_buffer = None
            self._pending_graph_lookup_validation = None
            self._pending_graph_embedding_validation = None

    def _consume_prefetched_embeddings(
        self, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        if self._prefetch_state is None:
            raise RuntimeError("PLE prefetch state is missing")
        embeddings, semantic_tokens, physical_tokens, future_lookup_ids = (
            self._prefetch_state
        )
        wait_for_prefetch = getattr(
            self.ple_embedding.ngram_embedding, "wait_for_prefetch", None
        )
        captured_disk = self._is_capturing() and isinstance(
            self.ple_embedding.ngram_embedding, Qwen4ExpDiskEmbedding
        )
        if captured_disk and self._validate_graph_staging:
            lookup_tokens = embeddings.shape[0]
            snapshot = self._graph_embedding_snapshot_buffers.get(lookup_tokens)
            if snapshot is None:
                snapshot = torch.empty_like(embeddings)
                self._graph_embedding_snapshot_buffers[lookup_tokens] = snapshot
            snapshot.copy_(embeddings)
        if not captured_disk:
            if wait_for_prefetch is not None:
                wait_for_prefetch()
            torch.cuda.current_stream().wait_stream(self._prefetch_stream)
            queue_prefill = getattr(
                self.ple_embedding.ngram_embedding, "queue_prefill", None
            )
            if queue_prefill is not None:
                queue_prefill(future_lookup_ids, self._prefetch_stream)
        embeddings = self.ple_embedding.ngram_embedding.reduce(embeddings)
        embeddings = embeddings * self.ple_embedding.ngram_embedding.weight_scale
        embeddings = self.ple_embedding._finish_embedding_lookup(
            embeddings,
            semantic_tokens,
            forward_batch,
            physical_tokens,
        )
        self._prefetch_state = None
        return embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        batch: _PLEBatch,
    ) -> torch.Tensor:
        hidden_states = hidden_states[: batch.processed_tokens]
        if self._prefetch_state is not None:
            embeddings = self._consume_prefetched_embeddings(forward_batch)
        else:
            embeddings = self.ple_embedding(batch, forward_batch)
        key, _ = self.key_proj(embeddings)
        value, _ = self.value_proj(embeddings)
        token_count = hidden_states.shape[0]
        hidden_size = self.hidden_size
        hc_count = self.hc_count
        if hidden_states.shape[-1] != hc_count * hidden_size:
            raise RuntimeError(
                "PLE hidden size does not match its hyper-connection layout: "
                f"expected {hc_count * hidden_size}, got {hidden_states.shape[-1]}"
            )
        key = key.reshape(token_count, hc_count, hidden_size)
        query = hidden_states.reshape(token_count, hc_count, hidden_size)
        key_normed = self._apply_ple_norm(self.norm_key, key)
        query_normed = self._apply_ple_norm(self.norm_query, query)
        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True)
        gate = gate / math.sqrt(hidden_size)
        fused_gate_value = False
        if batch.use_decode_fast_path:
            from sglang.kernels.ops.qwen4_ple import (
                can_fuse_qwen4_gate_value,
                fused_qwen4_gate_value,
            )

            fused_gate_value = can_fuse_qwen4_gate_value(gate, value)
        if fused_gate_value:
            gated_value = fused_qwen4_gate_value(gate, value)
        else:
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = torch.sigmoid(gate)
            gated_value = gate * value.unsqueeze(-2)
        gated_value_normed = self._apply_ple_norm(self.norm_conv, gated_value)
        gated_value = gated_value.flatten(-2)
        gated_value_normed = gated_value_normed.flatten(-2)
        conv_output = self._short_conv(
            gated_value_normed,
            forward_batch,
            batch,
        )
        output = gated_value + conv_output
        if not batch.use_decode_fast_path:
            output = torch.where(
                batch.valid_tokens.unsqueeze(-1),
                output,
                torch.zeros_like(output),
            )
        return _pad_token_rows(output, batch.physical_tokens)


class Qwen4ExpLayerExtensionMixin:
    def _init_qwen4_exp_layer_extensions(
        self,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.ple = None

        for attr_name in (
            "input_layernorm",
            "post_attention_layernorm",
            "layer_communicator",
        ):
            if hasattr(self, attr_name):
                delattr(self, attr_name)

        if (layer_id + 1) in config.ple_layer_ids:
            ple_layer_ids_sorted = sorted(set(config.ple_layer_ids))
            ple_layer_index = {
                abs_id: index for index, abs_id in enumerate(ple_layer_ids_sorted)
            }[layer_id + 1]
            # PLE is a sibling of the attn block (self.ple), so strip the block-type
            # segment like the dense mlp; else quant prefix misses ckpt skip-list → NaN.
            ple_prefix = prefix.replace(".linear_attn", "").replace(".self_attn", "")
            self.ple = Qwen4ExpPLELayer(
                config,
                quant_config=quant_config,
                prefix=f"{ple_prefix}.ple" if ple_prefix else "ple",
                layer_id=layer_id,
                ple_layer_index=ple_layer_index,
            )

        hc_config = HyperConnectionConfig(
            hc_count=self.hc_count,
            hidden_size=self.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            hc_per_branch_norm=True,
        )
        self.attn_hyper_connection = GatedResidual(
            hc_config,
            use_mix=True,
            use_combine=True,
        )
        self.mlp_hyper_connection = GatedResidual(
            hc_config,
            use_mix=True,
            use_combine=True,
        )

    def _prepare_qwen4_exp_attn(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        *,
        ple_batch: Optional[_PLEBatch],
    ):
        hc_dim = self.hc_count * self.hidden_size
        if hidden_states.shape[-1] != hc_dim:
            assert hidden_states.shape[-1] == self.hidden_size
            hidden_states = torch.cat(
                [hidden_states for _ in range(self.hc_count)], dim=-1
            )

        if self.ple is not None:
            if ple_batch is None:
                if not _get_ple_forward_mode(forward_batch).is_idle():
                    raise RuntimeError(
                        "non-idle Qwen4 PLE forward is missing its batch"
                    )
                self.ple.forward_idle(forward_batch)
            else:
                ple_query = (
                    hidden_states if residual is None else hidden_states + residual
                )
                hidden_states = hidden_states + self.ple(
                    ple_query, forward_batch, ple_batch
                )

        hidden_states, residual = self.attn_hyper_connection.mix(hidden_states)
        return hidden_states, residual

    def _prepare_qwen4_exp_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ):
        if not forward_batch.forward_mode.is_idle():
            hidden_states = attn_tp_all_reduce(hidden_states)
        hidden_states = self.attn_hyper_connection.combine(hidden_states, residual)
        hidden_states, residual = self.mlp_hyper_connection.mix(hidden_states)
        return hidden_states, residual

    def _qwen4_exp_use_dp_moe_gather(self) -> bool:
        return get_attention_dp_size() > 1 and get_moe_a2a_backend().is_none()

    def _qwen4_exp_use_attn_tp_a2a_scatter(self) -> bool:
        return get_parallel().attn_tp_size > 1 and not get_moe_a2a_backend().is_none()

    def _run_qwen4_exp_mlp(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if not self.config.num_experts:
            return self.mlp(hidden_states)

        use_dp_moe_gather = self._qwen4_exp_use_dp_moe_gather()
        use_attn_tp_a2a_scatter = self._qwen4_exp_use_attn_tp_a2a_scatter()

        if use_dp_moe_gather:
            hidden_states, local_hidden_states = (
                get_global_dp_buffer(get_tp_group()),
                hidden_states,
            )
            dp_gather_replicate(hidden_states, local_hidden_states, forward_batch)
        elif hidden_states.shape[0] == 0 and get_moe_a2a_backend().is_none():
            # Only safe to short-circuit an empty batch when the MoE holds no collective;
            # under deepep an idle DP rank must still join dispatch/combine or peers hang.
            return hidden_states

        attn_tp_chunks = None
        if use_attn_tp_a2a_scatter:
            attn_tp_size = get_parallel().attn_tp_size
            attn_tp_chunks = list(hidden_states.tensor_split(attn_tp_size))
            hidden_states = attn_tp_chunks[get_parallel().attn_tp_rank].contiguous()

        hidden_states = self.mlp(hidden_states, forward_batch)

        if use_dp_moe_gather:
            hidden_states, global_hidden_states = (
                get_local_dp_buffer(get_tp_group()),
                hidden_states,
            )
            if should_use_dp_reduce_scatterv():
                get_tp_group().reduce_scatterv(
                    global_hidden_states,
                    output=hidden_states,
                    sizes=get_dp_global_num_tokens(),
                )
            else:
                dp_scatter(hidden_states, global_hidden_states, forward_batch)
        elif use_attn_tp_a2a_scatter:
            assert attn_tp_chunks is not None
            gathered = [torch.empty_like(t) for t in attn_tp_chunks]
            attn_tp_all_gather(gathered, hidden_states.contiguous())
            hidden_states = torch.cat(gathered)

        return hidden_states

    def _postprocess_qwen4_exp_layer(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ):
        hidden_states = self.mlp_hyper_connection.combine(hidden_states, residual)
        return hidden_states, None


class Qwen4ExpLinearDecoderLayer(
    Qwen4ExpLayerExtensionMixin, Qwen3_5LinearDecoderLayer
):
    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
    ) -> None:
        super().__init__(config, layer_id, quant_config, prefix, alt_stream, is_nextn)
        self._init_qwen4_exp_layer_extensions(config, layer_id, quant_config, prefix)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        **kwargs,
    ):
        forward_batch = kwargs.get("forward_batch", None)

        hidden_states, residual = self._prepare_qwen4_exp_attn(
            hidden_states,
            residual,
            forward_batch,
            ple_batch=kwargs.get("ple_batch"),
        )

        if not forward_batch.forward_mode.is_idle():
            hidden_states = self.linear_attn(hidden_states, forward_batch)

        hidden_states, residual = self._prepare_qwen4_exp_mlp(
            hidden_states, residual, forward_batch
        )
        hidden_states = self._run_qwen4_exp_mlp(hidden_states, forward_batch)
        return self._postprocess_qwen4_exp_layer(hidden_states, residual, forward_batch)


class Qwen4ExpAttentionDecoderLayer(
    Qwen4ExpLayerExtensionMixin, Qwen3_5AttentionDecoderLayer
):
    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
    ) -> None:
        config.attn_output_gate = True
        super().__init__(config, layer_id, quant_config, prefix, alt_stream, is_nextn)
        from sglang.srt.layers.attention.qsa.config import is_qwen_qsa
        from sglang.srt.layers.attention.qsa.glue import build_qsa_indexer

        self.is_qsa = is_qwen_qsa(config)
        if self.is_qsa:
            self.indexer = build_qsa_indexer(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=f"{prefix}.indexer" if prefix else "indexer",
                rotary_emb=self.rotary_emb,
            )
        self._init_qwen4_exp_layer_extensions(config, layer_id, quant_config, prefix)

    def _compute_qsa_topk_indices(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.qsa.glue import (
            get_qsa_indexer_metadata,
            resolve_qsa_sparse_backend,
        )

        backend = get_attn_backend()
        sparse_backend = resolve_qsa_sparse_backend(backend)
        should_reuse = getattr(sparse_backend, "should_reuse_mtp_sparse_indices", None)
        if should_reuse is not None and should_reuse(forward_batch):
            # MTP decode steps reuse the draft-extend's target-aligned
            # selection; the indexer never runs inside the decode graph.
            return sparse_backend.lookup_mtp_sparse_indices(
                forward_batch, self.layer_id
            )
        indexer_metadata = get_qsa_indexer_metadata(
            backend, self.layer_id, forward_batch
        )
        topk_indices = self.indexer(
            hidden_states,
            positions,
            forward_batch,
            indexer_metadata,
        )
        should_capture = getattr(
            sparse_backend, "should_capture_mtp_sparse_indices", None
        )
        if should_capture is not None and should_capture(forward_batch):
            sparse_backend.capture_mtp_sparse_indices(
                topk_indices, forward_batch, self.layer_id, metadata=indexer_metadata
            )
        return topk_indices

    def self_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        overlap_indexer = (
            self.is_qsa
            and self.alt_stream is not None
            and get_is_capture_mode()
            and hidden_states.shape[0] < _QSA_INDEXER_OVERLAP_TOKEN_THRESHOLD
        )
        attention_kwargs = {}
        if overlap_indexer:
            # The indexer chain reads only hidden_states/positions and writes
            # QSA-private pool buffers, so it runs concurrently with the main
            # qkv projection + norm/rope chain.
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            with torch.cuda.stream(self.alt_stream):
                topk_indices = self._compute_qsa_topk_indices(
                    hidden_states, positions, forward_batch
                )

        q, k, v, gate = self._prepare_qkv_gate(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        if overlap_indexer:
            current_stream.wait_stream(self.alt_stream)
            # Allocated on alt_stream, consumed by attention on the current
            # stream; tell the caching allocator before alt_stream is reused.
            topk_indices.record_stream(current_stream)
            attention_kwargs["topk_indices"] = topk_indices
        elif self.is_qsa:
            attention_kwargs["topk_indices"] = self._compute_qsa_topk_indices(
                hidden_states, positions, forward_batch
            )

        attn_output = self.attn(q, k, v, forward_batch, **attention_kwargs)
        if gate is not None:
            if attn_output.is_cuda:
                # The strided 3D gate view feeds the kernel directly, so the
                # gate reshape copy disappears along with the sigmoid + mul.
                attn_output = fused_sigmoid_mul(attn_output, gate, inplace=True)
            else:
                gate = gate.reshape(gate.shape[0], -1) if gate.ndim == 3 else gate
                attn_output = attn_output * torch.sigmoid(gate)
        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ):
        hidden_states, residual = self._prepare_qwen4_exp_attn(
            hidden_states,
            residual,
            forward_batch,
            ple_batch=kwargs.get("ple_batch"),
        )

        if not forward_batch.forward_mode.is_idle():
            hidden_states = self.self_attention(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        hidden_states, residual = self._prepare_qwen4_exp_mlp(
            hidden_states, residual, forward_batch
        )
        hidden_states = self._run_qwen4_exp_mlp(hidden_states, forward_batch)
        return self._postprocess_qwen4_exp_layer(hidden_states, residual, forward_batch)


ALL_DECODER_LAYER_TYPES = {
    "attention": Qwen4ExpAttentionDecoderLayer,
    "full_attention": Qwen4ExpAttentionDecoderLayer,
    "linear_attention": Qwen4ExpLinearDecoderLayer,
}


class Qwen4ExpModel(Qwen3_5ForCausalLM):
    decoder_layer_types = ALL_DECODER_LAYER_TYPES
    supports_model_batch_hook = False
    supports_cuda_graph_replay_hook = False
    supports_storage_lifecycle_hook = False

    def _build_embed_tokens(self, config: Qwen4ExpTextConfig) -> nn.Module:
        return VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            use_attn_tp_group=is_dp_attention_enabled(),
        )

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        is_nextn: bool = False,
    ) -> None:
        super().__init__(config, quant_config, prefix, is_nextn)
        disk_storage = resolve_ple_storage(config, default="gpu") == "disk"
        self.supports_model_batch_hook = disk_storage
        self.supports_cuda_graph_replay_hook = disk_storage
        self.supports_storage_lifecycle_hook = disk_storage
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.has_ple = bool(config.ple_layer_ids)
        self.ple_ngram_size = int(config.ngram_size) if self.has_ple else None
        self.ple_ngram_eos_token_id = (
            int(config.eos_token_id) if self.ple_ngram_size is not None else None
        )
        self._ple_prefill_context_staging = None
        if hasattr(self, "norm"):
            delattr(self, "norm")
        hc_config = HyperConnectionConfig(
            hc_count=self.hc_count,
            hidden_size=self.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            hc_per_branch_norm=True,
        )
        self.hyper_connection_mixer = GatedResidual(hc_config, use_combine=False)

    def _ple_layers(self):
        for layer in self.layers:
            ple = getattr(layer, "ple", None)
            if ple is not None:
                yield ple

    def prepare_model_batch(self, schedule_batch, forward_batch: ForwardBatch) -> None:
        disk_layers = [
            ple
            for ple in self._ple_layers()
            if isinstance(ple.ple_embedding.ngram_embedding, Qwen4ExpDiskEmbedding)
        ]
        if not disk_layers:
            return
        contexts = None
        if forward_batch.forward_mode.is_extend_without_speculative():
            embedding = disk_layers[0].ple_embedding.ngram_embedding
            remaining = embedding.prefill_buffer_tokens
            ngram_size = self.ple_ngram_size
            eos_token_id = self.ple_ngram_eos_token_id
            if remaining > 0 and ngram_size is not None and eos_token_id is not None:
                staging = getattr(self, "_ple_prefill_context_staging", None)
                if (
                    staging is None
                    or staging.shape[0] < remaining
                    or staging.shape[1] != ngram_size
                ):
                    staging = torch.empty(
                        (remaining, ngram_size),
                        dtype=torch.long,
                        device="cpu",
                        pin_memory=forward_batch.input_ids.device.type == "cuda",
                    )
                    self._ple_prefill_context_staging = staging
                staging_array = staging.numpy()
                row_start = 0
                for req in schedule_batch.reqs:
                    extend_range = req.extend_range
                    if extend_range is None or remaining <= 0:
                        continue
                    tokens = req.origin_input_ids
                    start = int(extend_range.end)
                    end = min(len(tokens), start + remaining)
                    count = end - start
                    if count <= 0:
                        continue
                    history_start = start - ngram_size + 1
                    target = staging_array[row_start : row_start + count]
                    target.fill(eos_token_id)
                    for column in range(ngram_size):
                        first_valid = max(0, -history_start - column)
                        token_start = history_start + first_valid + column
                        valid_count = count - first_valid
                        if valid_count > 0:
                            target[first_valid:, column] = tokens[
                                token_start : token_start + valid_count
                            ]
                    row_start += count
                    remaining -= count
                if row_start:
                    contexts = staging[:row_start].to(forward_batch.input_ids.device)
        for ple in disk_layers:
            ple.set_disk_prefill_contexts(contexts)

    def prepare_cuda_graph_replay(self, replay: CudaGraphReplayInput) -> None:
        # DecodeCudaGraphRunner calls this pair around target-model replay. Qwen4
        # draft workers do not receive disk settings and their MTP model has no
        # PLE layer, so the draft and multi-step draft runners need no hook.
        disk_layers = [
            ple
            for ple in self._ple_layers()
            if isinstance(ple.ple_embedding.ngram_embedding, Qwen4ExpDiskEmbedding)
        ]
        if not disk_layers:
            return
        batch = _prepare_ple_batch(
            replay.input_ids,
            replay.runtime_forward_batch,
            ngram_size=self.ple_ngram_size,
            ngram_eos_token_id=self.ple_ngram_eos_token_id,
            replay=replay,
        )
        try:
            for ple in disk_layers:
                lookup_ids = None
                if batch is not None:
                    ngram_ids = ple.ple_embedding.compute_ngram_ids(batch)
                    lookup_ids, _ = ple.ple_embedding._prepare_embedding_lookup(
                        ngram_ids,
                        replay.runtime_forward_batch,
                        batch.physical_tokens,
                    )
                ple.prepare_cuda_graph_replay(batch, lookup_ids)
        except BaseException:
            for ple in disk_layers:
                ple.reset_cuda_graph_replay()
            raise

    def wait_cuda_graph_replay(self) -> None:
        disk_layers = [
            ple
            for ple in self._ple_layers()
            if isinstance(ple.ple_embedding.ngram_embedding, Qwen4ExpDiskEmbedding)
        ]
        for ple in disk_layers:
            ple.wait_cuda_graph_replay()

    def finish_cuda_graph_replay(self) -> None:
        for ple in self._ple_layers():
            if isinstance(ple.ple_embedding.ngram_embedding, Qwen4ExpDiskEmbedding):
                ple.finish_cuda_graph_replay()

    def reset_cuda_graph_replay(self) -> None:
        for ple in self._ple_layers():
            if isinstance(ple.ple_embedding.ngram_embedding, Qwen4ExpDiskEmbedding):
                ple.reset_cuda_graph_replay()

    def close(self) -> None:
        for ple in self._ple_layers():
            embedding = ple.ple_embedding.ngram_embedding
            if isinstance(embedding, Qwen4ExpDiskEmbedding):
                try:
                    ple.validate_cuda_graph_replay()
                finally:
                    embedding.close()

    def prepare_weight_reload(self) -> None:
        for ple in self._ple_layers():
            embedding = ple.ple_embedding.ngram_embedding
            if isinstance(embedding, Qwen4ExpDiskEmbedding):
                embedding.prepare_weight_reload()

    def resume_storage(self) -> None:
        for ple in self._ple_layers():
            embedding = ple.ple_embedding.ngram_embedding
            if isinstance(embedding, Qwen4ExpDiskEmbedding):
                embedding.resume_storage()

    @staticmethod
    def _prewarm_cuda_graph_jit_kernels(
        model: nn.Module, *, quantization: Optional[str]
    ) -> None:
        from sglang.kernels.ops.attention import qsa_indexer
        from sglang.kernels.ops.elementwise import fast_topk, hc_combine
        from sglang.kernels.ops.layernorm import grouped_gemma_rmsnorm
        from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer

        for module in model.modules():
            if isinstance(module, QSAIndexer):
                if module.block_topk in (512, 2048):
                    fast_topk._jit_fast_topk_module(module.block_topk)
                qsa_dtype = module.q_layernorm.weight.dtype
                rotary_emb = module.rotary_emb
                if (
                    qsa_dtype in (torch.bfloat16, torch.float16)
                    and module.index_head_dim in (64, 128, 256)
                    and rotary_emb.rotary_dim % 2 == 0
                    and not getattr(rotary_emb, "mrope_interleaved_glm", False)
                    and len(getattr(rotary_emb, "mrope_section", None) or ()) in (0, 3)
                    and getattr(rotary_emb, "cos_sin_cache", None) is not None
                ):
                    qsa_indexer._jit_qsa_indexer_module(
                        qsa_dtype,
                        module.index_head_dim,
                        rotary_emb.is_neox_style,
                    )
            elif isinstance(module, GatedResidual) and getattr(
                module, "_jit_combine_ok", False
            ):
                hc_combine._jit_hc_combine_module(
                    module.hc_count,
                    module.hidden_size,
                    module.block_inject_weight.weight.dtype,
                )
            elif isinstance(module, Qwen4ExpPLEGroupedNorm):
                group_size = module._jit_group_size
                if group_size is not None and module.weight.dtype in (
                    torch.bfloat16,
                    torch.float16,
                ):
                    grouped_gemma_rmsnorm._jit_grouped_gemma_rmsnorm_module(
                        group_size, module.weight.dtype
                    )

        if quantization == "fp8":
            from sglang.kernels.ops.gemm.fp8_blockwise_gemm import (
                _jit_fp8_blockwise_module,
            )

            _jit_fp8_blockwise_module()

    def prewarm_cuda_graphs(
        self, model_runner, *, capture_decode_cuda_graph: bool
    ) -> None:
        ple_layers = [
            module for module in self.modules() if isinstance(module, Qwen4ExpPLELayer)
        ]
        for module in ple_layers:
            module.reset_cuda_graph_capture_buffers()
        if not is_sm120_supported() or is_sm121():
            return
        graph_config = model_runner.server_args.cuda_graph_config
        max_tokens = 0
        if graph_config.prefill.backend != Backend.DISABLED:
            max_tokens = max(
                max_tokens,
                max(graph_config.prefill.bs or (graph_config.prefill.max_bs or 0,)),
            )
        if (
            capture_decode_cuda_graph
            and graph_config.decode.backend != Backend.DISABLED
        ):
            decode_width = model_runner.decode_num_tokens_per_req(
                num_draft_tokens=model_runner.server_args.speculative_num_draft_tokens
            )
            max_tokens = max(
                max_tokens,
                max(graph_config.decode.bs or (graph_config.decode.max_bs or 0,))
                * decode_width,
            )

        for module in ple_layers:
            lookup_tokens = max_tokens
            if module.ple_embedding.gather_dp_tokens:
                # Graph runners pad every DP rank to the captured local token
                # count before setting global_dp_buffer_len. The sum therefore
                # cannot exceed the local capture size times attention DP.
                lookup_tokens *= get_attention_dp_size()
            module.prepare_cuda_graph_prefetch_buffer(
                lookup_tokens, torch.device(model_runner.device)
            )

        tic = time.perf_counter()
        self._prewarm_cuda_graph_jit_kernels(
            self,
            quantization=model_runner.model_config.quantization,
        )
        logger.info(
            "Prewarmed Qwen4-Exp SM120 JIT kernels in %.2f seconds",
            time.perf_counter() - tic,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        ple_batch = (
            _prepare_ple_batch(
                input_ids,
                forward_batch,
                ngram_size=self.ple_ngram_size,
                ngram_eos_token_id=self.ple_ngram_eos_token_id,
            )
            if self.has_ple
            else None
        )
        residual = None
        aux_hidden_states = []
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            if i + 1 < self.end_layer:
                next_ple = getattr(self.layers[i + 1], "ple", None)
                if next_ple is not None:
                    next_ple.start_prefetch(ple_batch, forward_batch)
            with get_global_expert_distribution_recorder().with_current_layer(i):
                hidden_states, residual = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                    forward_batch=forward_batch,
                    ple_batch=ple_batch,
                    captured_last_layer_outputs=(
                        aux_hidden_states
                        if getattr(layer, "_is_layer_to_capture", False)
                        else None
                    ),
                )

        _commit_ple_batch(ple_batch, forward_batch)

        hc_hidden_states = hidden_states
        hidden_states, _ = self.hyper_connection_mixer.mix(hidden_states)
        if not forward_batch.forward_mode.is_idle():
            return hidden_states, hc_hidden_states

        if len(aux_hidden_states) == 0:
            return hidden_states
        return hidden_states, aux_hidden_states


class Qwen4ExpVLModel(Qwen4ExpModel):
    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        self.last_hc_hidden_states = None

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[Any] = None,
        input_deepstack_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.last_hc_hidden_states = None
        # mm routine passes input_ids=None; PLE needs the real ids.
        if input_ids is None:
            input_ids = forward_batch.input_ids
        model_output = super().forward(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            inputs_embeds=input_embeds,
        )
        if isinstance(model_output, tuple):
            hidden_states, self.last_hc_hidden_states = model_output
            return hidden_states
        return model_output


class Qwen4ExpForConditionalGeneration(Qwen3VLForConditionalGeneration):
    packed_modules_mapping = Qwen3_5ForCausalLM.packed_modules_mapping
    hf_to_sglang_mapper = None
    supports_model_batch_hook = False
    supports_cuda_graph_replay_hook = False
    supports_storage_lifecycle_hook = False

    def __init__(
        self,
        config: Qwen4ExpConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        language_model_cls=Qwen4ExpVLModel,
    ) -> None:
        super().__init__(config, quant_config, prefix, language_model_cls)
        self.supports_model_batch_hook = self.model.supports_model_batch_hook
        self.supports_cuda_graph_replay_hook = (
            self.model.supports_cuda_graph_replay_hook
        )
        self.supports_storage_lifecycle_hook = (
            self.model.supports_storage_lifecycle_hook
        )
        rope_config = getattr(self.config, "rope_parameters", None) or getattr(
            self.config, "rope_scaling", {}
        )
        self.is_mrope_enabled = (
            "mrope_section" in rope_config and not self.language_model_only
        )
        self.deepstack_visual_indexes = (
            self.visual.deepstack_visual_indexes if self.visual is not None else []
        )

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        hc_hidden_states = self.model.last_hc_hidden_states
        if hc_hidden_states is not None and isinstance(output, LogitsProcessorOutput):
            output.hidden_states = hc_hidden_states
        return output

    def prepare_cuda_graph_replay(self, replay: CudaGraphReplayInput) -> None:
        self.model.prepare_cuda_graph_replay(replay)

    def prepare_model_batch(self, schedule_batch, forward_batch: ForwardBatch) -> None:
        self.model.prepare_model_batch(schedule_batch, forward_batch)

    def wait_cuda_graph_replay(self) -> None:
        self.model.wait_cuda_graph_replay()

    def finish_cuda_graph_replay(self) -> None:
        self.model.finish_cuda_graph_replay()

    def reset_cuda_graph_replay(self) -> None:
        self.model.reset_cuda_graph_replay()

    def close(self) -> None:
        self.model.close()

    def prepare_weight_reload(self) -> None:
        self.model.prepare_weight_reload()

    def resume_storage(self) -> None:
        self.model.resume_storage()

    def _load_qwen4_exp_ple_buffer(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        buffers: dict,
        loaded_buffers: Set[str],
    ) -> bool:
        if ".ple.ple_embedding." not in name:
            return False
        buffer_name = name.rsplit(".", 1)[-1]
        if buffer_name.startswith("hashstats_"):
            return True
        if buffer_name == "token_lookup":
            return True
        if buffer_name not in {
            "layer_multipliers",
            "ngram_heads_offsets",
            "ngram_heads_vocab_sizes",
            "weight_scale",
        }:
            return False
        buffer = buffers.get(name)
        if buffer is None:
            return False
        if buffer.shape != loaded_weight.shape:
            raise ValueError(
                f"Shape mismatch for {name}: expected {tuple(buffer.shape)}, "
                f"got {tuple(loaded_weight.shape)}"
            )
        buffer.copy_(loaded_weight.to(device=buffer.device, dtype=buffer.dtype))
        loaded_buffers.add(name)
        return True

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            # qwen4 checkpoints are qwen3.5 format (head-first in_proj). These merge head-first
            # into the fused in_proj_qkvz/in_proj_ba, which is correct because the linear-attn
            # layer uses Qwen3_5GatedDeltaNet (head-first forward), not qwen3-next's group-first.
            ("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),
            ("in_proj_qkvz.", "in_proj_z.", 3),
            ("in_proj_ba.", "in_proj_b.", 0),
            ("in_proj_ba.", "in_proj_a.", 1),
        ]

        num_experts = getattr(self.config, "num_experts", None)
        expert_params_mapping = (
            FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=num_experts,
            )
            if num_experts is not None
            else []
        )
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale_inv",
            "_weight_scale_inv",
            ".input_scale_inv",
            "_input_scale_inv",
            "_weight_scale",
            "_input_scale",
        )

        def load_fused_expert_weights(
            name: str,
            params_dict: dict,
            loaded_weight: torch.Tensor,
            shard_id: str,
            num_experts: int,
        ) -> bool:
            if name not in params_dict:
                return False
            param = params_dict[name]
            weight_loader = param.weight_loader
            for expert_id in range(num_experts):
                weight_loader(
                    param,
                    loaded_weight[expert_id],
                    name,
                    shard_id,
                    expert_id,
                )
            return True

        def copy_ple_rows_to_tp_embedding(
            emb, loaded_weight: torch.Tensor, row_start: int, row_end: int
        ) -> None:
            tp_start = emb.shard_indices.org_vocab_start_index
            tp_end = emb.shard_indices.org_vocab_end_index
            ov_start = max(row_start, tp_start)
            ov_end = min(row_end, tp_end)
            if ov_start < ov_end:
                local_start = ov_start - tp_start
                src_start = ov_start - row_start
                n_rows = ov_end - ov_start
                emb.weight.data[local_start : local_start + n_rows].copy_(
                    loaded_weight[src_start : src_start + n_rows].to(
                        device=emb.weight.device, dtype=emb.weight.dtype
                    )
                )

        def load_qwen4_exp_ple_shard(name: str, loaded_weight: torch.Tensor) -> bool:
            if ".ngram_embedding.shard_" not in name:
                return False
            import re

            match = re.search(r"\.ngram_embedding\.shard_(\d+)\.weight$", name)
            if not match:
                return False
            shard_idx = int(match.group(1))
            mod_prefix = name[: name.index(".ngram_embedding.shard_")]
            ple_mod = ple_modules.get(mod_prefix)
            if ple_mod is None:
                return False
            emb = ple_mod.ngram_embedding
            emb_dtype = (
                emb.weight_dtype
                if isinstance(emb, Qwen4ExpDiskEmbedding)
                else emb.weight.dtype
            )
            if (
                loaded_weight.dtype == torch.float8_e4m3fn
                and emb_dtype != torch.float8_e4m3fn
            ):
                if isinstance(emb, Qwen4ExpPinnedHostEmbedding):
                    # offload gathers from pinned host memory; a swapped-in
                    # pageable tensor would fault in the Triton kernel.
                    raise ValueError(
                        "fp8 PLE auto-switch is unsupported with "
                        "--ple-storage pinned; set "
                        'text_config.ple_embedding_dtype="float8_e4m3fn" instead'
                    )
                logger.info(
                    "PLE embedding switched to fp8 storage: %s (%s)",
                    mod_prefix,
                    tuple(emb.weight.data.shape),
                )
                old_weight_data = emb.weight.data
                # StartupWeightLoadManager enforces tensor identity/dtype; this
                # swap breaks that contract if the model is ever enrolled.
                emb.weight = torch.nn.Parameter(
                    torch.empty_like(old_weight_data, dtype=torch.float8_e4m3fn),
                    requires_grad=False,
                )
                del old_weight_data
                # params_dict was snapshotted before the loop; drop the stale
                # entry or it pins the old bf16 storage until load end.
                params_dict.pop(f"{mod_prefix}.ngram_embedding.weight", None)
                torch.cuda.empty_cache()
            if (
                emb_dtype == torch.float8_e4m3fn
                and loaded_weight.dtype != torch.float8_e4m3fn
            ):
                if not getattr(load_qwen4_exp_ple_shard, "_warned_downcast", False):
                    load_qwen4_exp_ple_shard._warned_downcast = True
                    logger.warning(
                        "PLE checkpoint shards are %s but the embedding storage "
                        "is fp8 (ple_embedding_dtype / fp8 quant config); "
                        "downcasting is lossy",
                        loaded_weight.dtype,
                    )
            shard_size = (
                emb.org_vocab_size + ple_num_sync_shards - 1
            ) // ple_num_sync_shards
            shard_start = shard_idx * shard_size
            actual_rows = loaded_weight.shape[0]
            shard_end = shard_start + actual_rows
            if isinstance(emb, Qwen4ExpDiskEmbedding):
                emb.add_checkpoint_shard(name, loaded_weight, shard_start, shard_end)
            else:
                copy_ple_rows_to_tp_embedding(
                    emb, loaded_weight, shard_start, shard_end
                )
            loaded_shard_params.add(f"{mod_prefix}.ngram_embedding.weight")
            return True

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        buffers = dict(self.named_buffers())

        ple_modules = {
            mod_name: mod
            for mod_name, mod in self.named_modules()
            if isinstance(mod, Qwen4ExpNGramEmbedding)
        }
        text_config = getattr(self.config, "text_config", self.config)
        ple_num_sync_shards = int(
            getattr(
                text_config,
                "split_ngram_parts",
                getattr(self.config, "split_ngram_parts", 512),
            )
        )
        loaded_params: Set[str] = set()
        loaded_buffers: Set[str] = set()
        loaded_shard_params: Set[str] = set()
        skipped_visual_count = 0

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "mtp" in name:
                continue
            if "visual" in name and self.language_model_only:
                skipped_visual_count += 1
                continue
            if "language_model" in name:
                name = name.replace("model.language_model.", "model.")
            if ".self_attn." in name:
                name = name.replace(".self_attn", "")
            if name.endswith(".k_proj.k_scale"):
                name = name.replace(".k_proj.k_scale", ".attn.k_scale")
            elif name.endswith(".v_proj.v_scale"):
                name = name.replace(".v_proj.v_scale", ".attn.v_scale")

            if self._load_qwen4_exp_ple_buffer(
                name, loaded_weight, buffers, loaded_buffers
            ):
                continue
            if load_qwen4_exp_ple_shard(name, loaded_weight):
                continue
            if ".ple.ple_embedding.ngram_embedding." in name and name.endswith(
                ".weight"
            ):
                raise ValueError(
                    f"unsupported PLE weight layout (expected shard_N shards): {name}"
                )

            if (
                self.config.tie_word_embeddings
                and self.pp_group.is_last_rank
                and "model.embed_tokens.weight" in name
                and "lm_head.weight" in params_dict
            ):
                lm_head_param = params_dict["lm_head.weight"]
                weight_loader = getattr(
                    lm_head_param, "weight_loader", default_weight_loader
                )
                weight_loader(lm_head_param, loaded_weight)

            layer_id = get_layer_id(name)
            if layer_id is not None and (
                layer_id < self.start_layer or layer_id >= self.end_layer
            ):
                continue

            is_fused_expert = (
                "experts.gate_up_proj" in name or "experts.down_proj" in name
            )

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "visual" in name or "mlp.experts" in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if (
                    mapped_name.endswith(ignore_suffixes)
                    and mapped_name not in params_dict
                ):
                    continue
                if mapped_name not in params_dict:
                    continue
                param = params_dict[mapped_name]
                param.weight_loader(param, loaded_weight, shard_id)
                name = mapped_name
                break
            else:
                is_expert_weight = False
                current_expert_params_mapping = (
                    fused_expert_params_mapping
                    if is_fused_expert
                    else expert_params_mapping
                )
                for mapping in current_expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    if "visual" in name or self.config.encoder_only:
                        continue
                    is_expert_weight = True
                    mapped_name = name.replace(weight_name, param_name)
                    if is_fused_expert:
                        if "experts.gate_up_proj" in name:
                            gate_weight, up_weight = loaded_weight.chunk(2, dim=-2)
                            if not load_fused_expert_weights(
                                mapped_name,
                                params_dict,
                                gate_weight,
                                "w1",
                                num_experts,
                            ):
                                raise KeyError(f"Parameter {mapped_name} not found")
                            if not load_fused_expert_weights(
                                mapped_name,
                                params_dict,
                                up_weight,
                                "w3",
                                num_experts,
                            ):
                                raise KeyError(f"Parameter {mapped_name} not found")
                        else:
                            if not load_fused_expert_weights(
                                mapped_name,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            ):
                                raise KeyError(f"Parameter {mapped_name} not found")
                    else:
                        if (
                            mapped_name.endswith(ignore_suffixes)
                            and mapped_name not in params_dict
                        ):
                            continue
                        param = params_dict[mapped_name]
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight,
                            mapped_name,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    name = mapped_name
                    break
                else:
                    if is_expert_weight:
                        continue
                    if "visual" in name:
                        name = name.replace("attn.qkv.", "attn.qkv_proj.")
                        name = name.replace("model.visual.", "visual.")
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue
                    if name.endswith("_scale") and name not in params_dict:
                        assert (
                            abs(loaded_weight.item() - 1.0) < 1e-6
                        ), f"Expected 1.0, got {loaded_weight.item()} in skipped {name}"
                        continue
                    if name in params_dict:
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning(
                            "Parameter %s not found while loading Qwen4-Exp VL weights",
                            name,
                        )
                        continue
            loaded_params.add(name)

        loaded_params.update(loaded_buffers)
        loaded_params.update(loaded_shard_params)

        if skipped_visual_count > 0:
            logger.info(
                f"[language_model_only] Qwen4 load_weights: skipped "
                f"{skipped_visual_count} visual weights"
            )

        for module in self.modules():
            if isinstance(module, Qwen3_5GatedDeltaNet):
                module.finalize_fused_in_proj()
            elif isinstance(module, Qwen4ExpDiskEmbedding):
                module.finalize_image()

        return loaded_params

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        text_config = getattr(config, "text_config", config)
        if getattr(text_config, "num_experts", None) is None:
            return None
        return ModelConfigForExpertLocation(
            num_layers=text_config.num_hidden_layers,
            num_logical_experts=text_config.num_experts,
            num_groups=None,
        )


EntryClass = [Qwen4ExpForConditionalGeneration]
