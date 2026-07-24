import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import lru_cache
from threading import Lock
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import scipy  # type: ignore
import torch
import torch_npu
import vllm.envs as envs_vllm
from torch import nn
from vllm.config import CUDAGraphMode, VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
    is_v1_kv_transfer_group,
)
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadataBuilder
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.backend import (
    AttentionBackend,  # type: ignore
    AttentionCGSupport,
    MLAAttentionImpl,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend import envs
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import (
    _EXTRA_CTX,
    StagedSFAGraphKey,
    StagedSFAQueryProfile,
)
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel.common_cp import AscendPCPMetadata
from vllm_ascend.attention.mla_v1 import MAX_O_PROJ_PREFETCH_SIZE, MLAPO_MAX_SUPPORTED_TOKENS
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    ascend_chunked_prefill_workspace_size,
    enable_cp,
    get_lmcache_sparse_cached_tokens,
    maybe_save_kv_layer_to_connector,
    staged_sfa_connector_supports_sparse_load,
    trans_rope_weight,
    transdata,
    wait_for_kv_layer_from_connector,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.distributed.kv_transfer.sparse_offload import _prof as _dsa_prof
from vllm_ascend.distributed.kv_transfer.sparse_offload.prepare_sparse_indices import (
    prepare_sparse_indices,
)
from vllm_ascend.distributed.utils import all_gather_async
from vllm_ascend.ops.layer_shard_linear import (
    is_hidden_layer,
    post_process_after_loading_for_shard_weight_series,
    reach_layer_for_shard_weight_series,
    register_all_layers_to_shard_weight_series,
)
from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_mla
from vllm_ascend.ops.triton.rope import rope_forward_triton_siso
from vllm_ascend.quantization.methods import AscendW8A8LinearMethod
from vllm_ascend.utils import (
    ACL_FORMAT_FRACTAL_ND,
    StagedSFARouteAction,
    StagedSFARouteReason,
    _round_up,
    dispose_layer,
    enable_dsa_cp,
    enable_dsa_cp_with_layer_shard,
    enable_dsa_cp_with_o_proj_tp,
    get_weight_prefetch_method,
    maybe_trans_nz,
    staged_sfa_graph_capture_sizes,
    staged_sfa_graph_configured,
)
from vllm_ascend.worker.npu_input_batch import NPUInputBatch

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

# token count limits within bmm_transpose operator
BMM_TRANS_MAX_SUPPORTED_TOKENS = 1024
# Fence the first sparse load once in each worker process by default.
_LMCACHE_SPARSE_WAIT_SYNC_ONCE = os.getenv("VLLM_ASCEND_LMCACHE_SPARSE_WAIT_SYNC_ONCE", "1").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_lmcache_sparse_wait_sync_once_done = False
_lmcache_sparse_wait_sync_once_lock = Lock()


def _staged_sfa_profile_scope(name: str):
    if torch.autograd._profiler_enabled():
        return torch.profiler.record_function(name)
    return nullcontext()


@dataclass(frozen=True, slots=True)
class _TensorBinding:
    address: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device

    @property
    def layout(self) -> tuple[Any, ...]:
        return self.shape, self.stride, self.dtype, self.device

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "_TensorBinding":
        return cls(
            tensor.data_ptr(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.device,
        )


@dataclass(frozen=True, slots=True)
class _StagedSFALayerBinding:
    bridge: tuple[_TensorBinding, ...]
    kv_cache: tuple[_TensorBinding, ...]
    remap_boundary: _TensorBinding
    producer_event_id: int


@dataclass(slots=True)
class _StagedSFACaptureState:
    """Own the immutable capture contract for one local SFA layer."""

    producer_event: Any | None = None
    remap_boundary: torch.Tensor | None = None
    runtime: tuple[Any, ...] | None = None
    initialized_cache_capacity: int = 0
    bindings: dict[StagedSFAGraphKey, _StagedSFALayerBinding] = field(
        default_factory=dict,
    )

    def register(
        self,
        key: StagedSFAGraphKey,
        bridge: tuple[torch.Tensor, ...],
        kv_cache: tuple[torch.Tensor, ...],
    ) -> None:
        if key in self.bindings:
            raise RuntimeError(f"staged SFA graph key was captured twice: {key}")
        if self.producer_event is None or self.remap_boundary is None:
            raise RuntimeError("staged SFA capture storage is incomplete")

        binding = _StagedSFALayerBinding(
            tuple(_TensorBinding.from_tensor(tensor) for tensor in bridge),
            tuple(_TensorBinding.from_tensor(tensor) for tensor in kv_cache),
            _TensorBinding.from_tensor(self.remap_boundary),
            id(self.producer_event),
        )
        if self.bindings:
            existing = next(iter(self.bindings.values()))
            if (
                binding.kv_cache != existing.kv_cache
                or binding.producer_event_id != existing.producer_event_id
                or binding.remap_boundary.address
                != existing.remap_boundary.address
                or binding.remap_boundary.layout[1:]
                != existing.remap_boundary.layout[1:]
                or tuple(tensor.layout for tensor in binding.bridge)
                != tuple(tensor.layout for tensor in existing.bridge)
            ):
                raise RuntimeError(
                    "staged SFA capture bindings changed between graph keys"
                )
        self.bindings[key] = binding

    def seal(self, expected_keys: tuple[StagedSFAGraphKey, ...]) -> None:
        expected = frozenset(expected_keys)
        missing = expected.difference(self.bindings)
        unexpected = self.bindings.keys() - expected
        if (
            self.producer_event is None
            or self.remap_boundary is None
            or self.runtime is None
            or missing
            or unexpected
        ):
            raise RuntimeError(
                "staged SFA capture state is incomplete: "
                f"missing_keys={tuple(key.request_capacity for key in missing)}, "
                f"unexpected_keys={tuple(key.request_capacity for key in unexpected)}"
            )

def _sync_compute_stream_after_lmcache_sparse_wait() -> None:
    global _lmcache_sparse_wait_sync_once_done

    if not _LMCACHE_SPARSE_WAIT_SYNC_ONCE or _lmcache_sparse_wait_sync_once_done:
        return

    with _lmcache_sparse_wait_sync_once_lock:
        if _lmcache_sparse_wait_sync_once_done:
            return
        if not (hasattr(torch, "npu") and hasattr(torch.npu, "current_stream")):
            return

        torch.npu.current_stream().synchronize()
        _lmcache_sparse_wait_sync_once_done = True


def _dsa_topk_to_2d_indices(topk_indices: torch.Tensor) -> torch.Tensor:
    if topk_indices.dim() == 3 and topk_indices.shape[1] == 1:
        return topk_indices[:, 0, :]
    if topk_indices.dim() == 2:
        return topk_indices
    return topk_indices.reshape(topk_indices.shape[0], -1)


@lru_cache(maxsize=1)
def _decode_window_save_window_size() -> int:
    value = os.environ.get("LMCACHE_DECODE_WINDOW_SAVE_WINDOW_SIZE", "0")
    try:
        return max(0, int(value or 0))
    except ValueError:
        return 0


def _prepare_sfa_remap_boundary(
    attn_metadata: Any,
    request_ids: Any,
    *,
    is_dummy_run: bool,
    index_topk: int,
    cached_tokens: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Fill the stable Graph-A remap-boundary input once per step.

    Connector metadata and request/row mapping are host objects and therefore
    cannot be frozen into the captured runnable. Resolve them eagerly on CPU,
    then copy the final per-row boundary into the builder-owned NPU tensor.
    """
    boundary = attn_metadata.decode_remap_boundary
    if boundary is None:
        raise RuntimeError("[SFA sparse remap] boundary storage is unavailable.")
    if attn_metadata.decode_remap_boundary_ready:
        return boundary

    prompt_rows = attn_metadata.prompt_lens_cpu_rows
    row_req_indices = attn_metadata.decode_req_indices_cpu
    seq_lens_cpu = attn_metadata.seq_lens_cpu
    if prompt_rows is None or row_req_indices is None or seq_lens_cpu is None:
        raise RuntimeError("[SFA sparse remap] CPU metadata is incomplete.")

    prompt_rows_np = np.asarray(prompt_rows, dtype=np.int32).reshape(-1)
    row_req_indices_np = np.asarray(
        row_req_indices,
        dtype=np.int64,
    ).reshape(-1)
    seq_lens = [int(value) for value in seq_lens_cpu.tolist()]
    if int(boundary.numel()) != int(prompt_rows_np.size) or row_req_indices_np.size != prompt_rows_np.size:
        raise RuntimeError(
            "[SFA sparse remap] boundary shapes differ: "
            f"boundary={tuple(boundary.shape)}, "
            f"prompt_rows={tuple(prompt_rows_np.shape)}, "
            f"row_req_indices={tuple(row_req_indices_np.shape)}."
        )

    decode_request_indices = sorted(
        {int(request_index) for request_index in row_req_indices_np if int(request_index) >= 0}
    )
    for request_index in decode_request_indices:
        if request_index >= len(seq_lens):
            raise RuntimeError(
                "[SFA staged graph POC] decode row references request "
                f"{request_index}, but only {len(seq_lens)} sequence lengths "
                "are available."
            )

    cached_tokens_by_request: dict[int, int] = {}
    if not is_dummy_run:
        if cached_tokens is None:
            if decode_request_indices:
                if request_ids is None:
                    raise RuntimeError("[SFA sparse remap] active request IDs are unavailable.")
                request_ids = list(request_ids)
                if decode_request_indices[-1] >= len(request_ids):
                    raise RuntimeError("[SFA sparse remap] active request IDs do not cover all decode rows.")
                decode_request_ids = [request_ids[index] for index in decode_request_indices]
                resolved_tokens = get_lmcache_sparse_cached_tokens(decode_request_ids)
                cached_tokens_by_request = dict(zip(decode_request_indices, resolved_tokens, strict=True))
        else:
            if len(cached_tokens) != len(decode_request_indices):
                raise RuntimeError(
                    f"[SFA_ROUTE] action=fatal reason={StagedSFARouteReason.FRONTIER_COUNT_MISMATCH.value}"
                )
            cached_tokens_by_request = dict(zip(decode_request_indices, cached_tokens, strict=True))
    decode_window_size = _decode_window_save_window_size()
    boundary_rows = prompt_rows_np.copy()
    for row_index, request_index_value in enumerate(row_req_indices_np):
        request_index = int(request_index_value)
        if request_index < 0:
            continue
        cached_for_request = cached_tokens_by_request.get(request_index)
        if decode_window_size > 0:
            current_position = max(seq_lens[request_index] - 1, 0)
            row_boundary = current_position // decode_window_size * decode_window_size
            if cached_for_request is not None:
                row_boundary = min(row_boundary, cached_for_request)
            boundary_rows[row_index] = row_boundary
        elif cached_for_request is not None:
            boundary_rows[row_index] = cached_for_request

    _validate_dsa_scratch_capacity(
        boundary_rows,
        row_req_indices_np,
        getattr(attn_metadata, "decode_scratch_base_cpu", None),
        index_topk,
        getattr(attn_metadata, "decode_scratch_capacity", None),
    )

    boundary.copy_(torch.from_numpy(boundary_rows))
    attn_metadata.decode_remap_boundary_ready = True
    return boundary


def _validate_dsa_scratch_capacity(
    boundary_rows: Any,
    row_req_indices: Any,
    scratch_base_rows: Any,
    index_topk: int,
    scratch_capacity: int | None = None,
) -> None:
    """Validate that compact DSA scratch cannot alias live KV positions."""
    width = int(index_topk)
    if width <= 0:
        raise RuntimeError(f"DSA compact scratch requires a positive index_topk, got {width}.")

    boundaries = np.asarray(boundary_rows, dtype=np.int64).reshape(-1)
    request_rows = np.asarray(row_req_indices, dtype=np.int64).reshape(-1)
    if scratch_base_rows is None:
        scratch_bases = np.zeros_like(boundaries)
    else:
        scratch_bases = np.asarray(
            scratch_base_rows,
            dtype=np.int64,
        ).reshape(-1)
    if not (boundaries.size == request_rows.size == scratch_bases.size):
        raise RuntimeError(
            "DSA compact scratch metadata shapes differ: "
            f"boundaries={boundaries.size}, request_rows={request_rows.size}, "
            f"scratch_bases={scratch_bases.size}."
        )

    active_rows = np.flatnonzero(request_rows >= 0)
    bases_by_request: dict[int, set[int]] = {}
    for row_index_value in active_rows:
        row_index = int(row_index_value)
        request_index = int(request_rows[row_index])
        scratch_base = int(scratch_bases[row_index])
        boundary = int(boundaries[row_index])
        if scratch_base < 0 or scratch_base % width != 0:
            raise RuntimeError(
                "DSA compact scratch base must be non-negative and "
                f"index_topk-aligned: row={row_index}, "
                f"request={request_index}, scratch_base={scratch_base}, "
                f"index_topk={width}."
            )
        request_bases = bases_by_request.setdefault(request_index, set())
        if scratch_base in request_bases:
            raise RuntimeError(
                "DSA compact scratch rows for one request overlap: "
                f"row={row_index}, request={request_index}, "
                f"scratch_base={scratch_base}."
            )
        request_bases.add(scratch_base)

        required_boundary = scratch_base + width
        if scratch_capacity is not None and required_boundary > int(scratch_capacity):
            raise RuntimeError(
                "DSA compact scratch row exceeds its physical reservation: "
                f"row={row_index}, request={request_index}, "
                f"scratch_base={scratch_base}, index_topk={width}, "
                f"required_capacity={required_boundary}, "
                f"reserved_capacity={int(scratch_capacity)}."
            )
        if boundary < required_boundary:
            raise RuntimeError(
                "DSA compact scratch would alias live KV positions: "
                f"row={row_index}, request={request_index}, "
                f"boundary={boundary}, scratch_base={scratch_base}, "
                f"index_topk={width}, required_boundary="
                f"{required_boundary}."
            )


def _prepare_dsa_sparse_lmcache_payload(
    attn_metadata: Any,
    selected_packed: torch.Tensor,
    *,
    index_topk: int,
    validate_once: bool = False,
) -> tuple[torch.Tensor, list[str], torch.Tensor | None]:
    """Build and validate the row-aligned sparse LMCache retrieve payload."""
    if validate_once and getattr(attn_metadata, "staged_sfa_payload_validated", False) is True:
        return (
            selected_packed,
            attn_metadata.decode_request_ids_compact,
            attn_metadata.decode_target_slot_mapping,
        )

    if selected_packed.dim() != 2:
        raise RuntimeError(
            f"DSA sparse LMCache selected tokens must be rank 2, got shape={tuple(selected_packed.shape)}."
        )
    if int(selected_packed.shape[1]) != int(index_topk):
        raise RuntimeError(
            "DSA sparse LMCache selected-token width does not match "
            f"index_topk: selected={tuple(selected_packed.shape)}, "
            f"index_topk={int(index_topk)}."
        )

    valid_row_indices = getattr(
        attn_metadata,
        "decode_valid_row_indices",
        None,
    )
    if valid_row_indices is None:
        raise RuntimeError("DSA sparse LMCache payload has no valid-row mapping.")
    selected_for_wait = selected_packed
    if int(valid_row_indices.numel()) != int(selected_for_wait.shape[0]):
        raise RuntimeError(
            "DSA sparse LMCache payload row count differs from its valid-row "
            f"mapping: selected_rows={int(selected_for_wait.shape[0])}, "
            f"valid_rows={int(valid_row_indices.numel())}."
        )

    request_ids = getattr(
        attn_metadata,
        "decode_request_ids_compact",
        None,
    )
    if request_ids is None:
        raise RuntimeError("DSA sparse LMCache payload has no row-aligned request IDs.")
    num_rows = int(selected_for_wait.shape[0])
    if len(request_ids) != num_rows:
        raise RuntimeError(
            "DSA sparse LMCache payload row count differs from request IDs: "
            f"selected_rows={num_rows}, request_ids={len(request_ids)}."
        )

    target_slot_mapping = getattr(
        attn_metadata,
        "decode_target_slot_mapping",
        None,
    )
    scratch_base_compact = getattr(
        attn_metadata,
        "decode_scratch_base_compact",
        None,
    )
    if scratch_base_compact is not None and target_slot_mapping is None:
        raise RuntimeError("Row-specific DSA scratch requires an explicit target-slot mapping.")
    if target_slot_mapping is not None and tuple(target_slot_mapping.shape) != tuple(selected_for_wait.shape):
        raise RuntimeError(
            "DSA sparse LMCache target-slot shape differs from selected "
            f"tokens: targets={tuple(target_slot_mapping.shape)}, "
            f"selected={tuple(selected_for_wait.shape)}."
        )
    if validate_once:
        attn_metadata.staged_sfa_payload_validated = True
    return selected_for_wait, request_ids, target_slot_mapping


def _dsa_mask_padding_sparse_rows(
    topk_indices: torch.Tensor,
    row_req_indices: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep graph padding rows from referencing freed DSA logical blocks."""
    topk_2d = _dsa_topk_to_2d_indices(topk_indices)
    num_rows = int(topk_2d.shape[0])
    if row_req_indices is None:
        return topk_indices, topk_2d
    row_req_indices = row_req_indices[:num_rows].to(device=topk_indices.device)
    if int(row_req_indices.numel()) < num_rows:
        pad = torch.full(
            (num_rows - int(row_req_indices.numel()),),
            -1,
            dtype=row_req_indices.dtype,
            device=topk_indices.device,
        )
        row_req_indices = torch.cat((row_req_indices, pad), dim=0)
    padding_mask = row_req_indices < 0
    if not topk_indices.is_contiguous():
        topk_indices = topk_indices.contiguous()
        topk_2d = _dsa_topk_to_2d_indices(topk_indices)
    topk_2d.masked_fill_(padding_mask.reshape(-1, 1), 0)
    return topk_indices, topk_2d


def _dsa_build_target_slot_mapping(
    block_table: torch.Tensor,
    row_req_indices: torch.Tensor,
    scratch_base: torch.Tensor,
    width: int,
    block_size: int,
    *,
    scratch_capacity: int,
    position_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build per-row target slots for compact DSA scratch loads."""
    if width <= 0 or row_req_indices.numel() == 0:
        return torch.empty(
            (int(row_req_indices.numel()), max(width, 0)),
            dtype=torch.long,
            device=block_table.device,
        )

    if block_size <= 0:
        raise RuntimeError(f"DSA compact scratch block_size must be positive, got {block_size}.")
    table_capacity = int(block_table.shape[1]) * int(block_size)
    if scratch_capacity <= 0 or scratch_capacity > table_capacity:
        raise RuntimeError(
            "DSA compact scratch reservation exceeds the block-table "
            f"capacity: reserved_capacity={scratch_capacity}, "
            f"table_capacity={table_capacity}."
        )

    row_req_indices = row_req_indices.to(device=block_table.device, dtype=torch.long)
    scratch_base = scratch_base.to(device=block_table.device, dtype=torch.long)
    block_table_rows = block_table.index_select(0, row_req_indices).to(torch.long)
    if position_offsets is None:
        position_offsets = torch.arange(
            width,
            dtype=torch.long,
            device=block_table.device,
        )
    positions = scratch_base.reshape(-1, 1) + position_offsets[:width].reshape(1, -1)
    logical_blocks = positions // block_size
    offsets = positions % block_size
    # CPU metadata has already proved every logical block is inside both the
    # scheduler reservation and this table. Do not clamp an invalid block into
    # another row: gather must preserve the exact scratch destination.
    physical_blocks = block_table_rows.gather(1, logical_blocks)
    return physical_blocks * block_size + offsets


def _dsa_indexer_layer_name(layer_name: str) -> str:
    return layer_name.rsplit(".", 1)[0] + ".indexer.k_cache"


def _dsa_index_lmcache_enabled() -> bool:
    if envs.VLLM_ASCEND_DSA_DISABLE_INDEX_LMCACHE:
        return False
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return False
    connector = get_kv_transfer_group()
    return bool(getattr(connector, "supports_dsa_index_lmcache", False))


class AscendSFABackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        # HACK(Ronald1995): vllm `initialize_kv_cache` method in model runner v2 make
        # attention name assertion, we just set name to FLASH_ATTN to avoid assertion error.
        # rectify this when vllm disable the assertion.
        return "ASCEND_SFA" if not envs_vllm.VLLM_USE_V2_MODEL_RUNNER else "FLASH_ATTN"

    @staticmethod
    def get_builder_cls():
        if enable_cp():
            from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFACPMetadataBuilder

            return AscendSFACPMetadataBuilder
        return AscendSFAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int, head_size: int) -> tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_impl_cls() -> type["AscendSFAImpl"]:
        if enable_cp():
            from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFACPImpl

            return AscendSFACPImpl
        return AscendSFAImpl

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [128]


@dataclass
class DSACPContext:
    num_tokens: int
    num_tokens_pad: int
    local_start: int
    local_end: int
    local_end_with_pad: int
    slot_mapping_cp: torch.Tensor
    actual_seq_lengths_query: torch.Tensor
    actual_seq_lengths_key: torch.Tensor


@dataclass
class AscendSFAMetadata:
    """Metadata for MLACommon.

    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|
    num_actual_tokens: int  # Number of tokens excluding padding.
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    cum_query_lens: torch.Tensor
    block_table: torch.Tensor
    sin: torch.Tensor
    cos: torch.Tensor

    # For logging.
    num_input_tokens: int = 0  # Number of tokens including padding.
    # The dimension of the attention heads
    head_dim: int | None = None
    attn_mask: torch.Tensor = None
    # chunked prefill by default if no attn_states passed
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill
    dsa_cp_context: DSACPContext | None = None
    # DSA two-group mode: the indexer KV group's own block table / slot mapping.
    # None in single-group mode (indexer shares the latent's block ids).
    indexer_block_table: torch.Tensor | None = None
    indexer_slot_mapping: torch.Tensor | None = None
    reshape_cache_event: torch.npu.Event = None
    sfa_cp_metadata: AscendPCPMetadata | None = None
    num_decodes: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0

    # DSA latent offload (GLM5.1): request ids and prompt lengths per request, used to
    # key LMCache and to split the indexer top-k into prefill (LMCache) vs decode
    # (resident) sources. None unless latent offload is enabled.
    # HW-VERIFY: confirm the source — req_ids/prompt_lens live on the runner's
    # input_batch, not on CommonAttentionMetadata; the runner may need to thread them
    # in (see sparse_offload/INTEGRATION.md section B).
    req_ids: list[str] | None = None
    prompt_lens: torch.Tensor | None = None
    decode_req_indices: torch.Tensor | None = None
    decode_req_indices_cpu: Any = None
    decode_valid_row_indices: torch.Tensor | None = None
    decode_valid_rows_all: bool = False
    decode_req_indices_compact: torch.Tensor | None = None
    decode_req_indices_compact_cpu: Any = None
    decode_request_ids_compact: list[str] | None = None
    decode_row_offsets: torch.Tensor | None = None
    decode_scratch_base: torch.Tensor | None = None
    decode_scratch_base_compact: torch.Tensor | None = None
    decode_scratch_base_cpu: Any = None
    decode_scratch_capacity: int | None = None
    decode_target_slot_mapping: torch.Tensor | None = None
    need_sparse_lmcache_payload: bool = False
    staged_sfa_payload_validated: bool = False
    prompt_lens_cpu_rows: Any = None
    decode_remap_boundary: torch.Tensor | None = None
    decode_remap_boundary_ready: bool = False
    decode_remap_boundary_cpu_rows: Any = None


M = TypeVar("M", bound=AscendSFAMetadata)


class AscendSFAMetadataBuilder(MLACommonMetadataBuilder[AscendSFAMetadata]):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(
        self,
        kv_cache_spec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[AscendSFAMetadata] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            metadata_cls if metadata_cls is not None else AscendSFAMetadata,
            supports_dcp_with_varlen,
        )

        self.block_size = vllm_config.cache_config.block_size
        self.max_blocks = (vllm_config.model_config.max_model_len + self.block_size - 1) // self.block_size

        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, (
                f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"
            )
        self.reorder_batch_threshold = self.decode_threshold
        self.attn_mask_builder = AttentionMaskBuilder(self.device)
        self.rope_dim = self.model_config.hf_text_config.qk_rope_head_dim
        self.enable_dsa_cp = enable_dsa_cp()
        self.dsa_shrink_latent = int(envs.VLLM_ASCEND_DSA_SHRINK_LATENT) if envs.VLLM_ASCEND_DSA_UNBUNDLE else 0
        hf_config = self.model_config.hf_config
        hf_text_config = self.model_config.hf_text_config
        self.index_topk = int(
            getattr(
                hf_text_config or hf_config,
                "topk_tokens",
                getattr(hf_text_config or hf_config, "index_topk", 2048),
            )
        )
        self._dsa_target_position_offsets = (
            torch.arange(self.index_topk, dtype=torch.long, device=device) if self.dsa_shrink_latent else None
        )

        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.actual_seq_lengths_query = torch.zeros(max_num_reqs + 1, dtype=torch.int32, device=device)
        self.actual_seq_lengths_key = torch.empty_like(self.actual_seq_lengths_query)
        # Staged SHRINK_LATENT=2 graph input. The address must survive metadata
        # rebuilds across decode steps, so keep one builder-owned device buffer
        # and overwrite only its contents before Graph A replay.
        self.decode_remap_boundary = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=device,
        )
        self.decode_valid_row_indices = torch.empty_like(self.decode_remap_boundary)
        self.decode_scratch_base = torch.empty_like(self.decode_remap_boundary)

    @staticmethod
    def determine_chunked_prefill_workspace_size(vllm_config: VllmConfig) -> int:
        return ascend_chunked_prefill_workspace_size(vllm_config)

    @classmethod
    def get_cudagraph_support(
        cls: type["AscendSFAMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # Explicit override in case the underlying builder specialized this getter.
        # @override omitted only because of mypy limitation due to type variable.
        return AttentionCGSupport.UNIFORM_BATCH

    def reorder_batch(self, input_batch: "NPUInputBatch", scheduler_output: "SchedulerOutput") -> bool:
        # No need to reorder for Ascend SFA
        return False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendSFAMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        num_input_tokens = common_attn_metadata.num_input_tokens

        block_table = common_attn_metadata.block_table_tensor[:num_reqs]
        slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]
        input_positions = common_attn_metadata.positions[:num_input_tokens].long()

        # DSA two-group mode: mirror the indexer group's table/slots (same
        # slicing as the latent's) so the impl can address the indexer cache.
        indexer_block_table = None
        indexer_slot_mapping = None
        if common_attn_metadata.indexer_block_table_tensor is not None:
            indexer_block_table = common_attn_metadata.indexer_block_table_tensor[:num_reqs]
            indexer_slot_mapping = common_attn_metadata.indexer_slot_mapping[:num_input_tokens]

        # DSA shrink-latent: expand per-request prompt lengths to per-row cache
        # boundaries for sparse-index preparation. Decode rows start at the
        # prompt length by default;
        # decode-window mode later replaces those rows with current_window_start.
        # Prefill and padding rows get 0 and stay untouched by the remap.
        prompt_lens_rows = None
        decode_req_indices_rows = None
        decode_valid_row_indices = None
        decode_valid_rows_all = False
        decode_req_indices_compact = None
        decode_req_indices_compact_cpu = None
        decode_request_ids_compact = None
        decode_row_offsets_rows = None
        decode_scratch_base_rows = None
        decode_scratch_base_compact = None
        decode_scratch_capacity = (
            (self.index_topk * self.decode_threshold + self.block_size - 1) // self.block_size * self.block_size
        )
        decode_target_slot_mapping = None
        need_sparse_lmcache_payload = False
        num_decode_rows = 0
        plens_cpu = common_attn_metadata.prompt_lens_cpu if self.dsa_shrink_latent else None
        if plens_cpu is not None:
            rows = np.zeros(num_input_tokens, dtype=np.int32)
            req_rows = np.full(num_input_tokens, -1, dtype=np.int32)
            row_offsets = np.zeros(num_input_tokens, dtype=np.int32)
            n_real = min(len(plens_cpu), num_reqs)
            qsl = common_attn_metadata.query_start_loc_cpu[: n_real + 1].numpy()
            computed = common_attn_metadata.num_computed_tokens_cpu[:n_real].numpy()
            for r in range(n_real):
                s, e = int(qsl[r]), int(qsl[r + 1])
                plen = int(plens_cpu[r])
                first_decode = max(s, s + plen - int(computed[r]))
                if first_decode < e:
                    count = e - first_decode
                    offsets = np.arange(count, dtype=np.int32)
                    rows[first_decode:e] = plen
                    req_rows[first_decode:e] = r
                    row_offsets[first_decode:e] = offsets
            num_decode_rows = int(np.count_nonzero(req_rows >= 0))
            prompt_lens_rows = torch.from_numpy(rows).to(block_table.device)
            decode_req_indices_rows = torch.from_numpy(req_rows).to(block_table.device)
            scratch_base_np = row_offsets.astype(np.int32) * self.index_topk
            # Plain decode has one row per request and uses the legacy per-request
            # sparse slot mapping. Only MTP/spec rows need disjoint scratch bases
            # and explicit target-slot tensors.
            needs_row_scratch_base = bool(np.any(scratch_base_np))
            if num_decode_rows > 0:
                self.decode_scratch_base[:num_input_tokens].copy_(torch.from_numpy(scratch_base_np))
                decode_scratch_base_rows = self.decode_scratch_base[:num_input_tokens]
            if needs_row_scratch_base:
                decode_row_offsets_rows = torch.from_numpy(row_offsets).to(block_table.device)
            need_sparse_lmcache_payload = self.dsa_shrink_latent != 3 and staged_sfa_connector_supports_sparse_load()
            valid_row_indices_np = np.flatnonzero(req_rows >= 0).astype(np.int32)
            if valid_row_indices_np.size:
                decode_valid_rows_all = int(valid_row_indices_np.size) == int(num_input_tokens)
                valid_req_indices_np = req_rows[valid_row_indices_np].astype(np.int64)
                valid_scratch_base_np = scratch_base_np[valid_row_indices_np]
                decode_req_indices_compact_cpu = valid_req_indices_np
                req_ids = common_attn_metadata.request_ids
                if req_ids is not None:
                    decode_request_ids_compact = [req_ids[int(req_idx)] for req_idx in valid_req_indices_np]
                # The fused kernel assigns each complete source row to one AIV.
                valid_row_count = int(valid_row_indices_np.size)
                self.decode_valid_row_indices[:valid_row_count].copy_(torch.from_numpy(valid_row_indices_np))
                decode_valid_row_indices = self.decode_valid_row_indices[:valid_row_count]
                decode_req_indices_compact = torch.from_numpy(valid_req_indices_np).to(block_table.device)
                if needs_row_scratch_base:
                    required_capacity = int(valid_scratch_base_np.max()) + self.index_topk
                    if required_capacity > decode_scratch_capacity:
                        raise RuntimeError(
                            "DSA compact scratch rows exceed the scheduler "
                            "reservation: required_capacity="
                            f"{required_capacity}, reserved_capacity="
                            f"{decode_scratch_capacity}, "
                            f"decode_threshold={self.decode_threshold}."
                        )
                    decode_scratch_base_compact = torch.from_numpy(valid_scratch_base_np).to(block_table.device)
                    decode_target_slot_mapping = _dsa_build_target_slot_mapping(
                        block_table,
                        decode_req_indices_compact,
                        decode_scratch_base_compact,
                        self.index_topk,
                        self.block_size,
                        scratch_capacity=decode_scratch_capacity,
                        position_offsets=self._dsa_target_position_offsets,
                    )

        cum_query_lens = common_attn_metadata.query_start_loc[1 : num_reqs + 1]
        seq_lens = common_attn_metadata.seq_lens[:num_reqs]
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu[:num_reqs]

        cos, sin = get_cos_and_sin_mla(input_positions, True)

        dsa_cp_context = None
        if self.enable_dsa_cp:
            global_tp_size = get_tp_group().world_size
            num_tokens = num_input_tokens
            num_tokens_pad = _round_up(num_tokens, global_tp_size)
            num_tokens_per_device = num_tokens_pad // global_tp_size
            local_start = get_tp_group().rank_in_group * num_tokens_per_device
            local_end_with_pad = local_start + num_tokens_per_device
            local_end = min(local_end_with_pad, num_actual_tokens)

            pad_size = num_tokens_pad - cos.shape[0]
            assert cos.shape == sin.shape, f"cos.shape must be equal to sin.shape, got {cos.shape} and {sin.shape}"

            if pad_size > 0:
                cos = nn.functional.pad(cos, (0, 0, 0, 0, 0, 0, 0, pad_size))
                sin = nn.functional.pad(sin, (0, 0, 0, 0, 0, 0, 0, pad_size))

            pad_size_slot = num_tokens_pad - slot_mapping.shape[0]
            if pad_size_slot > 0:
                slot_mapping = nn.functional.pad(slot_mapping, (0, pad_size_slot), value=-1)
            else:
                slot_mapping = slot_mapping[:num_tokens_pad]
            slot_mapping_cp = slot_mapping[local_start:local_end_with_pad]

            cos = cos[local_start:local_end_with_pad]
            sin = sin[local_start:local_end_with_pad]

            assert cos.shape[0] == num_tokens_per_device, (
                f"cos.shape[0] must be equal to num_tokens_per_device, \
                    got {cos.shape[0]} and {num_tokens_per_device}"
            )
            assert slot_mapping_cp.shape[0] == num_tokens_per_device, (
                f"slot_mapping_cp.shape[0] must be equal to num_tokens_per_device, \
                    got {slot_mapping_cp.shape[0]} and {num_tokens_per_device}"
            )
            assert slot_mapping.shape[0] == num_tokens_pad, (
                f"slot_mapping.shape[0] must be equal to num_tokens_pad, \
                    got {slot_mapping.shape[0]} and {num_tokens_pad}"
            )

            actual_seq_lengths_query = self.actual_seq_lengths_query
            actual_seq_lengths_key = self.actual_seq_lengths_key

            num_segs = cum_query_lens.shape[0]
            last_token = 0
            cum = 0
            for i in range(0, num_segs):
                global_start = last_token
                global_end = cum_query_lens[i].item()
                last_token = global_end

                req_local_start = max(global_start, local_start)
                req_local_end = min(global_end, local_end_with_pad)
                num_local_tokens = req_local_end - req_local_start

                if num_local_tokens > 0:
                    cum += num_local_tokens
                    actual_seq_lengths_query[i] = cum

                    offset = global_end - req_local_end
                    actual_seq_lengths_key[i] = seq_lens[i].item() - offset
                else:
                    actual_seq_lengths_query[i] = cum
                    actual_seq_lengths_key[i] = 0

            actual_seq_lengths_query = actual_seq_lengths_query[:num_reqs]
            actual_seq_lengths_key = actual_seq_lengths_key[:num_reqs]

            dsa_cp_context = DSACPContext(
                num_tokens=num_tokens,
                num_tokens_pad=num_tokens_pad,
                local_start=local_start,
                local_end=local_end,
                local_end_with_pad=local_end_with_pad,
                slot_mapping_cp=slot_mapping_cp,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
            )

        return self.metadata_cls(  # type: ignore
            num_input_tokens=common_attn_metadata.num_input_tokens,
            num_actual_tokens=num_actual_tokens,
            cum_query_lens=cum_query_lens,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            slot_mapping=slot_mapping,
            head_dim=self.model_config.get_head_size(),
            attn_mask=self.attn_mask_builder.get_attention_mask(self.model_config),
            attn_state=common_attn_metadata.attn_state,
            block_table=block_table,
            sin=sin[:num_input_tokens],
            cos=cos[:num_input_tokens],
            dsa_cp_context=dsa_cp_context,
            indexer_block_table=indexer_block_table,
            indexer_slot_mapping=indexer_slot_mapping,
            # DSA latent offload: best-effort; getattr -> None when not threaded in yet
            # (harmless unless the feature is enabled). HW-VERIFY the real source.
            req_ids=getattr(common_attn_metadata, "request_ids", None),
            prompt_lens=prompt_lens_rows,
            decode_req_indices=decode_req_indices_rows,
            decode_req_indices_cpu=req_rows if decode_req_indices_rows is not None else None,
            decode_valid_row_indices=decode_valid_row_indices,
            decode_valid_rows_all=decode_valid_rows_all,
            decode_req_indices_compact=decode_req_indices_compact,
            decode_req_indices_compact_cpu=decode_req_indices_compact_cpu,
            decode_request_ids_compact=decode_request_ids_compact,
            decode_row_offsets=decode_row_offsets_rows,
            decode_scratch_base=decode_scratch_base_rows,
            decode_scratch_base_compact=decode_scratch_base_compact,
            decode_scratch_base_cpu=(scratch_base_np if plens_cpu is not None else None),
            decode_scratch_capacity=decode_scratch_capacity,
            decode_target_slot_mapping=decode_target_slot_mapping,
            need_sparse_lmcache_payload=need_sparse_lmcache_payload,
            prompt_lens_cpu_rows=rows if plens_cpu is not None else None,
            decode_remap_boundary=self.decode_remap_boundary[:num_input_tokens],
            decode_remap_boundary_ready=False,
            num_decode_tokens=num_decode_rows,
        )

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
    ):
        if attn_state in {AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding}:
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
        else:
            raise NotImplementedError("Currently we only support building dummy metadata for DecodeOnly state")

        attn_metadata.attn_state = attn_state
        return attn_metadata


class AscendSFAImpl(MLAAttentionImpl):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    # Supports forward using the all-gather o_proj weight for decode requests when Sharded CP is enabled.
    o_proj_full_pool: torch.Tensor | None = None

    # q_hadamard and k_hadamard tensor shared when dsa c8 enabled
    q_hadamard: torch.Tensor | None = None
    k_hadamard: torch.Tensor | None = None

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        # MLA Args
        self.q_lora_rank = kwargs["q_lora_rank"]
        self.kv_lora_rank = kwargs["kv_lora_rank"]
        self.qk_nope_head_dim = kwargs["qk_nope_head_dim"]
        self.qk_rope_head_dim = kwargs["qk_rope_head_dim"]
        self.qk_head_dim = kwargs["qk_head_dim"]
        self.v_head_dim = kwargs["v_head_dim"]
        self.rotary_emb = kwargs["rotary_emb"]
        self.q_proj = kwargs["q_proj"] if self.q_lora_rank is None else kwargs["q_b_proj"]
        self.fused_qkv_a_proj = kwargs.get("fused_qkv_a_proj")
        self.kv_b_proj = kwargs["kv_b_proj"]
        self.o_proj = kwargs["o_proj"]
        self.indexer = kwargs["indexer"]
        self.kv_a_proj_with_mqa = kwargs.get("kv_a_proj_with_mqa")
        self.kv_a_layernorm = kwargs.get("kv_a_layernorm")
        self.q_a_layernorm = kwargs.get("q_a_layernorm")
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tp_group().rank_in_group
        self.q_b_proj = kwargs["q_b_proj"]

        ascend_config = get_ascend_config()
        self.enable_shared_expert_dp = ascend_config.enable_shared_expert_dp

        # The MLAPO operator fuses the pre-processing steps on Q/K/V in MLA into a single operator
        # NOTE: it imposes a limit on the number of input tokens and conflicts with FlashComm
        self.enable_mlapo = envs.VLLM_ASCEND_ENABLE_MLAPO

        assert self.indexer is not None, "Indexer is required for DSA."

        self.local_num_heads = self.num_heads
        self.vllm_config = get_current_vllm_config()
        self.is_kv_producer = (
            self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.is_kv_producer
        )

        # indexer param
        self.n_head: int = self.indexer.n_head  # 64
        self.head_dim: int = self.indexer.head_dim  # 128
        hf_config = self.vllm_config.model_config.hf_config
        hf_text_config = getattr(self.vllm_config.model_config, "hf_text_config", None)
        self.index_topk = int(
            getattr(
                self.indexer,
                "topk_tokens",
                getattr(hf_text_config or hf_config, "index_topk", 2048),
            )
        )
        self.wq_b = self.indexer.wq_b
        self.wk = self.indexer.wk
        self.weights_proj = self.indexer.weights_proj
        self.k_norm = self.indexer.k_norm
        self.cp_size = 1
        self.is_rope_neox_style = True
        self.use_torch_npu_lightning_indexer = False
        if self.vllm_config.model_config.hf_config.model_type in ["glm_moe_dsa"]:
            self.is_rope_neox_style = False
            self.use_torch_npu_lightning_indexer = True

        # DSA latent offload Route-1 pragmatic (M-B): latent written to the
        # PagedLatentPool instead of the (shrunk) vLLM paged latent cache.
        self.dsa_offload_free_paged = bool(
            envs.VLLM_ASCEND_ENABLE_DSA_LATENT_OFFLOAD and envs.VLLM_ASCEND_DSA_OFFLOAD_FREE_PAGED
        )
        self.dsa_offload_unbundle = bool(envs.VLLM_ASCEND_DSA_UNBUNDLE)
        # Step B staging (1 = B2 compact-scratch read; 2 = +B1 freeing).
        self.dsa_shrink_latent = int(envs.VLLM_ASCEND_DSA_SHRINK_LATENT) if self.dsa_offload_unbundle else 0
        self.enable_staged_sfa_graph = staged_sfa_graph_configured(self.vllm_config)
        self._staged_sfa_graph_capture_sizes = (
            staged_sfa_graph_capture_sizes(self.vllm_config) if self.enable_staged_sfa_graph else ()
        )
        self._staged_sfa_capture_state = _StagedSFACaptureState()
        self._staged_sfa_row_indices: torch.Tensor | None = None
        # dsa c8
        self.use_sparse_c8_indexer = ascend_config.enable_sparse_c8
        if self.use_sparse_c8_indexer:
            self.c8_k_cache_dtype = torch.int8
            self.c8_k_scale_cache_dtype = torch.float16

        # Effective in SFA when FlashComm is enabled.
        self.enable_dsa_cp = enable_dsa_cp()

        # Enable layer sharding via DSA-CP on the P node in the PD-disaggregated setup.
        self.enable_dsa_cp_with_layer_shard = enable_dsa_cp_with_layer_shard()

        # Improves glm5 accuracy after enabling dsa-cp in scenarios with strict accuracy requirements,
        # especially for customized cases, at the cost of performance degradation due to extra communication.
        self.enable_dsa_cp_strict_accuracy = (
            self.enable_dsa_cp_with_layer_shard
            and self.vllm_config.model_config.hf_config.model_type in ["glm_moe_dsa"]
        )

        # use original TP o_proj weight in PD mix stage, and full gather
        # for o_proj weight for prefill stage.
        self.enable_dsa_cp_with_o_proj_tp = enable_dsa_cp_with_o_proj_tp()

        if self.enable_dsa_cp:
            self.local_num_heads = self.num_heads * self.tp_size
            if self.enable_dsa_cp_with_layer_shard:
                self.layer_sharding_kwargs = []
                for layer_name in get_ascend_config().layer_sharding or []:
                    if layer_name in kwargs:
                        self.layer_sharding_kwargs.append(kwargs[layer_name])
                    else:
                        logger.warning_once(
                            f"[SFAImpl init] Layer '{layer_name}' not found in kwargs for layer sharding, "
                            "skipping sharding configuration"
                        )
                register_all_layers_to_shard_weight_series(self.layer_sharding_kwargs)

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        # NOTE: We currently do not support quant kv_b_proj.
        assert isinstance(self.kv_b_proj.quant_method, UnquantizedLinearMethod)
        # NOTE: Weight will be reshaped next, we need to revert and transpose it.
        kv_b_proj_weight = torch_npu.npu_format_cast(self.kv_b_proj.weight.data, ACL_FORMAT_FRACTAL_ND).T
        assert kv_b_proj_weight.shape == (
            self.kv_lora_rank,
            self.local_num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        ), (
            f"{kv_b_proj_weight.shape=}, "
            f"{self.kv_lora_rank=}, "
            f"{self.local_num_heads=}, "
            f"{self.qk_nope_head_dim=}, "
            f"{self.v_head_dim=}"
        )
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.local_num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        W_UK, W_UV = kv_b_proj_weight.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # NOTE: When we make a incontiguous weight contiguous, a new address will be allocated for the weight,
        # in graph + RL scenario, we only capture the graph once, and the weight address is expected to be the same
        # across iterations, so we need to copy the weight to the original address after making it contiguous.
        if not hasattr(self, "W_UV"):
            # Convert from (L, N, V) to (N, L, V)
            self.W_UV = W_UV.transpose(0, 1).contiguous()
            # Convert from (L, N, P) to (N, P, L)
            self.W_UK_T = W_UK.permute(1, 2, 0).contiguous()
        else:
            self.W_UV.copy_(W_UV.transpose(0, 1).contiguous())
            self.W_UK_T.copy_(W_UK.permute(1, 2, 0).contiguous())

        # TODO(zzzzwwjj): Currently, torch.ops._C_ascend.batch_matmul_transpose cannot support weight nz
        # self.W_UV = maybe_trans_nz(self.W_UV)

        # Dispose kv_b_proj since it is replaced by W_UV and W_UK_T to save memory
        dispose_layer(self.kv_b_proj)
        if self.enable_dsa_cp:
            if self.enable_dsa_cp_with_layer_shard:
                for layer in self.layer_sharding_kwargs or []:
                    if is_hidden_layer(layer):
                        post_process_after_loading_for_shard_weight_series(layer)
            else:
                self._init_o_proj_tp_full_params()

        if self.enable_mlapo:
            quant_method = getattr(
                getattr(self.fused_qkv_a_proj, "quant_method", None),
                "quant_method",
                None,
            )
            reasons = []
            if self.fused_qkv_a_proj is None or not isinstance(quant_method, AscendW8A8LinearMethod):
                reasons.append(
                    "Currently mlapo only supports W8A8 quantization in SFA scenario."
                    "Some layers in your model are not quantized with W8A8,"
                    "thus mlapo is disabled for these layers."
                )
            if self.enable_dsa_cp:
                reasons.append("Currently mlapo does not support SFA with CP,thus mlapo is disabled for these layers.")
            if reasons:
                self.enable_mlapo = False
                for msg in reasons:
                    logger.warning_once(msg)
            else:
                self._process_weights_for_fused_mlapo(act_dtype)
        if not self.enable_mlapo:
            # if mlapo, W_UK_T can't trans nz
            self.W_UK_T = maybe_trans_nz(self.W_UK_T)

        if self.use_sparse_c8_indexer and AscendSFAImpl.q_hadamard is None:
            AscendSFAImpl.q_hadamard = torch.tensor(scipy.linalg.hadamard(128), dtype=torch.bfloat16, device="npu") / (
                128**0.5
            )
        if self.use_sparse_c8_indexer and AscendSFAImpl.k_hadamard is None:
            AscendSFAImpl.k_hadamard = torch.tensor(scipy.linalg.hadamard(128), dtype=torch.bfloat16, device="npu") / (
                128**0.5
            )

    # Processing the input parameters for MLAPO by reordering and transposing
    # QKV(and part of Q) weight, applying RoPE-related dimension transformations,
    # and handling quantization parameters.
    def _process_weights_for_fused_mlapo(self, act_dtype: torch.dtype):
        assert self.kv_a_proj_with_mqa is None
        assert self.fused_qkv_a_proj is not None

        kv_a_proj_wt = self.fused_qkv_a_proj.weight.data[..., self.q_lora_rank :].contiguous()
        q_a_proj_wt = self.fused_qkv_a_proj.weight.data[..., : self.q_lora_rank].contiguous()

        kv_a_proj_wt = kv_a_proj_wt.t().contiguous()
        kv_a_proj_wt = trans_rope_weight(kv_a_proj_wt, self.qk_rope_head_dim)
        kv_a_proj_wt = kv_a_proj_wt.t().contiguous()
        wd_qkv = torch.cat((kv_a_proj_wt, q_a_proj_wt), dim=-1)
        wd_qkv = wd_qkv.t().contiguous()
        wd_qkv = transdata(wd_qkv, block_size=(16, 32)).unsqueeze(0).contiguous()
        self.wd_qkv = torch_npu.npu_format_cast(wd_qkv, 29)

        kv_a_proj_deq_scl = self.fused_qkv_a_proj.deq_scale[self.q_lora_rank :].contiguous()
        q_a_proj_deq_scl = self.fused_qkv_a_proj.deq_scale[: self.q_lora_rank].contiguous()
        kv_a_proj_deq_scl = kv_a_proj_deq_scl.reshape(self.kv_lora_rank + self.qk_rope_head_dim, -1).contiguous()
        kv_a_proj_deq_scl = trans_rope_weight(kv_a_proj_deq_scl, self.qk_rope_head_dim)
        kv_a_proj_deq_scl = kv_a_proj_deq_scl.view(self.kv_lora_rank + self.qk_rope_head_dim).contiguous()
        self.deq_scale_qkv = torch.cat((kv_a_proj_deq_scl, q_a_proj_deq_scl), dim=-1).contiguous()

        kv_a_proj_qt_bias = self.fused_qkv_a_proj.quant_bias[self.q_lora_rank :].contiguous()
        q_a_proj_qt_bias = self.fused_qkv_a_proj.quant_bias[: self.q_lora_rank].contiguous()

        kv_a_proj_qt_bias = kv_a_proj_qt_bias.reshape(self.kv_lora_rank + self.qk_rope_head_dim, -1).contiguous()
        kv_a_proj_qt_bias = trans_rope_weight(kv_a_proj_qt_bias, self.qk_rope_head_dim)
        kv_a_proj_qt_bias = kv_a_proj_qt_bias.view(self.kv_lora_rank + self.qk_rope_head_dim).contiguous()
        self.quant_bias_qkv = torch.cat((kv_a_proj_qt_bias, q_a_proj_qt_bias), dim=-1).contiguous()

        wu_q = self.q_proj.weight.data
        wu_q = wu_q.t().reshape(self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim, -1)
        wu_q = trans_rope_weight(wu_q, self.qk_rope_head_dim)
        wu_q = wu_q.reshape(self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim), -1)
        wu_q = transdata(wu_q, block_size=(16, 32)).unsqueeze(0).contiguous()
        self.wu_q = torch_npu.npu_format_cast(wu_q, 29)

        qb_deq_scl = self.q_proj.deq_scale.data
        qb_deq_scl = qb_deq_scl.reshape(self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim, -1)
        qb_deq_scl = trans_rope_weight(qb_deq_scl, self.qk_rope_head_dim)
        self.qb_deq_scl = qb_deq_scl.reshape(self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim))

        qb_qt_bias = self.q_proj.quant_bias.data
        qb_qt_bias = qb_qt_bias.reshape(self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim, -1)
        qb_qt_bias = trans_rope_weight(qb_qt_bias, self.qk_rope_head_dim)
        self.qb_qt_bias = qb_qt_bias.reshape(self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim))

        device = self.q_proj.weight.device
        self.gamma1 = self.q_a_layernorm.weight.data  # type: ignore[union-attr]
        self.beta1 = self.q_a_layernorm.bias.data  # type: ignore[union-attr]
        self.gamma2 = self.kv_a_layernorm.weight.data  # type: ignore[union-attr]
        self.quant_scale0 = self.fused_qkv_a_proj.input_scale.data
        self.quant_offset0 = self.fused_qkv_a_proj.input_offset.data
        self.quant_scale1 = self.q_proj.input_scale.data
        self.quant_offset1 = self.q_proj.input_offset.data
        self.ctkv_scale = torch.tensor([1], dtype=act_dtype, device=device)
        self.q_nope_scale = torch.tensor([1], dtype=act_dtype, device=device)

        # On KV consumers (decode-only) MLAPO uses the transformed weights built above;
        # the original fused_qkv_a_proj/q_proj weights and quant params are no longer
        # referenced, so drop them to save memory.
        if (
            self.vllm_config.kv_transfer_config is not None
            and self.vllm_config.kv_transfer_config.is_kv_consumer
            and self.vllm_config.scheduler_config.max_num_batched_tokens <= MLAPO_MAX_SUPPORTED_TOKENS
        ):
            self.fused_qkv_a_proj.weight = None
            self.fused_qkv_a_proj.deq_scale = None
            self.fused_qkv_a_proj.quant_bias = None
            self.q_proj.weight = None
            self.q_proj.deq_scale = None
            self.q_proj.quant_bias = None
            torch.npu.empty_cache()

    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: M,
        k_scale: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        raise NotImplementedError("forward_mha is not supported for SFA attention. Use forward() instead.")

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: M,
        layer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError("forward_mqa is not supported for SFA attention. Use forward() instead.")

    def rope_single(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D = x.shape
        S = 1
        x = x.view(B, N, S, D)
        x = torch_npu.npu_interleave_rope(x, cos, sin)
        return x.view(B, N, D)

    def _init_o_proj_tp_full_params(self):
        """
        Initialize TP-mode and Full-mode parameters for o_proj weight,
        preparing for weight switching in PD mix stage.

        For PD mix stage:
        - Use original TP o_proj weight for decode phase
        - Need full-gather o_proj weight from all TP ranks for prefill phase
        """
        if AscendSFAImpl.o_proj_full_pool is None:
            sample = self.o_proj.weight
            AscendSFAImpl.o_proj_full_pool = torch.empty(
                (sample.shape[0] * self.tp_size, sample.shape[1]), dtype=sample.dtype, device=sample.device
            )

        # Save TP-mode parameters (original sharded weights)
        self.o_proj_tp_weight = self.o_proj.weight.clone().detach()
        self.o_proj_tp_aclnn_input_scale = self.o_proj.aclnn_input_scale.clone().detach()
        self.o_proj_tp_aclnn_input_scale_reciprocal = self.o_proj.aclnn_input_scale_reciprocal.clone().detach()
        self.o_proj_tp_aclnn_input_offset = self.o_proj.aclnn_input_offset.clone().detach()

        # Initially switch to TP mode for graph capture
        self.o_proj.weight.set_(self.o_proj_tp_weight)
        self.o_proj.aclnn_input_scale.set_(self.o_proj_tp_aclnn_input_scale)
        self.o_proj.aclnn_input_scale_reciprocal.set_(self.o_proj_tp_aclnn_input_scale_reciprocal)
        self.o_proj.aclnn_input_offset.set_(self.o_proj_tp_aclnn_input_offset)

        # Precompute Full-mode quantization parameters by repeating TP parameters across all TP ranks
        self.o_proj_full_aclnn_input_scale = self.o_proj.aclnn_input_scale.repeat(self.tp_size)
        self.o_proj_full_aclnn_input_scale_reciprocal = self.o_proj.aclnn_input_scale_reciprocal.repeat(self.tp_size)
        self.o_proj_full_aclnn_input_offset = self.o_proj.aclnn_input_offset.repeat(self.tp_size)

    def _handle_o_proj_weight_switch_and_forward(
        self,
        attn_output: torch.Tensor,
        output: torch.Tensor,
        o_proj_full_handle: torch.distributed.Work | None,
        should_shard_weight: bool,
    ) -> tuple[torch.Tensor, bool]:
        """
        Handle o_proj weight switching between TP-mode and Full-mode, and execute forward computation.
        """
        # Gather o_proj weight from all TP ranks for Full-mode computation
        if should_shard_weight:
            # Wait for the completion of o_proj weight all-gather operation
            if o_proj_full_handle is not None:
                o_proj_full_handle.wait()

            # Switch o_proj to Full-mode (gathered weight from all TP ranks)
            self.o_proj.weight.set_(AscendSFAImpl.o_proj_full_pool)
            self.o_proj.aclnn_input_scale.set_(self.o_proj_full_aclnn_input_scale)
            self.o_proj.aclnn_input_scale_reciprocal.set_(self.o_proj_full_aclnn_input_scale_reciprocal)
            self.o_proj.aclnn_input_offset.set_(self.o_proj_full_aclnn_input_offset)

            # Apply quantization method and execute forward computation
            output[...] = self.o_proj.quant_method.quant_method.apply(self.o_proj, attn_output)

            # Switch o_proj back to TP-mode for subsequent decode operations
            self.o_proj.weight.set_(self.o_proj_tp_weight)
            self.o_proj.aclnn_input_scale.set_(self.o_proj_tp_aclnn_input_scale)
            self.o_proj.aclnn_input_scale_reciprocal.set_(self.o_proj_tp_aclnn_input_scale_reciprocal)
            self.o_proj.aclnn_input_offset.set_(self.o_proj_tp_aclnn_input_offset)

            return output, False
        else:
            # For decode scenario: perform all-to-all communication on o_proj input activations
            # Reshape for all-to-all: [batch * seq, tp_size, head_dim] -> [tp_size, batch * seq, head_dim]
            send = (
                attn_output.view(-1, self.tp_size, self.num_heads * self.v_head_dim)
                .permute(1, 0, 2)
                .reshape(-1, self.num_heads * self.v_head_dim)
            )

            attn_output = torch.empty_like(send)
            torch.distributed.all_to_all_single(attn_output, send, group=get_tp_group().device_group)

            return attn_output, True

    def _get_full_kv(self, k, attn_metadata):
        return k

    def exec_kv(
        self,
        kv_no_split: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple,
        slots: torch.Tensor,
        attn_metadata: M,
    ):
        B = kv_no_split.shape[0]
        N = self.num_kv_heads
        S = 1
        # npu_kv_rmsnorm_rope_cache needs [B, N, S, D]
        kv_no_split = kv_no_split.view(B, N, S, self.kv_lora_rank + self.qk_rope_head_dim)
        cache_mode = "PA"

        if self.enable_dsa_cp:
            _, _, k_pe, k_nope = torch_npu.npu_kv_rmsnorm_rope_cache(
                kv_no_split,
                self.kv_a_layernorm.weight,  # type: ignore[union-attr]
                cos,
                sin,
                slots.to(torch.int64),
                kv_cache[1],
                kv_cache[0],
                epsilon=self.kv_a_layernorm.variance_epsilon,  # type: ignore[union-attr]
                cache_mode=cache_mode,
                is_output_kv=True,
            )
            return k_pe, k_nope
        else:
            # is_output_kv=True returns the freshly-computed latent (k_pe, k_nope) in
            # addition to caching it, so the DSA-offload forward can store/gather it
            # without a paged read-back. The op still caches to kv_cache[0]/[1] here;
            # suppressing that paged write is Stage2-B step 10b. Returning the latent is
            # harmless to the non-offload path (it ignores the returns).
            _, _, k_pe, k_nope = torch_npu.npu_kv_rmsnorm_rope_cache(
                kv_no_split,
                self.kv_a_layernorm.weight,  # type: ignore[union-attr]
                cos,
                sin,
                slots.to(torch.int64),
                kv_cache[1],
                kv_cache[0],
                epsilon=self.kv_a_layernorm.variance_epsilon,  # type: ignore[union-attr]
                cache_mode=cache_mode,
                is_output_kv=True,
            )
            return k_pe, k_nope

    # Return `ql_nope`, `q_pe`
    def _q_proj_and_k_up_proj(self, x):
        q_nope, q_pe = (
            self.q_proj(x)[0]
            .view(-1, self.local_num_heads, self.qk_head_dim)
            .split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        )

        # Convert from (B, N, P) to (N, B, P)
        q_nope = q_nope.transpose(0, 1)
        # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
        ql_nope = torch.bmm(q_nope, self.W_UK_T)
        # Convert from (N, B, L) to (B, N, L)
        return ql_nope.transpose(0, 1), q_pe

    def _v_up_proj(self, x):
        num_input_tokens, _, _ = x.shape
        if (
            x.dtype in [torch.float16, torch.bfloat16]
            and hasattr(torch.ops._C_ascend, "batch_matmul_transpose")
            and num_input_tokens <= BMM_TRANS_MAX_SUPPORTED_TOKENS
        ):
            x = x.view(-1, self.local_num_heads, self.kv_lora_rank)
            res = torch.empty((num_input_tokens, self.local_num_heads, self.v_head_dim), dtype=x.dtype, device=x.device)
            torch.ops._C_ascend.batch_matmul_transpose(x, self.W_UV, res)
            x = res.reshape(-1, self.local_num_heads * self.v_head_dim)
        else:
            # Convert from (B, N, L) to (N, B, L)
            x = x.view(-1, self.local_num_heads, self.kv_lora_rank).transpose(0, 1)
            # # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
            x = torch.bmm(x, self.W_UV)
            # # Convert from (N, B, V) to (B, N * V)
            x = x.transpose(0, 1).reshape(-1, self.local_num_heads * self.v_head_dim)
        return x

    def _sfa_preprocess_with_mlapo(
        self,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        slot_mapping: torch.Tensor,
        num_input_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        k_nope, k_pe = kv_cache[0], kv_cache[1]
        ql_nope = torch.empty(
            (num_input_tokens, self.W_UK_T.shape[0], k_nope.shape[-1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        q_pe = torch.empty(
            (num_input_tokens, self.W_UK_T.shape[0], k_pe.shape[-1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        q_c = torch.empty(
            (num_input_tokens, self.q_lora_rank),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops._C_ascend.mla_preprocess(
            hidden_states,
            self.wd_qkv,
            self.deq_scale_qkv,
            self.gamma1,
            self.beta1,
            self.wu_q,
            self.qb_deq_scl,
            self.gamma2,
            cos,
            sin,
            self.W_UK_T,
            k_nope,
            k_pe,
            slot_mapping,
            quant_scale0=self.quant_scale0,
            quant_offset0=self.quant_offset0,
            bias0=self.quant_bias_qkv,
            quant_scale1=self.quant_scale1,
            quant_offset1=self.quant_offset1,
            bias1=self.qb_qt_bias,
            ctkv_scale=self.ctkv_scale,
            q_nope_scale=self.q_nope_scale,
            cache_mode="krope_ctkv",
            quant_mode="per_tensor_quant_asymm",
            enable_inner_out=True,
            q_out0=ql_nope,
            kv_cache_out0=k_nope,
            q_out1=q_pe,
            kv_cache_out1=k_pe,
            inner_out=q_c,
        )
        return hidden_states, ql_nope, q_pe, q_c

    def indexer_select_pre_process(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        k_li, _ = self.wk(x)  # [b,s,7168] @ [7168,128] = [b,s,128]
        k_li = self.k_norm(k_li).unsqueeze(1)
        k_li = k_li.view(-1, 1, self.head_dim)

        if HAS_TRITON:
            cos = cos.view(-1, self.qk_rope_head_dim)
            sin = sin.view(-1, self.qk_rope_head_dim)
            k_li = rope_forward_triton_siso(
                k_li, cos, sin, rope_dim=self.qk_rope_head_dim, is_neox_style=self.is_rope_neox_style
            )
        else:
            k_li_pe, k_li_nope = torch.split(
                k_li, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
            )

            cos = cos.view(-1, 1, 1, self.qk_rope_head_dim)
            sin = sin.view(-1, 1, 1, self.qk_rope_head_dim)

            k_li_pe = k_li_pe.unsqueeze(2)
            k_li_pe = torch_npu.npu_rotary_mul(k_li_pe, cos, sin)
            k_li_pe = k_li_pe.squeeze(2)

            k_li = torch.cat([k_li_pe, k_li_nope], dim=-1)  # [b*s,128]

        if self.use_sparse_c8_indexer:
            k_li = k_li @ AscendSFAImpl.k_hadamard
            k_li, k_li_scale = torch_npu.npu_dynamic_quant(k_li.view(-1, self.head_dim), dst_type=self.c8_k_cache_dtype)
            k_li_scale = k_li_scale.to(self.c8_k_scale_cache_dtype)  # [b*s,]
            k_li_scale = k_li_scale.unsqueeze(-1)  # [b*s,1]
        else:
            k_li_scale = None

        return k_li, k_li_scale

    def indexer_select_post_process(
        self,
        x: torch.Tensor,
        q_c: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attn_metadata: M | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        indexer_block_table_override: torch.Tensor | None = None,
    ):
        # DSA two-group mode: the indexer cache has its own block ids; fall back
        # to the (shared) latent block table in single-group mode.
        if indexer_block_table_override is not None:
            indexer_block_table = indexer_block_table_override
        else:
            assert attn_metadata is not None
            indexer_block_table = (
                attn_metadata.indexer_block_table
                if attn_metadata.indexer_block_table is not None
                else attn_metadata.block_table
            )
        weights, _ = self.weights_proj(x)

        q_li, _ = self.wq_b(q_c)  # [b,s,1536] @ [1536,64*128] = [b,s,64*128]
        q_li = q_li.view(-1, self.n_head, self.head_dim)  # [n_toks,64,128]
        if HAS_TRITON:
            q_li = rope_forward_triton_siso(
                q_li, cos, sin, rope_dim=self.qk_rope_head_dim, is_neox_style=self.is_rope_neox_style
            )
        else:
            q_li_pe, q_li_nope = torch.split(
                q_li, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
            )  # [b,s,64,64+64]

            q_li_pe = q_li_pe.unsqueeze(2)
            q_li_pe = torch_npu.npu_rotary_mul(q_li_pe, cos, sin)
            q_li_pe = q_li_pe.squeeze(2)
            q_li = torch.cat([q_li_pe, q_li_nope], dim=-1)  # [b*s,64,128]

        if self.use_sparse_c8_indexer:
            q_li_shape_ori = q_li.shape
            q_li = q_li @ AscendSFAImpl.q_hadamard
            q_li, q_li_scale = torch_npu.npu_dynamic_quant(q_li.view(-1, self.head_dim), dst_type=self.c8_k_cache_dtype)
            q_li_scale = q_li_scale.to(self.c8_k_scale_cache_dtype)

        # DSV3.2 currently has graph compilation issues when using torch_npu.npu.lightning_indexer.
        # So two branches are maintained temporarily.
        # TODO: torch.ops._C_ascend.npu_lightning_indexer needs to be removed.
        if self.use_sparse_c8_indexer:
            assert len(kv_cache) == 4
            weights = weights.to(torch.float16)
            topk_indices = torch.ops._C_ascend.npu_lightning_indexer_quant(
                query=q_li.view(q_li_shape_ori),
                key=kv_cache[2],
                weights=weights,
                query_dequant_scale=q_li_scale.view(q_li_shape_ori[:-1]),
                key_dequant_scale=kv_cache[3].squeeze(2),  # B S N D -> B S D
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=indexer_block_table,
                query_quant_mode=0,
                key_quant_mode=0,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=self.index_topk,
                sparse_mode=3,
            )
        elif self.use_torch_npu_lightning_indexer:
            topk_indices, _ = torch_npu.npu_lightning_indexer(
                query=q_li,
                key=kv_cache[2],
                weights=weights,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=indexer_block_table,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=self.index_topk,
                sparse_mode=3,
            )
        else:
            topk_indices = torch.ops._C_ascend.npu_lightning_indexer(
                query=q_li,
                key=kv_cache[2],
                weights=weights,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=indexer_block_table,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=self.index_topk,
                sparse_mode=3,
            )
        return topk_indices

    def _execute_sparse_flash_attention_process(
        self,
        ql_nope,
        q_pe,
        kv_cache,
        topk_indices,
        attn_metadata,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        kv_override=None,
        key_rope_override=None,
        block_table_override=None,
        layer_name: str | None = None,
        trace_label: str = "native",
        padding_rows_zeroed: bool = False,
    ):
        # DSA latent offload: when overrides are given, read latent from the A1 scratch
        # (kv_override/key_rope_override) via the scratch block_table instead of the
        # full paged latent cache. Used by the decode-gather path.
        if block_table_override is not None:
            block_table = block_table_override
        else:
            assert attn_metadata is not None
            block_table = attn_metadata.block_table

        if kv_override is not None:
            kv = kv_override
            key_rope = key_rope_override
        else:
            kv = kv_cache[0]
            key_rope = kv_cache[1]

        _dsa_decode_sparse_fa = (
            self.dsa_shrink_latent
            and attn_metadata is not None
            and block_table is not None
            and attn_metadata.num_decode_tokens > 0
            and attn_metadata.attn_state
            in (
                AscendAttentionState.DecodeOnly,
                AscendAttentionState.SpecDecoding,
            )
        )
        topk_2d = None
        if _dsa_decode_sparse_fa and not padding_rows_zeroed:
            topk_indices, topk_2d = _dsa_mask_padding_sparse_rows(
                topk_indices,
                getattr(attn_metadata, "decode_req_indices", None),
            )

        if _dsa_decode_sparse_fa:
            if topk_2d is None:
                topk_2d = _dsa_topk_to_2d_indices(topk_indices)
            topk_rows = int(topk_2d.shape[0])
            block_table_rows = int(block_table.shape[0])
            batch_size = int(actual_seq_lengths_query.numel())
            if block_table_rows != batch_size:
                decode_req_indices = getattr(attn_metadata, "decode_req_indices", None)
                decode_req_indices_sample = None
                if decode_req_indices is not None:
                    decode_req_indices_sample = (
                        decode_req_indices[: min(topk_rows, 8)].detach().to(device="cpu").tolist()
                    )
                raise RuntimeError(
                    "DSA sparse FA block_table batch dimension mismatch: "
                    f"layer={layer_name} trace_label={trace_label} "
                    f"attn_state={attn_metadata.attn_state} "
                    f"topk_shape={tuple(topk_indices.shape)} "
                    f"topk_rows={topk_rows} "
                    f"block_table_shape={tuple(block_table.shape)} "
                    f"block_table_rows={block_table_rows} "
                    f"batch_size={batch_size} "
                    f"num_decode_tokens={attn_metadata.num_decode_tokens} "
                    f"decode_req_indices_shape="
                    f"{tuple(decode_req_indices.shape) if decode_req_indices is not None else None} "
                    f"decode_req_indices_sample={decode_req_indices_sample}"
                )
        attn_output = torch.ops._C_ascend.npu_sparse_flash_attention(
            query=ql_nope,
            key=kv,
            value=kv,
            sparse_indices=topk_indices,
            scale_value=self.scale,
            sparse_block_size=1,
            block_table=block_table,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_key,
            query_rope=q_pe,
            key_rope=key_rope,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
        )
        return attn_output

    def _cross_layer_ineligible_reason(
        self,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: M,
    ) -> str | None:
        """Return why this step cannot use a fixed-capacity Q1 graph key."""
        forward_context = get_forward_context()
        runtime_mode = getattr(
            forward_context,
            "cudagraph_runtime_mode",
            None,
        )
        staged_dummy_run = bool(
            getattr(
                forward_context,
                "staged_sfa_graph_dummy_run",
                False,
            )
        )
        if runtime_mode != CUDAGraphMode.PIECEWISE and not (staged_dummy_run and runtime_mode == CUDAGraphMode.NONE):
            return "the runtime graph mode is not PIECEWISE"

        batch_size = int(hidden_states.shape[0])
        capture_sizes = getattr(
            self,
            "_staged_sfa_graph_capture_sizes",
            None,
        )
        if capture_sizes is None:
            # Compatibility for lightweight test/downstream implementations.
            capture_sizes = staged_sfa_graph_capture_sizes(self.vllm_config)
        if batch_size not in capture_sizes:
            return "the exact Q=1 batch size is not a configured staged SFA graph key"
        graph_key = StagedSFAGraphKey(
            token_capacity=batch_size,
            request_capacity=batch_size,
            query_profile=StagedSFAQueryProfile.DECODE_Q1,
            max_query_len=1,
        )
        authorized_key = getattr(
            forward_context,
            "staged_sfa_graph_key",
            None,
        )
        if authorized_key != graph_key:
            return "the runner did not authorize this exact staged SFA graph key"

        batch_descriptor = getattr(
            forward_context,
            "batch_descriptor",
            None,
        )
        if (
            batch_descriptor != graph_key.to_legacy_batch_descriptor()
            or batch_descriptor.uniform
            or batch_descriptor.has_lora
            or batch_descriptor.num_reqs is not None
            or batch_descriptor.num_active_loras != 0
        ):
            return "the PIECEWISE descriptor does not match the exact Q=1 staged SFA graph key"

        if self.vllm_config.speculative_config is not None:
            return "speculative decoding is enabled"
        if self.vllm_config.lora_config is not None:
            return "LoRA is configured"
        if attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
            return "only DecodeOnly is supported"
        actual_rows = int(attn_metadata.num_actual_tokens)
        if (
            attn_metadata.num_input_tokens != batch_size
            or actual_rows <= 0
            or actual_rows > batch_size
            or attn_metadata.num_decode_tokens != actual_rows
        ):
            return "the real Q=1 row count exceeds the staged graph capacity"
        if self.dsa_shrink_latent != 2:
            return "SHRINK_LATENT must be 2"
        if (
            not staged_dummy_run
            and not staged_sfa_connector_supports_sparse_load()
        ):
            return "the active connector does not support staged sparse selective loads"
        if self.enable_mlapo:
            return "MLAPO is enabled"
        weight_prefetch_method = get_weight_prefetch_method()
        if weight_prefetch_method is not None and weight_prefetch_method.mla_sfa_prefetch_enable:
            return "weight prefetch is enabled"
        if self.enable_dsa_cp:
            return "DSA context parallelism is enabled"
        if self.enable_dsa_cp_with_o_proj_tp:
            return "DSA o_proj tensor parallelism is enabled"
        if self.use_sparse_c8_indexer:
            return "the sparse C8 indexer is enabled"
        if self.dsa_offload_free_paged:
            return "the free-paged offload path is enabled"
        if envs.VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY:
            return "the DSA parity path is enabled"
        if getattr(forward_context, "dsa_offload_manager", None) is not None:
            return "the DSA offload manager is active"
        if getattr(forward_context, "dsa_adapter_cache", None) is not None:
            return "the DSA adapter cache is active"
        if len(kv_cache) != 3:
            return "the POC requires exactly three KV tensors"
        if any(cache.ndim != 4 for cache in kv_cache):
            return "the POC requires rank-4 PA_BSND KV tensors"
        if self.num_kv_heads != 1 or any(int(cache.shape[-2]) != 1 for cache in kv_cache):
            return "the POC requires one KV head in every cache tensor"
        expected_hidden_dims = (
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.head_dim,
        )
        if tuple(int(cache.shape[-1]) for cache in kv_cache) != tuple(int(dim) for dim in expected_hidden_dims):
            return "the staged KV cache hidden dimensions do not match SFA"
        cache_block_sizes = {int(cache.shape[1]) for cache in kv_cache}
        if len(cache_block_sizes) != 1:
            return "the staged KV cache block sizes do not agree"
        configured_block_size = int(self.vllm_config.cache_config.block_size)
        if next(iter(cache_block_sizes)) != configured_block_size:
            return "the staged KV cache block size does not match the configured block size"
        if any(int(cache.shape[0]) < batch_size for cache in kv_cache):
            return "the staged KV caches do not have one safe dummy block per request"
        if len({cache.device for cache in kv_cache}) != 1:
            return "the staged KV caches are on different devices"
        cache_dtypes = {cache.dtype for cache in kv_cache}
        if len(cache_dtypes) != 1:
            return "the staged KV caches must share one dtype"
        if next(iter(cache_dtypes)) not in (
            torch.float16,
            torch.bfloat16,
        ):
            return "the staged KV cache dtype must be float16 or bfloat16"
        if self.q_lora_rank is None or self.fused_qkv_a_proj is None:
            return "the native Q-LoRA preprocessing path is unavailable"
        if self.q_a_layernorm is None:
            return "q_a_layernorm is unavailable"

        required_row_tensors = (
            attn_metadata.cos,
            attn_metadata.sin,
            attn_metadata.slot_mapping,
            attn_metadata.cum_query_lens,
            attn_metadata.seq_lens,
            attn_metadata.indexer_slot_mapping,
        )
        if any(tensor is None for tensor in required_row_tensors):
            return "required fixed-shape attention metadata is unavailable"
        if any(int(tensor.shape[0]) != batch_size for tensor in required_row_tensors):
            return "the fixed-shape attention row count does not match the graph key"
        if (
            attn_metadata.block_table is None
            or attn_metadata.indexer_block_table is None
            or int(attn_metadata.block_table.shape[0]) != batch_size
            or int(attn_metadata.indexer_block_table.shape[0]) != batch_size
        ):
            return "the native block-table row count does not match the graph key"
        if (
            not staged_dummy_run
            and not attn_metadata.need_sparse_lmcache_payload
        ):
            return "the v1 sparse LMCache payload path is unavailable"
        valid_rows = attn_metadata.decode_valid_row_indices
        scratch_base = attn_metadata.decode_scratch_base
        if (
            valid_rows is None
            or int(valid_rows.numel()) != actual_rows
            or scratch_base is None
            or int(scratch_base.numel()) != batch_size
            or attn_metadata.decode_scratch_base_compact is not None
            or attn_metadata.decode_target_slot_mapping is not None
        ):
            return "the fused remap inputs do not match the Q1 graph capacity"

        request_ids = attn_metadata.decode_request_ids_compact
        full_request_ids = attn_metadata.req_ids
        if (
            request_ids is None
            or full_request_ids is None
            or len(request_ids) != actual_rows
            or len(full_request_ids) != actual_rows
            or tuple(request_ids) != tuple(full_request_ids)
            or len(set(request_ids)) != actual_rows
        ):
            return "the compact LMCache request ids are not the unique native request order"
        if (
            attn_metadata.prompt_lens_cpu_rows is None
            or attn_metadata.decode_req_indices_cpu is None
            or attn_metadata.seq_lens_cpu is None
            or attn_metadata.decode_remap_boundary is None
            or int(attn_metadata.decode_remap_boundary.shape[0]) != batch_size
        ):
            return "the persistent remap-boundary metadata is unavailable"

        prompt_rows = np.asarray(
            attn_metadata.prompt_lens_cpu_rows,
            dtype=np.int64,
        ).reshape(-1)
        request_rows = np.asarray(
            attn_metadata.decode_req_indices_cpu,
            dtype=np.int64,
        ).reshape(-1)
        scratch_bases = np.asarray(
            attn_metadata.decode_scratch_base_cpu,
            dtype=np.int64,
        ).reshape(-1)
        seq_lens_cpu = attn_metadata.seq_lens_cpu
        if isinstance(seq_lens_cpu, torch.Tensor):
            if seq_lens_cpu.device.type != "cpu":
                return "sequence-length validation metadata is not on CPU"
            seq_rows = seq_lens_cpu.detach().numpy().reshape(-1)
        else:
            seq_rows = np.asarray(seq_lens_cpu).reshape(-1)
        expected_request_rows = np.full(batch_size, -1, dtype=np.int64)
        expected_request_rows[:actual_rows] = np.arange(actual_rows, dtype=np.int64)
        if (
            prompt_rows.size != batch_size
            or np.any(prompt_rows[:actual_rows] < self.index_topk)
            or np.any(prompt_rows[actual_rows:] != 0)
            or request_rows.size != batch_size
            or not np.array_equal(request_rows, expected_request_rows)
            or scratch_bases.size != batch_size
            or np.any(scratch_bases != 0)
            or seq_rows.size != batch_size
            or np.any(seq_rows[:actual_rows] < prompt_rows[:actual_rows])
            or np.any(seq_rows[actual_rows:] != 0)
        ):
            return "the Q1 CPU row metadata does not match the graph capacity"
        return None

    def _submit_sfa_save_operations(
        self,
        save_operations: list[tuple[str, list[torch.Tensor]]],
    ) -> None:
        for layer_name, kv_caches in save_operations:
            maybe_save_kv_layer_to_connector(
                layer_name,
                kv_caches,
            )

    def _cross_layer_pre_compute(
        self,
        hidden_states: torch.Tensor,
        kv_cache_nope: torch.Tensor,
        kv_cache_pe: torch.Tensor,
        indexer_cache: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        slot_mapping: torch.Tensor,
        indexer_slot_mapping: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        indexer_block_table: torch.Tensor,
        remap_boundary: torch.Tensor,
        valid_row_indices: torch.Tensor,
        scratch_base: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Pre-retrieval compute captured by the outer PIECEWISE graph."""
        assert self.fused_qkv_a_proj is not None
        assert self.q_lora_rank is not None
        assert self.q_a_layernorm is not None

        weight_prefetch_method = get_weight_prefetch_method()
        weight_prefetch_method.maybe_prefetch_mla_or_sla_weight_in_current_stream(
            inputs=self.fused_qkv_a_proj.weight,
            dependency=hidden_states,
        )
        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_no_split = qkv_lora.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            dim=-1,
        )
        q_c = self.q_a_layernorm(q_c)
        k_li, k_li_scale = self.indexer_select_pre_process(
            x=hidden_states,
            cos=cos,
            sin=sin,
        )
        assert k_li_scale is None
        kv_cache = (kv_cache_nope, kv_cache_pe, indexer_cache)
        self.exec_kv(
            kv_no_split,
            cos,
            sin,
            kv_cache,
            slot_mapping,
            None,
        )

        ql_nope, q_pe = self._q_proj_and_k_up_proj(q_c)
        q_pe = self.rope_single(q_pe, cos, sin)
        k_li = self._get_full_kv(k_li, None)

        torch_npu.npu_scatter_nd_update_(
            indexer_cache.view(-1, k_li.shape[-1]),
            indexer_slot_mapping.view(-1, 1),
            k_li.view(-1, k_li.shape[-1]),
        )
        topk_indices = self.indexer_select_post_process(
            x=hidden_states,
            q_c=q_c,
            kv_cache=kv_cache,
            attn_metadata=None,
            cos=cos,
            sin=sin,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            indexer_block_table_override=indexer_block_table,
        )
        topk_indices, selected_packed = prepare_sparse_indices(
            topk_indices,
            remap_boundary,
            need_packed=True,
            scratch_base=scratch_base,
            valid_row_indices=valid_row_indices,
        )
        topk_2d = _dsa_topk_to_2d_indices(topk_indices)
        topk_2d.masked_fill_(
            remap_boundary[: topk_2d.shape[0]].reshape(-1, 1) == 0,
            0,
        )
        assert selected_packed is not None
        return (
            ql_nope,
            q_pe,
            topk_indices,
            selected_packed,
        )

    def _cross_layer_post_compute(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        topk_indices: torch.Tensor,
        kv_cache_nope: torch.Tensor,
        kv_cache_pe: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        block_table: torch.Tensor,
        output: torch.Tensor,
        *,
        trace_label: str,
    ) -> torch.Tensor:
        kv_cache = (kv_cache_nope, kv_cache_pe)
        attn_output = self._execute_sparse_flash_attention_process(
            ql_nope,
            q_pe,
            kv_cache,
            topk_indices,
            None,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            block_table_override=block_table,
            trace_label=trace_label,
        )
        attn_output = self._v_up_proj(attn_output)
        weight_prefetch_method = get_weight_prefetch_method()
        weight_prefetch_method.maybe_prefetch_mla_or_sla_weight_in_current_stream(
            inputs=self.o_proj.weight,
            dependency=attn_output,
            max_size=MAX_O_PROJ_PREFETCH_SIZE,
            linear_layer=self.o_proj,
        )
        output[...] = self.o_proj(attn_output)[0]
        return output

    def _cross_layer_kv_cache(
        self,
        layer_name: str,
        kv_cache: tuple[torch.Tensor, ...],
    ) -> tuple[tuple[torch.Tensor, ...], str | None, bool]:
        index_layer_name = _dsa_indexer_layer_name(layer_name) if self.dsa_offload_unbundle else None
        index_enabled = bool(index_layer_name is not None and _dsa_index_lmcache_enabled())
        if self.dsa_offload_unbundle and len(kv_cache) < 3:
            index_cache = getattr(self, "_dsa_idx_cache_t", None)
            if index_cache is None:
                context = get_forward_context()
                assert index_layer_name is not None
                registered = context.no_compile_layers[index_layer_name].kv_cache[context.virtual_engine]
                index_cache = registered[0] if isinstance(registered, (tuple, list)) else registered
                self._dsa_idx_cache_t = index_cache
            kv_cache = (*kv_cache, index_cache)
        return kv_cache, index_layer_name, index_enabled

    def _cross_layer_empty_outputs(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Native retrieve/post are no-ops. Keep their ignored outputs
        # contiguous, but cap them at the largest real staged batch so memory
        # profiling cannot materialize prompt-sized bridge tensors.
        num_tokens = self._staged_sfa_graph_capture_sizes[-1]

        return (
            hidden_states.new_empty((num_tokens, self.local_num_heads, self.kv_lora_rank)),
            hidden_states.new_empty((num_tokens, self.local_num_heads, self.qk_rope_head_dim)),
            torch.empty(
                (num_tokens, 1, self.index_topk),
                dtype=torch.int32,
                device=hidden_states.device,
            ),
            torch.empty(
                (num_tokens, self.index_topk),
                dtype=torch.int32,
                device=hidden_states.device,
            ),
        )

    def reset_staged_sfa_capture(self) -> None:
        self._staged_sfa_capture_state = _StagedSFACaptureState()
        self._dsa_idx_cache_t = None

    def seal_staged_sfa_capture(
        self,
        graph_keys: tuple[StagedSFAGraphKey, ...],
    ) -> None:
        self._staged_sfa_capture_state.seal(graph_keys)

    def _pad_cross_layer_bridge_output(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        capacity = self._staged_sfa_graph_capture_sizes[-1]
        rows = tensor.shape[0]
        if rows == capacity:
            return tensor.contiguous()
        if rows > capacity:
            raise RuntimeError(
                f"staged SFA bridge output exceeds its configured capacity: rows={rows}, capacity={capacity}"
            )
        padding = tensor.new_empty((capacity - rows, *tensor.shape[1:]))
        return torch.cat((tensor, padding), dim=0)

    def cross_layer_graph_pre(
        self,
        layer_name: str,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: M | None,
        need_gather_q_kv: bool,
        output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run graph A, or the complete native path outside staged replay."""
        context = get_forward_context()
        if attn_metadata is None:
            self.forward(
                layer_name,
                hidden_states,
                kv_cache,
                attn_metadata,
                need_gather_q_kv,
                output,
            )
            return self._cross_layer_empty_outputs(hidden_states)
        kv_cache, index_layer_name, index_enabled = self._cross_layer_kv_cache(layer_name, kv_cache)
        graph_key = getattr(context, "staged_sfa_graph_key", None)
        if graph_key is None:
            self.forward(
                layer_name,
                hidden_states,
                kv_cache,
                attn_metadata,
                need_gather_q_kv,
                output,
            )
            return self._cross_layer_empty_outputs(hidden_states)
        route = getattr(context, "staged_sfa_route", None)
        if route is None or route.action != StagedSFARouteAction.STAGED or route.graph_key != graph_key:
            raise RuntimeError(f"[SFA_ROUTE] action=fatal reason={StagedSFARouteReason.RUNNER_LAYER_MISMATCH.value}")
        reason = self._cross_layer_ineligible_reason(hidden_states, kv_cache, attn_metadata)
        if reason is not None:
            raise RuntimeError(
                f"[SFA cross-layer graph] runner-authorized key became ineligible in {layer_name}: {reason}"
            )

        is_dummy = bool(getattr(context, "staged_sfa_graph_dummy_run", False))
        state = self._staged_sfa_capture_state
        initialized_capacity = state.initialized_cache_capacity
        if is_dummy and graph_key.request_capacity > initialized_capacity:
            for cache in kv_cache:
                cache[initialized_capacity : graph_key.request_capacity].zero_()
            state.initialized_cache_capacity = graph_key.request_capacity
        if is_dummy and context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
            # ACL capture cannot include the host copy in boundary preparation.
            # The immediately preceding eager warmup filled this stable buffer.
            capture_boundary = state.remap_boundary
            boundary = attn_metadata.decode_remap_boundary
            if (
                capture_boundary is None
                or boundary is None
                or capture_boundary.data_ptr() != boundary.data_ptr()
                or capture_boundary.shape != boundary.shape
            ):
                raise RuntimeError(
                    "[SFA cross-layer graph] remap boundary was not prepared in stable storage by eager warmup"
                )
            remap_boundary = capture_boundary
        else:
            remap_boundary = _prepare_sfa_remap_boundary(
                attn_metadata,
                attn_metadata.req_ids,
                is_dummy_run=is_dummy,
                index_topk=self.index_topk,
                cached_tokens=getattr(
                    getattr(context, "staged_sfa_route", None),
                    "frontiers",
                    None,
                ),
            )
            if is_dummy:
                state.remap_boundary = remap_boundary
        valid_row_indices = self._staged_sfa_row_indices
        if valid_row_indices is None:
            if context.cudagraph_runtime_mode != CUDAGraphMode.NONE:
                raise RuntimeError("staged SFA row indices were not created by eager warmup")
            valid_row_indices = torch.arange(
                self._staged_sfa_graph_capture_sizes[-1],
                dtype=torch.int32,
                device=hidden_states.device,
            )
            self._staged_sfa_row_indices = valid_row_indices
        valid_row_indices = valid_row_indices[: graph_key.request_capacity]
        scratch_base = attn_metadata.decode_scratch_base
        assert scratch_base is not None
        outputs = self._cross_layer_pre_compute(
            hidden_states,
            kv_cache[0],
            kv_cache[1],
            kv_cache[2],
            attn_metadata.cos,
            attn_metadata.sin,
            attn_metadata.slot_mapping,
            attn_metadata.indexer_slot_mapping,
            attn_metadata.cum_query_lens,
            attn_metadata.seq_lens,
            attn_metadata.indexer_block_table,
            remap_boundary,
            valid_row_indices,
            scratch_base,
        )
        outputs = (
            self._pad_cross_layer_bridge_output(outputs[0]),
            self._pad_cross_layer_bridge_output(outputs[1]),
            self._pad_cross_layer_bridge_output(outputs[2]),
            self._pad_cross_layer_bridge_output(outputs[3]),
        )
        producer_event = state.producer_event
        if producer_event is None:
            if context.cudagraph_runtime_mode != CUDAGraphMode.NONE:
                raise RuntimeError("staged SFA producer event was not created by eager warmup")
            producer_event = torch.npu.Event()
            state.producer_event = producer_event
        attn_metadata.reshape_cache_event = producer_event
        producer_event.record()
        state.runtime = (
            layer_name,
            kv_cache,
            index_layer_name,
            index_enabled,
        )
        if is_dummy and context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
            state.register(graph_key, outputs, kv_cache)
        return outputs

    def cross_layer_lmcache_retrieve(
        self,
        layer_name: str,
        next_layer_name: str,
        selected_packed: torch.Tensor,
        attn_metadata: M | None,
        context: Any,
    ) -> None:
        with _staged_sfa_profile_scope("sfa_cross_layer::lmcache_retrieve"):
            graph_key = getattr(context, "staged_sfa_graph_key", None)
            if attn_metadata is None or graph_key is None:
                return
            if getattr(context, "staged_sfa_graph_dummy_run", False):
                if next_layer_name:
                    next_metadata = context.attn_metadata[next_layer_name]
                    _prepare_sfa_remap_boundary(
                        next_metadata,
                        next_metadata.req_ids,
                        is_dummy_run=True,
                        index_topk=self.index_topk,
                    )
                return
            route = context.staged_sfa_route
            state = self._staged_sfa_capture_state
            index_enabled = bool(state.runtime and state.runtime[3])
            producer_event = state.producer_event
            if producer_event is not None:
                attn_metadata.reshape_cache_event = producer_event
            selected, request_ids, target_slots = _prepare_dsa_sparse_lmcache_payload(
                attn_metadata,
                selected_packed[: attn_metadata.num_decode_tokens],
                index_topk=self.index_topk,
                validate_once=True,
            )
            wait_for_kv_layer_from_connector(
                layer_name,
                selected_tokens=selected,
                token_start_index=None,
                request_ids=request_ids,
                target_slot_mapping=target_slots,
                payload_event=producer_event,
            )
            if _LMCACHE_SPARSE_WAIT_SYNC_ONCE and not _lmcache_sparse_wait_sync_once_done:
                _sync_compute_stream_after_lmcache_sparse_wait()
            if next_layer_name:
                next_metadata = context.attn_metadata[next_layer_name]
                _prepare_sfa_remap_boundary(
                    next_metadata,
                    next_metadata.req_ids,
                    is_dummy_run=False,
                    index_topk=self.index_topk,
                    cached_tokens=route.frontiers,
                )
                if index_enabled:
                    wait_for_kv_layer_from_connector(_dsa_indexer_layer_name(next_layer_name))

    def bootstrap_cross_layer(self, layer_name: str) -> None:
        """Prepare layer zero before the first captured island is launched."""
        with _staged_sfa_profile_scope("sfa_cross_layer::bootstrap"):
            context = get_forward_context()
            metadata = context.attn_metadata[layer_name]
            is_dummy = bool(getattr(context, "staged_sfa_graph_dummy_run", False))
            _prepare_sfa_remap_boundary(
                metadata,
                metadata.req_ids,
                is_dummy_run=is_dummy,
                index_topk=self.index_topk,
                cached_tokens=(
                    None
                    if is_dummy
                    else context.staged_sfa_route.frontiers
                ),
            )
            runtime = self._staged_sfa_capture_state.runtime
            if not is_dummy and runtime and runtime[2] is not None and runtime[3]:
                wait_for_kv_layer_from_connector(runtime[2])

    def cross_layer_graph_post(
        self,
        layer_name: str,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        topk_indices: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: M | None,
        output: torch.Tensor,
    ) -> None:
        graph_key = getattr(get_forward_context(), "staged_sfa_graph_key", None)
        if attn_metadata is None or graph_key is None:
            return
        rows = graph_key.request_capacity
        kv_cache, _, _ = self._cross_layer_kv_cache(layer_name, kv_cache)
        self._cross_layer_post_compute(
            ql_nope[:rows],
            q_pe[:rows],
            topk_indices[:rows],
            kv_cache[0],
            kv_cache[1],
            attn_metadata.cum_query_lens,
            attn_metadata.seq_lens,
            attn_metadata.block_table,
            output,
            trace_label="cross_layer",
        )

    def submit_cross_layer_save(self) -> None:
        runtime = self._staged_sfa_capture_state.runtime
        if runtime is None:
            return
        layer_name, kv_cache, index_layer_name, index_enabled = runtime
        if bool(self.dsa_shrink_latent) and _decode_window_save_window_size() == 0:
            return
        maybe_save_kv_layer_to_connector(layer_name, [kv_cache[0], kv_cache[1]])
        if index_layer_name is not None and index_enabled:
            maybe_save_kv_layer_to_connector(index_layer_name, [kv_cache[2]])

    def forward(
        self,
        layer_name,
        hidden_states: torch.Tensor,  # query in unified attn
        kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attn_metadata: M,
        need_gather_q_kv: bool = False,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            # Profiling run.
            if self.enable_dsa_cp_with_layer_shard and not _EXTRA_CTX.in_profile_run:
                for layer in self.layer_sharding_kwargs or []:
                    if is_hidden_layer(layer):
                        reach_layer_for_shard_weight_series(layer)
            return output.fill_(0)

        _dsa_prof.set_step_kind(attn_metadata.attn_state == AscendAttentionState.DecodeOnly)
        _sfa_t = _dsa_prof.begin("sfa_fwd")
        _is_pure_decode = attn_metadata.attn_state in (
            AscendAttentionState.DecodeOnly,
            AscendAttentionState.SpecDecoding,
        )
        _sparse_indices_padding_zeroed = False
        index_layer_name = _dsa_indexer_layer_name(layer_name) if self.dsa_offload_unbundle else None
        index_lmcache_enabled = (
            self.dsa_offload_unbundle and index_layer_name is not None and _dsa_index_lmcache_enabled()
        )
        if self.dsa_offload_unbundle and len(kv_cache) < 3:
            # Un-bundled: the indexer key is its own KV group (DeepseekV32IndexerCache).
            # layer_name is the inner MLAAttention name (...self_attn.attn); the indexer
            # cache is the sibling ...self_attn.indexer.k_cache. Re-assemble a 3-tuple so
            # the indexer read/write (kv_cache[2]) work unchanged — both groups share the
            # request's block ids, so attn_metadata.block_table/slot_mapping address both.
            # NOTE: in two-group mode the indexer group has its own block table and
            # slot mapping; the shared-block assumption only applies to legacy layouts.
            # The indexer KV tensor is allocated once at startup; cache the ref to avoid a
            # per-layer no_compile_layers dict lookup + tuple rebuild on the decode path.
            _idx_t = getattr(self, "_dsa_idx_cache_t", None)
            if _idx_t is None:
                _fc_ub = get_forward_context()
                assert index_layer_name is not None
                _idx_name = index_layer_name
                _idx_cache = _fc_ub.no_compile_layers[_idx_name].kv_cache[_fc_ub.virtual_engine]
                _idx_t = _idx_cache[0] if isinstance(_idx_cache, (tuple, list)) else _idx_cache
                self._dsa_idx_cache_t = _idx_t
            kv_cache = (kv_cache[0], kv_cache[1], _idx_t)

        cos = attn_metadata.cos
        sin = attn_metadata.sin
        slot_mapping = attn_metadata.slot_mapping
        # DSA two-group mode: the indexer cache write must use the indexer
        # group's own slots; falls back to the shared slots in single-group mode.
        idx_slot_mapping = (
            attn_metadata.indexer_slot_mapping if attn_metadata.indexer_slot_mapping is not None else slot_mapping
        )
        slot_mapping_cp = None
        if self.enable_dsa_cp:
            assert attn_metadata.dsa_cp_context is not None
            slot_mapping_cp = attn_metadata.dsa_cp_context.slot_mapping_cp
            actual_seq_lengths_query = attn_metadata.dsa_cp_context.actual_seq_lengths_query
            actual_seq_lengths_key = attn_metadata.dsa_cp_context.actual_seq_lengths_key
        else:
            actual_seq_lengths_query = attn_metadata.cum_query_lens
            actual_seq_lengths_key = attn_metadata.seq_lens

        # Inputs and outputs may be padded for CUDA graphs
        num_input_tokens = attn_metadata.num_input_tokens
        output_padded = output

        # all-gather o_proj weight for prefill stage of PD mix node
        o_proj_full_handle = None
        # if is PD mix stage, using original TP o_proj weight, and also need to full gather for o_proj
        # weight for prefill stage.
        full_gather_o_proj_enabled = self.enable_dsa_cp_with_o_proj_tp and attn_metadata.attn_state not in {
            AscendAttentionState.DecodeOnly,
            AscendAttentionState.SpecDecoding,
        }

        # run mlapo ops when dsa-cp is disabled, and ensure that num_tokens satisfies the count limitation
        if self.enable_mlapo and num_input_tokens <= MLAPO_MAX_SUPPORTED_TOKENS:
            hidden_states, ql_nope, q_pe, q_c = self._sfa_preprocess_with_mlapo(
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                cos=cos,
                sin=sin,
                slot_mapping=slot_mapping,
                num_input_tokens=num_input_tokens,
            )
            k_li, k_li_scale = self.indexer_select_pre_process(x=hidden_states, cos=cos, sin=sin)
        # native
        else:
            assert self.fused_qkv_a_proj is not None, "q lora is required for DSA."
            weight_prefetch_method = get_weight_prefetch_method()
            weight_prefetch_method.maybe_prefetch_mla_or_sla_weight_in_current_stream(
                inputs=self.fused_qkv_a_proj.weight, dependency=hidden_states
            )
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_no_split = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            assert self.q_a_layernorm is not None, "q_a_layernorm must be initialized"
            q_c = self.q_a_layernorm(q_c)

            k_li, k_li_scale = self.indexer_select_pre_process(x=hidden_states, cos=cos, sin=sin)

            # Step B2: in compact-scratch mode the connector load is driven by
            # the post-indexer call (with selected_tokens). Calling here too
            # would advance the per-request layerwise retriever TWICE per layer
            # (this one with a dense arange) and desync it — skip whenever the
            # batch has decode rows (mixed steps included).
            if not (self.dsa_shrink_latent and attn_metadata.num_decode_tokens > 0):
                wait_for_kv_layer_from_connector(layer_name)

            if self.enable_dsa_cp:
                assert slot_mapping_cp is not None
                k_pe, k_nope = self.exec_kv(kv_no_split, cos, sin, kv_cache, slot_mapping_cp, attn_metadata)
            else:
                _fc = get_forward_context()
                _dsa_mgr_xkv = getattr(_fc, "dsa_offload_manager", None)
                if self.dsa_offload_free_paged and _dsa_mgr_xkv is not None:
                    # FREE_PAGED (prefill + decode): write latent into the PagedLatentPool
                    # (not the 1-block dummy kv_cache[0]/[1]); the op writes
                    # ckv_cache/k_cache at the pool's own slots. positions = arange(ctx,
                    # ctx+qlen) per request handles both prefill chunks and decode (qlen=1).
                    # HW-VERIFY: pool tensors are paged-layout for the op.
                    _qsl = torch.cat([attn_metadata.cum_query_lens.new_zeros(1), attn_metadata.cum_query_lens])
                    _ctx = attn_metadata.seq_lens - (_qsl[1:] - _qsl[:-1])
                    with _dsa_prof.section("exec_kv_slots"):
                        _pslots, _pknope, _pkpe = _dsa_mgr_xkv.pool_exec_kv_slots(
                            layer_name,
                            _fc.dsa_req_ids,
                            _qsl,
                            _ctx,
                            decode=attn_metadata.attn_state == AscendAttentionState.DecodeOnly,
                        )
                    with _dsa_prof.section("exec_kv_op"):
                        k_pe, k_nope = self.exec_kv(kv_no_split, cos, sin, (_pknope, _pkpe), _pslots, attn_metadata)
                else:
                    with _dsa_prof.section("exec_kv"):
                        k_pe, k_nope = self.exec_kv(kv_no_split, cos, sin, kv_cache, slot_mapping, attn_metadata)

            if self.enable_dsa_cp:
                assert k_pe is not None
                assert k_nope is not None
                assert k_li is not None
                async_op = self.enable_dsa_cp_with_layer_shard or full_gather_o_proj_enabled
                # support all_gather kv async for communication calculation overlap
                if not self.use_sparse_c8_indexer:
                    fused_kv_no_split, kv_ag_handle = all_gather_async(
                        torch.cat(
                            [
                                k_pe.view(-1, k_pe.shape[-1]),
                                k_nope.view(-1, k_nope.shape[-1]),
                                k_li.view(-1, k_li.shape[-1]),
                            ],
                            dim=1,
                        ),
                        get_tp_group(),
                        async_op=async_op,
                    )
                else:
                    # due to different dtypes, we have to split commu pass
                    assert k_li_scale is not None
                    fused_kv_no_split, _ = all_gather_async(
                        torch.cat(
                            [
                                k_pe.view(-1, k_pe.shape[-1]),
                                k_nope.view(-1, k_nope.shape[-1]),
                            ],
                            dim=1,
                        ),
                        get_tp_group(),
                        async_op=async_op,
                    )
                    k_li, _ = all_gather_async(
                        k_li,
                        get_tp_group(),
                        async_op=async_op,
                    )
                    k_li_scale, kv_ag_handle = all_gather_async(
                        k_li_scale,
                        get_tp_group(),
                        async_op=async_op,
                    )

            ql_nope, q_pe = self._q_proj_and_k_up_proj(q_c)
            q_pe = self.rope_single(q_pe, cos, sin)

            if self.enable_dsa_cp:
                if kv_ag_handle is not None:
                    kv_ag_handle.wait()

                if self.enable_dsa_cp_with_layer_shard:
                    for layer in self.layer_sharding_kwargs or []:
                        if is_hidden_layer(layer):
                            reach_layer_for_shard_weight_series(layer)
                elif full_gather_o_proj_enabled:
                    _, o_proj_full_handle = all_gather_async(
                        self.o_proj_tp_weight, get_tp_group(), output=AscendSFAImpl.o_proj_full_pool
                    )

                if kv_cache is not None:
                    assert fused_kv_no_split is not None
                    if not self.use_sparse_c8_indexer:
                        k_pe, k_nope, k_li = fused_kv_no_split.split(
                            [self.qk_rope_head_dim, self.kv_lora_rank, self.head_dim], dim=-1
                        )
                    else:
                        k_pe, k_nope = fused_kv_no_split.split([self.qk_rope_head_dim, self.kv_lora_rank], dim=-1)
                    k_nope = k_nope.view(k_nope.shape[0], 1, -1)
                    k_pe = k_pe.view(k_pe.shape[0], 1, -1)
                    DeviceOperator.reshape_and_cache(
                        key=k_nope[: attn_metadata.num_actual_tokens],
                        value=k_pe[: attn_metadata.num_actual_tokens],
                        key_cache=kv_cache[0],
                        value_cache=kv_cache[1],
                        slot_mapping=slot_mapping[: attn_metadata.num_actual_tokens],
                    )

            k_li = self._get_full_kv(k_li, attn_metadata)

        if kv_cache is not None:
            if index_lmcache_enabled:
                # A cold shared-cache decode needs prompt index rows before
                # top-k selection. The group-1 wait is a no-op when resident
                # and does not advance the group-0 latent-layer cursor.
                with _dsa_prof.section("lmc_index_retrieve"):
                    wait_for_kv_layer_from_connector(index_layer_name)

            if self.is_kv_producer:
                attn_metadata.reshape_cache_event = torch.npu.Event()
            torch_npu.npu_scatter_nd_update_(
                kv_cache[2].view(-1, k_li.shape[-1]), idx_slot_mapping.view(-1, 1), k_li.view(-1, k_li.shape[-1])
            )  # b, s, n, d
            if self.use_sparse_c8_indexer:
                assert len(kv_cache) == 4
                assert k_li_scale is not None
                torch_npu.npu_scatter_nd_update_(
                    kv_cache[3].view(-1, k_li_scale.shape[-1]),
                    idx_slot_mapping.view(-1, 1),
                    k_li_scale.view(-1, k_li_scale.shape[-1]),
                )
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event.record()

        with _dsa_prof.section("indexer"):
            topk_indices = self.indexer_select_post_process(
                x=hidden_states,
                q_c=q_c,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                cos=cos,
                sin=sin,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
            )

        # DSA top-k dump: capture raw positions before compact-scratch remap.
        _dsa_topk_dumper = getattr(get_forward_context(), "dsa_topk_dumper", None)
        if _dsa_topk_dumper is not None:
            _dsa_topk_dumper.capture_layer(layer_name, topk_indices)

        # DSA Step B2 (compact-scratch decode): the indexer just produced topk.
        # Remap LMCache-selected entries to compact scratch rows [0..n_ret)
        # (the request's first ceil(k/block_size) latent blocks) and have
        # LMCache scatter exactly those tokens into scratch. Live-cache entries
        # keep their absolute positions and are read in place via the same
        # block table. Decode-window mode uses current_window_start as the
        # cache boundary instead of prompt_len.
        # All fixed-shape device math — no D2H sync. No-op without a connector.
        if self.dsa_shrink_latent and attn_metadata.prompt_lens is not None and attn_metadata.num_decode_tokens > 0:
            # _remap_boundary is per row. Decode rows carry prompt_len by
            # default; decode-window mode replaces it with current_window_start.
            # Prefill/padding rows carry 0 and stay untouched, so this also
            # covers mixed chunked-prefill + decode steps.
            # The packed front-list only feeds LMCache's selected_tokens; skip building
            # it (and its scatter) when no v1 connector will consume it (profiling /
            # no-offload runs). Production with an LMCache connector is unchanged.
            _need_packed = attn_metadata.need_sparse_lmcache_payload
            _topk_rows = int(topk_indices.shape[0])
            _scratch_base = attn_metadata.decode_scratch_base
            _valid_row_indices = attn_metadata.decode_valid_row_indices
            if _scratch_base is None or _valid_row_indices is None:
                raise RuntimeError(
                    "DSA sparse-index metadata is incomplete: "
                    "decode_valid_row_indices and decode_scratch_base are required"
                )
            _scratch_base = _scratch_base[:_topk_rows]
            if _scratch_base.device != topk_indices.device:
                _scratch_base = _scratch_base.to(device=topk_indices.device)
            _forward_context = get_forward_context()
            _boundary_request_ids = attn_metadata.req_ids
            if _boundary_request_ids is None:
                _boundary_request_ids = getattr(
                    _forward_context,
                    "dsa_req_ids",
                    None,
                )
            _remap_boundary = _prepare_sfa_remap_boundary(
                attn_metadata,
                _boundary_request_ids,
                is_dummy_run=bool(
                    getattr(
                        _forward_context,
                        "staged_sfa_graph_dummy_run",
                        False,
                    )
                ),
                index_topk=self.index_topk,
            )
            _padding_row_req_indices = attn_metadata.decode_req_indices if _is_pure_decode else None
            with _dsa_prof.section("prepare_sparse_indices"):
                topk_indices, _sel_packed = prepare_sparse_indices(
                    topk_indices,
                    _remap_boundary,
                    need_packed=_need_packed,
                    scratch_base=_scratch_base,
                    valid_row_indices=_valid_row_indices,
                    row_req_indices=_padding_row_req_indices,
                )
            _sparse_indices_padding_zeroed = _padding_row_req_indices is not None
            # Stage 3 = isolation diagnostic: remap + FA on (garbage) scratch but
            # NO LMCache call. Output is expected wrong; only crash/no-crash
            # matters (crash => our remap/FA, clean => LMCache transfer kernel).
            if self.dsa_shrink_latent != 3 and _sel_packed is not None:
                (
                    _selected_for_wait,
                    _request_ids_for_wait,
                    _target_slot_mapping_for_wait,
                ) = _prepare_dsa_sparse_lmcache_payload(
                    attn_metadata,
                    _sel_packed,
                    index_topk=self.index_topk,
                )
                _wait_fn = wait_for_kv_layer_from_connector
                with _dsa_prof.section("lmc_retrieve"):
                    _wait_fn(
                        layer_name,
                        selected_tokens=_selected_for_wait,
                        target_slot_mapping=_target_slot_mapping_for_wait,
                        request_ids=_request_ids_for_wait,
                    )
                if _LMCACHE_SPARSE_WAIT_SYNC_ONCE and not _lmcache_sparse_wait_sync_once_done:
                    _sync_compute_stream_after_lmcache_sparse_wait()

        # DSA latent KV offload (GLM5.1), single-card native non-CP path only:
        #   * prefill steps  -> store this layer's prompt latent, use native attention;
        #   * DecodeOnly step -> gather indexer-selected latent into the A1 scratch and
        #     run sparse attention against it. With ASSERT_PARITY, also run the native
        #     path and log the max-abs output diff, driving generation with the native
        #     result so a wrong scratch path can't corrupt output. Falls back to native
        #     when disabled or on unsupported paths (CP / sparse-c8 / mlapo).
        _dsa_fc = get_forward_context()
        _dsa_mgr = getattr(_dsa_fc, "dsa_offload_manager", None)
        _dsa_adapter = getattr(_dsa_fc, "dsa_adapter_cache", None)
        _dsa_on_native_path = not (self.enable_mlapo and num_input_tokens <= MLAPO_MAX_SUPPORTED_TOKENS)
        _dsa_supported = (
            _dsa_mgr is not None and not self.enable_dsa_cp and not self.use_sparse_c8_indexer and _dsa_on_native_path
        )
        # Adapter latent cache (separate flag). Needs per-request ids in the forward
        # context (absent in dummy/profile runs -> skip -> native).
        _adapter_supported = (
            _dsa_adapter is not None
            and getattr(_dsa_fc, "dsa_req_ids", None) is not None
            and not self.enable_dsa_cp
            and not self.use_sparse_c8_indexer
            and _dsa_on_native_path
        )
        if _dsa_mgr is not None and not _dsa_supported:
            # One-time heads-up if offload is enabled but this path can't use it, so a
            # missing [DSA-PARITY] log on the box is self-explanatory.
            logger.warning_once(
                "[DSA] latent offload enabled but inactive on this path "
                f"(dsa_cp={self.enable_dsa_cp}, sparse_c8={self.use_sparse_c8_indexer}, "
                f"mlapo_native={_dsa_on_native_path}); using native attention."
            )

        attn_output = None
        if _adapter_supported:
            # Adapter-backed latent hot cache: FA reads the resident pool in place
            # (zero-copy), the adapter owns residency (hit/miss) + eviction.
            _ac = _dsa_adapter
            _req_ids_a = _dsa_fc.dsa_req_ids
            _kn_a = k_nope.reshape(-1, self.kv_lora_rank)
            _kp_a = k_pe.reshape(-1, self.qk_rope_head_dim)
            if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                with _dsa_prof.section("ad_prep"):
                    # computed per layer (fresh): a cross-layer memo of these went
                    # stale on batch changes (wrong size) and bought no TPOT, so it was
                    # removed -- correctness over a non-win micro-opt.
                    _req_slots_a = _ac.req_slots_tensor(_req_ids_a)
                    _cur_pos_a = (attn_metadata.seq_lens.to(torch.long) - 1).tolist()
                    _topk2d = topk_indices[:, 0, :] if topk_indices.dim() == 3 else topk_indices
                with _dsa_prof.section("ad_insert"):
                    _insert_meta_op = False
                    for _b in range(len(_req_ids_a)):
                        # insert this step's generated token (one row per request);
                        # returns True only when it ran adapter metadata kernels (new
                        # block: load + mark_dirty) -- the only thing that races.
                        _insert_meta_op |= _ac.insert_decode_token(
                            layer_name, _req_ids_a[_b], int(_cur_pos_a[_b]), _kn_a[_b], _kp_a[_b]
                        )
                # WORKAROUND: the adapter's native metadata kernels (mark_dirty / load)
                # don't order with retrieve's load
                # on the device -> retrieve reads torn slot metadata -> bad slot ->
                # block_table OOB -> device hang. mark_dirty is once-per-block now, so
                # only block-allocation steps run those kernels; sync ONLY then. Normal
                # in-block steps do an ordered pool write and need no sync. Remove
                # entirely once the native kernels enforce their own device-side order.
                if _insert_meta_op and hasattr(torch, "npu"):
                    torch.npu.synchronize()
                with _dsa_prof.section("ad_retrieve"):
                    _res_a = _ac.retrieve(layer_name, _req_slots_a, _topk2d)
                with _dsa_prof.section("ad_fa"):
                    adapter_out = self._execute_sparse_flash_attention_process(
                        ql_nope,
                        q_pe,
                        kv_cache,
                        _res_a.sparse_indices.unsqueeze(1),
                        attn_metadata,
                        actual_seq_lengths_query,
                        _res_a.seq_lens,
                        kv_override=_res_a.knope_pool,
                        key_rope_override=_res_a.kpe_pool,
                        block_table_override=_res_a.block_table,
                        layer_name=layer_name,
                        trace_label="adapter",
                    )
                with _dsa_prof.section("ad_release"):
                    _ac.release_after_fa(layer_name, _res_a.loaded_ids)
                _dsa_prof.step()
                if envs.VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY:
                    native_out = self._execute_sparse_flash_attention_process(
                        ql_nope,
                        q_pe,
                        kv_cache,
                        topk_indices,
                        attn_metadata,
                        actual_seq_lengths_query,
                        actual_seq_lengths_key,
                        layer_name=layer_name,
                        trace_label="adapter_parity_native",
                        padding_rows_zeroed=_sparse_indices_padding_zeroed,
                    )
                    diff = (native_out.float() - adapter_out.float()).abs().max()
                    logger.info("[DSA-ADAPTER-PARITY] layer=%s max_abs_diff=%s", layer_name, float(diff))
                    attn_output = native_out  # generation uses the native result
                else:
                    attn_output = adapter_out
            else:
                # prefill: store this layer's prompt latent into the adapter backend so
                # decode-time retrieve can fetch prefill-selected blocks; attention
                # itself uses the native prefill path (attn_output stays None).
                _qsl_a = torch.cat([attn_metadata.cum_query_lens.new_zeros(1), attn_metadata.cum_query_lens])
                _ctx_a = attn_metadata.seq_lens - (_qsl_a[1:] - _qsl_a[:-1])
                _ac.store_prefill(layer_name, _req_ids_a, _qsl_a, _ctx_a, _kn_a, _kp_a)

        if attn_output is None and _dsa_supported:
            from vllm_ascend.distributed.kv_transfer.sparse_offload import sfa_hooks as _dsa_hooks

            _block_size = kv_cache[0].shape[1]
            # latent for this step's tokens comes straight from exec_kv's return
            # (is_output_kv=True), aligned with token order — no paged read-back, so this
            # no longer depends on the latent being resident in the paged cache (10b).
            _kn = k_nope.reshape(-1, self.kv_lora_rank)
            _kp = k_pe.reshape(-1, self.qk_rope_head_dim)
            if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                # store this step's token into the growing decode pool, then gather the
                # selected latent (prefill from LMCache, decode from pool) into scratch.
                _cur_pos = attn_metadata.seq_lens.to(torch.long) - 1
                with _dsa_prof.section("gather"):
                    s_knope, s_kpe, c_idx, s_bt, s_kv = _dsa_hooks.gather_decode(
                        _dsa_mgr,
                        layer_name,
                        _dsa_fc.dsa_req_ids,
                        topk_indices,
                        _dsa_fc.dsa_prompt_lens,
                        _cur_pos,
                        _block_size,
                        _kn,
                        _kp,
                        store_current=not self.dsa_offload_free_paged,
                    )
                # kernel expects sparse_indices as 3-D [num_tokens, 1, topk].
                with _dsa_prof.section("kernel"):
                    scratch_out = self._execute_sparse_flash_attention_process(
                        ql_nope,
                        q_pe,
                        kv_cache,
                        c_idx.unsqueeze(1),
                        attn_metadata,
                        actual_seq_lengths_query,
                        s_kv,
                        kv_override=s_knope,
                        key_rope_override=s_kpe,
                        block_table_override=s_bt,
                        layer_name=layer_name,
                        trace_label="lmcache_scratch",
                    )
                _dsa_prof.step()
                if envs.VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY and not self.dsa_offload_free_paged:
                    native_out = self._execute_sparse_flash_attention_process(
                        ql_nope,
                        q_pe,
                        kv_cache,
                        topk_indices,
                        attn_metadata,
                        actual_seq_lengths_query,
                        actual_seq_lengths_key,
                        layer_name=layer_name,
                        trace_label="lmcache_parity_native",
                        padding_rows_zeroed=_sparse_indices_padding_zeroed,
                    )
                    diff = (native_out.float() - scratch_out.float()).abs().max()
                    logger.info("[DSA-PARITY] layer=%s max_abs_diff=%s", layer_name, float(diff))
                    attn_output = native_out  # safe: generation uses the native result
                else:
                    attn_output = scratch_out
            else:
                # prefill: (1) offload prompt latent to LMCache; (2) ALSO scatter it into
                # the self-managed PagedLatentPool and run prefill attention from the pool
                # (Route 1 / R1b). The vLLM paged latent is still written by the op, so
                # the parity path can compare pool-attn vs native-paged-attn.
                _qsl = torch.cat([attn_metadata.cum_query_lens.new_zeros(1), attn_metadata.cum_query_lens])
                _ctx = attn_metadata.seq_lens - (_qsl[1:] - _qsl[:-1])
                _dsa_hooks.store_prefill(_dsa_mgr, layer_name, _dsa_fc.dsa_req_ids, _qsl, _ctx, _kn, _kp)
                _dsa_mgr.populate_pool_layer(_dsa_fc.dsa_req_ids, layer_name, _qsl, _ctx, _kn, _kp)
                _p_knope, _p_kpe, _p_bt = _dsa_mgr.pool_attn_args(
                    layer_name, _dsa_fc.dsa_req_ids, attn_metadata.block_table.shape[1]
                )
                pool_out = self._execute_sparse_flash_attention_process(
                    ql_nope,
                    q_pe,
                    kv_cache,
                    topk_indices,
                    attn_metadata,
                    actual_seq_lengths_query,
                    actual_seq_lengths_key,
                    kv_override=_p_knope,
                    key_rope_override=_p_kpe,
                    block_table_override=_p_bt,
                    layer_name=layer_name,
                    trace_label="pool_prefill",
                    padding_rows_zeroed=_sparse_indices_padding_zeroed,
                )
                if envs.VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY and not self.dsa_offload_free_paged:
                    native_out = self._execute_sparse_flash_attention_process(
                        ql_nope,
                        q_pe,
                        kv_cache,
                        topk_indices,
                        attn_metadata,
                        actual_seq_lengths_query,
                        actual_seq_lengths_key,
                        layer_name=layer_name,
                        trace_label="pool_prefill_parity_native",
                        padding_rows_zeroed=_sparse_indices_padding_zeroed,
                    )
                    diff = (native_out.float() - pool_out.float()).abs().max()
                    logger.info("[DSA-PARITY-PREFILL] layer=%s max_abs_diff=%s", layer_name, float(diff))
                    attn_output = native_out  # safe: generation uses the native result
                else:
                    attn_output = pool_out

        if attn_output is None:
            with _dsa_prof.section("fa"):
                attn_output = self._execute_sparse_flash_attention_process(
                    ql_nope,
                    q_pe,
                    kv_cache,
                    topk_indices,
                    attn_metadata,
                    actual_seq_lengths_query,
                    actual_seq_lengths_key,
                    layer_name=layer_name,
                    trace_label="native",
                    padding_rows_zeroed=_sparse_indices_padding_zeroed,
                )
            # one step per layer-call on the native (user) path so the profiler
            # logs mean ms/layer-call periodically (mirrors the manager path).
            _dsa_prof.step()

        attn_output = self._v_up_proj(attn_output)
        weight_prefetch_method = get_weight_prefetch_method()
        weight_prefetch_method.maybe_prefetch_mla_or_sla_weight_in_current_stream(
            inputs=self.o_proj.weight,
            dependency=attn_output,
            max_size=MAX_O_PROJ_PREFETCH_SIZE,
            linear_layer=self.o_proj,
        )

        if self.enable_dsa_cp_with_o_proj_tp:
            # When using SFA-CP with pd mixed, o_proj has two cases:
            # 1. prefill: o_proj is a TP weight, we need to all-gather o_proj weight to switch TP=1.
            # 2. decode: all-to-all the hidden_state before the o_proj forward.
            result, require_o_proj_forward = self._handle_o_proj_weight_switch_and_forward(
                attn_output=attn_output,
                output=output,
                o_proj_full_handle=o_proj_full_handle,
                should_shard_weight=full_gather_o_proj_enabled,
            )
            if not require_o_proj_forward:
                _dsa_prof.end(_sfa_t)
                return result
            attn_output = result

        if self.enable_dsa_cp_strict_accuracy:
            send = (
                attn_output.view(-1, self.tp_size, self.num_heads * self.v_head_dim)
                .permute(1, 0, 2)
                .reshape(-1, self.num_heads * self.v_head_dim)
            )

            attn_output = torch.empty_like(send)
            torch.distributed.all_to_all_single(attn_output, send, group=get_tp_group().device_group)

        output[...] = self.o_proj(attn_output)[0]

        # Offload to LMCache. Legacy un-bundled connectors save only the latent
        # (k_nope, k_pe). Connectors declaring DSA index LMCache support also
        # Save the sibling indexer layer whenever the LMCache indexer path is
        # enabled. Bundled path saves the whole tuple as before.
        # Shrink-latent: a pure-decode step's latent lives in the resident tail and is
        # never reloaded from LMCache, so saving it every decode layer is redundant
        # connector work (scales with batch). Skip save on steps with no prefill tokens
        # gated per step (num_prefills is shared by all layers), so the layerwise save
        # generator is never created that step and wait_for_save tolerates its absence.
        # NOTE: the SFA builder never populates attn_metadata.num_prefills (stays at
        # its dataclass default 0 on every step, prefill included), so gating on it
        # skipped the save unconditionally. Gate on attn_state instead, which the
        # builder does set: pure-decode steps are DecodeOnly/SpecDecoding.
        _decode_window_save_enabled = _decode_window_save_window_size() > 0
        _skip_decode_save = bool(self.dsa_shrink_latent) and _is_pure_decode and not _decode_window_save_enabled
        save_operations: list[tuple[str, list[torch.Tensor]]] = []
        if not _skip_decode_save:
            if self.dsa_offload_unbundle and len(kv_cache) >= 2:
                save_operations.append((layer_name, [kv_cache[0], kv_cache[1]]))
                if len(kv_cache) >= 3 and index_layer_name is not None and index_lmcache_enabled:
                    save_operations.append((index_layer_name, [kv_cache[2]]))
            else:
                save_operations.append((layer_name, list(kv_cache)))

        self._submit_sfa_save_operations(save_operations)

        _dsa_prof.end(_sfa_t)
        return output_padded
