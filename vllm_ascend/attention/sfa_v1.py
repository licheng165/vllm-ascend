import json
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
from vllm_ascend.attention.mtp_dw_diag import (
    diagnostic_int_checksum,
    diagnostic_values_to_list,
    scratch_live_slot_aliases,
    scratch_target_safety,
)
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


def _mtp_dw_diag_enabled() -> bool:
    return envs.VLLM_ASCEND_MTP_DW_DIAG


def _mtp_dw_event(stage: str, **fields: Any) -> None:
    if not _mtp_dw_diag_enabled():
        return
    payload = {"schema": 1, "stage": stage, "owner": "vllm_ascend_sfa"}
    payload.update(fields)
    logger.info("[MTP_DW] %s", json.dumps(payload, separators=(",", ":")))


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
                or binding.bridge != existing.bridge
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


def _update_dsa_split_boundary_in_place(
    attn_metadata: Any,
    cached_tokens: list[int] | None,
    decode_window_size: int,
) -> torch.Tensor:
    """Update the builder-owned row boundary without temporary device tensors."""
    split_boundary = attn_metadata.split_boundary
    boundary_cpu = attn_metadata.decode_split_boundary_cpu
    boundary_cpu_tensor = attn_metadata.decode_split_boundary_cpu_tensor
    row_req_indices_cpu = attn_metadata.decode_req_indices_cpu
    if split_boundary is None or boundary_cpu is None or boundary_cpu_tensor is None or row_req_indices_cpu is None:
        raise RuntimeError(
            "DSA sparse boundary backing storage is incomplete. Rebuild "
            "attention metadata with the configured max_num_batched_tokens."
        )

    num_rows = int(split_boundary.shape[0])
    if (
        len(boundary_cpu) < num_rows
        or int(boundary_cpu_tensor.shape[0]) < num_rows
        or len(row_req_indices_cpu) < num_rows
    ):
        raise RuntimeError(
            "DSA sparse boundary active view exceeds its backing storage: "
            f"num_rows={num_rows}, boundary_cpu={len(boundary_cpu)}, "
            f"boundary_tensor={int(boundary_cpu_tensor.shape[0])}, "
            f"row_req_indices={len(row_req_indices_cpu)}."
        )

    seq_lens_cpu = attn_metadata.seq_lens_cpu
    num_reqs = int(seq_lens_cpu.shape[0])
    row_req_indices = np.asarray(
        row_req_indices_cpu[:num_rows],
        dtype=np.int32,
    )
    valid_rows = row_req_indices >= 0
    if np.any(row_req_indices[valid_rows] >= num_reqs):
        bad_row = int(
            np.flatnonzero(
                valid_rows & (row_req_indices >= num_reqs)
            )[0]
        )
        raise RuntimeError(
            "DSA sparse row references a request outside seq_lens: "
            f"row={bad_row}, "
            f"request_index={int(row_req_indices[bad_row])}, "
            f"num_reqs={num_reqs}."
        )

    has_cached_frontier = cached_tokens is not None
    if (
        has_cached_frontier
        and decode_window_size <= 0
        and len(cached_tokens) == 0
        and attn_metadata.num_decode_tokens > 0
    ):
        raise RuntimeError("LMCache sparse remap has decode rows but no request boundaries")

    if np.any(valid_rows) and (
        has_cached_frontier or decode_window_size > 0
    ):
        if has_cached_frontier:
            cached_count = min(len(cached_tokens), num_reqs)
            if cached_count == num_reqs:
                request_boundaries = np.asarray(
                    cached_tokens[:cached_count],
                    dtype=np.int32,
                )
            else:
                request_boundaries = np.zeros(
                    num_reqs,
                    dtype=np.int32,
                )
                request_boundaries[:cached_count] = np.asarray(
                    cached_tokens[:cached_count],
                    dtype=np.int32,
                )
        else:
            request_boundaries = np.empty(
                num_reqs,
                dtype=np.int32,
            )
        if decode_window_size > 0:
            if isinstance(seq_lens_cpu, torch.Tensor):
                seq_lens = seq_lens_cpu.detach().numpy()
            else:
                seq_lens = np.asarray(seq_lens_cpu)
            current_positions = np.maximum(
                seq_lens[:num_reqs].astype(np.int64, copy=False) - 1,
                0,
            )
            window_starts = (
                current_positions // decode_window_size
                * decode_window_size
            )
            if has_cached_frontier:
                np.minimum(
                    window_starts,
                    request_boundaries,
                    out=request_boundaries,
                    casting="unsafe",
                )
            else:
                request_boundaries[:] = window_starts
        boundary_cpu[:num_rows][valid_rows] = request_boundaries[
            row_req_indices[valid_rows]
        ]

    split_boundary.copy_(boundary_cpu_tensor[:num_rows])
    attn_metadata.decode_split_boundary = split_boundary
    return split_boundary


def _resolve_sparse_cached_tokens_by_request(
    attn_metadata: Any,
    request_ids: Any,
) -> list[int]:
    """Resolve strict connector frontiers in the native request order."""
    row_req_indices = attn_metadata.decode_req_indices_cpu
    if row_req_indices is None:
        raise RuntimeError("[SFA sparse remap] row/request mapping is unavailable.")
    decode_request_indices = sorted(
        {
            int(request_index)
            for request_index in row_req_indices
            if int(request_index) >= 0
        }
    )
    request_ids = list(request_ids) if request_ids is not None else []
    if decode_request_indices and decode_request_indices[-1] >= len(request_ids):
        raise RuntimeError(
            "[SFA sparse remap] active request IDs do not cover all decode rows."
        )
    decode_request_ids = [
        request_ids[request_index] for request_index in decode_request_indices
    ]
    resolved = get_lmcache_sparse_cached_tokens(decode_request_ids)
    cached_tokens = [0] * int(attn_metadata.seq_lens_cpu.shape[0])
    for request_index, committed_end in zip(
        decode_request_indices, resolved, strict=True
    ):
        cached_tokens[request_index] = int(committed_end)
    return cached_tokens


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
    if isinstance(seq_lens_cpu, torch.Tensor):
        seq_lens = seq_lens_cpu.detach().numpy().reshape(-1)
    else:
        seq_lens = np.asarray(seq_lens_cpu).reshape(-1)
    if int(boundary.numel()) != int(prompt_rows_np.size) or row_req_indices_np.size != prompt_rows_np.size:
        raise RuntimeError(
            "[SFA sparse remap] boundary shapes differ: "
            f"boundary={tuple(boundary.shape)}, "
            f"prompt_rows={tuple(prompt_rows_np.shape)}, "
            f"row_req_indices={tuple(row_req_indices_np.shape)}."
        )

    valid_rows = row_req_indices_np >= 0
    decode_request_indices_np = np.unique(
        row_req_indices_np[valid_rows]
    )
    if (
        decode_request_indices_np.size
        and int(decode_request_indices_np[-1]) >= len(seq_lens)
    ):
        request_index = int(decode_request_indices_np[-1])
        raise RuntimeError(
            "[SFA staged graph POC] decode row references request "
            f"{request_index}, but only {len(seq_lens)} sequence lengths "
            "are available."
        )
    decode_request_indices = decode_request_indices_np.tolist()

    cached_tokens_by_request = np.zeros(len(seq_lens), dtype=np.int32)
    cached_request_mask = np.zeros(len(seq_lens), dtype=np.bool_)
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
                cached_tokens_by_request[
                    decode_request_indices_np
                ] = np.asarray(resolved_tokens, dtype=np.int32)
                cached_request_mask[decode_request_indices_np] = True
        else:
            if len(cached_tokens) != len(decode_request_indices):
                raise RuntimeError(
                    f"[SFA_ROUTE] action=fatal reason={StagedSFARouteReason.FRONTIER_COUNT_MISMATCH.value}"
                )
            cached_tokens_by_request[
                decode_request_indices_np
            ] = np.asarray(cached_tokens, dtype=np.int32)
            cached_request_mask[decode_request_indices_np] = True
    decode_window_size = _decode_window_save_window_size()
    boundary_rows = prompt_rows_np.copy()
    if decode_request_indices_np.size:
        request_boundaries = np.zeros(
            len(seq_lens),
            dtype=np.int32,
        )
        if decode_window_size > 0:
            current_positions = np.maximum(
                seq_lens.astype(np.int64, copy=False) - 1,
                0,
            )
            request_boundaries[:] = (
                current_positions // decode_window_size
                * decode_window_size
            )
            np.minimum(
                request_boundaries,
                cached_tokens_by_request,
                out=request_boundaries,
                where=cached_request_mask,
            )
        else:
            request_boundaries[cached_request_mask] = (
                cached_tokens_by_request[cached_request_mask]
            )
        rows_with_dynamic_boundary = valid_rows.copy()
        if decode_window_size <= 0:
            rows_with_dynamic_boundary[valid_rows] = (
                cached_request_mask[
                    row_req_indices_np[valid_rows]
                ]
            )
        boundary_rows[rows_with_dynamic_boundary] = (
            request_boundaries[
                row_req_indices_np[rows_with_dynamic_boundary]
            ]
        )

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
    """Validate request-level union scratch cannot alias live KV positions."""
    width = int(index_topk)
    if width <= 0:
        raise RuntimeError(f"DSA compact scratch requires a positive index_topk, got {width}.")

    boundaries = np.asarray(boundary_rows, dtype=np.int64).reshape(-1)
    request_rows = np.asarray(row_req_indices, dtype=np.int64).reshape(-1)
    if boundaries.size != request_rows.size:
        raise RuntimeError(
            "DSA compact scratch metadata shapes differ: "
            f"boundaries={boundaries.size}, request_rows={request_rows.size}."
        )

    if scratch_capacity is None or int(scratch_capacity) < width:
        raise RuntimeError(
            "DSA request-union scratch reservation is missing or too small: "
            f"scratch_capacity={scratch_capacity}, index_topk={width}."
        )
    capacity = int(scratch_capacity)
    for request_index in sorted(
        {int(value) for value in request_rows if int(value) >= 0}
    ):
        rows = np.flatnonzero(request_rows == request_index)
        if rows.size * width > capacity:
            raise RuntimeError(
                "DSA request-union scratch reservation is too small: "
                f"request={request_index}, rows={rows.size}, "
                f"index_topk={width}, scratch_capacity={capacity}."
            )
        request_boundaries = boundaries[rows]
        if np.any(
            (request_boundaries != 0)
            & (request_boundaries < capacity)
        ):
            raise RuntimeError(
                "DSA request-union scratch would alias live KV positions: "
                f"request={request_index}, boundaries="
                f"{request_boundaries.tolist()}, "
                f"scratch_capacity={capacity}."
            )


def _fixed_staged_decode_mtp(
    row_req_indices: Any,
    request_count: int,
    row_count: int,
    *,
    pure_decode: bool,
) -> int | None:
    """Return the MTP for the request-major staged layout, else fall back."""
    if (
        not pure_decode
        or request_count <= 0
        or row_req_indices is None
    ):
        return None
    request_rows = np.asarray(row_req_indices, dtype=np.int64).reshape(-1)
    if request_rows.size < row_count:
        return None
    request_rows = request_rows[:row_count]
    valid_rows = request_rows[request_rows >= 0]
    if valid_rows.size:
        if np.any(valid_rows >= request_count):
            return None
        counts = np.bincount(valid_rows, minlength=request_count)
        max_mtp = int(counts.max(initial=0))
        if max_mtp > 2:
            raise RuntimeError(
                "staged sparse-index preparation only supports MTP=1 or "
                f"MTP=2; got MTP={max_mtp}"
            )
    if row_count % request_count:
        return None
    mtp = row_count // request_count
    if mtp not in (1, 2):
        return None
    expected = np.repeat(
        np.arange(request_count, dtype=np.int64),
        mtp,
    )
    if not np.array_equal(request_rows, expected):
        return None
    return mtp


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
    decode_current_positions_cpu: Any = None
    split_boundary: torch.Tensor | None = None
    decode_split_boundary_cpu: Any = None
    decode_split_boundary_cpu_tensor: torch.Tensor | None = None
    decode_split_boundary: torch.Tensor | None = None
    decode_scratch_base: torch.Tensor | None = None
    decode_scratch_base_compact: torch.Tensor | None = None
    decode_scratch_base_cpu: Any = None
    decode_scratch_capacity: int | None = None
    decode_target_slot_mapping: torch.Tensor | None = None
    decode_selected_tokens: torch.Tensor | None = None
    decode_selected_counts: torch.Tensor | None = None
    decode_union_mapping_workspace: torch.Tensor | None = None
    need_sparse_lmcache_payload: bool = False
    staged_sfa_payload_validated: bool = False
    prompt_lens_cpu_rows: Any = None
    decode_remap_boundary: torch.Tensor | None = None
    decode_remap_boundary_ready: bool = False


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
        if self.dsa_shrink_latent and self.decode_threshold > 2:
            raise ValueError(
                "staged sparse-index preparation only supports MTP=1 or "
                f"MTP=2; got MTP={self.decode_threshold}"
            )
        hf_config = self.model_config.hf_config
        hf_text_config = self.model_config.hf_text_config
        self.index_topk = int(
            getattr(
                hf_text_config or hf_config,
                "topk_tokens",
                getattr(hf_text_config or hf_config, "index_topk", 2048),
            )
        )
        if self.dsa_shrink_latent and self.index_topk % self.block_size:
            raise ValueError(
                "DSA index_topk must be an integer multiple of block_size: "
                f"index_topk={self.index_topk}, block_size={self.block_size}. "
                "Configure index_topk to N * block_size."
            )
        self._dsa_target_position_offsets = (
            torch.arange(self.index_topk, dtype=torch.long, device=device) if self.dsa_shrink_latent else None
        )

        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.scratch_capacity = self.decode_threshold * self.index_topk
        if self.dsa_shrink_latent:
            max_num_rows = vllm_config.scheduler_config.max_num_batched_tokens
            if not isinstance(max_num_rows, int) or max_num_rows <= 0:
                raise ValueError(
                    "DSA sparse metadata requires a positive integer "
                    "max_num_batched_tokens, got "
                    f"{max_num_rows!r}. Configure scheduler "
                    "max_num_batched_tokens before initializing attention."
                )
            if not isinstance(max_num_reqs, int) or max_num_reqs <= 0:
                raise ValueError(f"DSA sparse metadata requires a positive integer max_num_seqs, got {max_num_reqs!r}.")
            self._dsa_max_num_rows = max_num_rows
            self._dsa_max_num_reqs = max_num_reqs

            # Builder-owned backing storage. Metadata instances expose only
            # active-prefix views, while storage addresses remain stable across
            # scheduler steps for a later staged-graph merge.
            self._dsa_prompt_lens_cpu = np.empty(
                max_num_rows,
                dtype=np.int32,
            )
            self._dsa_split_boundary_cpu = np.empty(max_num_rows, dtype=np.int32)
            self._dsa_req_indices_cpu = np.empty(max_num_rows, dtype=np.int32)
            self._dsa_row_offsets_cpu = np.empty(max_num_rows, dtype=np.int32)
            self._dsa_current_positions_cpu = np.empty(max_num_rows, dtype=np.int64)
            self._dsa_valid_row_indices_cpu = np.empty(max_num_rows, dtype=np.int32)
            self._dsa_compact_req_indices_cpu = np.empty(max_num_rows, dtype=np.int32)
            self._dsa_prompt_lens_cpu_tensor = torch.from_numpy(
                self._dsa_prompt_lens_cpu
            )
            self._dsa_split_boundary_cpu_tensor = torch.from_numpy(self._dsa_split_boundary_cpu)
            self._dsa_req_indices_cpu_tensor = torch.from_numpy(self._dsa_req_indices_cpu)
            self._dsa_row_offsets_cpu_tensor = torch.from_numpy(self._dsa_row_offsets_cpu)
            self._dsa_valid_row_indices_cpu_tensor = torch.from_numpy(
                self._dsa_valid_row_indices_cpu
            )
            self._dsa_compact_req_indices_cpu_tensor = torch.from_numpy(
                self._dsa_compact_req_indices_cpu
            )
            self._dsa_prompt_lens = torch.empty(
                max_num_rows,
                dtype=torch.int32,
                device=device,
            )
            self._dsa_split_boundary = torch.empty(max_num_rows, dtype=torch.int32, device=device)
            self._dsa_req_indices = torch.empty(max_num_rows, dtype=torch.int32, device=device)
            self._dsa_row_offsets = torch.empty(max_num_rows, dtype=torch.int32, device=device)
            self._dsa_selected_tokens = torch.empty(
                (max_num_reqs, self.scratch_capacity),
                dtype=torch.int32,
                device=device,
            )
            # One 64-byte row per request prevents different AIVs from
            # updating counts in the same cacheline.
            self._dsa_selected_counts = torch.empty((max_num_reqs, 16), dtype=torch.int32, device=device)
            self._dsa_target_slots = torch.empty(
                (max_num_reqs, self.scratch_capacity),
                dtype=torch.long,
                device=device,
            )
            self._dsa_union_mapping = torch.empty(
                (max_num_reqs, self.scratch_capacity),
                dtype=torch.int32,
                device=device,
            )
            fixed_query_starts = np.arange(
                max_num_reqs + 1,
                dtype=np.int32,
            )
            self._dsa_fixed_query_starts_cpu = np.stack(
                (
                    fixed_query_starts,
                    fixed_query_starts * 2,
                )
            )
        else:
            self._dsa_max_num_rows = 0
            self._dsa_max_num_reqs = 0
            self._dsa_prompt_lens_cpu = None
            self._dsa_split_boundary_cpu = None
            self._dsa_req_indices_cpu = None
            self._dsa_row_offsets_cpu = None
            self._dsa_current_positions_cpu = None
            self._dsa_valid_row_indices_cpu = None
            self._dsa_compact_req_indices_cpu = None
            self._dsa_prompt_lens_cpu_tensor = None
            self._dsa_split_boundary_cpu_tensor = None
            self._dsa_req_indices_cpu_tensor = None
            self._dsa_row_offsets_cpu_tensor = None
            self._dsa_valid_row_indices_cpu_tensor = None
            self._dsa_compact_req_indices_cpu_tensor = None
            self._dsa_prompt_lens = None
            self._dsa_split_boundary = None
            self._dsa_req_indices = None
            self._dsa_row_offsets = None
            self._dsa_selected_tokens = None
            self._dsa_selected_counts = None
            self._dsa_target_slots = None
            self._dsa_union_mapping = None
            self._dsa_fixed_query_starts_cpu = None
        self._dsa_fixed_layout_signature = None
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
        self.decode_req_indices_compact = torch.empty_like(
            self.decode_remap_boundary
        )
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
        if self.dsa_shrink_latent:
            if num_input_tokens > self._dsa_max_num_rows:
                raise RuntimeError(
                    "DSA sparse row metadata capacity exceeded: "
                    f"num_input_tokens={num_input_tokens}, "
                    f"max_num_batched_tokens={self._dsa_max_num_rows}. "
                    "Increase scheduler max_num_batched_tokens."
                )
            if num_reqs > self._dsa_max_num_reqs:
                raise RuntimeError(
                    "DSA sparse request metadata capacity exceeded: "
                    f"num_reqs={num_reqs}, "
                    f"max_num_seqs={self._dsa_max_num_reqs}. "
                    "Increase scheduler max_num_seqs."
                )

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
        row_offsets_rows = None
        decode_scratch_base_rows = None
        decode_scratch_base_compact = None
        decode_scratch_capacity = self.scratch_capacity
        decode_target_slot_mapping = None
        decode_selected_tokens = None
        decode_selected_counts = None
        decode_union_mapping_workspace = None
        need_sparse_lmcache_payload = False
        num_decode_rows = 0
        decode_req_indices_cpu = None
        split_boundary_cpu = None
        split_boundary_cpu_tensor = None
        split_boundary_rows = None
        current_positions = None
        plens_cpu = common_attn_metadata.prompt_lens_cpu if self.dsa_shrink_latent else None
        if plens_cpu is not None:
            assert self._dsa_prompt_lens_cpu is not None
            assert self._dsa_split_boundary_cpu is not None
            assert self._dsa_req_indices_cpu is not None
            assert self._dsa_row_offsets_cpu is not None
            assert self._dsa_current_positions_cpu is not None
            assert self._dsa_valid_row_indices_cpu is not None
            assert self._dsa_compact_req_indices_cpu is not None
            assert self._dsa_prompt_lens_cpu_tensor is not None
            assert self._dsa_split_boundary_cpu_tensor is not None
            assert self._dsa_req_indices_cpu_tensor is not None
            assert self._dsa_row_offsets_cpu_tensor is not None
            assert self._dsa_valid_row_indices_cpu_tensor is not None
            assert self._dsa_compact_req_indices_cpu_tensor is not None
            assert self._dsa_prompt_lens is not None
            assert self._dsa_split_boundary is not None
            assert self._dsa_req_indices is not None
            assert self._dsa_row_offsets is not None

            plens_cpu = np.asarray(plens_cpu, dtype=np.int32)
            rows = self._dsa_prompt_lens_cpu[:num_input_tokens]
            boundary_rows = self._dsa_split_boundary_cpu[:num_input_tokens]
            req_rows = self._dsa_req_indices_cpu[:num_input_tokens]
            row_offsets = self._dsa_row_offsets_cpu[:num_input_tokens]
            valid_rows = self._dsa_valid_row_indices_cpu[:num_input_tokens]
            compact_req_indices = self._dsa_compact_req_indices_cpu[
                :num_input_tokens
            ]
            current_positions = (
                self._dsa_current_positions_cpu[:num_input_tokens]
                if envs.VLLM_ASCEND_MTP_DW_DEEP_DIAG and self.dsa_shrink_latent == 2
                else None
            )
            if current_positions is not None:
                current_positions.fill(0)
            n_real = min(len(plens_cpu), num_reqs)
            computed = common_attn_metadata.num_computed_tokens_cpu[:n_real].numpy()
            fixed_decode_width = 0
            if (
                common_attn_metadata.attn_state
                == AscendAttentionState.DecodeOnly
            ):
                fixed_decode_width = 1
            elif (
                common_attn_metadata.attn_state
                == AscendAttentionState.SpecDecoding
            ):
                fixed_decode_width = self.decode_threshold
            if fixed_decode_width in (1, 2):
                qsl = common_attn_metadata.query_start_loc_cpu[
                    : n_real + 1
                ].numpy()
                assert self._dsa_fixed_query_starts_cpu is not None
                fixed_query_starts = (
                    self._dsa_fixed_query_starts_cpu[
                        fixed_decode_width - 1,
                        : n_real + 1,
                    ]
                )
            else:
                qsl = None
                fixed_query_starts = None
            fixed_width_decode = (
                not self.enable_dsa_cp
                and fixed_decode_width in (1, 2)
                and num_input_tokens
                == num_reqs * fixed_decode_width
                and num_actual_tokens
                == len(plens_cpu) * fixed_decode_width
                and n_real == len(plens_cpu)
                and np.all(computed >= plens_cpu[:n_real])
                and np.array_equal(qsl, fixed_query_starts)
            )
            if fixed_width_decode:
                num_decode_rows = num_actual_tokens
                signature = (
                    fixed_decode_width,
                    num_input_tokens,
                    tuple(common_attn_metadata.request_ids or ()),
                    tuple(map(int, plens_cpu)),
                )
                if signature != self._dsa_fixed_layout_signature:
                    rows.fill(0)
                    rows[:num_decode_rows].reshape(
                        n_real,
                        fixed_decode_width,
                    )[:] = plens_cpu[:, None]
                    boundary_rows[:] = rows
                    req_rows.fill(-1)
                    request_indices = np.arange(
                        n_real,
                        dtype=np.int32,
                    )
                    req_rows[:num_decode_rows].reshape(
                        n_real,
                        fixed_decode_width,
                    )[:] = request_indices[:, None]
                    row_offsets.fill(0)
                    row_offsets[:num_decode_rows].reshape(
                        n_real,
                        fixed_decode_width,
                    )[:] = np.arange(
                        fixed_decode_width,
                        dtype=np.int32,
                    )
                    valid_rows.fill(0)
                    valid_rows[:num_decode_rows] = np.arange(
                        num_decode_rows,
                        dtype=np.int32,
                    )
                    compact_req_indices.fill(-1)
                    compact_req_indices[:num_decode_rows].reshape(
                        n_real,
                        fixed_decode_width,
                    )[:] = request_indices[:, None]
                    self._dsa_prompt_lens[:num_input_tokens].copy_(
                        self._dsa_prompt_lens_cpu_tensor[:num_input_tokens]
                    )
                    self._dsa_split_boundary[:num_input_tokens].copy_(
                        self._dsa_split_boundary_cpu_tensor[:num_input_tokens]
                    )
                    self._dsa_req_indices[:num_input_tokens].copy_(
                        self._dsa_req_indices_cpu_tensor[:num_input_tokens]
                    )
                    self._dsa_row_offsets[:num_input_tokens].copy_(
                        self._dsa_row_offsets_cpu_tensor[:num_input_tokens]
                    )
                    self.decode_valid_row_indices[:num_decode_rows].copy_(
                        self._dsa_valid_row_indices_cpu_tensor[
                            :num_decode_rows
                        ]
                    )
                    self.decode_req_indices_compact[:num_decode_rows].copy_(
                        self._dsa_compact_req_indices_cpu_tensor[
                            :num_decode_rows
                        ]
                    )
                    self._dsa_fixed_layout_signature = signature
                if current_positions is not None:
                    current_positions[:num_decode_rows].reshape(
                        n_real,
                        fixed_decode_width,
                    )[:] = (
                        computed[:n_real, None]
                        + np.arange(
                            fixed_decode_width,
                            dtype=np.int32,
                        )
                    )
            else:
                self._dsa_fixed_layout_signature = None
                rows.fill(0)
                boundary_rows.fill(0)
                req_rows.fill(-1)
                row_offsets.fill(0)
                valid_rows.fill(0)
                compact_req_indices.fill(-1)
                qsl = common_attn_metadata.query_start_loc_cpu[
                    : n_real + 1
                ].numpy()
                for r in range(n_real):
                    s, e = int(qsl[r]), int(qsl[r + 1])
                    plen = int(plens_cpu[r])
                    first_decode = max(s, s + plen - int(computed[r]))
                    if first_decode < e:
                        count = e - first_decode
                        rows[first_decode:e] = plen
                        boundary_rows[first_decode:e] = plen
                        req_rows[first_decode:e] = r
                        computed_start = int(computed[r]) + first_decode - s
                        for row_index in range(first_decode, e):
                            offset = row_index - first_decode
                            row_offsets[row_index] = offset
                            compact_index = num_decode_rows + offset
                            valid_rows[compact_index] = row_index
                            compact_req_indices[compact_index] = r
                            if current_positions is not None:
                                current_positions[row_index] = (
                                    computed_start + offset
                                )
                        num_decode_rows += count
                self._dsa_prompt_lens[:num_input_tokens].copy_(
                    self._dsa_prompt_lens_cpu_tensor[:num_input_tokens]
                )
                self._dsa_split_boundary[:num_input_tokens].copy_(
                    self._dsa_split_boundary_cpu_tensor[:num_input_tokens]
                )
                self._dsa_req_indices[:num_input_tokens].copy_(
                    self._dsa_req_indices_cpu_tensor[:num_input_tokens]
                )
                self._dsa_row_offsets[:num_input_tokens].copy_(
                    self._dsa_row_offsets_cpu_tensor[:num_input_tokens]
                )
                if num_decode_rows:
                    self.decode_valid_row_indices[:num_decode_rows].copy_(
                        self._dsa_valid_row_indices_cpu_tensor[
                            :num_decode_rows
                        ]
                    )
                    self.decode_req_indices_compact[:num_decode_rows].copy_(
                        self._dsa_compact_req_indices_cpu_tensor[
                            :num_decode_rows
                        ]
                    )

            split_boundary_cpu = boundary_rows
            decode_req_indices_cpu = req_rows
            split_boundary_cpu_tensor = self._dsa_split_boundary_cpu_tensor[:num_input_tokens]
            split_boundary_rows = self._dsa_split_boundary[:num_input_tokens]
            prompt_lens_rows = self._dsa_prompt_lens[:num_input_tokens]
            decode_req_indices_rows = self._dsa_req_indices[:num_input_tokens]
            row_offsets_rows = self._dsa_row_offsets[:num_input_tokens]
            decode_valid_rows_all = num_decode_rows == num_input_tokens
            if num_decode_rows:
                decode_valid_row_indices = self.decode_valid_row_indices[
                    :num_decode_rows
                ]
                decode_req_indices_compact = self.decode_req_indices_compact[
                    :num_decode_rows
                ]
                decode_req_indices_compact_cpu = compact_req_indices[
                    :num_decode_rows
                ]
            need_sparse_lmcache_payload = (
                self.dsa_shrink_latent != 3
                and staged_sfa_connector_supports_sparse_load()
            )
            if num_decode_rows:
                assert self._dsa_target_slots is not None
                assert self._dsa_selected_tokens is not None
                assert self._dsa_selected_counts is not None
                assert self._dsa_union_mapping is not None
                req_ids = common_attn_metadata.request_ids
                if req_ids is not None:
                    decode_request_ids_compact = list(req_ids[:num_reqs])
                # This ordered-unique list is also the fused kernel's ownership
                # map: one AIV owns each complete source row.
                decode_target_slot_mapping = self._dsa_target_slots[:num_reqs]
                decode_selected_tokens = self._dsa_selected_tokens[:num_reqs]
                decode_selected_counts = self._dsa_selected_counts[:num_reqs]
                decode_union_mapping_workspace = self._dsa_union_mapping[
                    :num_reqs
                ]

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
            decode_req_indices_cpu=decode_req_indices_cpu,
            decode_valid_row_indices=decode_valid_row_indices,
            decode_valid_rows_all=decode_valid_rows_all,
            decode_req_indices_compact=decode_req_indices_compact,
            decode_req_indices_compact_cpu=decode_req_indices_compact_cpu,
            decode_request_ids_compact=decode_request_ids_compact,
            decode_row_offsets=row_offsets_rows,
            decode_current_positions_cpu=(
                current_positions if decode_req_indices_rows is not None else None
            ),
            decode_split_boundary_cpu=split_boundary_cpu,
            split_boundary=split_boundary_rows,
            decode_split_boundary_cpu_tensor=split_boundary_cpu_tensor,
            decode_scratch_base=decode_scratch_base_rows,
            decode_scratch_base_compact=decode_scratch_base_compact,
            decode_scratch_base_cpu=None,
            decode_scratch_capacity=decode_scratch_capacity,
            decode_target_slot_mapping=decode_target_slot_mapping,
            decode_selected_tokens=decode_selected_tokens,
            decode_selected_counts=decode_selected_counts,
            decode_union_mapping_workspace=decode_union_mapping_workspace,
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
        self.block_size = self.vllm_config.cache_config.block_size
        speculative_config = self.vllm_config.speculative_config
        self.decode_threshold = 1 + (
            speculative_config.num_speculative_tokens
            if speculative_config is not None
            else 0
        )
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
        if self.dsa_shrink_latent and self.decode_threshold > 2:
            raise ValueError(
                "staged sparse-index preparation only supports MTP=1 or "
                f"MTP=2; got MTP={self.decode_threshold}"
            )
        self.enable_staged_sfa_graph = staged_sfa_graph_configured(self.vllm_config)
        self._staged_sfa_graph_capture_sizes = (
            staged_sfa_graph_capture_sizes(self.vllm_config) if self.enable_staged_sfa_graph else ()
        )
        self._staged_sfa_capture_state = _StagedSFACaptureState()
        self._staged_sfa_bridge_buffers: tuple[torch.Tensor, ...] | None = None
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
        """Return why this step cannot use its authorized fixed-layout graph."""
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

        token_capacity = int(hidden_states.shape[0])
        capture_sizes = getattr(
            self,
            "_staged_sfa_graph_capture_sizes",
            None,
        )
        if capture_sizes is None:
            # Compatibility for lightweight test/downstream implementations.
            capture_sizes = staged_sfa_graph_capture_sizes(self.vllm_config)
        authorized_key = getattr(
            forward_context,
            "staged_sfa_graph_key",
            None,
        )
        if authorized_key is None or authorized_key.token_capacity != token_capacity:
            return "the runner did not authorize this staged SFA token capacity"
        graph_key = authorized_key
        if graph_key.max_query_len > 2:
            return "staged sparse-index preparation only supports MTP=1 or MTP=2"
        if graph_key.query_profile == StagedSFAQueryProfile.DECODE_Q1:
            if graph_key.max_query_len != 1 or graph_key.request_capacity != token_capacity:
                return "the Q1 staged SFA graph key is structurally invalid"
        elif graph_key.query_profile == StagedSFAQueryProfile.SPEC_FIXED:
            if (
                graph_key.max_query_len != self.decode_threshold
                or graph_key.max_query_len <= 1
                or graph_key.request_capacity * graph_key.max_query_len
                != token_capacity
            ):
                return "the fixed-width MTP staged SFA graph key is structurally invalid"
        else:
            return "the staged SFA query profile is unsupported"

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
            return "the PIECEWISE descriptor does not match the staged SFA graph key"

        if self.vllm_config.lora_config is not None:
            return "LoRA is configured"
        expected_state = (
            AscendAttentionState.DecodeOnly
            if graph_key.query_profile == StagedSFAQueryProfile.DECODE_Q1
            else AscendAttentionState.SpecDecoding
        )
        if attn_metadata.attn_state != expected_state:
            return "the attention state does not match the staged query profile"
        actual_rows = int(attn_metadata.num_actual_tokens)
        actual_requests = len(attn_metadata.decode_request_ids_compact or ())
        if (
            attn_metadata.num_input_tokens != token_capacity
            or actual_rows <= 0
            or actual_rows > token_capacity
            or attn_metadata.num_decode_tokens != actual_rows
            or actual_requests <= 0
            or actual_requests > graph_key.request_capacity
            or actual_rows != actual_requests * graph_key.max_query_len
        ):
            return "the real decode layout does not match the fixed staged graph width"
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
        if any(int(cache.shape[0]) < graph_key.request_capacity for cache in kv_cache):
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

        required_token_tensors = (
            attn_metadata.cos,
            attn_metadata.sin,
            attn_metadata.slot_mapping,
            attn_metadata.indexer_slot_mapping,
        )
        if any(tensor is None for tensor in required_token_tensors):
            return "required fixed-shape attention metadata is unavailable"
        if any(int(tensor.shape[0]) != token_capacity for tensor in required_token_tensors):
            return "the fixed-shape attention row count does not match the graph key"
        if (
            attn_metadata.cum_query_lens is None
            or attn_metadata.seq_lens is None
            or int(attn_metadata.cum_query_lens.shape[0])
            != graph_key.request_capacity
            or int(attn_metadata.seq_lens.shape[0])
            != graph_key.request_capacity
        ):
            return "the request metadata does not match the graph key"
        if (
            attn_metadata.block_table is None
            or attn_metadata.indexer_block_table is None
            or int(attn_metadata.block_table.shape[0])
            != graph_key.request_capacity
            or int(attn_metadata.indexer_block_table.shape[0])
            != graph_key.request_capacity
        ):
            return "the native block-table row count does not match the graph key"
        if (
            not staged_dummy_run
            and not attn_metadata.need_sparse_lmcache_payload
        ):
            return "the v1 sparse LMCache payload path is unavailable"
        if (
            attn_metadata.decode_req_indices is None
            or int(attn_metadata.decode_req_indices.numel()) != token_capacity
            or attn_metadata.decode_selected_tokens is None
            or int(attn_metadata.decode_selected_tokens.shape[0])
            != graph_key.request_capacity
            or attn_metadata.decode_selected_counts is None
            or int(attn_metadata.decode_selected_counts.shape[0])
            != graph_key.request_capacity
            or attn_metadata.decode_target_slot_mapping is None
            or int(attn_metadata.decode_target_slot_mapping.shape[0])
            != graph_key.request_capacity
            or attn_metadata.decode_union_mapping_workspace is None
            or int(attn_metadata.decode_union_mapping_workspace.shape[0])
            != graph_key.request_capacity
            or attn_metadata.decode_union_mapping_workspace.shape
            != attn_metadata.decode_selected_tokens.shape
        ):
            return "the request-union remap buffers do not match the graph key"

        request_ids = attn_metadata.decode_request_ids_compact
        full_request_ids = attn_metadata.req_ids
        if (
            request_ids is None
            or full_request_ids is None
            or len(request_ids) != actual_requests
            or len(full_request_ids) != actual_requests
            or tuple(request_ids) != tuple(full_request_ids)
            or len(set(request_ids)) != actual_requests
        ):
            return "the compact LMCache request ids are not the unique native request order"
        if (
            attn_metadata.prompt_lens_cpu_rows is None
            or attn_metadata.decode_req_indices_cpu is None
            or attn_metadata.seq_lens_cpu is None
            or attn_metadata.decode_remap_boundary is None
            or int(attn_metadata.decode_remap_boundary.shape[0])
            != token_capacity
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
        seq_lens_cpu = attn_metadata.seq_lens_cpu
        if isinstance(seq_lens_cpu, torch.Tensor):
            if seq_lens_cpu.device.type != "cpu":
                return "sequence-length validation metadata is not on CPU"
            seq_rows = seq_lens_cpu.detach().numpy().reshape(-1)
        else:
            seq_rows = np.asarray(seq_lens_cpu).reshape(-1)
        expected_request_rows = np.full(token_capacity, -1, dtype=np.int64)
        expected_request_rows[:actual_rows] = np.repeat(
            np.arange(actual_requests, dtype=np.int64),
            graph_key.max_query_len,
        )
        if (
            prompt_rows.size != token_capacity
            or np.any(prompt_rows[actual_rows:] != 0)
            or request_rows.size != token_capacity
            or not np.array_equal(request_rows, expected_request_rows)
            or seq_rows.size != graph_key.request_capacity
        ):
            return "the CPU row metadata does not match the staged graph layout"
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
        row_req_indices: torch.Tensor,
        request_block_table: torch.Tensor,
        selected_packed: torch.Tensor,
        selected_counts: torch.Tensor,
        target_slot_mapping: torch.Tensor,
        local_to_union_workspace: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
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
        (
            topk_indices,
            selected_packed,
            selected_count_values,
            target_slot_mapping,
        ) = prepare_sparse_indices(
            topk_indices,
            remap_boundary,
            row_req_indices=row_req_indices,
            request_block_table=request_block_table,
            selected_packed=selected_packed,
            selected_counts=selected_counts,
            target_slot_mapping=target_slot_mapping,
            block_size=self.block_size,
            need_packed=True,
            clear_invalid_rows=True,
            local_to_union_workspace=local_to_union_workspace,
            staged_mtp=int(topk_indices.shape[0])
            // int(request_block_table.shape[0]),
        )
        assert selected_packed is not None
        assert selected_count_values is not None
        assert target_slot_mapping is not None
        return (
            ql_nope,
            q_pe,
            topk_indices,
            selected_packed,
            selected_count_values,
            target_slot_mapping,
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
    ) -> tuple[torch.Tensor, ...]:
        return self._ensure_staged_sfa_bridge_buffers(hidden_states)

    def reset_staged_sfa_capture(self) -> None:
        self._staged_sfa_capture_state = _StagedSFACaptureState()
        self._dsa_idx_cache_t = None
        self._staged_sfa_bridge_buffers = None

    def seal_staged_sfa_capture(
        self,
        graph_keys: tuple[StagedSFAGraphKey, ...],
    ) -> None:
        self._staged_sfa_capture_state.seal(graph_keys)

    def _ensure_staged_sfa_bridge_buffers(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        buffers = getattr(self, "_staged_sfa_bridge_buffers", None)
        max_tokens = self._staged_sfa_graph_capture_sizes[-1]
        max_requests = max_tokens // self.decode_threshold
        scratch_capacity = self.decode_threshold * self.index_topk
        if buffers is None:
            context = get_forward_context()
            if (
                getattr(context, "cudagraph_runtime_mode", CUDAGraphMode.NONE)
                != CUDAGraphMode.NONE
            ):
                raise RuntimeError(
                    "staged SFA bridge storage was not allocated by eager "
                    "warmup before graph capture/replay"
                )
            buffers = (
                hidden_states.new_empty(
                    (
                        max_tokens,
                        self.local_num_heads,
                        self.kv_lora_rank,
                    )
                ),
                hidden_states.new_empty(
                    (
                        max_tokens,
                        self.local_num_heads,
                        self.qk_rope_head_dim,
                    )
                ),
                torch.empty(
                    (max_tokens, 1, self.index_topk),
                    dtype=torch.int32,
                    device=hidden_states.device,
                ),
                torch.empty(
                    (max_requests, scratch_capacity),
                    dtype=torch.int32,
                    device=hidden_states.device,
                ),
                torch.empty(
                    (max_requests,),
                    dtype=torch.int32,
                    device=hidden_states.device,
                ),
                torch.empty(
                    (max_requests, scratch_capacity),
                    dtype=torch.long,
                    device=hidden_states.device,
                ),
            )
            self._staged_sfa_bridge_buffers = buffers
        if any(tensor.device != hidden_states.device for tensor in buffers):
            raise RuntimeError(
                "staged SFA bridge storage moved to a different device"
            )
        if buffers[0].dtype != hidden_states.dtype:
            raise RuntimeError(
                "staged SFA bridge storage dtype differs from hidden states"
            )
        return buffers

    def _copy_to_staged_sfa_bridge(
        self,
        hidden_states: torch.Tensor,
        outputs: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        buffers = self._ensure_staged_sfa_bridge_buffers(hidden_states)
        if len(outputs) != len(buffers):
            raise RuntimeError(
                "staged SFA pre returned an unexpected bridge arity: "
                f"{len(outputs)}"
            )
        for source, destination in zip(outputs, buffers, strict=True):
            rows = int(source.shape[0])
            if (
                rows > int(destination.shape[0])
                or tuple(source.shape[1:])
                != tuple(destination.shape[1:])
            ):
                raise RuntimeError(
                    "staged SFA bridge output exceeds its fixed storage: "
                    f"source={tuple(source.shape)}, "
                    f"destination={tuple(destination.shape)}"
                )
            destination[:rows].copy_(source)
        return buffers

    def cross_layer_graph_pre(
        self,
        layer_name: str,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: M | None,
        need_gather_q_kv: bool,
        output: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
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
        row_req_indices = attn_metadata.decode_req_indices
        selected_packed = attn_metadata.decode_selected_tokens
        selected_counts = attn_metadata.decode_selected_counts
        target_slots = attn_metadata.decode_target_slot_mapping
        local_to_union_workspace = (
            attn_metadata.decode_union_mapping_workspace
        )
        if any(
            value is None
            for value in (
                row_req_indices,
                selected_packed,
                selected_counts,
                target_slots,
                local_to_union_workspace,
            )
        ):
            raise RuntimeError("staged SFA request-union buffers are unavailable")
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
            row_req_indices,
            attn_metadata.block_table,
            selected_packed,
            selected_counts,
            target_slots,
            local_to_union_workspace,
        )
        outputs = self._copy_to_staged_sfa_bridge(
            hidden_states,
            outputs,
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
        selected_counts: torch.Tensor,
        target_slots: torch.Tensor,
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
            request_ids = attn_metadata.decode_request_ids_compact
            if request_ids is None:
                raise RuntimeError("staged SFA request ids are unavailable")
            request_count = len(request_ids)
            wait_for_kv_layer_from_connector(
                layer_name,
                selected_tokens=selected_packed[:request_count],
                token_start_index=None,
                request_ids=request_ids,
                target_slot_mapping=target_slots[:request_count],
                selected_token_counts=selected_counts[:request_count],
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
        rows = graph_key.token_capacity
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

        # DSA Step B2 (compact-scratch decode): the indexer just produced topk.
        # Remap LMCache-selected entries to compact scratch rows [0..n_ret)
        # (the request's first ceil(k/block_size) latent blocks) and have
        # LMCache scatter exactly those tokens into scratch. Live-cache entries
        # keep their absolute positions and are read in place via the same
        # block table. Decode-window mode uses current_window_start as the
        # cache boundary instead of prompt_len.
        # All fixed-shape device math — no D2H sync. No-op without a connector.
        if (
            self.dsa_shrink_latent
            and attn_metadata.split_boundary is not None
            and attn_metadata.num_decode_tokens > 0
            and (attn_metadata.need_sparse_lmcache_payload or self.dsa_shrink_latent == 3)
        ):
            # _split_boundary is per row. Decode rows start at prompt length by
            # default; decode-window mode replaces it with current_window_start.
            # Prefill/padding rows carry 0 and stay untouched, so this also
            # covers mixed chunked-prefill + decode steps.
            # The packed front-list only feeds LMCache's selected_tokens; skip building
            # it (and its scatter) when no v1 connector will consume it (profiling /
            # no-offload runs). Production with an LMCache connector is unchanged.
            _need_packed = attn_metadata.need_sparse_lmcache_payload
            _topk_rows = int(topk_indices.shape[0])
            _split_boundary = attn_metadata.split_boundary
            _decode_window_size = _decode_window_save_window_size()
            _diag_remap_build = False
            _diag_current_window_start = None
            _diag_committed_end = None
            _cached_split_boundary = attn_metadata.decode_split_boundary
            if (
                _cached_split_boundary is not None
                and _cached_split_boundary.shape == _split_boundary.shape
                and _cached_split_boundary.device == _split_boundary.device
                and _cached_split_boundary.dtype == _split_boundary.dtype
            ):
                _split_boundary = _cached_split_boundary
            else:
                _lmcache_cached_tokens = _resolve_sparse_cached_tokens_by_request(
                    attn_metadata,
                    attn_metadata.req_ids,
                )
                if _lmcache_cached_tokens is not None or _decode_window_size > 0:
                    _split_boundary = _update_dsa_split_boundary_in_place(
                        attn_metadata,
                        _lmcache_cached_tokens,
                        _decode_window_size,
                    )
                    if _decode_window_size > 0 and _mtp_dw_diag_enabled():
                        # Diagnostics only; these host values are intentionally
                        # outside the production allocation-free contract.
                        _diag_current_window_start = [
                            max(int(seq_len) - 1, 0) // _decode_window_size * _decode_window_size
                            for seq_len in attn_metadata.seq_lens_cpu
                        ]
                    if _lmcache_cached_tokens is not None:
                        _diag_committed_end = _lmcache_cached_tokens
                else:
                    # Connector frontier expansion is step metadata, not layer
                    # data. Reuse it for every remaining SFA layer in this step.
                    attn_metadata.decode_split_boundary = _split_boundary
                _diag_remap_build = True
            _absolute_topk_for_diag = topk_indices
            _row_req_indices = attn_metadata.decode_req_indices
            if _row_req_indices is None:
                raise RuntimeError("DSA union remap requires row request indices")
            _selected_token_counts = attn_metadata.decode_selected_counts
            _staged_mtp = _fixed_staged_decode_mtp(
                attn_metadata.decode_req_indices_cpu,
                int(attn_metadata.block_table.shape[0]),
                _topk_rows,
                pure_decode=_is_pure_decode,
            )
            with _dsa_prof.section("prepare_sparse_indices"):
                (
                    topk_indices,
                    _sel_packed,
                    _selected_token_counts,
                    _target_slot_mapping,
                ) = prepare_sparse_indices(
                    topk_indices,
                    _split_boundary,
                    row_req_indices=_row_req_indices,
                    request_block_table=attn_metadata.block_table,
                    selected_packed=attn_metadata.decode_selected_tokens,
                    selected_counts=attn_metadata.decode_selected_counts,
                    target_slot_mapping=attn_metadata.decode_target_slot_mapping,
                    block_size=self.block_size,
                    need_packed=_need_packed,
                    clear_invalid_rows=_is_pure_decode,
                    local_to_union_workspace=(
                        attn_metadata.decode_union_mapping_workspace
                        if _staged_mtp is not None
                        else None
                    ),
                    staged_mtp=_staged_mtp,
                )
            _sparse_indices_padding_zeroed = _is_pure_decode
            _diag_context = get_forward_context() if _mtp_dw_diag_enabled() and _diag_remap_build else None
            if _diag_context is not None and getattr(_diag_context, "mtp_dw_diag_req_ids", None):
                _diag_req_ids = getattr(_diag_context, "dsa_req_ids", None)
                _diag_sampled_req_ids = getattr(_diag_context, "mtp_dw_diag_req_ids", set())
                _diag_row_req_indices = getattr(attn_metadata, "decode_req_indices_cpu", None)
                if _diag_row_req_indices is None:
                    _diag_row_req_indices = getattr(attn_metadata, "decode_req_indices", None)
                _diag_row_req_indices_list = diagnostic_values_to_list(_diag_row_req_indices)
                _diag_positions = (attn_metadata.seq_lens.to(torch.long) - 1).detach().cpu().tolist()
                _diag_boundaries = _split_boundary.detach().cpu().tolist()
                _diag_prompt_lens = (
                    _diag_context.dsa_prompt_lens.detach().cpu().tolist()
                    if getattr(_diag_context, "dsa_prompt_lens", None) is not None
                    else []
                )
                _diag_windows = (
                    diagnostic_values_to_list(_diag_current_window_start)
                    if _diag_current_window_start is not None
                    else []
                )
                _diag_committed = (
                    diagnostic_values_to_list(_diag_committed_end) if _diag_committed_end is not None else []
                )
                _diag_scratch = []
                _diag_absolute = _absolute_topk_for_diag.detach().cpu()
                _diag_packed = _sel_packed.detach().cpu() if _sel_packed is not None else None
                _diag_deep_req_ids = (
                    (getattr(_diag_context, "mtp_dw_deep_diag_req_ids", None) or set())
                    if envs.VLLM_ASCEND_MTP_DW_DEEP_DIAG and self.dsa_shrink_latent == 2
                    else set()
                )
                _diag_deep_emitted_req_ids = getattr(_diag_context, "mtp_dw_deep_diag_emitted_req_ids", None)
                if _diag_deep_req_ids:
                    if _diag_deep_emitted_req_ids is None:
                        _diag_deep_emitted_req_ids = set()
                        _diag_context.mtp_dw_deep_diag_emitted_req_ids = _diag_deep_emitted_req_ids
                    _diag_deep_req_ids = _diag_deep_req_ids - _diag_deep_emitted_req_ids
                _diag_row_offsets = (
                    diagnostic_values_to_list(getattr(attn_metadata, "decode_row_offsets", None))
                    if _diag_deep_req_ids
                    else []
                )
                _diag_current_positions = (
                    diagnostic_values_to_list(getattr(attn_metadata, "decode_current_positions_cpu", None))
                    if _diag_deep_req_ids
                    else []
                )
                _diag_target_slots = (
                    diagnostic_values_to_list(getattr(attn_metadata, "decode_target_slot_mapping", None))
                    if _diag_deep_req_ids
                    else []
                )
                _diag_selected_counts = (
                    _selected_token_counts.detach().cpu().tolist()
                    if _diag_deep_req_ids and _selected_token_counts is not None
                    else []
                )
                _diag_compact_row = (
                    {
                        source_row: int(req_index)
                        for source_row, req_index in enumerate(_diag_row_req_indices_list)
                        if int(req_index) >= 0
                    }
                    if _diag_deep_req_ids
                    else {}
                )
                _diag_deep_emitted_this_layer: set[str] = set()
                _diag_deep_payloads: dict[str, dict[str, Any]] = {}
                seen_scratch: dict[str, set[int]] = {}
                for row, req_index in enumerate(_diag_row_req_indices_list):
                    if req_index < 0 or row >= len(_diag_boundaries):
                        continue
                    req_id = (
                        str(_diag_req_ids[req_index])
                        if _diag_req_ids is not None and req_index < len(_diag_req_ids)
                        else None
                    )
                    if req_id not in _diag_sampled_req_ids:
                        continue
                    boundary = int(_diag_boundaries[row])
                    absolute_row = _diag_absolute[row].reshape(-1)
                    selected_absolute = absolute_row[(absolute_row >= 0) & (absolute_row < boundary)]
                    packed_row = (
                        _diag_packed[int(req_index)].reshape(-1)
                        if _diag_packed is not None and int(req_index) < _diag_packed.shape[0]
                        else torch.empty(0, dtype=torch.long)
                    )
                    scratch_base = int(_diag_scratch[row]) if row < len(_diag_scratch) else None
                    current_position = (
                        int(_diag_current_positions[row])
                        if row < len(_diag_current_positions)
                        else int(_diag_positions[req_index])
                        if req_index < len(_diag_positions)
                        else 0
                    )
                    prompt_len = int(_diag_prompt_lens[row]) if row < len(_diag_prompt_lens) else current_position
                    distance = min(
                        current_position % _decode_window_size,
                        (-current_position) % _decode_window_size,
                    )
                    sample_row = current_position - prompt_len < 3 or distance <= 4
                    _post_commit_req_ids = getattr(
                        _diag_context,
                        "mtp_dw_diag_post_commit_req_ids",
                        None,
                    )
                    if _post_commit_req_ids is not None and req_id in _post_commit_req_ids:
                        sample_row = True
                    committed = int(_diag_committed[req_index]) if req_index < len(_diag_committed) else None
                    current_window = int(_diag_windows[req_index]) if req_index < len(_diag_windows) else None
                    if req_id in _diag_deep_req_ids:
                        _diag_deep_emitted_this_layer.add(req_id)
                        row_offset = int(_diag_row_offsets[row]) if row < len(_diag_row_offsets) else 0
                        effective_scratch_base = scratch_base or 0
                        compact_row = _diag_compact_row.get(row)
                        selected_count = (
                            int(_diag_selected_counts[compact_row])
                            if compact_row is not None and compact_row < len(_diag_selected_counts)
                            else int(selected_absolute.numel())
                        )
                        packed_values = packed_row.tolist()
                        absolute_values = absolute_row.tolist()
                        selected_absolute_values = selected_absolute.tolist()
                        payload_width = len(packed_values)
                        block_size = int(kv_cache[0].shape[1])
                        block_table_row = attn_metadata.block_table[int(req_index)].detach().cpu().tolist()
                        scratch_safety = scratch_target_safety(
                            block_table_row,
                            effective_scratch_base,
                            selected_count,
                            boundary,
                            current_position,
                            block_size,
                        )
                        actual_target_slots = (
                            [int(value) for value in _diag_target_slots[compact_row]]
                            if compact_row is not None and compact_row < len(_diag_target_slots)
                            else None
                        )
                        if actual_target_slots is not None:
                            consumed_target_slots = actual_target_slots[:selected_count]
                            actual_aliases = sorted(
                                set(consumed_target_slots).intersection(scratch_safety["live_slots"])
                            )
                        else:
                            consumed_target_slots = []
                            actual_aliases = []
                        try:
                            derived_target_slots, live_slots, _ = scratch_live_slot_aliases(
                                block_table_row,
                                range(
                                    effective_scratch_base,
                                    effective_scratch_base + payload_width,
                                ),
                                boundary,
                                current_position,
                                block_size,
                            )
                            target_slots = actual_target_slots or derived_target_slots
                            consumed_target_slots = target_slots[:selected_count]
                            aliases = sorted(set(consumed_target_slots).intersection(live_slots))
                            if target_slots != derived_target_slots:
                                _mtp_dw_event(
                                    "fail",
                                    invariant="deep_target_slot_mapping",
                                    tp_rank=self.tp_rank,
                                    tp_world=self.tp_size,
                                    req=req_id,
                                    row=row,
                                    req_index=int(req_index),
                                    derived_sample=derived_target_slots[:8],
                                    actual_sample=target_slots[:8],
                                )
                        except ValueError as error:
                            _mtp_dw_event(
                                "fail",
                                invariant="deep_physical_slot_mapping",
                                tp_rank=self.tp_rank,
                                tp_world=self.tp_size,
                                req=req_id,
                                row=row,
                                req_index=int(req_index),
                                row_offset=row_offset,
                                scratch_base=effective_scratch_base,
                                current_position=current_position,
                                prompt_len=prompt_len,
                                committed_end=committed,
                                boundary=boundary,
                                detail=str(error),
                            )
                            target_slots = actual_target_slots or []
                            live_slots = scratch_safety["live_slots"]
                            aliases = actual_aliases
                        deep_common = {
                            "tp_rank": self.tp_rank,
                            "tp_world": self.tp_size,
                            "worker_rank": self.tp_rank,
                            "layer": layer_name,
                            "req": req_id,
                            "frontier": current_position,
                            "window_start": current_window,
                            "window_end": committed,
                            "kv_group": 0,
                            "row": row,
                            "req_index": int(req_index),
                            "row_offset": row_offset,
                            "scratch_base": effective_scratch_base,
                            "current_position": current_position,
                            "prompt_len": prompt_len,
                            "committed_end": committed,
                            "boundary": boundary,
                        }
                        _mtp_dw_event(
                            "deep",
                            event="row_mapping",
                            **deep_common,
                            selection_width=len(absolute_values),
                            lmcache_selected_count=selected_count,
                            selected_absolute_sample=selected_absolute_values[:8],
                            selected_absolute_checksum=diagnostic_int_checksum(selected_absolute_values),
                            live_physical_count=len(live_slots),
                            live_physical_sample=live_slots[:8],
                            live_physical_checksum=diagnostic_int_checksum(live_slots),
                            checksum_scope="first32",
                        )
                        _mtp_dw_event(
                            "deep",
                            event="scratch_target_safety",
                            **deep_common,
                            selected_count=selected_count,
                            target_logical_start=scratch_safety["target_logical_start"],
                            target_logical_end=scratch_safety["target_logical_end"],
                            target_block_start=scratch_safety["target_block_start"],
                            target_block_end=scratch_safety["target_block_end"],
                            target_block_values=scratch_safety["target_block_values"][:8],
                            target_block_checksum=diagnostic_int_checksum(
                                value for value in scratch_safety["target_block_values"] if value is not None
                            ),
                            valid_logical_end=scratch_safety["valid_logical_end"],
                            target_within_committed=scratch_safety["target_within_committed"],
                            target_beyond_current_sequence=scratch_safety["target_beyond_current_sequence"],
                            target_unmapped_count=scratch_safety["target_unmapped_count"],
                            target_live_intersection_count=len(scratch_safety["target_live_intersection"]),
                            target_live_intersection_sample=scratch_safety["target_live_intersection"][:8],
                            actual_target_live_intersection_count=len(aliases),
                            actual_target_live_intersection_sample=aliases[:8],
                        )
                        payload = _diag_deep_payloads.setdefault(
                            req_id,
                            {
                                "common": {
                                    "tp_rank": self.tp_rank,
                                    "tp_world": self.tp_size,
                                    "worker_rank": self.tp_rank,
                                    "layer": layer_name,
                                    "req": req_id,
                                    "frontier": committed,
                                    "window_start": (
                                        max(0, committed - _decode_window_size) if committed is not None else None
                                    ),
                                    "window_end": committed,
                                    "kv_group": 0,
                                    "committed_end": committed,
                                    "boundary": boundary,
                                },
                                "selection": [],
                                "slots": [],
                                "payload_count": 0,
                                "target_count": 0,
                                "selected_count": 0,
                                "aliases": set(),
                                "rows": [],
                            },
                        )
                        remaining = max(0, 32 - len(payload["selection"]))
                        consumed_packed_values = packed_values[:selected_count]
                        payload["selection"].extend(consumed_packed_values[:remaining])
                        payload["slots"].extend(consumed_target_slots[:remaining])
                        payload["payload_count"] += len(consumed_packed_values)
                        payload["target_count"] += len(consumed_target_slots)
                        payload["selected_count"] += selected_count
                        payload["aliases"].update(aliases)
                        payload["rows"].append(row)
                        if aliases:
                            _mtp_dw_event(
                                "fail",
                                invariant="scratch_live_slot_alias",
                                **deep_common,
                                intersection_count=len(aliases),
                                intersection_sample=aliases[:8],
                            )
                    if req_id is not None and scratch_base is not None:
                        bases = seen_scratch.setdefault(req_id, set())
                        if scratch_base in bases:
                            _mtp_dw_event(
                                "fail",
                                req=req_id,
                                frontier=current_position,
                                invariant="distinct_mtp_scratch_bases",
                                row=row,
                                scratch_base=scratch_base,
                            )
                        bases.add(scratch_base)
                    if (
                        committed is not None
                        and current_window is not None
                        and boundary != min(current_window, committed)
                    ):
                        _mtp_dw_event(
                            "fail",
                            req=req_id,
                            frontier=current_position,
                            invariant="remap_boundary",
                            current_window_start=current_window,
                            committed_end=committed,
                            remap_boundary=boundary,
                        )
                    if not sample_row:
                        continue
                    _mtp_dw_event(
                        "remap",
                        req=req_id,
                        frontier=current_position,
                        row=row,
                        req_index=int(req_index),
                        current_position=current_position,
                        window_start=(int(_diag_windows[req_index]) if req_index < len(_diag_windows) else None),
                        window_end=None,
                        committed_end=(int(_diag_committed[req_index]) if req_index < len(_diag_committed) else None),
                        remap_boundary=boundary,
                        scratch_base=scratch_base,
                        selected_absolute_count=int(selected_absolute.numel()),
                        selected_absolute_min=(int(selected_absolute.min()) if selected_absolute.numel() else None),
                        selected_absolute_max=(int(selected_absolute.max()) if selected_absolute.numel() else None),
                        selected_absolute_sample=selected_absolute[:8].tolist(),
                        selected_packed_count=int(selected_absolute.numel()),
                        selected_packed_min=(
                            int(packed_row[: selected_absolute.numel()].min()) if selected_absolute.numel() else None
                        ),
                        selected_packed_max=(
                            int(packed_row[: selected_absolute.numel()].max()) if selected_absolute.numel() else None
                        ),
                        selected_packed_sample=packed_row[: min(8, selected_absolute.numel())].tolist(),
                    )
                for payload in _diag_deep_payloads.values():
                    packed_values = payload["selection"]
                    target_slots = payload["slots"]
                    aliases = sorted(payload["aliases"])
                    _mtp_dw_event(
                        "deep",
                        event="connector_payload",
                        **payload["common"],
                        rows=payload["rows"],
                        row_count=len(payload["rows"]),
                        payload_count=payload["payload_count"],
                        lmcache_selected_count=payload["selected_count"],
                        selection_sample=packed_values[:8],
                        selection_checksum=diagnostic_int_checksum(packed_values),
                        target_physical_count=payload["target_count"],
                        target_slot_sample=target_slots[:8],
                        target_slot_checksum=diagnostic_int_checksum(target_slots),
                        checksum_scope="first32",
                        live_slot_intersection_count=len(aliases),
                        live_slot_intersection_sample=aliases[:8],
                    )
                if _diag_deep_emitted_req_ids is not None and _diag_deep_emitted_this_layer:
                    _diag_deep_emitted_req_ids.update(_diag_deep_emitted_this_layer)
            # Stage 3 = isolation diagnostic: remap + FA on (garbage) scratch but
            # NO LMCache call. Output is expected wrong; only crash/no-crash
            # matters (crash => our remap/FA, clean => LMCache transfer kernel).
            if self.dsa_shrink_latent != 3 and _sel_packed is not None:
                _selected_for_wait = _sel_packed
                _target_slot_mapping_for_wait = attn_metadata.decode_target_slot_mapping
                _request_ids_for_wait = attn_metadata.decode_request_ids_compact
                _wait_fn = wait_for_kv_layer_from_connector
                with _dsa_prof.section("lmc_retrieve"):
                    _wait_fn(
                        layer_name,
                        selected_tokens=_selected_for_wait,
                        target_slot_mapping=_target_slot_mapping_for_wait,
                        request_ids=_request_ids_for_wait,
                        selected_token_counts=_selected_token_counts,
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
