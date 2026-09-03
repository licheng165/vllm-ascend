from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group, is_v1_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    LayerwisePrefillCallbackMetadata,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

from vllm_ascend import envs
from vllm_ascend.utils import (
    AscendDeviceType,
    StagedSFARouteReason,
    get_ascend_config,
    get_ascend_device_type,
)

logger = init_logger(__name__)

_DSA_LMCACHE_TRACE = envs.VLLM_ASCEND_DSA_LMCACHE_TRACE


def _dsa_lmcache_log_layer(layer_name: str) -> bool:
    return _DSA_LMCACHE_TRACE and "layers.0." in layer_name


def _tensor_like_debug(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        flat = value.flatten()
        try:
            head = flat[:4].tolist() if flat.numel() else None
            tail = flat[-4:].tolist() if flat.numel() > 4 else head
        except Exception as exc:
            head = None
            tail = None
            value_error = f"{type(exc).__name__}: {exc}"
        else:
            value_error = None
        return {
            "type": "Tensor",
            "shape": list(value.shape),
            "numel": int(value.numel()),
            "device": str(value.device),
            "dtype": str(value.dtype),
            "head": head,
            "tail": tail,
            "value_error": value_error,
        }
    if isinstance(value, (list, tuple)):
        first = value[0] if value else None
        return {
            "type": type(value).__name__,
            "len": len(value),
            "first": _tensor_like_debug(first),
        }
    return {"type": type(value).__name__}


def ascend_chunked_prefill_workspace_size(vllm_config: VllmConfig) -> int:
    scheduler_config = vllm_config.scheduler_config
    cache_config = vllm_config.cache_config
    model_config = vllm_config.model_config

    chunked_prefill_workspace_size = min(
        # Make sure there is enough for 8 full length request or at least
        # 4 pages of cache per request
        max(8 * model_config.max_model_len, 4 * scheduler_config.max_num_seqs * cache_config.block_size),
        # For long-context models try not to over-allocate limiting
        # kv-cache space, limiting it to 128k tokens,
        # which would result in the workspace being:
        #   2*(576)*(128*1024) = 288mb
        # (assuming 576 MLA head dim, and fp16)
        # which would result in up-projected context being
        #   2*(192*128)*(128*1024) = 6gb
        # (assuming 192 QK head dim, 128 heads, and fp16)
        128 * 1024,
    )

    chunked_prefill_workspace_size = max(
        chunked_prefill_workspace_size,
        scheduler_config.max_num_seqs * cache_config.block_size,
    )

    return chunked_prefill_workspace_size


def using_paged_attention(runtime_shape: int, vllm_config: VllmConfig) -> bool:
    if vllm_config.speculative_config is not None:
        return False
    if get_ascend_device_type() == AscendDeviceType.A5:
        return False
    from vllm.config.compilation import CUDAGraphMode

    cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
    if cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
        return False

    return runtime_shape in get_ascend_config().pa_shape_list


@lru_cache(maxsize=1)
def enable_cp():
    prefill_config = get_current_vllm_config().parallel_config
    return prefill_config.prefill_context_parallel_size > 1 or prefill_config.decode_context_parallel_size > 1


@dataclass
class AscendPrefillContextParallelMetadata:
    """
    Metadata for Prefill Context Parallelism (PCP) in CommonAttentionMetadata.

    Contains index tensors and sequence lengths for PCP operations.
    """

    pcp_allgather_restore_idx: torch.Tensor = None

    num_actual_tokens_pcp_padded: int = 0

    num_computed_tokens_of_pcp_dcp: list[list[list[int]]] | None = None

    q_head_idx_tensor: torch.Tensor = None

    q_tail_idx_tensor: torch.Tensor = None

    kv_with_q_head_nomask_idx_tensor: torch.Tensor = None

    kv_with_q_head_mask_idx_tensor: torch.Tensor = None

    kv_with_q_tail_nomask_idx_tensor: torch.Tensor = None

    kv_with_q_tail_mask_idx_tensor: torch.Tensor = None

    attn_mask_seqlens: torch.Tensor = None

    head_attn_nomask_seqlens: torch.Tensor = None

    tail_attn_nomask_seqlens: torch.Tensor = None

    q_full_idx: torch.Tensor = None

    # original query_lens before pcp split
    query_lens_pcp_full_cpu: torch.Tensor = None

    # original max_query_len before pcp split
    max_query_len_pcp_full: int = 0

    # the following attributes are specifically used in hybrid-attn models.
    pcp_use_hybrid_attn: bool = False

    pcp_unpad_mask: torch.Tensor = None

    # to get the right order of query in prefill per rank
    pcp_fa_query_idx: torch.Tensor = None

    # restore the full sequence across all pcp ranks
    # when entering from linear-attention to attention
    pcp_enter_fa_restore_idx: torch.Tensor = None

    # scatter the full sequence across all pcp ranks
    # when exiting from attention to linear-attention
    pcp_exit_fa_scatter_idx: torch.Tensor = None

    # the number of tokens padded in linear-attn per rank
    pcp_padded_tokens_fla: int = 0


@dataclass
class AscendCommonAttentionMetadata(CommonAttentionMetadata):
    """
    Per-batch attention metadata, shared across layers and backends.
    AttentionMetadataBuilder instances use it to construct per-layer metadata.

    For many of the tensors we keep both NPU and CPU versions.
    """

    # CPU tensor of sequence lengths for host-side operations.
    # E.g., tensor([128, 256, 64]) for 3 requests with different seq lengths.
    seq_lens_cpu: torch.Tensor = None

    # CPU tensor of already computed tokens count per request.
    # E.g., tensor([100, 200, 50]) means req0 has 100 tokens already computed.
    num_computed_tokens_cpu: torch.Tensor = None

    # Number of decode tokens per request, used for speculative decoding.
    # E.g., 1 for normal decoding, >1 for speculative decoding.
    decode_token_per_req: int = 1

    # Actual query sequence lengths for each token in the batch (CPU list).
    # E.g., [1, 1, 1, 128] for 3 decode tokens and 1 prefill with 128 tokens.
    actual_seq_lengths_q: list[int] = field(default_factory=list)

    # NPU tensor of position indices for rotary embeddings computation.
    # E.g., tensor([0, 1, 2, ...]) indicating token positions in sequence.
    positions: torch.Tensor = None

    # Current attention state (e.g., ChunkedPrefill, DecodeOnly).
    attn_state: Any = None

    # Padding size for graph capture, -1 means not in graph mode.
    graph_pad_size: int = -1

    # Total number of tokens including padding, used for padding operations.
    num_input_tokens: int = 0

    # Metadata for Prefill Context Parallelism (PCP) operations.
    prefill_context_parallel_metadata: AscendPrefillContextParallelMetadata | None = None

    # DSA two-group mode: the indexer KV group's own block table / slot mapping
    # (the indexer cache has its own block ids when latent and indexer are
    # separate groups). None in single-group mode.
    indexer_block_table_tensor: torch.Tensor | None = None
    indexer_slot_mapping: torch.Tensor | None = None
    layerwise_prefill_callback_metadata: tuple[
        LayerwisePrefillCallbackMetadata, ...
    ] = ()
    # DSA shrink-latent: per-request prompt lengths (CPU, length num_reqs); the
    # SFA builder expands them per ROW (decode rows -> plen, prefill/padding
    # rows -> 0 = no remap).
    prompt_lens_cpu: Any = None
    request_ids: list[str] | None = None
    cold_compact_resumes: tuple[bool, ...] = ()
    # Stable request-owned rows for the experimental persistent sparse
    # scratch cache. Padding/graph-dummy rows contain -1.
    resident_state_indices: torch.Tensor | None = None
    resident_state_generations: torch.Tensor | None = None
    resident_state_indices_cpu: Any = None
    resident_state_generations_cpu: Any = None

    # TODO: Remove it when vLLM no longer uses this function.
    def unpadded(self, num_actual_tokens: int, num_actual_reqs: int) -> "AscendCommonAttentionMetadata":
        # This only use to eagle now. It will be use to enforce_eager in future.
        return AscendCommonAttentionMetadata(
            query_start_loc=self.query_start_loc[: num_actual_reqs + 1],
            query_start_loc_cpu=self.query_start_loc_cpu[: num_actual_reqs + 1],
            seq_lens=self.seq_lens[:num_actual_reqs],
            seq_lens_cpu=self.seq_lens_cpu[:num_actual_reqs],
            num_computed_tokens_cpu=self.num_computed_tokens_cpu[:num_actual_reqs],
            num_reqs=num_actual_reqs,
            num_actual_tokens=num_actual_tokens,
            max_query_len=self.max_query_len,
            decode_token_per_req=self.decode_token_per_req,
            # NOTE: keep all tokens for block_table_tensor and slot_mapping otherwise
            # there will be error about shape mismatch during reshape and cache.
            # This is really strange since vLLM slices them as well
            block_table_tensor=self.block_table_tensor,
            slot_mapping=self.slot_mapping,
            indexer_block_table_tensor=self.indexer_block_table_tensor,
            indexer_slot_mapping=self.indexer_slot_mapping,
            causal=self.causal,
            actual_seq_lengths_q=self.actual_seq_lengths_q[:num_actual_tokens],
            positions=self.positions,
            attn_state=self.attn_state,
            graph_pad_size=-1,  # It should be -1 when not run in fullgraph mode.
            num_input_tokens=self.num_input_tokens,
            prefill_context_parallel_metadata=self.prefill_context_parallel_metadata,
            layerwise_prefill_callback_metadata=(
                self.layerwise_prefill_callback_metadata
            ),
            max_seq_len=self.max_seq_len,
            request_ids=(self.request_ids[:num_actual_reqs] if self.request_ids is not None else None),
            cold_compact_resumes=self.cold_compact_resumes[
                :num_actual_reqs
            ],
            resident_state_indices=(
                self.resident_state_indices[:num_actual_reqs]
                if self.resident_state_indices is not None
                else None
            ),
            resident_state_generations=(
                self.resident_state_generations[:num_actual_reqs]
                if self.resident_state_generations is not None
                else None
            ),
            resident_state_indices_cpu=(
                self.resident_state_indices_cpu[:num_actual_reqs]
                if self.resident_state_indices_cpu is not None
                else None
            ),
            resident_state_generations_cpu=(
                self.resident_state_generations_cpu[:num_actual_reqs]
                if self.resident_state_generations_cpu is not None
                else None
            ),
        )


def filter_chunked_req_indices(
    seq_len: torch.Tensor,
    mask_for_non_zero_chunk: list[bool] | None,
) -> torch.Tensor:
    """
    filter the reqs which are doing real chunk_prefill.

    Args:
        seq_len: contains multi-req length: [req0_len, req1_len, ...]
        mask_for_non_zero_chunk: [True, False, True, False, ...]
    Returns:
        filtered_indices: the real chunked req's indices
    """
    assert mask_for_non_zero_chunk is not None and len(seq_len) == len(mask_for_non_zero_chunk)
    offsets = torch.cumsum(torch.cat([torch.tensor([0]), seq_len[:-1]]), dim=0)
    filtered_indices = torch.cat(
        [
            torch.arange(offsets[i], offsets[i] + seq_len[i])
            for i in range(len(mask_for_non_zero_chunk))
            if mask_for_non_zero_chunk[i]
        ]
    )
    return filtered_indices


def split_decodes_and_prefills(
    common_attn_metadata: AscendCommonAttentionMetadata,
    decode_threshold: int = 1,
) -> tuple[int, int, int, int]:
    """
    Assuming a reordered batch, finds the boundary between prefill and decode
    requests.
    While pcp > 1, query_lens is split across pcp ranks, so we pass in the
    original query_lens and max_query_len to distinguish prefills and decodes.

    Args:
        common_attn_metadata: AscendCommonAttentionMetadata object containing the
            batch metadata.
        decode_threshold: The maximum query length to be considered a decode.

    Returns:
        num_decodes: The number of decode requests.
        num_prefills: The number of prefill requests.
        num_decode_tokens: The number of tokens in the decode requests.
        num_prefill_tokens: The number of tokens in the prefill requests.
    """
    long_seq_metadata = common_attn_metadata.prefill_context_parallel_metadata
    query_lens_pcp_full = long_seq_metadata.query_lens_pcp_full_cpu if long_seq_metadata else None
    max_query_len_pcp_full = long_seq_metadata.max_query_len_pcp_full if long_seq_metadata else 0
    max_query_len = common_attn_metadata.max_query_len if max_query_len_pcp_full == 0 else max_query_len_pcp_full
    num_reqs = common_attn_metadata.num_reqs
    num_tokens = common_attn_metadata.num_actual_tokens
    query_start_loc = common_attn_metadata.query_start_loc_cpu

    if max_query_len <= decode_threshold:
        return num_reqs, 0, num_tokens, 0

    query_lens = (query_start_loc[1:] - query_start_loc[:-1]) if query_lens_pcp_full is None else query_lens_pcp_full
    is_prefill = query_lens > decode_threshold
    if not torch.any(is_prefill):
        return num_reqs, 0, num_tokens, 0

    first_prefill = is_prefill.int().argmax(dim=-1).item()
    num_decodes = first_prefill
    num_prefills = num_reqs - num_decodes
    num_decode_tokens = query_start_loc[first_prefill].item()
    num_prefill_tokens = num_tokens - num_decode_tokens
    return (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens)


def staged_sfa_connector_supports_sparse_load() -> bool:
    """Whether the active v1 connector satisfies the staged SFA contract."""
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return False
    try:
        connector = get_kv_transfer_group()
        return bool(
            getattr(connector, "supports_staged_sfa_sparse_load", False)
            and getattr(connector, "uses_layerwise_model_callbacks", False)
            and callable(getattr(connector, "wait_for_layer_load", None))
        )
    except Exception:
        return False


def _dsa_remap_frontier(load_spec: Any) -> int:
    """Derive the operator boundary without changing LMCache commit progress."""
    if load_spec is None or not getattr(load_spec, "can_load", False):
        return 0
    remap_value = getattr(load_spec, "dsa_remap_frontier", None)
    committed_value = (
        remap_value
        if remap_value is not None
        else getattr(load_spec, "dsa_committed_end", None)
    )
    committed_value = (
        committed_value
        if committed_value is not None
        else getattr(load_spec, "lmcache_cached_tokens", 0)
    )
    committed_end = int(committed_value or 0)
    scratch_capacity = getattr(load_spec, "dsa_scratch_capacity", None)
    if scratch_capacity is not None and committed_end < int(scratch_capacity):
        return 0
    return committed_end


def get_lmcache_sparse_cached_tokens(request_ids: Any) -> list[int]:
    """Return a proven remap frontier for every active request.

    Sparse-decode metadata contributes a boundary derived from committed
    LMCache progress and the fixed scratch capacity. A loadable dense-prefix
    request contributes zero because its first decoder step intentionally
    waits for the full prefix to become resident before attention.
    """
    if request_ids is None:
        raise RuntimeError("[SFA sparse remap] active request IDs are unavailable.")
    normalized_request_ids = [str(req_id) for req_id in list(request_ids)]
    if len(set(normalized_request_ids)) != len(normalized_request_ids):
        raise RuntimeError("[SFA sparse remap] frontier lookup requires unique native request IDs.")
    if not normalized_request_ids:
        return []
    if not staged_sfa_connector_supports_sparse_load():
        raise RuntimeError(
            "[SFA sparse remap] the active connector does not advertise the "
            "staged sparse selective-load/frontier contract."
        )

    get_metadata = getattr(get_kv_transfer_group(), "_get_connector_metadata", None)
    if not callable(get_metadata):
        raise RuntimeError("[SFA sparse remap] connector frontier metadata is unavailable.")
    try:
        metadata = get_metadata()
    except Exception as exc:
        raise RuntimeError("[SFA sparse remap] connector frontier metadata lookup failed.") from exc

    reason, frontiers, _ = staged_sfa_metadata_sparse_route(
        metadata,
        normalized_request_ids,
    )
    if reason not in (
        StagedSFARouteReason.ELIGIBLE,
        StagedSFARouteReason.DENSE_PREFIX_HIT,
        StagedSFARouteReason.MIXED_CONNECTOR_LOAD,
    ):
        raise RuntimeError(
            "[SFA sparse remap] connector frontier validation failed: "
            f"{reason.value}."
        )
    return list(frontiers)


def staged_sfa_metadata_sparse_route(
    metadata: Any,
    request_ids: Any,
) -> tuple[StagedSFARouteReason, tuple[int, ...], tuple[bool, ...]]:
    """Validate main request metadata and resolve ordered remap frontiers."""
    if metadata is None or request_ids is None:
        return StagedSFARouteReason.MISSING_CONNECTOR_METADATA, (), ()
    active_request_ids = [str(req_id) for req_id in request_ids]
    if not active_request_ids or len(set(active_request_ids)) != len(active_request_ids):
        return StagedSFARouteReason.INVALID_REQUEST_IDS, (), ()
    active_request_id_set = set(active_request_ids)
    main_by_req: dict[str, Any] = {}
    try:
        metadata_requests = tuple(getattr(metadata, "requests", ()))
    except (TypeError, ValueError):
        return StagedSFARouteReason.INVALID_FRONTIER, (), ()
    for request in metadata_requests:
        req_id = str(getattr(request, "req_id", ""))
        if req_id not in active_request_id_set:
            continue
        if getattr(request, "is_decode_window_save", False):
            continue
        if req_id in main_by_req:
            return StagedSFARouteReason.DUPLICATE_MAIN_METADATA, (), ()
        main_by_req[req_id] = request

    if set(main_by_req) != active_request_id_set:
        return StagedSFARouteReason.MISSING_CONNECTOR_METADATA, (), ()

    dense_request_ids: set[str] = set()
    sparse_request_ids: set[str] = set()
    frontiers: list[int] = []
    cold_resumes: list[bool] = []
    for req_id in active_request_ids:
        request = main_by_req[req_id]
        try:
            current_released = int(request.dsa_current_released_frontier)
            nonresident = int(request.dsa_nonresident_frontier)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return StagedSFARouteReason.INVALID_FRONTIER, (), ()
        if current_released < 0 or nonresident < 0 or current_released > nonresident:
            return StagedSFARouteReason.INVALID_FRONTIER, (), ()

        if not getattr(request, "is_sparse_decode", False):
            if current_released != 0 or nonresident != 0:
                return StagedSFARouteReason.DENSE_PREFIX_NOT_RESIDENT, (), ()
            dense_request_ids.add(req_id)
            frontiers.append(0)
            cold_resumes.append(False)
            continue

        sparse_request_ids.add(req_id)
        load_spec = getattr(request, "load_spec", None)
        if load_spec is None:
            return StagedSFARouteReason.SPARSE_LOAD_UNAVAILABLE, (), ()
        can_load = bool(getattr(load_spec, "can_load", False))
        try:
            cached = int(getattr(load_spec, "lmcache_cached_tokens", 0) or 0)
            committed_raw = getattr(load_spec, "dsa_committed_end", None)
            committed = int(
                committed_raw
                if committed_raw is not None
                else (cached if can_load else 0)
            )
            remap_frontier = _dsa_remap_frontier(load_spec)
        except (TypeError, ValueError, OverflowError):
            return StagedSFARouteReason.INVALID_FRONTIER, (), ()
        if (
            cached < 0
            or committed < 0
            or committed > cached
            or remap_frontier < 0
            or remap_frontier > committed
        ):
            return StagedSFARouteReason.INVALID_FRONTIER, (), ()
        if can_load:
            if nonresident > remap_frontier:
                return StagedSFARouteReason.INVALID_FRONTIER, (), ()
        elif current_released > 0 or nonresident > 0:
            return StagedSFARouteReason.SPARSE_LOAD_UNAVAILABLE, (), ()
        frontiers.append(remap_frontier if can_load else 0)
        cold_resumes.append(
            bool(getattr(load_spec, "dsa_cold_compact_resume", False))
        )

    cold_resume_tuple = tuple(cold_resumes) if any(cold_resumes) else ()
    if dense_request_ids and sparse_request_ids:
        return (
            StagedSFARouteReason.MIXED_CONNECTOR_LOAD,
            tuple(frontiers),
            cold_resume_tuple,
        )
    if sparse_request_ids:
        return (
            StagedSFARouteReason.ELIGIBLE,
            tuple(frontiers),
            cold_resume_tuple,
        )
    return StagedSFARouteReason.DENSE_PREFIX_HIT, tuple(frontiers), ()


def staged_sfa_metadata_sparse_load(
    metadata: Any,
    request_ids: Any,
) -> tuple[StagedSFARouteReason, tuple[int, ...]]:
    """Preserve the existing frontier-only helper contract."""
    reason, frontiers, _ = staged_sfa_metadata_sparse_route(
        metadata, request_ids
    )
    return reason, frontiers


def wait_for_kv_layer_from_connector(
    layer_name: str,
    selected_tokens=None,
    token_start_index=None,
    request_ids=None,
    target_slot_mapping=None,
    selected_token_counts=None,
    payload_event=None,
):
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return

    connector = get_kv_transfer_group()
    should_log = _dsa_lmcache_log_layer(layer_name)
    wait_kwargs = {}
    if target_slot_mapping is not None:
        wait_kwargs["target_slot_mapping"] = target_slot_mapping
    if selected_token_counts is not None:
        wait_kwargs["selected_token_counts"] = selected_token_counts
    if payload_event is not None:
        wait_kwargs["payload_event"] = payload_event
    if selected_tokens is not None and request_ids is not None and not should_log:
        try:
            connector.wait_for_layer_load(
                layer_name,
                selected_tokens,
                token_start_index,
                request_ids,
                **wait_kwargs,
            )
        except Exception:
            logger.exception(
                "[DSA_INDEX_LMCACHE] connector_wait_error layer=%s "
                "connector=%s selected=%s token_start_index=%s request_ids=%s "
                "target_slot_mapping=%s attn_metadata=%s",
                layer_name,
                type(connector).__name__,
                _tensor_like_debug(selected_tokens),
                _tensor_like_debug(token_start_index),
                request_ids,
                _tensor_like_debug(target_slot_mapping),
                None,
            )
            raise
        return

    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None:
        return

    if should_log:
        logger.info(
            "[DSA_INDEX_LMCACHE] connector_wait_enter layer=%s "
            "connector=%s selected=%s token_start_index=%s request_ids=%s "
            "target_slot_mapping=%s attn_metadata=%s",
            layer_name,
            type(connector).__name__,
            _tensor_like_debug(selected_tokens),
            _tensor_like_debug(token_start_index),
            request_ids,
            _tensor_like_debug(target_slot_mapping),
            type(attn_metadata).__name__,
        )

    # DSA selective load: pass the indexer's top-k positions so LMCache loads only the
    # selected prefill latent for this decode step. Falls back to a whole-layer load when
    # not provided (prefill / non-sparse).
    try:
        if selected_tokens is not None:
            # Per-row request identity, in the SAME order as `selected_tokens` (both are
            # sliced from input_batch-row-ordered data). LMCache uses this to pair each
            # decode request to its own selected-token row by req_id instead of by loop
            # position, so a divergence between the connector-metadata order and the
            # runner's batch order can no longer mis-pair (or IndexError) at higher batch.
            dsa_req_ids = getattr(forward_context, "dsa_req_ids", None)
            if request_ids is None and dsa_req_ids is not None:
                request_ids = list(dsa_req_ids[: selected_tokens.shape[0]])
            connector.wait_for_layer_load(
                layer_name,
                selected_tokens,
                token_start_index,
                request_ids,
                **wait_kwargs,
            )
        else:
            connector.wait_for_layer_load(layer_name)
    except Exception:
        logger.exception(
            "[DSA_INDEX_LMCACHE] connector_wait_error layer=%s "
            "connector=%s selected=%s token_start_index=%s request_ids=%s "
            "target_slot_mapping=%s attn_metadata=%s",
            layer_name,
            type(connector).__name__,
            _tensor_like_debug(selected_tokens),
            _tensor_like_debug(token_start_index),
            request_ids,
            _tensor_like_debug(target_slot_mapping),
            type(attn_metadata).__name__,
        )
        raise

    if should_log:
        logger.info(
            "[DSA_INDEX_LMCACHE] connector_wait_done layer=%s connector=%s",
            layer_name,
            type(connector).__name__,
        )


def maybe_save_kv_layer_to_connector(
    layer_name: str,
    kv_cache_layer: list[torch.Tensor],
):
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return

    connector = get_kv_transfer_group()

    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None:
        return
    # TODO: assert ascendMetadata
    should_log = _dsa_lmcache_log_layer(layer_name)
    if should_log:
        logger.info(
            "[DSA_INDEX_LMCACHE] connector_save_enter layer=%s connector=%s kv_cache_layer=%s attn_metadata=%s",
            layer_name,
            type(connector).__name__,
            _tensor_like_debug(kv_cache_layer),
            type(attn_metadata).__name__,
        )
    try:
        connector.save_kv_layer(layer_name, kv_cache_layer, attn_metadata)
    except Exception:
        logger.exception(
            "[DSA_INDEX_LMCACHE] connector_save_error layer=%s connector=%s kv_cache_layer=%s attn_metadata=%s",
            layer_name,
            type(connector).__name__,
            _tensor_like_debug(kv_cache_layer),
            type(attn_metadata).__name__,
        )
        raise
    if should_log:
        logger.info(
            "[DSA_INDEX_LMCACHE] connector_save_done layer=%s connector=%s",
            layer_name,
            type(connector).__name__,
        )


def _get_layerwise_prefill_p_node_connector():
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        raise RuntimeError(
            "Layerwise-prefill P-node callback requires a v1 KV connector."
        )
    connector = get_kv_transfer_group()
    if getattr(connector, "supports_layerwise_prefill_p_node", False) is not True:
        raise RuntimeError(
            "The active KV connector stopped supporting the complete "
            "layerwise-prefill P-node protocol."
        )
    return connector


def wait_for_layerwise_prefill_from_connector(
    callbacks: tuple[LayerwisePrefillCallbackMetadata, ...],
) -> None:
    """Synchronously wait for canonical rows before one eager SFA layer."""

    if not callbacks:
        return
    connector = _get_layerwise_prefill_p_node_connector()
    wait = getattr(connector, "wait_for_layerwise_prefill_load", None)
    if not callable(wait):
        raise RuntimeError(
            "Layerwise-prefill P-node connector has no synchronous wait callback."
        )
    for callback in callbacks:
        wait(callback)


def save_layerwise_prefill_to_connector(
    callback: LayerwisePrefillCallbackMetadata,
    kv_layer: Any,
    attn_metadata: Any,
) -> None:
    """Synchronously save one canonical row after one eager SFA layer."""

    connector = _get_layerwise_prefill_p_node_connector()
    save = getattr(connector, "save_layerwise_prefill_kv_layer", None)
    if not callable(save):
        raise RuntimeError(
            "Layerwise-prefill P-node connector has no synchronous save callback."
        )
    save(callback, kv_layer, attn_metadata)


def round_up(val: int, align: int) -> int:
    if align == 0:
        return 0
    return -(val // -align) * align


def trans_rope_weight(weight, rope_dim):
    if rope_dim == 0:
        return weight.contiguous()
    nope_part = weight[..., :-rope_dim, :]
    rope_part = weight[..., -rope_dim:, :]
    reordered_rope_part = torch.cat((rope_part[..., ::2, :], rope_part[..., 1::2, :]), dim=-2)
    return torch.cat((nope_part, reordered_rope_part), dim=-2).contiguous()


def transdata(nd_mat, block_size: tuple = (16, 16)):
    r = round_up(nd_mat.shape[0], block_size[0])
    c = round_up(nd_mat.shape[1], block_size[1])
    r_pad = r - nd_mat.shape[0]
    c_pad = c - nd_mat.shape[1]
    nd_mat = F.pad(nd_mat, (0, r_pad, 0, c_pad))
    nz_mat = torch.permute(
        torch.reshape(
            nd_mat,
            (r // block_size[0], block_size[0], c // block_size[1], block_size[1]),
        ),
        [2, 0, 1, 3],
    )
    nz_mat = torch.reshape(nz_mat, (nz_mat.shape[0], nz_mat.shape[1] * nz_mat.shape[2], nz_mat.shape[3]))
    return nz_mat


def enabling_mlapo(vllm_config: VllmConfig) -> bool:
    is_decode_instance = (
        vllm_config.kv_transfer_config is not None
        and vllm_config.kv_transfer_config.is_kv_consumer
        and not vllm_config.kv_transfer_config.is_kv_producer
    )
    return bool(envs.VLLM_ASCEND_ENABLE_MLAPO and is_decode_instance)
