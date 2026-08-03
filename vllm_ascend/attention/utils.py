from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group, is_v1_kv_transfer_group
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
    # DSA shrink-latent: per-request prompt lengths (CPU, length num_reqs); the
    # SFA builder expands them per ROW (decode rows -> plen, prefill/padding
    # rows -> 0 = no remap).
    prompt_lens_cpu: Any = None
    request_ids: list[str] | None = None
    # Authoritative Scheduler route indexed by request index:
    # request_index -> (route_state, remap_end).
    dsa_route_table: dict[int, tuple[str, int]] | None = None

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
            causal=self.causal,
            actual_seq_lengths_q=self.actual_seq_lengths_q[:num_actual_tokens],
            positions=self.positions,
            attn_state=self.attn_state,
            graph_pad_size=-1,  # It should be -1 when not run in fullgraph mode.
            num_input_tokens=self.num_input_tokens,
            prefill_context_parallel_metadata=self.prefill_context_parallel_metadata,
            max_seq_len=self.max_seq_len,
            request_ids=(self.request_ids[:num_actual_reqs] if self.request_ids is not None else None),
            dsa_route_table=(
                {
                    request_index: route
                    for request_index, route in self.dsa_route_table.items()
                    if request_index < num_actual_reqs
                }
                if self.dsa_route_table is not None
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


def get_lmcache_sparse_cached_tokens(request_ids: Any) -> list[int]:
    """Return a proven remap frontier for every active request.

    Sparse-decode metadata contributes its committed LMCache frontier. A
    loadable dense-prefix request contributes zero because its first decoder
    step intentionally waits for the full prefix to become resident before
    attention, so compact-scratch remapping must remain disabled for it.
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

    cached_by_req: dict[str, int] = {}
    for request in getattr(metadata, "requests", ()):
        is_sparse_decode = bool(getattr(request, "is_sparse_decode", False))
        load_spec = getattr(request, "load_spec", None)
        is_dense_prefix_load = bool(
            not is_sparse_decode
            and load_spec is not None
            and getattr(load_spec, "can_load", False)
        )
        if not is_sparse_decode and not is_dense_prefix_load:
            continue
        req_id = str(getattr(request, "req_id", ""))
        if not req_id:
            raise RuntimeError(
                "[SFA sparse remap] connector remap metadata has an empty request ID."
            )
        if req_id in cached_by_req:
            raise RuntimeError(
                "[SFA sparse remap] connector remap metadata contains a "
                f"duplicate request ID: {req_id!r}."
            )
        if (
            is_dense_prefix_load
            or load_spec is None
            or not getattr(load_spec, "can_load", False)
        ):
            cached_by_req[req_id] = 0
        else:
            route_state = getattr(load_spec, "dsa_route_state", None)
            remap_end = (
                getattr(load_spec, "dsa_remap_end", None)
                if route_state == "sparse"
                else None
            )
            cached_by_req[req_id] = int(
                remap_end
                if remap_end is not None
                else (
                    getattr(load_spec, "dsa_committed_end", None)
                    if getattr(load_spec, "dsa_committed_end", None) is not None
                    else getattr(load_spec, "lmcache_cached_tokens", 0)
                )
            )

    missing = [req_id for req_id in normalized_request_ids if req_id not in cached_by_req]
    if missing:
        raise RuntimeError(
            f"[SFA sparse remap] connector metadata has no proven sparse frontier for active requests: {missing!r}."
        )
    return [cached_by_req[req_id] for req_id in normalized_request_ids]


def staged_sfa_metadata_sparse_load(
    metadata: Any,
    request_ids: Any,
) -> tuple[StagedSFARouteReason, tuple[int, ...]]:
    """Classify active connector metadata and return ordered frontiers."""
    if metadata is None or request_ids is None:
        return StagedSFARouteReason.MISSING_CONNECTOR_METADATA, ()
    active_request_ids = [str(req_id) for req_id in request_ids]
    if not active_request_ids or len(set(active_request_ids)) != len(active_request_ids):
        return StagedSFARouteReason.INVALID_REQUEST_IDS, ()
    active_request_id_set = set(active_request_ids)
    sparse_frontiers: dict[str, int] = {}
    dense_request_ids: set[str] = set()
    matched_request_ids: set[str] = set()
    for request in getattr(metadata, "requests", ()):
        req_id = str(getattr(request, "req_id", ""))
        if req_id not in active_request_id_set:
            continue
        matched_request_ids.add(req_id)
        load_spec = getattr(request, "load_spec", None)
        if getattr(request, "is_sparse_decode", False):
            if req_id in sparse_frontiers:
                return StagedSFARouteReason.DUPLICATE_SPARSE_LOAD, ()
            route_state = getattr(load_spec, "dsa_route_state", None)
            remap_end = (
                getattr(load_spec, "dsa_remap_end", None)
                if route_state == "sparse"
                else None
            )
            sparse_frontiers[req_id] = int(
                remap_end
                if remap_end is not None
                else (
                    getattr(load_spec, "dsa_committed_end", None)
                    if getattr(load_spec, "dsa_committed_end", None)
                    is not None
                    else (
                        getattr(load_spec, "lmcache_cached_tokens", 0)
                        if getattr(load_spec, "can_load", False)
                        else 0
                    )
                )
                or 0
            )
            continue
        if getattr(load_spec, "can_load", False):
            dense_request_ids.add(req_id)

    if dense_request_ids.intersection(sparse_frontiers):
        return StagedSFARouteReason.DUPLICATE_SPARSE_LOAD, ()
    loadable_request_ids = dense_request_ids.union(sparse_frontiers)
    if loadable_request_ids == active_request_id_set:
        if dense_request_ids and sparse_frontiers:
            return StagedSFARouteReason.MIXED_CONNECTOR_LOAD, ()
        if sparse_frontiers:
            return StagedSFARouteReason.ELIGIBLE, tuple(sparse_frontiers[req_id] for req_id in active_request_ids)
        return StagedSFARouteReason.DENSE_PREFIX_HIT, ()
    if matched_request_ids != active_request_id_set:
        return StagedSFARouteReason.MISSING_CONNECTOR_METADATA, ()
    return StagedSFARouteReason.SPARSE_LOAD_UNAVAILABLE, ()


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
