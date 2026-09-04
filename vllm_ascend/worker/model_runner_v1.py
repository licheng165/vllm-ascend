#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/vllm/worker/gpu_model_runner.py
#

import json
import math
import os
import sys
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import copy, deepcopy
from dataclasses import dataclass
from multiprocessing import Manager
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from vllm.compilation.counter import compilation_counter
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import CompilationMode, CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size, tensor_model_parallel_all_gather
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
    is_v1_kv_transfer_group,
)
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    LayerwisePrefillCallbackMetadata,
)
from vllm.distributed.parallel_state import get_dcp_group, get_dp_group, get_pcp_group, get_pp_group, get_tp_group
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.model_loader import get_model
from vllm.sequence import IntermediateTensors
from vllm.utils.import_utils import LazyLoader
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.mem_utils import DeviceMemoryProfiler
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.attention.selector import get_attn_backend  # type: ignore
from vllm.v1.core.dsa_shared_pool import (
    LAYERWISE_PREFILL_BANK_COUNT,
    MAX_ALLOCATION_GENERATION,
    DSABlockAllocationMode,
)
from vllm.v1.core.kv_cache_utils import get_layerwise_prefill_max_tokens
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    DSAExecutionRow,
    DSAKVRow,
    DSAKVTopology,
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    ECConnectorOutput,
    KVConnectorOutput,
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput,
    SamplerOutput,
    make_empty_encoder_model_runner_output,
)
from vllm.v1.sample.logits_processor import build_logitsprocs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID, RejectionSampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker import mamba_utils
from vllm.v1.worker.cp_utils import (
    get_total_cp_world_size,
)
from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput, GPUModelRunner
from vllm.v1.worker.ubatch_utils import (
    UBatchSlices,
    maybe_create_ubatch_slices,
)
from vllm.v1.worker.utils import AttentionGroup

import vllm_ascend.envs as envs_ascend

# yapf: enable
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.mtp_dw_diag import (
    post_commit_sample_requests,
    scheduled_decode_requests,
)
from vllm_ascend.attention.target_sfa_diagnostics import (
    target_tail_boundary,
)
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    get_lmcache_sparse_cached_tokens,
    staged_sfa_connector_supports_sparse_load,
    staged_sfa_metadata_sparse_route,
    using_paged_attention,
)

# yapf conflicts with isort for this block
# yapf: disable
from vllm_ascend.compilation.acl_graph import (
    ACLGraphWrapper,
    reset_graph_params,
    set_draft_graph_params,
    set_graph_params,
    update_full_graph_params,
)
from vllm_ascend.distributed.kv_transfer.sparse_offload.resident_sparse_cache import (
    MAX_INT16_SCRATCH_CAPACITY,
    ResidentRequestStateRegistry,
)
from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
from vllm_ascend.eplb.core.eplb_device_transfer_loader import D2DExpertWeightLoader
from vllm_ascend.eplb.core.eplb_worker import EplbProcess
from vllm_ascend.eplb.eplb_updator import EplbUpdator
from vllm_ascend.eplb.utils import model_register
from vllm_ascend.ops.rotary_embedding import set_cos_and_sin, update_cos_sin
from vllm_ascend.patch.worker.patch_draft_quarot import patch_load_weights
from vllm_ascend.patch.worker.patch_module import patch_torch_npu_argsort
from vllm_ascend.quantization.utils import enable_fa_quant
from vllm_ascend.sample.sampler import AscendSampler
from vllm_ascend.spec_decode import get_spec_decode_method
from vllm_ascend.spec_decode.draft_proposer import AscendDraftModelProposer
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer
from vllm_ascend.spec_decode.medusa_proposer import AscendMedusaProposer
from vllm_ascend.spec_decode.ngram_proposer import AscendNgramProposer
from vllm_ascend.spec_decode.suffix_proposer import AscendSuffixDecodingProposer
from vllm_ascend.utils import (
    StagedSFARouteAction,
    StagedSFARouteDecision,
    StagedSFARouteReason,
    calc_split_factor,
    check_gdn_layer,
    enable_sp,
    enable_sp_by_pass,
    global_stream,
    is_drafter_moe_model,
    is_moe_model,
    lmhead_tp_enable,
    parse_layer_idx,
    set_weight_prefetch_method,
    sparse_kv_cache_has_indexer,
    staged_sfa_graph_capture_sizes,
    staged_sfa_graph_configuration_errors,
    staged_sfa_graph_configured,
)
from vllm_ascend.worker.block_table import MultiGroupBlockTable
from vllm_ascend.worker.dsa_shared_pool import (
    reshape_dsa_layerwise_prefill_raw,
    reshape_dsa_shared_pool_raw,
)
from vllm_ascend.worker.npu_input_batch import NPUInputBatch
from vllm_ascend.worker.pcp_utils import PCPManager

from vllm_ascend.ascend_forward_context import (  # isort: skip
    MoECommType,
    StagedSFAGraphKey,
    get_mc2_tokens_capacity,
    select_moe_comm_method,
    set_ascend_forward_context,
    set_mc2_mask,
    set_mc2_tokens_capacity,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import RoutedExpertsCapturer

if TYPE_CHECKING:
    import xgrammar as xgr  # type: ignore[import-untyped]
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")


def _staged_sfa_dummy_remap_boundaries(
    seq_lens: Any,
    query_width: int,
    index_topk: int,
) -> np.ndarray:
    """Build safe synthetic remap boundaries for staged graph capture."""
    boundaries = (
        np.asarray(seq_lens, dtype=np.int32).reshape(-1)
        - int(query_width)
    )
    scratch_capacity = int(query_width) * int(index_topk)
    boundaries[boundaries < scratch_capacity] = 0
    return boundaries


def _mtp_dw_diag_enabled() -> bool:
    return envs_ascend.VLLM_ASCEND_MTP_DW_DIAG


def _mtp_dw_window_size() -> int:
    try:
        return max(
            int(os.getenv("LMCACHE_DECODE_WINDOW_SAVE_WINDOW_SIZE", "0")), 0
        )
    except ValueError:
        return 0


def _mtp_dw_event(stage: str, **fields: Any) -> None:
    if not _mtp_dw_diag_enabled():
        return
    payload = {"schema": 1, "stage": stage, "owner": "vllm_ascend_runner"}
    payload.update(fields)
    logger.info("[MTP_DW] %s", json.dumps(payload, separators=(",", ":")))


def _mtp_dw_for_requests(
    owner: Any,
    scheduler_output: SchedulerOutput,
    stage: str,
    event: str,
    req_ids: set[str] | None = None,
    **fields: Any,
) -> None:
    if not _mtp_dw_diag_enabled():
        return
    for req_id in scheduler_output.num_scheduled_tokens:
        if req_ids is not None and req_id not in req_ids:
            continue
        frontier = getattr(owner, "_mtp_dw_diag_current_frontiers", {}).get(
            req_id
        )
        _mtp_dw_event(
            stage, req=req_id, event=event, frontier=frontier, **fields
        )


def _mtp_dw_sample_requests(
    owner: Any, scheduler_output: SchedulerOutput
) -> set[str]:
    if not _mtp_dw_diag_enabled():
        return set()
    counts = getattr(owner, "_mtp_dw_diag_step_counts", None)
    if counts is None:
        counts = {}
        owner._mtp_dw_diag_step_counts = counts
    window_size = max(_mtp_dw_window_size(), 1)
    sampled: set[str] = set()
    frontiers: dict[str, int] = {}
    for req_id, scheduled in scheduler_output.num_scheduled_tokens.items():
        req_index = owner.input_batch.req_id_to_index.get(req_id)
        if req_index is None:
            continue
        frontier = int(
            owner.input_batch.num_computed_tokens_cpu[req_index] + scheduled
        )
        frontiers[req_id] = frontier
        step = counts.get(req_id, 0)
        counts[req_id] = step + 1
        distance = min(frontier % window_size, (-frontier) % window_size)
        if step < 3 or distance <= 4:
            sampled.add(req_id)
    owner._mtp_dw_diag_current_frontiers = frontiers
    return sampled


# if true, allow tensor initialization and casting with internal format (e.g., NZ)
torch.npu.config.allow_internal_format = True

AttnMetadataDict: TypeAlias = dict[str, AttentionMetadata]
# list when ubatching is enabled
PerLayerAttnMetadata: TypeAlias = list[AttnMetadataDict] | AttnMetadataDict


SEQ_LEN_WITH_MAX_PA_WORKSPACE = 6144
_STAGED_SFA_ROUTE_ACTIONS = tuple(StagedSFARouteAction)


@dataclass
class GraphCaptureContext:
    stream: torch.npu.Stream


@dataclass(frozen=True)
class _LayerwisePrefillSlabSpec:
    logical_names: tuple[str, ...]
    latent_names: tuple[str, ...]
    indexer_names: tuple[str, ...]
    dtype: torch.dtype
    block_size: int
    num_kv_heads: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    index_head_dim: int
    latent_page_size_bytes: int
    indexer_page_size_bytes: int
    bundle_page_size_bytes: int
    slab_size_bytes: int


@contextmanager
def graph_capture(device: torch.device):
    """
    `graph_capture` is a context manager which should surround the code that
    is capturing the NPU graph. Its main purpose is to ensure that the
    some operations will be run after the graph is captured, before the graph
    is replayed. It returns a `GraphCaptureContext` object which contains the
    necessary data for the graph capture. Currently, it only contains the
    stream that the graph capture is running on. This stream is set to the
    current NPU stream when the context manager is entered and reset to the
    default stream when the context manager is exited. This is to ensure that
    the graph capture is running on a separate stream from the default stream,
    in order to explicitly distinguish the kernels to capture
    from other kernels possibly launched on background in the default stream.
    """
    graph_capture_context = GraphCaptureContext(torch.npu.Stream(device=device))
    stream = graph_capture_context.stream

    # we use nullcontext now
    maybe_ca_context = nullcontext()

    # ensure all initialization operations complete before attempting to
    # capture the graph on another stream
    curr_stream = torch.npu.current_stream()
    if curr_stream != stream:
        stream.wait_stream(curr_stream)

    with torch.npu.stream(stream), maybe_ca_context:
        yield graph_capture_context


def get_tp_context(drafter):
    return getattr(drafter, "tp_group_context", nullcontext())


class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: "SchedulerOutput"
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    spec_decode_common_attn_metadata: AscendCommonAttentionMetadata | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    attn_metadata: "PerLayerAttnMetadata"
    positions: torch.Tensor
    ec_connector_output: "ECConnectorOutput | None"
    cudagraph_stats: CUDAGraphStat | None
    batch_desc: BatchDescriptor
    staged_sfa_graph_key: StagedSFAGraphKey | None


def _fixed_decode_layout_arrays(
    max_num_reqs: int,
    query_width: int,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute request-major CPU metadata for fixed-width decode."""
    if max_num_reqs <= 0 or query_width not in (1, 2):
        raise ValueError(
            "fixed decode layout requires positive requests and "
            f"MTP=1 or MTP=2, got requests={max_num_reqs}, "
            f"MTP={query_width}"
        )
    request_indices = np.repeat(
        np.arange(max_num_reqs, dtype=dtype),
        query_width,
    )
    position_offsets = np.tile(
        np.arange(query_width, dtype=dtype),
        max_num_reqs,
    )
    cumulative_tokens = (
        np.arange(1, max_num_reqs + 1, dtype=dtype)
        * query_width
    )
    return request_indices, position_offsets, cumulative_tokens


def _fill_fixed_decode_positions(
    positions: np.ndarray,
    computed_tokens: np.ndarray,
    position_offsets: np.ndarray,
    num_reqs: int,
    query_width: int,
) -> None:
    """Fill request-major positions without per-step repeat/cumsum arrays."""
    if query_width not in (1, 2):
        raise ValueError(
            "fixed decode positions only support MTP=1 or MTP=2, "
            f"got MTP={query_width}"
        )
    num_tokens = num_reqs * query_width
    if (
        positions.size != num_tokens
        or computed_tokens.size < num_reqs
        or position_offsets.size < num_tokens
    ):
        raise ValueError(
            "fixed decode position buffers do not match the layout: "
            f"positions={positions.size}, computed={computed_tokens.size}, "
            f"offsets={position_offsets.size}, requests={num_reqs}, "
            f"MTP={query_width}"
        )
    positions.reshape(num_reqs, query_width)[:] = (
        computed_tokens[:num_reqs, None]
    )
    positions += position_offsets[:num_tokens]


class NPUModelRunner(GPUModelRunner):
    supports_layerwise_prefill_p_node = True

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # TODO(qcs): These manual pad and unpad for GPUModelRunner are
        # used to expand some buffers, which need to be reverted after
        # the following PR is merged:
        # https://github.com/vllm-project/vllm/pull/28988
        max_pcp_pad_tokens = (
            vllm_config.parallel_config.prefill_context_parallel_size * 2 * vllm_config.scheduler_config.max_num_seqs
        )
        vllm_config.scheduler_config.max_num_batched_tokens += max_pcp_pad_tokens
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)

        # NOTE: For FULL mode we change +1 to +2 to reserve extra space for padding.
        # See _pad_query_start_loc_for_fia.
        self.query_start_loc = self._make_buffer(
            self.max_num_reqs + 2,  # type: ignore[has-type]
            dtype=torch.int32,
        )

        # Now, query_start_loc is padded.
        # But gdn needs an unpadded one.
        # gdn_query_start_loc is an unpadded version of query_start_loc.
        # TODO delete it if fia's check is removed.
        self._has_gdn = check_gdn_layer(vllm_config)
        if self._has_gdn:
            self.gdn_query_start_loc = self._make_buffer(
                self.max_num_reqs + 1,  # type: ignore[has-type]
                dtype=torch.int32,
            )

        vllm_config.scheduler_config.max_num_batched_tokens -= max_pcp_pad_tokens
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank

        self.sampler = AscendSampler()
        self.attn_state: AscendAttentionState | None = None
        self._staged_sfa_impls: tuple[tuple[str, Any], ...] = ()
        self._staged_sfa_graph_capture_sizes = staged_sfa_graph_capture_sizes(
            vllm_config
        )
        # Ephemeral output of the existing batch/DP coordination pass.
        self._staged_sfa_dp_route_action: StagedSFARouteAction | None = None
        self._dp_batch_sync_buffers: dict[int, torch.Tensor] = {}
        self._staged_sfa_startup_capture_attempted = False
        self._profiling_cudagraph_memory = False

        # Ascend-specific configurations
        self.ascend_config = get_ascend_config()
        set_weight_prefetch_method(self.ascend_config.weight_prefetch_config)
        # Dump / PrecisionDebugger configuration now comes from AscendConfig
        dump_cfg = self.ascend_config.dump_config_path
        self.debugger = None
        if dump_cfg is not None:
            if self.model_config.enforce_eager:
                from msprobe.pytorch import PrecisionDebugger

                self.debugger = PrecisionDebugger(dump_cfg)
            else:
                raise RuntimeError("Dumping/debugging only works in eager mode.")
        # use_hybrid_blocks: if hybrid blocks is used.
        self.use_hybrid_blocks: bool = False
        self.need_accepted_tokens: bool = False

        self.is_multimodal_model = self.model_config.is_multimodal_model
        self.block_size = vllm_config.cache_config.block_size
        # Set up Attention
        self.use_sparse = hasattr(vllm_config.model_config, "hf_text_config") and hasattr(
            vllm_config.model_config.hf_text_config, "index_topk"
        )
        self.dsa_index_topk = 0
        if self.use_sparse:
            self.dsa_index_topk = int(self.model_config.hf_text_config.index_topk)
            self.sparse_head_dim = (
                self.model_config.hf_text_config.kv_lora_rank,
                self.model_config.hf_text_config.qk_rope_head_dim,
                self.model_config.hf_text_config.index_head_dim,
            )
        # DSA latent offload Route-1 pragmatic (M-B): when enabled, the SFA paged cache
        # holds only the indexer key (latent goes to the self-managed PagedLatentPool),
        # so the per-token page shrinks ~5.5x and GPU KV cache size grows.
        import vllm_ascend.envs as envs_ascend

        self.dsa_free_paged = bool(
            self.use_sparse
            and envs_ascend.VLLM_ASCEND_ENABLE_DSA_LATENT_OFFLOAD
            and envs_ascend.VLLM_ASCEND_DSA_OFFLOAD_FREE_PAGED
        )
        # Proper route P1: split the SFA KV into latent + indexer KV cache groups.
        self.dsa_unbundle = bool(self.use_sparse and envs_ascend.VLLM_ASCEND_DSA_UNBUNDLE)
        if self.dsa_unbundle:
            logger.info("DSA un-bundle enabled: latent and indexer use separate KV cache groups.")
        # Step A: latent and indexer become two REAL KV cache groups with separate
        # block tables and per-group block pools (the vLLM side is gated by the
        # same env var). Requires UNBUNDLE. Prerequisite for freeing latent blocks
        # at end of prefill while the indexer stays resident.
        self.dsa_two_groups = bool(self.dsa_unbundle and envs_ascend.VLLM_ASCEND_DSA_TWO_GROUPS)
        self.dsa_kv_topology: DSAKVTopology | None = None
        self.dsa_kv_rows_by_layer_name: dict[str, DSAKVRow] = {}
        self.dsa_kv_executions_by_ordinal: dict[int, DSAExecutionRow] = {}
        self._dsa_kv_specs_by_layer_name: dict[str, KVCacheSpec] = {}
        if self.dsa_two_groups:
            logger.info("DSA two-group mode enabled: separate block tables/pools for latent and indexer.")
        elif envs_ascend.VLLM_ASCEND_DSA_TWO_GROUPS:
            logger.warning("VLLM_ASCEND_DSA_TWO_GROUPS requires VLLM_ASCEND_DSA_UNBUNDLE=1; ignoring.")
        self.dsa_shared_pool = bool(
            self.dsa_two_groups and envs_ascend.VLLM_ASCEND_DSA_SHARED_POOL
        )
        if self.dsa_shared_pool:
            logger.info("DSA shared bundle pool enabled for latent/indexer KV cache.")
        elif envs_ascend.VLLM_ASCEND_DSA_SHARED_POOL:
            logger.warning(
                "VLLM_ASCEND_DSA_SHARED_POOL requires DSA_UNBUNDLE=1 and "
                "DSA_TWO_GROUPS=1; ignoring."
            )
        # Step B staging (1 = B2 compact-scratch decode read; 2 = +B1 freeing).
        self.dsa_shrink_latent = (
            int(envs_ascend.VLLM_ASCEND_DSA_SHRINK_LATENT) if self.dsa_two_groups else 0
        )
        if self.dsa_shrink_latent:
            logger.info("DSA shrink-latent stage %d enabled (B2 compact-scratch decode).", self.dsa_shrink_latent)
        # dsa c8
        self.use_sparse_c8_indexer = self.ascend_config.enable_sparse_c8
        if self.use_sparse_c8_indexer:
            self.c8_k_cache_dtype = torch.int8
            self.c8_k_scale_cache_dtype = torch.float16

        self.attn_backend = get_attn_backend(
            0,
            self.dtype,
            None,
            use_mla=self.model_config.use_mla,
            use_sparse=self.use_sparse,
            use_mm_prefix=self.model_config is not None and self.model_config.is_mm_prefix_lm,
        )

        try:
            self.dcp_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
            self.pcp_size = get_pcp_group().world_size
            self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        except Exception:
            self.dcp_size = 1
            self.dcp_rank = 0
            self.pcp_size = 1
            self.pcp_rank = 0
        if self.pcp_size > 1:
            self.model_config.max_model_len += 2 * self.pcp_size * self.max_num_reqs
        max_buffer_num_tokens = self.max_num_tokens
        if self.pcp_size * self.dcp_size > 1:
            max_buffer_num_tokens = self.max_num_tokens + self.max_num_reqs * 2 * self.pcp_size
            self.pcp_manager = PCPManager(
                self.pcp_size,
                self.pcp_rank,
                self.dcp_size,
                self.dcp_rank,
                max_buffer_num_tokens,
                self.max_num_reqs,
                self.device,
                self.vllm_config,
                self.use_async_scheduling,
                self.pin_memory,
                self.use_sparse,
            )
            # TODO(zhenwenqi) after https://github.com/vllm-project/vllm/pull/28988 is merged, we can delete this
            self.input_ids = self._make_buffer(max_buffer_num_tokens, dtype=torch.int32)
            self.positions = self._make_buffer(max_buffer_num_tokens, dtype=torch.int64)

        self._set_up_drafter()

        # kv role
        self.is_kv_producer = False
        self.is_kv_consumer = False
        if vllm_config.kv_transfer_config is not None:
            self.is_kv_producer = vllm_config.kv_transfer_config.is_kv_producer
            self.is_kv_consumer = vllm_config.kv_transfer_config.is_kv_consumer

        set_cos_and_sin(vllm_config, self.max_num_reqs, self.uniform_decode_query_len, self.dtype, self.device)
        set_mc2_tokens_capacity(vllm_config, self.max_num_reqs, self.uniform_decode_query_len)
        set_mc2_mask(vllm_config, self.device)
        self.decode_threshold = 1 + (self.speculative_config.num_speculative_tokens if self.speculative_config else 0)
        if self.decode_threshold in (1, 2):
            (
                self._fixed_decode_req_indices,
                self._fixed_decode_position_offsets,
                self._fixed_decode_cu_num_tokens,
            ) = _fixed_decode_layout_arrays(
                self.max_num_reqs,
                self.decode_threshold,
                self.arange_np.dtype,
            )
        else:
            self._fixed_decode_req_indices = None
            self._fixed_decode_position_offsets = None
            self._fixed_decode_cu_num_tokens = None

        self.use_aclgraph = self._use_aclgraph()

        eplb_config = self.ascend_config.eplb_config
        self.dynamic_eplb = eplb_config.dynamic_eplb
        self.eplb_enable = self.dynamic_eplb or (eplb_config.expert_map_path is not None)
        if self.dynamic_eplb:
            self.is_eplb_warmuped = False
            self.policy_type = eplb_config.eplb_policy_type
            self.eplb_loader = D2DExpertWeightLoader()
            self.manager = Manager()
            self.shared_dict = self.manager.dict({"expert_map": None, "moe_load": None, "expert_maps": None})
            self.eplb_process = EplbProcess(shared_dict=self.shared_dict, policy_type=self.policy_type, enable_d2d=True)
            self.process = self.eplb_process._launch_process()
            self.eplb_updator = EplbUpdator(eplb_config, self.eplb_loader, self.eplb_process, self.process)
        # Input Batch
        # NOTE(Chen): Ideally, we should initialize the input batch inside
        # `initialize_kv_cache` based on the kv cache config. However, as in
        # https://github.com/vllm-project/vllm/pull/18298, due to some unknown
        # reasons, we have to initialize the input batch before `load_model`,
        # quantization + weight offloading will fail otherwise. As a temporary
        # solution, we initialize the input batch here, and re-initialize it
        # in `initialize_kv_cache` if the block_sizes here is different from
        # the block_sizes in the kv cache config.
        self.input_batch = NPUInputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=max(self.model_config.max_model_len, self.max_encoder_len),
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.block_size],
            kernel_block_sizes=[[self.cache_config.block_size]],
            is_spec_decode=bool(self.vllm_config.speculative_config),
            logitsprocs=build_logitsprocs(
                self.vllm_config,
                self.device,
                self.pin_memory,
                self.is_pooling_model,
                self.vllm_config.model_config.logits_processors,
            ),
            is_pooling_model=self.is_pooling_model,
            num_speculative_tokens=(
                self.vllm_config.speculative_config.num_speculative_tokens if self.vllm_config.speculative_config else 0
            ),
            cp_kv_cache_interleave_size=self.parallel_config.cp_kv_cache_interleave_size,
            layerwise_prefill_p_node=self.layerwise_prefill_p_node,
        )
        self.num_draft_tokens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
        # here we use int32
        self.sampled_token_ids_pinned_cpu = torch.empty(
            (self.max_num_reqs, 1),
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        # for cleancode , actually the three attrs is defined in gpu_model_runner
        self.execute_model_state: ExecuteModelState | None = None
        # None in the first PP rank. The rest are set after load_model.
        self.intermediate_tensors: IntermediateTensors | None = None
        self.reorder_batch_threshold: int | None = None
        self.long_seq_metadata = None
        self.query_lens: torch.Tensor | None = None
        self.cpu_slot_mapping = None
        self.sampling_done_event: torch.npu.Event | None = None

        # self.cudagraph_batch_sizes sorts in ascending order.
        if (
            self.compilation_config.cudagraph_capture_sizes
            and self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            self.cudagraph_batch_sizes = sorted(self.compilation_config.cudagraph_capture_sizes)
        else:
            self.cudagraph_batch_sizes = []
        self.mamba_state_idx: dict[str, int] = {}
        self._mamba_copy_bufs: mamba_utils.MambaCopyBuffers | None = None
        # The disabled path still uses compact scratch, but retrieves the
        # complete split-boundary union.
        self.dsa_resident_cache = bool(
            self.dsa_shrink_latent
            and envs_ascend.VLLM_ASCEND_DSA_RESIDENT_CACHE
        )
        self._resident_state_registry: ResidentRequestStateRegistry | None = None
        self._resident_state_indices = None
        self._resident_state_generations = None
        self._resident_scratch_capacity = (
            self.decode_threshold * self.dsa_index_topk
        )
        if self.dsa_resident_cache:
            if not (
                0
                < self._resident_scratch_capacity
                < MAX_INT16_SCRATCH_CAPACITY
            ):
                raise ValueError(
                    "resident sparse cache stores scratch slots in signed "
                    "int16 and requires 0 < MTP * index_topk < "
                    f"{MAX_INT16_SCRATCH_CAPACITY}; got "
                    f"{self._resident_scratch_capacity}"
                )
            self._resident_state_registry = ResidentRequestStateRegistry(
                self.max_num_reqs
            )
            self._resident_state_indices = self._make_buffer(
                self.max_num_reqs, dtype=torch.int32
            )
            self._resident_state_generations = self._make_buffer(
                self.max_num_reqs, dtype=torch.int64
            )
            logger.info(
                "DSA sorted resident scratch reuse enabled: capacity=%d, "
                "state_slots=%d.",
                self._resident_scratch_capacity,
                self.max_num_reqs,
            )

    @property
    def use_cp(self) -> bool:
        return self.pcp_size * self.dcp_size > 1

    def _init_device_properties(self) -> None:
        self.num_sms = None

    def _sync_device(self) -> None:
        torch.npu.synchronize()

    @staticmethod
    def _validate_layerwise_prefill_bank_metadata(
        request_id: str,
        block_ids: tuple[list[int], ...] | None,
        block_ids_by_bank: tuple[tuple[list[int], ...], ...] | None,
        allocation_mode: DSABlockAllocationMode | None,
        allocation_generation: int | None,
    ) -> None:
        if allocation_mode != DSABlockAllocationMode.PREFILL_CHILD:
            raise RuntimeError(
                "Layerwise-prefill P-node request has the wrong allocation "
                f"mode: request_id={request_id}, mode={allocation_mode}."
            )
        if (
            not isinstance(allocation_generation, int)
            or isinstance(allocation_generation, bool)
            or not 0 < allocation_generation <= MAX_ALLOCATION_GENERATION
        ):
            raise RuntimeError(
                "Layerwise-prefill P-node request is missing a valid allocation "
                f"generation: request_id={request_id}."
            )
        if block_ids is None or len(block_ids) != 2:
            raise RuntimeError(
                "Layerwise-prefill P-node request must carry two KV groups: "
                f"request_id={request_id}."
            )
        if (
            block_ids_by_bank is None
            or len(block_ids_by_bank) != LAYERWISE_PREFILL_BANK_COUNT
            or any(len(bank) != len(block_ids) for bank in block_ids_by_bank)
        ):
            raise RuntimeError(
                "Layerwise-prefill P-node request is missing its two-bank KV "
                f"metadata: request_id={request_id}."
            )
        if block_ids_by_bank[0] != block_ids:
            raise RuntimeError(
                "Layerwise-prefill P-node primary blocks differ from bank 0: "
                f"request_id={request_id}."
            )
        if any(
            len(bank_group) != len(primary_group)
            for bank in block_ids_by_bank
            for bank_group, primary_group in zip(bank, block_ids)
        ):
            raise RuntimeError(
                "Layerwise-prefill P-node bank group lengths differ from the "
                f"primary allocation: request_id={request_id}."
            )

    def _layerwise_prefill_table_addresses(
        self,
    ) -> tuple[tuple[tuple[int, int], ...], ...]:
        tables = getattr(
            self.input_batch,
            "layerwise_prefill_block_tables",
            None,
        )
        if tables is None:
            raise RuntimeError(
                "Layerwise-prefill PIECEWISE requires the fixed per-bank "
                "block tables."
            )
        addresses = []
        for bank_table in tables:
            bank_addresses = []
            for block_table in bank_table.block_tables:
                slot = block_table.slot_mapping.gpu
                device_tensor = block_table.get_device_tensor()
                bank_addresses.append(
                    (
                        slot.data_ptr(),
                        (
                            device_tensor.data_ptr()
                            if device_tensor is not None
                            else 0
                        ),
                    )
                )
            addresses.append(tuple(bank_addresses))
        return tuple(addresses)

    def _validate_layerwise_prefill_piecewise_addresses(self) -> None:
        """Enforce the PIECEWISE fixed-address guarantee.

        Graph capture bakes the per-bank block-table and slot-mapping
        addresses into the captured kernels; any later rebind would make
        replay dereference stale storage. The fixed buffers are recorded
        once and every refresh must keep their data pointers unchanged.
        """
        addresses = self._layerwise_prefill_table_addresses()
        recorded = getattr(
            self,
            "_layerwise_prefill_recorded_addresses",
            None,
        )
        if recorded is None:
            self._layerwise_prefill_recorded_addresses = addresses
            return
        if addresses != recorded:
            raise RuntimeError(
                "Layerwise-prefill PIECEWISE fixed-address buffers were "
                "rebound after capture; graph replay would dereference "
                "stale addresses."
            )

    def _validate_layerwise_prefill_runtime(
        self,
        topology: DSAKVTopology | None = None,
    ) -> Any:
        if not getattr(self, "layerwise_prefill_p_node", False):
            return None
        self._ensure_layerwise_prefill_p_node_supported()
        if self.compilation_config.cudagraph_mode in (
            CUDAGraphMode.FULL,
            CUDAGraphMode.FULL_DECODE_ONLY,
        ):
            raise RuntimeError(
                "Layerwise-prefill P-node execution rejects FULL and "
                "FULL_DECODE_ONLY graph modes: full-model replay bypasses "
                "the per-layer transfer-window callbacks."
            )
        if self.compilation_config.cudagraph_mode is CUDAGraphMode.PIECEWISE:
            self._validate_layerwise_prefill_piecewise_addresses()
        topology = topology or getattr(self, "dsa_kv_topology", None)
        if (
            topology is None
            or len(topology.rows_by_group) != 2
            or not topology.rows_by_group[0]
            or not topology.rows_by_group[1]
        ):
            raise RuntimeError(
                "Layerwise-prefill P-node execution requires the canonical "
                "two-group DSA KV topology."
            )
        if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
            raise RuntimeError(
                "Layerwise-prefill P-node execution requires a v1 KV connector."
            )
        transfer_config = self.vllm_config.kv_transfer_config
        if transfer_config is None or transfer_config.is_kv_producer is not True:
            raise RuntimeError(
                "Layerwise-prefill P-node execution requires a KV producer role."
            )
        connector = get_kv_transfer_group()
        if getattr(connector, "supports_layerwise_prefill_p_node", False) is not True:
            raise RuntimeError(
                "The active KV connector does not support the complete "
                "layerwise-prefill P-node protocol: connector="
                f"{type(connector).__name__}, eager_callbacks="
                f"{getattr(connector, 'supports_layerwise_prefill_eager_callbacks', False)}, "
                "indexer_persistence="
                f"{getattr(connector, 'supports_dsa_index_lmcache', False)}, "
                "transfer_window="
                f"{getattr(connector, 'supports_layerwise_prefill_transfer_window', False)}. "
                "Check that the deployed LMCache and LMCache-Ascend builds "
                "provide the layerwise-prefill P-node capabilities."
            )
        if not callable(
            getattr(connector, "wait_for_layerwise_prefill_load", None)
        ) or not callable(
            getattr(connector, "save_layerwise_prefill_kv_layer", None)
        ):
            raise RuntimeError(
                "The active KV connector advertises layerwise-prefill P-node "
                "support without synchronous eager callbacks."
            )
        return connector

    def _validate_layerwise_prefill_scheduler_output(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        if not getattr(self, "layerwise_prefill_p_node", False):
            return

        for request in scheduler_output.scheduled_new_reqs:
            self._validate_layerwise_prefill_bank_metadata(
                request.req_id,
                request.block_ids,
                request.block_ids_by_bank,
                request.block_allocation_mode,
                request.allocation_generation,
            )

        cached = scheduler_output.scheduled_cached_reqs
        if not cached.req_ids:
            return
        fields = (
            cached.new_block_ids,
            cached.new_block_ids_by_bank,
            cached.new_block_allocation_modes,
            cached.allocation_generations,
        )
        if any(field is None or len(field) != len(cached.req_ids) for field in fields):
            raise RuntimeError(
                "Layerwise-prefill cached requests are missing bank, allocation "
                "mode, or generation metadata."
            )
        assert cached.new_block_ids_by_bank is not None
        assert cached.new_block_allocation_modes is not None
        assert cached.allocation_generations is not None
        for index, request_id in enumerate(cached.req_ids):
            request = self.requests.get(request_id)
            if request is None:
                raise RuntimeError(
                    "Layerwise-prefill cached request has no worker state: "
                    f"request_id={request_id}."
                )
            block_ids_by_bank = cached.new_block_ids_by_bank[index]
            allocation_mode = cached.new_block_allocation_modes[index]
            allocation_generation = cached.allocation_generations[index]
            block_ids = cached.new_block_ids[index]
            resumed = request_id in cached.resumed_req_ids
            if resumed:
                if block_ids is None or block_ids_by_bank is None:
                    raise RuntimeError(
                        "Resumed layerwise-prefill request is missing replacement "
                        f"bank metadata: request_id={request_id}."
                    )
                self._validate_layerwise_prefill_bank_metadata(
                    request_id,
                    block_ids,
                    block_ids_by_bank,
                    allocation_mode,
                    allocation_generation,
                )
                if allocation_generation == request.allocation_generation:
                    raise RuntimeError(
                        "Resumed layerwise-prefill request reused its allocation "
                        f"generation: request_id={request_id}."
                    )
                continue

            self._validate_layerwise_prefill_bank_metadata(
                request_id,
                request.block_ids,
                request.block_ids_by_bank,
                request.block_allocation_mode,
                request.allocation_generation,
            )
            if allocation_generation != request.allocation_generation:
                raise RuntimeError(
                    "Layerwise-prefill cached request changed or omitted its "
                    f"allocation generation: request_id={request_id}."
                )
            has_delta = any(
                value is not None
                for value in (block_ids, block_ids_by_bank, allocation_mode)
            )
            if has_delta:
                self._validate_layerwise_prefill_bank_metadata(
                    request_id,
                    block_ids,
                    block_ids_by_bank,
                    allocation_mode,
                    allocation_generation,
                )

    def _update_states(self, scheduler_output: SchedulerOutput) -> None:
        self._validate_layerwise_prefill_runtime()
        self._validate_layerwise_prefill_scheduler_output(scheduler_output)
        registry = self._resident_state_registry
        if registry is not None:
            registry.release(tuple(scheduler_output.finished_req_ids))
        super()._update_states(scheduler_output)

    def _prepare_resident_request_state(
        self,
        *,
        num_reqs: int,
        num_reqs_padded: int,
        is_dummy: bool,
        resident_compatible: bool = True,
        remap_frontiers: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, Any, Any]:
        registry = self._resident_state_registry
        indices = self._resident_state_indices
        generations = self._resident_state_generations
        if registry is None or indices is None or generations is None:
            return None, None, None, None

        indices.np[:num_reqs_padded].fill(-1)
        generations.np[:num_reqs_padded].fill(-1)
        if not is_dummy and num_reqs:
            request_ids = tuple(self.input_batch.req_ids[:num_reqs])
            if any(request_id is None for request_id in request_ids):
                raise RuntimeError(
                    "resident sparse cache received an empty request id"
                )
            if remap_frontiers is not None and len(remap_frontiers) != num_reqs:
                raise RuntimeError(
                    "resident sparse cache remap frontier count differs from "
                    f"the active request count: {len(remap_frontiers)} != "
                    f"{num_reqs}"
                )
            if not resident_compatible:
                # Generic decode/prefill unions may overwrite scratch without
                # updating sorted state. The next resident step must be cold.
                registry.invalidate(request_ids)  # type: ignore[arg-type]
            block_table = self.input_batch.block_table[0]
            scratch_blocks = cdiv(
                self._resident_scratch_capacity,
                block_table.block_size,
            )
            eligible_rows = []
            eligible_request_ids = []
            inactive_request_ids = []
            signatures = []
            for row in range(num_reqs):
                remap_frontier = (
                    remap_frontiers[row]
                    if remap_frontiers is not None
                    else None
                )
                if remap_frontier == 0:
                    # A zero boundary is the no-remap path even if the block
                    # allocator has already materialized the full prefix.
                    inactive_request_ids.append(request_ids[row])
                    continue
                if block_table.num_blocks_per_row[row] < scratch_blocks:
                    if remap_frontier is not None:
                        raise RuntimeError(
                            "resident sparse cache received a nonzero remap "
                            "frontier without a complete scratch prefix for "
                            f"request row {row}: frontier="
                            f"{remap_frontier}, needs {scratch_blocks} "
                            "blocks, has "
                            f"{block_table.num_blocks_per_row[row]}"
                        )
                    inactive_request_ids.append(request_ids[row])
                    continue
                eligible_rows.append(row)
                eligible_request_ids.append(request_ids[row])
                signatures.append(
                    tuple(
                        map(
                            int,
                            block_table.block_table.np[
                                row, :scratch_blocks
                            ],
                        )
                    )
                )
            if resident_compatible and inactive_request_ids:
                # If a preempted/restarted request used to own a resident row,
                # make a later zero-to-nonzero boundary transition cold even
                # when the allocator reuses the same physical block ids.
                registry.invalidate(inactive_request_ids)  # type: ignore[arg-type]
            if resident_compatible and eligible_rows:
                state_rows, state_generations = registry.bind(
                    eligible_request_ids,  # type: ignore[arg-type]
                    signatures,
                )
                indices.np[eligible_rows] = state_rows
                generations.np[eligible_rows] = state_generations

        indices.copy_to_gpu(num_reqs_padded)
        generations.copy_to_gpu(num_reqs_padded)
        return (
            indices.gpu[:num_reqs_padded],
            generations.gpu[:num_reqs_padded],
            indices.np[:num_reqs_padded],
            generations.np[:num_reqs_padded],
        )

    def _set_up_drafter(self):
        # Set up speculative decoding.
        self.drafter: (
            AscendNgramProposer
            | AscendEagleProposer
            | AscendDraftModelProposer
            | AscendSuffixDecodingProposer
            | AscendMedusaProposer
            | None
        ) = None
        self.actual_seq_lengths_q: list[int] = []
        self.decode_token_per_req = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            assert spec_token_num > 0
            self.decode_token_per_req = 1 + spec_token_num
            if get_pp_group().is_last_rank:
                self.drafter = self._get_drafter()
                if self.speculative_config.method == "eagle3":
                    assert isinstance(self.drafter, AscendEagleProposer)
                    self.use_aux_hidden_state_outputs = self.drafter.eagle3_use_aux_hidden_state
                self.rejection_sampler = RejectionSampler(self.sampler)
        self.discard_request_indices = self._make_buffer(self.max_num_reqs, dtype=torch.int64)
        self.num_discarded_requests = 0

    def _get_drafter(self):
        return get_spec_decode_method(self.speculative_config.method, self.vllm_config, self.device, self)

    def _use_aclgraph(self) -> bool:
        return (
            self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            and self.compilation_config.mode == CompilationMode.VLLM_COMPILE
            and not self.model_config.enforce_eager
        )

    def _skip_all_reduce_across_dp_group(self, is_draft_model=False) -> bool:
        """
        Decide whether to skip the all-reduce across the data-parallel (DP) group.

        Skipping is applicable for all dense models and for moe models only on ranks
        that act as KV consumers. We skip the DP all-reduce when either:
        - Both the prefill and decode communication methods are MC2 (or FUSED_MC2), or
        - Decode requires MC2 and ascend_config.recompute_scheduler_enable is True.
        """
        # For dense models, since we don't actually need dp communication, we simply skip it.
        # This usually happens when main model is moe while eagle draft model is dense.
        is_context_moe_model = (
            is_drafter_moe_model(self.vllm_config) if is_draft_model else is_moe_model(self.vllm_config)
        )
        if not is_context_moe_model:
            return True

        # Only applicable to MoE models on KV consumer ranks.
        if not self.is_kv_consumer:
            return False

        def needs_mc2(num_tokens: int) -> bool:
            return select_moe_comm_method(num_tokens, self.vllm_config) in {MoECommType.MC2, MoECommType.FUSED_MC2}

        # Determine whether decode must use MC2. Use max cudagraph capture size
        # if available, otherwise use the maximal uniform decode token count.
        if self.compilation_config.cudagraph_capture_sizes:
            potential_max_tokens = self.compilation_config.max_cudagraph_capture_size
        else:
            potential_max_tokens = self.max_num_reqs * self.uniform_decode_query_len
        decode_must_use_mc2 = needs_mc2(potential_max_tokens)

        # For prefill, use the scheduler's max_num_batched_tokens for a single
        # batch.
        prefill_must_use_mc2 = needs_mc2(self.vllm_config.scheduler_config.max_num_batched_tokens)

        # Skip all-reduce if decode requires MC2 and either prefill also
        # requires MC2 or recompute-based scheduler is enabled.
        return decode_must_use_mc2 and (prefill_must_use_mc2 or self.ascend_config.recompute_scheduler_enable)

    def _sync_metadata_across_dp(
        self, num_tokens: int, with_prefill: bool = False, is_draft_model: bool = False
    ) -> tuple[int, torch.Tensor | None, bool]:
        # TODO: In vLLM, the only thing that needs to be synced is num_tokens, but in
        # our case, we still need to sync the other two flags as well. So we need to
        # include them in the all_reduce operation, and more over, we CANNOT skip it
        # even if we are running in eager mode, which harms performance.
        # FIXME: Restore the `or self.vllm_config.model_config.enforce_eager` here
        # immediately once the other two flags are no longer needed.
        if self.dp_size == 1:
            return num_tokens, None, with_prefill

        if self._skip_all_reduce_across_dp_group(is_draft_model):
            num_tokens_after_padding = torch.tensor([num_tokens] * self.dp_size, device="cpu", dtype=torch.int32)
            return num_tokens, num_tokens_after_padding, with_prefill

        # Sync num_tokens, with_prefill across dp ranks
        num_tokens_tensor = torch.tensor(
            [num_tokens if i == self.dp_rank else 0 for i in range(self.dp_size)], dtype=torch.int32, device="cpu"
        )

        flags_tensor = torch.tensor([int(with_prefill)], dtype=torch.int32, device="cpu")

        packed_tensor = torch.cat([num_tokens_tensor, flags_tensor])
        # use cpu_group to avoid cpu synchronization issue.
        # it can be overlapped with main moell execution on npu.
        dist.all_reduce(packed_tensor, group=get_dp_group().cpu_group)

        # Unpack the results
        num_tokens_across_dp = packed_tensor[:-1]
        synced_flags = packed_tensor[-1:]
        max_tokens_across_dp = torch.max(num_tokens_across_dp).item()
        global_with_prefill = bool(synced_flags[0])

        # Create a tensor for num_tokens_after_padding
        num_tokens_after_padding = torch.tensor([max_tokens_across_dp] * self.dp_size, device="cpu", dtype=torch.int32)

        return max_tokens_across_dp, num_tokens_after_padding, global_with_prefill

    def get_model(self) -> nn.Module:
        # get raw model out of the aclgraph wrapper.
        if isinstance(self.model, ACLGraphWrapper):
            return self.model.unwrap()
        return self.model

    def _pad_query_start_loc_for_fia(
        self,
        num_tokens_padded: int,
        num_reqs_padded: int,
        num_reqs: int,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        batch_desc_num_reqs: int | None = None,
    ) -> int:
        """
        This function is only designed to satisfied the constraint that when the layout is TND,
        the first dimension of `hidden_states` must equal the last element of `actual_seq_lengths_q`.
        """
        # TODO: need refactor later, related to vllm PR #34043 this pr delete func
        # relax_for_mixed_batch_cudagraphs, num_reqs no longer equals the actual number of requests.
        if cudagraph_runtime_mode == CUDAGraphMode.FULL and \
            self.compilation_config.cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs_padded = num_reqs
        else:
            num_reqs_padded = batch_desc_num_reqs if batch_desc_num_reqs is not None else num_reqs

        if num_tokens_padded == num_reqs_padded * self.uniform_decode_query_len:
            # Uniform-batch case: num_reqs must be no greater than num_reqs_padded
            assert num_reqs <= num_reqs_padded

            if num_reqs < num_reqs_padded:
                last_loc = self.query_start_loc.np[num_reqs]
                self.query_start_loc.np[num_reqs + 1 : num_reqs_padded + 1] = (
                    self.arange_np[1 : num_reqs_padded + 1 - num_reqs] * self.uniform_decode_query_len + last_loc
                )
                self.query_start_loc.copy_to_gpu()
        else:
            # Mixed-batch case: num_reqs must equal num_reqs_padded
            assert num_reqs == num_reqs_padded

            # Insert a dummy request instead of setting query_start_loc[num_reqs] = num_tokens_padded directly
            self.query_start_loc.np[num_reqs_padded + 1] = num_tokens_padded
            num_reqs_padded = num_reqs_padded + 1
            self.query_start_loc.copy_to_gpu()

        return num_reqs_padded

    @staticmethod
    def _fia_request_capacity(
        staged_sfa_graph_key: StagedSFAGraphKey | None,
        batch_desc: BatchDescriptor,
    ) -> int | None:
        """Return the request, rather than token, capacity for FIA padding."""
        if staged_sfa_graph_key is not None:
            return staged_sfa_graph_key.request_capacity
        return batch_desc.num_reqs

    def _refresh_layerwise_prefill_block_tables(
        self,
    ) -> tuple[MultiGroupBlockTable, ...]:
        """Populate the fixed bank-1 table in current request-row order."""

        tables = getattr(
            self.input_batch,
            "layerwise_prefill_block_tables",
            (self.input_batch.block_table,),
        )
        if not getattr(self, "layerwise_prefill_p_node", False):
            return tables
        if len(tables) != LAYERWISE_PREFILL_BANK_COUNT:
            raise RuntimeError(
                "Layerwise-prefill P node requires exactly two fixed block "
                "tables."
            )

        for request_index, request_id in enumerate(self.input_batch.req_ids):
            request = self.requests[request_id]
            self._validate_layerwise_prefill_bank_metadata(
                request_id,
                request.block_ids,
                request.block_ids_by_bank,
                request.block_allocation_mode,
                request.allocation_generation,
            )
            assert request.block_ids_by_bank is not None
            tables[1].add_row(request.block_ids_by_bank[1], request_index)
        return tables

    @staticmethod
    def _layerwise_prefill_common_attn_metadata(
        common_attn_metadata: AscendCommonAttentionMetadata,
        execution: DSAExecutionRow,
        request_generations: tuple[tuple[str, int], ...],
        block_table_getter: Any,
    ) -> AscendCommonAttentionMetadata:
        """Lower one canonical execution's groups to independent banks."""

        layer_metadata = copy(common_attn_metadata)
        latent = execution.latent
        (
            layer_metadata.block_table_tensor,
            layer_metadata.slot_mapping,
        ) = block_table_getter(latent.kv_group, latent.bank)
        layer_metadata.indexer_block_table_tensor = None
        layer_metadata.indexer_slot_mapping = None
        if execution.indexer is not None:
            indexer = execution.indexer
            (
                layer_metadata.indexer_block_table_tensor,
                layer_metadata.indexer_slot_mapping,
            ) = block_table_getter(indexer.kv_group, indexer.bank)
        layer_metadata.layerwise_prefill_callback_metadata = (
            LayerwisePrefillCallbackMetadata.for_execution(
                execution,
                request_generations,
            )
        )
        return layer_metadata

    def _layerwise_prefill_draft_common_attn_metadata(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        draft_layer_name: str,
        request_generations: tuple[tuple[str, int], ...],
        block_table_getter: Any,
    ) -> AscendCommonAttentionMetadata:
        draft_row = self.dsa_kv_rows_by_layer_name.get(draft_layer_name)
        if draft_row is None or draft_row.kv_group != 0:
            raise RuntimeError(
                "Layerwise-prefill draft layer is absent from the canonical "
                f"LATENT topology: {draft_layer_name!r}."
            )
        draft_execution = self.dsa_kv_executions_by_ordinal.get(
            draft_row.execution_ordinal
        )
        if draft_execution is None or draft_execution.latent != draft_row:
            raise RuntimeError(
                "Layerwise-prefill draft layer has no canonical execution: "
                f"{draft_layer_name!r}."
            )
        return self._layerwise_prefill_common_attn_metadata(
            common_attn_metadata,
            draft_execution,
            request_generations,
            block_table_getter,
        )

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[torch.Tensor, SpecDecodeMetadata | None, int]:
        """
        :return: tuple[
            logits_indices,
            spec_decode_metadata,
            total_num_scheduled_tokens,
        ]
        """
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        block_tables_by_bank = self._refresh_layerwise_prefill_block_tables()
        for block_table in block_tables_by_bank:
            block_table.commit_block_table(num_reqs)

        # Get the attention state.
        if not scheduler_output.scheduled_spec_decode_tokens:
            num_valid_tokens = num_scheduled_tokens
        else:
            num_valid_tokens = np.array(
                [
                    scheduler_output.num_scheduled_tokens[i]
                    - len(scheduler_output.scheduled_spec_decode_tokens.get(i, []))
                    for i in self.input_batch.req_ids
                ],
                dtype=np.int32,
            )
        attn_state = self._build_attn_state(num_reqs, num_scheduled_tokens, num_valid_tokens)

        # Determine if it's a splitfuse batch
        with_prefill = attn_state not in [AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding]
        self.with_prefill = with_prefill

        # Get positions.
        positions_np = self.positions.np[:total_num_scheduled_tokens]
        fixed_decode_width = (
            1
            if attn_state == AscendAttentionState.DecodeOnly
            else self.decode_threshold
            if attn_state == AscendAttentionState.SpecDecoding
            else 0
        )
        uniform_fixed_decode = (
            not self.use_cp
            and fixed_decode_width in (1, 2)
            and total_num_scheduled_tokens
            == num_reqs * fixed_decode_width
            and np.all(
                num_scheduled_tokens[:num_reqs]
                == fixed_decode_width
            )
        )
        if uniform_fixed_decode:
            if fixed_decode_width == 1:
                req_indices = self.arange_np[:num_reqs]
                cu_num_tokens = self.arange_np[1 : num_reqs + 1]
                positions_np[:] = (
                    self.input_batch.num_computed_tokens_cpu[:num_reqs]
                )
            else:
                assert self._fixed_decode_req_indices is not None
                assert self._fixed_decode_position_offsets is not None
                assert self._fixed_decode_cu_num_tokens is not None
                req_indices = self._fixed_decode_req_indices[
                    :total_num_scheduled_tokens
                ]
                cu_num_tokens = self._fixed_decode_cu_num_tokens[
                    :num_reqs
                ]
                _fill_fixed_decode_positions(
                    positions_np,
                    self.input_batch.num_computed_tokens_cpu,
                    self._fixed_decode_position_offsets,
                    num_reqs,
                    fixed_decode_width,
                )
        else:
            req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)
            cu_num_tokens, arange = self._get_cumsum_and_arange(num_scheduled_tokens)
            np.add(self.input_batch.num_computed_tokens_cpu[req_indices], arange, out=positions_np)

        for block_table in block_tables_by_bank:
            block_table.compute_slot_mapping(req_indices, positions_np)
            block_table.commit_slot_mapping(total_num_scheduled_tokens)

        if self.use_cp:
            self.pcp_manager.init_batch_info(
                num_scheduled_tokens,
                self.input_batch.num_reqs,
            )

        # for pcp, prefill mtp should use origin scheduleroutput ,
        if self.speculative_config and self.use_cp:
            self.pcp_manager.generate_pcp_mtp_input(
                total_num_scheduled_tokens,
                scheduler_output.num_scheduled_tokens,
                with_prefill,
                self.input_batch,
                self.arange_np,
                req_indices,
                positions_np,
                cu_num_tokens,
                self._draft_token_ids,  # type: ignore[has-type]
                scheduler_output,
                self.num_spec_tokens,
            )

        if self.pcp_size > 1:
            num_scheduled_tokens[:num_reqs], position_pcp = self.pcp_manager.update_tokens_for_pcp(
                num_scheduled_tokens[:num_reqs], self.arange_np
            )
            # Re-update after PCP split sequences.
            total_num_scheduled_tokens = sum(num_scheduled_tokens[:num_reqs])
            req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)
            cu_num_tokens, _ = self._get_cumsum_and_arange(num_scheduled_tokens)
            positions_np = self.positions.np[:total_num_scheduled_tokens]
            np.add(
                self.input_batch.num_computed_tokens_cpu[req_indices],
                position_pcp[:total_num_scheduled_tokens],
                out=positions_np,
            )
        if self.pcp_size > 1 and self.pcp_manager.pcp_use_hybrid_attn:
            assert self.pcp_manager.num_scheduled_tokens_padded is not None
            self.query_lens = torch.from_numpy(self.pcp_manager.num_scheduled_tokens_padded)
        else:
            self.query_lens = torch.from_numpy(num_scheduled_tokens)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
        token_indices_tensor = torch.from_numpy(token_indices)
        # Prepare input_ids.
        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(
            self.input_batch.token_ids_cpu_tensor.flatten(),
            0,
            token_indices_tensor,
            out=self.input_ids.cpu[:total_num_scheduled_tokens],
        )
        if self.enable_prompt_embeds:
            is_token_ids = self.input_batch.is_token_ids_tensor.flatten()
            torch.index_select(
                is_token_ids, 0, token_indices_tensor, out=self.is_token_ids.cpu[:total_num_scheduled_tokens]
            )

        # Because we did not pre-allocate a massive prompt_embeds CPU tensor on
        # the InputBatch, we need to fill in the prompt embeds into the expected
        # spots in the GpuModelRunner's pre-allocated prompt_embeds tensor.
        if self.input_batch.req_prompt_embeds and (self.is_multimodal_model or self.enable_prompt_embeds):
            output_idx = 0
            for req_idx in range(num_reqs):
                num_sched = num_scheduled_tokens[req_idx]

                # Skip if this request doesn't have embeddings
                if req_idx not in self.input_batch.req_prompt_embeds:
                    output_idx += num_sched
                    continue

                # Skip if no tokens scheduled
                if num_sched <= 0:
                    output_idx += num_sched
                    continue

                req_embeds = self.input_batch.req_prompt_embeds[req_idx]
                start_pos = self.input_batch.num_computed_tokens_cpu[req_idx]

                # Skip if trying to read beyond available embeddings
                if start_pos >= req_embeds.shape[0]:
                    output_idx += num_sched
                    continue

                # Copy available embeddings
                end_pos = start_pos + num_sched
                actual_end = min(end_pos, req_embeds.shape[0])
                actual_num_sched = actual_end - start_pos

                if actual_num_sched > 0:
                    self.inputs_embeds.cpu[output_idx : output_idx + actual_num_sched].copy_(
                        req_embeds[start_pos:actual_end]
                    )

                output_idx += num_sched

        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
        self.query_start_loc.copy_to_gpu()

        # Now, query_start_loc is padded.
        # But gdn needs an unpadded one.
        # gdn_query_start_loc is an unpadded version of query_start_loc.
        # TODO delete it if fia's check is removed.
        if self._has_gdn:
            self.gdn_query_start_loc.np[0] = 0
            self.gdn_query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
            self.gdn_query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
            self.gdn_query_start_loc.copy_to_gpu()

        self.seq_lens.np[:num_reqs] = self.input_batch.num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens
        self.seq_lens.cpu[num_reqs:].fill_(0)
        self.seq_lens.copy_to_gpu()

        # Fill unused with -1. Needed for reshape_and_cache in attention_cp
        self.query_start_loc.gpu[num_reqs + 1 :].fill_(-1)

        # Copy the tensors to the NPU.
        self._prepare_input_ids(scheduler_output, total_num_scheduled_tokens, cu_num_tokens)
        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self._calc_mrope_positions(scheduler_output)
            self.mrope_positions.gpu.copy_(
                self.mrope_positions.cpu,
                non_blocking=True,
            )
        elif self.uses_xdrope_dim > 0:
            self._calc_xdrope_positions(scheduler_output)
            # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
            self.xdrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.xdrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True,
            )
        else:
            # Common case (1D positions)
            self.positions.copy_to_gpu(total_num_scheduled_tokens)

        # Record the index of requests that should not be sampled,
        # so that we could clear the sampled tokens before returning
        num_tokens = [self.requests[r].num_tokens for r in self.input_batch.req_ids]
        num_tokens_np = np.array(num_tokens, dtype=np.int32)
        base_num_reqs = self.input_batch.num_reqs
        num_reqs = base_num_reqs
        tokens_original = None
        if self.pcp_size > 1:
            # while pcp > 1, we need the original num_scheduled_tokens before split
            # to calculate discard_requests_mask
            tokens_original = [scheduler_output.num_scheduled_tokens[i] for i in self.input_batch.req_ids]
            original_seq_lens_np = self.input_batch.num_computed_tokens_cpu[:num_reqs] + np.array(
                tokens_original, dtype=np.int32
            )
            discard_requests_mask = original_seq_lens_np < num_tokens_np
        else:
            discard_requests_mask = self.seq_lens.np[:num_reqs] < num_tokens_np

        discard_request_indices = np.nonzero(discard_requests_mask)[0]
        self.num_discarded_requests = len(discard_request_indices)
        self.discard_request_indices.np[: self.num_discarded_requests] = discard_request_indices
        if self.num_discarded_requests:
            self.discard_request_indices.copy_to_gpu(self.num_discarded_requests)
        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            spec_decode_metadata = None
            num_draft_tokens = None
            num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
            if self.use_cp:
                logits_indices = self.pcp_manager.get_logits_indices(cu_num_tokens, num_reqs, tokens_original)
                logits_indices = logits_indices.pin_memory().to(self.device, non_blocking=True)
            else:
                logits_indices = self.query_start_loc.gpu[1 : num_reqs + 1] - 1
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            # For chunked prefills, use -1 as mask rather than 0, as guided
            # decoding may rollback speculative tokens.
            new_schedule_reqs = [x.req_id for x in scheduler_output.scheduled_new_reqs]
            num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
            for (
                req_id,
                draft_token_ids,
            ) in scheduler_output.scheduled_spec_decode_tokens.items():
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)
                if (self.is_kv_consumer and req_id in new_schedule_reqs) or \
                   (self.input_batch.num_computed_tokens_cpu[req_idx] >= \
                    self.input_batch.num_prompt_tokens[req_idx]):
                    num_decode_draft_tokens[req_idx] = len(draft_token_ids)
                else:
                    num_decode_draft_tokens[req_idx] = -1

            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens,
                cu_num_tokens,
                num_pcp_pads=self.pcp_manager.num_pcp_pads_cpu[:num_reqs] if self.pcp_size > 1 else None,
            )
            logits_indices = spec_decode_metadata.logits_indices
            num_sampled_tokens = num_draft_tokens + 1

            # For DECODE only cuda graph of some attention backends (e.g., GDN).
            self.num_decode_draft_tokens.np[:num_reqs] = num_decode_draft_tokens
            self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
            self.num_decode_draft_tokens.copy_to_gpu()
        # save logits_indices for pcp spec decode usage
        self.logits_indices = logits_indices

        # Hot-Swap lora model
        if self.lora_config:
            assert np.sum(num_sampled_tokens) <= self.vllm_config.scheduler_config.max_num_batched_tokens
            self.set_active_loras(self.input_batch, num_scheduled_tokens, num_sampled_tokens)
        if lmhead_tp_enable():
            max_num_reqs_across_dp = self.max_num_reqs * self.uniform_decode_query_len
            logits_indices = nn.functional.pad(logits_indices, (0, max_num_reqs_across_dp - logits_indices.shape[0]))

        return (
            logits_indices,
            spec_decode_metadata,
            total_num_scheduled_tokens,
        )

    def _build_attn_state(self, num_reqs, num_scheduled_tokens, num_valid_tokens):
        if np.all(self.input_batch.num_computed_tokens_cpu[:num_reqs] == 0):
            attn_state = AscendAttentionState.PrefillNoCache
        # We assume it is the decode stage, where prefill occurs but only one token is not hit in cache.
        elif np.all(num_scheduled_tokens == 1):
            attn_state = AscendAttentionState.DecodeOnly
            if self.speculative_config and self.speculative_config.method == "mtp":
                # SpecDecoding now supports seq_len=1 and seq_len=2
                # In Prefilling Decoding Disaggregation scenario, SpecDecoding need to supports seq_len=1
                attn_state = AscendAttentionState.SpecDecoding
        # Speculative decoding.
        elif np.all(num_valid_tokens == 1):
            if self.speculative_config:
                attn_state = AscendAttentionState.SpecDecoding
            else:
                attn_state = AscendAttentionState.ChunkedPrefill
        # splitfuse
        elif self.scheduler_config.enable_chunked_prefill:
            attn_state = AscendAttentionState.ChunkedPrefill
        else:
            attn_state = AscendAttentionState.PrefillCacheHit

        # For the overlay of the PCP feature and the eagle3, attn_state needs to be recovered
        # TODO: Resolved the conflict between the sunset of attn_state and the PCP that requires this interface.
        if attn_state == AscendAttentionState.SpecDecoding and self.speculative_config.method != "mtp":
            self.attn_state = AscendAttentionState.ChunkedPrefill  # type: ignore
        else:
            self.attn_state = attn_state  # type: ignore

        return attn_state

    def _sanitize_placeholder_input_ids_for_forward(
        self,
        scheduler_output: SchedulerOutput,
        num_forward_tokens: int,
    ) -> None:
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        if not scheduled_spec_tokens:
            return
        if not any(
            PLACEHOLDER_TOKEN_ID in token_ids
            for token_ids in scheduled_spec_tokens.values()
        ):
            return

        input_ids = self.input_ids.gpu[:num_forward_tokens]
        input_ids.masked_fill_(input_ids == PLACEHOLDER_TOKEN_ID, 0)

    def _calc_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
        num_pcp_pads: np.ndarray | None,
    ) -> SpecDecodeMetadata:
        # Inputs:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]
        # Outputs:
        # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
        # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,
        #                            206, 207, 208]
        # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
        # bonus_logits_indices:     [  3,   4,   7,   8,  10]

        # Compute the logits indices.
        # [4, 1, 3, 1, 2]
        num_sampled_tokens = num_draft_tokens + 1
        # Step 1. [4, 5, 8, 9, 11]
        cu_num_sampled_tokens = np.cumsum(num_sampled_tokens, dtype=np.int32)
        total_num_sampled_tokens = cu_num_sampled_tokens[-1]
        # Step 2. [0, 0, 0, 0, 4, 5, 5, 5, 8, 9, 9]
        cumsums_offsets = np.repeat(cu_num_sampled_tokens - num_sampled_tokens, num_sampled_tokens)
        # Step 3. [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
        arange = self.arange_np[:total_num_sampled_tokens] - cumsums_offsets
        # Step 4. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
        logits_indices = np.repeat(cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)
        # Step 5. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
        logits_indices += arange

        # while pcp > 1, decode results may contain padding (from pcp all-gather),
        # update logits_indices after getting draft_token_ids from ori logits_indices
        if self.pcp_size > 1:
            cu_num_scheduled_tokens = cu_num_scheduled_tokens * self.pcp_size - num_pcp_pads
            logits_indices_pcp = np.repeat(cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)
            logits_indices_pcp += arange
            logits_indices_pcp = torch.from_numpy(logits_indices_pcp).pin_memory().to(self.device, non_blocking=True)

        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the draft logits indices.
        # [3, 3, 5, 5, 6]
        cu_num_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)
        total_num_draft_tokens = cu_num_draft_tokens[-1]
        # [0, 0, 0, 3, 3, 5]
        cumsums_offsets = np.repeat(cu_num_draft_tokens - num_draft_tokens, num_draft_tokens)
        # [0, 1, 2, 0, 1, 0]
        arange = self.arange_np[:total_num_draft_tokens] - cumsums_offsets
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens)
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += arange

        # TODO: Optimize the CPU -> NPU copy.
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).pin_memory().to(self.device, non_blocking=True)
        cu_num_sampled_tokens = torch.from_numpy(cu_num_sampled_tokens).pin_memory().to(self.device, non_blocking=True)
        logits_indices = torch.from_numpy(logits_indices).pin_memory().to(self.device, non_blocking=True)
        target_logits_indices = torch.from_numpy(target_logits_indices).pin_memory().to(self.device, non_blocking=True)
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices).pin_memory().to(self.device, non_blocking=True)

        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = self.input_ids.gpu[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]
        if self.pcp_size > 1:
            logits_indices = logits_indices_pcp
        return SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            cu_num_sampled_tokens=cu_num_sampled_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )

    # TODO: Once the PCP features are complete, it will fully inherit the classes from the VLLM community.
    def propose_draft_token_ids(
        self,
        valid_sampled_token_ids: torch.Tensor | list[list[int]],
        sampling_metadata: SamplingMetadata,
        scheduler_output: "SchedulerOutput",
        spec_decode_metadata: SpecDecodeMetadata,
        spec_decode_common_attn_metadata: AscendCommonAttentionMetadata,
        positions: torch.Tensor,
        num_scheduled_tokens: int,
        hidden_states: torch.Tensor,
        aux_hidden_states: torch.Tensor = None,
        sample_hidden_states: torch.Tensor = None,
        target_model_batch_desc: BatchDescriptor = None,
        target_staged_sfa_graph_key: StagedSFAGraphKey | None = None,
    ) -> list[list[int]] | None:
        if not self.drafter:
            # Speculative decoding is not enabled.
            draft_token_ids = None
        elif isinstance(self.drafter, (AscendNgramProposer, AscendSuffixDecodingProposer)):
            draft_token_ids = self.drafter.propose(valid_sampled_token_ids)
        elif isinstance(self.drafter, AscendMedusaProposer):
            draft_token_ids = self.drafter.propose(
                valid_sampled_token_ids, sampling_metadata, spec_decode_metadata, sample_hidden_states
            )
        elif self.speculative_config.use_eagle() or self.speculative_config.uses_draft_model():
            common_attn_metadata = spec_decode_common_attn_metadata
            sampled_token_ids = valid_sampled_token_ids

            if self.vllm_config.speculative_config.disable_padded_drafter_batch:
                # When padded-batch is disabled, the sampled_token_ids should be
                # the cpu-side list[list[int]] of valid sampled tokens for each
                # request, with invalid requests having empty lists.
                assert isinstance(sampled_token_ids, list), (
                    "sampled_token_ids should be a python list whenpadded-batch is disabled."
                )
                assert self.drafter is not None
                next_token_ids = self.drafter.prepare_next_token_ids_cpu(
                    sampled_token_ids, self.requests, self.input_batch, scheduler_output.num_scheduled_tokens
                )
            else:
                # When using padded-batch, the sampled_token_ids should be
                # the gpu tensor of sampled tokens for each request, of shape
                # (num_reqs, num_spec_tokens + 1) with rejected tokens having
                # value -1.
                assert isinstance(sampled_token_ids, torch.Tensor), (
                    "sampled_token_ids should be a torch.Tensor whenpadded-batch is enabled."
                )
                assert self.drafter is not None
                next_token_ids, valid_sampled_tokens_count = self.drafter.prepare_next_token_ids_padded(
                    common_attn_metadata,
                    sampled_token_ids,
                    self.requests,
                    self.input_batch,
                    self.discard_request_indices.gpu,
                    self.num_discarded_requests,
                )
                self._copy_valid_sampled_token_count(next_token_ids, valid_sampled_tokens_count)

            req_scheduled_tokens = scheduler_output.num_scheduled_tokens
            if self.use_cp:
                long_seq_metadata = self.long_seq_metadata  # type: ignore
                input_ids_pcp_full = self.pcp_manager.input_ids_pcp_full.gpu
                query_start_loc_pcp_full = self.pcp_manager.query_start_loc_pcp_full.gpu
                query_start_loc_pcp_full_cpu = self.pcp_manager.query_start_loc_pcp_full.cpu
                num_reqs = self.input_batch.num_reqs
                num_prefill_reqs = self.pcp_manager.num_prefill_reqs
                num_decode_reqs = self.pcp_manager.num_decode_reqs
            else:
                long_seq_metadata = None  # type: ignore
                num_prefill_reqs = 0
                num_decode_reqs = 0

            num_rejected_tokens_gpu = None
            if spec_decode_metadata is None:
                # update pcp related params
                if self.pcp_size > 1:
                    token_indices_to_sample = query_start_loc_pcp_full[1 : num_reqs + 1] - 1
                    target_token_ids = input_ids_pcp_full[:num_scheduled_tokens]
                    target_positions = self._get_positions(num_scheduled_tokens)
                    target_hidden_states = hidden_states
                    if self.use_aux_hidden_state_outputs:
                        target_hidden_states = torch.cat([h for h in aux_hidden_states], dim=-1)
                else:
                    token_indices_to_sample = None
                    # input_ids can be None for multimodal models.
                    target_token_ids = self.input_ids.gpu[:num_scheduled_tokens]
                    target_positions = self._get_positions(num_scheduled_tokens)
                    if self.use_aux_hidden_state_outputs:
                        target_hidden_states = torch.cat([h[:num_scheduled_tokens] for h in aux_hidden_states], dim=-1)
                    else:
                        target_hidden_states = hidden_states[:num_scheduled_tokens]
            else:
                if self.pcp_size > 1:
                    assert common_attn_metadata is not None
                    common_attn_metadata.query_start_loc_cpu[: num_reqs + 1] = query_start_loc_pcp_full_cpu[
                        : num_reqs + 1
                    ]
                    assert common_attn_metadata is not None
                    common_attn_metadata.query_start_loc[: num_reqs + 1] = query_start_loc_pcp_full[: num_reqs + 1]
                if self.vllm_config.speculative_config.disable_padded_drafter_batch:
                    # NOTE: Currently, MTP-fullgraph is incompatibility with pcp
                    token_indices_to_sample = None
                    assert self.drafter is not None
                    common_attn_metadata, token_indices = self.drafter.prepare_inputs(
                        common_attn_metadata, sampled_token_ids, spec_decode_metadata.num_draft_tokens
                    )
                else:
                    assert self.drafter is not None
                    common_attn_metadata, token_indices, token_indices_to_sample, num_rejected_tokens_gpu = (
                        self.drafter.prepare_inputs_padded(
                            common_attn_metadata, spec_decode_metadata, valid_sampled_tokens_count
                        )
                    )
                if self.pcp_size > 1:
                    target_token_ids = input_ids_pcp_full[token_indices]
                    target_positions = positions
                    target_hidden_states = hidden_states
                    if self.use_aux_hidden_state_outputs:
                        target_hidden_states = torch.cat([h for h in aux_hidden_states], dim=-1)
                else:
                    target_token_ids = self.input_ids.gpu[token_indices]
                    target_positions = self._get_positions(token_indices)
                    if self.use_aux_hidden_state_outputs:
                        target_hidden_states = torch.cat([h[token_indices] for h in aux_hidden_states], dim=-1)
                    else:
                        target_hidden_states = hidden_states[token_indices]
            assert self.drafter is not None
            draft_token_ids = self.drafter._propose(
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                token_indices_to_sample=token_indices_to_sample,
                common_attn_metadata=common_attn_metadata,
                target_model_batch_desc=target_model_batch_desc,
                sampling_metadata=sampling_metadata,
                req_scheduled_tokens=req_scheduled_tokens,
                long_seq_metadata=long_seq_metadata,
                num_prefill_reqs=num_prefill_reqs,
                num_decode_reqs=num_decode_reqs,
                scheduler_output=scheduler_output,
                num_scheduled_tokens=num_scheduled_tokens,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                target_staged_sfa_graph_key=target_staged_sfa_graph_key,
            )
        else:
            raise ValueError(f"Unknown speculative decoding method: {self.speculative_config.method}")

        return draft_token_ids

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        if self.vllm_config.model_config.enable_return_routed_experts:
            capturer = RoutedExpertsCapturer.get_instance()
            if capturer is not None:
                capturer.clear_buffer()
            else:
                logger.warning("RoutedExpertsCapturer is not initialized.")
        if self.execute_model_state is not None:
            raise RuntimeError("State error: sample_tokens() must be called after execute_model() returns None.")
        # self._draft_token_ids is None when `input_fits_in_drafter=False`
        # and there is no draft tokens scheduled. so it need to update the
        # spec_decoding info in scheduler_output with async_scheduling.
        # use deepcopy to avoid the modification has influence on the
        # scheduler_output in engine core process.
        # TODO(Ronald1995): deepcopy is expensive when there is a large
        # number of requests, optimize it later.
        if (
            self.use_async_scheduling and self.num_spec_tokens and self._draft_token_ids is None  # type: ignore[has-type]
        ):
            scheduler_output = deepcopy(scheduler_output)
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        with record_function_or_nullcontext("prepare input"):
            with self.synchronize_input_prep():
                # Update persistent batch states.
                self._update_states(scheduler_output)

                if has_ec_transfer() and get_ec_transfer().is_producer:
                    with self.maybe_get_ec_connector_output(
                        scheduler_output,
                        encoder_cache=self.encoder_cache,
                    ) as ec_connector_output:
                        self._execute_mm_encoder(scheduler_output)
                        return make_empty_encoder_model_runner_output(scheduler_output)

                if not num_scheduled_tokens:
                    if (
                        self.parallel_config.distributed_executor_backend == "external_launcher"
                        and self.parallel_config.data_parallel_size > 1
                    ):
                        # this is a corner case when both external launcher
                        # and DP are enabled, num_scheduled_tokens could be
                        # 0, and has_unfinished_requests in the outer loop
                        # returns True. before returning early here we call
                        # dummy run to ensure coordinate_batch_across_dp
                        # is called into to avoid out of sync issues.
                        self._dummy_run(1)
                    if not has_kv_transfer_group():
                        # Return empty ModelRunnerOutput if no work to do.
                        return EMPTY_MODEL_RUNNER_OUTPUT
                    return self.kv_connector_no_forward(scheduler_output, self.vllm_config)
                if self.cache_config.kv_sharing_fast_prefill:
                    assert not self.num_prompt_logprobs, (
                        "--kv-sharing-fast-prefill produces incorrect "
                        "logprobs for prompt tokens, tokens, please disable "
                        "it when the requests need prompt logprobs"
                    )

                num_reqs = self.input_batch.num_reqs
                req_ids = self.input_batch.req_ids
                tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
                num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
                max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())

                (
                    logits_indices,
                    spec_decode_metadata,
                    total_num_scheduled_tokens,
                ) = self._prepare_inputs(
                    scheduler_output,
                    num_scheduled_tokens_np,
                )

                num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens
                if self.pcp_size > 1:
                    num_tokens_unpadded = self.pcp_manager.total_num_sampled_tokens_pcp
                cascade_attn_prefix_lens = None
                # Disable cascade attention when using microbatching (DBO)
                if self.cascade_attn_enabled and not self.parallel_config.enable_dbo:
                    # Pre-compute cascade attention prefix lengths
                    cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(
                        num_scheduled_tokens_np,
                        self.input_batch.num_computed_tokens_cpu[:num_reqs],
                        scheduler_output.num_common_prefix_blocks,
                    )

                staged_sfa_local_route = self._staged_sfa_local_route(
                    num_tokens_unpadded=num_tokens_unpadded,
                    num_reqs=num_reqs,
                    num_scheduled_tokens=num_scheduled_tokens_np,
                    index_topk=self.dsa_index_topk,
                    has_cascade_attention=(
                        cascade_attn_prefix_lens is not None
                    ),
                    request_ids=self.input_batch.req_ids[:num_reqs],
                    num_computed_tokens=(
                        self.input_batch.num_computed_tokens_cpu[:num_reqs]
                    ),
                    prompt_lens=self.input_batch.num_prompt_tokens[:num_reqs],
                    kv_connector_metadata=(
                        scheduler_output.kv_connector_metadata
                    ),
                )

                (
                    cudagraph_mode,
                    batch_desc,
                    should_ubatch,
                    num_tokens_across_dp,
                    cudagraph_stats,
                ) = self._determine_batch_execution_and_padding(
                    num_tokens=num_tokens_unpadded,
                    num_reqs=num_reqs,
                    num_scheduled_tokens_np=num_scheduled_tokens_np,
                    max_num_scheduled_tokens=max_num_scheduled_tokens,
                    use_cascade_attn=cascade_attn_prefix_lens is not None,
                    force_eager=self.model_config.enforce_eager,
                    num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
                    staged_sfa_route_action=(
                        staged_sfa_local_route.action
                        if self._staged_sfa_graph_capture_sizes
                        else None
                    ),
                )
                dp_route_action = self._staged_sfa_dp_route_action

                logger.debug(
                    "Running batch with cudagraph_mode: %s, batch_descriptor: %s, "
                    "should_ubatch: %s, num_tokens_across_dp: %s",
                    cudagraph_mode,
                    batch_desc,
                    should_ubatch,
                    num_tokens_across_dp,
                )

                num_tokens_padded = batch_desc.num_tokens
                staged_sfa_route = self._staged_sfa_live_route(
                    local_route=staged_sfa_local_route,
                    dp_route_action=dp_route_action,
                    cudagraph_mode=cudagraph_mode,
                    batch_descriptor=batch_desc,
                    num_tokens_unpadded=num_tokens_unpadded,
                    num_tokens_padded=num_tokens_padded,
                    num_reqs=num_reqs,
                    should_ubatch=should_ubatch,
                )
                staged_sfa_graph_key = self._apply_staged_sfa_route(
                    staged_sfa_route
                )
                if (
                    self._staged_sfa_graph_capture_sizes
                    and staged_sfa_graph_key is None
                ):
                    cudagraph_mode = CUDAGraphMode.NONE
                num_reqs_padded = batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
                ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
                    should_ubatch,
                    num_scheduled_tokens_np,
                    num_tokens_padded,
                    num_reqs_padded,
                    self.parallel_config.num_ubatches,
                )

                pad_attn = cudagraph_mode == CUDAGraphMode.FULL

                # NOTE(Angazenn): According to https://github.com/vllm-project/vllm/pull/30877,
                # there should be a corresponding 'postprocess_mamba'. However, it is called inside
                # '_update_states_after_model_execute', which is not overridden in vLLM-Ascend.
                # We simply utilize the implementation in vLLM.
                if self.cache_config.mamba_cache_mode == "align":
                    mamba_utils.preprocess_mamba(
                        scheduler_output,
                        self.kv_cache_config,
                        self.cache_config,
                        self.mamba_state_idx,
                        self.input_batch,
                        self.requests,
                        self.compilation_config.static_forward_context,
                        self.model.get_mamba_state_copy_func(),
                        self._get_mamba_copy_bufs(),
                    )

                use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
                ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices

                if (
                    cudagraph_mode == CUDAGraphMode.FULL
                    or staged_sfa_graph_key is not None
                    or (enable_sp() and not self.model_config.use_mla)
                    and self.pcp_size * self.dcp_size == 1
                ):
                    # Currently, Graph Mode and SP will both pad num_tokens,
                    # Another possible condition is num_tokens_padded != num_tokens_unpadded
                    # but this scope is way too big and the consequences are unpredictable
                    old_num_reqs_padded = num_reqs_padded
                    num_reqs_padded = self._pad_query_start_loc_for_fia(
                        num_tokens_padded,
                        num_reqs_padded,
                        num_reqs,
                        cudagraph_mode,
                        self._fia_request_capacity(
                            staged_sfa_graph_key,
                            batch_desc,
                        ),
                    )
                    if enable_sp() and num_tokens_padded == num_tokens_unpadded:
                        if num_reqs_padded > old_num_reqs_padded:
                            num_reqs_padded = old_num_reqs_padded
                            self.query_start_loc.np[num_reqs_padded + 1] = 0

                (attn_metadata, spec_decode_common_attn_metadata) = self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded
                    if not (self.use_cp and self.pcp_manager.pcp_use_hybrid_attn)
                    else total_num_scheduled_tokens,
                    num_tokens_padded=num_tokens_padded,
                    num_reqs=num_reqs,
                    num_reqs_padded=num_reqs_padded,
                    max_query_len=max_num_scheduled_tokens,
                    ubatch_slices=ubatch_slices_attn,
                    logits_indices=logits_indices,
                    use_spec_decode=use_spec_decode,
                    num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                    num_scheduled_tokens_np=num_scheduled_tokens_np,
                    cascade_attn_prefix_lens=cascade_attn_prefix_lens,
                    cold_compact_resumes=(
                        staged_sfa_route.cold_compact_resumes
                    ),
                    resident_remap_frontiers=(
                        staged_sfa_route.frontiers
                        if staged_sfa_route.action
                        == StagedSFARouteAction.STAGED
                        else None
                    ),
                )

                self._sanitize_placeholder_input_ids_for_forward(
                    scheduler_output,
                    num_tokens_padded
                    if not (
                        self.use_cp
                        and self.pcp_manager.pcp_use_hybrid_attn
                    )
                    else total_num_scheduled_tokens,
                )

            (
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
                ec_connector_output,
            ) = self._preprocess(
                scheduler_output,
                num_tokens_padded
                if not (self.use_cp and self.pcp_manager.pcp_use_hybrid_attn)
                else total_num_scheduled_tokens,
                intermediate_tensors,
            )

            # update global cos, sin
            update_cos_sin(positions)

        if self.dynamic_eplb:
            with record_function_or_nullcontext("EPLB weight D2D"):
                self.eplb_updator.forward_before()

        # Set cudagraph mode to none if calc_kv_scales is true.
        # KV scales calculation involves dynamic operations that are incompatible
        # with CUDA graph capture.
        if self.calculate_kv_scales:  # type: ignore[has-type]
            cudagraph_mode = CUDAGraphMode.NONE
            # Mark KV scales as calculated after the first forward pass
            self.calculate_kv_scales = False  # type: ignore[has-type]
        # prevent debugger is None
        if self.debugger is not None:
            dbg_cfg = getattr(self.debugger, "config", None)
            dump_level = str(getattr(dbg_cfg, "level", "L1")).upper() if dbg_cfg is not None else "L1"
            if dump_level in ("L0", "MIX"):
                self.debugger.start(model=self.model)
            else:
                self.debugger.start()
        if self.ascend_config.enable_async_exponential:
            self.sampler.do_async_exponential(
                b_s=logits_indices.shape[0],
                head_dim=self.model_config.get_vocab_size(),
                generators=self.input_batch.sampling_metadata.generators,
            )

        # Encoder-decoder models can only compile the pure decode steps where no
        # encoder inputs are present. Use eager for the first pass.
        num_encoder_reqs = len(scheduler_output.scheduled_encoder_inputs)
        has_encoder_input = self.model_config.is_encoder_decoder and num_encoder_reqs > 0

        # DSA latent offload (GLM5.1): pass the manager + per-request ids/prompt lengths
        # (from input_batch, batch order) into the forward context so AscendSFAImpl can
        # drive store/gather. No-op when the feature is off.
        dsa_offload_manager = getattr(self, "dsa_offload_manager", None)
        dsa_adapter_cache = getattr(self, "dsa_adapter_cache", None)
        dsa_req_ids = None
        dsa_prompt_lens = None
        # Thread per-request ids/prompt lengths (input_batch row order) into the
        # forward context whenever a DSA sparse path needs them: the offload manager
        # (Option A), the shrink-latent LMCache path (keys selected-token rows by
        # req_id), AND the adapter latent cache.
        if dsa_offload_manager is not None or self.dsa_shrink_latent or dsa_adapter_cache is not None:
            num_reqs = self.input_batch.num_reqs
            dsa_req_ids = self.input_batch.req_ids[:num_reqs]
            dsa_prompt_lens = torch.from_numpy(self.input_batch.num_prompt_tokens[:num_reqs])

        # Run forward pass
        clear_kv_metadata = self.speculative_config is None
        diag_enabled = _mtp_dw_diag_enabled()
        diag_req_ids: set[str] | None = None
        diag_post_commit_req_ids: set[str] | None = None
        diag_deep_req_ids: set[str] | None = None
        if diag_enabled:
            diag_req_ids = _mtp_dw_sample_requests(self, scheduler_output)
            previous_frontiers = getattr(
                self, "_mtp_dw_diag_committed_frontiers", None
            )
            active_req_ids = (
                {str(req_id) for req_id in dsa_req_ids}
                if dsa_req_ids is not None
                else set()
            )
            if previous_frontiers is not None:
                for req_id in set(previous_frontiers) - active_req_ids:
                    previous_frontiers.pop(req_id, None)
            self._mtp_dw_diag_current_req_ids = diag_req_ids
            _mtp_dw_for_requests(
                self,
                scheduler_output,
                "config",
                "target_forward",
                req_ids=diag_req_ids,
                mtp_enabled=self.speculative_config is not None,
                decode_window_size=_mtp_dw_window_size(),
                diag_enabled=True,
            )
            _mtp_dw_for_requests(
                self,
                scheduler_output,
                "finalize",
                "target_forward",
                req_ids=diag_req_ids,
                deferred=not clear_kv_metadata,
                order=0,
            )
        with (
            record_function_or_nullcontext("forward"),
            set_ascend_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                aclgraph_runtime_mode=cudagraph_mode,
                batch_descriptor=batch_desc,
                num_actual_tokens=scheduler_output.total_num_scheduled_tokens,
                model_instance=self.model,
                max_tokens_across_pcp=0 if self.pcp_size == 1 else self.pcp_manager.max_num_tokens_across_pcp,
                skip_compiled=has_encoder_input,
                dsa_offload_manager=dsa_offload_manager,
                dsa_req_ids=dsa_req_ids,
                dsa_prompt_lens=dsa_prompt_lens,
                dsa_adapter_cache=dsa_adapter_cache,
                mtp_dw_diag_req_ids=diag_req_ids,
                mtp_dw_diag_post_commit_req_ids=diag_post_commit_req_ids,
                mtp_dw_deep_diag_req_ids=diag_deep_req_ids,
                staged_sfa_route=staged_sfa_route,
                staged_sfa_graph_key=staged_sfa_graph_key,
            ),
            self.maybe_get_kv_connector_output(
                scheduler_output,
                **(
                    {"defer_finalize": not clear_kv_metadata}
                ),
            ) as kv_connector_output,
        ):
            # Connector metadata is bound by maybe_get_kv_connector_output's
            # __enter__. Sample committed frontiers only after that point so the
            # first forward that can retrieve a new window also forces a remap
            # diagnostic for the same window.
            if (
                self._staged_sfa_graph_capture_sizes
                and staged_sfa_graph_key is None
            ):
                self._synchronize_staged_sfa_capture_unsafe_loads()
            if diag_enabled and dsa_req_ids is not None:
                decode_requests = scheduled_decode_requests(
                    dsa_req_ids,
                    scheduler_output.num_scheduled_tokens,
                    self.input_batch.num_computed_tokens_cpu,
                    self.input_batch.num_prompt_tokens,
                )
                decode_req_ids = [
                    req_id for _, req_id in decode_requests
                ]
                committed_frontiers = (
                    get_lmcache_sparse_cached_tokens(decode_req_ids)
                    if decode_req_ids
                    else None
                )
                if committed_frontiers is not None:
                    decode_committed_frontiers = [
                        int(committed) for committed in committed_frontiers
                    ]
                    if decode_req_ids:
                        if previous_frontiers is None:
                            previous_frontiers = {}
                            self._mtp_dw_diag_committed_frontiers = (
                                previous_frontiers
                            )
                        diag_post_commit_req_ids = post_commit_sample_requests(
                            previous_frontiers,
                            decode_req_ids,
                            decode_committed_frontiers,
                        )
                        if envs_ascend.VLLM_ASCEND_MTP_DW_DEEP_DIAG:
                            # Sample every committed-window advance. The connector
                            # keys content probes by frontier, so this preserves one
                            # bounded probe per group and window.
                            diag_deep_req_ids = diag_post_commit_req_ids
                        diag_req_ids.update(diag_post_commit_req_ids)
                        forward_context = get_forward_context()
                        forward_context.mtp_dw_diag_req_ids = diag_req_ids
                        forward_context.mtp_dw_diag_post_commit_req_ids = (
                            diag_post_commit_req_ids
                        )
                        forward_context.mtp_dw_deep_diag_req_ids = (
                            diag_deep_req_ids
                        )
            if staged_sfa_graph_key is not None:
                first_layer_name, first_impl = self._staged_sfa_impls[0]
                first_impl.bootstrap_cross_layer(first_layer_name)
            hidden_states = self._model_forward(
                num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs
            )
            target_diag_session = getattr(
                get_forward_context(),
                "_target_sfa_diag_session",
                None,
            )
            if envs_ascend.VLLM_ASCEND_MTP_DRAFT_DEBUG:
                target_tail_boundary(
                    target_diag_session,
                    "model_forward",
                    hidden_states,
                )
            self._target_sfa_diag_session = target_diag_session
            if staged_sfa_graph_key is not None:
                for _, impl in self._staged_sfa_impls:
                    impl.submit_cross_layer_save()
        with record_function_or_nullcontext("post process"):
            aux_hidden_states = None
            if self.use_aux_hidden_state_outputs:
                hidden_states, aux_hidden_states = hidden_states
            if self.pcp_size > 1:
                # NOTE we must `slice` hidden_states because pcp_allgather_restore_idx
                # ignores the padding from CUDA Graph.
                hidden_states = self.pcp_manager.get_restore_hidden_states(hidden_states)
                if aux_hidden_states is not None:
                    aux_hidden_states = [
                        self.pcp_manager.get_restore_hidden_states(aux_hidden_states_pcp)
                        for aux_hidden_states_pcp in aux_hidden_states
                    ]

            if not self.broadcast_pp_output:
                # Common case.
                if not get_pp_group().is_last_rank:
                    # Return the intermediate tensors.
                    assert isinstance(hidden_states, IntermediateTensors)
                    hidden_states.kv_connector_output = kv_connector_output
                    self.kv_connector_output = kv_connector_output
                    if self.debugger is not None:
                        self.debugger.stop()
                        self.debugger.step()
                    return hidden_states
                if self.is_pooling_model:
                    # Return the pooling output.
                    output = self._pool(
                        hidden_states, num_scheduled_tokens, num_scheduled_tokens_np, kv_connector_output
                    )
                    output.kv_connector_output = kv_connector_output
                    if self.debugger is not None:
                        self.debugger.stop()
                        self.debugger.step()
                    return output

                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
            else:
                # Rare case.
                assert not self.is_pooling_model

                if not get_pp_group().is_last_rank:
                    sample_hidden_states = hidden_states[logits_indices]
                    get_pp_group().send_tensor_dict(hidden_states.tensors, all_gather_group=get_tp_group())
                    logits = None
                else:
                    sample_hidden_states = hidden_states[logits_indices]
                    logits = self.model.compute_logits(sample_hidden_states)

                model_output_broadcast_data: dict[str, Any] = {}
                if logits is not None:
                    model_output_broadcast_data["logits"] = logits.contiguous()
                broadcasted = get_pp_group().broadcast_tensor_dict(
                    model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
                )
                assert broadcasted is not None
                logits = broadcasted["logits"]

            if envs_ascend.VLLM_ASCEND_MTP_DRAFT_DEBUG:
                target_tail_boundary(
                    getattr(self, "_target_sfa_diag_session", None),
                    "logits",
                    logits,
                )

            # Apply structured output bitmasks if present
            self.execute_model_state = ExecuteModelState(
                scheduler_output,
                logits,
                spec_decode_metadata,
                spec_decode_common_attn_metadata,
                hidden_states,
                sample_hidden_states,
                aux_hidden_states,
                attn_metadata,
                positions,
                ec_connector_output,
                cudagraph_stats,
                batch_desc,
                staged_sfa_graph_key,
            )
            self.kv_connector_output = kv_connector_output
        return None

    @torch.inference_mode()
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None

        if self.execute_model_state is None:
            # Nothing to do (PP non-final rank case), output isn't used.
            # receive sampled token ids from the last PP rank when using
            # async scheduling + pipeline parallelism so downstream code
            # (e.g., PCP input preparation) can access them.
            if self.use_async_scheduling and get_pp_group().world_size > 1:
                self._pp_receive_prev_sampled_token_ids_to_input_batch()
            if not kv_connector_output:
                return None  # noqa
            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            if kv_connector_output.is_empty():
                return EMPTY_MODEL_RUNNER_OUTPUT

            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return output

        # Unpack ephemeral state.
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            attn_metadata,
            positions,
            ec_connector_output,
            cudagraph_stats,
            batch_desc,
            staged_sfa_graph_key,
        ) = self.execute_model_state
        # Clear ephemeral state.
        self.execute_model_state = None

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            # here we are different from gpu_model_runner,
            # the apply_grammar_bitmask uses torch.compile to optimize this,ascend does not support it now
            logits_dtype = logits.dtype
            logits = logits.to("cpu").float()
            apply_grammar_bitmask(scheduler_output, grammar_output, self.input_batch, logits)
            logits = logits.to(self.device).to(logits_dtype)

        with record_function_or_nullcontext("sample_token"):
            sampler_output = self._sample(logits, spec_decode_metadata)
        if envs_ascend.VLLM_ASCEND_MTP_DRAFT_DEBUG:
            target_tail_boundary(
                getattr(self, "_target_sfa_diag_session", None),
                "sampling",
                sampler_output,
            )

        if self.need_accepted_tokens:
            if self.sampling_done_event is None:
                self.sampling_done_event = torch.npu.Event()

            assert self.sampling_done_event is not None
            self.sampling_done_event.record()

        def propose_draft_token_ids(sampled_token_ids):
            assert spec_decode_common_attn_metadata is not None
            draft_diag_scope = (
                self.drafter.mtp_draft_diagnostic_scope()
                if getattr(self.drafter, "method", None) == "mtp"
                else nullcontext()
            )
            with draft_diag_scope:
                self._draft_token_ids = self.propose_draft_token_ids(
                    sampled_token_ids,
                    self.input_batch.sampling_metadata,
                    scheduler_output,
                    spec_decode_metadata,
                    spec_decode_common_attn_metadata,
                    positions,
                    scheduler_output.total_num_scheduled_tokens,
                    hidden_states,
                    aux_hidden_states,
                    sample_hidden_states,
                    batch_desc,
                    staged_sfa_graph_key,
                )
            self._copy_draft_token_ids_to_cpu(scheduler_output)

        (
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = self._bookkeeping_sync(
            scheduler_output,
            sampler_output,
            logits,
            hidden_states,
            scheduler_output.total_num_scheduled_tokens,
            spec_decode_metadata,
        )
        _mtp_dw_for_requests(
            self,
            scheduler_output,
            "finalize",
            "bookkeeping",
            req_ids=getattr(self, "_mtp_dw_diag_current_req_ids", set()),
            deferred=self.speculative_config is not None,
            order=1,
        )

        with record_function_or_nullcontext("draft_token"):
            if self.speculative_config:
                use_padded_batch = (
                    self.speculative_config
                    and (self.speculative_config.use_eagle() or self.speculative_config.uses_draft_model())
                    and not self.speculative_config.disable_padded_drafter_batch
                )
                if use_padded_batch:
                    # EAGLE speculative decoding can use the GPU sampled tokens
                    # as inputs, and does not need to wait for bookkeeping to finish.
                    propose_draft_token_ids(sampler_output.sampled_token_ids)
                if not use_padded_batch:
                    # ngram and other speculative decoding methods use the sampled
                    # tokens on the CPU, so they are run after bookkeeping.
                    propose_draft_token_ids(valid_sampled_token_ids)

                if _mtp_dw_diag_enabled():
                    draft_counts = {}
                    if self._draft_token_ids is not None:
                        for index, req_id in enumerate(
                            scheduler_output.num_scheduled_tokens
                        ):
                            try:
                                draft_counts[req_id] = len(
                                    self._draft_token_ids[index]
                                )
                            except (IndexError, TypeError):
                                draft_counts[req_id] = None
                    diag_req_ids = getattr(
                        self, "_mtp_dw_diag_current_req_ids", set()
                    )
                    for req_id in scheduler_output.num_scheduled_tokens:
                        if req_id not in diag_req_ids:
                            continue
                        _mtp_dw_event(
                            "finalize",
                            req=req_id,
                            frontier=self._mtp_dw_diag_current_frontiers.get(
                                req_id
                            ),
                            event="draft_proposal",
                            deferred=True,
                            order=2,
                            draft_count=draft_counts.get(req_id),
                        )

            if has_kv_transfer_group():
                if self.speculative_config:
                    completed_decode_window_saves = self.finalize_kv_connector()
                    diag_req_ids = getattr(
                        self, "_mtp_dw_diag_current_req_ids", set()
                    )
                    for req_id in scheduler_output.num_scheduled_tokens:
                        if req_id not in diag_req_ids:
                            continue
                        _mtp_dw_event(
                            "finalize",
                            req=req_id,
                            frontier=self._mtp_dw_diag_current_frontiers.get(
                                req_id
                            ),
                            event="connector_finalize",
                            deferred=True,
                            order=3,
                            completed_window_end=(
                                completed_decode_window_saves.get(req_id)
                            ),
                        )
                    if completed_decode_window_saves:
                        if kv_connector_output is None:
                            kv_connector_output = KVConnectorOutput()
                        for req_id, window_end in completed_decode_window_saves.items():
                            kv_connector_output.completed_decode_window_saves[
                                req_id
                            ] = max(
                                kv_connector_output.completed_decode_window_saves.get(
                                    req_id, 0
                                ),
                                window_end,
                            )

        if self.model_config.enable_return_routed_experts:
            capturer = RoutedExpertsCapturer.get_instance()
            if capturer is not None:
                capturer.save_captured_experts(indices=self.cpu_slot_mapping)
            else:
                logger.warning("RoutedExpertsCapturer is not initialized.")

        model_runner_output = ModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            kv_connector_output=kv_connector_output,
            pooler_output=[],
            ec_connector_output=ec_connector_output if self.supports_mm_inputs else None,
            cudagraph_stats=cudagraph_stats,
        )

        if self.dynamic_eplb:
            with record_function_or_nullcontext("EPLB update"):
                self.eplb_updator.forward_end()

        if self.debugger is not None:
            self.debugger.stop()
            self.debugger.step()

        if self.need_accepted_tokens:
            assert self.sampling_done_event is not None
            with (
                record_function_or_nullcontext("async_state_update"),
                torch.npu.stream(global_stream()),
            ):
                global_stream().wait_event(self.sampling_done_event)
                self._update_states_after_model_execute(sampler_output.sampled_token_ids, scheduler_output)

        # In async scheduling + PP, broadcast sampled token ids from the
        # last PP rank so other PP ranks can receive them without going
        # through the scheduler/engine IPC path.
        if self.use_async_scheduling:
            pp = get_pp_group()
            if pp.world_size > 1 and pp.is_last_rank:
                self._pp_broadcast_prev_sampled_token_ids(sampler_output.sampled_token_ids)

        if not self.use_async_scheduling:
            return model_runner_output
        return AsyncGPUModelRunnerOutput(
            model_runner_output=model_runner_output,
            sampled_token_ids=sampler_output.sampled_token_ids,
            logprobs_tensors=sampler_output.logprobs_tensors,
            invalid_req_indices=invalid_req_indices,
            async_output_copy_stream=self.async_output_copy_stream,
            vocab_size=self.input_batch.vocab_size,
        )

    # overwrite _sample for lmhead_tp_enable and need_accepted_tokens
    def _sample(self, logits, spec_decode_metadata):
        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        if spec_decode_metadata is None:
            if lmhead_tp_enable() and logits is not None:
                logits = logits[: self.input_batch.num_reqs]
            return self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )

        if lmhead_tp_enable() and logits is not None:
            logits = logits[: len(spec_decode_metadata.logits_indices)]
        sampler_output = self.rejection_sampler(
            spec_decode_metadata,
            None,  # draft_probs
            logits,
            sampling_metadata,
        )
        return sampler_output

    # TODO: remove this func after eagle_proposer is refactored and
    #  _bookkeeping_sync is moved after propose_draft_token_ids
    def _bookkeeping_sync(
        self,
        scheduler_output: "SchedulerOutput",
        sampler_output: SamplerOutput,
        logits: torch.Tensor | None,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> tuple[
        LogprobsLists | None,
        list[list[int]],
        dict[str, LogprobsTensors | None],
        list[str],
        dict[str, int],
        list[int],
    ]:
        # TODO: implement PR 28597 from vllm
        discard_sampled_tokens_req_indices = self.discard_request_indices.np[: self.num_discarded_requests]
        for i in discard_sampled_tokens_req_indices:
            gen = self.input_batch.generators.get(int(i))
            if gen is not None:
                gen.set_offset(gen.get_offset() - 4)

        # Copy some objects so they don't get modified after returning.
        # This is important when using async scheduling.
        req_ids_output_copy = self.input_batch.req_ids.copy()
        req_id_to_index_output_copy = self.input_batch.req_id_to_index.copy()

        num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
        sampled_token_ids = sampler_output.sampled_token_ids
        logprobs_tensors = sampler_output.logprobs_tensors
        invalid_req_indices = []
        cu_num_tokens: list[int] | None = None
        if not self.use_async_scheduling:
            # Get the valid generated tokens.
            max_gen_len = sampled_token_ids.shape[-1]
            if max_gen_len == 1:
                # No spec decode tokens.
                valid_sampled_token_ids = self._to_list(sampled_token_ids)
                # Mask out the sampled tokens that should not be sampled.
                for i in discard_sampled_tokens_req_indices:
                    valid_sampled_token_ids[int(i)].clear()
            else:
                # Includes spec decode tokens.
                valid_sampled_token_ids, cu_num_tokens = RejectionSampler.parse_output(
                    sampled_token_ids,
                    self.input_batch.vocab_size,
                    discard_sampled_tokens_req_indices,
                    logprobs_tensors=logprobs_tensors,
                )
        else:
            valid_sampled_token_ids = []
            invalid_req_indices = discard_sampled_tokens_req_indices.tolist()
            invalid_req_indices_set = set(invalid_req_indices)

            if self.num_spec_tokens <= 0:
                assert sampled_token_ids.shape[-1] == 1
                # Cache the sampled tokens on the NPU and avoid CPU sync.
                # These will be copied into input_ids in the next step
                # when preparing inputs.
                self.input_batch.prev_sampled_token_ids = sampled_token_ids

            self.input_batch.prev_req_id_to_index = {
                req_id: i for i, req_id in enumerate(self.input_batch.req_ids) if i not in invalid_req_indices_set
            }

        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        # NOTE(woosuk): As an exception, when using PP, the scheduler sends
        # the sampled tokens back, because there's no direct communication
        # between the first-stage worker and the last-stage worker.
        req_ids = self.input_batch.req_ids
        for req_idx in range(num_sampled_tokens):
            if self.use_async_scheduling:
                sampled_ids = [-1] if req_idx not in invalid_req_indices_set else None
            else:
                sampled_ids = valid_sampled_token_ids[req_idx]

            num_sampled_ids: int = len(sampled_ids) if sampled_ids else 0

            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + num_sampled_ids
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}"
            )

            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
            self.input_batch.is_token_ids[req_idx, start_idx:end_idx] = True
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx

            req_id = req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        logprobs_lists = (
            logprobs_tensors.tolists(cu_num_tokens)
            if not self.use_async_scheduling and logprobs_tensors is not None
            else None
        )

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:num_scheduled_tokens],
            scheduler_output.num_scheduled_tokens,
        )

        return (
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        )

    # all-gather one hidden-states in sp scene
    @staticmethod
    def _all_gather_hidden_states(hidden_states):
        hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
        pad_size = get_forward_context().pad_size
        if pad_size > 0:
            hidden_states = hidden_states[:-pad_size, :]

        return hidden_states

    # all-gather a list of hidden-states in sp scene
    @staticmethod
    def _all_gather_hidden_states_list(hidden_states_list):
        return [NPUModelRunner._all_gather_hidden_states(hidden_states) for hidden_states in hidden_states_list]

    # all-gather hidden-states in last layer with aux-hidden-states in sp scene
    @staticmethod
    def _all_gather_hidden_states_and_aux(hidden_states):
        if isinstance(hidden_states, tuple):
            return (
                NPUModelRunner._all_gather_hidden_states(hidden_states[0]),
                NPUModelRunner._all_gather_hidden_states_list(hidden_states[1]),
            )
        return NPUModelRunner._all_gather_hidden_states(hidden_states)

    def _model_forward(
        self,
        num_tokens_padded: int,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ):
        assert self.model is not None
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )
        forward_context = get_forward_context()
        assert forward_context is not None
        if (
            forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
            and not forward_context.capturing
            and not self.use_sparse
        ):
            assert positions is not None
            update_full_graph_params(
                self.attn_backend,
                self.update_stream,
                forward_context,
                num_tokens_padded,
                self.vllm_config,
                self.speculative_config,
                positions.shape[0],
            )
        if get_forward_context().flash_comm_v1_enabled and not isinstance(hidden_states, IntermediateTensors):
            hidden_states = self._all_gather_hidden_states_and_aux(hidden_states)
        return hidden_states

    def _pad_for_sequence_parallelism(self, num_scheduled_tokens: int) -> int:
        # Pad tokens to multiple of tensor_parallel_size when
        # enabled collective fusion for SP
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        if enable_sp(self.vllm_config) or enable_sp_by_pass():
            return round_up(num_scheduled_tokens, tp_size)
        return num_scheduled_tokens

    def _sync_batch_across_dp(
        self,
        num_tokens_padded: int | None = None,
        cudagraph_mode: int = 0,
        allow_dp_padding: bool = False,
        staged_sfa_route_action: StagedSFARouteAction | None = None,
    ) -> tuple[
        bool,
        torch.Tensor | None,
        int,
        StagedSFARouteAction | None,
    ]:
        """
        Coordinates amongst all DP ranks to determine if and how the full batch
        should be split into microbatches.

        Args:
            num_tokens_padded: Number of tokens including any non-DP padding (CUDA graphs,
                TP, etc)
            cudagraph_mode: The cudagraph mode for this rank (0=NONE, 1=PIECEWISE, 2=FULL)
            staged_sfa_route_action: Optional staged-SFA admission verdict for
                this rank, packed into the same DP collective.

        Returns: tuple[
            ubatch_slices: if this is set then all DP ranks have agreed to
            microbatch
            num_tokens_after_padding: A tensor containing the total number of
            tokens per-microbatch for each DP rank including padding. Will be
            padded up to the max value across all DP ranks when allow_dp_padding
            is True.
            synced_cudagraph_mode: The synchronized cudagraph mode (min across ranks)
            synced_staged_sfa_route_action: The strongest staged-SFA verdict
                across ranks, or None when staged SFA is not configured.
        ]

        """

        # TODO: In vLLM, the only thing that needs to be synced is num_tokens, but in
        # our case, we still need to sync the other two flags as well. So we need to
        # include them in the all_reduce operation, and more over, we CANNOT skip it
        # even if we are running in eager mode, which harms performance.
        # FIXME: Restore the `or self.vllm_config.model_config.enforce_eager` here
        # immediately once the other two flags are no longer needed.

        if self.dp_size == 1:
            return False, None, cudagraph_mode, staged_sfa_route_action

        rows = 3 if staged_sfa_route_action is not None else 2
        tensor = self._dp_batch_sync_buffers.get(rows)
        if tensor is None or tensor.shape[1] != self.dp_size:
            tensor = torch.empty(
                rows,
                self.dp_size,
                device="cpu",
                dtype=torch.int32,
            )
            self._dp_batch_sync_buffers[rows] = tensor
        tensor.zero_()
        num_tokens_across_dp = tensor[0]

        if self._skip_all_reduce_across_dp_group():
            num_tokens_across_dp.fill_(num_tokens_padded)
            return (
                False,
                num_tokens_across_dp,
                cudagraph_mode,
                staged_sfa_route_action,
            )

        tensor[0, self.dp_rank] = num_tokens_padded
        tensor[1, self.dp_rank] = cudagraph_mode
        if staged_sfa_route_action is not None:
            tensor[2, self.dp_rank] = _STAGED_SFA_ROUTE_ACTIONS.index(
                staged_sfa_route_action
            )
        dist.all_reduce(tensor, group=get_dp_group().cpu_group)

        max_num_tokens = int(num_tokens_across_dp.max().item())
        synced_route_action = (
            _STAGED_SFA_ROUTE_ACTIONS[int(tensor[2, :].max().item())]
            if staged_sfa_route_action is not None
            else None
        )

        if allow_dp_padding:
            num_tokens_across_dp.fill_(max_num_tokens)

        # Synchronize cudagraph_mode across ranks (take min)
        synced_cudagraph_mode = _post_process_cudagraph_mode(tensor)
        return (
            False,
            num_tokens_across_dp,
            synced_cudagraph_mode,
            synced_route_action,
        )

    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        num_reqs: int,
        num_scheduled_tokens_np: np.ndarray,
        max_num_scheduled_tokens: int,
        use_cascade_attn: bool,
        allow_microbatching: bool = False,
        force_eager: bool = False,
        # For cudagraph capture TODO(lucas): Refactor how we capture cudagraphs (will
        # be improved in model runner v2)
        force_uniform_decode: bool | None = None,
        force_has_lora: bool | None = None,
        force_num_active_loras: int | None = None,
        num_encoder_reqs: int = 0,
        staged_sfa_route_action: StagedSFARouteAction | None = None,
    ) -> tuple[
        CUDAGraphMode,
        BatchDescriptor,
        bool,
        torch.Tensor | None,
        CUDAGraphStat | None,
    ]:
        num_tokens_padded = self._pad_for_sequence_parallelism(num_tokens)
        is_all_decode = np.all(self.input_batch.num_computed_tokens_cpu[:num_reqs] > 0)
        uniform_decode = (
            (
                (is_all_decode if self.speculative_config else True)
                and (max_num_scheduled_tokens == self.uniform_decode_query_len)
                and (num_tokens == max_num_scheduled_tokens * num_reqs)
            )
            if force_uniform_decode is None
            else force_uniform_decode
        )
        # Encoder-decoder models only support CG for decoder_step > 0 (no enc_output
        # is present). Also, chunked-prefill is disabled, so batch are uniform.
        has_encoder_output = self.model_config.is_encoder_decoder and num_encoder_reqs > 0
        num_active_loras = (
            force_num_active_loras
            if force_num_active_loras is not None
            else len(self.input_batch.lora_id_to_lora_request)
        )
        has_lora = num_active_loras > 0 if force_has_lora is None else force_has_lora

        # ruff: noqa: E731
        def dispatch_cudagraph(num_tokens, disable_full=False, valid_modes=None):
            if force_eager:
                return (CUDAGraphMode.NONE, BatchDescriptor(num_tokens_padded))

            return self.cudagraph_dispatcher.dispatch(
                num_tokens=num_tokens,
                has_lora=has_lora,
                uniform_decode=uniform_decode,
                valid_modes=valid_modes,
                invalid_modes={CUDAGraphMode.FULL} if disable_full else None,
                num_active_loras=num_active_loras,
            )

        cudagraph_mode, batch_descriptor = dispatch_cudagraph(num_tokens_padded, use_cascade_attn or has_encoder_output)
        num_tokens_padded = batch_descriptor.num_tokens
        if enable_sp(self.vllm_config):
            assert batch_descriptor.num_tokens % self.vllm_config.parallel_config.tensor_parallel_size == 0, (
                "Sequence parallelism requires num_tokens to be a multiple of tensor parallel size"
            )
        # Extra coordination when running data-parallel since we need to coordinate
        # across ranks
        should_ubatch, num_tokens_across_dp = False, None
        if self.vllm_config.parallel_config.data_parallel_size > 1:
            (
                _,
                num_tokens_across_dp,
                synced_cudagraph_mode,
                staged_sfa_route_action,
            ) = self._sync_batch_across_dp(
                num_tokens_padded=num_tokens_padded,
                cudagraph_mode=cudagraph_mode.value,
                allow_dp_padding=(cudagraph_mode != CUDAGraphMode.NONE) or enable_sp(self.vllm_config),
                staged_sfa_route_action=staged_sfa_route_action,
            )

            # Extract DP padding if there is any
            if num_tokens_across_dp is not None:
                dp_rank = self.parallel_config.data_parallel_rank
                agreed_num_tokens = int(num_tokens_across_dp[dp_rank].item())
                if (
                    agreed_num_tokens != num_tokens_padded
                    or synced_cudagraph_mode != cudagraph_mode.value
                ):
                    num_tokens_padded = agreed_num_tokens
                    cudagraph_mode, batch_descriptor = dispatch_cudagraph(
                        num_tokens_padded,
                        valid_modes={CUDAGraphMode(synced_cudagraph_mode)},
                    )
                    # The synchronized count must identify an exact graph key.
                    assert batch_descriptor.num_tokens == num_tokens_padded
        cudagraph_stats = None
        if self.vllm_config.observability_config.cudagraph_metrics:
            cudagraph_stats = CUDAGraphStat(
                num_unpadded_tokens=num_tokens,
                num_padded_tokens=batch_descriptor.num_tokens,
                num_paddings=batch_descriptor.num_tokens - num_tokens,
                runtime_mode=str(cudagraph_mode),
            )
        self._staged_sfa_dp_route_action = staged_sfa_route_action

        return (
            cudagraph_mode,
            batch_descriptor,
            should_ubatch,
            num_tokens_across_dp,
            cudagraph_stats,
        )

    def _build_attention_metadata(
        self,
        num_tokens: int,
        num_reqs: int,
        max_query_len: int,
        num_tokens_padded: int | None = None,
        num_reqs_padded: int | None = None,
        ubatch_slices: UBatchSlices | None = None,
        logits_indices: torch.Tensor | None = None,
        use_spec_decode: bool = False,
        for_cudagraph_capture: bool = False,
        staged_sfa_graph_dummy_run: bool = False,
        num_scheduled_tokens: dict[str, int] | None = None,
        num_scheduled_tokens_np: np.ndarray | None = None,
        cascade_attn_prefix_lens: list[list[int]] | None = None,
        cold_compact_resumes: tuple[bool, ...] = (),
        resident_remap_frontiers: tuple[int, ...] | None = None,
    ) -> tuple[PerLayerAttnMetadata, CommonAttentionMetadata | None]:
        """
        :return: tuple[attn_metadata, spec_decode_common_attn_metadata]
        """
        # Attention metadata is not needed for attention free models
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return {}, None
        num_tokens_padded = num_tokens_padded or num_tokens
        num_reqs_padded = num_reqs_padded or num_reqs
        attn_metadata: PerLayerAttnMetadata = {}
        if ubatch_slices is not None:
            attn_metadata = [dict() for _ in range(len(ubatch_slices))]
        if for_cudagraph_capture:
            # For some attention backends (e.g. FA) with sliding window models we need
            # to make sure the backend see a max_seq_len that is larger to the sliding
            # window size when capturing to make sure the correct kernel is selected.
            max_seq_len = self.max_model_len
        else:
            max_seq_len = self.seq_lens.np[:num_reqs].max().item()
        if use_spec_decode and self.need_accepted_tokens:
            self.num_accepted_tokens.np[:num_reqs] = self.input_batch.num_accepted_tokens_cpu[:num_reqs]
            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()

        kv_cache_groups = self.kv_cache_config.kv_cache_groups

        def _get_pcp_metadata(block_table_tensor):
            if not self.use_cp:
                return None, block_table_tensor
            return self.pcp_manager.generate_pcp_metadata(
                num_tokens,
                self.query_lens,
                self.input_batch,
                num_scheduled_tokens_np,
                block_table_tensor,
                num_reqs_padded,
                num_reqs,
            )

        def _get_block_table_and_slot_mapping(
            kv_cache_gid: int,
            bank: int = 0,
        ):
            assert num_reqs_padded is not None and num_tokens_padded is not None
            kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
            if self.pcp_size > 1:
                total_num_pcp_pads = sum(self.pcp_manager.num_pcp_pads_cpu[:num_reqs])
                if self.pcp_manager.pcp_use_hybrid_attn:
                    num_scheduled_tokens_padded = self.pcp_manager.num_scheduled_tokens_padded
                    assert num_scheduled_tokens_padded is not None
                    maybe_pcp_full_tokens = sum(num_scheduled_tokens_padded) * self.pcp_size - total_num_pcp_pads
                else:
                    maybe_pcp_full_tokens = num_tokens * self.pcp_size - total_num_pcp_pads
            else:
                maybe_pcp_full_tokens = num_tokens_padded
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                blk_table_tensor = torch.zeros(
                    (num_reqs_padded, 1),
                    dtype=torch.int32,
                    device=self.device,
                )
                slot_mapping = torch.zeros(
                    (num_tokens_padded,),
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                block_tables_by_bank = getattr(
                    self.input_batch,
                    "layerwise_prefill_block_tables",
                    (self.input_batch.block_table,),
                )
                if bank < 0 or bank >= len(block_tables_by_bank):
                    raise RuntimeError(
                        f"Invalid layerwise-prefill physical bank {bank}."
                    )
                blk_table = block_tables_by_bank[bank][kv_cache_gid]
                slot_mapping = blk_table.slot_mapping.gpu[:maybe_pcp_full_tokens]
                maybe_num_reqs_padded = num_reqs_padded * self.decode_token_per_req if self.use_cp else num_reqs_padded
                blk_table_tensor = blk_table.get_device_tensor()[:maybe_num_reqs_padded]

                # Fill unused with -1. Needed for reshape_and_cache in full cuda
                # graph mode. `blk_table_tensor` -1 to match mamba PAD_SLOT_ID
                if self.pcp_size == 1:
                    if num_tokens < num_tokens_padded:
                        slot_mapping[num_tokens:num_tokens_padded].fill_(-1)
                    if num_reqs < num_reqs_padded:
                        blk_table_tensor[num_reqs:num_reqs_padded].fill_(0)
            if self.pcp_size > 1:
                slot_mapping = self.pcp_manager.get_padded_slot_mapping(
                    num_tokens,
                    num_tokens_padded,
                    slot_mapping,
                )
            if self.model_config.enable_return_routed_experts and kv_cache_gid == 0:
                self.cpu_slot_mapping = slot_mapping.cpu().numpy()
            return blk_table_tensor, slot_mapping

        block_table_gid_0, slot_mapping_gid_0 = _get_block_table_and_slot_mapping(0)
        self.long_seq_metadata, block_table_gid_0 = _get_pcp_metadata(block_table_gid_0)

        staged_dummy_prompt_lens = None
        staged_dummy_computed_tokens = None
        staged_dummy_request_ids = None
        resident_state_enabled = self._resident_state_registry is not None
        scheduled = (
            np.asarray(num_scheduled_tokens_np).reshape(-1)
            if staged_sfa_graph_dummy_run or resident_state_enabled
            else None
        )
        if staged_sfa_graph_dummy_run:
            query_width = self.decode_threshold
            assert scheduled is not None
            if (
                num_tokens != num_reqs * query_width
                or num_tokens_padded != num_reqs_padded * query_width
                or num_reqs > num_reqs_padded
                or scheduled.shape != (num_reqs,)
                or not np.all(scheduled == query_width)
            ):
                raise RuntimeError(
                    'The staged SFA graph dummy metadata does not match its '
                    f'fixed query width {query_width}.'
                )
            staged_dummy_prompt_lens = (
                _staged_sfa_dummy_remap_boundaries(
                    self.seq_lens.np[:num_reqs],
                    query_width,
                    self.dsa_index_topk,
                )
            )
            if staged_dummy_prompt_lens.shape != (num_reqs,):
                raise RuntimeError(
                    'The staged SFA graph dummy remap boundary shape differs '
                    f'from num_reqs={num_reqs}: '
                    f'{staged_dummy_prompt_lens.shape}.'
                )
            staged_dummy_computed_tokens = torch.zeros_like(
                self.input_batch.num_computed_tokens_cpu_tensor[
                    :num_reqs_padded
                ]
            )
            staged_dummy_computed_tokens[:num_reqs].copy_(
                torch.from_numpy(staged_dummy_prompt_lens).to(
                    dtype=staged_dummy_computed_tokens.dtype
                )
            )
            staged_dummy_request_ids = self._staged_sfa_dummy_request_ids(
                num_reqs
            )

        resident_compatible = (
            not resident_state_enabled
            or staged_sfa_graph_dummy_run
            or (
                self.attn_state
                in (
                    AscendAttentionState.DecodeOnly,
                    AscendAttentionState.SpecDecoding,
                )
                and num_tokens == num_reqs * self.decode_threshold
                and num_tokens_padded == num_reqs_padded * self.decode_threshold
                and scheduled is not None
                and scheduled.shape == (num_reqs,)
                and np.all(scheduled == self.decode_threshold)
            )
        )
        (
            resident_state_indices,
            resident_state_generations,
            resident_state_indices_cpu,
            resident_state_generations_cpu,
        ) = self._prepare_resident_request_state(
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
            is_dummy=staged_sfa_graph_dummy_run,
            resident_compatible=resident_compatible,
            remap_frontiers=resident_remap_frontiers,
        )

        cm_base = AscendCommonAttentionMetadata(
            query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1],
            query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs_padded + 1],
            seq_lens=self.seq_lens.gpu[:num_reqs_padded],
            # TODO
            seq_lens_cpu=self.seq_lens.cpu[:num_reqs_padded],
            # TODO
            num_computed_tokens_cpu=(
                staged_dummy_computed_tokens
                if staged_dummy_computed_tokens is not None
                else self.input_batch.num_computed_tokens_cpu_tensor[
                    :num_reqs_padded
                ]
            ),
            num_reqs=num_reqs_padded,
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            block_table_tensor=block_table_gid_0,
            slot_mapping=slot_mapping_gid_0,
            causal=True,
            num_input_tokens=num_tokens_padded,
            actual_seq_lengths_q=self.actual_seq_lengths_q,
            positions=self.positions.gpu,
            attn_state=self.attn_state,
            decode_token_per_req=self.decode_token_per_req,
            prefill_context_parallel_metadata=self.long_seq_metadata,
            request_ids=(
                staged_dummy_request_ids
                if staged_dummy_request_ids is not None
                else (
                    self.input_batch.req_ids[:num_reqs]
                    if self.dsa_shrink_latent
                    else None
                )
            ),
            cold_compact_resumes=cold_compact_resumes,
            resident_state_indices=resident_state_indices,
            resident_state_generations=resident_state_generations,
            resident_state_indices_cpu=resident_state_indices_cpu,
            resident_state_generations_cpu=resident_state_generations_cpu,
        )

        if logits_indices is not None and self.cache_config.kv_sharing_fast_prefill:
            cm_base.num_logits_indices = logits_indices.size(0)
            cm_base.logits_indices_padded = self._prepare_kv_sharing_fast_prefill(logits_indices)

        def _build_attn_group_metadata(
            kv_cache_gid: int,
            attn_gid: int,
            common_attn_metadata: CommonAttentionMetadata,
            ubid: int | None = None,
            layer_names: tuple[str, ...] | None = None,
        ) -> None:
            attn_group = self.attn_groups[kv_cache_gid][attn_gid]
            builder = attn_group.get_metadata_builder(ubid or 0)
            cascade_attn_prefix_len = (
                cascade_attn_prefix_lens[kv_cache_gid][attn_gid] if cascade_attn_prefix_lens else 0
            )

            extra_attn_metadata_args = {}
            if use_spec_decode and isinstance(builder, GDNAttentionMetadataBuilder):
                assert ubid is None, "UBatching not supported with GDN yet"
                patch_torch_npu_argsort()
                extra_attn_metadata_args = dict(
                    num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs_padded],
                    num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[:num_reqs_padded],
                )

            if for_cudagraph_capture:
                attn_metadata_i = builder.build_for_cudagraph_capture(common_attn_metadata)
            else:
                attn_metadata_i = builder.build(
                    common_prefix_len=cascade_attn_prefix_len,
                    common_attn_metadata=common_attn_metadata,
                    **extra_attn_metadata_args,
                )
                # NOTE(zxr): Due to the Triton operator does not deal with -1 padding in FullGraph mode,
                # the padding needs to be changed from -1 to 0 to avoid writing invalid mamba block.
                if self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs() \
                    and isinstance(builder, GDNAttentionMetadataBuilder) and attn_metadata_i.num_prefills == 0:
                    if attn_metadata_i.num_decodes == 0 and attn_metadata_i.num_spec_decodes > 0:
                        attn_metadata_i.spec_state_indices_tensor[attn_metadata_i.num_spec_decodes:].fill_(0)

            if ubid is None:
                assert isinstance(attn_metadata, dict)
                attn_metadata_dict = attn_metadata
            else:
                assert isinstance(attn_metadata, list)
                attn_metadata_dict = attn_metadata[ubid]

            for layer_name in layer_names or tuple(attn_group.layer_names):
                attn_metadata_dict[layer_name] = attn_metadata_i

        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        spec_decode_common_attn_metadata = None
        layerwise_request_generations = (
            self.get_layerwise_prefill_request_generations()
            if getattr(self, "layerwise_prefill_p_node", False)
            else ()
        )
        for kv_cache_gid, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups):
            cm = copy(cm_base)  # shallow copy
            # Basically only the encoder seq_lens, block_table and slot_mapping change
            # for each kv_cache_group.
            cm.encoder_seq_lens, cm.encoder_seq_lens_cpu = self._get_encoder_seq_lens(
                num_scheduled_tokens or {},
                kv_cache_group.kv_cache_spec,
                num_reqs_padded,
            )

            # Now, query_start_loc is padded.
            # But gdn needs an unpadded one.
            # gdn_query_start_loc is an unpadded version of query_start_loc.
            # TODO delete it if fia's check is removed.
            if self._has_gdn:
                attn_group = self.attn_groups[kv_cache_gid][0]
                builder = attn_group.get_metadata_builder(0)
                if use_spec_decode and isinstance(builder, GDNAttentionMetadataBuilder):
                    cm.query_start_loc_cpu = self.gdn_query_start_loc.cpu[: num_reqs_padded + 1]
                    cm.query_start_loc = self.gdn_query_start_loc.gpu[: num_reqs_padded + 1]

            if kv_cache_gid > 0:
                cm.block_table_tensor, cm.slot_mapping = _get_block_table_and_slot_mapping(kv_cache_gid)
            elif self.dsa_two_groups and len(self.kv_cache_config.kv_cache_groups) == 2:
                # DSA two-group mode: hand the indexer group's (group 1) table and
                # slots to the latent (SFA) builder; it mirrors them into its
                # metadata so the impl can read/write the indexer cache, which now
                # has its own block ids.
                cm.indexer_block_table_tensor, cm.indexer_slot_mapping = _get_block_table_and_slot_mapping(1)
                if self.dsa_shrink_latent:
                    # B2 compact-scratch decode: hand per-request prompt lengths
                    # (CPU) to the SFA builder, which expands them to per-ROW
                    # values (decode rows -> plen, prefill/padding rows -> 0 =
                    # no remap) and ships them to device.
                    plens_np = (
                        staged_dummy_prompt_lens.copy()
                        if staged_dummy_prompt_lens is not None
                        else self.input_batch.num_prompt_tokens[:num_reqs]
                    )
                    cm.prompt_lens_cpu = plens_np
            if self.speculative_config and spec_decode_common_attn_metadata is None:
                if isinstance(self.drafter, AscendEagleProposer | AscendDraftModelProposer):
                    draft_layer_name = self.drafter.attn_layer_names[0]
                    if draft_layer_name in kv_cache_group.layer_names:
                        if getattr(self, "layerwise_prefill_p_node", False):
                            spec_decode_common_attn_metadata = (
                                self._layerwise_prefill_draft_common_attn_metadata(
                                    cm,
                                    draft_layer_name,
                                    layerwise_request_generations,
                                    _get_block_table_and_slot_mapping,
                                )
                            )
                        else:
                            spec_decode_common_attn_metadata = cm
                else:
                    spec_decode_common_attn_metadata = cm

            for attn_gid in range(len(self.attn_groups[kv_cache_gid])):
                if not getattr(self, "layerwise_prefill_p_node", False):
                    _build_attn_group_metadata(kv_cache_gid, attn_gid, cm)
                    continue

                attn_group = self.attn_groups[kv_cache_gid][attn_gid]
                if kv_cache_gid != 0:
                    _build_attn_group_metadata(kv_cache_gid, attn_gid, cm)
                    continue
                for layer_name in attn_group.layer_names:
                    row = self.dsa_kv_rows_by_layer_name.get(layer_name)
                    if row is None or row.kv_group != 0:
                        raise RuntimeError(
                            "Layerwise-prefill attention layer is absent from "
                            f"the canonical LATENT topology: {layer_name!r}."
                        )
                    execution = self.dsa_kv_executions_by_ordinal.get(
                        row.execution_ordinal
                    )
                    if execution is None or execution.latent != row:
                        raise RuntimeError(
                            "Layerwise-prefill attention layer has no canonical "
                            f"execution: {layer_name!r}."
                        )
                    layer_cm = self._layerwise_prefill_common_attn_metadata(
                        cm,
                        execution,
                        layerwise_request_generations,
                        _get_block_table_and_slot_mapping,
                    )
                    _build_attn_group_metadata(
                        kv_cache_gid,
                        attn_gid,
                        layer_cm,
                        layer_names=(layer_name,),
                    )
        if self.is_mm_prefix_lm:
            req_doc_ranges = {}
            for req_id in self.input_batch.req_ids:
                image_doc_ranges = []
                req_state = self.requests[req_id]
                for mm_feature in req_state.mm_features:
                    pos_info = mm_feature.mm_position
                    img_doc_range = pos_info.extract_embeds_range()
                    image_doc_ranges.extend(img_doc_range)
                req_idx = self.input_batch.req_id_to_index[req_id]
                req_doc_ranges[req_idx] = image_doc_ranges

            if isinstance(attn_metadata, list):
                for ub_metadata in attn_metadata:
                    for _metadata in ub_metadata.values():
                        _metadata.mm_prefix_range = req_doc_ranges  # type: ignore[attr-defined]
            else:
                for _metadata in attn_metadata.values():
                    _metadata.mm_prefix_range = req_doc_ranges  # type: ignore[attr-defined]

        if spec_decode_common_attn_metadata is not None and (
            num_reqs != num_reqs_padded or num_tokens != num_tokens_padded
        ):
            # Currently the drafter still only uses piecewise cudagraphs (and modifies
            # the attention metadata in directly), and therefore does not want to use
            # padded attention metadata.
            spec_decode_common_attn_metadata = spec_decode_common_attn_metadata.unpadded(num_tokens, num_reqs)
        return attn_metadata, spec_decode_common_attn_metadata

    def _should_build_dummy_attn_metadata(
        self,
        force_attention: bool = False,
        is_profile: bool = False,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
    ) -> bool:
        """
        Determine whether attention metadata should be built during dummy_run.
        SubClass can override this to add custom conditions.
        """
        # If force_attention is True, we always capture attention, Otherwise,
        # it only happens for cudagraph_runtime_mode=FULL.
        return force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL

    def _staged_sfa_dummy_batch_size(
        self,
        *,
        is_profile: bool,
        cudagraph_runtime_mode: CUDAGraphMode,
        allow_eager: bool,
        num_active_loras: int,
        num_tokens_unpadded: int,
        num_tokens_padded: int,
        num_reqs: int,
        num_scheduled_tokens: np.ndarray,
        batch_descriptor: BatchDescriptor,
        dp_route_action: StagedSFARouteAction | None,
    ) -> int | None:
        '''Return the fixed token capacity for a staged decode dummy batch.'''
        query_width = 1 + int(
            getattr(self.speculative_config, "num_speculative_tokens", 0)
        )
        if (
            not self._staged_sfa_graph_capture_sizes
            or is_profile
            or cudagraph_runtime_mode
            not in (CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE)
            or (
                cudagraph_runtime_mode == CUDAGraphMode.NONE
                and not allow_eager
            )
            or num_active_loras != 0
            or dp_route_action != StagedSFARouteAction.STAGED
        ):
            return None

        batch_size = int(num_tokens_padded)
        if (
            batch_size not in self._staged_sfa_graph_capture_sizes
            or batch_descriptor != BatchDescriptor(num_tokens=batch_size)
            or num_tokens_unpadded <= 0
            or num_tokens_unpadded != num_reqs * query_width
            or num_reqs * query_width > batch_size
        ):
            return None

        scheduled = np.asarray(num_scheduled_tokens).reshape(-1)
        if (
            scheduled.shape != (num_reqs,)
            or not np.all(scheduled == query_width)
        ):
            return None
        return batch_size

    def _staged_sfa_local_route(
        self,
        *,
        num_tokens_unpadded: int,
        num_reqs: int,
        num_scheduled_tokens: np.ndarray,
        index_topk: int,
        has_cascade_attention: bool,
        request_ids: Any,
        kv_connector_metadata: Any,
        num_computed_tokens: Any = None,
        prompt_lens: Any = None,
    ) -> StagedSFARouteDecision:
        """Classify local scheduler/connector state before DP coordination."""
        def native(reason):
            return StagedSFARouteDecision(
                StagedSFARouteAction.SAFE_NATIVE,
                reason,
            )
        if not self._staged_sfa_graph_capture_sizes:
            return native(StagedSFARouteReason.NOT_CONFIGURED)
        query_width = 1 + int(
            getattr(self.speculative_config, "num_speculative_tokens", 0)
        )
        expected_state = (
            AscendAttentionState.DecodeOnly
            if query_width == 1
            else AscendAttentionState.SpecDecoding
        )
        if self.attn_state != expected_state:
            return native(StagedSFARouteReason.NOT_DECODE)
        metadata_reason, frontiers, cold_resumes = (
            staged_sfa_metadata_sparse_route(
                kv_connector_metadata,
                request_ids,
            )
        )
        if metadata_reason not in (
            StagedSFARouteReason.ELIGIBLE,
            StagedSFARouteReason.DENSE_PREFIX_HIT,
            StagedSFARouteReason.MIXED_CONNECTOR_LOAD,
        ):
            return StagedSFARouteDecision(
                StagedSFARouteAction.FATAL,
                metadata_reason,
            )
        if getattr(self, "calculate_kv_scales", False):
            return native(StagedSFARouteReason.RUNTIME_MODE)
        if getattr(self.vllm_config, "lora_config", None) is not None:
            return native(StagedSFARouteReason.LORA)
        if has_cascade_attention:
            return native(StagedSFARouteReason.CASCADE)
        batch_size = int(num_tokens_unpadded)
        capture_sizes = self._staged_sfa_graph_capture_sizes
        if (
            batch_size <= 0
            or not capture_sizes
            or batch_size > capture_sizes[-1]
            or num_reqs * query_width != batch_size
        ):
            return native(StagedSFARouteReason.UNSUPPORTED_BATCH)
        scheduled = np.asarray(num_scheduled_tokens).reshape(-1)
        if scheduled.shape != (num_reqs,) or not np.all(
            scheduled == query_width
        ):
            return native(StagedSFARouteReason.NON_Q1)
        if len(frontiers) != num_reqs:
            return StagedSFARouteDecision(
                StagedSFARouteAction.FATAL,
                StagedSFARouteReason.FRONTIER_COUNT_MISMATCH,
            )
        if query_width == 1:
            if any(cold_resumes):
                computed = np.asarray(num_computed_tokens).reshape(-1)
                prompts = np.asarray(prompt_lens).reshape(-1)
                if (
                    len(cold_resumes) != num_reqs
                    or computed.shape != (num_reqs,)
                    or prompts.shape != (num_reqs,)
                    or any(
                        int(computed[i]) != int(prompts[i]) - 1
                        or frontiers[i] != int(computed[i])
                        for i, resume in enumerate(cold_resumes)
                        if resume
                    )
                ):
                    return native(StagedSFARouteReason.COLD_COMPACT_LAYOUT)
        else:
            cold_resumes = ()
        scratch_capacity = query_width * index_topk
        if any(
            frontier != 0 and frontier < scratch_capacity
            for frontier in frontiers
        ):
            return StagedSFARouteDecision(
                StagedSFARouteAction.FATAL,
                StagedSFARouteReason.FRONTIER_TOO_SHORT,
            )
        return StagedSFARouteDecision(
            StagedSFARouteAction.STAGED,
            metadata_reason,
            frontiers=frontiers,
            cold_compact_resumes=cold_resumes,
        )

    def _synchronize_staged_sfa_capture_unsafe_loads(self) -> None:
        """Keep background cold loads out of serving-time graph capture."""
        if not has_kv_transfer_group():
            return
        connector = get_kv_transfer_group()
        synchronize = getattr(
            connector,
            "synchronize_staged_sfa_capture_unsafe_loads",
            None,
        )
        local_error: BaseException | None = None
        try:
            if callable(synchronize):
                synchronize()
            elif bool(
                getattr(
                    connector,
                    "supports_dsa_compact_external_load",
                    False,
                )
            ):
                local_error = RuntimeError(
                    "The staged SFA native fallback requires an LMCache "
                    "connector with a capture-unsafe load barrier. Update "
                    "LMCache before enabling asynchronous DSA cold compact "
                    "loads."
                )
        except BaseException as exc:
            local_error = exc

        cpu_groups = []
        tp_group = get_tp_group()
        if tp_group.world_size > 1:
            cpu_groups.append(tp_group.cpu_group)
        # Internal-DP replicas have independent scheduler and connector flow;
        # idle replicas in _dummy_run do not enter this conditional barrier.
        if not cpu_groups:
            if local_error is not None:
                raise RuntimeError(
                    "The staged SFA capture-unsafe load barrier failed"
                ) from local_error
            return

        failure = torch.tensor(
            [int(local_error is not None)],
            dtype=torch.int32,
        )
        for cpu_group in cpu_groups:
            dist.all_reduce(failure, op=dist.ReduceOp.MAX, group=cpu_group)
        if int(failure.item()) != 0:
            if local_error is not None:
                raise RuntimeError(
                    "The staged SFA capture-unsafe load barrier failed "
                    "on this worker"
                ) from local_error
            raise RuntimeError(
                "The staged SFA capture-unsafe load barrier failed on a peer "
                "worker"
            )

    def _staged_sfa_live_route(
        self,
        *,
        local_route: StagedSFARouteDecision,
        dp_route_action: StagedSFARouteAction | None,
        cudagraph_mode: CUDAGraphMode,
        batch_descriptor: BatchDescriptor,
        num_tokens_unpadded: int,
        num_tokens_padded: int,
        num_reqs: int,
        should_ubatch: bool,
    ) -> StagedSFARouteDecision:
        """Bind a DP-agreed local route to one captured graph capacity."""
        if (
            dp_route_action is not None
            and dp_route_action != StagedSFARouteAction.STAGED
        ):
            if local_route.action == dp_route_action:
                return local_route
            return StagedSFARouteDecision(
                dp_route_action,
                StagedSFARouteReason.RUNTIME_PARALLELISM,
            )
        if local_route.action != StagedSFARouteAction.STAGED:
            return local_route
        if cudagraph_mode != CUDAGraphMode.PIECEWISE:
            return StagedSFARouteDecision(
                StagedSFARouteAction.SAFE_NATIVE,
                StagedSFARouteReason.RUNTIME_MODE,
            )
        if should_ubatch:
            return StagedSFARouteDecision(
                StagedSFARouteAction.SAFE_NATIVE,
                StagedSFARouteReason.UBATCH,
            )
        batch_size = int(num_tokens_unpadded)
        capacity = int(num_tokens_padded)
        query_width = self.decode_threshold
        if capacity % query_width:
            return StagedSFARouteDecision(
                StagedSFARouteAction.SAFE_NATIVE,
                StagedSFARouteReason.PADDED_BATCH,
            )
        graph_key = (
            StagedSFAGraphKey.exact_q1(capacity)
            if query_width == 1
            else StagedSFAGraphKey.fixed_spec(
                capacity // query_width,
                query_width,
            )
        )
        if (
            batch_size <= 0
            or batch_size != num_reqs * query_width
            or batch_size > capacity
            or capacity
            not in self._staged_sfa_graph_capture_sizes
        ):
            return StagedSFARouteDecision(
                StagedSFARouteAction.SAFE_NATIVE,
                StagedSFARouteReason.PADDED_BATCH,
            )
        if batch_descriptor != graph_key.to_legacy_batch_descriptor():
            return StagedSFARouteDecision(
                StagedSFARouteAction.SAFE_NATIVE,
                StagedSFARouteReason.BATCH_DESCRIPTOR,
            )
        return StagedSFARouteDecision(
            StagedSFARouteAction.STAGED,
            StagedSFARouteReason.ELIGIBLE,
            graph_key=graph_key,
            frontiers=local_route.frontiers,
            cold_compact_resumes=local_route.cold_compact_resumes,
        )

    def _apply_staged_sfa_route(
        self,
        route: StagedSFARouteDecision,
    ) -> StagedSFAGraphKey | None:
        if route.action == StagedSFARouteAction.STAGED:
            return route.graph_key
        message = (
            f"[SFA_ROUTE] action={route.action.value} "
            f"reason={route.reason.value}"
        )
        if route.action in (
            StagedSFARouteAction.RECOMPUTE,
            StagedSFARouteAction.FATAL,
        ):
            raise RuntimeError(message)
        logged = getattr(self, "_staged_sfa_logged_routes", None)
        if logged is None:
            logged = self._staged_sfa_logged_routes = set()
        if route.reason not in logged and route.reason != StagedSFARouteReason.NOT_CONFIGURED:
            logged.add(route.reason)
            logger.info(message)
        return None

    @staticmethod
    def _staged_sfa_query_start_locs(
        request_count: int,
        *,
        query_width: int,
        dtype: np.dtype,
    ) -> np.ndarray:
        if request_count <= 0 or query_width <= 0:
            raise ValueError(
                'The staged SFA request count and query width must be positive.'
            )
        return (
            np.arange(request_count + 1, dtype=dtype)
            * query_width
        )

    @staticmethod
    def _staged_sfa_q1_query_start_locs(
        batch_size: int,
        *,
        dtype: np.dtype,
    ) -> np.ndarray:
        return NPUModelRunner._staged_sfa_query_start_locs(
            batch_size,
            query_width=1,
            dtype=dtype,
        )

    @staticmethod
    def _staged_sfa_dummy_request_ids(batch_size: int) -> list[str]:
        if batch_size <= 0:
            raise ValueError('The staged SFA dummy batch size must be positive.')
        return [
            f'staged-sfa-graph-dummy-{request_index}'
            for request_index in range(batch_size)
        ]

    def _prepare_staged_sfa_dummy_block_tables(
        self,
        *,
        batch_size: int,
        positions: np.ndarray,
    ) -> None:
        '''Give every staged dummy token a non-aliasing physical slot.'''
        positions = np.asarray(positions, dtype=np.int64).reshape(-1)
        if (
            batch_size <= 0
            or positions.size % batch_size
            or np.any(positions < 0)
        ):
            raise RuntimeError(
                'Staged SFA dummy positions must contain a fixed positive '
                'number of non-negative positions per request, got '
                f'batch_size={batch_size}, shape={positions.shape}.'
            )
        query_width = positions.size // batch_size

        block_tables = getattr(
            self.input_batch.block_table,
            'block_tables',
            None,
        )
        if block_tables is None or len(block_tables) != 2:
            raise RuntimeError(
                'The staged SFA dummy batch requires exactly two KV block '
                'tables (latent and indexer).'
            )

        configured_blocks_per_group = getattr(
            self.kv_cache_config,
            'num_blocks_per_group',
            None,
        )
        if configured_blocks_per_group is None:
            available_blocks_per_group = (
                getattr(self.kv_cache_config, 'num_blocks', None),
            ) * len(block_tables)
        else:
            try:
                available_blocks_per_group = tuple(
                    configured_blocks_per_group
                )
            except TypeError as exc:
                raise RuntimeError(
                    'The staged SFA dummy batch requires one KV block count '
                    'per group.'
                ) from exc
            if len(available_blocks_per_group) != len(block_tables):
                raise RuntimeError(
                    'The staged SFA dummy batch requires one KV block count '
                    f'per group: groups={len(block_tables)}, '
                    f'counts={len(available_blocks_per_group)}.'
                )

        req_indices = np.repeat(
            np.arange(batch_size, dtype=np.int32),
            query_width,
        )
        max_position = int(positions.max()) if positions.size else -1
        for group_index, block_table in enumerate(block_tables):
            available_blocks = available_blocks_per_group[group_index]
            if (
                not isinstance(available_blocks, (int, np.integer))
                or int(available_blocks) < batch_size
            ):
                raise RuntimeError(
                    'The staged SFA dummy batch requires one physical block '
                    f'per request in KV group {group_index}: '
                    f'batch_size={batch_size}, '
                    f'available_blocks={available_blocks!r}.'
                )
            if (
                int(block_table.max_num_reqs) < batch_size
                or int(block_table.max_num_blocks_per_req) <= 0
            ):
                raise RuntimeError(
                    'The staged SFA dummy block-table capacity is too small '
                    f'for group {group_index}: batch_size={batch_size}, '
                    f'max_num_reqs={block_table.max_num_reqs}, '
                    'max_num_blocks_per_req='
                    f'{block_table.max_num_blocks_per_req}.'
                )

            logical_table_width = int(
                block_table.block_table.np.shape[1]
            )
            logical_block_size = int(block_table.block_size)
            cp_world_size = max(
                1,
                int(getattr(block_table, 'dcp_world_size', 1))
                * int(getattr(block_table, 'pcp_world_size', 1)),
            )
            logical_capacity = (
                logical_table_width
                * logical_block_size
                * cp_world_size
            )
            if max_position >= logical_capacity:
                raise RuntimeError(
                    'The staged SFA dummy position exceeds the logical '
                    f'block-table capacity for KV group {group_index}: '
                    f'max_position={max_position}, '
                    f'logical_capacity={logical_capacity}, '
                    f'table_width={logical_table_width}, '
                    f'logical_block_size={logical_block_size}, '
                    f'cp_world_size={cp_world_size}.'
                )

            physical_blocks = np.arange(
                batch_size,
                dtype=block_table.block_table.np.dtype,
            ).reshape(-1, 1)
            block_table.block_table.np[:batch_size, :] = physical_blocks
            block_table.num_blocks_per_row[:batch_size] = (
                block_table.max_num_blocks_per_req
            )
            block_table.commit_block_table(batch_size, force=True)
            block_table.compute_slot_mapping(req_indices, positions)
            block_table.commit_slot_mapping(positions.size)

            slots = block_table.slot_mapping.np[: positions.size]
            if (
                np.any(slots < 0)
                or np.unique(slots).size != positions.size
            ):
                raise RuntimeError(
                    'The staged SFA dummy slot mapping aliases requests in '
                    f'KV group {group_index}: slots={slots.tolist()}.'
                )
    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        remove_lora: bool = True,
        is_graph_capturing: bool = False,
        num_active_loras: int = 0,
        profile_seq_lens: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # only support eager mode and piecewise graph now
        assert cudagraph_runtime_mode is None or cudagraph_runtime_mode.valid_runtime_modes()
        # If cudagraph_mode.decode_mode() == FULL and
        # cudagraph_mode.separate_routine(). This means that we are using
        # different graphs and/or modes for mixed prefill-decode batches vs.
        # uniform decode batches. A uniform decode batch means that all
        # requests have identical query length, except a potential virtual
        # request (shorter) in the batch account for padding.
        # Uniform decode batch could either be common pure decode, where
        # max_query_len == 1, or speculative decode, where
        # max_query_len == 1 + num_spec_decode_tokens.

        # When setting max_query_len = 1, we switch to and capture the optimized
        # routine of FA2 for pure decode, i.e., Flashdecode + an optimization
        # for GQA/MQA.
        staged_capture_candidate = (
            not uniform_decode
            and skip_eplb
            and not remove_lora
            and bool(self._staged_sfa_graph_capture_sizes)
        )
        max_query_len = (
            self.decode_threshold
            if staged_capture_candidate
            else self.uniform_decode_query_len
            if uniform_decode
            else num_tokens
        )
        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        if create_mixed_batch:
            raise NotImplementedError("create_mixed_batch is used for warmup deepgemm, vllm-ascend does not need it")
        elif uniform_decode or staged_capture_candidate:
            assert not create_mixed_batch
            num_reqs = min(max_num_reqs, cdiv(num_tokens, max_query_len))
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        else:
            num_reqs = min(num_tokens, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs

        if not is_profile and self.dynamic_eplb:
            self.eplb_updator.forward_before()

        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
        self.query_lens = torch.from_numpy(num_scheduled_tokens)
        num_tokens_unpadded = int(num_scheduled_tokens.sum())
        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
        staged_capture = staged_capture_candidate
        dp_idle = uniform_decode and cudagraph_runtime_mode is None and self.dp_size > 1
        dummy_route_action = (
            StagedSFARouteAction.STAGED
            if self._staged_sfa_graph_capture_sizes
            and not is_profile
            and (staged_capture or dp_idle)
            else None
        )
        (
            _cudagraph_mode,
            batch_desc,
            _,
            num_tokens_across_dp,
            _,
        ) = self._determine_batch_execution_and_padding(
            num_tokens=num_tokens_unpadded,
            num_reqs=num_reqs,
            num_scheduled_tokens_np=num_scheduled_tokens,
            max_num_scheduled_tokens=max_query_len,
            use_cascade_attn=False,
            allow_microbatching=allow_microbatching,
            force_eager=is_profile or (cudagraph_runtime_mode == CUDAGraphMode.NONE),
            # `force_uniform_decode` is used for cudagraph capture; because for
            # capturing mixed prefill-decode batches, we sometimes use
            # num_tokens == num_reqs which looks like a uniform decode batch to the
            # dispatcher; but we actually want to capture a piecewise cudagraph
            force_uniform_decode=uniform_decode,
            # `force_has_lora` is used for cudagraph capture; because LoRA is
            # activated later in the context manager, but we need to know the
            # LoRA state when determining the batch descriptor for capture
            force_has_lora=num_active_loras > 0,
            force_num_active_loras=num_active_loras,
            staged_sfa_route_action=dummy_route_action,
        )
        dp_route_action = self._staged_sfa_dp_route_action
        if self.use_cp:
            self.pcp_manager.init_batch_info(
                num_scheduled_tokens,
                num_reqs,
            )
            if self.speculative_config:
                self.pcp_manager.query_lens_pcp_full.cpu[:num_reqs] = torch.from_numpy(num_scheduled_tokens)
                self.pcp_manager.query_lens_pcp_full.copy_to_gpu()
        if cudagraph_runtime_mode is None:
            cudagraph_runtime_mode = _cudagraph_mode
        else:
            assert cudagraph_runtime_mode == _cudagraph_mode, (
                f"Cudagraph runtime mode mismatch in dummy_run. "
                f"Expected {_cudagraph_mode}, but got {cudagraph_runtime_mode}."
            )
        num_tokens_padded = batch_desc.num_tokens
        num_reqs_padded = batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
        if num_tokens_across_dp is not None and num_tokens_padded != num_tokens:
            # pad is needed if the pad of `num_tokens` is triggered inside CudagraphDispatcher
            num_tokens_across_dp[:] = num_tokens_padded
            num_scheduled_tokens = num_scheduled_tokens.repeat(num_reqs_padded)
        # vllm-ascend does not support ubatch now
        ubatch_slices, ubatch_slices_padded = None, None
        attn_metadata: PerLayerAttnMetadata | None = None
        staged_sfa_dummy_batch_size = self._staged_sfa_dummy_batch_size(
            is_profile=is_profile,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            allow_eager=staged_capture,
            num_active_loras=num_active_loras,
            num_tokens_unpadded=num_tokens_unpadded,
            num_tokens_padded=num_tokens_padded,
            num_reqs=num_reqs,
            num_scheduled_tokens=num_scheduled_tokens,
            batch_descriptor=batch_desc,
            dp_route_action=dp_route_action,
        )
        staged_sfa_graph_dummy_run = staged_sfa_dummy_batch_size is not None
        if (
            dummy_route_action is not None
            and not staged_sfa_graph_dummy_run
        ):
            fallback_action = (
                dp_route_action
                if dp_route_action != StagedSFARouteAction.STAGED
                else StagedSFARouteAction.SAFE_NATIVE
            )
            self._apply_staged_sfa_route(
                StagedSFARouteDecision(
                    fallback_action,
                    StagedSFARouteReason.RUNTIME_PARALLELISM,
                )
            )
            cudagraph_runtime_mode = CUDAGraphMode.NONE
        # Build attention metadata for dummy_run
        if (
            self._should_build_dummy_attn_metadata(
                force_attention,
                is_profile,
                cudagraph_runtime_mode,
            )
            or staged_sfa_graph_dummy_run
        ):
            if create_mixed_batch:
                raise NotImplementedError(
                    "create_mixed_batch is used for warmup deepgemm, vllm-ascend does not need it"
                )
            self.attn_state = AscendAttentionState.DecodeOnly
            if self.speculative_config and self.speculative_config.method == "mtp":
                # `AscendAttentionState.SpecDecoding` is only designed for mla
                if self.vllm_config.model_config.use_mla:
                    self.attn_state = AscendAttentionState.SpecDecoding
                else:
                    self.attn_state = AscendAttentionState.ChunkedPrefill
            # The reason why we use a fixed seq_len rather than max_query_len is that
            # _npu_paged_attention_get_workspace only returns max workspace with specific
            # seq_lens. We use this seq_len only when capturing graph, and still use max_query_len
            # in inference. This will be removed once npu_fused_infer_attention_score
            # outperforms _npu_paged_attention on all cases.
            # The staged SFA POC reuses 6144 only as bounded dummy data. Its
            # indexer/SFA capacity is fixed by the max-model-length block-table
            # width, while seq_lens remains a live tensor input during replay.
            # That makes changing lengths plausible, but the torch_npu
            # lightning-indexer branch still requires live numerical parity.
            if profile_seq_lens is not None:
                seq_lens = profile_seq_lens
            else:
                seq_lens = (
                    SEQ_LEN_WITH_MAX_PA_WORKSPACE
                    if staged_sfa_graph_dummy_run
                    or (
                        is_graph_capturing
                        and using_paged_attention(
                            num_tokens,
                            self.vllm_config,
                        )
                    )
                    else max_query_len
                )  # type: ignore[assignment]
            self.seq_lens.np[:num_reqs_padded] = seq_lens
            self.seq_lens.np[num_reqs_padded:] = 0
            self.seq_lens.copy_to_gpu()

            if staged_sfa_graph_dummy_run:
                self._prepare_staged_sfa_dummy_block_tables(
                    batch_size=num_reqs,
                    positions=(
                        self.seq_lens.np[:num_reqs]
                        .astype(np.int64)
                        .reshape(-1, 1)
                        - self.decode_threshold
                        + np.arange(
                            self.decode_threshold,
                            dtype=np.int64,
                        ).reshape(1, -1)
                    ).reshape(-1),
                )
                query_start_locs = self._staged_sfa_query_start_locs(
                    num_reqs,
                    query_width=self.decode_threshold,
                    dtype=self.query_start_loc.np.dtype,
                )
                self.query_start_loc.np[
                    : num_reqs + 1
                ] = query_start_locs
            else:
                cum_num_tokens, _ = self._get_cumsum_and_arange(
                    num_scheduled_tokens
                )
                self.query_start_loc.np[
                    1 : num_reqs_padded + 1
                ] = cum_num_tokens
            self.query_start_loc.copy_to_gpu()
            num_reqs_padded = self._pad_query_start_loc_for_fia(
                num_tokens_padded,
                num_reqs_padded,
                num_reqs,
                cudagraph_runtime_mode,
                (
                    staged_sfa_dummy_batch_size // self.decode_threshold
                    if staged_sfa_graph_dummy_run
                    else batch_desc.num_reqs
                ),
            )

            pad_attn = cudagraph_runtime_mode == CUDAGraphMode.FULL
            attn_metadata, _ = self._build_attention_metadata(
                num_tokens=num_tokens_unpadded,
                num_tokens_padded=num_tokens_padded,
                num_reqs=num_reqs,
                num_reqs_padded=num_reqs_padded,
                max_query_len=(
                    self.decode_threshold
                    if staged_sfa_graph_dummy_run
                    else max_query_len
                ),
                ubatch_slices=ubatch_slices_padded if pad_attn else ubatch_slices,
                for_cudagraph_capture=is_graph_capturing,
                num_scheduled_tokens_np=num_scheduled_tokens,
                staged_sfa_graph_dummy_run=staged_sfa_graph_dummy_run,
            )

        with self.maybe_dummy_run_with_lora(
            self.lora_config,
            num_scheduled_tokens,
            num_sampled_tokens,
            remove_lora,
            # TODO: The next line is a temporary workaround
            # to fix the accuracy issue of test_llama32_lora.py,
            # which is introduced by vllm-project/vllm#32005
            num_active_loras=(self.lora_config.max_loras if self.lora_config is not None else num_active_loras),
        ):
            # Make sure padding doesn't exceed max_num_tokens
            assert num_tokens_padded <= self.max_num_tokens
            if self.is_multimodal_model and not self.model_config.is_encoder_decoder or self.enable_prompt_embeds:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
            else:
                input_ids = self.input_ids.gpu[:num_tokens_padded]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions.gpu[:, :num_tokens_padded]
            elif self.uses_xdrope_dim > 0:
                positions = self.xdrope_positions.gpu[:, :num_tokens_padded]
            else:
                positions = self.positions.gpu[:num_tokens_padded]

            # update global cos, sin
            update_cos_sin(positions)

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                # When PP and flashcomm1 are enabled, during dummy_run the estimated space should divide num_tokens by
                # tp_size; otherwise, on non-first PP ranks it would effectively perform an extra all-gather, leading
                # to incorrect memory estimation and potentially causing OOM.
                intermediate_tokens = num_tokens_padded
                if enable_sp():
                    tp_size = get_tensor_model_parallel_world_size()
                    intermediate_tokens = (num_tokens_padded + tp_size - 1) // tp_size
                if self.intermediate_tensors is None:
                    max_actual_tokens = self.max_num_tokens
                    if enable_sp():
                        max_actual_tokens = (self.max_num_tokens + tp_size - 1) // tp_size
                    self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                        batch_size=max_actual_tokens, dtype=self.dtype, device=self.device
                    )
                intermediate_tensors = IntermediateTensors(
                    {k: v[:intermediate_tokens] for k, v in self.intermediate_tensors.items()}
                )

            need_dummy_logits = not is_profile and lmhead_tp_enable()
            max_num_reqs_across_dp = max_num_reqs * self.uniform_decode_query_len
            dummy_indices = torch.zeros(max_num_reqs_across_dp, dtype=torch.int32)

            def dummy_compute_logits(hidden_states):
                if not need_dummy_logits:
                    return None
                return self.model.compute_logits(hidden_states[dummy_indices])

            def dummy_drafter_compute_logits(hidden_states):
                if not need_dummy_logits or self.drafter is None:
                    return
                if hasattr(self.drafter, "model") and hasattr(self.drafter.model, "compute_logits"):
                    return self.drafter.model.compute_logits(hidden_states[dummy_indices])

            staged_dummy_key = None
            if staged_sfa_dummy_batch_size is not None:
                staged_dummy_key = (
                    StagedSFAGraphKey.exact_q1(staged_sfa_dummy_batch_size)
                    if self.decode_threshold == 1
                    else StagedSFAGraphKey.fixed_spec(
                        staged_sfa_dummy_batch_size // self.decode_threshold,
                        self.decode_threshold,
                    )
                )
            with set_ascend_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                in_profile_run=is_profile,
                num_actual_tokens=num_tokens_padded,
                aclgraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=batch_desc,
                model_instance=self.model,
                dsa_offload_manager=getattr(self, "dsa_offload_manager", None),
                dsa_adapter_cache=getattr(self, "dsa_adapter_cache", None),
                staged_sfa_graph_dummy_run=staged_sfa_graph_dummy_run,
                staged_sfa_route=(
                    StagedSFARouteDecision(
                        StagedSFARouteAction.STAGED,
                        StagedSFARouteReason.ELIGIBLE,
                        staged_dummy_key,
                    )
                    if staged_dummy_key is not None
                    else None
                ),
                staged_sfa_graph_key=staged_dummy_key,
            ):
                if staged_dummy_key is not None and self._staged_sfa_impls:
                    first_layer_name, first_impl = self._staged_sfa_impls[0]
                    first_impl.bootstrap_cross_layer(first_layer_name)
                outputs = self._model_forward(
                    num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds
                )
            if self.use_aux_hidden_state_outputs:
                hidden_states, _ = outputs
            else:
                hidden_states = outputs
            dummy_compute_logits(hidden_states)

            if self.drafter:
                draft_num_tokens = num_tokens_padded
                draft_num_reqs = num_reqs_padded
                draft_runtime_mode = cudagraph_runtime_mode
                draft_batch_descriptor = batch_desc
                staged_mtp_draft_graph = bool(
                    staged_dummy_key is not None
                    and getattr(
                        self.drafter,
                        "use_staged_mtp_draft_graph",
                        False,
                    )
                )
                if staged_mtp_draft_graph:
                    draft_num_tokens = staged_dummy_key.request_capacity
                    draft_num_reqs = staged_dummy_key.request_capacity
                    draft_batch_descriptor = BatchDescriptor(
                        num_tokens=draft_num_tokens,
                    )
                    draft_runtime_mode = (
                        CUDAGraphMode.FULL
                        if cudagraph_runtime_mode
                        == CUDAGraphMode.PIECEWISE
                        else CUDAGraphMode.NONE
                    )
                self.drafter.dummy_run(
                    num_tokens=draft_num_tokens,
                    with_prefill=with_prefill,
                    num_reqs=draft_num_reqs,
                    num_tokens_across_dp=num_tokens_across_dp,
                    aclgraph_runtime_mode=draft_runtime_mode,
                    batch_descriptor=draft_batch_descriptor,
                    dummy_compute_logits=dummy_drafter_compute_logits,
                    in_graph_capturing=not force_attention,
                    is_profile=is_profile,
                    **(
                        {
                            "staged_mtp_draft_graph":
                            staged_mtp_draft_graph,
                        }
                        if hasattr(
                            self.drafter,
                            "use_staged_mtp_draft_graph",
                        )
                        else {}
                    ),
                )
            if is_profile and self.dynamic_eplb:
                self.model.clear_all_moe_loads()
            if self.dynamic_eplb:
                self.eplb_updator.forward_end()
            return hidden_states, hidden_states

    @torch.inference_mode()
    def _dummy_sampler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        output = None

        # For profile, have maximum num_reqs and that collectively have
        # maximum num_tokens.
        min_tokens_per_req = self.max_num_tokens // self.max_num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * self.max_num_reqs
        num_scheduled_tokens_list[-1] += self.max_num_tokens % self.max_num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        # TODO: need to rum a dummy sampler for generate task
        hidden_states = hidden_states[logit_indices]
        output = self.model.compute_logits(hidden_states)
        return output

    def profile_run(self) -> None:
        self.eplb_warmup()
        mc2_tokens_capacity = get_mc2_tokens_capacity()
        if self.max_num_tokens > mc2_tokens_capacity and select_moe_comm_method(
            mc2_tokens_capacity, self.vllm_config
        ) in {MoECommType.MC2, MoECommType.FUSED_MC2}:
            self._dummy_run(mc2_tokens_capacity, with_prefill=True, is_profile=True)
        origin_max_num_tokens = self.max_num_tokens
        # in the pcp scenario, the split sequence needs to be used for profile run
        # TODO: after the vllm pcp function is launched, this logic needs to be brought up to the community
        if self.pcp_size > 1:
            self.max_num_tokens = math.ceil(self.max_num_tokens / (self.pcp_size * 2)) * 2
        super().profile_run()
        self.max_num_tokens = origin_max_num_tokens

    def eplb_warmup(self):
        if self.dynamic_eplb and not self.is_eplb_warmuped:
            self.is_eplb_warmuped = True
            self.eplb_adaptor = VllmEplbAdaptor(model=self.model)
            self.eplb_loader.set_adator(self.eplb_adaptor)
            self.eplb_updator.set_adaptor(self.eplb_adaptor)
            self.eplb_updator.warm_up_eplb()

    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)

        with DeviceMemoryProfiler() as m:  # noqa: SIM117
            if self.eplb_enable:
                self.vllm_config.parallel_config.enable_eplb = True
            self.model: nn.Module = get_model(vllm_config=self.vllm_config)
            if self.dynamic_eplb:
                model_register(self.model)
            if self.drafter:
                logger.info("Loading drafter model...")
                if self.vllm_config.quant_config is not None:
                    patch_load_weights(self.vllm_config)
                with get_tp_context(self.drafter):
                    self.drafter.load_model(self.model)
                if self.use_aux_hidden_state_outputs:
                    from vllm.model_executor.models.interfaces import supports_eagle3
                    if not supports_eagle3(self.model):
                        raise RuntimeError(
                            "Model does not support EAGLE3 interface but "
                            "aux_hidden_state_outputs was requested"
                        )
                    aux_layers = self.model.get_eagle3_default_aux_hidden_state_layers()
                    self.model.set_aux_hidden_state_layers(aux_layers)

            if self.lora_config:
                self.model = self.load_lora_model(self.model, self.vllm_config, self.device)
        self.model_memory_usage = m.consumed_memory
        logger.info("Loading model weights took %.4f GB", m.consumed_memory / float(2**30))

        # wrap the model with full graph wrapper if needed.
        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.update_stream: torch.npu.Stream = torch.npu.Stream()
            self.model = ACLGraphWrapper(self.model, self.vllm_config, runtime_mode=CUDAGraphMode.FULL)

    def _validate_sfa_layerwise_connector_cudagraph_mode(self) -> None:
        """Reject full-model replay that would bypass layerwise retrieval."""
        staged_graph_configured = staged_sfa_graph_configured(
            self.vllm_config
        )
        if (
            envs_ascend.VLLM_ASCEND_SFA_STAGED_GRAPH
            and not staged_graph_configured
        ):
            errors = staged_sfa_graph_configuration_errors(
                self.vllm_config
            )
            details = "; ".join(errors) or "unknown incompatibility"
            raise ValueError(
                "VLLM_ASCEND_SFA_STAGED_GRAPH was explicitly requested, "
                "but this configuration is unsupported: " + details
            )
        if staged_graph_configured:
            if not self.use_sparse:
                raise ValueError(
                    "The staged SFA graph path requires sparse attention."
                )
            if (
                not self._profiling_cudagraph_memory
                and not staged_sfa_connector_supports_sparse_load()
            ):
                raise ValueError(
                    "The staged SFA graph path requires an LMCache connector "
                    "that advertises layerwise batched sparse selective loads, "
                    "reliable per-request frontier metadata, and a consumer "
                    "role (kv_both or kv_consumer)."
                )
        mode = self.compilation_config.cudagraph_mode
        if not self.use_sparse or not mode.has_full_cudagraphs():
            return
        if not has_kv_transfer_group():
            return
        connector = get_kv_transfer_group()
        if not bool(
            getattr(connector, "uses_layerwise_model_callbacks", False)
        ):
            return
        raise ValueError(
            "SFA with a layerwise KV connector does not support FULL or "
            "FULL_DECODE_ONLY graph mode: full-model replay bypasses the "
            "Python per-layer retrieval callbacks. Use PIECEWISE graph mode."
        )

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        if getattr(self, "layerwise_prefill_p_node", False):
            self._validate_layerwise_prefill_runtime(
                kv_cache_config.dsa_kv_topology
            )
            self._log_layerwise_prefill_startup(kv_cache_config)
        if staged_sfa_graph_configured(self.vllm_config) and getattr(
            self,
            "_staged_sfa_startup_capture_attempted",
            False,
        ):
            raise RuntimeError(
                "[SFA cross-layer graph] KV cache cannot be reinitialized "
                "after graph capture has started; restart the worker to rebuild "
                "captured addresses."
            )
        self._validate_sfa_layerwise_connector_cudagraph_mode()
        dsa_kv_topology = kv_cache_config.dsa_kv_topology
        kv_cache_config = deepcopy(kv_cache_config)
        kv_cache_config.dsa_kv_topology = dsa_kv_topology
        self.kv_cache_config = kv_cache_config
        self._validate_and_cache_dsa_kv_topology(kv_cache_config)
        self._mamba_copy_bufs = None
        self.may_add_encoder_only_layers_to_kv_cache_config()
        self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)
        # NOTE(cmq): initialize_attn_backend must before using self.attn_groups
        self.initialize_attn_backend(kv_cache_config)
        self.use_hybrid_blocks = len(self.attn_groups) > 1
        # NOTE: Currently, we determine whether we need `num_accepted_tokens` through `MambaSpec`.
        self.need_accepted_tokens = any(
            [isinstance(attn_group[0].kv_cache_spec, MambaSpec) for attn_group in self.attn_groups]
        )

        self.may_reinitialize_input_batch(kv_cache_config)
        kv_caches = self.initialize_kv_cache_tensors(kv_cache_config)
        # TODO: refactor the logic of attention
        # Initialize drafter attention group initialization
        if self.speculative_config and (
            self.speculative_config.use_eagle() or self.speculative_config.uses_draft_model()
        ):
            assert isinstance(self.drafter, AscendEagleProposer | AscendDraftModelProposer)
            block_size = (self.kernel_block_sizes[0] if isinstance(
            self.kernel_block_sizes, list) else self.kernel_block_sizes)
            self.drafter.initialize_attn_backend(kv_cache_config, block_size)

        if has_kv_transfer_group() and not self._profiling_cudagraph_memory:
            kv_transfer_group = get_kv_transfer_group()
            kv_caches_to_register = kv_caches
            disable_dsa_index_lmcache = bool(
                envs_ascend.VLLM_ASCEND_DSA_DISABLE_INDEX_LMCACHE
            )
            requires_full_dsa_kv_caches = bool(
                getattr(kv_transfer_group, "requires_full_dsa_kv_caches", False)
            )
            supports_dsa_index_lmcache = bool(
                getattr(kv_transfer_group, "supports_dsa_index_lmcache", False)
            )
            register_full_dsa_kv_caches = (
                not disable_dsa_index_lmcache
                and (requires_full_dsa_kv_caches or supports_dsa_index_lmcache)
            )
            if self.dsa_unbundle and not register_full_dsa_kv_caches:
                # Un-bundled: the indexer layer registers a 1-tuple (key only, no
                # value). The KV connector only offloads the latent, and LMCache's
                # permute requires >=2 tensors per entry, so register latent layers
                # only and keep the indexer out of the connector entirely.
                kv_caches_to_register = {
                    name: kv
                    for name, kv in kv_caches.items()
                    if not (isinstance(kv, (tuple, list)) and len(kv) < 2)
                }
                logger.info(
                    "DSA un-bundle: registering %d/%d KV layers with the connector "
                    "(latent only; indexer kept resident; "
                    "requires_full_dsa_kv_caches=%s "
                    "supports_dsa_index_lmcache=%s "
                    "disable_dsa_index_lmcache=%s).",
                    len(kv_caches_to_register),
                    len(kv_caches),
                    requires_full_dsa_kv_caches,
                    supports_dsa_index_lmcache,
                    disable_dsa_index_lmcache,
                )
            elif self.dsa_unbundle:
                logger.info(
                    "DSA un-bundle: registering all %d KV layers with the "
                    "group-aware connector for latent/indexer sub-dispatch "
                    "(requires_full_dsa_kv_caches=%s "
                    "supports_dsa_index_lmcache=%s "
                    "disable_dsa_index_lmcache=%s).",
                    len(kv_caches_to_register),
                    requires_full_dsa_kv_caches,
                    supports_dsa_index_lmcache,
                    disable_dsa_index_lmcache,
                )
            kv_transfer_group.register_kv_caches(kv_caches_to_register)

        self._maybe_init_dsa_latent_offload()

        if self.model_config.enable_return_routed_experts:
            self.init_routed_experts_capturer()

    def _validate_and_cache_dsa_kv_topology(
        self, kv_cache_config: KVCacheConfig
    ) -> None:
        self.dsa_kv_topology = None
        self.dsa_kv_rows_by_layer_name = {}
        self.dsa_kv_executions_by_ordinal = {}
        if not getattr(self, "dsa_two_groups", False):
            return

        topology = kv_cache_config.dsa_kv_topology
        if topology is None:
            raise ValueError(
                "DSA two-group mode requires a canonical DSA KV topology."
            )
        if len(kv_cache_config.kv_cache_groups) != 2:
            raise ValueError(
                "DSA KV topology requires exactly two configured KV cache groups, "
                f"got {len(kv_cache_config.kv_cache_groups)}."
            )
        if len(topology.rows_by_group) != 2:
            raise ValueError(
                "DSA KV topology requires exactly two row groups, "
                f"got {len(topology.rows_by_group)}."
            )

        rows_by_layer_name: dict[str, DSAKVRow] = {}
        rows_by_registration: dict[tuple[int, int], DSAKVRow] = {}
        for kv_group, rows in enumerate(topology.rows_by_group):
            for row in rows:
                if row.kv_group != kv_group:
                    raise ValueError(
                        "DSA KV topology row group mismatch for "
                        f"{row.layer_name!r}: row group={row.kv_group}, "
                        f"container group={kv_group}."
                    )
                if row.layer_name in rows_by_layer_name:
                    raise ValueError(
                        "DSA KV topology has duplicate layer name "
                        f"{row.layer_name!r}."
                    )
                registration_key = (row.execution_ordinal, row.kv_group)
                if registration_key in rows_by_registration:
                    raise ValueError(
                        "DSA KV topology has duplicate execution/group "
                        f"registration {registration_key}."
                    )
                rows_by_layer_name[row.layer_name] = row
                rows_by_registration[registration_key] = row

        executions_by_ordinal = {
            execution.execution_ordinal: execution
            for execution in topology.executions
        }
        if len(executions_by_ordinal) != len(topology.executions):
            raise ValueError("DSA KV topology has duplicate execution ordinals.")
        expected_execution_ordinals = {
            row.execution_ordinal for row in topology.rows_by_group[0]
        }
        if set(executions_by_ordinal) != expected_execution_ordinals:
            raise ValueError(
                "DSA KV topology has a partial execution table: "
                f"expected={sorted(expected_execution_ordinals)}, "
                f"got={sorted(executions_by_ordinal)}."
            )
        for registration_key, row in rows_by_registration.items():
            execution = executions_by_ordinal.get(row.execution_ordinal)
            execution_row = None
            if execution is not None:
                execution_row = (
                    execution.latent if row.kv_group == 0 else execution.indexer
                )
            if execution_row != row:
                raise ValueError(
                    "DSA KV topology execution row mismatch for registration "
                    f"{registration_key}."
                )

        configured_specs = self._get_layer_kv_cache_specs(kv_cache_config)
        configured_groups: dict[str, int] = {}
        for kv_group, group in enumerate(kv_cache_config.kv_cache_groups):
            for layer_name in group.layer_names:
                if layer_name in configured_groups:
                    raise ValueError(
                        "DSA KV cache configuration has duplicate layer name "
                        f"{layer_name!r}."
                    )
                configured_groups[layer_name] = kv_group

        missing_topology_rows = sorted(
            set(configured_groups) - set(rows_by_layer_name)
        )
        if missing_topology_rows:
            raise ValueError(
                "DSA KV topology is missing registered cache layers: "
                f"{missing_topology_rows}."
            )
        registered_specs = getattr(self, "_dsa_kv_specs_by_layer_name", {})
        missing_registered_specs = sorted(
            set(configured_groups) - set(registered_specs)
        )
        if missing_registered_specs:
            raise ValueError(
                "DSA two-group mode is missing model-provided registrations for "
                f"cache layers: {missing_registered_specs}."
            )

        for layer_name, kv_group in configured_groups.items():
            row = rows_by_layer_name[layer_name]
            if row.kv_group != kv_group:
                raise ValueError(
                    "DSA KV topology/config group mismatch for "
                    f"{layer_name!r}: topology={row.kv_group}, config={kv_group}."
                )
            registered_spec = registered_specs[layer_name]
            registration = registered_spec.dsa_kv_registration
            assert registration is not None
            if (
                registration.execution_ordinal != row.execution_ordinal
                or registration.kv_group != row.kv_group
            ):
                raise ValueError(
                    "DSA KV topology/model registration mismatch for "
                    f"{layer_name!r}: topology=({row.execution_ordinal}, "
                    f"{row.kv_group}), model=({registration.execution_ordinal}, "
                    f"{registration.kv_group})."
                )
            if registered_spec != configured_specs[layer_name]:
                raise ValueError(
                    "DSA KV topology/config spec mismatch for "
                    f"{layer_name!r}."
                )

        self.dsa_kv_topology = topology
        self.dsa_kv_rows_by_layer_name = rows_by_layer_name
        self.dsa_kv_executions_by_ordinal = executions_by_ordinal
        logger.info_once(
            "DSA KV topology initialized: signature=%s executions=%d "
            "latent_rows=%d indexer_rows=%d.",
            topology.signature,
            len(topology.executions),
            len(topology.rows_by_group[0]),
            len(topology.rows_by_group[1]),
            scope="local",
        )

    def _maybe_init_dsa_latent_offload(self) -> None:
        """Build the DSA latent-offload manager (GLM5.1) when enabled.

        Gated by ``VLLM_ASCEND_ENABLE_DSA_LATENT_OFFLOAD`` and a DSA model; a no-op
        otherwise. Buffers are sized from the same config whose bytes were already
        reserved out of the KV budget in ``determine_available_memory``, so they fit.
        HW-VERIFY: the offloaded layers are the MLAAttention layers (DSA reuses MLA);
        confirm the count matches ``num_hidden_layers`` (the MTP layer may add one).
        """
        # Import the sparse_offload package only when the feature (or its
        # introspection probe) is on, so baseline serving never depends on it.
        import vllm_ascend.envs as envs_ascend

        self.dsa_offload_manager = None

        # Adapter-backed latent hot cache: its OWN flag, independent of the offload
        # manager below, so build it before that early return. Reachable as
        # self.dsa_adapter_cache and threaded into the forward context.
        self.dsa_adapter_cache = None
        if envs_ascend.VLLM_ASCEND_DSA_USE_ADAPTER_CACHE:
            from vllm_ascend.distributed.kv_transfer.sparse_offload.adapter_cache import (
                build_adapter_cache,
            )

            _mla = get_layers_from_vllm_config(self.vllm_config, MLAAttention)
            self.dsa_adapter_cache = build_adapter_cache(
                self.vllm_config, list(_mla.keys()), self.device
            )
            if self.dsa_adapter_cache is not None:
                logger.info(
                    "DSA adapter latent cache enabled for %d MLA layers", len(_mla)
                )

        if not envs_ascend.VLLM_ASCEND_ENABLE_DSA_LATENT_OFFLOAD:
            return

        from vllm_ascend.distributed.kv_transfer.sparse_offload.runner_integration import (
            build_manager,
            config_from_vllm,
        )

        mla_layers = get_layers_from_vllm_config(self.vllm_config, MLAAttention)
        layer_names = list(mla_layers.keys())

        config = config_from_vllm(self.vllm_config, device=self.device)
        if config is None:
            return
        # backend=None -> in-memory reference backend; swap for the LMCache adapter
        # once it lands (no other change needed).
        self.dsa_offload_manager = build_manager(config, layer_names, backend=None)
        logger.info("DSA latent offload enabled for %d MLA layers", len(layer_names))

    def _align_memory(self, tensor: torch.Tensor, alignment: int) -> torch.Tensor:
        data_ptr = tensor.data_ptr()
        aligned_addr = (data_ptr + alignment - 1) // alignment * alignment
        offset = (aligned_addr - data_ptr) // tensor.element_size()
        return tensor[int(offset) :]

    def initialize_kv_cache_tensors(self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Initialize the memory buffer for KV cache.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        # Initialize the memory buffer for KV cache
        kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)
        # Change the memory buffer to the desired shape
        kv_caches = self._reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors)

        # Set up cross-layer KV cache sharing
        for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
            logger.debug("%s reuses KV cache of %s", layer_name, target_layer_name)
            kv_caches[layer_name] = kv_caches[target_layer_name]

        from vllm.v1.worker.utils import bind_kv_cache

        num_attn_module = 2 if self.model_config.hf_text_config.model_type == "longcat_flash" else 1
        bind_kv_cache(kv_caches, self.compilation_config.static_forward_context, self.kv_caches, num_attn_module)
        return kv_caches

    def _get_layer_kv_cache_specs(self, kv_cache_config: KVCacheConfig) -> dict[str, KVCacheSpec]:
        layer_kv_cache_spec: dict[str, KVCacheSpec] = {}
        for group_kv_cache_spec in kv_cache_config.kv_cache_groups:
            group_spec = group_kv_cache_spec.kv_cache_spec
            for layer_name in group_kv_cache_spec.layer_names:
                if isinstance(group_spec, UniformTypeKVCacheSpecs):
                    layer_kv_cache_spec[layer_name] = group_spec.kv_cache_specs[layer_name]
                else:
                    layer_kv_cache_spec[layer_name] = group_spec
        return layer_kv_cache_spec

    def _get_attention_kv_cache_dims(self, layer_name: str, kv_cache_spec: AttentionSpec) -> tuple[int, int]:
        if isinstance(kv_cache_spec, MLAAttentionSpec):
            attn_layers = get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,
                [layer_name],
            )
            attn_layer = attn_layers[layer_name]
            if not isinstance(attn_layer, MLAAttention):
                raise TypeError(
                    f"Expected MLAAttention layer for {layer_name}, got {type(attn_layer).__name__}."
                )
            return attn_layer.kv_lora_rank, attn_layer.qk_rope_head_dim

        head_size_v = kv_cache_spec.head_size_v if hasattr(kv_cache_spec, "head_size_v") else kv_cache_spec.head_size
        return kv_cache_spec.head_size, head_size_v

    def _log_layerwise_prefill_startup(
        self,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        """Log one §13.1 startup line per P-node worker."""

        slab_spec = self._get_layerwise_prefill_slab_spec(kv_cache_config)
        topology = kv_cache_config.dsa_kv_topology
        assert topology is not None
        connector = get_kv_transfer_group()
        logger.info(
            "Layerwise-prefill P worker: residency_mode=PREFILL_LAYERWISE "
            "topology_signature=%s latent_layers=%d indexer_layers=%d "
            "producer_executions=%d latent_page_bytes=%d "
            "indexer_page_bytes=%d bundle_page_bytes=%d "
            "latent_blocks_per_bundle=%d indexer_blocks_per_bundle=%d "
            "available_kv_bytes=%d parent_capacity=%d child_capacity=%d "
            "slab_bytes=%d max_tokens=%d graph_mode=%s "
            "connector_transfer_window=%s connector_sync_callbacks=%s "
            "connector_indexer_persistence=%s.",
            topology.signature,
            len(topology.rows_by_group[0]),
            len(topology.rows_by_group[1]),
            sum(
                1
                for execution in topology.executions
                if execution.indexer is not None
            ),
            slab_spec.latent_page_size_bytes,
            slab_spec.indexer_page_size_bytes,
            slab_spec.bundle_page_size_bytes,
            slab_spec.bundle_page_size_bytes // slab_spec.latent_page_size_bytes,
            slab_spec.bundle_page_size_bytes
            // slab_spec.indexer_page_size_bytes,
            slab_spec.slab_size_bytes,
            kv_cache_config.num_blocks,
            len(topology.rows_by_group[0]) * kv_cache_config.num_blocks,
            slab_spec.slab_size_bytes,
            get_layerwise_prefill_max_tokens(kv_cache_config),
            self.compilation_config.cudagraph_mode.name,
            getattr(
                connector,
                "supports_layerwise_prefill_transfer_window",
                False,
            ),
            getattr(
                connector,
                "supports_layerwise_prefill_eager_callbacks",
                False,
            ),
            getattr(connector, "supports_dsa_index_lmcache", False),
        )

    def _get_layerwise_prefill_slab_spec(
        self,
        kv_cache_config: KVCacheConfig,
    ) -> _LayerwisePrefillSlabSpec:
        topology = kv_cache_config.dsa_kv_topology
        if topology is None or len(topology.rows_by_group) != 2:
            raise RuntimeError(
                "Layerwise-prefill global slab requires canonical two-group "
                "DSA topology."
            )
        if len(kv_cache_config.kv_cache_groups) != 2:
            raise RuntimeError(
                "Layerwise-prefill global slab requires exactly two KV groups."
            )

        latent_names = tuple(row.layer_name for row in topology.rows_by_group[0])
        indexer_names = tuple(row.layer_name for row in topology.rows_by_group[1])
        expected_names_by_group = (set(latent_names), set(indexer_names))
        specs_by_group: dict[int, AttentionSpec] = {}
        for group in kv_cache_config.kv_cache_groups:
            group_names = set(group.layer_names)
            matching_groups = [
                group_index
                for group_index, expected_names in enumerate(expected_names_by_group)
                if group_names == expected_names
            ]
            if len(matching_groups) != 1 or not isinstance(
                group.kv_cache_spec, AttentionSpec
            ):
                raise RuntimeError(
                    "Layerwise-prefill KV groups do not match the final "
                    "LATENT/INDEXER topology and attention specs."
                )
            specs_by_group[matching_groups[0]] = group.kv_cache_spec
        if set(specs_by_group) != {0, 1}:
            raise RuntimeError(
                "Layerwise-prefill final LATENT/INDEXER specs are incomplete."
            )

        latent_spec = specs_by_group[0]
        indexer_spec = specs_by_group[1]
        latent_dims = getattr(latent_spec, "sparse_head_dim", None)
        indexer_dims = getattr(indexer_spec, "sparse_head_dim", None)
        if (
            latent_dims is None
            or len(latent_dims) != 2
            or indexer_dims is None
            or len(indexer_dims) != 1
        ):
            raise RuntimeError(
                "Layerwise-prefill final specs must describe two LATENT planes "
                "and one INDEXER plane."
            )
        kv_lora_rank, qk_rope_head_dim = map(int, latent_dims)
        index_head_dim = int(indexer_dims[0])
        if (
            latent_spec.dtype != indexer_spec.dtype
            or latent_spec.block_size != indexer_spec.block_size
            or latent_spec.num_kv_heads != indexer_spec.num_kv_heads
            or latent_spec.head_size != kv_lora_rank + qk_rope_head_dim
            or indexer_spec.head_size != index_head_dim
        ):
            raise RuntimeError(
                "Layerwise-prefill final LATENT/INDEXER specs have incompatible "
                "dtype, block, head, or sparse dimensions."
            )

        latent_page = latent_spec.page_size_bytes
        indexer_page = indexer_spec.page_size_bytes
        bundle_page = math.lcm(latent_page, indexer_page)
        slab_size = (
            len(latent_names) * kv_cache_config.num_blocks + 1
        ) * bundle_page
        logical_names = (*latent_names, *indexer_names)
        if len(kv_cache_config.kv_cache_tensors) != 1:
            raise RuntimeError(
                "Layerwise-prefill P node requires exactly one global raw KV "
                "tensor."
            )
        tensor_spec = kv_cache_config.kv_cache_tensors[0]
        if (
            tensor_spec.size != slab_size
            or len(tensor_spec.shared_by) != len(logical_names)
            or len(set(tensor_spec.shared_by)) != len(logical_names)
            or set(tensor_spec.shared_by) != set(logical_names)
        ):
            raise RuntimeError(
                "Layerwise-prefill global raw tensor disagrees with the final "
                "topology, parent capacity, or single-null layout."
            )
        return _LayerwisePrefillSlabSpec(
            logical_names=logical_names,
            latent_names=latent_names,
            indexer_names=indexer_names,
            dtype=latent_spec.dtype,
            block_size=latent_spec.block_size,
            num_kv_heads=latent_spec.num_kv_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            index_head_dim=index_head_dim,
            latent_page_size_bytes=latent_page,
            indexer_page_size_bytes=indexer_page,
            bundle_page_size_bytes=bundle_page,
            slab_size_bytes=slab_size,
        )

    def _allocate_layerwise_prefill_kv_cache_tensor(
        self,
        kv_cache_config: KVCacheConfig,
        alignment: int,
    ) -> dict[str, tuple[torch.Tensor]]:
        slab_spec = self._get_layerwise_prefill_slab_spec(kv_cache_config)
        raw_backing = torch.zeros(
            slab_spec.slab_size_bytes + alignment,
            dtype=torch.uint8,
            device=self.device,
        )
        raw = self._align_memory(raw_backing, alignment)[
            : slab_spec.slab_size_bytes
        ]
        self._layerwise_prefill_global_raw_backing = raw_backing
        self._layerwise_prefill_global_raw = raw
        self._layerwise_prefill_slab_spec = slab_spec
        shared_raw = (raw,)
        return {name: shared_raw for name in slab_spec.logical_names}

    def _reshape_layerwise_prefill_kv_cache_tensor(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_raw_tensors: dict[str, tuple[torch.Tensor]],
    ) -> dict[str, tuple[torch.Tensor, ...]]:
        slab_spec = getattr(self, "_layerwise_prefill_slab_spec", None)
        if slab_spec is None:
            slab_spec = self._get_layerwise_prefill_slab_spec(kv_cache_config)
        raw = getattr(self, "_layerwise_prefill_global_raw", None)
        if raw is None:
            raise RuntimeError(
                "Layerwise-prefill global raw tensor lost its aligned owner."
            )
        if any(
            len(tensors) != 1 or tensors[0] is not raw
            for tensors in kv_cache_raw_tensors.values()
        ) or set(kv_cache_raw_tensors) != set(slab_spec.logical_names):
            raise RuntimeError(
                "Layerwise-prefill logical raw bindings do not share the one "
                "global slab."
            )

        latent_view, indexer_view = reshape_dsa_layerwise_prefill_raw(
            raw,
            slab_spec.dtype,
            slab_spec.block_size,
            slab_spec.num_kv_heads,
            slab_spec.kv_lora_rank,
            slab_spec.qk_rope_head_dim,
            slab_spec.index_head_dim,
            latent_page_size_bytes=slab_spec.latent_page_size_bytes,
            indexer_page_size_bytes=slab_spec.indexer_page_size_bytes,
        )
        return {
            **{name: latent_view for name in slab_spec.latent_names},
            **{name: indexer_view for name in slab_spec.indexer_names},
        }

    def _allocate_kv_cache_tensors(self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Initializes the KV cache buffer with the correct size. The buffer needs
        to be reshaped to the desired shape before being used by the models.

        NOTE: To support prefill disaggregation, we need to split kvcache tensor into
        k_cache and v cache, and the addr of both are aligned by 2M

        Args:
            kv_cache_config: The KV cache config
        Returns:
            dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
            dict[str, tuple(torch.Tensor, torch.Tensor)] A map between layer names
            to their corresponding memory buffer for K cache and V cache.
        """
        # init kv cache tensors
        kv_cache_raw_tensors: dict[str, torch.Tensor | torch.Tensor | None | None] = {}
        # prefill disaggregation need the addr of cache tensor be aligned with 2M
        alignment = 2 * 1024 * 1024
        if getattr(self, "layerwise_prefill_p_node", False):
            self.hybrid_with_attn_and_mamba = False
            return self._allocate_layerwise_prefill_kv_cache_tensor(
                kv_cache_config,
                alignment,
            )
        self._layerwise_prefill_global_raw_backing = None
        self._layerwise_prefill_global_raw = None
        self._layerwise_prefill_slab_spec = None
        layer_kv_cache_spec = self._get_layer_kv_cache_specs(kv_cache_config)
        # If some tensors are shared by linear layers and attention layers,
        # the same tensor format must be maintained even if some layers
        # have only linear or attention layers, for example, the mtp layer.
        self.hybrid_with_attn_and_mamba = False
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            use_mamba, use_attn = False, False
            for layer_name in kv_cache_tensor.shared_by:
                if isinstance(layer_kv_cache_spec[layer_name], MambaSpec):
                    use_mamba = True
                if isinstance(layer_kv_cache_spec[layer_name], AttentionSpec):
                    use_attn = True
            self.hybrid_with_attn_and_mamba = self.hybrid_with_attn_and_mamba or (use_mamba and use_attn)
            if (
                self.dsa_shared_pool
                and self.use_sparse
                and use_attn
                and not use_mamba
                and any("indexer" in ln for ln in kv_cache_tensor.shared_by)
                and any("indexer" not in ln for ln in kv_cache_tensor.shared_by)
            ):
                if self.use_sparse_c8_indexer:
                    raise RuntimeError("DSA shared pool does not support sparse C8 indexer.")
                if self.vllm_config.kv_transfer_config is None:
                    raw_tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=self.device)
                else:
                    cache_size_aligned = kv_cache_tensor.size + alignment
                    raw_tensor = torch.zeros(cache_size_aligned, dtype=torch.int8, device=self.device)
                    raw_tensor = self._align_memory(raw_tensor, alignment)[: kv_cache_tensor.size]
                for layer_name_inner in kv_cache_tensor.shared_by:
                    kv_cache_raw_tensors[layer_name_inner] = (raw_tensor,)
                continue
            for idx in range(len(kv_cache_tensor.shared_by)):
                layer_name = kv_cache_tensor.shared_by[idx]
                if (
                    "linear_attn" in layer_name or self.hybrid_with_attn_and_mamba
                ) and layer_name not in kv_cache_raw_tensors:
                    # for mamba linear attention or attn-linear hybrid
                    if self.vllm_config.kv_transfer_config is None:
                        tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=self.device)
                    else:
                        cache_size_aligned = kv_cache_tensor.size + alignment
                        tensor = torch.zeros(cache_size_aligned, dtype=torch.int8, device=self.device)
                        tensor = self._align_memory(tensor, alignment)[: kv_cache_tensor.size]

                    for layer_name_inner in kv_cache_tensor.shared_by:
                        # shared the kvcache for all shared layers
                        kv_cache_raw_tensors[layer_name_inner] = tensor
                elif "attn" in layer_name and layer_name not in kv_cache_raw_tensors and not use_mamba:
                    # NOTE: We need to init k cache tensor (nope cache tensor in mla) and
                    # v cache tensor (rope cache tensor in mla) separately to support prefill disaggregation,
                    # as it only support the 0-dim of kv_cache is `num_blocks`.
                    # For deepseek mla, we need to spilt cache tensor accrodding to the nope head dim
                    # and rope head dim.
                    current_kv_cache_spec = layer_kv_cache_spec[layer_name]
                    assert isinstance(current_kv_cache_spec, AttentionSpec)

                    dsa_k_tensor_size = None
                    dsa_k_scale_tensor_size = None
                    unbundle_indexer = False
                    if self.dsa_unbundle and self.use_sparse:
                        # Proper route P1: each layer's tensor is allocated per its own
                        # group — latent (k_nope + k_pe) or indexer (single vector).
                        # Size from num_blocks * true per-block bytes (NOT a proportional
                        # split of kv_cache_tensor.size, which carries page padding from
                        # two groups of different page sizes -> odd, mis-aligned bytes).
                        # Discriminate by LAYER NAME (grouping may rewrite sparse_head_dim).
                        kv_lora_rank, qk_rope_head_dim, index_head_dim = self.sparse_head_dim
                        elt = get_dtype_size(self.kv_cache_dtype)
                        bs = current_kv_cache_spec.block_size
                        # Use THIS tensor's allocated budget (kv_cache_tensor.size), not
                        # kv_cache_config.num_blocks (over-counted across two groups of
                        # different page sizes -> OOM). nb*page == size, so k+v fits.
                        if any("indexer" in ln for ln in kv_cache_tensor.shared_by):
                            unbundle_indexer = True
                            k_tensor_size = int(kv_cache_tensor.size)  # whole = indexer cache
                            v_tensor_size = 0
                        else:
                            # Derive nb from the TRUE latent page (block_size*(512+64)*elt),
                            # NOT spec.page_size_bytes (grouping unifies it to the small
                            # indexer page -> nb overcounted -> OOM).
                            latent_page = bs * (kv_lora_rank + qk_rope_head_dim) * elt
                            nb = int(kv_cache_tensor.size) // latent_page
                            k_tensor_size = nb * bs * kv_lora_rank * elt
                            v_tensor_size = nb * bs * qk_rope_head_dim * elt
                    elif self.use_sparse and self.dsa_free_paged:
                        # DSA offload (M-B): the page holds ONLY the indexer key; the
                        # latent k/v are 1-block dummies (exec_kv writes the
                        # PagedLatentPool instead). This is what shrinks per-token memory.
                        dtb = get_dtype_size(self.kv_cache_dtype)
                        bs = current_kv_cache_spec.block_size
                        k_tensor_size = bs * self.sparse_head_dim[0] * dtb  # 1-block dummy
                        v_tensor_size = bs * self.sparse_head_dim[1] * dtb  # 1-block dummy
                        dsa_k_tensor_size = int(kv_cache_tensor.size)  # whole page = indexer
                    elif self.use_sparse:
                        # for deepseek v3.2, we split the kv cache according to the corresponding ratio
                        # Shared-indexer consumers own no indexer key plane, so their
                        # tensor splits into latent k/v only.
                        has_indexer_cache = sparse_kv_cache_has_indexer(
                            layer_kv_cache_spec[layer_name]
                        )
                        if has_indexer_cache:
                            sparse_kv_cache_ratio = layer_kv_cache_spec[layer_name].sparse_kv_cache_ratio
                            k_tensor_size = int(kv_cache_tensor.size // sparse_kv_cache_ratio[0])
                            v_tensor_size = int(kv_cache_tensor.size // sparse_kv_cache_ratio[1])
                            dsa_k_tensor_size = int(kv_cache_tensor.size // sparse_kv_cache_ratio[2])
                            if self.use_sparse_c8_indexer:
                                dsa_k_scale_tensor_size = int(kv_cache_tensor.size // sparse_kv_cache_ratio[3])
                        else:
                            assert not self.use_sparse_c8_indexer
                            k_dim, v_dim, _ = layer_kv_cache_spec[layer_name].sparse_head_dim
                            k_tensor_split_factor, v_tensor_split_factor = calc_split_factor([k_dim, v_dim])
                            k_tensor_size = int(kv_cache_tensor.size // k_tensor_split_factor)
                            v_tensor_size = int(kv_cache_tensor.size // v_tensor_split_factor)
                    else:
                        k_dim, v_dim = self._get_attention_kv_cache_dims(layer_name, current_kv_cache_spec)
                        assert k_dim > 0 and v_dim > 0
                        kv_head_dim_list = [
                            k_dim,
                            v_dim,
                        ]
                        if self.is_kv_consumer and enable_fa_quant(self.vllm_config):
                            k_tensor_split_factor, v_tensor_split_factor = (
                                self.vllm_config.quant_config.get_kv_quant_split_factor(layer_name, kv_head_dim_list)
                            )
                        else:
                            k_tensor_split_factor, v_tensor_split_factor = calc_split_factor(kv_head_dim_list)
                        k_tensor_size = int(kv_cache_tensor.size // k_tensor_split_factor)
                        v_tensor_size = int(kv_cache_tensor.size // v_tensor_split_factor)

                    # for other attentions, e.g., self_attn, sliding window attn
                    if self.vllm_config.kv_transfer_config is None:
                        k_tensor = torch.zeros(k_tensor_size, dtype=torch.int8, device=self.device)
                        v_tensor = torch.zeros(v_tensor_size, dtype=torch.int8, device=self.device)
                        #### for deepseek sparse attention
                        if dsa_k_tensor_size is not None:
                            dsa_k_tensor = torch.zeros(dsa_k_tensor_size, dtype=torch.int8, device=self.device)
                        if dsa_k_scale_tensor_size is not None:
                            dsa_k_scale_tensor = torch.zeros(
                                dsa_k_scale_tensor_size, dtype=torch.int8, device=self.device
                            )
                    else:
                        k_tensor = torch.zeros(k_tensor_size + alignment, dtype=torch.int8, device=self.device)
                        v_tensor = torch.zeros(v_tensor_size + alignment, dtype=torch.int8, device=self.device)
                        k_tensor = self._align_memory(k_tensor, alignment)[:k_tensor_size]
                        v_tensor = self._align_memory(v_tensor, alignment)[:v_tensor_size]
                        #### for deepseek sparse attention
                        if dsa_k_tensor_size is not None:
                            dsa_k_tensor = torch.zeros(
                                dsa_k_tensor_size + alignment, dtype=torch.int8, device=self.device
                            )
                            dsa_k_tensor = self._align_memory(dsa_k_tensor, alignment)[:dsa_k_tensor_size]
                        if dsa_k_scale_tensor_size is not None:
                            dsa_k_scale_tensor = torch.zeros(
                                dsa_k_scale_tensor_size + alignment, dtype=torch.int8, device=self.device
                            )
                            dsa_k_scale_tensor = self._align_memory(
                                dsa_k_scale_tensor, alignment
                            )[:dsa_k_scale_tensor_size]

                    for layer_name_inner in kv_cache_tensor.shared_by:
                        # shared the attn kvcache for all shared layers
                        if "attn" in layer_name_inner and "linear_attn" not in layer_name_inner:
                            if self.dsa_unbundle and self.use_sparse:
                                kv_cache_raw_tensors[layer_name_inner] = (
                                    (k_tensor,) if unbundle_indexer else (k_tensor, v_tensor)
                                )
                            elif self.use_sparse:
                                if self.use_sparse_c8_indexer:
                                    kv_cache_raw_tensors[layer_name_inner] = (
                                        k_tensor, v_tensor, dsa_k_tensor, dsa_k_scale_tensor
                                    )
                                else:
                                    kv_cache_raw_tensors[layer_name_inner] = (k_tensor, v_tensor, dsa_k_tensor)
                            else:
                                kv_cache_raw_tensors[layer_name_inner] = (k_tensor, v_tensor)
        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_cache_raw_tensors.keys()), "Some layers are not correctly initialized"

        return kv_cache_raw_tensors

    def _reshape_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Reshape the KV cache tensors to the desired shape and dtype.

        Args:
            kv_cache_config: The KV cache config
            kv_cache_raw_tensors: The KV cache buffer of each layer, with
                correct size but uninitialized shape.
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        if getattr(self, "layerwise_prefill_p_node", False):
            return self._reshape_layerwise_prefill_kv_cache_tensor(
                kv_cache_config,
                kv_cache_raw_tensors,  # type: ignore[arg-type]
            )  # type: ignore[return-value]

        kv_caches: dict[str, torch.Tensor] = {}
        layer_kv_cache_spec = self._get_layer_kv_cache_specs(kv_cache_config)
        for group in self._kv_cache_spec_attn_group_iterator():
            attn_backend = group.backend
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue

                current_kv_cache_spec = layer_kv_cache_spec[layer_name]

                # Proper route P1: each layer is its own group (latent k_nope+k_pe, or
                # indexer single vector) — reshape directly, no 3-tuple split.
                if self.dsa_unbundle and self.use_sparse and isinstance(
                    current_kv_cache_spec, AttentionSpec
                ):
                    spec = current_kv_cache_spec
                    raws = kv_cache_raw_tensors[layer_name]
                    bs, nh = spec.block_size, spec.num_kv_heads
                    elt = get_dtype_size(spec.dtype)
                    kv_lora_rank, qk_rope_head_dim, index_head_dim = self.sparse_head_dim
                    if self.dsa_shared_pool and len(raws) == 1:
                        kv_caches[layer_name] = reshape_dsa_shared_pool_raw(
                            raws[0],
                            spec.dtype,
                            bs,
                            nh,
                            kv_lora_rank,
                            qk_rope_head_dim,
                            index_head_dim,
                            is_indexer="indexer" in layer_name,
                        )
                        continue
                    # Discriminate by LAYER NAME (grouping may rewrite sparse_head_dim).
                    if "indexer" in layer_name:  # single vector cache
                        nb = raws[0].numel() // (bs * nh * index_head_dim * elt)
                        kv_caches[layer_name] = (
                            raws[0].view(spec.dtype).view(nb, bs, nh, index_head_dim),
                        )
                    else:  # latent group: (k_nope, k_pe)
                        nb = raws[0].numel() // (bs * nh * kv_lora_rank * elt)
                        k_nope = raws[0].view(spec.dtype).view(nb, bs, nh, kv_lora_rank)
                        k_pe = raws[1].view(spec.dtype).view(nb, bs, nh, qk_rope_head_dim)
                        kv_caches[layer_name] = (k_nope, k_pe)
                    continue

                # TODO: remove this after the OOM issue is located and fixed, otherwise, some model may
                # encounter OOM issue
                if isinstance(current_kv_cache_spec, AttentionSpec):
                    has_indexer_cache = sparse_kv_cache_has_indexer(current_kv_cache_spec)
                    if self.use_sparse:
                        if self.use_sparse_c8_indexer:
                            raw_k_tensor, raw_v_tensor, raw_dsa_k_tensor, raw_dsa_k_scale_tensor = kv_cache_raw_tensors[  # type: ignore
                                layer_name]
                            assert raw_dsa_k_tensor is not None
                            assert raw_dsa_k_scale_tensor is not None
                            sum_page_size_bytes = (
                                raw_k_tensor.numel()
                                + raw_v_tensor.numel()
                                + raw_dsa_k_tensor.numel()
                                + raw_dsa_k_scale_tensor.numel()
                            )
                        elif not has_indexer_cache:
                            # Shared-indexer consumer: latent k/v only.
                            assert not self.use_sparse_c8_indexer
                            raw_k_tensor, raw_v_tensor = kv_cache_raw_tensors[  # type: ignore
                                layer_name]
                            sum_page_size_bytes = raw_k_tensor.numel() + raw_v_tensor.numel()
                        else:
                            raw_k_tensor, raw_v_tensor, raw_dsa_k_tensor = kv_cache_raw_tensors[  # type: ignore
                                layer_name]
                            assert raw_dsa_k_tensor is not None
                            if self.dsa_free_paged:
                                # latent k/v are 1-block dummies (not part of the page);
                                # the page = indexer only, so num_blocks derives from it.
                                sum_page_size_bytes = raw_dsa_k_tensor.numel()
                            else:
                                sum_page_size_bytes = (
                                    raw_k_tensor.numel() + raw_v_tensor.numel() + raw_dsa_k_tensor.numel()
                                )
                    elif self.use_hybrid_blocks and self.hybrid_with_attn_and_mamba:
                        # Currently, we ensure that the same kvcache format is used even if there
                        # is no shared layer, such as the full attention mtp layer of qwen3.5, etc.
                        raw_k_tensor, raw_v_tensor = kv_cache_raw_tensors[layer_name], kv_cache_raw_tensors[layer_name]
                        sum_page_size_bytes = raw_k_tensor.numel()
                    else:
                        raw_k_tensor, raw_v_tensor = kv_cache_raw_tensors[  # type: ignore
                            layer_name
                        ]
                        sum_page_size_bytes = raw_k_tensor.numel() + raw_v_tensor.numel()
                    assert raw_k_tensor is not None
                    assert raw_v_tensor is not None
                    assert sum_page_size_bytes % current_kv_cache_spec.page_size_bytes == 0
                    num_blocks = sum_page_size_bytes // current_kv_cache_spec.page_size_bytes

                    # `num_blocks` is the number of blocks the model runner can use.
                    # `kv_cache_config.num_blocks` is the number of blocks that
                    # KVCacheManager may allocate.
                    # Since different GPUs may have different number of layers and
                    # different memory capacities, `num_blocks` can be different on
                    # different GPUs, and `kv_cache_config.num_blocks` is set to
                    # the min of all `num_blocks`. Verify it here.
                    assert num_blocks >= kv_cache_config.num_blocks

                    if hasattr(attn_backend, "get_supported_kernel_block_sizes") and self.use_hybrid_blocks:
                        block_size = attn_backend.get_supported_kernel_block_sizes()[0]

                        block_size_chunk = current_kv_cache_spec.block_size // block_size
                        kv_cache_shape = attn_backend.get_kv_cache_shape(
                            num_blocks * block_size_chunk,
                            block_size,
                            current_kv_cache_spec.num_kv_heads,
                            current_kv_cache_spec.head_size,
                        )
                        if self.hybrid_with_attn_and_mamba:
                            attn_tensor_page_size = int(np.prod(kv_cache_shape[1:])) * get_dtype_size(
                                current_kv_cache_spec.dtype
                            )
                            conv_block_padding_size = raw_k_tensor.numel() - attn_tensor_page_size * 2
                            raw_kv_tensor = raw_k_tensor[conv_block_padding_size:]
                            raw_k_tensor = raw_kv_tensor[:attn_tensor_page_size]
                            raw_v_tensor = raw_kv_tensor[attn_tensor_page_size:]
                    else:
                        kv_cache_shape = attn_backend.get_kv_cache_shape(
                            num_blocks,
                            current_kv_cache_spec.block_size,
                            current_kv_cache_spec.num_kv_heads,
                            current_kv_cache_spec.head_size,
                        )
                    if not isinstance(current_kv_cache_spec, MLAAttentionSpec):
                        k_shape = kv_cache_shape[1:]
                        if hasattr(current_kv_cache_spec, "head_size_v"):
                            v_shape = (*kv_cache_shape[1:-1], current_kv_cache_spec.head_size_v)
                        else:
                            v_shape = k_shape
                    else:
                        # k_cache: nope_cache    v_cache: rope_cache
                        mla_num_blocks, mla_block_size, num_kv_heads, _ = kv_cache_shape
                        k_dim, v_dim = self._get_attention_kv_cache_dims(layer_name, current_kv_cache_spec)
                        # DSA offload (M-B): latent is in the PagedLatentPool, so the
                        # paged k/v are 1-block dummies (the op writes the pool, not these).
                        latent_blocks = 1 if self.dsa_free_paged else mla_num_blocks
                        k_shape = (
                            latent_blocks,
                            mla_block_size,
                            num_kv_heads,
                            k_dim,
                        )
                        v_shape = (
                            latent_blocks,
                            mla_block_size,
                            num_kv_heads,
                            v_dim,
                        )
                    k_cache_dtype = v_cache_dtype = current_kv_cache_spec.dtype
                    if self.is_kv_consumer and enable_fa_quant(self.vllm_config):
                        k_cache_dtype, v_cache_dtype = self.vllm_config.quant_config.get_kv_quant_dtype(
                            layer_name, current_kv_cache_spec.dtype, self.model_config
                        )
                    k_cache = raw_k_tensor.view(k_cache_dtype).view(k_shape)
                    v_cache = raw_v_tensor.view(v_cache_dtype).view(v_shape)

                    if self.use_sparse:
                        if not has_indexer_cache:
                            # Shared-indexer consumer: no dsa_k plane was
                            # allocated for this layer.
                            kv_caches[layer_name] = (k_cache, v_cache)
                        else:
                            dsa_k_cache_shape = (
                                num_blocks,
                                current_kv_cache_spec.block_size,
                                current_kv_cache_spec.num_kv_heads,
                                self.model_config.hf_text_config.index_head_dim,
                            )
                            if self.use_sparse_c8_indexer:
                                # dsa_k
                                dsa_k_cache = raw_dsa_k_tensor.view(self.c8_k_cache_dtype).view(dsa_k_cache_shape)
                                # dsa_k_scale
                                dsa_k_scale_cache_shape = (
                                    num_blocks,
                                    current_kv_cache_spec.block_size,
                                    current_kv_cache_spec.num_kv_heads,
                                    1,
                                )
                                assert raw_dsa_k_scale_tensor is not None
                                dsa_k_scale_cache = (
                                    raw_dsa_k_scale_tensor
                                    .view(self.c8_k_scale_cache_dtype)
                                    .view(dsa_k_scale_cache_shape)
                                )
                                kv_caches[layer_name] = (k_cache, v_cache, dsa_k_cache, dsa_k_scale_cache)
                            else:
                                # dsa_k
                                dsa_k_cache = raw_dsa_k_tensor.view(current_kv_cache_spec.dtype).view(dsa_k_cache_shape)
                                kv_caches[layer_name] = (k_cache, v_cache, dsa_k_cache)
                    else:
                        kv_caches[layer_name] = (k_cache, v_cache)
                elif isinstance(current_kv_cache_spec, MambaSpec):
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    assert raw_tensor is not None
                    assert raw_tensor.numel() % current_kv_cache_spec.page_size_bytes == 0
                    num_blocks = raw_tensor.numel() // current_kv_cache_spec.page_size_bytes
                    assert num_blocks >= kv_cache_config.num_blocks

                    # `num_blocks` is the number of blocks the model runner can use.
                    # `kv_cache_config.num_blocks` is the number of blocks that
                    # KVCacheManager may allocate.
                    # Since different GPUs may have different number of layers and
                    # different memory capacities, `num_blocks` can be different on
                    # different GPUs, and `kv_cache_config.num_blocks` is set to
                    # the min of all `num_blocks`. Verify it here.

                    state_tensors = []
                    target_idx = 0
                    start_idx = 0
                    # NOTE(zxr): in order to keep all tensor contiguous, we align ssm and kv block
                    # with same page size, so have to add extra padding block for kv, the overall
                    # layout of hybrid kv_cache on Ascend is:
                    # tensor1: [(kv_padding), conv           , ...]
                    # tensor2: [k           , ssm            , ...]
                    # tensor3: [v           , (mamba_padding), ...]
                    for shape, dtype in zip(current_kv_cache_spec.shapes, current_kv_cache_spec.dtypes):
                        # normally, there is conv state and ssm state in this loop. And there is only
                        # a conv state in some special models.
                        target_shape = (num_blocks, *shape)

                        target_idx += math.prod(target_shape) * get_dtype_size(dtype)
                        tensor = raw_tensor[start_idx:target_idx].view(dtype).view(target_shape)
                        start_idx = target_idx
                        state_tensors.append(tensor)
                    kv_caches[layer_name] = state_tensors
                else:
                    raise ValueError("Unknown KV cache spec type.")

        return kv_caches

    def may_reinitialize_input_batch(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Re-initialize the input batch if the block sizes are different from
        `[self.cache_config.block_size]`. This usually happens when there
        are multiple KV cache groups.

        Args:
            kv_cache_config: The KV cache configuration.
        """
        block_sizes = [
            kv_cache_group.kv_cache_spec.block_size
            for kv_cache_group in kv_cache_config.kv_cache_groups
            if not isinstance(kv_cache_group.kv_cache_spec, EncoderOnlyAttentionSpec)
        ]

        # Generate kernel_block_sizes that matches each block_size
        # For attention backends that support virtual block splitting,
        # use the supported block sizes from the backend
        # For other backends (like Mamba), use [0] (no splitting)
        self.kernel_block_sizes = []
        for kv_cache_group_id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                # All layers in the UniformTypeKVCacheSpecs have the same type,
                # Pick an arbitrary one to dispatch.
                kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                continue
            elif isinstance(kv_cache_spec, AttentionSpec):
                # This is an attention backend that supports virtual
                # block splitting. Get the supported block sizes from
                # the backend.
                try:
                    attn_groups = self.attn_groups[kv_cache_group_id]
                except IndexError:
                    attn_groups = None
                if attn_groups and self.use_hybrid_blocks:
                    # Use the backend's supported block size list
                    backend = attn_groups[0].backend
                    supported_sizes = backend.get_supported_kernel_block_sizes()
                    # If no specific sizes supported, use cache config
                    # block_size
                    kernel_block_size_list = supported_sizes if supported_sizes else [self.cache_config.block_size]
                else:
                    # Fallback to cache config block_size if no backend found
                    kernel_block_size_list = [self.cache_config.block_size]
                self.kernel_block_sizes.append(kernel_block_size_list)
            else:
                # This is likely Mamba or other non-attention cache,
                # no splitting.
                # NOTE: set kernel_block_sizes to 0 to disable slotmapping computation
                # of mamba block. In this case, BlockTable.block_size will never equal
                # to kernel_block_sizes[0]
                self.kernel_block_sizes.append([0])

        max_num_blocks = []
        max_model_len = max(self.max_model_len, self.max_encoder_len)
        for i, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            if isinstance(kv_cache_group.kv_cache_spec, EncoderOnlyAttentionSpec):
                continue
            max_num_blocks_per_req = cdiv(max_model_len, block_sizes[i] * get_total_cp_world_size())
            if self.dsa_shared_pool and isinstance(kv_cache_group.kv_cache_spec, AttentionSpec):
                if kv_cache_group.kv_cache_spec.head_size > self.sparse_head_dim[-1]:
                    blocks_per_bundle = 2
                else:
                    blocks_per_bundle = 9
                max_num_blocks_per_req = cdiv(
                    max_num_blocks_per_req, blocks_per_bundle
                ) * blocks_per_bundle
            if isinstance(kv_cache_group.kv_cache_spec, MambaSpec):
                mamba_blocks_per_req = (
                    max_num_blocks_per_req if self.cache_config.enable_prefix_caching else 1
                ) + kv_cache_group.kv_cache_spec.num_speculative_blocks

                max_num_blocks_per_req = max(max_num_blocks_per_req, mamba_blocks_per_req)
            max_num_blocks.append(max_num_blocks_per_req)

        if block_sizes != [self.cache_config.block_size] or self.kernel_block_sizes != [[self.cache_config.block_size]]:
            assert self.offload_config.uva.cpu_offload_gb == 0, (
                "Cannot re-initialize the input batch when CPU weight "
                "offloading is enabled. See https://github.com/vllm-project/vllm/pull/18298 "  # noqa: E501
                "for more details."
            )
            self.input_batch = NPUInputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=max_model_len,
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                pin_memory=self.pin_memory,
                vocab_size=self.model_config.get_vocab_size(),
                block_sizes=block_sizes,
                is_spec_decode=bool(self.vllm_config.speculative_config),
                logitsprocs=self.input_batch.logitsprocs,
                is_pooling_model=self.is_pooling_model,
                num_speculative_tokens=(
                    self.vllm_config.speculative_config.num_speculative_tokens
                    if self.vllm_config.speculative_config
                    else 0
                ),
                kernel_block_sizes=self.kernel_block_sizes,
                max_num_blocks_per_req=max_num_blocks,
                layerwise_prefill_p_node=self.layerwise_prefill_p_node,
            )

    def initialize_attn_backend(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize the attention backends and attention metadata builders.
        """
        assert len(self.attn_groups) == 0, "Attention backends are already initialized"

        class AttentionGroupKey(NamedTuple):
            attn_backend: type[AttentionBackend]
            kv_cache_spec: KVCacheSpec

        def get_attn_backends_for_group(
            kv_cache_group_spec: KVCacheGroupSpec,
        ) -> tuple[dict[AttentionGroupKey, list[str]], set[type[AttentionBackend]]]:
            layers = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase, kv_cache_group_spec.layer_names)
            attn_backends = {}
            attn_backend_layers = defaultdict(list)
            # Dedupe based on full class name; this is a bit safer than
            # using the class itself as the key because when we create dynamic
            # attention backend subclasses (e.g. ChunkedLocalAttention) unless
            # they are cached correctly, there will be different objects per
            # layer.
            for layer_name in kv_cache_group_spec.layer_names:
                attn_backend = layers[layer_name].get_attn_backend()
                full_cls_name = attn_backend.full_cls_name()
                layer_kv_cache_spec = kv_cache_group_spec.kv_cache_spec
                if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                    layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]
                key = (full_cls_name, layer_kv_cache_spec)
                attn_backends[key] = AttentionGroupKey(attn_backend, layer_kv_cache_spec)
                attn_backend_layers[key].append(layer_name)
            return (
                {attn_backends[k]: v for k, v in attn_backend_layers.items()},
                set(group_key.attn_backend for group_key in attn_backends.values()),
            )

        def create_attn_groups(
            attn_backends_map: dict[AttentionBackend, list[str]], kv_cache_group_id: int
        ) -> list[AttentionGroup]:
            attn_groups: list[AttentionGroup] = []
            for (attn_backend, kv_cache_spec), layer_names in attn_backends_map.items():
                attn_metadata_builders = []
                attn_metadata_builders.append(
                    attn_backend.get_builder_cls()(
                        kv_cache_spec,
                        layer_names,
                        self.vllm_config,
                        self.device,
                    )
                )
                attn_group = AttentionGroup(
                    attn_backend, layer_names, kv_cache_spec, kv_cache_group_id, attn_metadata_builders
                )
                attn_groups.append(attn_group)
            return attn_groups

        attention_backend_maps = []
        attention_backend_list = []
        for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
            attn_backends = get_attn_backends_for_group(kv_cache_group_spec)
            attention_backend_maps.append(attn_backends[0])
            attention_backend_list.append(attn_backends[1])

        self._check_and_update_cudagraph_mode(attention_backend_list, kv_cache_config.kv_cache_groups)

        for i, kv_cache_group_spec in enumerate(kv_cache_config.kv_cache_groups):
            attn_backends = get_attn_backends_for_group(  # type: ignore
                kv_cache_group_spec
            )
            self.attn_groups.append(create_attn_groups(attn_backends[0], i))

        # Calculate reorder batch threshold (if needed)
        self.calculate_reorder_batch_threshold()

    def calculate_reorder_batch_threshold(self) -> None:
        """
        Check that if any backends reorder batches; that the reordering
        is compatible (e.g., decode threshold is the same)
        """
        for group in self._attn_group_iterator():
            attn_metadata_builder_i = group.get_metadata_builder()
            if hasattr(attn_metadata_builder_i, "reorder_batch_threshold"):  # noqa
                # check that if any backends reorder batches; that the reordering
                # is compatible (e.g., decode threshold is the same)
                reorder_batch_threshold_i = attn_metadata_builder_i.reorder_batch_threshold
                if reorder_batch_threshold_i is not None:  # noqa
                    if self.reorder_batch_threshold is not None:
                        if reorder_batch_threshold_i != self.reorder_batch_threshold:
                            raise ValueError(
                                f"Attention backend reorders decodes with "
                                f"threshold {reorder_batch_threshold_i} but other "
                                f"backend uses threshold "
                                f"{self.reorder_batch_threshold}"
                            )
                    else:
                        self.reorder_batch_threshold = reorder_batch_threshold_i  # noqa

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """

        if has_ec_transfer() and get_ec_transfer().is_producer:
            return {}

        kv_cache_spec: dict[str, KVCacheSpec] = {}
        attn_layers = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase)
        # NOTE: Must process Attention/MLAAttention before MambaBase to maintain
        # ordering expected by graph parameter update logic in attention backends.
        mamba_layers: dict[str, MambaBase] = {}
        attn_layer_names = set()
        for layer_name, attn_module in attn_layers.items():
            if isinstance(attn_module, Attention):
                if (kv_tgt_layer := attn_module.kv_sharing_target_layer_name) is not None:
                    # The layer doesn't need its own KV cache and will use that of
                    # the target layer. We skip creating a KVCacheSpec for it, so
                    # that KV cache management logic will act as this layer does
                    # not exist, and doesn't allocate KV cache for the layer. This
                    # enables the memory saving of cross-layer kv sharing, allowing
                    # a given amount of memory to accommodate longer context lengths
                    # or enable more requests to be processed simultaneously.
                    self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
                    continue

                if spec := attn_module.get_kv_cache_spec(self.vllm_config):
                    kv_cache_spec[layer_name] = spec
                    attn_layer_names.add(layer_name)

            elif isinstance(attn_module, MLAAttention):
                model_spec = attn_module.get_kv_cache_spec(self.vllm_config)
                dsa_kv_registration = model_spec.dsa_kv_registration
                if self.use_sparse:
                    # `MLAAttentionSpec` is temporarily patched to `AscendMLAAttentionSpec`.
                    # Re-importing it at runtime will therefore resolve to the patched class.
                    # Rename it here to make this behavior explicit.
                    from vllm.v1.kv_cache_interface import MLAAttentionSpec as AscendMLAAttentionSpec
                    # TODO(rjg-lyh): when kv_cache_spec's refactor is ready,
                    # implement it by creating a new kv_cache_spec class
                    if self.dsa_unbundle:
                        # Proper route P1: this MLA layer owns ONLY the latent
                        # (k_nope + k_pe); the indexer key is a separate KV group
                        # emitted from the DeepseekV32IndexerCache layer below.
                        kv_lora_rank, qk_rope_head_dim = self.sparse_head_dim[:2]
                        kv_cache_spec[layer_name] = AscendMLAAttentionSpec(
                            block_size=self.block_size,
                            num_kv_heads=1,
                            head_size=kv_lora_rank + qk_rope_head_dim,
                            sparse_head_dim=(kv_lora_rank, qk_rope_head_dim),
                            dtype=self.kv_cache_dtype,
                            cache_dtype_str=self.vllm_config.cache_config.cache_dtype,
                            cache_sparse_c8=False,
                            dsa_kv_registration=dsa_kv_registration,
                        )
                    elif self.dsa_free_paged:
                        # DSA offload (M-B): paged cache holds ONLY the indexer key;
                        # latent lives in the PagedLatentPool. Per-token page shrinks
                        # from sum(704) to index_head_dim -> ~5.5x more blocks.
                        if not getattr(attn_module.impl, "has_indexer", True):
                            raise NotImplementedError(
                                "DSA free-paged offload requires every sparse "
                                "layer to own an indexer; shared-indexer "
                                f"consumer layer {layer_name} has none."
                            )
                        index_head_dim = self.sparse_head_dim[-1]
                        kv_cache_spec[layer_name] = AscendMLAAttentionSpec(
                            block_size=self.block_size,
                            num_kv_heads=1,
                            head_size=index_head_dim,
                            sparse_head_dim=(index_head_dim,),
                            dtype=self.kv_cache_dtype,
                            cache_dtype_str=self.vllm_config.cache_config.cache_dtype,
                            cache_sparse_c8=False,
                            dsa_kv_registration=dsa_kv_registration,
                        )
                    elif not getattr(attn_module.impl, "has_indexer", True):
                        # Bundled shared-indexer consumer (GLM-5.2): the layer
                        # reuses a producer's top-k and owns no indexer key, so
                        # its page covers only the latent (k_nope + k_pe).
                        kv_lora_rank, qk_rope_head_dim = self.sparse_head_dim[:2]
                        kv_cache_spec[layer_name] = AscendMLAAttentionSpec(
                            block_size=self.block_size,
                            num_kv_heads=1,
                            head_size=kv_lora_rank + qk_rope_head_dim,
                            sparse_head_dim=(kv_lora_rank, qk_rope_head_dim, 0),
                            dtype=self.kv_cache_dtype,
                            cache_dtype_str=self.vllm_config.cache_config.cache_dtype,
                            cache_sparse_c8=False,
                            dsa_kv_registration=dsa_kv_registration,
                        )
                    else:
                        kv_cache_spec[layer_name] = AscendMLAAttentionSpec(
                            block_size=self.block_size,
                            num_kv_heads=1,
                            head_size=sum(self.sparse_head_dim),
                            sparse_head_dim=self.sparse_head_dim,
                            dtype=self.kv_cache_dtype,
                            cache_dtype_str=self.vllm_config.cache_config.cache_dtype,
                            cache_sparse_c8=self.use_sparse_c8_indexer,
                            dsa_kv_registration=dsa_kv_registration,
                        )
                else:
                    assert isinstance(model_spec, MLAAttentionSpec)
                    from vllm.v1.kv_cache_interface import MLAAttentionSpec as AscendMLAAttentionSpec
                    if getattr(attn_module.impl, "fa_quant_layer", False):
                        head_size = attn_module.head_size + attn_module.qk_rope_head_dim
                        dtype, cache_dtype_str = attn_module.impl.dtype, None
                    else:
                        head_size = model_spec.head_size
                        dtype = model_spec.dtype
                        cache_dtype_str = model_spec.cache_dtype_str
                    kv_cache_spec[layer_name] = AscendMLAAttentionSpec(
                        block_size=model_spec.block_size,
                        num_kv_heads=model_spec.num_kv_heads,
                        head_size=head_size,
                        dtype=dtype,
                        cache_dtype_str=cache_dtype_str,
                        dsa_kv_registration=dsa_kv_registration,
                    )

            elif (
                self.dsa_unbundle
                and (registration := getattr(attn_module, "dsa_kv_registration", None))
                is not None
                and registration.kv_group == 1
            ):
                # Proper route P1: the indexer key cache becomes its own KV group
                # (so the latent group's blocks can be freed independently later).
                from vllm.v1.kv_cache_interface import MLAAttentionSpec as AscendMLAAttentionSpec
                index_head_dim = self.sparse_head_dim[-1]
                model_spec = attn_module.get_kv_cache_spec(self.vllm_config)
                kv_cache_spec[layer_name] = AscendMLAAttentionSpec(
                    block_size=self.block_size,
                    num_kv_heads=1,
                    head_size=index_head_dim,
                    sparse_head_dim=(index_head_dim,),
                    dtype=self.kv_cache_dtype,
                    cache_dtype_str=self.vllm_config.cache_config.cache_dtype,
                    cache_sparse_c8=self.use_sparse_c8_indexer,
                    dsa_kv_registration=registration,
                )

            elif isinstance(attn_module, MambaBase):
                mamba_layers[layer_name] = attn_module

        if len(mamba_layers) > 0:
            mamba_page_size_padded = 0
            for layer_name, mamba_module in mamba_layers.items():
                if spec := mamba_module.get_kv_cache_spec(self.vllm_config):
                    kv_cache_spec[layer_name] = spec
                    mamba_page_size_padded = spec.page_size_bytes
            # align attn_page_size to mamba_page_size_padded
            for layer_name in attn_layer_names:
                if kv_cache_spec[layer_name].page_size_bytes < mamba_page_size_padded:
                    object.__setattr__(kv_cache_spec[layer_name], "page_size_padded", mamba_page_size_padded)

        self._dsa_kv_specs_by_layer_name = {
            name: spec
            for name, spec in kv_cache_spec.items()
            if spec.dsa_kv_registration is not None
        }

        if self.use_sparse and not self.dsa_two_groups:
            # Startup artifact for shared-indexer models (GLM-5.2): report the
            # registered cache topology derived from runtime construction. The
            # expected values (e.g. 79 LATENT / 22 INDEXER with MTP1) are a
            # validation gate, never hardcoded production constants.
            latent_names = [
                name
                for name, spec in kv_cache_spec.items()
                if isinstance(spec, MLAAttentionSpec)
                and "indexer" not in name
                and getattr(spec, "sparse_head_dim", None) is not None
            ]
            indexer_names = [
                name
                for name, spec in kv_cache_spec.items()
                if isinstance(spec, MLAAttentionSpec) and "indexer" in name
            ]
            if indexer_names:
                num_hidden_layers = getattr(
                    self.model_config.hf_text_config, "num_hidden_layers", None
                )
                model_layers = sorted(
                    {
                        int(name.split(".")[2])
                        for name in indexer_names + latent_names
                        if len(name.split(".")) > 2 and name.split(".")[2].isdigit()
                    }
                )
                logger.info(
                    "DSA cache registration: latent_layers=%d indexer_layers=%d "
                    "indexer_model_layers=%s mtp_layers=%s",
                    len(latent_names),
                    len(indexer_names),
                    [
                        int(name.split(".")[2])
                        for name in indexer_names
                        if num_hidden_layers is not None
                        and int(name.split(".")[2]) < num_hidden_layers
                    ],
                    [
                        int(name.split(".")[2])
                        for name in indexer_names
                        if num_hidden_layers is not None
                        and int(name.split(".")[2]) >= num_hidden_layers
                    ],
                )
                if len(latent_names) != len(model_layers):
                    raise ValueError(
                        "DSA cache registration is inconsistent: "
                        f"{len(latent_names)} latent layers vs "
                        f"{len(model_layers)} model layers with a registered "
                        "cache. Shared consumers must still register LATENT."
                    )

        return kv_cache_spec

    def _check_and_update_cudagraph_mode(
        self,
        attention_backends: list[set[type[AttentionBackend]]],
        kv_cache_groups: list[KVCacheGroupSpec],
    ) -> None:
        with update_pass_config(self):
            super()._check_and_update_cudagraph_mode(attention_backends, kv_cache_groups)

        # NOTE: Since aclgraph_batch_sizes cannot be determined until here,
        # we set the graph params right before initializing the keys.
        if self.use_aclgraph:
            set_graph_params(self.cudagraph_batch_sizes)
            if self.speculative_config:
                draft_graph_sizes = set(self.cudagraph_batch_sizes)
                if staged_sfa_graph_configured(self.vllm_config):
                    query_width = (
                        1
                        + self.speculative_config.num_speculative_tokens
                    )
                    draft_graph_sizes.update(
                        size // query_width
                        for size in staged_sfa_graph_capture_sizes(
                            self.vllm_config
                        )
                    )
                set_draft_graph_params(sorted(draft_graph_sizes))



    def _collect_staged_sfa_impls(self) -> tuple[tuple[str, Any], ...]:
        """Return each target-model staged SFA implementation exactly once."""
        attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,
        )
        target_layer_count = getattr(
            self.vllm_config.model_config.hf_text_config,
            "num_hidden_layers",
            None,
        )
        staged_impls: dict[int, tuple[str, Any]] = {}
        for layer_name, attn_layer in attn_layers.items():
            impl = getattr(attn_layer, "impl", None)
            if impl is None or not getattr(
                impl,
                "enable_staged_sfa_graph",
                False,
            ):
                continue
            canonical_name = getattr(
                attn_layer,
                "layer_name",
                layer_name,
            )
            layer_index = parse_layer_idx(canonical_name)
            if (
                isinstance(target_layer_count, int)
                and layer_index is not None
                and layer_index >= target_layer_count
            ):
                logger.info(
                    "[SFA cross-layer graph] excluding non-target attention "
                    "from the target capture registry: layer=%s index=%d "
                    "target_layers=%d",
                    canonical_name,
                    layer_index,
                    target_layer_count,
                )
                continue
            staged_impls.setdefault(
                id(impl),
                (canonical_name, impl),
            )
        return tuple(staged_impls.values())

    def _reset_staged_sfa_startup_capture(self) -> None:
        """Discard staged state before the one real startup capture."""
        self._staged_sfa_impls = ()
        for _layer_name, impl in self._collect_staged_sfa_impls():
            impl.reset_staged_sfa_capture()





    def capture_model(self) -> int:
        staged_graph_configured = staged_sfa_graph_configured(
            self.vllm_config
        )
        if staged_graph_configured and getattr(
            self,
            "_staged_sfa_startup_capture_attempted",
            False,
        ):
            raise RuntimeError(
                "[SFA cross-layer graph] startup graph capture was already "
                "attempted; retrying could reuse stale outer graph entries."
            )
        if staged_graph_configured:
            self._staged_sfa_startup_capture_attempted = True
            self._reset_staged_sfa_startup_capture()
        with _torch_cuda_wrapper(), _replace_gpu_model_runner_function_wrapper(
            GPUModelRunner.__module__,
        ):
            graph_memory_bytes = GPUModelRunner.capture_model(self)
        if staged_graph_configured:
            self._staged_sfa_impls = self._collect_staged_sfa_impls()
            if not self._staged_sfa_impls:
                raise RuntimeError(
                    "[SFA cross-layer graph] no local SFA layers were captured"
                )
            query_width = 1 + int(
                getattr(
                    self.vllm_config.speculative_config,
                    "num_speculative_tokens",
                    0,
                )
            )
            runtime_query_width = int(
                getattr(self, "decode_threshold", query_width)
            )
            if runtime_query_width != query_width:
                raise RuntimeError(
                    "[SFA cross-layer graph] configured and runtime query "
                    "widths differ: "
                    f"configured={query_width}, runtime={runtime_query_width}"
                )
            capture_sizes = staged_sfa_graph_capture_sizes(self.vllm_config)
            if query_width > 1 and any(
                size % query_width for size in capture_sizes
            ):
                raise RuntimeError(
                    "[SFA cross-layer graph] fixed-width MTP capture sizes "
                    f"must be divisible by query_width={query_width}: "
                    f"sizes={capture_sizes}"
                )
            graph_keys = tuple(
                (
                    StagedSFAGraphKey.exact_q1(size)
                    if query_width == 1
                    else StagedSFAGraphKey.fixed_spec(
                        size // query_width,
                        query_width,
                    )
                )
                for size in capture_sizes
            )
            for layer_name, impl in self._staged_sfa_impls:
                try:
                    impl.seal_staged_sfa_capture(graph_keys)
                except RuntimeError as exc:
                    raise RuntimeError(
                        "[SFA cross-layer graph] eager warmup/capture was "
                        f"incomplete for {layer_name}: {exc}"
                    ) from exc
            # The normal retrieve split creates one outer island per target
            # layer plus the model tail.  Target diagnostics add graph-external
            # input and output boundaries around every target layer, creating
            # two additional islands per layer.  Keep the exact-count check so
            # a genuinely incomplete debug capture still fails at startup.
            expected_outer_islands = len(self._staged_sfa_impls) + 1
            if envs_ascend.VLLM_ASCEND_MTP_DRAFT_DEBUG:
                expected_outer_islands += 2 * len(self._staged_sfa_impls)
            graph_entry_count = ACLGraphWrapper.seal_staged_entries(
                graph_keys,
                expected_outer_islands,
            )
            draft_graph_count = 0
            if (
                getattr(self, "drafter", None) is not None
                and getattr(
                    self.drafter,
                    "use_staged_mtp_draft_graph",
                    False,
                )
            ):
                draft_graph_count = (
                    self.drafter.seal_staged_mtp_draft_graphs(
                        tuple(
                            graph_key.request_capacity
                            for graph_key in graph_keys
                        )
                    )
                )
            logger.info(
                "[SFA cross-layer graph] captured retrieve-split outer graphs "
                "for %d local SFA layers and %d keys; entries=%d, "
                "draft_full_graphs=%d",
                len(self._staged_sfa_impls),
                len(graph_keys),
                graph_entry_count,
                draft_graph_count,
            )
        return graph_memory_bytes

    def profile_cudagraph_memory(self) -> int:
        """Measure staged ACL graphs before final KV-cache sizing."""
        if not staged_sfa_graph_configured(self.vllm_config):
            raise RuntimeError(
                "ACL graph memory profiling is only enabled for staged SFA"
            )
        original_pools = {
            wrapper: wrapper.graph_pool
            for wrapper in ACLGraphWrapper._all_instances
        }
        saved_num_cudagraph_captured = (
            compilation_counter.num_cudagraph_captured
        )
        completed = False
        self._profiling_cudagraph_memory = True
        try:
            with (
                _torch_cuda_wrapper(),
                _replace_gpu_model_runner_function_wrapper(
                    GPUModelRunner.__module__,
                ),
            ):
                result = GPUModelRunner.profile_cudagraph_memory(self)
                completed = True
                return result
        finally:
            self._profiling_cudagraph_memory = False
            reset_graph_params()
            if not completed:
                set_cudagraph_capturing_enabled(False)
                ACLGraphWrapper.clear_all_graphs()
                for wrapper, graph_pool in original_pools.items():
                    wrapper.graph_pool = graph_pool
                for key_set in self.cudagraph_dispatcher.cudagraph_keys.values():
                    key_set.clear()
                self.cudagraph_dispatcher.keys_initialized = False
                compilation_counter.num_cudagraph_captured = (
                    saved_num_cudagraph_captured
                )
                self._cleanup_profiling_kv_cache()
            self._reset_staged_sfa_startup_capture()

    def _prepare_multimodal_fields(self):
        """
        Ensures specific multimodal tensors are on CPU.
        This is necessary for fields like 'grid_thw' which are converted to numpy
        inside the model's forward pass.
        """
        if not self.multimodal_cpu_fields:
            return

        req_ids = self.input_batch.req_ids
        for req_id in req_ids:
            req = self.requests.get(req_id)
            if req is None:
                continue

            mm_data = getattr(req, "multimodal_data", None)
            if not mm_data:
                continue

            for field in self.multimodal_cpu_fields:
                if field in mm_data:
                    tensor = mm_data[field]
                    if isinstance(tensor, torch.Tensor) and tensor.device.type != "cpu":
                        mm_data[field] = tensor.cpu()


def _post_process_cudagraph_mode(tensor: torch.Tensor) -> int:
    """
    Synchronize cudagraph_mode across DP ranks by taking the minimum.
    If any rank has NONE (0), all ranks use NONE.
    This ensures all ranks send consistent values (all padded or all unpadded).
    """
    return int(tensor[1, :].min().item())


@contextmanager
def _torch_cuda_wrapper():
    class _EventPlaceholder:
        def __init__(self, *args, **kwargs) -> None:
            self.record = lambda: None
            self.synchronize = lambda: None

    class _StreamPlaceholder:
        def __init__(self, *args, **kwargs) -> None:
            pass

    try:
        # replace cuda APIs with xpu APIs, this should work by default
        torch.Event = torch.npu.Event
        torch.cuda.Event = torch.npu.Event
        torch.cuda.Stream = torch.npu.Stream
        torch.cuda.default_stream = torch.npu.default_stream
        torch.cuda.current_stream = torch.npu.current_stream
        torch.cuda.stream = torch.npu.stream
        torch.cuda.synchronize = torch.npu.synchronize
        torch.cuda.mem_get_info = torch.npu.mem_get_info
        yield
    except Exception as e:
        torch.cuda.Event = _EventPlaceholder
        torch.cuda.Stream = _StreamPlaceholder
        torch.cuda.default_stream = _StreamPlaceholder
        torch.cuda.current_stream = _StreamPlaceholder
        torch.cuda.stream = _StreamPlaceholder
        torch.cuda.synchronize = _StreamPlaceholder
        torch.cuda.mem_get_info = _StreamPlaceholder
        raise RuntimeError(f"NPUModelRunner init failed, error is {e}")
    finally:
        # if anything goes wrong, just patch it with a placeholder
        torch.cuda.Event = _EventPlaceholder
        torch.cuda.Stream = torch.cuda.Stream
        torch.cuda.default_stream = torch.npu.default_stream
        torch.cuda.current_stream = torch.npu.current_stream
        torch.cuda.stream = torch.npu.stream
        torch.cuda.synchronize = torch.npu.synchronize
        torch.cuda.mem_get_info = torch.npu.mem_get_info


@contextmanager
def _replace_gpu_model_runner_function_wrapper(target_module_name):
    target_module = sys.modules[target_module_name]
    original_graph_capture = target_module.graph_capture
    original_graph_wrapper = getattr(
        target_module,
        "CUDAGraphWrapper",
        None,
    )
    try:
        setattr(target_module, "graph_capture", graph_capture)  # noqa: B010
        setattr(target_module, "CUDAGraphWrapper", ACLGraphWrapper)  # noqa: B010
        yield
    except Exception as e:
        raise RuntimeError(f"NPUModelRunner failed, error is {e}")
    finally:
        target_module.graph_capture = original_graph_capture
        if original_graph_wrapper is None:
            delattr(target_module, "CUDAGraphWrapper")
        else:
            target_module.CUDAGraphWrapper = original_graph_wrapper


# TODO: remove it when flash_comm1 is removed
@contextmanager
def update_pass_config(model_runner):
    try:
        original_pass_config_sp = model_runner.compilation_config.pass_config.enable_sp
        model_runner.compilation_config.pass_config.enable_sp = enable_sp(model_runner.vllm_config)
        yield
    finally:
        model_runner.compilation_config.pass_config.enable_sp = original_pass_config_sp
