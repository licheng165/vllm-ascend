# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

import re

import torch
from torch import nn
from vllm.config import CacheConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import AttentionMetadata  # type: ignore

from vllm_ascend import envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.utils import is_vl_model, parse_layer_idx, vllm_version_is


class IndexerWrapper(nn.Module):
    """
    A wrapper of Indexer for Deepseek v3.2.
    This wrapper is currently used to solve the fp8 hard code issue of vllm's deepseek_v2.py.
    It wraps the original Indexer, inherits its module weights
    (including wq_b, wk, weights_proj, k_norm)
    while deletes the unused topk_indices_buffer and k_cache to save memory.
    TODO: Will be removed once original Indexer supports different quantization methods.
    """

    def __init__(self, vllm_indexer: nn.Module) -> None:
        super().__init__()

        self.n_head: int = vllm_indexer.n_head  # 64
        self.head_dim: int = vllm_indexer.head_dim  # 128
        self.topk_tokens: int = vllm_indexer.topk_tokens  # 2048
        self.q_lora_rank: int = vllm_indexer.q_lora_rank  # 1536
        self.wq_b = vllm_indexer.wq_b
        self.wk = vllm_indexer.wk
        self.weights_proj = vllm_indexer.weights_proj
        self.k_norm = vllm_indexer.k_norm
        self.softmax_scale = vllm_indexer.softmax_scale
        vllm_indexer.topk_indices_buffer = None  # delete topk_indices_buffer
        vllm_indexer.k_cache = None  # delete k_cache

    def forward(self):
        return


class AscendMultiHeadLatentAttention(MultiHeadLatentAttentionWrapper):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        mla_modules: MLAModules,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.prefix = prefix
        hf_config = get_current_vllm_config().model_config.hf_text_config
        self.enable_shared_expert_dp = get_ascend_config().enable_shared_expert_dp
        self.tp_size = get_tensor_model_parallel_world_size()
        self.layers = hf_config.num_hidden_layers
        if mla_modules.indexer is not None:
            ascend_indexer = IndexerWrapper(mla_modules.indexer)
        else:
            ascend_indexer = None
        # Shared-indexer (GLM-5.2): IndexerWrapper drops the original Indexer's
        # topk_indices_buffer, so the shared buffer must be captured from
        # mla_modules (which holds an independent reference) and passed to the
        # SFA impl explicitly. Shared-consumer layers (indexer=None, skip_topk)
        # reach the producer-written buffer only through this path.
        self.skip_topk = mla_modules.skip_topk
        self.topk_indices_buffer = mla_modules.topk_indices_buffer
        self.mla_attn = MLAAttention(
            num_heads=num_heads,
            scale=scale,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            kv_b_proj=mla_modules.kv_b_proj,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            use_sparse=mla_modules.is_sparse,
            indexer=ascend_indexer,
            topk_indices_buffer=self.topk_indices_buffer,
            skip_topk=self.skip_topk,
            # extra args
            rotary_emb=mla_modules.rotary_emb,
            fused_qkv_a_proj=mla_modules.fused_qkv_a_proj,
            q_b_proj=mla_modules.q_b_proj,
            q_a_layernorm=mla_modules.q_a_layernorm,
            q_proj=mla_modules.q_proj,
            kv_a_proj_with_mqa=mla_modules.kv_a_proj_with_mqa,
            kv_a_layernorm=mla_modules.kv_a_layernorm,
            o_proj=mla_modules.o_proj,
            layer_name=f"{prefix}.attn",
        )

        original_process_weights = self.mla_attn.process_weights_after_loading

        def wrapped_process_weights(act_dtype: torch.dtype):
            from vllm_ascend.attention.sfa_v1 import AscendSFAImpl

            if not isinstance(self.mla_attn.impl, AscendSFAImpl):
                original_process_weights(act_dtype)
            self.mla_attn.impl.process_weights_after_loading(act_dtype)

        self.mla_attn.process_weights_after_loading = wrapped_process_weights

        # For VL models (e.g. Kimi K2.5), inputs_embeds at layer 0 comes from
        # the vision encoder as full [N, H] — it has NOT been reduce-scattered.
        # We detect this statically at init time (not at runtime via shape checks,
        # which break graph-mode compilation) so the branch is a constant to dynamo.
        vllm_config = get_current_vllm_config()
        _is_vl = is_vl_model(vllm_config)
        _layer_idx = parse_layer_idx(prefix)
        self.is_vl_first_layer = bool(_is_vl and _layer_idx == 0)
        self.next_layer_name = (
            re.sub(r"layers\.\d+", f"layers.{_layer_idx + 1}", prefix, count=1) + ".attn"
            if _layer_idx is not None and _layer_idx + 1 < self.layers
            else ""
        )
        self.use_cross_layer_sfa = getattr(self.mla_attn.impl, "enable_staged_sfa_graph", False) is True
        self.target_sfa_debug = bool(
            self.use_cross_layer_sfa
            and envs_ascend.VLLM_ASCEND_MTP_DRAFT_DEBUG
        )

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor | None = None,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        hidden_dim = hidden_states.shape[-1]

        if _EXTRA_CTX.flash_comm_v1_enabled and self.tp_size > 1 and self.is_vl_first_layer:
            need_gather_q_kv = False
            n_out = hidden_states.shape[0] // self.tp_size
            output = torch.empty((n_out, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
        else:
            need_gather_q_kv = _EXTRA_CTX.flash_comm_v1_enabled
            output = torch.empty(hidden_states.shape, dtype=hidden_states.dtype, device=hidden_states.device)

        if self.use_cross_layer_sfa:
            impl = self.mla_attn.impl
            if self.target_sfa_debug:
                torch.ops.vllm.sfa_target_layer_diag(
                    hidden_states,
                    self.prefix,
                    "input",
                )
            (
                ql_nope,
                q_pe,
                topk_indices,
                selected_packed,
                selected_counts,
                target_slots,
            ) = torch.ops.vllm.sfa_forward_pre(
                hidden_states,
                need_gather_q_kv,
                output,
                self.prefix,
                impl.local_num_heads,
                impl.kv_lora_rank,
                impl.qk_rope_head_dim,
                impl.index_topk,
                impl._staged_sfa_graph_capture_sizes[-1],
                (
                    impl._staged_sfa_graph_capture_sizes[-1]
                    // impl.decode_threshold
                ),
                impl.decode_threshold * impl.index_topk,
            )
            torch.ops.vllm.sfa_lmcache_retrieve(
                selected_packed,
                selected_counts,
                target_slots,
                output,
                self.prefix,
                self.next_layer_name,
            )
            torch.ops.vllm.sfa_forward_post(
                ql_nope,
                q_pe,
                topk_indices,
                selected_packed,
                selected_counts,
                target_slots,
                output,
                self.prefix,
            )
            if self.target_sfa_debug:
                torch.ops.vllm.sfa_target_layer_diag(
                    output,
                    self.prefix,
                    "output",
                )
        else:
            torch.ops.vllm.mla_forward(hidden_states, need_gather_q_kv, output, self.prefix)
        output = output.view(-1, hidden_dim)
        return output


def mla_forward(
    hidden_states: torch.Tensor,
    need_gather_q_kv: bool,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    if forward_context.attn_metadata:
        attn_metadata = forward_context.attn_metadata[self.mla_attn.layer_name]
    else:
        attn_metadata = forward_context.attn_metadata
    kv_cache = self.mla_attn.kv_cache[forward_context.virtual_engine if vllm_version_is("0.18.0") else 0]
    self.mla_attn.impl.forward(
        self.mla_attn.layer_name, hidden_states, kv_cache, attn_metadata, need_gather_q_kv, output
    )
    return


def mla_forward_fake(
    hidden_states: torch.Tensor,
    need_gather_q_kv: bool,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="mla_forward",
    op_func=mla_forward,
    mutates_args=["output"],
    fake_impl=mla_forward_fake,
    dispatch_key="PrivateUse1",
)


def _mla_runtime_metadata(layer_name: str):
    forward_context: ForwardContext = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]
    attn_metadata = (
        forward_context.attn_metadata[layer.mla_attn.layer_name]
        if forward_context.attn_metadata
        else forward_context.attn_metadata
    )
    return forward_context, layer, attn_metadata


def _mla_runtime_state(layer_name: str):
    forward_context, layer, attn_metadata = _mla_runtime_metadata(layer_name)
    virtual_engine = forward_context.virtual_engine if vllm_version_is("0.18.0") else 0
    return layer.mla_attn.impl, layer.mla_attn.layer_name, layer.mla_attn.kv_cache[virtual_engine], attn_metadata


StagedSFABridge = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def sfa_forward_pre(
    hidden_states: torch.Tensor,
    need_gather_q_kv: bool,
    output: torch.Tensor,
    layer_name: str,
    local_num_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    index_topk: int,
    token_capacity: int,
    request_capacity: int,
    scratch_capacity: int,
) -> StagedSFABridge:
    impl, attn_layer_name, kv_cache, attn_metadata = _mla_runtime_state(layer_name)
    return impl.cross_layer_graph_pre(
        attn_layer_name,
        hidden_states,
        kv_cache,
        attn_metadata,
        need_gather_q_kv,
        output,
    )


def sfa_forward_pre_fake(
    hidden_states: torch.Tensor,
    need_gather_q_kv: bool,
    output: torch.Tensor,
    layer_name: str,
    local_num_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    index_topk: int,
    token_capacity: int,
    request_capacity: int,
    scratch_capacity: int,
) -> StagedSFABridge:
    return (
        hidden_states.new_empty((token_capacity, local_num_heads, kv_lora_rank)),
        hidden_states.new_empty((token_capacity, local_num_heads, qk_rope_head_dim)),
        torch.empty(
            (token_capacity, 1, index_topk),
            dtype=torch.int32,
            device=hidden_states.device,
        ),
        torch.empty(
            (request_capacity, scratch_capacity),
            dtype=torch.int32,
            device=hidden_states.device,
        ),
        torch.empty(
            (request_capacity,),
            dtype=torch.int32,
            device=hidden_states.device,
        ),
        torch.empty(
            (request_capacity, scratch_capacity),
            dtype=torch.long,
            device=hidden_states.device,
        ),
    )


def sfa_lmcache_retrieve(
    selected_packed: torch.Tensor,
    selected_counts: torch.Tensor,
    target_slots: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    next_layer_name: str,
) -> None:
    context, layer, attn_metadata = _mla_runtime_metadata(layer_name)
    layer.mla_attn.impl.cross_layer_lmcache_retrieve(
        layer.mla_attn.layer_name,
        next_layer_name,
        selected_packed,
        selected_counts,
        target_slots,
        attn_metadata,
        context,
    )


def sfa_lmcache_retrieve_fake(
    selected_packed: torch.Tensor,
    selected_counts: torch.Tensor,
    target_slots: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    next_layer_name: str,
) -> None:
    return


def sfa_forward_post(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    topk_indices: torch.Tensor,
    selected_packed: torch.Tensor,
    selected_counts: torch.Tensor,
    target_slots: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    impl, attn_layer_name, kv_cache, attn_metadata = _mla_runtime_state(layer_name)
    impl.cross_layer_graph_post(
        attn_layer_name,
        ql_nope,
        q_pe,
        topk_indices,
        kv_cache,
        attn_metadata,
        output,
    )


def sfa_forward_post_fake(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    topk_indices: torch.Tensor,
    selected_packed: torch.Tensor,
    selected_counts: torch.Tensor,
    target_slots: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


def sfa_target_layer_diag(
    tensor: torch.Tensor,
    layer_name: str,
    phase: str,
) -> None:
    context, layer, attn_metadata = _mla_runtime_metadata(layer_name)
    layer.mla_attn.impl.target_sfa_diagnostic_boundary(
        layer.mla_attn.layer_name,
        phase,
        tensor,
        attn_metadata,
        context,
    )


def sfa_target_layer_diag_fake(
    tensor: torch.Tensor,
    layer_name: str,
    phase: str,
) -> None:
    return


direct_register_custom_op(
    op_name="sfa_forward_pre",
    op_func=sfa_forward_pre,
    mutates_args=["output"],
    fake_impl=sfa_forward_pre_fake,
    dispatch_key="PrivateUse1",
)
direct_register_custom_op(
    op_name="sfa_lmcache_retrieve",
    op_func=sfa_lmcache_retrieve,
    mutates_args=["output"],
    fake_impl=sfa_lmcache_retrieve_fake,
    dispatch_key="PrivateUse1",
)
direct_register_custom_op(
    op_name="sfa_forward_post",
    op_func=sfa_forward_post,
    mutates_args=["output"],
    fake_impl=sfa_forward_post_fake,
    dispatch_key="PrivateUse1",
)
direct_register_custom_op(
    op_name="sfa_target_layer_diag",
    op_func=sfa_target_layer_diag,
    mutates_args=["tensor"],
    fake_impl=sfa_target_layer_diag_fake,
    dispatch_key="PrivateUse1",
)
