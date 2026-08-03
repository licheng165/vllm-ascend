import math
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import vllm.envs as envs_vllm
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_dp_group, get_ep_group, get_tensor_model_parallel_world_size
from vllm.forward_context import BatchDescriptor, get_forward_context, set_forward_context

import vllm_ascend.envs as envs_ascend
from vllm_ascend.utils import (
    AscendDeviceType,
    StagedSFARouteDecision,
    enable_sp,
    flashcomm2_enable,
    get_ascend_device_type,
    has_layer_idx,
    is_drafter_moe_model,
    is_moe_model,
    speculative_enable_dispatch_gmm_combine_decode,
    vllm_version_is,
)


class MoECommType(Enum):
    ALLGATHER = 0
    MC2 = 1
    ALLTOALL = 2
    FUSED_MC2 = 3


class StagedSFAQueryProfile(str, Enum):
    """Structural query layouts that require distinct staged SFA graphs."""

    DECODE_Q1 = "decode_q1"
    SPEC_FIXED = "spec_fixed"


@dataclass(frozen=True)
class StagedSFAGraphKey:
    """Shape/topology key for a staged SFA outer-graph plan.

    Actual token/request counts and sequence lengths are dynamic buffer
    contents. Capacities and query topology are structural and must never
    collapse onto the same outer graph entry.
    """

    token_capacity: int
    request_capacity: int
    query_profile: StagedSFAQueryProfile
    max_query_len: int

    def __post_init__(self) -> None:
        if self.token_capacity <= 0 or self.request_capacity <= 0 or self.max_query_len <= 0:
            raise ValueError("Staged SFA graph capacities must be positive.")
        if self.query_profile == StagedSFAQueryProfile.DECODE_Q1:
            if (
                self.max_query_len != 1
                or self.token_capacity != self.request_capacity
            ):
                raise ValueError(
                    "DECODE_Q1 requires equal token/request capacity and "
                    "max_query_len=1."
                )
        elif (
            self.max_query_len <= 1
            or self.token_capacity
            != self.request_capacity * self.max_query_len
        ):
            raise ValueError(
                "SPEC_FIXED requires token_capacity == request_capacity * "
                "max_query_len with max_query_len > 1."
            )

    @classmethod
    def exact_q1(cls, size: int) -> "StagedSFAGraphKey":
        """Construct an equal token/request-capacity Q=1 key."""
        return cls(
            token_capacity=size,
            request_capacity=size,
            query_profile=StagedSFAQueryProfile.DECODE_Q1,
            max_query_len=1,
        )

    @classmethod
    def fixed_spec(
        cls,
        request_capacity: int,
        query_width: int,
    ) -> "StagedSFAGraphKey":
        """Construct a fixed-width speculative-decode graph key."""
        if query_width <= 1:
            raise ValueError("SPEC_FIXED query_width must be greater than one.")
        return cls(
            token_capacity=request_capacity * query_width,
            request_capacity=request_capacity,
            query_profile=StagedSFAQueryProfile.SPEC_FIXED,
            max_query_len=query_width,
        )

    def to_legacy_batch_descriptor(self) -> BatchDescriptor:
        """Adapt the structural key to normalized PIECEWISE dispatch."""
        return BatchDescriptor(num_tokens=self.token_capacity)


STAGED_SFA_SINGLETON_GRAPH_KEY = StagedSFAGraphKey.exact_q1(1)


@contextmanager
def set_ascend_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    virtual_engine: int = 0,
    num_tokens: int = 0,
    num_tokens_across_dp: torch.Tensor | None = None,
    in_profile_run: bool = False,
    num_actual_tokens: int | None = None,
    aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    model_instance: torch.nn.Module = None,
    is_draft_model=False,
    skip_compiled: bool = False,
    max_tokens_across_pcp: int = 0,
    draft_attn_metadatas=None,
    dsa_offload_manager=None,
    dsa_req_ids=None,
    dsa_prompt_lens=None,
    dsa_adapter_cache=None,
    mtp_dw_diag_req_ids=None,
    mtp_dw_diag_post_commit_req_ids=None,
    mtp_dw_deep_diag_req_ids=None,
    staged_sfa_graph_dummy_run: bool = False,
    staged_sfa_route: StagedSFARouteDecision | None = None,
    staged_sfa_graph_key: StagedSFAGraphKey | None = None,
):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    We add some additional param into forward_context.
    """
    forward_context_kwargs = {
        "attn_metadata": attn_metadata,
        "vllm_config": vllm_config,
        "num_tokens": num_tokens,
        "num_tokens_across_dp": num_tokens_across_dp,
        "cudagraph_runtime_mode": aclgraph_runtime_mode,
        "batch_descriptor": batch_descriptor,
        "skip_compiled": skip_compiled,
    }
    if vllm_version_is("0.18.0"):
        forward_context_kwargs["virtual_engine"] = virtual_engine

    with set_forward_context(**forward_context_kwargs):
        forward_context = get_forward_context()
        forward_context.draft_attn_metadatas = draft_attn_metadatas
        # DSA latent offload manager (None unless GLM5.1 latent offload is enabled);
        # read by AscendSFAImpl to drive prefill store / decode gather. req_ids and
        # prompt_lens (per request, batch order) come from the runner's input_batch.
        forward_context.dsa_offload_manager = dsa_offload_manager
        forward_context.dsa_req_ids = dsa_req_ids
        forward_context.dsa_prompt_lens = dsa_prompt_lens
        # Adapter-backed latent hot cache (None unless VLLM_ASCEND_DSA_USE_ADAPTER_CACHE).
        forward_context.dsa_adapter_cache = dsa_adapter_cache
        if mtp_dw_diag_req_ids is not None:
            forward_context.mtp_dw_diag_req_ids = mtp_dw_diag_req_ids
            forward_context.mtp_dw_diag_post_commit_req_ids = (
                mtp_dw_diag_post_commit_req_ids
            )
            forward_context.mtp_dw_deep_diag_req_ids = mtp_dw_deep_diag_req_ids
        # True only for the explicit one-token eager warmup / graph-capture
        # passes used by the staged SFA proof of concept. Connector generators
        # and save hooks must not advance during either dummy pass.
        forward_context.staged_sfa_graph_dummy_run = staged_sfa_graph_dummy_run
        forward_context.staged_sfa_route = staged_sfa_route
        forward_context.staged_sfa_graph_key = staged_sfa_graph_key

        from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method

        max_num_tokens = int(num_tokens_across_dp.max().item()) if num_tokens_across_dp is not None else num_tokens
        moe_comm_type = select_moe_comm_method(max_num_tokens, vllm_config, is_draft_model)

        forward_context.moe_comm_type = moe_comm_type
        forward_context.moe_comm_method = get_moe_comm_method(moe_comm_type)

        tp_world_size = get_tensor_model_parallel_world_size()

        forward_context.in_profile_run = in_profile_run

        # NOTE: This cannot be set using set_forward_context
        # due to multiple warmups before actual capturing
        forward_context.capturing = False

        # TODO: remove it when torch_npu.npu_mm_reduce_scatter_base supports tp_size >= 16.
        mmrs_fusion = tp_world_size <= 8

        # set for sequence parallelism, 1000 is the batch size concurrency threshold
        # for enabling the flashcomm_v1 or sequence_parallelism feature.
        # Currently, it is an empirical value. In normal scenarios, if the concurrency
        # exceeds this threshold, the performance benefits can be maximized.
        # Conversely, if the concurrency is below the threshold,
        # the performance may degrade due to the switching of communication methods.

        # main model and drafter model may have different architecture
        is_context_moe_model = is_drafter_moe_model(vllm_config) if is_draft_model else is_moe_model(vllm_config)
        if is_context_moe_model:
            flash_comm_v1_enabled = enable_sp(vllm_config) and num_tokens is not None
            mmrs_fusion = False
        elif is_draft_model:
            # TODO: for dense drafter, `sp` is redundant and is not compatible with `dp` and `graph`.
            # Disable it to avoid more problems.
            flash_comm_v1_enabled = False
        else:
            flash_comm_v1_enabled = enable_sp(vllm_config) and num_tokens is not None and num_tokens > 1000
        forward_context.mmrs_fusion = mmrs_fusion
        forward_context.num_tokens = num_tokens
        forward_context.flash_comm_v1_enabled = flash_comm_v1_enabled
        # TODO(Levi-JQ): another PR to normalize the enabling logic for sp/fc2
        forward_context.flashcomm_v2_enabled = flashcomm2_enable() and tp_world_size > 1 and num_tokens is not None

        forward_context.pad_size = 0
        if forward_context.flash_comm_v1_enabled or forward_context.flashcomm_v2_enabled:
            pad_size = (tp_world_size - (num_tokens % tp_world_size)) % tp_world_size
            forward_context.pad_size = pad_size

        # set this for rope forward_oot using
        forward_context.is_first_layer = True

        # set layer_idx to enable optimization features that depend on this information.
        # This is only applicable to models that contain these necessary attributes.
        forward_context.layer_idx = None
        if has_layer_idx(model_instance):
            forward_context.layer_idx = model_instance.model.start_layer

        forward_context.prefetch_mlp_gate_up_proj = False
        forward_context.prefetch_mlp_down_proj = False
        forward_context.model_instance = model_instance
        forward_context.is_draft_model = is_draft_model

        if num_tokens is None and attn_metadata is not None:
            num_tokens = attn_metadata.num_actual_tokens

        dp_world_size = get_dp_group().world_size
        if dp_world_size > 1 and forward_context.dp_metadata is not None:
            max_tokens_across_dp = forward_context.dp_metadata.max_tokens_across_dp_cpu.item()
            if forward_context.flash_comm_v1_enabled or forward_context.flashcomm_v2_enabled:
                padded_length = (max_tokens_across_dp + tp_world_size - 1) // tp_world_size * tp_world_size
                pad_size = padded_length - num_tokens
                forward_context.padded_length = padded_length
                forward_context.pad_size = pad_size
        else:
            max_tokens_across_dp = num_tokens

        forward_context.max_tokens_across_dp = max_tokens_across_dp
        forward_context.max_tokens_across_pcp = max_tokens_across_pcp

        if num_tokens is not None:
            if num_actual_tokens is None:
                num_actual_tokens = num_tokens
            # NOTE: token num which need to pad to when mc2
            forward_context.padded_num_tokens = math.ceil(max_tokens_across_dp / tp_world_size) * tp_world_size
            reserved_mc2_mask = get_mc2_mask()
            if reserved_mc2_mask is not None:
                mc2_mask = reserved_mc2_mask[: forward_context.padded_num_tokens]
                mc2_mask[:num_actual_tokens] = True
                mc2_mask[num_actual_tokens:] = False
                forward_context.mc2_mask = mc2_mask
        try:
            yield
        finally:
            pass


_mc2_tokens_capacity: int | None = None
_reserved_mc2_mask: torch.Tensor | None = None


def set_mc2_tokens_capacity(vllm_config, max_num_reqs, uniform_decode_query_len):
    global _mc2_tokens_capacity
    if _mc2_tokens_capacity is not None:
        return
    if vllm_config.compilation_config.cudagraph_capture_sizes:
        max_num_tokens = vllm_config.compilation_config.max_cudagraph_capture_size
    else:
        # NOTE: To save memory, we cap the max number of tokens to 512.
        max_num_tokens = min(max_num_reqs * uniform_decode_query_len, 512)
    tp_size = vllm_config.parallel_config.tensor_parallel_size
    # Use integer arithmetic for ceiling division.
    num_tokens_per_tp_rank = (max_num_tokens + tp_size - 1) // tp_size
    _mc2_tokens_capacity = num_tokens_per_tp_rank * tp_size


def get_mc2_tokens_capacity():
    return _mc2_tokens_capacity


def set_mc2_mask(vllm_config, device):
    global _reserved_mc2_mask
    if _reserved_mc2_mask is not None:
        return
    if is_moe_model(vllm_config):
        _reserved_mc2_mask = torch.zeros(get_mc2_tokens_capacity(), dtype=torch.bool, device=device)
    else:
        _reserved_mc2_mask = None


def get_mc2_mask():
    return _reserved_mc2_mask


def select_moe_comm_method(num_tokens: int, vllm_config: VllmConfig, is_draft_model=False) -> MoECommType | None:
    """Select the MoE communication method according to parallel settings,
    device generation, token count, and quantization.

    1. Non-MoE models return `None`.
    2. Without expert parallel, fall back to all-gather.
    3. On A2 with expert parallel, pick MC2 when tokens fit the MC2 capacity
       and the DP size is large enough; otherwise use all-gather.
    4. On A3 with expert parallel, prefer fused MC2 when using w8a8_dynamic
       quantization with small EP size, no dynamic_eplb, and not in MTP
       mode; otherwise use MC2 within capacity or all-to-all.
    5. On 310P, always use all-gather.

    Args:
        num_tokens (int): The number of tokens in the current batch.
        vllm_config (VllmConfig): Runtime configuration for the model.
        is_draft_model (bool): Whether the model runs in MTP mode (disables fused MC2).

    Raises:
        ValueError: If the soc version is unsupported.

    Returns:
        MoECommType | None: The selected MoE communication method.
    """
    if not is_moe_model(vllm_config):
        return None
    mc2_tokens_capacity = get_mc2_tokens_capacity()
    soc_version = get_ascend_device_type()
    quant_type = getattr(
        vllm_config.model_config.hf_text_config,
        "moe_quantize",
        getattr(vllm_config.model_config.hf_text_config, "quantize", None),
    )

    if not vllm_config.parallel_config.enable_expert_parallel or get_ep_group().world_size == 1:
        moe_comm_type = MoECommType.ALLGATHER
    elif soc_version in {AscendDeviceType.A2}:
        num_experts = vllm_config.model_config.get_num_experts()
        ep_world_size = (
            vllm_config.parallel_config.world_size_across_dp // vllm_config.parallel_config.pipeline_parallel_size
        )
        num_experts_per_device = num_experts // ep_world_size
        if num_experts_per_device <= 24 and ep_world_size >= 16 and num_tokens <= mc2_tokens_capacity:
            moe_comm_type = MoECommType.MC2
        else:
            moe_comm_type = MoECommType.ALLGATHER

    elif soc_version in {AscendDeviceType.A3}:
        # TODO: drop the EP-size guard when dispatch_ffn_combine supports larger EP sizes
        # TODO: drop speculative method guard when dispatch_gmm_combine_decode supports w16a16
        fused_mc2_enable = envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2
        dispatch_ffn_combine_enable = get_ep_group().world_size <= 32 and (not is_draft_model)
        if num_tokens <= mc2_tokens_capacity:
            fused_decode_enable = fused_mc2_enable
            if envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2 == 1:
                fused_decode_enable = fused_mc2_enable and dispatch_ffn_combine_enable
            elif envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2 == 2:
                fused_decode_enable = (
                    fused_mc2_enable
                    and speculative_enable_dispatch_gmm_combine_decode(vllm_config)
                    and quant_type == "w8a8_dynamic"
                )
            moe_comm_type = MoECommType.FUSED_MC2 if fused_decode_enable else MoECommType.MC2
        else:
            fused_prefill_enable = fused_mc2_enable
            if envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2 == 1:
                fused_prefill_enable = fused_mc2_enable and dispatch_ffn_combine_enable
            elif envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2 == 2:
                fused_prefill_enable = False
            moe_comm_type = MoECommType.FUSED_MC2 if fused_prefill_enable else MoECommType.ALLTOALL
    elif soc_version in {AscendDeviceType._310P}:
        moe_comm_type = MoECommType.ALLGATHER
    elif soc_version in {AscendDeviceType.A5}:
        if num_tokens <= mc2_tokens_capacity and vllm_config.parallel_config.world_size_across_dp > 1:
            moe_comm_type = MoECommType.MC2
        else:
            moe_comm_type = MoECommType.ALLTOALL
    else:
        raise ValueError(f"Unsupported soc_version: {soc_version}")
    return moe_comm_type


class _ExtraForwardContextProxy:
    """Unified forward-context access for v1/v2 model runners."""

    extra_attrs = (
        "capturing",
        "moe_comm_type",
        "moe_comm_method",
        "mmrs_fusion",
        "num_tokens",
        "flash_comm_v1_enabled",
        "flashcomm_v2_enabled",
        "pad_size",
        "padded_length",
        "num_tokens_across_dp",
        "mc2_mask",
        "is_draft_model",
        "prefetch_mlp_gate_up_proj",
        "prefetch_mlp_down_proj",
        "model_instance",
        "layer_idx",
        "max_tokens_across_dp",
        "max_tokens_across_pcp",
        "num_accept_tokens",
        "in_profile_run",
        "padded_num_tokens",
    )

    def check_extra_attr(self, name: str):
        if name not in self.extra_attrs:
            raise AttributeError(
                f"{name} is not extra forward context attribute, "
                "please get/set it from vllm's _forward_context directly."
            )

    @staticmethod
    def _ctx():
        return get_forward_context()

    def __getattr__(self, name: str) -> Any:
        self.check_extra_attr(name)
        ctx = self._ctx()
        if envs_vllm.VLLM_USE_V2_MODEL_RUNNER:
            return ctx.additional_kwargs[name]
        return getattr(ctx, name)

    def __setattr__(self, name: str, value: Any) -> None:
        self.check_extra_attr(name)
        ctx = self._ctx()
        if envs_vllm.VLLM_USE_V2_MODEL_RUNNER:
            ctx.additional_kwargs[name] = value
        else:
            setattr(ctx, name, value)


# usage: from vllm_ascend.ascend_forward_context import _EXTRA_CTX
_EXTRA_CTX = _ExtraForwardContextProxy()
