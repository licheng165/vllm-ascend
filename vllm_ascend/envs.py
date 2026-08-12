#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# This file is mainly Adapted from vllm-project/vllm/vllm/envs.py
# Copyright 2023 The vLLM team.
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
#

import os
from collections.abc import Callable
from typing import Any

# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition

env_variables: dict[str, Callable[[], Any]] = {
    # max compile thread number for package building. Usually, it is set to
    # the number of CPU cores. If not set, the default value is None, which
    # means all number of CPU cores will be used.
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    # The build type of the package. It can be one of the following values:
    # Release, Debug, RelWithDebugInfo. If not set, the default value is Release.
    "CMAKE_BUILD_TYPE": lambda: os.getenv("CMAKE_BUILD_TYPE"),
    # Whether to compile custom kernels. If not set, the default value is True.
    # If set to False, the custom kernels will not be compiled.
    # This configuration option should only be set to False when running UT
    # scenarios in an environment without an NPU. Do not set it to False in
    # other scenarios.
    "COMPILE_CUSTOM_KERNELS": lambda: bool(int(os.getenv("COMPILE_CUSTOM_KERNELS", "1"))),
    # The CXX compiler used for compiling the package. If not set, the default
    # value is None, which means the system default CXX compiler will be used.
    "CXX_COMPILER": lambda: os.getenv("CXX_COMPILER", None),
    # The C compiler used for compiling the package. If not set, the default
    # value is None, which means the system default C compiler will be used.
    "C_COMPILER": lambda: os.getenv("C_COMPILER", None),
    # The version of the Ascend chip. It's used for package building.
    # If not set, we will query chip info through `npu-smi`.
    # Please make sure that the version is correct.
    "SOC_VERSION": lambda: os.getenv("SOC_VERSION", None),
    # If set, vllm-ascend will print verbose logs during compilation
    "VERBOSE": lambda: bool(int(os.getenv("VERBOSE", "0"))),
    # The home path for CANN toolkit. If not set, the default value is
    # /usr/local/Ascend/ascend-toolkit/latest
    "ASCEND_HOME_PATH": lambda: os.getenv("ASCEND_HOME_PATH", None),
    # The path for HCCL library, it's used by pyhccl communicator backend. If
    # not set, the default value is libhccl.so.
    "HCCL_SO_PATH": lambda: os.getenv("HCCL_SO_PATH", None),
    # The version of vllm is installed. This value is used for developers who
    # installed vllm from source locally. In this case, the version of vllm is
    # usually changed. For example, if the version of vllm is "0.9.0", but when
    # it's installed from source, the version of vllm is usually set to "0.9.1".
    # In this case, developers need to set this value to "0.9.0" to make sure
    # that the correct package is installed.
    "VLLM_VERSION": lambda: os.getenv("VLLM_VERSION", None),
    # Whether to enable MatmulAllReduce fusion kernel when tensor parallel is enabled.
    # this feature is supported in A2, and eager mode will get better performance.
    "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE", "0"))),
    # Whether to enable FlashComm optimization when tensor parallel is enabled.
    # This feature will get better performance when concurrency is large.
    "VLLM_ASCEND_ENABLE_FLASHCOMM1": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "0"))),
    # Whether to enable FLASHCOMM2. Setting it to 0 disables the feature, while setting it to 1 or above enables it.
    # The specific value set will be used as the O-matrix TP group size for flashcomm2.
    # For a detailed introduction to the parameters and the differences and applicable scenarios
    # between this feature and FLASHCOMM1, please refer to the feature guide in the documentation.
    "VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE": lambda: int(os.getenv("VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE", 0)),
    # Whether to enable msMonitor tool to monitor the performance of vllm-ascend.
    "MSMONITOR_USE_DAEMON": lambda: bool(int(os.getenv("MSMONITOR_USE_DAEMON", "0"))),
    # Whether to enable MLAPO optimization for DeepSeek W8A8 series models.
    # This option is enabled by default. MLAPO can improve performance, but
    # it will consume more NPU memory. If reducing NPU memory usage is a higher priority
    # for your DeepSeek W8A8 scene, then disable it.
    "VLLM_ASCEND_ENABLE_MLAPO": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_MLAPO", "1"))),
    # Whether to enable weight cast format to FRACTAL_NZ.
    # 0: close nz;
    # 1: only quant case enable nz;
    # 2: enable nz as long as possible.
    "VLLM_ASCEND_ENABLE_NZ": lambda: int(os.getenv("VLLM_ASCEND_ENABLE_NZ", 1)),
    # Decide whether we should enable CP parallelism.
    "VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL", "0"))),
    # Whether to anbale dynamic EPLB
    "DYNAMIC_EPLB": lambda: os.getenv("DYNAMIC_EPLB", "false").lower(),
    # Whether to enable fused mc2(`dispatch_gmm_combine_decode`/`dispatch_ffn_combine` operator)
    # 0, or not set: default ALLTOALL and MC2 will be used.
    # 1: ALLTOALL and MC2 might be replaced by `dispatch_ffn_combine` operator.
    # `dispatch_ffn_combine` can be used only for moe layer with W8A8, EP<=32, non-mtp, non-dynamic-eplb.
    # 2: MC2 might be replaced by `dispatch_gmm_combine_decode` operator.
    # `dispatch_gmm_combine_decode` can be used only for **decode node** moe layer
    # with W8A8. And MTP layer must be W8A8.
    "VLLM_ASCEND_ENABLE_FUSED_MC2": lambda: int(os.getenv("VLLM_ASCEND_ENABLE_FUSED_MC2", "0")),
    # Whether to anbale balance scheduling
    "VLLM_ASCEND_BALANCE_SCHEDULING": lambda: bool(int(os.getenv("VLLM_ASCEND_BALANCE_SCHEDULING", "0"))),
    # use fused op transpose_kv_cache_by_block, default is True
    "VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK": lambda: bool(
        int(os.getenv("VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK", "1"))
    ),
    # Enable DSA latent KV offload for GLM5.1 (GlmMoeDsa): at prefill end the MLA
    # latent KV is offloaded to LMCache and only the indexer-selected top-k tokens
    # are gathered back per decode step, while the indexer-key cache stays resident.
    # Only effective for DSA / sparse-attention (SFA backend) models. Default off.
    "VLLM_ASCEND_ENABLE_DSA_LATENT_OFFLOAD": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_DSA_LATENT_OFFLOAD", "0"))),
    # DSA latent offload staging gate (bring-up). 0 (Stage 1/2): keep the paged latent
    # write so the offload path can be compared against native sparse attention.
    # 1 (Stage 3): stop writing the paged latent (it lives only in LMCache + decode
    # store) to actually free NPU memory. Default 0. Only effective with the offload
    # flag on AND the SFA kernel-redirect wiring done (see sparse_offload/INTEGRATION.md).
    "VLLM_ASCEND_DSA_OFFLOAD_FREE_PAGED": lambda: bool(int(os.getenv("VLLM_ASCEND_DSA_OFFLOAD_FREE_PAGED", "0"))),
    # DSA latent offload parity check (bring-up). When 1, the SFA decode path runs both
    # the scratch-gather and the native paged sparse attention and logs/asserts their
    # outputs match. Use during Step 2 of bring-up; disable in production (it doubles
    # the attention cost). Default 0.
    "VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY": lambda: bool(int(os.getenv("VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY", "0"))),
    # DSA un-bundle (proper route P1). When 1, the SFA KV cache is split into TWO
    # vLLM-managed KV cache groups: the MLA latent (k_nope+k_pe) and the indexer key,
    # instead of the bundled 3-tuple. This is the prerequisite for freeing the latent
    # blocks after prefill (P2) in a graph-compatible way. Default 0 (bundled path).
    "VLLM_ASCEND_DSA_UNBUNDLE": lambda: bool(int(os.getenv("VLLM_ASCEND_DSA_UNBUNDLE", "0"))),
    # DSA Step A: latent and indexer become two REAL KV cache groups with separate
    # block tables and per-group block pools (the vLLM fork gates its side on the
    # same variable, read as a raw env there). Requires VLLM_ASCEND_DSA_UNBUNDLE=1
    # and --no-enable-prefix-caching. Prerequisite for freeing latent blocks at
    # end of prefill (DSA latent offload P2). Default 0.
    "VLLM_ASCEND_DSA_TWO_GROUPS": lambda: bool(int(os.getenv("VLLM_ASCEND_DSA_TWO_GROUPS", "0"))),
    # DSA shared bundle pool. Requires DSA_UNBUNDLE=1 and DSA_TWO_GROUPS=1.
    # vLLM owns one physical bundle allocator while exposing two logical block
    # tables: latent (k_nope+k_pe) and indexer. vLLM-Ascend backs sibling latent
    # and indexer layers with one raw tensor laid out as
    # [all k_nope pages][all k_pe pages].
    "VLLM_ASCEND_DSA_SHARED_POOL": lambda: bool(int(os.getenv("VLLM_ASCEND_DSA_SHARED_POOL", "1"))),
    # Debug/compat switch: disable DSA indexer LMCache/index-offload hooks.
    # When enabled, unbundled indexer 1-tuple caches stay resident and are not
    # registered with LMCache connectors that cannot permute 1-tuple KV entries.
    "VLLM_ASCEND_DSA_DISABLE_INDEX_LMCACHE": lambda: bool(int(os.getenv("VLLM_ASCEND_DSA_DISABLE_INDEX_LMCACHE", "0"))),
    # DSA Step B staging. Requires TWO_GROUPS=1 + the LMCache connector.
    # 1 (B2): decode reads prefill-selected latent from the compact scratch
    #   (request's first ceil(k/block_size) latent blocks, filled by LMCache);
    #   decode-selected latent is read in place (absolute positions). Latent
    #   blocks are NOT freed yet — outputs must match the resident path.
    # 2 (B2+B1): additionally free the latent blocks [k .. prompt) at end of
    #   prefill (the actual memory saving). Default 0.
    "VLLM_ASCEND_DSA_SHRINK_LATENT": lambda: int(os.getenv("VLLM_ASCEND_DSA_SHRINK_LATENT", "0")),
    # Experimental SFA graph-capture proof of concept. When enabled, exact-Q1
    # decode is captured across layers, with selective LMCache retrieval as
    # the eager split operation.
    # Unsupported live batch shapes keep using the existing eager SFA forward;
    # incompatible model/runtime features fail fast during startup capture so
    # an explicitly requested POC cannot silently remain inactive.
    "VLLM_ASCEND_SFA_STAGED_GRAPH": lambda: bool(int(os.getenv("VLLM_ASCEND_SFA_STAGED_GRAPH", "0"))),
    # Comma-separated positive exact-Q1 batch sizes, bounded by scheduler
    # capacity. Defaults to singleton capture; not sensitive.
    "VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES": lambda: os.getenv(
        "VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES",
        "1",
    ),
    # Number of blocks for the self-managed paged latent pool (Route 1). 0 = derive a
    # default from max_num_seqs. The pool holds prefill latent during prefill (freed
    # after) + decode latent, sized far below full-context. Tune up for long prompts.
    "VLLM_ASCEND_DSA_LATENT_POOL_BLOCKS": lambda: int(os.getenv("VLLM_ASCEND_DSA_LATENT_POOL_BLOCKS", "0")),
    # Storage device for the in-memory reference offload backend (the LMCache stand-in
    # used until the real adapter lands). "npu" keeps latent in device memory (no
    # memory relief, correctness-only); "cpu" stages latent in host RAM, simulating
    # an off-NPU LMCache — pair with Stage 2 to actually free NPU memory. Default npu.
    "VLLM_ASCEND_DSA_OFFLOAD_BACKEND_DEVICE": lambda: os.getenv("VLLM_ASCEND_DSA_OFFLOAD_BACKEND_DEVICE", "npu"),
    # Adapter-backed DSA latent hot cache (bring-up; default OFF). When on, decode
    # retrieves selected latent from an on-NPU pool (KVCacheAdapter) read in place by
    # the sparse-attn kernel, instead of the scratch-gather offload-manager path.
    "VLLM_ASCEND_DSA_USE_ADAPTER_CACHE": lambda: bool(int(os.getenv("VLLM_ASCEND_DSA_USE_ADAPTER_CACHE", "0"))),
    # Adapter pool headroom over the per-step working set (>=1; larger -> more
    # cross-step reuse and more NPU memory). See adapter_cache.py sizing notes.
    "VLLM_ASCEND_DSA_ADAPTER_POOL_RATIO": lambda: float(os.getenv("VLLM_ASCEND_DSA_ADAPTER_POOL_RATIO", "1.5")),
    # Concurrency the adapter pool is sized for without thrash (0 -> max_num_seqs).
    "VLLM_ASCEND_DSA_ADAPTER_CONCURRENCY_CAP": lambda: int(os.getenv("VLLM_ASCEND_DSA_ADAPTER_CONCURRENCY_CAP", "0")),
    # Back the adapter latent pool with LMCache (host KV store) instead of the
    # in-memory reference backend. Default OFF: the in-memory backend keeps the CPU
    # parity path and a no-LMCache A/B baseline. On -> evicted pool blocks spill to
    # LMCache and misses reload from it (see adapter_cache.build_adapter_cache).
    "VLLM_ASCEND_DSA_USE_LMCACHE_BACKEND": lambda: bool(int(os.getenv("VLLM_ASCEND_DSA_USE_LMCACHE_BACKEND", "0"))),
    # DSA/LMCache trace logging. Default OFF because tensor summaries can force
    # NPU->CPU synchronization on the decode hot path.
    "VLLM_ASCEND_DSA_LMCACHE_TRACE": lambda: bool(
        int(os.getenv("VLLM_ASCEND_DSA_LMCACHE_TRACE", "0"))
    ),
    # Emit bounded diagnostics for MTP and LMCache decode-window interaction.
    # Default off because sampled tensor summaries can synchronize NPU and CPU.
    "VLLM_ASCEND_MTP_DW_DIAG": lambda: bool(
        int(os.getenv("VLLM_ASCEND_MTP_DW_DIAG", "0"))
    ),
    # Keep this many completed decode-window saves pending before publishing
    # them to the scheduler. The scheduler uses the published frontier for
    # both the next DSA split boundary and saved-block release, so both lag
    # together. Only active when LMCache decode-window save is enabled.
    "VLLM_ASCEND_LMCACHE_DECODE_WINDOW_SAVE_COMMIT_DELAY_WINDOWS": lambda: max(
        int(
            os.getenv(
                "VLLM_ASCEND_LMCACHE_DECODE_WINDOW_SAVE_COMMIT_DELAY_WINDOWS",
                "0",
            )
        ),
        0,
    ),
    # Emit one CPU-synchronized, first-post-commit mapping diagnostic for the
    # SHRINK_LATENT=2 compact-scratch path. Requires MTP_DW_DIAG. Default off;
    # diagnostic only, with no inference fallback or output changes.
    "VLLM_ASCEND_MTP_DW_DEEP_DIAG": lambda: bool(
        int(os.getenv("VLLM_ASCEND_MTP_DW_DIAG", "0"))
        and int(os.getenv("VLLM_ASCEND_MTP_DW_DEEP_DIAG", "0"))
    ),
    # Host CPU budget (GiB) PER LAYER for the LMCache adapter backend. 0 (default) =
    # auto-size from the per-layer pinned-bundle need (num_logical_blocks * bundle,
    # with headroom); set >0 to override. Total host = this x number of MLA layers.
    "VLLM_ASCEND_DSA_LMCACHE_CPU_GB": lambda: float(os.getenv("VLLM_ASCEND_DSA_LMCACHE_CPU_GB", "0")),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
