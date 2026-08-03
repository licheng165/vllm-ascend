# SPDX-License-Identifier: Apache-2.0
"""DSA context-length threshold routing configuration and startup validation.

This is the single authoritative place where the
``VLLM_ASCEND_DSA_CONTEXT_LENGTH_THRESHOLD`` environment variable is parsed and
the design's startup prerequisites (03_GLM51_DSA提示词阈值分流详细设计
section 5) are validated.  The normalized config is written into
``VllmConfig.additional_config['dsa']`` and consumed by the vLLM Scheduler's
``DSAController`` (see vllm.v1.core.sched.dsa_controller).

Design section 4.1 explicitly forbids each of the four repos from reading the
env var independently; the platform config here is the one parser, and the
Scheduler only reads the normalized ``additional_config['dsa']``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from vllm.logger import logger

from vllm_ascend import envs


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        return value
    return ((value + alignment - 1) // alignment) * alignment


@dataclass
class DSAThresholdConfig:
    """Normalized DSA threshold routing configuration."""

    threshold: int = 0
    max_model_len: int = 0
    block_size: int = 0
    chunk_size: int = 0
    window_size: int = 0
    index_topk: int = 0
    query_width: int = 1
    scratch_capacity: int = 0
    minimum_valid_boundary: int = 0
    first_reclaiming_frontier: int = 0
    node_role: str = "standalone"
    deployment_mode: str = "standalone"
    data_compatibility_fingerprint: str = ""
    instance_capability_digest: str = ""
    native_boundary_validation: bool = False

    @property
    def enabled(self) -> bool:
        return self.threshold > 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_threshold() -> int:
    raw = envs.VLLM_ASCEND_DSA_CONTEXT_LENGTH_THRESHOLD
    if raw is None or raw == "":
        return 0
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        raise ValueError(
            "VLLM_ASCEND_DSA_CONTEXT_LENGTH_THRESHOLD must be an integer, "
            f"got {raw!r}"
        )
    if value < 0:
        raise ValueError(
            "VLLM_ASCEND_DSA_CONTEXT_LENGTH_THRESHOLD must be >= 0, "
            f"got {value}"
        )
    return value


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _read_lmcache_chunk_size(*, required: bool) -> int:
    raw = os.getenv("LMCACHE_CHUNK_SIZE")
    source = "LMCACHE_CHUNK_SIZE"
    if raw is None or raw == "":
        config_path = os.getenv("LMCACHE_CONFIG_FILE")
        if config_path:
            source = f"LMCACHE_CONFIG_FILE ({config_path})"
            try:
                config = yaml.safe_load(
                    Path(config_path).read_text(encoding="utf-8")
                )
            except (OSError, yaml.YAMLError) as exc:
                if required:
                    raise ValueError(
                        "Unable to read LMCache chunk_size from "
                        f"{source}: {exc}"
                    ) from exc
                config = None
            if config is not None and not isinstance(config, dict):
                if required:
                    raise ValueError(f"{source} must contain a YAML mapping")
                config = None
            raw = config.get("chunk_size") if config else None

    if raw is None or raw == "":
        return 256
    try:
        chunk_size = int(raw)
    except (TypeError, ValueError) as exc:
        if not required:
            return 256
        raise ValueError(f"LMCache chunk_size from {source} must be an integer") from exc
    if chunk_size <= 0:
        if not required:
            return 256
        raise ValueError(f"LMCache chunk_size from {source} must be positive")
    return chunk_size


def _read_node_role(vllm_config: Any) -> tuple[str, str]:
    """Read dsa_deployment_mode / dsa_node_role from kv_connector_extra_config.

    The scheduler-side LMCache connector is the unique owner that *removes*
    these keys before constructing the LMCache config (design section 14.6);
    here we only READ them for capability building without consuming them.
    """
    ktc = getattr(vllm_config, "kv_transfer_config", None)
    extra = {}
    if ktc is not None:
        extra = getattr(ktc, "kv_connector_extra_config", None) or {}
    deployment = str(
        extra.get("dsa_deployment_mode", "standalone")
    ).lower()
    node_role = str(extra.get("dsa_node_role", deployment)).lower()
    valid_pairs = {
        ("standalone", "standalone"),
        ("pd", "prefill"),
        ("pd", "decode"),
    }
    if (deployment, node_role) not in valid_pairs:
        raise ValueError(
            "Invalid DSA deployment configuration: expected "
            "standalone/standalone, pd/prefill, or pd/decode; got "
            f"deployment_mode={deployment!r}, node_role={node_role!r}"
        )
    return deployment, node_role


def build_dsa_threshold_config(vllm_config: Any) -> DSAThresholdConfig:
    """Parse the env var, validate prerequisites, and build the config.

    Performs fail-fast validation per design section 5.  Returns a config with
    ``threshold == 0`` (disabled) when the env var is unset/0.
    """
    threshold = _parse_threshold()

    model_config = getattr(vllm_config, "model_config", None)
    cache_config = getattr(vllm_config, "cache_config", None)

    block_size = int(getattr(cache_config, "block_size", 0) or 0) if cache_config else 0
    max_model_len = (
        int(getattr(model_config, "max_model_len", 0) or 0) if model_config else 0
    )

    # Frontier parameters come from the LMCache runtime config / hf config.
    hf_config = getattr(model_config, "hf_text_config", None) if model_config else None
    index_topk = int(getattr(hf_config, "index_topk", 0) or 0) if hf_config else 0
    speculative_config = getattr(vllm_config, "speculative_config", None)
    num_speculative_tokens = int(
        getattr(speculative_config, "num_speculative_tokens", 0) or 0
    )
    query_width = 1 + max(num_speculative_tokens, 0)
    scratch_capacity = query_width * index_topk if index_topk > 0 else 0

    chunk_size = _read_lmcache_chunk_size(required=threshold > 0)
    window_size = _env_int("LMCACHE_DECODE_WINDOW_SAVE_WINDOW_SIZE", 0)

    deployment, node_role = _read_node_role(vllm_config)

    cfg = DSAThresholdConfig(
        threshold=threshold,
        max_model_len=max_model_len,
        block_size=block_size,
        chunk_size=chunk_size,
        window_size=window_size,
        index_topk=index_topk,
        query_width=query_width,
        scratch_capacity=scratch_capacity,
        minimum_valid_boundary=scratch_capacity,
        first_reclaiming_frontier=_align_up(scratch_capacity + 1, chunk_size),
        node_role=node_role,
        deployment_mode=deployment,
        native_boundary_validation=True,
    )

    # _validate_startup_prerequisites(vllm_config, cfg)

    # Build capability fingerprints (design section 4.2).
    cfg.data_compatibility_fingerprint = _data_compatibility_fingerprint(cfg, vllm_config)
    cfg.instance_capability_digest = _instance_capability_digest(cfg, vllm_config)

    _emit_startup_log(cfg)
    return cfg


def _validate_startup_prerequisites(vllm_config: Any, cfg: DSAThresholdConfig) -> None:
    """Fail-fast validation.  Two scopes (design section 5):

    * Global (always, even threshold=0): async scheduling must be off when any
      shrink-latent release path is possible; native SFA boundary validation
      must be enabled.
    * Positive-threshold-only: the full dependency matrix (UNBUNDLE,
      TWO_GROUPS, SHRINK_LATENT, LMCache flags, alignment, capability
      handshake).
    """
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    async_scheduling = bool(getattr(scheduler_config, "async_scheduling", False))
    shrink_latent = int(envs.VLLM_ASCEND_DSA_SHRINK_LATENT)
    unbundle = bool(envs.VLLM_ASCEND_DSA_UNBUNDLE)
    two_groups = bool(envs.VLLM_ASCEND_DSA_TWO_GROUPS)

    # ---- Global: async scheduling + shrink-latent -----------------------
    # Any path that may release latent (including the threshold=0 LEGACY
    # release path) is incompatible with async scheduling in the first version
    # because request.num_computed_tokens is optimistically advanced and async
    # queues add output placeholders.  Future async support requires an
    # execution-sequence barrier and worker-confirmed materialization frontier
    # (design section 5); simply removing this check is unsafe.
    if shrink_latent and async_scheduling:
        raise ValueError(
            "VLLM_ASCEND_DSA_SHRINK_LATENT requires async scheduling to be "
            "disabled (--no-async-scheduling) in the first version. "
            "num_computed_tokens is optimistically advanced under async and "
            "would race with latent release. See design section 5."
        )

    if not cfg.enabled:
        return

    # ---- Positive-threshold-only dependency matrix ----------------------
    cache_config = getattr(vllm_config, "cache_config", None)
    prefix_caching = bool(getattr(cache_config, "enable_prefix_caching", False))

    errors: list[str] = []

    if not unbundle:
        errors.append("VLLM_ASCEND_DSA_UNBUNDLE must be 1")
    if not two_groups:
        errors.append("VLLM_ASCEND_DSA_TWO_GROUPS must be 1")
    if shrink_latent != 2:
        errors.append("VLLM_ASCEND_DSA_SHRINK_LATENT must be 2")
    if not _env_bool("LMCACHE_ENABLE_SPARSE_ATTENTION"):
        errors.append("LMCACHE_ENABLE_SPARSE_ATTENTION must be true")
    if not _env_bool("LMCACHE_USE_LAYERWISE"):
        errors.append("LMCACHE_USE_LAYERWISE must be true")
    if not _env_bool("LMCACHE_DSA_TWO_GROUPS"):
        errors.append("LMCACHE_DSA_TWO_GROUPS must be true")
    if not _env_bool("LMCACHE_SAVE_UNFULL_CHUNK"):
        errors.append("LMCACHE_SAVE_UNFULL_CHUNK must be true")
    if _env_int("LMCACHE_DECODE_WINDOW_SAVE_WINDOW_SIZE", 0) <= 0:
        errors.append("LMCACHE_DECODE_WINDOW_SAVE_WINDOW_SIZE must be > 0")
    if prefix_caching:
        errors.append("vLLM prefix cache must be disabled")
    if cfg.chunk_size <= 0 or cfg.block_size <= 0:
        errors.append("chunk_size and block_size must be positive")
    elif cfg.chunk_size % cfg.block_size != 0:
        errors.append(
            f"chunk_size ({cfg.chunk_size}) % block_size ({cfg.block_size}) != 0"
        )
    if cfg.window_size > 0 and cfg.chunk_size > 0 and cfg.window_size % cfg.chunk_size != 0:
        errors.append(
            f"window_size ({cfg.window_size}) % chunk_size ({cfg.chunk_size}) != 0"
        )
    if cfg.index_topk > 0 and cfg.block_size > 0 and cfg.index_topk % cfg.block_size != 0:
        errors.append(
            f"index_topk ({cfg.index_topk}) % block_size ({cfg.block_size}) != 0"
        )
    if cfg.query_width not in (1, 2):
        errors.append(
            f"query_width must be 1 or 2 (got {cfg.query_width})"
        )
    if cfg.scratch_capacity <= 0:
        errors.append("scratch_capacity could not be derived (index_topk unset?)")

    # Four-repo flag consistency: vLLM Core only checks TWO_GROUPS, Ascend also
    # requires UNBUNDLE; divergence causes tensor interpretation errors.
    if unbundle != two_groups and (unbundle or two_groups):
        # two_groups requires unbundle; covered above, keep explicit warning.
        pass

    if errors:
        raise ValueError(
            "VLLM_ASCEND_DSA_CONTEXT_LENGTH_THRESHOLD is set to a positive "
            "value but startup prerequisites are not met:\n  - "
            + "\n  - ".join(errors)
        )

    # Warnings (non-fatal).
    if cfg.max_model_len and cfg.threshold > cfg.max_model_len:
        logger.warning(
            "VLLM_ASCEND_DSA_CONTEXT_LENGTH_THRESHOLD (%d) > max_model_len "
            "(%d): all legal requests will stay resident; sparse routing is "
            "effectively disabled. This is a valid way to force-close "
            "request-level sparse.",
            cfg.threshold,
            cfg.max_model_len,
        )
    if cfg.scratch_capacity > 0 and cfg.threshold < cfg.scratch_capacity:
        logger.warning(
            "VLLM_ASCEND_DSA_CONTEXT_LENGTH_THRESHOLD (%d) < scratch_capacity "
            "(%d): requests may enter PROMOTING but source frontier will not "
            "create reclaimable blocks until rolling coverage exceeds scratch. "
            "Low thresholds are legal; release is gated independently by the "
            "source frontier (design section 3.2).",
            cfg.threshold,
            cfg.scratch_capacity,
        )


def _data_compatibility_fingerprint(cfg: DSAThresholdConfig, vllm_config: Any) -> str:
    model_config = getattr(vllm_config, "model_config", None)
    payload = {
        "model": str(getattr(model_config, "model", "")) if model_config else "",
        "dtype": str(getattr(model_config, "dtype", "")) if model_config else "",
        "quant": str(getattr(model_config, "quantization", "")) if model_config else "",
        "bs": cfg.block_size,
        "cs": cfg.chunk_size,
        "ws": cfg.window_size,
        "tk": cfg.index_topk,
        "qw": cfg.query_width,
        "sc": cfg.scratch_capacity,
        "abi": "dsa-v1",
        "ns": "v2",
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _instance_capability_digest(cfg: DSAThresholdConfig, vllm_config: Any) -> str:
    parallel_config = getattr(vllm_config, "parallel_config", None)
    payload = {
        "role": cfg.node_role,
        "mode": cfg.deployment_mode,
        "tp": int(getattr(parallel_config, "tensor_parallel_size", 1) or 1),
        "pp": int(getattr(parallel_config, "pipeline_parallel_size", 1) or 1),
        "dp": int(getattr(parallel_config, "enable_expert_parallel", False)),
        "native_validation": cfg.native_boundary_validation,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _emit_startup_log(cfg: DSAThresholdConfig) -> None:
    logger.info(
        "[DSA_THRESHOLD] raw_threshold=%s normalized_threshold=%d enabled=%s "
        "max_model_len=%d block_size=%d chunk_size=%d window_size=%d "
        "index_topk=%d query_width=%d scratch_capacity=%d "
        "minimum_valid_boundary=%d first_reclaiming_frontier=%d "
        "node_role=%s deployment_mode=%s "
        "data_compatibility_fingerprint=%s instance_capability_digest=%s",
        os.getenv("VLLM_ASCEND_DSA_CONTEXT_LENGTH_THRESHOLD", "0"),
        cfg.threshold,
        cfg.enabled,
        cfg.max_model_len,
        cfg.block_size,
        cfg.chunk_size,
        cfg.window_size,
        cfg.index_topk,
        cfg.query_width,
        cfg.scratch_capacity,
        cfg.minimum_valid_boundary,
        cfg.first_reclaiming_frontier,
        cfg.node_role,
        cfg.deployment_mode,
        cfg.data_compatibility_fingerprint,
        cfg.instance_capability_digest,
    )


def apply_dsa_threshold_config(vllm_config: Any) -> DSAThresholdConfig:
    """Build the config and write it into additional_config['dsa'].

    Called from NPUPlatform.check_and_update_config.
    """
    cfg = build_dsa_threshold_config(vllm_config)
    if vllm_config.additional_config is None:
        vllm_config.additional_config = {}
    vllm_config.additional_config["dsa"] = cfg.to_dict()
    return cfg
