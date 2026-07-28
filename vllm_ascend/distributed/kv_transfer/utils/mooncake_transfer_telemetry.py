#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
"""Structured Mooncake transfer telemetry helper.

See ``Mooncake分布式传输性能打点详细设计.md`` for the event schema and the
invariants enforced here.

Design invariants
-----------------
1. The helper is *stateless*: it never keeps counters, histograms or locks.
2. It is best-effort: any failure inside it MUST NOT change the Mooncake
   data-plane control flow or mask the original exception/result.
3. Callers MUST cache the env switch at initialization time and gate all
   timing/payload construction behind that cached flag, so the disabled
   path performs no JSON, no clock reads and no byte sums. The check
   inside ``emit_mooncake_transfer_metric`` is only a defensive guard.
4. Each target Mooncake synchronous API call emits at most one
   ``MOONCAKE_TRANSFER_METRIC`` line. No request ID, session, IP, key or
   raw address is written into Prometheus labels (this module emits logs
   only, never Prometheus metrics).
5. Failed calls must write ``null`` for the bytes/items that cannot be
   observed, never a fabricated ``0``.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip

from vllm_ascend import envs

logger = init_logger(__name__)

SCHEMA_VERSION = "mooncake_transfer.v1"
LOG_PREFIX = "MOONCAKE_TRANSFER_METRIC"

# Network-type annotation per the design's MVP rule. ``network_type_source``
# must always accompany ``network_type`` so consumers know the confidence
# level.
NETWORK_TYPE_NPU_SIDE = "npu_side"
NETWORK_TYPE_HOST_SIDE = "host_side"
NETWORK_TYPE_UNKNOWN = "unknown"

SOURCE_CONNECTOR_INFERENCE = "connector_inference"
SOURCE_CONTROL_PLANE = "control_plane_semantics"
SOURCE_NOT_EXPOSED = "not_exposed_by_python_api"


def is_enabled() -> bool:
    """Return the cached-by-caller switch. Read once at init time."""
    return bool(envs.VLLM_ASCEND_MOONCAKE_TRANSFER_METRICS)


def _local_host() -> str:
    try:
        return str(get_ip())
    except Exception:
        return ""


def _make_event_id(
    *,
    event: str,
    start_perf_ns: int | None,
    end_wall_time_ns: int | None,
    remote_request_id: str | None,
    local_host: str,
) -> str:
    """Build the per-event dedup key.

    ``api_call_end`` events use the monotonic start timestamp so they are
    unique per actual API call; ``publish_ready`` events have no API call
    and use the wall time + request id.
    """
    pid = os.getpid()
    thread_id = threading.get_native_id()
    if event == "publish_ready":
        return f"{local_host}:{pid}:publish_ready:{remote_request_id}:{end_wall_time_ns}"
    return f"{local_host}:{pid}:{thread_id}:{start_perf_ns}"


def emit_mooncake_transfer_metric(payload: dict[str, Any]) -> None:
    """Emit one structured Mooncake transfer metric line.

    ``payload`` must already contain the event-specific fields. This helper
    fills in the common process/thread/host fields and serializes the JSON.

    Any exception is swallowed: telemetry must never change business
    behavior. The env check here is a defensive guard only; callers gate
    construction behind the cached switch.
    """
    if not envs.VLLM_ASCEND_MOONCAKE_TRANSFER_METRICS:
        return
    try:
        local_host = _local_host()
        full_payload: dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "pid": os.getpid(),
            "thread": threading.current_thread().name,
            "thread_id": threading.get_native_id(),
            "local_host": local_host,
            **payload,
        }
        if "event_id" not in full_payload:
            full_payload["event_id"] = _make_event_id(
                event=full_payload.get("event", "api_call_end"),
                start_perf_ns=full_payload.get("start_perf_ns_ref"),
                end_wall_time_ns=full_payload.get("end_wall_time_ns"),
                remote_request_id=full_payload.get("remote_request_id"),
                local_host=local_host,
            )
        full_payload.pop("start_perf_ns_ref", None)
        logger.info(
            "%s %s",
            LOG_PREFIX,
            json.dumps(full_payload, separators=(",", ":"), sort_keys=True),
        )
    except Exception:
        logger.debug("Failed to emit Mooncake transfer metric", exc_info=True)


def safe_summarize_store_input(
    keys: list[str],
    addrs: list[list[int]],
    sizes: list[list[int]],
) -> dict[str, Any]:
    """Best-effort summary of a Store PUT/GET input batch.

    Never raises: any malformed shape returns null fields so the original
    Mooncake API is still called with the original arguments. The summary
    only contains::

        (
            object_count,
            slice_count,
            per_object_bytes,
            total_bytes,
        )
        input_shape_valid

    For PUT ``total_bytes`` is the attempted payload; for GET it is the
    destination buffer capacity (see the design doc for the distinction).
    """
    object_count: int | None = None
    slice_count: int | None = None
    per_object_bytes: list[int] | None = None
    total_bytes: int | None = None
    input_shape_valid = True
    try:
        object_count = len(keys)
        if len(addrs) != object_count or len(sizes) != object_count:
            input_shape_valid = False
        else:
            per_object_bytes = []
            slice_count = 0
            total_bytes = 0
            for s in sizes:
                if not isinstance(s, (list, tuple)):
                    input_shape_valid = False
                    per_object_bytes = None
                    slice_count = None
                    total_bytes = None
                    break
                slice_count += len(s)
                obj_bytes = 0
                ok = True
                for v in s:
                    if not isinstance(v, (int, float)) or isinstance(v, bool):
                        ok = False
                        break
                    obj_bytes += int(v)
                if not ok:
                    input_shape_valid = False
                    per_object_bytes = None
                    slice_count = None
                    total_bytes = None
                    break
                per_object_bytes.append(obj_bytes)
                total_bytes += obj_bytes
    except Exception:
        input_shape_valid = False
        object_count = None
        slice_count = None
        per_object_bytes = None
        total_bytes = None

    return {
        "object_count": object_count,
        "slice_count": slice_count,
        "per_object_bytes": per_object_bytes,
        "total_bytes": total_bytes,
        "input_shape_valid": input_shape_valid,
    }


def build_p2p_read_metric_payload(
    *,
    start_perf_ns: int,
    end_perf_ns: int,
    start_wall_time_ns: int,
    end_wall_time_ns: int,
    attempted_bytes: int,
    completed_bytes: int | None,
    attempted_items: int,
    completed_items: int | None,
    outcome: str,
    return_code: int | None,
    exception_type: str | None,
    request_id: str | None,
    remote_request_id: str,
    local_engine_id: str,
    peer_engine_id: str | None,
    remote_host: str,
    session_id: str,
    same_host: bool,
    num_blocks: int,
    num_groups: int,
    tp_rank: int | None,
    dp_rank_global: int | None,
    dp_rank_local: int | None,
    kv_role: str | None,
    pull_offset: int | None,
    local_memory_type: str,
    network_type: str,
    network_type_source: str,
) -> dict[str, Any]:
    """Build the P2P ``api_call_end`` payload for a sync READ call."""
    duration_ns = end_perf_ns - start_perf_ns
    return_code_counts: dict[str, int] | None = None
    if return_code is not None:
        return_code_counts = {str(return_code): 1}
    payload: dict[str, Any] = {
        "event": "api_call_end",
        "phase": "data_plane",
        "component": "p2p",
        "connector": "MooncakeConnectorV1",
        "backend": "transfer_engine",
        "operation": "read",
        "outcome": outcome,
        "start_wall_time_ns": start_wall_time_ns,
        "end_wall_time_ns": end_wall_time_ns,
        "duration_ns": duration_ns,
        "attempted_bytes": attempted_bytes,
        "buffer_capacity_bytes": None,
        "completed_bytes": completed_bytes,
        "bytes_semantics": "logical_payload",
        "attempted_items": attempted_items,
        "completed_items": completed_items,
        "item_unit": "descriptor",
        "descriptor_count": attempted_items,
        "return_code_counts": return_code_counts,
        "exception_type": exception_type,
        "request_id": request_id,
        "remote_request_id": remote_request_id,
        "local_engine_id": local_engine_id,
        "peer_engine_id": peer_engine_id,
        "dp_rank_global": dp_rank_global,
        "dp_rank_local": dp_rank_local,
        "tp_rank": tp_rank,
        "kv_role": kv_role,
        "rank": tp_rank,
        "remote_host": remote_host,
        "same_host": same_host,
        "session_id": session_id,
        "num_blocks": num_blocks,
        "num_groups": num_groups,
        "pull_offset": pull_offset,
        "network_type": network_type,
        "network_type_source": network_type_source,
        "local_memory_type": local_memory_type,
        "remote_memory_type": "npu_inferred",
        "configured_protocol": None,
        "fabric_mem_enabled": None,
        "start_perf_ns_ref": start_perf_ns,
    }
    return payload


def build_publish_ready_payload(
    *,
    end_wall_time_ns: int,
    remote_request_id: str,
    local_engine_id: str,
    num_blocks: int,
    kv_role: str,
    tp_rank: int | None,
    dp_rank_global: int | None,
    dp_rank_local: int | None,
) -> dict[str, Any]:
    """Build the P-side ``publish_ready`` control-plane event."""
    return {
        "event": "publish_ready",
        "phase": "control_plane",
        "component": "p2p",
        "connector": "MooncakeConnectorV1",
        "backend": "transfer_engine",
        "operation": "read",
        "outcome": None,
        "duration_ns": None,
        "start_wall_time_ns": None,
        "end_wall_time_ns": end_wall_time_ns,
        "remote_request_id": remote_request_id,
        "local_engine_id": local_engine_id,
        "peer_engine_id": None,
        "kv_role": kv_role,
        "rank": tp_rank,
        "dp_rank_global": dp_rank_global,
        "dp_rank_local": dp_rank_local,
        "tp_rank": tp_rank,
        "num_blocks": num_blocks,
        "start_perf_ns_ref": None,
    }


def _store_common_fields(
    *,
    configured_protocol: str,
    fabric_mem_enabled: bool,
    tp_rank: int | None,
    dp_rank_global: int | None,
    dp_rank_local: int | None,
) -> dict[str, Any]:
    return {
        "component": "store",
        "connector": "AscendStoreConnector",
        "backend": "distributed_store",
        "configured_protocol": configured_protocol,
        "fabric_mem_enabled": fabric_mem_enabled,
        "rank": tp_rank,
        "dp_rank_global": dp_rank_global,
        "dp_rank_local": dp_rank_local,
        "tp_rank": tp_rank,
        "kv_role": None,
        "local_memory_type": "npu_inferred",
        "remote_memory_type": "mooncake_managed",
        "same_host": None,
        "session_id": None,
    }


def build_store_put_metric_payload(
    *,
    start_perf_ns: int,
    end_perf_ns: int,
    start_wall_time_ns: int,
    end_wall_time_ns: int,
    input_summary: dict[str, Any],
    result: list[int] | None,
    exception_type: str | None,
    configured_protocol: str,
    fabric_mem_enabled: bool,
    tp_rank: int | None,
    dp_rank_global: int | None,
    dp_rank_local: int | None,
) -> dict[str, Any]:
    """Build the Store PUT ``api_call_end`` payload."""
    return _build_store_data_payload(
        operation="put",
        bytes_semantics="accepted_logical_payload",
        start_perf_ns=start_perf_ns,
        end_perf_ns=end_perf_ns,
        start_wall_time_ns=start_wall_time_ns,
        end_wall_time_ns=end_wall_time_ns,
        input_summary=input_summary,
        result=result,
        exception_type=exception_type,
        success_predicate=lambda v: v == 0,
        configured_protocol=configured_protocol,
        fabric_mem_enabled=fabric_mem_enabled,
        tp_rank=tp_rank,
        dp_rank_global=dp_rank_global,
        dp_rank_local=dp_rank_local,
    )


def build_store_get_metric_payload(
    *,
    start_perf_ns: int,
    end_perf_ns: int,
    start_wall_time_ns: int,
    end_wall_time_ns: int,
    input_summary: dict[str, Any],
    result: list[int] | None,
    exception_type: str | None,
    configured_protocol: str,
    fabric_mem_enabled: bool,
    tp_rank: int | None,
    dp_rank_global: int | None,
    dp_rank_local: int | None,
) -> dict[str, Any]:
    """Build the Store GET ``api_call_end`` payload.

    GET ``sizes`` is destination buffer capacity, not requested payload, so
    ``attempted_bytes`` is null and the total goes to
    ``buffer_capacity_bytes``.
    """
    payload = _build_store_data_payload(
        operation="get",
        bytes_semantics="returned_logical_payload",
        start_perf_ns=start_perf_ns,
        end_perf_ns=end_perf_ns,
        start_wall_time_ns=start_wall_time_ns,
        end_wall_time_ns=end_wall_time_ns,
        input_summary=input_summary,
        result=result,
        exception_type=exception_type,
        success_predicate=lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0,
        configured_protocol=configured_protocol,
        fabric_mem_enabled=fabric_mem_enabled,
        tp_rank=tp_rank,
        dp_rank_global=dp_rank_global,
        dp_rank_local=dp_rank_local,
    )
    # For GET, sum(sizes) is buffer capacity, not attempted payload.
    payload["buffer_capacity_bytes"] = input_summary.get("total_bytes")
    payload["attempted_bytes"] = None
    return payload


def _build_store_data_payload(
    *,
    operation: str,
    bytes_semantics: str,
    start_perf_ns: int,
    end_perf_ns: int,
    start_wall_time_ns: int,
    end_wall_time_ns: int,
    input_summary: dict[str, Any],
    result: list[int] | None,
    exception_type: str | None,
    success_predicate,
    configured_protocol: str,
    fabric_mem_enabled: bool,
    tp_rank: int | None,
    dp_rank_global: int | None,
    dp_rank_local: int | None,
) -> dict[str, Any]:
    duration_ns = end_perf_ns - start_perf_ns
    object_count = input_summary.get("object_count")
    slice_count = input_summary.get("slice_count")
    per_object_bytes = input_summary.get("per_object_bytes")
    input_shape_valid = input_summary.get("input_shape_valid", True)
    total_bytes = input_summary.get("total_bytes")

    outcome = "error"
    completed_bytes: int | None = None
    completed_items: int | None = None
    success_count = 0
    failure_count = 0
    return_code_counts: dict[str, int] | None = None
    invalid_result = False

    if exception_type is not None:
        outcome = "error"
        completed_bytes = None
        completed_items = None
    elif result is None:
        outcome = "error"
        invalid_result = True
    else:
        return_code_counts = {}
        for v in result:
            key = str(v)
            return_code_counts[key] = return_code_counts.get(key, 0) + 1
        if len(result) == 0 and (object_count == 0 or object_count is None):
            # Zero-key zero-result is a successful no-op.
            outcome = "success"
            success_count = 0
            failure_count = 0
            completed_bytes = 0
            completed_items = 0
        elif object_count is not None and len(result) != object_count:
            outcome = "error"
            invalid_result = True
            success_count = sum(1 for v in result if success_predicate(v))
            failure_count = len(result) - success_count if result else 0
        else:
            success_count = sum(1 for v in result if success_predicate(v))
            failure_count = len(result) - success_count
            if success_count == 0:
                outcome = "error"
            elif failure_count == 0:
                outcome = "success"
            else:
                outcome = "partial"

            completed_bytes = 0
            for i, v in enumerate(result):
                if not success_predicate(v):
                    continue
                if operation == "get":
                    completed_bytes += int(v)
                else:
                    # put: per_object_bytes may be None on bad shape
                    if per_object_bytes is not None and i < len(per_object_bytes):
                        completed_bytes += int(per_object_bytes[i])
                    elif total_bytes is not None and success_count > 0:
                        # Fallback only when shape was valid; otherwise we
                        # cannot attribute bytes safely.
                        completed_bytes = total_bytes
                        break
            completed_items = success_count

    payload: dict[str, Any] = {
        "event": "api_call_end",
        "phase": "data_plane",
        "operation": operation,
        "outcome": outcome,
        "start_wall_time_ns": start_wall_time_ns,
        "end_wall_time_ns": end_wall_time_ns,
        "duration_ns": duration_ns,
        "attempted_bytes": total_bytes if operation == "put" else None,
        "buffer_capacity_bytes": None,
        "completed_bytes": completed_bytes,
        "bytes_semantics": bytes_semantics,
        "attempted_items": object_count,
        "completed_items": completed_items,
        "item_unit": "object",
        "descriptor_count": slice_count,
        "return_code_counts": return_code_counts,
        "exception_type": exception_type,
        "object_count": object_count,
        "slice_count": slice_count,
        "success_count": success_count,
        "failure_count": failure_count,
        "hit_count": None,
        "miss_count": None,
        "error_count": failure_count,
        "input_shape_valid": input_shape_valid,
        "invalid_result": invalid_result,
        "network_type": NETWORK_TYPE_UNKNOWN,
        "network_type_source": SOURCE_NOT_EXPOSED,
        "start_perf_ns_ref": start_perf_ns,
    }
    payload.update(
        _store_common_fields(
            configured_protocol=configured_protocol,
            fabric_mem_enabled=fabric_mem_enabled,
            tp_rank=tp_rank,
            dp_rank_global=dp_rank_global,
            dp_rank_local=dp_rank_local,
        )
    )
    return payload


def build_store_exists_metric_payload(
    *,
    start_perf_ns: int,
    end_perf_ns: int,
    start_wall_time_ns: int,
    end_wall_time_ns: int,
    keys: list[str],
    result: list[int] | None,
    exception_type: str | None,
    configured_protocol: str,
    fabric_mem_enabled: bool,
    tp_rank: int | None,
    dp_rank_global: int | None,
    dp_rank_local: int | None,
) -> dict[str, Any]:
    """Build the Store EXISTS ``api_call_end`` (control-plane) payload."""
    duration_ns = end_perf_ns - start_perf_ns
    object_count = len(keys) if keys is not None else None
    hit_count = 0
    miss_count = 0
    error_count = 0
    return_code_counts: dict[str, int] | None = None
    outcome = "error"
    completed_items: int | None = None
    invalid_result = False

    if exception_type is not None:
        outcome = "error"
    elif result is None:
        outcome = "error"
        invalid_result = True
    else:
        return_code_counts = {}
        for v in result:
            key = str(v)
            return_code_counts[key] = return_code_counts.get(key, 0) + 1
            if v == 1:
                hit_count += 1
            elif v == 0:
                miss_count += 1
            else:
                error_count += 1
        if object_count is not None and len(result) != object_count and not (len(result) == 0 and object_count == 0):
            outcome = "error"
            invalid_result = True
        elif error_count == 0:
            outcome = "success"
        elif hit_count + miss_count == 0:
            outcome = "error"
        else:
            outcome = "partial"
        completed_items = hit_count + miss_count

    payload: dict[str, Any] = {
        "event": "api_call_end",
        "phase": "control_plane",
        "operation": "exists",
        "outcome": outcome,
        "start_wall_time_ns": start_wall_time_ns,
        "end_wall_time_ns": end_wall_time_ns,
        "duration_ns": duration_ns,
        "attempted_bytes": 0,
        "buffer_capacity_bytes": None,
        "completed_bytes": 0,
        "bytes_semantics": "control_plane_no_payload",
        "attempted_items": object_count,
        "completed_items": completed_items,
        "item_unit": "key",
        "descriptor_count": None,
        "return_code_counts": return_code_counts,
        "exception_type": exception_type,
        "object_count": object_count,
        "slice_count": None,
        "success_count": hit_count + miss_count,
        "failure_count": error_count,
        "hit_count": hit_count,
        "miss_count": miss_count,
        "error_count": error_count,
        "input_shape_valid": True,
        "invalid_result": invalid_result,
        "network_type": NETWORK_TYPE_HOST_SIDE,
        "network_type_source": SOURCE_CONTROL_PLANE,
        "start_perf_ns_ref": start_perf_ns,
    }
    payload.update(
        _store_common_fields(
            configured_protocol=configured_protocol,
            fabric_mem_enabled=fabric_mem_enabled,
            tp_rank=tp_rank,
            dp_rank_global=dp_rank_global,
            dp_rank_local=dp_rank_local,
        )
    )
    return payload


def classify_p2p_network_type(local_memory_type: str) -> tuple[str, str]:
    """Infer network_type for a P2P READ from the local runtime device kind."""
    if local_memory_type in ("npu", "cuda", "ascend"):
        return NETWORK_TYPE_NPU_SIDE, SOURCE_CONNECTOR_INFERENCE
    return NETWORK_TYPE_UNKNOWN, SOURCE_CONNECTOR_INFERENCE


def now_perf_ns() -> int:
    return time.perf_counter_ns()


def now_wall_ns() -> int:
    return time.time_ns()
