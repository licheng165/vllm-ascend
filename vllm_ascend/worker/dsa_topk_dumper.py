"""DSA indexer top-k dump for GLM5.1 decode diagnostics.

Implements the design in ``GLM5.1_Indexer选中Token采集详细设计.md``.
Captures the raw (pre-remap) top-k token positions selected by each
target-model SFA layer during decode, maps them to vocabulary token IDs,
and writes JSONL records for offline head/tail distribution analysis.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import socket
import threading
import time
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_input_batch import InputBatch

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_MAX_STEPS = 128
DEFAULT_MAX_REQUESTS = 1
DEFAULT_EXPECTED_K = 2048
DEFAULT_MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024
WRITER_QUEUE_DEPTH = 2
_INT32_BYTES = 4
_DIR_MODE = 0o700
_FILE_MODE = 0o600


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _parse_tp_ranks(raw: str | None) -> set[int] | None:
    if raw is None or raw.strip() == "":
        return {0}
    raw = raw.strip()
    if raw.lower() == "all":
        return None
    return {int(x) for x in raw.split(",") if x.strip()}


@dataclass(frozen=True)
class DSATopKDumpConfig:
    enabled: bool
    dump_dir: str
    tag: str
    request_id_substr: str
    max_steps: int
    max_requests: int
    tp_ranks: set[int] | None
    include_mtp: bool
    expected_k: int
    max_file_bytes: int

    @classmethod
    def from_env(cls) -> "DSATopKDumpConfig":
        enabled = bool(int(os.getenv("VLLM_ASCEND_DSA_TOPK_DUMP", "0")))
        dump_dir = os.getenv("VLLM_ASCEND_DSA_TOPK_DUMP_DIR", "")
        tag = os.getenv("VLLM_ASCEND_DSA_TOPK_DUMP_TAG", "")
        substr = os.getenv("VLLM_ASCEND_DSA_TOPK_DUMP_REQUEST_ID_SUBSTR", "")
        max_steps = int(os.getenv("VLLM_ASCEND_DSA_TOPK_DUMP_MAX_STEPS", str(DEFAULT_MAX_STEPS)))
        max_requests = int(os.getenv("VLLM_ASCEND_DSA_TOPK_DUMP_MAX_REQUESTS", str(DEFAULT_MAX_REQUESTS)))
        tp_ranks = _parse_tp_ranks(os.getenv("VLLM_ASCEND_DSA_TOPK_DUMP_TP_RANKS"))
        include_mtp = bool(int(os.getenv("VLLM_ASCEND_DSA_TOPK_DUMP_INCLUDE_MTP", "0")))
        expected_k = int(os.getenv("VLLM_ASCEND_DSA_TOPK_DUMP_EXPECTED_K", str(DEFAULT_EXPECTED_K)))
        max_file_bytes = int(
            os.getenv("VLLM_ASCEND_DSA_TOPK_DUMP_MAX_FILE_BYTES", str(DEFAULT_MAX_FILE_BYTES))
        )
        return cls(
            enabled=enabled,
            dump_dir=dump_dir,
            tag=tag,
            request_id_substr=substr,
            max_steps=max_steps,
            max_requests=max_requests,
            tp_ranks=tp_ranks,
            include_mtp=include_mtp,
            expected_k=expected_k,
            max_file_bytes=max_file_bytes,
        )

    def validate(self) -> None:
        if not self.enabled:
            return
        if not os.path.isabs(self.dump_dir):
            raise ValueError(
                f"VLLM_ASCEND_DSA_TOPK_DUMP_DIR must be an absolute path, got '{self.dump_dir}'"
            )
        if self.max_steps <= 0:
            raise ValueError(f"MAX_STEPS must be positive, got {self.max_steps}")
        if self.max_requests <= 0:
            raise ValueError(f"MAX_REQUESTS must be positive, got {self.max_requests}")
        if self.include_mtp:
            raise ValueError(
                "VLLM_ASCEND_DSA_TOPK_DUMP_INCLUDE_MTP=1 is not supported in this version; "
                "MTP predictor token-ID schema is not implemented yet."
            )
        if self.max_file_bytes <= 0:
            raise ValueError(f"MAX_FILE_BYTES must be positive, got {self.max_file_bytes}")
        os.makedirs(self.dump_dir, mode=_DIR_MODE, exist_ok=True)
        test_file = os.path.join(self.dump_dir, f".dump_test_{os.getpid()}")
        try:
            fd = os.open(test_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _FILE_MODE)
            os.close(fd)
            os.unlink(test_file)
        except OSError as exc:
            raise ValueError(
                f"Cannot create test file in dump dir '{self.dump_dir}': {exc}"
            ) from exc
        usage = os.statvfs(self.dump_dir)
        free_bytes = usage.f_bavail * usage.f_frsize
        if free_bytes < self.max_file_bytes:
            raise ValueError(
                f"Insufficient disk space in '{self.dump_dir}': "
                f"free={free_bytes}, required>={self.max_file_bytes}"
            )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RowDescriptor:
    internal_request_id: str
    request_index: int
    source_flat_row_index: int
    query_position: int
    request_query_offset: int
    row_kind: str
    prompt_len: int
    num_tokens_no_spec_before_step: int
    scheduled_spec_ids: tuple[int, ...]


@dataclass
class CapturePlan:
    forward_id: int
    rows: list[RowDescriptor]
    layer_names: list[str]
    device_row_indices: torch.Tensor | None = None
    remap_boundary_rows: np.ndarray | None = None


@dataclass
class StepRecord:
    manifest: dict[str, Any]
    selections: list[dict[str, Any]]
    step_end: dict[str, Any]
    future: Future[bool] = field(default_factory=Future)


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def reserved_bytes(
    num_layers: int,
    max_requests: int,
    decode_rows: int,
    index_topk: int,
) -> int:
    return num_layers * max_requests * decode_rows * index_topk * _INT32_BYTES


def _compute_position_stats(
    positions: np.ndarray,
    query_position: int,
    prompt_len: int,
    num_tokens_no_spec: int,
    num_spec_ids: int,
) -> dict[str, Any]:
    k = len(positions)
    valid = (positions >= 0) & (positions <= query_position)
    valid_count = int(valid.sum())
    invalid_count = k - valid_count

    valid_positions = positions[valid]
    denom = max(query_position, 1)
    relative = valid_positions.astype(np.float64) / denom
    bin_ids = np.minimum(9, (relative * 10).astype(np.int32))
    decile = np.bincount(bin_ids, minlength=10).tolist()

    head_10 = int(((relative >= 0.0) & (relative < 0.1)).sum())
    tail_10 = int((relative >= 0.9).sum())
    middle_80 = valid_count - head_10 - tail_10

    prompt_count = int(((valid_positions >= 0) & (valid_positions < prompt_len)).sum())
    spec_start = num_tokens_no_spec
    spec_end = spec_start + num_spec_ids
    accepted_gen = int(
        ((valid_positions >= prompt_len) & (valid_positions < spec_start)).sum()
    )
    scheduled_spec = int(((valid_positions >= spec_start) & (valid_positions < spec_end)).sum())
    unique_count = int(len(np.unique(valid_positions)))

    return {
        "valid_count": valid_count,
        "invalid_position_count": invalid_count,
        "position_decile_counts": decile,
        "head_10pct_count": head_10,
        "middle_80pct_count": middle_80,
        "tail_10pct_count": tail_10,
        "prompt_position_count": prompt_count,
        "accepted_generated_position_count": accepted_gen,
        "scheduled_speculative_position_count": scheduled_spec,
        "unique_position_count": unique_count,
    }


def _map_token_ids(
    positions: np.ndarray,
    query_position: int,
    row: RowDescriptor,
    input_batch: "InputBatch",
) -> list[int | None]:
    k = len(positions)
    result: list[int | None] = [None] * k
    valid = (positions >= 0) & (positions <= query_position)
    valid_positions = positions[valid]
    valid_flat = np.flatnonzero(valid)

    spec_start = row.num_tokens_no_spec_before_step
    spec_ids = row.scheduled_spec_ids
    spec_end = spec_start + len(spec_ids)

    is_spec = (valid_positions >= spec_start) & (valid_positions < spec_end)
    for j in np.flatnonzero(is_spec):
        result[int(valid_flat[j])] = spec_ids[int(valid_positions[j]) - spec_start]

    is_history = ~is_spec
    hist_positions = valid_positions[is_history]
    hist_flat = valid_flat[is_history]
    is_real = input_batch.is_token_ids[row.request_index, hist_positions]
    for j in np.flatnonzero(is_real):
        result[int(hist_flat[j])] = int(
            input_batch.token_ids_cpu[row.request_index, int(hist_positions[j])]
        )
    return result


def _compute_manifest_hash(manifest: dict[str, Any]) -> str:
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _compute_record_id(
    session_id: str,
    forward_id: int,
    subforward_id: int,
    internal_request_id: str,
    layer_name: str,
    source_flat_row_index: int,
) -> str:
    raw = (
        f"{session_id}/{forward_id}/{subforward_id}/"
        f"{internal_request_id}/{layer_name}/{source_flat_row_index}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# JSONL Writer
# ---------------------------------------------------------------------------


class DSATopKJSONLWriter:
    def __init__(self, file_path: str, max_file_bytes: int) -> None:
        self._file_path = file_path
        self._max_file_bytes = max_file_bytes
        self._fp = open(file_path, "a", encoding="utf-8")
        os.chmod(file_path, _FILE_MODE)
        self._queue: queue.Queue[StepRecord | None] = queue.Queue(maxsize=WRITER_QUEUE_DEPTH)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="dsa-topk-writer", daemon=False)
        self._writer_error: Exception | None = None
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                self._write_step_record(item)
                item.future.set_result(True)
            except Exception as exc:
                self._writer_error = exc
                if not item.future.done():
                    item.future.set_exception(exc)

    def _write_line(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, separators=(",", ":"))
        self._fp.write(line + "\n")

    def _write_step_record(self, step: StepRecord) -> None:
        self._write_line(step.manifest)
        for sel in step.selections:
            self._write_line(sel)
        self._write_line(step.step_end)
        self._fp.flush()
        os.fsync(self._fp.fileno())
        size = self._fp.tell()
        if size > self._max_file_bytes:
            raise RuntimeError(
                f"Dump file '{self._file_path}' exceeded MAX_FILE_BYTES: "
                f"{size} > {self._max_file_bytes}"
            )

    def write_session_start(self, session_info: dict[str, Any]) -> None:
        self._write_line(session_info)
        self._fp.flush()
        os.fsync(self._fp.fileno())

    def enqueue_step(self, step: StepRecord) -> None:
        if self._writer_error is not None:
            raise self._writer_error
        self._queue.put(step)

    def write_line_direct(self, record: dict[str, Any]) -> None:
        self._write_line(record)
        self._fp.flush()
        os.fsync(self._fp.fileno())

    def close(self) -> None:
        if self._writer_error is not None:
            self._queue.put(None)
            self._thread.join(timeout=30)
            self._fp.close()
            raise self._writer_error
        self._queue.put(None)
        self._thread.join(timeout=30)
        self._fp.close()


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class DSATopKDumpManager:
    def __init__(
        self,
        config: DSATopKDumpConfig,
        layer_names: list[str],
        index_topk: int,
        device: torch.device,
        num_speculative_tokens: int,
        max_num_seqs: int,
        model_info: dict[str, Any] | None = None,
    ) -> None:
        self._config = config
        self._layer_names = list(layer_names)
        self._index_topk = index_topk
        self._device = device
        self._num_speculative_tokens = num_speculative_tokens
        self._max_num_seqs = max_num_seqs
        self._model_info = model_info or {}

        self._layer_name_to_slot: dict[str, int] = {
            name: i for i, name in enumerate(layer_names)
        }
        self._num_layers = len(layer_names)

        self._session_id = uuid.uuid4().hex[:8]
        self._hostname = socket.gethostname()
        self._pid = os.getpid()
        self._rank_suffix = self._safe_rank_suffix()

        self._forward_id = 0
        self._request_decode_step: dict[str, int] = {}
        self._matched_internal_request_ids: set[str] = set()

        decode_rows = 1 + num_speculative_tokens
        self._max_rows = config.max_requests * decode_rows
        self._npu_buffer: torch.Tensor = torch.zeros(
            (self._num_layers, self._max_rows, index_topk),
            dtype=torch.int32,
            device=device,
        )

        self._current_plan: CapturePlan | None = None
        self._captured_layer_slots: set[int] = set()

        self._writer: DSATopKJSONLWriter | None = None
        self._init_writer()

    def _safe_rank_suffix(self) -> str:
        try:
            from vllm.distributed.utils import get_worker_rank_suffix

            return get_worker_rank_suffix()
        except Exception:
            return f"tp{self._safe_tp_rank()}"

    @staticmethod
    def _safe_tp_rank() -> int:
        try:
            from vllm.distributed import get_tensor_model_parallel_rank

            return get_tensor_model_parallel_rank()
        except Exception:
            return 0

    def _init_writer(self) -> None:
        tag = self._config.tag
        safe_tag = tag.replace(os.sep, "_") if tag else "default"
        safe_host = self._hostname.replace(".", "-")
        fname = (
            f"dsa_indexer_topk__{safe_tag}__{safe_host}"
            f"__{self._rank_suffix}__pid-{self._pid}__session-{self._session_id}.jsonl"
        )
        path = os.path.join(self._config.dump_dir, fname)
        self._writer = DSATopKJSONLWriter(path, self._config.max_file_bytes)
        session_info = self._build_session_start(path)
        self._writer.write_session_start(session_info)
        logger.info(
            "[DSA_TOPK_DUMP] session %s started, file=%s, layers=%d, k=%d",
            self._session_id,
            path,
            self._num_layers,
            self._index_topk,
        )

    def _build_session_start(self, file_path: str) -> dict[str, Any]:
        info: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "session_start",
            "session_id": self._session_id,
            "start_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "deployment_tag": self._config.tag,
            "hostname": self._hostname,
            "pid": self._pid,
            "rank_suffix": self._rank_suffix,
            "model": self._model_info.get("model", ""),
            "model_type": self._model_info.get("model_type", ""),
            "index_topk": self._index_topk,
            "num_hidden_layers": self._model_info.get("num_hidden_layers", self._num_layers),
            "num_speculative_tokens": self._num_speculative_tokens,
            "target_sfa_layer_names": list(self._layer_names),
            "request_correlation_surrogate": self._config.request_id_substr,
            "max_steps": self._config.max_steps,
            "max_requests": self._config.max_requests,
            "max_file_bytes": self._config.max_file_bytes,
            "capture_scope": "decode_raw_absolute_before_remap",
        }
        return info

    # -- lifecycle ----------------------------------------------------------

    def begin_forward(
        self,
        input_batch: "InputBatch",
        scheduler_output: Any,
        attn_metadata: Any,
        positions_cpu: np.ndarray,
        query_start_loc_cpu: np.ndarray,
    ) -> CapturePlan | None:
        forward_id = self._forward_id
        self._forward_id += 1

        rows = self._build_row_descriptors(
            input_batch, scheduler_output, attn_metadata, positions_cpu, query_start_loc_cpu
        )
        if not rows:
            self._current_plan = None
            return None

        layer_names = list(self._layer_names)
        device_row_indices = torch.tensor(
            [r.source_flat_row_index for r in rows],
            dtype=torch.int64,
            device=self._device,
        )

        remap_boundary_rows = None
        boundary = getattr(attn_metadata, "decode_remap_boundary", None)
        boundary_cpu = getattr(attn_metadata, "decode_remap_boundary_cpu_rows", None)
        if boundary_cpu is not None:
            remap_boundary_rows = np.asarray(boundary_cpu, dtype=np.int32).copy()

        plan = CapturePlan(
            forward_id=forward_id,
            rows=rows,
            layer_names=layer_names,
            device_row_indices=device_row_indices,
            remap_boundary_rows=remap_boundary_rows,
        )
        self._current_plan = plan
        self._captured_layer_slots = set()
        return plan

    def _build_row_descriptors(
        self,
        input_batch: "InputBatch",
        scheduler_output: Any,
        attn_metadata: Any,
        positions_cpu: np.ndarray,
        query_start_loc_cpu: np.ndarray,
    ) -> list[RowDescriptor]:
        substr = self._config.request_id_substr
        num_decode_tokens = getattr(attn_metadata, "num_decode_tokens", 0)
        if num_decode_tokens <= 0:
            return []

        compact_req_indices = getattr(attn_metadata, "decode_req_indices_compact_cpu", None)
        if compact_req_indices is None:
            return []

        compact_req_indices = np.asarray(compact_req_indices).reshape(-1)
        valid_row_indices = getattr(attn_metadata, "decode_valid_row_indices", None)
        if valid_row_indices is not None:
            valid_row_indices_np = (
                valid_row_indices.cpu().numpy()
                if isinstance(valid_row_indices, torch.Tensor)
                else np.asarray(valid_row_indices).reshape(-1)
            )
        else:
            valid_row_indices_np = compact_req_indices.astype(np.int64)

        request_ids_list = getattr(attn_metadata, "decode_request_ids_compact", None)
        if request_ids_list is None:
            common_req_ids = getattr(input_batch, "req_ids", None)
            if common_req_ids is not None:
                request_ids_list = [common_req_ids[int(ri)] for ri in compact_req_indices]
            else:
                request_ids_list = []

        if len(request_ids_list) != len(compact_req_indices):
            return []

        rows: list[RowDescriptor] = []
        for idx, (req_idx, src_row) in enumerate(
            zip(compact_req_indices, valid_row_indices_np, strict=False)
        ):
            req_idx = int(req_idx)
            src_row = int(src_row)
            if req_idx < 0:
                continue
            if idx >= len(request_ids_list):
                break
            internal_req_id = request_ids_list[idx]
            if substr and substr not in internal_req_id:
                continue

            if internal_req_id in self._matched_internal_request_ids:
                pass
            elif len(self._matched_internal_request_ids) >= self._config.max_requests:
                raise RuntimeError(
                    f"[DSA_TOPK_DUMP] session-wide MAX_REQUESTS={self._config.max_requests} "
                    f"exceeded by new request '{internal_req_id}'"
                )
            else:
                self._matched_internal_request_ids.add(internal_req_id)

            step = self._request_decode_step.get(internal_req_id, 0)
            if step >= self._config.max_steps:
                continue

            query_position = int(positions_cpu[src_row]) if src_row < len(positions_cpu) else 0
            req_start = (
                int(query_start_loc_cpu[req_idx])
                if req_idx < len(query_start_loc_cpu)
                else 0
            )
            request_query_offset = src_row - req_start

            prompt_len = int(input_batch.num_prompt_tokens[req_idx])
            num_tokens_no_spec = int(input_batch.num_tokens_no_spec[req_idx])

            spec_tokens = scheduler_output.scheduled_spec_decode_tokens.get(internal_req_id, [])
            spec_ids = tuple(int(t) for t in spec_tokens)

            spec_start = num_tokens_no_spec
            spec_end = spec_start + len(spec_ids)
            if spec_start <= query_position < spec_end:
                row_kind = "target_speculative_input"
            else:
                row_kind = "target_input"

            rows.append(
                RowDescriptor(
                    internal_request_id=internal_req_id,
                    request_index=req_idx,
                    source_flat_row_index=src_row,
                    query_position=query_position,
                    request_query_offset=request_query_offset,
                    row_kind=row_kind,
                    prompt_len=prompt_len,
                    num_tokens_no_spec_before_step=num_tokens_no_spec,
                    scheduled_spec_ids=spec_ids,
                )
            )
        return rows

    # -- capture (hot path) -------------------------------------------------

    def capture_layer(
        self,
        layer_name: str,
        raw_topk_indices: torch.Tensor,
    ) -> None:
        plan = self._current_plan
        if plan is None:
            return
        slot = self._layer_name_to_slot.get(layer_name)
        if slot is None:
            raise RuntimeError(f"[DSA_TOPK_DUMP] unknown layer '{layer_name}'")
        if slot in self._captured_layer_slots:
            raise RuntimeError(
                f"[DSA_TOPK_DUMP] duplicate layer call for '{layer_name}' "
                f"in forward {plan.forward_id}"
            )
        num_rows = len(plan.rows)
        if num_rows == 0 or plan.device_row_indices is None:
            return

        src = raw_topk_indices
        if src.dim() == 3 and src.shape[1] == 1:
            gathered = src[plan.device_row_indices, 0, :]
        else:
            gathered = src[plan.device_row_indices, :]
        dest = self._npu_buffer[slot, :num_rows, :]
        dest.copy_(gathered)
        self._captured_layer_slots.add(slot)

    # -- finish (cold path) -------------------------------------------------

    def finish_forward(self, input_batch: "InputBatch") -> None:
        plan = self._current_plan
        if plan is None:
            return
        try:
            num_rows = len(plan.rows)
            num_captured = len(self._captured_layer_slots)
            if num_rows == 0 or num_captured == 0:
                self._current_plan = None
                return

            cpu_positions = (
                self._npu_buffer[:num_captured, :num_rows, :].cpu().numpy()
            )
            captured_slots_sorted = sorted(self._captured_layer_slots)
            step_record = self._build_step_record(
                plan, captured_slots_sorted, cpu_positions, input_batch
            )
            assert self._writer is not None
            self._writer.enqueue_step(step_record)
            step_record.future.result(timeout=300)
        finally:
            self._current_plan = None
            self._captured_layer_slots = set()

    def abort_forward(self) -> None:
        self._current_plan = None
        self._captured_layer_slots = set()

    def _build_step_record(
        self,
        plan: CapturePlan,
        captured_slots: list[int],
        cpu_positions: np.ndarray,
        input_batch: "InputBatch",
    ) -> StepRecord:
        captured_layer_names = [self._layer_names[s] for s in captured_slots]
        expected_rows_manifest: list[dict[str, Any]] = []
        for row in plan.rows:
            expected_rows_manifest.append(
                {
                    "internal_request_id": row.internal_request_id,
                    "source_flat_row_index": row.source_flat_row_index,
                    "query_position": row.query_position,
                    "request_query_offset": row.request_query_offset,
                    "row_kind": row.row_kind,
                }
            )
        expected_count = len(captured_layer_names) * len(plan.rows)
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "forward_manifest",
            "session_id": self._session_id,
            "forward_id": plan.forward_id,
            "subforward_id": 0,
            "model_role": "target",
            "correlation_id": self._config.request_id_substr,
            "expected_layer_names": captured_layer_names,
            "expected_rows": expected_rows_manifest,
            "expected_selection_record_count": expected_count,
        }
        manifest_sha = _compute_manifest_hash(manifest)
        manifest["manifest_sha256"] = manifest_sha

        selections: list[dict[str, Any]] = []
        for layer_slot_idx, layer_slot in enumerate(captured_slots):
            layer_name = self._layer_names[layer_slot]
            layer_index = layer_slot
            for row_idx, row in enumerate(plan.rows):
                positions = cpu_positions[layer_slot_idx, row_idx, :]
                stats = _compute_position_stats(
                    positions,
                    row.query_position,
                    row.prompt_len,
                    row.num_tokens_no_spec_before_step,
                    len(row.scheduled_spec_ids),
                )
                token_ids = _map_token_ids(positions, row.query_position, row, input_batch)

                boundary_val = None
                if plan.remap_boundary_rows is not None:
                    boundary_val = int(plan.remap_boundary_rows[row.source_flat_row_index])

                step = self._request_decode_step.get(row.internal_request_id, 0)

                sel: dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION,
                    "record_type": "selection",
                    "session_id": self._session_id,
                    "forward_id": plan.forward_id,
                    "subforward_id": 0,
                    "manifest_sha256": manifest_sha,
                    "record_id": _compute_record_id(
                        self._session_id,
                        plan.forward_id,
                        0,
                        row.internal_request_id,
                        layer_name,
                        row.source_flat_row_index,
                    ),
                    "correlation_id": self._config.request_id_substr,
                    "request_decode_step": step,
                    "internal_request_id": row.internal_request_id,
                    "actual_phase": "decode",
                    "model_role": "target",
                    "layer_index": layer_index,
                    "layer_name": layer_name,
                    "layer_call_index": 0,
                    "source_flat_row_index": row.source_flat_row_index,
                    "request_query_offset": row.request_query_offset,
                    "row_kind": row.row_kind,
                    "query_position": row.query_position,
                    "eligible_context_len": row.query_position + 1,
                    "batch_final_seq_len": row.query_position + 1,
                    "prompt_len": row.prompt_len,
                    "pre_step_num_computed_tokens": row.num_tokens_no_spec_before_step
                    - (1 if row.row_kind == "target_input" else 0),
                    "num_tokens_no_spec_before_step": row.num_tokens_no_spec_before_step,
                    "remap_boundary": boundary_val,
                    "index_topk": self._index_topk,
                    "selected_positions": positions.astype(np.int32).tolist(),
                    "selected_vocab_token_ids": token_ids,
                    "capture_time_ns": time.time_ns(),
                }
                sel.update(stats)
                selections.append(sel)

        for row in plan.rows:
            self._request_decode_step[row.internal_request_id] = (
                self._request_decode_step.get(row.internal_request_id, 0) + 1
            )

        actual_set = sorted(
            sel["record_id"] for sel in selections
        )
        actual_set_hash = hashlib.sha256(
            json.dumps(actual_set).encode()
        ).hexdigest()

        step_end: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "step_end",
            "session_id": self._session_id,
            "forward_id": plan.forward_id,
            "subforward_id": 0,
            "model_role": "target",
            "manifest_sha256": manifest_sha,
            "actual_record_set_sha256": actual_set_hash,
            "expected_selection_record_count": expected_count,
            "selection_record_count": len(selections),
            "captured_layer_count": len(captured_layer_names),
            "captured_query_row_count": len(plan.rows),
            "complete": True,
        }
        return StepRecord(manifest=manifest, selections=selections, step_end=step_end)

    # -- close --------------------------------------------------------------

    def close(self) -> None:
        if self._writer is None:
            return
        for req_id in sorted(self._matched_internal_request_ids):
            if self._request_decode_step.get(req_id, 0) >= self._config.max_steps:
                self._writer.write_line_direct(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "record_type": "capture_stop",
                        "correlation_id": self._config.request_id_substr,
                        "internal_request_id": req_id,
                        "reason": "max_steps",
                        "captured_steps": self._request_decode_step[req_id],
                    }
                )
        self._writer.write_line_direct(
            {
                "schema_version": SCHEMA_VERSION,
                "record_type": "session_end",
                "session_id": self._session_id,
                "complete_steps": self._forward_id,
                "writer_errors": 0,
            }
        )
        self._writer.close()
        self._writer = None
