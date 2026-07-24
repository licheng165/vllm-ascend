"""CPU unit tests for DSA top-k dump module.

Tests the pure functions (token mapping, position stats, config) without NPU.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest

from vllm_ascend.worker.dsa_topk_dumper import (
    DSATopKDumpConfig,
    RowDescriptor,
    _compute_manifest_hash,
    _compute_position_stats,
    _compute_record_id,
    _map_token_ids,
    reserved_bytes,
)


class TestConfig:
    def test_disabled_by_default(self):
        os.environ.pop("VLLM_ASCEND_DSA_TOPK_DUMP", None)
        cfg = DSATopKDumpConfig.from_env()
        assert not cfg.enabled

    def test_basic_parsing(self):
        os.environ["VLLM_ASCEND_DSA_TOPK_DUMP"] = "1"
        os.environ["VLLM_ASCEND_DSA_TOPK_DUMP_DIR"] = tempfile.gettempdir()
        os.environ["VLLM_ASCEND_DSA_TOPK_DUMP_TAG"] = "test"
        os.environ["VLLM_ASCEND_DSA_TOPK_DUMP_REQUEST_ID_SUBSTR"] = "req-001"
        os.environ["VLLM_ASCEND_DSA_TOPK_DUMP_MAX_STEPS"] = "64"
        os.environ["VLLM_ASCEND_DSA_TOPK_DUMP_MAX_REQUESTS"] = "2"
        os.environ["VLLM_ASCEND_DSA_TOPK_DUMP_TP_RANKS"] = "0,1"
        os.environ["VLLM_ASCEND_DSA_TOPK_DUMP_INCLUDE_MTP"] = "0"
        os.environ["VLLM_ASCEND_DSA_TOPK_DUMP_EXPECTED_K"] = "2048"
        cfg = DSATopKDumpConfig.from_env()
        assert cfg.enabled
        assert cfg.max_steps == 64
        assert cfg.max_requests == 2
        assert cfg.tp_ranks == {0, 1}
        assert cfg.expected_k == 2048
        cfg.validate()

    def test_tp_ranks_all(self):
        os.environ["VLLM_ASCEND_DSA_TOPK_DUMP_TP_RANKS"] = "all"
        cfg = DSATopKDumpConfig.from_env()
        assert cfg.tp_ranks is None

    def test_include_mtp_fails(self):
        os.environ["VLLM_ASCEND_DSA_TOPK_DUMP_INCLUDE_MTP"] = "1"
        cfg = DSATopKDumpConfig.from_env()
        with pytest.raises(ValueError, match="not supported"):
            cfg.validate()

    def test_max_steps_must_be_positive(self):
        os.environ["VLLM_ASCEND_DSA_TOPK_DUMP_MAX_STEPS"] = "0"
        cfg = DSATopKDumpConfig.from_env()
        with pytest.raises(ValueError, match="positive"):
            cfg.validate()


class TestReservedBytes:
    def test_basic_calculation(self):
        result = reserved_bytes(78, 1, 2, 2048)
        assert result == 78 * 1 * 2 * 2048 * 4


class TestPositionStats:
    def test_basic_stats(self):
        positions = np.array([0, 50, 99, 100, -1], dtype=np.int32)
        stats = _compute_position_stats(positions, 100, 50, 100, 1)
        assert stats["valid_count"] == 4
        assert stats["invalid_position_count"] == 1
        assert sum(stats["position_decile_counts"]) == stats["valid_count"]
        assert stats["head_10pct_count"] + stats["middle_80pct_count"] + stats["tail_10pct_count"] == 4
        assert stats["prompt_position_count"] == 1
        assert stats["unique_position_count"] == 4

    def test_all_valid(self):
        positions = np.array([0, 1, 2, 3], dtype=np.int32)
        stats = _compute_position_stats(positions, 3, 2, 3, 1)
        assert stats["valid_count"] == 4
        assert stats["invalid_position_count"] == 0


class TestMapTokenIds:
    def _make_input_batch(self, token_ids, is_token_ids):
        batch = MagicMock()
        batch.token_ids_cpu = np.array([token_ids], dtype=np.int32)
        batch.is_token_ids = np.array([is_token_ids], dtype=bool)
        return batch

    def test_history_mapping(self):
        tokens = [100, 200, 300, 400]
        is_ids = [True, True, True, True]
        batch = self._make_input_batch(tokens, is_ids)
        positions = np.array([0, 1, 2, 3], dtype=np.int32)
        row = RowDescriptor(
            internal_request_id="req-0",
            request_index=0,
            source_flat_row_index=0,
            query_position=3,
            request_query_offset=0,
            row_kind="target_input",
            prompt_len=2,
            num_tokens_no_spec_before_step=4,
            scheduled_spec_ids=(),
        )
        result = _map_token_ids(positions, 3, row, batch)
        assert result == [100, 200, 300, 400]

    def test_speculative_overlay(self):
        tokens = [100, 200, 0, 0]
        is_ids = [True, True, False, False]
        batch = self._make_input_batch(tokens, is_ids)
        positions = np.array([0, 1, 2, 3], dtype=np.int32)
        row = RowDescriptor(
            internal_request_id="req-0",
            request_index=0,
            source_flat_row_index=0,
            query_position=3,
            request_query_offset=0,
            row_kind="target_speculative_input",
            prompt_len=2,
            num_tokens_no_spec_before_step=2,
            scheduled_spec_ids=(500, 600),
        )
        result = _map_token_ids(positions, 3, row, batch)
        assert result == [100, 200, 500, 600]

    def test_prompt_embedding_null(self):
        tokens = [0, 200]
        is_ids = [False, True]
        batch = self._make_input_batch(tokens, is_ids)
        positions = np.array([0, 1], dtype=np.int32)
        row = RowDescriptor(
            internal_request_id="req-0",
            request_index=0,
            source_flat_row_index=0,
            query_position=1,
            request_query_offset=0,
            row_kind="target_input",
            prompt_len=2,
            num_tokens_no_spec_before_step=2,
            scheduled_spec_ids=(),
        )
        result = _map_token_ids(positions, 1, row, batch)
        assert result[0] is None
        assert result[1] == 200

    def test_invalid_position(self):
        tokens = [100]
        is_ids = [True]
        batch = self._make_input_batch(tokens, is_ids)
        positions = np.array([-1, 0], dtype=np.int32)
        row = RowDescriptor(
            internal_request_id="req-0",
            request_index=0,
            source_flat_row_index=0,
            query_position=0,
            request_query_offset=0,
            row_kind="target_input",
            prompt_len=1,
            num_tokens_no_spec_before_step=1,
            scheduled_spec_ids=(),
        )
        result = _map_token_ids(positions, 0, row, batch)
        assert result[0] is None
        assert result[1] == 100


class TestManifestHash:
    def test_deterministic(self):
        manifest = {"a": 1, "b": [1, 2, 3]}
        h1 = _compute_manifest_hash(manifest)
        h2 = _compute_manifest_hash(manifest)
        assert h1 == h2

    def test_different_content(self):
        assert _compute_manifest_hash({"a": 1}) != _compute_manifest_hash({"a": 2})


class TestRecordId:
    def test_unique_per_layer(self):
        rid1 = _compute_record_id("sess", 1, 0, "req-0", "layer0", 0)
        rid2 = _compute_record_id("sess", 1, 0, "req-0", "layer1", 0)
        assert rid1 != rid2

    def test_unique_per_row(self):
        rid1 = _compute_record_id("sess", 1, 0, "req-0", "layer0", 0)
        rid2 = _compute_record_id("sess", 1, 0, "req-0", "layer0", 1)
        assert rid1 != rid2
