# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for mooncake_transfer_telemetry helper.

These tests are CPU-only and do not require NPU hardware.
"""

import json
import logging
import os
import unittest
from unittest.mock import patch

from vllm_ascend.distributed.kv_transfer.utils import mooncake_transfer_telemetry as mc


def _capture_log_records():
    """Capture log records emitted to the telemetry helper logger."""
    records: list[logging.LogRecord] = []

    class _Handler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Handler(level=logging.INFO)
    logger = logging.getLogger(mc.__name__)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return records, handler


def _extract_metric_json(record: logging.LogRecord) -> dict | None:
    msg = record.getMessage()
    prefix = mc.LOG_PREFIX + " "
    if not msg.startswith(prefix):
        return None
    return json.loads(msg[len(prefix) :])


class TestSafeSummarizeStoreInput(unittest.TestCase):
    def test_valid_input(self):
        summary = mc.safe_summarize_store_input(
            keys=["a", "b"],
            addrs=[[1], [2]],
            sizes=[[100, 200], [300]],
        )
        self.assertTrue(summary["input_shape_valid"])
        self.assertEqual(summary["object_count"], 2)
        self.assertEqual(summary["slice_count"], 3)
        self.assertEqual(summary["per_object_bytes"], [300, 300])
        self.assertEqual(summary["total_bytes"], 600)

    def test_mismatched_lengths(self):
        summary = mc.safe_summarize_store_input(
            keys=["a", "b"],
            addrs=[[1]],
            sizes=[[100]],
        )
        self.assertFalse(summary["input_shape_valid"])
        self.assertIsNone(summary["total_bytes"])

    def test_empty_keys(self):
        summary = mc.safe_summarize_store_input(keys=[], addrs=[], sizes=[])
        self.assertTrue(summary["input_shape_valid"])
        self.assertEqual(summary["object_count"], 0)
        self.assertEqual(summary["total_bytes"], 0)

    def test_zero_key_zero_slices(self):
        summary = mc.safe_summarize_store_input(keys=[], addrs=[], sizes=[])
        self.assertEqual(summary["slice_count"], 0)

    def test_nested_sizes_with_non_numbers(self):
        summary = mc.safe_summarize_store_input(
            keys=["a"],
            addrs=[[1]],
            sizes=[["bad"]],  # type: ignore[list-item]
        )
        self.assertFalse(summary["input_shape_valid"])
        self.assertIsNone(summary["total_bytes"])


class TestP2PReadPayload(unittest.TestCase):
    def test_success(self):
        payload = mc.build_p2p_read_metric_payload(
            start_perf_ns=1000,
            end_perf_ns=2000,
            start_wall_time_ns=5000,
            end_wall_time_ns=6000,
            attempted_bytes=1024,
            completed_bytes=1024,
            attempted_items=4,
            completed_items=4,
            outcome="success",
            return_code=0,
            exception_type=None,
            request_id="d-req",
            remote_request_id="p-req",
            local_engine_id="d-eng",
            peer_engine_id="p-eng",
            remote_host="10.0.0.1",
            session_id="10.0.0.1:1234",
            same_host=False,
            num_blocks=8,
            num_groups=4,
            tp_rank=3,
            dp_rank_global=0,
            dp_rank_local=0,
            kv_role="kv_consumer",
            pull_offset=0,
            local_memory_type="npu",
            network_type="npu_side",
            network_type_source="connector_inference",
        )
        self.assertEqual(payload["outcome"], "success")
        self.assertEqual(payload["duration_ns"], 1000)
        self.assertEqual(payload["completed_bytes"], 1024)
        self.assertEqual(payload["attempted_bytes"], 1024)
        self.assertEqual(payload["return_code_counts"], {"0": 1})
        self.assertEqual(payload["network_type"], "npu_side")

    def test_failure_null_bytes(self):
        payload = mc.build_p2p_read_metric_payload(
            start_perf_ns=1000,
            end_perf_ns=2000,
            start_wall_time_ns=5000,
            end_wall_time_ns=6000,
            attempted_bytes=1024,
            completed_bytes=None,
            attempted_items=4,
            completed_items=None,
            outcome="error",
            return_code=-1,
            exception_type="RuntimeError",
            request_id="d-req",
            remote_request_id="p-req",
            local_engine_id="d-eng",
            peer_engine_id="p-eng",
            remote_host="10.0.0.1",
            session_id="10.0.0.1:1234",
            same_host=False,
            num_blocks=8,
            num_groups=4,
            tp_rank=3,
            dp_rank_global=0,
            dp_rank_local=0,
            kv_role="kv_consumer",
            pull_offset=0,
            local_memory_type="npu",
            network_type="npu_side",
            network_type_source="connector_inference",
        )
        self.assertEqual(payload["outcome"], "error")
        self.assertIsNone(payload["completed_bytes"])
        self.assertIsNone(payload["completed_items"])
        self.assertEqual(payload["return_code_counts"], {"-1": 1})


class TestStorePutPayload(unittest.TestCase):
    def _summary(self):
        return mc.safe_summarize_store_input(
            keys=["a", "b", "c", "d"],
            addrs=[[1], [2], [3], [4]],
            sizes=[[32], [32], [32], [32]],
        )

    def test_all_success(self):
        payload = mc.build_store_put_metric_payload(
            start_perf_ns=100,
            end_perf_ns=200,
            start_wall_time_ns=500,
            end_wall_time_ns=600,
            input_summary=self._summary(),
            result=[0, 0, 0, 0],
            exception_type=None,
            configured_protocol="ascend",
            fabric_mem_enabled=False,
            tp_rank=0,
            dp_rank_global=0,
            dp_rank_local=0,
        )
        self.assertEqual(payload["outcome"], "success")
        self.assertEqual(payload["success_count"], 4)
        self.assertEqual(payload["failure_count"], 0)
        self.assertEqual(payload["completed_bytes"], 128)
        self.assertEqual(payload["attempted_bytes"], 128)

    def test_partial(self):
        payload = mc.build_store_put_metric_payload(
            start_perf_ns=100,
            end_perf_ns=200,
            start_wall_time_ns=500,
            end_wall_time_ns=600,
            input_summary=self._summary(),
            result=[0, -704, 0, -800],
            exception_type=None,
            configured_protocol="ascend",
            fabric_mem_enabled=False,
            tp_rank=0,
            dp_rank_global=0,
            dp_rank_local=0,
        )
        self.assertEqual(payload["outcome"], "partial")
        self.assertEqual(payload["success_count"], 2)
        self.assertEqual(payload["failure_count"], 2)
        self.assertEqual(payload["completed_bytes"], 64)
        self.assertIn("-704", payload["return_code_counts"])
        self.assertIn("-800", payload["return_code_counts"])

    def test_all_fail(self):
        payload = mc.build_store_put_metric_payload(
            start_perf_ns=100,
            end_perf_ns=200,
            start_wall_time_ns=500,
            end_wall_time_ns=600,
            input_summary=self._summary(),
            result=[-600, -600, -600, -600],
            exception_type=None,
            configured_protocol="ascend",
            fabric_mem_enabled=False,
            tp_rank=0,
            dp_rank_global=0,
            dp_rank_local=0,
        )
        self.assertEqual(payload["outcome"], "error")
        self.assertEqual(payload["completed_bytes"], 0)

    def test_exception(self):
        payload = mc.build_store_put_metric_payload(
            start_perf_ns=100,
            end_perf_ns=200,
            start_wall_time_ns=500,
            end_wall_time_ns=600,
            input_summary=self._summary(),
            result=None,
            exception_type="RuntimeError",
            configured_protocol="ascend",
            fabric_mem_enabled=False,
            tp_rank=0,
            dp_rank_global=0,
            dp_rank_local=0,
        )
        self.assertEqual(payload["outcome"], "error")
        self.assertIsNone(payload["completed_bytes"])
        self.assertIsNone(payload["completed_items"])

    def test_zero_key_zero_result_noop(self):
        summary = mc.safe_summarize_store_input(keys=[], addrs=[], sizes=[])
        payload = mc.build_store_put_metric_payload(
            start_perf_ns=100,
            end_perf_ns=200,
            start_wall_time_ns=500,
            end_wall_time_ns=600,
            input_summary=summary,
            result=[],
            exception_type=None,
            configured_protocol="ascend",
            fabric_mem_enabled=False,
            tp_rank=0,
            dp_rank_global=0,
            dp_rank_local=0,
        )
        self.assertEqual(payload["outcome"], "success")
        self.assertEqual(payload["completed_bytes"], 0)

    def test_nonempty_keys_empty_result(self):
        summary = mc.safe_summarize_store_input(keys=["a"], addrs=[[1]], sizes=[[32]])
        payload = mc.build_store_put_metric_payload(
            start_perf_ns=100,
            end_perf_ns=200,
            start_wall_time_ns=500,
            end_wall_time_ns=600,
            input_summary=summary,
            result=[],
            exception_type=None,
            configured_protocol="ascend",
            fabric_mem_enabled=False,
            tp_rank=0,
            dp_rank_global=0,
            dp_rank_local=0,
        )
        self.assertEqual(payload["outcome"], "error")
        self.assertTrue(payload["invalid_result"])


class TestStoreGetPayload(unittest.TestCase):
    def test_success_uses_return_values(self):
        summary = mc.safe_summarize_store_input(
            keys=["a", "b"],
            addrs=[[1], [2]],
            sizes=[[64], [64]],
        )
        payload = mc.build_store_get_metric_payload(
            start_perf_ns=100,
            end_perf_ns=200,
            start_wall_time_ns=500,
            end_wall_time_ns=600,
            input_summary=summary,
            result=[48, 32],
            exception_type=None,
            configured_protocol="ascend",
            fabric_mem_enabled=False,
            tp_rank=0,
            dp_rank_global=0,
            dp_rank_local=0,
        )
        self.assertEqual(payload["outcome"], "success")
        # GET attempted_bytes must be null, not buffer capacity
        self.assertIsNone(payload["attempted_bytes"])
        self.assertEqual(payload["buffer_capacity_bytes"], 128)
        self.assertEqual(payload["completed_bytes"], 80)

    def test_partial(self):
        summary = mc.safe_summarize_store_input(
            keys=["a", "b"],
            addrs=[[1], [2]],
            sizes=[[64], [64]],
        )
        payload = mc.build_store_get_metric_payload(
            start_perf_ns=100,
            end_perf_ns=200,
            start_wall_time_ns=500,
            end_wall_time_ns=600,
            input_summary=summary,
            result=[48, -704],
            exception_type=None,
            configured_protocol="ascend",
            fabric_mem_enabled=False,
            tp_rank=0,
            dp_rank_global=0,
            dp_rank_local=0,
        )
        self.assertEqual(payload["outcome"], "partial")
        self.assertEqual(payload["completed_bytes"], 48)
        self.assertIn("-704", payload["return_code_counts"])


class TestStoreExistsPayload(unittest.TestCase):
    def test_hit_miss_error(self):
        payload = mc.build_store_exists_metric_payload(
            start_perf_ns=100,
            end_perf_ns=200,
            start_wall_time_ns=500,
            end_wall_time_ns=600,
            keys=["a", "b", "c"],
            result=[1, 0, -900],
            exception_type=None,
            configured_protocol="ascend",
            fabric_mem_enabled=False,
            tp_rank=0,
            dp_rank_global=0,
            dp_rank_local=0,
        )
        self.assertEqual(payload["outcome"], "partial")
        self.assertEqual(payload["hit_count"], 1)
        self.assertEqual(payload["miss_count"], 1)
        self.assertEqual(payload["error_count"], 1)
        self.assertEqual(payload["attempted_bytes"], 0)
        self.assertEqual(payload["completed_bytes"], 0)
        self.assertEqual(payload["phase"], "control_plane")

    def test_all_success(self):
        payload = mc.build_store_exists_metric_payload(
            start_perf_ns=100,
            end_perf_ns=200,
            start_wall_time_ns=500,
            end_wall_time_ns=600,
            keys=["a", "b"],
            result=[1, 0],
            exception_type=None,
            configured_protocol="ascend",
            fabric_mem_enabled=False,
            tp_rank=0,
            dp_rank_global=0,
            dp_rank_local=0,
        )
        self.assertEqual(payload["outcome"], "success")

    def test_zero_key_zero_result_noop(self):
        payload = mc.build_store_exists_metric_payload(
            start_perf_ns=100,
            end_perf_ns=200,
            start_wall_time_ns=500,
            end_wall_time_ns=600,
            keys=[],
            result=[],
            exception_type=None,
            configured_protocol="ascend",
            fabric_mem_enabled=False,
            tp_rank=0,
            dp_rank_global=0,
            dp_rank_local=0,
        )
        self.assertEqual(payload["outcome"], "success")


class TestEmitDisabled(unittest.TestCase):
    def test_disabled_no_log(self):
        records, handler = _capture_log_records()
        try:
            with patch.object(mc.envs, "VLLM_ASCEND_MOONCAKE_TRANSFER_METRICS", False):
                mc.emit_mooncake_transfer_metric({"event": "api_call_end"})
            metric_records = [r for r in records if mc.LOG_PREFIX in r.getMessage()]
            self.assertEqual(len(metric_records), 0)
        finally:
            mc.logger.removeHandler(handler)


class TestEmitEnabled(unittest.TestCase):
    def setUp(self):
        os.environ["VLLM_ASCEND_MOONCAKE_TRANSFER_METRICS"] = "1"

    def tearDown(self):
        os.environ.pop("VLLM_ASCEND_MOONCAKE_TRANSFER_METRICS", None)

    def test_emits_valid_json(self):
        records, handler = _capture_log_records()
        try:
            payload = mc.build_publish_ready_payload(
                end_wall_time_ns=999,
                remote_request_id="p-req",
                local_engine_id="p-eng",
                num_blocks=10,
                kv_role="kv_producer",
                tp_rank=0,
                dp_rank_global=0,
                dp_rank_local=0,
            )
            mc.emit_mooncake_transfer_metric(payload)
            metric_records = [r for r in records if mc.LOG_PREFIX in r.getMessage()]
            self.assertEqual(len(metric_records), 1)
            data = _extract_metric_json(metric_records[0])
            self.assertIsNotNone(data)
            self.assertEqual(data["schema"], "mooncake_transfer.v1")
            self.assertEqual(data["event"], "publish_ready")
            self.assertEqual(data["remote_request_id"], "p-req")
            self.assertIn("pid", data)
            self.assertIn("thread_id", data)
        finally:
            mc.logger.removeHandler(handler)

    def test_helper_swallows_errors(self):
        records, handler = _capture_log_records()
        try:
            with patch("builtins.json.dumps", side_effect=ValueError("boom")):
                mc.emit_mooncake_transfer_metric({"event": "api_call_end"})
            # Must not raise; debug log is acceptable
        finally:
            mc.logger.removeHandler(handler)


class TestClassifyNetworkType(unittest.TestCase):
    def test_npu(self):
        net, src = mc.classify_p2p_network_type("npu")
        self.assertEqual(net, "npu_side")
        self.assertEqual(src, "connector_inference")

    def test_cuda(self):
        net, src = mc.classify_p2p_network_type("cuda")
        self.assertEqual(net, "npu_side")

    def test_cpu(self):
        net, src = mc.classify_p2p_network_type("cpu")
        self.assertEqual(net, "unknown")


if __name__ == "__main__":
    unittest.main()
