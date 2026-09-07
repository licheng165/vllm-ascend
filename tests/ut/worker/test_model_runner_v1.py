import unittest
from contextlib import nullcontext
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

import vllm_ascend.worker.model_runner_v1 as model_runner_module
from vllm_ascend.ascend_forward_context import (
    STAGED_SFA_SINGLETON_GRAPH_KEY,
    StagedSFAGraphKey,
    StagedSFAQueryProfile,
)
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.utils import (
    StagedSFARouteAction,
    StagedSFARouteReason,
)
from vllm_ascend.worker.block_table import MultiGroupBlockTable
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class TestFixedDecodeLayoutArrays(unittest.TestCase):
    def test_q1_layout_is_identity(self):
        req_indices, offsets, cumulative = (
            model_runner_module._fixed_decode_layout_arrays(
                3,
                1,
                np.dtype(np.int64),
            )
        )

        np.testing.assert_array_equal(
            req_indices,
            np.array([0, 1, 2], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            offsets,
            np.zeros(3, dtype=np.int64),
        )
        np.testing.assert_array_equal(
            cumulative,
            np.array([1, 2, 3], dtype=np.int64),
        )

    def test_mtp2_layout_is_request_major(self):
        req_indices, offsets, cumulative = (
            model_runner_module._fixed_decode_layout_arrays(
                3,
                2,
                np.dtype(np.int32),
            )
        )

        np.testing.assert_array_equal(
            req_indices,
            np.array([0, 0, 1, 1, 2, 2], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            offsets,
            np.array([0, 1, 0, 1, 0, 1], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            cumulative,
            np.array([2, 4, 6], dtype=np.int32),
        )

    def test_layout_rejects_mtp_above_two(self):
        with self.assertRaisesRegex(ValueError, "MTP=3"):
            model_runner_module._fixed_decode_layout_arrays(
                3,
                3,
                np.dtype(np.int32),
            )

    def test_layout_rejects_empty_request_capacity(self):
        with self.assertRaisesRegex(ValueError, "requests=0"):
            model_runner_module._fixed_decode_layout_arrays(
                0,
                2,
                np.dtype(np.int32),
            )

    def test_mtp2_positions_follow_each_request_frontier(self):
        positions = np.empty(6, dtype=np.int64)
        _, offsets, _ = (
            model_runner_module._fixed_decode_layout_arrays(
                3,
                2,
                np.dtype(np.int64),
            )
        )

        model_runner_module._fill_fixed_decode_positions(
            positions,
            np.array([100, 200, 300], dtype=np.int64),
            offsets,
            3,
            2,
        )

        np.testing.assert_array_equal(
            positions,
            np.array([100, 101, 200, 201, 300, 301]),
        )

    def test_positions_reject_mismatched_buffers(self):
        with self.assertRaisesRegex(
            ValueError,
            "do not match",
        ):
            model_runner_module._fill_fixed_decode_positions(
                np.empty(3, dtype=np.int64),
                np.array([100, 200], dtype=np.int64),
                np.array([0, 1, 0, 1], dtype=np.int64),
                2,
                2,
            )


class TestMTPPlaceholderForwardInputs(unittest.TestCase):
    @staticmethod
    def _build_runner():
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.input_ids = SimpleNamespace(
            cpu=torch.tensor([11, -1, 33, -1], dtype=torch.int32),
            gpu=torch.tensor([11, -1, 33, -1], dtype=torch.int32),
        )
        return runner

    def test_placeholder_tokens_are_sanitized_only_for_forward(self):
        runner = self._build_runner()
        scheduler_output = SimpleNamespace(
            scheduled_spec_decode_tokens={"req0": [-1]},
        )

        runner._sanitize_placeholder_input_ids_for_forward(
            scheduler_output,
            num_forward_tokens=4,
        )

        self.assertEqual(runner.input_ids.gpu.tolist(), [11, 0, 33, 0])
        self.assertEqual(runner.input_ids.cpu.tolist(), [11, -1, 33, -1])
        self.assertEqual(
            scheduler_output.scheduled_spec_decode_tokens,
            {"req0": [-1]},
        )

    def test_forward_input_is_unchanged_without_placeholder_metadata(self):
        runner = self._build_runner()
        scheduler_output = SimpleNamespace(
            scheduled_spec_decode_tokens={"req0": [7]},
        )

        runner._sanitize_placeholder_input_ids_for_forward(
            scheduler_output,
            num_forward_tokens=4,
        )

        self.assertEqual(runner.input_ids.gpu.tolist(), [11, -1, 33, -1])

    def test_async_real_draft_survives_forward_sanitization(self):
        runner = self._build_runner()
        runner.device = torch.device("cpu")
        runner.pcp_size = 1
        runner.arange_np = np.arange(4)
        # Real drafts have been scattered by _prepare_input_ids. The second
        # request has no previous proposal, so its bootstrap draft remains -1.
        runner.input_ids.gpu[:] = torch.tensor([11, 42, 33, -1])
        scheduler_output = SimpleNamespace(
            scheduled_spec_decode_tokens={"req0": [-1], "new": [-1]},
        )
        with patch.object(torch.Tensor, "pin_memory", lambda tensor: tensor):
            metadata = runner._calc_spec_decode_metadata(
                np.array([1, 1]),
                np.array([2, 4]),
                None,
            )
        runner._sanitize_placeholder_input_ids_for_forward(scheduler_output, 4)

        self.assertEqual(metadata.draft_token_ids.tolist(), [42, -1])
        self.assertEqual(runner.input_ids.gpu.tolist(), [11, 42, 33, 0])
        self.assertEqual(scheduler_output.scheduled_spec_decode_tokens["req0"], [-1])

    def test_invalid_verification_width_fails_before_device_preparation(self):
        for counts, drafts, bad_req in (
            ({"short": 1}, {"short": [-1]}, "short"),
            ({"valid": 2, "short": 1}, {"valid": [42], "short": [-1]}, "short"),
            ({"valid": 2}, {"missing": []}, "missing"),
        ):
            with self.subTest(counts=counts, drafts=drafts):
                runner = NPUModelRunner.__new__(NPUModelRunner)
                runner.input_batch = SimpleNamespace(num_reqs=len(counts))
                schedule = SimpleNamespace(
                    total_num_scheduled_tokens=sum(counts.values()),
                    num_scheduled_tokens=counts,
                    scheduled_spec_decode_tokens=drafts,
                )
                with (
                    patch.object(runner, "_refresh_layerwise_prefill_block_tables") as refresh,
                    patch.object(runner, "_prepare_input_ids") as prepare_ids,
                    self.assertRaisesRegex(RuntimeError, f"Invalid speculative schedule: req={bad_req}"),
                ):
                    runner._prepare_inputs(schedule, np.array(list(counts.values())))
                refresh.assert_not_called()
                prepare_ids.assert_not_called()

    def test_metadata_rejects_request_local_underflow_before_gather(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.pcp_size = 1
        for drafts, cumulative, bad_row in (([1], [1], 0), ([1, 1], [2, 3], 1)):
            with self.subTest(cumulative=cumulative), self.assertRaisesRegex(
                ValueError, f"scheduled span: row={bad_row}"
            ):
                # No input buffer exists: validation must precede any gather.
                runner._calc_spec_decode_metadata(np.array(drafts), np.array(cumulative), None)

    def test_mixed_prefill_and_verify_rows_stay_in_their_request_spans(self):
        runner = self._build_runner()
        runner.device = torch.device("cpu")
        runner.pcp_size = 1
        runner.arange_np = np.arange(6)
        runner.input_ids.gpu = torch.tensor([11, 12, 13, 14, 21, 42])
        with patch.object(torch.Tensor, "pin_memory", lambda tensor: tensor):
            metadata = runner._calc_spec_decode_metadata(
                np.array([0, 1]), np.array([4, 6]), None,
            )
        self.assertEqual(metadata.logits_indices.tolist(), [3, 4, 5])
        self.assertEqual(metadata.draft_token_ids.tolist(), [42])


class TestMTPAcceptanceDiagnostics(unittest.TestCase):
    def setUp(self):
        self.runner = NPUModelRunner.__new__(NPUModelRunner)
        self.runner.speculative_config = SimpleNamespace(method="mtp")
        self.runner.pcp_size = 1
        self.runner.use_async_scheduling = True
        self.runner.requests = {"req0": object()}
        self.runner.input_batch = SimpleNamespace(
            req_ids=["req0"],
            vocab_size=100,
            sampling_metadata=SimpleNamespace(all_greedy=True),
        )
        self.runner.input_ids = SimpleNamespace(gpu=torch.tensor([11, 42]))
        self.runner.positions = SimpleNamespace(gpu=torch.tensor([131621, 131622]))
        self.runner.num_discarded_requests = 0
        self.runner.discard_request_indices = SimpleNamespace(np=np.array([], dtype=np.int32))
        self.scheduler_output = SimpleNamespace(
            scheduled_spec_decode_tokens={"req0": [-1]},
            num_scheduled_tokens={"req0": 2},
        )
        self.metadata = SimpleNamespace(
            draft_token_ids=torch.tensor([42]),
            num_draft_tokens=[1],
            logits_indices=torch.tensor([0, 1]),
        )
        self.logits = torch.zeros(2, 100)
        self.logits[0, 42] = 1
        self.logits[1, 99] = 1
        self.output = SimpleNamespace(sampled_token_ids=torch.tensor([[42, 99]]))
        for patcher in (
            patch.object(model_runner_module.envs_ascend, "VLLM_ASCEND_MTP_ACCEPT_DIAG", True),
            patch.object(model_runner_module, "get_tp_group", return_value=SimpleNamespace(rank_in_group=0)),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def trace(self):
        with patch.object(model_runner_module.logger, "info") as log:
            self.runner._trace_mtp_acceptance(self.scheduler_output, self.metadata, self.logits, self.output)
        return [call.args[0] % call.args[1:] for call in log.call_args_list]

    def test_reports_device_candidates_not_scheduler_placeholders(self):
        self.runner._draft_token_ids = torch.tensor([[88]])  # Never read the next proposal.
        (message,) = self.trace()
        self.assertIn("draft_ids=[42]", message)
        self.assertIn("raw_output=[42, 99] accepted=1 worker_placeholder=False", message)
        self.assertIn("input_ids=[11, 42] positions=[131621, 131622]", message)
        self.assertIn("raw_target_argmax=[42, 99] logits_nan=False", message)
        self.assertIn("scheduler_placeholder=True", message)
        self.assertIn("logits_indices=[0, 1] num_scheduled_tokens=2 num_drafts=1", message)

    def test_reports_negative_metadata_indices_without_clamping(self):
        self.scheduler_output.num_scheduled_tokens["req0"] = 1
        self.runner.input_ids.gpu = torch.tensor([42, 0])
        self.runner.positions.gpu = torch.tensor([8386, 0])
        # Simulate corrupt metadata for the diagnostic itself. The real
        # metadata builder now rejects this layout before a device gather.
        self.metadata.logits_indices = torch.tensor([-1, 0])
        cpu = MagicMock(wraps=torch.Tensor.cpu)
        with patch.object(torch.Tensor, "cpu", lambda tensor: cpu(tensor)):
            (message,) = self.trace()
        cpu.assert_called_once()
        self.assertIn("verify_step=0", message)
        self.assertIn("input_ids=[0, 42] positions=[0, 8386]", message)
        self.assertIn("logits_nan=False", message)
        self.assertIn("logits_indices=[-1, 0] num_scheduled_tokens=1 num_drafts=1", message)
        self.assertEqual(self.metadata.logits_indices.tolist(), [-1, 0])

    def test_reports_rejection_and_actual_bootstrap_placeholder(self):
        self.output.sampled_token_ids[:] = torch.tensor([[99, -1]])
        self.metadata.draft_token_ids[:] = -1
        self.logits[1, 99] = float("nan")
        (message,) = self.trace()
        self.assertIn("draft_ids=[-1] raw_output=[99, -1] accepted=0 worker_placeholder=True", message)
        self.assertIn("logits_nan=True", message)

    def test_disabled_and_other_tp_ranks_do_not_touch_device_tensors(self):
        self.scheduler_output = self.metadata = self.logits = self.output = object()
        for enabled, rank in ((False, 0), (True, 1)):
            with (
                patch.object(model_runner_module.envs_ascend, "VLLM_ASCEND_MTP_ACCEPT_DIAG", enabled),
                patch.object(model_runner_module, "get_tp_group", return_value=SimpleNamespace(rank_in_group=rank)),
                patch.object(torch, "cat", side_effect=AssertionError("unexpected device work")),
            ):
                self.assertEqual(self.trace(), [])

    def test_sampling_is_bounded_and_finished_state_is_pruned(self):
        self.runner._mtp_accept_diag_steps = {"finished": 800, "req0": 3}
        with patch.object(torch, "cat", side_effect=AssertionError("unexpected device work")):
            self.assertEqual(self.trace(), [])
        self.assertNotIn("finished", self.runner._mtp_accept_diag_steps)
        self.runner._mtp_accept_diag_steps["req0"] = 128
        self.assertEqual(len(self.trace()), 1)
        self.runner.num_discarded_requests = 1
        self.runner.discard_request_indices.np = np.array([0])
        self.assertEqual(self.trace(), [])

    def test_pcp_layout_is_not_indexed_as_local_inputs(self):
        self.runner.pcp_size = 2
        with patch.object(torch, "cat", side_effect=AssertionError("unexpected device work")):
            self.assertEqual(self.trace(), [])

    def test_request_cap_defers_without_starving_later_rows(self):
        req_ids = [f"req{i}" for i in range(6)]
        self.runner.requests = dict.fromkeys(req_ids, object())
        self.runner.input_batch.req_ids = req_ids
        self.scheduler_output.num_scheduled_tokens = dict.fromkeys(req_ids, 2)
        self.metadata.num_draft_tokens = [1] * 6
        self.metadata.draft_token_ids = torch.tensor([42] * 6)
        self.metadata.logits_indices = torch.arange(12)
        self.runner.input_ids.gpu = torch.tensor([11, 42] * 6)
        self.runner.positions.gpu = torch.arange(12)
        self.logits = torch.zeros(12, 100)
        self.output.sampled_token_ids = torch.tensor([[42, 99]] * 6)
        messages = [self.trace() for _ in range(5)]
        self.assertTrue(all(len(batch) <= 4 for batch in messages))
        self.assertTrue(any("req=req5 " in message for batch in messages for message in batch))

    def test_reordered_mixed_batch_uses_flat_draft_offsets(self):
        self.runner.requests = {"req0": object(), "empty": object(), "req2": object()}
        self.runner.input_batch.req_ids = ["req2", "empty", "req0"]
        self.scheduler_output.num_scheduled_tokens = {"req0": 2, "empty": 1, "req2": 3}
        self.metadata.num_draft_tokens = [2, 0, 1]
        self.metadata.draft_token_ids = torch.tensor([21, 22, 42])
        self.metadata.logits_indices = torch.tensor([0, 1, 2, 3, 5, 4])
        self.runner.input_ids.gpu = torch.tensor([10, 21, 22, 9, 11, 42])
        self.runner.positions.gpu = torch.tensor([10, 11, 12, 7, 20, 21])
        self.output.sampled_token_ids = torch.tensor([[21, 22, 23], [8, -1, -1], [99, -1, -1]])
        self.logits = torch.zeros(6, 100)
        first, second = self.trace()
        self.assertIn("req=req2", first)
        self.assertIn("draft_ids=[21, 22] raw_output=[21, 22, 23] accepted=2", first)
        self.assertIn("logits_indices=[0, 1, 2] num_scheduled_tokens=3 num_drafts=2", first)
        self.assertIn("req=req0", second)
        self.assertIn("draft_ids=[42] raw_output=[99, -1, -1] accepted=0", second)
        self.assertIn("input_ids=[42, 11] positions=[21, 20]", second)
        self.assertIn("logits_indices=[5, 4] num_scheduled_tokens=2 num_drafts=1", second)

    def test_zero_decode_window_does_not_sample_every_step(self):
        owner = SimpleNamespace(
            input_batch=SimpleNamespace(
                req_id_to_index={"req0": 0},
                num_computed_tokens_cpu=np.array([131621]),
            )
        )
        output = SimpleNamespace(num_scheduled_tokens={"req0": 2})
        with (
            patch.object(model_runner_module, "_mtp_dw_diag_enabled", return_value=True),
            patch.object(model_runner_module, "_mtp_dw_window_size", return_value=0),
        ):
            samples = [model_runner_module._mtp_dw_sample_requests(owner, output) for _ in range(5)]
        self.assertEqual(samples, [{"req0"}] * 3 + [set(), set()])


class TestStagedSFADummyRemapBoundaries(unittest.TestCase):
    def test_short_synthetic_sequences_use_no_remap_sentinel(self):
        for query_width, seq_lens, expected in (
            (
                1,
                [1, 2048, 2049, 4097],
                [0, 0, 2048, 4096],
            ),
            (
                2,
                [2, 4097, 4098, 8194],
                [0, 0, 4096, 8192],
            ),
        ):
            with self.subTest(query_width=query_width):
                boundaries = (
                    model_runner_module
                    ._staged_sfa_dummy_remap_boundaries(
                        np.asarray(seq_lens, dtype=np.int32),
                        query_width,
                        2048,
                    )
                )
                np.testing.assert_array_equal(
                    boundaries,
                    np.asarray(expected, dtype=np.int32),
                )


class TestResidentRequestState(unittest.TestCase):
    @staticmethod
    def _build_runner():
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner._resident_state_registry = (
            model_runner_module.ResidentRequestStateRegistry(3)
        )
        runner._resident_state_indices = SimpleNamespace(
            np=np.zeros(3, dtype=np.int32),
            gpu=torch.zeros(3, dtype=torch.int32),
            copy_to_gpu=MagicMock(),
        )
        runner._resident_state_generations = SimpleNamespace(
            np=np.zeros(3, dtype=np.int64),
            gpu=torch.zeros(3, dtype=torch.int64),
            copy_to_gpu=MagicMock(),
        )
        runner._resident_scratch_capacity = 4096

        block_ids = np.full((2, 32), -1, dtype=np.int32)
        block_ids[0, :26] = np.arange(100, 126, dtype=np.int32)
        block_ids[1] = np.arange(200, 232, dtype=np.int32)
        block_table = SimpleNamespace(
            block_size=128,
            num_blocks_per_row=np.array([26, 32], dtype=np.int32),
            block_table=SimpleNamespace(np=block_ids),
        )
        runner.input_batch = SimpleNamespace(
            req_ids=["short", "full"],
            block_table=[block_table],
        )
        return runner, block_table

    def test_zero_boundary_stays_cold_until_remap_activates(self):
        runner, block_table = self._build_runner()

        _, _, indices, generations = runner._prepare_resident_request_state(
            num_reqs=2,
            num_reqs_padded=3,
            is_dummy=False,
            remap_frontiers=(0, 4096),
        )

        self.assertEqual(indices.tolist()[0], -1)
        self.assertEqual(generations.tolist()[0], -1)
        self.assertGreaterEqual(indices.tolist()[1], 0)
        self.assertGreaterEqual(generations.tolist()[1], 0)
        self.assertEqual(indices.tolist()[2], -1)
        full_state = int(indices[1])
        full_generation = int(generations[1])

        with self.assertRaisesRegex(
            RuntimeError,
            "nonzero remap frontier without a complete scratch prefix",
        ):
            runner._prepare_resident_request_state(
                num_reqs=2,
                num_reqs_padded=3,
                is_dummy=False,
                remap_frontiers=(4096, 4096),
            )

        block_table.block_table.np[0, 26:] = np.arange(
            126, 132, dtype=np.int32
        )
        block_table.num_blocks_per_row[0] = 32
        _, _, indices, generations = runner._prepare_resident_request_state(
            num_reqs=2,
            num_reqs_padded=3,
            is_dummy=False,
            remap_frontiers=(0, 4096),
        )
        self.assertEqual(int(indices[0]), -1)
        self.assertEqual(int(generations[0]), -1)

        _, _, indices, generations = runner._prepare_resident_request_state(
            num_reqs=2,
            num_reqs_padded=3,
            is_dummy=False,
            remap_frontiers=(4096, 4096),
        )

        self.assertGreaterEqual(indices.tolist()[0], 0)
        self.assertGreaterEqual(generations.tolist()[0], 0)
        self.assertEqual(int(indices[1]), full_state)
        self.assertEqual(int(generations[1]), full_generation)

        short_state = int(indices[0])
        short_generation = int(generations[0])
        block_table.num_blocks_per_row[0] = 26
        _, _, indices, generations = runner._prepare_resident_request_state(
            num_reqs=2,
            num_reqs_padded=3,
            is_dummy=False,
            remap_frontiers=(0, 4096),
        )
        self.assertEqual(int(indices[0]), -1)
        self.assertEqual(int(generations[0]), -1)

        block_table.num_blocks_per_row[0] = 32
        _, _, indices, generations = runner._prepare_resident_request_state(
            num_reqs=2,
            num_reqs_padded=3,
            is_dummy=False,
            remap_frontiers=(4096, 4096),
        )
        self.assertEqual(int(indices[0]), short_state)
        self.assertGreater(int(generations[0]), short_generation)

    def test_generic_fallback_invalidates_resident_generation(self):
        runner, _ = self._build_runner()
        _, _, indices, generations = runner._prepare_resident_request_state(
            num_reqs=2,
            num_reqs_padded=2,
            is_dummy=False,
            remap_frontiers=(4096, 4096),
        )
        state = indices.copy()
        resident_generation = generations.copy()

        _, _, indices, generations = runner._prepare_resident_request_state(
            num_reqs=2,
            num_reqs_padded=2,
            is_dummy=False,
            resident_compatible=False,
            remap_frontiers=(4096, 4096),
        )
        np.testing.assert_array_equal(indices, [-1, -1])
        np.testing.assert_array_equal(generations, [-1, -1])

        _, _, indices, generations = runner._prepare_resident_request_state(
            num_reqs=2,
            num_reqs_padded=2,
            is_dummy=False,
            remap_frontiers=(4096, 4096),
        )
        np.testing.assert_array_equal(indices, state)
        self.assertTrue(np.all(generations > resident_generation))


class TestNPUModelRunnerKVCache(unittest.TestCase):
    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.use_sparse = False
        runner.dsa_shared_pool = False
        runner.dsa_unbundle = False
        runner.use_sparse_c8_indexer = False
        runner.use_hybrid_blocks = False
        runner.hybrid_with_attn_and_mamba = False
        runner.runner_only_attn_layers = set()
        runner.is_kv_consumer = False
        runner.vllm_config = MagicMock()
        runner.vllm_config.kv_transfer_config = None
        runner.model_config = MagicMock()
        runner.model_config.use_mla = True
        backend = MagicMock()
        backend.get_kv_cache_shape.side_effect = lambda num_blocks, block_size, num_kv_heads, head_size: (
            2,
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
        )
        runner.attn_backend = backend
        return runner

    def test_allocate_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )

        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        k_cache_raw, v_cache_raw = kv_cache_raw_tensors["draft_attn"]

        self.assertEqual(k_cache_raw.numel(), kv_cache_spec.page_size_bytes)
        self.assertEqual(v_cache_raw.numel(), kv_cache_spec.page_size_bytes)

    def test_reshape_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )
        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        runner._kv_cache_spec_attn_group_iterator = lambda: [
            SimpleNamespace(
                kv_cache_spec=kv_cache_spec,
                backend=runner.attn_backend,
                layer_names=["draft_attn"],
            )
        ]

        kv_caches = runner._reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors)
        k_cache, v_cache = kv_caches["draft_attn"]

        self.assertEqual(k_cache.shape, (2, 16, 8, 64))
        self.assertEqual(v_cache.shape, (2, 16, 8, 64))


class TestStagedSFAGraphKey(unittest.TestCase):
    def test_structural_dimensions_do_not_collide(self):
        base = STAGED_SFA_SINGLETON_GRAPH_KEY
        variants = (
            StagedSFAGraphKey.exact_q1(2),
            StagedSFAGraphKey.fixed_spec(1, 2),
            StagedSFAGraphKey.fixed_spec(1, 3),
        )
        self.assertEqual(
            len({base, StagedSFAGraphKey(**base.__dict__)}),
            1,
        )
        self.assertTrue(all(variant != base for variant in variants))
        self.assertEqual(len(set(variants)), len(variants))

    def test_invalid_structural_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            StagedSFAGraphKey(
                token_capacity=2,
                request_capacity=1,
                query_profile=StagedSFAQueryProfile.DECODE_Q1,
                max_query_len=1,
            )
        with self.assertRaises(ValueError):
            StagedSFAGraphKey(
                token_capacity=2,
                request_capacity=2,
                query_profile=StagedSFAQueryProfile.SPEC_FIXED,
                max_query_len=2,
            )

    def test_key_is_frozen(self):
        with self.assertRaises(FrozenInstanceError):
            STAGED_SFA_SINGLETON_GRAPH_KEY.token_capacity = 2

    def test_structural_keys_adapt_to_legacy_descriptor(self):
        descriptor = STAGED_SFA_SINGLETON_GRAPH_KEY.to_legacy_batch_descriptor()
        self.assertEqual(descriptor, BatchDescriptor(num_tokens=1))
        self.assertIsNone(descriptor.num_reqs)
        self.assertFalse(descriptor.uniform)
        self.assertFalse(descriptor.has_lora)

        batch_descriptor = StagedSFAGraphKey(
            token_capacity=2,
            request_capacity=2,
            query_profile=StagedSFAQueryProfile.DECODE_Q1,
            max_query_len=1,
        ).to_legacy_batch_descriptor()
        self.assertEqual(batch_descriptor, BatchDescriptor(num_tokens=2))

        spec_key = StagedSFAGraphKey.fixed_spec(2, 3)
        self.assertEqual(
            spec_key.to_legacy_batch_descriptor(),
            BatchDescriptor(num_tokens=6),
        )

    def test_fixed_spec_keeps_request_and_token_capacity_distinct(self):
        key = StagedSFAGraphKey.fixed_spec(4, 3)
        self.assertEqual(key.request_capacity, 4)
        self.assertEqual(key.token_capacity, 12)
        self.assertEqual(key.max_query_len, 3)
        self.assertEqual(key.query_profile, StagedSFAQueryProfile.SPEC_FIXED)

    def test_mtp_fia_padding_uses_request_not_token_capacity(self):
        key = StagedSFAGraphKey.fixed_spec(1, 2)
        batch_desc = BatchDescriptor(num_tokens=2)

        self.assertEqual(
            NPUModelRunner._fia_request_capacity(key, batch_desc),
            1,
        )

    def test_fixed_spec_rejects_q1_width(self):
        with self.assertRaises(ValueError):
            StagedSFAGraphKey.fixed_spec(4, 1)


class TestQueryStartPadding(unittest.TestCase):
    @staticmethod
    def _build_runner():
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.uniform_decode_query_len = 1
        runner.compilation_config = SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
        )
        runner.arange_np = np.arange(8, dtype=np.int32)
        runner.query_start_loc = SimpleNamespace(
            np=np.array([0, 1, 2, -1, -1, -1, -1, -1], dtype=np.int32),
            copy_to_gpu=MagicMock(),
        )
        return runner

    def test_exact_q1_capacity_reuses_uploaded_query_starts(self):
        runner = self._build_runner()

        padded_reqs = runner._pad_query_start_loc_for_fia(
            num_tokens_padded=2,
            num_reqs_padded=2,
            num_reqs=2,
            batch_desc_num_reqs=2,
        )

        self.assertEqual(padded_reqs, 2)
        runner.query_start_loc.copy_to_gpu.assert_not_called()

    def test_padded_q1_capacity_uploads_padding(self):
        runner = self._build_runner()

        padded_reqs = runner._pad_query_start_loc_for_fia(
            num_tokens_padded=4,
            num_reqs_padded=4,
            num_reqs=2,
            batch_desc_num_reqs=4,
        )

        self.assertEqual(padded_reqs, 4)
        np.testing.assert_array_equal(
            runner.query_start_loc.np[:5],
            np.arange(5, dtype=np.int32),
        )
        runner.query_start_loc.copy_to_gpu.assert_called_once_with()


class TestStagedSFADummyBatch(unittest.TestCase):
    @staticmethod
    def _build_runner():
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.vllm_config = object()
        runner.speculative_config = None
        runner.decode_threshold = 1
        runner.parallel_config = SimpleNamespace(data_parallel_size=1)
        runner.attn_state = AscendAttentionState.DecodeOnly
        runner._staged_sfa_graph_capture_sizes = (1, 4)
        runner._dp_batch_sync_buffers = {}
        return runner

    @staticmethod
    def _eligibility_kwargs(batch_size=4):
        return {
            "is_profile": False,
            "cudagraph_runtime_mode": CUDAGraphMode.PIECEWISE,
            "allow_eager": True,
            "num_active_loras": 0,
            "num_tokens_unpadded": batch_size,
            "num_tokens_padded": batch_size,
            "num_reqs": batch_size,
            "num_scheduled_tokens": np.ones(batch_size, dtype=np.int32),
            "batch_descriptor": BatchDescriptor(num_tokens=batch_size),
            "dp_route_action": StagedSFARouteAction.STAGED,
        }

    def test_exact_q1_capture_sizes_are_staged(self):
        runner = self._build_runner()
        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                model_runner_module,
                "staged_sfa_graph_capture_sizes",
                return_value=(1, 4),
            ),
        ):
            for runtime_mode in (
                CUDAGraphMode.NONE,
                CUDAGraphMode.PIECEWISE,
            ):
                kwargs = self._eligibility_kwargs()
                kwargs["cudagraph_runtime_mode"] = runtime_mode
                with self.subTest(runtime_mode=runtime_mode):
                    self.assertEqual(
                        runner._staged_sfa_dummy_batch_size(**kwargs),
                        4,
                    )

    def test_dp_dummy_uses_agreed_graph_capacity(self):
        runner = self._build_runner()
        kwargs = self._eligibility_kwargs(batch_size=1)
        kwargs.update(
            num_tokens_padded=4,
            batch_descriptor=BatchDescriptor(num_tokens=4),
            allow_eager=False,
        )
        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                model_runner_module,
                "staged_sfa_graph_capture_sizes",
                return_value=(1, 4),
            ),
        ):
            self.assertEqual(
                runner._staged_sfa_dummy_batch_size(**kwargs),
                4,
            )
            kwargs["cudagraph_runtime_mode"] = CUDAGraphMode.NONE
            self.assertIsNone(
                runner._staged_sfa_dummy_batch_size(**kwargs)
            )

    def test_native_execution_waits_for_capture_unsafe_connector_loads(self):
        runner = self._build_runner()
        barrier = MagicMock()
        connector = SimpleNamespace(
            synchronize_staged_sfa_capture_unsafe_loads=barrier,
        )
        with (
            patch.object(
                model_runner_module,
                "has_kv_transfer_group",
                return_value=True,
            ),
            patch.object(
                model_runner_module,
                "get_kv_transfer_group",
                return_value=connector,
            ),
            patch.object(
                model_runner_module,
                "get_tp_group",
                return_value=SimpleNamespace(world_size=1),
            ),
        ):
            runner._synchronize_staged_sfa_capture_unsafe_loads()

        barrier.assert_called_once_with()

    def test_native_load_barrier_does_not_collect_across_internal_dp(self):
        runner = self._build_runner()
        runner.parallel_config.data_parallel_size = 2
        barrier = MagicMock()
        connector = SimpleNamespace(
            synchronize_staged_sfa_capture_unsafe_loads=barrier,
        )
        with (
            patch.object(
                model_runner_module,
                "has_kv_transfer_group",
                return_value=True,
            ),
            patch.object(
                model_runner_module,
                "get_kv_transfer_group",
                return_value=connector,
            ),
            patch.object(
                model_runner_module,
                "get_tp_group",
                return_value=SimpleNamespace(world_size=1),
            ),
            patch.object(model_runner_module, "get_dp_group") as get_dp_group,
            patch.object(model_runner_module.dist, "all_reduce") as all_reduce,
        ):
            runner._synchronize_staged_sfa_capture_unsafe_loads()

        barrier.assert_called_once_with()
        get_dp_group.assert_not_called()
        all_reduce.assert_not_called()

    def test_native_execution_fails_closed_without_load_barrier(self):
        runner = self._build_runner()
        connector = SimpleNamespace(
            supports_dsa_compact_external_load=True,
        )
        with (
            patch.object(
                model_runner_module,
                "has_kv_transfer_group",
                return_value=True,
            ),
            patch.object(
                model_runner_module,
                "get_kv_transfer_group",
                return_value=connector,
            ),
            patch.object(
                model_runner_module,
                "get_tp_group",
                return_value=SimpleNamespace(world_size=1),
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "capture-unsafe load barrier failed",
            ),
        ):
            runner._synchronize_staged_sfa_capture_unsafe_loads()

    def test_native_execution_propagates_peer_load_barrier_failure(self):
        runner = self._build_runner()
        barrier = MagicMock()
        cpu_group = object()

        def report_peer_failure(failure, **_kwargs):
            failure.fill_(1)

        with (
            patch.object(
                model_runner_module,
                "has_kv_transfer_group",
                return_value=True,
            ),
            patch.object(
                model_runner_module,
                "get_kv_transfer_group",
                return_value=SimpleNamespace(
                    synchronize_staged_sfa_capture_unsafe_loads=barrier,
                ),
            ),
            patch.object(
                model_runner_module,
                "get_tp_group",
                return_value=SimpleNamespace(
                    world_size=2,
                    cpu_group=cpu_group,
                ),
            ),
            patch.object(
                model_runner_module.dist,
                "all_reduce",
                side_effect=report_peer_failure,
            ) as all_reduce,
            self.assertRaisesRegex(RuntimeError, "peer worker"),
        ):
            runner._synchronize_staged_sfa_capture_unsafe_loads()

        barrier.assert_called_once_with()
        self.assertIs(all_reduce.call_args.kwargs["group"], cpu_group)

    def test_dp_sync_agrees_route_in_existing_collective(self):
        runner = self._build_runner()
        runner.dp_size = 2
        runner.dp_rank = 0
        runner._skip_all_reduce_across_dp_group = MagicMock(return_value=False)

        def peer(action):
            def all_reduce(tensor, group):
                tensor[0, 1] = 4
                tensor[1, 1] = CUDAGraphMode.PIECEWISE.value
                tensor[2, 1] = tuple(StagedSFARouteAction).index(action)

            return all_reduce

        buffer_addresses = set()
        for peer_action in StagedSFARouteAction:
            with (
                self.subTest(peer_action=peer_action),
                patch.object(
                    model_runner_module,
                    "get_dp_group",
                    return_value=SimpleNamespace(cpu_group=object()),
                ),
                patch.object(
                    model_runner_module.dist,
                    "all_reduce",
                    side_effect=peer(peer_action),
                ),
            ):
                _, sizes, _, action = runner._sync_batch_across_dp(
                    num_tokens_padded=1,
                    cudagraph_mode=CUDAGraphMode.PIECEWISE.value,
                    allow_dp_padding=True,
                    staged_sfa_route_action=StagedSFARouteAction.STAGED,
                )
                self.assertEqual(sizes.tolist(), [4, 4])
                self.assertEqual(action, peer_action)
                buffer_addresses.add(sizes.data_ptr())

        self.assertEqual(len(buffer_addresses), 1)
        self.assertEqual(
            runner._skip_all_reduce_across_dp_group.call_count,
            len(StagedSFARouteAction),
        )

    def test_dp_sync_keeps_non_staged_collective_shape(self):
        runner = self._build_runner()
        runner.dp_size = 2
        runner.dp_rank = 0
        runner._skip_all_reduce_across_dp_group = MagicMock(return_value=False)

        def all_reduce(tensor, group):
            self.assertEqual(tuple(tensor.shape), (2, 2))
            tensor[0, 1] = 4
            tensor[1, 1] = CUDAGraphMode.PIECEWISE.value

        with (
            patch.object(
                model_runner_module,
                "get_dp_group",
                return_value=SimpleNamespace(cpu_group=object()),
            ),
            patch.object(
                model_runner_module.dist,
                "all_reduce",
                side_effect=all_reduce,
            ),
        ):
            _, sizes, mode, action = runner._sync_batch_across_dp(
                num_tokens_padded=1,
                cudagraph_mode=CUDAGraphMode.PIECEWISE.value,
                allow_dp_padding=True,
            )

        self.assertEqual(sizes.tolist(), [4, 4])
        self.assertEqual(mode, CUDAGraphMode.PIECEWISE.value)
        self.assertIsNone(action)

    def test_dp_redispatches_only_when_agreement_changes_the_key(self):
        runner = self._build_runner()
        runner.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=2,
                tensor_parallel_size=2,
            ),
            observability_config=SimpleNamespace(cudagraph_metrics=False),
        )
        runner.parallel_config = SimpleNamespace(data_parallel_rank=0)
        runner.input_batch = SimpleNamespace(
            num_computed_tokens_cpu=np.ones(1, dtype=np.int32),
            lora_id_to_lora_request={},
        )
        runner.model_config = SimpleNamespace(is_encoder_decoder=False)
        runner.uniform_decode_query_len = 1
        runner._pad_for_sequence_parallelism = MagicMock(
            side_effect=lambda value: value
        )
        runner.cudagraph_dispatcher = SimpleNamespace(
            dispatch=MagicMock(
                side_effect=lambda num_tokens, **_: (
                    CUDAGraphMode.PIECEWISE,
                    BatchDescriptor(num_tokens=num_tokens),
                )
            )
        )

        for agreed_size, expected_calls in ((1, 1), (4, 2)):
            runner.cudagraph_dispatcher.dispatch.reset_mock()
            runner._sync_batch_across_dp = MagicMock(
                return_value=(
                    False,
                    torch.tensor([agreed_size, agreed_size]),
                    CUDAGraphMode.PIECEWISE.value,
                    StagedSFARouteAction.STAGED,
                )
            )
            with (
                self.subTest(agreed_size=agreed_size),
                patch.object(model_runner_module, "enable_sp", return_value=False),
            ):
                _, descriptor, _, _, _ = (
                    runner._determine_batch_execution_and_padding(
                        num_tokens=1,
                        num_reqs=1,
                        num_scheduled_tokens_np=np.ones(1, dtype=np.int32),
                        max_num_scheduled_tokens=1,
                        use_cascade_attn=False,
                        staged_sfa_route_action=StagedSFARouteAction.STAGED,
                    )
                )

            self.assertEqual(descriptor.num_tokens, agreed_size)
            self.assertEqual(
                runner.cudagraph_dispatcher.dispatch.call_count,
                expected_calls,
            )

    def test_padded_non_q1_and_unsupported_batches_fall_back(self):
        runner = self._build_runner()
        cases = {
            "unsupported_size": {
                "num_tokens_unpadded": 2,
                "num_tokens_padded": 2,
                "num_reqs": 2,
                "num_scheduled_tokens": np.ones(2, dtype=np.int32),
                "batch_descriptor": BatchDescriptor(num_tokens=2),
            },
            "token_padding": {
                "num_tokens_padded": 8,
                "batch_descriptor": BatchDescriptor(num_tokens=8),
            },
            "non_q1": {
                "num_scheduled_tokens": np.array([1, 1, 2, 0]),
            },
            "uniform_descriptor": {
                "batch_descriptor": BatchDescriptor(
                    num_tokens=4,
                    uniform=True,
                ),
            },
            "dp_fallback": {
                "dp_route_action": StagedSFARouteAction.SAFE_NATIVE,
            },
            "profile": {"is_profile": True},
        }
        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                model_runner_module,
                "staged_sfa_graph_capture_sizes",
                return_value=(1, 4),
            ),
        ):
            for case_name, overrides in cases.items():
                kwargs = self._eligibility_kwargs()
                kwargs.update(overrides)
                with self.subTest(case=case_name):
                    self.assertIsNone(runner._staged_sfa_dummy_batch_size(**kwargs))

    def test_live_route_accepts_dp_padding_before_mutation(self):
        runner = self._build_runner()
        request_ids = [f"req-{index}" for index in range(4)]
        local_kwargs = {
            "num_tokens_unpadded": 4,
            "num_reqs": 4,
            "num_scheduled_tokens": np.ones(4, dtype=np.int32),
            "index_topk": 2048,
            "has_cascade_attention": False,
            "request_ids": request_ids,
            "kv_connector_metadata": SimpleNamespace(
                requests=[
                    SimpleNamespace(
                        req_id=req_id,
                        is_sparse_decode=True,
                        dsa_current_released_frontier=4096,
                        dsa_nonresident_frontier=4096,
                        load_spec=SimpleNamespace(
                            can_load=True,
                            lmcache_cached_tokens=4096,
                        ),
                    )
                    for req_id in request_ids
                ]
            ),
        }
        final_kwargs = {
            "dp_route_action": StagedSFARouteAction.STAGED,
            "cudagraph_mode": CUDAGraphMode.PIECEWISE,
            "batch_descriptor": BatchDescriptor(num_tokens=4),
            "num_tokens_unpadded": 4,
            "num_tokens_padded": 4,
            "num_reqs": 4,
            "should_ubatch": False,
        }
        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                model_runner_module,
                "staged_sfa_graph_capture_sizes",
                return_value=(1, 4),
            ),
        ):
            local_route = runner._staged_sfa_local_route(**local_kwargs)
            route = runner._staged_sfa_live_route(
                local_route=local_route,
                **final_kwargs,
            )
            self.assertEqual(route.action, StagedSFARouteAction.STAGED)
            self.assertEqual(route.reason, StagedSFARouteReason.ELIGIBLE)
            self.assertEqual(route.graph_key, StagedSFAGraphKey.exact_q1(4))
            self.assertEqual(route.frontiers, (4096,) * 4)

            one_request_kwargs = dict(local_kwargs)
            one_request_kwargs.update(
                num_tokens_unpadded=1,
                num_reqs=1,
                num_scheduled_tokens=np.ones(1, dtype=np.int32),
                request_ids=request_ids[:1],
                kv_connector_metadata=SimpleNamespace(requests=local_kwargs["kv_connector_metadata"].requests[:1]),
            )
            padded_route = runner._staged_sfa_live_route(
                local_route=runner._staged_sfa_local_route(**one_request_kwargs),
                **{
                    **final_kwargs,
                    "num_tokens_unpadded": 1,
                    "num_reqs": 1,
                },
            )
            self.assertEqual(
                padded_route.graph_key,
                StagedSFAGraphKey.exact_q1(4),
            )

            for name, overrides in {
                "multi_token": {
                    "num_scheduled_tokens": np.array(
                        [1, 1, 2, 0],
                        dtype=np.int32,
                    ),
                },
                "cascade": {"has_cascade_attention": True},
                "dense_prefix_hit": {
                    "kv_connector_metadata": SimpleNamespace(
                        requests=[
                            SimpleNamespace(
                                req_id=req_id,
                                is_sparse_decode=False,
                                dsa_current_released_frontier=0,
                                dsa_nonresident_frontier=0,
                                load_spec=SimpleNamespace(can_load=True),
                            )
                            for req_id in request_ids
                        ]
                    ),
                },
                "mixed_connector_load": {
                    "kv_connector_metadata": SimpleNamespace(
                        requests=[
                            SimpleNamespace(
                                req_id=req_id,
                                is_sparse_decode=index >= 2,
                                dsa_current_released_frontier=(
                                    4096 if index >= 2 else 0
                                ),
                                dsa_nonresident_frontier=(
                                    4096 if index >= 2 else 0
                                ),
                                load_spec=SimpleNamespace(
                                    can_load=True,
                                    lmcache_cached_tokens=4096,
                                ),
                            )
                            for index, req_id in enumerate(request_ids)
                        ]
                    ),
                },
                "short_frontier": {
                    "kv_connector_metadata": SimpleNamespace(
                        requests=[
                            SimpleNamespace(
                                req_id=req_id,
                                is_sparse_decode=True,
                                dsa_current_released_frontier=0,
                                dsa_nonresident_frontier=0,
                                load_spec=SimpleNamespace(
                                    can_load=True,
                                    lmcache_cached_tokens=1024,
                                ),
                            )
                            for req_id in request_ids
                        ]
                    ),
                },
            }.items():
                rejected = dict(local_kwargs)
                rejected.update(overrides)
                with self.subTest(name=name):
                    route = runner._staged_sfa_local_route(**rejected)
                    self.assertEqual(
                        route.action,
                        (
                            StagedSFARouteAction.FATAL
                            if name == "short_frontier"
                            else (
                                StagedSFARouteAction.STAGED
                                if name in (
                                    "dense_prefix_hit",
                                    "mixed_connector_load",
                                )
                                else StagedSFARouteAction.SAFE_NATIVE
                            )
                        ),
                    )
                    if name == "dense_prefix_hit":
                        self.assertEqual(
                            route.reason,
                            StagedSFARouteReason.DENSE_PREFIX_HIT,
                        )
                    elif name == "mixed_connector_load":
                        self.assertEqual(
                            route.reason,
                            StagedSFARouteReason.MIXED_CONNECTOR_LOAD,
                        )
                        self.assertEqual(route.frontiers, (0, 0, 4096, 4096))
                    elif name == "short_frontier":
                        self.assertEqual(
                            route.reason,
                            StagedSFARouteReason.FRONTIER_TOO_SHORT,
                        )

            ubatch_route = runner._staged_sfa_live_route(
                local_route=local_route,
                **{**final_kwargs, "should_ubatch": True},
            )
            self.assertEqual(
                ubatch_route.reason,
                StagedSFARouteReason.UBATCH,
            )

            peer_fallback = runner._staged_sfa_live_route(
                local_route=local_route,
                **{
                    **final_kwargs,
                    "dp_route_action": StagedSFARouteAction.SAFE_NATIVE,
                },
            )
            self.assertEqual(
                peer_fallback.reason,
                StagedSFARouteReason.RUNTIME_PARALLELISM,
            )

            runner.attn_state = AscendAttentionState.PrefillNoCache
            route = runner._staged_sfa_local_route(**local_kwargs)
            self.assertEqual(route.action, StagedSFARouteAction.SAFE_NATIVE)
            self.assertEqual(route.reason, StagedSFARouteReason.NOT_DECODE)

    def test_non_native_routes_report_stable_action_and_reason(self):
        runner = self._build_runner()
        for action in (
            StagedSFARouteAction.RECOMPUTE,
            StagedSFARouteAction.FATAL,
        ):
            route = model_runner_module.StagedSFARouteDecision(
                action,
                StagedSFARouteReason.SPARSE_LOAD_UNAVAILABLE,
            )

            with (
                self.subTest(action=action),
                self.assertRaisesRegex(
                    RuntimeError,
                    rf"\[SFA_ROUTE\] action={action.value} "
                    r"reason=sparse_load_unavailable",
                ),
            ):
                runner._apply_staged_sfa_route(route)

    def test_q1_cold_compact_resume_uses_staged_graph(self):
        runner = self._build_runner()
        metadata = SimpleNamespace(
            requests=[
                SimpleNamespace(
                    req_id="cold",
                    is_sparse_decode=True,
                    dsa_current_released_frontier=0,
                    dsa_nonresident_frontier=8192,
                    load_spec=SimpleNamespace(
                        can_load=True,
                        lmcache_cached_tokens=8193,
                        dsa_committed_end=8192,
                        dsa_cold_compact_resume=True,
                    ),
                )
            ]
        )
        kwargs = dict(
            num_tokens_unpadded=1,
            num_reqs=1,
            num_scheduled_tokens=np.ones(1, dtype=np.int32),
            index_topk=2048,
            has_cascade_attention=False,
            request_ids=["cold"],
            kv_connector_metadata=metadata,
            num_computed_tokens=np.array([8192], dtype=np.int32),
            prompt_lens=np.array([8193], dtype=np.int32),
        )

        route = runner._staged_sfa_local_route(**kwargs)
        self.assertEqual(route.action, StagedSFARouteAction.STAGED)
        self.assertEqual(route.frontiers, (8192,))
        self.assertEqual(route.cold_compact_resumes, (True,))
        live = runner._staged_sfa_live_route(
            local_route=route,
            dp_route_action=StagedSFARouteAction.STAGED,
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
            batch_descriptor=BatchDescriptor(num_tokens=1),
            num_tokens_unpadded=1,
            num_tokens_padded=1,
            num_reqs=1,
            should_ubatch=False,
        )
        self.assertEqual(live.graph_key, StagedSFAGraphKey.exact_q1(1))
        self.assertEqual(live.cold_compact_resumes, (True,))

        for name, override in {
            "computed": {"num_computed_tokens": np.array([8191])},
            "prompt": {"prompt_lens": np.array([8194])},
            "frontier": {
                "kv_connector_metadata": SimpleNamespace(
                    requests=[
                        SimpleNamespace(
                            req_id="cold",
                            is_sparse_decode=True,
                            dsa_current_released_frontier=0,
                            dsa_nonresident_frontier=7936,
                            load_spec=SimpleNamespace(
                                can_load=True,
                                dsa_committed_end=7936,
                                dsa_cold_compact_resume=True,
                            ),
                        )
                    ]
                )
            },
        }.items():
            with self.subTest(name=name):
                rejected = runner._staged_sfa_local_route(
                    **{**kwargs, **override}
                )
                self.assertEqual(
                    rejected.action, StagedSFARouteAction.SAFE_NATIVE
                )
                self.assertEqual(
                    rejected.reason,
                    StagedSFARouteReason.COLD_COMPACT_LAYOUT,
                )

    def test_native_route_logs_once_per_reason(self):
        runner = self._build_runner()
        route = model_runner_module.StagedSFARouteDecision(
            StagedSFARouteAction.SAFE_NATIVE,
            StagedSFARouteReason.DENSE_PREFIX_HIT,
        )

        with patch.object(model_runner_module.logger, "info") as log:
            self.assertIsNone(runner._apply_staged_sfa_route(route))
            self.assertIsNone(runner._apply_staged_sfa_route(route))

        log.assert_called_once_with("[SFA_ROUTE] action=safe_native reason=dense_prefix_hit")

    def test_native_q1_rows_have_unique_ids_and_query_starts(self):
        request_ids = NPUModelRunner._staged_sfa_dummy_request_ids(4)
        query_start_locs = NPUModelRunner._staged_sfa_q1_query_start_locs(
            4,
            dtype=np.dtype(np.int32),
        )

        self.assertEqual(
            request_ids,
            [
                "staged-sfa-graph-dummy-0",
                "staged-sfa-graph-dummy-1",
                "staged-sfa-graph-dummy-2",
                "staged-sfa-graph-dummy-3",
            ],
        )
        self.assertEqual(len(set(request_ids)), 4)
        np.testing.assert_array_equal(
            query_start_locs,
            np.arange(5, dtype=np.int32),
        )

    def test_fixed_mtp_query_starts_preserve_request_rows(self):
        np.testing.assert_array_equal(
            NPUModelRunner._staged_sfa_query_start_locs(
                3,
                query_width=2,
                dtype=np.dtype(np.int32),
            ),
            np.array([0, 2, 4, 6], dtype=np.int32),
        )

    def test_fixed_width_mtp_dummy_batches_are_staged(self):
        runner = self._build_runner()
        runner.speculative_config = SimpleNamespace(
            method="mtp",
            num_speculative_tokens=1,
        )
        runner.decode_threshold = 2
        runner.attn_state = AscendAttentionState.SpecDecoding
        kwargs = self._eligibility_kwargs()
        kwargs.update(
            num_reqs=2,
            num_scheduled_tokens=np.full(2, 2, dtype=np.int32),
        )
        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                model_runner_module,
                "staged_sfa_graph_capture_sizes",
                return_value=(1, 4),
            ),
        ):
            self.assertEqual(runner._staged_sfa_dummy_batch_size(**kwargs), 4)

    def test_fixed_width_mtp_live_route_uses_request_capacity(self):
        runner = self._build_runner()
        runner.speculative_config = SimpleNamespace(
            method="mtp",
            num_speculative_tokens=1,
        )
        runner.decode_threshold = 2
        runner.attn_state = AscendAttentionState.SpecDecoding
        request_ids = ["req-0", "req-1"]
        local = runner._staged_sfa_local_route(
            num_tokens_unpadded=4,
            num_reqs=2,
            num_scheduled_tokens=np.full(2, 2, dtype=np.int32),
            index_topk=2048,
            has_cascade_attention=False,
            request_ids=request_ids,
            kv_connector_metadata=SimpleNamespace(
                requests=[
                    SimpleNamespace(
                        req_id=req_id,
                        is_sparse_decode=True,
                        dsa_current_released_frontier=4096,
                        dsa_nonresident_frontier=4096,
                        load_spec=SimpleNamespace(
                            can_load=True,
                            lmcache_cached_tokens=4096,
                            dsa_cold_compact_resume=True,
                        ),
                    )
                    for req_id in request_ids
                ]
            ),
        )
        route = runner._staged_sfa_live_route(
            local_route=local,
            dp_route_action=StagedSFARouteAction.STAGED,
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
            batch_descriptor=BatchDescriptor(num_tokens=4),
            num_tokens_unpadded=4,
            num_tokens_padded=4,
            num_reqs=2,
            should_ubatch=False,
        )
        self.assertEqual(local.cold_compact_resumes, ())
        self.assertEqual(route.action, StagedSFARouteAction.STAGED)
        self.assertEqual(route.graph_key, StagedSFAGraphKey.fixed_spec(2, 2))

    def test_mtp_request_three_and_five_use_different_graph_capacities(self):
        query_width = 2
        runner = self._build_runner()
        runner.speculative_config = SimpleNamespace(
            method="mtp",
            num_speculative_tokens=1,
        )
        runner.decode_threshold = query_width
        runner.attn_state = AscendAttentionState.SpecDecoding
        runner._staged_sfa_graph_capture_sizes = (
            4 * query_width,
            8 * query_width,
        )
        for actual_requests, graph_request_capacity in ((3, 4), (5, 8)):
            with self.subTest(actual_requests=actual_requests):
                request_ids = [
                    f"req-{index}" for index in range(actual_requests)
                ]
                local = runner._staged_sfa_local_route(
                    num_tokens_unpadded=actual_requests * query_width,
                    num_reqs=actual_requests,
                    num_scheduled_tokens=np.full(
                        actual_requests,
                        query_width,
                        dtype=np.int32,
                    ),
                    index_topk=2048,
                    has_cascade_attention=False,
                    request_ids=request_ids,
                    kv_connector_metadata=SimpleNamespace(
                        requests=[
                            SimpleNamespace(
                                req_id=req_id,
                                is_sparse_decode=True,
                                dsa_current_released_frontier=131_584,
                                dsa_nonresident_frontier=131_584,
                                load_spec=SimpleNamespace(
                                    can_load=True,
                                    lmcache_cached_tokens=131_584,
                                ),
                            )
                            for req_id in request_ids
                        ]
                    ),
                )
                graph_token_capacity = (
                    graph_request_capacity * query_width
                )
                route = runner._staged_sfa_live_route(
                    local_route=local,
                    dp_route_action=StagedSFARouteAction.STAGED,
                    cudagraph_mode=CUDAGraphMode.PIECEWISE,
                    batch_descriptor=BatchDescriptor(
                        num_tokens=graph_token_capacity
                    ),
                    num_tokens_unpadded=actual_requests * query_width,
                    num_tokens_padded=graph_token_capacity,
                    num_reqs=actual_requests,
                    should_ubatch=False,
                )

                self.assertEqual(
                    route.action,
                    StagedSFARouteAction.STAGED,
                )
                self.assertEqual(
                    route.graph_key,
                    StagedSFAGraphKey.fixed_spec(
                        graph_request_capacity,
                        query_width,
                    ),
                )
                self.assertGreater(
                    route.graph_key.request_capacity,
                    actual_requests,
                )

    def test_zero_frontier_uses_graph_with_all_kv_resident(self):
        runner = self._build_runner()
        request_ids = ["req-0"]
        local = runner._staged_sfa_local_route(
            num_tokens_unpadded=1,
            num_reqs=1,
            num_scheduled_tokens=np.ones(1, dtype=np.int32),
            index_topk=2048,
            has_cascade_attention=False,
            request_ids=request_ids,
            kv_connector_metadata=SimpleNamespace(
                requests=[
                    SimpleNamespace(
                        req_id="req-0",
                        is_sparse_decode=True,
                        dsa_current_released_frontier=0,
                        dsa_nonresident_frontier=0,
                        load_spec=SimpleNamespace(
                            can_load=True,
                            lmcache_cached_tokens=257,
                            dsa_committed_end=257,
                            dsa_scratch_capacity=4096,
                        ),
                    )
                ]
            ),
        )
        self.assertEqual(local.action, StagedSFARouteAction.STAGED)
        self.assertEqual(local.cold_compact_resumes, ())
        self.assertEqual(local.frontiers, (0,))

    def test_two_group_dummy_rows_use_noncolliding_physical_slots(self):
        runner = self._build_runner()
        runner.kv_cache_config = SimpleNamespace(
            num_blocks=8,
            num_blocks_per_group=[8, 8],
        )
        runner.input_batch = SimpleNamespace(
            block_table=MultiGroupBlockTable(
                max_num_reqs=4,
                max_model_len=16,
                max_num_batched_tokens=4,
                pin_memory=False,
                device=torch.device("cpu"),
                block_sizes=[4, 8],
                kernel_sizes=[[4], [8]],
                max_num_blocks=[4, 2],
            )
        )

        positions = np.array([8, 9, 10, 11], dtype=np.int64)
        runner._prepare_staged_sfa_dummy_block_tables(
            batch_size=4,
            positions=positions,
        )

        expected_slots = (
            np.array([0, 5, 10, 15], dtype=np.int64),
            np.array([0, 9, 18, 27], dtype=np.int64),
        )
        for group_index, block_table in enumerate(runner.input_batch.block_table.block_tables):
            expected_rows = np.broadcast_to(
                np.arange(4, dtype=np.int32).reshape(-1, 1),
                (4, block_table.max_num_blocks_per_req),
            )
            np.testing.assert_array_equal(
                block_table.block_table.np[:4],
                expected_rows,
            )
            np.testing.assert_array_equal(
                block_table.slot_mapping.np[:4],
                expected_slots[group_index],
            )

    def test_two_group_mtp_dummy_rows_get_one_slot_per_token(self):
        runner = self._build_runner()
        runner.kv_cache_config = SimpleNamespace(
            num_blocks=8,
            num_blocks_per_group=[8, 8],
        )
        runner.input_batch = SimpleNamespace(
            block_table=MultiGroupBlockTable(
                max_num_reqs=4,
                max_model_len=16,
                max_num_batched_tokens=8,
                pin_memory=False,
                device=torch.device("cpu"),
                block_sizes=[4, 8],
                kernel_sizes=[[4], [8]],
                max_num_blocks=[4, 2],
            )
        )
        runner._prepare_staged_sfa_dummy_block_tables(
            batch_size=2,
            positions=np.array([8, 9, 10, 11], dtype=np.int64),
        )
        expected_slots = (
            np.array([0, 1, 6, 7], dtype=np.int64),
            np.array([0, 1, 10, 11], dtype=np.int64),
        )
        for group_index, block_table in enumerate(
            runner.input_batch.block_table.block_tables
        ):
            np.testing.assert_array_equal(
                block_table.slot_mapping.np[:4],
                expected_slots[group_index],
            )
            self.assertEqual(
                np.unique(block_table.slot_mapping.np[:4]).size,
                4,
            )

    def test_dummy_block_rows_require_enough_physical_blocks(self):
        runner = self._build_runner()
        runner.kv_cache_config = SimpleNamespace(num_blocks=3)
        runner.input_batch = SimpleNamespace(
            block_table=MultiGroupBlockTable(
                max_num_reqs=4,
                max_model_len=16,
                max_num_batched_tokens=4,
                pin_memory=False,
                device=torch.device("cpu"),
                block_sizes=[4, 8],
                kernel_sizes=[[4], [8]],
                max_num_blocks=[4, 2],
            )
        )

        with self.assertRaisesRegex(RuntimeError, "one physical block"):
            runner._prepare_staged_sfa_dummy_block_tables(
                batch_size=4,
                positions=np.arange(4),
            )

    def test_dummy_block_rows_reject_asymmetric_group_pool(self):
        runner = self._build_runner()
        runner.kv_cache_config = SimpleNamespace(
            num_blocks=8,
            num_blocks_per_group=[3, 8],
        )
        runner.input_batch = SimpleNamespace(
            block_table=MultiGroupBlockTable(
                max_num_reqs=4,
                max_model_len=16,
                max_num_batched_tokens=4,
                pin_memory=False,
                device=torch.device("cpu"),
                block_sizes=[4, 8],
                kernel_sizes=[[4], [8]],
                max_num_blocks=[4, 2],
            )
        )

        with self.assertRaisesRegex(
            RuntimeError,
            r"KV group 0: .*available_blocks=3",
        ):
            runner._prepare_staged_sfa_dummy_block_tables(
                batch_size=4,
                positions=np.arange(4),
            )

    def test_dummy_position_rejects_logical_row_overflow(self):
        runner = self._build_runner()
        runner.kv_cache_config = SimpleNamespace(
            num_blocks=8,
            num_blocks_per_group=[8, 8],
        )
        runner.input_batch = SimpleNamespace(
            block_table=MultiGroupBlockTable(
                max_num_reqs=4,
                max_model_len=16,
                max_num_batched_tokens=4,
                pin_memory=False,
                device=torch.device("cpu"),
                block_sizes=[4, 8],
                kernel_sizes=[[4], [8]],
                max_num_blocks=[4, 2],
            )
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "logical block-table capacity for KV group 0",
        ):
            runner._prepare_staged_sfa_dummy_block_tables(
                batch_size=4,
                positions=np.array([16, 1, 2, 3], dtype=np.int64),
            )

    def test_dummy_position_capacity_includes_cp_world_size(self):
        runner = self._build_runner()
        runner.kv_cache_config = SimpleNamespace(
            num_blocks=8,
            num_blocks_per_group=[8, 8],
        )
        block_table = MultiGroupBlockTable(
            max_num_reqs=4,
            max_model_len=16,
            max_num_batched_tokens=4,
            pin_memory=False,
            device=torch.device("cpu"),
            block_sizes=[4, 8],
            kernel_sizes=[[4], [8]],
            max_num_blocks=[4, 2],
        )
        for group_table in block_table.block_tables:
            group_table.dcp_world_size = 2
            group_table.dcp_rank = 0
            group_table.pcp_world_size = 1
            group_table.pcp_rank = 0
        runner.input_batch = SimpleNamespace(block_table=block_table)

        runner._prepare_staged_sfa_dummy_block_tables(
            batch_size=4,
            positions=np.array([16, 18, 20, 22], dtype=np.int64),
        )

        for group_table in block_table.block_tables:
            self.assertEqual(
                np.unique(group_table.slot_mapping.np[:4]).size,
                4,
            )


class TestSFALayerwiseGraphModeCompatibility(unittest.TestCase):
    @staticmethod
    def _build_runner(mode, *, use_sparse=True):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.use_sparse = use_sparse
        runner._profiling_cudagraph_memory = False
        runner.compilation_config = SimpleNamespace(cudagraph_mode=mode)
        return runner

    def test_internal_data_parallel_staged_graph_is_accepted(self):
        runner = self._build_runner(CUDAGraphMode.PIECEWISE)
        runner.vllm_config = object()
        runner.parallel_config = SimpleNamespace(data_parallel_size=2)

        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                model_runner_module,
                "staged_sfa_connector_supports_sparse_load",
                return_value=True,
            ),
        ):
            runner._validate_sfa_layerwise_connector_cudagraph_mode()

    def test_explicit_unsupported_staged_graph_request_is_rejected(self):
        runner = self._build_runner(CUDAGraphMode.PIECEWISE)
        runner.vllm_config = object()
        runner.parallel_config = SimpleNamespace(data_parallel_size=1)

        with (
            patch.object(
                model_runner_module.envs_ascend,
                "VLLM_ASCEND_SFA_STAGED_GRAPH",
                True,
            ),
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=False,
            ),
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configuration_errors",
                return_value=("only fixed-width MTP speculative decoding is supported",),
            ),
            self.assertRaisesRegex(ValueError, "MTP"),
        ):
            runner._validate_sfa_layerwise_connector_cudagraph_mode()

    def test_staged_graph_requires_sparse_load_connector_capability(self):
        runner = self._build_runner(CUDAGraphMode.PIECEWISE)
        runner.vllm_config = object()
        runner.parallel_config = SimpleNamespace(data_parallel_size=1)

        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                model_runner_module,
                "staged_sfa_connector_supports_sparse_load",
                return_value=False,
            ),
            self.assertRaisesRegex(
                ValueError,
                "batched sparse selective loads",
            ),
        ):
            runner._validate_sfa_layerwise_connector_cudagraph_mode()

        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                model_runner_module,
                "staged_sfa_connector_supports_sparse_load",
                return_value=True,
            ),
        ):
            runner._validate_sfa_layerwise_connector_cudagraph_mode()

    def test_profiling_defers_connector_capability_validation(self):
        runner = self._build_runner(CUDAGraphMode.PIECEWISE)
        runner.vllm_config = object()
        runner.parallel_config = SimpleNamespace(data_parallel_size=1)
        runner._profiling_cudagraph_memory = True

        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                model_runner_module,
                "staged_sfa_connector_supports_sparse_load",
                return_value=False,
            ),
        ):
            runner._validate_sfa_layerwise_connector_cudagraph_mode()

    def test_full_graph_modes_are_rejected_for_layerwise_connector(self):
        connector = SimpleNamespace(uses_layerwise_model_callbacks=True)
        for mode in (
            CUDAGraphMode.FULL,
            CUDAGraphMode.FULL_DECODE_ONLY,
        ):
            with (
                self.subTest(mode=mode),
                patch.object(
                    model_runner_module,
                    "has_kv_transfer_group",
                    return_value=True,
                ),
                patch.object(
                    model_runner_module,
                    "get_kv_transfer_group",
                    return_value=connector,
                ),
                self.assertRaisesRegex(ValueError, "PIECEWISE"),
            ):
                runner = self._build_runner(mode)
                runner.vllm_config = object()
                runner.parallel_config = SimpleNamespace(data_parallel_size=1)
                with patch.object(
                    model_runner_module,
                    "staged_sfa_graph_configured",
                    return_value=False,
                ):
                    runner._validate_sfa_layerwise_connector_cudagraph_mode()

    def test_compatible_modes_and_connectors_are_accepted(self):
        cases = (
            (
                CUDAGraphMode.PIECEWISE,
                True,
                True,
                True,
            ),
            (
                CUDAGraphMode.FULL,
                False,
                True,
                True,
            ),
            (
                CUDAGraphMode.FULL,
                True,
                False,
                True,
            ),
            (
                CUDAGraphMode.FULL,
                True,
                True,
                False,
            ),
        )
        for mode, use_sparse, has_connector, uses_layerwise in cases:
            with (
                self.subTest(
                    mode=mode,
                    use_sparse=use_sparse,
                    has_connector=has_connector,
                    uses_layerwise=uses_layerwise,
                ),
                patch.object(
                    model_runner_module,
                    "has_kv_transfer_group",
                    return_value=has_connector,
                ),
                patch.object(
                    model_runner_module,
                    "get_kv_transfer_group",
                    return_value=SimpleNamespace(
                        uses_layerwise_model_callbacks=uses_layerwise,
                    ),
                ),
            ):
                runner = self._build_runner(
                    mode,
                    use_sparse=use_sparse,
                )
                runner.vllm_config = object()
                runner.parallel_config = SimpleNamespace(data_parallel_size=1)
                with patch.object(
                    model_runner_module,
                    "staged_sfa_graph_configured",
                    return_value=False,
                ):
                    runner._validate_sfa_layerwise_connector_cudagraph_mode()


class TestStagedSFAStartupCaptureValidation(unittest.TestCase):
    def setUp(self):
        configured = patch.object(
            model_runner_module,
            "staged_sfa_graph_configured",
            return_value=True,
        )
        capture_sizes = patch.object(
            model_runner_module,
            "staged_sfa_graph_capture_sizes",
            return_value=(1, 2),
        )
        configured.start()
        capture_sizes.start()
        self.addCleanup(configured.stop)
        self.addCleanup(capture_sizes.stop)

    @staticmethod
    def _build_runner():
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.compilation_config = SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
        )
        runner.vllm_config = SimpleNamespace(
            compilation_config=runner.compilation_config,
            model_config=SimpleNamespace(
                use_mla=True,
                hf_text_config=SimpleNamespace(index_topk=2048),
            ),
            kv_transfer_config=object(),
            speculative_config=None,
            lora_config=None,
        )
        runner._staged_sfa_startup_capture_attempted = False
        runner._profiling_cudagraph_memory = False
        return runner

    def test_capture_model_preserves_result_and_validates_cross_layer_warmup(self):
        runner = self._build_runner()
        calls = []
        draft_seal = MagicMock(return_value=2)
        runner.drafter = SimpleNamespace(
            use_staged_mtp_draft_graph=True,
            seal_staged_mtp_draft_graphs=draft_seal,
        )
        impl = SimpleNamespace(
            seal_staged_sfa_capture=MagicMock(),
        )
        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                model_runner_module,
                "_torch_cuda_wrapper",
                return_value=nullcontext(),
            ),
            patch.object(
                model_runner_module,
                "_replace_gpu_model_runner_function_wrapper",
                return_value=nullcontext(),
            ),
            patch.object(
                runner,
                "_reset_staged_sfa_startup_capture",
                side_effect=lambda: calls.append("reset"),
            ),
            patch.object(
                model_runner_module.GPUModelRunner,
                "capture_model",
                side_effect=lambda _runner: calls.append("parent") or 123,
            ) as parent_capture,
            patch.object(
                model_runner_module.ACLGraphWrapper,
                "seal_staged_entries",
                return_value=4,
            ) as seal_entries,
            patch.object(
                runner,
                "_collect_staged_sfa_impls",
                return_value=(("layer-0", impl),),
            ),
        ):
            result = runner.capture_model()

        self.assertEqual(result, 123)
        self.assertEqual(calls, ["reset", "parent"])
        self.assertTrue(runner._staged_sfa_startup_capture_attempted)
        self.assertEqual(runner._staged_sfa_impls, (("layer-0", impl),))
        parent_capture.assert_called_once_with(runner)
        graph_keys = tuple(
            StagedSFAGraphKey.exact_q1(size) for size in (1, 2)
        )
        impl.seal_staged_sfa_capture.assert_called_once_with(graph_keys)
        seal_entries.assert_called_once_with(graph_keys, 2)
        draft_seal.assert_called_once_with((1, 2))

    def test_capture_model_seals_target_staged_graph_when_draft_graph_is_off(self):
        runner = self._build_runner()
        runner.vllm_config.speculative_config = SimpleNamespace(
            num_speculative_tokens=1,
        )
        runner.decode_threshold = 2
        draft_seal = MagicMock()
        runner.drafter = SimpleNamespace(
            use_staged_mtp_draft_graph=False,
            seal_staged_mtp_draft_graphs=draft_seal,
        )
        impl = SimpleNamespace(
            seal_staged_sfa_capture=MagicMock(),
        )
        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_capture_sizes",
                return_value=(2, 4),
            ),
            patch.object(
                model_runner_module,
                "_torch_cuda_wrapper",
                return_value=nullcontext(),
            ),
            patch.object(
                model_runner_module,
                "_replace_gpu_model_runner_function_wrapper",
                return_value=nullcontext(),
            ),
            patch.object(
                runner,
                "_reset_staged_sfa_startup_capture",
            ),
            patch.object(
                model_runner_module.GPUModelRunner,
                "capture_model",
                return_value=123,
            ),
            patch.object(
                model_runner_module.ACLGraphWrapper,
                "seal_staged_entries",
                return_value=4,
            ) as seal_entries,
            patch.object(
                runner,
                "_collect_staged_sfa_impls",
                return_value=(("layer-0", impl),),
            ),
        ):
            result = runner.capture_model()

        graph_keys = tuple(
            StagedSFAGraphKey.fixed_spec(requests, 2)
            for requests in (1, 2)
        )
        self.assertEqual(result, 123)
        impl.seal_staged_sfa_capture.assert_called_once_with(graph_keys)
        seal_entries.assert_called_once_with(graph_keys, 2)
        draft_seal.assert_not_called()

    def test_capture_model_counts_target_debug_split_islands(self):
        runner = self._build_runner()
        impls = tuple(
            (
                f"layer-{index}",
                SimpleNamespace(seal_staged_sfa_capture=MagicMock()),
            )
            for index in range(3)
        )
        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                model_runner_module.envs_ascend,
                "VLLM_ASCEND_MTP_DRAFT_DEBUG",
                True,
            ),
            patch.object(
                model_runner_module,
                "_torch_cuda_wrapper",
                return_value=nullcontext(),
            ),
            patch.object(
                model_runner_module,
                "_replace_gpu_model_runner_function_wrapper",
                return_value=nullcontext(),
            ),
            patch.object(runner, "_reset_staged_sfa_startup_capture"),
            patch.object(
                model_runner_module.GPUModelRunner,
                "capture_model",
                return_value=123,
            ),
            patch.object(
                model_runner_module.ACLGraphWrapper,
                "seal_staged_entries",
                return_value=14,
            ) as seal_entries,
            patch.object(
                runner,
                "_collect_staged_sfa_impls",
                return_value=impls,
            ),
        ):
            result = runner.capture_model()

        graph_keys = tuple(
            StagedSFAGraphKey.exact_q1(size) for size in (1, 2)
        )
        self.assertEqual(result, 123)
        # Normal layout: 3 target islands + 1 tail. Diagnostics add an input
        # and output split around each of the 3 target layers: 4 + 2 * 3 = 10.
        seal_entries.assert_called_once_with(graph_keys, 10)

    def test_collect_staged_sfa_impls_excludes_mtp_draft_layer(self):
        runner = self._build_runner()
        runner.vllm_config.model_config.hf_text_config.num_hidden_layers = 78
        target_impl = SimpleNamespace(enable_staged_sfa_graph=True)
        draft_impl = SimpleNamespace(enable_staged_sfa_graph=True)
        target_layer = SimpleNamespace(
            impl=target_impl,
            layer_name="model.layers.77.self_attn.attn",
        )
        draft_layer = SimpleNamespace(
            impl=draft_impl,
            layer_name="model.layers.78.self_attn.attn",
        )

        with patch.object(
            model_runner_module,
            "get_layers_from_vllm_config",
            return_value={
                target_layer.layer_name: target_layer,
                draft_layer.layer_name: draft_layer,
            },
        ):
            collected = runner._collect_staged_sfa_impls()

        self.assertEqual(
            collected,
            ((target_layer.layer_name, target_impl),),
        )

    def test_capture_model_rejects_a_missing_configured_key(self):
        runner = self._build_runner()
        impl = SimpleNamespace(
            seal_staged_sfa_capture=MagicMock(
                side_effect=RuntimeError("missing_keys=(2,)"),
            ),
        )
        with (
            patch.object(
                model_runner_module,
                "_torch_cuda_wrapper",
                return_value=nullcontext(),
            ),
            patch.object(
                model_runner_module,
                "_replace_gpu_model_runner_function_wrapper",
                return_value=nullcontext(),
            ),
            patch.object(runner, "_reset_staged_sfa_startup_capture"),
            patch.object(
                model_runner_module.GPUModelRunner,
                "capture_model",
                return_value=0,
            ),
            patch.object(
                runner,
                "_collect_staged_sfa_impls",
                return_value=(("layer-0", impl),),
            ),
            self.assertRaisesRegex(
                RuntimeError,
                r"layer-0.*missing_keys=.*2",
            ),
        ):
            runner.capture_model()

    def test_failed_parent_capture_cannot_retry_stale_outer_graphs(self):
        runner = self._build_runner()
        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                model_runner_module,
                "_torch_cuda_wrapper",
                return_value=nullcontext(),
            ),
            patch.object(
                model_runner_module,
                "_replace_gpu_model_runner_function_wrapper",
                return_value=nullcontext(),
            ),
            patch.object(
                runner,
                "_reset_staged_sfa_startup_capture",
            ),
            patch.object(
                model_runner_module.GPUModelRunner,
                "capture_model",
                side_effect=RuntimeError("capture failed"),
            ) as parent_capture,
        ):
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                runner.capture_model()
            with self.assertRaisesRegex(
                RuntimeError,
                "startup graph capture was already attempted",
            ):
                runner.capture_model()

        self.assertTrue(runner._staged_sfa_startup_capture_attempted)
        parent_capture.assert_called_once_with(runner)

    def test_second_capture_attempt_is_rejected_before_parent(self):
        runner = self._build_runner()
        runner._staged_sfa_startup_capture_attempted = True
        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                runner,
                "_reset_staged_sfa_startup_capture",
            ) as reset,
            patch.object(
                model_runner_module.GPUModelRunner,
                "capture_model",
            ) as parent_capture,
            self.assertRaisesRegex(
                RuntimeError,
                "startup graph capture was already attempted",
            ),
        ):
            runner.capture_model()

        reset.assert_not_called()
        parent_capture.assert_not_called()

    def test_graph_memory_profile_resets_temporary_capture_state(self):
        runner = self._build_runner()
        with (
            patch.object(
                model_runner_module,
                "_torch_cuda_wrapper",
                return_value=nullcontext(),
            ),
            patch.object(
                model_runner_module,
                "_replace_gpu_model_runner_function_wrapper",
                return_value=nullcontext(),
            ),
            patch.object(
                model_runner_module.GPUModelRunner,
                "profile_cudagraph_memory",
                side_effect=lambda _runner: (
                    self.assertTrue(runner._profiling_cudagraph_memory)
                    or 123
                ),
            ) as parent_profile,
            patch.object(
                runner,
                "_reset_staged_sfa_startup_capture",
            ) as reset,
            patch.object(
                model_runner_module,
                "reset_graph_params",
            ) as reset_params,
        ):
            result = runner.profile_cudagraph_memory()

        self.assertEqual(result, 123)
        self.assertFalse(runner._profiling_cudagraph_memory)
        parent_profile.assert_called_once_with(runner)
        reset.assert_called_once_with()
        reset_params.assert_called_once_with()

    def test_graph_memory_profile_cleans_up_after_failure(self):
        runner = self._build_runner()
        runner.cudagraph_dispatcher = SimpleNamespace(
            cudagraph_keys={
                CUDAGraphMode.PIECEWISE: {object()},
            },
            keys_initialized=True,
        )
        with (
            patch.object(
                model_runner_module,
                "_torch_cuda_wrapper",
                return_value=nullcontext(),
            ),
            patch.object(
                model_runner_module,
                "_replace_gpu_model_runner_function_wrapper",
                return_value=nullcontext(),
            ),
            patch.object(
                model_runner_module.GPUModelRunner,
                "profile_cudagraph_memory",
                side_effect=RuntimeError("profile failed"),
            ),
            patch.object(
                model_runner_module.ACLGraphWrapper,
                "clear_all_graphs",
            ) as clear_graphs,
            patch.object(
                runner,
                "_cleanup_profiling_kv_cache",
            ) as cleanup_cache,
            patch.object(
                runner,
                "_reset_staged_sfa_startup_capture",
            ) as reset,
            patch.object(
                model_runner_module,
                "set_cudagraph_capturing_enabled",
            ) as set_capture_enabled,
            patch.object(
                model_runner_module,
                "reset_graph_params",
            ) as reset_params,
            self.assertRaisesRegex(RuntimeError, "profile failed"),
        ):
            runner.profile_cudagraph_memory()

        self.assertFalse(runner._profiling_cudagraph_memory)
        clear_graphs.assert_called_once_with()
        cleanup_cache.assert_called_once_with()
        reset.assert_called_once_with()
        reset_params.assert_called_once_with()
        set_capture_enabled.assert_called_once_with(False)
        self.assertFalse(runner.cudagraph_dispatcher.keys_initialized)
        self.assertFalse(
            runner.cudagraph_dispatcher.cudagraph_keys[
                CUDAGraphMode.PIECEWISE
            ]
        )

    def test_kv_cache_reinitialization_after_capture_is_rejected(self):
        runner = self._build_runner()
        runner._staged_sfa_startup_capture_attempted = True

        with (
            patch.object(
                runner,
                "_validate_sfa_layerwise_connector_cudagraph_mode",
            ) as validate,
            self.assertRaisesRegex(
                RuntimeError,
                "KV cache cannot be reinitialized after graph capture",
            ),
        ):
            runner.initialize_kv_cache(object())

        validate.assert_not_called()

if __name__ == "__main__":
    unittest.main()
