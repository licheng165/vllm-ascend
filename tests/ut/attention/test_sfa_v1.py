import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.forward_context import BatchDescriptor

from tests.ut.attention.utils import patch_distributed_groups
from tests.ut.base import TestBase
from vllm_ascend.ascend_forward_context import (
    STAGED_SFA_SINGLETON_GRAPH_KEY,
    StagedSFAGraphKey,
)
from vllm_ascend.attention.attention_v1 import AscendAttentionState

if "torch_npu._inductor" not in sys.modules:
    sys.modules["torch_npu._inductor"] = MagicMock()

import vllm_ascend.attention.sfa_v1 as sfa_v1
import vllm_ascend.attention.target_sfa_diagnostics as target_diag
import vllm_ascend.attention.utils as attention_utils
from vllm_ascend.attention.sfa_v1 import (
    AscendSFABackend,
    AscendSFAImpl,
    AscendSFAMetadata,
    AscendSFAMetadataBuilder,
    _update_dsa_split_boundary_in_place,
)
from vllm_ascend.utils import (
    StagedSFARouteAction,
    StagedSFARouteDecision,
    StagedSFARouteReason,
    enable_dsa_cp,
)


def test_sfa_metadata_declares_cached_decode_split_boundary() -> None:
    field = AscendSFAMetadata.__dataclass_fields__["decode_split_boundary"]
    assert field.default is None


def test_sparse_boundary_updates_preallocated_storage_in_place():
    boundary_cpu = torch.tensor([9, 9, 19, 0], dtype=torch.int32)
    boundary = torch.empty(4, dtype=torch.int32)
    boundary.copy_(boundary_cpu)
    metadata = SimpleNamespace(
        split_boundary=boundary,
        decode_split_boundary_cpu=boundary_cpu.numpy(),
        decode_split_boundary_cpu_tensor=boundary_cpu,
        decode_req_indices_cpu=torch.tensor([0, 0, 1, -1], dtype=torch.int32).numpy(),
        seq_lens_cpu=torch.tensor([513, 770], dtype=torch.int32),
        num_decode_tokens=3,
        decode_split_boundary=None,
    )
    address = metadata.split_boundary.data_ptr()

    with (
        patch.object(
            sfa_v1.torch,
            "tensor",
            side_effect=AssertionError("unexpected torch.tensor"),
        ),
        patch.object(
            sfa_v1.torch,
            "arange",
            side_effect=AssertionError("unexpected torch.arange"),
        ),
        patch.object(
            sfa_v1.torch.nn.functional,
            "pad",
            side_effect=AssertionError("unexpected pad"),
        ),
    ):
        actual = _update_dsa_split_boundary_in_place(
            metadata,
            cached_tokens=[512, 768],
            decode_window_size=256,
        )

    assert actual.data_ptr() == address
    assert metadata.decode_split_boundary.data_ptr() == address
    assert actual.tolist() == [512, 512, 768, 0]


def test_sparse_boundary_short_frontier_preserves_zero_pad_semantics():
    boundary_cpu = torch.tensor([9, 19], dtype=torch.int32)
    boundary = boundary_cpu.clone()
    metadata = SimpleNamespace(
        split_boundary=boundary,
        decode_split_boundary_cpu=boundary_cpu.numpy(),
        decode_split_boundary_cpu_tensor=boundary_cpu,
        decode_req_indices_cpu=torch.tensor([0, 1], dtype=torch.int32).numpy(),
        seq_lens_cpu=torch.tensor([10, 20], dtype=torch.int32),
        num_decode_tokens=2,
        decode_split_boundary=None,
    )

    actual = _update_dsa_split_boundary_in_place(
        metadata,
        cached_tokens=[8],
        decode_window_size=0,
    )

    assert actual.tolist() == [8, 0]


def test_lmcache_load_stat_aggregates_mtp_rows_and_resets():
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.index_topk = 4
    impl._lmcache_load_stat_enabled = True
    impl._lmcache_load_stat_tokens = None
    impl._lmcache_load_stat_denominator = 0
    impl._lmcache_load_stat_rows = 0
    impl._lmcache_load_stat_calls = 0
    impl._lmcache_load_stat_last_log = 10.0

    with (
        patch.object(sfa_v1, "monotonic", side_effect=[10.5, 11.2]),
        patch.object(sfa_v1.logger, "info") as log,
    ):
        impl._record_lmcache_load_stat(
            "model.layers.78.self_attn",
            torch.tensor([[1, 99], [2, 99]], dtype=torch.int32),
            request_count=2,
            decode_rows=4,
        )
        impl._record_lmcache_load_stat(
            "model.layers.78.self_attn",
            torch.tensor([[3, 99], [4, 99]], dtype=torch.int32),
            request_count=2,
            decode_rows=4,
        )

    log.assert_called_once()
    args = log.call_args.args
    assert args[1:] == (
        "model.layers.78.self_attn",
        pytest.approx(1.2),
        10,
        4,
        8,
        32,
        pytest.approx(10 / 32),
        2,
    )
    assert impl._lmcache_load_stat_tokens.item() == 0
    assert impl._lmcache_load_stat_denominator == 0
    assert impl._lmcache_load_stat_rows == 0
    assert impl._lmcache_load_stat_calls == 0
    assert impl._lmcache_load_stat_last_log == pytest.approx(11.2)


def test_lmcache_load_stat_disabled_does_not_touch_tensor_or_clock():
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl._lmcache_load_stat_enabled = False
    impl._lmcache_load_stat_tokens = None

    with patch.object(sfa_v1, "monotonic") as clock:
        impl._record_lmcache_load_stat(
            "model.layers.0.self_attn",
            torch.tensor([3], dtype=torch.int32),
            request_count=1,
            decode_rows=1,
        )

    clock.assert_not_called()
    assert impl._lmcache_load_stat_tokens is None


def test_target_sfa_diagnostics_save_layer_io_and_retrieve_state():
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.tp_rank = 3
    impl.block_size = 4
    impl._sorted_resident_state = None
    impl._sorted_resident_workspace_views = {}
    impl.index_topk = 2
    impl._staged_sfa_bridge_buffers = (
        torch.arange(8, dtype=torch.float32).reshape(2, 1, 4),
        torch.arange(4, dtype=torch.float32).reshape(2, 1, 2),
        torch.tensor([[[0, 5]], [[1, 6]]], dtype=torch.int32),
        torch.tensor([[0, 1, 5, 6]], dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32),
        torch.tensor([[12, 13, 14, 15]], dtype=torch.long),
    )
    kv_cache = (
        torch.arange(20 * 4 * 2, dtype=torch.float32).reshape(20, 4, 1, 2),
        torch.arange(20 * 4, dtype=torch.float32).reshape(20, 4, 1, 1),
        torch.empty(20, 4, 1, 1),
    )
    impl._staged_sfa_capture_state = SimpleNamespace(
        runtime=("model.layers.0.self_attn.attn", kv_cache, None, False),
        remap_boundary=torch.tensor([4, 4], dtype=torch.int32),
    )
    raw_topk = torch.tensor([[2, 3, 5, 6]], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_actual_tokens=2,
        num_decode_tokens=2,
        req_ids=["request-0"],
        decode_request_ids_compact=["request-0"],
        decode_req_indices=torch.tensor([0, 0], dtype=torch.int32),
        block_table=torch.tensor([[10, 11, 12, 13]], dtype=torch.int32),
        resident_state_indices=None,
        decode_union_mapping_workspace=raw_topk,
    )
    context = SimpleNamespace(
        staged_sfa_graph_key=object(),
        staged_sfa_graph_dummy_run=False,
    )

    with (
        TemporaryDirectory() as temp_dir,
        patch.dict(
            os.environ,
            {"VLLM_ASCEND_MTP_DRAFT_DEBUG": "1"},
        ),
        patch.object(
            target_diag,
            "MTP_DRAFT_DIAG_ROOT",
            Path(temp_dir),
        ),
        patch.object(
            sfa_v1.torch.npu,
            "is_current_stream_capturing",
            return_value=False,
        ),
        patch.object(sfa_v1.torch.npu, "synchronize"),
    ):
        input_tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        impl.target_sfa_diagnostic_boundary(
            "model.layers.0.self_attn.attn",
            "input",
            input_tensor,
            metadata,
            context,
        )
        diagnostic = impl._target_sfa_diag_pre_retrieve(
            "model.layers.0.self_attn.attn",
            impl._staged_sfa_bridge_buffers[3],
            impl._staged_sfa_bridge_buffers[4],
            impl._staged_sfa_bridge_buffers[5],
            metadata,
            context,
            1,
        )
        impl._target_sfa_diag_post_retrieve(diagnostic)
        output_tensor = input_tensor + 1
        impl.target_sfa_diagnostic_boundary(
            "model.layers.0.self_attn.attn",
            "output",
            output_tensor,
            metadata,
            context,
        )
        target_diag.target_tail_boundary(
            context._target_sfa_diag_session,
            "model_forward",
            output_tensor + 1,
        )

        session = context._target_sfa_diag_session
        saved_input = torch.load(
            target_diag.target_sfa_path(session, 0, "input"),
            weights_only=True,
        )
        saved_pre = torch.load(
            target_diag.target_sfa_path(session, 0, "pre"),
            weights_only=True,
        )
        saved_lmcache = torch.load(
            target_diag.target_sfa_path(session, 0, "lmcache"),
            weights_only=True,
        )
        saved_output = torch.load(
            target_diag.target_sfa_path(session, 0, "output"),
            weights_only=True,
        )
        saved_model_output = torch.load(
            session.output_dir / "target_model_forward.pt",
            weights_only=True,
        )

    assert torch.equal(saved_input["tensor"], input_tensor)
    assert saved_input["metadata"]["req_ids"] == ["request-0"]
    assert saved_pre["physical_block_ids"] == [3, 10, 11]
    assert torch.equal(
        saved_pre["topk_indices"],
        impl._staged_sfa_bridge_buffers[2],
    )
    assert torch.equal(saved_pre["raw_topk"], raw_topk.reshape(2, 1, 2))
    assert torch.equal(
        saved_pre["remap_boundary"],
        impl._staged_sfa_capture_state.remap_boundary,
    )
    assert saved_pre["cache_before_lmcache"][0]["physical_block_ids"] == [3, 10, 11]
    assert saved_lmcache["cache_after_lmcache"][0]["physical_block_ids"] == [3, 10, 11]
    assert torch.equal(saved_output["tensor"], output_tensor)
    assert torch.equal(saved_model_output["value"], output_tensor + 1)


def test_target_sfa_diagnostics_skip_dummy_graph_capture():
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    context = SimpleNamespace(
        staged_sfa_graph_key=object(),
        staged_sfa_graph_dummy_run=True,
    )
    with (
        patch.dict(
            os.environ,
            {"VLLM_ASCEND_MTP_DRAFT_DEBUG": "1"},
        ),
        patch.object(sfa_v1.torch.npu, "synchronize") as synchronize,
    ):
        impl.target_sfa_diagnostic_boundary(
            "model.layers.0.self_attn.attn",
            "input",
            torch.ones(1, 2),
            SimpleNamespace(),
            context,
        )

    synchronize.assert_not_called()


def test_sparse_boundary_gathers_by_request_for_mtp_rows():
    boundary_cpu = torch.tensor(
        [11, 11, 22, 22],
        dtype=torch.int32,
    )
    boundary = boundary_cpu.clone()
    metadata = SimpleNamespace(
        split_boundary=boundary,
        decode_split_boundary_cpu=boundary_cpu.numpy(),
        decode_split_boundary_cpu_tensor=boundary_cpu,
        decode_req_indices_cpu=np.array(
            [1, 0, 1, 0],
            dtype=np.int32,
        ),
        seq_lens_cpu=torch.tensor([129, 257], dtype=torch.int32),
        num_decode_tokens=4,
        decode_split_boundary=None,
    )

    actual = _update_dsa_split_boundary_in_place(
        metadata,
        cached_tokens=[120, 240],
        decode_window_size=0,
    )

    assert actual.tolist() == [240, 120, 240, 120]


def test_sparse_boundary_gathers_decode_window_without_cached_frontier():
    boundary_cpu = torch.tensor(
        [11, 11, 22, 22, 0],
        dtype=torch.int32,
    )
    metadata = SimpleNamespace(
        split_boundary=boundary_cpu.clone(),
        decode_split_boundary_cpu=boundary_cpu.numpy(),
        decode_split_boundary_cpu_tensor=boundary_cpu,
        decode_req_indices_cpu=np.array(
            [0, 0, 1, 1, -1],
            dtype=np.int32,
        ),
        seq_lens_cpu=torch.tensor([513, 770], dtype=torch.int32),
        num_decode_tokens=4,
        decode_split_boundary=None,
    )

    actual = _update_dsa_split_boundary_in_place(
        metadata,
        cached_tokens=None,
        decode_window_size=256,
    )

    assert actual.tolist() == [512, 512, 768, 768, 0]


def test_sparse_boundary_rejects_request_outside_seq_lens():
    boundary_cpu = torch.tensor([11, 22], dtype=torch.int32)
    metadata = SimpleNamespace(
        split_boundary=boundary_cpu.clone(),
        decode_split_boundary_cpu=boundary_cpu.numpy(),
        decode_split_boundary_cpu_tensor=boundary_cpu,
        decode_req_indices_cpu=np.array([0, 2], dtype=np.int32),
        seq_lens_cpu=torch.tensor([513, 770], dtype=torch.int32),
        num_decode_tokens=2,
        decode_split_boundary=None,
    )

    with pytest.raises(
        RuntimeError,
        match="request outside seq_lens",
    ):
        _update_dsa_split_boundary_in_place(
            metadata,
            cached_tokens=[512, 768],
            decode_window_size=0,
        )


def test_sparse_boundary_rejects_empty_frontiers_with_decode_rows():
    boundary_cpu = torch.tensor([11], dtype=torch.int32)
    metadata = SimpleNamespace(
        split_boundary=boundary_cpu.clone(),
        decode_split_boundary_cpu=boundary_cpu.numpy(),
        decode_split_boundary_cpu_tensor=boundary_cpu,
        decode_req_indices_cpu=np.array([0], dtype=np.int32),
        seq_lens_cpu=torch.tensor([513], dtype=torch.int32),
        num_decode_tokens=1,
        decode_split_boundary=None,
    )

    with pytest.raises(
        RuntimeError,
        match="no request boundaries",
    ):
        _update_dsa_split_boundary_in_place(
            metadata,
            cached_tokens=[],
            decode_window_size=0,
        )


def test_sparse_boundary_prefers_explicit_committed_end():
    from vllm_ascend.attention import utils as attention_utils

    metadata = SimpleNamespace(
        requests=[
            SimpleNamespace(
                req_id="resident",
                is_sparse_decode=True,
                dsa_current_released_frontier=0,
                dsa_nonresident_frontier=0,
                load_spec=SimpleNamespace(
                    can_load=True,
                    lmcache_cached_tokens=3072,
                    dsa_committed_end=3072,
                    dsa_scratch_capacity=4096,
                ),
            ),
            SimpleNamespace(
                req_id="scratch-full",
                is_sparse_decode=True,
                dsa_current_released_frontier=0,
                dsa_nonresident_frontier=0,
                load_spec=SimpleNamespace(
                    can_load=True,
                    lmcache_cached_tokens=4096,
                    dsa_committed_end=4096,
                    dsa_scratch_capacity=4096,
                ),
            ),
            SimpleNamespace(
                req_id="offloaded",
                is_sparse_decode=True,
                dsa_current_released_frontier=0,
                dsa_nonresident_frontier=8192,
                load_spec=SimpleNamespace(
                    can_load=True,
                    lmcache_cached_tokens=8192,
                    dsa_committed_end=8192,
                ),
            ),
        ]
    )
    connector = SimpleNamespace(
        supports_staged_sfa_sparse_load=True,
        uses_layerwise_model_callbacks=True,
        wait_for_layer_load=lambda *_args, **_kwargs: None,
        _get_connector_metadata=lambda: metadata,
    )
    with (
        patch.object(attention_utils, "has_kv_transfer_group", return_value=True),
        patch.object(attention_utils, "is_v1_kv_transfer_group", return_value=True),
        patch.object(attention_utils, "get_kv_transfer_group", return_value=connector),
    ):
        assert attention_utils.get_lmcache_sparse_cached_tokens(
            ["resident", "scratch-full", "offloaded"]
        ) == [0, 4096, 8192]


def _staged_route(frontiers=(4096,)):
    return StagedSFARouteDecision(
        StagedSFARouteAction.STAGED,
        StagedSFARouteReason.ELIGIBLE,
        STAGED_SFA_SINGLETON_GRAPH_KEY,
        frontiers,
    )


class TestLMCacheSparseWaitSync(TestBase):
    def setUp(self):
        self.original_once_done = sfa_v1._lmcache_sparse_wait_sync_once_done
        sfa_v1._lmcache_sparse_wait_sync_once_done = False

    def tearDown(self):
        sfa_v1._lmcache_sparse_wait_sync_once_done = self.original_once_done

    def test_once_mode_synchronizes_only_first_sparse_wait(self):
        stream = MagicMock()
        with (
            patch.object(sfa_v1, "_LMCACHE_SPARSE_WAIT_SYNC_ONCE", True),
            patch.object(
                sfa_v1.torch.npu,
                "current_stream",
                return_value=stream,
            ) as current_stream,
        ):
            sfa_v1._sync_compute_stream_after_lmcache_sparse_wait()
            sfa_v1._sync_compute_stream_after_lmcache_sparse_wait()

        current_stream.assert_called_once_with()
        stream.synchronize.assert_called_once_with()
        self.assertTrue(sfa_v1._lmcache_sparse_wait_sync_once_done)

    def test_completed_mode_does_not_touch_npu_stream(self):
        sfa_v1._lmcache_sparse_wait_sync_once_done = True
        with patch.object(sfa_v1.torch.npu, "current_stream") as current_stream:
            sfa_v1._sync_compute_stream_after_lmcache_sparse_wait()

        current_stream.assert_not_called()

    def test_disabled_mode_does_not_synchronize(self):
        with (
            patch.object(sfa_v1, "_LMCACHE_SPARSE_WAIT_SYNC_ONCE", False),
            patch.object(sfa_v1.torch.npu, "current_stream") as current_stream,
        ):
            sfa_v1._sync_compute_stream_after_lmcache_sparse_wait()

        current_stream.assert_not_called()
        self.assertFalse(sfa_v1._lmcache_sparse_wait_sync_once_done)

    def test_sync_compute_stream_skips_when_npu_unavailable(self):
        with (
            patch.object(sfa_v1, "_LMCACHE_SPARSE_WAIT_SYNC_ONCE", True),
            patch.object(sfa_v1.torch, "npu", None),
        ):
            sfa_v1._sync_compute_stream_after_lmcache_sparse_wait()

        self.assertFalse(sfa_v1._lmcache_sparse_wait_sync_once_done)


class TestDSASparsePadding(TestBase):
    def test_trailing_graph_padding_is_zeroed_in_place(self):
        topk = torch.arange(4 * 64, dtype=torch.int32).reshape(4, 1, 64)
        original_actual = topk[:2].clone()
        input_ptr = topk.data_ptr()

        result, result_2d = sfa_v1._dsa_mask_padding_sparse_rows(
            topk,
            torch.tensor([0, 1, -1, -1], dtype=torch.int32),
        )

        self.assertEqual(result.data_ptr(), input_ptr)
        self.assertEqual(result_2d.data_ptr(), input_ptr)
        self.assertTrue(torch.equal(result[:2], original_actual))
        self.assertEqual(torch.count_nonzero(result[2:]).item(), 0)


class TestLMCacheSparseFrontier(TestBase):
    @staticmethod
    def _remap_frontiers(metadata: object, request_ids: list[str]) -> list[int]:
        connector = SimpleNamespace(_get_connector_metadata=lambda: metadata)
        with (
            patch.object(
                attention_utils,
                "staged_sfa_connector_supports_sparse_load",
                return_value=True,
            ),
            patch.object(
                attention_utils,
                "get_kv_transfer_group",
                return_value=connector,
            ),
        ):
            return attention_utils.get_lmcache_sparse_cached_tokens(request_ids)

    def test_invalid_or_duplicate_request_identity_fails_closed(self):
        sparse = SimpleNamespace(
            req_id="req-0",
            is_sparse_decode=True,
            dsa_current_released_frontier=0,
            dsa_nonresident_frontier=0,
            load_spec=SimpleNamespace(
                can_load=True,
                lmcache_cached_tokens=128,
            ),
        )
        metadata = SimpleNamespace(requests=[sparse])
        self.assertEqual(
            attention_utils.staged_sfa_metadata_sparse_load(
                metadata,
                ["req-0", "req-0"],
            ),
            (StagedSFARouteReason.INVALID_REQUEST_IDS, ()),
        )

        metadata.requests.append(sparse)
        self.assertEqual(
            attention_utils.staged_sfa_metadata_sparse_load(
                metadata,
                ["req-0"],
            ),
            (StagedSFARouteReason.DUPLICATE_MAIN_METADATA, ()),
        )

    def test_missing_active_request_frontier_fails_closed(self):
        metadata = SimpleNamespace(
            requests=[
                SimpleNamespace(
                    req_id="req-0",
                    is_sparse_decode=True,
                    dsa_current_released_frontier=0,
                    dsa_nonresident_frontier=0,
                    load_spec=SimpleNamespace(
                        can_load=True,
                        lmcache_cached_tokens=128,
                    ),
                )
            ]
        )
        self.assertEqual(
            attention_utils.staged_sfa_metadata_sparse_load(
                metadata,
                ["req-0", "req-1"],
            ),
            (StagedSFARouteReason.MISSING_CONNECTOR_METADATA, ()),
        )

    def test_save_only_metadata_is_ignored_and_main_must_be_unique(self):
        main = SimpleNamespace(
            req_id="req-0",
            is_sparse_decode=False,
            is_decode_window_save=False,
            dsa_current_released_frontier=0,
            dsa_nonresident_frontier=0,
            load_spec=None,
        )
        save_only = SimpleNamespace(
            req_id="req-0",
            is_sparse_decode=False,
            is_decode_window_save=True,
            load_spec=None,
        )
        metadata = SimpleNamespace(requests=[save_only, main])
        self.assertEqual(
            attention_utils.staged_sfa_metadata_sparse_load(
                metadata,
                ["req-0"],
            ),
            (StagedSFARouteReason.DENSE_PREFIX_HIT, (0,)),
        )

        metadata.requests.append(main)
        self.assertEqual(
            attention_utils.staged_sfa_metadata_sparse_load(
                metadata,
                ["req-0"],
            ),
            (StagedSFARouteReason.DUPLICATE_MAIN_METADATA, ()),
        )

    def test_nonresident_dense_or_unloadable_sparse_fails_closed(self):
        request = SimpleNamespace(
            req_id="req-0",
            is_sparse_decode=False,
            dsa_current_released_frontier=0,
            dsa_nonresident_frontier=4096,
            load_spec=None,
        )
        metadata = SimpleNamespace(requests=[request])
        self.assertEqual(
            attention_utils.staged_sfa_metadata_sparse_load(
                metadata,
                ["req-0"],
            ),
            (StagedSFARouteReason.DENSE_PREFIX_NOT_RESIDENT, ()),
        )

        request.is_sparse_decode = True
        request.load_spec = SimpleNamespace(can_load=False)
        self.assertEqual(
            attention_utils.staged_sfa_metadata_sparse_load(
                metadata,
                ["req-0"],
            ),
            (StagedSFARouteReason.SPARSE_LOAD_UNAVAILABLE, ()),
        )

    def test_invalid_frontier_invariants_fail_closed(self):
        valid_load = {
            "can_load": True,
            "lmcache_cached_tokens": 8192,
            "dsa_committed_end": 8192,
            "dsa_remap_frontier": 8192,
        }
        cases = {
            "missing_current": SimpleNamespace(
                req_id="req-0",
                is_sparse_decode=True,
                dsa_nonresident_frontier=0,
                load_spec=SimpleNamespace(**valid_load),
            ),
            "negative_nonresident": SimpleNamespace(
                req_id="req-0",
                is_sparse_decode=True,
                dsa_current_released_frontier=0,
                dsa_nonresident_frontier=-1,
                load_spec=SimpleNamespace(**valid_load),
            ),
            "current_above_nonresident": SimpleNamespace(
                req_id="req-0",
                is_sparse_decode=True,
                dsa_current_released_frontier=8192,
                dsa_nonresident_frontier=4096,
                load_spec=SimpleNamespace(**valid_load),
            ),
            "committed_above_cached": SimpleNamespace(
                req_id="req-0",
                is_sparse_decode=True,
                dsa_current_released_frontier=0,
                dsa_nonresident_frontier=0,
                load_spec=SimpleNamespace(
                    can_load=True,
                    lmcache_cached_tokens=4096,
                    dsa_committed_end=8192,
                ),
            ),
            "nonresident_above_remap": SimpleNamespace(
                req_id="req-0",
                is_sparse_decode=True,
                dsa_current_released_frontier=0,
                dsa_nonresident_frontier=8192,
                load_spec=SimpleNamespace(
                    can_load=True,
                    lmcache_cached_tokens=8192,
                    dsa_committed_end=8192,
                    dsa_remap_frontier=7936,
                ),
            ),
        }
        for name, request in cases.items():
            with self.subTest(name=name):
                self.assertEqual(
                    attention_utils.staged_sfa_metadata_sparse_load(
                        SimpleNamespace(requests=[request]),
                        ["req-0"],
                    ),
                    (StagedSFARouteReason.INVALID_FRONTIER, ()),
                )

    def test_frontiers_preserve_native_request_order(self):
        metadata = SimpleNamespace(
            requests=[
                SimpleNamespace(
                    req_id="req-1",
                    is_sparse_decode=True,
                    dsa_current_released_frontier=0,
                    dsa_nonresident_frontier=0,
                    load_spec=SimpleNamespace(
                        can_load=True,
                        lmcache_cached_tokens=256,
                    ),
                ),
                SimpleNamespace(
                    req_id="req-0",
                    is_sparse_decode=True,
                    dsa_current_released_frontier=0,
                    dsa_nonresident_frontier=0,
                    load_spec=SimpleNamespace(
                        can_load=True,
                        lmcache_cached_tokens=128,
                    ),
                ),
            ]
        )
        self.assertEqual(
            attention_utils.staged_sfa_metadata_sparse_load(
                metadata,
                ["req-0", "req-1"],
            ),
            (StagedSFARouteReason.ELIGIBLE, (128, 256)),
        )

    def test_dense_prefix_hit_is_not_a_sparse_graph_step(self):
        metadata = SimpleNamespace(
            requests=[
                SimpleNamespace(
                    req_id="req-0",
                    is_sparse_decode=False,
                    dsa_current_released_frontier=0,
                    dsa_nonresident_frontier=0,
                    load_spec=SimpleNamespace(
                        can_load=True,
                        lmcache_cached_tokens=18879,
                    ),
                )
            ]
        )

        self.assertEqual(
            attention_utils.staged_sfa_metadata_sparse_load(metadata, ["req-0"]),
            (StagedSFARouteReason.DENSE_PREFIX_HIT, (0,)),
        )
        self.assertEqual(
            self._remap_frontiers(metadata, ["req-0"]),
            [0],
        )
        metadata.requests[0].is_sparse_decode = True
        self.assertEqual(
            attention_utils.staged_sfa_metadata_sparse_load(metadata, ["req-0"]),
            (StagedSFARouteReason.ELIGIBLE, (18879,)),
        )
        metadata.requests[0].load_spec.can_load = False
        self.assertEqual(
            attention_utils.staged_sfa_metadata_sparse_load(metadata, ["req-0"]),
            (StagedSFARouteReason.ELIGIBLE, (0,)),
        )

    def test_sparse_route_prefers_committed_boundary_over_load_length(self):
        metadata = SimpleNamespace(
            requests=[
                SimpleNamespace(
                    req_id="resident",
                    is_sparse_decode=True,
                    dsa_current_released_frontier=0,
                    dsa_nonresident_frontier=0,
                    load_spec=SimpleNamespace(
                        can_load=True,
                        lmcache_cached_tokens=3072,
                        dsa_committed_end=3072,
                        dsa_scratch_capacity=4096,
                    ),
                ),
                SimpleNamespace(
                    req_id="offloaded",
                    is_sparse_decode=True,
                    dsa_current_released_frontier=8192,
                    dsa_nonresident_frontier=8192,
                    load_spec=SimpleNamespace(
                        can_load=True,
                        lmcache_cached_tokens=8192,
                        dsa_committed_end=8192,
                        dsa_scratch_capacity=4096,
                    ),
                ),
            ]
        )
        self.assertEqual(
            attention_utils.staged_sfa_metadata_sparse_load(
                metadata,
                ["resident", "offloaded"],
            ),
            (StagedSFARouteReason.ELIGIBLE, (0, 8192)),
        )

    def test_cold_compact_resume_excludes_recomputed_last_prompt_token(self):
        metadata = SimpleNamespace(
            requests=[
                SimpleNamespace(
                    req_id="cold-compact",
                    is_sparse_decode=True,
                    dsa_current_released_frontier=0,
                    dsa_nonresident_frontier=8192,
                    load_spec=SimpleNamespace(
                        can_load=True,
                        lmcache_cached_tokens=8193,
                        dsa_committed_end=8193,
                        dsa_remap_frontier=8192,
                        dsa_cold_compact_load=False,
                        dsa_cold_compact_resume=True,
                    ),
                )
            ]
        )

        self.assertEqual(
            attention_utils.staged_sfa_metadata_sparse_load(
                metadata,
                ["cold-compact"],
            ),
            (StagedSFARouteReason.ELIGIBLE, (8192,)),
        )
        self.assertEqual(
            self._remap_frontiers(metadata, ["cold-compact"]),
            [8192],
        )
        self.assertEqual(
            attention_utils.staged_sfa_metadata_sparse_route(
                metadata, ["cold-compact"]
            ),
            (StagedSFARouteReason.ELIGIBLE, (8192,), (True,)),
        )

    def test_mixed_resident_sparse_row_uses_zero_frontier(self):
        metadata = SimpleNamespace(
            requests=[
                SimpleNamespace(
                    req_id="dense",
                    is_sparse_decode=False,
                    dsa_current_released_frontier=0,
                    dsa_nonresident_frontier=0,
                    load_spec=SimpleNamespace(can_load=True),
                ),
                SimpleNamespace(
                    req_id="sparse",
                    is_sparse_decode=True,
                    dsa_current_released_frontier=0,
                    dsa_nonresident_frontier=0,
                    load_spec=SimpleNamespace(can_load=False),
                ),
            ]
        )

        self.assertEqual(
            attention_utils.staged_sfa_metadata_sparse_load(
                metadata,
                ["dense", "sparse"],
            ),
            (StagedSFARouteReason.MIXED_CONNECTOR_LOAD, (0, 0)),
        )

    def test_native_remap_frontiers_preserve_dense_sparse_request_order(
        self,
    ) -> None:
        metadata = SimpleNamespace(
            requests=[
                SimpleNamespace(
                    req_id="sparse",
                    is_sparse_decode=True,
                    dsa_current_released_frontier=0,
                    dsa_nonresident_frontier=0,
                    load_spec=SimpleNamespace(
                        can_load=True,
                        lmcache_cached_tokens=8192,
                        dsa_committed_end=7936,
                    ),
                ),
                SimpleNamespace(
                    req_id="dense",
                    is_sparse_decode=False,
                    dsa_current_released_frontier=0,
                    dsa_nonresident_frontier=0,
                    load_spec=SimpleNamespace(
                        can_load=True,
                        lmcache_cached_tokens=120000,
                    ),
                ),
            ]
        )
        self.assertEqual(
            self._remap_frontiers(metadata, ["dense", "sparse"]),
            [0, 7936],
        )

    def test_sparse_wait_forwards_existing_payload_event(self):
        connector = MagicMock()
        event = object()
        selected = torch.ones(1, 4, dtype=torch.int32)
        target_slots = torch.arange(4).view(1, 4)
        with (
            patch.object(attention_utils, "has_kv_transfer_group", return_value=True),
            patch.object(attention_utils, "is_v1_kv_transfer_group", return_value=True),
            patch.object(attention_utils, "get_kv_transfer_group", return_value=connector),
            patch.object(attention_utils, "_dsa_lmcache_log_layer", return_value=False),
        ):
            attention_utils.wait_for_kv_layer_from_connector(
                "layer-0",
                selected_tokens=selected,
                request_ids=["req-0"],
                target_slot_mapping=target_slots,
                payload_event=event,
            )

        connector.wait_for_layer_load.assert_called_once_with(
            "layer-0",
            selected,
            None,
            ["req-0"],
            target_slot_mapping=target_slots,
            payload_event=event,
        )

    def test_scratch_reservation_and_table_capacity_fail_closed(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "reservation is too small",
        ):
            sfa_v1._validate_dsa_scratch_capacity(
                [8, 8],
                [0, 0],
                [0, 4],
                4,
                scratch_capacity=7,
            )

        # Boundary zero means all KV stays resident. A nonzero boundary must
        # begin after the complete fixed-width request scratch reservation.
        sfa_v1._validate_dsa_scratch_capacity(
            [0, 0],
            [0, 0],
            None,
            4,
            scratch_capacity=8,
        )
        with self.assertRaisesRegex(RuntimeError, "alias live KV"):
            sfa_v1._validate_dsa_scratch_capacity(
                [4, 4],
                [0, 0],
                None,
                4,
                scratch_capacity=8,
            )

        with self.assertRaisesRegex(
            RuntimeError,
            "block-table capacity",
        ):
            sfa_v1._dsa_build_target_slot_mapping(
                torch.tensor([[0]], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int64),
                torch.tensor([4], dtype=torch.int64),
                4,
                4,
                scratch_capacity=8,
            )

    def test_fixed_staged_decode_layout_selects_q1_and_mtp2(self):
        self.assertEqual(
            sfa_v1._fixed_staged_decode_mtp(
                [0, 1, 2],
                3,
                3,
                pure_decode=True,
            ),
            1,
        )
        self.assertEqual(
            sfa_v1._fixed_staged_decode_mtp(
                [0, 0, 1, 1],
                2,
                4,
                pure_decode=True,
            ),
            2,
        )

    def test_fixed_staged_decode_layout_falls_back_for_mixed_or_irregular(self):
        self.assertIsNone(
            sfa_v1._fixed_staged_decode_mtp(
                [0, -1, 1],
                2,
                3,
                pure_decode=False,
            )
        )
        self.assertIsNone(
            sfa_v1._fixed_staged_decode_mtp(
                [0, 1, 0, 1],
                2,
                4,
                pure_decode=True,
            )
        )

    def test_fixed_staged_decode_layout_rejects_mtp_above_two(self):
        with self.assertRaisesRegex(RuntimeError, "got MTP=3"):
            sfa_v1._fixed_staged_decode_mtp(
                [0, 0, 0, 1, 1, 1],
                2,
                6,
                pure_decode=True,
            )

    def test_fixed_staged_decode_layout_falls_back_for_dp_padding(self):
        self.assertIsNone(
            sfa_v1._fixed_staged_decode_mtp(
                [0] + [-1] * 72,
                1,
                73,
                pure_decode=True,
            )
        )


@pytest.mark.parametrize("mtp", [1, 2])
def test_sorted_resident_helper_uses_active_fixed_address_prefix(mtp):
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.block_size = 128
    impl._sorted_resident_state = MagicMock()
    impl._sorted_resident_workspace = (
        sfa_v1.allocate_sorted_resident_workspace(
            4,
            mtp,
            device=torch.device("cpu"),
        )
    )
    impl._sorted_resident_workspace_views = {}
    requests = 2
    topk = torch.zeros(
        (requests * mtp, 1, sfa_v1.INDEX_TOPK),
        dtype=torch.int32,
    )
    split_boundary = torch.full(
        (requests * mtp,),
        4096,
        dtype=torch.int32,
    )
    row_req_indices = torch.arange(
        requests,
        dtype=torch.int32,
    ).repeat_interleave(mtp)
    block_table = torch.zeros((requests, 32), dtype=torch.int32)
    state_indices = torch.arange(requests, dtype=torch.int32)
    state_generations = torch.ones(requests, dtype=torch.int64)

    with (
        patch.object(
            sfa_v1,
            "prepare_resident_sharded_union_",
        ) as union,
        patch.object(
            sfa_v1,
            "prepare_sorted_resident_cache_fused_",
        ) as finalize,
    ):
        active_workspace = sfa_v1.sorted_resident_workspace_prefix(
            impl._sorted_resident_workspace,
            requests,
        )
        finalize.return_value = (
            active_workspace.miss_tokens,
            active_workspace.miss_counts[:, 0],
            active_workspace.target_slots,
        )
        outputs = impl._prepare_sorted_resident_sparse_cache(
            topk,
            split_boundary,
            row_req_indices,
            block_table,
            state_indices,
            state_generations,
            mtp=mtp,
        )

    union_workspace = union.call_args.args[6]
    finalize_workspace = finalize.call_args.args[5]
    assert union_workspace is finalize_workspace
    assert union_workspace.shard_packed.shape[0] == requests
    assert (
        union_workspace.shard_packed.data_ptr()
        == impl._sorted_resident_workspace.shard_packed.data_ptr()
    )
    assert outputs[0] is topk
    assert outputs[1] is active_workspace.miss_tokens
    assert outputs[2].data_ptr() == active_workspace.miss_counts.data_ptr()
    assert outputs[3] is active_workspace.target_slots


@pytest.mark.parametrize(
    "mtp,shards_per_row,expected",
    [
        (1, 1, 1),
        (1, 2, 2),
        (1, 4, 4),
        (2, 1, 2),
        (2, 2, 4),
        (2, 4, 8),
    ],
)
def test_configured_resident_shards_are_request_wide(
    mtp, shards_per_row, expected
):
    with patch.object(
        sfa_v1.envs,
        "VLLM_ASCEND_DSA_RESIDENT_SHARDS_PER_ROW",
        shards_per_row,
    ):
        assert sfa_v1._configured_resident_shards(mtp) == (
            shards_per_row,
            expected,
        )


def test_configured_resident_shards_default_to_four_per_row():
    with patch.object(
        sfa_v1.envs,
        "VLLM_ASCEND_DSA_RESIDENT_SHARDS_PER_ROW",
        4,
    ):
        assert sfa_v1._configured_resident_shards(1) == (4, 4)
        assert sfa_v1._configured_resident_shards(2) == (4, 8)


@pytest.mark.parametrize(
    "enabled,need_packed,decode_threshold,mtp,uses_sorted",
    [
        (True, True, 1, 1, True),
        (True, True, 2, 2, True),
        (True, True, 2, 1, False),
        (False, True, 1, 1, False),
        (True, True, 1, None, False),
        (True, False, 1, 1, False),
    ],
)
def test_decode_sparse_planner_routes_only_fixed_decode_to_resident(
    enabled,
    need_packed,
    decode_threshold,
    mtp,
    uses_sorted,
):
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.block_size = 128
    impl.dsa_resident_cache = enabled
    impl.decode_threshold = decode_threshold
    expected = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
    impl._prepare_sorted_resident_sparse_cache = MagicMock(
        return_value=expected
    )
    tensors = [MagicMock() for _ in range(9)]

    with patch.object(
        sfa_v1,
        "prepare_sparse_indices",
        return_value=expected,
    ) as ordinary:
        actual = impl._prepare_decode_sparse_indices(
            *tensors,
            local_to_union_workspace=MagicMock(),
            shard_packed_workspace=MagicMock(),
            shard_mapping_workspace=MagicMock(),
            shard_counts_workspace=MagicMock(),
            staged_mtp=mtp,
            need_packed=need_packed,
            clear_invalid_rows=True,
        )

    assert actual is expected
    assert impl._prepare_sorted_resident_sparse_cache.called is uses_sorted
    assert ordinary.called is not uses_sorted


def test_decode_sparse_planner_debug_snapshots_raw_topk_before_resident():
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.block_size = 128
    impl.dsa_resident_cache = True
    impl.decode_threshold = 2
    expected = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
    impl._prepare_sorted_resident_sparse_cache = MagicMock(
        return_value=expected
    )
    topk = torch.tensor(
        [[[7, 8]], [[9, 10]]],
        dtype=torch.int32,
    )
    raw_topk_workspace = torch.full((1, 4), -1, dtype=torch.int32)
    tensors = [topk, *(MagicMock() for _ in range(8))]

    with patch.dict(
        os.environ,
        {"VLLM_ASCEND_MTP_DRAFT_DEBUG": "1"},
    ):
        actual = impl._prepare_decode_sparse_indices(
            *tensors,
            local_to_union_workspace=raw_topk_workspace,
            shard_packed_workspace=MagicMock(),
            shard_mapping_workspace=MagicMock(),
            shard_counts_workspace=MagicMock(),
            staged_mtp=2,
            need_packed=True,
            clear_invalid_rows=True,
        )

    assert actual is expected
    assert torch.equal(raw_topk_workspace, topk.reshape(1, 4))
    impl._prepare_sorted_resident_sparse_cache.assert_called_once()


class TestStagedSFAGraphPoc(TestBase):
    def setUp(self):
        super().setUp()
        capture_sizes = patch.object(
            sfa_v1,
            "staged_sfa_graph_capture_sizes",
            return_value=(1, 4),
        )
        connector_support = patch.object(
            sfa_v1,
            "staged_sfa_connector_supports_sparse_load",
            return_value=True,
        )
        capture_sizes.start()
        connector_support.start()
        self.addCleanup(capture_sizes.stop)
        self.addCleanup(connector_support.stop)

    @staticmethod
    def _make_eligible_impl():
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        impl.dsa_shrink_latent = 2
        impl.dsa_resident_cache = True
        impl.num_kv_heads = 1
        impl.local_num_heads = 2
        impl.kv_lora_rank = 2
        impl.qk_rope_head_dim = 2
        impl.head_dim = 2
        impl.index_topk = 4
        impl.decode_threshold = 1
        impl.enable_mlapo = False
        impl.enable_dsa_cp = False
        impl.enable_dsa_cp_with_o_proj_tp = False
        impl.use_sparse_c8_indexer = False
        impl.dsa_offload_free_paged = False
        impl.q_lora_rank = 4
        impl.fused_qkv_a_proj = MagicMock()
        impl.q_a_layernorm = MagicMock()
        impl.vllm_config = MagicMock()
        impl.vllm_config.cache_config.block_size = 128
        impl.vllm_config.speculative_config = None
        impl.vllm_config.lora_config = None
        impl._staged_sfa_capture_state = sfa_v1._StagedSFACaptureState()
        impl._staged_sfa_graph_capture_sizes = (1, 4)
        impl._staged_sfa_bridge_buffers = None
        # Shared-indexer construction state (GLM-5.2): producers by default.
        impl.has_indexer = True
        impl.skip_topk = False
        impl.topk_indices_buffer = None
        impl.index_cache_enabled = False
        impl.use_index_cache = False
        impl.layer_name = "model.layers.0.self_attn.attn"
        return impl

    @staticmethod
    def _make_eligible_kv_cache(
        *,
        dtype=torch.bfloat16,
        device="cpu",
        block_size=128,
        num_blocks=2,
    ):
        return (
            torch.empty(num_blocks, block_size, 1, 2, dtype=dtype, device=device),
            torch.empty(num_blocks, block_size, 1, 2, dtype=dtype, device=device),
            torch.empty(num_blocks, block_size, 1, 2, dtype=dtype, device=device),
        )

    @staticmethod
    def _make_pre_outputs(token_rows: int = 1, request_rows: int = 1):
        return (
            torch.empty(token_rows, 2, 2),
            torch.empty(token_rows, 2, 2),
            torch.empty(token_rows, 1, 4, dtype=torch.int32),
            torch.empty(request_rows, 4, dtype=torch.int32),
            torch.empty(request_rows, dtype=torch.int32),
            torch.empty(request_rows, 4, dtype=torch.long),
        )

    @staticmethod
    def _make_decode_metadata(batch_size: int = 1, query_width: int = 1):
        num_tokens = batch_size * query_width
        scratch_capacity = 4 * query_width
        metadata = MagicMock()
        metadata.attn_state = AscendAttentionState.SpecDecoding if query_width > 1 else AscendAttentionState.DecodeOnly
        metadata.num_input_tokens = num_tokens
        metadata.num_actual_tokens = num_tokens
        metadata.num_decode_tokens = num_tokens
        metadata.cos = torch.ones(num_tokens, 2)
        metadata.sin = torch.zeros(num_tokens, 2)
        metadata.slot_mapping = torch.arange(num_tokens)
        metadata.indexer_slot_mapping = torch.arange(num_tokens)
        metadata.cum_query_lens = torch.arange(1, batch_size + 1) * query_width
        metadata.seq_lens = torch.full((batch_size,), 8 + query_width)
        metadata.seq_lens_cpu = torch.full((batch_size,), 8 + query_width)
        metadata.block_table = torch.arange(batch_size).view(batch_size, 1)
        metadata.indexer_block_table = torch.arange(batch_size).view(
            batch_size,
            1,
        )
        metadata.prompt_lens = torch.full(
            (num_tokens,),
            8,
            dtype=torch.int32,
        )
        metadata.prompt_lens_cpu_rows = [8] * num_tokens
        metadata.decode_req_indices = torch.arange(
            batch_size,
            dtype=torch.int32,
        ).repeat_interleave(query_width)
        metadata.decode_req_indices_cpu = metadata.decode_req_indices.tolist()
        metadata.decode_req_indices_compact_cpu = np.arange(
            batch_size,
            dtype=np.int64,
        )
        metadata.need_sparse_lmcache_payload = True
        metadata.decode_valid_rows_all = True
        metadata.decode_valid_row_indices = torch.arange(
            num_tokens,
            dtype=torch.int32,
        )
        metadata.decode_scratch_base = torch.zeros(
            num_tokens,
            dtype=torch.int32,
        )
        metadata.decode_scratch_base_compact = None
        metadata.decode_scratch_base_cpu = [0] * num_tokens
        metadata.decode_scratch_capacity = scratch_capacity
        metadata.decode_selected_tokens = torch.empty(
            batch_size, scratch_capacity, dtype=torch.int32
        )
        metadata.decode_selected_counts = torch.empty(
            batch_size, 16, dtype=torch.int32
        )
        metadata.decode_target_slot_mapping = torch.empty(
            batch_size, scratch_capacity, dtype=torch.long
        )
        metadata.decode_union_mapping_workspace = torch.empty(
            batch_size, scratch_capacity, dtype=torch.int32
        )
        metadata.decode_shard_packed_workspace = torch.empty(
            batch_size, 2, scratch_capacity, dtype=torch.int32
        )
        metadata.decode_shard_mapping_workspace = torch.empty_like(
            metadata.decode_shard_packed_workspace
        )
        metadata.decode_shard_counts_workspace = torch.empty(
            batch_size, 2, 16, dtype=torch.int32
        )
        metadata.resident_state_indices = torch.arange(
            batch_size,
            dtype=torch.int32,
        )
        metadata.resident_state_generations = torch.ones(
            batch_size,
            dtype=torch.int64,
        )
        metadata.decode_request_ids_compact = [f"req-{row}" for row in range(batch_size)]
        metadata.req_ids = list(metadata.decode_request_ids_compact)
        metadata.decode_remap_boundary = torch.empty(
            num_tokens,
            dtype=torch.int32,
        )
        metadata.decode_remap_boundary_ready = False
        return metadata

    def test_cross_layer_pre_uses_native_path_without_authorized_key(self):
        impl = self._make_eligible_impl()
        impl.local_num_heads = 2
        impl.forward = MagicMock()
        impl._cross_layer_kv_cache = MagicMock(return_value=(self._make_eligible_kv_cache(), "index-0", True))
        context = SimpleNamespace(staged_sfa_graph_key=None)
        hidden_states = torch.empty(16, 4)
        output = torch.empty_like(hidden_states)

        with patch.object(sfa_v1, "get_forward_context", return_value=context):
            outputs = impl.cross_layer_graph_pre(
                "layer-0",
                hidden_states,
                self._make_eligible_kv_cache(),
                self._make_decode_metadata(),
                False,
                output,
            )

        impl.forward.assert_called_once()
        self.assertEqual(
            [tuple(tensor.shape[:1]) for tensor in outputs],
            [(4,)] * 6,
        )
        self.assertTrue(all(tensor.is_contiguous() for tensor in outputs))

    def test_cross_layer_pre_fails_if_authorized_key_becomes_ineligible(self):
        impl = self._make_eligible_impl()
        impl._cross_layer_kv_cache = MagicMock(return_value=(self._make_eligible_kv_cache(), "index-0", True))
        impl._cross_layer_ineligible_reason = MagicMock(return_value="changed metadata")
        context = SimpleNamespace(
            staged_sfa_graph_key=STAGED_SFA_SINGLETON_GRAPH_KEY,
            staged_sfa_route=_staged_route(),
        )

        with (
            patch.object(sfa_v1, "get_forward_context", return_value=context),
            self.assertRaisesRegex(RuntimeError, "changed metadata"),
        ):
            impl.cross_layer_graph_pre(
                "layer-0",
                torch.empty(1, 4),
                self._make_eligible_kv_cache(),
                self._make_decode_metadata(),
                False,
                torch.empty(1, 4),
            )

    def test_cross_layer_capture_reuses_eager_boundary_storage(self):
        impl = self._make_eligible_impl()
        kv_cache = self._make_eligible_kv_cache()
        impl._staged_sfa_capture_state.initialized_cache_capacity = 1
        impl._staged_sfa_capture_state.producer_event = MagicMock()
        impl._cross_layer_kv_cache = MagicMock(return_value=(kv_cache, "index-0", True))
        impl._cross_layer_ineligible_reason = MagicMock(return_value=None)
        impl._cross_layer_pre_compute = MagicMock(
            return_value=self._make_pre_outputs()
        )
        eager_metadata = self._make_decode_metadata()
        capture_metadata = self._make_decode_metadata()
        capture_metadata.decode_remap_boundary = eager_metadata.decode_remap_boundary
        contexts = (
            SimpleNamespace(
                staged_sfa_graph_key=STAGED_SFA_SINGLETON_GRAPH_KEY,
                staged_sfa_route=_staged_route(),
                staged_sfa_graph_dummy_run=True,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            ),
            SimpleNamespace(
                staged_sfa_graph_key=STAGED_SFA_SINGLETON_GRAPH_KEY,
                staged_sfa_route=_staged_route(),
                staged_sfa_graph_dummy_run=True,
                cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            ),
        )

        with (
            patch.object(
                sfa_v1,
                "get_forward_context",
                # The eager call resolves the context once in graph_pre and
                # once more while allocating its fixed bridge storage. Replay
                # reuses that storage and resolves the context only once.
                side_effect=(contexts[0], contexts[0], contexts[1]),
            ),
            patch.object(
                sfa_v1,
                "_prepare_sfa_remap_boundary",
                return_value=eager_metadata.decode_remap_boundary,
            ) as prepare_boundary,
        ):
            outputs = [
                impl.cross_layer_graph_pre(
                    "layer-0",
                    torch.empty(1, 4),
                    kv_cache,
                    metadata,
                    False,
                    torch.empty(1, 4),
                )
                for metadata in (eager_metadata, capture_metadata)
            ]

        prepare_boundary.assert_called_once_with(
            eager_metadata,
            eager_metadata.req_ids,
            is_dummy_run=True,
            index_topk=impl.index_topk,
            cached_tokens=(4096,),
        )
        self.assertIs(
            impl._staged_sfa_capture_state.remap_boundary,
            eager_metadata.decode_remap_boundary,
        )
        self.assertIs(
            impl._cross_layer_pre_compute.call_args_list[1].args[11],
            eager_metadata.decode_remap_boundary,
        )
        self.assertEqual(
            impl._staged_sfa_capture_state.bindings.keys(),
            {STAGED_SFA_SINGLETON_GRAPH_KEY},
        )
        self.assertTrue(all(tensor.shape[0] == 4 for result in outputs for tensor in result))

    def test_cross_layer_padding_uses_fixed_graph_rows(self):
        impl = self._make_eligible_impl()
        graph_key = StagedSFAGraphKey.exact_q1(4)
        metadata = self._make_decode_metadata(4)
        metadata.num_actual_tokens = metadata.num_decode_tokens = 1
        metadata.decode_req_indices[1:] = -1
        impl._staged_sfa_capture_state.producer_event = MagicMock()
        impl._cross_layer_kv_cache = MagicMock(
            return_value=(
                self._make_eligible_kv_cache(num_blocks=4),
                "index-0",
                True,
            )
        )
        impl._cross_layer_ineligible_reason = MagicMock(return_value=None)
        impl._cross_layer_pre_compute = MagicMock(
            return_value=self._make_pre_outputs(4, 4)
        )
        context = SimpleNamespace(
            staged_sfa_graph_key=graph_key,
            staged_sfa_route=StagedSFARouteDecision(
                StagedSFARouteAction.STAGED,
                StagedSFARouteReason.ELIGIBLE,
                graph_key,
                (4096,),
            ),
            staged_sfa_graph_dummy_run=False,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        )

        with (
            patch.object(sfa_v1, "get_forward_context", return_value=context),
            patch.object(
                sfa_v1,
                "_prepare_sfa_remap_boundary",
                return_value=metadata.decode_remap_boundary,
            ),
        ):
            impl.cross_layer_graph_pre(
                "layer-0",
                torch.empty(4, 4),
                self._make_eligible_kv_cache(num_blocks=4),
                metadata,
                False,
                torch.empty(4, 4),
            )

        args = impl._cross_layer_pre_compute.call_args.args
        # The graph keeps four fixed rows, while -1 prevents its three
        # padding rows from participating in the request-level union.
        self.assertEqual(args[12].tolist(), [0, -1, -1, -1])

    def test_three_and_five_request_batches_both_have_padding_slots(self):
        """Padding alone cannot explain a failure unique to request five."""
        query_width = 2
        for actual_requests, request_capacity in ((3, 4), (5, 8)):
            with self.subTest(actual_requests=actual_requests):
                actual_tokens = actual_requests * query_width
                token_capacity = request_capacity * query_width
                padding_tokens = token_capacity - actual_tokens
                impl = self._make_eligible_impl()
                impl.decode_threshold = query_width
                graph_key = StagedSFAGraphKey.fixed_spec(
                    request_capacity,
                    query_width,
                )
                metadata = self._make_decode_metadata(token_capacity)
                metadata.num_actual_tokens = actual_tokens
                metadata.decode_req_indices = torch.cat(
                    (
                        torch.arange(actual_requests).repeat_interleave(
                            query_width
                        ),
                        torch.full((padding_tokens,), -1),
                    )
                ).to(torch.int32)
                metadata.slot_mapping[actual_tokens:].fill_(-1)
                metadata.indexer_slot_mapping[actual_tokens:].fill_(-1)
                metadata.req_ids = [
                    f"req-{index}" for index in range(actual_requests)
                ]
                impl._staged_sfa_capture_state.producer_event = MagicMock()
                impl._cross_layer_kv_cache = MagicMock(
                    return_value=(
                        self._make_eligible_kv_cache(
                            num_blocks=request_capacity
                        ),
                        "index-0",
                        True,
                    )
                )
                impl._cross_layer_ineligible_reason = MagicMock(
                    return_value=None
                )
                impl._cross_layer_pre_compute = MagicMock(
                    return_value=self._make_pre_outputs(
                        token_capacity,
                        request_capacity,
                    )
                )
                context = SimpleNamespace(
                    staged_sfa_graph_key=graph_key,
                    staged_sfa_route=StagedSFARouteDecision(
                        StagedSFARouteAction.STAGED,
                        StagedSFARouteReason.ELIGIBLE,
                        graph_key,
                        (131_584,) * actual_requests,
                    ),
                    staged_sfa_graph_dummy_run=False,
                    cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                )

                with (
                    patch.object(
                        sfa_v1,
                        "get_forward_context",
                        return_value=context,
                    ),
                    patch.object(
                        sfa_v1,
                        "_prepare_sfa_remap_boundary",
                        return_value=metadata.decode_remap_boundary,
                    ),
                ):
                    impl.cross_layer_graph_pre(
                        "layer-0",
                        torch.empty(token_capacity, 4),
                        self._make_eligible_kv_cache(
                            num_blocks=request_capacity
                        ),
                        metadata,
                        False,
                        torch.empty(token_capacity, 4),
                    )

                args = impl._cross_layer_pre_compute.call_args.args
                self.assertEqual(
                    torch.count_nonzero(args[6] < 0).item(),
                    padding_tokens,
                )
                self.assertEqual(
                    torch.count_nonzero(args[7] < 0).item(),
                    padding_tokens,
                )

    def test_bridge_storage_is_preallocated_and_reused_for_q1(self):
        impl = self._make_eligible_impl()
        hidden_states = torch.empty(1, 4)
        outputs = self._make_pre_outputs()
        context = SimpleNamespace(
            cudagraph_runtime_mode=CUDAGraphMode.NONE
        )
        with patch.object(
            sfa_v1,
            "get_forward_context",
            return_value=context,
        ):
            first = impl._copy_to_staged_sfa_bridge(
                hidden_states,
                outputs,
            )
            first_addresses = tuple(tensor.data_ptr() for tensor in first)
            second = impl._copy_to_staged_sfa_bridge(
                hidden_states,
                outputs,
            )
        self.assertEqual(
            tuple(tensor.data_ptr() for tensor in second),
            first_addresses,
        )
        self.assertEqual(
            [tensor.shape[0] for tensor in second],
            [4, 4, 4, 4, 4, 4],
        )

    def test_bridge_storage_separates_mtp_token_and_request_capacity(self):
        impl = self._make_eligible_impl()
        impl.decode_threshold = 2
        impl._staged_sfa_graph_capture_sizes = (2, 8)
        hidden_states = torch.empty(4, 4)
        outputs = (
            torch.empty(4, 2, 2),
            torch.empty(4, 2, 2),
            torch.empty(4, 1, 4, dtype=torch.int32),
            torch.empty(2, 8, dtype=torch.int32),
            torch.empty(2, dtype=torch.int32),
            torch.empty(2, 8, dtype=torch.long),
        )
        with patch.object(
            sfa_v1,
            "get_forward_context",
            return_value=SimpleNamespace(
                cudagraph_runtime_mode=CUDAGraphMode.NONE
            ),
        ):
            bridge = impl._copy_to_staged_sfa_bridge(
                hidden_states,
                outputs,
            )
        self.assertEqual(
            [tensor.shape[0] for tensor in bridge],
            [8, 8, 8, 4, 4, 4],
        )
        self.assertEqual(tuple(bridge[3].shape), (4, 8))
        self.assertEqual(tuple(bridge[5].shape), (4, 8))

    def test_capture_state_seals_exact_keys(self):
        state = sfa_v1._StagedSFACaptureState(
            producer_event=object(),
            remap_boundary=torch.empty(1, dtype=torch.int32),
            runtime=("layer-0",),
        )
        key = STAGED_SFA_SINGLETON_GRAPH_KEY
        state.register(
            key,
            tuple(torch.empty(1) for _ in range(6)),
            self._make_eligible_kv_cache(),
        )
        state.seal((key,))

        with self.assertRaisesRegex(RuntimeError, "missing_keys=.*2"):
            state.seal((key, StagedSFAGraphKey.exact_q1(2)))

    def test_capture_state_rejects_binding_drift(self):
        state = sfa_v1._StagedSFACaptureState(
            producer_event=object(),
            remap_boundary=torch.empty(4, dtype=torch.int32),
            runtime=("layer-0",),
        )
        bridge = tuple(torch.empty(4) for _ in range(6))
        first_cache = self._make_eligible_kv_cache()
        state.register(
            STAGED_SFA_SINGLETON_GRAPH_KEY,
            bridge,
            first_cache,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "bindings changed between graph keys",
        ):
            state.register(
                StagedSFAGraphKey.exact_q1(2),
                bridge,
                self._make_eligible_kv_cache(),
            )

    def test_capture_reset_discards_cached_index_tensor(self):
        impl = self._make_eligible_impl()
        old_state = impl._staged_sfa_capture_state
        impl._dsa_idx_cache_t = torch.empty(1)

        impl.reset_staged_sfa_capture()

        self.assertIsNot(impl._staged_sfa_capture_state, old_state)
        self.assertIsNone(impl._dsa_idx_cache_t)

    def test_dummy_cache_initialization_grows_with_capture_key(self):
        impl = self._make_eligible_impl()
        kv_cache = tuple(torch.ones_like(cache) for cache in self._make_eligible_kv_cache(num_blocks=4))
        impl._staged_sfa_capture_state.producer_event = MagicMock()
        impl._cross_layer_kv_cache = MagicMock(return_value=(kv_cache, "index-0", True))
        impl._cross_layer_ineligible_reason = MagicMock(return_value=None)
        impl._cross_layer_pre_compute = MagicMock(
            return_value=self._make_pre_outputs()
        )

        def run(batch_size: int) -> None:
            key = StagedSFAGraphKey.exact_q1(batch_size)
            metadata = self._make_decode_metadata(batch_size)
            context = SimpleNamespace(
                staged_sfa_graph_key=key,
                staged_sfa_route=StagedSFARouteDecision(
                    StagedSFARouteAction.STAGED,
                    StagedSFARouteReason.ELIGIBLE,
                    key,
                ),
                staged_sfa_graph_dummy_run=True,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            with (
                patch.object(sfa_v1, "get_forward_context", return_value=context),
                patch.object(
                    sfa_v1,
                    "_prepare_sfa_remap_boundary",
                    return_value=metadata.decode_remap_boundary,
                ),
            ):
                impl.cross_layer_graph_pre(
                    "layer-0",
                    torch.empty(batch_size, 4),
                    kv_cache,
                    metadata,
                    False,
                    torch.empty(batch_size, 4),
                )

        run(2)
        self.assertTrue(all(torch.count_nonzero(cache[:2]) == 0 for cache in kv_cache))
        self.assertTrue(all(torch.count_nonzero(cache[2:]) > 0 for cache in kv_cache))
        run(4)
        self.assertTrue(all(torch.count_nonzero(cache) == 0 for cache in kv_cache))
        self.assertEqual(
            impl._staged_sfa_capture_state.initialized_cache_capacity,
            4,
        )

    def test_cross_layer_retrieve_prefetches_next_index(self):
        impl = self._make_eligible_impl()
        graph_key = StagedSFAGraphKey.exact_q1(4)
        impl._cross_layer_kv_cache = MagicMock(return_value=(self._make_eligible_kv_cache(), "index-0", True))
        impl._staged_sfa_capture_state.producer_event = object()
        impl._staged_sfa_capture_state.runtime = (None, None, None, True)
        metadata = self._make_decode_metadata()
        next_metadata = self._make_decode_metadata()
        context = SimpleNamespace(
            staged_sfa_graph_key=graph_key,
            staged_sfa_route=StagedSFARouteDecision(
                StagedSFARouteAction.STAGED,
                StagedSFARouteReason.ELIGIBLE,
                graph_key,
                (4096,),
            ),
            staged_sfa_graph_dummy_run=False,
            attn_metadata={"layer-1.attn": next_metadata},
        )
        waits = []
        with (
            patch.object(
                sfa_v1,
                "wait_for_kv_layer_from_connector",
                side_effect=lambda name, *args, **kwargs: waits.append(name),
            ) as wait_for_layer,
            patch.object(sfa_v1, "_sync_compute_stream_after_lmcache_sparse_wait"),
            patch.object(sfa_v1, "_prepare_sfa_remap_boundary") as prepare_boundary,
        ):
            impl.cross_layer_lmcache_retrieve(
                "layer-0",
                "layer-1.attn",
                torch.ones(4, 4, dtype=torch.int32),
                torch.ones(4, dtype=torch.int32),
                torch.zeros(4, 4, dtype=torch.long),
                metadata,
                context,
            )

        self.assertEqual(waits, ["layer-0", "layer-1.indexer.k_cache"])
        self.assertIs(
            metadata.reshape_cache_event,
            impl._staged_sfa_capture_state.producer_event,
        )
        self.assertIs(
            wait_for_layer.call_args_list[0].kwargs["payload_event"],
            impl._staged_sfa_capture_state.producer_event,
        )
        self.assertNotIn(
            "require_complete_sparse_load",
            wait_for_layer.call_args_list[0].kwargs,
        )
        self.assertEqual(
            wait_for_layer.call_args_list[0]
            .kwargs["selected_token_counts"]
            .tolist(),
            [1],
        )
        prepare_boundary.assert_called_once_with(
            next_metadata,
            next_metadata.req_ids,
            is_dummy_run=False,
            index_topk=impl.index_topk,
            cached_tokens=(4096,),
        )

    def test_cross_layer_dummy_retrieve_only_prepares_next_boundary(self):
        impl = self._make_eligible_impl()
        metadata = self._make_decode_metadata()
        next_metadata = self._make_decode_metadata()
        context = SimpleNamespace(
            staged_sfa_graph_key=StagedSFAGraphKey.exact_q1(4),
            staged_sfa_graph_dummy_run=True,
            attn_metadata={"layer-1.attn": next_metadata},
        )
        with (
            patch.object(sfa_v1, "_prepare_sfa_remap_boundary") as prepare_boundary,
            patch.object(sfa_v1, "wait_for_kv_layer_from_connector") as wait_for_layer,
        ):
            impl.cross_layer_lmcache_retrieve(
                "layer-0",
                "layer-1.attn",
                torch.ones(4, 4, dtype=torch.int32),
                torch.ones(4, dtype=torch.int32),
                torch.zeros(4, 4, dtype=torch.long),
                metadata,
                context,
            )

        prepare_boundary.assert_called_once_with(
            next_metadata,
            next_metadata.req_ids,
            is_dummy_run=True,
            index_topk=impl.index_topk,
        )
        wait_for_layer.assert_not_called()

    def test_cross_layer_post_ignores_padded_bridge_rows(self):
        impl = self._make_eligible_impl()
        kv_cache = self._make_eligible_kv_cache()
        impl._cross_layer_kv_cache = MagicMock(return_value=(kv_cache, "index-0", True))
        impl._cross_layer_post_compute = MagicMock()
        context = SimpleNamespace(
            staged_sfa_graph_key=STAGED_SFA_SINGLETON_GRAPH_KEY,
        )

        with patch.object(sfa_v1, "get_forward_context", return_value=context):
            impl.cross_layer_graph_post(
                "layer-0",
                torch.empty(4, 2, 4),
                torch.empty(4, 2, 2),
                torch.empty(4, 1, 16, dtype=torch.int32),
                kv_cache,
                self._make_decode_metadata(),
                torch.empty(1, 4),
            )

        args = impl._cross_layer_post_compute.call_args.args
        self.assertEqual([tensor.shape[0] for tensor in args[:3]], [1] * 3)

    def test_cross_layer_bootstrap_prepares_boundary_before_index_wait(self):
        impl = self._make_eligible_impl()
        impl._staged_sfa_capture_state.runtime = (
            None,
            None,
            "layer-0.indexer.k_cache",
            True,
        )
        metadata = self._make_decode_metadata()
        context = SimpleNamespace(
            attn_metadata={"layer-0.attn": metadata},
            staged_sfa_route=_staged_route(),
            staged_sfa_graph_key=STAGED_SFA_SINGLETON_GRAPH_KEY,
        )
        order = []
        with (
            patch.object(sfa_v1, "get_forward_context", return_value=context),
            patch.object(
                sfa_v1,
                "_prepare_sfa_remap_boundary",
                side_effect=lambda *args, **kwargs: order.append("boundary"),
            ),
            patch.object(
                sfa_v1,
                "wait_for_kv_layer_from_connector",
                side_effect=lambda *args, **kwargs: order.append("index"),
            ) as wait_for_layer,
        ):
            impl.bootstrap_cross_layer("layer-0.attn")

        self.assertEqual(order, ["boundary", "index"])
        wait_for_layer.assert_called_once_with("layer-0.indexer.k_cache")

    def test_cross_layer_dummy_bootstrap_skips_index_wait(self):
        impl = self._make_eligible_impl()
        metadata = self._make_decode_metadata()
        context = SimpleNamespace(
            attn_metadata={"layer-0.attn": metadata},
            staged_sfa_graph_dummy_run=True,
        )
        with (
            patch.object(sfa_v1, "get_forward_context", return_value=context),
            patch.object(sfa_v1, "_prepare_sfa_remap_boundary") as prepare_boundary,
            patch.object(sfa_v1, "wait_for_kv_layer_from_connector") as wait_for_layer,
        ):
            impl.bootstrap_cross_layer("layer-0.attn")

        prepare_boundary.assert_called_once_with(
            metadata,
            metadata.req_ids,
            is_dummy_run=True,
            index_topk=impl.index_topk,
            cached_tokens=None,
        )
        wait_for_layer.assert_not_called()

    def test_remap_boundary_is_resolved_once_per_step(self):
        metadata = self._make_decode_metadata()
        metadata.prompt_lens_cpu_rows = [1000]
        metadata.seq_lens_cpu = torch.tensor([1025])
        original_address = metadata.decode_remap_boundary.data_ptr()

        with (
            patch.object(
                sfa_v1,
                "_decode_window_save_window_size",
                return_value=256,
            ),
            patch.object(sfa_v1, "get_lmcache_sparse_cached_tokens") as lookup,
        ):
            first = sfa_v1._prepare_sfa_remap_boundary(
                metadata,
                ["req-0"],
                is_dummy_run=False,
                index_topk=4,
                cached_tokens=(900,),
            )
            second = sfa_v1._prepare_sfa_remap_boundary(
                metadata,
                ["req-0"],
                is_dummy_run=False,
                index_topk=4,
                cached_tokens=(900,),
            )

        self.assertIs(first, second)
        self.assertEqual(first.data_ptr(), original_address)
        self.assertEqual(first.tolist(), [900])
        lookup.assert_not_called()

    def test_remap_boundary_ignores_dp_padding_rows(self):
        metadata = self._make_decode_metadata(batch_size=2)
        metadata.prompt_lens_cpu_rows = [1000, 0]
        metadata.decode_req_indices_cpu = [0, -1]
        metadata.seq_lens_cpu = torch.tensor([1025, 0])

        with patch.object(
            sfa_v1,
            "_decode_window_save_window_size",
            return_value=256,
        ):
            boundary = sfa_v1._prepare_sfa_remap_boundary(
                metadata,
                ["req-0"],
                is_dummy_run=False,
                index_topk=4,
                cached_tokens=(900,),
            )

        self.assertEqual(boundary.tolist(), [900, 0])

    def test_dummy_remap_boundary_ignores_empty_route_frontiers(self):
        metadata = self._make_decode_metadata()

        boundary = sfa_v1._prepare_sfa_remap_boundary(
            metadata,
            ["req-0"],
            is_dummy_run=True,
            index_topk=4,
            cached_tokens=(),
        )

        self.assertEqual(boundary.tolist(), [8])

    def test_native_remap_boundary_retains_connector_frontier_lookup(self):
        metadata = self._make_decode_metadata()
        metadata.prompt_lens_cpu_rows = [1000]
        metadata.seq_lens_cpu = torch.tensor([1025])

        with (
            patch.object(
                sfa_v1,
                "_decode_window_save_window_size",
                return_value=256,
            ),
            patch.object(
                sfa_v1,
                "get_lmcache_sparse_cached_tokens",
                return_value=[900],
            ) as lookup,
        ):
            boundary = sfa_v1._prepare_sfa_remap_boundary(
                metadata,
                ["req-0"],
                is_dummy_run=False,
                index_topk=4,
            )

        self.assertEqual(boundary.tolist(), [900])
        lookup.assert_called_once_with(["req-0"])

    def test_native_mixed_remap_looks_up_decode_requests_only(self):
        metadata = self._make_decode_metadata(batch_size=2)
        metadata.prompt_lens_cpu_rows = [100, 0]
        metadata.decode_req_indices_cpu = [0, -1]
        metadata.seq_lens_cpu = torch.tensor([110, 6400])

        with (
            patch.object(
                sfa_v1,
                "_decode_window_save_window_size",
                return_value=0,
            ),
            patch.object(
                sfa_v1,
                "get_lmcache_sparse_cached_tokens",
                return_value=[90],
            ) as lookup,
        ):
            boundary = sfa_v1._prepare_sfa_remap_boundary(
                metadata,
                ["decode-req", "prefill-req"],
                is_dummy_run=False,
                index_topk=4,
            )

        self.assertEqual(boundary.tolist(), [90, 0])
        lookup.assert_called_once_with(["decode-req"])

    def test_native_frontiers_are_aligned_after_filtering_prefill(self):
        metadata = self._make_decode_metadata(batch_size=2)
        metadata.decode_req_indices_cpu = [0, -1]

        with patch.object(
            sfa_v1,
            "get_lmcache_sparse_cached_tokens",
            return_value=[90],
        ) as lookup:
            cached_tokens = sfa_v1._resolve_sparse_cached_tokens_by_request(
                metadata,
                ["decode-req", "prefill-req"],
            )

        self.assertEqual(cached_tokens, [90, 0])
        lookup.assert_called_once_with(["decode-req"])

    def test_remap_boundary_uses_unique_request_ids_for_mtp_rows(self):
        metadata = self._make_decode_metadata()
        metadata.prompt_lens_cpu_rows = [100, 100, 200, 200]
        metadata.decode_req_indices_cpu = [0, 0, 1, 1]
        metadata.seq_lens_cpu = torch.tensor([110, 210])
        metadata.decode_scratch_base_cpu = [0, 4, 0, 4]
        metadata.decode_scratch_capacity = 8
        metadata.decode_remap_boundary = torch.empty(4, dtype=torch.int32)
        metadata.decode_remap_boundary_ready = False

        with patch.object(
            sfa_v1,
            "_decode_window_save_window_size",
            return_value=0,
        ):
            boundary = sfa_v1._prepare_sfa_remap_boundary(
                metadata,
                ["req-0", "req-1"],
                is_dummy_run=False,
                index_topk=4,
                cached_tokens=(90, 180),
            )

        self.assertEqual(boundary.tolist(), [90, 90, 180, 180])

    def test_remap_boundary_rejects_scratch_live_alias(self):
        metadata = self._make_decode_metadata()
        metadata.prompt_lens_cpu_rows = [100, 100]
        metadata.decode_req_indices_cpu = [0, 0]
        metadata.seq_lens_cpu = torch.tensor([110])
        metadata.decode_scratch_base_cpu = [0, 4]
        metadata.decode_scratch_capacity = 8
        metadata.decode_remap_boundary = torch.empty(2, dtype=torch.int32)
        metadata.decode_remap_boundary_ready = False

        with (
            patch.object(
                sfa_v1,
                "_decode_window_save_window_size",
                return_value=0,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "would alias live KV positions",
            ),
        ):
            sfa_v1._prepare_sfa_remap_boundary(
                metadata,
                ["req-0"],
                is_dummy_run=False,
                index_topk=4,
                cached_tokens=(7,),
            )

        self.assertFalse(metadata.decode_remap_boundary_ready)

    def test_sparse_lmcache_payload_preserves_duplicate_request_rows(self):
        metadata = self._make_decode_metadata()
        selected = torch.arange(12, dtype=torch.int32).view(3, 4)
        targets = torch.arange(12, dtype=torch.int64).view(3, 4) + 32
        metadata.decode_request_ids_compact = ["req-0", "req-0", "req-1"]
        metadata.decode_valid_row_indices = torch.arange(3, dtype=torch.int32)
        metadata.decode_scratch_base_compact = torch.tensor([0, 4, 0])
        metadata.decode_target_slot_mapping = targets

        payload = sfa_v1._prepare_dsa_sparse_lmcache_payload(
            metadata,
            selected,
            index_topk=4,
        )

        self.assertIs(payload[0], selected)
        self.assertEqual(payload[1], ["req-0", "req-0", "req-1"])
        self.assertIs(payload[1], metadata.decode_request_ids_compact)
        self.assertIs(payload[2], targets)

    def test_staged_sparse_payload_validates_once_per_shared_metadata(self):
        metadata = self._make_decode_metadata()
        metadata.staged_sfa_payload_validated = False
        selected = torch.arange(4, dtype=torch.int32).view(1, 4)

        sfa_v1._prepare_dsa_sparse_lmcache_payload(
            metadata,
            selected,
            index_topk=4,
            validate_once=True,
        )
        metadata.decode_valid_row_indices = None
        payload = sfa_v1._prepare_dsa_sparse_lmcache_payload(
            metadata,
            selected,
            index_topk=4,
            validate_once=True,
        )

        self.assertTrue(metadata.staged_sfa_payload_validated)
        self.assertIs(payload[0], selected)

    def test_eligibility_accepts_single_native_piecewise_decode(self):
        impl = self._make_eligible_impl()
        metadata = self._make_decode_metadata()
        forward_context = MagicMock()
        forward_context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
        forward_context.capturing = False
        forward_context.staged_sfa_graph_dummy_run = False
        forward_context.staged_sfa_graph_key = STAGED_SFA_SINGLETON_GRAPH_KEY
        forward_context.batch_descriptor = BatchDescriptor(
            num_tokens=1,
            num_reqs=None,
            uniform=False,
        )
        forward_context.dsa_offload_manager = None
        forward_context.dsa_adapter_cache = None

        with (
            patch.object(
                sfa_v1,
                "get_forward_context",
                return_value=forward_context,
            ),
            patch.object(
                sfa_v1,
                "get_weight_prefetch_method",
                return_value=None,
            ),
            patch.object(
                sfa_v1.envs,
                "VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY",
                False,
            ),
        ):
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(dtype=dtype):
                    reason = impl._cross_layer_ineligible_reason(
                        torch.empty(1, 4, dtype=dtype),
                        self._make_eligible_kv_cache(dtype=dtype),
                        metadata,
                    )
                    self.assertIsNone(reason)
            metadata.prompt_lens_cpu_rows = [1]
            reason = impl._cross_layer_ineligible_reason(
                torch.empty(1, 4, dtype=torch.bfloat16),
                self._make_eligible_kv_cache(dtype=torch.bfloat16),
                metadata,
            )
            self.assertIsNone(reason)
            metadata.need_sparse_lmcache_payload = False
            reason = impl._cross_layer_ineligible_reason(
                torch.empty(1, 4, dtype=torch.bfloat16),
                self._make_eligible_kv_cache(dtype=torch.bfloat16),
                metadata,
            )
            self.assertEqual(
                reason,
                "the v1 sparse LMCache payload path is unavailable",
            )
            metadata.need_sparse_lmcache_payload = True
            with patch.object(
                sfa_v1,
                "staged_sfa_connector_supports_sparse_load",
                return_value=False,
            ):
                reason = impl._cross_layer_ineligible_reason(
                    torch.empty(1, 4, dtype=torch.bfloat16),
                    self._make_eligible_kv_cache(dtype=torch.bfloat16),
                    metadata,
                )
            self.assertEqual(
                reason,
                "the active connector does not support staged sparse "
                "selective loads",
            )

    def _eligibility_context(self):
        forward_context = MagicMock()
        forward_context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
        forward_context.capturing = False
        forward_context.staged_sfa_graph_dummy_run = False
        forward_context.staged_sfa_graph_key = STAGED_SFA_SINGLETON_GRAPH_KEY
        forward_context.batch_descriptor = BatchDescriptor(
            num_tokens=1,
            num_reqs=None,
            uniform=False,
        )
        forward_context.dsa_offload_manager = None
        forward_context.dsa_adapter_cache = None
        return forward_context

    def test_shared_consumer_eligible_with_two_latent_planes(self):
        impl = self._make_eligible_impl()
        impl.has_indexer = False
        impl.skip_topk = True
        impl.topk_indices_buffer = torch.empty(8, 4, dtype=torch.int32)
        metadata = self._make_decode_metadata()
        kv_cache = self._make_eligible_kv_cache(dtype=torch.bfloat16)

        with (
            patch.object(
                sfa_v1,
                "get_forward_context",
                return_value=self._eligibility_context(),
            ),
            patch.object(
                sfa_v1,
                "get_weight_prefetch_method",
                return_value=None,
            ),
            patch.object(
                sfa_v1.envs,
                "VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY",
                False,
            ),
        ):
            reason = impl._cross_layer_ineligible_reason(
                torch.empty(1, 4, dtype=torch.bfloat16),
                kv_cache,
                metadata,
            )
        self.assertIsNone(reason)

        # A producer-style three-plane tuple is not the consumer contract.
        reason = impl._cross_layer_ineligible_reason(
            torch.empty(1, 4, dtype=torch.bfloat16),
            (*kv_cache, kv_cache[0]),
            metadata,
        )
        self.assertIsNotNone(reason)

    def test_shared_consumer_requires_shared_topk_buffer(self):
        impl = self._make_eligible_impl()
        impl.has_indexer = False
        impl.skip_topk = True
        impl.topk_indices_buffer = None
        metadata = self._make_decode_metadata()

        with (
            patch.object(
                sfa_v1,
                "get_forward_context",
                return_value=self._eligibility_context(),
            ),
            patch.object(
                sfa_v1,
                "get_weight_prefetch_method",
                return_value=None,
            ),
            patch.object(
                sfa_v1.envs,
                "VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY",
                False,
            ),
        ):
            reason = impl._cross_layer_ineligible_reason(
                torch.empty(1, 4, dtype=torch.bfloat16),
                self._make_eligible_kv_cache(dtype=torch.bfloat16),
                metadata,
            )
        self.assertEqual(
            reason,
            "the shared-consumer staged graph requires the shared top-k buffer",
        )

    def test_producer_requires_indexer_plane(self):
        impl = self._make_eligible_impl()
        metadata = self._make_decode_metadata()
        latent_only = self._make_eligible_kv_cache(dtype=torch.bfloat16)[:2]

        with (
            patch.object(
                sfa_v1,
                "get_forward_context",
                return_value=self._eligibility_context(),
            ),
            patch.object(
                sfa_v1,
                "get_weight_prefetch_method",
                return_value=None,
            ),
            patch.object(
                sfa_v1.envs,
                "VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY",
                False,
            ),
        ):
            reason = impl._cross_layer_ineligible_reason(
                torch.empty(1, 4, dtype=torch.bfloat16),
                latent_only,
                metadata,
            )
        self.assertIsNotNone(reason)

    def test_capture_binding_seals_shared_topk_buffer(self):
        impl = self._make_eligible_impl()
        impl.index_cache_enabled = True
        state = impl._staged_sfa_capture_state
        state.producer_event = object()
        state.remap_boundary = torch.empty(1, dtype=torch.int32)
        bridge = self._make_pre_outputs()
        kv_cache = self._make_eligible_kv_cache()
        key_a = MagicMock()
        key_b = MagicMock()

        buffer_one = torch.empty(8, 4, dtype=torch.int32)
        buffer_two = torch.empty(8, 4, dtype=torch.int32)
        state.register(key_a, bridge, kv_cache, topk_buffer=buffer_one)
        with self.assertRaisesRegex(RuntimeError, "capture bindings changed between graph keys"):
            state.register(key_b, bridge, kv_cache, topk_buffer=buffer_two)
        # The same buffer keeps the contract stable.
        state.register(key_b, bridge, kv_cache, topk_buffer=buffer_one)
        # And omitting it after sealing one is also a change.
        key_c = MagicMock()
        with self.assertRaisesRegex(RuntimeError, "capture bindings changed between graph keys"):
            state.register(key_c, bridge, kv_cache, topk_buffer=None)

    def test_indexcache_buffer_roundtrip_helpers(self):
        impl = self._make_eligible_impl()
        impl.dsa_shrink_latent = 0
        impl.topk_indices_buffer = torch.zeros(8, 4, dtype=torch.int32)
        impl._indexcache_topk_staging = None

        produced = torch.arange(8, dtype=torch.int32).view(8, 1, 1).expand(8, 1, 4)
        impl._update_indexcache_topk_indices(produced.contiguous())
        read_back = impl._get_indexcache_topk_indices(8)
        self.assertEqual(read_back.shape, (8, 1, 4))
        self.assertTrue(torch.equal(read_back, produced))
        self.assertEqual(read_back.data_ptr(), impl.topk_indices_buffer.data_ptr())

        # 2-D buffer rows are unsqueezed to the 3-D kernel layout.
        impl.topk_indices_buffer = torch.zeros(8, 4, dtype=torch.int32)
        produced_2d = torch.ones(8, 4, dtype=torch.int32)
        impl._update_indexcache_topk_indices(produced_2d)
        self.assertEqual(impl._get_indexcache_topk_indices(8).shape, (8, 1, 4))

        # No buffer configured: update is a no-op, read fails closed.
        impl.topk_indices_buffer = None
        impl._update_indexcache_topk_indices(produced_2d)
        with self.assertRaisesRegex(RuntimeError, "topk_indices_buffer"):
            impl._get_indexcache_topk_indices(8)

    def test_shared_consumer_native_uses_local_hidden_state_rows(self):
        local_rows = 512
        padded_rows = 4096
        impl = self._make_eligible_impl()
        impl.has_indexer = False
        impl.skip_topk = True
        impl.topk_indices_buffer = torch.zeros(padded_rows, 4, dtype=torch.int32)
        impl.dsa_offload_unbundle = False
        impl.is_kv_producer = False
        impl.dsa_shrink_latent = 0
        impl.enable_dsa_cp = True
        impl.enable_dsa_cp_strict_accuracy = False

        impl.fused_qkv_a_proj = MagicMock()
        impl.fused_qkv_a_proj.weight = torch.empty(1)
        impl.fused_qkv_a_proj.return_value = (
            torch.empty(local_rows, impl.q_lora_rank + impl.kv_lora_rank + impl.qk_rope_head_dim),
        )
        impl.q_a_layernorm = MagicMock(side_effect=lambda value: value)
        impl.exec_kv = MagicMock(
            return_value=(
                torch.empty(local_rows, 1, impl.qk_rope_head_dim),
                torch.empty(local_rows, 1, impl.kv_lora_rank),
            )
        )
        impl._q_proj_and_k_up_proj = MagicMock(
            return_value=(
                torch.empty(local_rows, 1, 2),
                torch.empty(local_rows, 1, impl.qk_rope_head_dim),
            )
        )
        impl.rope_single = MagicMock(side_effect=lambda value, cos, sin: value)
        impl._execute_sparse_flash_attention_process = MagicMock(
            return_value=torch.empty(local_rows, 1, 2)
        )
        impl._v_up_proj = MagicMock(side_effect=lambda value: value)
        impl.o_proj = MagicMock(return_value=(torch.empty(local_rows, 4),))
        impl.o_proj.weight = torch.empty(1)
        impl._submit_sfa_save_operations = MagicMock()

        metadata = SimpleNamespace(
            cos=torch.ones(local_rows, 2),
            sin=torch.zeros(local_rows, 2),
            slot_mapping=torch.arange(local_rows),
            indexer_slot_mapping=None,
            cum_query_lens=torch.tensor([0, local_rows]),
            seq_lens=torch.tensor([local_rows]),
            num_input_tokens=padded_rows,
            num_actual_tokens=local_rows,
            num_decode_tokens=0,
            attn_state=AscendAttentionState.ChunkedPrefill,
            split_boundary=None,
            dsa_cp_context=SimpleNamespace(
                slot_mapping_cp=torch.arange(local_rows),
                actual_seq_lengths_query=torch.tensor([0, local_rows]),
                actual_seq_lengths_key=torch.tensor([local_rows]),
            ),
        )
        hidden_states = torch.empty(local_rows, 4)
        output = torch.empty_like(hidden_states)
        kv_cache = (
            torch.empty(1, 128, 1, impl.kv_lora_rank),
            torch.empty(1, 128, 1, impl.qk_rope_head_dim),
        )
        context = SimpleNamespace(dsa_offload_manager=None, dsa_adapter_cache=None)
        prefetch = MagicMock()

        with (
            patch.object(sfa_v1, "get_forward_context", return_value=context),
            patch.object(sfa_v1, "get_weight_prefetch_method", return_value=prefetch),
            patch.object(sfa_v1, "wait_for_kv_layer_from_connector"),
            patch.object(sfa_v1, "get_tp_group", return_value=MagicMock()),
            patch.object(
                sfa_v1,
                "all_gather_async",
                side_effect=lambda tensor, group, **kwargs: (tensor, None),
            ),
            patch.object(sfa_v1.DeviceOperator, "reshape_and_cache"),
        ):
            impl.forward("model.layers.1.self_attn.attn", hidden_states, kv_cache, metadata, output=output)

        sparse_indices = impl._execute_sparse_flash_attention_process.call_args.args[3]
        self.assertEqual(sparse_indices.shape, (local_rows, 1, 4))

    def test_forward_remap_diagnostic_sampling(self):
        impl = self._make_eligible_impl()
        impl.has_indexer = False
        impl.skip_topk = True
        impl.topk_indices_buffer = torch.tensor([[1, 7, 25, 26]], dtype=torch.int32)
        impl._indexcache_topk_staging = torch.empty_like(impl.topk_indices_buffer)
        impl.dsa_offload_unbundle = False
        impl.is_kv_producer = False
        impl.enable_dsa_cp_strict_accuracy = False
        # Stub device computation, but execute forward's remap and diagnostics.
        impl.enable_mlapo = True
        hidden_states = torch.zeros(1, 4)
        ql_nope, q_pe, *_ = self._make_pre_outputs()
        impl._sfa_preprocess_with_mlapo = MagicMock(
            return_value=(hidden_states, ql_nope, q_pe, hidden_states)
        )
        remapped = torch.tensor([[[0, 1, 25, 26]]], dtype=torch.int32)
        selected = torch.tensor([[1, 7, -1, -1]], dtype=torch.int32)
        counts = torch.tensor([2], dtype=torch.int32)
        targets = torch.tensor([[0, 1, -1, -1]], dtype=torch.long)
        impl._prepare_decode_sparse_indices = MagicMock(return_value=(remapped, selected, counts, targets))
        impl._execute_sparse_flash_attention_process = MagicMock(return_value=hidden_states)
        impl._v_up_proj = MagicMock(side_effect=lambda value: value)
        impl.o_proj = MagicMock(return_value=(torch.ones_like(hidden_states),))
        impl._submit_sfa_save_operations = MagicMock()

        metadata = self._make_decode_metadata()
        metadata.layerwise_prefill_callback_metadata = ()
        metadata.split_boundary = torch.empty(1, dtype=torch.int32)
        metadata.decode_split_boundary_cpu_tensor = torch.empty_like(metadata.split_boundary)
        metadata.decode_split_boundary_cpu = metadata.decode_split_boundary_cpu_tensor.numpy()
        context = SimpleNamespace(
            dsa_offload_manager=None,
            dsa_adapter_cache=None,
            dsa_req_ids=["req-0"],
            mtp_dw_diag_req_ids={"req-0"},
        )
        kv_cache = self._make_eligible_kv_cache()

        # position, prompt length, post-commit, startup/post-commit sample, boundary sample
        cases = (
            (27, 24, False, False, False),
            (28, 24, False, False, True),
            (31, 24, False, False, True),
            (32, 24, False, False, True),
            (33, 24, False, False, True),
            (36, 24, False, False, True),
            (37, 24, False, False, False),
            (40, 40, False, True, False),
            (41, 40, False, True, False),
            (42, 40, False, True, False),
            (43, 40, False, False, False),
            (40, 24, False, False, False),
            (40, 24, True, True, False),
        )
        for diag_enabled, window in ((False, 0), (False, 16), (True, 0), (True, 16)):
            for position, prompt_len, post_commit, startup_or_commit, near_boundary in cases:
                with (
                    self.subTest(
                        diag=diag_enabled,
                        window=window,
                        position=position,
                        prompt=prompt_len,
                        post_commit=post_commit,
                    ),
                    patch.object(sfa_v1.envs, "VLLM_ASCEND_MTP_DW_DIAG", diag_enabled),
                    patch.object(sfa_v1.envs, "VLLM_ASCEND_MTP_DW_DEEP_DIAG", False),
                    patch.object(sfa_v1, "_decode_window_save_window_size", return_value=window),
                    patch.object(sfa_v1, "get_forward_context", return_value=context),
                    patch.object(sfa_v1, "get_weight_prefetch_method", return_value=MagicMock()),
                    patch.object(sfa_v1, "get_lmcache_sparse_cached_tokens", return_value=[24]),
                    patch.object(sfa_v1, "wait_for_kv_layer_from_connector") as wait,
                    patch.object(sfa_v1, "_LMCACHE_SPARSE_WAIT_SYNC_ONCE", False),
                    patch.object(sfa_v1, "_mtp_dw_event") as event,
                ):
                    metadata.decode_split_boundary = None
                    metadata.seq_lens.fill_(position + 1)
                    metadata.seq_lens_cpu.fill_(position + 1)
                    context.dsa_prompt_lens = torch.tensor([prompt_len])
                    context.mtp_dw_diag_post_commit_req_ids = {"req-0"} if post_commit else None
                    output = torch.empty_like(hidden_states)

                    result = impl.forward(impl.layer_name, hidden_states, kv_cache, metadata, output=output)

                    self.assertIs(result, output)
                    self.assertTrue(torch.equal(output, torch.ones_like(output)))
                    self.assertEqual(len(impl._submit_sfa_save_operations.call_args.args[0]), int(window > 0))
                    boundary = 16 if window == 16 and position < 32 else 24
                    self.assertEqual(metadata.decode_split_boundary.tolist(), [boundary])
                    self.assertIs(impl._prepare_decode_sparse_indices.call_args.args[1], metadata.split_boundary)
                    self.assertIs(impl._execute_sparse_flash_attention_process.call_args.args[3], remapped)
                    wait.assert_called_once_with(
                        impl.layer_name,
                        selected_tokens=selected,
                        target_slot_mapping=targets,
                        request_ids=["req-0"],
                        selected_token_counts=counts,
                    )
                    sample = diag_enabled and (startup_or_commit or (window > 0 and near_boundary))
                    if sample:
                        event.assert_called_once()
                        self.assertEqual(event.call_args.args, ("remap",))
                        fields = event.call_args.kwargs
                        self.assertEqual(fields["current_position"], position)
                        self.assertEqual(fields["window_start"], position // 16 * 16 if window else None)
                        self.assertEqual(fields["committed_end"], 24)
                        self.assertEqual(fields["remap_boundary"], boundary)
                        self.assertEqual(fields["selected_absolute_sample"], [1, 7])
                    else:
                        event.assert_not_called()

    def test_shared_indexer_reuse_map_prefers_preceding_producer(self):
        # The reuse map is derived from indexer_types; consumers must map to
        # the nearest PRECEDING full producer. logger.info_once deduplicates
        # via lru_cache over (msg, *args), so every arg must also be hashable
        # (regression test for the unhashable-list startup crash) and free of
        # per-layer values (otherwise every layer would emit its own copy).
        captured = []

        def fake_info_once(msg, *args, scope="process"):
            captured.append((msg, args))

        indexer_types = ["full", "full", "shared", "shared", "full", "shared"]
        config = SimpleNamespace(indexer_types=indexer_types)
        with patch.object(sfa_v1.logger, "info_once", fake_info_once):
            sfa_v1._log_shared_indexer_reuse_map((config,))

        self.assertEqual(len(captured), 1)
        msg, args = captured[0]
        for arg in args:
            hash(arg)
        formatted = msg % args
        self.assertIn("producers=(0, 1, 4)", formatted)
        self.assertIn("consumers=(2, 3, 5)", formatted)
        self.assertIn("source_producer_per_consumer={2: 1, 3: 1, 5: 4}", formatted)

    def test_eligibility_accepts_exact_multi_request_q1_batch(self):
        batch_size = 4
        impl = self._make_eligible_impl()
        metadata = self._make_decode_metadata(batch_size)
        graph_key = StagedSFAGraphKey.exact_q1(batch_size)
        forward_context = MagicMock(
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            staged_sfa_graph_dummy_run=False,
            staged_sfa_graph_key=graph_key,
            batch_descriptor=graph_key.to_legacy_batch_descriptor(),
            dsa_offload_manager=None,
            dsa_adapter_cache=None,
        )

        with (
            patch.object(sfa_v1, "get_forward_context", return_value=forward_context),
            patch.object(sfa_v1, "get_weight_prefetch_method", return_value=None),
            patch.object(
                sfa_v1.envs,
                "VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY",
                False,
            ),
        ):
            reason = impl._cross_layer_ineligible_reason(
                torch.empty(batch_size, 4, dtype=torch.bfloat16),
                self._make_eligible_kv_cache(
                    dtype=torch.bfloat16,
                    num_blocks=batch_size,
                ),
                metadata,
            )

        self.assertIsNone(reason)

    def test_eligibility_accepts_fixed_width_mtp_batch(self):
        impl = self._make_eligible_impl()
        impl.decode_threshold = 2
        impl.vllm_config.speculative_config = SimpleNamespace(
            num_speculative_tokens=1
        )
        metadata = self._make_decode_metadata(4)
        metadata.attn_state = AscendAttentionState.SpecDecoding
        metadata.cum_query_lens = torch.tensor([2, 4])
        metadata.seq_lens = torch.tensor([9, 9])
        metadata.seq_lens_cpu = torch.tensor([9, 9])
        metadata.block_table = torch.arange(2).view(2, 1)
        metadata.indexer_block_table = torch.arange(2).view(2, 1)
        metadata.decode_req_indices = torch.tensor(
            [0, 0, 1, 1], dtype=torch.int32
        )
        metadata.decode_req_indices_cpu = [0, 0, 1, 1]
        metadata.decode_request_ids_compact = ["req-0", "req-1"]
        metadata.req_ids = ["req-0", "req-1"]
        metadata.decode_selected_tokens = torch.empty(
            2, 8, dtype=torch.int32
        )
        metadata.decode_selected_counts = torch.empty(
            2, 16, dtype=torch.int32
        )
        metadata.decode_target_slot_mapping = torch.empty(
            2, 8, dtype=torch.long
        )
        metadata.decode_union_mapping_workspace = torch.empty(
            2, 8, dtype=torch.int32
        )
        metadata.decode_shard_packed_workspace = torch.empty(
            2, 2, 8, dtype=torch.int32
        )
        metadata.decode_shard_mapping_workspace = torch.empty_like(
            metadata.decode_shard_packed_workspace
        )
        metadata.decode_shard_counts_workspace = torch.empty(
            2, 2, 16, dtype=torch.int32
        )
        metadata.resident_state_indices = torch.arange(
            2,
            dtype=torch.int32,
        )
        metadata.resident_state_generations = torch.ones(
            2,
            dtype=torch.int64,
        )
        graph_key = StagedSFAGraphKey.fixed_spec(2, 2)
        context = SimpleNamespace(
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            staged_sfa_graph_dummy_run=False,
            staged_sfa_graph_key=graph_key,
            batch_descriptor=graph_key.to_legacy_batch_descriptor(),
            dsa_offload_manager=None,
            dsa_adapter_cache=None,
        )
        with (
            patch.object(sfa_v1, "get_forward_context", return_value=context),
            patch.object(
                sfa_v1,
                "get_weight_prefetch_method",
                return_value=None,
            ),
            patch.object(
                sfa_v1.envs,
                "VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY",
                False,
            ),
        ):
            reason = impl._cross_layer_ineligible_reason(
                torch.empty(4, 4, dtype=torch.bfloat16),
                self._make_eligible_kv_cache(
                    dtype=torch.bfloat16,
                    num_blocks=2,
                ),
                metadata,
            )
        self.assertIsNone(reason)

    def test_eligibility_rejects_invalid_cache_contract(self):
        impl = self._make_eligible_impl()
        metadata = self._make_decode_metadata()
        forward_context = MagicMock()
        forward_context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
        forward_context.capturing = False
        forward_context.staged_sfa_graph_dummy_run = False
        forward_context.staged_sfa_graph_key = STAGED_SFA_SINGLETON_GRAPH_KEY
        forward_context.batch_descriptor = BatchDescriptor(
            num_tokens=1,
            num_reqs=None,
            uniform=False,
        )
        forward_context.dsa_offload_manager = None
        forward_context.dsa_adapter_cache = None
        valid = self._make_eligible_kv_cache()
        invalid_contracts = (
            (
                "rank",
                (
                    torch.empty(2, 128, 2, dtype=torch.bfloat16),
                    valid[1],
                    valid[2],
                ),
                "rank-4 PA_BSND",
            ),
            (
                "head_axis",
                (
                    torch.empty(2, 128, 2, 2, dtype=torch.bfloat16),
                    valid[1],
                    valid[2],
                ),
                "one KV head",
            ),
            (
                "hidden_dim",
                (
                    torch.empty(2, 128, 1, 3, dtype=torch.bfloat16),
                    valid[1],
                    valid[2],
                ),
                "hidden dimensions",
            ),
            (
                "different_block_sizes",
                (
                    valid[0],
                    valid[1],
                    torch.empty(2, 64, 1, 2, dtype=torch.bfloat16),
                ),
                "block sizes do not agree",
            ),
            (
                "wrong_configured_block_size",
                self._make_eligible_kv_cache(block_size=64),
                "configured block size",
            ),
            (
                "different_devices",
                (
                    valid[0],
                    valid[1],
                    torch.empty(
                        2,
                        128,
                        1,
                        2,
                        dtype=torch.bfloat16,
                        device="meta",
                    ),
                ),
                "different devices",
            ),
            (
                "different_dtypes",
                (
                    valid[0],
                    valid[1],
                    torch.empty(2, 128, 1, 2, dtype=torch.float16),
                ),
                "share one dtype",
            ),
            (
                "unsupported_dtype",
                self._make_eligible_kv_cache(dtype=torch.float32),
                "must be float16 or bfloat16",
            ),
        )

        with (
            patch.object(
                sfa_v1,
                "get_forward_context",
                return_value=forward_context,
            ),
            patch.object(
                sfa_v1,
                "get_weight_prefetch_method",
                return_value=None,
            ),
            patch.object(
                sfa_v1.envs,
                "VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY",
                False,
            ),
        ):
            for name, kv_cache, expected_reason in invalid_contracts:
                with self.subTest(name=name):
                    reason = impl._cross_layer_ineligible_reason(
                        torch.empty(1, 4, dtype=torch.bfloat16),
                        kv_cache,
                        metadata,
                    )
                    self.assertIn(expected_reason, reason)

    def test_eligibility_rejects_weight_prefetch(self):
        impl = self._make_eligible_impl()
        metadata = self._make_decode_metadata()
        forward_context = MagicMock()
        forward_context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
        forward_context.staged_sfa_graph_dummy_run = False
        forward_context.staged_sfa_graph_key = STAGED_SFA_SINGLETON_GRAPH_KEY
        forward_context.batch_descriptor = BatchDescriptor(
            num_tokens=1,
            num_reqs=None,
            uniform=False,
        )
        forward_context.dsa_offload_manager = None
        forward_context.dsa_adapter_cache = None
        weight_prefetch_method = MagicMock()
        weight_prefetch_method.mla_sfa_prefetch_enable = True

        with (
            patch.object(
                sfa_v1,
                "get_forward_context",
                return_value=forward_context,
            ),
            patch.object(
                sfa_v1,
                "get_weight_prefetch_method",
                return_value=weight_prefetch_method,
            ),
            patch.object(
                sfa_v1.envs,
                "VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY",
                False,
            ),
        ):
            reason = impl._cross_layer_ineligible_reason(
                torch.empty(1, 4),
                self._make_eligible_kv_cache(),
                metadata,
            )

        self.assertEqual(reason, "weight prefetch is enabled")

    def test_eligibility_accepts_explicit_eager_dummy_warmup(self):
        impl = self._make_eligible_impl()
        metadata = self._make_decode_metadata()
        metadata.need_sparse_lmcache_payload = False
        forward_context = MagicMock()
        forward_context.cudagraph_runtime_mode = CUDAGraphMode.NONE
        forward_context.capturing = False
        forward_context.staged_sfa_graph_dummy_run = True
        forward_context.staged_sfa_graph_key = STAGED_SFA_SINGLETON_GRAPH_KEY
        forward_context.batch_descriptor = BatchDescriptor(
            num_tokens=1,
            num_reqs=None,
            uniform=False,
        )
        forward_context.dsa_offload_manager = None
        forward_context.dsa_adapter_cache = None

        with (
            patch.object(
                sfa_v1,
                "get_forward_context",
                return_value=forward_context,
            ),
            patch.object(
                sfa_v1,
                "get_weight_prefetch_method",
                return_value=None,
            ),
            patch.object(
                sfa_v1.envs,
                "VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY",
                False,
            ),
            patch.object(
                sfa_v1,
                "staged_sfa_connector_supports_sparse_load",
                return_value=False,
            ) as connector_support,
        ):
            reason = impl._cross_layer_ineligible_reason(
                torch.empty(1, 4),
                self._make_eligible_kv_cache(),
                metadata,
            )

        self.assertIsNone(reason)
        connector_support.assert_not_called()

    def test_eligibility_accepts_real_rows_within_graph_capacity(self):
        capacity = 4
        impl = self._make_eligible_impl()
        metadata = self._make_decode_metadata(capacity)
        metadata.num_actual_tokens = metadata.num_decode_tokens = 1
        metadata.seq_lens[1:] = 0
        metadata.seq_lens_cpu[1:] = 0
        metadata.prompt_lens_cpu_rows = [8, 0, 0, 0]
        metadata.decode_req_indices[1:] = -1
        metadata.decode_req_indices_cpu = [0, -1, -1, -1]
        metadata.decode_valid_rows_all = False
        metadata.decode_valid_row_indices = torch.tensor(
            [0],
            dtype=torch.int32,
        )
        metadata.decode_request_ids_compact = ["req-0"]
        metadata.req_ids = ["req-0"]
        graph_key = StagedSFAGraphKey.exact_q1(capacity)
        forward_context = SimpleNamespace(
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            staged_sfa_graph_dummy_run=False,
            staged_sfa_graph_key=graph_key,
            batch_descriptor=graph_key.to_legacy_batch_descriptor(),
            dsa_offload_manager=None,
            dsa_adapter_cache=None,
        )

        with (
            patch.object(
                sfa_v1,
                "get_forward_context",
                return_value=forward_context,
            ),
            patch.object(
                sfa_v1,
                "get_weight_prefetch_method",
                return_value=None,
            ),
            patch.object(
                sfa_v1.envs,
                "VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY",
                False,
            ),
        ):
            reason = impl._cross_layer_ineligible_reason(
                torch.empty(capacity, 4, dtype=torch.bfloat16),
                self._make_eligible_kv_cache(
                    dtype=torch.bfloat16,
                    num_blocks=capacity,
                ),
                metadata,
            )

        self.assertIsNone(reason)


@pytest.mark.parametrize(
    "shrink_latent,staged,resident",
    [
        (0, False, False),
        (2, False, False),
        (2, False, True),
        (2, True, False),
        (2, True, True),
    ],
    ids=["no-shrink", "ordinary-eager", "resident-eager", "ordinary-staged", "resident-staged"],
)
@pytest.mark.parametrize("consumer_has_indexer", [False, True], ids=["shared", "runtime-skip"])
@pytest.mark.parametrize("buffer_ndim", [2, 3])
@pytest.mark.parametrize("query_width", [1, 2], ids=["q1", "mtp1"])
def test_indexcache_consumers_preserve_shared_raw_topk(
    shrink_latent, staged, resident, consumer_has_indexer, buffer_ndim, query_width
):
    shared = torch.zeros(4, 4, dtype=torch.int32)
    if buffer_ndim == 3:
        shared = shared.unsqueeze(1)
    hf_config = SimpleNamespace(
        indexer_types=["full", "shared", "shared"],
        index_topk=4,
        index_n_heads=2,
        index_head_dim=2,
        model_type="glm_moe_dsa",
    )
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=128),
        speculative_config=SimpleNamespace(num_speculative_tokens=1) if query_width == 2 else None,
        kv_transfer_config=None,
        model_config=SimpleNamespace(hf_config=hf_config, hf_text_config=hf_config),
        scheduler_config=SimpleNamespace(max_num_seqs=4 // query_width),
    )
    indexer = SimpleNamespace(
        n_head=2,
        head_dim=2,
        topk_tokens=4,
        wq_b=MagicMock(),
        wk=MagicMock(),
        weights_proj=MagicMock(),
        k_norm=MagicMock(),
    )
    layers = []
    with (
        patch.object(sfa_v1, "get_current_vllm_config", return_value=config),
        patch.object(
            sfa_v1,
            "get_ascend_config",
            return_value=SimpleNamespace(
                enable_shared_expert_dp=False,
                enable_sparse_c8=False,
            ),
        ),
        patch.object(sfa_v1, "get_tensor_model_parallel_world_size", return_value=1),
        patch.object(sfa_v1, "get_tp_group", return_value=SimpleNamespace(rank_in_group=0)),
        patch.object(sfa_v1, "enable_dsa_cp", return_value=False),
        patch.object(sfa_v1, "enable_dsa_cp_with_layer_shard", return_value=False),
        patch.object(sfa_v1, "enable_dsa_cp_with_o_proj_tp", return_value=False),
        patch.object(sfa_v1, "staged_sfa_graph_configured", return_value=False),
        patch.object(sfa_v1.envs, "VLLM_ASCEND_DSA_UNBUNDLE", True),
        patch.object(sfa_v1.envs, "VLLM_ASCEND_DSA_SHRINK_LATENT", shrink_latent),
        patch.object(sfa_v1.envs, "VLLM_ASCEND_DSA_RESIDENT_CACHE", resident),
        patch.object(sfa_v1.envs, "VLLM_ASCEND_ENABLE_MLAPO", False),
        # Resident kernels/state are stubbed for the toy top-k width.
        patch.object(sfa_v1, "INDEX_TOPK", 4),
        patch.object(sfa_v1, "allocate_sorted_resident_state"),
        patch.object(sfa_v1, "allocate_sorted_resident_workspace"),
        patch.object(torch, "empty_like", wraps=torch.empty_like) as allocate_topk,
    ):
        for layer_index in range(3):
            impl = AscendSFAImpl(
                num_heads=2,
                head_size=4,
                scale=1.0,
                num_kv_heads=1,
                alibi_slopes=None,
                sliding_window=None,
                kv_cache_dtype="auto",
                logits_soft_cap=None,
                attn_type="decoder",
                kv_sharing_target_layer_name=None,
                q_lora_rank=4,
                kv_lora_rank=2,
                qk_nope_head_dim=2,
                qk_rope_head_dim=2,
                qk_head_dim=4,
                v_head_dim=2,
                rotary_emb=None,
                q_b_proj=MagicMock(),
                kv_b_proj=MagicMock(),
                o_proj=MagicMock(),
                fused_qkv_a_proj=MagicMock(),
                q_a_layernorm=MagicMock(),
                indexer=indexer if layer_index == 0 or consumer_has_indexer else None,
                skip_topk=layer_index != 0,
                topk_indices_buffer=shared,
                layer_name=f"model.layers.{layer_index}.self_attn.attn",
            )
            layers.append(impl)

    # Only shrink-enabled consumers allocate private storage, before serving.
    assert allocate_topk.call_count == (2 if shrink_latent else 0)
    assert layers[0]._indexcache_topk_staging is None
    if shrink_latent:
        staging_ptrs = [impl._indexcache_topk_staging.data_ptr() for impl in layers[1:]]
        assert len(set([shared.data_ptr(), *staging_ptrs])) == 3
        for impl in layers[1:]:
            assert impl._indexcache_topk_staging.shape == shared.shape
            assert impl._indexcache_topk_staging.dtype == shared.dtype
            assert impl._indexcache_topk_staging.device == shared.device
            assert impl._indexcache_topk_staging.is_contiguous()
    else:
        assert all(impl._indexcache_topk_staging is None for impl in layers)
        with patch.object(torch.Tensor, "copy_", side_effect=AssertionError("no-shrink consumer copy")):
            for rows in (4, 2, 4):
                for impl in layers[1:]:
                    assert impl._get_indexcache_topk_indices(rows).data_ptr() == shared.data_ptr()

    hidden = torch.zeros(4, 4)
    qkv = torch.zeros(4, 8)
    q = torch.zeros(4, 2, 2)
    k_li = torch.zeros(4, 1, 2)
    produced = torch.empty(4, 1, 4, dtype=torch.int32)
    kv_caches = [TestStagedSFAGraphPoc._make_eligible_kv_cache(num_blocks=8 // query_width) for _ in layers]
    for impl in layers:
        assert impl.dsa_shrink_latent == shrink_latent
        assert impl.dsa_resident_cache == resident
        assert impl.decode_threshold == query_width
        impl.fused_qkv_a_proj.side_effect = lambda x: (qkv[: x.shape[0]],)
        impl.q_a_layernorm.side_effect = lambda x: x
        impl.exec_kv = MagicMock(return_value=(None, None))
        impl._q_proj_and_k_up_proj = MagicMock(side_effect=lambda x: (q[: x.shape[0]], q[: x.shape[0]]))
        impl.rope_single = MagicMock(side_effect=lambda x, cos, sin: x)
        impl.indexer_select_pre_process = MagicMock(side_effect=lambda **kw: (k_li[: kw["x"].shape[0]], None))
        impl.indexer_select_post_process = MagicMock(side_effect=AssertionError("consumer ran indexer"))
        impl._execute_sparse_flash_attention_process = MagicMock(wraps=impl._execute_sparse_flash_attention_process)
        impl._v_up_proj = MagicMock(side_effect=lambda x: x)
        impl.o_proj.side_effect = lambda x: (x,)
        impl._submit_sfa_save_operations = MagicMock()

    # Both NPU planners overwrite their input. Emulate that contract on CPU,
    # leaving the real producer/consumer paths and planner dispatch intact.
    def mutate_plan(topk, *args, **kwargs):
        assert kwargs["mtp" if resident else "staged_mtp"] == query_width
        assert torch.equal(topk, raw[:rows].unsqueeze(1))
        metadata.decode_selected_tokens.copy_(topk.view(request_count, 4 * query_width))
        topk.fill_(layer_index)
        metadata.decode_selected_counts.fill_(4 * query_width)
        metadata.decode_target_slot_mapping.fill_(layer_index)
        return (
            topk,
            metadata.decode_selected_tokens,
            metadata.decode_selected_counts[:, 0],
            metadata.decode_target_slot_mapping,
        )

    context = SimpleNamespace(dsa_offload_manager=None, dsa_adapter_cache=None)
    with (
        patch.object(sfa_v1, "get_forward_context", return_value=context),
        patch.object(sfa_v1, "get_weight_prefetch_method", return_value=MagicMock()),
        patch.object(sfa_v1.torch_npu, "npu_scatter_nd_update_"),
        patch.object(
            torch.ops._C_ascend,
            "npu_sparse_flash_attention",
            side_effect=lambda **kw: hidden[: kw["query"].shape[0]],
            create=True,
        ),
        patch.object(sfa_v1, "_dsa_mask_padding_sparse_rows", side_effect=AssertionError("unexpected padding rewrite")),
        patch.object(sfa_v1, "_dsa_index_lmcache_enabled", return_value=False),
        patch.object(sfa_v1, "wait_for_kv_layer_from_connector"),
        patch.object(sfa_v1, "_decode_window_save_window_size", return_value=0),
        patch.object(sfa_v1, "_LMCACHE_SPARSE_WAIT_SYNC_ONCE", False),
        patch.object(sfa_v1.envs, "VLLM_ASCEND_MTP_DW_DIAG", False),
        patch.object(sfa_v1.envs, "VLLM_ASCEND_MTP_DRAFT_DEBUG", False),
        patch.object(sfa_v1, "prepare_sparse_indices", side_effect=mutate_plan) as ordinary,
        patch.object(AscendSFAImpl, "_prepare_sorted_resident_sparse_cache", side_effect=mutate_plan) as sorted_plan,
    ):
        for step, rows in enumerate((4, 2, 4)):
            raw = torch.arange(16, dtype=torch.int32).view(4, 4) + 64 + step * 16
            produced.copy_(raw.unsqueeze(1))
            layers[0].indexer_select_post_process = MagicMock(return_value=produced[:rows])
            expected_shared = shared.view(4, 4).clone()
            expected_shared[:rows].copy_(raw[:rows])
            request_count = rows // query_width
            metadata = TestStagedSFAGraphPoc._make_decode_metadata(request_count, query_width=query_width)
            metadata.seq_lens.fill_(128 + query_width)
            metadata.seq_lens_cpu.fill_(128 + query_width)
            metadata.prompt_lens.fill_(128)
            metadata.prompt_lens_cpu_rows = [128] * rows
            metadata.block_table = torch.arange(request_count * 2).view(request_count, 2)
            metadata.indexer_block_table = metadata.block_table
            metadata.slot_mapping = (
                metadata.block_table[:, 1:] * 128 + torch.arange(query_width)
            ).reshape(rows)
            metadata.indexer_slot_mapping = metadata.slot_mapping
            metadata.layerwise_prefill_callback_metadata = ()
            metadata.split_boundary = torch.full((rows,), 128, dtype=torch.int32)
            metadata.decode_split_boundary = metadata.split_boundary
            output = torch.empty(rows, 4)
            plans = []
            for layer_index, (impl, kv_cache) in enumerate(zip(layers, kv_caches)):
                if not impl.has_indexer:
                    kv_cache = kv_cache[:2]
                with (
                    patch.object(torch, "empty", side_effect=AssertionError("serving allocation")),
                    patch.object(torch, "empty_like", side_effect=AssertionError("serving allocation")),
                    patch.object(torch.Tensor, "clone", side_effect=AssertionError("serving clone")),
                    patch.object(torch.Tensor, "contiguous", side_effect=AssertionError("contiguous is not isolation")),
                ):
                    if staged:
                        outputs = impl._cross_layer_pre_compute(
                            hidden[:rows],
                            kv_cache[0],
                            kv_cache[1],
                            kv_cache[2] if impl.has_indexer else None,
                            metadata.cos,
                            metadata.sin,
                            metadata.slot_mapping,
                            metadata.indexer_slot_mapping,
                            metadata.cum_query_lens,
                            metadata.seq_lens,
                            metadata.indexer_block_table,
                            metadata.split_boundary,
                            metadata.decode_req_indices,
                            metadata.block_table,
                            metadata.decode_selected_tokens,
                            metadata.decode_selected_counts,
                            metadata.decode_target_slot_mapping,
                            metadata.decode_union_mapping_workspace,
                            metadata.decode_shard_packed_workspace,
                            metadata.decode_shard_mapping_workspace,
                            metadata.decode_shard_counts_workspace,
                            metadata.resident_state_indices,
                            metadata.resident_state_generations,
                        )
                        plan = outputs[2]
                    else:
                        impl.forward(impl.layer_name, hidden[:rows], kv_cache, metadata, output=output)
                        plan = impl._execute_sparse_flash_attention_process.call_args.args[3]
                plans.append(plan)
                assert torch.equal(shared.view(4, 4), expected_shared)
                if shrink_latent:
                    assert torch.equal(metadata.decode_selected_tokens, raw[:rows].view(request_count, 4 * query_width))
                assert plan.shape == (rows, 1, 4)
                if layer_index:
                    # Shrink Graph A retains private storage; full-resident
                    # attention reads the shared address without a planner.
                    expected_ptr = staging_ptrs[layer_index - 1] if shrink_latent else shared.data_ptr()
                    assert plan.data_ptr() == expected_ptr
                for previous_index, previous_plan in enumerate(plans):
                    if shrink_latent:
                        assert torch.all(previous_plan == previous_index)
                    else:
                        assert torch.equal(previous_plan, raw[:rows].unsqueeze(1))
            layers[0].indexer_select_post_process.assert_called_once()

    assert ordinary.call_count == (9 if shrink_latent and not resident else 0)
    assert sorted_plan.call_count == (9 if resident else 0)
    for impl in layers[1:]:
        impl.indexer_select_post_process.assert_not_called()


class TestAscendSFABackend(TestBase):
    def test_get_name(self):
        self.assertEqual(AscendSFABackend.get_name(), "ASCEND_SFA")

    def test_get_builder_cls(self):
        with patch.object(sfa_v1, "enable_cp", return_value=False):
            self.assertEqual(AscendSFABackend.get_builder_cls(), AscendSFAMetadataBuilder)

    def test_get_kv_cache_shape(self):
        result = AscendSFABackend.get_kv_cache_shape(2, 4, 8, 128)
        self.assertEqual(result, (2, 4, 8, 128))

    def test_get_impl_cls(self):
        with patch.object(sfa_v1, "enable_cp", return_value=False):
            result = AscendSFABackend.get_impl_cls()
        self.assertEqual(result, AscendSFAImpl)


class TestAscendSFAMetadata(TestBase):
    def test_ascend_sfa_metadata_default(self):
        num_actual_tokens = 100
        slot_mapping = torch.randn(100, 4, 1024)
        seq_lens = torch.tensor([30, 50])
        cum_query_lens = torch.tensor([0, 30, 80])
        block_table = torch.randint(0, 100, (100, 4))

        rope_dim = 32
        max_seq_len = int(seq_lens.max().item())
        sin = torch.randn(max_seq_len, rope_dim)
        cos = torch.randn(max_seq_len, rope_dim)

        num_input_tokens = 2
        head_dim = None
        attn_mask = None
        attn_state = AscendAttentionState.ChunkedPrefill

        metadata = AscendSFAMetadata(
            num_actual_tokens=num_actual_tokens,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens,
            cum_query_lens=cum_query_lens,
            block_table=block_table,
            sin=sin,
            cos=cos,
            num_input_tokens=num_input_tokens,
            head_dim=head_dim,
            attn_mask=attn_mask,
            attn_state=attn_state,
        )

        self.assertEqual(metadata.num_actual_tokens, num_actual_tokens)
        self.assertIs(metadata.slot_mapping, slot_mapping)
        self.assertTrue(torch.equal(metadata.seq_lens, seq_lens))
        self.assertTrue(torch.equal(metadata.cum_query_lens, cum_query_lens))
        self.assertIs(metadata.block_table, block_table)
        self.assertIs(metadata.sin, sin)
        self.assertIs(metadata.cos, cos)
        self.assertEqual(metadata.num_input_tokens, num_input_tokens)
        self.assertIs(metadata.head_dim, head_dim)
        self.assertIs(metadata.attn_mask, attn_mask)
        self.assertEqual(metadata.attn_state, attn_state)


class TestAscendSFAMetadataBuilder(TestBase):
    @patch("vllm.distributed.parallel_state._TP", new_callable=lambda: MagicMock(spec=GroupCoordinator))
    def setUp(self, mock_tp):
        mock_tp.world_size = 2
        mock_tp.rank_in_group = MagicMock()
        mock_tp.device_group = MagicMock()

        self.mock_cfg = MagicMock()

        self.mock_cfg.parallel_config = MagicMock()
        self.mock_cfg.parallel_config.tensor_parallel_size = 1
        self.mock_cfg.parallel_config.prefill_context_parallel_size = 1
        self.mock_cfg.parallel_config.decode_context_parallel_size = 1

        self.mock_cfg.compilation_config = MagicMock()
        self.mock_cfg.compilation_config.pass_config = MagicMock()
        self.mock_cfg.compilation_config.pass_config.enable_sp = False

        self.mock_cfg.speculative_config.num_speculative_tokens = 0

        self.patcher = patch("vllm.config.get_current_vllm_config", return_value=self.mock_cfg)
        self.patcher.start()

        # Mock parent class __init__ to avoid complex initialization,
        # but still set the essential attributes that child class needs
        def mock_parent_init(
            self, kv_cache_spec, layer_names, vllm_config, device, metadata_cls, supports_dcp_with_varlen
        ):
            self.metadata_cls = metadata_cls
            self.kv_cache_spec = kv_cache_spec
            self.model_config = vllm_config.model_config
            self.vllm_config = vllm_config
            self.device = device
            self.chunked_prefill_workspace_size = 128 * 1024
            self.chunked_prefill_workspace = torch.empty(
                (self.chunked_prefill_workspace_size, vllm_config.model_config.get_head_size()),
                dtype=vllm_config.model_config.dtype,
                device=device,
            )

        self.parent_init_patcher = patch(
            "vllm.model_executor.layers.attention.mla_attention.MLACommonMetadataBuilder.__init__", mock_parent_init
        )
        self.parent_init_patcher.start()

        if hasattr(enable_dsa_cp, "cache_clear"):
            enable_dsa_cp.cache_clear()

    def tearDown(self):
        self.patcher.stop()
        self.parent_init_patcher.stop()

    @patch("vllm_ascend.attention.sfa_v1.is_v1_kv_transfer_group")
    @patch("vllm_ascend.attention.sfa_v1.has_kv_transfer_group")
    @patch("vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla")
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp")
    def test_dsa_sparse_metadata_reuses_builder_storage(
        self,
        mock_enable_dsa_cp,
        mock_get_cos_and_sin_mla,
        mock_has_kv_transfer_group,
        mock_is_v1_kv_transfer_group,
    ):
        mock_enable_dsa_cp.return_value = False
        mock_has_kv_transfer_group.return_value = True
        mock_is_v1_kv_transfer_group.return_value = True
        mock_get_cos_and_sin_mla.side_effect = lambda positions, _: (
            torch.zeros_like(positions),
            torch.zeros_like(positions),
        )

        kv_cache_spec = MagicMock()
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        vllm_config.model_config.hf_text_config.topk_tokens = 16
        vllm_config.speculative_config.num_speculative_tokens = 1
        vllm_config.scheduler_config.max_num_seqs = 4
        vllm_config.scheduler_config.max_num_batched_tokens = 8

        with patch.dict(
            os.environ,
            {
                "VLLM_ASCEND_DSA_UNBUNDLE": "1",
                "VLLM_ASCEND_DSA_SHRINK_LATENT": "2",
            },
        ):
            builder = AscendSFAMetadataBuilder(
                kv_cache_spec=kv_cache_spec,
                layer_names=["layer1", "layer2"],
                vllm_config=vllm_config,
                device=torch.device("cpu"),
            )
        builder.attn_mask_builder.get_attention_mask = MagicMock(return_value=None)

        def common_metadata(
            query_start_loc,
            computed,
            prompt_lens,
            request_ids,
            cold_compact_resumes=(),
        ):
            num_reqs = len(request_ids)
            num_tokens = int(query_start_loc[-1])
            return SimpleNamespace(
                num_reqs=num_reqs,
                num_actual_tokens=num_tokens,
                num_input_tokens=num_tokens,
                block_table_tensor=torch.zeros((num_reqs, 4), dtype=torch.int32),
                slot_mapping=torch.arange(num_tokens, dtype=torch.int64),
                positions=torch.arange(num_tokens, dtype=torch.int64),
                indexer_block_table_tensor=None,
                indexer_slot_mapping=None,
                prompt_lens_cpu=prompt_lens,
                query_start_loc_cpu=torch.tensor(query_start_loc, dtype=torch.int32),
                num_computed_tokens_cpu=torch.tensor(computed, dtype=torch.int32),
                query_start_loc=torch.tensor(query_start_loc, dtype=torch.int32),
                seq_lens=torch.tensor(computed, dtype=torch.int32),
                seq_lens_cpu=torch.tensor(computed, dtype=torch.int32),
                request_ids=request_ids,
                attn_state=AscendAttentionState.DecodeOnly,
                cold_compact_resumes=cold_compact_resumes,
            )

        ordinary_transition = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_metadata(
                [0, 1], [8], [9], ["ordinary"]
            ),
        )
        assert ordinary_transition.num_decode_tokens == 0

        cold_transition = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_metadata(
                [0, 1], [8], [9], ["cold"], (True,)
            ),
        )
        assert cold_transition.decode_req_indices.tolist() == [0]
        assert cold_transition.decode_row_offsets.tolist() == [0]
        assert cold_transition.split_boundary.tolist() == [9]
        assert cold_transition.num_decode_tokens == 1
        with patch.object(
            sfa_v1, "_decode_window_save_window_size", return_value=0
        ):
            boundary = sfa_v1._prepare_sfa_remap_boundary(
                cold_transition,
                ["cold"],
                is_dummy_run=False,
                index_topk=16,
                cached_tokens=(8,),
            )
        assert boundary.tolist() == [8]

        first = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_metadata(
                [0, 2, 4],
                [10, 20],
                [9, 19],
                ["r0", "r1"],
            ),
        )
        addresses = (
            first.split_boundary.data_ptr(),
            first.decode_req_indices.data_ptr(),
            first.decode_row_offsets.data_ptr(),
            first.decode_selected_tokens.data_ptr(),
            first.decode_selected_counts.data_ptr(),
            first.decode_target_slot_mapping.data_ptr(),
            first.decode_union_mapping_workspace.data_ptr(),
            first.decode_shard_packed_workspace.data_ptr(),
            first.decode_shard_mapping_workspace.data_ptr(),
            first.decode_shard_counts_workspace.data_ptr(),
        )
        assert first.decode_req_indices.tolist() == [0, 0, 1, 1]
        assert first.decode_row_offsets.tolist() == [0, 1, 0, 1]
        assert first.num_decode_tokens == 4

        second = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_metadata(
                [0, 1],
                [5],
                [4],
                ["r2"],
            ),
        )
        second_addresses = (
            second.split_boundary.data_ptr(),
            second.decode_req_indices.data_ptr(),
            second.decode_row_offsets.data_ptr(),
            second.decode_selected_tokens.data_ptr(),
            second.decode_selected_counts.data_ptr(),
            second.decode_target_slot_mapping.data_ptr(),
            second.decode_union_mapping_workspace.data_ptr(),
            second.decode_shard_packed_workspace.data_ptr(),
            second.decode_shard_mapping_workspace.data_ptr(),
            second.decode_shard_counts_workspace.data_ptr(),
        )

        assert second_addresses == addresses
        assert second.split_boundary.tolist() == [4]
        assert second.decode_req_indices.tolist() == [0]
        assert second.decode_row_offsets.tolist() == [0]
        assert second.num_decode_tokens == 1

        third = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_metadata(
                [0, 2, 4],
                [12, 22],
                [11, 21],
                ["r3", "r4"],
            ),
        )
        assert third.decode_req_indices.tolist() == [0, 0, 1, 1]
        assert third.decode_row_offsets.tolist() == [0, 1, 0, 1]
        assert third.split_boundary.tolist() == [11, 11, 21, 21]

        with self.assertRaisesRegex(RuntimeError, "max_num_batched_tokens=8"):
            builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_metadata(
                    [0, 9],
                    [9],
                    [8],
                    ["too-large"],
                ),
            )

        with self.assertRaisesRegex(RuntimeError, "max_num_seqs=4"):
            builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_metadata(
                    [0, 1, 2, 3, 4, 5],
                    [1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0],
                    ["r0", "r1", "r2", "r3", "r4"],
                ),
            )

    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_ascend_sfa_metadata_builder_default(self):
        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(
            kv_cache_spec=kv_cache_spec, layer_names=layer_names, vllm_config=vllm_config, device=device
        )

        assert builder.device == device
        assert builder.vllm_config == vllm_config

    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    @patch("vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla")
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_ascend_sfa_metadata_builder_build(
        self,
        mock_enable_dsa_cp,
        mock_get_cos_and_sin_mla,
        mock_get_current_vllm_config,
    ):
        mock_enable_dsa_cp.return_value = False

        cfg = MagicMock()
        cfg.model_config = MagicMock()
        cfg.model_config.hf_text_config = MagicMock()

        mock_get_current_vllm_config.return_value = cfg
        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(
            kv_cache_spec=kv_cache_spec, layer_names=layer_names, vllm_config=vllm_config, device=device
        )

        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 10
        common_attn_metadata.num_actual_tokens = 100
        common_attn_metadata.query_start_loc = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.query_start_loc_cpu = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.slot_mapping = torch.randn(100, 4, 1024)
        common_attn_metadata.seq_lens_cpu = torch.tensor([2] * 10)
        common_attn_metadata.positions = torch.randn(100)
        common_attn_metadata.attn_mask = None
        common_attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
        common_attn_metadata.block_table_tensor = torch.randn(100, 4)
        common_attn_metadata.cos = None
        common_attn_metadata.sin = None
        common_attn_metadata.num_input_tokens = 100

        mock_get_cos_and_sin_mla.return_value = (torch.randn(100), torch.randn(100))

        metadata = builder.build(
            common_prefix_len=10,
            common_attn_metadata=common_attn_metadata,
        )

        assert isinstance(metadata, AscendSFAMetadata)
        assert metadata.num_actual_tokens == common_attn_metadata.num_actual_tokens
        assert metadata.slot_mapping.shape == (100, 4, 1024)

    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    @patch("vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_ascend_sfa_metadata_builder_build_for_graph_capture(
        self, mock_get_cos_and_sin_mla, mock_get_current_vllm_config
    ):
        cfg = MagicMock()
        cfg.model_config = MagicMock()
        cfg.model_config.hf_text_config = MagicMock()

        mock_get_current_vllm_config.return_value = cfg

        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(
            kv_cache_spec=kv_cache_spec, layer_names=layer_names, vllm_config=vllm_config, device=device
        )

        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 10
        common_attn_metadata.num_actual_tokens = 100
        common_attn_metadata.query_start_loc = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.query_start_loc_cpu = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.slot_mapping = torch.randn(100, 4, 1024)
        common_attn_metadata.seq_lens_cpu = torch.tensor([2] * 10)
        common_attn_metadata.positions = torch.randn(100)
        common_attn_metadata.attn_mask = None
        common_attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
        common_attn_metadata.block_table_tensor = torch.randn(100, 4)
        common_attn_metadata.cos = None
        common_attn_metadata.sin = None
        common_attn_metadata.num_input_tokens = 100

        mock_get_cos_and_sin_mla.return_value = (torch.randn(100), torch.randn(100))

        attn_metadata = builder.build_for_graph_capture(
            common_attn_metadata=common_attn_metadata,
            attn_state=AscendAttentionState.DecodeOnly,
        )

        assert isinstance(attn_metadata, AscendSFAMetadata)
        assert attn_metadata.attn_state == AscendAttentionState.DecodeOnly

    @patch("vllm_ascend.attention.sfa_v1.staged_sfa_connector_supports_sparse_load", return_value=True)
    @patch("vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla")
    def test_q1_sparse_rows_reuse_builder_storage(self, mock_get_cos_and_sin_mla, _):
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        vllm_config.model_config.hf_text_config.topk_tokens = 32
        vllm_config.speculative_config = None
        vllm_config.scheduler_config.max_num_seqs = 4
        vllm_config.scheduler_config.max_num_batched_tokens = 4
        with patch.dict(
            os.environ,
            {
                "VLLM_ASCEND_DSA_UNBUNDLE": "1",
                "VLLM_ASCEND_DSA_SHRINK_LATENT": "2",
            },
        ):
            builder = AscendSFAMetadataBuilder(
                kv_cache_spec=MagicMock(),
                layer_names=["layer1"],
                vllm_config=vllm_config,
                device=torch.device("cpu"),
            )
        builder.enable_dsa_cp = False

        common = MagicMock()
        common.num_reqs = 4
        common.num_actual_tokens = 2
        common.num_input_tokens = 4
        common.block_table_tensor = torch.zeros((4, 4), dtype=torch.int32)
        common.slot_mapping = torch.arange(4, dtype=torch.int32)
        common.positions = torch.arange(4, dtype=torch.long)
        common.indexer_block_table_tensor = None
        common.indexer_slot_mapping = None
        common.prompt_lens_cpu = np.array([128, 256], dtype=np.int32)
        common.request_ids = ["req0", "req1"]
        common.query_start_loc = torch.arange(5, dtype=torch.int32)
        common.query_start_loc_cpu = torch.arange(5, dtype=torch.int32)
        common.num_computed_tokens_cpu = torch.tensor([128, 256, 0, 0], dtype=torch.int32)
        common.seq_lens = torch.tensor([129, 257, 0, 0], dtype=torch.int32)
        common.seq_lens_cpu = common.seq_lens.cpu()
        common.attn_state = AscendAttentionState.DecodeOnly
        mock_get_cos_and_sin_mla.return_value = (
            torch.zeros((4, 1)),
            torch.zeros((4, 1)),
        )

        first = builder.build(0, common)
        prompt_lens_version = first.prompt_lens._version
        second = builder.build(0, common)

        self.assertEqual(first.prompt_lens.data_ptr(), second.prompt_lens.data_ptr())
        self.assertEqual(first.decode_req_indices.data_ptr(), second.decode_req_indices.data_ptr())
        self.assertEqual(second.prompt_lens._version, prompt_lens_version)
        torch.testing.assert_close(
            second.prompt_lens,
            torch.tensor([128, 256, 0, 0], dtype=torch.int32),
        )
        torch.testing.assert_close(
            second.decode_req_indices,
            torch.tensor([0, 1, -1, -1], dtype=torch.int32),
        )
        torch.testing.assert_close(
            second.decode_valid_row_indices,
            torch.tensor([0, 1], dtype=torch.int32),
        )
        self.assertFalse(second.decode_valid_rows_all)

        common.num_computed_tokens_cpu = torch.tensor([127, 256, 0, 0], dtype=torch.int32)
        prompt_remainder = builder.build(0, common)

        self.assertEqual(prompt_remainder.num_decode_tokens, 1)
        torch.testing.assert_close(
            prompt_remainder.decode_valid_row_indices,
            torch.tensor([1], dtype=torch.int32),
        )

        common.num_computed_tokens_cpu = torch.tensor([128, 384, 0, 0], dtype=torch.int32)
        common.prompt_lens_cpu = np.array([128, 384], dtype=np.int32)
        common.request_ids = ["req0", "req2"]
        updated = builder.build(0, common)

        self.assertEqual(first.prompt_lens.data_ptr(), updated.prompt_lens.data_ptr())
        self.assertGreater(updated.prompt_lens._version, prompt_lens_version)
        torch.testing.assert_close(
            updated.prompt_lens,
            torch.tensor([128, 384, 0, 0], dtype=torch.int32),
        )
        self.assertEqual(updated.decode_request_ids_compact, ["req0", "req2"])

        common.attn_state = AscendAttentionState.ChunkedPrefill
        common.num_actual_tokens = 4
        common.query_start_loc = torch.tensor([0, 2, 4, 4, 4], dtype=torch.int32)
        common.query_start_loc_cpu = common.query_start_loc.cpu()
        common.num_computed_tokens_cpu = torch.tensor([127, 383, 0, 0], dtype=torch.int32)
        builder.build(0, common)

        common.attn_state = AscendAttentionState.DecodeOnly
        common.num_actual_tokens = 2
        common.query_start_loc = torch.arange(5, dtype=torch.int32)
        common.query_start_loc_cpu = common.query_start_loc.cpu()
        common.num_computed_tokens_cpu = torch.tensor([128, 384, 0, 0], dtype=torch.int32)
        restored = builder.build(0, common)

        torch.testing.assert_close(
            restored.decode_valid_row_indices,
            torch.tensor([0, 1], dtype=torch.int32),
        )

    @patch("vllm_ascend.attention.sfa_v1.staged_sfa_connector_supports_sparse_load", return_value=True)
    @patch("vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla")
    def test_mtp2_sparse_rows_reuse_fixed_layout_storage(self, mock_get_cos_and_sin_mla, _):
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        vllm_config.model_config.hf_text_config.topk_tokens = 32
        vllm_config.speculative_config = SimpleNamespace(
            method="mtp",
            num_speculative_tokens=1,
        )
        vllm_config.scheduler_config.max_num_seqs = 2
        vllm_config.scheduler_config.max_num_batched_tokens = 4
        with patch.dict(
            os.environ,
            {
                "VLLM_ASCEND_DSA_UNBUNDLE": "1",
                "VLLM_ASCEND_DSA_SHRINK_LATENT": "2",
            },
        ):
            builder = AscendSFAMetadataBuilder(
                kv_cache_spec=MagicMock(),
                layer_names=["layer1"],
                vllm_config=vllm_config,
                device=torch.device("cpu"),
            )
        builder.enable_dsa_cp = False

        common = MagicMock()
        common.num_reqs = 2
        common.num_actual_tokens = 2
        common.num_input_tokens = 4
        common.block_table_tensor = torch.zeros((2, 4), dtype=torch.int32)
        common.slot_mapping = torch.arange(4, dtype=torch.int32)
        common.positions = torch.arange(4, dtype=torch.long)
        common.indexer_block_table_tensor = None
        common.indexer_slot_mapping = None
        common.prompt_lens_cpu = np.array([128], dtype=np.int32)
        common.request_ids = ["req0"]
        common.query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32)
        common.query_start_loc_cpu = common.query_start_loc.cpu()
        common.num_computed_tokens_cpu = torch.tensor(
            [128, 0],
            dtype=torch.int32,
        )
        common.seq_lens = torch.tensor([130, 0], dtype=torch.int32)
        common.seq_lens_cpu = common.seq_lens.cpu()
        common.attn_state = AscendAttentionState.SpecDecoding
        mock_get_cos_and_sin_mla.return_value = (
            torch.zeros((4, 1)),
            torch.zeros((4, 1)),
        )

        first = builder.build(0, common)
        prompt_lens_version = first.prompt_lens._version
        request_rows_version = first.decode_req_indices._version
        row_offsets_version = first.decode_row_offsets._version
        second = builder.build(0, common)

        self.assertEqual(
            first.prompt_lens.data_ptr(),
            second.prompt_lens.data_ptr(),
        )
        self.assertEqual(
            second.prompt_lens._version,
            prompt_lens_version,
        )
        self.assertEqual(
            second.decode_req_indices._version,
            request_rows_version,
        )
        self.assertEqual(
            second.decode_row_offsets._version,
            row_offsets_version,
        )
        torch.testing.assert_close(
            second.prompt_lens,
            torch.tensor([128, 128, 0, 0], dtype=torch.int32),
        )
        torch.testing.assert_close(
            second.decode_req_indices,
            torch.tensor([0, 0, -1, -1], dtype=torch.int32),
        )
        torch.testing.assert_close(
            second.decode_row_offsets,
            torch.tensor([0, 1, 0, 0], dtype=torch.int32),
        )
        torch.testing.assert_close(
            second.decode_valid_row_indices,
            torch.tensor([0, 1], dtype=torch.int32),
        )
        torch.testing.assert_close(
            second.decode_req_indices_compact,
            torch.tensor([0, 0], dtype=torch.int32),
        )
        self.assertFalse(second.decode_valid_rows_all)

        common.prompt_lens_cpu = np.array([160], dtype=np.int32)
        common.request_ids = ["req1"]
        common.num_computed_tokens_cpu = torch.tensor(
            [160, 0],
            dtype=torch.int32,
        )
        common.seq_lens = torch.tensor([162, 0], dtype=torch.int32)
        common.seq_lens_cpu = common.seq_lens.cpu()
        updated = builder.build(0, common)

        self.assertGreater(
            updated.prompt_lens._version,
            prompt_lens_version,
        )
        torch.testing.assert_close(
            updated.prompt_lens,
            torch.tensor([160, 160, 0, 0], dtype=torch.int32),
        )
        self.assertEqual(
            updated.decode_request_ids_compact,
            ["req1"],
        )

        common.num_computed_tokens_cpu = torch.tensor(
            [159, 0],
            dtype=torch.int32,
        )
        prompt_remainder = builder.build(0, common)

        self.assertEqual(prompt_remainder.num_decode_tokens, 1)
        torch.testing.assert_close(
            prompt_remainder.decode_valid_row_indices,
            torch.tensor([1], dtype=torch.int32),
        )
        torch.testing.assert_close(
            prompt_remainder.decode_req_indices,
            torch.tensor([-1, 0, -1, -1], dtype=torch.int32),
        )

        common.num_computed_tokens_cpu = torch.tensor(
            [160, 0],
            dtype=torch.int32,
        )
        restored = builder.build(0, common)

        self.assertEqual(restored.num_decode_tokens, 2)
        torch.testing.assert_close(
            restored.decode_valid_row_indices,
            torch.tensor([0, 1], dtype=torch.int32),
        )
        torch.testing.assert_close(
            restored.decode_req_indices,
            torch.tensor([0, 0, -1, -1], dtype=torch.int32),
        )

        common.attn_state = AscendAttentionState.SpecDecoding
        common.num_actual_tokens = 1
        common.num_input_tokens = 2
        common.slot_mapping = torch.arange(2, dtype=torch.int32)
        common.positions = torch.arange(2, dtype=torch.long)
        common.query_start_loc = torch.tensor(
            [0, 1, 2],
            dtype=torch.int32,
        )
        common.query_start_loc_cpu = common.query_start_loc.cpu()
        mock_get_cos_and_sin_mla.return_value = (
            torch.zeros((2, 1)),
            torch.zeros((2, 1)),
        )
        single_row_mtp_step = builder.build(0, common)

        self.assertEqual(single_row_mtp_step.num_decode_tokens, 1)
        torch.testing.assert_close(
            single_row_mtp_step.decode_req_indices,
            torch.tensor([0, -1], dtype=torch.int32),
        )
        torch.testing.assert_close(
            single_row_mtp_step.decode_valid_row_indices,
            torch.tensor([0], dtype=torch.int32),
        )
