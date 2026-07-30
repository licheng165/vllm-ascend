"""Unit tests for per-request DSA offload routing helpers (design 15.2).

These cover the pure routing logic in vllm_ascend.attention.utils that consumes
LMCache's ``dsa_offload_routes`` table: resident/promoting/offloaded
classification, frontier derivation, staged-graph eligibility, and the
route-aware strict-frontier lookup. The connector is mocked, so no NPU/KV
transfer runtime is required.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Stub torch_npu for CPU test environments (real on NPU CI).
if "torch_npu" not in sys.modules:
    import importlib.util

    _npu = MagicMock()
    _npu.__spec__ = importlib.util.spec_from_loader("torch_npu", loader=None)
    sys.modules["torch_npu"] = _npu

from vllm_ascend.attention import utils as attention_utils
from vllm_ascend.utils import StagedSFARouteReason

DSA_OFF = attention_utils.DSA_ROUTE_STATE_OFFLOADED
DSA_RES = attention_utils.DSA_ROUTE_STATE_RESIDENT
DSA_PRO = attention_utils.DSA_ROUTE_STATE_PROMOTING
DSARouteSnapshot = attention_utils.DSARouteSnapshot


def _route(state=DSA_RES, committed_end=0, generation=0, window_anchor=0):
    return SimpleNamespace(
        state=state,
        committed_end=committed_end,
        generation=generation,
        window_anchor=window_anchor,
    )


def _metadata(requests=(), routes=None):
    md = SimpleNamespace(requests=list(requests))
    if routes is not None:
        md.dsa_offload_routes = routes
    return md


@pytest.fixture
def patched_connector():
    """Patch the connector metadata accessor used by the route helpers."""
    connector = MagicMock()
    connector.supports_staged_sfa_sparse_load = True
    connector.uses_layerwise_model_callbacks = True
    connector.wait_for_layer_load = MagicMock()
    connector._get_connector_metadata = MagicMock(return_value=_metadata())
    with patch.object(attention_utils, "has_kv_transfer_group", return_value=True), \
         patch.object(attention_utils, "is_v1_kv_transfer_group", return_value=True), \
         patch.object(attention_utils, "get_kv_transfer_group", return_value=connector):
        yield connector


# ---------------------------------------------------------------------------
# get_dsa_offload_routes
# ---------------------------------------------------------------------------


def test_get_routes_returns_none_without_table(patched_connector):
    patched_connector._get_connector_metadata.return_value = _metadata(routes=None)
    assert attention_utils.get_dsa_offload_routes(["a", "b"]) is None


def test_get_routes_aligned_to_request_order(patched_connector):
    patched_connector._get_connector_metadata.return_value = _metadata(
        routes={"a": _route(DSA_OFF, 8192, 1, 256), "b": _route(DSA_RES)}
    )
    snaps = attention_utils.get_dsa_offload_routes(["b", "a"])
    assert [s.state for s in snaps] == [DSA_RES, DSA_OFF]
    assert snaps[1].committed_end == 8192


def test_get_routes_missing_request_defaults_resident(patched_connector):
    patched_connector._get_connector_metadata.return_value = _metadata(
        routes={"a": _route(DSA_OFF, 4096, 1, 256)}
    )
    snaps = attention_utils.get_dsa_offload_routes(["a", "ghost"])
    assert snaps[0].is_offloaded
    assert snaps[1].state == DSA_RES
    assert snaps[1].committed_end == 0


def test_get_routes_rejects_duplicate_ids(patched_connector):
    patched_connector._get_connector_metadata.return_value = _metadata(
        routes={"a": _route(DSA_OFF, 8, 1, 0)}
    )
    with pytest.raises(RuntimeError, match="unique native request IDs"):
        attention_utils.get_dsa_offload_routes(["a", "a"])


# ---------------------------------------------------------------------------
# Pure classification helpers
# ---------------------------------------------------------------------------


def test_dsa_route_frontier_only_offloaded_contributes():
    snaps = [
        DSARouteSnapshot(DSA_OFF, 8192, 1, 256),
        DSARouteSnapshot(DSA_RES, 0, 0, 0),
        DSARouteSnapshot(DSA_PRO, 0, 1, 256),
        DSARouteSnapshot(DSA_OFF, 0, 1, 256),  # offloaded but zero frontier
    ]
    assert attention_utils.dsa_route_frontier(snaps) == [8192, 0, 0, 0]


def test_batch_has_offloaded_and_promoting_helpers():
    off = DSARouteSnapshot(DSA_OFF, 8, 1, 0)
    res = DSARouteSnapshot(DSA_RES, 0, 0, 0)
    pro = DSARouteSnapshot(DSA_PRO, 0, 1, 0)
    assert attention_utils.batch_has_offloaded_decode_row([res, off])
    assert not attention_utils.batch_has_offloaded_decode_row([res, pro])
    assert not attention_utils.batch_has_offloaded_decode_row(None)
    assert attention_utils.all_decode_requests_offloaded([off, off])
    assert not attention_utils.all_decode_requests_offloaded([off, res])
    assert attention_utils.batch_has_promoting_request([res, pro])
    assert not attention_utils.batch_has_promoting_request([res, off])


# ---------------------------------------------------------------------------
# get_lmcache_sparse_cached_tokens (route-aware)
# ---------------------------------------------------------------------------


def test_cached_tokens_route_aware(patched_connector):
    patched_connector._get_connector_metadata.return_value = _metadata(
        routes={
            "a": _route(DSA_OFF, 8192, 1, 256),
            "b": _route(DSA_RES),
            "c": _route(DSA_PRO, 0, 1, 256),
        }
    )
    frontiers = attention_utils.get_lmcache_sparse_cached_tokens(["a", "b", "c"])
    assert frontiers == [8192, 0, 0]


def test_cached_tokens_resident_does_not_raise_in_mixed_batch(patched_connector):
    # Design 8.2: a resident row in a mixed batch must not trigger the
    # "no proven frontier" failure.
    patched_connector._get_connector_metadata.return_value = _metadata(
        routes={"a": _route(DSA_OFF, 4096, 1, 256), "b": _route(DSA_RES)}
    )
    assert attention_utils.get_lmcache_sparse_cached_tokens(["a", "b"]) == [4096, 0]


# ---------------------------------------------------------------------------
# staged_sfa_metadata_sparse_load (route-aware)
# ---------------------------------------------------------------------------


def test_staged_classifier_all_offloaded_eligible(patched_connector):
    patched_connector._get_connector_metadata.return_value = _metadata(
        routes={
            "a": _route(DSA_OFF, 8192, 1, 256),
            "b": _route(DSA_OFF, 4608, 1, 256),
        }
    )
    reason, frontiers = attention_utils.staged_sfa_metadata_sparse_load(
        patched_connector._get_connector_metadata(), ["a", "b"]
    )
    assert reason is StagedSFARouteReason.ELIGIBLE
    assert frontiers == (8192, 4608)


def test_staged_classifier_mixed_route_is_safe_native(patched_connector):
    patched_connector._get_connector_metadata.return_value = _metadata(
        routes={"a": _route(DSA_OFF, 8192, 1, 256), "b": _route(DSA_RES)}
    )
    reason, frontiers = attention_utils.staged_sfa_metadata_sparse_load(
        patched_connector._get_connector_metadata(), ["a", "b"]
    )
    assert reason is StagedSFARouteReason.MIXED_DSA_ROUTE
    assert frontiers == ()


def test_staged_classifier_all_resident_is_safe_native(patched_connector):
    patched_connector._get_connector_metadata.return_value = _metadata(
        routes={"a": _route(DSA_RES), "b": _route(DSA_PRO, 0, 1, 256)}
    )
    reason, _ = attention_utils.staged_sfa_metadata_sparse_load(
        patched_connector._get_connector_metadata(), ["a", "b"]
    )
    assert reason is StagedSFARouteReason.MIXED_DSA_ROUTE


# ---------------------------------------------------------------------------
# build_dsa_compact_route (compact payload compression, design 14.3)
# ---------------------------------------------------------------------------


def test_build_compact_route_skips_resident_and_aligns_rows():
    snapshots = [
        DSARouteSnapshot(DSA_OFF, 8192, 1, 256),  # native req 0 -> compact 0
        DSARouteSnapshot(DSA_RES, 0, 0, 0),  # native req 1 -> skipped
        DSARouteSnapshot(DSA_OFF, 4608, 1, 256),  # native req 2 -> compact 1
    ]
    # Per-row native request indices (MTP2: req0 has two rows, then req1, req2).
    decode_req_indices = [0, 0, 1, 2]
    out = attention_utils.build_dsa_compact_route(snapshots, decode_req_indices)
    assert out is not None
    row_to_off, compact_to_native, offloaded_native = out
    assert row_to_off == [0, 0, -1, 1]  # both req0 rows -> compact 0; req1 -> -1
    assert compact_to_native == [0, 2]
    assert offloaded_native == [0, 2]


def test_build_compact_route_none_when_no_offloaded():
    snapshots = [
        DSARouteSnapshot(DSA_RES, 0, 0, 0),
        DSARouteSnapshot(DSA_PRO, 0, 1, 256),
    ]
    assert attention_utils.build_dsa_compact_route(snapshots, [0, 1]) is None


def test_build_compact_route_none_without_table():
    assert attention_utils.build_dsa_compact_route(None, [0]) is None


def test_compact_payload_matches_native_for_offloaded_rows():
    """The compact block-table view must match the native-order result for
    offloaded rows and skip resident rows entirely (design 14.3). Verified
    against the torch reference oracle of prepare_sparse_indices."""
    import torch

    from vllm_ascend.distributed.kv_transfer.sparse_offload.prepare_sparse_indices import (  # noqa: E501
        _prepare_sparse_indices_torch,
    )

    snapshots = [
        DSARouteSnapshot(DSA_OFF, 8, 1, 0),
        DSARouteSnapshot(DSA_RES, 0, 0, 0),
        DSARouteSnapshot(DSA_OFF, 40, 1, 0),
    ]
    decode_req_indices = [0, 1, 2]
    row_to_off, compact_to_native, offloaded_native = (
        attention_utils.build_dsa_compact_route(snapshots, decode_req_indices)
    )

    topk = torch.tensor([[[5, 6]], [[15, 16]], [[35, 36]]])
    boundary = torch.tensor([8, 0, 40])
    native_block_table = torch.tensor(
        [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]
    )

    # Native-order reference (resident row present, count 0).
    _, _, n_counts, n_targets = _prepare_sparse_indices_torch(
        topk,
        boundary,
        row_req_indices=torch.tensor([0, 1, 2]),
        request_block_table=native_block_table,
        block_size=2,
        need_packed=True,
    )
    # Compact view: gather offloaded rows, remap row indices.
    compact_block_table = native_block_table.index_select(
        0, torch.tensor(offloaded_native)
    )
    _, _, c_counts, c_targets = _prepare_sparse_indices_torch(
        topk,
        boundary,
        row_req_indices=torch.tensor(row_to_off),
        request_block_table=compact_block_table,
        block_size=2,
        need_packed=True,
    )
    # Compact payload only has offloaded rows, in native order.
    n_counts = n_counts.tolist()
    n_targets = n_targets.tolist()
    assert c_counts.tolist() == [n_counts[0], n_counts[2]]
    assert c_targets.tolist() == [n_targets[0], n_targets[2]]
    assert compact_to_native == [0, 2]
