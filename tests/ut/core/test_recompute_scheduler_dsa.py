# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

from vllm.v1.core.sched.dsa_controller import DSAController, DSAControllerConfig
from vllm.v1.core.sched.dsa_types import DSARouteState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

from vllm_ascend.core.recompute_scheduler import RecomputeScheduler


def _make_scheduler() -> RecomputeScheduler:
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.requests = {}
    scheduler.is_kv_producer = False
    scheduler.is_hybrid_model = False
    scheduler.is_mtp_kv_consumer = False
    scheduler.num_spec_tokens = 0
    scheduler.log_stats = False
    scheduler._enqueue_waiting_request = Mock()
    scheduler.connector = None
    scheduler.kv_cache_manager = SimpleNamespace(remove_saved_decode_window_blocks=Mock(return_value=0))
    scheduler.dsa_controller = DSAController(
        config=DSAControllerConfig(
            threshold=8192,
            max_model_len=179840,
            block_size=128,
            chunk_size=256,
            window_size=256,
            index_topk=2048,
            query_width=2,
            scratch_capacity=4096,
            node_role="decode",
            deployment_mode="pd",
            data_compatibility_fingerprint="data-fingerprint",
            instance_capability_digest="instance-digest",
        )
    )
    scheduler._dsa_pending_decode_window_releases = {}
    return scheduler


def test_long_decoder_completion_has_dsa_state_and_route() -> None:
    scheduler = _make_scheduler()
    request = SimpleNamespace(
        request_id="long-decoder-request",
        resumable=False,
        num_tokens=16384,
        dsa_state=None,
    )

    scheduler.add_request(request)

    assert request.dsa_state is not None
    assert request.dsa_state.route_state == DSARouteState.PROMOTING
    scheduler._enqueue_waiting_request.assert_called_once_with(request)

    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = {request.request_id: 1}
    scheduler_output.total_num_scheduled_tokens = 1
    scheduler._attach_dsa_route_snapshots(scheduler_output)

    assert scheduler_output.dsa_routes is not None
    route = scheduler_output.dsa_routes[request.request_id]
    assert route.request_key == request.dsa_state.request_key
    assert route.route_state == DSARouteState.PROMOTING
    assert scheduler_output.dsa_data_compatibility_fingerprint == "data-fingerprint"
    assert scheduler_output.dsa_instance_capability_digest == "instance-digest"

    scheduler._update_from_kv_xfer_finished(
        KVConnectorOutput(completed_decode_window_saves={request.request_id: 16384})
    )

    scheduler.kv_cache_manager.remove_saved_decode_window_blocks.assert_not_called()
    pending_key, pending_frontiers = scheduler._dsa_pending_decode_window_releases[request.request_id]
    assert pending_key == request.dsa_state.request_key
    assert list(pending_frontiers) == [16384]
