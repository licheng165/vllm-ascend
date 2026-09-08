# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm import SamplingParams
from vllm.v1.core.dsa_shared_pool import DSABlockAllocationMode
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.request import Request, RequestStatus

from vllm_ascend.patch.platform.patch_balance_schedule import BalanceScheduler


class _ChildBlocks:
    block_ids = ([11, 12],)
    block_ids_by_bank = (([11], [12]),)

    def get_block_ids(self, allow_none: bool = False):
        return self.block_ids

    def get_block_ids_by_bank(self, allow_none: bool = False):
        return self.block_ids_by_bank

    def get_allocation_mode(self):
        return DSABlockAllocationMode.PREFILL_CHILD

    def get_allocation_generation(self):
        return 1


class _KVCacheManager:
    def __init__(self):
        self.blocks = _ChildBlocks()
        self.generations = []
        self.empty_kv_cache_blocks = SimpleNamespace(blocks=())

    def new_step_starts(self) -> None:
        pass

    def get_computed_blocks(self, request):
        return self.empty_kv_cache_blocks, 0

    def allocate_slots(self, request, num_new_tokens, **kwargs):
        self.generations.append(kwargs.get("allocation_generation"))
        return self.blocks

    def get_blocks(self, request_id):
        return self.blocks

    def get_num_common_prefix_blocks(self, request_id):
        return [0]


@pytest.fixture
def scheduler() -> BalanceScheduler:
    scheduler = BalanceScheduler.__new__(BalanceScheduler)
    scheduler.max_num_scheduled_tokens = 256
    scheduler.max_num_encoder_input_tokens = 0
    scheduler.max_num_running_reqs = 32
    scheduler.max_model_len = 179_840
    scheduler.num_lookahead_tokens = 0
    scheduler.need_mamba_block_aligned_split = False
    scheduler.is_encoder_decoder = False
    scheduler.use_eagle = False
    scheduler.use_pp = False
    scheduler.use_v2_model_runner = False
    scheduler.layerwise_prefill_p_node = True
    scheduler._last_allocation_generation = 0
    scheduler._request_allocation_generations = {}
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.scheduler_config = SimpleNamespace(
        long_prefill_token_threshold=0,
        enable_chunked_prefill=True,
        async_scheduling=False,
    )
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler.balance_queue = [torch.tensor([0])]
    scheduler.lora_config = None
    scheduler.connector = None
    scheduler.connector_prefix_cache_stats = None
    scheduler.ec_connector = None
    scheduler.log_stats = False
    scheduler.running = []
    scheduler.waiting = create_request_queue(SchedulingPolicy.FCFS)
    scheduler.prev_step_scheduled_req_ids = set()
    scheduler.finished_req_ids = set()
    scheduler.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
    scheduler.kv_cache_manager = _KVCacheManager()
    scheduler.encoder_cache_manager = SimpleNamespace(
        allocate=lambda request, index: None,
        free=lambda request: None,
        get_freed_mm_hashes=lambda: set(),
    )
    scheduler._update_after_schedule = lambda output: None
    return scheduler


def test_balance_scheduler_preserves_prefill_child_generation(scheduler) -> None:
    request = Request(
        request_id="req-120k",
        prompt_token_ids=list(range(128)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
    )
    scheduler.waiting.add_request(request)

    first = scheduler.schedule()
    new_request = first.scheduled_new_reqs[0]
    assert new_request.block_ids_by_bank == _ChildBlocks.block_ids_by_bank
    assert new_request.block_allocation_mode is DSABlockAllocationMode.PREFILL_CHILD
    assert new_request.allocation_generation == 1

    second = scheduler.schedule()
    assert scheduler.kv_cache_manager.generations == [1, 1]
    assert second.scheduled_cached_reqs.allocation_generations == [1]
    assert second.scheduled_cached_reqs.new_block_ids_by_bank == [
        _ChildBlocks.block_ids_by_bank
    ]


@pytest.mark.parametrize(
    ("load_kv_async", "capability", "compact_intent"),
    [
        pytest.param(True, True, True, id="cold-full-compact-load"),
        pytest.param(True, False, False, id="capability-false"),
        pytest.param(True, None, False, id="capability-absent"),
        pytest.param(False, True, False, id="no-async-load"),
        # Feature-off connectors advertise neither compact support nor async load.
        pytest.param(False, False, False, id="feature-off"),
    ],
)
def test_balance_scheduler_compact_external_load(scheduler, load_kv_async, capability, compact_intent) -> None:
    scheduler.layerwise_prefill_p_node = False
    scheduler.num_lookahead_tokens = 8
    scheduler.kv_cache_manager.blocks = KVCacheBlocks(([KVCacheBlock(block_id=11)],))
    allocate_slots = Mock(wraps=scheduler.kv_cache_manager.allocate_slots)
    scheduler.kv_cache_manager.allocate_slots = allocate_slots
    request = Request(
        request_id="cold-external",
        prompt_token_ids=list(range(8192)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
    )
    # A full external hit leaves the last prompt token for local computation.
    external_tokens = request.num_tokens - 1
    connector = SimpleNamespace(
        get_num_new_matched_tokens=Mock(return_value=(external_tokens, load_kv_async)),
        update_state_after_alloc=Mock(),
        build_connector_meta=Mock(return_value=object()),
    )
    if capability is not None:
        connector.supports_dsa_compact_external_load = capability
    scheduler.connector = connector
    scheduler.waiting.add_request(request)

    output = scheduler.schedule()

    connector.get_num_new_matched_tokens.assert_called_once_with(request, 0)
    allocate_slots.assert_called_once_with(
        request,
        0 if load_kv_async else 1,
        num_new_computed_tokens=0,
        new_computed_blocks=scheduler.kv_cache_manager.empty_kv_cache_blocks,
        num_lookahead_tokens=0,
        num_external_computed_tokens=external_tokens,
        delay_cache_blocks=load_kv_async,
        num_encoder_tokens=0,
        dsa_compact_external_load=compact_intent,
        allocation_generation=None,
    )
    connector.update_state_after_alloc.assert_called_once_with(
        request, scheduler.kv_cache_manager.blocks, external_tokens
    )
    connector.build_connector_meta.assert_called_once_with(output)
    assert output.kv_connector_metadata is connector.build_connector_meta.return_value
    assert request.num_external_computed_tokens == external_tokens
    assert scheduler._request_allocation_generations == {}
    if load_kv_async:
        assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        assert list(scheduler.waiting) == [request]
        assert scheduler.running == []
        assert output.scheduled_new_reqs == []
        assert output.num_scheduled_tokens == {}
        assert output.total_num_scheduled_tokens == 0
    else:
        assert request.status == RequestStatus.RUNNING
        assert not scheduler.waiting
        assert scheduler.running == [request]
        assert request.num_computed_tokens == external_tokens
        assert output.num_scheduled_tokens == {request.request_id: 1}
        assert output.total_num_scheduled_tokens == 1
        new_request = output.scheduled_new_reqs[0]
        assert new_request.block_ids == scheduler.kv_cache_manager.blocks.get_block_ids()
        assert new_request.block_ids_by_bank is None
        assert new_request.allocation_generation is None
