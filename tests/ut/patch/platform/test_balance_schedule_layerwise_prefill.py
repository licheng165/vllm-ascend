# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from vllm import SamplingParams
from vllm.v1.core.dsa_shared_pool import DSABlockAllocationMode
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.request import Request

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


def test_balance_scheduler_preserves_prefill_child_generation() -> None:
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
