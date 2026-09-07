# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import PretrainedConfig
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID
from vllm.v1.structured_output import StructuredOutputManager

from vllm_ascend.core.recompute_scheduler import AsyncRecomputeScheduler, RecomputeScheduler

BLOCK_SIZE = 16


@pytest.fixture(autouse=True)
def _initialize_block_hashing():
    init_none_hash(sha256)


def _make_scheduler(
    num_tokens: int,
    *,
    scheduler_cls=AsyncRecomputeScheduler,
    matched_tokens: int = 0,
    async_load: bool = False,
    num_kv_groups: int = 1,
    token_budget: int = 128,
    num_spec_tokens: int = 1,
    prefix_caching: bool = False,
) -> RecomputeScheduler:
    connector = MagicMock(spec=KVConnectorBase_V1)
    connector.get_num_new_matched_tokens.return_value = (matched_tokens, async_load)
    connector.build_connector_meta.return_value = None
    connector.take_events.return_value = None
    connector.request_finished.return_value = (False, None)
    max_model_len = max(num_tokens + 64, token_budget)
    num_blocks = num_kv_groups * ((max_model_len + BLOCK_SIZE - 1) // BLOCK_SIZE) + 8

    # Follow test_scheduler_dynamic_batch's offline config setup, without
    # replacing the scheduler, Request, KVCacheManager, or their update methods.
    with (
        patch.object(ModelConfig, "__post_init__", return_value=None),
        patch.object(SpeculativeConfig, "__post_init__", return_value=None),
        patch.object(VllmConfig, "__post_init__", return_value=None),
        patch(
            "vllm.v1.core.sched.scheduler.KVConnectorFactory.create_connector",
            return_value=connector,
        ),
    ):
        model_config = ModelConfig(
            model="scheduler-test",
            tokenizer="scheduler-test",
            max_model_len=max_model_len,
            skip_tokenizer_init=True,
        )
        model_config.hf_config = PretrainedConfig(is_encoder_decoder=False)
        model_config.hf_text_config = model_config.hf_config
        model_config.multimodal_config = None
        cache_config = CacheConfig(
            block_size=BLOCK_SIZE,
            enable_prefix_caching=prefix_caching,
        )
        cache_config.num_gpu_blocks = num_blocks
        config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            device_config=DeviceConfig(device="cpu"),
            scheduler_config=SchedulerConfig(
                max_num_seqs=1,
                max_num_batched_tokens=token_budget,
                max_model_len=max_model_len,
                enable_chunked_prefill=True,
                async_scheduling=scheduler_cls is AsyncRecomputeScheduler,
                is_encoder_decoder=False,
            ),
            speculative_config=SpeculativeConfig(method="mtp", num_speculative_tokens=num_spec_tokens),
            kv_transfer_config=KVTransferConfig(kv_connector="MockKVConnector", kv_role="kv_consumer"),
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    [f"layer_{group}"],
                    FullAttentionSpec(
                        block_size=BLOCK_SIZE,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float32,
                    ),
                )
                for group in range(num_kv_groups)
            ],
        )
        scheduler = scheduler_cls(
            vllm_config=config,
            kv_cache_config=kv_cache_config,
            block_size=BLOCK_SIZE,
            structured_output_manager=StructuredOutputManager(config),
            log_stats=True,
        )
    assert isinstance(scheduler.kv_cache_manager, KVCacheManager)
    assert scheduler.is_mtp_kv_consumer and scheduler.use_eagle
    assert scheduler.num_lookahead_tokens == num_spec_tokens
    return scheduler


def _make_request(num_tokens: int, request_id: str = "request") -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=[1] * num_tokens,
        sampling_params=SamplingParams(max_tokens=32, ignore_eos=True),
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def _update_output(scheduler, output: SchedulerOutput, token_ids: list[int]):
    req_ids = list(output.num_scheduled_tokens)
    assert len(req_ids) == 1
    return scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_ids[0]: 0},
            sampled_token_ids=[token_ids],
        ),
    )


def _assert_decode(output: SchedulerOutput, request: Request, start: int) -> None:
    req_id = request.request_id
    assert output.num_scheduled_tokens == {req_id: 2}
    assert output.scheduled_spec_decode_tokens == {req_id: [PLACEHOLDER_TOKEN_ID]}
    # Each decode must retain a non-draft row, including with outputs in flight.
    assert output.num_scheduled_tokens[req_id] - len(output.scheduled_spec_decode_tokens[req_id]) == 1
    if output.scheduled_new_reqs:
        assert output.scheduled_new_reqs[0].num_computed_tokens == start
    else:
        assert output.scheduled_cached_reqs.num_computed_tokens == [start]


@pytest.mark.parametrize(
    ("num_tokens", "async_load", "num_kv_groups"),
    [(6, False, 1), (8385, False, 1), (131621, True, 2)],
    ids=["sync-hit-6", "sync-hit-8385", "async-receive-131621-two-groups"],
)
@pytest.mark.parametrize("output_first", [False, True], ids=["schedule-ahead", "output-first"])
def test_async_mtp_cache_hit_interleavings(num_tokens, async_load, num_kv_groups, output_first):
    scheduler = _make_scheduler(
        num_tokens,
        matched_tokens=num_tokens - 1,
        async_load=async_load,
        num_kv_groups=num_kv_groups,
    )
    request = _make_request(num_tokens)
    scheduler.add_request(request)
    assert request.num_computed_tokens == 0
    assert request.spec_token_ids == [PLACEHOLDER_TOKEN_ID]

    if async_load:
        receiving = scheduler.schedule()
        assert receiving.num_scheduled_tokens == {}
        assert receiving.scheduled_spec_decode_tokens == {}
        assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        assert request.num_computed_tokens == num_tokens - 1
        block_ids = scheduler.kv_cache_manager.get_block_ids(request.request_id)
        assert len(block_ids) == num_kv_groups
        assert all(len(group) == (num_tokens - 1 + BLOCK_SIZE - 1) // BLOCK_SIZE for group in block_ids)
        assert scheduler.schedule().num_scheduled_tokens == {}
        scheduler.update_from_output(
            receiving,
            ModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                kv_connector_output=KVConnectorOutput(finished_recving={request.request_id}),
            ),
        )
        assert request.request_id in scheduler.finished_recving_kv_req_ids

    first = scheduler.schedule()
    _assert_decode(first, request, num_tokens - 1)
    assert request.status == RequestStatus.RUNNING
    assert request.num_cached_tokens == num_tokens - 1
    assert request.num_computed_tokens == num_tokens + 1
    assert request.num_output_placeholders == 2
    scheduler.connector.get_num_new_matched_tokens.assert_called_once_with(request, 0)
    assert not scheduler.finished_recving_kv_req_ids

    if output_first:
        _update_output(scheduler, first, [100])  # Reject the admission placeholder.
        assert request.num_computed_tokens == num_tokens
        assert request.num_output_placeholders == 0

    second = scheduler.schedule()
    _assert_decode(second, request, num_tokens if output_first else num_tokens + 1)
    if output_first:
        _update_output(scheduler, second, [101, 102])  # Accept the next draft.
    else:
        _update_output(scheduler, first, [100])
        assert request.num_output_placeholders == 2
    assert request.num_computed_tokens == num_tokens + 2

    third = scheduler.schedule()
    _assert_decode(third, request, num_tokens + 2)
    if not output_first:
        _update_output(scheduler, second, [101, 102])
    _update_output(scheduler, third, [103])
    assert list(request.output_token_ids) == [100, 101, 102, 103]
    assert request.num_computed_tokens == request.num_tokens - 1 == num_tokens + 3
    assert request.num_output_placeholders == 0


@pytest.mark.parametrize("num_tokens", [6, 8385])
def test_sync_mtp_cache_hit_output_and_draft_updates(num_tokens):
    scheduler = _make_scheduler(num_tokens, scheduler_cls=RecomputeScheduler, matched_tokens=num_tokens - 1)
    request = _make_request(num_tokens)
    scheduler.add_request(request)

    first = scheduler.schedule()
    _assert_decode(first, request, num_tokens - 1)
    assert request.spec_token_ids == []
    assert request.num_output_placeholders == 0
    result = _update_output(scheduler, first, [100])
    stats = result[0].scheduler_stats.spec_decoding_stats
    assert stats.num_draft_tokens == 1
    assert stats.num_accepted_tokens == 0
    assert request.num_computed_tokens == num_tokens

    scheduler.update_draft_token_ids(DraftTokenIds([request.request_id], [[101]]))
    second = scheduler.schedule()
    assert second.num_scheduled_tokens == {request.request_id: 2}
    assert second.scheduled_spec_decode_tokens == {request.request_id: [101]}
    assert second.scheduled_cached_reqs.num_computed_tokens == [num_tokens]
    assert request.spec_token_ids == []
    result = _update_output(scheduler, second, [101, 102])
    assert result[0].scheduler_stats.spec_decoding_stats.num_accepted_tokens == 1
    assert request.num_computed_tokens == request.num_tokens - 1 == num_tokens + 2
    assert list(request.output_token_ids) == [100, 101, 102]
    assert request.num_output_placeholders == 0


@pytest.mark.parametrize("scheduler_cls", [AsyncRecomputeScheduler, RecomputeScheduler])
@pytest.mark.parametrize(
    ("token_budget", "num_drafts"),
    [(5, 0), (6, 0), (7, 1)],
    ids=["before-ordinary-boundary", "exact-ordinary-boundary", "one-draft"],
)
def test_mtp_cache_hit_budget_clipping(scheduler_cls, token_budget, num_drafts):
    num_tokens, matched_tokens = 38, 32
    scheduler = _make_scheduler(
        num_tokens,
        scheduler_cls=scheduler_cls,
        matched_tokens=matched_tokens,
        token_budget=token_budget,
        num_spec_tokens=3,
    )
    request = _make_request(num_tokens)
    scheduler.add_request(request)
    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {request.request_id: token_budget}
    assert output.total_num_scheduled_tokens == token_budget
    assert output.scheduled_new_reqs[0].num_computed_tokens == matched_tokens
    expected_drafts = {request.request_id: [PLACEHOLDER_TOKEN_ID] * num_drafts} if num_drafts else {}
    assert output.scheduled_spec_decode_tokens == expected_drafts
    is_prefill_chunk = token_budget < num_tokens - matched_tokens
    assert request.is_prefill_chunk == is_prefill_chunk
    _update_output(scheduler, output, [] if is_prefill_chunk else [100])
    assert request.num_computed_tokens == min(matched_tokens + token_budget, num_tokens)
    assert request.num_output_placeholders == 0


@pytest.mark.parametrize("external_tokens", [0, BLOCK_SIZE], ids=["local-only", "local-plus-external"])
def test_mtp_real_prefix_cache_hit(external_tokens):
    num_tokens = 3 * BLOCK_SIZE + 1
    scheduler = _make_scheduler(num_tokens, prefix_caching=True)
    seed = _make_request(num_tokens, "seed")
    scheduler.add_request(seed)
    seed_output = scheduler.schedule()
    _update_output(scheduler, seed_output, [100])
    scheduler.finish_requests(seed.request_id, RequestStatus.FINISHED_ABORTED)

    request = _make_request(num_tokens)
    blocks, local_tokens = scheduler.kv_cache_manager.get_computed_blocks(request)
    # MTP/Eagle drops the last matched block to recompute its hidden states.
    assert local_tokens == 2 * BLOCK_SIZE
    assert len(blocks.get_block_ids()[0]) == 2
    scheduler.connector.get_num_new_matched_tokens.return_value = (external_tokens, False)
    scheduler.add_request(request)
    assert request.num_computed_tokens == 0

    output = scheduler.schedule()
    effective_tokens = local_tokens + external_tokens
    scheduler.connector.get_num_new_matched_tokens.assert_called_with(request, local_tokens)
    assert output.scheduled_new_reqs[0].num_computed_tokens == effective_tokens
    assert output.scheduled_new_reqs[0].block_ids[0][:2] == blocks.get_block_ids()[0]
    assert output.num_scheduled_tokens == {request.request_id: num_tokens + 1 - effective_tokens}
    assert output.scheduled_spec_decode_tokens == {request.request_id: [PLACEHOLDER_TOKEN_ID]}
    assert request.num_cached_tokens == effective_tokens
    _update_output(scheduler, output, [100])
    assert request.num_computed_tokens == num_tokens
    assert request.num_output_placeholders == 0
    _assert_decode(scheduler.schedule(), request, num_tokens)
