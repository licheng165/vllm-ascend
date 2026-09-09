# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import math
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.core.dsa_shared_pool import DSABlockAllocationMode
from vllm.v1.kv_cache_interface import (
    DSAExecutionRow,
    DSAKVRow,
    DSAKVTopology,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)

import vllm_ascend.attention.sfa_v1 as sfa_module
import vllm_ascend.spec_decode.eagle_proposer as proposer_module
import vllm_ascend.worker.model_runner_v1 as model_runner_module
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.sfa_v1 import AscendSFAMetadata, AscendSFAMetadataBuilder
from vllm_ascend.patch.platform.patch_kv_cache_interface import (
    AscendMLAAttentionSpec,
)
from vllm_ascend.spec_decode.eagle_proposer import SpecDecodeBaseProposer
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.npu_input_batch import NPUInputBatch

_PRODUCER_EXECUTIONS = (0, 1, 2, *(6 + 4 * index for index in range(18)), 78)


def _topology() -> DSAKVTopology:
    latent_rows = tuple(
        DSAKVRow(f"latent.{execution}", execution, 0, execution, execution % 2) for execution in range(79)
    )
    indexer_rows = tuple(
        DSAKVRow(
            f"indexer.{execution}",
            execution,
            1,
            row_ordinal,
            row_ordinal % 2,
        )
        for row_ordinal, execution in enumerate(_PRODUCER_EXECUTIONS)
    )
    indexers = {row.execution_ordinal: row for row in indexer_rows}
    executions = tuple(
        DSAExecutionRow(
            latent.execution_ordinal,
            latent,
            indexers.get(latent.execution_ordinal),
        )
        for latent in latent_rows
    )
    return DSAKVTopology(executions, (latent_rows, indexer_rows), "test-79-22")


def _global_slab_config(parent_capacity: int = 2) -> KVCacheConfig:
    topology = _topology()
    latent_names = [row.layer_name for row in topology.rows_by_group[0]]
    indexer_names = [row.layer_name for row in topology.rows_by_group[1]]
    latent_spec = AscendMLAAttentionSpec(
        block_size=1,
        num_kv_heads=1,
        head_size=9,
        sparse_head_dim=(8, 1),
        dtype=torch.float32,
    )
    indexer_spec = AscendMLAAttentionSpec(
        block_size=1,
        num_kv_heads=1,
        head_size=1,
        sparse_head_dim=(1,),
        dtype=torch.float32,
    )
    bundle_page = math.lcm(
        latent_spec.page_size_bytes,
        indexer_spec.page_size_bytes,
    )
    slab_size = (len(latent_names) * parent_capacity + 1) * bundle_page
    return KVCacheConfig(
        num_blocks=parent_capacity,
        kv_cache_tensors=[
            KVCacheTensor(
                size=slab_size,
                shared_by=[*latent_names, *indexer_names],
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(latent_names, latent_spec),
            KVCacheGroupSpec(indexer_names, indexer_spec),
        ],
        dsa_kv_topology=topology,
    )


def _runner() -> NPUModelRunner:
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.device = torch.device("cpu")
    runner.layerwise_prefill_p_node = True
    runner.dsa_shared_pool = True
    runner.dsa_unbundle = True
    runner.use_sparse = True
    runner.use_sparse_c8_indexer = False
    runner.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(is_kv_producer=True)
    )
    return runner


def test_global_slab_has_one_allocation_owner_reshape_and_null() -> None:
    runner = _runner()
    config = _global_slab_config()
    topology = config.dsa_kv_topology
    assert topology is not None

    with patch.object(
        model_runner_module.torch,
        "zeros",
        wraps=torch.zeros,
    ) as allocate:
        raw_bindings = runner._allocate_kv_cache_tensors(config)

    assert allocate.call_count == 1
    assert len(raw_bindings) == 79 + 22
    shared_entries = {id(value) for value in raw_bindings.values()}
    assert len(shared_entries) == 1
    raw = next(iter(raw_bindings.values()))[0]
    owner = runner._layerwise_prefill_global_raw_backing
    assert runner._layerwise_prefill_global_raw is raw
    assert owner is not raw
    assert raw.untyped_storage().data_ptr() == owner.untyped_storage().data_ptr()
    assert raw.data_ptr() % (2 * 1024 * 1024) == 0

    slab_spec = runner._layerwise_prefill_slab_spec
    slot_count = raw.numel() // slab_spec.bundle_page_size_bytes
    assert slot_count == 79 * config.num_blocks + 1

    original_reshape = model_runner_module.reshape_dsa_layerwise_prefill_raw
    with patch.object(
        model_runner_module,
        "reshape_dsa_layerwise_prefill_raw",
        wraps=original_reshape,
    ) as reshape:
        logical_caches = runner._reshape_kv_cache_tensors(config, raw_bindings)

    assert reshape.call_count == 1
    latent_views = {id(logical_caches[row.layer_name]) for row in topology.rows_by_group[0]}
    indexer_views = {id(logical_caches[row.layer_name]) for row in topology.rows_by_group[1]}
    assert len(latent_views) == 1
    assert len(indexer_views) == 1
    latent = logical_caches[topology.rows_by_group[0][0].layer_name]
    indexer = logical_caches[topology.rows_by_group[1][0].layer_name]
    assert len(latent) == 2
    assert len(indexer) == 1
    assert latent[0].shape[0] == slot_count
    assert indexer[0].shape[0] == slot_count * 9
    raw_storage = raw.untyped_storage().data_ptr()
    assert latent[0].untyped_storage().data_ptr() == raw_storage
    assert latent[1].untyped_storage().data_ptr() == raw_storage
    assert indexer[0].untyped_storage().data_ptr() == raw_storage
    assert latent[0].data_ptr() == raw.data_ptr()
    expected_rope_offset = (
        latent[0].shape[0] * latent[0].shape[1] * latent[0].shape[2] * latent[0].shape[3] * latent[0].element_size()
    )
    assert latent[1].data_ptr() - raw.data_ptr() == expected_rope_offset
    assert indexer[0].data_ptr() == raw.data_ptr()
    element_size = latent[0].element_size()
    assert latent[0].stride(0) * element_size == (
        slab_spec.block_size * slab_spec.num_kv_heads * slab_spec.kv_lora_rank * element_size
    )
    assert latent[1].stride(0) * element_size == (
        slab_spec.block_size * slab_spec.num_kv_heads * slab_spec.qk_rope_head_dim * element_size
    )
    assert indexer[0].stride(0) * element_size == (
        slab_spec.block_size * slab_spec.num_kv_heads * slab_spec.index_head_dim * element_size
    )


def test_startup_log_reports_p_node_observability() -> None:
    runner = _runner()
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE)
    runner.dsa_kv_topology = _topology()
    config = _global_slab_config()
    connector = SimpleNamespace(
        supports_layerwise_prefill_p_node=True,
        supports_layerwise_prefill_transfer_window=True,
        supports_layerwise_prefill_eager_callbacks=True,
        supports_dsa_index_lmcache=True,
    )
    logged = []

    def capture_info(message, *args, **_kwargs):
        logged.append(message % args if args else message)

    with (
        patch.object(model_runner_module, "get_kv_transfer_group", return_value=connector),
        patch.object(model_runner_module.logger, "info", capture_info),
    ):
        runner._log_layerwise_prefill_startup(config)

    message = next(
        message
        for message in logged
        if "Layerwise-prefill P worker" in message
    )
    assert "residency_mode=PREFILL_LAYERWISE" in message
    assert "latent_layers=79 indexer_layers=22" in message
    assert "producer_executions=22" in message
    assert "parent_capacity=2 child_capacity=158" in message
    assert "connector_transfer_window=True" in message
    assert "connector_indexer_persistence=True" in message


def test_global_slab_rejects_a_missing_single_null_bundle() -> None:
    runner = _runner()
    config = _global_slab_config()
    config.kv_cache_tensors[0].size -= math.lcm(
        config.kv_cache_groups[0].kv_cache_spec.page_size_bytes,
        config.kv_cache_groups[1].kv_cache_spec.page_size_bytes,
    )

    with pytest.raises(RuntimeError, match="single-null"):
        runner._allocate_kv_cache_tensors(config)


@pytest.mark.parametrize("execution_ordinal", [6, 78])
def test_execution_6_and_78_lower_groups_independently(
    execution_ordinal: int,
) -> None:
    topology = _topology()
    execution = topology.executions[execution_ordinal]
    common = SimpleNamespace(
        block_table_tensor=object(),
        slot_mapping=object(),
        indexer_block_table_tensor=object(),
        indexer_slot_mapping=object(),
    )
    tensors = {
        (0, 0): (object(), object()),
        (1, 1): (object(), object()),
    }
    getter = MagicMock(side_effect=lambda group, bank: tensors[(group, bank)])

    lowered = NPUModelRunner._layerwise_prefill_common_attn_metadata(
        common,
        execution,
        (("request", 17),),
        getter,
    )

    assert execution.latent.bank == 0
    assert execution.indexer is not None
    assert execution.indexer.bank == 1
    assert getter.call_args_list == [call(0, 0), call(1, 1)]
    assert lowered.block_table_tensor is tensors[(0, 0)][0]
    assert lowered.slot_mapping is tensors[(0, 0)][1]
    assert lowered.indexer_block_table_tensor is tensors[(1, 1)][0]
    assert lowered.indexer_slot_mapping is tensors[(1, 1)][1]
    assert tuple(callback.row for callback in lowered.layerwise_prefill_callback_metadata) == (
        execution.latent,
        execution.indexer,
    )
    assert all(
        callback.request_generations == (("request", 17),) for callback in lowered.layerwise_prefill_callback_metadata
    )


def test_mtp_draft_common_metadata_uses_execution_78_banks() -> None:
    runner = _runner()
    topology = _topology()
    runner.dsa_kv_rows_by_layer_name = {
        row.layer_name: row
        for rows in topology.rows_by_group
        for row in rows
    }
    runner.dsa_kv_executions_by_ordinal = {
        execution.execution_ordinal: execution
        for execution in topology.executions
    }
    common = SimpleNamespace(
        block_table_tensor=object(),
        slot_mapping=object(),
        indexer_block_table_tensor=object(),
        indexer_slot_mapping=object(),
    )
    tensors = {
        (0, 0): (object(), object()),
        (1, 1): (object(), object()),
    }
    getter = MagicMock(side_effect=lambda group, bank: tensors[(group, bank)])

    lowered = runner._layerwise_prefill_draft_common_attn_metadata(
        common,
        "latent.78",
        (("request", 17),),
        getter,
    )

    callbacks = lowered.layerwise_prefill_callback_metadata
    assert callbacks[0].execution.execution_ordinal == 78
    assert tuple(callback.row.bank for callback in callbacks) == (0, 1)
    assert lowered.indexer_block_table_tensor is tensors[(1, 1)][0]
    assert lowered.indexer_slot_mapping is tensors[(1, 1)][1]


@pytest.fixture
def mtp_metadata_runner(monkeypatch):
    runner = _runner()
    runner.kv_cache_config = _global_slab_config()
    topology = runner.kv_cache_config.dsa_kv_topology
    runner.dsa_kv_rows_by_layer_name = {row.layer_name: row for rows in topology.rows_by_group for row in rows}
    runner.dsa_kv_executions_by_ordinal = {execution.execution_ordinal: execution for execution in topology.executions}
    runner.pcp_size = runner.dcp_size = 1
    runner._has_gdn = runner.is_mm_prefix_lm = False
    runner.dsa_two_groups = True
    runner.dsa_shrink_latent = 0
    runner.decode_token_per_req = 1
    runner.need_accepted_tokens = False
    runner.attn_state = AscendAttentionState.ChunkedPrefill
    runner.actual_seq_lengths_q = [2]
    runner._resident_state_registry = runner._resident_state_indices = runner._resident_state_generations = None
    runner.model_config = SimpleNamespace(enable_return_routed_experts=False, uses_mrope=False, get_head_size=lambda: 8)
    runner.vllm_config.model_config = runner.model_config
    runner.vllm_config.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE)
    runner.speculative_config = SimpleNamespace(method="mtp")
    for name, values in (("seq_lens", [2, 0]), ("query_start_loc", [0, 2, 2]), ("positions", [0, 1, 0, 0])):
        cpu = torch.tensor(values, dtype=torch.int32)
        setattr(runner, name, SimpleNamespace(cpu=cpu, gpu=cpu.clone(), np=cpu.numpy()))
    runner.input_batch = NPUInputBatch(
        max_num_reqs=2,
        max_model_len=8,
        max_num_batched_tokens=4,
        device=runner.device,
        pin_memory=False,
        vocab_size=32,
        block_sizes=[1, 1],
        kernel_block_sizes=[[1], [1]],
        layerwise_prefill_p_node=True,
    )
    runner.input_batch._req_ids = ["request"]
    runner.input_batch.num_computed_tokens_cpu_tensor.zero_()
    banks = (([3, 4, 5, 6], [7, 8, 9, 10]), ([13, 14, 15, 16], [17, 18, 19, 20]))
    runner.requests = {
        "request": SimpleNamespace(
            block_ids=banks[0],
            block_ids_by_bank=banks,
            block_allocation_mode=DSABlockAllocationMode.PREFILL_CHILD,
            allocation_generation=17,
        )
    }
    for table, blocks in zip(runner.input_batch.layerwise_prefill_block_tables, banks):
        table.add_row(blocks, 0)
        table.commit_block_table(1)
        for group_table in table.block_tables:
            group_table.compute_slot_mapping(np.array([0, 0]), np.array([0, 1]))
        table.commit_slot_mapping(2)

    # Run the real SFA builder on CPU, replacing only device/model setup.
    builder = AscendSFAMetadataBuilder.__new__(AscendSFAMetadataBuilder)
    builder.metadata_cls = AscendSFAMetadata
    builder.model_config = runner.model_config
    builder.dsa_shrink_latent = 0
    builder.enable_dsa_cp = False
    builder.scratch_capacity = 4
    builder.decode_remap_boundary = torch.zeros(4, dtype=torch.int32)
    builder.attn_mask_builder = SimpleNamespace(get_attention_mask=lambda _: None)
    monkeypatch.setattr(sfa_module, "get_cos_and_sin_mla", lambda positions, _: (positions, positions))
    latent_names = runner.kv_cache_config.kv_cache_groups[0].layer_names
    runner.attn_groups = [[SimpleNamespace(layer_names=latent_names, get_metadata_builder=lambda *_: builder)], []]

    proposer = model_runner_module.AscendEagleProposer.__new__(model_runner_module.AscendEagleProposer)
    runner.drafter = proposer
    proposer.runner = runner
    proposer.vllm_config = runner.vllm_config
    proposer.method = "mtp"
    proposer.pcp_size = proposer.dcp_size = 1
    proposer.uses_mrope = proposer.needs_extra_input_slots = proposer.supports_mm_inputs = False
    proposer.uses_xdrope_dim = 0
    proposer.use_staged_mtp_draft_graph = proposer.use_cuda_graph = proposer.parallel_drafting = False
    proposer.num_speculative_tokens = 3
    proposer.kernel_block_size = 1
    proposer.max_model_len = 8
    proposer.attn_layer_names = ["latent.78"]
    proposer.draft_attn_groups = [SimpleNamespace(get_metadata_builder=lambda: builder)]
    proposer.arange = torch.arange(5, dtype=torch.int32)
    proposer.token_arange_np = proposer.arange.numpy()
    proposer.input_ids = torch.zeros(4, dtype=torch.int64)
    proposer.positions = torch.zeros(4, dtype=torch.int32)
    proposer.hidden_states = torch.zeros((4, 2))
    proposer.token_indices_to_sample = torch.zeros(2, dtype=torch.int64)
    proposer.slot_mapping_group = [torch.full((4,), -1, dtype=torch.int32) for _ in range(3)]
    proposer.indexer_slot_mapping_group = [torch.full((4,), -1, dtype=torch.int32) for _ in range(3)]
    runner._sync_metadata_across_dp = lambda num_tokens, **_: (num_tokens, None, None)
    runner.model = object()
    context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE)
    monkeypatch.setattr(proposer_module, "set_ascend_forward_context", lambda *_, **__: nullcontext(context))
    monkeypatch.setattr(proposer_module, "get_forward_context", lambda: context)
    monkeypatch.setattr(proposer_module, "is_pin_memory_available", lambda: False)
    monkeypatch.setattr(proposer_module, "HAS_TRITON", False)
    return runner


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("prepare", ["first_prefill", "unpadded", "padded"])
def test_mtp_runner_and_proposer_share_step_local_callback_identity(mtp_metadata_runner, padded, prepare):
    runner = mtp_metadata_runner
    proposer = runner.drafter
    previous_callbacks = None
    for _ in range(2):
        with patch.object(
            model_runner_module.LayerwisePrefillCallbackMetadata,
            "for_execution",
            wraps=model_runner_module.LayerwisePrefillCallbackMetadata.for_execution,
        ) as for_execution:
            regular, common = runner._build_attention_metadata(
                num_tokens=2,
                num_reqs=1,
                max_query_len=2,
                num_tokens_padded=4 if padded else 2,
                num_reqs_padded=2 if padded else 1,
            )

        callbacks = regular["latent.78"].layerwise_prefill_callback_metadata
        # Value equality is insufficient: the async backend authenticates `is`.
        assert common.layerwise_prefill_callback_metadata[0] is callbacks[0]
        assert common.layerwise_prefill_callback_metadata[1] is callbacks[1]
        assert common.layerwise_prefill_callback_metadata is callbacks
        assert [(cb.row.kv_group, cb.row.row_ordinal, cb.row.bank) for cb in callbacks] == [(0, 78, 0), (1, 21, 1)]
        assert for_execution.call_count == 79
        assert sorted(args.args[0].execution_ordinal for args in for_execution.call_args_list) == list(range(79))
        if previous_callbacks is not None:
            assert callbacks == previous_callbacks
            assert all(current is not previous for current, previous in zip(callbacks, previous_callbacks))
        previous_callbacks = callbacks
        assert len(regular) == 79
        for execution in runner.kv_cache_config.dsa_kv_topology.executions:
            metadata = regular[execution.latent.layer_name]
            rows = metadata.layerwise_prefill_callback_metadata
            assert rows[0].row is execution.latent
            if execution.indexer is None:
                assert len(rows) == 1
                assert metadata.indexer_block_table is None
                assert metadata.indexer_slot_mapping is None
            else:
                assert len(rows) == 2
                assert rows[1].row is execution.indexer

        assert common.block_table_tensor[0, :4].tolist() == [3, 4, 5, 6]
        assert common.indexer_block_table_tensor[0, :4].tolist() == [17, 18, 19, 20]
        assert common.slot_mapping[:2].tolist() == [3, 4]
        assert common.indexer_slot_mapping[:2].tolist() == [17, 18]
        if prepare == "unpadded":
            common, indices = proposer.prepare_inputs(common, [[11, 12]], [1])
            assert indices.tolist() == [0, 1]
        elif prepare == "padded":
            common, indices, _, _ = proposer.prepare_inputs_padded(
                common,
                SimpleNamespace(cu_num_draft_tokens=torch.tensor([1], dtype=torch.int32)),
                torch.tensor([2], dtype=torch.int32),
            )
            assert indices.tolist() == [0, 1]
        assert common.layerwise_prefill_callback_metadata is callbacks

        proposer._runnable = MagicMock(return_value=torch.tensor([[11, 12, 13]]))
        with patch.object(proposer, "shallow_copy_metadata", wraps=proposer.shallow_copy_metadata) as shallow_copy:
            proposer._propose(
                target_token_ids=torch.tensor([1, 2]),
                target_positions=torch.tensor([0, 1]),
                target_hidden_states=torch.zeros((2, 2)),
                next_token_ids=torch.tensor([11]),
                token_indices_to_sample=None,
                common_attn_metadata=common,
                target_model_batch_desc=None,
                sampling_metadata=None,
            )
        assert shallow_copy.call_count == 2
        drafts = proposer._runnable.call_args.kwargs["multi_steps_attn_metadata"]
        assert len(drafts) == 3
        for step, draft in enumerate(drafts):
            metadata = draft["latent.78"]
            assert metadata.layerwise_prefill_callback_metadata is callbacks
            assert metadata.layerwise_prefill_callback_metadata[0] is callbacks[0]
            assert metadata.layerwise_prefill_callback_metadata[1] is callbacks[1]
            assert metadata.slot_mapping.data_ptr() == proposer.slot_mapping_group[step].data_ptr()
            assert metadata.indexer_slot_mapping.data_ptr() == proposer.indexer_slot_mapping_group[step].data_ptr()
            assert metadata.block_table[0, :4].tolist() == [3, 4, 5, 6]
            assert metadata.indexer_block_table[0, :4].tolist() == [17, 18, 19, 20]
            assert metadata.slot_mapping.tolist() == ([3, 4] if step == 0 else [4 + step, -1])
            assert metadata.indexer_slot_mapping.tolist() == ([17, 18] if step == 0 else [18 + step, -1])
        assert proposer.slot_mapping_group[0][:2].tolist() == [3, 4]
        assert proposer.indexer_slot_mapping_group[0][:2].tolist() == [17, 18]


def test_mtp_draft_steps_own_independent_indexer_slot_mappings() -> None:
    proposer = SpecDecodeBaseProposer.__new__(SpecDecodeBaseProposer)
    proposer.pcp_size = 1
    proposer.dcp_size = 1
    proposer.uses_mrope = False
    proposer.kernel_block_size = 4
    proposer.max_model_len = 64
    proposer.method = "mtp"
    proposer.arange = torch.arange(3, dtype=torch.int32)
    proposer.token_arange_np = torch.arange(3, dtype=torch.int32).numpy()
    proposer.slot_mapping_group = [
        torch.full((4,), -1, dtype=torch.int32),
        torch.full((4,), -1, dtype=torch.int32),
    ]
    proposer.indexer_slot_mapping_group = [
        torch.full((4,), 7, dtype=torch.int32),
        torch.full((4,), -1, dtype=torch.int32),
    ]
    old_common = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([1], dtype=torch.int32),
        num_computed_tokens_cpu=torch.tensor([0], dtype=torch.int32),
        positions=torch.tensor([0], dtype=torch.int32),
        block_table_tensor=torch.tensor([[3]], dtype=torch.int32),
        slot_mapping=proposer.slot_mapping_group[0],
        indexer_block_table_tensor=torch.tensor([[8]], dtype=torch.int32),
        indexer_slot_mapping=proposer.indexer_slot_mapping_group[0],
        layerwise_prefill_callback_metadata=(object(),),
    )
    builder = MagicMock()
    builder.build_for_drafting.side_effect = (
        lambda common_attn_metadata, draft_index: SimpleNamespace(
            indexer_slot_mapping=common_attn_metadata.indexer_slot_mapping,
            draft_index=draft_index,
        )
    )
    attn_group = MagicMock()
    attn_group.get_metadata_builder.return_value = builder

    common, metadata = proposer.attn_update_stack_num_spec_norm(
        draft_step=1,
        old_attn_metadata=object(),
        old_common_metadata=old_common,
        batch_size=1,
        input_batch_size=1,
        used_update_positions=torch.tensor([0], dtype=torch.int32),
        aclgraph_runtime_mode=CUDAGraphMode.NONE,
        attn_group=attn_group,
    )

    assert common.indexer_slot_mapping is proposer.indexer_slot_mapping_group[1]
    assert metadata.indexer_slot_mapping is proposer.indexer_slot_mapping_group[1]
    assert common.indexer_slot_mapping.tolist() == [33, -1, -1, -1]
    assert proposer.indexer_slot_mapping_group[0].tolist() == [7, 7, 7, 7]


@pytest.mark.parametrize("execution_ordinal", [3, 4, 5])
def test_shared_consumers_have_no_group_1_lowering_or_callback(
    execution_ordinal: int,
) -> None:
    execution = _topology().executions[execution_ordinal]
    common = SimpleNamespace(
        block_table_tensor=object(),
        slot_mapping=object(),
        indexer_block_table_tensor=object(),
        indexer_slot_mapping=object(),
    )
    latent_tensors = (object(), object())
    getter = MagicMock(return_value=latent_tensors)

    lowered = NPUModelRunner._layerwise_prefill_common_attn_metadata(
        common,
        execution,
        (("request", 9),),
        getter,
    )

    assert execution.indexer is None
    getter.assert_called_once_with(0, execution.latent.bank)
    assert lowered.block_table_tensor is latent_tensors[0]
    assert lowered.slot_mapping is latent_tensors[1]
    assert lowered.indexer_block_table_tensor is None
    assert lowered.indexer_slot_mapping is None
    assert len(lowered.layerwise_prefill_callback_metadata) == 1
    assert lowered.layerwise_prefill_callback_metadata[0].row is execution.latent


def test_bank_refresh_uses_stage_2_request_metadata_for_table_and_slots() -> None:
    runner = _runner()
    primary = MagicMock()
    bank_1 = MagicMock()
    runner.input_batch = SimpleNamespace(
        req_ids=["request-b", "request-a"],
        block_table=primary,
        layerwise_prefill_block_tables=(primary, bank_1),
    )
    runner.requests = {
        "request-a": SimpleNamespace(
            block_ids=([1], [2]),
            block_ids_by_bank=(([1], [2]), ([11], [12])),
            block_allocation_mode=DSABlockAllocationMode.PREFILL_CHILD,
            allocation_generation=7,
        ),
        "request-b": SimpleNamespace(
            block_ids=([3], [4]),
            block_ids_by_bank=(([3], [4]), ([13], [14])),
            block_allocation_mode=DSABlockAllocationMode.PREFILL_CHILD,
            allocation_generation=8,
        ),
    }

    tables = runner._refresh_layerwise_prefill_block_tables()

    assert tables == (primary, bank_1)
    assert bank_1.add_row.call_args_list == [
        call(([13], [14]), 0),
        call(([11], [12]), 1),
    ]


def test_missing_generation_fails_before_runner_state_mutation() -> None:
    runner = _runner()
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE)
    runner.dsa_kv_topology = _topology()
    runner._resident_state_registry = MagicMock()
    runner.requests = {}
    connector = SimpleNamespace(
        supports_layerwise_prefill_p_node=True,
        wait_for_layerwise_prefill_load=lambda _metadata: None,
        save_layerwise_prefill_kv_layer=lambda *_args: None,
    )
    new_request = SimpleNamespace(
        req_id="request",
        block_ids=([1], [2]),
        block_ids_by_bank=(([1], [2]), ([11], [12])),
        block_allocation_mode=DSABlockAllocationMode.PREFILL_CHILD,
        allocation_generation=None,
    )
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[new_request],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        finished_req_ids={"finished"},
    )

    with (
        patch.object(model_runner_module, "has_kv_transfer_group", return_value=True),
        patch.object(model_runner_module, "is_v1_kv_transfer_group", return_value=True),
        patch.object(model_runner_module, "get_kv_transfer_group", return_value=connector),
        pytest.raises(RuntimeError, match="allocation generation"),
    ):
        runner._update_states(scheduler_output)

    runner._resident_state_registry.release.assert_not_called()


def test_resumed_request_must_roll_generation_before_state_mutation() -> None:
    runner = _runner()
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE)
    runner.dsa_kv_topology = _topology()
    runner._resident_state_registry = MagicMock()
    runner.requests = {
        "request": SimpleNamespace(
            block_ids=([1], [2]),
            block_ids_by_bank=(([1], [2]), ([11], [12])),
            block_allocation_mode=DSABlockAllocationMode.PREFILL_CHILD,
            allocation_generation=7,
        )
    }
    connector = SimpleNamespace(
        supports_layerwise_prefill_p_node=True,
        wait_for_layerwise_prefill_load=lambda _metadata: None,
        save_layerwise_prefill_kv_layer=lambda *_args: None,
    )
    cached = SimpleNamespace(
        req_ids=["request"],
        resumed_req_ids={"request"},
        new_block_ids=[([3], [4])],
        new_block_ids_by_bank=[(([3], [4]), ([13], [14]))],
        new_block_allocation_modes=[DSABlockAllocationMode.PREFILL_CHILD],
        allocation_generations=[7],
    )
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        finished_req_ids={"finished"},
    )

    with (
        patch.object(model_runner_module, "has_kv_transfer_group", return_value=True),
        patch.object(model_runner_module, "is_v1_kv_transfer_group", return_value=True),
        patch.object(model_runner_module, "get_kv_transfer_group", return_value=connector),
        pytest.raises(RuntimeError, match="reused its allocation generation"),
    ):
        runner._update_states(scheduler_output)

    runner._resident_state_registry.release.assert_not_called()


@pytest.mark.parametrize("failure", ["topology", "connector"])
def test_runtime_prerequisites_fail_before_state_mutation(failure: str) -> None:
    runner = _runner()
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE)
    runner.dsa_kv_topology = None if failure == "topology" else _topology()
    runner._resident_state_registry = MagicMock()
    runner.requests = {}
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        finished_req_ids={"finished"},
    )

    with (
        patch.object(model_runner_module, "has_kv_transfer_group", return_value=False),
        pytest.raises(RuntimeError),
    ):
        runner._update_states(scheduler_output)

    runner._resident_state_registry.release.assert_not_called()


def test_runtime_rejects_missing_connector_topology_and_piecewise() -> None:
    runner = _runner()
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE)
    topology = _topology()

    with pytest.raises(RuntimeError, match="canonical two-group"):
        runner._validate_layerwise_prefill_runtime(None)

    runner.dsa_kv_topology = topology
    with (
        patch.object(model_runner_module, "has_kv_transfer_group", return_value=False),
        pytest.raises(RuntimeError, match="v1 KV connector"),
    ):
        runner._validate_layerwise_prefill_runtime()

    runner.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    with pytest.raises(RuntimeError, match="rejects FULL"):
        runner._validate_layerwise_prefill_runtime()


def test_piecewise_fixed_addresses_are_stable_across_refresh() -> None:
    runner = _runner()
    input_batch = NPUInputBatch(
        max_num_reqs=2,
        max_model_len=8,
        max_num_batched_tokens=8,
        device=torch.device("cpu"),
        pin_memory=False,
        vocab_size=32,
        block_sizes=[2, 4],
        kernel_block_sizes=[[2], [4]],
        max_num_blocks_per_req=[4, 2],
        layerwise_prefill_p_node=True,
    )
    runner.input_batch = input_batch

    runner._validate_layerwise_prefill_piecewise_addresses()
    runner._validate_layerwise_prefill_piecewise_addresses()
    # Refresh bank 1 in place: content changes, storage addresses do not.
    input_batch.layerwise_prefill_block_tables[1].add_row(
        ([10, 11], [20]),
        row_idx=0,
    )
    runner._validate_layerwise_prefill_piecewise_addresses()

    # A rebind to fresh tables must fail closed.
    rebound = NPUInputBatch(
        max_num_reqs=2,
        max_model_len=8,
        max_num_batched_tokens=8,
        device=torch.device("cpu"),
        pin_memory=False,
        vocab_size=32,
        block_sizes=[2, 4],
        kernel_block_sizes=[[2], [4]],
        max_num_blocks_per_req=[4, 2],
        layerwise_prefill_p_node=True,
    )
    runner.input_batch = rebound
    with pytest.raises(RuntimeError, match="rebound"):
        runner._validate_layerwise_prefill_piecewise_addresses()


def test_piecewise_initialization_seals_final_two_group_batch() -> None:
    runner = _runner()
    config = _global_slab_config()
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.PIECEWISE)
    runner.cache_config = SimpleNamespace(block_size=1)
    runner.offload_config = SimpleNamespace(uva=SimpleNamespace(cpu_offload_gb=0))
    runner.model_config = SimpleNamespace(get_vocab_size=lambda: 32, enable_return_routed_experts=False)
    runner.max_model_len = runner.max_encoder_len = runner.max_num_tokens = 8
    runner.max_num_reqs = 2
    runner.sparse_head_dim = (8, 1)
    runner.pin_memory = False
    runner.is_pooling_model = False
    runner.speculative_config = runner.vllm_config.speculative_config = None
    runner._profiling_cudagraph_memory = False
    runner.input_batch = NPUInputBatch(
        max_num_reqs=2,
        max_model_len=8,
        max_num_batched_tokens=8,
        device=torch.device("cpu"),
        pin_memory=False,
        vocab_size=32,
        block_sizes=[1],
        kernel_block_sizes=[[1]],
        max_num_blocks_per_req=[8],
        layerwise_prefill_p_node=True,
    )
    temporary_batch = runner.input_batch
    temporary_addresses = runner._layerwise_prefill_table_addresses()
    backend = SimpleNamespace(get_supported_kernel_block_sizes=lambda: [1])
    runner.attn_groups = [
        [SimpleNamespace(kv_cache_spec=group.kv_cache_spec, backend=backend)] for group in config.kv_cache_groups
    ]
    connector = SimpleNamespace(
        supports_layerwise_prefill_p_node=True,
        supports_dsa_index_lmcache=True,
        wait_for_layerwise_prefill_load=lambda _: None,
        save_layerwise_prefill_kv_layer=lambda *_: None,
        register_kv_caches=MagicMock(),
    )
    with (
        patch.object(model_runner_module, "has_kv_transfer_group", return_value=True),
        patch.object(model_runner_module, "is_v1_kv_transfer_group", return_value=True),
        patch.object(model_runner_module, "get_kv_transfer_group", return_value=connector),
        patch.object(model_runner_module, "staged_sfa_graph_configured", return_value=False),
        patch.object(model_runner_module, "get_total_cp_world_size", return_value=1),
        patch.object(runner, "_log_layerwise_prefill_startup"),
        patch.object(runner, "_validate_sfa_layerwise_connector_cudagraph_mode"),
        patch.object(runner, "_validate_and_cache_dsa_kv_topology"),
        patch.object(runner, "may_add_encoder_only_layers_to_kv_cache_config"),
        patch.object(runner, "maybe_add_kv_sharing_layers_to_kv_cache_groups"),
        patch.object(runner, "initialize_attn_backend"),
        patch.object(runner, "initialize_kv_cache_tensors", return_value={}),
        patch.object(runner, "_maybe_init_dsa_latent_offload"),
    ):
        runner.initialize_kv_cache(config)
        assert runner.input_batch is not temporary_batch
        addresses = runner._layerwise_prefill_table_addresses()
        assert addresses != temporary_addresses
        assert all(len(bank) == 2 for bank in addresses)
        assert runner._layerwise_prefill_recorded_addresses == addresses
        runner._validate_layerwise_prefill_runtime(config.dsa_kv_topology)
        final_batch = runner.input_batch
        with pytest.raises(RuntimeError, match="cannot be reinitialized"):
            runner.initialize_kv_cache(config)
        assert runner.input_batch is final_batch


def test_runtime_rejects_a_consumer_only_connector_role() -> None:
    runner = _runner()
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE)
    runner.dsa_kv_topology = _topology()
    runner.vllm_config.kv_transfer_config.is_kv_producer = False

    with (
        patch.object(model_runner_module, "has_kv_transfer_group", return_value=True),
        patch.object(model_runner_module, "is_v1_kv_transfer_group", return_value=True),
        pytest.raises(RuntimeError, match="requires a KV producer role"),
    ):
        runner._validate_layerwise_prefill_runtime()


def test_feature_off_does_not_query_connector_or_shadow_banks() -> None:
    runner = _runner()
    runner.layerwise_prefill_p_node = False
    primary = MagicMock()
    runner.input_batch = SimpleNamespace(
        block_table=primary,
        layerwise_prefill_block_tables=(primary,),
    )

    with patch.object(model_runner_module, "has_kv_transfer_group") as has_connector:
        assert runner._validate_layerwise_prefill_runtime() is None
        assert runner._refresh_layerwise_prefill_block_tables() == (primary,)

    has_connector.assert_not_called()


@pytest.mark.parametrize("window", [False, True])
@pytest.mark.parametrize("abort_kind", ["missing", "noncallable", "base_noop", "concrete", "inherited_concrete"])
def test_runtime_transfer_window_requires_concrete_abort(monkeypatch, window, abort_kind) -> None:
    runner = _runner()
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE)
    runner.dsa_kv_topology = _topology()

    def base_abort(self):
        pass

    def concrete_abort(self):
        raise AssertionError("runtime validation must not call abort")

    monkeypatch.setattr(
        model_runner_module.KVConnectorBase_V1, "abort_layerwise_prefill_step", base_abort, raising=False
    )

    class ParentConnector(SimpleNamespace):
        if abort_kind == "inherited_concrete":
            abort_layerwise_prefill_step = concrete_abort
        elif abort_kind == "base_noop":
            abort_layerwise_prefill_step = model_runner_module.KVConnectorBase_V1.abort_layerwise_prefill_step

    class Connector(ParentConnector):
        if abort_kind == "concrete":
            abort_layerwise_prefill_step = concrete_abort
        elif abort_kind == "noncallable":
            abort_layerwise_prefill_step = None

    connector = Connector(
        supports_layerwise_prefill_p_node=True,
        supports_layerwise_prefill_transfer_window=window,
        wait_for_layerwise_prefill_load=lambda _: None,
        save_layerwise_prefill_kv_layer=lambda *_: None,
    )
    monkeypatch.setattr(model_runner_module, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(model_runner_module, "is_v1_kv_transfer_group", lambda: True)
    monkeypatch.setattr(model_runner_module, "get_kv_transfer_group", lambda: connector)

    if window and abort_kind in ("missing", "noncallable", "base_noop"):
        with pytest.raises(RuntimeError, match="concrete abort_layerwise_prefill_step"):
            runner._validate_layerwise_prefill_runtime()
    else:
        assert runner._validate_layerwise_prefill_runtime() is connector
