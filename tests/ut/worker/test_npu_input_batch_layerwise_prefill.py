# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch

from vllm_ascend.worker.npu_input_batch import NPUInputBatch


def _make_input_batch(layerwise_prefill: bool) -> NPUInputBatch:
    return NPUInputBatch(
        max_num_reqs=2,
        max_model_len=8,
        max_num_batched_tokens=8,
        device=torch.device("cpu"),
        pin_memory=False,
        vocab_size=32,
        block_sizes=[2, 4],
        kernel_block_sizes=[[2], [4]],
        max_num_blocks_per_req=[4, 2],
        layerwise_prefill_p_node=layerwise_prefill,
    )


def test_feature_off_reuses_only_the_primary_block_table() -> None:
    input_batch = _make_input_batch(layerwise_prefill=False)

    assert input_batch.layerwise_prefill_block_tables == (input_batch.block_table,)


def test_layerwise_prefill_owns_two_fixed_multigroup_bank_tables() -> None:
    input_batch = _make_input_batch(layerwise_prefill=True)
    tables = input_batch.layerwise_prefill_block_tables

    assert len(tables) == 2
    assert tables[0] is input_batch.block_table
    assert tables[0] is not tables[1]
    assert [len(table.block_tables) for table in tables] == [2, 2]

    tables[1].add_row(([10, 11], [20]), row_idx=0)
    assert tables[1][0].num_blocks_per_row[0] == 2
    assert tables[1][1].num_blocks_per_row[0] == 1
    assert tables[0][0].num_blocks_per_row[0] == 0
