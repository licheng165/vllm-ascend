# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from copy import copy
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import CUDAGraphMode

from tests.ut.worker.test_layerwise_prefill_p_node import (
    mtp_metadata_runner as mtp_metadata_runner,
)
from vllm_ascend.attention import sfa_v1 as sfa
from vllm_ascend.ops import rotary_embedding as rope


class _RopeTable:
    def __init__(self, values):
        self.values = values
        self.gathers = 0

    def __getitem__(self, positions):
        self.gathers += 1
        return self.values[positions]


@pytest.fixture
def rope_runner(request, monkeypatch):
    runner = request.getfixturevalue("mtp_metadata_runner")
    values = torch.arange(128, dtype=torch.float32).reshape(32, 4)
    monkeypatch.setattr(rope, "_cos_cache", _RopeTable(values))
    monkeypatch.setattr(rope, "_sin_cache", _RopeTable(-values))
    monkeypatch.setattr(rope, "_cos_mla", torch.empty(4, 1, 1, 4))
    monkeypatch.setattr(rope, "_sin_mla", torch.empty(4, 1, 1, 4))
    monkeypatch.setattr(sfa, "get_cos_and_sin_mla", rope.get_cos_and_sin_mla)
    return runner


def _build(runner, padded=False, **kwargs):
    return runner._build_attention_metadata(
        num_tokens=2,
        num_reqs=1,
        max_query_len=2,
        num_tokens_padded=4 if padded else 2,
        num_reqs_padded=2 if padded else 1,
        **kwargs,
    )


@pytest.mark.parametrize("num_groups", [1, 3])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE])
def test_p_node_builds_rope_once_per_group(rope_runner, num_groups, padded, mode):
    runner = rope_runner
    runner.vllm_config.compilation_config.cudagraph_mode = mode
    group = runner.attn_groups[0][0]
    builder = group.get_metadata_builder()
    builders = [copy(builder) for _ in range(num_groups)]
    runner.attn_groups[0] = [
        SimpleNamespace(
            layer_names=group.layer_names[index::num_groups],
            get_metadata_builder=lambda *_, builder=builder: builder,
        )
        for index, builder in enumerate(builders)
    ]
    for group_index, builder in enumerate(builders):
        builder.attn_mask_builder = SimpleNamespace(
            get_attention_mask=lambda _, group_index=group_index: torch.tensor(group_index)
        )
        builder.build = MagicMock(wraps=builder.build)

    # Inference tensors have no version counter. Reusing their identity/storage
    # across forwards must never serve as a position-equality cache key.
    with torch.inference_mode():
        runner.positions.gpu = runner.positions.gpu.clone()
        previous = None
        address = runner.positions.gpu.data_ptr()
        for step in range(3):
            runner.positions.gpu.copy_(torch.tensor([2 * step, 2 * step + 1, 0, 0]))
            metadata, draft_common = _build(runner, padded, use_spec_decode=True)
            assert len(metadata) == len({id(value) for value in metadata.values()}) == 79
            assert runner.positions.gpu.data_ptr() == address
            positions = runner.positions.gpu[: 4 if padded else 2].long()
            expected_cos = rope._cos_cache.values[positions, None, None]
            expected_sin = rope._sin_cache.values[positions, None, None]
            callbacks = []
            for execution in runner.kv_cache_config.dsa_kv_topology.executions:
                value = metadata[execution.latent.layer_name]
                expected_group = group.layer_names.index(execution.latent.layer_name) % num_groups
                torch.testing.assert_close(value.attn_mask, torch.tensor(expected_group))
                torch.testing.assert_close(value.cos, expected_cos)
                torch.testing.assert_close(value.sin, expected_sin)
                assert value.cos.data_ptr() == rope._cos_mla.data_ptr()
                assert value.sin.data_ptr() == rope._sin_mla.data_ptr()
                bank = runner.input_batch.layerwise_prefill_block_tables[execution.latent.bank][0]
                torch.testing.assert_close(value.block_table, bank.get_device_tensor()[: value.seq_lens.numel()])
                torch.testing.assert_close(value.slot_mapping[:2], bank.slot_mapping.gpu[:2])
                if execution.indexer is None:
                    assert value.indexer_block_table is value.indexer_slot_mapping is None
                else:
                    indexer = runner.input_batch.layerwise_prefill_block_tables[execution.indexer.bank][1]
                    torch.testing.assert_close(
                        value.indexer_block_table, indexer.get_device_tensor()[: value.seq_lens.numel()]
                    )
                    torch.testing.assert_close(value.indexer_slot_mapping[:2], indexer.slot_mapping.gpu[:2])
                rows = value.layerwise_prefill_callback_metadata
                assert rows[0].row is execution.latent
                if execution.indexer is not None:
                    assert rows[1].row is execution.indexer
                callbacks.extend(rows)
                if previous is not None:
                    old = previous[execution.latent.layer_name].layerwise_prefill_callback_metadata
                    assert rows == old
                    assert all(current is not prior for current, prior in zip(rows, old))
            assert len({id(callback) for callback in callbacks}) == 101
            assert (
                draft_common.layerwise_prefill_callback_metadata
                is metadata["latent.78"].layerwise_prefill_callback_metadata
            )
            metadata["latent.0"].decode_remap_boundary_ready = True
            assert not metadata["latent.1"].decode_remap_boundary_ready
            assert rope._cos_cache.gathers == rope._sin_cache.gathers == (step + 1) * num_groups
            assert sum(builder.build.call_count for builder in builders) == (step + 1) * num_groups
            previous = metadata


@pytest.mark.parametrize("padded", [False, True])
def test_optimized_metadata_matches_independent_layer_builds(rope_runner, padded):
    runner = rope_runner
    optimized, _ = _build(runner, padded)
    assert rope._cos_cache.gathers == rope._sin_cache.gathers == 1
    builder = runner.attn_groups[0][0].get_metadata_builder()
    # Custom metadata factories must use the unoptimized path.
    builder.metadata_cls = lambda **kwargs: sfa.AscendSFAMetadata(**kwargs)
    reference, _ = _build(runner, padded)
    assert rope._cos_cache.gathers == rope._sin_cache.gathers == 1 + 79
    for layer_name, value in optimized.items():
        for field in fields(value):
            actual = getattr(value, field.name)
            expected = getattr(reference[layer_name], field.name)
            if isinstance(actual, torch.Tensor):
                torch.testing.assert_close(actual, expected)
            else:
                assert actual == expected, (layer_name, field.name)


def test_generic_eager_and_capture_keep_building(rope_runner):
    runner = rope_runner
    runner.layerwise_prefill_p_node = False
    runner.max_model_len = 8
    builder = runner.attn_groups[0][0].get_metadata_builder()
    builder.reorder_batch_threshold = 2
    builder.build = MagicMock(wraps=builder.build)
    for capture in (False, True, False):
        runner.positions.gpu.add_(1)
        metadata, _ = _build(runner, for_cudagraph_capture=capture)
        assert metadata["latent.0"] is metadata["latent.78"]
        expected = rope._cos_cache.values[runner.positions.gpu[:2].long(), None, None]
        torch.testing.assert_close(metadata["latent.0"].cos, expected)
    assert builder.build.call_count == rope._cos_cache.gathers == rope._sin_cache.gathers == 3


def test_p_node_capture_does_not_reuse_full_metadata(rope_runner):
    runner = rope_runner
    runner.max_model_len = 8
    runner.attn_groups[0][0].get_metadata_builder().reorder_batch_threshold = 2
    _build(runner, for_cudagraph_capture=True)
    assert rope._cos_cache.gathers == rope._sin_cache.gathers == 79


def test_use_cache_still_gathers_changed_positions(rope_runner):
    positions = torch.tensor([0, 1])
    first_cos, first_sin = rope.get_cos_and_sin_mla(positions, True)
    positions.add_(2)
    second_cos, second_sin = rope.get_cos_and_sin_mla(positions, True)
    assert first_cos.data_ptr() == second_cos.data_ptr()
    assert first_sin.data_ptr() == second_sin.data_ptr()
    torch.testing.assert_close(second_cos, rope._cos_cache.values[positions, None, None])
    torch.testing.assert_close(second_sin, rope._sin_cache.values[positions, None, None])
    assert rope._cos_cache.gathers == rope._sin_cache.gathers == 2


@pytest.mark.parametrize("fallback", ["cp", "shrink", "subclass"])
def test_specialized_builders_are_not_reused(rope_runner, fallback):
    runner = rope_runner
    builder = runner.attn_groups[0][0].get_metadata_builder()
    # Exercise the runner's guard without constructing device CP/decode state.
    if fallback == "subclass":

        class SpecializedBuilder(sfa.AscendSFAMetadataBuilder):
            pass

        builder.__class__ = SpecializedBuilder
    else:
        setattr(builder, "enable_dsa_cp" if fallback == "cp" else "dsa_shrink_latent", True)
    builder.build = MagicMock(side_effect=lambda **_: object())
    metadata, _ = _build(runner)
    assert len(metadata) == builder.build.call_count == 79


def test_failed_build_cannot_retain_a_template(rope_runner, monkeypatch):
    runner = rope_runner
    lower = runner._layerwise_prefill_common_attn_metadata

    def fail_after_first(common, execution, *args):
        if execution.execution_ordinal == 1:
            raise RuntimeError("aborted metadata build")
        return lower(common, execution, *args)

    with monkeypatch.context() as patcher:
        patcher.setattr(runner, "_layerwise_prefill_common_attn_metadata", fail_after_first)
        with pytest.raises(RuntimeError, match="aborted metadata build"):
            _build(runner)
    assert rope._cos_cache.gathers == 1
    runner.positions.gpu.add_(3)
    metadata, _ = _build(runner, padded=True)
    expected = rope._cos_cache.values[runner.positions.gpu.long(), None, None]
    torch.testing.assert_close(metadata["latent.78"].cos, expected)
    assert rope._cos_cache.gathers == rope._sin_cache.gathers == 2


def test_draft_builds_and_uncached_rope_are_outside_target_template(rope_runner):
    runner = rope_runner
    target, common = _build(runner)
    builder = runner.attn_groups[0][0].get_metadata_builder()
    callbacks = target["latent.78"].layerwise_prefill_callback_metadata
    positions = torch.tensor([2, 3], dtype=torch.int32)
    common.positions = positions
    independent = []
    for step in range(3):
        positions.add_(1)
        draft = builder.build_for_drafting(common, draft_index=step)
        assert draft is not target["latent.78"]
        assert draft.layerwise_prefill_callback_metadata is callbacks
        expected = rope._cos_cache.values[positions.long(), None, None]
        torch.testing.assert_close(draft.cos, expected)
        # Callers requesting independent buffers must still get independent
        # gathers, not a view into the target's fixed-address output buffers.
        cos, sin = rope.get_cos_and_sin_mla(positions.long(), use_cache=False)
        independent.append((cos, sin, expected.clone()))
        assert cos.data_ptr() != target["latent.78"].cos.data_ptr()
        assert sin.data_ptr() != target["latent.78"].sin.data_ptr()
    for cos, sin, expected in independent:
        torch.testing.assert_close(cos, expected)
        torch.testing.assert_close(sin, -expected)
    assert len({cos.data_ptr() for cos, _, _ in independent}) == 3
    assert rope._cos_cache.gathers == rope._sin_cache.gathers == 1 + 3 + 3
    runner.positions.gpu.add_(5)
    next_target, _ = _build(runner)
    torch.testing.assert_close(
        next_target["latent.78"].cos,
        rope._cos_cache.values[runner.positions.gpu[:2].long(), None, None],
    )
    assert rope._cos_cache.gathers == rope._sin_cache.gathers == 8
