# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    LayerwisePrefillCallbackMetadata,
)
from vllm.v1.kv_cache_interface import DSAExecutionRow, DSAKVRow

import vllm_ascend.attention.sfa_v1 as sfa_v1
import vllm_ascend.attention.utils as attention_utils
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata


def _producer_callbacks() -> tuple[LayerwisePrefillCallbackMetadata, ...]:
    latent = DSAKVRow("latent.6", 6, 0, 6, 0)
    indexer = DSAKVRow("indexer.6", 6, 1, 3, 1)
    execution = DSAExecutionRow(6, latent, indexer)
    return LayerwisePrefillCallbackMetadata.for_execution(
        execution,
        (("request", 23),),
    )


def _consumer_callbacks() -> tuple[LayerwisePrefillCallbackMetadata, ...]:
    latent = DSAKVRow("latent.3", 3, 0, 3, 1)
    execution = DSAExecutionRow(3, latent, None)
    return LayerwisePrefillCallbackMetadata.for_execution(
        execution,
        (("request", 23),),
    )


def test_eager_forward_wires_callbacks_around_all_kv_and_sfa_work() -> None:
    source = inspect.getsource(AscendSFAImpl.forward)

    ordered_operations = (
        "_wait_for_layerwise_prefill_rows",
        "self.exec_kv(",
        "self._execute_sparse_flash_attention_process(",
        "_submit_layerwise_prefill_saves(",
        "submit_layerwise_prefill_load_to_connector(",
        "_save_layerwise_prefill_rows(",
        "self._v_up_proj(",
        "self.o_proj(attn_output)",
        "_finish_layerwise_prefill_saves(",
    )
    offsets = [source.index(operation) for operation in ordered_operations]
    assert offsets == sorted(offsets)


def test_producer_waits_and_saves_latent_then_indexer_synchronously() -> None:
    callbacks = _producer_callbacks()
    metadata = SimpleNamespace(layerwise_prefill_callback_metadata=callbacks)
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.has_indexer = True
    kv_cache = tuple(torch.empty(1) for _ in range(3))
    events: list[str] = []
    connector = MagicMock()
    connector.supports_layerwise_prefill_p_node = True
    connector.wait_for_layerwise_prefill_load.side_effect = lambda callback: events.append(
        f"wait-{callback.row.kv_group}"
    )
    connector.save_layerwise_prefill_kv_layer.side_effect = lambda callback, _kv_layer, _metadata: events.append(
        f"save-{callback.row.kv_group}"
    )

    with (
        patch.object(attention_utils, "has_kv_transfer_group", return_value=True),
        patch.object(attention_utils, "is_v1_kv_transfer_group", return_value=True),
        patch.object(attention_utils, "get_kv_transfer_group", return_value=connector),
    ):
        active = impl._wait_for_layerwise_prefill_rows("latent.6", metadata)
        events.extend(("exec-kv", "sfa"))
        impl._save_layerwise_prefill_rows(active, kv_cache, metadata)
        events.extend(("v-up", "o-proj"))

    assert events == [
        "wait-0",
        "wait-1",
        "exec-kv",
        "sfa",
        "save-0",
        "save-1",
        "v-up",
        "o-proj",
    ]
    latent_save, indexer_save = connector.save_layerwise_prefill_kv_layer.call_args_list
    assert latent_save.args[1][0] is kv_cache[0]
    assert latent_save.args[1][1] is kv_cache[1]
    assert indexer_save.args[1] == [kv_cache[2]]


def test_shared_consumer_has_no_indexer_wait_or_save() -> None:
    callbacks = _consumer_callbacks()
    metadata = SimpleNamespace(layerwise_prefill_callback_metadata=callbacks)
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.has_indexer = False
    kv_cache = (torch.empty(1), torch.empty(1))
    connector = MagicMock()
    connector.supports_layerwise_prefill_p_node = True

    with (
        patch.object(attention_utils, "has_kv_transfer_group", return_value=True),
        patch.object(attention_utils, "is_v1_kv_transfer_group", return_value=True),
        patch.object(attention_utils, "get_kv_transfer_group", return_value=connector),
    ):
        active = impl._wait_for_layerwise_prefill_rows("latent.3", metadata)
        impl._save_layerwise_prefill_rows(active, kv_cache, metadata)

    connector.wait_for_layerwise_prefill_load.assert_called_once_with(callbacks[0])
    connector.save_layerwise_prefill_kv_layer.assert_called_once()
    save_args = connector.save_layerwise_prefill_kv_layer.call_args.args
    assert save_args[0] is callbacks[0]
    assert save_args[1][0] is kv_cache[0]
    assert save_args[1][1] is kv_cache[1]


def test_shared_consumer_records_the_producer_cache_event() -> None:
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.is_kv_producer = True
    event = MagicMock()
    metadata = SimpleNamespace(reshape_cache_event=event)

    impl._record_reshape_cache_event((torch.empty(1), torch.empty(1)), metadata)

    event.record.assert_called_once_with()


def _transfer_window_connector() -> MagicMock:
    connector = MagicMock()
    connector.supports_layerwise_prefill_p_node = True
    connector.supports_layerwise_prefill_transfer_window = True
    return connector


def test_transfer_window_splits_submit_and_finish_around_hcom() -> None:
    callbacks = _producer_callbacks()
    metadata = SimpleNamespace(layerwise_prefill_callback_metadata=callbacks)
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.has_indexer = True
    kv_cache = tuple(torch.empty(1) for _ in range(3))
    connector = _transfer_window_connector()
    events: list[str] = []
    connector.submit_layerwise_prefill_save.side_effect = (
        lambda callback, _kv_layer, _metadata: events.append(
            f"submit-save-{callback.row.kv_group}"
        )
    )
    connector.submit_layerwise_prefill_load.side_effect = lambda _callback: events.append(
        "submit-load"
    )
    connector.finish_layerwise_prefill_save.side_effect = lambda callback: events.append(
        f"finish-{callback.row.kv_group}"
    )

    with (
        patch.object(attention_utils, "has_kv_transfer_group", return_value=True),
        patch.object(attention_utils, "is_v1_kv_transfer_group", return_value=True),
        patch.object(attention_utils, "get_kv_transfer_group", return_value=connector),
    ):
        assert attention_utils.layerwise_prefill_transfer_window_active() is True
        impl._submit_layerwise_prefill_saves(callbacks, kv_cache, metadata)
        attention_utils.submit_layerwise_prefill_load_to_connector(callbacks[0])
        events.extend(("v-up", "o-proj", "hcom"))
        impl._finish_layerwise_prefill_saves(callbacks)

    assert events == [
        "submit-save-0",
        "submit-save-1",
        "submit-load",
        "v-up",
        "o-proj",
        "hcom",
        "finish-0",
        "finish-1",
    ]
    latent_submit, indexer_submit = (
        connector.submit_layerwise_prefill_save.call_args_list
    )
    assert latent_submit.args[1][0] is kv_cache[0]
    assert latent_submit.args[1][1] is kv_cache[1]
    assert indexer_submit.args[1] == [kv_cache[2]]
    connector.save_layerwise_prefill_kv_layer.assert_not_called()
    connector.submit_layerwise_prefill_load.assert_called_once_with(callbacks[0])


def test_transfer_window_shared_consumer_only_submits_latent() -> None:
    callbacks = _consumer_callbacks()
    metadata = SimpleNamespace(layerwise_prefill_callback_metadata=callbacks)
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.has_indexer = False
    kv_cache = (torch.empty(1), torch.empty(1))
    connector = _transfer_window_connector()

    with (
        patch.object(attention_utils, "has_kv_transfer_group", return_value=True),
        patch.object(attention_utils, "is_v1_kv_transfer_group", return_value=True),
        patch.object(attention_utils, "get_kv_transfer_group", return_value=connector),
    ):
        impl._submit_layerwise_prefill_saves(callbacks, kv_cache, metadata)
        attention_utils.submit_layerwise_prefill_load_to_connector(callbacks[0])
        impl._finish_layerwise_prefill_saves(callbacks)

    connector.submit_layerwise_prefill_save.assert_called_once()
    submit_args = connector.submit_layerwise_prefill_save.call_args.args
    assert submit_args[0] is callbacks[0]
    assert submit_args[1][0] is kv_cache[0]
    assert submit_args[1][1] is kv_cache[1]
    connector.finish_layerwise_prefill_save.assert_called_once_with(callbacks[0])


def test_sync_connectors_do_not_activate_the_transfer_window() -> None:
    connector = MagicMock()
    connector.supports_layerwise_prefill_p_node = True
    connector.supports_layerwise_prefill_transfer_window = False
    callbacks = _producer_callbacks()

    with (
        patch.object(attention_utils, "has_kv_transfer_group", return_value=True),
        patch.object(attention_utils, "is_v1_kv_transfer_group", return_value=True),
        patch.object(attention_utils, "get_kv_transfer_group", return_value=connector),
    ):
        assert attention_utils.layerwise_prefill_transfer_window_active() is False
        with pytest.raises(RuntimeError, match="transfer window"):
            attention_utils.submit_layerwise_prefill_load_to_connector(callbacks[0])
        with pytest.raises(RuntimeError, match="transfer window"):
            attention_utils.finish_layerwise_prefill_save_to_connector(callbacks[0])
        with pytest.raises(RuntimeError, match="transfer window"):
            attention_utils.submit_layerwise_prefill_save_to_connector(
                callbacks[0], [], object()
            )


def test_unpadding_preserves_indexer_mapping_and_callbacks() -> None:
    callbacks = _producer_callbacks()
    indexer_block_table = torch.tensor([[11, 12], [13, 14]])
    indexer_slot_mapping = torch.tensor([21, 22, 23, 24])
    metadata = AscendCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2, 4], dtype=torch.int32),
        seq_lens=torch.tensor([2, 2], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([2, 2], dtype=torch.int32),
        num_computed_tokens_cpu=torch.tensor([0, 0], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=4,
        max_query_len=2,
        max_seq_len=2,
        block_table_tensor=torch.tensor([[1, 2], [3, 4]]),
        slot_mapping=torch.tensor([1, 2, 3, 4]),
        actual_seq_lengths_q=[2, 4],
        positions=torch.arange(4),
        indexer_block_table_tensor=indexer_block_table,
        indexer_slot_mapping=indexer_slot_mapping,
        layerwise_prefill_callback_metadata=callbacks,
    )

    unpadded = metadata.unpadded(num_actual_tokens=2, num_actual_reqs=1)

    assert unpadded.indexer_block_table_tensor is indexer_block_table
    assert unpadded.indexer_slot_mapping is indexer_slot_mapping
    assert unpadded.layerwise_prefill_callback_metadata is callbacks


def test_empty_callback_metadata_is_connector_free() -> None:
    metadata = SimpleNamespace(layerwise_prefill_callback_metadata=())
    impl = AscendSFAImpl.__new__(AscendSFAImpl)

    with (
        patch.object(attention_utils, "has_kv_transfer_group") as has_connector,
        patch.object(sfa_v1, "wait_for_layerwise_prefill_from_connector") as wait,
    ):
        assert impl._wait_for_layerwise_prefill_rows("latent.0", metadata) == ()
        attention_utils.wait_for_layerwise_prefill_from_connector(())

    has_connector.assert_not_called()
    wait.assert_not_called()
