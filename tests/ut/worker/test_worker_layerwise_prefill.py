# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, AsyncModelRunnerOutput

import vllm_ascend.worker.worker as worker_module
from vllm_ascend.worker.worker import NPUWorker


@pytest.fixture
def worker(monkeypatch):
    worker = NPUWorker.__new__(NPUWorker)
    worker._pp_send_work = []
    worker.model_runner = Mock()
    monkeypatch.setattr(worker_module.envs_ascend, "MSMONITOR_USE_DAEMON", False)
    monkeypatch.setattr(worker_module, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True))
    return worker


@pytest.fixture
def transfer(monkeypatch):
    connector = SimpleNamespace(
        supports_layerwise_prefill_transfer_window=True,
        abort_layerwise_prefill_step=Mock(),
        clear_connector_metadata=Mock(),
        metadata=object(),
    )
    has_group = Mock(return_value=True)
    get_group = Mock(return_value=connector)
    synchronize = Mock()
    monkeypatch.setattr(worker_module, "has_kv_transfer_group", has_group)
    monkeypatch.setattr(worker_module, "get_kv_transfer_group", get_group)
    monkeypatch.setattr(torch.npu, "synchronize", synchronize)
    yield SimpleNamespace(connector=connector, has_group=has_group, get_group=get_group)
    synchronize.assert_not_called()
    connector.clear_connector_metadata.assert_not_called()


@pytest.mark.parametrize("method", ["execute_model", "sample_tokens"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_failure_waits_for_pending_d2h_before_reporting_or_next_batch(worker, transfer, method, error_type):
    connector = transfer.connector
    metadata = connector.metadata
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=1)
    grammar_output = object()
    model_error = error_type("model failed with D2H pending")
    pending_d2h = Event()
    abort_entered = Event()
    release_d2h = Event()
    reported = Event()
    next_batch = Event()
    events = []

    def execute_model(*_args):
        assert not pending_d2h.is_set()
        if reported.is_set():
            events.append("next_batch")
            next_batch.set()
            return EMPTY_MODEL_RUNNER_OUTPUT
        pending_d2h.set()
        events.append("target")
        if method == "execute_model":
            raise model_error
        # MTP target forward defers connector finalization until sampling.
        return None

    def sample_tokens(_grammar_output):
        assert torch.is_inference_mode_enabled()
        assert pending_d2h.is_set()
        events.append("sampling")
        raise model_error

    def abort():
        assert connector.metadata is metadata
        assert pending_d2h.is_set()
        if method == "sample_tokens":
            assert torch.is_inference_mode_enabled()
        events.append("abort")
        abort_entered.set()
        assert release_d2h.wait(timeout=5)
        pending_d2h.clear()
        events.append("drained")

    worker.model_runner.execute_model.side_effect = execute_model
    worker.model_runner.sample_tokens.side_effect = sample_tokens
    connector.abort_layerwise_prefill_step.side_effect = abort

    def serialized_calls():
        try:
            result = worker.execute_model(scheduler_output)
            if method == "sample_tokens":
                assert result is None
                connector.abort_layerwise_prefill_step.assert_not_called()
                worker.sample_tokens(grammar_output)
        except BaseException as error:
            assert error is model_error
            assert not pending_d2h.is_set()
            assert connector.metadata is metadata
            events.append("reported")
            reported.set()
        else:
            pytest.fail("worker did not propagate the runner failure")
        assert worker.execute_model(scheduler_output) is EMPTY_MODEL_RUNNER_OUTPUT

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(serialized_calls)
        try:
            assert abort_entered.wait(timeout=5)
            assert pending_d2h.is_set()
            assert not reported.is_set()
            assert not next_batch.is_set()
            assert not future.done()
            assert connector.metadata is metadata
            connector.clear_connector_metadata.assert_not_called()
        finally:
            release_d2h.set()
        future.result(timeout=5)

    expected = ["target"] + (["sampling"] if method == "sample_tokens" else [])
    assert events == [*expected, "abort", "drained", "reported", "next_batch"]
    connector.abort_layerwise_prefill_step.assert_called_once_with()
    assert worker.model_runner.execute_model.call_args_list == [call(scheduler_output, None)] * 2
    if method == "sample_tokens":
        worker.model_runner.sample_tokens.assert_called_once_with(grammar_output)
    else:
        worker.model_runner.sample_tokens.assert_not_called()


@pytest.mark.parametrize("method", ["execute_model", "sample_tokens"])
@pytest.mark.parametrize("capability", [False, None, 1, "missing", "no_group"])
def test_failure_without_transfer_window_has_no_cleanup(worker, transfer, method, capability):
    connector = transfer.connector
    if capability == "missing":
        del connector.supports_layerwise_prefill_transfer_window
    elif capability == "no_group":
        transfer.has_group.return_value = False
    else:
        connector.supports_layerwise_prefill_transfer_window = capability
    error = RuntimeError("runner failure")
    getattr(worker.model_runner, method).side_effect = error

    with pytest.raises(RuntimeError) as raised:
        getattr(worker, method)(SimpleNamespace(total_num_scheduled_tokens=1))

    assert raised.value is error
    transfer.has_group.assert_called_once_with()
    if capability == "no_group":
        transfer.get_group.assert_not_called()
    else:
        transfer.get_group.assert_called_once_with()
    connector.abort_layerwise_prefill_step.assert_not_called()


@pytest.mark.parametrize("method", ["execute_model", "sample_tokens"])
@pytest.mark.parametrize("output_kind", ["none", "output", "async_output"])
def test_success_never_queries_or_aborts_transfer(worker, transfer, method, output_kind):
    output = {
        "none": None,
        "output": EMPTY_MODEL_RUNNER_OUTPUT,
        "async_output": Mock(spec=AsyncModelRunnerOutput),
    }[output_kind]
    getattr(worker.model_runner, method).return_value = output

    assert getattr(worker, method)(SimpleNamespace(total_num_scheduled_tokens=1)) is output

    transfer.has_group.assert_not_called()
    transfer.get_group.assert_not_called()
    transfer.connector.abort_layerwise_prefill_step.assert_not_called()
    if output_kind == "async_output":
        output.get_output.assert_not_called()


@pytest.mark.parametrize("method", ["execute_model", "sample_tokens"])
@pytest.mark.parametrize("abort_error_type", [ValueError, RuntimeError, SystemExit])
def test_abort_error_preserves_model_error_only_when_safely_drained(worker, transfer, method, abort_error_type):
    model_error = RuntimeError("original model failure")
    abort_error = abort_error_type("drained ACK mismatch or fatal unknown fence")
    getattr(worker.model_runner, method).side_effect = model_error
    transfer.connector.abort_layerwise_prefill_step.side_effect = abort_error
    expected = model_error if abort_error_type is ValueError else abort_error

    with pytest.raises(type(expected)) as raised:
        getattr(worker, method)(SimpleNamespace(total_num_scheduled_tokens=1))

    assert raised.value is expected
    if expected is abort_error:
        assert raised.value.__context__ is model_error
    else:
        frames = []
        traceback = raised.value.__traceback__
        while traceback is not None:
            frames.append(traceback.tb_frame.f_code.co_name)
            traceback = traceback.tb_next
        assert method in frames
        assert "_abort_layerwise_prefill_step" not in frames
    transfer.connector.abort_layerwise_prefill_step.assert_called_once_with()
