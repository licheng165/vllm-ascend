# Standard
import json
import os
from dataclasses import dataclass
from typing import Any

import regex as re
import torch

# Third Party
from vllm.config import ParallelConfig
from vllm.logger import logger
from vllm.utils.network_utils import get_ip

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend import Backend
from vllm_ascend.distributed.kv_transfer.utils import mooncake_transfer_telemetry as mc_telemetry
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te

DEFAULT_GLOBAL_SEGMENT_SIZE = 1073741824  # 1.0 GiB
DEFAULT_LOCAL_BUFFER_SIZE = 1073741824  # 1.0 GiB


class MooncakeBackend(Backend):
    def __init__(self, parallel_config: ParallelConfig):
        try:
            from mooncake.store import MooncakeDistributedStore  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run vLLM with MooncakeConnector."
            ) from e
        self.config = MooncakeStoreConfig.load_from_env()
        self.store = MooncakeDistributedStore()
        self.rank = parallel_config.rank
        if self.config.protocol == "ascend":
            local_hostname = get_ip()
            # ASCEND_ENABLE_USE_FABRIC_MEM: Enable unified memory address direct transmission scheme
            # and only can be used for 800 I/T A3 series.
            # Required supporting hardware versions are as follows:
            if os.getenv("ASCEND_ENABLE_USE_FABRIC_MEM", "0") != "1":
                transfer_engine = global_te.get_transfer_engine(local_hostname, device_name=None)
                self.local_seg = local_hostname + ":" + str(transfer_engine.get_rpc_port())
                ret = self.store.setup(
                    self.local_seg,
                    self.config.metadata_server,
                    self.config.global_segment_size,
                    self.config.local_buffer_size,
                    self.config.protocol,
                    self.config.device_name,
                    self.config.master_server_address,
                    transfer_engine.get_engine(),
                )
            else:
                self.local_seg = local_hostname
                ret = self.store.setup(
                    self.local_seg,
                    self.config.metadata_server,
                    self.config.global_segment_size,
                    0,
                    self.config.protocol,
                    self.config.device_name,
                    self.config.master_server_address,
                )

        if ret != 0:
            msg = "Initialize mooncake failed."
            logger.error(msg)
            raise RuntimeError(msg)

        self._mc_metrics_enabled = mc_telemetry.is_enabled()
        self._mc_tp_rank: int | None = None
        self._mc_dp_rank_global: int | None = getattr(parallel_config, "data_parallel_rank", None)
        self._mc_dp_rank_local: int | None = getattr(parallel_config, "data_parallel_rank_local", None)
        try:
            from vllm.distributed import get_tensor_model_parallel_rank

            self._mc_tp_rank = get_tensor_model_parallel_rank()
        except Exception:
            self._mc_tp_rank = None

    def set_device(self):
        device = torch.device(f"npu:{self.rank}")
        torch.npu.set_device(device)

    def register_buffer(self, ptrs: list[int], lengths: list[int]):
        if os.getenv("ASCEND_ENABLE_USE_FABRIC_MEM", "0") != "1":
            global_te.register_buffer(ptrs, lengths)

    def _mc_fabric_enabled(self) -> bool:
        return os.getenv("ASCEND_ENABLE_USE_FABRIC_MEM", "0") == "1"

    def _mc_store_common_kwargs(self) -> dict[str, Any]:
        return dict(
            configured_protocol=self.config.protocol,
            fabric_mem_enabled=self._mc_fabric_enabled(),
            tp_rank=self._mc_tp_rank,
            dp_rank_global=self._mc_dp_rank_global,
            dp_rank_local=self._mc_dp_rank_local,
        )

    def exists(self, keys: list[str]) -> list[int]:
        enabled = self._mc_metrics_enabled
        _start_wall = _start_perf = 0
        if enabled:
            _start_wall = mc_telemetry.now_wall_ns()
            _start_perf = mc_telemetry.now_perf_ns()
        result: list[int] | None = None
        _exc_type: str | None = None
        try:
            result = self.store.batch_is_exist(keys)
        except Exception as e:
            _exc_type = type(e).__name__
            raise
        finally:
            if enabled:
                try:
                    _end_perf = mc_telemetry.now_perf_ns()
                    _end_wall = mc_telemetry.now_wall_ns()
                    payload = mc_telemetry.build_store_exists_metric_payload(
                        start_perf_ns=_start_perf,
                        end_perf_ns=_end_perf,
                        start_wall_time_ns=_start_wall,
                        end_wall_time_ns=_end_wall,
                        keys=keys,
                        result=result if _exc_type is None else None,
                        exception_type=_exc_type,
                        **self._mc_store_common_kwargs(),
                    )
                    mc_telemetry.emit_mooncake_transfer_metric(payload)
                except Exception:
                    logger.debug("Failed to build Mooncake EXISTS metric", exc_info=True)
        return result  # type: ignore[return-value]

    def put(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        enabled = self._mc_metrics_enabled
        _start_wall = _start_perf = 0
        input_summary = None
        if enabled:
            input_summary = mc_telemetry.safe_summarize_store_input(keys, addrs, sizes)
            _start_wall = mc_telemetry.now_wall_ns()
            _start_perf = mc_telemetry.now_perf_ns()
        result: list[int] | None = None
        _exc_type: str | None = None
        try:
            res = self.store.batch_put_from_multi_buffers(keys, addrs, sizes)
            result = list(res) if res is not None else None
            for value in res:
                if value < 0:
                    logger.error(f"Failed to put key {keys},res:{res}")
        except Exception as e:
            _exc_type = type(e).__name__
            logger.error(f"Failed to put key {keys},error:{e}")
        finally:
            if enabled:
                try:
                    _end_perf = mc_telemetry.now_perf_ns()
                    _end_wall = mc_telemetry.now_wall_ns()
                    payload = mc_telemetry.build_store_put_metric_payload(
                        start_perf_ns=_start_perf,
                        end_perf_ns=_end_perf,
                        start_wall_time_ns=_start_wall,
                        end_wall_time_ns=_end_wall,
                        input_summary=input_summary,
                        result=result,
                        exception_type=_exc_type,
                        **self._mc_store_common_kwargs(),
                    )
                    mc_telemetry.emit_mooncake_transfer_metric(payload)
                except Exception:
                    logger.debug("Failed to build Mooncake PUT metric", exc_info=True)

    def get(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        enabled = self._mc_metrics_enabled
        _start_wall = _start_perf = 0
        input_summary = None
        if enabled:
            input_summary = mc_telemetry.safe_summarize_store_input(keys, addrs, sizes)
            _start_wall = mc_telemetry.now_wall_ns()
            _start_perf = mc_telemetry.now_perf_ns()
        result: list[int] | None = None
        _exc_type: str | None = None
        try:
            res = self.store.batch_get_into_multi_buffers(keys, addrs, sizes)
            result = list(res) if res is not None else None
            for value in res:
                if value < 0:
                    logger.error(f"Failed to get key {keys}, res:{res}")
        except Exception as e:
            _exc_type = type(e).__name__
            logger.error(f"Failed to get key {keys}, error:{e}")
        finally:
            if enabled:
                try:
                    _end_perf = mc_telemetry.now_perf_ns()
                    _end_wall = mc_telemetry.now_wall_ns()
                    payload = mc_telemetry.build_store_get_metric_payload(
                        start_perf_ns=_start_perf,
                        end_perf_ns=_end_perf,
                        start_wall_time_ns=_start_wall,
                        end_wall_time_ns=_end_wall,
                        input_summary=input_summary,
                        result=result,
                        exception_type=_exc_type,
                        **self._mc_store_common_kwargs(),
                    )
                    mc_telemetry.emit_mooncake_transfer_metric(payload)
                except Exception:
                    logger.debug("Failed to build Mooncake GET metric", exc_info=True)


@dataclass
class MooncakeStoreConfig:
    metadata_server: str
    global_segment_size: int | str
    local_buffer_size: int
    protocol: str
    device_name: str
    master_server_address: str

    @staticmethod
    def from_file(file_path: str) -> "MooncakeStoreConfig":
        with open(file_path) as file:
            config = json.load(file)
        return MooncakeStoreConfig(
            metadata_server=config.get("metadata_server"),
            global_segment_size=_parse_global_segment_size(
                config.get("global_segment_size", DEFAULT_GLOBAL_SEGMENT_SIZE)
            ),
            local_buffer_size=_parse_global_segment_size(config.get("local_buffer_size", DEFAULT_LOCAL_BUFFER_SIZE)),
            protocol=config.get("protocol", "ascend"),
            device_name=config.get("device_name", ""),
            master_server_address=config.get("master_server_address"),
        )

    @staticmethod
    def load_from_env() -> "MooncakeStoreConfig":
        config_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if not config_path:
            raise ValueError("The environment variable 'MOONCAKE_CONFIG_PATH' is not set.")
        return MooncakeStoreConfig.from_file(config_path)


def _parse_global_segment_size(value) -> int:
    """
    Parse storage size strings with support for units: GB, MB, KB, B

    Args:
        value: Input value (int, str, or other convertible types)

    Returns:
        int: Size in bytes

    Raises:
        ValueError: For invalid format, missing number, or negative values
        TypeError: For unsupported input types
    """

    if isinstance(value, int):
        return value
    elif not isinstance(value, str):
        try:
            return int(value)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Unsupported type for global_segment_size: {type(value)}") from e

    cleaned_input = value.strip().lower()
    if not cleaned_input:
        raise ValueError("global segment size cannot be empty.")

    UNIT_MULTIPLIERS = {
        "gb": 1024**3,  # 1 GB = 1024^3 bytes
        "mb": 1024**2,  # 1 MB = 1024^2 bytes
        "kb": 1024,  # 1 KB = 1024 bytes
        "b": 1,  # 1 B = 1 byte
    }
    pattern = r"^\s*([\d.]+)\s*(gb|mb|kb|b)?\s*$"
    match = re.match(pattern, cleaned_input)

    if not match:
        raise ValueError(f"Invalid format: '{value}'")

    number_str = match.group(1)
    unit = match.group(2) or "b"

    multiplier = UNIT_MULTIPLIERS[unit]
    return _convert_to_bytes(number_str, multiplier, value)


def _convert_to_bytes(number_str: str, multiplier: int, original_input: str) -> int:
    """
    Convert numeric string to byte count

    Args:
        number_str: Numeric portion of input
        multiplier: Unit conversion factor
        original_input: Original input string (for error messages)

    Returns:
        int: Byte count

    Raises:
        ValueError: For invalid numbers or negative results
    """
    try:
        numeric_value = float(number_str)
    except ValueError:
        raise ValueError(f"Invalid numeric value '{number_str}' in: '{original_input}'")
    # Calculate byte count
    try:
        byte_count = int(numeric_value * multiplier)
    except OverflowError:
        raise ValueError(f"Storage size too large: '{original_input}'")
    return byte_count
