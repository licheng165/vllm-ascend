# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm_ascend.dsa_threshold_config import build_dsa_threshold_config


def _make_vllm_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=179840,
            hf_text_config=SimpleNamespace(index_topk=2048),
            model="GLM-5.1",
            dtype="bfloat16",
            quantization="ascend",
        ),
        cache_config=SimpleNamespace(
            block_size=128,
            enable_prefix_caching=False,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        speculative_config=SimpleNamespace(num_speculative_tokens=1),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8,
            pipeline_parallel_size=1,
            enable_expert_parallel=False,
        ),
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "dsa_deployment_mode": "pd",
                "dsa_node_role": "decode",
            }
        ),
    )


def test_build_config_uses_final_geometry_and_lmcache_yaml(tmp_path, monkeypatch) -> None:
    config_file = tmp_path / "lmcache.yaml"
    config_file.write_text("chunk_size: 256\n", encoding="utf-8")
    monkeypatch.setenv("VLLM_ASCEND_DSA_CONTEXT_LENGTH_THRESHOLD", "8192")
    monkeypatch.setenv("LMCACHE_CONFIG_FILE", str(config_file))
    monkeypatch.delenv("LMCACHE_CHUNK_SIZE", raising=False)

    config = build_dsa_threshold_config(_make_vllm_config())

    assert config.block_size == 128
    assert config.chunk_size == 256
    assert config.query_width == 2
    assert config.scratch_capacity == 4096
    assert config.first_reclaiming_frontier == 4352
    assert config.deployment_mode == "pd"
    assert config.node_role == "decode"


def test_build_config_defaults_to_standalone(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_ASCEND_DSA_CONTEXT_LENGTH_THRESHOLD", "0")
    vllm_config = _make_vllm_config()
    vllm_config.kv_transfer_config.kv_connector_extra_config = {}

    config = build_dsa_threshold_config(vllm_config)

    assert config.deployment_mode == "standalone"
    assert config.node_role == "standalone"


@pytest.mark.parametrize(
    "extra_config",
    [
        {"dsa_deployment_mode": "pd"},
        {"dsa_node_role": "prefill"},
        {
            "dsa_deployment_mode": "standalone",
            "dsa_node_role": "decode",
        },
        {"dsa_deployment_mode": "pd", "dsa_node_role": "standalone"},
    ],
)
def test_build_config_rejects_inconsistent_role(
    monkeypatch,
    extra_config: dict[str, str],
) -> None:
    monkeypatch.setenv("VLLM_ASCEND_DSA_CONTEXT_LENGTH_THRESHOLD", "0")
    vllm_config = _make_vllm_config()
    vllm_config.kv_transfer_config.kv_connector_extra_config = extra_config

    with pytest.raises(ValueError, match="Invalid DSA deployment"):
        build_dsa_threshold_config(vllm_config)
