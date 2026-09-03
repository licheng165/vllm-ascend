# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Shared-indexer (GLM-5.2) KV cache registration tests.

With the checkpoint-aware construction, shared-consumer layers have
``impl.has_indexer == False``. The model runner must register a
latent-only KV cache spec for them (no indexer key plane), and the
bundled allocation/reshape paths must skip the dsa_k plane for those
layers.
"""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.layers.attention import MLAAttention
from vllm.v1.core.kv_cache_utils import build_dsa_kv_topology
from vllm.v1.kv_cache_interface import (
    DSAKVRegistration,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MLAAttentionSpec,
)

import vllm_ascend.worker.model_runner_v1 as model_runner_module
from vllm_ascend.utils import sparse_kv_cache_has_indexer
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
INDEX_HEAD_DIM = 128


def _attn_layer(has_indexer: bool, execution_ordinal: int = 3) -> MLAAttention:
    module = MLAAttention.__new__(MLAAttention)
    torch.nn.Module.__init__(module)
    module.impl = SimpleNamespace(has_indexer=has_indexer)
    module.kv_lora_rank = KV_LORA_RANK
    module.qk_rope_head_dim = QK_ROPE_HEAD_DIM
    module.get_kv_cache_spec = MagicMock(
        return_value=MLAAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
            dtype=torch.float32,
            dsa_kv_registration=DSAKVRegistration(execution_ordinal, 0),
        )
    )
    return module


class DeepseekV32IndexerCache:
    def __init__(self, execution_ordinal: int):
        self.dsa_kv_registration = DSAKVRegistration(execution_ordinal, 1)
        self.get_kv_cache_spec = MagicMock(
            return_value=MLAAttentionSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=INDEX_HEAD_DIM,
                dtype=torch.float32,
                dsa_kv_registration=DSAKVRegistration(execution_ordinal, 1),
            )
        )


def _attn_group(layer_name: str, spec):
    return SimpleNamespace(
        layer_names=[layer_name],
        kv_cache_spec=spec,
        backend=SimpleNamespace(
            get_kv_cache_shape=lambda num_blocks, block_size, num_kv_heads, head_size: (
                num_blocks,
                block_size,
                num_kv_heads,
                head_size,
            )
        ),
    )


class _RunnerMixin:
    def _build_runner(self, *, use_sparse_c8_indexer: bool = False):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.use_sparse = True
        runner.use_hybrid_blocks = False
        runner.hybrid_with_attn_and_mamba = False
        runner.block_size = 16
        runner.sparse_head_dim = (KV_LORA_RANK, QK_ROPE_HEAD_DIM, INDEX_HEAD_DIM)
        runner.kv_cache_dtype = torch.float32
        runner.shared_kv_cache_layers = {}
        runner.dsa_unbundle = False
        runner.dsa_two_groups = False
        runner.dsa_free_paged = False
        runner.dsa_shared_pool = False
        runner.use_sparse_c8_indexer = use_sparse_c8_indexer
        runner.ascend_config = MagicMock()
        runner.runner_only_attn_layers = set()
        runner.is_kv_consumer = False
        runner.device = torch.device("cpu")
        runner.vllm_config = MagicMock()
        runner.model_config = SimpleNamespace(
            hf_text_config=SimpleNamespace(
                kv_lora_rank=KV_LORA_RANK,
                qk_rope_head_dim=QK_ROPE_HEAD_DIM,
                index_head_dim=INDEX_HEAD_DIM,
                num_hidden_layers=78,
            )
        )
        runner.vllm_config.cache_config.cache_dtype = "auto"
        return runner


class TestSparseKVCacheHasIndexer(unittest.TestCase):
    def test_producer_spec_has_indexer(self):
        spec = SimpleNamespace(sparse_head_dim=(KV_LORA_RANK, QK_ROPE_HEAD_DIM, INDEX_HEAD_DIM))
        self.assertTrue(sparse_kv_cache_has_indexer(spec))

    def test_consumer_spec_has_no_indexer(self):
        spec = SimpleNamespace(sparse_head_dim=(KV_LORA_RANK, QK_ROPE_HEAD_DIM, 0))
        self.assertFalse(sparse_kv_cache_has_indexer(spec))

    def test_non_sparse_spec_has_no_indexer(self):
        self.assertFalse(sparse_kv_cache_has_indexer(SimpleNamespace(sparse_head_dim=None)))
        self.assertFalse(sparse_kv_cache_has_indexer(SimpleNamespace()))


@patch("vllm_ascend.worker.model_runner_v1.has_ec_transfer", return_value=False)
@patch("vllm_ascend.worker.model_runner_v1.get_layers_from_vllm_config")
class TestGetKVCacheSpecSharedIndexer(_RunnerMixin, unittest.TestCase):
    def _run(self, mock_get_layers, has_indexer: bool):
        runner = self._build_runner()
        layer_name = "model.layers.3.self_attn.attn"
        mock_get_layers.return_value = {layer_name: _attn_layer(has_indexer)}
        return runner.get_kv_cache_spec()[layer_name]

    def test_consumer_registers_latent_only_spec(self, mock_get_layers, _mock_ec):
        spec = self._run(mock_get_layers, has_indexer=False)
        self.assertEqual(spec.sparse_head_dim, (KV_LORA_RANK, QK_ROPE_HEAD_DIM, 0))
        self.assertEqual(spec.head_size, KV_LORA_RANK + QK_ROPE_HEAD_DIM)
        self.assertFalse(spec.cache_sparse_c8)

    def test_producer_registers_full_spec(self, mock_get_layers, _mock_ec):
        spec = self._run(mock_get_layers, has_indexer=True)
        self.assertEqual(
            spec.sparse_head_dim,
            (KV_LORA_RANK, QK_ROPE_HEAD_DIM, INDEX_HEAD_DIM),
        )
        self.assertEqual(spec.head_size, KV_LORA_RANK + QK_ROPE_HEAD_DIM + INDEX_HEAD_DIM)

    def test_free_paged_rejects_consumer(self, mock_get_layers, _mock_ec):
        runner = self._build_runner()
        runner.dsa_free_paged = True
        layer_name = "model.layers.3.self_attn.attn"
        mock_get_layers.return_value = {layer_name: _attn_layer(False)}
        with self.assertRaisesRegex(NotImplementedError, "free-paged"):
            runner.get_kv_cache_spec()

    def test_unbundled_specs_preserve_model_registrations(self, mock_get_layers, _mock_ec):
        runner = self._build_runner()
        runner.dsa_unbundle = True
        runner.dsa_two_groups = True
        latent_name = "opaque.latent.cache"
        indexer_name = "opaque.physical.cache"
        mock_get_layers.return_value = {
            latent_name: _attn_layer(True, execution_ordinal=6),
            indexer_name: DeepseekV32IndexerCache(execution_ordinal=78),
        }

        specs = runner.get_kv_cache_spec()

        self.assertEqual(
            specs[latent_name].dsa_kv_registration,
            DSAKVRegistration(execution_ordinal=6, kv_group=0),
        )
        self.assertEqual(
            specs[indexer_name].dsa_kv_registration,
            DSAKVRegistration(execution_ordinal=78, kv_group=1),
        )


class TestAllocateReshapeSharedIndexer(_RunnerMixin, unittest.TestCase):
    def _allocate_and_reshape(self, *, has_indexer: bool):
        runner = self._build_runner()
        layer_name = "model.layers.3.self_attn.attn"
        if has_indexer:
            sparse_head_dim = (KV_LORA_RANK, QK_ROPE_HEAD_DIM, INDEX_HEAD_DIM)
        else:
            sparse_head_dim = (KV_LORA_RANK, QK_ROPE_HEAD_DIM, 0)
        with patch(
            "vllm_ascend.worker.model_runner_v1.get_layers_from_vllm_config",
            return_value={layer_name: _attn_layer(has_indexer)},
        ):
            spec = runner.get_kv_cache_spec()[layer_name]
            self.assertEqual(spec.sparse_head_dim, sparse_head_dim)

            num_blocks = 2
            kv_cache_config = KVCacheConfig(
                num_blocks=num_blocks,
                kv_cache_tensors=[
                    KVCacheTensor(
                        size=spec.page_size_bytes * num_blocks,
                        shared_by=[layer_name],
                    )
                ],
                kv_cache_groups=[KVCacheGroupSpec(layer_names=[layer_name], kv_cache_spec=spec)],
            )
            raw_caches = runner._allocate_kv_cache_tensors(kv_cache_config)
            with patch.object(
                NPUModelRunner,
                "_kv_cache_spec_attn_group_iterator",
                lambda self: iter([_attn_group(layer_name, spec)]),
            ):
                kv_caches = runner._reshape_kv_cache_tensors(kv_cache_config, raw_caches)
        return spec, raw_caches[layer_name], kv_caches[layer_name]

    def test_consumer_allocates_only_latent_planes(self):
        spec, raws, kv_cache = self._allocate_and_reshape(has_indexer=False)
        # two latent planes, no indexer plane
        self.assertEqual(len(raws), 2)
        self.assertEqual(len(kv_cache), 2)
        k_nope, k_pe = kv_cache
        self.assertEqual(k_nope.shape[-1], KV_LORA_RANK)
        self.assertEqual(k_pe.shape[-1], QK_ROPE_HEAD_DIM)
        self.assertEqual(
            raws[0].numel() + raws[1].numel(),
            spec.page_size_bytes * 2,
        )

    def test_producer_allocates_latent_and_indexer_planes(self):
        spec, raws, kv_cache = self._allocate_and_reshape(has_indexer=True)
        self.assertEqual(len(raws), 3)
        self.assertEqual(len(kv_cache), 3)
        k_nope, k_pe, dsa_k = kv_cache
        self.assertEqual(k_nope.shape[-1], KV_LORA_RANK)
        self.assertEqual(k_pe.shape[-1], QK_ROPE_HEAD_DIM)
        self.assertEqual(dsa_k.shape[-1], INDEX_HEAD_DIM)


def _registered_glm52_specs() -> dict[str, KVCacheSpec]:
    producer_executions = {0, 1, 2} | {6 + 4 * i for i in range(18)} | {78}
    specs: dict[str, KVCacheSpec] = {
        f"latent.execution.{execution}": MLAAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
            dtype=torch.float32,
            dsa_kv_registration=DSAKVRegistration(execution, 0),
        )
        for execution in range(79)
    }
    specs.update(
        {
            f"physical.indexer.{execution}": MLAAttentionSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=INDEX_HEAD_DIM,
                dtype=torch.float32,
                dsa_kv_registration=DSAKVRegistration(execution, 1),
            )
            for execution in producer_executions
        }
    )
    return specs


def _topology_runner_and_config():
    specs = _registered_glm52_specs()
    topology = build_dsa_kv_topology(specs)
    latent_names = [row.layer_name for row in topology.rows_by_group[0]]
    indexer_names = [row.layer_name for row in topology.rows_by_group[1]]
    config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(latent_names, specs[latent_names[0]]),
            KVCacheGroupSpec(indexer_names, specs[indexer_names[0]]),
        ],
        dsa_kv_topology=topology,
    )
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.dsa_two_groups = True
    runner._dsa_kv_specs_by_layer_name = specs
    return runner, config


class TestDSAKVTopologyInitialization(unittest.TestCase):
    def test_validates_and_caches_glm52_topology(self):
        runner, config = _topology_runner_and_config()

        with patch.object(model_runner_module.logger, "info_once") as info_once:
            runner._validate_and_cache_dsa_kv_topology(config)

        self.assertIs(runner.dsa_kv_topology, config.dsa_kv_topology)
        self.assertEqual(len(runner.dsa_kv_rows_by_layer_name), 79 + 22)
        self.assertEqual(len(runner.dsa_kv_executions_by_ordinal), 79)
        execution_6 = runner.dsa_kv_executions_by_ordinal[6]
        self.assertEqual(execution_6.latent.row_ordinal, 6)
        self.assertEqual(execution_6.indexer.row_ordinal, 3)
        execution_78 = runner.dsa_kv_executions_by_ordinal[78]
        self.assertEqual(execution_78.latent.row_ordinal, 78)
        self.assertEqual(execution_78.indexer.row_ordinal, 21)
        info_once.assert_called_once()
        self.assertEqual(info_once.call_args.args[2:], (79, 79, 22))

    def test_missing_topology_fails_closed(self):
        runner, config = _topology_runner_and_config()
        config.dsa_kv_topology = None

        with self.assertRaisesRegex(ValueError, "requires.*topology"):
            runner._validate_and_cache_dsa_kv_topology(config)

    def test_partial_topology_fails_closed(self):
        runner, config = _topology_runner_and_config()
        topology = config.dsa_kv_topology
        assert topology is not None
        config.dsa_kv_topology = replace(
            topology,
            executions=topology.executions[:-1],
            rows_by_group=(
                topology.rows_by_group[0][:-1],
                topology.rows_by_group[1][:-1],
            ),
        )

        with self.assertRaisesRegex(ValueError, "missing registered cache layers"):
            runner._validate_and_cache_dsa_kv_topology(config)

    def test_registration_mismatch_fails_closed(self):
        runner, config = _topology_runner_and_config()
        layer_name = "latent.execution.6"
        runner._dsa_kv_specs_by_layer_name[layer_name] = replace(
            runner._dsa_kv_specs_by_layer_name[layer_name],
            dsa_kv_registration=DSAKVRegistration(7, 0),
        )

        with self.assertRaisesRegex(ValueError, "registration mismatch"):
            runner._validate_and_cache_dsa_kv_topology(config)

    def test_feature_off_ignores_missing_topology(self):
        runner, config = _topology_runner_and_config()
        runner.dsa_two_groups = False
        config.dsa_kv_topology = None

        runner._validate_and_cache_dsa_kv_topology(config)

        self.assertIsNone(runner.dsa_kv_topology)
        self.assertEqual(runner.dsa_kv_rows_by_layer_name, {})
        self.assertEqual(runner.dsa_kv_executions_by_ordinal, {})


if __name__ == "__main__":
    unittest.main()
