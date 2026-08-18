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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.layers.attention import MLAAttention
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)

from vllm_ascend.utils import sparse_kv_cache_has_indexer
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
INDEX_HEAD_DIM = 128


def _attn_layer(has_indexer: bool) -> MLAAttention:
    module = MLAAttention.__new__(MLAAttention)
    torch.nn.Module.__init__(module)
    module.impl = SimpleNamespace(has_indexer=has_indexer)
    module.kv_lora_rank = KV_LORA_RANK
    module.qk_rope_head_dim = QK_ROPE_HEAD_DIM
    return module


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


if __name__ == "__main__":
    unittest.main()
