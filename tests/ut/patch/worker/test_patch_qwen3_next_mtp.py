# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.patch.worker.patch_qwen3_next_mtp import bind_kv_cache


@pytest.mark.parametrize("indexer_first", [False, True])
def test_bind_kv_cache_keeps_same_index_latent_and_indexer_caches(indexer_first):
    latent_name = "model.layers.7.self_attn.attn"
    indexer_name = "model.layers.7.self_attn.indexer.k_cache"
    backing = torch.empty(2)
    latent_cache = backing[:1]
    indexer_cache = backing[1:]
    cache_items = [
        (latent_name, latent_cache),
        (indexer_name, indexer_cache),
    ]
    if indexer_first:
        cache_items.reverse()
    kv_caches = dict(cache_items)
    forward_context = {
        latent_name: SimpleNamespace(),
        indexer_name: SimpleNamespace(),
    }
    runner_kv_caches = []

    bind_kv_cache(kv_caches, forward_context, runner_kv_caches)

    assert len(runner_kv_caches) == 2
    assert runner_kv_caches[0] is latent_cache
    assert runner_kv_caches[1] is indexer_cache
    assert latent_cache.untyped_storage().data_ptr() == indexer_cache.untyped_storage().data_ptr()
    assert forward_context[latent_name].kv_cache[0] is latent_cache
    assert forward_context[indexer_name].kv_cache[0] is indexer_cache


def test_bind_kv_cache_preserves_single_glm51_cache_behavior():
    layer_name = "model.layers.7.self_attn.attn"
    kv_cache = torch.empty(1)
    forward_context = {layer_name: SimpleNamespace()}
    runner_kv_caches = []

    bind_kv_cache({layer_name: kv_cache}, forward_context, runner_kv_caches)

    assert len(runner_kv_caches) == 1
    assert runner_kv_caches[0] is kv_cache
    assert forward_context[layer_name].kv_cache[0] is kv_cache


@pytest.mark.parametrize(
    ("first_name", "second_name"),
    [
        ("model.layers.7.self_attn", "model.layers.7.cross_attn"),
        (
            "model.layers.7.cross_attn.attn",
            "model.layers.7.self_attn.indexer.k_cache",
        ),
    ],
    ids=["bart", "non-sibling-indexer"],
)
def test_bind_kv_cache_preserves_non_dsa_duplicate_index_behavior(first_name, second_name):
    first_cache = torch.empty(1)
    second_cache = torch.empty(1)
    kv_caches = {
        first_name: first_cache,
        second_name: second_cache,
    }
    forward_context = {
        first_name: SimpleNamespace(),
        second_name: SimpleNamespace(),
    }
    runner_kv_caches = []

    bind_kv_cache(kv_caches, forward_context, runner_kv_caches)

    assert len(runner_kv_caches) == 1
    assert runner_kv_caches[0] is first_cache
    assert forward_context[first_name].kv_cache[0] is first_cache
    assert forward_context[second_name].kv_cache[0] is second_cache
