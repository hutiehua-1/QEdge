"""Exercise original host modules without executing accelerator package imports."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


with patch.dict(os.environ, {k: v for k, v in os.environ.items() if not k.startswith('DNDX_V3_')}, clear=True):
    config = load('qedge_host_config', 'src/config.py')
sampling = load('qedge_host_sampling', 'src/sampling_params.py')
blocks = load('qedge_host_blocks', 'src/data/block_manager.py')
hashes = load('qedge_host_hash', 'src/data/prefix/hash_util.py')
radix = load('qedge_host_radix', 'src/data/prefix/radix_tree.py')
hybrid = load('qedge_host_hybrid', 'src/data/hybrid.py')


class ConfigTests(unittest.TestCase):
    def test_invalid_capacity_and_block_alignment(self):
        for kwargs in [{'block_size':128}, {'max_num_seqs':0}, {'num_kvcache_blocks':-1}, {'gpu_memory_utilization':1.1}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                config.EngineConfig(**kwargs).validate()

    def test_env_boolean_and_overrides(self):
        clean = {k: v for k, v in os.environ.items() if not k.startswith('DNDX_V3_')}
        with patch.dict(os.environ, clean, clear=True):
            with patch.dict(os.environ, {'DNDX_V3_PREFIX_CACHING':'yes','DNDX_V3_BLOCK_SIZE':'512'}):
                value = config.EngineConfig.from_env()
                self.assertTrue(value.enable_prefix_caching)
                self.assertEqual(value.block_size,512)
            with patch.dict(os.environ, {'DNDX_V3_PREFIX_CACHING':'maybe'}):
                with self.assertRaises(ValueError):
                    config.EngineConfig.from_env()

    def test_sampling_rejects_invalid_probabilities_and_budgets(self):
        for kwargs in [{'top_p':0}, {'top_p':1.1}, {'min_p':-0.1}, {'min_tokens':3,'max_tokens':2}, {'temperature':-1}, {'top_k':0}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                sampling.SamplingParams(**kwargs)


class BlockTests(unittest.TestCase):
    def test_shared_page_only_released_after_last_owner(self):
        pool = blocks.BlockManager(1,256)
        bid = pool.allocate_fresh()
        pool.acquire_hit(bid)
        pool.release_to_cached_or_free(bid,keep_cached=False)
        self.assertIn(bid,pool.used_ids)
        self.assertEqual(pool.num_free,0)
        pool.release_to_cached_or_free(bid,keep_cached=False)
        self.assertEqual(pool.num_free,1)
        self.assertFalse(pool.used_ids)

    def test_cache_hit_reclaims_without_duplicate_ownership(self):
        pool = blocks.BlockManager(1,256)
        bid = pool.allocate_fresh()
        pool.blocks[bid].update(123,[1,2])
        pool.release_to_cached_or_free(bid,keep_cached=True)
        self.assertEqual(pool.num_cached,1)
        pool.acquire_hit(bid)
        self.assertEqual(pool.num_cached,0)
        self.assertEqual(pool.blocks[bid].token_ids,[1,2])
        self.assertEqual(pool.blocks[bid].ref_count,1)

    def test_eviction_notifies_index_and_clears_snapshot(self):
        pool = blocks.BlockManager(1,256)
        evicted=[]
        pool.on_evict=evicted.append
        bid=pool.allocate_fresh()
        pool.blocks[bid].update(42,[7])
        pool.blocks[bid].hybrid_snapshot=object()
        pool.release_to_cached_or_free(bid,keep_cached=True)
        self.assertEqual(pool.allocate_fresh(),bid)
        self.assertEqual(evicted,[bid])
        self.assertIsNone(pool.blocks[bid].hybrid_snapshot)
        self.assertEqual(pool.blocks[bid].hash,-1)
        self.assertEqual(pool.available_slots,0)

    def test_live_page_cannot_be_evicted(self):
        pool=blocks.BlockManager(1,256)
        pool.allocate_fresh()
        with self.assertRaises(RuntimeError):
            pool.allocate_fresh()


class HashTests(unittest.TestCase):
    def test_media_identity_changes_dependent_hash_chain(self):
        first=hashes.compute_block_hash([1,2],extra_keys=('mm:image-a:0',))
        other=hashes.compute_block_hash([1,2],extra_keys=('mm:image-b:0',))
        self.assertNotEqual(first,other)
        self.assertNotEqual(hashes.compute_block_hash([3,4],first),hashes.compute_block_hash([3,4],other))

    def test_media_only_marks_intersecting_blocks(self):
        seq=SimpleNamespace(block_size=4,multimodal_spans=((4,7,'image-a'),),multimodal_hash='image-a')
        self.assertEqual(hashes.multimodal_block_extra_keys(seq,0),())
        self.assertEqual(hashes.multimodal_block_extra_keys(seq,1),('mm:image-a:0',))
        self.assertEqual(hashes.multimodal_block_extra_keys(seq,2),())

    def test_extra_keys_are_length_delimited(self):
        self.assertNotEqual(hashes.compute_block_hash([1],extra_keys=('ab','c')),hashes.compute_block_hash([1],extra_keys=('a','bc')))


class RadixTests(unittest.TestCase):
    def test_split_preserves_shared_prefix(self):
        tree=radix.RadixTree()
        tree.insert([1,2,3],[10,11,12])
        matched,_=tree.insert([1,2,4],[10,11,13])
        self.assertEqual(matched,2)
        self.assertEqual(tree.match_prefix([1,2,3])[0],[10,11,12])
        self.assertEqual(tree.match_prefix([1,2,4])[0],[10,11,13])
        self.assertEqual(tree.match_prefix([1,2,9])[0],[10,11])

    def test_referenced_path_survives_eviction(self):
        tree=radix.RadixTree()
        _,active=tree.insert([1,2],[10,11])
        tree.inc_ref(active)
        tree.insert([3],[12])
        self.assertEqual(tree.evict(3),[12])
        self.assertTrue(tree.contains_block(10))
        tree.dec_ref(active)
        self.assertEqual(set(tree.evict(2)),{10,11})
        self.assertFalse(tree.contains_block(10))
        self.assertFalse(tree.contains_block(11))


class HybridTests(unittest.TestCase):
    def test_unknown_layer_type_fails_explicitly(self):
        with self.assertRaises(ValueError):
            hybrid.HybridLayerLayout.from_hf_config(SimpleNamespace(layer_types=['unknown']))


if __name__ == '__main__':
    unittest.main()
