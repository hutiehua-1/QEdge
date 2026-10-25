"""统一 KV 资源层：管理物理页、GPU 内存池和多种前缀缓存后端。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from transformers import PretrainedConfig

from .block_manager import BlockManager
from .prefix import PrefixBackend, RadixPrefixIndex, make_prefix_index
from .sequence import Sequence
from .hybrid import HybridLayerLayout


@dataclass(slots=True)
class KVCacheStats:
    """Prefix/KV lifecycle counters, following the xsy cache API shape.

    The counters describe the v3 physical page cache; no second
    ``past_key_values`` store is introduced.
    """

    queries: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.queries if self.queries else 0.0


@dataclass(slots=True)
class PrefixAllocation:
    """一次 prefix 查询与容量规划的结果，供 can_allocate/allocate 复用。"""

    hit_blocks: tuple[int, ...]
    num_cached_blocks: int


class KVManager:
    def __init__(
        self,
        hf_config: PretrainedConfig,
        *,
        num_blocks: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
        prefix_backend: PrefixBackend = "hash",
        enable_full_prefix_caching: bool = True,
        prefix_snapshot_interval_blocks: int = 32,
        prefix_admission_mode: str = "always",
        prefix_admission_window: int = 4,
        prefix_admission_probe_interval: int = 32,
        prefix_admission_min_token_hit_rate: float = 0.05,
    ) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, got {num_blocks}")

        self.block_size = block_size
        self.device = device
        self.dtype = dtype
        self.hf_config = hf_config
        self.layout = HybridLayerLayout.from_hf_config(hf_config)
        self.prefix_backend: PrefixBackend = prefix_backend
        self.enable_full_prefix_caching = enable_full_prefix_caching
        if prefix_snapshot_interval_blocks <= 0:
            raise ValueError("prefix_snapshot_interval_blocks must be positive")
        self.prefix_snapshot_interval_blocks = prefix_snapshot_interval_blocks
        self.prefix_admission_mode = prefix_admission_mode
        self.prefix_admission_window = prefix_admission_window
        self.prefix_admission_probe_interval = prefix_admission_probe_interval
        self.prefix_admission_min_token_hit_rate = prefix_admission_min_token_hit_rate
        self._allocation_plans: dict[int, PrefixAllocation] = {}
        self._admission_enabled = True
        self._admission_window_queries = 0
        self._admission_window_prompt_tokens = 0
        self._admission_window_hit_tokens = 0
        self._admission_disabled_queries = 0
        self.num_blocks_total = num_blocks
        self.num_blocks_schedulable = num_blocks

        self.blocks = BlockManager(self.num_blocks_schedulable, block_size)
        self.index = make_prefix_index(prefix_backend)
        self.stats = KVCacheStats()
        # Hash 后端使用页池 LRU 并同步失效索引；Radix 后端优先回收树的冷叶节点。
        self.blocks.on_evict = self._on_evict
        if isinstance(self.index, RadixPrefixIndex):
            self.blocks.reclaim_fn = self._reclaim_radix
        # 全局KV Cache的Tensor
        self.kv_cache = self._alloc_pool(hf_config, num_blocks, block_size, device, dtype)
        self.last_num_cached_tokens = 0
        self.last_prefix_hit_blocks = 0
        self.last_prefix_admitted = False
        # Full block indexes cannot reuse a short system prompt when the image
        # starts in the same 256-token page.  Keep one image-independent,
        # copy-on-write partial page per stable token prefix.
        self._partial_prefixes: dict[tuple[int, ...], tuple[torch.Tensor, object]] = {}

    def _partial_prefix_key(self, seq: Sequence) -> tuple[int, ...] | None:
        count = seq.reusable_prefix_tokens
        if (
            not seq.prefix_cache_eligible
            or count <= 0
            or count >= self.block_size
            or count > seq.num_prompt_tokens
        ):
            return None
        return tuple(seq.token_ids[:count])

    def publish_partial_prefix(self, seq: Sequence, snapshot: object) -> None:
        """Save an image-independent prefix shorter than one physical page."""
        key = self._partial_prefix_key(seq)
        if key is None or not seq.prefix_cache_admitted or not seq.block_table:
            return
        count = len(key)
        block_id = seq.block_table[0]
        page = self.kv_cache[:, :, block_id, :count].detach().clone()
        self._partial_prefixes[key] = (page, snapshot)

    def _on_evict(self, block_id: int) -> None:
        self.index.invalidate(block_id, self.blocks)
        self.stats.evictions += 1

    def _reclaim_radix(self, num_pages: int) -> list[int]:
        pages = self.index.reclaim_pages(num_pages, self.blocks)
        self.stats.evictions += len(pages)
        return pages

    @staticmethod
    def _alloc_pool(
        hf: PretrainedConfig,
        num_blocks: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        layout = HybridLayerLayout.from_hf_config(hf)
        nkv = layout.num_key_value_heads
        hd = layout.head_dim
        layers = layout.num_full_attention_layers
        block_bytes = 2 * layers * block_size * nkv * hd * dtype.itemsize
        kv = torch.empty(
            2, layers, num_blocks, block_size, nkv, hd, dtype=dtype, device=device
        )
        # print(
        #     f"KV pool: {num_blocks} blocks × {block_size} tok "
        #     f"({num_blocks * block_bytes / (1024**3):.2f} GiB)"
        # )
        return kv

    # 计算出能开多少个block
    @classmethod
    def auto_num_blocks(
        cls,
        hf: PretrainedConfig,
        *,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device,
        gpu_memory_utilization: float,
        reserve_bytes: int = 0,
    ) -> int:
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        layout = HybridLayerLayout.from_hf_config(hf)
        nkv = layout.num_key_value_heads
        hd = layout.head_dim
        layers = layout.num_full_attention_layers
        block_bytes = 2 * layers * block_size * nkv * hd * dtype.itemsize
        free, total = torch.cuda.mem_get_info(device)
        available = int(total * gpu_memory_utilization) - (total - free)
        # 预留给 CG 的字节数由调用方按实际捕获足迹传入（见 Engine._probe_cg_footprint）。
        available -= reserve_bytes
        return max(4, available // block_bytes)

    @property
    def num_free(self) -> int:
        return self.blocks.num_free

    def _make_allocation_plan(self, seq: Sequence) -> PrefixAllocation | None:
        need = seq.num_blocks
        if need == 0:
            return PrefixAllocation((), 0)
        hit = (
            self.index.match(seq, self.blocks)
            if seq.prefix_cache_eligible
            and getattr(self, "enable_full_prefix_caching", True) else []
        )
        # 至少重算最后一个块以取得 prompt 末尾 logits。
        max_cached = min(len(hit), max(0, need - 1))
        while max_cached > 0 and self.blocks.blocks[hit[max_cached - 1]].hybrid_snapshot is None:
            max_cached -= 1
        while max_cached >= 0:
            new_needed = need - max_cached
            reclaimable = sum(
                1 for b in hit[:max_cached] if b in self.blocks.cached_ids
            )
            # 命中的 cached 页会直接转成 used，不能再作为新页容量；其余
            # cached 页都可由 BlockManager 的 LRU 淘汰。旧实现漏算这些可
            # 淘汰页，池满后 can_allocate 永远为 False，调度器因而空转。
            evictable = self.blocks.num_cached - reclaimable
            if self.blocks.num_free + evictable >= new_needed:
                return PrefixAllocation(tuple(hit), max_cached)
            max_cached -= 1
        return None

    def _fit_prefix(self, seq: Sequence) -> int:
        """兼容测试/诊断入口；调度热路径使用并复用 PrefixAllocation。"""
        plan = self._make_allocation_plan(seq)
        return -1 if plan is None else plan.num_cached_blocks

    def _select_admission(self, *, hit_tokens: int) -> bool:
        if self.prefix_backend == "none":
            return False
        if self.prefix_admission_mode == "always" or hit_tokens > 0:
            return True
        if self._admission_enabled:
            return True
        # 低命中窗口仍周期性放行一个 miss，检测工作负载是否发生变化。
        return (
            self._admission_disabled_queries % self.prefix_admission_probe_interval
        ) == 0

    def _update_admission_window(self, *, prompt_tokens: int, hit_tokens: int) -> None:
        if self.prefix_admission_mode != "adaptive":
            return
        self._admission_window_queries += 1
        self._admission_window_prompt_tokens += prompt_tokens
        self._admission_window_hit_tokens += hit_tokens
        if not self._admission_enabled:
            self._admission_disabled_queries += 1
        if self._admission_window_queries < self.prefix_admission_window:
            return
        denominator = max(1, self._admission_window_prompt_tokens)
        token_hit_rate = self._admission_window_hit_tokens / denominator
        self._admission_enabled = (
            token_hit_rate >= self.prefix_admission_min_token_hit_rate
        )
        if self._admission_enabled:
            self._admission_disabled_queries = 0
        self._admission_window_queries = 0
        self._admission_window_prompt_tokens = 0
        self._admission_window_hit_tokens = 0

    @property
    def prefix_admission_enabled(self) -> bool:
        return self.prefix_admission_mode == "always" or self._admission_enabled

    def make_prefix_cache_stats(self) -> KVCacheStats:
        """Return and reset interval counters, matching the xsy wrapper API."""
        stats = self.stats
        self.stats = KVCacheStats()
        return stats

    def reset_prefix_cache_stats(self) -> None:
        self.stats = KVCacheStats()

    @property
    def usage(self) -> float:
        total = self.num_blocks_total
        return 0.0 if total <= 0 else 1.0 - (self.blocks.num_free / total)

    def reset_prefix_cache(self) -> bool:
        """Drop all published prefix metadata when no request owns a page."""
        if self.blocks.used_ids:
            return False
        for bid in list(self.blocks.cached_ids):
            self.blocks._take_cached_to_free(bid, notify=False)
        self.index = make_prefix_index(self.prefix_backend)
        self._allocation_plans.clear()
        self.blocks.on_evict = self._on_evict
        if isinstance(self.index, RadixPrefixIndex):
            self.blocks.reclaim_fn = self._reclaim_radix
        self.last_num_cached_tokens = 0
        self.last_prefix_hit_blocks = 0
        self.last_prefix_admitted = False
        self._partial_prefixes.clear()
        return True

    def can_allocate(self, seq: Sequence) -> bool:
        plan = self._make_allocation_plan(seq)
        if plan is None:
            self._allocation_plans.pop(seq.seq_id, None)
            return False
        self._allocation_plans[seq.seq_id] = plan
        return True

    def allocate(self, seq: Sequence) -> None:
        seq.block_table.clear()
        seq.num_cached_tokens = 0
        need = seq.num_blocks
        if need == 0:
            self.last_num_cached_tokens = 0
            return

        plan = self._allocation_plans.pop(seq.seq_id, None)
        if plan is None:
            plan = self._make_allocation_plan(seq)
        if plan is None:
            raise RuntimeError("not enough KV blocks for sequence")
        max_cached = plan.num_cached_blocks
        hit = plan.hit_blocks
        self.stats.queries += 1
        self.stats.hits += int(max_cached > 0)
        self.stats.misses += int(max_cached == 0)
        self.last_prefix_hit_blocks = max_cached
        hit_tokens = max_cached * self.block_size
        partial_key = self._partial_prefix_key(seq) if max_cached == 0 else None
        partial = self._partial_prefixes.get(partial_key) if partial_key is not None else None
        if partial is not None:
            hit_tokens = len(partial_key)
        seq.prefix_cache_admitted = (
            seq.prefix_cache_eligible
            and self._select_admission(hit_tokens=hit_tokens)
        )
        self.last_prefix_admitted = seq.prefix_cache_admitted
        self._update_admission_window(
            prompt_tokens=seq.num_prompt_tokens, hit_tokens=hit_tokens
        )

        for bid in hit[:max_cached]:
            self.blocks.acquire_hit(bid)
            seq.block_table.append(bid)

        for _ in range(max_cached, need):
            seq.block_table.append(self.blocks.allocate_fresh())

        seq.num_cached_tokens = max_cached * self.block_size
        if partial is not None:
            cached_page, snapshot = partial
            count = len(partial_key)
            destination = self.kv_cache[:, :, seq.block_table[0], :count]
            destination.copy_(cached_page, non_blocking=True)
            seq.num_cached_tokens = count
            seq.runtime_prefix_snapshot = snapshot
            # Convert the full-block miss recorded above into a useful partial hit.
            self.stats.hits += 1
            self.stats.misses -= 1
        self.last_num_cached_tokens = seq.num_cached_tokens

        if isinstance(self.index, RadixPrefixIndex) and max_cached > 0:
            self.index.pin_match(seq, max_cached)

    def can_append(self, seq: Sequence) -> bool:
        return self.blocks.can_append(seq)

    def may_append(self, seq: Sequence) -> None:
        self.blocks.may_append(seq)

    def sync_prefix(self, seq: Sequence) -> None:
        """只把本步新填满的块写入前缀索引；未满块或未跨块边界则直接返回。"""
        if (
            self.prefix_backend == "none"
            or not self.enable_full_prefix_caching
            or not seq.prefix_cache_admitted
            or seq.num_scheduled_tokens <= 0
        ):
            return
        prev = seq.num_cached_tokens - seq.num_scheduled_tokens
        start = prev // self.block_size
        end = min(seq.num_full_blocks(), seq.num_cached_tokens // self.block_size)
        if start >= end:
            return
        self.index.publish(seq, end, self.blocks, start=start)

    def deallocate(self, seq: Sequence, *, publish: bool = True) -> None:
        # completion block 没有 Hybrid 恢复点，保留它只会污染索引和 LRU。
        n_full = min(
            seq.num_prompt_tokens // self.block_size,
            seq.num_full_blocks(),
        ) if publish and seq.prefix_cache_admitted and self.enable_full_prefix_caching else 0
        if publish and n_full > 0 and self.prefix_backend != "none":
            self.index.publish(seq, n_full, self.blocks)

        self.index.release_seq(seq.seq_id)

        published = set(seq.block_table[:n_full]) if n_full > 0 else set()
        for bid in list(seq.block_table):
            keep = bid in published and self.prefix_backend != "none"
            # hash：释放前摘索引；radix：先降引用，零引用后再摘树叶。
            if not keep and self.prefix_backend == "hash":
                self.index.invalidate(bid, self.blocks)
            self.blocks.release_to_cached_or_free(bid, keep_cached=keep)
            if not keep and self.prefix_backend == "radix":
                self.index.invalidate(bid, self.blocks)

        seq.block_table.clear()
        seq.num_cached_tokens = 0
