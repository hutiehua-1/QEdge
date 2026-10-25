from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import count
from typing import Any

from ..sampling_params import SamplingParams

_SEQ_IDS = count(1)


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


def slots_for_range(
    block_table: list[int],
    start: int,
    end: int,
    block_size: int,
) -> list[int]:
    """计算 [start, end) 内每个 token 的物理 slot。

    块内 slot 连续，按物理块循环 + ``extend(range(...))``（C 层），避免逐 token
    的 Python ``//`` ``%`` 运算：8192 token 的 prefill 块从 ~1.1ms 降到 ~0.12ms。
    """
    n = end - start
    if n == 0:
        return []
    b = start // block_size
    off = start - b * block_size
    if off + n <= block_size:
        base = block_table[b] * block_size
        return list(range(base + off, base + off + n))
    slots: list[int] = []
    last = (end - 1) // block_size
    while b <= last:
        lo = start if b == start // block_size else b * block_size
        hi = end if b == last else (b + 1) * block_size
        base = block_table[b] * block_size
        slots.extend(range(base + lo - b * block_size, base + hi - b * block_size))
        b += 1
    return slots


@dataclass
class Sequence:
    """单个请求：包含 token、物理块表和调度游标。"""

    token_ids: list[int]
    block_size: int
    seq_id: int = 0
    status: SequenceStatus = SequenceStatus.WAITING
    block_table: list[int] = field(default_factory=list)
    num_cached_tokens: int = 0
    num_scheduled_tokens: int = 0
    need_sample: bool = False  # 本步 prefill 是否已覆盖全部 prompt，需要采样
    is_prefill: bool = True
    num_prompt_tokens: int = 0
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    max_tokens: int = 128
    min_tokens: int = 0
    ignore_eos: bool = False
    last_token: int = 0
    # Qwen3.5 多模态与混合缓存元数据。它们必须跟随请求而非 ModelRunner，
    # 这样抢占、恢复和请求切换时的状态归属才是显式的。
    multimodal_hash: str | None = None
    # (start_token, end_token, content_hash)。只有与媒体 placeholder 相交的
    # block 才把 content_hash 纳入 key，随后由 parent hash 传播媒体依赖。
    multimodal_spans: tuple[tuple[int, int, str], ...] = ()
    prefix_cache_eligible: bool = True
    prefix_cache_admitted: bool = True
    # 图片前的稳定文本前缀可以不足一个物理页；KVManager 用 copy-on-write
    # 部分页缓存复用它，避免为了 256-token 页边界向 prompt 注入 padding。
    reusable_prefix_tokens: int = 0
    runtime_prefix_snapshot: Any = None
    image_grid_thw: tuple[int, int, int] | None = None
    mrope_position_delta: int = 0
    multimodal_data: dict[str, Any] | None = None
    recurrent_state_id: int | None = None
    conv_state_id: int | None = None
    # 过渡阶段由 ModelRunner 持有的 Transformers 混合 cache 与请求输入。
    runtime_cache: Any = None
    runtime_attention_mask: Any = None
    runtime_inputs: dict[str, Any] | None = None
    runtime_mask_buffer: Any = None
    runtime_embeds: Any = None
    runtime_position_ids: Any = None
    runtime_rope_deltas: Any = None

    @classmethod
    def from_prompt(
        cls,
        token_ids: list[int],
        *,
        block_size: int,
        sampling: SamplingParams | None = None,
        multimodal_hash: str | None = None,
        multimodal_spans: tuple[tuple[int, int, str], ...] = (),
        prefix_cache_eligible: bool = True,
        reusable_prefix_tokens: int = 0,
        image_grid_thw: tuple[int, int, int] | None = None,
        multimodal_data: dict[str, Any] | None = None,
    ) -> Sequence:
        sp = sampling or SamplingParams()
        ids = copy(token_ids)
        if not ids:
            raise ValueError("empty prompt")
        return cls(
            token_ids=ids,
            block_size=block_size,
            seq_id=next(_SEQ_IDS),
            num_prompt_tokens=len(ids),
            last_token=ids[-1],
            temperature=sp.temperature,
            top_p=sp.top_p,
            top_k=sp.top_k,
            min_p=sp.min_p,
            max_tokens=sp.max_tokens,
            min_tokens=sp.min_tokens,
            ignore_eos=sp.ignore_eos,
            multimodal_hash=multimodal_hash,
            multimodal_spans=multimodal_spans,
            prefix_cache_eligible=prefix_cache_eligible,
            reusable_prefix_tokens=reusable_prefix_tokens,
            image_grid_thw=image_grid_thw,
            multimodal_data=multimodal_data,
        )

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens

    @property
    def num_blocks(self) -> int:
        n = self.num_tokens
        if n == 0:
            return 0
        return (n + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        if self.num_blocks == 0:
            return 0
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def num_full_blocks(self) -> int:
        return self.num_tokens // self.block_size

    def block_token_ids(self, i: int) -> list[int]:
        assert 0 <= i < self.num_blocks
        start = i * self.block_size
        end = min(start + self.block_size, self.num_tokens)
        return self.token_ids[start:end]

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)
        self.last_token = token_id
