"""链式块哈希；热路径优先使用速度高于 blake2b 的 xxhash。

语义为 ``h_i = H(tokens_i | h_{i-1})``，初始值 ``h_{-1} = -1``。
"""

from __future__ import annotations

import struct

import numpy as np

try:
    import xxhash
except ImportError:  # pragma: no cover
    xxhash = None  # type: ignore


def compute_block_hash(
    token_ids: list[int] | np.ndarray,
    prefix: int = -1,
    extra_keys: tuple[str, ...] = (),
) -> int:
    if xxhash is not None:
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little", signed=False))
        arr = np.asarray(token_ids, dtype=np.int32)
        h.update(arr.tobytes())
        for key in extra_keys:
            encoded = key.encode()
            h.update(len(encoded).to_bytes(4, "little"))
            h.update(encoded)
        return int(h.intdigest())
    # 未安装 xxhash 时回退到相同输入布局的 blake2b 实现。
    import hashlib

    h = hashlib.blake2b(digest_size=8)
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little", signed=False))
    h.update(struct.pack(f"<{len(token_ids)}i", *token_ids))
    for key in extra_keys:
        encoded = key.encode()
        h.update(len(encoded).to_bytes(4, "little"))
        h.update(encoded)
    return int.from_bytes(h.digest(), "little", signed=False)


def multimodal_block_extra_keys(seq, block_index: int) -> tuple[str, ...]:
    """返回只影响当前 block 的媒体 key；后续 block 由链式 parent 继承。"""
    start = block_index * seq.block_size
    end = start + seq.block_size
    if seq.multimodal_spans:
        return tuple(
            f"mm:{identifier}:{span_start - start}"
            for span_start, span_end, identifier in seq.multimodal_spans
            if end > span_start and start < span_end
        )
    # 兼容直接构造 Sequence 的调用者：没有位置元数据时采用旧的保守语义，
    # 让媒体标识从首块开始影响整个 hash chain。
    if seq.multimodal_hash is not None and block_index == 0:
        return (f"mm:{seq.multimodal_hash}:0",)
    return ()


def multimodal_hash_seed(value: str | None) -> int:
    """存在媒体输入时，用图像标识作为链式 token 哈希的初始种子。"""
    if value is None:
        return -1
    try:
        return int(value[:16], 16)
    except (TypeError, ValueError):
        import hashlib
        return int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8).digest(), "little")
