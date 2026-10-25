"""前缀缓存后端：关闭、Hash 索引或 Radix Tree 索引。"""

from __future__ import annotations

from typing import Literal

from .hash_index import HashPrefixIndex
from .noop import NoopPrefixIndex
from .radix_index import RadixPrefixIndex

PrefixBackend = Literal["none", "hash", "radix"]


def make_prefix_index(backend: PrefixBackend):
    if backend == "hash":
        return HashPrefixIndex()
    if backend == "radix":
        return RadixPrefixIndex()
    return NoopPrefixIndex()


__all__ = ["PrefixBackend", "make_prefix_index"]
