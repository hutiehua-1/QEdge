"""Qwen3.5 各层替换器共享的轻量兼容工具。"""

from __future__ import annotations

from torch import nn


def layer_type(layer: nn.Module) -> str | None:
    """兼容不同 Transformers API 版本，读取 Qwen3.5 层类型。

    Transformers 5.12 将 ``block_type`` 改名为 ``layer_type``。将兼容逻辑
    集中在这里，可避免每个层替换器重复进行版本判断。
    """
    return getattr(layer, "block_type", None) or getattr(layer, "layer_type", None)
