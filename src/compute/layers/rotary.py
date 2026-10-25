"""Qwen3.5 多模态旋转位置编码层。

与纯文本 Qwen3 不同，Qwen3.5 会在上游组合时间、高度和宽度三个方向的频率。
本模块直接使用预计算的 cos/sin 张量，避免重建纯文本缓存而破坏 MRoPE 语义。
"""

from __future__ import annotations

import torch
from torch import nn

from .qwen35_ops import apply_qwen35_mrope


class Qwen35MRoPE(nn.Module):
    def __init__(self, unsqueeze_dim: int = 1) -> None:
        super().__init__()
        self.unsqueeze_dim = unsqueeze_dim

    def forward(
        self, query: torch.Tensor, key: torch.Tensor,
        cos: torch.Tensor, sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_qwen35_mrope(
            query, key, cos, sin, unsqueeze_dim=self.unsqueeze_dim
        )
