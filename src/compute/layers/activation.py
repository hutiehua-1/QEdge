"""基于本地 FlashInfer 自定义算子的 Qwen3.5 激活层。"""

from __future__ import annotations

import torch
from torch import nn

from .ops import silu_and_mul


class SiluAndMul(nn.Module):
    """对 ``[..., 2 * intermediate]`` 输入执行融合 SwiGLU 激活。"""

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return silu_and_mul(values)
