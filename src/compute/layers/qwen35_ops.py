"""Qwen3.5 使用的归一化与多轴旋转位置编码算子。"""

from __future__ import annotations

import torch
from torch import nn
from .ops import gemma_rmsnorm


class Qwen35RMSNorm(nn.Module):
    """复用原始权重，调用 `(1 + weight)` 语义的融合 RMSNorm kernel。"""

    def __init__(self, reference: nn.Module) -> None:
        super().__init__()
        # 直接挂接原 Parameter，替换模块时不复制权重，也不改变 state_dict 数值。
        self.weight = reference.weight
        self.eps = reference.eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        values = hidden_states.reshape(-1, shape[-1])
        return gemma_rmsnorm(values, self.weight, self.eps).reshape(shape)


def _rotate_half(values: torch.Tensor) -> torch.Tensor:
    """按 Qwen/NeoX 布局交换并旋转最后一维的两个半区。"""
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_qwen35_mrope(query: torch.Tensor, key: torch.Tensor,
                       cos: torch.Tensor, sin: torch.Tensor,
                       unsqueeze_dim: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """应用已经由 Qwen3.5 rotary embedding 合成好的多轴 cos/sin。

    文本位置以及图像/视频的 T/H/W 位置选择发生在上游 rotary module；这里负责
    对 Q/K 执行最终旋转，因而同时兼容纯文本和多模态 MRoPE。
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = query[..., :rotary_dim], query[..., rotary_dim:]
    k_rot, k_pass = key[..., :rotary_dim], key[..., rotary_dim:]
    query = q_rot * cos + _rotate_half(q_rot) * sin
    key = k_rot * cos + _rotate_half(k_rot) * sin
    if q_pass.shape[-1]:
        query = torch.cat((query, q_pass), dim=-1)
    if k_pass.shape[-1]:
        key = torch.cat((key, k_pass), dim=-1)
    return query, key


def replace_qwen35_norms(model: nn.Module) -> int:
    """替换语言主干内全部 Qwen3.5 RMSNorm，返回替换数量。"""
    language_model = model.model.language_model
    count = 0
    for layer in language_model.layers:
        for name in ("input_layernorm", "post_attention_layernorm"):
            setattr(layer, name, Qwen35RMSNorm(getattr(layer, name)))
            count += 1
        attention = getattr(layer, "self_attn", None)
        # 分页 attention 将原官方模块保存在 reference 中。
        attention = getattr(attention, "reference", attention)
        if attention is not None:
            for name in ("q_norm", "k_norm"):
                if hasattr(attention, name):
                    setattr(attention, name, Qwen35RMSNorm(getattr(attention, name)))
                    count += 1
    if hasattr(language_model, "norm"):
        language_model.norm = Qwen35RMSNorm(language_model.norm)
        count += 1
    return count
