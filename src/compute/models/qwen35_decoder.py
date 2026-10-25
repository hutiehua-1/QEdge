"""自研 Qwen3.5 decoder 组合层与融合 MLP。"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from ..layers.activation import SiluAndMul
from ..layers.utils import layer_type


class FusedQwen35MLP(nn.Module):
    """合并 gate/up GEMM，并使用 FlashInfer SiLU×mul。"""

    def __init__(self, reference: nn.Module) -> None:
        super().__init__()
        hidden = reference.gate_proj.in_features
        intermediate = reference.gate_proj.out_features
        self.gate_up_proj = nn.Linear(hidden, 2 * intermediate, bias=False,
                                      device=reference.gate_proj.weight.device,
                                      dtype=reference.gate_proj.weight.dtype)
        with torch.no_grad():
            self.gate_up_proj.weight.copy_(
                torch.cat([reference.gate_proj.weight, reference.up_proj.weight], dim=0)
            )
        self.down_proj = reference.down_proj
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        gate_up = self.gate_up_proj(hidden_states).reshape(-1, self.gate_up_proj.out_features)
        activated = self.act_fn(gate_up)
        return self.down_proj(activated).reshape(*shape[:-1], -1)


class Qwen35DecoderLayer(nn.Module):
    """保持官方输入契约的项目自有 decoder layer。"""

    def __init__(self, reference: nn.Module) -> None:
        super().__init__()
        self.block_type = layer_type(reference)
        if self.block_type is None:
            raise ValueError("Qwen3.5 decoder layer has no layer type")
        self.layer_type = self.block_type
        self.input_layernorm = reference.input_layernorm
        self.post_attention_layernorm = reference.post_attention_layernorm
        self.mlp = FusedQwen35MLP(reference.mlp)
        if self.block_type == "linear_attention":
            self.linear_attn = reference.linear_attn
        else:
            self.self_attn = reference.self_attn

    def forward(self, hidden_states: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor],
                attention_mask: torch.Tensor | None = None,
                position_ids: torch.Tensor | None = None,
                past_key_values: Any = None, **kwargs: Any) -> torch.Tensor:
        residual = hidden_states
        normalized = self.input_layernorm(hidden_states)
        if self.block_type == "linear_attention":
            mixed = self.linear_attn(
                hidden_states=normalized, cache_params=past_key_values,
                attention_mask=attention_mask, **kwargs,
            )
        else:
            mixed, _ = self.self_attn(
                hidden_states=normalized, position_embeddings=position_embeddings,
                attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, **kwargs,
            )
        hidden_states = residual + mixed
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


def install_qwen35_decoder(model: nn.Module) -> int:
    """原地替换语言主干的 24 个 decoder layer。"""
    layers = model.model.language_model.layers
    for index, layer in enumerate(layers):
        layers[index] = Qwen35DecoderLayer(layer)
    return len(layers)
