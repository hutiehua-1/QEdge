from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HybridLayerLayout:
    """从 Qwen3.5 文本配置中提取的混合缓存拓扑。"""

    full_attention_layers: tuple[int, ...]
    linear_attention_layers: tuple[int, ...]
    num_key_value_heads: int
    head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int

    @classmethod
    def from_hf_config(cls, config: Any) -> "HybridLayerLayout":
        text = getattr(config, "text_config", config)
        layer_types = tuple(text.layer_types)
        full = tuple(i for i, kind in enumerate(layer_types) if kind == "full_attention")
        linear = tuple(i for i, kind in enumerate(layer_types) if kind == "linear_attention")
        if len(full) + len(linear) != len(layer_types):
            raise ValueError(f"unsupported Qwen3.5 layer types: {layer_types!r}")
        return cls(
            full_attention_layers=full,
            linear_attention_layers=linear,
            num_key_value_heads=int(text.num_key_value_heads),
            head_dim=int(text.head_dim),
            linear_num_key_heads=int(text.linear_num_key_heads),
            linear_num_value_heads=int(text.linear_num_value_heads),
            linear_key_head_dim=int(text.linear_key_head_dim),
            linear_value_head_dim=int(text.linear_value_head_dim),
            linear_conv_kernel_dim=int(text.linear_conv_kernel_dim),
        )

    @property
    def num_full_attention_layers(self) -> int:
        return len(self.full_attention_layers)

    @property
    def num_linear_attention_layers(self) -> int:
        return len(self.linear_attention_layers)
