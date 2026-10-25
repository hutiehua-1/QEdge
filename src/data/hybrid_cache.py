"""Qwen3.5 混合缓存：Full Attention 外置分页，DeltaNet 状态保留。"""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer


class ExternalPagedLayer(DynamicLayer):
    """只记录长度而不保存 K/V；真实张量位于 KVManager 的物理页池。"""

    def __init__(self) -> None:
        super().__init__()
        self.cumulative_length = 0

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        if not self.is_initialized:
            self.dtype, self.device = key_states.dtype, key_states.device
            self.keys = torch.empty(0, dtype=self.dtype, device=self.device)
            self.values = torch.empty(0, dtype=self.dtype, device=self.device)
            self.is_initialized = True
        self.cumulative_length += key_states.shape[-2]
        return key_states, value_states

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.cumulative_length + query_length, 0

    def get_max_length(self) -> int:
        return -1


class Qwen35HybridCache(DynamicCache):
    """18 层持有 conv/recurrent state，6 个 full 层仅持有长度。"""

    def __init__(self, config) -> None:
        super().__init__(config=config)
        text = config.get_text_config(decoder=True)
        for index, kind in enumerate(text.layer_types):
            if kind == "full_attention":
                self.layers[index] = ExternalPagedLayer()

    def restore_from(self, snapshot: "Qwen35HybridCache") -> None:
        """保持所有 GPU Tensor 地址不变，原地恢复图捕获前的混合状态。"""
        for target, source in zip(self.layers, snapshot.layers):
            if isinstance(target, ExternalPagedLayer):
                target.cumulative_length = source.cumulative_length
                continue
            for state_idx, value in source.conv_states.items():
                if value is not None:
                    target.conv_states[state_idx].copy_(value)
            for state_idx, value in source.recurrent_states.items():
                if value is not None:
                    target.recurrent_states[state_idx].copy_(value)
            target.has_previous_state.update(source.has_previous_state)

    def reset_in_place(self) -> None:
        """请求结束后原地重置状态，同时保持 CUDA Graph 使用的张量地址不变。"""
        for layer in self.layers:
            if isinstance(layer, ExternalPagedLayer):
                layer.cumulative_length = 0
                continue
            for value in layer.conv_states.values():
                if value is not None:
                    value.zero_()
            for value in layer.recurrent_states.values():
                if value is not None:
                    value.zero_()
            for state_idx in layer.has_previous_state:
                layer.has_previous_state[state_idx] = False
