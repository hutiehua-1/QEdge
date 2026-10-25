"""Qwen3.5 Full Attention 的分页 KV 数据通路。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from ...data.sequence import Sequence, slots_for_range
from ...data.kv_ops import norm_mrope_store_kvcache, store_kvcache
from .qwen35_ops import apply_qwen35_mrope
from .utils import layer_type


@dataclass
class PagedAttentionContext:
    """由 ModelRunner 在一次模型调用前绑定的请求页表上下文。"""

    sequence: Sequence | None = None
    start: int = 0
    end: int = 0
    slot_mapping: torch.Tensor | None = None
    block_table: torch.Tensor | None = None
    cache_seqlens: torch.Tensor | None = None

    def bind(self, sequence: Sequence, start: int, end: int) -> None:
        if start < 0 or end < start or end > sequence.num_tokens:
            raise ValueError("invalid paged attention range")
        self.sequence, self.start, self.end = sequence, start, end

    def clear(self) -> None:
        self.sequence = None
        self.start = self.end = 0
        self.slot_mapping = self.block_table = self.cache_seqlens = None

    def bind_batch(self, *, slot_mapping: torch.Tensor, block_table: torch.Tensor,
                   cache_seqlens: torch.Tensor) -> None:
        self.slot_mapping = slot_mapping
        self.block_table = block_table
        self.cache_seqlens = cache_seqlens


class PagedQwen35Attention(nn.Module):
    """复用官方权重，把 MRoPE 后的 K/V 写入 KVManager 物理页。"""

    def __init__(self, reference: nn.Module, kv_cache: torch.Tensor,
                 context: PagedAttentionContext, paged_layer: int,
                 fused_norm_mrope_store: bool = True) -> None:
        super().__init__()
        self.reference = reference
        self.kv_cache = kv_cache
        self.context = context
        self.paged_layer = paged_layer
        self.fused_norm_mrope_store = fused_norm_mrope_store
        self.config = reference.config
        self.layer_idx = reference.layer_idx
        self.is_causal = reference.is_causal
        self.q_size = reference.q_proj.out_features
        self.k_size = reference.k_proj.out_features
        self.v_size = reference.v_proj.out_features
        self.qkv_proj = nn.Linear(
            reference.q_proj.in_features,
            self.q_size + self.k_size + self.v_size,
            bias=False, device=reference.q_proj.weight.device,
            dtype=reference.q_proj.weight.dtype,
        )
        with torch.no_grad():
            self.qkv_proj.weight.copy_(torch.cat((
                reference.q_proj.weight, reference.k_proj.weight, reference.v_proj.weight,
            ), dim=0))

    def forward(self, hidden_states: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor],
                attention_mask: torch.Tensor | None,
                past_key_values: Any = None, **kwargs: Any):
        ref = self.reference
        seq = self.context.sequence
        # 非调度入口（例如独立 HF generate）保持官方实现兼容。
        if seq is None:
            return ref(hidden_states=hidden_states, position_embeddings=position_embeddings,
                       attention_mask=attention_mask, past_key_values=past_key_values, **kwargs)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, ref.head_dim)
        projected = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = torch.split(
            projected, (self.q_size, self.k_size, self.v_size), dim=-1
        )
        query, gate = torch.chunk(
            query_states.view(*input_shape, -1, ref.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)
        query = query.view(hidden_shape)
        key = key_states.view(hidden_shape)
        value = value_states.view(hidden_shape)

        if self.context.end - self.context.start != hidden_states.shape[1]:
            raise RuntimeError("paged context and attention input length differ")
        slots = self.context.slot_mapping
        if slots is None:
            slots = torch.tensor(
                slots_for_range(seq.block_table, self.context.start, self.context.end, seq.block_size),
                device=hidden_states.device, dtype=torch.long,
            )
        else:
            slots = slots[: hidden_states.shape[1]].long()
        k_pool = self.kv_cache[0, self.paged_layer]
        v_pool = self.kv_cache[1, self.paged_layer]
        if self.fused_norm_mrope_store:
            cos, sin = position_embeddings
            q = norm_mrope_store_kvcache(
                query, key, value, cos, sin, ref.q_norm.weight, ref.k_norm.weight,
                k_pool.reshape(-1, key.shape[2], key.shape[-1]),
                v_pool.reshape(-1, value.shape[2], value.shape[-1]),
                slots, ref.q_norm.eps,
            ).contiguous()
            # ExternalPagedLayer 只记录序列长度，不会保留这里的 K/V 张量。
            if past_key_values is not None:
                past_key_values.update(key.transpose(1, 2), value.transpose(1, 2), ref.layer_idx)
        else:
            query = ref.q_norm(query).transpose(1, 2)
            key = ref.k_norm(key).transpose(1, 2)
            value = value.transpose(1, 2)
            query, key = apply_qwen35_mrope(query, key, *position_embeddings)
            if past_key_values is not None:
                past_key_values.update(key, value, ref.layer_idx)
            store_kvcache(
                key.transpose(1, 2)[0], value.transpose(1, 2)[0],
                k_pool.reshape(-1, key.shape[1], key.shape[-1]),
                v_pool.reshape(-1, value.shape[1], value.shape[-1]), slots,
            )
            q = query.transpose(1, 2).contiguous()

        table = self.context.block_table
        if table is None:
            table = torch.tensor([seq.block_table], device=hidden_states.device, dtype=torch.int32)
        if hidden_states.shape[1] > 1:
            query_length = hidden_states.shape[1]
            key_length = self.context.end
            cu_q = torch.tensor([0, query_length], device=hidden_states.device, dtype=torch.int32)
            cu_k = torch.tensor([0, key_length], device=hidden_states.device, dtype=torch.int32)
            output = flash_attn_varlen_func(
                q[0], k_pool, v_pool, cu_q, cu_k, query_length, key_length,
                causal=True, softmax_scale=ref.scaling, block_table=table,
            ).unsqueeze(0)
        elif hidden_states.shape[1] == 1:
            cache_len = self.context.cache_seqlens
            if cache_len is None:
                cache_len = torch.tensor([self.context.end], device=hidden_states.device,
                                         dtype=torch.int32)
            output = flash_attn_with_kvcache(
                q, k_pool, v_pool, cache_seqlens=cache_len,
                block_table=table, causal=True, softmax_scale=ref.scaling,
            )
        else:
            raise RuntimeError("invalid paged attention input")
        output = output.reshape(*input_shape, -1)
        return ref.o_proj(output * torch.sigmoid(gate)), None


def install_paged_attention(model: nn.Module, kv_cache: torch.Tensor,
                            context: PagedAttentionContext,
                            fused_norm_mrope_store: bool = True) -> int:
    """将语言模型中 6 个 full_attention 层替换为分页实现。"""
    layers = model.model.language_model.layers
    page = 0
    for layer in layers:
        if layer_type(layer) == "full_attention":
            layer.self_attn = PagedQwen35Attention(
                layer.self_attn, kv_cache, context, page, fused_norm_mrope_store
            )
            page += 1
    return page
