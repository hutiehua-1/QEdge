"""基于 FLA 的 Qwen3.5 Gated DeltaNet 算子。

本模块保留与 checkpoint 兼容的原始层引用，并提供可选的权重加载后投影合并，
用于测试不同硬件上的 GEMM 性能。
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from .causal_conv import causal_conv_decode, causal_conv_prefill
from .utils import layer_type


class FLAGatedDeltaNet(nn.Module):
    def __init__(self, reference: nn.Module, *, merge_projections: bool = False,
                 use_fla_prefill: bool = True) -> None:
        super().__init__()
        self.reference = reference
        self.layer_idx = reference.layer_idx
        self.merge_projections = merge_projections
        self.use_fla_prefill = use_fla_prefill
        if merge_projections:
            self.qkv_size = reference.in_proj_qkv.out_features
            self.z_size = reference.in_proj_z.out_features
            self.beta_size = reference.in_proj_b.out_features
            self.gate_size = reference.in_proj_a.out_features
            self.qkvz_proj = self._merge_linear(reference.in_proj_qkv, reference.in_proj_z)
            self.ba_proj = self._merge_linear(reference.in_proj_b, reference.in_proj_a)

    @staticmethod
    def _merge_linear(first: nn.Linear, second: nn.Linear) -> nn.Linear:
        if first.in_features != second.in_features:
            raise ValueError("merged DeltaNet projections must share input width")
        has_bias = first.bias is not None or second.bias is not None
        merged = nn.Linear(
            first.in_features, first.out_features + second.out_features,
            bias=has_bias, device=first.weight.device, dtype=first.weight.dtype,
        )
        with torch.no_grad():
            merged.weight.copy_(torch.cat((first.weight, second.weight), dim=0))
            if has_bias:
                merged.bias[:first.out_features].zero_()
                merged.bias[first.out_features:].zero_()
                if first.bias is not None:
                    merged.bias[:first.out_features].copy_(first.bias)
                if second.bias is not None:
                    merged.bias[first.out_features:].copy_(second.bias)
        return merged

    def forward(self, hidden_states: torch.Tensor, cache_params: Any = None,
                attention_mask: torch.Tensor | None = None, **kwargs):
        # 参考 prefill 已使用 grouped-conv；在当前硬件上其完整 GDN 路径比
        # FLA chunk 更快。FLA prefill 仍可显式开启以测试 causal_conv 快速路径。
        if hidden_states.shape[1] > 1 and not self.use_fla_prefill:
            return self.reference(
                hidden_states=hidden_states,
                cache_params=cache_params,
                attention_mask=attention_mask,
                **kwargs,
            )
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )

        ref = self.reference
        if attention_mask is not None and not isinstance(attention_mask, dict):
            hidden_states = hidden_states * attention_mask[:, -hidden_states.shape[1]:, None]
        batch, seq_len, _ = hidden_states.shape
        cached = cache_params is not None and cache_params.has_previous_state(ref.layer_idx)

        if self.merge_projections:
            qkvz = self.qkvz_proj(hidden_states)
            mixed_qkv, z = torch.split(qkvz, [self.qkv_size, self.z_size], dim=-1)
            ba = self.ba_proj(hidden_states)
            beta_logits, gate_logits = torch.split(
                ba, [self.beta_size, self.gate_size], dim=-1
            )
            z = z.reshape(batch, seq_len, -1, ref.head_v_dim)
        else:
            mixed_qkv = ref.in_proj_qkv(hidden_states)
            z = ref.in_proj_z(hidden_states).reshape(batch, seq_len, -1, ref.head_v_dim)
            beta_logits = ref.in_proj_b(hidden_states)
            gate_logits = ref.in_proj_a(hidden_states)
        weight = ref.conv1d.weight.squeeze(1)

        if cached and seq_len == 1 and not cache_params.layers[ref.layer_idx].record_past:
            conv_state = cache_params.layers[ref.layer_idx].conv_states[0]
            mixed_qkv, _ = causal_conv_decode(
                mixed_qkv, conv_state, weight, ref.conv1d.bias, ref.activation,
            )
        else:
            initial_conv = (
                cache_params.layers[ref.layer_idx].conv_states[0] if cached else None
            )
            mixed_qkv, final_conv = causal_conv_prefill(
                mixed_qkv, initial_conv, weight, ref.conv1d.bias, ref.activation,
            )
            if cache_params is not None and final_conv is not None:
                cache_params.update_conv_state(
                    final_conv, ref.layer_idx, conv_kernel_size=ref.conv_kernel_size
                )

        query, key, value = torch.split(
            mixed_qkv, [ref.key_dim, ref.key_dim, ref.value_dim], dim=-1
        )
        query = query.reshape(batch, seq_len, -1, ref.head_k_dim)
        key = key.reshape(batch, seq_len, -1, ref.head_k_dim)
        value = value.reshape(batch, seq_len, -1, ref.head_v_dim)

        recurrent = (
            cache_params.layers[ref.layer_idx].recurrent_states[0] if cached else None
        )
        common = dict(
            q=query, k=key, v=value, g=gate_logits, beta=beta_logits,
            initial_state=recurrent, output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            A_log=ref.A_log.float(), dt_bias=ref.dt_bias,
            use_beta_sigmoid_in_kernel=True,
        )
        if cached and seq_len == 1:
            core, final_recurrent = fused_recurrent_gated_delta_rule(**common)
        else:
            core, final_recurrent = chunk_gated_delta_rule(**common)
        if cache_params is not None and final_recurrent is not None:
            cache_params.update_recurrent_state(final_recurrent, ref.layer_idx)

        core = core.reshape(-1, ref.head_v_dim)
        core = ref.norm(core, z.reshape(-1, ref.head_v_dim))
        return ref.out_proj(core.reshape(batch, seq_len, -1))


def install_fla_gated_deltanet(model: nn.Module, *, merge_projections: bool = False,
                               use_fla_prefill: bool = True) -> int:
    count = 0
    for layer in model.model.language_model.layers:
        if layer_type(layer) != "linear_attention":
            continue
        current = layer.linear_attn
        if isinstance(current, FLAGatedDeltaNet):
            continue
        layer.linear_attn = FLAGatedDeltaNet(
            current, merge_projections=merge_projections,
            use_fla_prefill=use_fla_prefill,
        )
        count += 1
    return count
