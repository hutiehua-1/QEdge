"""因果卷积后端选择，并提供不受二进制 ABI 影响的 FLA 回退路径。"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

try:
    from causal_conv1d import causal_conv1d_fn as _cuda_prefill
    from causal_conv1d import causal_conv1d_update as _cuda_update
except (ImportError, OSError):
    _cuda_prefill = _cuda_update = None


TORCH_CONV_PREFILL_MAX_TOKENS = int(
    os.getenv("DNDX_V3_TORCH_CONV_PREFILL_MAX_TOKENS", "768")
)
CAUSAL_CONV_BACKEND = (
    "causal_conv1d" if _cuda_prefill is not None
    else f"torch_grouped_conv<={TORCH_CONV_PREFILL_MAX_TOKENS}+fla_triton"
)
CAUSAL_CONV_DECODE_BACKEND = (
    "causal_conv1d" if _cuda_update is not None else "fla_triton"
)


def causal_conv_prefill(x: torch.Tensor, state: torch.Tensor | None,
                        weight: torch.Tensor, bias: torch.Tensor | None,
                        activation: str):
    """输入输出布局固定为 [B,T,D]，卷积状态布局为 [B,D,W]。"""
    if _cuda_prefill is not None:
        width = weight.shape[1]
        initial = None if state is None else state[..., -(width - 1):]
        output, final = _cuda_prefill(
            x.transpose(1, 2).contiguous(), weight, bias,
            initial_states=initial, return_final_states=True, activation=activation,
        )
        return output.transpose(1, 2), final
    # On ZW810E the vendor grouped-conv kernel is substantially faster than
    # FLA's Triton kernel for short, fresh prefills. Stateful continuation stays
    # on FLA to preserve its exact cache semantics.
    limit = TORCH_CONV_PREFILL_MAX_TOKENS
    if state is None and limit > 0 and x.shape[1] <= limit:
        width = weight.shape[1]
        channels_first = x.transpose(1, 2)
        output = F.conv1d(
            channels_first, weight.unsqueeze(1), bias,
            padding=width - 1, groups=weight.shape[0],
        )[..., :x.shape[1]]
        if activation in {"silu", "swish"}:
            output = F.silu(output)
        elif activation not in {None, "identity"}:
            raise ValueError(f"unsupported causal conv activation: {activation}")
        if x.shape[1] >= width:
            final = channels_first[..., -width:].contiguous()
        else:
            final = F.pad(channels_first, (width - x.shape[1], 0))
        return output.transpose(1, 2), final
    from fla.modules.conv import causal_conv1d
    return causal_conv1d(
        x, weight=weight, bias=bias, initial_state=state,
        output_final_state=True, activation=activation, backend="triton",
    )


def causal_conv_decode(x: torch.Tensor, state: torch.Tensor,
                       weight: torch.Tensor, bias: torch.Tensor | None,
                       activation: str):
    if _cuda_update is not None:
        output = _cuda_update(
            x.transpose(1, 2).contiguous(), state, weight, bias, activation
        )
        return output.transpose(1, 2), state
    from fla.modules.conv.triton import causal_conv1d_update
    return causal_conv1d_update(
        x, state, weight=weight, bias=bias, activation=activation
    )
