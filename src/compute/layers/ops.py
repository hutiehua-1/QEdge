"""外部融合 kernel 的 torch.library 编译边界。"""

from __future__ import annotations

import os
from pathlib import Path

import torch

os.environ.setdefault(
    "FLASHINFER_WORKSPACE_BASE",
    str(Path(__file__).resolve().parents[3] / ".runtime_cache"),
)
from flashinfer.activation import silu_and_mul as _flashinfer_silu_and_mul
from flashinfer.norm import gemma_rmsnorm as _flashinfer_gemma_rmsnorm


@torch.library.custom_op("dndx_v3::silu_and_mul", mutates_args=())
def silu_and_mul(values: torch.Tensor) -> torch.Tensor:
    """把 FlashInfer 激活 kernel 作为不可继续 trace 的叶子算子。"""
    return _flashinfer_silu_and_mul(values)


@silu_and_mul.register_fake
def _silu_and_mul_fake(values: torch.Tensor) -> torch.Tensor:
    shape = (*values.shape[:-1], values.shape[-1] // 2)
    return torch.empty(shape, dtype=values.dtype, device=values.device)


@torch.library.custom_op("dndx_v3::gemma_rmsnorm", mutates_args=())
def gemma_rmsnorm(values: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3.5 `(1 + weight)` RMSNorm 的融合 kernel 边界。"""
    return _flashinfer_gemma_rmsnorm(values, weight, eps)


@gemma_rmsnorm.register_fake
def _gemma_rmsnorm_fake(values: torch.Tensor, weight: torch.Tensor,
                        eps: float) -> torch.Tensor:
    return torch.empty_like(values)
