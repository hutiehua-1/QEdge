from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < D
    key_offsets = idx * key_stride + offsets
    value_offsets = idx * value_stride + offsets
    key = tl.load(key_ptr + key_offsets, mask=mask, other=0.0)
    value = tl.load(value_ptr + value_offsets, mask=mask, other=0.0)
    cache_offsets = slot * D + offsets
    tl.store(k_cache_ptr + cache_offsets, key, mask=mask)
    tl.store(v_cache_ptr + cache_offsets, value, mask=mask)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()

    n, num_heads, head_dim = key.shape
    d = num_heads * head_dim
    assert slot_mapping.numel() == n

    block = triton.next_power_of_2(d)
    _store_kvcache_kernel[(n,)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        d,
        block,
    )


@triton.jit
def _norm_mrope_store_kernel(
    q_ptr, k_ptr, v_ptr, cos_ptr, sin_ptr, q_weight_ptr, k_weight_ptr,
    q_out_ptr, k_cache_ptr, v_cache_ptr, slots_ptr,
    q_stride_t: tl.constexpr, q_stride_h: tl.constexpr,
    k_stride_t: tl.constexpr, k_stride_h: tl.constexpr,
    v_stride_t: tl.constexpr, v_stride_h: tl.constexpr,
    QH: tl.constexpr, KH: tl.constexpr, D: tl.constexpr, RD: tl.constexpr,
    EPS: tl.constexpr, BLOCK: tl.constexpr,
):
    """融合 Q/K RMSNorm、MRoPE 以及分页 K/V 写入。"""
    token, head = tl.program_id(0), tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    half = RD // 2
    rot_mask = offs < RD
    cos = tl.load(cos_ptr + token * RD + offs, mask=rot_mask, other=1.0).to(tl.float32)
    sin = tl.load(sin_ptr + token * RD + offs, mask=rot_mask, other=0.0).to(tl.float32)
    pair = tl.where(offs < half, offs + half, offs - half)
    if head < QH:
        base = token * q_stride_t + head * q_stride_h
        q = tl.load(q_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(q * q, axis=0) / D + EPS)
        w = 1.0 + tl.load(q_weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        qp = tl.load(q_ptr + base + pair, mask=rot_mask, other=0.0).to(tl.float32)
        wp = 1.0 + tl.load(q_weight_ptr + pair, mask=rot_mask, other=0.0).to(tl.float32)
        rotated = tl.where(offs < half, -(qp * inv * wp), qp * inv * wp)
        q_result = tl.where(rot_mask, q * inv * w * cos + rotated * sin, q * inv * w)
        tl.store(q_out_ptr + (token * QH + head) * D + offs,
                 q_result, mask=mask)
    if head < KH:
        base = token * k_stride_t + head * k_stride_h
        k = tl.load(k_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(k * k, axis=0) / D + EPS)
        w = 1.0 + tl.load(k_weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        kp = tl.load(k_ptr + base + pair, mask=rot_mask, other=0.0).to(tl.float32)
        wp = 1.0 + tl.load(k_weight_ptr + pair, mask=rot_mask, other=0.0).to(tl.float32)
        rotated = tl.where(offs < half, -(kp * inv * wp), kp * inv * wp)
        k_result = tl.where(rot_mask, k * inv * w * cos + rotated * sin, k * inv * w)
        slot = tl.load(slots_ptr + token)
        valid = mask & (slot >= 0)
        cache_base = (slot * KH + head) * D
        tl.store(k_cache_ptr + cache_base + offs,
                 k_result, mask=valid)
        v = tl.load(v_ptr + token * v_stride_t + head * v_stride_h + offs,
                    mask=mask, other=0.0)
        tl.store(v_cache_ptr + cache_base + offs, v, mask=valid)


def norm_mrope_store_kvcache(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor, q_weight: torch.Tensor,
    k_weight: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
    slot_mapping: torch.Tensor, eps: float,
) -> torch.Tensor:
    """返回旋转后的 Q，并将旋转 K/V 直接写入物理页。"""
    batch, seqlen, q_heads, head_dim = query.shape
    if batch != 1:
        raise ValueError("fused Qwen3.5 kernel requires batch_size=1")
    kv_heads, tokens = key.shape[2], batch * seqlen
    output = torch.empty((tokens, q_heads, head_dim), device=query.device, dtype=query.dtype)
    rotary_dim = cos.shape[-1]
    cos, sin = cos.reshape(tokens, rotary_dim).contiguous(), sin.reshape(tokens, rotary_dim).contiguous()
    block = triton.next_power_of_2(head_dim)
    _norm_mrope_store_kernel[(tokens, max(q_heads, kv_heads))](
        query, key, value, cos, sin, q_weight, k_weight, output,
        k_cache, v_cache, slot_mapping,
        query.stride(1), query.stride(2), key.stride(1), key.stride(2),
        value.stride(1), value.stride(2), q_heads, kv_heads, head_dim, rotary_dim, eps, block,
    )
    return output.view(batch, seqlen, q_heads, head_dim)
