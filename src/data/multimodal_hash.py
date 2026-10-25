from __future__ import annotations

import hashlib
import json
from typing import Any

import torch


def multimodal_cache_key(image: Any, *, grid_thw: tuple[int, int, int] | None = None,
                         processor_config: dict[str, Any] | None = None) -> str:
    """基于媒体内容生成缓存键；仅使用图像占位 token 无法区分不同图片。"""
    digest = hashlib.sha256()
    if hasattr(image, "convert") and hasattr(image, "save"):
        # 对解码后的规范 RGB 像素直接哈希，避免 PNG 重编码；调用方应在输入
        # 搬到 GPU 前完成这一步，使其不触发 generate 热路径的设备同步。
        rgb = image.convert("RGB")
        digest.update(str(rgb.size).encode())
        digest.update(rgb.tobytes())
    elif isinstance(image, (bytes, bytearray, memoryview)):
        digest.update(bytes(image))
    elif hasattr(image, "detach") and hasattr(image, "cpu"):
        # evaluation_wrapper 已完成图像预处理时，使用像素张量内容区分请求。
        tensor = image.detach().contiguous().cpu()
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    else:
        raise TypeError("image must be PIL-compatible, bytes-like, or a tensor")
    metadata = {"grid_thw": grid_thw, "processor": processor_config or {}}
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def multimodal_token_spans(
    token_type_ids: Any,
    identifiers: list[str] | tuple[str, ...],
) -> tuple[tuple[int, int, str], ...]:
    """从 processor 的 ``mm_token_type_ids`` 提取连续媒体 token 区间。"""
    if token_type_ids is None or not identifiers:
        return ()
    values = token_type_ids.detach().cpu().reshape(-1).tolist()
    spans: list[tuple[int, int, str]] = []
    start: int | None = None
    run_index = 0
    for index, value in enumerate(values + [0]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            identifier = identifiers[min(run_index, len(identifiers) - 1)]
            spans.append((start, index, identifier))
            start = None
            run_index += 1
    return tuple(spans)
