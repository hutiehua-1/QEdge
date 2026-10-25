"""Qwen3.5 Gemma 风格 RMSNorm 的公共导出入口。"""

from .qwen35_ops import Qwen35RMSNorm, replace_qwen35_norms

__all__ = ["Qwen35RMSNorm", "replace_qwen35_norms"]
