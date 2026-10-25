"""Qwen3.5 计算层。

计算层负责模型执行、CUDA Graph 和缓存写入，不参与请求生命周期管理。
当前实现参考 vLLM 中已经验证的 Qwen3.5 视觉、Gated DeltaNet、分页注意力
和 CUDA Graph 处理方式，但运行时不依赖 vLLM。
"""
from .executor import ModelExecutor
from .model_runner import ModelRunner

__all__ = ["ModelExecutor", "ModelRunner"]
