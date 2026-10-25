# 第三方依赖与来源待核对项

下表盘点运行依赖和用途，不等于宣称这些项目的代码已被复制进 QEdge，也不替代各项目许可证。

| 上游 | 在项目中的用途 |
|---|---|
| [PyTorch](https://github.com/pytorch/pytorch) | 张量、图执行、自定义算子 |
| [Transformers](https://github.com/huggingface/transformers) | Qwen3.5 模型、Processor、缓存 API |
| [Triton](https://github.com/triton-lang/triton) | 融合 kernel |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | Attention 调用；PPU 环境需匹配适配构建 |
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer) | SiLU 与 Gemma RMSNorm |
| [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) | Gated DeltaNet 与卷积路径 |
| [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) | 可选卷积实现 |
| [safetensors](https://github.com/huggingface/safetensors) | 权重读取 |
| [NumPy](https://github.com/numpy/numpy)、[Pillow](https://github.com/python-pillow/Pillow) | 主机数据与图片 |
| [python-xxhash](https://github.com/ifduyue/python-xxhash) | 可选块哈希 |

项目名、算法相似或 API 兼容不能证明独立创作，也不能证明复制。本次保持原源码内容，逐文件 SHA256 在 `docs/source-manifest.json`。作者需进一步提供改编来源；确认前不能把本文件当作完整 NOTICE 合规证明。
