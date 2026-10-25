# 安装与环境边界

## 主机检查

整理时使用 Windows、Python 3.13；本机没有 PyTorch。CI 使用 Python 3.10 与 3.12，只安装 NumPy 并运行纯主机检查。Python 3.10 是源码联合类型等语法的最低要求，不意味着全部加速依赖都支持该版本。

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python scripts/check_source.py
```

## 真实推理

目标是 Linux + PPU/SAIL 适配环境，代码通过 `torch.cuda`、Triton 和适配后的算子包执行。没有已验证的通用 Windows、CPU、原生 NVIDIA 或其他平台运行矩阵。

1. 根据 [平头哥开发者中心](https://developer.t-head.cn/) 文档配置厂商推荐的系统、驱动和 SDK。
2. 准备厂商匹配的 PyTorch、Triton、FlashAttention、FlashInfer 与 FLA；当前项目导入 `flash_attn`、`flashinfer`、`fla`，同名公开包不保证支持目标设备。
3. 补齐 Transformers（含 Qwen3.5 模型与缓存 API）、safetensors、NumPy、Pillow。可选 `causal-conv1d` 与 `xxhash` 分别用于卷积快速路径与哈希加速。
4. 准备本地模型目录和数据，按 README 的显式真实后端命令运行少量样本。
5. 记录环境后再扩大实验。原始包缺少环境锁，尚不能提供诚实的固定版本组合。

依赖盘点在 `requirements.txt`。不要在现有厂商环境直接执行整文件升级，以免安装器把适配包替换为不兼容的通用构建。`causal-conv1d` 的存在与否会影响路径，`xxhash` 的存在与否会改变哈希算法；对照实验必须固定依赖组合。

## 复现时需要记录

硬件型号/数量/显存，OS、驱动、SAIL、Python、PyTorch、Triton、FlashAttention、FlashInfer、FLA、Transformers 的版本与来源，模型 revision，数据集文件 SHA256，代码提交号，所有 `DNDX_V3_*` 环境变量，命令、预热和样本顺序。可把 `python -m pip freeze` 输出保存到本地实验目录，但不上传凭据、内部镜像地址或机器路径。

## 常见问题

- 缺少 `flash_attn` / `flashinfer`：先核对厂商构建，关闭一个算子开关并不能消除模块顶层导入依赖。
- 加载失败却有结果：检查是否用了 `auto` 并回退 dummy；真实结果 `backend` 应为 `v3`。
- 显存不足：检查 `num_kvcache_blocks` 的实际默认值 8192；注释中的 64 并非当前配置。可在记录清楚后降低 `DNDX_V3_NUM_KVCACHE_BLOCKS`，但本次未给出已验证的设备推荐值。
- 算子签名或缓存 API 不匹配：记录实际版本，不能仅以“最新版”代替兼容性检查。
