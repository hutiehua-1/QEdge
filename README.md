# 灵枢 QEdge

面向 Qwen3.5-2B 多模态推理的实验性引擎，围绕分页 KV、混合状态管理、前缀复用与融合算子组织执行路径。项目来自 DNDX 多模态推理调优实践，目标运行环境为平头哥 PPU / T-Head SAIL 适配环境。

[English](README.en.md) · [安装说明](docs/installation.md) · [架构](docs/architecture.md) · [配置](docs/configuration.md) · [评测](docs/benchmarking.md) · [博客](docs/blogs/README.md)

## 当前状态

这是从最新版 `v1.0.zip` 整理的发布候选，42 个 Python 文件保持原样，出处见 [源码清单](docs/source-manifest.json)。主机侧逻辑提供独立测试；本次整理未执行 PPU 推理、算子正确性或性能复测。

- 外部保留比赛的 `VLMModel.generate_with_metrics` 契约，`transformers` 后端名称实际选择自定义引擎。
- 默认开启分页 Attention、融合 Norm/MRoPE/KV 写入、FLA DeltaNet Decode、CUDA Graph 和系统前缀缓存。
- 默认关闭完整前缀缓存、FLA Prefill、GDN 投影合并和 `torch.compile` Prefill。
- 默认最大并行请求数为 1；尚不宣称通用在线服务或跨平台即装即用。
- 本次未擅自给未知来源代码添加许可证，公开发布前须完成 [许可确认](LICENSE-STATUS.md)。

## 快速开始

真实推理需要 Linux 上与目标加速器匹配的软件栈。现有代码使用 `torch.cuda` 接口；本地 Windows CPU 环境仅用于下方的主机测试。完整版本要求见 [安装说明](docs/installation.md)。

1. 按厂商文档准备 PPU/SAIL 和兼容的 PyTorch、Triton、FlashAttention、FlashInfer、FLA。`requirements.txt` 是依赖盘点，不是已验证的环境锁，不建议在厂商环境直接覆盖安装通用 CUDA 包。
2. 将合法取得的 Qwen3.5-2B 模型放在本地 `./Qwen3.5-2B`，公开 MMBench 开发集放在 `./datasets/mmbench/`，获取入口见 [数据说明](docs/data.md)。
3. 在仓库根目录运行：

```bash
python benchmark_public.py \
  --dataset-path ./datasets/mmbench/mmbench_dev_en.tsv \
  --model-path ./Qwen3.5-2B \
  --backend transformers \
  --num-samples 20 \
  --output result_dev_en.json
```

必须显式传入 `--backend transformers`。`auto` 在模型加载失败时可能降级为 dummy；dummy 输出不能作为性能或准确率证据。`transformers` 仍会加载加速依赖，不能用这个入口检验无 GPU 的环境。

### 无加速器的主机测试

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python scripts/check_source.py
```

测试覆盖配置校验、采样边界、页生命周期、多模态块哈希隔离和 Radix 索引。测试按文件加载真实纯逻辑源码，避免执行 `src/__init__.py` 导致加速依赖导入；不替代完整包导入或设备测试。

## 历史评测记录

下表来自随压缩包保存的 `result_dev_en.json`，**未经本次独立复现**。逐样本记录重算与汇总一致，完整结果未收录入仓库。

| 项目 | 历史记录值 |
|---|---:|
| 英文开发集样本数 | 2000 |
| 正确数 / 准确率 | 1613 / 80.65% |
| 平均 TTFT | 60.449 ms |
| 平均样本解码吞吐 | 680.489 tokens/s |
| 公开输出校验失败数 | 0 |

原始记录时间为 `2026-09-01T13:04:01.579879`，未记录时区、完整软硬件版本或模型 revision。不能据此认定硬件型号、端到端系统吞吐或比赛最终成绩，也不能与其他规模的 baseline 直接计算加速比。详见 [结果摘要](docs/results/v1-historical-summary.json) 与 [评测口径](docs/benchmarking.md)。

## 目录

```text
evaluation_wrapper.py       比赛兼容接口、消息构造与计时
benchmark_public.py         公开开发集自测
src/control/                Engine 与 Scheduler
src/data/                   页池、前缀索引、混合状态、批次
src/compute/                Executor、ModelRunner、模型与算子
tests/                      不依赖加速器的主机逻辑检查
docs/                       架构、复现、博客和出处
```

欢迎按 [贡献指南](CONTRIBUTING.md) 提供可复现的问题和最小修改。第三方依赖、数据及模型各自的使用条款仍然适用，见 [第三方说明](THIRD_PARTY_NOTICES.md)。
