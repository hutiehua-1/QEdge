# Qwen3.5 推理优化：算子融合与 Prefill/Decode 路径选择

> 摘要：减少 kernel 调用和中间读写是推理优化的常见方向，但融合越多并不一定越快。QEdge 把自有融合实现、第三方算子集成和执行阶段选路结合起来，并用开关保留可比较的路径。
>
> 状态：基于 v1.0 的实现解读；没有在本文中报告未经验证的单算子加速比。

## 先明确算子的来源

QEdge 中同时存在自己组织的 Triton kernel、模型结构封装，以及 FlashInfer、FlashAttention、FLA 等第三方调用。它们承担不同职责。

例如 `src/data/kv_ops.py` 中的融合路径把 Q/K Norm、MRoPE 和分页 KV 写入组织到同一个实现中；`src/compute/layers/ops.py` 则为 FlashInfer 的 SiLU 和 Gemma RMSNorm 建立 `torch.library.custom_op` 边界。后者是第三方算子的集成，不应全部写成自研 kernel。

## 融合解决什么问题

连续的小算子可能各自产生调度成本和中间张量读写。如果数据在相邻步骤中具有明确的使用关系，融合可以减少这些环节。

```text
分离路径：Q/K Norm → MRoPE → KV 写入
融合路径：在同一 kernel 中完成相邻步骤
```

但融合也会改变寄存器、线程组织和访存行为。我们只能把“减少调用与中间流量”当作优化假设，不能仅凭融合后的代码较短就认定硬件利用率提高。正确的论证需要参考输出对照、形状矩阵、设备微基准与端到端结果。

## Prefill 和 Decode 的计算形态不同

Prefill 一次处理多个输入 token，Decode 通常每步只推进一个 token。一个阶段更好的实现不一定适合另一个阶段。

Gated DeltaNet 封装保留参考层。当输入长度大于 1 且 `use_fla_prefill` 关闭时，它返回参考 Prefill 路径；单 token Decode 则可以进入 FLA 的融合递归计算。项目默认启用 `enable_fla_gdn`，关闭 `enable_fla_prefill`。

这组默认值表明项目采取了分阶段选路，不能写成“全流程使用 FLA 加速”。代码注释中的特定设备观察也不等于具有环境和命令记录的可复现实验。

## 图执行仍需处理可变状态

Decode 重复执行类似计算，为图捕获与回放提供了机会。项目包含持久化缓冲区、采样与图执行相关路径，意图减少重复执行的组织开销。

然而，模型的缓存状态会随 token 推进；捕获过程本身也可能改变状态。因此，相关实现需要保存和恢复捕获时的混合状态，确认回放读取的是当前请求的 token 与位置。启用 CUDA Graph 开关本身不足以证明所有形状都命中图路径。

图执行验证应区分首次捕获、稳态回放、回退路径和请求切换。报告时需说明是否包含编译和初始化成本。

## 给每项优化准备独立证据

建议以同一代码版本、模型与数据为起点，逐项比较融合 Norm/MRoPE/KV、图执行、系统前缀缓存和 FLA Prefill。每轮保留实际环境变量、答案准确率、TTFT、样本吞吐、端到端耗时与显存。

有两点容易误读：第一，`max_num_seqs=1` 时的样本吞吐不是多请求服务吞吐；第二，prompt 重排和输出规范化可能影响文本与准确率，需要与纯计算优化分开检查。

随包的历史结果记录了整体运行表现，但缺少同条件消融。因此，现阶段可准确说明已经集成哪些路径，不能把整体收益归因到某一个融合算子。

## PPU 环境的复现提醒

代码使用 `torch.cuda` 接口和多个算子包，运行依赖厂商适配构建。直接安装公开同名 CUDA 包并不保证能在 PPU 上运行。复现应从 [平头哥开发者中心](https://developer.t-head.cn/) 的环境说明开始，记录实际 SAIL、驱动和算子版本。

## 源码与上游

- [融合 KV 操作](../../src/data/kv_ops.py)
- [FlashInfer 调用边界](../../src/compute/layers/ops.py)
- [Gated DeltaNet 选路](../../src/compute/layers/gated_deltanet.py)
- [ModelRunner 与图执行](../../src/compute/model_runner.py)
- [FLA](https://github.com/fla-org/flash-linear-attention)、[FlashInfer](https://github.com/flashinfer-ai/flashinfer)、[FlashAttention](https://github.com/Dao-AILab/flash-attention)
