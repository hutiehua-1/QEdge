# 基于 QEdge 的优化方向

贡献应解决真实调用链中的问题。文档示例修正属于维护工作，不作为 QEdge 的性能优化成果。以下方向基于现有源码和上游接口检查，尚未形成性能结论。

## 1. 避免 DeltaNet 的重复 Q/K head 展开

QEdge 的 `src/compute/layers/gated_deltanet.py` 在 value head 数量大于 key head 数量时，使用 `repeat_interleave` 展开 Q/K，然后传入 FLA。上游 FLA 的 chunk 和 fused recurrent 接口已支持 Grouped Value Attention：`HV` 可以大于 `H`，且要求 `HV % H == 0`。

建议先在 QEdge 验证直接传递未展开的 Q/K，减少扩展张量和中间读写。这是 QEdge 自身可优先开展的优化，不应把上游已有能力包装成新增上游功能。

验证要求：使用实际模型的 H/HV/K/V，固定归一化、gate 与 beta 开关；比较原展开路径与GVA路径的输出及最终状态；覆盖 Prefill、单token Decode、非空初始状态；记录目标PPU上的算子延迟、端到端TTFT、吞吐和显存。如果厂商FLA分支不支持GVA，需要定位具体版本与缺口，再决定向哪个仓库贡献。

## 2. FLA Decode 的调用方预分配输出缓冲区

上游 `fused_recurrent_gated_delta_rule_fwd` 每次调用通过 `torch.empty_like(v)` 创建输出，并在需要最终状态时分配新的 FP32 状态。QEdge 的 Decode 每步调用这个算子，又通过缓存接口更新递归状态。值得测量的是：调用方持久化输出/状态缓冲区，能否减少分配与后续拷贝，改善图回放集成。

上游贡献建议：先提出可选输出缓冲区接口的 RFC，保持当前默认分配行为；明确形状、dtype、device、layout 与别名约束，以及 `initial_state` 和最终状态是否允许共享存储。不要未经分析直接实现原地状态更新：kernel按V维分块，读写依赖及训练语义必须单独核实。

证据要求：先用 profiler 区分分配、拷贝和 kernel 耗时。需要同时覆盖 state_v_first、等长/变长、非空初始状态和不请求最终状态的情况，并检查图捕获/回放中的地址稳定性。若收益不成立，则不提交性能PR。此方向改变公开API，应先遵守FLA的设计讨论要求。

## 3. 混合前缀快照的选择性复制

QEdge 的 `model_runner.py` 在部分前缀发布及完整快照路径中使用深拷贝。应先确认快照包含哪些状态、哪些历史已由分页KV持有，避免复制可重用或不需要的部分。

建议先在QEdge记录快照字节数、创建/恢复延迟与命中率，再设计只包含恢复必需状态的快照。必须验证递归状态、卷积窗口和MRoPE边界一致性，并覆盖页淘汰和请求切换。只有确认上游缓存API阻碍这一需求时，才向Transformers或FLA提出接口改进。

## 优先顺序

先验证方向1，改动局部、参考路径明确；然后测量方向2和3的真实成本。获得正确性与性能证据后，再决定提交QEdge优化PR、上游RFC或上游实现PR。当前没有设备测量数据，不提交宣称加速的改动。
