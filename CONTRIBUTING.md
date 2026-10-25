# 贡献指南

感谢帮助改进 QEdge。先检索已有 issue/PR，一次修改解决一个明确问题。

## 报告问题

提供代码版本、平台和依赖版本、最小输入/配置、预期与实际行为。性能问题附可复现命令、样本规模、预热与 before/after；不要上传私有图片、答案、权重或访问凭据。

## 验证

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python scripts/check_source.py
```

修改状态管理应覆盖命中/未命中、图片变化、状态恢复与淘汰边界；修改 kernel 必须在目标设备上对照参考输出，标明容差和形状。性能声明需同硬件、同模型、同输入的 before/after。不能以 CPU 测试代替 kernel 验证。

文档变更检查命令、路径和默认值；推理行为修改先添加能复现问题的回归测试。说明实际跑过和未跑的检查，不填写虚构测试结果。AI 辅助贡献也需要作者理解并能解释修改。

## 许可

项目仍在确认源码权利与最终许可证。贡献前先阅读 `LICENSE-STATUS.md`；保留引入代码的来源与所需版权声明。公开接收贡献前应先确定统一许可政策。
