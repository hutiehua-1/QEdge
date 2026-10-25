"""用于基准指标的轻量 CUDA 分阶段计时器。

计时过程只记录 CUDA Event，不在推理热路径中同步。引擎在请求结束时统一同步，
随后由 :meth:`snapshot` 将事件对汇总为稳定耗时。将统计逻辑独立于
``ModelRunner``，可以让执行器只关注推理流程本身。
"""

from __future__ import annotations

from collections import defaultdict

import torch


class CudaStageProfiler:
    """累计各 CUDA 阶段耗时，且不在每个阶段结束时触发同步。"""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._total_ms: dict[str, float] = defaultdict(float)
        self._calls: dict[str, int] = defaultdict(int)
        self._pending: dict[
            str, list[tuple[torch.cuda.Event, torch.cuda.Event]]
        ] = defaultdict(list)

    def start(self) -> torch.cuda.Event | None:
        """记录阶段起始事件；关闭统计时返回 ``None``。"""
        if not self.enabled:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def end(self, name: str, start: torch.cuda.Event | None) -> None:
        """记录命名阶段的结束事件，实际耗时延迟到快照阶段计算。"""
        if start is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._pending[name].append((start, end))

    def reset(self) -> None:
        """清空当前请求累计的事件和统计结果。"""
        self._total_ms.clear()
        self._calls.clear()
        self._pending.clear()

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        """在调用方完成 CUDA 同步后，返回各阶段累计耗时。"""
        for name, pairs in self._pending.items():
            self._total_ms[name] += sum(
                float(start.elapsed_time(end)) for start, end in pairs
            )
            self._calls[name] += len(pairs)
        self._pending.clear()
        return {
            name: {
                "total_ms": round(total, 4),
                "calls": self._calls[name],
                "avg_ms": round(total / max(self._calls[name], 1), 4),
            }
            for name, total in self._total_ms.items()
        }
