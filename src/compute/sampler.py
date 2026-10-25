from __future__ import annotations

import torch
from torch import nn


@torch.compile(dynamic=True, fullgraph=False)
def sample_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """GPU 采样器；temperature 为零时走专用贪心快速路径。"""
    if temperature <= 1e-5:
        return logits.argmax(dim=-1)
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)
        remove = (sorted_probs.cumsum(dim=-1) - sorted_probs) > top_p
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        position = torch.multinomial(sorted_probs, 1)
        return sorted_indices.gather(1, position).squeeze(1)
    return torch.multinomial(probabilities, 1).squeeze(1)


class BatchSampler(nn.Module):
    """批量 greedy/temperature/top-p 采样，避免逐请求 GPU 同步。"""

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor,
                top_ps: torch.Tensor, *, all_greedy: bool = False,
                use_top_p: bool = False) -> torch.Tensor:
        if all_greedy:
            return logits.argmax(dim=-1)
        greedy = temperatures <= 1e-5
        safe = torch.where(greedy, torch.ones_like(temperatures), temperatures)
        probs = torch.softmax(logits.float() / safe[:, None], dim=-1)
        # 是否启用 top-p 来自 CPU 请求参数，不读取 GPU 标量。
        if use_top_p:
            vals, idx = torch.sort(probs, descending=True, dim=-1)
            remove = (vals.cumsum(-1) - vals) > top_ps[:, None]
            vals = vals.masked_fill(remove, 0)
            sampled = idx.gather(1, torch.multinomial(vals, 1)).squeeze(1)
        else:
            sampled = torch.multinomial(probs, 1).squeeze(1)
        return torch.where(greedy, logits.argmax(dim=-1), sampled)
