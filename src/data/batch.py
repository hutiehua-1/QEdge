from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .sequence import Sequence, slots_for_range


@dataclass
class ScheduledBatch:
    """调度器产出的本步批次。compute 层只消费这个显式接口，不触碰 Sequence 内部。"""

    prefills: list[Sequence] = field(default_factory=list)
    decodes: list[Sequence] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.prefills and not self.decodes


class Batch:
    """常驻 GPU buffer：prefill / decode / CUDA Graph 共用同一套静态张量。

    每步只 copy_ 实际数据进 buffer，避免反复分配 GPU 张量；padding 区域保持
    捕获时的安全值（slot=-1、seqlen=0、block=0），使 CUDA Graph 重放无需额外清理。
    """

    def __init__(
        self,
        *,
        max_bs: int,
        max_tokens: int,
        max_blocks: int,
        vocab_size: int,
        device: torch.device,
        persistent_host_staging: bool = True,
    ) -> None:
        self.max_bs = max_bs
        self.max_tokens = max_tokens
        self.max_blocks = max_blocks
        self.vocab_size = vocab_size
        self.device = device
        self.persistent_host_staging = persistent_host_staging

        self.input_ids = torch.zeros(max_tokens, dtype=torch.int64, device=device)
        self.positions = torch.zeros(max_tokens, dtype=torch.int64, device=device)
        self.slot_mapping = torch.full(
            (max_tokens,), -1, dtype=torch.int32, device=device
        )
        self.cache_seqlens = torch.zeros(max_bs, dtype=torch.int32, device=device)
        self.cu_seqlens_q = torch.zeros(max_bs + 1, dtype=torch.int32, device=device)
        self.cu_seqlens_k = torch.zeros(max_bs + 1, dtype=torch.int32, device=device)
        self.block_tables = torch.zeros(
            max_bs, max_blocks, dtype=torch.int32, device=device
        )

        self.temperatures = torch.zeros(max_bs, dtype=torch.float32, device=device)
        self.top_ps = torch.zeros(max_bs, dtype=torch.float32, device=device)
        self.top_ks = torch.zeros(max_bs, dtype=torch.int32, device=device)
        self.min_ps = torch.zeros(max_bs, dtype=torch.float32, device=device)
        self.all_greedy = True
        self.any_greedy = False
        self.use_top_k = False
        self.use_top_p = False
        self.use_min_p = False
        self.max_top_k = 0
        # 每个常驻 GPU buffer 对应一个可复用的固定页内存镜像，避免每生成
        # 一个 token 都重新申请 pinned memory。
        self._host_staging: dict[tuple[int, torch.dtype], torch.Tensor] = {}
        # Decode 期间页表和采样参数通常保持不变；缓存 CPU 签名，避免每个
        # token 都重复填充 pinned staging 并发起 H2D copy。
        self._block_table_signature: tuple[tuple[int, ...], ...] | None = None
        self._sampling_signature: tuple[tuple[float, float, int, float], ...] | None = None

    def _cp(self, dst: torch.Tensor, data: list, dtype: torch.dtype) -> None:
        if not self.persistent_host_staging:
            dst.copy_(torch.tensor(data, dtype=dtype, pin_memory=True), non_blocking=True)
            return
        storage = dst.untyped_storage()
        key = (storage.data_ptr(), dtype)
        capacity = storage.nbytes() // torch.empty((), dtype=dtype).element_size()
        cpu = self._host_staging.get(key)
        if cpu is None or cpu.numel() < capacity:
            cpu = torch.empty(capacity, dtype=dtype, pin_memory=True)
            self._host_staging[key] = cpu
        # NumPy 直接写入常驻固定页内存，随后再异步拷贝至 GPU。
        cpu.numpy()[:len(data)] = data
        dst.copy_(cpu[:len(data)].view_as(dst), non_blocking=True)

    def build_decode(
        self,
        seqs: list[Sequence],
        *,
        token_ids_gpu: torch.Tensor | None = None,
    ) -> None:
        bs = len(seqs)
        block_size = seqs[0].block_size
        # 当 next token 已由 ModelRunner 保存在 GPU 上时，不再经过 Python
        # list/pinned staging；input_ids 仍保留为 fallback 路径。
        if token_ids_gpu is None:
            self._cp(self.input_ids[:bs], [s.last_token for s in seqs], torch.int64)
        self._cp(
            self.slot_mapping[:bs],
            [self._decode_slot(s, block_size) for s in seqs],
            torch.int32,
        )
        self._cp(self.cache_seqlens[:bs], [len(s) for s in seqs], torch.int32)
        self._fill_block_tables(seqs)
        self._fill_sampling(seqs)

    def build_prefill(self, seqs: list[Sequence]) -> tuple[int, int, int, bool]:
        block_size = seqs[0].block_size
        slots: list[int] = []
        cu_q = [0]
        cu_k = [0]
        max_q = 0
        max_k = 0
        for s in seqs:
            start = s.num_cached_tokens
            n = s.num_scheduled_tokens
            end = start + n
            slots.extend(slots_for_range(s.block_table, start, end, block_size))
            cu_q.append(cu_q[-1] + n)
            cu_k.append(cu_k[-1] + end)
            max_q = max(max_q, n)
            max_k = max(max_k, end)

        n = len(slots)
        self._cp(self.slot_mapping[:n], slots, torch.int32)

        has_cache = cu_k[-1] > cu_q[-1]
        # Paged prefill 无论是否命中前缀都需要完整 block table；不能依赖零填充。
        self._fill_block_tables(seqs)
        self._fill_sampling(seqs)
        return n, max_q, max_k, has_cache

    def _fill_block_tables(self, seqs: list[Sequence]) -> None:
        signature = tuple(tuple(s.block_table) for s in seqs)
        if signature == self._block_table_signature:
            return
        width = self.max_blocks
        flat: list[int] = []
        for s in seqs:
            flat.extend(s.block_table)
            flat.extend([-1] * (width - len(s.block_table)))
        self._cp(self.block_tables[: len(seqs)].reshape(-1), flat, torch.int32)
        self._block_table_signature = signature

    def _fill_sampling(self, seqs: list[Sequence]) -> None:
        bs = len(seqs)
        vocab = self.vocab_size
        temps = [s.temperature for s in seqs]
        top_ps = [1.0 if s.top_p is None else s.top_p for s in seqs]
        top_ks = [
            vocab if (s.top_k is None or s.top_k <= 0) else min(s.top_k, vocab)
            for s in seqs
        ]
        min_ps = [0.0 if s.min_p is None else s.min_p for s in seqs]

        signature = tuple(zip(temps, top_ps, top_ks, min_ps))
        if signature == self._sampling_signature:
            return

        self._cp(self.temperatures[:bs], temps, torch.float32)
        self._cp(self.top_ps[:bs], top_ps, torch.float32)
        self._cp(self.top_ks[:bs], top_ks, torch.int32)
        self._cp(self.min_ps[:bs], min_ps, torch.float32)

        self.all_greedy = all(t <= 1e-5 for t in temps)
        self.any_greedy = any(t <= 1e-5 for t in temps)
        self.use_top_k = any(k < vocab for k in top_ks)
        self.use_top_p = any(p < 1.0 for p in top_ps)
        self.use_min_p = any(m > 0.0 for m in min_ps)
        self.max_top_k = max(top_ks) if self.use_top_k else 0
        self._sampling_signature = signature

    @staticmethod
    def _decode_slot(seq: Sequence, block_size: int) -> int:
        n = len(seq)
        return seq.block_table[(n - 1) // block_size] * block_size + (n - 1) % block_size
