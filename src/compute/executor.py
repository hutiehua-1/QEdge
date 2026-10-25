"""组装模型算子、分页缓存资源和模型执行器。"""

from __future__ import annotations

import gc
import io
from contextlib import redirect_stdout
from dataclasses import replace

import torch

from ..config import EngineConfig
from ..data import KVManager, ScheduledBatch
from .model_runner import ModelRunner
from .layers import (
    PagedAttentionContext,
    PagedQwen35Attention,
    install_fla_gated_deltanet,
    install_paged_attention,
    replace_qwen35_norms,
)
from .models import install_qwen35_decoder


class ModelExecutor:
    """持有计算层资源，并提供统一的调度批次执行接口。

    安装顺序不能改变：分页注意力先包装原始 attention，融合归一化再更新包装
    内的引用，最后在 checkpoint 权重加载完成后替换 FLA DeltaNet。
    """
    def __init__(self, model, hf_config, config: EngineConfig, device: torch.device) -> None:
        self.model, self.config, self.device = model, config, device
        self.hf_config = hf_config
        self.dtype = getattr(torch, config.dtype)
        self.kv: KVManager | None = None
        self.runner: ModelRunner | None = None
        self._mem_baseline = torch.cuda.memory_reserved(device)
        self.attention_context = PagedAttentionContext()
        self.paged_attention_layers = 0
        self.fused_norm_layers = 0
        self.decoder_layers = 0
        self.fla_gdn_layers = 0
        self._model_optimizations_installed = False

    def _prefix_backend(self) -> str:
        enabled = (
            self.config.enable_prefix_caching
            or self.config.enable_system_prefix_caching
        )
        return self.config.prefix_backend if enabled else "none"

    def _make_kv(self, blocks: int) -> KVManager:
        # KVManager 只持有 Full Attention 的 K/V 页；DeltaNet 循环状态和
        # 卷积状态由 Qwen35HybridCache 单独管理。
        return KVManager(
            self.hf_config, num_blocks=blocks, block_size=self.config.block_size,
            device=self.device, dtype=self.dtype, prefix_backend=self._prefix_backend(),
            enable_full_prefix_caching=self.config.enable_prefix_caching,
            prefix_snapshot_interval_blocks=self.config.prefix_snapshot_interval_blocks,
            prefix_admission_mode=self.config.prefix_admission_mode,
            prefix_admission_window=self.config.prefix_admission_window,
            prefix_admission_probe_interval=(
                self.config.prefix_admission_probe_interval
            ),
            prefix_admission_min_token_hit_rate=(
                self.config.prefix_admission_min_token_hit_rate
            ),
        )

    def _resolve_num_blocks(self, reserve_bytes: int) -> int: # 看看能分配多少个blocks
        config = self.config
        blocks_per_sequence = (
            config.max_model_len + config.block_size - 1
        ) // config.block_size
        minimum = max(4, config.max_num_seqs * blocks_per_sequence)
        blocks = KVManager.auto_num_blocks( # 能分配多少个block
            self.hf_config, block_size=config.block_size, dtype=self.dtype,
            device=self.device, gpu_memory_utilization=config.gpu_memory_utilization,
            reserve_bytes=reserve_bytes,
        )
        if blocks < minimum:
            raise RuntimeError(
                f"KV memory budget only fits {blocks} blocks, but max_model_len requires {minimum}"
            )
        return blocks

    def _install_model_optimizations(self) -> None:
        config = self.config
        assert self.kv is not None
        if self._model_optimizations_installed:
            for module in self.model.modules():
                if isinstance(module, PagedQwen35Attention):
                    module.kv_cache = self.kv.kv_cache
                    module.context = self.attention_context
            return
        self.paged_attention_layers = (
            install_paged_attention(
                self.model, self.kv.kv_cache, self.attention_context,
                fused_norm_mrope_store=config.enable_fused_norm_mrope_store,
            )
            if config.enable_paged_attention else 0
        )
        already_fused = bool(getattr(self.model, "_v3_fused_decoder", False))
        self.fused_norm_layers = (
            0 if already_fused else replace_qwen35_norms(self.model)
        ) if config.enable_fused_decoder else 0
        self.decoder_layers = (
            0 if already_fused else install_qwen35_decoder(self.model)
        ) if config.enable_fused_decoder else 0
        if self.decoder_layers:
            self.model._v3_fused_decoder = True
        self.fla_gdn_layers = (
            install_fla_gated_deltanet(
                self.model, merge_projections=config.enable_merged_gdn_projections,
                use_fla_prefill=config.enable_fla_prefill,
            ) if config.enable_fla_gdn else 0
        )
        self._model_optimizations_installed = True

    # 用来估算 CUDA Graph、固定 buffer、采样器等运行时对象需要占用多少显存
    # 这样在分配 KV Cache 时，不会把显存全部占满，导致后续创建 CUDA Graph 或运行时 buffer 时 OOM。
    def _probe_cg_footprint(self) -> int:
        if not (
            self.config.auto_probe_runtime_memory
            and self.config.enable_cuda_graph
            and self.config.enable_paged_attention
            and not self.config.enforce_eager
        ):
            return int(self.config.kv_runtime_reserve_gb * 1024**3)
        fixed_reserve = int(self.config.kv_runtime_reserve_gb * 1024**3)
        probe_config = replace(self.config, enable_cuda_graph=False) # 临时关闭CUDA Graph
        with redirect_stdout(io.StringIO()): # 屏蔽输出的情况下
            probe_kv = self._make_kv(max(4, self.config.max_num_seqs)) # 临时创建最小的KV
            probe_runner = ModelRunner( # 创建ModelRunner，runner初始化时会创建元数据，这个函数计算的就是这个值
                self.model, self.device, attention_context=self.attention_context,
                config=probe_config, kv=probe_kv,
            )
        torch.cuda.synchronize(self.device) # 等待CUDA初始化完成
        footprint = torch.cuda.memory_reserved(self.device) - self._mem_baseline # 计算差值
        probe_runner.destroy() # 
        del probe_runner, probe_kv # 删除资源
        gc.collect()
        torch.cuda.empty_cache() # 清空显存
        return max(fixed_reserve, int(footprint))

    def init_kv(self) -> KVManager:
        if self.kv is not None and self.runner is not None:
            return self.kv
        if self.config.num_kvcache_blocks > 0:
            blocks = self.config.num_kvcache_blocks
        else:
            # 当用户没有显式配置 num_kvcache_blocks 时，_resolve_num_blocks() 会根据可用 GPU 显存计算能创建多少个 KV block。
            reserve = self._probe_cg_footprint()
            blocks = self._resolve_num_blocks(reserve)
            # print(
            #     f"Runtime footprint reserved: {reserve / 2**30:.3f} GiB; "
            #     f"KV pool sized to {blocks} blocks"
            # )
        # 真实分配Tensor
        self.kv = self._make_kv(blocks)
        self._install_model_optimizations() # 构建模型
        self.runner = ModelRunner(
            self.model, self.device, attention_context=self.attention_context,
            config=self.config, kv=self.kv,
        )
        return self.kv

    def reset_kv(self) -> KVManager:
        self.destroy_runtime()
        return self.init_kv()

    def destroy_runtime(self) -> None:
        if self.runner is not None:
            self.runner.destroy()
        self.runner = None
        self.kv = None
        self.attention_context.clear()
        gc.collect()
        torch.cuda.empty_cache()

    def run(self, batch: ScheduledBatch) -> tuple[list[int], list[int]]:
        assert self.runner is not None, "init_kv() must be called before run()"
        return self.runner.run(batch)

    def destroy(self) -> None:
        self.destroy_runtime()
        self.model = None
