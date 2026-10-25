from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


@dataclass(frozen=True)
class EngineConfig:
    """Qwen3.5 推理引擎配置。

    配置项按显存策略、算子开关、执行上限和前缀缓存策略组织。所有影响基准
    性能的优化都提供环境变量开关，便于逐项测试收益和额外开销。
    """

    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.90
    # 64 × 256 = 16K token slots，覆盖本项目单请求 8K 上下文并留出余量。
    # 0 可恢复按剩余显存自动计算，但会挤占视觉/FLA 编译 workspace。
    num_kvcache_blocks: int = 8192
    kv_runtime_reserve_gb: float = 1.0
    auto_probe_runtime_memory: bool = True
    torch_compile_prefill: bool = False
    compile_mode: str = "reduce-overhead"
    enable_paged_attention: bool = True
    enable_fused_norm_mrope_store: bool = True
    enable_fused_decoder: bool = True
    enable_fla_gdn: bool = True
    # Transformers 的多 token 路径已使用 vendor grouped-conv，且在当前
    # 硬件上整体快于 FLA chunk GDN；Decode 仍使用 FLA fused recurrent。
    enable_fla_prefill: bool = False
    # Qwen3.5-2B 上合并后的不对称大 GEMM 比四个专用 GEMM 慢约 2%，默认关闭。
    enable_merged_gdn_projections: bool = False
    enable_persistent_batch: bool = True
    direct_weight_load: bool = True
    enable_cuda_graph: bool = True
    enable_graph_greedy_sampling: bool = True
    max_model_len: int = 8192
    max_num_seqs: int = 1
    max_num_batched_tokens: int = 8192
    block_size: int = 256
    enforce_eager: bool = False
    enable_prefix_caching: bool = False
    # Cache the stable text before the image even when it is shorter than one
    # 256-token physical page. This avoids full multimodal snapshot overhead.
    enable_system_prefix_caching: bool = True
    prefix_backend: str = "hash"
    # Hybrid/DeltaNet 状态比 Full Attention KV page 更重；每 4 个物理块保存
    # 一个恢复点，同时调度器会额外保存 prompt 尾部最后一个可复用边界。
    prefix_snapshot_interval_blocks: int = 32
    # adaptive 会在低复用窗口停止为普通 miss 创建昂贵的 Hybrid snapshot，
    # 但周期性放行一个探测请求，以便工作负载变化后重新启用缓存。
    prefix_admission_mode: str = "adaptive"
    prefix_admission_window: int = 4
    prefix_admission_probe_interval: int = 32
    prefix_admission_min_token_hit_rate: float = 0.05
    mm_processor_cache_gb: float = 0.0
    # 将稳定任务说明放到图片之前，使其 KV 不依赖当前图片。只调整已识别的
    # 单选题模板，普通单行/自由格式 prompt 保持原消息结构。
    reorder_multimodal_prompt: bool = True

    @classmethod
    def from_env(cls) -> "EngineConfig":
        """读取 ``DNDX_V3_*`` 环境变量并生成经过校验的配置。"""
        value = cls(
            dtype=os.getenv("DNDX_V3_DTYPE", cls.dtype),
            gpu_memory_utilization=float(os.getenv(
                "DNDX_V3_GPU_MEMORY_UTILIZATION", cls.gpu_memory_utilization
            )),
            num_kvcache_blocks=int(os.getenv(
                "DNDX_V3_NUM_KVCACHE_BLOCKS", cls.num_kvcache_blocks
            )),
            kv_runtime_reserve_gb=float(os.getenv(
                "DNDX_V3_KV_RESERVE_GB", cls.kv_runtime_reserve_gb
            )),
            auto_probe_runtime_memory=_env_bool(
                "DNDX_V3_AUTO_PROBE_RUNTIME_MEMORY", cls.auto_probe_runtime_memory
            ),
            torch_compile_prefill=_env_bool(
                "DNDX_V3_TORCH_COMPILE_PREFILL", cls.torch_compile_prefill
            ),
            compile_mode=os.getenv("DNDX_V3_COMPILE_MODE", cls.compile_mode),
            enable_paged_attention=_env_bool(
                "DNDX_V3_PAGED_ATTENTION", cls.enable_paged_attention
            ),
            enable_fused_norm_mrope_store=_env_bool(
                "DNDX_V3_FUSED_NORM_MROPE_STORE", cls.enable_fused_norm_mrope_store
            ),
            enable_fused_decoder=_env_bool("DNDX_V3_FUSED_DECODER", cls.enable_fused_decoder),
            enable_fla_gdn=_env_bool("DNDX_V3_FLA_GDN", cls.enable_fla_gdn),
            enable_fla_prefill=_env_bool(
                "DNDX_V3_FLA_PREFILL", cls.enable_fla_prefill
            ),
            enable_merged_gdn_projections=_env_bool(
                "DNDX_V3_MERGED_GDN_PROJECTIONS", cls.enable_merged_gdn_projections
            ),
            enable_persistent_batch=_env_bool(
                "DNDX_V3_PERSISTENT_BATCH", cls.enable_persistent_batch
            ),
            direct_weight_load=_env_bool("DNDX_V3_DIRECT_WEIGHT_LOAD", cls.direct_weight_load),
            enable_cuda_graph=_env_bool("DNDX_V3_CUDA_GRAPH", cls.enable_cuda_graph),
            enable_graph_greedy_sampling=_env_bool(
                "DNDX_V3_GRAPH_GREEDY_SAMPLING", cls.enable_graph_greedy_sampling
            ),
            max_model_len=int(os.getenv("DNDX_V3_MAX_MODEL_LEN", cls.max_model_len)),
            max_num_seqs=int(os.getenv("DNDX_V3_MAX_NUM_SEQS", cls.max_num_seqs)),
            max_num_batched_tokens=int(os.getenv(
                "DNDX_V3_MAX_BATCHED_TOKENS", cls.max_num_batched_tokens
            )),
            block_size=int(os.getenv("DNDX_V3_BLOCK_SIZE", cls.block_size)),
            enforce_eager=_env_bool("DNDX_V3_ENFORCE_EAGER", cls.enforce_eager),
            enable_prefix_caching=_env_bool("DNDX_V3_PREFIX_CACHING", cls.enable_prefix_caching),
            enable_system_prefix_caching=_env_bool(
                "DNDX_V3_SYSTEM_PREFIX_CACHING", cls.enable_system_prefix_caching
            ),
            prefix_backend=os.getenv("DNDX_V3_PREFIX_BACKEND", cls.prefix_backend),
            prefix_snapshot_interval_blocks=int(os.getenv(
                "DNDX_V3_PREFIX_SNAPSHOT_INTERVAL_BLOCKS",
                cls.prefix_snapshot_interval_blocks,
            )),
            prefix_admission_mode=os.getenv(
                "DNDX_V3_PREFIX_ADMISSION", cls.prefix_admission_mode
            ),
            prefix_admission_window=int(os.getenv(
                "DNDX_V3_PREFIX_ADMISSION_WINDOW", cls.prefix_admission_window
            )),
            prefix_admission_probe_interval=int(os.getenv(
                "DNDX_V3_PREFIX_ADMISSION_PROBE_INTERVAL",
                cls.prefix_admission_probe_interval,
            )),
            prefix_admission_min_token_hit_rate=float(os.getenv(
                "DNDX_V3_PREFIX_ADMISSION_MIN_TOKEN_HIT_RATE",
                cls.prefix_admission_min_token_hit_rate,
            )),
            mm_processor_cache_gb=float(os.getenv("DNDX_V3_MM_CACHE_GB", cls.mm_processor_cache_gb)),
            reorder_multimodal_prompt=_env_bool(
                "DNDX_V3_REORDER_MM_PROMPT", cls.reorder_multimodal_prompt
            ),
        )
        value.validate()
        return value

    def validate(self) -> None:
        """在模型加载前检查配置，尽早暴露不支持的组合。"""
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        limits = (
            self.max_model_len, self.max_num_seqs,
            self.max_num_batched_tokens, self.block_size,
        )
        if min(limits) <= 0:
            raise ValueError("length, batch and block settings must be positive")
        if self.mm_processor_cache_gb < 0:
            raise ValueError("mm_processor_cache_gb cannot be negative")
        if self.num_kvcache_blocks < 0:
            raise ValueError("num_kvcache_blocks cannot be negative")
        if self.kv_runtime_reserve_gb < 0:
            raise ValueError("kv_runtime_reserve_gb cannot be negative")
        if self.block_size % 256 != 0:
            raise ValueError("block_size must be a multiple of 256 for Sail paged FlashAttention")
        if self.prefix_backend not in {"hash", "radix"}:
            raise ValueError("prefix_backend must be hash or radix")
        if self.prefix_snapshot_interval_blocks <= 0:
            raise ValueError("prefix_snapshot_interval_blocks must be positive")
        if self.prefix_admission_mode not in {"always", "adaptive"}:
            raise ValueError("prefix_admission_mode must be always or adaptive")
        if self.prefix_admission_window <= 0:
            raise ValueError("prefix_admission_window must be positive")
        if self.prefix_admission_probe_interval <= 0:
            raise ValueError("prefix_admission_probe_interval must be positive")
        if not 0.0 <= self.prefix_admission_min_token_hit_rate <= 1.0:
            raise ValueError("prefix_admission_min_token_hit_rate must be in [0, 1]")
        if self.compile_mode not in {"default", "reduce-overhead", "max-autotune"}:
            raise ValueError("unsupported torch.compile mode")


DEFAULT_CONFIG = EngineConfig.from_env()
