# 配置表

以下默认值与环境变量由当前 `src/config.py` 的语法树生成；以可执行值为准，避免沿用旧注释。

| 字段 | 默认值 | 环境变量 |
|---|---|---|
| `dtype` | `bfloat16` | `DNDX_V3_DTYPE` |
| `gpu_memory_utilization` | `0.9` | `DNDX_V3_GPU_MEMORY_UTILIZATION` |
| `num_kvcache_blocks` | `8192` | `DNDX_V3_NUM_KVCACHE_BLOCKS` |
| `kv_runtime_reserve_gb` | `1.0` | `DNDX_V3_KV_RESERVE_GB` |
| `auto_probe_runtime_memory` | `True` | `DNDX_V3_AUTO_PROBE_RUNTIME_MEMORY` |
| `torch_compile_prefill` | `False` | `DNDX_V3_TORCH_COMPILE_PREFILL` |
| `compile_mode` | `reduce-overhead` | `DNDX_V3_COMPILE_MODE` |
| `enable_paged_attention` | `True` | `DNDX_V3_PAGED_ATTENTION` |
| `enable_fused_norm_mrope_store` | `True` | `DNDX_V3_FUSED_NORM_MROPE_STORE` |
| `enable_fused_decoder` | `True` | `DNDX_V3_FUSED_DECODER` |
| `enable_fla_gdn` | `True` | `DNDX_V3_FLA_GDN` |
| `enable_fla_prefill` | `False` | `DNDX_V3_FLA_PREFILL` |
| `enable_merged_gdn_projections` | `False` | `DNDX_V3_MERGED_GDN_PROJECTIONS` |
| `enable_persistent_batch` | `True` | `DNDX_V3_PERSISTENT_BATCH` |
| `direct_weight_load` | `True` | `DNDX_V3_DIRECT_WEIGHT_LOAD` |
| `enable_cuda_graph` | `True` | `DNDX_V3_CUDA_GRAPH` |
| `enable_graph_greedy_sampling` | `True` | `DNDX_V3_GRAPH_GREEDY_SAMPLING` |
| `max_model_len` | `8192` | `DNDX_V3_MAX_MODEL_LEN` |
| `max_num_seqs` | `1` | `DNDX_V3_MAX_NUM_SEQS` |
| `max_num_batched_tokens` | `8192` | `DNDX_V3_MAX_BATCHED_TOKENS` |
| `block_size` | `256` | `DNDX_V3_BLOCK_SIZE` |
| `enforce_eager` | `False` | `DNDX_V3_ENFORCE_EAGER` |
| `enable_prefix_caching` | `False` | `DNDX_V3_PREFIX_CACHING` |
| `enable_system_prefix_caching` | `True` | `DNDX_V3_SYSTEM_PREFIX_CACHING` |
| `prefix_backend` | `hash` | `DNDX_V3_PREFIX_BACKEND` |
| `prefix_snapshot_interval_blocks` | `32` | `DNDX_V3_PREFIX_SNAPSHOT_INTERVAL_BLOCKS` |
| `prefix_admission_mode` | `adaptive` | `DNDX_V3_PREFIX_ADMISSION` |
| `prefix_admission_window` | `4` | `DNDX_V3_PREFIX_ADMISSION_WINDOW` |
| `prefix_admission_probe_interval` | `32` | `DNDX_V3_PREFIX_ADMISSION_PROBE_INTERVAL` |
| `prefix_admission_min_token_hit_rate` | `0.05` | `DNDX_V3_PREFIX_ADMISSION_MIN_TOKEN_HIT_RATE` |
| `mm_processor_cache_gb` | `0.0` | `DNDX_V3_MM_CACHE_GB` |
| `reorder_multimodal_prompt` | `True` | `DNDX_V3_REORDER_MM_PROMPT` |

布尔环境变量接受 `1/true/yes/on` 或 `0/false/no/off`，大小写不敏感。非法值会抛出异常。

例如，在 Linux 中比较完整前缀缓存：

```bash
DNDX_V3_PREFIX_CACHING=1 python benchmark_public.py --backend transformers --dataset-path ./datasets/mmbench/mmbench_dev_en.tsv --model-path ./Qwen3.5-2B --num-samples 20 --output result_prefix.json
```

这个例子是待在目标环境执行的消融命令，不代表本次已验证其收益。块大小必须为 256 的倍数。设 `num_kvcache_blocks=0` 可选择自动估算，真实内存风险仍需设备验证。

原源码注释中存在与实际值不一致的历史描述，例如 8192 个 KV 块旁仍提到 64，32 块快照间隔旁仍提到 4。本次保留源文件以便校验，文档使用实际默认值。
