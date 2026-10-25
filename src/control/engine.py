from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from ..config import EngineConfig
from ..compute.models import validate_qwen35_config
from ..compute import ModelExecutor
from ..compute.models.loader import load_qwen35_direct, load_qwen35_fast
from ..data import Sequence
from ..sampling_params import SamplingParams
from .scheduler import Scheduler


_MARK = r"(?:\*{1,3}|_{1,3}|`{1,3})*"
_CHOICE = rf"{_MARK}\s*[\(\[（【]?\s*([ABCD])\s*[\)\]）】]?\s*{_MARK}"
_CHOICE_PATTERNS = (
    re.compile(r"(?:\\boxed|boxed)\s*\{\s*([ABCD])\s*\}", re.IGNORECASE),
    re.compile(
        r"(?:final\s*)?(?:answer|choice|option|答案|选项|选择|正确答案|最终答案)"
        rf"\s*(?:(?:is|为|是|[:：])\s*)*{_CHOICE}",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:I(?:'d|\s+would)?\s+(?:choose|select)|my\s+(?:choice|answer)\s+is)"
        rf"\s*[:：]?\s*{_CHOICE}",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:我(?:会)?选|我认为(?:是)?|应(?:该)?选|请选择|故选|所以选|选|答案为|答案是)"
        rf"\s*(?:[:：]\s*)?{_CHOICE}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"^\s*{_CHOICE}\s*(?:[\.。,:：\)\]）】\s]|$)",
        re.IGNORECASE | re.MULTILINE,
    ),
)


def _extract_choice(text: str) -> str | None:
    for pattern in _CHOICE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    return None


class Qwen35Engine:
    """同步 Qwen3.5 推理引擎的控制层。

    职责边界与 dzyy-vllm v3 的 ``Engine`` 保持一致：构造多模态输入、请求
    入队、驱动调度循环并整理性能指标。模型执行、CUDA Graph 和缓存写入由
    ``ModelExecutor``/``ModelRunner`` 负责；整个运行过程不导入 vLLM。
    """

    def __init__(self, model_path: str, config: EngineConfig) -> None:
        startup_start = time.perf_counter()
        import torch
        from transformers import AutoConfig, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("v3 Qwen3.5 backend requires a CUDA GPU")
        # TODO
        # self.first_token_at_perf_counter: float | None = None
        self.torch = torch
        self.device = torch.device("cuda:0")
        self.model_path = str(Path(model_path))
        self.config = config
        self.hf_config = AutoConfig.from_pretrained(
            self.model_path, local_files_only=True, trust_remote_code=True
        )
        self.cache_layout = validate_qwen35_config(self.hf_config)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, local_files_only=True, trust_remote_code=True
        )
        dtype = getattr(torch, config.dtype)
        self.hf_config._attn_implementation = "sdpa"
        # 加载模型权重
        loader = load_qwen35_direct if config.direct_weight_load else load_qwen35_fast
        self.model = loader(
            self.model_path, dtype=dtype, device=self.device
        )
        self.tokenizer = self.processor.tokenizer
        # streamer 每步只需要一个 CPU token；复用固定 buffer，避免 decode
        # 热路径反复构造小 Tensor。
        self.stream_token_cpu = torch.empty(1, dtype=torch.long)
        eos = getattr(self.model.generation_config, "eos_token_id", None)
        if eos is None:
            eos = self.tokenizer.eos_token_id
        eos = eos if isinstance(eos, (list, tuple)) else [eos]
        self.eos_token_ids = {int(token_id) for token_id in eos}
        self.executor = ModelExecutor(self.model, self.hf_config, config, self.device)
        # 分配全局KV Tensor，由executor实施
        kv = self.executor.init_kv()
        self.scheduler = Scheduler(
            kv, max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
            eos_token_id=next(iter(self.eos_token_ids)), mix_prefill_decode=True,
        )
        torch.cuda.synchronize(self.device)
        self.startup_seconds = time.perf_counter() - startup_start
        self.model_memory_mb = torch.cuda.memory_allocated(self.device) / 2**20
        self.model_reserved_mb = torch.cuda.memory_reserved(self.device) / 2**20

    def generate(self, **generation_kwargs: Any):
        """提供与 Transformers ``generate`` 相近的单请求生成接口。

        输入张量、采样参数和 streamer 均由 evaluation wrapper 传入；返回值
        与 HF 一致，包含 prompt token 和新生成 token。v3 特有指标由
        :meth:`generation_meta` 在计时结束后读取，不污染兼容接口。
        """
        torch = self.torch
        multimodal_hash = generation_kwargs.get("_dndx_multimodal_hash")
        multimodal_spans = generation_kwargs.get("_dndx_multimodal_spans", ())
        inputs = {
            key: value for key, value in generation_kwargs.items()
            if key not in {
                # 这些东西不属于输入
                "max_new_tokens", "temperature", "top_p", "do_sample",
                "use_cache", "streamer", "_dndx_multimodal_hash",
                "_dndx_multimodal_spans",
            }
        }
        
        input_ids = inputs.get("input_ids")
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("v3 generate currently requires one input_ids sequence")
        
        requested_max_tokens = int(generation_kwargs.get("max_new_tokens", 64))
        if requested_max_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        max_tokens = max(4, requested_max_tokens)
        
        temperature = float(generation_kwargs.get("temperature", 0.0))
        top_p = float(generation_kwargs.get("top_p", 1.0))
        streamer = generation_kwargs.get("streamer")

        # 判断请求是否适合 Prefix Cache
        pixel_values = inputs.get("pixel_values")
        prefix_cache_eligible = not (
            (self.config.enable_prefix_caching or self.config.enable_system_prefix_caching)
            and pixel_values is not None
            and multimodal_hash is None
        )

        # prompt 只从 GPU 回传一次：同一份 CPU Tensor 同时服务 Sequence 和
        # streamer，避免原实现的 ``tolist()`` 与 ``cpu()`` 两次搬运。
        prompt_ids_cpu = input_ids.detach().cpu()
        seq = Sequence.from_prompt(
            prompt_ids_cpu[0].tolist(), block_size=self.config.block_size, # [[1,2,3]]->[1,2,3]
            sampling=SamplingParams(
                max_tokens=max_tokens,
                min_tokens=4,
                temperature=temperature,
                top_p=top_p,
            ),
            multimodal_hash=multimodal_hash,
            multimodal_spans=tuple(multimodal_spans),
            prefix_cache_eligible=prefix_cache_eligible,
            reusable_prefix_tokens=(
                int(multimodal_spans[0][0]) if multimodal_spans else 0
            ),
        )
        seq.runtime_inputs = inputs
        self.scheduler.add(seq)

        # 把streamer看成一个输出流就行（tokenid->text），下面是设置streamer，告诉它prompt就不要输出了
        if streamer is not None:
            # HF generate 会先推送 prompt；skip_prompt=True 的 streamer 据此
            # 丢弃输入 token，随后只向调用方暴露新生成文本。
            streamer.put(prompt_ids_cpu)
        
        generated = self._run_generation(seq, streamer)

        # evaluation_wrapper 只需要 HF 风格的二维 token Tensor，不依赖其设备。
        # 在 CPU 拼接可避免生成完成后额外发起 GPU 分配和 H2D copy。
        output_ids = torch.cat((
            prompt_ids_cpu,
            torch.tensor([generated], dtype=prompt_ids_cpu.dtype),
        ), dim=1)
        return output_ids

    def _run_generation(self, seq: Sequence, streamer: Any | None) -> list[int]:
        """驱动调度循环，并把新增 token 依次推送给文本 streamer。"""
        torch = self.torch
        # 记录 token_ids 中下一个需要推送的位置，避免每轮生成都创建
        # generated_now 及其切片。
        stream_cursor = seq.num_prompt_tokens
        try:
            with torch.inference_mode():
                while not self.scheduler.is_finished():
                    # 控制层决定本轮执行哪些请求，计算层只负责执行；序列状态和
                    # KV 页所有权只能由调度器修改，避免两个模块同时维护状态。
                    batch = self.scheduler.schedule()
                    if batch.is_empty():
                        raise RuntimeError(
                            "scheduler made no progress while requests remain; "
                            "check KV cache capacity accounting"
                        )
                    prefill_ids, decode_ids = self.executor.run(batch)
                    if batch.prefills:      # prefill阶段
                        self.scheduler.postprocess(batch.prefills, prefill_ids, is_prefill=True)
                        #TODO
                        # if (
                        #     prefill_ids
                        #     and batch.prefills[0].need_sample
                        #     and self.first_token_at_perf_counter is None
                        # ):
                        #     self.first_token_at_perf_counter = time.perf_counter()
                    if batch.decodes:       # decode阶段
                        self.scheduler.postprocess(batch.decodes, decode_ids, is_prefill=False)

                    if (
                        not seq.is_finished
                        and seq.num_completion_tokens >= seq.min_tokens
                    ):
                        generated_text = self.tokenizer.decode(
                            seq.token_ids[seq.num_prompt_tokens:],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )
                        if _extract_choice(generated_text) is not None:
                            self.scheduler.finish(seq)

                    # ModelRunner 已将采样结果同步回 CPU；这里只补充本轮新生成的
                    # token，模拟原 baseline 中 generate() 向 streamer 推送数据。
                    if streamer is not None:
                        while stream_cursor < len(seq.token_ids):
                            token_id = seq.token_ids[stream_cursor]
                            self.stream_token_cpu[0] = token_id
                            streamer.put(self.stream_token_cpu)
                            stream_cursor += 1
        finally:
            if streamer is not None:
                streamer.end()
            # pending GPU token 只服务当前请求；Graph 运行期间保持地址稳定，
            # 请求结束后再统一释放，避免跨请求累积。
            assert self.executor.runner is not None
            self.executor.runner.clear_pending_tokens()
        return list(seq.token_ids[seq.num_prompt_tokens:])

    def reset_request_metrics(self) -> None:
        """在 benchmark 开始计时前重置请求级统计。"""
        assert self.executor.runner is not None
        self.executor.runner.reset_stage_metrics()
        self.scheduler.kv.reset_prefix_cache_stats()
        self.first_token_at_perf_counter = None
        self.torch.cuda.reset_peak_memory_stats(self.device)

    def generation_meta(self, generated: list[int]) -> dict[str, Any]:
        """整理 v3 特有的结束原因、显存占用和优化开关。"""
        torch = self.torch
        hit_eos = bool(generated) and generated[-1] in self.eos_token_ids
        return {
            "finish_reason": "stop" if hit_eos else "length",
            "stage_metrics": self.executor.runner.stage_metrics(),
            "startup_seconds": self.startup_seconds,
            "model_memory_mb": round(self.model_memory_mb, 3),
            "model_reserved_mb": round(self.model_reserved_mb, 3),
            "request_peak_memory_mb": round(
                torch.cuda.max_memory_allocated(self.device) / 2**20, 3
            ),
            "request_extra_peak_memory_mb": round(
                torch.cuda.max_memory_allocated(self.device) / 2**20 - self.model_memory_mb, 3
            ),
            "optimizations": {
                "paged_attention": self.config.enable_paged_attention,
                "fused_norm_mrope_store": self.config.enable_fused_norm_mrope_store,
                "fused_decoder": self.config.enable_fused_decoder,
                "fla_gdn": self.config.enable_fla_gdn,
                "fla_prefill": self.config.enable_fla_prefill,
                "merged_gdn_projections": self.config.enable_merged_gdn_projections,
                "cuda_graph": self.executor.runner.enable_cuda_graph,
                "graph_greedy_sampling": self.config.enable_graph_greedy_sampling,
                "direct_weight_load": self.config.direct_weight_load,
                "persistent_batch": self.config.enable_persistent_batch,
                "causal_conv_backend": self._causal_conv_backend(),
            },
            "kv_cache": {
                "prefix_backend": self.scheduler.kv.prefix_backend,
                "prefix_cache_enabled": self.config.enable_prefix_caching,
                "system_prefix_cache_enabled": self.config.enable_system_prefix_caching,
                "snapshot_interval_blocks": self.scheduler.kv.prefix_snapshot_interval_blocks,
                "admission_mode": self.scheduler.kv.prefix_admission_mode,
                "admission_enabled": self.scheduler.kv.prefix_admission_enabled,
                "request_admitted": self.scheduler.kv.last_prefix_admitted,
                "admission_probe_interval": (
                    self.scheduler.kv.prefix_admission_probe_interval
                ),
                "queries": self.scheduler.kv.stats.queries,
                "hits": self.scheduler.kv.stats.hits,
                "misses": self.scheduler.kv.stats.misses,
                "evictions": self.scheduler.kv.stats.evictions,
                "hit_rate": self.scheduler.kv.stats.hit_rate,
                "usage": self.scheduler.kv.usage,
                "last_prefix_hit_blocks": self.scheduler.kv.last_prefix_hit_blocks,
                "last_cached_tokens": self.scheduler.kv.last_num_cached_tokens,
                "free_blocks": self.scheduler.kv.num_free,
            },
        }

    def _causal_conv_backend(self) -> str:
        from ..compute.layers.causal_conv import (
            CAUSAL_CONV_BACKEND,
            CAUSAL_CONV_DECODE_BACKEND,
        )
        if not self.config.enable_fla_prefill:
            return f"transformers_prefill+{CAUSAL_CONV_DECODE_BACKEND}_decode"
        return CAUSAL_CONV_BACKEND

    def destroy(self) -> None:
        """释放模型引用和 CUDA 缓存，供评测进程安全退出。"""
        if getattr(self, "executor", None) is not None:
            self.executor.destroy()
            self.executor = None
        if getattr(self, "model", None) is not None:
            del self.model
        if hasattr(self, "torch"):
            self.torch.cuda.empty_cache()
