"""执行调度器产出的 Qwen3.5 prefill 与 decode 批次。

执行器持有请求级静态 GPU buffer 和可复用的 decode CUDA Graph。调度策略与
KV 页所有权仍由 control/data 层维护；本模块只把 ``ScheduledBatch`` 转换成
模型调用和 token ID。模型结构沿用 Transformers，热路径数值算子则由
:mod:`src.compute.layers` 中的本地实现替换。
"""

from __future__ import annotations

import torch
import copy
import os
import time

from ..data import Batch, Qwen35HybridCache, ScheduledBatch, Sequence
from .sampler import BatchSampler
from .profiler import CudaStageProfiler


class ModelRunner:
    """单个 CUDA 设备上一个模型副本的有状态执行器。"""

    def __init__(self, model, device: torch.device, *, attention_context=None,
                 config=None, kv=None) -> None:
        self.model, self.device = model, device
        self.attention_context = attention_context
        self.sampler = BatchSampler().to(device)
        self.copy_stream = torch.cuda.Stream(device=device)
        self.copy_event = torch.cuda.Event()
        self.sample_cpu = torch.empty(256, dtype=torch.int64, pin_memory=True)
        # 固定 GPU slot，供下一轮 decode 直接作为 input_ids；seq_id 只映射
        # 到 slot，不在请求间释放底层 Tensor，满足 CUDA Graph 固定地址要求。
        # CPU mirror 仍由 sample_cpu 提供，scheduler 的权威状态不变。
        self.pending_next_tokens_gpu = torch.empty(
            config.max_num_seqs, dtype=torch.long, device=device
        )
        self.pending_token_slots: dict[int, int] = {}
        self.batch = Batch(
            max_bs=config.max_num_seqs, max_tokens=config.max_num_batched_tokens,
            max_blocks=(config.max_model_len + config.block_size - 1) // config.block_size,
            vocab_size=model.config.get_text_config(decoder=True).vocab_size, device=device,
            persistent_host_staging=config.enable_persistent_batch,
        )
        self.max_model_len = config.max_model_len
        self.enable_cuda_graph = (
            config.enable_cuda_graph and config.enable_paged_attention and not config.enforce_eager
        )
        self.decode_graph: torch.cuda.CUDAGraph | None = None
        self.decode_graph_cache: object | None = None
        self.decode_graph_logits: torch.Tensor | None = None
        self.decode_graph_ids: torch.Tensor | None = None
        self.decode_graph_samples_greedy = False
        self.enable_graph_greedy_sampling = config.enable_graph_greedy_sampling
        self.reusable_hybrid_cache: Qwen35HybridCache | None = None
        # token、页表和 cache_seqlens 已由 Batch 常驻；位置也必须保持固定地址。
        self.decode_position_ids = torch.zeros((4, 1, 1), dtype=torch.long, device=device)
        self.decode_position_seq_id: int | None = None
        # 单请求 decode 没有 padding；显式映射可绕过 HF 每步读取 Python cache 长度造 mask。
        self.decode_attention_masks = {"full_attention": None, "linear_attention": None}
        self.kv = kv
        self.use_paged_attention = config.enable_paged_attention
        self.enable_prefix_caching = config.enable_prefix_caching
        self.enable_system_prefix_caching = config.enable_system_prefix_caching
        self.profiler = CudaStageProfiler(
            enabled=os.getenv("DNDX_V3_PROFILE_STAGES", "0") == "1"
        )
        self.prefill_language_model = self.model.model.language_model
        if config.torch_compile_prefill:
            self.prefill_language_model = torch.compile(
                self.model.model.language_model,
                mode=config.compile_mode,
                dynamic=True,
                fullgraph=False,
            )

    def _sample_gpu(self, logits: list[torch.Tensor], seqs: list[Sequence]) -> torch.Tensor:
        """在 GPU 上为每条序列采样一个 token。"""
        if not logits:
            return torch.empty(0, dtype=torch.long, device=self.device)
        merged = torch.cat(logits, dim=0)
        all_greedy = all(s.temperature <= 1e-5 for s in seqs)
        use_top_p = any((s.top_p or 1.0) < 1.0 for s in seqs)
        if all_greedy:
            ids = merged.argmax(dim=-1)
        else:
            temperatures = self.batch.temperatures[:len(seqs)]
            top_ps = self.batch.top_ps[:len(seqs)]
            ids = self.sampler(
                merged, temperatures, top_ps,
                all_greedy=False, use_top_p=use_top_p,
            )
        return ids

    def _publish_sample_ids(self, ids: torch.Tensor, seqs: list[Sequence]) -> list[int]:
        """保留 GPU next-token，并同步生成 scheduler 使用的 CPU mirror。"""
        for index, seq in enumerate(seqs):
            slot = self.pending_token_slots.get(seq.seq_id)
            if slot is None:
                used = set(self.pending_token_slots.values())
                slot = next(
                    (candidate for candidate in range(self.pending_next_tokens_gpu.numel())
                     if candidate not in used),
                    None,
                )
                if slot is None:
                    raise RuntimeError("no GPU pending-token slot available")
                self.pending_token_slots[seq.seq_id] = slot
            # 这是唯一的 GPU-side token staging；下一轮直接读取 pending。
            self.pending_next_tokens_gpu[slot].copy_(ids[index])
        return self._read_sample_ids(ids, len(seqs))

    def clear_pending_tokens(self) -> None:
        """请求结束后释放 seq_id 映射，保留固定 GPU slot 地址。"""
        self.pending_token_slots.clear()

    def _sample(self, logits: list[torch.Tensor], seqs: list[Sequence]) -> list[int]:
        """GPU 采样，同时发布 GPU next-token 和 CPU mirror。"""
        ids = self._sample_gpu(logits, seqs)
        if not seqs:
            return []
        return self._publish_sample_ids(ids, seqs)

    def _read_sample_ids(self, ids: torch.Tensor, count: int) -> list[int]:
        """在独立 CUDA Stream 上回传采样结果，仅等待这一次小数据拷贝。"""
        current = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current)
        with torch.cuda.stream(self.copy_stream):
            self.sample_cpu[:count].copy_(ids, non_blocking=True)
        self.copy_event.record(self.copy_stream)
        self.copy_event.synchronize()
        sampled = self.sample_cpu[:count].tolist()
        if os.getenv("DNDX_V3_DEBUG_TOKENS") == "1":
            print(f"[v3-token] {sampled}", flush=True)
        return sampled

    @torch.inference_mode()
    def run_prefill(self, seqs: list[Sequence]) -> list[int]:
        """执行单个请求的 prompt 分块；多模态预处理只计算一次。"""
        if not seqs:
            return []
        if len(seqs) != 1:
            raise ValueError("v3 ModelRunner supports exactly one prefill sequence")

        seq = seqs[0]
        stage_start = self._stage_start()

        # 1. 把文本 token 转成 embedding；
        # 2. 如果请求中包含图片，调用视觉编码器；
        # 3. 将图片 embedding 填入 image placeholder；
        # 4. 计算 Qwen3.5 多模态位置编码；
        # 5. 创建请求的 Hybrid Cache；
        # 6. 创建后续 decode 使用的 attention mask buffer。
        if seq.runtime_embeds is None:
            prepare_start = self._stage_start()
            self._prepare_multimodal(seq) # 整个prompt的编码（包括图片），然后存进runtime_embed
            self._stage_end("prefill_prepare_multimodal", prepare_start)
            self._restore_prefix_snapshot(seq) #  如果该请求命中了 prefix cache，需要恢复除了普通 Attention K/V 之外的状态

        start = seq.num_cached_tokens
        end = start + seq.num_scheduled_tokens # 本次prefill的范围
        self.batch.build_prefill([seq]) # 构建本轮 Paged Attention 元数据，例如token ID、位置编码、token对应的KV Slot
        if self.attention_context is not None:
            self.attention_context.bind(seq, start, end)
            self.attention_context.bind_batch(
                slot_mapping=self.batch.slot_mapping[:seq.num_scheduled_tokens],
                block_table=self.batch.block_tables[:1],
                cache_seqlens=self.batch.cache_seqlens[:1],
            )
        model_start = self._stage_start()
        out = self.prefill_language_model( # 执行核心计算
            inputs_embeds=seq.runtime_embeds[:, start:end], # 只传本轮调度的 prompt chunk
            position_ids=seq.runtime_position_ids[:, :, start:end],
            attention_mask=seq.runtime_attention_mask[:, :end], # attention_mask 描述的是“当前请求到目前为止，哪些 key/value 位置是有效的”
            past_key_values=seq.runtime_cache, use_cache=True, return_dict=True,
        )
        self._stage_end("prefill_language_model", model_start)
        reusable_boundary = (
            seq.num_prompt_tokens // seq.block_size
        ) * seq.block_size
        if (
            self.enable_prefix_caching
            and seq.prefix_cache_admitted
            and end % seq.block_size == 0
            and end <= reusable_boundary
        ): # 保存hybrid快照
            block_id = seq.block_table[end // seq.block_size - 1]
            block = self.kv.blocks.blocks[block_id]
            block.hybrid_snapshot = copy.deepcopy(seq.runtime_cache)
            if seq.runtime_rope_deltas is not None:
                block.rope_delta_snapshot = seq.runtime_rope_deltas.detach().clone()
        if ( #  保存不足一页的系统前缀
            self.enable_system_prefix_caching
            and seq.prefix_cache_admitted
            and end == seq.reusable_prefix_tokens
            and end < seq.block_size
        ):
            self.kv.publish_partial_prefix(seq, copy.deepcopy(seq.runtime_cache))
        sample_start = self._stage_start()
        if seq.need_sample: #  判断本轮是否应该采样，表示prompt处理完毕
            logits = self.model.lm_head(out.last_hidden_state[:, -1:, :])[:, -1, :]
        else: 
            logits = torch.zeros(
                (1, self.model.config.text_config.vocab_size),
                device=self.device,
                dtype=out.last_hidden_state.dtype,
            )

        sampled = self._sample([logits], [seq]) # GPU 采样并发布结果
        self._stage_end("prefill_lm_head_and_sample", sample_start)
        self._stage_end("prefill_total", stage_start)
        return sampled

    def _restore_prefix_snapshot(self, seq: Sequence) -> None:
        """恢复命中边界的 DeltaNet/conv 和 M-RoPE 请求级状态。"""
        if not seq.num_cached_tokens: # 如果没有缓存的状态
            return
        if seq.runtime_prefix_snapshot is not None: # 对应不足一个完整物理页的系统前缀
            seq.runtime_cache.restore_from(seq.runtime_prefix_snapshot) # 恢复缓存
            return
        block_index = seq.num_cached_tokens // seq.block_size - 1 # 定位缓存边界对应的最后一个 block
        block_id = seq.block_table[block_index]
        block = self.kv.blocks.blocks[block_id] # 拿到KV Cache
        snapshot = block.hybrid_snapshot # 拿到hybrid_snapshot
        if snapshot is None:
            raise RuntimeError("prefix page has no HybridCache snapshot")
        # CUDA Graph 捕获依赖 reusable_hybrid_cache 内部 Tensor 地址稳定。
        # Prefix hit 时原地恢复状态；仅兼容非分页 DynamicCache 路径时
        # 才创建独立副本。
        if hasattr(seq.runtime_cache, "restore_from"): # 优先原地恢复，可以做到CUDA Graph复用
            seq.runtime_cache.restore_from(snapshot)
        else:
            seq.runtime_cache = copy.deepcopy(snapshot)
        if block.rope_delta_snapshot is not None:
            seq.runtime_rope_deltas = block.rope_delta_snapshot.detach().clone()
            self.model.model.rope_deltas = seq.runtime_rope_deltas

    @torch.inference_mode()
    def run_decode(self, seqs: list[Sequence]) -> list[int]:
        """为本轮唯一序列执行一次自回归 decode。"""
        if not seqs:
            return []
        if len(seqs) != 1:
            raise ValueError("v3 ModelRunner supports exactly one decode sequence")

        seq = seqs[0]
        stage_start = self._stage_start()

        # prefill/decode 上一步发布的 token 保持在 GPU。读取但不删除，（取得上一轮生成的token）
        # 以保证 CUDA Graph 跨轮次使用同一个 input 地址
        # pending_next_tokens_gpu 是一个常驻 GPU 的 token 缓冲区。
        slot = self.pending_token_slots.get(seq.seq_id)
        # pending_token是上一轮生成的token
        pending_token = (
            self.pending_next_tokens_gpu[slot:slot + 1]
            if slot is not None else None
        )
        # 准备本轮模型推理所需的数据，包括KV Cache对应的block、slot mapping、attention相关数据
        self.batch.build_decode([seq], token_ids_gpu=pending_token)
        length = seq.num_tokens
        seq.runtime_attention_mask = seq.runtime_mask_buffer[:, :length] # 避免每轮 decode 都创建新的完整 mask，所以用缓冲区
        # Graph 在 replay 尾部把位置原地推进 1；仅新请求首步重置。
        # Eager 路径没有 Graph 内更新，因此仍逐步设置当前位置。
        if not self.enable_cuda_graph or self.decode_position_seq_id != seq.seq_id:
            self._reset_decode_position_ids(seq)
            self.decode_position_seq_id = seq.seq_id
        # 要推理1个token
        token = (
            pending_token.view(1, 1)
            if pending_token is not None
            else self.batch.input_ids[:1].view(1, 1)
        )
        if self.attention_context is not None:
            self.attention_context.bind(seq, seq.num_tokens - 1, seq.num_tokens)
            self.attention_context.bind_batch(
                slot_mapping=self.batch.slot_mapping[:1],
                block_table=self.batch.block_tables[:1],
                cache_seqlens=self.batch.cache_seqlens[:1],
            )
        if self.enable_cuda_graph:
            graph_result, is_sampled = self._run_graph_decode(seq, token)
            if is_sampled:
                # Greedy Graph 已把 argmax 写回固定 token 输入；这里只做
                # scheduler/streamer 必需的单标量 D2H 回读。
                sampled = (
                    self._read_sample_ids(graph_result, 1)
                    if slot is not None else self._publish_sample_ids(graph_result, [seq])
                )
            else:
                sampled = self._sample([graph_result], [seq])
        else:
            out = self.model(
                input_ids=token,
                attention_mask=self.decode_attention_masks,
                position_ids=self.decode_position_ids,
                past_key_values=seq.runtime_cache,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
            seq.runtime_cache = out.past_key_values # 保存KV
            sampled = self._sample([out.logits[:, -1, :]], [seq])
        self._stage_end("decode_total", stage_start)
        return sampled

    def _reset_decode_position_ids(self, seq: Sequence) -> None:
        """将固定 Graph 输入重置为当前请求的四路 MRoPE 位置。"""
        self.decode_position_ids.fill_(seq.num_tokens - 1)
        delta = seq.runtime_rope_deltas
        if delta is not None:
            self.decode_position_ids[1:].add_(delta.to(self.device))

    def _run_graph_decode(self, seq: Sequence, token: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """捕获或重放 batch size 为 1 的 decode CUDA Graph。

        Qwen3.5 除分页注意力 KV 外还有 DeltaNet 循环状态。捕获动作本身会
        执行一次模型，因此首次真实 replay 前必须原地恢复缓存快照。保持缓存
        张量地址稳定后，后续请求才能安全复用同一个 Graph。
        """

        #  第一次decode_graph
        #  如果还没有 Graph，就要经历：warmup → 同步 → 保存缓存快照 → capture → 恢复缓存 → replay
        if self.decode_graph is None:
            stage_start = self._stage_start()
            # 新请求拥有不同的 DeltaNet 状态地址，必须重新捕获。
            warm_cache = copy.deepcopy(seq.runtime_cache)
            self.model(
                input_ids=token, attention_mask=self.decode_attention_masks,
                position_ids=self.decode_position_ids,
                past_key_values=warm_cache, use_cache=True,
                logits_to_keep=1, return_dict=True,
            )
            torch.cuda.synchronize(self.device)
            # 保存 capture 前的正确状态，用它恢复缓存
            capture_snapshot = copy.deepcopy(seq.runtime_cache)
            capture_token = token.clone()
            self.decode_graph = torch.cuda.CUDAGraph()
            self.decode_graph_cache = seq.runtime_cache
            # 🌟捕获一次完整的 GPU decode 计算，并把它保存到 self.decode_graph 中，供后续通过 replay() 重复执行
            with torch.cuda.graph(self.decode_graph):
                out = self.model(
                    input_ids=token, attention_mask=self.decode_attention_masks,
                    position_ids=self.decode_position_ids,
                    past_key_values=seq.runtime_cache, use_cache=True,
                    logits_to_keep=1, return_dict=True,
                )
                self.decode_graph_logits = out.logits[:, -1, :] # 形状是[1,vocab_size], 每个数表示对应词表 token 的未归一化分数
                # 判断是否在 Graph 内做 greedy sampling
                self.decode_graph_samples_greedy = (
                    self.enable_graph_greedy_sampling and seq.temperature <= 1e-5
                )
                if self.decode_graph_samples_greedy:
                    self.decode_graph_ids = self.decode_graph_logits.argmax(dim=-1) # 如果贪心采样
                    token.copy_(self.decode_graph_ids.view_as(token))
                # 下一次 replay 使用相同地址读取下一位置。
                self.decode_position_ids.add_(1)
            seq.runtime_cache = out.past_key_values
            # capture 已经执行过一次；恢复固定地址状态，再以 replay 产生本步真实结果。
            seq.runtime_cache.restore_from(capture_snapshot)
            # capture 本身执行了位置推进；首次真实 replay 前恢复当前步位置。
            self._reset_decode_position_ids(seq)
            token.copy_(capture_token)
            self.decode_graph.replay() # 开始replay
            self._stage_end("cuda_graph_capture_and_first_replay", stage_start)
        else: # 🌟已经存储了CUDA Graph
            stage_start = self._stage_start()
            assert self.decode_graph is not None and self.decode_graph_logits is not None
            self.decode_graph.replay()
            self._stage_end("cuda_graph_replay", stage_start)
        assert self.decode_graph_logits is not None
        if self.decode_graph_samples_greedy:
            assert self.decode_graph_ids is not None
            return self.decode_graph_ids, True
        return self.decode_graph_logits, False

    def run(self, batch: ScheduledBatch) -> tuple[list[int], list[int]]:
        """执行调度批次中的 prefill/decode，并在结束后解除元数据绑定。"""
        try:
            return self.run_prefill(batch.prefills), self.run_decode(batch.decodes)
        finally:
            if self.attention_context is not None:
                self.attention_context.clear()
    def _prepare_multimodal(self, seq: Sequence) -> None:
        # 把 processor 产生的文本、图片等原始输入，转换成语言模型能够直接使用的 embedding、四通道位置编码、attention mask 和缓存对象。
        # 保存在seq.runtime中
        """仅编码一次图像，并保留视觉 embedding 与四通道位置编码。"""
        inputs = seq.runtime_inputs or {} # 原始的模型输入，尚未经过 ModelRunner 的多模态组合
        base = self.model.model
        ids = inputs["input_ids"] # 来自于输入
        embeds = base.get_input_embeddings()(ids)
        if inputs.get("pixel_values") is not None: # 图片
            #  pixel_values：经过图像 processor 处理后的图片； image_grid_thw：图片视觉网格的时间、高度、宽度信息。
            #  得到所有图片的视觉 token embedding
            vision = base.get_image_features(
                inputs["pixel_values"], inputs.get("image_grid_thw"), return_dict=True
            ).pooler_output
            # 合并图片特征并统一设备、精度；如果有多个图片，就拼接（dim=0），能不能
            image_embeds = torch.cat(vision, dim=0).to(embeds.device, embeds.dtype)
            # 多模态 prompt 中，图片通常会先表现为一组特殊占位 token：“请描述这张图片：<image><image><image>...”
            #  get_placeholder_mask() 根据 input_ids 找到这些图片 token 对应的位置。
            # 
            #   input_ids: [普通, 普通, 图片, 图片, 图片, 普通]
            #   image_mask: [False, False, True, True, True, False]
            image_mask, _ = base.get_placeholder_mask(
                ids, inputs_embeds=embeds, image_features=image_embeds
            )
            # 用图片特征替换占位 embedding
            embeds = embeds.masked_scatter(image_mask, image_embeds) # 得到最终的emb
        # 计算位置编码，其中图片有三个维度（thw），用于标识上下/左右关系
        positions = base.compute_3d_position_ids(
            input_ids=ids, inputs_embeds=embeds,
            image_grid_thw=inputs.get("image_grid_thw"), # 整个图片的维度
            video_grid_thw=inputs.get("video_grid_thw"),
            attention_mask=inputs.get("attention_mask"),
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
        )

        # base.compute_3d_position_ids除了返回 positions，还可能把计算出的 RoPE 偏移保存到：base.rope_deltas
        # rope_deltas是多模态 prompt 的位置偏移量，多模态的pos不是简单的0 1 2 3 4 5（具体看overleaf）
        model_rope_deltas = getattr(base, "rope_deltas", None)
        seq.runtime_rope_deltas = (
            model_rope_deltas.detach().clone() if model_rope_deltas is not None else None
        )
        # language_model 需要 text + 三路视觉位置，共四个 channel。
        # 创建文本位置通道
        # 先生成：[0, 1, 2, ..., sequence_length - 1]。原始形状为：[sequence_length]。经过.view(1, 1, -1)，变成：[1, 1, sequence_length]
        # 两个前导维度分别对应：位置通道数 = 1；batch size = 1
        # ids.shape[1]代表token长度，torch.arange生成1...n，得到纯文本位置编码，即四个通道中的通道1
        text_pos = torch.arange(ids.shape[1], device=ids.device).view(1, 1, -1)
        seq.runtime_position_ids = torch.cat([text_pos, positions], dim=0) # 通道1和234拼接
        seq.runtime_embeds = embeds
        if self.use_paged_attention:
            if self.reusable_hybrid_cache is None:
                self.reusable_hybrid_cache = Qwen35HybridCache(self.model.config)
            else:
                self.reusable_hybrid_cache.reset_in_place()
            seq.runtime_cache = self.reusable_hybrid_cache
        else:
            from transformers.cache_utils import DynamicCache
            seq.runtime_cache = DynamicCache(config=self.model.config.text_config)
        seq.runtime_attention_mask = inputs.get("attention_mask")
        seq.runtime_mask_buffer = torch.ones(
            (1, self.max_model_len), dtype=seq.runtime_attention_mask.dtype, device=self.device
        )

    def _stage_start(self):
        return self.profiler.start()

    def _stage_end(self, name: str, start) -> None:
        self.profiler.end(name, start)

    def reset_stage_metrics(self) -> None:
        self.profiler.reset()

    def stage_metrics(self) -> dict[str, dict[str, float | int]]:
        # Engine 在读取快照前已统一同步设备，此处不会额外阻塞热路径。
        return self.profiler.snapshot()

    def destroy(self) -> None:
        self.decode_graph = None
        self.decode_graph_cache = None
        self.decode_graph_logits = None
        self.decode_graph_ids = None
        self.reusable_hybrid_cache = None
