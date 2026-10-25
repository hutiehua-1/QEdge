"""Qwen3.5 快速加载：跳过必然被 safetensors 覆盖的随机初始化。"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
from torch import nn
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForImageTextToText

from .qwen35_decoder import install_qwen35_decoder
from ..layers.qwen35_ops import replace_qwen35_norms


@contextmanager
def skip_parameter_initialization():
    """模型构造期间关闭 Linear/Embedding 的 reset_parameters。"""
    linear, embedding = nn.Linear.reset_parameters, nn.Embedding.reset_parameters
    nn.Linear.reset_parameters = lambda self: None
    nn.Embedding.reset_parameters = lambda self: None
    try:
        yield
    finally:
        nn.Linear.reset_parameters, nn.Embedding.reset_parameters = linear, embedding


def load_qwen35_fast(model_path: str, *, dtype: torch.dtype, device: torch.device):
    """直接按目标 dtype 加载本地分片，避免 fp32 中间模型和随机初始化。"""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with skip_parameter_initialization():
            model = AutoModelForImageTextToText.from_pretrained(
                model_path, local_files_only=True, trust_remote_code=True,
                dtype=dtype, low_cpu_mem_usage=True,
            )
    finally:
        torch.set_default_dtype(previous)
    return model.to(device).eval()


def _weight_files(model_path: str) -> list[Path]:
    files = sorted(Path(model_path).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors weights found in {model_path}")
    return files


def _iter_weight_shards(model_path: str):
    """后台预取下一分片，使磁盘读取与当前分片 H2D copy 重叠。"""
    files = _weight_files(model_path)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="v3-weight-prefetch") as pool:
        future = pool.submit(load_file, str(files[0]), "cpu")
        for index in range(len(files)):
            tensors = future.result()
            if index + 1 < len(files):
                future = pool.submit(load_file, str(files[index + 1]), "cpu")
            yield tensors


# 拷贝参数，其中gate/up特殊处理
# name是checkpoint中的真实参数
def _copy_weight(parameters: dict[str, nn.Parameter], name: str,
                 value: torch.Tensor, device: torch.device) -> bool:
    """写普通参数，或将 HF gate/up 权重直接写入融合参数的对应半区。"""
    # target是算子融合后的模型
    target_name = name
    target_slice: slice | None = None
    # 如果是gate/up
    if ".mlp.gate_proj.weight" in name or ".mlp.up_proj.weight" in name:
        source_is_gate = ".gate_proj." in name
        source = "gate_proj.weight" if source_is_gate else "up_proj.weight"
        target_name = name.replace(source, "gate_up_proj.weight")
        # target是算子融合后模型参数的值
        target = parameters.get(target_name)
        if target is None:
            return False
        # 把gate和up的参数写到一起，Gate 写前半区，Up 写后半区
        width = target.shape[0] // 2
        # target_slice代表这次要写的目标parameter的范围
        target_slice = slice(0, width) if source_is_gate else slice(width, 2 * width)
    target = parameters.get(target_name)
    if target is None:
        return False
    destination = target if target_slice is None else target[target_slice]
    # 写入参数
    destination.copy_(value.to(device=device, dtype=destination.dtype), non_blocking=True)
    return True


def load_qwen35_direct(model_path: str, *, dtype: torch.dtype, device: torch.device):
    """先构造最终 Decoder，再把 safetensors 直接流式写入目标设备参数。"""
    config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with skip_parameter_initialization(), torch.device(device): # 代表不要初始化
            model = AutoModelForImageTextToText.from_config(config, trust_remote_code=True)
        # 替换算子融合后的组件
        replace_qwen35_norms(model)
        install_qwen35_decoder(model)
        # 获取算子替换后的模型parameters
        parameters = dict(model.named_parameters())
        loaded: set[str] = set()
        unmatched: list[str] = []
        with torch.no_grad():
            for shard in _iter_weight_shards(model_path):
                for name, value in shard.items():
                    # 将checkpoint的参数copy进模型
                    if _copy_weight(parameters, name, value, device):
                        loaded.add(name)
                    else:
                        unmatched.append(name)
        torch.cuda.synchronize(device)
    finally:
        torch.set_default_dtype(previous)
    unexpected = [name for name in unmatched if not name.startswith("mtp.")]
    if unexpected:
        raise RuntimeError(f"direct loader left unmatched inference weights: {unexpected[:8]}")
    model._v3_fused_decoder = True
    model._v3_loaded_weight_count = len(loaded)
    model._v3_unmatched_weights = unmatched
    return model.eval()
