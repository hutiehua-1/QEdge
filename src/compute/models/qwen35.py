from __future__ import annotations

from typing import Any

from ...data import HybridLayerLayout


def validate_qwen35_config(config: Any) -> HybridLayerLayout:
    if getattr(config, "model_type", None) != "qwen3_5":
        raise ValueError("expected a qwen3_5 model config")
    architectures = tuple(getattr(config, "architectures", ()) or ())
    if "Qwen3_5ForConditionalGeneration" not in architectures:
        raise ValueError("expected Qwen3_5ForConditionalGeneration weights")
    return HybridLayerLayout.from_hf_config(config)
