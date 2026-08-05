"""Standalone ``Qwen3VLTextConfig`` ported from ``transformers``.

Kept dependency-free (no ``PreTrainedConfig``) so the encoder package can be
moved/embedded anywhere.  ``from_pretrained`` understands both the flat
``qwen3_vl_text`` layout and the nested ``{"text_config": {...}}`` layout used
inside a full ``Qwen3VLConfig``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class Qwen3VLTextConfig:
    """Architecture hyper-parameters of the Qwen3-VL text (language) model.

    Defaults mirror ``transformers.models.qwen3_vl.configuration_qwen3_vl``.
    The real values are read from the checkpoint's ``config.json`` at load time.
    """
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 22016
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 128000
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    rope_parameters: dict[str, Any] = field(default_factory=lambda: {
        "rope_type": "default",
        "rope_theta": 500000.0,
        "mrope_section": [24, 20, 20],
        "mrope_interleaved": False,
    })
    attention_bias: bool = False
    attention_dropout: float = 0.0
    pad_token_id: Optional[int] = None
    tie_word_embeddings: bool = True
    model_type: str = "qwen3_vl_text"

    # -- construction helpers ------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Qwen3VLTextConfig":
        known = {f for f in cls.__dataclass_fields__}
        kw = {k: v for k, v in d.items() if k in known}
        # transformers stores RoPE knobs under ``rope_scaling``; fold them into
        # our ``rope_parameters`` so real checkpoints (e.g. MiniMax-H3's
        # Qwen3-VL-32B with rope_theta=5e6, interleaved MRoPE) load correctly.
        rs = d.get("rope_scaling")
        if isinstance(rs, dict) and kw.get("rope_parameters") is None:
            rp = dict(kw.get("rope_parameters") or {
                "rope_type": "default", "rope_theta": 500000.0,
                "mrope_section": [24, 20, 20], "mrope_interleaved": False})
            if "rope_theta" in d:
                rp["rope_theta"] = d["rope_theta"]
            for k in ("rope_type", "rope_theta", "mrope_section",
                      "mrope_interleaved"):
                if k in rs:
                    rp[k] = rs[k]
            kw["rope_parameters"] = rp
        return cls(**kw)

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "Qwen3VLTextConfig":
        cfg_path = Path(model_dir) / "config.json"
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        if isinstance(raw.get("text_config"), dict):
            return cls.from_dict(raw["text_config"])
        # Flat qwen3_vl_text config (or a full config with only text keys).
        return cls.from_dict(raw)


@dataclass
class Qwen3VLVisionConfig:
    """Vision-tower hyper-parameters (mirrors transformers Qwen3VLVisionConfig)."""
    depth: int = 27
    hidden_size: int = 1152
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 3584
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: tuple[int, ...] = (8, 16, 24)
    initializer_range: float = 0.02
    model_type: str = "qwen3_vl_vision"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Qwen3VLVisionConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Qwen3VLConfig:
    """Full Qwen3-VL config (text + vision + multimodal token ids)."""
    text_config: Optional[dict] = None
    vision_config: Optional[dict] = None
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    tie_word_embeddings: bool = False
    model_type: str = "qwen3_vl"

    def __post_init__(self):
        self.text = Qwen3VLTextConfig.from_dict(self.text_config or {})
        self.vision = Qwen3VLVisionConfig.from_dict(self.vision_config or {})

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "Qwen3VLConfig":
        raw = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})
