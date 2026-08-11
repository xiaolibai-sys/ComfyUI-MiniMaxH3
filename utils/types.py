"""Typed data-transfer payloads for the MiniMax H3 streaming runner.

Centralises the structured data flowing between the custom nodes and the
self-contained runtime (``minimax_h3`` package).  Style follows
``ComfyUI-BerniniRWrapper/utils/types.py``: frozen dataclasses for immutable
payloads, enums for closed option sets, small property helpers so consumers
never repeat dtype/device arithmetic.

ComfyUI native containers (LATENT, CONDITIONING, IMAGE, AUDIO) keep their
external shape; only the custom ``MINIMAX_H3_*`` sockets are represented as
typed dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import torch


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class LoadingMode(str, Enum):
    """How DiT block weights are held during a sampling run."""
    STREAMING = "Streaming"   # load block-group -> compute -> release, group by group
    FULL = "Full"             # load every block to VRAM once and keep resident


class SwapMode(str, Enum):
    """BlockSwap residency strategy for the DiT blocks."""
    RING_BUFFER = "ring_buffer"   # fixed GPU ring pool + pinned CPU home pool + disk prefetch
    NONE = "none"                 # no swap (Full loading)


class CondKind(str, Enum):
    """Modality tag of a packed-sequence segment (matches the DiT layout)."""
    TEXT = "text"
    VIDEO = "video"
    AUDIO = "audio"
    COND = "cond"          # fl2va keyframe rows
    REF_IMG = "ref_img"    # ref2va image/video reference rows
    REF_AUDIO = "ref_audio"


@dataclass(frozen=True)
class AdaLNCacheKey:
    """Exact signature for one AdaLN modulation entry."""
    sigma: float
    unique_timesteps: tuple[float, ...]
    has_visual_cond: bool
    has_audio_cond: bool
    shift_video: float = 12.0
    shift_audio: float = 3.0


@dataclass(frozen=True)
class AdaLNCacheEntry:
    """Baked AdaLN outputs for one exact sigma/timestep signature."""
    block_mods: tuple[tuple[torch.Tensor, ...], ...]
    final_mods: tuple[torch.Tensor, ...]


@dataclass
class AdaLNCache:
    """High-cohesion AdaLN cache owned by the sampling/refiner layer."""
    entries: dict[AdaLNCacheKey, AdaLNCacheEntry] = field(default_factory=dict)

    def add(self, key: AdaLNCacheKey, entry: AdaLNCacheEntry) -> None:
        self.entries[key] = entry


# ---------------------------------------------------------------------------
# BlockSwap / streaming configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class H3BlockSwap:
    """Ring-buffer VRAM<->RAM<->disk block-swap knobs (mirrors BerniniBlockSwap).

    ``block_to_swap`` is the number of DiT blocks kept OFF the GPU; the
    resident window is ``total_blocks - block_to_swap`` (per block ≈ 1.2 GB
    BF16 / ≈ 0.6 GB int8).  The off-GPU blocks live in a fixed pinned CPU
    "home" pool, so host RAM stays flat for the whole run.
    """
    enabled: bool = True
    block_to_swap: int = 47             # DiT blocks off GPU (50 - 47 = 3 resident)
    hot_blocks: int = 0                 # leading DiT blocks kept resident on GPU
    prefetch: bool = True               # disk -> RAM prefetch of the next window
    prefetch_count: int = 2             # home slots read ahead from disk
    pin_memory: bool = True             # pinned CPU staging; only the small prefetch/stage pools are pinned (home stays pageable), so the locked footprint is tiny and Windows-safe
    disk_workers: int = 2               # background reader threads
    auto_vram: bool = True              # estimate reserve/runtime LoRA before pool allocation
    vram_reserve_mb: float = 0.0        # non-weight VRAM reserved before block pool allocation
    runtime_lora_total_mb: float = 0.0  # runtime LoRA A/B bytes shared across all 50 blocks
    runtime_lora_fixed_mb: float = 0.0  # AdaLN/final/token-refiner runtime LoRA fixed bytes
    loading_mode: str = LoadingMode.STREAMING.value
    dtype: str = "bfloat16"             # bfloat16 | float16 | float32

    def window_size(self, total_blocks: int) -> int:
        """DiT blocks resident on GPU at once (>= 1, <= total)."""
        return max(1, min(total_blocks - self.block_to_swap, total_blocks))

    @property
    def torch_dtype(self) -> torch.dtype:
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.dtype]


@dataclass(frozen=True)
class H3TeaCache:
    """TeaCache knobs (mirrors BerniniTeaCache): skip near-identical block runs."""
    start_block: int = 3
    max_skip_blocks: int = 15
    rel_l1_thresh: float = 0.08
    warmup_steps: int = 1
    cooldown_steps: int = 2


@dataclass(frozen=True)
class EncoderStreamConfig:
    """Streaming knobs forwarded to the VENDORED encoder
    (``models/text_encoder``, self-contained)."""
    group_size: int = 2
    loading_mode: str = LoadingMode.STREAMING.value
    prefetch: bool = True
    prefetch_count: int = 1
    disk_workers: int = 2
    pin_memory: bool = True
    dtype: str = "float32"
    full_precision_mm: bool = True
    weight_path: str = ""               # standalone ComfyUI safetensors ('' = model_dir)

    def to_encoder_config(self) -> Any:
        """Build the vendored encoder's ``StreamConfig`` (lazy import)."""
        from ..models.text_encoder.types import StreamConfig
        return StreamConfig(
            group_size=self.group_size,
            loading_mode=self.loading_mode,
            prefetch=self.prefetch,
            prefetch_count=self.prefetch_count,
            disk_workers=self.disk_workers,
            pin_memory=self.pin_memory,
            device="cuda",
            dtype=self.dtype,
            full_precision_mm=self.full_precision_mm,
            weight_path=self.weight_path or None,
        )


# ---------------------------------------------------------------------------
# Attention backend config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AttentionConfig:
    """Attention backend selection (mirrors BerniniR_AttentionConfig)."""
    backend: str = "sageattn2"
    force_backend: bool = False
    available: tuple = ()
    best: str = "sdpa_math"

    def make_override(self):
        from ..attention.backends import create_attention_override
        return create_attention_override(self.backend, force_backend=self.force_backend)


# ---------------------------------------------------------------------------
# Slot entry: unified ring-buffer payload (plain or quantized)
# ---------------------------------------------------------------------------

def _params_extra_fields(params) -> tuple[dict, dict]:
    """Split a comfy_kitchen layout ``Params`` dataclass into tensor extras
    and meta extras (mirrors BerniniRWrapper).  ``scale``/``orig_dtype``/
    ``orig_shape`` are first-class SlotEntry fields; anything else the layout
    carries (nvfp4 ``block_scale``, ``transposed``, ``convrot`` ...) is
    captured here so slot copies stay layout-agnostic."""
    import dataclasses
    tensor_extras: dict = {}
    meta_extras: dict = {}
    if not dataclasses.is_dataclass(params):
        return tensor_extras, meta_extras
    for f in dataclasses.fields(params):
        if f.name in ("scale", "orig_dtype", "orig_shape"):
            continue
        v = getattr(params, f.name)
        if isinstance(v, torch.Tensor):
            tensor_extras[f.name] = v
        else:
            meta_extras[f.name] = v
    return tensor_extras, meta_extras


class SlotEntry:
    """One slot buffer: either a plain tensor or the components of a
    comfy_kitchen ``QuantizedTensor`` (``_qdata`` + ``_params``).  The swap
    engine never branches on parameter type at the pool level."""

    __slots__ = ("is_qt", "data", "scale", "layout_cls", "orig_dtype",
                 "orig_shape", "lora", "extra", "meta")

    def __init__(self, *, data, scale=None, layout_cls="", orig_dtype=None,
                 orig_shape=None, lora=None, extra=None, meta=None):
        self.is_qt = scale is not None
        self.data = data
        self.scale = scale
        self.layout_cls = layout_cls
        self.orig_dtype = orig_dtype
        self.orig_shape = orig_shape
        self.lora = lora
        self.extra = extra or {}
        self.meta = meta or {}

    @classmethod
    def from_qt(cls, qt) -> "SlotEntry":
        t_extras, m_extras = _params_extra_fields(qt._params)
        return cls(
            data=qt._qdata, scale=qt._params.scale, layout_cls=qt._layout_cls,
            orig_dtype=qt._params.orig_dtype, orig_shape=tuple(qt.shape),
            extra=t_extras, meta=m_extras)

    @classmethod
    def empty_like_entry(cls, entry: "SlotEntry", device,
                         pin_memory: bool = False) -> "SlotEntry":
        if entry.is_qt:
            pin = device == "cpu" and pin_memory
            return cls(
                data=torch.empty_like(entry.data, device=device, pin_memory=pin),
                scale=torch.empty_like(entry.scale, device=device, pin_memory=pin),
                layout_cls=entry.layout_cls,
                orig_dtype=entry.orig_dtype,
                orig_shape=entry.orig_shape,
                extra={n: torch.empty_like(t, device=device, pin_memory=pin)
                       for n, t in entry.extra.items()},
                meta=dict(entry.meta))
        return cls(data=torch.empty_like(entry.data, device=device,
                                         pin_memory=(device == "cpu" and pin_memory)))

    def copy_from(self, src: "SlotEntry", non_blocking: bool = False) -> None:
        if self.is_qt:
            self.data.copy_(src.data, non_blocking=non_blocking)
            self.scale.copy_(src.scale, non_blocking=non_blocking)
            for n, t in src.extra.items():
                if n in self.extra:
                    self.extra[n].copy_(t, non_blocking=non_blocking)
        else:
            self.data.copy_(src.data, non_blocking=non_blocking)

    def to_quantized_tensor(self):
        """Rebuild a fresh QuantizedTensor from this slot's components."""
        from comfy_kitchen.tensor import QuantizedTensor, get_layout_class
        layout_cls = get_layout_class(self.layout_cls)
        params = layout_cls.Params(scale=self.scale, orig_dtype=self.orig_dtype,
                                   orig_shape=self.orig_shape,
                                   **self.meta, **self.extra)
        return QuantizedTensor(self.data, self.layout_cls, params)

    def assign_to(self, module, leaf: str) -> None:
        """Wire this slot's data into ``module._parameters[leaf]`` (meta-safe,
        QuantizedTensor-safe: replaces the whole Parameter, never ``.data``)."""
        if self.is_qt:
            module._parameters[leaf] = torch.nn.Parameter(
                self.to_quantized_tensor(), requires_grad=False)
        else:
            module._parameters[leaf] = torch.nn.Parameter(
                self.data, requires_grad=False)



# ---------------------------------------------------------------------------
# LoRA payloads (fold-on-load, reserved in the block-swap buffers)
# ---------------------------------------------------------------------------

@dataclass
class LoraEntry:
    """One LoRA/DoRA delta targeting a block parameter (folded on load).

    Mirrors ``BerniniRWrapper``'s per-slot LoRA payloads: entries are attached
    to a block via ``BlockSwapManager.apply_lora`` and folded into the CPU
    slot once, when the block is next materialised from disk; the payload is
    then consumed so re-loads skip folding automatically.
    """
    target: str                     # param leaf, e.g. "attn.qkv_proj" / "mlp.fc1"
    a: Optional[torch.Tensor] = None        # LoRA A  [r, in]
    b: Optional[torch.Tensor] = None        # LoRA B  [out, r]
    alpha: float = 1.0
    strength: float = 1.0
    diff: Optional[torch.Tensor] = None     # DoRA direction diff (per row)
    diff_b: Optional[torch.Tensor] = None   # DoRA magnitude diff (per row)


@dataclass
class AdaLNOverride:
    """Full AdaLN table/projection replacement for pruned model LoRAs."""
    table: torch.Tensor
    block_weights: dict[int, torch.Tensor] = field(default_factory=dict)
    block_biases: dict[int, torch.Tensor] = field(default_factory=dict)
    final_weight: Optional[torch.Tensor] = None
    final_bias: Optional[torch.Tensor] = None


@dataclass
class H3Lora:
    """Parsed LoRA payload attached to a MiniMax H3 model handle."""
    path: str
    strength: float = 1.0
    block_groups: dict[int, list[LoraEntry]] = field(default_factory=dict)
    token_refiner_groups: dict[int, list[LoraEntry]] = field(default_factory=dict)
    final_adaln: Optional[LoraEntry] = None
    silu_grid_path: str = ""
    adaln_override: Optional[AdaLNOverride] = None


@dataclass
class H3LoraSet:
    """Multi-LoRA payload attached to a MiniMax H3 model handle."""
    loras: list[H3Lora] = field(default_factory=list)

    def add(self, lora: H3Lora) -> None:
        if self.loras and lora.silu_grid_path:
            current = self.silu_grid_path
            if current and current != lora.silu_grid_path:
                raise ValueError(
                    "MiniMax H3 multi-LoRA: all runtime AdaLN LoRAs must use "
                    "the same silu(t_emb) grid; got "
                    f"{current} and {lora.silu_grid_path}."
                )
        self.loras.append(lora)

    def __bool__(self) -> bool:
        return bool(self.loras)

    @property
    def block_groups(self) -> dict[int, list[LoraEntry]]:
        merged: dict[int, list[LoraEntry]] = {}
        for lora in self.loras:
            for idx, entries in lora.block_groups.items():
                merged.setdefault(idx, []).extend(entries)
        return merged

    @property
    def token_refiner_groups(self) -> dict[int, list[LoraEntry]]:
        merged: dict[int, list[LoraEntry]] = {}
        for lora in self.loras:
            for idx, entries in lora.token_refiner_groups.items():
                merged.setdefault(idx, []).extend(entries)
        return merged

    @property
    def final_adaln_entries(self) -> list[LoraEntry]:
        return [l.final_adaln for l in self.loras if l.final_adaln is not None]

    @property
    def silu_grid_path(self) -> str:
        for lora in self.loras:
            if lora.silu_grid_path:
                return lora.silu_grid_path
        return ""

    @property
    def has_adaln(self) -> bool:
        if self.adaln_override is not None:
            return True
        if self.final_adaln_entries:
            return True
        return any(
            e.target == "adaln_proj.linear"
            for entries in self.block_groups.values()
            for e in entries
        )

    def signature(self) -> list[dict]:
        return [
            dict(path=l.path, strength=l.strength, silu_grid_path=l.silu_grid_path)
            for l in self.loras
        ]

    @property
    def adaln_override(self) -> Optional[AdaLNOverride]:
        for lora in self.loras:
            if lora.adaln_override is not None:
                return lora.adaln_override
        return None



# ---------------------------------------------------------------------------
# Conditioning payload (consumed by the sampler)
# ---------------------------------------------------------------------------

@dataclass
class H3Conditioning:
    """Everything the DiT forward needs besides the AV latents."""
    text_states: torch.Tensor            # [1, L, text_dim] (unnormalized, layer 50)
    text_token_tags: torch.Tensor        # [1, L] int64, 0=vision-pad(video) 1=text
    keyframes: list = field(default_factory=list)      # fl2va: [{resolved_frame_index, latent}]
    refs: list = field(default_factory=list)           # ref2va: [{kind, latent, ...}]
    frame_count: Optional[int] = None
    visual_cond_noise_aug: float = 0.999
    audio_cond_noise_aug: float = 1.0
    seed: int = 0

    def to_payload(self) -> dict:
        payload = {
            "text_token_tags": self.text_token_tags,
            "visual_cond_noise_aug": self.visual_cond_noise_aug,
            "audio_cond_noise_aug": self.audio_cond_noise_aug,
            "seed": self.seed,
        }
        if self.keyframes:
            payload["keyframes"] = self.keyframes
            payload["frame_count"] = self.frame_count
            payload["cond_video_latents"] = [kf["latent"] for kf in self.keyframes]
        if self.refs:
            payload["refs"] = self.refs
            payload["cond_video_latents"] = [r["latent"] for r in self.refs if "latent" in r]
            payload["cond_audio_latents"] = [r["audio_latent"] for r in self.refs
                                             if r.get("audio_latent") is not None]
        return payload


@dataclass
class AVLatent:
    """Joint video+audio latent (NestedTensor-like pair)."""
    video: torch.Tensor    # [B, 24, T_lat, H/16, W/16]
    audio: torch.Tensor    # [B, 32, 2, T40]

    @property
    def shape(self) -> tuple:
        return self.video.shape, self.audio.shape


# ---------------------------------------------------------------------------
# Sampler output
# ---------------------------------------------------------------------------

@dataclass
class H3SampleResult:
    """Sampled denoised latents + metadata."""
    video: torch.Tensor
    audio: torch.Tensor
    steps: int
    sigmas: torch.Tensor
    swap_hits: int = 0          # blocks reused from the GPU ring
    swap_loads: int = 0         # blocks pulled from RAM/disk
    peak_vram_mb: float = 0.0
    d2h_stage: int = 0
    d2h_direct: int = 0
    d2h_host_register: int = 0
    d2h_sync: int = 0
    prebake_seconds: float = 0.0

    @property
    def av(self) -> AVLatent:
        return AVLatent(video=self.video, audio=self.audio)
