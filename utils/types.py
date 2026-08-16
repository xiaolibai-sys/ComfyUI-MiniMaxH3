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
    pin_memory: bool = True             # pinned CPU staging
    disk_workers: int = 2               # background reader threads
    auto_vram: bool = True              # estimate reserve/runtime LoRA before pool allocation
    vram_reserve_mb: float = 0.0        # non-weight VRAM reserved before block pool allocation
    runtime_lora_total_mb: float = 0.0  # runtime LoRA A/B bytes shared across all 50 blocks
    runtime_lora_fixed_mb: float = 0.0  # AdaLN/final/token-refiner runtime LoRA fixed bytes
    offload_dit: bool = False           # rolling VAE phase: keep a small RAM home pool
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


@dataclass(frozen=True)
class VAERef:
    """Paths only; the pack is loaded lazily (and cached) by decode/sample nodes."""

    video_path: str
    audio_path: str = ""


# ---------------------------------------------------------------------------
# Attention backend config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AttentionConfig:
    """Attention backend selection (mirrors BerniniR_AttentionConfig)."""
    backend: str = "sageattn2"
    force_backend: bool = False

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


@dataclass
class SwapBlock:
    """One swappable weight group (a DiT block)."""

    name: str
    module: torch.nn.Module
    keys: list[str] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    refs: list[tuple[torch.nn.Module, str, str]] = field(default_factory=list)
    templates: list[SlotEntry] = field(default_factory=list)
    lora: Optional[list] = None
    overrides: dict = field(default_factory=dict)

    def bytes_per_block(self) -> int:
        total = 0
        for t in self.templates:
            total += t.data.numel() * t.data.element_size()
            if t.scale is not None:
                total += t.scale.numel() * t.scale.element_size()
            for e in t.extra.values():
                total += e.numel() * e.element_size()
        return total


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

@dataclass(frozen=True)
class TextConditioning:
    """Qwen hidden states plus per-token modality tags."""

    states: torch.Tensor       # [1, L, text_dim]
    tags: torch.Tensor         # [1, L]


@dataclass(frozen=True)
class KeyframeCondition:
    """One FL2VA keyframe latent anchored at a resolved frame index."""

    resolved_frame_index: int
    latent: torch.Tensor


@dataclass(frozen=True)
class ReferenceCondition:
    """One ref2va image/video/audio reference block."""

    kind: str                                  # image | video | video_audio | audio
    latent: Optional[torch.Tensor] = None
    latent_t: int = 0
    latent_h: int = 0
    latent_w: int = 0
    ref_audio_t: int = 0
    audio_latent: Optional[torch.Tensor] = None

    def to_payload(self) -> dict:
        payload = {"kind": self.kind}
        if self.latent_h or self.latent_w:
            payload["latent_h"] = self.latent_h
            payload["latent_w"] = self.latent_w
        if self.latent is not None:
            payload["latent"] = self.latent
        if self.latent_t:
            payload["latent_t"] = self.latent_t
        if self.ref_audio_t:
            payload["ref_audio_t"] = self.ref_audio_t
        if self.audio_latent is not None:
            payload["audio_latent"] = self.audio_latent
        return payload


@dataclass(frozen=True)
class MediaConditioning:
    """Media anchors carried alongside text conditioning."""

    keyframes: tuple[KeyframeCondition, ...] = ()
    refs: tuple[ReferenceCondition, ...] = ()
    frame_count: Optional[int] = None

    def to_payload(self) -> dict:
        payload: dict = {}
        if self.keyframes:
            payload["keyframes"] = [
                {
                    "resolved_frame_index": kf.resolved_frame_index,
                    "latent": kf.latent,
                }
                for kf in self.keyframes
            ]
            payload["frame_count"] = self.frame_count
            payload["cond_video_latents"] = [
                kf.latent for kf in self.keyframes
            ]
        if self.refs:
            payload["refs"] = [r.to_payload() for r in self.refs]
            payload["cond_video_latents"] = [
                r.latent for r in self.refs if r.latent is not None
            ]
            payload["cond_audio_latents"] = [
                r.audio_latent for r in self.refs
                if r.audio_latent is not None
            ]
        return payload


@dataclass
class H3Conditioning:
    """Composed text + media conditioning consumed by the sampler."""

    text: TextConditioning
    media: MediaConditioning = field(default_factory=MediaConditioning)
    segment_texts: tuple[TextConditioning, ...] = ()
    segment_negative_texts: tuple[TextConditioning, ...] = ()
    visual_cond_noise_aug: float = 0.999
    audio_cond_noise_aug: float = 1.0
    seed: int = 0
    fl_constraint: Optional[dict] = None
    av_encoder: Optional[Any] = None

    def to_payload(self) -> dict:
        payload = {
            "text_token_tags": self.text.tags,
            "visual_cond_noise_aug": self.visual_cond_noise_aug,
            "audio_cond_noise_aug": self.audio_cond_noise_aug,
            "seed": self.seed,
        }
        payload.update(self.media.to_payload())
        return payload


@dataclass
class AVLatent:
    """Joint video+audio latent (NestedTensor-like pair)."""
    video: torch.Tensor    # [B, 24, T_lat, H/16, W/16]
    audio: torch.Tensor    # [B, 32, 2, T40]

    @property
    def shape(self) -> tuple:
        return self.video.shape, self.audio.shape


@dataclass(frozen=True)
class SamplerRequest:
    """One k-diffusion sampling run request."""

    latent: AVLatent
    seed: int = 0
    payload: Optional[dict] = None
    negative_payload: Optional[dict] = None
    positive_text: Optional[TextConditioning] = None
    negative_text: Optional[TextConditioning] = None


# ---------------------------------------------------------------------------
# Sampler output
# ---------------------------------------------------------------------------

@dataclass
class H3SampleResult:
    """Sampled denoised latents + metadata."""
    video: torch.Tensor
    audio: torch.Tensor
    steps: int
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


@dataclass(frozen=True)
class BlockSwapStats:
    """Snapshot of BlockSwap counters consumed by sampling runners."""

    swap_hits: int = 0
    swap_loads: int = 0
    home_size: int = 0
    total: int = 0
    window: int = 0
    hot: int = 0
    disk_reads: int = 0
    d2h_stage: int = 0
    d2h_direct: int = 0
    d2h_host_register: int = 0
    d2h_sync: int = 0


# ---------------------------------------------------------------------------
# Rolling FL2VA contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FLKeyframe:
    """One user-defined timeline keyframe."""

    time: float
    image_path: str = ""
    image: Optional[torch.Tensor] = None
    prompt: str = ""
    negative_prompt: str = ""
    note: str = ""


@dataclass(frozen=True)
class FLConstraint:
    """Parsed front-end FL Constraint data."""

    fps: int = 24
    keyframes: tuple[FLKeyframe, ...] = ()
    offload_dit: bool = False
    audio_loudness_match: bool = True
    global_negative_prompt: str = ""

    @classmethod
    def from_json(cls, raw: str) -> "FLConstraint":
        import json

        data = json.loads(raw or "{}")
        fps = int(data.get("fps") or 24)
        offload_dit = bool(data.get("offload_dit") or False)
        audio_loudness_match = bool(data.get("audio_loudness_match", True))
        global_negative_prompt = str(data.get("global_negative_prompt") or "")
        kfs = []
        for item in data.get("keyframes") or []:
            kfs.append(FLKeyframe(
                time=float(item.get("time") or 0.0),
                image_path=str(item.get("image_path") or ""),
                image=item.get("image"),
                prompt=str(item.get("prompt") or ""),
                negative_prompt=str(item.get("negative_prompt") or ""),
                note=str(item.get("note") or ""),
            ))
        return cls(
            fps=fps,
            keyframes=tuple(kfs),
            offload_dit=offload_dit,
            audio_loudness_match=audio_loudness_match,
            global_negative_prompt=global_negative_prompt,
        )


@dataclass(frozen=True)
class RollingSegment:
    """One legal FL2VA segment inside a rolling plan."""

    start_time: float
    end_time: float
    frame_count: int
    latent_t: int
    audio_t: int
    start_image: Optional[torch.Tensor] = None
    end_image: Optional[torch.Tensor] = None
    prompt: str = ""
    negative_prompt: str = ""
    note: str = ""


@dataclass(frozen=True)
class RollingPlan:
    """Fully resolved rolling segments for one sampling run."""

    segments: tuple[RollingSegment, ...]
    width: int
    height: int
    fps: int = 24


# ---------------------------------------------------------------------------
# Sampling / session contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SamplingConfig:
    """Sampler knobs extracted once from node inputs."""

    steps: int
    cfg: float
    seed: int
    sampler_name: str
    scheduler_name: str
    shift_video: float
    shift_audio: float
    use_adaln_cache: bool
    adaln_prebake_batch: int
    width: int
    height: int
    denoise: float = 1.0


@dataclass(frozen=True)
class RuntimeOptions:
    """Runtime injection data; never mutated after construction."""

    swap: Optional[H3BlockSwap] = None
    teacache: Optional[H3TeaCache] = None

    def make_teacache(self, model):
        """Attach TeaCache hooks to the model, or return None when disabled."""
        if self.teacache is None:
            return None
        from .teacache import TeaCache
        return TeaCache(
            model,
            start_block=self.teacache.start_block,
            max_skip_blocks=self.teacache.max_skip_blocks,
            rel_l1_thresh=self.teacache.rel_l1_thresh,
            warmup_steps=self.teacache.warmup_steps,
            cooldown_steps=self.teacache.cooldown_steps,
        )


@dataclass(frozen=True)
class SamplingAssets:
    """Everything a sampling run needs besides sampler knobs."""

    handle: Any
    positive: H3Conditioning
    negative: Optional[H3Conditioning]
    fl_constraint: FLConstraint
    av_encoder: Any
    runtime: RuntimeOptions = field(default_factory=RuntimeOptions)
    vram_spec: Optional[SequenceSpec] = None
    latent: Optional[AVLatent] = None


@dataclass(frozen=True)
class ForwardRequest:
    """One k-diffusion forward wrapper request."""

    model: Any
    cfg: float
    video_shape: tuple
    audio_shape: tuple
    positive_text: TextConditioning
    positive_payload: dict
    negative_text: Optional[TextConditioning] = None
    negative_payload: Optional[dict] = None
    shift_video: float = 12.0
    shift_audio: float = 3.0


@dataclass(frozen=True)
class SessionContext:
    """Runtime view handed to sampler/runner modules."""

    model: Any
    reader: Optional[Any]
    vae: Any
    positive_text: TextConditioning
    negative_text: Optional[TextConditioning]
    sigmas: torch.Tensor
    teacache: Optional[Any]
    adaln_cache: Optional[AdaLNCache]
    device: torch.device
    dtype: torch.dtype
    positive_payload: Optional[dict] = None
    negative_payload: Optional[dict] = None
    block_stats: Optional[BlockSwapStats] = None


@dataclass(frozen=True)
class SegmentRequest:
    """One rolling segment sampling request."""

    segment: RollingSegment
    start_latent: torch.Tensor
    end_latent: Optional[torch.Tensor] = None
    seed: int = 0


@dataclass(frozen=True)
class SegmentResult:
    """Sampled latent plus per-segment stats."""

    latent: AVLatent
    swap_hits: int = 0
    swap_loads: int = 0
    peak_vram_mb: float = 0.0


@dataclass(frozen=True)
class DecodedSegment:
    """Decoded rolling segment and the next segment's start latent."""

    video: torch.Tensor          # CPU [T,H,W,3]
    audio: torch.Tensor          # CPU waveform
    next_start_latent: torch.Tensor


@dataclass(frozen=True)
class RollingOutput:
    """Final rolling sampling result."""

    video: torch.Tensor
    audio: torch.Tensor
    stats: str
    segment_count: int
    peak_vram_mb: float


# ---------------------------------------------------------------------------
# VRAM / BlockSwap planning contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SequenceSpec:
    """Typed sequence used by VRAM estimators."""

    text_len: int
    latent_t: int
    latent_h: int
    latent_w: int
    audio_t: int
    media: MediaConditioning = field(default_factory=MediaConditioning)
    cfg: float = 1.0


@dataclass(frozen=True)
class VRAMEstimate:
    """Estimated non-weight VRAM for one sequence."""

    sequence_tokens: int
    activation_mb: float
    comfy_reserve_mb: float
    perf_headroom_mb: float
    runtime_lora_total_mb: float
    runtime_lora_fixed_mb: float

    @property
    def reserve_mb(self) -> float:
        return (
            self.activation_mb
            + self.comfy_reserve_mb
            + self.perf_headroom_mb
        )


@dataclass(frozen=True)
class PoolPlan:
    """Final BlockSwap pool geometry after VRAM planning."""

    block_mb: float
    free_mb: float
    effective_reserve_mb: float
    lora_per_slot_mb: float
    max_slots: int
    requested_slots: int
    window_size: int
    hot_blocks: int
    prefetch_count: int
    home_slots: int
    gpu_slots: int


@dataclass(frozen=True)
class SwapAllocation:
    """Effective swap config plus resolved pool geometry."""

    config: H3BlockSwap
    pool: PoolPlan


__all__ = [
    "LoadingMode",
    "CondKind",
    "AdaLNCacheKey",
    "AdaLNCacheEntry",
    "AdaLNCache",
    "H3BlockSwap",
    "H3TeaCache",
    "EncoderStreamConfig",
    "VAERef",
    "AttentionConfig",
    "SlotEntry",
    "SwapBlock",
    "LoraEntry",
    "AdaLNOverride",
    "H3Lora",
    "H3LoraSet",
    "TextConditioning",
    "KeyframeCondition",
    "ReferenceCondition",
    "MediaConditioning",
    "H3Conditioning",
    "AVLatent",
    "H3SampleResult",
    "BlockSwapStats",
    "FLKeyframe",
    "FLConstraint",
    "RollingSegment",
    "RollingPlan",
    "SamplingConfig",
    "RuntimeOptions",
    "SamplingAssets",
    "ForwardRequest",
    "SessionContext",
    "SamplerRequest",
    "SegmentRequest",
    "SegmentResult",
    "DecodedSegment",
    "RollingOutput",
    "SequenceSpec",
    "VRAMEstimate",
    "PoolPlan",
    "SwapAllocation",
    "_params_extra_fields",
]
