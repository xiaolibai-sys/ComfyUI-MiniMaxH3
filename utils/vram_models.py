"""Measured single-block activation peak models per attention backend.

All functions return MB for one DiT block at a given packed sequence length.
These models were calibrated on MiniMax H3 with bf16 and current attention
backends; full-forward planning must add persistent h, embeddings, final
layer, attention workspace and ComfyUI reserve separately.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from .types import H3BlockSwap, H3LoraSet

SAGE3_MAX_SEQ = 65536
CFG_SAFETY_MB = 256.0
CFG_FULL_EXTRA_MB = 5.0
PERSISTENT_BYTES_PER_TOKEN = 5376 * 2
FULL_FORWARD_EXTRA_INTERCEPT_MB = 30.0
FULL_FORWARD_EXTRA_SLOPE_MB = 0.0107
COMFY_RESERVE_MB = 1519.2
PERF_HEADROOM_MB = 2457.6


def linear_mb(n_tokens: int, slope: float, intercept: float) -> float:
    """Linear peak model in MB."""
    return intercept + slope * n_tokens


def quadratic_mb(
    n_tokens: int,
    intercept: float,
    slope: float,
    quadratic: float,
) -> float:
    """Quadratic peak model in MB."""
    return intercept + slope * n_tokens + quadratic * n_tokens * n_tokens


def sage1_sage2_sdpa_mb(n_tokens: int) -> float:
    """SageAttention1/2 and PyTorch SDPA no-copy path."""
    return linear_mb(n_tokens, 0.1265, 0.37)


def sdpa_attention_mb(n_tokens: int) -> float:
    """PyTorch SDPA attention-only peak."""
    return linear_mb(n_tokens, 0.01367188, 0.0)


def sage_attention2_peak_mb(n_tokens: int) -> float:
    """SageAttention2 attention-only peak."""
    return linear_mb(n_tokens, 0.04102065, 0.01348432)


def sage_attention3_peak_mb(n_tokens: int) -> float:
    """Sage3 attention upper-bound formula based on 128-token padding."""
    if n_tokens > SAGE3_MAX_SEQ:
        return sage_attention2_peak_mb(n_tokens)
    padded = ((n_tokens + 127) // 128) * 128
    groups = padded // 128
    return (
        83776 * padded + 224 * groups * padded + 14336 * groups
    ) / (1024 * 1024)


def flash_attention_peak_mb(n_tokens: int) -> float:
    """FlashAttention attention-only peak."""
    return linear_mb(n_tokens, 0.05491501, -0.14570944)


def flash_attention_mb(n_tokens: int) -> float:
    """FlashAttention path with q/k/v contiguous copies."""
    return linear_mb(n_tokens, 0.1745, 0.58)


def sage_attention3_mb(n_tokens: int) -> float:
    """SageAttention3 FP4/TMA path; falls back to Sage2 above the limit."""
    if n_tokens > SAGE3_MAX_SEQ:
        return sage1_sage2_sdpa_mb(n_tokens)
    return quadratic_mb(n_tokens, 227.6, 0.12388, 2.37e-6)


BACKEND_ACTIVATION_MODELS: dict[str, Callable[[int], float]] = {
    "sageattn1": sage1_sage2_sdpa_mb,
    "sageattn2": sage1_sage2_sdpa_mb,
    "sdpa": sage1_sage2_sdpa_mb,
    "flash_attention": flash_attention_mb,
    "sageattn3": sage_attention3_mb,
}

ATTENTION_PEAK_MODELS: dict[str, Callable[[int], float]] = {
    "sdpa": sdpa_attention_mb,
    "sageattn2": sage_attention2_peak_mb,
    "sageattn3": sage_attention3_peak_mb,
    "flash_attention": flash_attention_peak_mb,
}


def activation_peak_mb(
    backend: str,
    n_tokens: int,
    *,
    cfg: float = 1.0,
) -> float:
    """Single-block activation estimate including a small CFG safety margin."""
    model = BACKEND_ACTIVATION_MODELS.get(
        backend, sage1_sage2_sdpa_mb)
    peak = model(n_tokens)
    if cfg > 1.0:
        peak += CFG_SAFETY_MB
    return peak


def full_forward_activation_mb(
    backend: str,
    n_tokens: int,
    *,
    cfg: float = 1.0,
) -> float:
    """General full forward activation from block peak plus fixed components."""
    block_peak = BACKEND_ACTIVATION_MODELS.get(
        backend, sage1_sage2_sdpa_mb)(n_tokens)
    persistent_mb = (
        n_tokens * PERSISTENT_BYTES_PER_TOKEN / (1024 * 1024)
    )
    extra_mb = (
        FULL_FORWARD_EXTRA_INTERCEPT_MB
        + FULL_FORWARD_EXTRA_SLOPE_MB * n_tokens
    )
    peak = block_peak + persistent_mb + extra_mb
    if cfg > 1.0:
        peak += CFG_FULL_EXTRA_MB
    return peak


def total_vram_mb(
    backend: str,
    n_tokens: int,
    block_mb: float,
    gpu_slots: int,
    *,
    cfg: float = 1.0,
    lora_extra_mb: float = 0.0,
    periphery_mb: float = 512.0,
    comfy_reserve_mb: float = 1519.2,
    safety_mb: float = 1024.0,
) -> float:
    """Total VRAM budget model for one sampling configuration."""
    # block_mb should be the actual post-LoRA-fold block size. lora_extra_mb
    # covers runtime AdaLN/LoRA state that is not part of the folded block.
    weights_mb = (
        gpu_slots * block_mb
        + periphery_mb
        + lora_extra_mb
    )
    activation_mb = full_forward_activation_mb(
        backend, n_tokens, cfg=cfg)
    return weights_mb + activation_mb + comfy_reserve_mb + safety_mb


def estimate_sequence_length(text_len: int, latent, payload: dict) -> int:
    """Count packed DiT tokens for the current AV latent and conditioning."""
    n = (
        int(text_len)
        + int(
            latent.video.shape[2]
            * (latent.video.shape[3] // 2)
            * (latent.video.shape[4] // 2)
        )
        + int(latent.audio.shape[-1] * 2)
    )
    for kf in payload.get("keyframes") or []:
        z = kf["latent"]
        n += int((z.shape[-2] // 2) * (z.shape[-1] // 2))
    for ref in payload.get("refs") or []:
        kind = ref.get("kind")
        if kind in ("image", "video", "video_audio"):
            lat_h = ref.get("latent_h") or ref["latent"].shape[-2]
            lat_w = ref.get("latent_w") or ref["latent"].shape[-1]
            n += int(ref.get("latent_t", 1) * (lat_h // 2) * (lat_w // 2))
        n += int(ref.get("ref_audio_t", 0)) * 2
    return n


def _tensor_mb(t) -> float:
    if t is None:
        return 0.0
    return t.numel() * t.element_size() / (1024 * 1024)


def estimate_runtime_lora_mb(loras: H3LoraSet) -> tuple[float, float]:
    """Estimate runtime LoRA bytes that follow the block window and fixed extras."""
    total_mb = 0.0
    fixed_mb = 0.0
    for lora in loras.loras:
        runtime = lora.adaln_override is None
        for entries in lora.block_groups.values():
            for entry in entries:
                if runtime and entry.target == "adaln_proj.linear":
                    total_mb += (
                        _tensor_mb(entry.a)
                        + _tensor_mb(entry.b)
                        + _tensor_mb(entry.diff)
                        + _tensor_mb(entry.diff_b)
                    )
        if runtime and lora.final_adaln is not None:
            entry = lora.final_adaln
            total_mb += (
                _tensor_mb(entry.a)
                + _tensor_mb(entry.b)
                + _tensor_mb(entry.diff)
                + _tensor_mb(entry.diff_b)
            )
        for entries in lora.token_refiner_groups.values():
            for entry in entries:
                fixed_mb += (
                    _tensor_mb(entry.a)
                    + _tensor_mb(entry.b)
                    + _tensor_mb(entry.diff)
                    + _tensor_mb(entry.diff_b)
                )
    return total_mb, fixed_mb


def make_static_reserved_swap(
    swap: H3BlockSwap,
    backend: str,
    text_len: int,
    latent,
    payload: dict,
    *,
    cfg: float = 1.0,
    loras: H3LoraSet | None = None,
) -> H3BlockSwap:
    """Attach a static VRAM reserve before the block pool is allocated."""
    if not swap.auto_vram:
        return swap
    if swap.vram_reserve_mb > 0:
        return swap
    n_tokens = estimate_sequence_length(text_len, latent, payload)
    activation_mb = full_forward_activation_mb(
        backend, n_tokens, cfg=cfg)
    reserve_mb = (
        activation_mb + COMFY_RESERVE_MB + PERF_HEADROOM_MB
    )
    runtime_total, runtime_fixed = estimate_runtime_lora_mb(
        loras or H3LoraSet())
    return replace(
        swap,
        vram_reserve_mb=reserve_mb,
        runtime_lora_total_mb=runtime_total,
        runtime_lora_fixed_mb=runtime_fixed,
    )
