"""Measured single-block activation peak models per attention backend.

All functions return MB for one DiT block at a given packed sequence length.
These models were calibrated on MiniMax H3 with bf16 and current attention
backends; full-forward planning must add persistent h, embeddings, final
layer, attention workspace and ComfyUI reserve separately.
"""

from __future__ import annotations

from typing import Callable

from .types import H3LoraSet, SequenceSpec

SAGE3_MAX_SEQ = 65536
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


def sequence_token_count(spec: SequenceSpec) -> int:
    """Typed replacement for ``estimate_sequence_length``."""
    n = (
        int(spec.text_len)
        + int(spec.latent_t * (spec.latent_h // 2) * (spec.latent_w // 2))
        + int(spec.audio_t * 2)
    )
    for kf in spec.media.keyframes:
        z = kf.latent
        n += int((z.shape[-2] // 2) * (z.shape[-1] // 2))
    for ref in spec.media.refs:
        kind = ref.kind
        if kind in ("image", "video", "video_audio"):
            lat_h = ref.latent_h or ref.latent.shape[-2]
            lat_w = ref.latent_w or ref.latent.shape[-1]
            n += int(max(ref.latent_t, 1) * (lat_h // 2) * (lat_w // 2))
        n += int(ref.ref_audio_t) * 2
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
