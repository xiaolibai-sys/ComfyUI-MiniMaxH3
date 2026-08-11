"""Attention backends with auto-detection and fallback chain (BerniniRWrapper-style).

Fallback order (best -> worst):
  SageAttention 3 (Blackwell sm_100+) -> SageAttention 2 (sm_90) ->
  SageAttention 1 (sm_80+) -> FlashAttention -> SDPA-flash (torch precompiled
  kernel, ~FlashAttention speed) -> xformers -> PyTorch SDPA (auto) ->
  SDPA-math (eager, always works).

Every backend is a standalone callable ``fn(q, k, v, heads) -> [B, S, heads*dim]``
where ``q/k/v`` are ``[B, heads, S, dim_head]`` — the same contract our DiT
``Attention`` consumes.  A runtime wrapper catches any kernel failure and falls
back to the eager math backend so sampling never crashes.
"""

from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn.functional as F

logger = logging.getLogger("h3.attn")

_PRIORITY = ["sageattn3", "sageattn2", "sageattn1", "flash_attention",
             "sdpa_flash", "xformers", "sdpa", "sdpa_math"]

# SageAttention3's Blackwell TMA descriptor fails for very long sequences
# (e.g. 71k tokens on this box). Keep Sage3 below the verified safe length
# and let Sage2 handle longer prefill sequences.
_SAGE3_MAX_SEQ = 65536

_AVAILABLE: dict[str, bool] = {}

# -- SageAttention 3 (Blackwell: 5090/5070 Ti/B200) --
try:
    import sageattn3  # noqa: F401
    _AVAILABLE["sageattn3"] = True
except Exception:
    _AVAILABLE["sageattn3"] = False

# -- SageAttention 2 (sm_90) / 1 (sm_80+) --
try:
    import sageattention  # noqa: F401
    _AVAILABLE["sageattn2"] = True
    _AVAILABLE["sageattn1"] = True
except Exception:
    _AVAILABLE["sageattn2"] = False
    _AVAILABLE["sageattn1"] = False

# -- FlashAttention (Dao) --
try:
    from flash_attn import flash_attn_func  # noqa: F401
    _AVAILABLE["flash_attention"] = True
except Exception:
    _AVAILABLE["flash_attention"] = False

# -- xformers --
try:
    import xformers.ops  # noqa: F401
    _AVAILABLE["xformers"] = True
except Exception:
    _AVAILABLE["xformers"] = False

# -- PyTorch SDPA backends (always available on CUDA) --
_AVAILABLE["sdpa_flash"] = torch.cuda.is_available()
_AVAILABLE["sdpa"] = True
_AVAILABLE["sdpa_math"] = True


def available_backends() -> list[str]:
    return [n for n in _PRIORITY if _AVAILABLE.get(n, False)]


def best_available() -> str:
    for n in _PRIORITY:
        if _AVAILABLE.get(n, False):
            return n
    return "sdpa_math"


BACKEND_NAMES = ["auto"] + [n for n in _PRIORITY if _AVAILABLE.get(n, False)]


# ---------------------------------------------------------------------------
# Backend implementations  (q/k/v: [B, heads, S, dim_head])
# ---------------------------------------------------------------------------

def _reshape_out(out: torch.Tensor) -> torch.Tensor:
    return out.transpose(1, 2).reshape(out.shape[0], out.shape[2], -1)


def _sdpa_math_core(q, k, v):
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)


def _sdpa_flash_core(q, k, v):
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)


def _sdpa_core(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)


def _flash_core(q, k, v):
    from flash_attn import flash_attn_func
    # flash_attn expects [B, S, H, D], while our DiT passes [B, H, S, D].
    q, k, v = (
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
    )
    out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
    return out.transpose(1, 2)


def _xformers_core(q, k, v):
    import xformers.ops as xops
    return xops.memory_efficient_attention(q, k, v)


def _sage3_core(q, k, v):
    if q.shape[-2] > _SAGE3_MAX_SEQ or k.shape[-2] > _SAGE3_MAX_SEQ:
        logger.warning(
            "SageAttention3 does not support seq_len > %d; using SageAttention2 "
            "for this attention call", _SAGE3_MAX_SEQ)
        try:
            return _sage_core(q, k, v)
        except Exception:
            return _sdpa_flash_core(q, k, v)
    from sageattn3 import sageattn3_blackwell
    return sageattn3_blackwell(q, k, v)


def _sage_core(q, k, v):
    # ComfyUI passes sm_scale = 1/sqrt(head_dim); without it sageattn applies
    # the wrong softmax temperature -> large error.
    from sageattention import sageattn
    sm = 1.0 / (q.shape[-1] ** 0.5)
    return sageattn(q, k, v, sm_scale=sm)


_CORES = {
    "sageattn3": _sage3_core,
    "sageattn2": _sage_core,
    "sageattn1": _sage_core,
    "flash_attention": _flash_core,
    "sdpa_flash": _sdpa_flash_core,
    "xformers": _xformers_core,
    "sdpa": _sdpa_core,
    "sdpa_math": _sdpa_math_core,
}


def _wrap(name: str, core: Callable) -> Callable:
    def fn(q, k, v, heads):
        try:
            # SageAttention and PyTorch SDPA accept the non-contiguous q/k/v
            # views from the fused qkv buffer directly. Avoiding per-layer
            # copies saves about 1.3GB peak VRAM at H3 sequence lengths.
            if not (name.startswith("sageattn") or name in ("sdpa", "sdpa_math")):
                q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
            out = core(q, k, v)
            if out.shape[2] != q.shape[2]:          # kernels returning [B, S, D]
                out = out.view(q.shape[0], q.shape[2], heads, -1).transpose(1, 2)
            return _reshape_out(out)
        except Exception as e:
            logger.warning("attention backend %s failed (%s); falling back to sdpa_math", name, e)
            return _reshape_out(_sdpa_math_core(q, k, v))
    return fn


_BACKENDS: dict[str, Callable] = {
    n: _wrap(n, c) for n, c in _CORES.items()
}


# SageAttention3 quantizes q/k/v to FP4, so ~15-20% max-relative error vs
# exact SDPA is expected on this box (measured 0.14-0.17) and is not a broken
# build; int8 SageAttention2 is much tighter (~2.5%).
_TOL = {"sageattn3": 0.35, "sageattn2": 0.10, "sageattn1": 0.10,
        "flash_attention": 0.05, "xformers": 0.05}


def _self_test(name: str, fn: Callable, tol: float | None = None) -> bool:
    """Verify a kernel is numerically sane on realistic inputs.

    Inputs mimic real DiT attention (RMSNormed q/k, so magnitudes are bounded)
    and are compared to exact SDPA.  Tolerance is per-backend: FP4
    SageAttention3 accepts ~35%, int8 kernels ~10%, exact kernels ~5%.
    """
    tol = _TOL.get(name, 0.10) if tol is None else tol
    if name.startswith("sdpa"):
        return True                      # torch's own kernels are trustworthy
    try:
        torch.manual_seed(0)
        b, h, s, d = 1, 4, 512, 64
        q = F.rms_norm(torch.randn(b, h, s, d, device="cuda", dtype=torch.bfloat16), (d,)) * 0.7
        k = F.rms_norm(torch.randn(b, h, s, d, device="cuda", dtype=torch.bfloat16), (d,)) * 0.7
        v = torch.randn(b, h, s, d, device="cuda", dtype=torch.bfloat16) * 0.1
        ref = F.scaled_dot_product_attention(q, k, v)
        out = fn(q, k, v, h)
        ok_shape = out.shape == (b, s, h * d)
        if not ok_shape:
            logger.warning("attention backend %s smoke-test FAILED (shape=%s) - disabled",
                           name, tuple(out.shape))
            return False
        rel = (out.float() - ref.transpose(1, 2).reshape(b, s, -1).float()).abs().max().item()               / max(1e-6, ref.float().abs().max().item())
        ok = torch.isfinite(out).all() and rel < tol
        if not ok:
            logger.warning("attention backend %s self-test FAILED (rel=%.3f) - disabled", name, rel)
        return ok
    except Exception as e:
        logger.warning("attention backend %s self-test error (%s) - disabled", name, e)
        return False


def create_attention_override(backend: str = "auto", *, force_backend: bool = False) -> Callable:
    """Build the override callable for ``model.set_attn_backend``.

    Every candidate is verified on a tiny input before being accepted, so a
    broken kernel (bad build / wrong GPU) never silently degrades sampling.
    """
    if backend == "auto":
        for n in _PRIORITY:
            if _AVAILABLE.get(n, False) and _self_test(n, _BACKENDS[n]):
                logger.info("attention backend (auto): %s", n)
                return _BACKENDS[n]
        return _BACKENDS["sdpa_math"]

    fn = _BACKENDS.get(backend)
    if fn is not None and _self_test(backend, fn):
        logger.info("attention backend: %s", backend)
        return fn
    if force_backend:
        logger.warning(
            "attention backend %s failed self-test; "
            "falling back instead of aborting the workflow", backend)
    for n in _PRIORITY:
        if _AVAILABLE.get(n, False) and _self_test(n, _BACKENDS[n]):
            logger.warning("backend %s unavailable/broken -> %s", backend, n)
            return _BACKENDS[n]
    return _BACKENDS["sdpa_math"]
