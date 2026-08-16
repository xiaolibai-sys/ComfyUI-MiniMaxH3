"""Sampling core: the official ComfyUI k-diffusion loop over the packed H3 AV latent.

Mirrors ComfyUI-BerniniRWrapper's ``nodes/bernini_sampling.py``:

* ``H3ModelWrapper`` replaces ``CFGGuider`` — k-diffusion samplers call
  ``model(x, sigma, **extra_args)`` and get a denoised prediction back;
  guidance (cond/uncond combine) lives here.
* ``h3_sample`` drives a ``comfy.samplers.KSAMPLER`` with the H3
  ``flow_sigmas`` grid. Video advances on the video sigma clock while audio
  is integrated on the model's shifted audio clock.

The joint video+audio latent is packed into one flat vector
``cat([video.flatten(1), audio.flatten(1)], dim=1)``: packing is linear, so
the sampler can update the video and audio halves independently without
changing the packed layout.
"""

from __future__ import annotations

from dataclasses import replace

import torch
import comfy.samplers

from ..utils.types import (
    H3SampleResult,
    RollingOutput,
    SamplingAssets,
    SamplingConfig,
)

H3_SAMPLERS = list(comfy.samplers.KSampler.SAMPLERS)
H3_SCHEDULERS = ["flow_uniform"] + list(comfy.samplers.KSampler.SCHEDULERS)
ADALN_PREBAKE_UNSUPPORTED = {"dpm_adaptive"}


# ---------------------------------------------------------------------------
# Sampling entry point
# ---------------------------------------------------------------------------

def run_sampling(
    assets: SamplingAssets,
    config: SamplingConfig,
    preview_callback=None,
    disable_pbar: bool = False,
) -> H3SampleResult | RollingOutput:
    """Run H3 denoising through the official k-diffusion sampler loop."""
    negative = assets.negative
    latent = assets.latent
    if latent is None:
        raise ValueError("SamplingAssets.latent is required for run_sampling.")
    assert config.steps >= 1, "steps must be >= 1"
    assert 0.0 <= config.denoise <= 1.0, "denoise must be in [0, 1]"
    assert config.cfg >= 1.0, "cfg must be >= 1.0"
    assert config.sampler_name, "sampler_name is required"
    assert config.scheduler_name, "scheduler_name is required"
    has_segment_negative = bool(getattr(
        getattr(assets, "positive", None),
        "segment_negative_texts",
        None,
    ))
    if negative is None and not has_segment_negative:
        config = replace(config, cfg=1.0)
    if config.sampler_name in ADALN_PREBAKE_UNSUPPORTED:
        config = replace(config, use_adaln_cache=False)

    from ..utils.rolling import rolling_sample

    result = rolling_sample(
        assets,
        config,
        preview_callback=preview_callback,
        disable_pbar=disable_pbar,
    )
    return result
