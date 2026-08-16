"""Test-only helper for calling run_sampling with the legacy argument shape."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from h3rt.nodes.h3_sampling import run_sampling
from h3rt.utils.types import (
    FLConstraint,
    RuntimeOptions,
    SamplingAssets,
    SamplingConfig,
    SequenceSpec,
)


def h3_sample(
    handle,
    conditioning,
    latent,
    negative,
    steps,
    cfg,
    sampler_name,
    shift_video,
    denoise,
    seed,
    runtime,
    preview_callback=None,
    disable_pbar=False,
    use_adaln_cache=False,
    shift_audio=3.0,
    scheduler_name="normal",
    adaln_prebake_batch=3,
):
    if negative is None:
        cfg = 1.0

    text_len = conditioning.text.states.shape[1]
    latent_t, lat_h, lat_w = (
        latent.video.shape[2],
        latent.video.shape[3],
        latent.video.shape[4],
    )
    audio_t = latent.audio.shape[-1]
    vram_spec = SequenceSpec(
        text_len=text_len,
        latent_t=latent_t,
        latent_h=lat_h,
        latent_w=lat_w,
        audio_t=audio_t,
        media=conditioning.media,
        cfg=cfg,
    )
    assets = SamplingAssets(
        handle=handle,
        positive=conditioning,
        negative=negative,
        fl_constraint=(
            getattr(conditioning, "fl_constraint", None)
            or FLConstraint()
        ),
        av_encoder=getattr(conditioning, "av_encoder", None),
        runtime=runtime,
        vram_spec=vram_spec,
        latent=latent,
    )
    config = SamplingConfig(
        steps=steps,
        cfg=cfg,
        seed=seed,
        sampler_name=sampler_name,
        scheduler_name=scheduler_name,
        shift_video=shift_video,
        shift_audio=shift_audio,
        use_adaln_cache=use_adaln_cache,
        adaln_prebake_batch=adaln_prebake_batch,
        width=latent.video.shape[4] * 16,
        height=latent.video.shape[3] * 16,
        denoise=denoise,
    )
    return run_sampling(
        assets,
        config,
        preview_callback=preview_callback,
        disable_pbar=disable_pbar,
    )
