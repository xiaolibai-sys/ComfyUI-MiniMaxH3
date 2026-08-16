"""Single sampling-run runner built on SessionContext."""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import comfy.model_sampling
import comfy.samplers
import comfy.utils

from .types import (
    ForwardRequest,
    H3SampleResult,
    SamplerRequest,
    SamplingConfig,
    SessionContext,
)
from .native_samplers import _h3_sampler
from .wrapper import H3ModelWrapper

logger = logging.getLogger("h3.sampler")


class SamplerRunner:
    """Run one k-diffusion pass and collect BlockSwap/sampling stats."""

    def __init__(
        self,
        context: SessionContext,
        config: SamplingConfig,
    ):
        self.context = context
        self.config = config

    def run(
        self,
        request: SamplerRequest,
        preview_callback=None,
        disable_pbar: bool = False,
    ) -> H3SampleResult:
        context = self.context
        config = self.config
        device = context.device
        dtype = context.dtype
        assert len(context.sigmas) >= 2, "sigmas must contain at least one step"
        assert request.latent.video.ndim == 5, "video latent must be [B,C,T,H,W]"
        assert request.latent.audio.ndim == 4, "audio latent must be [B,C,Ch,T]"
        assert request.latent.video.shape[0] == 1, "batch size must be 1"
        assert request.latent.audio.shape[0] == 1, "audio batch size must be 1"

        forward = ForwardRequest(
            model=context.model,
            cfg=config.cfg,
            video_shape=request.latent.video.shape,
            audio_shape=request.latent.audio.shape,
            positive_text=(
                request.positive_text
                if request.positive_text is not None
                else context.positive_text
            ),
            positive_payload=(
                request.payload
                if request.payload is not None
                else context.positive_payload
            ),
            negative_text=(
                request.negative_text
                if request.negative_text is not None
                else context.negative_text
            ),
            negative_payload=(
                request.negative_payload
                if request.negative_payload is not None
                else context.negative_payload
            ),
            shift_video=config.shift_video,
            shift_audio=config.shift_audio,
        )
        wrapper = H3ModelWrapper(forward)

        latent_packed = wrapper.pack(
            request.latent.video.to(device, dtype),
            request.latent.audio.to(device, dtype),
        )

        gen = torch.Generator("cpu").manual_seed(request.seed)
        noise_v = torch.randn(
            request.latent.video.shape,
            generator=gen,
            dtype=torch.float32,
        )
        noise_a = torch.randn(
            request.latent.audio.shape,
            generator=gen,
            dtype=torch.float32,
        )
        noise = torch.cat([
            noise_v.reshape(noise_v.shape[0], -1),
            noise_a.reshape(noise_a.shape[0], -1),
        ], dim=1).to(device=device, dtype=dtype)

        total_steps = len(context.sigmas) - 1
        tc = context.teacache
        if tc is not None:
            tc.reset(total_steps)
        pbar = (
            None
            if preview_callback is not None
            else comfy.utils.ProgressBar(total_steps)
        )

        stats = context.block_stats
        base_hits = stats.swap_hits if stats is not None else 0
        base_loads = stats.swap_loads if stats is not None else 0
        base_d2h_stage = stats.d2h_stage if stats is not None else 0
        base_d2h_direct = stats.d2h_direct if stats is not None else 0
        base_d2h_host = stats.d2h_host_register if stats is not None else 0
        base_d2h_sync = stats.d2h_sync if stats is not None else 0

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        vram_before = (
            torch.cuda.memory_allocated()
            if torch.cuda.is_available()
            else 0
        )

        def callback(*args):
            if len(args) == 1 and isinstance(args[0], dict):
                info = args[0]
                step = int(info["i"])
                x0 = info["denoised"]
                x = info["x"]
                total = len(context.sigmas) - 1
            else:
                step, x0, x, total = args
            if tc is not None:
                tc.step()
            if pbar is not None:
                pbar.update_absolute(step + 1)
            if preview_callback is not None:
                x0_v, _ = wrapper.unpack(x0)
                x_v, _ = wrapper.unpack(x)
                preview_callback(step, x0_v, x_v, total)

        if getattr(comfy.model_sampling, "ModelSamplingAV", None) is None:
            logger.warning(
                "SamplerRunner: comfy.model_sampling.ModelSamplingAV missing; "
                "using native dual-clock fallback for %s.",
                config.sampler_name,
            )
            samp = _h3_sampler(config.sampler_name)
            out = samp(
                wrapper,
                latent_packed,
                context.sigmas,
                extra_args={"model_options": {}, "seed": request.seed},
                callback=callback,
                disable=disable_pbar,
            )
        else:
            samp = comfy.samplers.sampler_object(config.sampler_name)
            out = samp.sample(
                wrapper,
                context.sigmas,
                extra_args={"model_options": {}, "seed": request.seed},
                callback=callback,
                noise=noise,
                latent_image=latent_packed,
                denoise_mask=None,
                disable_pbar=disable_pbar,
            )
        out_v, out_a = wrapper.unpack(out)

        peak = (
            (torch.cuda.max_memory_allocated() - vram_before) / 2 ** 20
            if torch.cuda.is_available()
            else 0.0
        )
        return H3SampleResult(
            video=out_v.to(request.latent.video.dtype),
            audio=out_a.to(request.latent.audio.dtype),
            steps=total_steps,
            swap_hits=(stats.swap_hits - base_hits)
            if stats is not None else 0,
            swap_loads=(stats.swap_loads - base_loads)
            if stats is not None else 0,
            peak_vram_mb=peak,
            d2h_stage=(stats.d2h_stage - base_d2h_stage)
            if stats is not None else 0,
            d2h_direct=(stats.d2h_direct - base_d2h_direct)
            if stats is not None else 0,
            d2h_host_register=(
                stats.d2h_host_register - base_d2h_host)
            if stats is not None else 0,
            d2h_sync=(stats.d2h_sync - base_d2h_sync)
            if stats is not None else 0,
        )


__all__ = ["SamplerRunner"]
