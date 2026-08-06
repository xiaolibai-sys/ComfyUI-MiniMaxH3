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

import math
from typing import Optional
from types import SimpleNamespace

import torch
import comfy.model_management
import comfy.model_sampling
import comfy.samplers
import comfy.utils

from ..models.model import flow_sigmas, time_shift_sigma, time_shift_slope
from ..utils.injection import InjectionContext
from ..utils.lifecycle import collect_garbage
from ..utils.types import AVLatent, H3Conditioning, H3SampleResult

SHIFT_V = 12.0
SHIFT_A = 3.0


def _h3_native_sampler(model, x, sigmas, extra_args=None, callback=None,
                       disable=None, **kwargs):
    """Native MiniMax-H3 AV dual-schedule integrator.

    The video stream advances on the video sigma grid (default shift=12) while
    the audio stream advances on its own shifted sigma grid (default shift=3).
    This is the schedule used by the model's native training, so it is always
    used by the MiniMax-H3 KSampler instead of a single packed-latent
    schedule.  Overridable shifts are read from the wrapped model.
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    # KSAMPLER passes KSamplerX0Inpaint as `model`; the packed AV split point
    # lives on H3ModelWrapper at model.inner_model.
    nv = getattr(model, "_n_video", None)
    if nv is None:
        nv = getattr(getattr(model, "inner_model", None), "_n_video", None)
    shift_v, shift_a = SHIFT_V, SHIFT_A
    probe = model
    for _ in range(2):
        shift_v = float(getattr(probe, "_shift_video", shift_v))
        shift_a = float(getattr(probe, "_shift_audio", shift_a))
        probe = getattr(probe, "inner_model", None)
        if probe is None:
            break
    for i in range(len(sigmas) - 1):
        sv, sv_n = float(sigmas[i]), float(sigmas[i + 1])
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        out = (x - denoised) / sigmas[i]
        if nv is None:
            x = x + (sv_n - sv) * out
        else:
            xv, xa = x[..., :nv], x[..., nv:]
            ov, oa = out[..., :nv], out[..., nv:]
            xv = xv + (sv_n - sv) * ov
            slope = time_shift_slope(max(sv, 1e-6), shift_v, shift_a)
            xa = xa + (time_shift_sigma(sv_n, shift_v, shift_a) -
                       time_shift_sigma(sv, shift_v, shift_a)) * (oa / slope)
            x = torch.cat([xv, xa], dim=-1)
        if callback is not None:
            callback({"i": i, "denoised": denoised, "x": x,
                      "sigma": sigmas[i], "sigma_hat": sigmas[i]})
    return x


# ---------------------------------------------------------------------------
# k-diffusion interface shims
# ---------------------------------------------------------------------------

class _H3ModelSampling(comfy.model_sampling.CONST,
                       comfy.model_sampling.ModelSamplingDiscreteFlow):
    """Flow-matching sigma behaviour (CONST) + discrete-flow sigma range."""
    pass


class _InnerShim:
    """What ``KSAMPLER`` / k-diffusion reach for via ``model_wrap.inner_model``:
    only ``model_sampling`` is ever read (noise_scaling / inverse_noise_scaling
    / CONST isinstance checks)."""

    def __init__(self):
        self.model_sampling = _H3ModelSampling()


class _PatcherShim:
    """Minimal ModelPatcher stand-in: ``.model`` for KSAMPLER's log line and
    ``get_model_object("model_sampling")`` for SDE-family samplers."""

    def __init__(self, model, model_sampling):
        self.model = model
        self._model_sampling = model_sampling

    def get_model_object(self, name: str):
        if name == "model_sampling":
            return self._model_sampling
        raise AttributeError(f"_PatcherShim has no model object {name!r}")


class _SigmaCaptureModel:
    """Dummy k-diffusion model used to enumerate sampler sigma calls."""

    def __init__(self, model_sampling, shift_video=SHIFT_V, shift_audio=SHIFT_A):
        self.inner_model = SimpleNamespace(
            model_sampling=model_sampling,
            inner_model=SimpleNamespace(model_sampling=model_sampling),
        )
        self.model_patcher = _PatcherShim(self, model_sampling)
        self.cfg = 1.0
        self.captured = []
        self._shift_video = float(shift_video)
        self._shift_audio = float(shift_audio)

    def __call__(self, x, sigma, **kwargs):
        self.captured.append(float(sigma.flatten()[0]))
        return torch.zeros_like(x)


# ---------------------------------------------------------------------------
# Model wrapper — replaces CFGGuider for k-diffusion
# ---------------------------------------------------------------------------

class H3ModelWrapper:
    """k-diffusion entry point: packed noisy AV latent -> packed denoised prediction.

    Runs one forward per step (cond only, cfg == 1) or two (cond + uncond)
    and combines velocities with classifier-free guidance before converting
    to the denoised estimate ``x0 = x - sigma * v``.
    """

    def __init__(self, model, cfg: float, video_shape, audio_shape,
                 text_states: torch.Tensor, payload: dict,
                 neg_text_states: Optional[torch.Tensor] = None,
                 neg_payload: Optional[dict] = None,
                 shift_video: float = SHIFT_V, shift_audio: float = SHIFT_A):
        self.model = model
        self.cfg = float(cfg)
        self._video_shape = tuple(video_shape)
        self._audio_shape = tuple(audio_shape)
        self._n_video = math.prod(self._video_shape[1:])
        self._text_states = text_states
        self._payload = payload
        self._neg_text_states = neg_text_states
        self._neg_payload = neg_payload
        self._shift_video = float(shift_video)
        self._shift_audio = float(shift_audio)

        self.inner_model = _InnerShim()
        self.model_patcher = _PatcherShim(model, self.inner_model.model_sampling)

    # -- packed AV latent -------------------------------------------------

    def pack(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        return torch.cat([video.reshape(video.shape[0], -1),
                          audio.reshape(audio.shape[0], -1)], dim=1)

    def unpack(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        video = x[:, :self._n_video].reshape(self._video_shape)
        audio = x[:, self._n_video:].reshape(self._audio_shape)
        return video, audio

    # -- k-diffusion interface --------------------------------------------

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor, **extra_args) -> torch.Tensor:
        comfy.model_management.throw_exception_if_processing_interrupted()
        s = float(sigma.flatten()[0])
        x_v, x_a = self.unpack(x)

        v_v, v_a = self.model.velocity(
            x_v, x_a, s, self._text_states, self._payload,
            shift_video=self._shift_video, shift_audio=self._shift_audio)
        if self._neg_text_states is not None and self.cfg != 1.0:
            u_v, u_a = self.model.velocity(
                x_v, x_a, s, self._neg_text_states, self._neg_payload,
                shift_video=self._shift_video, shift_audio=self._shift_audio)
            v_v = u_v + self.cfg * (v_v - u_v)
            v_a = u_a + self.cfg * (v_a - u_a)

        x0_v = x_v - s * v_v
        x0_a = x_a - s * v_a
        return self.pack(x0_v, x0_a)


# ---------------------------------------------------------------------------
# Sampling entry point
# ---------------------------------------------------------------------------

def h3_sample(handle, conditioning: H3Conditioning, latent: AVLatent,
              negative: Optional[H3Conditioning], steps: int, cfg: float,
              sampler_name: str, shift_video: float, denoise: float, seed: int,
              injection: InjectionContext, preview_callback=None,
              disable_pbar: bool = False,
              use_adaln_cache: bool = False,
              shift_audio: float = SHIFT_A) -> H3SampleResult:
    """Run H3 denoising through the official k-diffusion sampler loop.

    ``handle`` is the lifecycle ``ModelHandle``; the model is loaded with the
    BlockSwap layout from ``injection.swap`` and always released on return.
    ``preview_callback`` follows ComfyUI's ``(step, x0, x, total)`` contract
    and receives the unpacked video latent.
    """
    if negative is None:
        cfg = 1.0

    from ..models.vae import unload_all_vaes
    from ..utils.encoder_use import unload_all_encoders
    unload_all_vaes()
    unload_all_encoders()

    device = handle.load_device
    dtype = (injection.swap or handle.swap).torch_dtype
    text_len = conditioning.text_states.shape[1]
    latent_t, lat_h, lat_w = latent.video.shape[2], latent.video.shape[3], latent.video.shape[4]
    audio_t = latent.audio.shape[-1]

    from ..models.model import PackedLayout
    from ..utils.stream import BlockReader
    from ..utils.lifecycle import detect_key_prefix, scan_dit_config
    from ..utils.config import MiniMaxH3DiTConfig

    payload = conditioning.to_payload()
    layout = PackedLayout(
        text_len, latent_t, lat_h, lat_w, audio_t,
        keyframes=payload.get("keyframes"),
        refs=payload.get("refs"),
        frame_count=payload.get("frame_count"),
    )
    payload["layout"] = layout

    neg_payload = None
    neg_layout = None
    if negative is not None:
        neg_payload = negative.to_payload()
        neg_layout = PackedLayout(
            text_len, latent_t, lat_h, lat_w, audio_t,
            keyframes=neg_payload.get("keyframes"),
            refs=neg_payload.get("refs"),
            frame_count=neg_payload.get("frame_count"),
        )
        neg_payload["layout"] = neg_layout

    if denoise <= 0.0:
        return H3SampleResult(video=latent.video, audio=latent.audio,
                              steps=0, sigmas=torch.zeros(0))
    if denoise >= 1.0:
        sigmas = flow_sigmas(steps, shift_video)
    else:
        sigmas = flow_sigmas(int(steps / denoise), shift_video)[-(steps + 1):]
    sigmas = sigmas.to(device)
    total_steps = len(sigmas) - 1

    adaln_cache = None
    reader = None
    if use_adaln_cache and total_steps > 0:
        samp = comfy.samplers.KSAMPLER(_h3_native_sampler)
        capture = _SigmaCaptureModel(
            _H3ModelSampling(),
            shift_video=shift_video,
            shift_audio=shift_audio)
        tiny = torch.zeros((1, 16), dtype=torch.float32)
        samp.sample(
            capture,
            sigmas.cpu(),
            extra_args={"model_options": {}, "seed": seed},
            callback=lambda *args: None,
            noise=torch.zeros_like(tiny),
            latent_image=tiny,
            denoise_mask=None,
            disable_pbar=True,
        )
        bake_sigmas = sorted(set(capture.captured))
        if not bake_sigmas:
            bake_sigmas = [float(s) for s in sigmas[:-1]]

        reader = BlockReader(handle.model_path)
        try:
            config = scan_dit_config(reader, MiniMaxH3DiTConfig())
            prefix = detect_key_prefix(reader)
            if handle.loras:
                has_adaln_lora = handle.loras.has_adaln
                if has_adaln_lora and config.adaln_curve_grid is None:
                    raise ValueError(
                        "AdaLN pre-bake is not supported with a full-model "
                        "LoRA fold yet; disable use_adaln_cache or use the "
                        "pruned/curve base with runtime AdaLN injection."
                    )
            from ..models.adaln import (
                AdaLNCacheBaker,
                AdaLNCachePlanner,
                bake_adaln_entry,
            )

            planner = AdaLNCachePlanner(shift_video, shift_audio)
            bake_plans = planner.build(
                bake_sigmas, payload, layout, neg_payload, neg_layout)
            bake_pbar = None
            if not disable_pbar and comfy.utils.PROGRESS_BAR_ENABLED:
                bake_pbar = comfy.utils.ProgressBar(
                    len(bake_plans) * (config.num_layers + 1))
            baker = AdaLNCacheBaker(
                reader, config, prefix, dtype, device, pbar=bake_pbar)
            adaln_cache = baker.bake(bake_plans)
        except Exception:
            reader.close()
            reader = None
            raise

    try:
        model = handle.load(swap_config=injection.swap,
                            include_adaln=not use_adaln_cache)
        if adaln_cache is not None:
            model.adaln_cache = adaln_cache
            if reader is not None:
                def bake_missing(key, unique_t):
                    entry = bake_adaln_entry(
                        reader, config, prefix, unique_t, dtype, device)
                    adaln_cache.entries[key] = entry
                    return entry
                model._adaln_bake_fallback = bake_missing
        tc = injection.make_teacache(model)
    except Exception:
        if reader is not None:
            reader.close()
            reader = None
        raise
    try:
        text_states = conditioning.text_states.to(device, handle.dtype)
        text_states = handle.preprocess_text(
            text_states, include_adaln=not use_adaln_cache)
        payload["text_token_tags"] = conditioning.text_token_tags.to(device)

        neg_text_states = None
        if negative is not None:
            neg_text_states = negative.text_states.to(device, handle.dtype)
            neg_text_states = handle.preprocess_text(
                neg_text_states, include_adaln=not use_adaln_cache)
            neg_payload["text_token_tags"] = negative.text_token_tags.to(device)

        wrapper = H3ModelWrapper(model, cfg, latent.video.shape, latent.audio.shape,
                                 text_states, payload, neg_text_states, neg_payload,
                                 shift_video=shift_video, shift_audio=shift_audio)

        latent_packed = wrapper.pack(latent.video.to(device, handle.dtype),
                                     latent.audio.to(device, handle.dtype))

        # official comfy.sample.prepare_noise: CPU generator, float32, then cast
        gen = torch.Generator("cpu").manual_seed(seed)
        noise = wrapper.pack(
            torch.randn(latent.video.shape, generator=gen, dtype=torch.float32),
            torch.randn(latent.audio.shape, generator=gen, dtype=torch.float32))
        noise = noise.to(device=device, dtype=handle.dtype)

        if tc is not None:
            tc.reset(total_steps)
        pbar = None if preview_callback is not None else comfy.utils.ProgressBar(total_steps)

        mgr = getattr(model, "_swap_mgr", None)
        base_hits = mgr.swap_hits if mgr is not None else 0
        base_loads = mgr.swap_loads if mgr is not None else 0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        vram_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        def callback(step, x0, x, total):
            if tc is not None:
                tc.step()
            if pbar is not None:
                pbar.update_absolute(step + 1)
            if preview_callback is not None:
                x0_v, _ = wrapper.unpack(x0)
                x_v, _ = wrapper.unpack(x)
                preview_callback(step, x0_v, x_v, total)

        samp = comfy.samplers.KSAMPLER(_h3_native_sampler)
        out = samp.sample(
            wrapper, sigmas,
            extra_args={"model_options": {}, "seed": seed},
            callback=callback,
            noise=noise,
            latent_image=latent_packed,
            denoise_mask=None,
            disable_pbar=disable_pbar,
        )
        out_v, out_a = wrapper.unpack(out)

        peak = (torch.cuda.max_memory_allocated() - vram_before) / 2 ** 20 \
            if torch.cuda.is_available() else 0.0
        return H3SampleResult(
            video=out_v.to(latent.video.dtype),
            audio=out_a.to(latent.audio.dtype),
            steps=total_steps,
            sigmas=sigmas.cpu(),
            swap_hits=(mgr.swap_hits - base_hits) if mgr is not None else 0,
            swap_loads=(mgr.swap_loads - base_loads) if mgr is not None else 0,
            peak_vram_mb=peak,
        )
    finally:
        if tc is not None:
            tc.detach()
        if reader is not None:
            reader.close()
            if model is not None:
                model._adaln_bake_fallback = None
        handle.unload()
        from ..models.vae import unload_all_vaes
        from ..utils.encoder_use import unload_all_encoders
        unload_all_vaes()
        unload_all_encoders()
        collect_garbage(aggressive=True)
