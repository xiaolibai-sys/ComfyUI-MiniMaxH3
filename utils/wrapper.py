"""k-diffusion wrapper for the packed MiniMax H3 AV latent."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Optional

import torch
import comfy.model_management
import comfy.model_sampling
import comfy.samplers

from ..models.model import flow_sigmas, time_shift_sigma
from .types import ForwardRequest

SHIFT_V = 12.0
SHIFT_A = 3.0


class _H3ModelSampling(comfy.model_sampling.CONST,
                       comfy.model_sampling.ModelSamplingDiscreteFlow):
    """Flow-matching sigma behaviour (CONST) + discrete-flow sigma range."""

    def __init__(self):
        super().__init__()
        self.set_parameters(shift=SHIFT_V, timesteps=1000, multiplier=1000)


class _InnerShim:
    """What ``KSAMPLER`` / k-diffusion reach for via ``model_wrap.inner_model``."""

    def __init__(self):
        self.model_sampling = _H3ModelSampling()


class _PatcherShim:
    """Minimal ModelPatcher stand-in used by SDE-family samplers."""

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
        self._audio_scale = float(shift_video / shift_audio)

    def __call__(self, x, sigma, **kwargs):
        self.captured.append(float(sigma.flatten()[0]))
        denoised = torch.zeros_like(x)
        post_fns = kwargs.get("model_options", {}).get(
            "sampler_post_cfg_function", [])
        if not post_fns:
            return denoised
        for fn in post_fns:
            args = {
                "denoised": denoised,
                "cond": None,
                "uncond": None,
                "cond_scale": self.cfg,
                "model": self,
                "uncond_denoised": denoised,
                "cond_denoised": denoised,
                "sigma": sigma,
                "model_options": kwargs.get("model_options", {}),
                "input": x,
            }
            denoised = fn(args)
        return denoised


def h3_sigmas(scheduler_name: str, steps: int, shift_video: float):
    """Build an H3 flow sigma grid using the selected scheduler."""
    if scheduler_name == "flow_uniform":
        return flow_sigmas(steps, shift_video)
    return comfy.samplers.calculate_sigmas(
        _H3ModelSampling(), scheduler_name, steps)


class H3ModelWrapper:
    """k-diffusion entry point: packed noisy AV latent -> packed denoised prediction.

    Runs one forward per step (cond only, cfg == 1) or two (cond + uncond)
    and combines velocities with classifier-free guidance before converting
    to the denoised estimate ``x0 = x - sigma * v``.
    """

    def __init__(self, request: ForwardRequest):
        self.model = request.model
        self.cfg = float(request.cfg)
        self._video_shape = tuple(request.video_shape)
        self._audio_shape = tuple(request.audio_shape)
        self._n_video = math.prod(self._video_shape[1:])
        self._text_states = request.positive_text.states
        self._payload = request.positive_payload
        self._neg_text_states = (
            request.negative_text.states
            if request.negative_text is not None
            else None
        )
        self._neg_payload = request.negative_payload
        self._shift_video = float(request.shift_video)
        self._shift_audio = float(request.shift_audio)
        self._audio_scale = float(request.shift_video / request.shift_audio)

        self.inner_model = _InnerShim()
        self.model_patcher = _PatcherShim(
            request.model, self.inner_model.model_sampling)

    # -- packed AV latent -------------------------------------------------

    def pack(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            video.reshape(video.shape[0], -1),
            audio.reshape(audio.shape[0], -1) * self._audio_scale,
        ], dim=1)

    def unpack(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        video = x[:, :self._n_video].reshape(self._video_shape)
        audio = (
            x[:, self._n_video:].reshape(self._audio_shape)
            / self._audio_scale
        )
        return video, audio

    # -- k-diffusion interface --------------------------------------------

    def _denoised(self, x_v, x_a_scaled, s, text_states, payload):
        sigma_a = float(time_shift_sigma(
            s, self._shift_video, self._shift_audio))
        audio_x = x_a_scaled * (sigma_a / max(s, 1e-6))
        audio_x = audio_x.reshape(self._audio_shape)
        v_v, v_a = self.model.velocity(
            x_v, audio_x, s, text_states, payload,
            shift_video=self._shift_video, shift_audio=self._shift_audio)
        x0_v = (
            x_v.reshape(x_v.shape[0], -1)
            - s * v_v.reshape(v_v.shape[0], -1)
        )
        c = self._audio_scale
        out_y = (
            (1.0 - c) * audio_x
            + (1.0 + (c - 1.0) * sigma_a) * v_a
        ).reshape(x_v.shape[0], -1)
        x0_a = x_a_scaled - s * out_y
        return torch.cat([x0_v, x0_a], dim=-1)

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor,
                 **extra_args) -> torch.Tensor:
        comfy.model_management.throw_exception_if_processing_interrupted()
        s = float(sigma.flatten()[0])
        x_v, x_a = self.unpack(x)
        x_a_scaled = x[..., self._n_video:]

        model_options = extra_args.get("model_options", {})
        cond_denoised = self._denoised(
            x_v, x_a_scaled, s, self._text_states, self._payload)
        uncond_denoised = None
        if self._neg_text_states is not None and (
            self.cfg != 1.0 or model_options.get("disable_cfg1_optimization")
        ):
            uncond_denoised = self._denoised(
                x_v, x_a_scaled, s,
                self._neg_text_states, self._neg_payload)

        if uncond_denoised is None:
            denoised = cond_denoised
        else:
            denoised = uncond_denoised + self.cfg * (
                cond_denoised - uncond_denoised)

        post_fns = model_options.get("sampler_post_cfg_function", [])
        if uncond_denoised is None and post_fns:
            uncond_denoised = cond_denoised
        for fn in post_fns:
            args = {
                "denoised": denoised,
                "cond": cond_denoised,
                "uncond": uncond_denoised,
                "cond_scale": self.cfg,
                "model": self.model,
                "uncond_denoised": uncond_denoised,
                "cond_denoised": cond_denoised,
                "sigma": sigma,
                "model_options": model_options,
                "input": x,
            }
            denoised = fn(args)
        return denoised


__all__ = [
    "H3ModelWrapper",
    "_H3ModelSampling",
    "_InnerShim",
    "_PatcherShim",
    "_SigmaCaptureModel",
    "h3_sigmas",
]
