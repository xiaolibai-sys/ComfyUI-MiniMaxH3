"""Native MiniMax H3 dual-schedule samplers.

Kept as a compatibility fallback for older ComfyUI builds that do not expose
the packed AV sampler path used by ``SamplerRunner``.
"""

from __future__ import annotations

import math

import torch
from comfy.k_diffusion.sampling import (
    default_noise_sampler,
    get_ancestral_step,
    to_d,
)

from ..models.model import time_shift_sigma, time_shift_slope

SHIFT_V = 12.0
SHIFT_A = 3.0


def _h3_native_sampler(model, x, sigmas, extra_args=None, callback=None,
                       disable=None, **kwargs):
    """Native MiniMax-H3 AV dual-schedule integrator."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
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


def _probe_nv_shifts(model, nv=None, shift_v=SHIFT_V, shift_a=SHIFT_A):
    if nv is None:
        nv = getattr(model, "_n_video", None)
        if nv is None:
            nv = getattr(getattr(model, "inner_model", None), "_n_video", None)
    probe = model
    for _ in range(2):
        shift_v = float(getattr(probe, "_shift_video", shift_v))
        shift_a = float(getattr(probe, "_shift_audio", shift_a))
        probe = getattr(probe, "inner_model", None)
        if probe is None:
            break
    return nv, shift_v, shift_a


def _audio_sigmas(sigmas, shift_v, shift_a):
    return time_shift_sigma(
        torch.as_tensor(sigmas, dtype=torch.float32, device=sigmas.device),
        shift_v,
        shift_a,
    )


def _slope(sigma, shift_v, shift_a):
    return time_shift_slope(max(float(sigma), 1e-6), shift_v, shift_a)


def _split(x, nv):
    if nv is None:
        return x, torch.zeros_like(x)
    return x[..., :nv], x[..., nv:]


def _dual_euler(model, x, sigmas, extra_args=None, callback=None,
                disable=None, **kwargs):
    extra_args = {} if extra_args is None else extra_args
    nv, shift_v, shift_a = _probe_nv_shifts(model)
    audio_sigmas = _audio_sigmas(sigmas, shift_v, shift_a)
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        sv, sv_next = float(sigmas[i]), float(sigmas[i + 1])
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        d = to_d(x, sigmas[i], denoised)
        xv, xa = _split(x, nv)
        dv, da = _split(d, nv)
        xv = xv + (sv_next - sv) * dv
        sa = float(audio_sigmas[i])
        sa_next = float(audio_sigmas[i + 1])
        xa = xa + (sa_next - sa) * (da / _slope(sv, shift_v, shift_a))
        x = torch.cat([xv, xa], dim=-1) if nv is not None else xv
        if callback is not None:
            callback({"i": i, "denoised": denoised, "x": x,
                      "sigma": sigmas[i], "sigma_hat": sigmas[i]})
    return x


def _dual_euler_ancestral(model, x, sigmas, extra_args=None, callback=None,
                          disable=None, eta=1.0, s_noise=1.0, **kwargs):
    extra_args = {} if extra_args is None else extra_args
    nv, shift_v, shift_a = _probe_nv_shifts(model)
    audio_sigmas = _audio_sigmas(sigmas, shift_v, shift_a)
    noise_sampler = default_noise_sampler(x, seed=extra_args.get("seed"))
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        sv, sv_next = float(sigmas[i]), float(sigmas[i + 1])
        sa, sa_next = float(audio_sigmas[i]), float(audio_sigmas[i + 1])
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        d = to_d(x, sigmas[i], denoised)
        xv, xa = _split(x, nv)
        dv, da = _split(d, nv)
        sv_down, sv_up = get_ancestral_step(sv, sv_next, eta)
        sa_down, sa_up = get_ancestral_step(sa, sa_next, eta)
        slope = _slope(sv, shift_v, shift_a)
        da_audio = da / slope
        if callback is not None:
            callback({"i": i, "denoised": denoised, "x": x,
                      "sigma": sigmas[i], "sigma_hat": sigmas[i]})
        noise = noise_sampler(sigmas[i], sigmas[i + 1])
        noise_v, noise_a = _split(noise, nv)
        if sv_down == 0:
            denoised_v, _ = _split(denoised, nv)
            xv = denoised_v
        else:
            xv = xv + (sv_down - sv) * dv + noise_v * s_noise * sv_up
        if sa_down == 0:
            xa = xa - sa * da_audio
        else:
            xa = xa + (sa_down - sa) * da_audio + noise_a * s_noise * sa_up
        x = torch.cat([xv, xa], dim=-1) if nv is not None else xv
    return x


def _dual_heun(model, x, sigmas, extra_args=None, callback=None,
               disable=None, **kwargs):
    extra_args = {} if extra_args is None else extra_args
    nv, shift_v, shift_a = _probe_nv_shifts(model)
    audio_sigmas = _audio_sigmas(sigmas, shift_v, shift_a)
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        sv, sv_next = float(sigmas[i]), float(sigmas[i + 1])
        sa, sa_next = float(audio_sigmas[i]), float(audio_sigmas[i + 1])
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        d1 = to_d(x, sigmas[i], denoised)
        xv, xa = _split(x, nv)
        dv1, da1 = _split(d1, nv)
        slope1 = _slope(sv, shift_v, shift_a)
        x2v = xv + (sv_next - sv) * dv1
        x2a = xa + (sa_next - sa) * (da1 / slope1)
        x2 = torch.cat([x2v, x2a], dim=-1) if nv is not None else x2v
        if callback is not None:
            callback({"i": i, "denoised": denoised, "x": x,
                      "sigma": sigmas[i], "sigma_hat": sigmas[i]})
        if sv_next == 0:
            xv = xv + (sv_next - sv) * dv1
            xa = xa + (sa_next - sa) * (da1 / slope1)
            x = torch.cat([xv, xa], dim=-1) if nv is not None else xv
            continue
        denoised2 = model(x2, sigmas[i + 1] * s_in, **extra_args)
        d2 = to_d(x2, sigmas[i + 1], denoised2)
        dv2, da2 = _split(d2, nv)
        slope2 = _slope(sv_next, shift_v, shift_a)
        xv = xv + (sv_next - sv) * (dv1 + dv2) / 2
        xa = xa + (sa_next - sa) * (da1 / slope1 + da2 / slope2) / 2
        x = torch.cat([xv, xa], dim=-1) if nv is not None else xv
    return x


def _dual_dpmpp_2m(model, x, sigmas, extra_args=None, callback=None,
                   disable=None, **kwargs):
    extra_args = {} if extra_args is None else extra_args
    nv, shift_v, shift_a = _probe_nv_shifts(model)
    audio_sigmas = _audio_sigmas(sigmas, shift_v, shift_a)
    s_in = x.new_ones([x.shape[0]])
    old_v = None
    old_a = None
    prev_t_v = None
    prev_t_a = None

    def t_v(sigma):
        return math.log(1.0 / max(float(sigma), 1e-12))

    def t_a(sigma):
        return math.log(1.0 / max(float(sigma), 1e-12))

    for i in range(len(sigmas) - 1):
        sv, sv_next = float(sigmas[i]), float(sigmas[i + 1])
        sa, sa_next = float(audio_sigmas[i]), float(audio_sigmas[i + 1])
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"i": i, "denoised": denoised, "x": x,
                      "sigma": sigmas[i], "sigma_hat": sigmas[i]})
        xv, xa = _split(x, nv)
        dv, da = _split(to_d(x, sigmas[i], denoised), nv)
        denoised_v, _ = _split(denoised, nv)
        slope = _slope(sv, shift_v, shift_a)
        audio_denoised = xa - sa * (da / slope)

        tv, tv_next = t_v(sv), t_v(sv_next)
        ta, ta_next = t_a(sa), t_a(sa_next)
        hv, ha = tv_next - tv, ta_next - ta

        if sv_next == 0:
            xv = denoised_v
        elif old_v is None:
            xv = (sv_next / sv) * xv - math.expm1(-hv) * denoised_v
        else:
            rv = (tv - prev_t_v) / hv
            denoised_d = (1 + 1 / (2 * rv)) * denoised_v - \
                (1 / (2 * rv)) * old_v
            xv = (sv_next / sv) * xv - math.expm1(-hv) * denoised_d

        if sa_next == 0:
            xa = audio_denoised
        elif old_a is None:
            xa = (sa_next / sa) * xa - math.expm1(-ha) * audio_denoised
        else:
            ra = (ta - prev_t_a) / ha
            denoised_d = (1 + 1 / (2 * ra)) * audio_denoised - \
                (1 / (2 * ra)) * old_a
            xa = (sa_next / sa) * xa - math.expm1(-ha) * denoised_d

        old_v = denoised_v
        old_a = audio_denoised
        prev_t_v = tv
        prev_t_a = ta
        x = torch.cat([xv, xa], dim=-1) if nv is not None else xv
    return x


H3_SAMPLER_FUNCTIONS = {
    "euler": _dual_euler,
    "euler_ancestral": _dual_euler_ancestral,
    "heun": _dual_heun,
    "dpmpp_2m": _dual_dpmpp_2m,
}


def _h3_sampler(sampler_name: str):
    return H3_SAMPLER_FUNCTIONS.get(sampler_name, _dual_euler)


__all__ = [
    "_h3_sampler",
    "H3_SAMPLER_FUNCTIONS",
]
