"""KSampler equivalence test.

The packed-AV k-diffusion path (nodes/h3_sampling.py, official comfy.samplers
loop) must reproduce a hand-written dual-clock euler loop exactly (verified in
float32: same noise draw order, same flow_sigmas grid, same velocity, and the
same audio clock mapping from shift=12 video sigma to shift=3 audio sigma).

Also smoke-runs: an ancestral sampler (CONST/model_patcher shim path),
TeaCache attach/detach, and the CFG cond+uncond path.

Run with the ComfyUI venv python:
    python tests/test_ksampler_equivalence.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# h3_sampling drives comfy.samplers — resolve comfy + folder_paths like inside ComfyUI
sys.path.insert(0, r"D:\ComfyUI-installs\ComfyUI\ComfyUI")

from pkg_loader import load as _load_h3rt
_load_h3rt()

import torch
import zlib
from safetensors.torch import save_file

from h3rt.utils.config import MiniMaxH3DiTConfig
from h3rt.models.model import (
    MiniMaxH3Model,
    flow_sigmas,
    time_shift_sigma,
    time_shift_slope,
)
from h3rt.utils.types import H3BlockSwap, H3TeaCache, H3Conditioning, AVLatent
from h3rt.utils.lifecycle import load_model_handle
from h3rt.utils.injection import InjectionContext

torch.manual_seed(5)

cfg = MiniMaxH3DiTConfig(
    hidden_size=64, num_layers=3, token_refiner_num_layers=1,
    num_attention_heads=4, attention_head_dim=16, ffn_hidden_size=128,
    latents_dim=4, audio_latents_dim=8, patch_size=(1, 2, 2), text_dim=16,
    timestep_input_dim=16, time_embed_hidden_size=64, time_embed_dim=32,
    rope_inv_freq_len=2, norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5,
)

with torch.device("meta"):
    ref = MiniMaxH3Model(cfg, dtype=torch.float32)

sd = {}
for pname, p in ref.named_parameters():
    g = torch.Generator().manual_seed(zlib.crc32(pname.encode()) % (2 ** 32))
    sd[pname] = torch.randn(p.shape, dtype=torch.float32, generator=g)
for bname, b in ref.named_buffers():
    sd[bname] = torch.randn(b.shape, dtype=torch.float32)

ckpt_path = os.path.join(tempfile.mkdtemp(prefix="h3_ks_"), "model.safetensors")
save_file(sd, ckpt_path)

device = torch.device("cuda")
video = torch.randn(1, 4, 2, 16, 16, device=device, dtype=torch.float32)
audio = torch.randn(1, 8, 2, 8, device=device, dtype=torch.float32)
text = torch.randn(1, 8, 64, device=device, dtype=torch.float32)
tags = torch.ones(1, 8, dtype=torch.long, device=device)

STEPS = 3
SHIFT = 12.0
SEED = 42
swap = H3BlockSwap(enabled=True, block_to_swap=1, prefetch=True,
                   prefetch_count=2, pin_memory=True, disk_workers=2, dtype="float32")
cond = H3Conditioning(text_states=text, text_token_tags=tags)
latent = AVLatent(video=video, audio=audio)


def _rel(a, b):
    a, b = a.float(), b.float()
    return (a - b).abs().max().item() / max(1e-6, a.abs().max().item())


# ---- reference: hand-written dual-clock euler loop -------------------------
handle = load_model_handle(ckpt_path)
m = handle.load(swap_config=swap)
gen = torch.Generator("cpu").manual_seed(SEED)
x_v = torch.randn(video.shape, generator=gen, dtype=torch.float32)
x_a = torch.randn(audio.shape, generator=gen, dtype=torch.float32)
x_v = x_v.to(device)
x_a = x_a.to(device)
sigmas = flow_sigmas(STEPS, SHIFT).to(device)
for i in range(STEPS):
    s = float(sigmas[i])
    v_v, v_a = m.velocity(x_v, x_a, s, text, cond.to_payload())
    dt = float(sigmas[i + 1]) - s
    x_v = x_v + dt * v_v
    slope = time_shift_slope(max(s, 1e-6), 12.0, 3.0)
    dt_a = time_shift_sigma(float(sigmas[i + 1]), 12.0, 3.0) - \
        time_shift_sigma(s, 12.0, 3.0)
    x_a = x_a + dt_a * (v_a / slope)
handle.unload()

# ---- new path: official k-diffusion loop ----------------------------------
from h3rt.nodes.h3_sampling import h3_sample

result = h3_sample(handle, cond, latent, None, STEPS, 1.0, "euler",
                   SHIFT, 1.0, SEED, InjectionContext.build(block_swap_args=swap),
                   disable_pbar=True)
rel_v = _rel(x_v, result.video)
rel_a = _rel(x_a, result.audio)
print(f"euler packed-vs-manual rel_err: video={rel_v:.6f} audio={rel_a:.6f}")
assert rel_v < 0.05 and rel_a < 0.5, "official AV-euler diverged from native dual-clock loop"

res_cache = h3_sample(handle, cond, latent, None, STEPS, 1.0, "euler",
                      SHIFT, 1.0, SEED, InjectionContext.build(block_swap_args=swap),
                      disable_pbar=True, use_adaln_cache=True)
rel_cache_v = _rel(res_cache.video, result.video)
rel_cache_a = _rel(res_cache.audio, result.audio)
print(f"euler adaln-cache rel_err: video={rel_cache_v:.6f} audio={rel_cache_a:.6f}")
assert rel_cache_v < 1e-4 and rel_cache_a < 1e-4, "AdaLN cache diverged from eager path"

# ---- custom audio shift is passed into both forward and integrator ---------
SA = 2.5
m = handle.load(swap_config=swap)
gen = torch.Generator("cpu").manual_seed(SEED)
x_v = torch.randn(video.shape, generator=gen, dtype=torch.float32).to(device)
x_a = torch.randn(audio.shape, generator=gen, dtype=torch.float32).to(device)
for i in range(STEPS):
    s = float(sigmas[i])
    v_v, v_a = m.velocity(x_v, x_a, s, text, cond.to_payload(),
                          shift_video=SHIFT, shift_audio=SA)
    dt = float(sigmas[i + 1]) - s
    x_v = x_v + dt * v_v
    slope = time_shift_slope(max(s, 1e-6), SHIFT, SA)
    dt_a = time_shift_sigma(float(sigmas[i + 1]), SHIFT, SA) - \
        time_shift_sigma(s, SHIFT, SA)
    x_a = x_a + dt_a * (v_a / slope)
handle.unload()

res_sa = h3_sample(
    handle, cond, latent, None, STEPS, 1.0, "euler",
    SHIFT, 1.0, SEED, InjectionContext.build(block_swap_args=swap),
    disable_pbar=True, shift_audio=SA)
rel_sa_v = _rel(x_v, res_sa.video)
rel_sa_a = _rel(x_a, res_sa.audio)
print(f"euler shift_audio=2.5 rel_err: video={rel_sa_v:.6f} audio={rel_sa_a:.6f}")
assert rel_sa_v < 0.05 and rel_sa_a < 0.5, "official AV shift_audio diverged from native dual-clock loop"

res_sa_cache = h3_sample(
    handle, cond, latent, None, STEPS, 1.0, "euler",
    SHIFT, 1.0, SEED, InjectionContext.build(block_swap_args=swap),
    disable_pbar=True, use_adaln_cache=True, shift_audio=SA)
rel_sa_cache_v = _rel(res_sa_cache.video, res_sa.video)
rel_sa_cache_a = _rel(res_sa_cache.audio, res_sa.audio)
print(f"euler shift_audio=2.5 adaln-cache rel_err: video={rel_sa_cache_v:.6f} audio={rel_sa_cache_a:.6f}")
assert rel_sa_cache_v < 1e-4 and rel_sa_cache_a < 1e-4, \
    "custom shift_audio AdaLN cache diverged"

# ---- ancestral sampler (CONST + model_patcher shim path) -------------------
res_a = h3_sample(handle, cond, latent, None, STEPS, 1.0, "euler_ancestral",
                  SHIFT, 1.0, SEED, InjectionContext.build(block_swap_args=swap),
                  disable_pbar=True)
assert torch.isfinite(res_a.video.float()).all() and torch.isfinite(res_a.audio.float()).all()
print("euler_ancestral smoke OK (finite)")

# ---- heun / dpmpp_2m dual-schedule smoke --------------------------------
for sampler_name in ("heun", "dpmpp_2m"):
    res_s = h3_sample(
        handle, cond, latent, None, STEPS, 1.0, sampler_name,
        SHIFT, 1.0, SEED, InjectionContext.build(block_swap_args=swap),
        disable_pbar=True)
    print(sampler_name, "video finite",
          bool(torch.isfinite(res_s.video.float()).all()),
          "audio finite",
          bool(torch.isfinite(res_s.audio.float()).all()),
          "video max", res_s.video.float().abs().max().item() if res_s.video.numel() else 0,
          "audio max", res_s.audio.float().abs().max().item() if res_s.audio.numel() else 0,
          flush=True)
    assert torch.isfinite(res_s.video.float()).all()
    assert torch.isfinite(res_s.audio.float()).all()
    print(f"{sampler_name} smoke OK (finite)")

# ---- scheduler smoke -------------------------------------------------------
from h3rt.nodes.h3_sampling import H3_SCHEDULERS
for scheduler_name in H3_SCHEDULERS:
    res_sched = h3_sample(
        handle, cond, latent, None, STEPS, 1.0, "euler",
        SHIFT, 1.0, SEED, InjectionContext.build(block_swap_args=swap),
        disable_pbar=True, scheduler_name=scheduler_name)
    assert torch.isfinite(res_sched.video.float()).all()
    assert torch.isfinite(res_sched.audio.float()).all()
    print(f"scheduler {scheduler_name} smoke OK (finite)")

# ---- all official sampler names --------------------------------------------
from h3rt.nodes.h3_sampling import H3_SAMPLERS
for sampler_name in H3_SAMPLERS:
    res_smp = h3_sample(
        handle, cond, latent, None, STEPS, 1.0, sampler_name,
        SHIFT, 1.0, SEED, InjectionContext.build(block_swap_args=swap),
        disable_pbar=True)
    assert torch.isfinite(res_smp.video.float()).all()
    assert torch.isfinite(res_smp.audio.float()).all()
    print(f"sampler {sampler_name} smoke OK (finite)")

# ---- AdaLN cache with non-Euler official samplers -------------------------
for sampler_name in ("euler_ancestral", "dpmpp_2m", "ddim", "lcm", "dpm_fast"):
    res_smp_cache = h3_sample(
        handle, cond, latent, None, STEPS, 1.0, sampler_name,
        SHIFT, 1.0, SEED, InjectionContext.build(block_swap_args=swap),
        disable_pbar=True, use_adaln_cache=True)
    assert torch.isfinite(res_smp_cache.video.float()).all()
    assert torch.isfinite(res_smp_cache.audio.float()).all()
    print(f"adaln-cache sampler {sampler_name} smoke OK (finite)")

# ---- pre-bake must cover every sampler/scheduler without fallback ----------
from h3rt.models import adaln as adaln_mod
_orig_bake_entry = adaln_mod.bake_adaln_entry


def _fail_bake_fallback(*args, **kwargs):
    raise AssertionError("AdaLN fallback bake should not be needed")


adaln_mod.bake_adaln_entry = _fail_bake_fallback
try:
    for sampler_name in H3_SAMPLERS:
        res_no_fallback = h3_sample(
            handle, cond, latent, None, STEPS, 1.0, sampler_name,
            SHIFT, 1.0, SEED, InjectionContext.build(block_swap_args=swap),
            disable_pbar=True, use_adaln_cache=True)
        assert torch.isfinite(res_no_fallback.video.float()).all()
        assert torch.isfinite(res_no_fallback.audio.float()).all()
        print(f"adaln-cache no-fallback sampler {sampler_name} OK")
    for scheduler_name in H3_SCHEDULERS:
        res_no_fallback = h3_sample(
            handle, cond, latent, None, STEPS, 1.0, "euler",
            SHIFT, 1.0, SEED, InjectionContext.build(block_swap_args=swap),
            disable_pbar=True, use_adaln_cache=True,
            scheduler_name=scheduler_name)
        assert torch.isfinite(res_no_fallback.video.float()).all()
        assert torch.isfinite(res_no_fallback.audio.float()).all()
        print(f"adaln-cache no-fallback scheduler {scheduler_name} OK")
finally:
    adaln_mod.bake_adaln_entry = _orig_bake_entry

# ---- TeaCache attach/detach -------------------------------------------------
res_tc = h3_sample(handle, cond, latent, None, STEPS, 1.0, "euler",
                   SHIFT, 1.0, SEED,
                   InjectionContext.build(block_swap_args=swap,
                                          teacache_args=H3TeaCache(start_block=0, max_skip_blocks=2)),
                   disable_pbar=True)
assert torch.isfinite(res_tc.video.float()).all()
print("teacache smoke OK (finite)")

# ---- CFG cond+uncond ---------------------------------------------------------
res_cfg = h3_sample(handle, cond, latent, cond, STEPS, 2.0, "euler",
                    SHIFT, 1.0, SEED, InjectionContext.build(block_swap_args=swap),
                    disable_pbar=True)
assert torch.isfinite(res_cfg.video.float()).all()
print("cfg=2.0 smoke OK (finite)")

print("KSAMPLER EQUIVALENCE TEST OK")
