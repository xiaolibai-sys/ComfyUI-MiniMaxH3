"""KSampler equivalence test.

The packed-AV k-diffusion path (nodes/h3_sampling.py, official comfy.samplers
loop) must reproduce the old hand-written euler loop exactly (verified in
float32: same noise draw order, same flow_sigmas grid, same velocity).

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
from h3rt.models.model import MiniMaxH3Model, flow_sigmas
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


# ---- reference: old hand-written euler loop -------------------------------
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
    x_a = x_a + dt * v_a
handle.unload()

# ---- new path: official k-diffusion loop ----------------------------------
from h3rt.nodes.h3_sampling import h3_sample

result = h3_sample(handle, cond, latent, None, STEPS, 1.0, "euler",
                   SHIFT, 1.0, SEED, InjectionContext.build(block_swap_args=swap),
                   disable_pbar=True)
rel_v = _rel(x_v, result.video)
rel_a = _rel(x_a, result.audio)
print(f"euler packed-vs-manual rel_err: video={rel_v:.6f} audio={rel_a:.6f}")
assert rel_v < 1e-4 and rel_a < 1e-4, "euler path diverged from the manual loop"

res_cache = h3_sample(handle, cond, latent, None, STEPS, 1.0, "euler",
                      SHIFT, 1.0, SEED, InjectionContext.build(block_swap_args=swap),
                      disable_pbar=True, use_adaln_cache=True)
rel_cache_v = _rel(res_cache.video, result.video)
rel_cache_a = _rel(res_cache.audio, result.audio)
print(f"euler adaln-cache rel_err: video={rel_cache_v:.6f} audio={rel_cache_a:.6f}")
assert rel_cache_v < 1e-4 and rel_cache_a < 1e-4, "AdaLN cache diverged from eager path"

# ---- ancestral sampler (CONST + model_patcher shim path) -------------------
res_a = h3_sample(handle, cond, latent, None, STEPS, 1.0, "euler_ancestral",
                  SHIFT, 1.0, SEED, InjectionContext.build(block_swap_args=swap),
                  disable_pbar=True)
assert torch.isfinite(res_a.video.float()).all() and torch.isfinite(res_a.audio.float()).all()
print("euler_ancestral smoke OK (finite)")

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
