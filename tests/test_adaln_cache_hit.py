"""Verify AdaLN pre-bake hits for every sampler/scheduler without fallback."""

import os
import sys
import tempfile
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, r"D:\ComfyUI-installs\ComfyUI\ComfyUI")

from pkg_loader import load as _load_h3rt
_load_h3rt()

import torch
import zlib
from safetensors.torch import save_file

from h3rt.models.model import MiniMaxH3Model
from h3rt.models import adaln
from h3rt.nodes.h3_sampling import (
    ADALN_PREBAKE_UNSUPPORTED,
    H3_SAMPLERS,
    H3_SCHEDULERS,
    h3_sample,
)
from h3rt.utils.config import MiniMaxH3DiTConfig
from h3rt.utils.injection import InjectionContext
from h3rt.utils.lifecycle import load_model_handle
from h3rt.utils.types import AVLatent, H3BlockSwap, H3Conditioning

torch.manual_seed(23)

cfg = MiniMaxH3DiTConfig(
    hidden_size=64, num_layers=3, token_refiner_num_layers=1,
    num_attention_heads=4, attention_head_dim=16, ffn_hidden_size=128,
    latents_dim=4, audio_latents_dim=8, patch_size=(1, 2, 2), text_dim=16,
    timestep_input_dim=16, time_embed_hidden_size=64, time_embed_dim=8,
    rope_inv_freq_len=2, norm_eps=1e-5, qk_norm_eps=1e-5,
    final_norm_eps=1e-5,
)

with torch.device("meta"):
    ref = MiniMaxH3Model(cfg, dtype=torch.float32)
sd = {}
for pname, p in ref.named_parameters():
    g = torch.Generator().manual_seed(zlib.crc32(pname.encode()) % (2 ** 32))
    sd[pname] = torch.randn(p.shape, dtype=torch.float32, generator=g)
for bname, b in ref.named_buffers():
    sd[bname] = torch.randn(b.shape, dtype=torch.float32)

ckpt_path = os.path.join(tempfile.mkdtemp(prefix="h3_cache_hit_"), "model.safetensors")
save_file(sd, ckpt_path)

device = torch.device("cuda")
video = torch.randn(1, 4, 2, 16, 16, device=device, dtype=torch.float32)
audio = torch.randn(1, 8, 2, 8, device=device, dtype=torch.float32)
text = torch.randn(1, 8, 16, device=device, dtype=torch.float32)
tags = torch.ones(1, 8, dtype=torch.long, device=device)
cond = H3Conditioning(text_states=text, text_token_tags=tags)
latent = AVLatent(video=video, audio=audio)
handle = load_model_handle(ckpt_path)
injection = InjectionContext.build(block_swap_args=H3BlockSwap(
    enabled=True, block_to_swap=1, prefetch=True, prefetch_count=2,
    pin_memory=True, disk_workers=2, dtype="float32"))

fallback_calls = []
orig_bake = adaln.bake_adaln_entry


def tracked(*args, **kwargs):
    fallback_calls.append((args, kwargs))
    return orig_bake(*args, **kwargs)


with mock.patch.object(adaln, "bake_adaln_entry", side_effect=tracked):
    for sampler_name in H3_SAMPLERS:
        fallback_calls.clear()
        res = h3_sample(
            handle, cond, latent, None, 3, 1.0, sampler_name,
            12.0, 1.0, 42, injection, disable_pbar=True,
            use_adaln_cache=True)
        assert torch.isfinite(res.video.float()).all()
        assert fallback_calls == [], f"{sampler_name} triggered fallback"
        if sampler_name in ADALN_PREBAKE_UNSUPPORTED:
            print(f"sampler {sampler_name}: eager AdaLN OK", flush=True)
        else:
            print(f"sampler {sampler_name}: cache hit OK", flush=True)

    for scheduler_name in H3_SCHEDULERS:
        fallback_calls.clear()
        res = h3_sample(
            handle, cond, latent, None, 3, 1.0, "euler",
            12.0, 1.0, 42, injection, disable_pbar=True,
            use_adaln_cache=True, scheduler_name=scheduler_name)
        assert torch.isfinite(res.video.float()).all()
        assert fallback_calls == [], f"{scheduler_name} triggered fallback"
        print(f"scheduler {scheduler_name}: cache hit OK", flush=True)

handle.unload()
print("ADALN CACHE HIT TEST OK", flush=True)
