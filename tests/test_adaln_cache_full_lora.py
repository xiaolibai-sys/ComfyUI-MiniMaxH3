"""AdaLN pre-bake with full-model original AdaLN LoRA fold."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, r"D:\ComfyUI-installs\ComfyUI\ComfyUI")

from pkg_loader import load as _load_h3rt
_load_h3rt()

import torch
import zlib
from safetensors.torch import save_file

from h3rt.models.model import MiniMaxH3Model
from h3rt.models.lora import parse_lora_h3
from h3_test_utils import h3_sample
from h3rt.utils.config import MiniMaxH3DiTConfig
from h3rt.utils.types import RuntimeOptions
from h3rt.utils.lifecycle import load_model_handle
from h3rt.utils.types import (
    AVLatent,
    H3BlockSwap,
    H3Conditioning,
    RuntimeOptions,
    TextConditioning,
)

torch.manual_seed(19)

cfg = MiniMaxH3DiTConfig(
    hidden_size=64, num_layers=3, token_refiner_num_layers=1,
    num_attention_heads=4, attention_head_dim=16, ffn_hidden_size=128,
    latents_dim=4, audio_latents_dim=8, patch_size=(1, 2, 2), text_dim=16,
    timestep_input_dim=16, time_embed_hidden_size=64, time_embed_dim=8,
    rope_inv_freq_len=2, norm_eps=1e-5, qk_norm_eps=1e-5,
    final_norm_eps=1e-5, adaln_curve_grid=None,
)

with torch.device("meta"):
    ref = MiniMaxH3Model(cfg, dtype=torch.float32)
ckpt_sd = {}
for pname, p in ref.named_parameters():
    g = torch.Generator().manual_seed(zlib.crc32(pname.encode()) % (2 ** 32))
    ckpt_sd[pname] = torch.randn(p.shape, dtype=torch.float32, generator=g)
for bname, b in ref.named_buffers():
    ckpt_sd[bname] = torch.randn(b.shape, dtype=torch.float32)

tmp = tempfile.mkdtemp(prefix="h3_adaln_cache_full_")
ckpt_path = os.path.join(tmp, "model.safetensors")
lora_path = os.path.join(tmp, "lora.safetensors")
save_file(ckpt_sd, ckpt_path)

blk = ref.blocks[0]
adaln_w = blk.adaln_proj.linear.weight
final_w = ref.final_layer.adaln_proj.linear.weight
lora_sd = {
    "blocks.0.adaln_proj.linear.lora_A.weight": torch.randn(4, 8),
    "blocks.0.adaln_proj.linear.lora_B.weight": torch.randn(adaln_w.shape[0], 4),
    "final_layer.adaln_proj.linear.lora_A.weight": torch.randn(4, 8),
    "final_layer.adaln_proj.linear.lora_B.weight": torch.randn(final_w.shape[0], 4),
}
save_file(lora_sd, lora_path)

device = torch.device("cuda")
swap = H3BlockSwap(enabled=True, block_to_swap=1, prefetch=True,
                   prefetch_count=2, pin_memory=True, disk_workers=2,
                   dtype="float32")
video = torch.randn(1, 4, 2, 16, 16, device=device, dtype=torch.float32)
audio = torch.randn(1, 8, 2, 8, device=device, dtype=torch.float32)
text = torch.randn(1, 8, 64, device=device, dtype=torch.float32)
tags = torch.ones(1, 8, dtype=torch.long, device=device)
cond = H3Conditioning(text=TextConditioning(states=text, tags=tags))
latent = AVLatent(video=video, audio=audio)

handle = load_model_handle(ckpt_path)
handle.loras.add(parse_lora_h3(lora_path, strength=1.0))
injection = RuntimeOptions(swap=swap)

res_eager = h3_sample(
    handle, cond, latent, None, 3, 1.0, "euler",
    12.0, 1.0, 42, injection, disable_pbar=True)
res_cached = h3_sample(
    handle, cond, latent, None, 3, 1.0, "euler",
    12.0, 1.0, 42, injection, disable_pbar=True, use_adaln_cache=True)


def _rel(a, b):
    a, b = a.float(), b.float()
    return (a - b).abs().max().item() / max(1e-6, a.abs().max().item())


rv = _rel(res_eager.video, res_cached.video)
ra = _rel(res_eager.audio, res_cached.audio)
print("rel video", rv, "audio", ra, flush=True)
assert rv < 1e-4
assert ra < 1e-4
handle.unload()
print("FULL MODEL + ORIGINAL ADALN LORA ADALN CACHE OK", flush=True)
