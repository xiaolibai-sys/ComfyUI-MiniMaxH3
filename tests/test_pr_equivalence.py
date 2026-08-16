"""Output-equivalence test: our DiT vs the ComfyUI PR model (#15224).

Builds both models with an identical tiny config and identical seeded random
weights, runs one velocity step with the same latents/text/payload, and
compares the video/audio outputs (the PR model returns the same flow-velocity
tuple ``[-video_out, -audio_out]``).

Run with the ComfyUI venv python (ComfyUI root auto-added for ``comfy.*``):
    python tests/test_pr_equivalence.py
"""

import os
import sys

_COMFY_ROOT = r"D:\ComfyUI-installs\ComfyUI\ComfyUI"
# ComfyUI root first, then our package root so `utils`/`models` resolve to ours
sys.path.insert(0, _COMFY_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pkg_loader import load as _load_h3rt
_load_h3rt()

import torch

from h3rt.utils.config import MiniMaxH3DiTConfig
from h3rt.models.model import MiniMaxH3Model as OurModel

torch.manual_seed(7)

cfg = MiniMaxH3DiTConfig(
    hidden_size=64, num_layers=3, token_refiner_num_layers=1,
    num_attention_heads=4, attention_head_dim=16, ffn_hidden_size=128,
    latents_dim=4, audio_latents_dim=8, patch_size=(1, 2, 2), text_dim=16,
    timestep_input_dim=16, time_embed_hidden_size=64, time_embed_dim=32,
    rope_inv_freq_len=2, norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5,
)

import comfy.ops  # noqa: E402
import comfy.ldm.minimax.model as _PR_MODULE  # noqa: E402
# this env has no xformers; route optimized_attention to SDPA (pytorch backend)
from comfy.ldm.modules.attention import attention_pytorch  # noqa: E402
_PR_MODULE.optimized_attention = attention_pytorch
from comfy.ldm.minimax.model import MiniMaxH3Model as PRModel  # noqa: E402

pr = PRModel(operations=comfy.ops.disable_weight_init, dtype=torch.bfloat16, 
    hidden_size=cfg.hidden_size, num_layers=cfg.num_layers,
    token_refiner_num_layers=cfg.token_refiner_num_layers,
    num_attention_heads=cfg.num_attention_heads,
    attention_head_dim=cfg.attention_head_dim,
    ffn_hidden_size=cfg.ffn_hidden_size, latents_dim=cfg.latents_dim,
    audio_latents_dim=cfg.audio_latents_dim, patch_size=cfg.patch_size,
    text_dim=cfg.text_dim, timestep_input_dim=cfg.timestep_input_dim,
    time_embed_hidden_size=cfg.time_embed_hidden_size,
    time_embed_dim=cfg.time_embed_dim, rope_inv_freq_len=cfg.rope_inv_freq_len,
    norm_eps=cfg.norm_eps, qk_norm_eps=cfg.qk_norm_eps,
    final_norm_eps=cfg.final_norm_eps,
)

with torch.device("meta"):
    ours = OurModel(cfg, dtype=torch.bfloat16)

device = torch.device("cuda")

# identical weights
state = {}
for pname, p in pr.named_parameters():
    state[pname] = torch.randn(p.shape, dtype=torch.bfloat16)
for pname, p in pr.named_parameters():
    p.data.copy_(state[pname])
for bname, b in pr.named_buffers():
    if b.dtype == torch.float32:
        b.data.copy_(torch.randn(b.shape, dtype=torch.float32))
    else:
        b.data.copy_(torch.randn(b.shape, dtype=torch.bfloat16))

for pname, p in ours.named_parameters():
    mod = ours.get_submodule(pname.rsplit(".", 1)[0]) if "." in pname else ours
    leaf = pname.rsplit(".", 1)[1] if "." in pname else pname
    mod._parameters[leaf] = torch.nn.Parameter(
        state[pname].to(device, p.dtype), requires_grad=False)
pr_bufs = {n: b for n, b in pr.named_buffers()}
for bname, b in ours.named_buffers():
    mod = ours.get_submodule(bname.rsplit(".", 1)[0]) if "." in bname else ours
    leaf = bname.rsplit(".", 1)[1] if "." in bname else bname
    mod._buffers[leaf] = pr_bufs[bname].to(device)

pr.requires_grad_(False)
pr.to(device)
ours.to(device)

T, H, W, AT = 2, 16, 16, 8
video = torch.randn(1, 4, T, H, W, device=device, dtype=torch.bfloat16)
audio = torch.randn(1, 8, 2, AT, device=device, dtype=torch.bfloat16)
text = torch.randn(1, 8, 64, device=device, dtype=torch.bfloat16)  # pre-refined (hidden)
tags = torch.ones(1, 8, dtype=torch.long, device=device)
payload = {"text_token_tags": tags}

sigma = 0.5
with torch.inference_mode():
    pr_out = pr([video, audio], torch.tensor([sigma * 1000.0], device=device),
                text, {}, payload)
    our_out = ours.velocity(video, audio, sigma, text, payload)

pv, pa = pr_out[0].float(), pr_out[1].float()
ov, oa = our_out[0].float(), our_out[1].float()

def rel_err(a, b):
    return (a - b).abs().max().item() / max(1e-6, b.abs().max().item())

rv, ra = rel_err(ov, pv), rel_err(oa, pa)
print(f"video velocity rel_err={rv:.6f}  audio velocity rel_err={ra:.6f}")
print(f"video shapes match: {tuple(ov.shape) == tuple(pv.shape)}")

assert tuple(ov.shape) == tuple(pv.shape), "video output shape mismatch"
assert tuple(oa.shape) == tuple(pa.shape), "audio output shape mismatch"
# eager RoPE vs triton kernel, SDPA vs optimized_attention -> expect ~1e-2
assert rv < 2e-2, f"video velocity diverged: {rv}"
assert ra < 2e-2, f"audio velocity diverged: {ra}"
print("PR EQUIVALENCE TEST OK")
