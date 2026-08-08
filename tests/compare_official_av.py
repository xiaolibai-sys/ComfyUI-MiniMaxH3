"""Compare official ComfyUI MiniMax H3 AV forward with local official_av mode."""

import os
import sys

_NODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"D:\ComfyUI-installs\ComfyUI\ComfyUI")
sys.path.insert(0, _NODE_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pkg_loader import load as _load_h3rt
_load_h3rt()

import torch
import comfy.ops
import comfy.ldm.minimax.model as _PR_MODULE
from comfy.ldm.modules.attention import attention_pytorch
_PR_MODULE.optimized_attention = attention_pytorch
from comfy.ldm.minimax.model import MiniMaxH3Model as OfficialModel
from comfy.ldm.minimax.model import PackedLayout as OfficialPackedLayout

from h3rt.models.model import MiniMaxH3Model as OurModel
from h3rt.utils.config import MiniMaxH3DiTConfig

torch.manual_seed(7)

cfg = MiniMaxH3DiTConfig(
    hidden_size=64, num_layers=3, token_refiner_num_layers=1,
    num_attention_heads=4, attention_head_dim=16, ffn_hidden_size=128,
    latents_dim=4, audio_latents_dim=8, patch_size=(1, 2, 2), text_dim=16,
    timestep_input_dim=16, time_embed_hidden_size=64, time_embed_dim=32,
    rope_inv_freq_len=2, norm_eps=1e-5, qk_norm_eps=1e-5,
    final_norm_eps=1e-5,
)

official = OfficialModel(
    operations=comfy.ops.disable_weight_init, dtype=torch.float32,
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
    ours = OurModel(cfg, dtype=torch.float32)

state = {}
for pname, p in official.named_parameters():
    state[pname] = torch.randn(p.shape, dtype=torch.float32)
for pname, p in official.named_parameters():
    p.data.copy_(state[pname])
for bname, b in official.named_buffers():
    if b.dtype == torch.float32:
        b.data.copy_(torch.randn(b.shape, dtype=torch.float32))
    else:
        b.data.copy_(torch.randn(b.shape, dtype=torch.bfloat16))
for pname, p in ours.named_parameters():
    mod = ours.get_submodule(pname.rsplit(".", 1)[0]) if "." in pname else ours
    leaf = pname.rsplit(".", 1)[1] if "." in pname else pname
    mod._parameters[leaf] = torch.nn.Parameter(state[pname], requires_grad=False)
official_bufs = {n: b for n, b in official.named_buffers()}
for bname, b in ours.named_buffers():
    mod = ours.get_submodule(bname.rsplit(".", 1)[0]) if "." in bname else ours
    leaf = bname.rsplit(".", 1)[1] if "." in bname else bname
    mod._buffers[leaf] = official_bufs[bname]

device = torch.device("cuda")
official.to(device).requires_grad_(False)
ours.to(device).requires_grad_(False)

video = torch.randn(1, 4, 2, 16, 16, device=device, dtype=torch.float32)
audio = torch.randn(1, 8, 2, 8, device=device, dtype=torch.float32)
text = torch.randn(1, 8, 16, device=device, dtype=torch.float32)
sigma = 0.5
shift_v, shift_a = 12.0, 3.0
scale = shift_v / shift_a

audio_scaled = audio * scale
from h3rt.models.model import time_shift_sigma, time_shift_slope
sigma_a = float(time_shift_sigma(sigma, shift_v, shift_a))
audio_internal = audio_scaled * (sigma_a / sigma)

payload = {
    "audio_scale": scale,
    "layout": OfficialPackedLayout(
        text.shape[1], video.shape[2], video.shape[3], video.shape[4],
        audio.shape[-1]),
}
timestep = torch.tensor([sigma * 1000], device=device)
official_out = official(
    [video, audio_scaled],
    timestep,
    text,
    transformer_options={
        "minimax_h3_sigma_shift_video": shift_v,
        "minimax_h3_sigma_shift_audio": shift_a,
    },
    minimax_payload=payload,
)
official_internal = official._forward(
    [video, audio_internal], timestep, text,
    transformer_options={
        "minimax_h3_sigma_shift_video": shift_v,
        "minimax_h3_sigma_shift_audio": shift_a,
    },
    minimax_payload=payload,
)
official_raw_internal = official._forward(
    [video, audio], timestep, text,
    transformer_options={
        "minimax_h3_sigma_shift_video": shift_v,
        "minimax_h3_sigma_shift_audio": shift_a,
    },
    minimax_payload=payload,
)
ours_raw = ours.velocity(
    video, audio, sigma, text, {},
    shift_video=shift_v, shift_audio=shift_a)
official_y = (
    (1.0 - scale) * audio_internal
    + (1.0 + (scale - 1.0) * sigma_a) * official_internal[1]
)

ours_out = ours.velocity(
    video, audio_scaled, sigma, text, {},
    shift_video=shift_v, shift_audio=shift_a)

print("official video", tuple(official_out[0].shape),
      official_out[0].float().norm().item())
print("official audio", tuple(official_out[1].shape),
      official_out[1].float().norm().item())
print("ours video", tuple(ours_out[0].shape),
      ours_out[0].float().norm().item())
print("ours audio", tuple(ours_out[1].shape),
      ours_out[1].float().norm().item())
print("official internal audio norm", official_internal[1].float().norm().item())
print("official_y audio norm", official_y.float().norm().item())
print("official_out audio norm", official_out[1].float().norm().item())

print("video rel", (ours_out[0].float() - official_out[0].float()).abs().max().item()
      / max(1e-6, official_out[0].float().abs().max().item()))
print("audio rel", (ours_out[1].float() - official_out[1].float()).abs().max().item()
      / max(1e-6, official_out[1].float().abs().max().item()))
print("internal audio rel",
      (ours_out[1].float() - official_internal[1].float()).abs().max().item()
      / max(1e-6, official_internal[1].float().abs().max().item()))
print("raw internal audio rel",
      (ours_raw[1].float() - official_raw_internal[1].float()).abs().max().item()
      / max(1e-6, official_raw_internal[1].float().abs().max().item()))
print("scaled input audio rel",
      (ours_out[1].float() - official_out[1].float()).abs().max().item()
      / max(1e-6, official_out[1].float().abs().max().item()))
