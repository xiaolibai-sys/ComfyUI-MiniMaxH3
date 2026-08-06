"""Complete pruned LoRA: AdaLN table/projection replacement through BlockSwap."""

import os
import sys
import tempfile
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, r"D:\ComfyUI-installs\ComfyUI\ComfyUI")
from pkg_loader import load as _load_h3rt
_load_h3rt()

import torch
from safetensors.torch import save_file

from h3rt.models.lora import parse_lora_h3
from h3rt.models.model import MiniMaxH3Model
from h3rt.utils.config import MiniMaxH3DiTConfig
from h3rt.utils.lifecycle import load_model_handle
from h3rt.utils.types import H3BlockSwap

torch.manual_seed(17)

cfg = MiniMaxH3DiTConfig(
    hidden_size=64, num_layers=3, token_refiner_num_layers=1,
    num_attention_heads=4, attention_head_dim=16, ffn_hidden_size=128,
    latents_dim=4, audio_latents_dim=8, patch_size=(1, 2, 2), text_dim=16,
    timestep_input_dim=16, time_embed_hidden_size=64, time_embed_dim=8,
    rope_inv_freq_len=2, norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5,
    adaln_curve_grid=1025,
)

with torch.device("meta"):
    ref = MiniMaxH3Model(cfg, dtype=torch.float32)
ckpt_sd = {}
for pname, p in ref.named_parameters():
    g = torch.Generator().manual_seed(zlib.crc32(pname.encode()) % (2 ** 32))
    ckpt_sd[pname] = torch.randn(p.shape, dtype=torch.float32, generator=g)
for bname, b in ref.named_buffers():
    ckpt_sd[bname] = torch.randn(b.shape, dtype=torch.float32)

tmp = tempfile.mkdtemp(prefix="h3_lora_override_")
ckpt_path = os.path.join(tmp, "base.safetensors")
lora_path = os.path.join(tmp, "complete.safetensors")
save_file(ckpt_sd, ckpt_path)

override_w = torch.randn(6 * 64 * 3, 8)
override_b = torch.randn(6 * 64 * 3)
override_t = torch.randn(1025, 8)
save_file({
    "blocks.0.attn.qkv_proj.lora_A.weight": torch.randn(4, 64),
    "blocks.0.attn.qkv_proj.lora_B.weight": torch.randn(192, 4),
    "blocks.0.adaln_proj.linear.weight": override_w,
    "blocks.0.adaln_proj.linear.bias": override_b,
    "adaln_t_table": override_t,
    "final_layer.adaln_proj.linear.weight": torch.randn(128, 8),
    "final_layer.adaln_proj.linear.bias": torch.randn(128),
}, lora_path)

handle = load_model_handle(ckpt_path)
handle.loras.add(parse_lora_h3(lora_path, strength=1.0))
swap = H3BlockSwap(enabled=True, block_to_swap=1, prefetch=False,
                   prefetch_count=0, pin_memory=True, disk_workers=1,
                   dtype="float32")
m = handle.load(swap_config=swap)
mgr = m._swap_mgr
mgr.prepare(0)
torch.cuda.synchronize()
mgr.end()

gslot = mgr._xfer._block_gpu[0]
gpu = mgr._xfer._gpu_pool[gslot]
assert torch.allclose(m.adaln_t_table.cpu(), override_t, atol=1e-6)
assert torch.allclose(
    gpu["adaln_proj.linear.weight"].data.cpu().float(), override_w, atol=1e-6)
assert torch.allclose(
    gpu["adaln_proj.linear.bias"].data.cpu().float(), override_b, atol=1e-6)
handle.unload()
print("COMPLETE LORA OVERRIDE LOAD OK")
