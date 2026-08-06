"""Multi-LoRA auto-fold vs manual-fold equivalence with BlockSwap+hot+prebake.

Builds one tiny pruned/curve H3 checkpoint and three random LoRAs. The
automatic path feeds all three through H3LoraSet + BlockSwap (with hot blocks
and AdaLN pre-bake). The manual path folds the same backbone deltas directly
into a second checkpoint and only re-injects AdaLN deltas at runtime, then
runs the same sampler configuration.
"""

import os
import sys
import tempfile
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, r"D:\ComfyUI-installs\ComfyUI\ComfyUI")
from pkg_loader import load as _load_h3rt
_load_h3rt()

import torch
import torch.nn as nn
from safetensors.torch import save_file

from h3rt.models.lora import fold_lora_into_module, parse_lora_h3
from h3rt.models.model import MiniMaxH3Model
from h3rt.nodes.h3_sampling import h3_sample
from h3rt.utils.config import MiniMaxH3DiTConfig
from h3rt.utils.injection import InjectionContext
from h3rt.utils.lifecycle import load_model_handle
from h3rt.utils.types import (
    AVLatent,
    H3BlockSwap,
    H3Conditioning,
    H3Lora,
    H3LoraSet,
)

torch.manual_seed(23)

cfg = MiniMaxH3DiTConfig(
    hidden_size=64, num_layers=10, token_refiner_num_layers=2,
    num_attention_heads=4, attention_head_dim=16, ffn_hidden_size=128,
    latents_dim=4, audio_latents_dim=8, patch_size=(1, 2, 2), text_dim=16,
    timestep_input_dim=16, time_embed_hidden_size=64, time_embed_dim=8,
    rope_inv_freq_len=2, norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5,
    adaln_curve_grid=8,
)

with torch.device("meta"):
    ref = MiniMaxH3Model(cfg, dtype=torch.float32)

ckpt_sd = {}
for pname, p in ref.named_parameters():
    g = torch.Generator().manual_seed(zlib.crc32(pname.encode()) % (2 ** 32))
    ckpt_sd[pname] = torch.randn(p.shape, dtype=torch.float32, generator=g)
for bname, b in ref.named_buffers():
    ckpt_sd[bname] = torch.randn(b.shape, dtype=torch.float32)


def bind_plain(model, sd):
    for pname, p in model.named_parameters():
        mod = model.get_submodule(pname.rsplit(".", 1)[0]) if "." in pname else model
        leaf = pname.rsplit(".", 1)[1] if "." in pname else pname
        mod._parameters[leaf] = nn.Parameter(sd[pname].clone(), requires_grad=False)
    for bname, b in model.named_buffers():
        mod = model.get_submodule(bname.rsplit(".", 1)[0]) if "." in bname else model
        leaf = bname.rsplit(".", 1)[1] if "." in bname else bname
        mod._buffers[leaf] = sd[bname].clone()


def add_lora(sd, base, weight, rank=4, grid_dim=None):
    a = torch.randn(rank, grid_dim if grid_dim is not None else weight.shape[1])
    b = torch.randn(weight.shape[0], rank)
    sd[f"{base}.lora_A.weight"] = a
    sd[f"{base}.lora_B.weight"] = b


tmp = tempfile.mkdtemp(prefix="h3_lora_multi_")
ckpt_path = os.path.join(tmp, "base.safetensors")
grid_path = os.path.join(tmp, "grid.safetensors")
save_file(ckpt_sd, ckpt_path)
save_file({"silu_t_emb_grid": torch.randn(9, 16)}, grid_path)

b0 = ref.blocks[0]
b1 = ref.blocks[5]
b2 = ref.blocks[9]
tr0 = ref.token_refiner.blocks[0]
tr1 = ref.token_refiner.blocks[1]

lora1_sd = {}
add_lora(lora1_sd, "blocks.0.attn.qkv_proj", b0.attn.qkv_proj.weight)
add_lora(lora1_sd, "blocks.0.adaln_proj.linear", b0.adaln_proj.linear.weight,
         grid_dim=16)
add_lora(lora1_sd, "token_refiner.blocks.0.attn.out_proj",
         tr0.attn.out_proj.weight)
add_lora(lora1_sd, "final_layer.adaln_proj.linear",
         ref.final_layer.adaln_proj.linear.weight, grid_dim=16)

lora2_sd = {}
add_lora(lora2_sd, "blocks.1.mlp.fc1", b1.mlp.fc1.weight)
add_lora(lora2_sd, "blocks.1.adaln_proj.linear", b1.adaln_proj.linear.weight,
         grid_dim=16)

lora3_sd = {}
add_lora(lora3_sd, "blocks.2.attn.out_proj", b2.attn.out_proj.weight)
add_lora(lora3_sd, "blocks.2.adaln_proj.linear", b2.adaln_proj.linear.weight,
         grid_dim=16)
add_lora(lora3_sd, "token_refiner.blocks.1.attn.qkv_proj",
         tr1.attn.qkv_proj.weight)
add_lora(lora3_sd, "final_layer.adaln_proj.linear",
         ref.final_layer.adaln_proj.linear.weight, grid_dim=16)

lora_paths = []
for i, sd in enumerate((lora1_sd, lora2_sd, lora3_sd), 1):
    path = os.path.join(tmp, f"lora{i}.safetensors")
    save_file(sd, path)
    lora_paths.append(path)

auto_loras = H3LoraSet()
for path in lora_paths:
    auto_loras.add(parse_lora_h3(path, strength=1.0,
                                 silu_grid_path=grid_path))

# ---- manual path: fold backbone into weights, keep AdaLN at runtime ------
manual = MiniMaxH3Model(cfg, dtype=torch.float32)
bind_plain(manual, ckpt_sd)
manual_loras = H3LoraSet()
for lora in auto_loras.loras:
    for idx, entries in lora.block_groups.items():
        folded = [e for e in entries if e.target != "adaln_proj.linear"]
        adaln = [e for e in entries if e.target == "adaln_proj.linear"]
        if folded:
            fold_lora_into_module(manual.blocks[idx], folded)
        if adaln:
            manual_loras.add(H3Lora(
                path=f"manual-block-{idx}",
                strength=lora.strength,
                block_groups={idx: adaln},
                silu_grid_path=grid_path,
            ))
    for idx, entries in lora.token_refiner_groups.items():
        if entries:
            fold_lora_into_module(manual.token_refiner.blocks[idx], entries)
    if lora.final_adaln is not None:
        manual_loras.add(H3Lora(
            path="manual-final",
            strength=lora.strength,
            final_adaln=lora.final_adaln,
            silu_grid_path=grid_path,
        ))

manual_sd = {}
for pname, p in manual.named_parameters():
    manual_sd[pname] = p.detach().cpu()
for bname, b in manual.named_buffers():
    manual_sd[bname] = b.detach().cpu()
manual_path = os.path.join(tmp, "manual_folded.safetensors")
save_file(manual_sd, manual_path)

swap = H3BlockSwap(enabled=True, block_to_swap=3, hot_blocks=1,
                   prefetch=True, prefetch_count=1, pin_memory=True,
                   disk_workers=2, dtype="float32")

device = torch.device("cuda")
video = torch.randn(1, 4, 2, 16, 16, device=device, dtype=torch.float32)
audio = torch.randn(1, 8, 2, 8, device=device, dtype=torch.float32)
text = torch.randn(1, 8, 64, device=device, dtype=torch.float32)
tags = torch.ones(1, 8, dtype=torch.long, device=device)
latent = AVLatent(video=video, audio=audio)
cond = H3Conditioning(text_states=text, text_token_tags=tags)


def run(handle):
    return h3_sample(
        handle, cond, latent, None, 5, 1.0, "euler",
        12.0, 1.0, 123, InjectionContext.build(block_swap_args=swap),
        disable_pbar=True, use_adaln_cache=True)


auto_handle = load_model_handle(ckpt_path)
auto_handle.loras = auto_loras
res_auto = run(auto_handle)

manual_handle = load_model_handle(manual_path)
manual_handle.loras = manual_loras
res_manual = run(manual_handle)


def rel(a, b):
    a, b = a.float(), b.float()
    return (a - b).abs().max().item() / max(1e-6, a.abs().max().item())


rel_v = rel(res_auto.video, res_manual.video)
rel_a = rel(res_auto.audio, res_manual.audio)
print(f"auto-vs-manual rel_err: video={rel_v:.6f} audio={rel_a:.6f}")
assert rel_v < 1e-3 and rel_a < 1e-3
print("MULTI-LORA 10-LAYER 5-STEP AUTO vs MANUAL + BLOCKSWAP + HOT + PREBAKE OK")
