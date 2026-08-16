"""GPU smoke test: tiny DiT + fake disk reader through the ring-buffer BlockSwap.

Run with the ComfyUI venv python:
    python tests/smoke_blockswap.py
Requires a CUDA GPU.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pkg_loader import load as _load_h3rt
_load_h3rt()

import torch

from h3rt.utils.config import MiniMaxH3DiTConfig
from h3rt.models.model import MiniMaxH3Model
from h3rt.utils.blockswap import BlockSwapManager, SwapBlock
from h3rt.utils.types import (
    H3BlockSwap,
    PoolPlan,
    SlotEntry,
    SwapAllocation,
)

torch.manual_seed(0)

cfg = MiniMaxH3DiTConfig(
    hidden_size=64, num_layers=3, token_refiner_num_layers=1,
    num_attention_heads=4, attention_head_dim=16, ffn_hidden_size=128,
    latents_dim=4, audio_latents_dim=8, patch_size=(1, 2, 2), text_dim=16,
    timestep_input_dim=16, time_embed_hidden_size=64, time_embed_dim=32,
    rope_inv_freq_len=2, norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5,
)

with torch.device("meta"):
    model = MiniMaxH3Model(cfg, dtype=torch.bfloat16)

shape_map = {}
for i, blk in enumerate(model.blocks):
    for pname, p in blk.named_parameters():
        shape_map[f"blocks.{i}.{pname}"] = (tuple(p.shape), p.dtype)
for pname, p in model.named_parameters():
    if not pname.startswith("blocks."):
        shape_map[pname] = (tuple(p.shape), p.dtype)
for bname, b in model.named_buffers():
    shape_map[bname] = (tuple(b.shape), b.dtype)


class FakeReader:
    def get_tensors(self, names):
        return {n: torch.randn(shape_map[n][0], dtype=torch.bfloat16) for n in names}

    def get_tensor_group(self, names):
        return {n: torch.randn(shape_map[n][0], dtype=torch.bfloat16) for n in names}

    def get_tensor_info(self, name):
        return shape_map[name]

    def all_keys(self):
        return list(shape_map)


reader = FakeReader()
device = torch.device("cuda")

for pname, p in model.named_parameters():
    if not pname.startswith("blocks."):
        mod = model.get_submodule(pname.rsplit(".", 1)[0]) if "." in pname else model
        leaf = pname.rsplit(".", 1)[1] if "." in pname else pname
        mod._parameters[leaf] = torch.nn.Parameter(
            torch.randn(shape_map[pname][0], dtype=p.dtype, device=device), requires_grad=False)
for bname, b in model.named_buffers():
    mod = model.get_submodule(bname.rsplit(".", 1)[0]) if "." in bname else model
    leaf = bname.rsplit(".", 1)[1] if "." in bname else bname
    mod._buffers[leaf] = torch.randn(shape_map[bname][0], dtype=b.dtype, device=device)

blocks = []
for i in range(cfg.num_layers):
    blk = model.blocks[i]
    keys, names, refs, templates = [], [], [], []
    for pname, p in blk.named_parameters():
        keys.append(f"blocks.{i}.{pname}")
        names.append(pname)
        mod = blk.get_submodule(pname.rsplit(".", 1)[0]) if "." in pname else blk
        leaf = pname.rsplit(".", 1)[1] if "." in pname else pname
        refs.append((mod, leaf, "param"))
        templates.append(SlotEntry(data=torch.empty(tuple(p.shape), dtype=torch.bfloat16)))
    blocks.append(SwapBlock(name=f"blocks.{i}", module=blk, keys=keys, names=names,
                            refs=refs, templates=templates))

block_mb = blocks[0].bytes_per_block() / 2 ** 20
allocation = SwapAllocation(
    config=H3BlockSwap(
        enabled=True,
        block_to_swap=cfg.num_layers - 2,
        prefetch=True,
        prefetch_count=2,
        pin_memory=True,
        disk_workers=2,
        dtype="bfloat16",
        offload_dit=True,
    ),
    pool=PoolPlan(
        block_mb=block_mb,
        free_mb=0.0,
        effective_reserve_mb=0.0,
        lora_per_slot_mb=0.0,
        max_slots=4,
        requested_slots=4,
        window_size=2,
        hot_blocks=0,
        prefetch_count=2,
        home_slots=2,
        gpu_slots=4,
    ),
)
mgr = BlockSwapManager(
    blocks, reader, device,
    prefetch=True,
    pin_memory=True,
    disk_workers=2,
    allocation=allocation,
    dtype=torch.bfloat16,
)
model._swap_mgr = mgr

T, H, W, AT = 2, 16, 16, 8
video = torch.randn(1, 4, T, H, W, device=device, dtype=torch.bfloat16)
audio = torch.randn(1, 8, 2, AT, device=device, dtype=torch.bfloat16)
text = torch.randn(1, 8, 64, device=device, dtype=torch.bfloat16)
tags = torch.ones(1, 8, dtype=torch.long, device=device)
payload = {"text_token_tags": tags}

v_v, v_a = model.velocity(video, audio, 0.5, text, payload)
print("velocity shapes:", tuple(v_v.shape), tuple(v_a.shape),
      "hits/loads:", mgr.swap_hits, mgr.swap_loads)
assert tuple(v_v.shape) == (1, 4, T, H, W)
assert tuple(v_a.shape) == (1, 8, 2, AT)

v_v2, v_a2 = model.velocity(video, audio, 0.3, text, payload)
print("2nd step hits/loads:", mgr.swap_hits, mgr.swap_loads)

# VAE-phase offload: DIT leaves the GPU ring, then the first window reloads.
mgr.offload_all()
print("gpu ring after offload_all:", mgr._xfer._gpu_storage is None)
assert mgr._xfer._gpu_storage is None
mgr.restore_initial()
v_v3, v_a3 = model.velocity(video, audio, 0.2, text, payload)
print("post-restore velocity shapes:", tuple(v_v3.shape), tuple(v_a3.shape))
assert tuple(v_v3.shape) == (1, 4, T, H, W)
assert tuple(v_a3.shape) == (1, 8, 2, AT)

# global registered home pool check
registered = mgr._xfer._home_storage_tokens is not None
print("home pool registered:", registered)
assert registered

mgr.apply_lora(0, [])
mgr.clear_lora(0)
mgr.end()
mgr.shutdown()
print("SMOKE TEST OK")
