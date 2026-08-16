"""Quantized (int8) forward test: comfy-kitchen QuantizedTensor slots vs plain bf16.

Builds the same tiny DiT twice; one with plain bf16 block slots, one with
int8_tensorwise QuantizedTensor slots (same underlying weights).  Runs one
velocity step through the ring-buffer BlockSwap for both and compares the
outputs within int8 quantization tolerance.

Run with the ComfyUI venv python:
    python tests/test_quant_forward.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pkg_loader import load as _load_h3rt
_load_h3rt()

import torch
import zlib

from h3rt.utils.config import MiniMaxH3DiTConfig
from h3rt.models.model import MiniMaxH3Model
from h3rt.utils.blockswap import BlockSwapManager, SwapBlock
from h3rt.utils.types import (
    H3BlockSwap,
    PoolPlan,
    SlotEntry,
    SwapAllocation,
)
from h3rt.models import quant

torch.manual_seed(11)

cfg = MiniMaxH3DiTConfig(
    hidden_size=64, num_layers=3, token_refiner_num_layers=1,
    num_attention_heads=4, attention_head_dim=16, ffn_hidden_size=128,
    latents_dim=4, audio_latents_dim=8, patch_size=(1, 2, 2), text_dim=16,
    timestep_input_dim=16, time_embed_hidden_size=64, time_embed_dim=32,
    rope_inv_freq_len=2, norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5,
)

device = torch.device("cuda")


def _weights_for(key: str, shape):
    import math
    g = torch.Generator().manual_seed(zlib.crc32(key.encode()) % (2 ** 32))
    if len(shape) == 1:
        # RMSNorm-style: weights near 1.0 keep activations O(1)
        return torch.full(shape, 1.0, dtype=torch.bfloat16) + 0.01 * torch.randn(shape, generator=g, dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(shape[-1])
    return scale * torch.randn(shape, generator=g, dtype=torch.bfloat16)


def build_runner(quantize: bool, direct: bool = False):
    with torch.device("meta"):
        model = MiniMaxH3Model(cfg, dtype=torch.bfloat16)
    for pname, p in model.named_parameters():
        if pname.startswith("blocks."):
            continue
        mod = model.get_submodule(pname.rsplit(".", 1)[0]) if "." in pname else model
        leaf = pname.rsplit(".", 1)[1] if "." in pname else pname
        mod._parameters[leaf] = torch.nn.Parameter(
            _weights_for(pname, p.shape).to(device, p.dtype), requires_grad=False)
    for bname, b in model.named_buffers():
        mod = model.get_submodule(bname.rsplit(".", 1)[0]) if "." in bname else model
        leaf = bname.rsplit(".", 1)[1] if "." in bname else bname
        mod._buffers[leaf] = _weights_for(bname, b.shape).to(device, b.dtype)

    from comfy_kitchen.tensor import QuantizedTensor as QT
    qdata_map = {}
    blocks = []
    for i in range(cfg.num_layers):
        blk = model.blocks[i]
        keys, names, refs, templates = [], [], [], []
        for pname, p in blk.named_parameters():
            key = f"blocks.{i}.{pname}"
            keys.append(key)
            names.append(pname)
            mod = blk.get_submodule(pname.rsplit(".", 1)[0]) if "." in pname else blk
            leaf = pname.rsplit(".", 1)[1] if "." in pname else pname
            refs.append((mod, leaf, "param"))
            t = _weights_for(key, p.shape)
            if quantize and p.ndim == 2:  # quantize only 2D weights (Linear)
                qt = QT.from_float(t, "TensorWiseINT8Layout")
                entry = SlotEntry.from_qt(qt)
                templates.append(SlotEntry(
                    data=torch.empty_like(entry.data),
                    scale=entry.scale.clone(),
                    layout_cls=entry.layout_cls,
                    orig_dtype=entry.orig_dtype,
                    orig_shape=entry.orig_shape,
                    extra={n: v.clone() for n, v in entry.extra.items()},
                    meta=dict(entry.meta)))
                qdata_map[key] = entry.data.clone()
            else:
                templates.append(SlotEntry(data=torch.empty(tuple(p.shape), dtype=torch.bfloat16)))
                qdata_map[key] = t
        blocks.append(SwapBlock(name=f"blocks.{i}", module=blk, keys=keys, names=names,
                                refs=refs, templates=templates))
        if quantize:
            for m in blk.modules():
                if isinstance(m, torch.nn.Linear):
                    quant.patch_linear(m)

    class FakeReader:
        def get_tensors(self, names):
            return {n: qdata_map[n] for n in names}

        def get_tensor_group(self, names):
            missing = [n for n in names if n not in qdata_map]
            if missing:
                raise KeyError(f"missing tensors: {missing}")
            return {n: qdata_map[n] for n in names}

        def get_tensor_info(self, name):
            t = qdata_map[name]
            return list(t.shape), t.dtype

        def all_keys(self):
            return list(qdata_map)

    if direct:
        # bind the ACTUAL int8 qdata directly (no swap) as the reference
        for i, (blk, sb) in enumerate(zip(model.blocks, blocks)):
            for n, (mod, leaf, kind), e in zip(sb.names, sb.refs, sb.templates):
                g = SlotEntry.empty_like_entry(e, device)
                g.copy_from(e)
                key = f"blocks.{i}.{n}"
                if key in qdata_map:
                    g.data.copy_(qdata_map[key].to(g.data.device))
                g.assign_to(mod, leaf)
        return model, None, qdata_map
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
            home_slots=cfg.num_layers - 2,
            gpu_slots=4,
        ),
    )
    mgr = BlockSwapManager(
        blocks, FakeReader(), device,
        prefetch=True,
        pin_memory=True,
        disk_workers=2,
        allocation=allocation,
        dtype=torch.bfloat16,
    )
    model._swap_mgr = mgr
    return model, mgr, qdata_map


T, H, W, AT = 2, 16, 16, 8
video = torch.randn(1, 4, T, H, W, device=device, dtype=torch.bfloat16)
audio = torch.randn(1, 8, 2, AT, device=device, dtype=torch.bfloat16)
text = torch.randn(1, 8, 64, device=device, dtype=torch.bfloat16)
payload = {"text_token_tags": torch.ones(1, 8, dtype=torch.long, device=device)}

model_a, mgr_a, _ = build_runner(quantize=False)
out_a = model_a.velocity(video, audio, 0.5, text, payload)
torch.cuda.synchronize()
print("PLAIN nan:", torch.isnan(out_a[0]).any().item(), torch.isnan(out_a[1]).any().item())
mgr_a.end()
mgr_a.shutdown()

model_b, mgr_b, _ = build_runner(quantize=True)
out_b = model_b.velocity(video, audio, 0.5, text, payload)
torch.cuda.synchronize()
print("QUANT swap nan:", torch.isnan(out_b[0]).any().item(), torch.isnan(out_b[1]).any().item())
mgr_b.end()
mgr_b.shutdown()

model_d, _, _ = build_runner(quantize=True, direct=True)
out_d = model_d.velocity(video, audio, 0.5, text, payload)
torch.cuda.synchronize()

# 1) swap-vs-direct with the SAME int8 weights -> swap path correctness (~0)
for name, a, b in (("video", out_b[0], out_d[0]), ("audio", out_b[1], out_d[1])):
    a, b = a.float(), b.float()
    rel = (a - b).abs().max().item() / max(1e-6, b.abs().max().item())
    print(f"{name} int8 swap-vs-direct rel_err={rel:.6f}")
    assert rel < 2e-2, f"swap path corrupts {name}: {rel}"

# 2) int8-vs-bf16 loose sanity (max-element error through 3 random blocks)
for name, a, b in (("video", out_a[0], out_b[0]), ("audio", out_a[1], out_b[1])):
    a, b = a.float(), b.float()
    rel = (a - b).abs().max().item() / max(1e-6, a.abs().max().item())
    print(f"{name} int8-vs-bf16 rel_err={rel:.5f}")
    assert rel < 0.5, f"{name} diverged: {rel}"
print("QUANT FORWARD TEST OK")
