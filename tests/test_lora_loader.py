"""LoRA loader tests: H3Lora parsing, live-module folding, pruned AdaLN delta."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, r"D:\ComfyUI-installs\ComfyUI\ComfyUI")
from pkg_loader import load as _load_h3rt
_load_h3rt()

import torch
import torch.nn as nn
import zlib
from safetensors.torch import save_file

from h3rt.models.lora import (
    AdalnLoraState,
    fold_lora_into_module,
    parse_lora_h3,
)
from h3rt.models.model import AdalnProj, MiniMaxH3Model
from h3_test_utils import h3_sample
from h3rt.utils.config import MiniMaxH3DiTConfig
from h3rt.utils.types import RuntimeOptions
from h3rt.utils.lifecycle import load_model_handle
from h3rt.utils.types import (
    AVLatent,
    H3BlockSwap,
    H3Conditioning,
    H3Lora,
    H3LoraSet,
    LoraEntry,
    TextConditioning,
)

torch.manual_seed(11)


# ---- H3Lora parsing covers blocks, token refiner, and final AdaLN ----------
sd = {
    "blocks.0.attn.qkv_proj.lora_A.weight": torch.randn(4, 16),
    "blocks.0.attn.qkv_proj.lora_B.weight": torch.randn(32, 4),
    "blocks.0.adaln_proj.linear.lora_A.weight": torch.randn(4, 16),
    "blocks.0.adaln_proj.linear.lora_B.weight": torch.randn(64, 4),
    "token_refiner.blocks.1.attn.out_proj.lora_A.weight": torch.randn(4, 16),
    "token_refiner.blocks.1.attn.out_proj.lora_B.weight": torch.randn(8, 4),
    "final_layer.adaln_proj.linear.lora_A.weight": torch.randn(4, 16),
    "final_layer.adaln_proj.linear.lora_B.weight": torch.randn(32, 4),
}
path = os.path.join(tempfile.mkdtemp(prefix="h3_lora_loader_"), "x.safetensors")
save_file(sd, path)
lora = parse_lora_h3(path, strength=0.5)
assert set(lora.block_groups) == {0}
assert {e.target for e in lora.block_groups[0]} == {
    "attn.qkv_proj", "adaln_proj.linear"}
assert set(lora.token_refiner_groups) == {1}
assert lora.token_refiner_groups[1][0].target == "attn.out_proj"
assert lora.final_adaln is not None and lora.final_adaln.target == "adaln_proj.linear"
print("H3Lora parsing OK", flush=True)

multi = H3LoraSet()
multi.add(parse_lora_h3(path, strength=0.5))
multi.add(parse_lora_h3(path, strength=0.3))
assert len(multi.loras) == 2
assert len(multi.block_groups[0]) == 4
assert len(multi.final_adaln_entries) == 2
assert len(multi.signature()) == 2
print("H3LoraSet multi-LoRA merge OK", flush=True)

grid_set = H3LoraSet()
grid_set.add(H3Lora(path="a", silu_grid_path="fl2va"))
grid_set.add(H3Lora(path="b", silu_grid_path="fl2va"))
assert grid_set.silu_grid_path == "fl2va"
try:
    grid_set.add(H3Lora(path="c", silu_grid_path="ref2va"))
except ValueError:
    pass
else:
    raise AssertionError("mixed silu grids should be rejected")
print("H3LoraSet silu grid consistency OK", flush=True)


# ---- live-module folding ------------------------------------------------
class Box(nn.Module):
    def __init__(self, inn, out):
        super().__init__()
        self.proj = nn.Linear(inn, out, bias=False)


inn, out, rank = 16, 8, 4
m = Box(inn, out)
w0 = m.proj.weight.detach().clone()
A = torch.randn(rank, inn)
B = torch.randn(out, rank)
fold_lora_into_module(m, [
    LoraEntry(target="proj", a=A, b=B, alpha=None, strength=0.5)
])
delta = B.float() @ A.float() * 0.5
assert torch.allclose(m.proj.weight.float(), w0.float() + delta, atol=1e-5)
print("fold_lora_into_module OK", flush=True)


# ---- pruned AdaLN delta matches eager and precomputed paths --------------
hidden, expand, modalities, M, t_dim = 4, 6, 3, 2, 8
proj = AdalnProj(t_dim, hidden, expand, modalities, apply_silu=False)
t_emb = torch.randn(M, t_dim)
entry = LoraEntry(
    target="adaln_proj.linear",
    a=torch.randn(rank, 16),
    b=torch.randn(expand * hidden * modalities, rank),
    alpha=None,
    strength=1.0,
)
state = AdalnLoraState(torch.zeros(9, 16), {}, [])
state.current = torch.randn(M, 16)
proj.attach_lora(state, [entry])

base = proj.linear(t_emb)
delta = state.entry_delta(entry, state.current)
exp_chunks = ((base + delta).view(
    M * modalities, expand * hidden).chunk(expand, dim=-1))
got = proj(t_emb)
for g, e in zip(got, exp_chunks):
    assert torch.allclose(g.float(), e.float(), atol=1e-5)

base_chunks = base.view(M * modalities, expand * hidden).chunk(expand, dim=-1)
applied = state.apply_to_mods(base_chunks, [entry], state.current)
for a, e in zip(applied, exp_chunks):
    assert torch.allclose(a.float(), e.float(), atol=1e-5)
print("pruned AdaLN delta OK", flush=True)

entry2 = LoraEntry(
    target="adaln_proj.linear",
    a=torch.randn(rank, 16),
    b=torch.randn(expand * hidden * modalities, rank),
    alpha=None,
    strength=0.5,
)
proj2 = AdalnProj(t_dim, hidden, expand, modalities, apply_silu=False)
state2 = AdalnLoraState(torch.zeros(9, 16), {}, [])
state2.current = torch.randn(M, 16)
proj2.attach_lora(state2, [entry, entry2])
base2 = proj2.linear(t_emb)
delta2 = state2.entry_delta(entry, state2.current) + \
    state2.entry_delta(entry2, state2.current)
exp2 = ((base2 + delta2).view(
    M * modalities, expand * hidden).chunk(expand, dim=-1))
got2 = proj2(t_emb)
for g, e in zip(got2, exp2):
    assert torch.allclose(g.float(), e.float(), atol=1e-5)
print("multiple AdaLN delta merge OK", flush=True)


# ---- ModelHandle + BlockSwap integration ---------------------------------
cfg = MiniMaxH3DiTConfig(
    hidden_size=64, num_layers=3, token_refiner_num_layers=1,
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

tmp = tempfile.mkdtemp(prefix="h3_lora_handle_")
ckpt_path = os.path.join(tmp, "model.safetensors")
lora_path = os.path.join(tmp, "lora.safetensors")
save_file(ckpt_sd, ckpt_path)

blk = ref.blocks[0]
adaln_w = blk.adaln_proj.linear.weight
qkv_w = blk.attn.qkv_proj.weight
final_w = ref.final_layer.adaln_proj.linear.weight
lora_sd = {
    "blocks.0.adaln_proj.linear.lora_A.weight": torch.randn(4, 16),
    "blocks.0.adaln_proj.linear.lora_B.weight": torch.randn(adaln_w.shape[0], 4),
    "blocks.0.attn.qkv_proj.lora_A.weight": torch.randn(4, qkv_w.shape[1]),
    "blocks.0.attn.qkv_proj.lora_B.weight": torch.randn(qkv_w.shape[0], 4),
    "final_layer.adaln_proj.linear.lora_A.weight": torch.randn(4, 16),
    "final_layer.adaln_proj.linear.lora_B.weight": torch.randn(final_w.shape[0], 4),
}
save_file(lora_sd, lora_path)
grid_path = os.path.join(tmp, "grid.safetensors")
save_file({"silu_t_emb_grid": torch.randn(9, 16)}, grid_path)

device = torch.device("cuda")
swap = H3BlockSwap(enabled=True, block_to_swap=1, prefetch=True,
                   prefetch_count=2, pin_memory=True, disk_workers=2,
                   dtype="float32")
handle = load_model_handle(ckpt_path)
handle.loras.add(parse_lora_h3(lora_path, strength=1.0,
                               silu_grid_path=grid_path))
m = handle.load(swap_config=swap)
mgr = m._swap_mgr
video = torch.randn(1, 4, 2, 16, 16, device=device, dtype=torch.float32)
audio = torch.randn(1, 8, 2, 8, device=device, dtype=torch.float32)
text = torch.randn(1, 8, 64, device=device, dtype=torch.float32)
payload = {"text_token_tags": torch.ones(1, 8, dtype=torch.long, device=device)}
out = m.velocity(video, audio, 0.5, text, payload)
torch.cuda.synchronize()
mgr.end()
assert torch.isfinite(out[0]).all() and torch.isfinite(out[1]).all()
assert getattr(m, "_lora_adaln", None) is not None
assert mgr.blocks[0].lora is None

latent = AVLatent(video=video, audio=audio)
cond = H3Conditioning(
    text=TextConditioning(
        states=text,
        tags=payload["text_token_tags"],
    )
)
res_eager = h3_sample(
    handle, cond, latent, None, 2, 1.0, "euler",
    12.0, 1.0, 123, RuntimeOptions(swap=swap),
    disable_pbar=True)
res_cached = h3_sample(
    handle, cond, latent, None, 2, 1.0, "euler",
    12.0, 1.0, 123, RuntimeOptions(swap=swap),
    disable_pbar=True, use_adaln_cache=True)


def _rel(a, b):
    a, b = a.float(), b.float()
    return (a - b).abs().max().item() / max(1e-6, a.abs().max().item())


assert _rel(res_eager.video, res_cached.video) < 1e-4
assert _rel(res_eager.audio, res_cached.audio) < 1e-4
handle.unload()
print("ModelHandle + BlockSwap LoRA integration OK", flush=True)
print("LoRA eager-vs-adaln-cache OK", flush=True)

# ---- fold-on-first-GPU + merged RAM + no re-fold --------------------------
swap2 = H3BlockSwap(enabled=True, block_to_swap=1, prefetch=False,
                    prefetch_count=0, pin_memory=True, disk_workers=1,
                    dtype="float32")
handle2 = load_model_handle(ckpt_path)
handle2.loras.add(parse_lora_h3(lora_path, strength=1.0,
                                silu_grid_path=grid_path))
m2 = handle2.load(swap_config=swap2)
mgr2 = m2._swap_mgr
mgr2.prepare(0)
torch.cuda.synchronize()
mgr2.end()

gslot0 = mgr2._xfer._block_gpu[0]
gpu0 = mgr2._xfer._gpu_pool[gslot0]
folded0 = {
    name: entry.data.detach().cpu().clone()
    for name, entry in gpu0.items()
    if not entry.is_qt
}
assert mgr2.blocks[0].lora is None

mgr2.after_compute(0)
mgr2.end()
hslot0 = mgr2._xfer._block_home.get(0)
assert hslot0 is not None
home0 = mgr2._xfer._home_pool[hslot0]
for name, folded in folded0.items():
    assert torch.equal(home0[name].data, folded), f"{name} RAM != folded GPU"

disk_before = mgr2._disk.disk_reads
mgr2.prepare(0)
torch.cuda.synchronize()
mgr2.end()
gslot_reload = mgr2._xfer._block_gpu[0]
gpu_reload = mgr2._xfer._gpu_pool[gslot_reload]
for name, folded in folded0.items():
    assert torch.equal(gpu_reload[name].data.cpu(), folded), \
        f"{name} reload != folded GPU"
assert mgr2._disk.disk_reads == disk_before
assert mgr2.blocks[0].lora is None
handle2.unload()
print("fold-on-first-GPU + RAM merge + no re-fold OK", flush=True)

print("LORA LOADER TEST OK")
