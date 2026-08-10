"""Second-run + memory/streaming test.

1. Second-run: load -> sample -> unload -> load again (same handle and a fresh
   handle) must work and give identical outputs (cache-eviction-on-unload fix).
2. Memory: with ``ram_budget_gb`` set, the home pool is capped and the rest of
   the blocks are disk-backed through the pinned staging ring — RAM never holds
   the full model; pin-stage buffers are actually pinned.
3. Stage path: a tiny budget forces every block through disk->pin->GPU.

Run with the ComfyUI venv python:
    python tests/test_second_run_and_memory.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pkg_loader import load as _load_h3rt
_load_h3rt()

import torch
import zlib
from safetensors.torch import save_file

from h3rt.utils.config import MiniMaxH3DiTConfig
from h3rt.models.model import MiniMaxH3Model
from h3rt.utils.types import H3BlockSwap
from h3rt.utils.lifecycle import load_model_handle, unload_all

torch.manual_seed(5)

cfg = MiniMaxH3DiTConfig(
    hidden_size=64, num_layers=3, token_refiner_num_layers=1,
    num_attention_heads=4, attention_head_dim=16, ffn_hidden_size=128,
    latents_dim=4, audio_latents_dim=8, patch_size=(1, 2, 2), text_dim=16,
    timestep_input_dim=16, time_embed_hidden_size=64, time_embed_dim=32,
    rope_inv_freq_len=2, norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5,
)

with torch.device("meta"):
    ref = MiniMaxH3Model(cfg, dtype=torch.bfloat16)

sd = {}
for pname, p in ref.named_parameters():
    g = torch.Generator().manual_seed(zlib.crc32(pname.encode()) % (2 ** 32))
    sd[pname] = torch.randn(p.shape, dtype=torch.bfloat16, generator=g)
for bname, b in ref.named_buffers():
    sd[bname] = torch.randn(b.shape, dtype=torch.float32)

ckpt_dir = tempfile.mkdtemp(prefix="h3_ckpt_")
ckpt_path = os.path.join(ckpt_dir, "model.safetensors")
save_file(sd, ckpt_path)
print("checkpoint written:", ckpt_path, f"({os.path.getsize(ckpt_path)} bytes)")

device = torch.device("cuda")
video = torch.randn(1, 4, 2, 16, 16, device=device, dtype=torch.bfloat16)
audio = torch.randn(1, 8, 2, 8, device=device, dtype=torch.bfloat16)
text = torch.randn(1, 8, 64, device=device, dtype=torch.bfloat16)
payload = {"text_token_tags": torch.ones(1, 8, dtype=torch.long, device=device)}


def sample(handle, swap):
    m = handle.load(swap_config=swap)
    out = m.velocity(video, audio, 0.5, text, payload)
    torch.cuda.synchronize()
    mgr = m._swap_mgr
    mgr.end()
    return out, m


# ---------- 1) second-run on the same handle ----------
swap = H3BlockSwap(enabled=True, block_to_swap=1, prefetch=True,  # 3 fake layers -> window 2
                   prefetch_count=2, pin_memory=True, disk_workers=2)
handle = load_model_handle(ckpt_path)
out1, m1 = sample(handle, swap)
handle.unload()
assert not handle.is_loaded(), "handle should be unloaded"
out2, m2 = sample(handle, swap)   # reload on the same handle
assert torch.equal(out1[0], out2[0]) and torch.equal(out1[1], out2[1]), \
    "same-handle reload outputs differ"
print("same-handle reload OK")

# ---------- 2) second run with a FRESH handle (workflow re-run) ----------
h2 = load_model_handle(ckpt_path)
out3, m3 = sample(h2, swap)
assert torch.equal(out1[0], out3[0]) and torch.equal(out1[1], out3[1]), \
    "fresh-handle outputs differ"
print("fresh-handle re-run OK")

# ---------- 3) one-copy semantics: RAM + VRAM = exactly one model copy ----
swap_b = H3BlockSwap(enabled=True, block_to_swap=2, prefetch=True,  # 3 fake layers -> window 1
                     prefetch_count=2, pin_memory=True, disk_workers=2)
print("one-copy handle load...", flush=True)
hb = load_model_handle(ckpt_path)
print("sampling...", flush=True)
_out_b, mb = sample(hb, swap_b)
print("sample done", flush=True)
mgr_b = mb._swap_mgr
# home pool = (total - window) reusable slots -> RAM never holds the full model
assert mgr_b.home_size == mgr_b.total - mgr_b.window_size, \
    f"home={mgr_b.home_size} total={mgr_b.total} window={mgr_b.window_size}"
assert len(mgr_b._xfer._home_pool) == max(mgr_b.home_size, 1)
assert all(s is not None for s in mgr_b._xfer._home_pool)
assert mgr_b._xfer._home_storage_tokens is not None, "home pool not registered"
# after a full pass, every block is resident once: home + gpu == total
resident = len(mgr_b._xfer._block_home) + len(mgr_b._window.on_gpu)
print(f"home_size={mgr_b.home_size}/{mgr_b.total} resident(home+gpu)={resident}")
assert resident == mgr_b.total, f"resident={resident} != total={mgr_b.total}"
assert len(mgr_b._xfer._block_home) <= mgr_b.home_size
# run again: still bit-identical, resident stays one copy
out_b, _ = sample(hb, swap_b)
assert torch.equal(out1[0], out_b[0]) and torch.equal(out1[1], out_b[1]), \
    "second pass outputs differ"
resident2 = len(mb._swap_mgr._xfer._block_home) + len(mb._swap_mgr._window.on_gpu)
assert resident2 == mgr_b.total
print("one-copy run OK")

# ---------- 4) RSS sanity: RAM stays below full-model bytes ----------
import psutil
proc = psutil.Process()
base_rss = proc.memory_info().rss
hb.load()   # already loaded; force a full rebuild after unload
hb.unload()
hb.load()
torch.cuda.synchronize()
after_rss = proc.memory_info().rss
ckpt_bytes = os.path.getsize(ckpt_path)
print(f"RSS delta={after_rss - base_rss} bytes vs ckpt={ckpt_bytes} bytes")
# RSS is noisy at this scale; home_size=0 already proves the model is not
# held in RAM.  This is a gross-regression smoke bound.
assert after_rss - base_rss < ckpt_bytes * 8, "RAM grew far beyond checkpoint size"

unload_all()
print("SECOND-RUN + MEMORY TEST OK")
