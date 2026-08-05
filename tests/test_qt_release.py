"""QuantizedTensor release test: no leak across load/sample/unload cycles.

Creates a tiny INT8 (comfy_quant) checkpoint on disk, loads it through the
full ModelHandle + ring-buffer BlockSwap path (QuantizedTensor slots), samples,
unloads, and asserts CUDA memory returns to baseline and RSS stays bounded
over repeated cycles.

Run with the ComfyUI venv python:
    python tests/test_qt_release.py
"""

import json
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
from h3rt.utils.lifecycle import load_model_handle, unload_all, collect_garbage

torch.manual_seed(9)

cfg = MiniMaxH3DiTConfig(
    hidden_size=256, num_layers=4, token_refiner_num_layers=1,
    num_attention_heads=4, attention_head_dim=64, ffn_hidden_size=512,
    latents_dim=4, audio_latents_dim=8, patch_size=(1, 2, 2), text_dim=16,
    timestep_input_dim=16, time_embed_hidden_size=256, time_embed_dim=96,
    rope_inv_freq_len=2, norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5,
)

with torch.device("meta"):
    ref = MiniMaxH3Model(cfg, dtype=torch.bfloat16)

from comfy_kitchen.tensor import QuantizedTensor as QT

sd = {}
for pname, p in ref.named_parameters():
    g = torch.Generator().manual_seed(zlib.crc32(pname.encode()) % (2 ** 32))
    if p.ndim == 2:
        w = torch.randn(p.shape, dtype=torch.bfloat16, generator=g)
        qt = QT.from_float(w, "TensorWiseINT8Layout")
        sd[pname] = qt._qdata
        base = pname[:-len("weight")]
        sd[base + "weight_scale"] = qt._params.scale
        conf = json.dumps({"format": "int8_tensorwise"}).encode("utf-8")
        sd[base + "comfy_quant"] = torch.tensor(list(conf), dtype=torch.uint8)
    else:
        sd[pname] = torch.randn(p.shape, dtype=torch.bfloat16, generator=g)
for bname, b in ref.named_buffers():
    sd[bname] = torch.randn(b.shape, dtype=torch.float32)

ckpt = os.path.join(tempfile.mkdtemp(prefix="h3_qt_"), "model.safetensors")
save_file(sd, ckpt)
print(f"int8 checkpoint written ({os.path.getsize(ckpt)} bytes)", flush=True)

device = torch.device("cuda")
video = torch.randn(1, 4, 2, 16, 16, device=device, dtype=torch.bfloat16)
audio = torch.randn(1, 8, 2, 8, device=device, dtype=torch.bfloat16)
text = torch.randn(1, 8, 256, device=device, dtype=torch.bfloat16)
payload = {"text_token_tags": torch.ones(1, 8, dtype=torch.long, device=device)}

swap = H3BlockSwap(enabled=True, block_to_swap=2, prefetch=True, prefetch_count=2,  # 4 fake layers -> window 2
                   pin_memory=True, disk_workers=2)

import psutil
proc = psutil.Process()
rss0 = proc.memory_info().rss
torch.cuda.synchronize()
alloc0 = torch.cuda.memory_allocated()

rss_prev = rss0
for cycle in range(3):
    handle = load_model_handle(ckpt)
    m = handle.load(swap_config=swap)
    out = m.velocity(video, audio, 0.5, text, payload)
    torch.cuda.synchronize()
    m._swap_mgr.end()
    # verify quantized weights are actually used (a param is a QuantizedTensor)
    qt_params = sum(1 for p in m.parameters() if type(p).__name__ == "QuantizedTensor")
    assert qt_params > 0, "no QuantizedTensor params bound"
    handle.unload()
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated()
    rss = proc.memory_info().rss
    rss_growth = rss - rss_prev
    print(f"cycle {cycle}: qt_params={qt_params} "
          f"vram_delta={alloc - alloc0} bytes rss_growth={rss_growth // 1024 // 1024} MiB", flush=True)
    assert alloc - alloc0 < 2 * 1024 * 1024, f"VRAM not released: {alloc - alloc0} bytes"
    # cycle 0 includes one-time triton kernel compilation; later cycles must
    # not accumulate RAM (that would be an unload leak)
    if cycle > 0:
        assert rss_growth < 40 * 1024 * 1024, f"RAM accumulating: {rss_growth // 1024 // 1024} MiB"
    rss_prev = rss

unload_all()
print("QT RELEASE TEST OK")
