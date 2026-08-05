"""NVFP4 encoder alignment: vendored quantized loading vs ComfyUI official.

Compares, for real quantized layers of the ComfyUI MiniMax-H3 Qwen3-VL-32B
NVFP4/AWQ checkpoint:

* dequantized weights: our ``LayerSpec``-built comfy-kitchen QuantizedTensor
  vs the official ``comfy.ops`` quantized Linear (same comfy-kitchen kernels).
* end-to-end text encoding: our vendored TextEncoder (NVFP4, streamed) runs
  one short prompt and sanity-checks shape/finiteness/statistics (full numeric
  parity with ComfyUI requires loading both 32B models, not possible in 23 GB
  RAM; dequant parity is the bit-exact contract that guarantees it).

Run with the ComfyUI venv python from the package root:
    python tests/test_nvfp4_align.py
"""

import json
import gc
import os
import sys

sys.path.insert(0, r"D:\ComfyUI-installs\ComfyUI\ComfyUI")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pkg_loader import load as _load_h3rt
_load_h3rt()

import torch
from safetensors import safe_open

WEIGHT = (r"D:\ComfyUI-installs\ComfyUI\ComfyUI\models\text_encoders"
          r"\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
MODEL_DIR = (r"D:\ComfyUI-installs\ComfyUI\ComfyUI\custom_nodes"
             r"\ComfyUI-MiniMaxH3\models\text_encoder\ModelsData\Minimax EncModel")


def rel_err(a, b):
    a, b = a.float(), b.float()
    return (a - b).abs().max().item() / max(1e-6, b.abs().max().item())


def official_dequant(prefix):
    """Reference dequantization replicating comfy.ops._load_quantized_module
    (nvfp4: qdata U8 + block_scale F8_E4M3 + tensor_scale F32; int8:
    qdata I8 + per-row scale) via the same comfy-kitchen layout."""
    from comfy_kitchen.tensor import QuantizedTensor, get_layout_class
    with safe_open(WEIGHT, framework="pt") as f:
        weight = f.get_tensor(prefix + "weight")
        conf = json.loads(bytes(f.get_tensor(prefix + "comfy_quant").tolist()).decode())
        fmt = conf["format"]
        if fmt == "nvfp4":
            layout = "TensorCoreNVFP4Layout"
            ts = f.get_tensor(prefix + "weight_scale_2")
            bs = f.get_tensor(prefix + "weight_scale").view(torch.float8_e4m3fn)
            params = dict(scale=ts, block_scale=bs)
        elif fmt == "int8_tensorwise":
            layout = "TensorWiseINT8Layout"
            sc = f.get_tensor(prefix + "weight_scale")
            params = dict(scale=sc)
        else:
            raise ValueError(fmt)
        cls = get_layout_class(layout)
        p = cls.Params(**params, orig_dtype=torch.bfloat16,
                       orig_shape=tuple(weight.shape))
        return QuantizedTensor(weight, layout, p).dequantize()


def our_dequant(prefix):
    from h3rt.models import quant
    from h3rt.utils.stream import BlockReader
    reader = BlockReader(WEIGHT)
    key = prefix + "weight"
    spec = quant.load_layer_spec(reader, key, read_weight=True)
    qt = quant.make_quantized_tensor(spec, tuple(spec.qdata.shape),
                                     torch.bfloat16)
    reader.close()
    return qt.dequantize()


def check_layer(name, prefix, tol=1e-4):
    a = our_dequant(prefix)
    b = official_dequant(prefix)
    assert tuple(a.shape) == tuple(b.shape), f"{name}: {tuple(a.shape)} vs {tuple(b.shape)}"
    err = rel_err(a, b)
    assert err < tol, f"{name}: dequant rel_err={err} > {tol}"
    print(f"  {name}: dequant rel_err={err:.6f}  OK", flush=True)


print("=== dequantized-weight parity (nvfp4 + int8 embed) ===", flush=True)
check_layer("q_proj L0", "model.layers.0.self_attn.q_proj.")
check_layer("gate_proj L0", "model.layers.0.mlp.gate_proj.")
check_layer("o_proj L24", "model.layers.24.self_attn.o_proj.")
check_layer("down_proj L49", "model.layers.49.mlp.down_proj.")

# embed_tokens is int8_tensorwise; official Embedding dequantizes the same way
from h3rt.models import quant as _q
from h3rt.utils.stream import BlockReader
reader = BlockReader(WEIGHT)
spec = _q.load_layer_spec(reader, "model.embed_tokens.weight", read_weight=True)
qt = _q.make_quantized_tensor(spec, tuple(spec.qdata.shape), torch.bfloat16)
ours_embed = qt.dequantize()
with safe_open(WEIGHT, framework="pt") as f:
    w8 = f.get_tensor("model.embed_tokens.weight")
    sc = f.get_tensor("model.embed_tokens.weight_scale")
from comfy_kitchen.tensor import QuantizedTensor as _QT, get_layout_class as _GLC
_cls = _GLC("TensorWiseINT8Layout")
_p = _cls.Params(scale=sc, orig_dtype=torch.bfloat16,
                 orig_shape=tuple(w8.shape))
off_embed = _QT(w8, "TensorWiseINT8Layout", _p).dequantize()
err = rel_err(ours_embed, off_embed)
assert err < 1e-4, f"embed dequant rel_err={err}"
print(f"  embed_tokens: dequant rel_err={err:.6f}  OK", flush=True)
reader.close()

print("=== end-to-end NVFP4 encode (vendored, streamed) ===", flush=True)
# End-to-end runs in a fresh subprocess: this 23 GB machine cannot hold two
# 14.6 GB safetensors mmaps plus the CUDA context in one process.
import subprocess
_script = r'''
import sys, os, torch
sys.path.insert(0, r"D:\ComfyUI-installs\ComfyUI\ComfyUI")
sys.path.insert(0, r"D:\ComfyUI-installs\ComfyUI\ComfyUI\custom_nodes\ComfyUI-MiniMaxH3")
import importlib.util
_p = r"D:\ComfyUI-installs\ComfyUI\ComfyUI\custom_nodes\ComfyUI-MiniMaxH3"
spec = importlib.util.spec_from_file_location("h3rt", os.path.join(_p, "__init__.py"))
m = importlib.util.module_from_spec(spec); sys.modules["h3rt"] = m; spec.loader.exec_module(m)
from h3rt.models.text_encoder.encoder import TextEncoder
from h3rt.models.text_encoder.types import StreamConfig, TextEncoderInput
cfg = StreamConfig(group_size=2, prefetch=True, prefetch_count=1,
                   pin_memory=True, disk_workers=2, device="cuda",
                   dtype="bfloat16", weight_path=r"__WEIGHT__")
enc = TextEncoder(r"__MODEL_DIR__", stream_config=cfg)
enc.model.norm = torch.nn.Identity()
out = enc.encode(TextEncoderInput(text="a cat sitting on a mat", max_length=64))
assert tuple(out.last_hidden_state.shape) == (1, 6, 5120)
assert torch.isfinite(out.last_hidden_state).all()
print("E2E hidden=%s finite=True mean_abs=%.4f" % (
    tuple(out.last_hidden_state.shape),
    out.last_hidden_state.float().abs().mean().item()))
img = torch.rand(1, 96, 64, 3, dtype=torch.float32)
out = enc.encode(TextEncoderInput(
    text="a cat", max_length=64,
    minimax_ref_items=[{"type": "image", "data": img}]))
assert tuple(out.last_hidden_state.shape) == (1, 16, 5120)
assert torch.isfinite(out.last_hidden_state).all()
print("E2E ref hidden=%s finite=True" % (tuple(out.last_hidden_state.shape),))
enc.destroy()
'''
_script = _script.replace("__WEIGHT__", WEIGHT).replace("__MODEL_DIR__", MODEL_DIR)
r = subprocess.run([sys.executable, "-c", _script], capture_output=True, text=True,
                   timeout=1200)
out_txt = (r.stdout + r.stderr)
print("  " + "\n  ".join(l for l in out_txt.splitlines()
                         if "E2E" in l or "Error" in l or "Traceback" in l or "assert" in l)[:800], flush=True)
assert r.returncode == 0 and "E2E hidden=(1, 6, 5120)" in out_txt and \
    "E2E ref hidden=(1, 16, 5120)" in out_txt, out_txt[-2000:]
print("NVFP4 ALIGN TEST OK")
