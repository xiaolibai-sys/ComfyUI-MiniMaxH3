"""LoRA/DoRA fold tests: key standardisation + merge-math parity with
BerniniRWrapper (including DoRA row-wise normalisation and Kohya orientation).

Run with the ComfyUI venv python:
    python tests/test_lora_fold.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pkg_loader import load as _load_h3rt
_load_h3rt()

import torch

from h3rt.models.lora import (
    standardize_lora_keys, _lora_delta, fold_lora_into_slot, parse_lora,
)
from h3rt.utils.types import LoraEntry, SlotEntry

torch.manual_seed(3)

# ---------------- 1) key standardisation -----------------------------------
cases = {
    "diffusion_model.blocks.3.attn.qkv_proj.lora_A.weight":
        "blocks.3.attn.qkv_proj.lora_A.weight",
    "transformer.blocks.0.mlp.fc1.lora_B.weight":
        "blocks.0.mlp.fc1.lora_B.weight",
    "blocks.2.attn.out_proj.lora_A.weight":
        "blocks.2.attn.out_proj.lora_A.weight",
    "lora_unet_blocks_0_self_attn_q.lora_A.weight":
        "blocks.0.attn.q.lora_A.weight",
    "lycoris_blocks_3_self_attn_q.lora_A.weight":
        "blocks.3.attn.q.lora_A.weight",
    "base_model.model.transformer.blocks.1.attn.qkv_proj.lora_down.weight":
        "blocks.1.attn.qkv_proj.lora_A.weight",
    "blocks.0.mlp.fc2.lora_up.weight": "blocks.0.mlp.fc2.lora_B.weight",
}
for src, exp in cases.items():
    got = standardize_lora_keys({src: torch.zeros(1)}) and list(
        standardize_lora_keys({src: torch.zeros(1)}))[0]
    assert got == exp, f"{src!r} -> {got!r}, expected {exp!r}"
print(f"key standardisation: {len(cases)} cases OK", flush=True)

# ---------------- 2) _lora_delta: diffusers + kohya orientations ------------
out, inn = 8, 16
rank = 4
base = torch.randn(out, inn)
A = torch.randn(rank, inn)          # diffusers
B = torch.randn(out, rank)
d1, _ = _lora_delta(A, B, torch.tensor(8.0), 0.5, base.shape)
exp1 = (B.float() @ A.float()) * (0.5 * 8.0 / rank)
assert torch.allclose(d1, exp1, atol=1e-5), "diffusers orientation mismatch"
d2, _ = _lora_delta(A.t(), B.t(), None, 1.0, base.shape)   # kohya (down=[in,r], up=[r,out])
exp2 = (A.t().float() @ B.t().float()).T
assert torch.allclose(d2, exp2, atol=1e-5), "kohya orientation mismatch"
print("lora delta orientations OK", flush=True)

# ---------------- 3) DoRA standardisation formula ---------------------------
# full DoRA: LoRA delta + row-wise magnitude renormalisation
w0 = torch.randn(out, inn)
dora_diff_b = torch.randn(out, 1)
A = torch.randn(rank, inn)
B = torch.randn(out, rank)
alpha = torch.tensor(8.0)
strength = 0.9
delta, _ = _lora_delta(A, B, alpha, strength, w0.shape)
w_temp = w0 + delta
init_norm = w0.norm(dim=1, keepdim=True).clamp(min=1e-8)
exp_dora = (init_norm + dora_diff_b).clamp(min=0.0) * w_temp / \
           w_temp.norm(dim=1, keepdim=True).clamp(min=1e-8)

entry = LoraEntry(target="attn.qkv_proj", a=A, b=B, alpha=alpha,
                  strength=strength, diff_b=dora_diff_b)
slot = {"attn.qkv_proj.weight": SlotEntry(data=w0.clone())}
block = type("B", (), {"lora": [entry]})()
fold_lora_into_slot(block, slot)
got = slot["attn.qkv_proj.weight"].data.float()
assert torch.allclose(got, exp_dora, atol=1e-5), "DoRA formula mismatch"
print("DoRA standardisation formula OK", flush=True)

# ---------------- 4) plain slot LoRA fold vs direct math --------------------
A = torch.randn(rank, inn)
B = torch.randn(out, rank)
alpha = torch.tensor(8.0)
strength = 0.7
w0 = torch.randn(out, inn)
delta, _ = _lora_delta(A, B, alpha, strength, w0.shape)
entry = LoraEntry(target="mlp.fc1", a=A, b=B, alpha=alpha, strength=strength)
slot = {"mlp.fc1.weight": SlotEntry(data=w0.clone())}
block = type("B", (), {"lora": [entry]})()
fold_lora_into_slot(block, slot)
got = slot["mlp.fc1.weight"].data.float()
assert torch.allclose(got, w0 + delta, atol=1e-5), "plain fold mismatch"
print("plain slot LoRA fold OK", flush=True)

# ---------------- 5) int8 quantized slot fold --------------------------------
from comfy_kitchen.tensor import QuantizedTensor as QT
wq = torch.randn(out, inn, dtype=torch.bfloat16)
qt = QT.from_float(wq, "TensorWiseINT8Layout")
entry = SlotEntry.from_qt(qt)
tpl = SlotEntry(data=torch.empty_like(entry.data), scale=entry.scale.clone(),
                layout_cls=entry.layout_cls, orig_dtype=entry.orig_dtype,
                orig_shape=entry.orig_shape,
                extra={n: v.clone() for n, v in entry.extra.items()},
                meta=dict(entry.meta))
tpl.data.copy_(entry.data)   # the prefetcher fills qdata; emulate it here
slot = {"mlp.fc2.weight": tpl}
entry_l = LoraEntry(target="mlp.fc2", a=A, b=B, alpha=alpha, strength=strength)
block = type("B", (), {"lora": [entry_l]})()
fold_lora_into_slot(block, slot)
ref = wq.float() + delta
# dequant folded slot and compare within int8 requant tolerance
dq = slot["mlp.fc2.weight"].to_quantized_tensor().dequantize().float()
rel = (dq - ref).abs().max().item() / ref.abs().max().item()
assert rel < 0.05, f"int8 fold rel_err={rel}"
print(f"int8 quantized slot fold rel_err={rel:.4f} OK", flush=True)

# ---------------- 6) per-head slicing on fused qkv_proj -----------------------
hd = 8          # heads * head_dim
K = 16
wqkv = torch.randn(3 * hd, K)
A = torch.randn(rank, K)
B = torch.randn(hd, rank)
alpha = torch.tensor(8.0)
strength = 0.6
delta, _ = _lora_delta(A, B, alpha, strength, (hd, K))
slot = {"attn.qkv_proj.weight": SlotEntry(data=wqkv.clone())}
entry = LoraEntry(target="attn.q", a=A, b=B, alpha=alpha, strength=strength)
block = type("B", (), {"lora": [entry]})()
fold_lora_into_slot(block, slot)
got = slot["attn.qkv_proj.weight"].data
exp = wqkv.clone()
exp[:hd] = exp[:hd] + delta
assert torch.allclose(got.float(), exp, atol=1e-5), "per-head q slice mismatch"
assert torch.equal(got[hd:], wqkv[hd:]), "k/v rows must be untouched"
print("per-head q-slice on fused qkv OK", flush=True)

# ---------------- 7) multiple DoRA sequential stacking -----------------------
w0 = torch.randn(out, inn)
A1, B1 = torch.randn(rank, inn), torch.randn(out, rank)
A2, B2 = torch.randn(rank, inn), torch.randn(out, rank)
diff_b1, diff_b2 = torch.randn(out, 1), torch.randn(out, 1)
slot = {"mlp.fc1.weight": SlotEntry(data=w0.clone())}
entries = [
    LoraEntry(target="mlp.fc1", a=A1, b=B1, alpha=None,
              strength=0.8, diff_b=diff_b1),
    LoraEntry(target="mlp.fc1", a=A2, b=B2, alpha=None,
              strength=0.6, diff_b=diff_b2),
]
block = type("B", (), {"lora": entries})()
fold_lora_into_slot(block, slot)

w = w0.float()
for A, B, diff_b, strength in (
    (A1, B1, diff_b1, 0.8),
    (A2, B2, diff_b2, 0.6),
):
    before_norm = w.norm(dim=1, keepdim=True).clamp(min=1e-8)
    w = w + strength * B.float() @ A.float()
    w = (before_norm + diff_b.float()).clamp(min=0.0) * w / \
        w.norm(dim=1, keepdim=True).clamp(min=1e-8)
assert torch.allclose(slot["mlp.fc1.weight"].data.float(), w, atol=1e-5)
print("multiple DoRA sequential stacking OK", flush=True)

# ---------------- 8) parse_lora roundtrip ------------------------------------
sd = {
    "transformer.blocks.0.attn.qkv_proj.lora_A.weight": torch.randn(rank, inn),
    "transformer.blocks.0.attn.qkv_proj.lora_B.weight": torch.randn(out, rank),
    "transformer.blocks.0.attn.qkv_proj.alpha": torch.tensor(8.0),
    "diffusion_model.blocks.2.mlp.fc1.lora_A.weight": torch.randn(rank, inn),
    "diffusion_model.blocks.2.mlp.fc1.diff_b": torch.randn(out, 1),
}
path = os.path.join(tempfile.mkdtemp(prefix="h3_lora_"), "x.safetensors")
from safetensors.torch import save_file
save_file(sd, path)
groups = parse_lora(path, strength=0.5)
assert set(groups) == {0, 2}, f"block grouping wrong: {set(groups)}"
assert groups[0][0].target == "attn.qkv_proj"
assert groups[0][0].a is not None and groups[0][0].b is not None
assert groups[2][0].target == "mlp.fc1" and groups[2][0].diff_b is not None
print("parse_lora grouping OK", flush=True)

print("LORA/DORA FOLD TEST OK")
