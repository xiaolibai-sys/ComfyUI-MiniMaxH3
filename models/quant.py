"""Weight-format auto-detection + comfy-kitchen quantized loading.

Supports the ComfyUI checkpoint convention used by the official Comfy-Org
weights: every quantized layer carries ``{layer}.comfy_quant`` (JSON bytes:
format / convrot flags) plus ``{layer}.weight`` (storage dtype) and scales
(``weight_scale``, ``weight_scale_2``, ``input_scale``, ``pre_quant_scale``).

Layers without quant metadata load as plain tensors.  The runtime keeps its
modules as ``nn.Linear``; when a quantized weight is bound, ``patch_linear``
wraps the module forward to dispatch through the comfy-kitchen kernels
(int8/convrot via ``ck.int8_linear``, nvfp4 by quantising the input then
letting ``F.linear`` dispatch through the layout handler).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import torch

from ..utils.types import SlotEntry


class _ShapeOnly:
    """Header-only weight placeholder (shape + dtype, no data)."""
    __slots__ = ("shape", "dtype")

    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype


_ck = None
_qt_cls = None
_get_layout = None

# format -> comfy-kitchen layout class name / storage dtype (mirrors
# comfy.quant_ops.QUANT_ALGOS, kept local so the runtime stays self-contained)
FORMAT_LAYOUT = {
    "float8_e4m3fn": "TensorCoreFP8E4M3Layout",
    "float8_e5m2": "TensorCoreFP8E5M2Layout",
    "mxfp8": "TensorCoreMXFP8Layout",
    "nvfp4": "TensorCoreNVFP4Layout",
    "int8_tensorwise": "TensorWiseINT8Layout",
    "convrot_w4a4": "TensorCoreConvRotW4A4Layout",
}
FORMAT_STORAGE = {
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
    "mxfp8": torch.float8_e4m3fn,
    "nvfp4": torch.uint8,
    "int8_tensorwise": torch.int8,
    "convrot_w4a4": torch.int8,
}


def _ensure_ck():
    global _ck, _qt_cls, _get_layout
    if _ck is None:
        import comfy_kitchen as _ck
        from comfy_kitchen.tensor import QuantizedTensor as _qt_cls
        from comfy_kitchen.tensor import get_layout_class as _get_layout
    return _ck, _qt_cls, _get_layout


def _resolve_layout_name(fmt: str) -> str:
    preferred = FORMAT_LAYOUT.get(fmt)
    if preferred is None:
        raise ValueError(f"unsupported quant format {fmt!r}")
    try:
        _get_layout(preferred)
        return preferred
    except KeyError:
        if fmt.startswith("float8_"):
            _get_layout("TensorCoreFP8Layout")
            return "TensorCoreFP8Layout"
        raise


def ck_available() -> bool:
    try:
        _ensure_ck()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Layer format detection
# ---------------------------------------------------------------------------

@dataclass
class LayerSpec:
    """One weight's on-disk representation."""
    is_quant: bool = False
    format: str = ""                 # nvfp4 | int8_tensorwise | convrot_w4a4 | fp8...
    qdata: Optional[torch.Tensor] = None
    scales: dict = field(default_factory=dict)        # layout Params kwargs
    meta: dict = field(default_factory=dict)          # convrot etc.
    extra_params: dict = field(default_factory=dict)  # input_scale / pre_quant_scale


def layer_prefix_of(full_key: str) -> str:
    """'blocks.0.attn.qkv_proj.weight' -> 'blocks.0.attn.qkv_proj.'"""
    return full_key[: full_key.rindex("weight")]


def is_quant_layer(reader, full_key: str) -> bool:
    if not full_key.endswith(".weight"):
        return False
    lp = layer_prefix_of(full_key)
    return reader.has(lp + "comfy_quant") or reader.has(lp + "weight_scale")


def load_layer_spec(reader, full_key: str, read_weight: bool = False) -> LayerSpec:
    """Read one weight + its quant metadata.

    ``read_weight=False`` (default, build-time) reads only the small metadata
    (``comfy_quant`` JSON + scales) and records the storage shape/dtype from
    the header; the actual weight bytes stream in later via the prefetcher.
    """
    if not full_key.endswith(".weight"):
        # biases / 1D weights are never quantized
        if read_weight:
            return LayerSpec(is_quant=False,
                             qdata=reader.get_tensors([full_key])[full_key])
        shape, dt = reader.get_tensor_info(full_key)
        return LayerSpec(is_quant=False, qdata=_ShapeOnly(shape, dt))
    lp = layer_prefix_of(full_key)
    if read_weight:
        weight = reader.get_tensors([full_key])[full_key]
    else:
        shape, dt = reader.get_tensor_info(full_key)
        weight = _ShapeOnly(shape, dt)

    conf = None
    if reader.has(lp + "comfy_quant"):
        raw = reader.get_tensors([lp + "comfy_quant"])[lp + "comfy_quant"]
        try:
            conf = json.loads(bytes(raw.tolist()).decode("utf-8"))
        except Exception:
            conf = None

    if conf is None:
        return LayerSpec(is_quant=False, qdata=weight)

    fmt = conf.get("format")
    if fmt is None:
        return LayerSpec(is_quant=False, qdata=weight)

    ck, qt, get_layout = _ensure_ck()
    layout_name = _resolve_layout_name(fmt)

    def pop_scale(name, view_dtype=None):
        key = lp + name
        if not reader.has(key):
            return None
        v = reader.get_tensors([key])[key]
        if view_dtype is not None:
            v = v.view(view_dtype)
        return v

    scales: dict = {}
    meta: dict = {}
    if fmt in ("float8_e4m3fn", "float8_e5m2"):
        scales["scale"] = pop_scale("weight_scale")
    elif fmt == "mxfp8":
        scales["scale"] = pop_scale("weight_scale", torch.float8_e8m0fnu)
    elif fmt == "nvfp4":
        ts = pop_scale("weight_scale_2")
        bs = pop_scale("weight_scale", torch.float8_e4m3fn)
        scales["scale"] = ts
        scales["block_scale"] = bs
    elif fmt == "int8_tensorwise":
        scales["scale"] = pop_scale("weight_scale")
        params_conf = conf.get("params", {}) or {}
        if conf.get("convrot", params_conf.get("convrot", False)):
            scales["convrot"] = True
            scales["convrot_groupsize"] = int(
                conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256)))
    elif fmt == "convrot_w4a4":
        scales["scale"] = pop_scale("weight_scale")
        params_conf = conf.get("params", {}) or {}
        scales["convrot_groupsize"] = int(
            conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256)))
        scales["quant_group_size"] = 64
        scales["linear_dtype"] = conf.get("linear_dtype", params_conf.get("linear_dtype", "int4"))
    else:
        raise ValueError(f"unsupported quant format {fmt!r} for {full_key}")

    # extra per-layer params (applied outside the matmul)
    extra = {}
    for name in ("input_scale", "pre_quant_scale"):
        v = pop_scale(name)
        if v is not None:
            extra[name] = v

    return LayerSpec(is_quant=True, format=fmt, qdata=weight, scales=scales,
                     meta=meta, extra_params=extra)


def make_quantized_tensor(spec: LayerSpec, orig_shape: tuple, orig_dtype: torch.dtype,
                          qdata: Optional[torch.Tensor] = None):
    """Build a comfy-kitchen QuantizedTensor from a LayerSpec."""
    ck, qt, get_layout = _ensure_ck()
    layout_name = _resolve_layout_name(spec.format)
    layout_cls = get_layout(layout_name)
    params = layout_cls.Params(**spec.scales, orig_dtype=orig_dtype,
                               orig_shape=tuple(orig_shape))
    storage_t = FORMAT_STORAGE[spec.format]
    if qdata is None:
        if isinstance(spec.qdata, _ShapeOnly):
            # meta: shape-only template, ZERO committed memory.  Windows
            # commits torch.empty() immediately; allocating a full buffer per
            # template for all 50 blocks would commit ~15 GB before the pools
            # are even built (access violation / OOM).
            qdata = torch.empty(spec.qdata.shape, dtype=storage_t, device="meta")
        else:
            qdata = spec.qdata
    if qdata.dtype != storage_t:
        qdata = qdata.to(storage_t)
    return qt(qdata, layout_name, params)


def slot_entry_template(spec: LayerSpec, orig_shape: tuple,
                        orig_dtype: torch.dtype) -> SlotEntry:
    """Template SlotEntry for pool pre-allocation (plain or quantized).

    For quantized layers the qdata buffer is allocated from the file's storage
    shape (header-only); scale/extra values are copied in now (small tensors).
    The prefetcher later fills only the qdata buffer.
    """
    if not spec.is_quant:
        return SlotEntry(data=torch.empty(orig_shape, dtype=orig_dtype, device="meta"))
    qt = make_quantized_tensor(spec, orig_shape, orig_dtype)
    entry = SlotEntry.from_qt(qt)
    return SlotEntry(
        data=torch.empty(entry.data.shape, dtype=entry.data.dtype, device="meta"),
        scale=entry.scale.clone() if entry.scale is not None else None,
        layout_cls=entry.layout_cls,
        orig_dtype=entry.orig_dtype,
        orig_shape=entry.orig_shape,
        extra={n: t.clone() for n, t in entry.extra.items()},
        meta=dict(entry.meta))


# ---------------------------------------------------------------------------
# Quantized forward dispatch (wraps nn.Linear instances on bind)
# ---------------------------------------------------------------------------

def patch_linear(module) -> None:
    """Wrap an ``nn.Linear`` instance so quantized weights dispatch through
    comfy-kitchen kernels.  Idempotent.  Plain weights keep the normal path."""
    if getattr(module, "_h3_quant_patched", False):
        return
    orig_fwd = module.forward

    def fwd(x):
        w = module.weight
        if isinstance(w, _qt_cls if _qt_cls is not None else ()):
            layout = w._layout_cls
            if getattr(module, "_h3_full_precision_mm", False):
                pqs = getattr(module, "_pre_quant_scale", None)
                if pqs is not None:
                    x = x * pqs.to(x.device).to(x.dtype)
                wd = w.dequantize().to(x.dtype)
                b = module.bias.to(x.dtype) if module.bias is not None else None
                return _F.linear(x, wd, b)
            if layout == "TensorWiseINT8Layout":
                ck, _, _ = _ensure_ck()
                from comfy_kitchen.tensor import TensorWiseINT8Layout
                qdata, scale = TensorWiseINT8Layout.get_plain_tensors(w)
                inp2 = x.reshape(-1, x.shape[-1]) if x.ndim > 2 else x
                out = ck.int8_linear(
                    inp2, qdata, scale, module.bias, inp2.dtype,
                    convrot=getattr(w._params, "convrot", False),
                    convrot_groupsize=getattr(w._params, "convrot_groupsize", 256))
                if x.ndim > 2:
                    out = out.reshape(*x.shape[:-1], -1)
                return out
            if layout.startswith("TensorCoreNVFP4"):
                pqs = getattr(module, "_pre_quant_scale", None)
                if pqs is not None:
                    x = x * pqs.to(x.device).to(x.dtype)
                isc = getattr(module, "_input_scale", None)
                ndim = x.ndim
                inp = x.reshape(-1, x.shape[-1]) if ndim > 2 else x
                xq = _qt_cls.from_float(inp, layout, scale=isc)
                out = _orig_linear(xq, w, module.bias)
                if ndim > 2:
                    out = out.reshape(*x.shape[:-1], -1)
                return out
            if layout.startswith("TensorCoreFP8"):
                return _orig_linear(x, w, module.bias)
        return orig_fwd(x)

    module._h3_quant_patched = True
    module._h3_orig_fwd = orig_fwd
    module.forward = fwd


import torch.nn.functional as _F


def _orig_linear(x, w, b):
    return _F.linear(x, w, b)


def bind_param(module, leaf, tensor_or_qt, extra_params=None) -> None:
    """Bind a checkpoint tensor / QuantizedTensor into a module parameter,
    attaching per-layer extras (pre_quant_scale / input_scale)."""
    if isinstance(tensor_or_qt, SlotEntry):
        tensor_or_qt.assign_to(module, leaf)
    else:
        module._parameters[leaf] = torch.nn.Parameter(tensor_or_qt, requires_grad=False)
    if isinstance(module, torch.nn.Linear):
        patch_linear(module)
    if extra_params:
        for name, t in extra_params.items():
            setattr(module, f"_{name}", t.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")))
