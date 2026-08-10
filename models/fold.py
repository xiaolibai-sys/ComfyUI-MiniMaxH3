"""Shared LoRA/DoRA delta math for backbone and AdaLN folding."""

from __future__ import annotations

from typing import Optional

import torch


def _lora_delta(A: torch.Tensor, B: torch.Tensor, alpha, strength: float,
                base_shape=None):
    """``strength * (alpha/rank) * delta`` in float32, both orientations."""
    A = A.to(torch.float32)
    B = B.to(torch.float32)
    if base_shape is not None and len(base_shape) == 2:
        out, in_ = int(base_shape[0]), int(base_shape[1])
        if B.shape[1] == A.shape[0] and (B @ A).shape == (out, in_):
            delta, rank = B @ A, A.shape[0]
        elif A.shape[1] == B.shape[0] and (A @ B).T.shape == (out, in_):
            delta, rank = (A @ B).T, A.shape[1]
        else:
            delta, rank = B @ A, A.shape[0]
    else:
        if A.shape[1] == B.shape[0]:
            delta, rank = (A @ B).T, A.shape[1]
        else:
            delta, rank = B @ A, A.shape[0]
    if alpha is not None:
        try:
            alpha_val = float(alpha.item() if alpha.numel() == 1 else alpha)
        except Exception:
            alpha_val = float(rank)
    else:
        alpha_val = float(rank)
    return delta * (strength * (alpha_val / rank)), rank


def _chunk_add_delta(w: torch.Tensor, entry, chunk_rows: int = 8192) -> None:
    """Add one LoRA delta in row chunks on ``w.device``."""
    if entry.a is None or entry.b is None:
        return
    A = entry.a.to(device=w.device, dtype=torch.float32)
    n_rows = w.shape[0]
    for start in range(0, n_rows, chunk_rows):
        end = min(start + chunk_rows, n_rows)
        rows = slice(start, end)
        B = entry.b[rows].to(device=w.device, dtype=torch.float32)
        d, _ = _lora_delta(A, B, entry.alpha, entry.strength,
                           (end - start, w.shape[1]))
        w[rows] += d


def _fold_entries_chunked(w: torch.Tensor, entries: list,
                          chunk_rows: int = 8192) -> torch.Tensor:
    """Chunked standard LoRA fold to avoid materialising huge B@A deltas."""
    for e in entries:
        _chunk_add_delta(w, e, chunk_rows)
    return w


def _fold_entries_dora_chunked(w: torch.Tensor, entries: list,
                               chunk_rows: int = 8192) -> torch.Tensor:
    """Chunked DoRA fold: row norms on GPU, deltas added per row chunk."""
    for e in entries:
        if e.diff_b is not None:
            before_norm = w.norm(dim=1, keepdim=True).clamp(min=1e-8)
            _chunk_add_delta(w, e, chunk_rows)
            after_norm = w.norm(dim=1, keepdim=True).clamp(min=1e-8)
            diff_b = e.diff_b.to(device=w.device, dtype=w.dtype)
            m = (before_norm + diff_b.float().reshape(-1, 1)).clamp(min=0.0)
            w = m * w / after_norm
        else:
            _chunk_add_delta(w, e, chunk_rows)
    return w


def fold_entries(w: torch.Tensor, entries: list) -> torch.Tensor:
    """LoRA + DoRA standardisation on one weight tensor (fp32)."""
    w = w.float()
    if w.dim() == 2:
        if any(e.diff_b is not None for e in entries):
            return _fold_entries_dora_chunked(w, entries)
        if not any(e.diff is not None for e in entries):
            return _fold_entries_chunked(w, entries)
    for e in entries:
        before_norm = None
        if e.diff_b is not None and w.dim() == 2:
            before_norm = w.norm(dim=1, keepdim=True).clamp(min=1e-8)
        if e.a is not None and e.b is not None:
            d, _ = _lora_delta(e.a, e.b, e.alpha, e.strength, w.shape)
            w = w + d.to(device=w.device, dtype=w.dtype)
        if before_norm is not None:
            diff_b = e.diff_b.to(device=w.device, dtype=w.dtype)
            m = (before_norm + diff_b.float().reshape(-1, 1)).clamp(min=0.0)
            w = m * w / w.norm(dim=1, keepdim=True).clamp(min=1e-8)
    diff = next((e.diff for e in entries if e.diff is not None), None)
    if diff is not None and w.dim() == 1:
        w = w + diff.to(device=w.device, dtype=w.dtype).float()
    return w


def project_lora_delta(a: torch.Tensor, b: torch.Tensor, alpha, strength: float,
                       inputs: torch.Tensor) -> Optional[torch.Tensor]:
    """Project low-rank ``A/B`` onto ``inputs`` and return the output delta.

    Supports diffusers ``A=[rank,in]`` / ``B=[out,rank]`` and kohya
    ``A=[in,rank]`` / ``B=[rank,out]`` orientations.  The result is always
    float32 and shaped like the projected output rows.
    """
    if a is None or b is None:
        return None
    A = a.to(device=inputs.device, dtype=torch.float32)
    B = b.to(device=inputs.device, dtype=torch.float32)
    x = inputs.to(device=inputs.device, dtype=torch.float32)
    in_dim = int(x.shape[-1])
    if A.shape[1] == in_dim:
        rank = A.shape[0]
        proj = x @ A.T
    elif A.shape[0] == in_dim:
        rank = A.shape[1]
        proj = x @ A
    else:
        raise ValueError(
            f"LoRA input dim mismatch: A={tuple(A.shape)} input={in_dim}")
    proj_dim = int(proj.shape[-1])
    if B.shape[1] == proj_dim:
        delta = proj @ B.T
    elif B.shape[0] == proj_dim:
        delta = proj @ B
    else:
        raise ValueError(
            f"LoRA output dim mismatch: B={tuple(B.shape)} proj={proj_dim}")
    if alpha is not None:
        try:
            alpha_val = float(alpha.item() if alpha.numel() == 1 else alpha)
        except Exception:
            alpha_val = float(rank)
    else:
        alpha_val = float(rank)
    return delta * (strength * (alpha_val / max(1, rank)))


def sum_projected_deltas(entries: list, inputs: torch.Tensor
                         ) -> Optional[torch.Tensor]:
    """Sum all low-rank projections for ``entries`` over ``inputs``."""
    delta = None
    for e in entries or []:
        d = project_lora_delta(e.a, e.b, e.alpha, e.strength, inputs)
        if d is None:
            continue
        delta = d if delta is None else delta + d
    return delta


def dequantize_weight(w):
    """Return ``(data, orig_dtype)`` for plain or comfy-kitchen weights."""
    orig_dtype = getattr(w, "orig_dtype", None)
    if orig_dtype is None and hasattr(w, "_params"):
        orig_dtype = getattr(w._params, "orig_dtype", None)
    if orig_dtype is None and hasattr(w, "data"):
        data = w.data
        if hasattr(data, "orig_dtype"):
            orig_dtype = data.orig_dtype
        elif hasattr(data, "_params"):
            orig_dtype = getattr(data._params, "orig_dtype", None)
    if orig_dtype is None:
        orig_dtype = getattr(w, "dtype", None)

    def _dequant(data):
        layout = getattr(data, "_layout_cls", None)
        params = getattr(data, "_params", None)
        if (
            layout == "TensorWiseINT8Layout"
            and params is not None
            and getattr(params, "convrot", False)
        ):
            from comfy_kitchen.backends.eager.quantization import (
                dequantize_int8_convrot_weight,
            )
            from comfy_kitchen.tensor import TensorWiseINT8Layout
            qdata, scale = TensorWiseINT8Layout.get_plain_tensors(data)
            return dequantize_int8_convrot_weight(
                qdata,
                scale,
                int(getattr(params, "convrot_groupsize", 256)),
            )
        return data.dequantize()

    if hasattr(w, "dequantize"):
        data = _dequant(w)
    elif hasattr(w, "data") and hasattr(w.data, "dequantize"):
        data = _dequant(w.data)
    else:
        data = w.data if hasattr(w, "data") else w
    if orig_dtype is not None:
        data = data.to(orig_dtype)
    return data, orig_dtype


def dequantize_and_fold(w, entries: list):
    """Dequantize when needed, apply ``fold_entries``, return float32 data."""
    data, orig_dtype = dequantize_weight(w)
    if entries:
        data = fold_entries(data, entries)
    return data, orig_dtype
