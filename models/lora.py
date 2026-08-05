"""LoRA/DoRA key standardisation + inline folding (BerniniRWrapper parity).

Ports the parts of ``ComfyUI-BerniniRWrapper/utils/lora.py`` relevant to the
MiniMax H3 DiT:

* ``standardize_lora_keys`` — converts common LoRA key formats (ComfyUI
  native, plain blocks, diffusers ``transformer.``, Kohya ``lora_unet_``,
  LyCORIS ``lycoris_blocks_``, Fun ``lora_unet__``) to
  ``blocks.N.<component>.weight`` so they match the swap-block names;
* ``_lora_delta`` — diffusers (``B@A``) AND kohya (``(A@B).T``) orientations,
  disambiguated against the base weight shape; ``scale = strength * alpha/rank``;
* DoRA standardisation — ``W_final = (||W0|| + diff_b) * W_temp / ||W_temp||``
  row-wise, ``||W0||`` computed on the base weight before the LoRA delta;
  ``diff`` on 1-D norm weights is additive; ``diff_b`` on a 1-D weight merges
  into the sibling ``.bias`` slot entry;
* quantized slots fold via ``requantize_from_float`` (best-effort; convrot
  warns and leaves the weight unmerged).

``parse_lora`` loads + standardises a LoRA file and groups the entries per
DiT block so they can be handed to ``BlockSwapManager.apply_lora``.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from ..utils.types import LoraEntry, SlotEntry

logger = logging.getLogger("h3.lora")


# ---------------------------------------------------------------------------
# Key standardisation
# ---------------------------------------------------------------------------

def _standardize_key(k: str) -> str:
    """Normalise one LoRA key to ``blocks.N.<module>.<suffix>`` form."""
    # LyCORIS / AI-Toolkit underscore format
    if k.startswith("lycoris_blocks_"):
        k = k.replace("lycoris_blocks_", "blocks.")
        k = k.replace("_self_attn_", ".self_attn.")
        k = k.replace("_ffn_net_0_proj", ".ffn.0")
        k = k.replace("_ffn_net_2", ".ffn.2")
        k = k.replace("to_out_0", "o")
    # Fun LoRA
    if k.startswith("lora_unet__"):
        k = k.replace("lora_unet__", "")
        k = k.replace("_blocks_", ".blocks.")
        k = k.replace("_self_attn_", ".self_attn.")
        k = k.replace("_q.", ".q.")
        k = k.replace("_k.", ".k.")
        k = k.replace("_v.", ".v.")
        k = k.replace("_o.", ".o.")
        k = k.replace("_ffn_", ".ffn.")
    # Common prefixes (iteratively, they can be nested:
    # e.g. base_model.model.transformer.blocks...)
    _PREFIXES = ("transformer.", "pipe.dit.", "base_model.model.", "diffusion_model.")
    while True:
        for pre in _PREFIXES:
            if k.startswith(pre):
                k = k[len(pre):]
                break
        else:
            break
    if k.startswith("lora_unet_") and not k.startswith("lora_unet__"):
        body = k[len("lora_unet_"):]
        parts = body.split(".")
        main_part = parts[0]
        tokens = main_part.split("_")
        path_tokens = []
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t == "blocks" and i + 1 < len(tokens) and tokens[i + 1].isdigit():
                path_tokens.extend(["blocks", tokens[i + 1]]); i += 2; continue
            if t in ("self", "cross") and i + 1 < len(tokens) and tokens[i + 1] == "attn":
                path_tokens.append(f"{t}_attn"); i += 2; continue
            if t in ("q", "k", "v", "o"):
                path_tokens.append(t); i += 1; continue
            if t == "ffn" and i + 1 < len(tokens) and tokens[i + 1].isdigit():
                path_tokens.extend(["ffn", tokens[i + 1]]); i += 2; continue
            path_tokens.append(t); i += 1
        k = ".".join(path_tokens) + ("." + ".".join(parts[1:]) if len(parts) > 1 else "")
    # Wan-style self_attn.q/k/v/o: q/k/v are per-head slices of the FUSED
    # attn.qkv_proj (the fold resolves the row ranges from the weight shape);
    # o maps to the full attn.out_proj
    k = k.replace("self_attn.q.", "attn.q.")
    k = k.replace("self_attn.k.", "attn.k.")
    k = k.replace("self_attn.v.", "attn.v.")
    k = k.replace("self_attn.o.", "attn.out_proj.")
    # suffix normalisation
    k = k.replace(".lora_down.weight", ".lora_A.weight")
    k = k.replace(".lora_up.weight", ".lora_B.weight")
    k = k.replace("lora_down", "lora_A")
    k = k.replace("lora_up", "lora_B")
    return k


def standardize_lora_keys(lora_sd: dict) -> dict:
    return {_standardize_key(k): v for k, v in lora_sd.items()}


# ---------------------------------------------------------------------------
# Merge math (BerniniRWrapper parity)
# ---------------------------------------------------------------------------

def _lora_delta(A: torch.Tensor, B: torch.Tensor, alpha, strength: float,
                base_shape=None):
    """``strength * (alpha/rank) * delta`` in float32, both orientations.

    Diffusers: ``lora_A=[rank,in]``, ``lora_B=[out,rank]`` -> ``B @ A``.
    Kohya: ``lora_down=[in,rank]`` (A), ``lora_up=[rank,out]`` (B) -> ``(A @ B).T``.
    ``base_shape`` validates the orientation for square matrices.
    """
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


def _fold_entries(w: torch.Tensor, entries: list) -> torch.Tensor:
    """LoRA + DoRA standardisation on one weight tensor (fp32)."""
    w = w.float()
    init_norm = None
    has_dora = any(e.diff_b is not None for e in entries)
    if has_dora and w.dim() == 2:
        init_norm = w.norm(dim=1, keepdim=True).clamp(min=1e-8)
    for e in entries:
        if e.a is None or e.b is None:
            continue
        d, _ = _lora_delta(e.a, e.b, e.alpha, e.strength, w.shape)
        w = w + d
    diff_b = next((e.diff_b for e in entries if e.diff_b is not None), None)
    diff = next((e.diff for e in entries if e.diff is not None), None)
    if diff_b is not None and w.dim() == 2 and init_norm is not None:
        m = (init_norm + diff_b.float().reshape(-1, 1)).clamp(min=0.0)
        w = m * w / w.norm(dim=1, keepdim=True).clamp(min=1e-8)
    elif diff is not None and w.dim() == 1:
        w = w + diff.float()
    return w


# ---------------------------------------------------------------------------
# Slot folding
# ---------------------------------------------------------------------------

def _find_slot_key(slot: dict, leaf: str):
    for k in slot:
        if k.endswith(f".{leaf}.weight") or k == f"{leaf}.weight":
            return k
    return None


# per-head LoRA targets on the FUSED qkv_proj: (slot leaf, qkv index)
_HEAD_TARGETS = {"attn.q": ("attn.qkv_proj", 0),
                 "attn.k": ("attn.qkv_proj", 1),
                 "attn.v": ("attn.qkv_proj", 2)}


def _resolve_target(slot: dict, leaf: str):
    """Return (slot_key, row_slice) for a target leaf.

    Whole-weight targets match the slot key directly; per-head targets
    (``attn.q/k/v``) resolve to the fused ``attn.qkv_proj`` with the row
    range derived from the weight shape (q/k/v each occupy rows/3)."""
    key = _find_slot_key(slot, leaf)
    if key is not None:
        return key, None
    if leaf in _HEAD_TARGETS:
        base, idx = _HEAD_TARGETS[leaf]
        key = _find_slot_key(slot, base)
        if key is not None:
            return key, idx
    return None, None


def _find_bias_key(slot: dict, weight_key: str):
    b = weight_key[: -len(".weight")] + ".bias"
    return b if b in slot else None


def fold_lora_into_slot(block, slot: dict) -> None:
    """Fold ``block.lora`` entries into ``slot`` (dict of param -> SlotEntry).

    Handles norm-only groups (no A/B): ``diff`` on a 1-D weight is additive;
    ``diff_b`` on a 1-D weight merges into the sibling ``.bias`` slot entry.
    """
    if not block.lora:
        return
    by_target: dict[str, list] = {}
    for e in block.lora:
        by_target.setdefault(e.target, []).append(e)

    for leaf, entries in by_target.items():
        key, qkv_idx = _resolve_target(slot, leaf)
        if key is None:
            continue
        entry = slot[key]
        hd = entry.data.shape[0] // 3 if qkv_idx is not None else None

        def _row_slice():
            return slice(qkv_idx * hd, (qkv_idx + 1) * hd)

        # DoRA bias: diff_b on a 1-D weight merges into the sibling .bias entry
        for e in entries:
            if e.diff_b is not None and entry.data.dim() == 1:
                bkey = _find_bias_key(slot, key)
                if bkey is not None:
                    b = slot[bkey]
                    b.data.copy_((b.data.float() + e.diff_b.float()).to(b.data.dtype))

        if entry.is_qt:
            # fold on the DEQUANTISED weight, then requantise with a fresh
            # scale (folding on raw int8 qdata would corrupt the result)
            qt = entry.to_quantized_tensor()
            w = qt.dequantize().float()
            if qkv_idx is not None:
                sl = _row_slice()
                w[sl] = _fold_entries(w[sl], entries)
            else:
                w = _fold_entries(w, entries)
            try:
                new_qt = qt.requantize_from_float(w.to(qt.dtype), scale="recalculate")
                entry.data.copy_(new_qt._qdata)
                entry.scale.copy_(new_qt._params.scale)
                for f in getattr(new_qt._params, "__dataclass_fields__", {}):
                    if f in ("scale", "orig_dtype", "orig_shape"):
                        continue
                    v = getattr(new_qt._params, f)
                    if isinstance(v, torch.Tensor) and f in entry.extra:
                        entry.extra[f].copy_(v)
            except Exception:
                logger.warning("quantized LoRA fold for %s failed (convrot?) - left unmerged", key)
        else:
            w = entry.data.float()
            if qkv_idx is not None:
                sl = _row_slice()
                w[sl] = _fold_entries(w[sl], entries)
            else:
                w = _fold_entries(w, entries)
            entry.data.copy_(w.to(entry.data.dtype))


# ---------------------------------------------------------------------------
# LoRA file parsing -> BlockSwapManager.apply_lora payload
# ---------------------------------------------------------------------------

def parse_lora(lora_path: str, strength: float = 1.0,
               block_prefix: str = "blocks."):
    """Load + standardise a LoRA file and group entries per DiT block index.

    Returns ``{block_idx: [LoraEntry, ...]}`` ready for
    ``BlockSwapManager.apply_lora(idx, entries)``.
    """
    from safetensors.torch import load_file
    sd = standardize_lora_keys(load_file(lora_path))
    groups: dict[str, dict] = {}
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        base = None
        kind = None
        for suffix, kk in ((".lora_A.weight", "a"), (".lora_B.weight", "b"),
                           (".alpha", "alpha"), (".diff_b", "diff_b"), (".diff", "diff")):
            if k.endswith(suffix):
                base = k[: -len(suffix)] + ".weight"
                kind = kk
                break
        if base is None:
            continue
        groups.setdefault(base, {})[kind] = v

    out: dict[int, list] = {}
    for base, parts in groups.items():
        if not base.startswith(block_prefix):
            continue
        # blocks.<i>.<component path>
        rest = base[len(block_prefix):]
        idx_s, _, comp = rest.partition(".")
        if not idx_s.isdigit():
            continue
        entry = LoraEntry(
            target=comp[: -len(".weight")] if comp.endswith(".weight") else comp,
            a=parts.get("a"), b=parts.get("b"),
            alpha=parts.get("alpha"), strength=strength,
            diff=parts.get("diff"), diff_b=parts.get("diff_b"),
        )
        out.setdefault(int(idx_s), []).append(entry)
    return out
