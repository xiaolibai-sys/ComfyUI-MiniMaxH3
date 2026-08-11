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
from types import SimpleNamespace
from typing import Optional

import torch

from .fold import (
    _lora_delta,
    dequantize_and_fold,
    dequantize_weight,
    fold_entries as _fold_entries,
    project_lora_delta,
)
from ..utils.types import AdaLNOverride, H3Lora, LoraEntry, SlotEntry

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
            w = None
            try:
                if qkv_idx is not None:
                    w, _ = dequantize_weight(qt)
                    sl = _row_slice()
                    w[sl] = _fold_entries(w[sl], entries)
                else:
                    w, _ = dequantize_and_fold(qt, entries)
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
            finally:
                del qt
                if w is not None:
                    del w
                if "new_qt" in locals():
                    del new_qt
        else:
            w = entry.data.float()
            if qkv_idx is not None:
                sl = _row_slice()
                w[sl] = _fold_entries(w[sl], entries)
            else:
                w = _fold_entries(w, entries)
            entry.data.copy_(w.to(entry.data.dtype))


def _group_lora_parts(sd: dict) -> dict[str, dict]:
    """Group standardized LoRA tensors by the base weight key."""
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
    return groups


def _make_entry(base: str, parts: dict, strength: float,
                target: Optional[str] = None) -> LoraEntry:
    if target is None:
        target = base[: -len(".weight")] if base.endswith(".weight") else base
    return LoraEntry(
        target=target,
        a=parts.get("a"), b=parts.get("b"),
        alpha=parts.get("alpha"), strength=strength,
        diff=parts.get("diff"), diff_b=parts.get("diff_b"),
    )


def _module_entry_of(t) -> SlotEntry:
    """Transient SlotEntry over a live parameter or quantized tensor."""
    inner = getattr(t, "data", t)
    if hasattr(inner, "_qdata") and hasattr(inner, "_params"):
        return SlotEntry.from_qt(inner)
    return SlotEntry(data=inner)


def fold_lora_into_module(module: torch.nn.Module, entries: list) -> None:
    """Fold LoRA entries directly into a live module's parameters."""
    if not entries:
        return
    slot = {
        pname: _module_entry_of(p)
        for pname, p in module.named_parameters()
    }
    fake_block = SimpleNamespace(lora=list(entries))
    fold_lora_into_slot(fake_block, slot)
    for pname, p in module.named_parameters():
        entry = slot.get(pname)
        if entry is not None and entry.is_qt:
            entry.assign_to(module, pname[: -len(".weight")])


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
    groups = _group_lora_parts(sd)

    out: dict[int, list] = {}
    for base, parts in groups.items():
        if not base.startswith(block_prefix):
            continue
        # blocks.<i>.<component path>
        rest = base[len(block_prefix):]
        idx_s, _, comp = rest.partition(".")
        if not idx_s.isdigit():
            continue
        comp = comp[: -len(".weight")] if comp.endswith(".weight") else comp
        entry = _make_entry(base, parts, strength, target=comp)
        out.setdefault(int(idx_s), []).append(entry)
    return out


def parse_lora_h3(lora_path: str, strength: float = 1.0,
                  silu_grid_path: str = "") -> H3Lora:
    """Parse a MiniMax H3 LoRA into all supported target groups."""
    from safetensors.torch import load_file
    sd = standardize_lora_keys(load_file(lora_path))
    groups = _group_lora_parts(sd)

    block_groups: dict[int, list[LoraEntry]] = {}
    token_refiner_groups: dict[int, list[LoraEntry]] = {}
    final_adaln: Optional[LoraEntry] = None
    override_table = None
    override_block_w: dict[int, torch.Tensor] = {}
    override_block_b: dict[int, torch.Tensor] = {}
    override_final_w = None
    override_final_b = None

    for base, parts in groups.items():
        if base.startswith("blocks."):
            rest = base[len("blocks."):]
            idx_s, _, comp = rest.partition(".")
            if not idx_s.isdigit():
                continue
            comp = comp[: -len(".weight")] if comp.endswith(".weight") else comp
            block_groups.setdefault(int(idx_s), []).append(
                _make_entry(base, parts, strength, target=comp))
        elif base.startswith("token_refiner.blocks."):
            rest = base[len("token_refiner.blocks."):]
            idx_s, _, comp = rest.partition(".")
            if not idx_s.isdigit():
                continue
            comp = comp[: -len(".weight")] if comp.endswith(".weight") else comp
            token_refiner_groups.setdefault(int(idx_s), []).append(
                _make_entry(base, parts, strength, target=comp))
        elif base == "final_layer.adaln_proj.linear.weight":
            final_adaln = _make_entry(
                base, parts, strength, target="adaln_proj.linear")

    for k, v in sd.items():
        if k == "adaln_t_table":
            override_table = v
        elif k.startswith("blocks.") and ".adaln_proj.linear.weight" in k:
            parts = k.split(".")
            override_block_w[int(parts[1])] = v
        elif k.startswith("blocks.") and ".adaln_proj.linear.bias" in k:
            parts = k.split(".")
            override_block_b[int(parts[1])] = v
        elif k == "final_layer.adaln_proj.linear.weight":
            override_final_w = v
        elif k == "final_layer.adaln_proj.linear.bias":
            override_final_b = v

    adaln_override = None
    if override_table is not None:
        adaln_override = AdaLNOverride(
            table=override_table,
            block_weights=override_block_w,
            block_biases=override_block_b,
            final_weight=override_final_w,
            final_bias=override_final_b,
        )

    return H3Lora(
        path=lora_path,
        strength=strength,
        block_groups=block_groups,
        token_refiner_groups=token_refiner_groups,
        final_adaln=final_adaln,
        silu_grid_path=silu_grid_path,
        adaln_override=adaln_override,
    )


def load_silu_grid(path: str) -> torch.Tensor:
    """Load the shared ``silu(t_emb)`` grid used by pruned AdaLN deltas."""
    from ..utils.stream import BlockReader
    if not path:
        raise ValueError(
            "pruned MiniMax H3 LoRA requires a silu(t_emb) grid; "
            "select h3_silu_temb_grid.safetensors"
        )
    reader = BlockReader(path)
    try:
        return reader.get_tensors(["silu_t_emb_grid"])["silu_t_emb_grid"]
    finally:
        reader.close()


class AdalnLoraState:
    """Low-rank AdaLN delta for pruned/curve H3 bases.

    The pruned base has an 8-dim ``adaln_t_table``, while the LoRA delta lives
    in the full 2688-dim ``silu(t_emb)`` space.  Instead of folding into the
    incompatible weight, this injects ``B @ A @ silu(t_emb)`` at runtime.
    """

    def __init__(self, grid: torch.Tensor,
                 block_entries: dict[int, list[LoraEntry]],
                 final_entries: list[LoraEntry]):
        self.grid = grid
        self.block_entries = dict(block_entries)
        self.final_entries = list(final_entries)
        self.current: Optional[torch.Tensor] = None
        self._moved: dict = {}

    def _move(self, t: torch.Tensor, device, dtype) -> torch.Tensor:
        key = (id(t), device, dtype)
        cached = self._moved.get(key)
        if cached is None:
            cached = t.to(device=device, dtype=dtype)
            self._moved[key] = cached
        return cached

    def silu_temb(self, unique_t, device, dtype) -> torch.Tensor:
        t = torch.tensor([float(v) for v in unique_t], dtype=torch.float32,
                         device=self.grid.device)
        pos = t.clamp(0.0, 1.0) * (self.grid.shape[0] - 1)
        i0 = pos.floor().long().clamp(max=self.grid.shape[0] - 2)
        rows = torch.lerp(self.grid[i0].float(),
                          self.grid[i0 + 1].float(), (pos - i0).unsqueeze(1))
        return rows.to(device=device, dtype=dtype)

    def set_current(self, unique_t, device, dtype) -> None:
        self.current = self.silu_temb(unique_t, device, dtype)

    def entry_delta(self, entry: Optional[LoraEntry],
                    st: torch.Tensor) -> Optional[torch.Tensor]:
        if entry is None or entry.a is None or entry.b is None:
            return None
        a = self._move(entry.a, st.device, torch.float32)
        b = self._move(entry.b, st.device, torch.float32)
        return project_lora_delta(
            a, b, entry.alpha, entry.strength, st)

    def apply_to_mods(self, mods, entries: Optional[list[LoraEntry]],
                      st: torch.Tensor):
        if mods is None or not entries:
            return mods
        delta = None
        for entry in entries:
            d = self.entry_delta(entry, st)
            if d is None:
                continue
            delta = d if delta is None else delta + d
        if delta is None:
            return mods
        hidden = mods[0].shape[-1]
        total_rows = mods[0].shape[0]
        delta = delta.reshape(total_rows, len(mods) * hidden)
        chunks = delta.chunk(len(mods), dim=-1)
        return tuple(
            (c + d.to(device=c.device, dtype=c.dtype)).to(c.dtype)
            for c, d in zip(mods, chunks)
        )


def attach_adaln_lora(model, state: AdalnLoraState) -> None:
    """Attach runtime AdaLN deltas to the model's eager and cache paths."""
    model._lora_adaln = state
    for i, blk in enumerate(model.blocks):
        if blk.adaln_proj is not None and i in state.block_entries:
            blk.adaln_proj.attach_lora(state, state.block_entries[i])
    if model.final_layer.adaln_proj is not None and state.final_entries:
        model.final_layer.adaln_proj.attach_lora(state, state.final_entries)
