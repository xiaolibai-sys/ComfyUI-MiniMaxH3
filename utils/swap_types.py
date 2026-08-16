"""Shared BlockSwap data types and storage helpers."""

from __future__ import annotations

import torch

from .types import SlotEntry, SwapBlock


def _dtype_of(t) -> torch.dtype:
    try:
        return t.dtype
    except Exception:
        return torch.float32


def free_module_storage(module: torch.nn.Module) -> None:
    """Replace every parameter/buffer with a zero-size tensor."""
    for key in list(module._parameters.keys()):
        p = module._parameters.get(key)
        if p is None:
            continue
        try:
            module._parameters[key] = torch.nn.Parameter(
                torch.empty((0,), dtype=_dtype_of(p)), requires_grad=False)
        except Exception:
            pass
    for key in list(module._buffers.keys()):
        b = module._buffers.get(key)
        if b is None:
            continue
        try:
            module._buffers[key] = torch.empty((0,), dtype=_dtype_of(b))
        except Exception:
            pass


def _entry_of(t) -> SlotEntry:
    """Transient SlotEntry view over a live param/buffer tensor."""
    if hasattr(t, "_qdata") and hasattr(t, "_params"):
        return SlotEntry.from_qt(t)
    return SlotEntry(data=t)
