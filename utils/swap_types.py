"""Shared BlockSwap data types and storage helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from .types import SlotEntry


@dataclass
class SwapBlock:
    """One swappable weight group (a DiT block)."""
    name: str
    module: torch.nn.Module
    keys: list[str] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    refs: list[tuple[torch.nn.Module, str, str]] = field(default_factory=list)
    templates: list[SlotEntry] = field(default_factory=list)
    lora: Optional[list] = None
    overrides: dict = field(default_factory=dict)

    def bytes_per_block(self) -> int:
        total = 0
        for t in self.templates:
            total += t.data.numel() * t.data.element_size()
            if t.scale is not None:
                total += t.scale.numel() * t.scale.element_size()
            for e in t.extra.values():
                total += e.numel() * e.element_size()
        return total


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
