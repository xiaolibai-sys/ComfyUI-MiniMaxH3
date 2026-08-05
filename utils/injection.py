"""Centralised injection context for the MiniMax H3 sampling pipeline.

Mirrors ComfyUI-BerniniRWrapper's ``utils/injection.py``: built once at the
start of ``h3_sample()`` from the optional config-node payloads
(``H3BlockSwap`` / ``H3TeaCache``), it is the single place that turns node
inputs into runtime behaviour — block-swap model loading and TeaCache block
hooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .types import H3BlockSwap, H3TeaCache


@dataclass
class InjectionContext:
    """All injection data extracted once at sampling start."""

    swap: Optional[H3BlockSwap] = None
    teacache: Optional[H3TeaCache] = None

    @classmethod
    def build(
        cls,
        block_swap_args: Optional[H3BlockSwap] = None,
        teacache_args: Optional[H3TeaCache] = None,
    ) -> "InjectionContext":
        return cls(swap=block_swap_args, teacache=teacache_args)

    def make_teacache(self, model):
        """Attach TeaCache hooks to the model, or return None when disabled."""
        if self.teacache is None:
            return None
        from .teacache import TeaCache
        return TeaCache(
            model,
            start_block=self.teacache.start_block,
            max_skip_blocks=self.teacache.max_skip_blocks,
            rel_l1_thresh=self.teacache.rel_l1_thresh,
            warmup_steps=self.teacache.warmup_steps,
            cooldown_steps=self.teacache.cooldown_steps,
        )
