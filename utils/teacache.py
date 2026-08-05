"""TeaCache — skip redundant transformer block computations during sampling.

Ported from ComfyUI-BerniniRWrapper's ``utils/teacache.py`` (itself from
WanVideoWrapper), adapted to the MiniMax H3 DiT: blocks take
``(x, t_emb, mod_segments, rope_freqs)`` and there are no context windows,
so a single cache slot is enough.  When consecutive sampling steps produce
near-identical hidden states at ``start_block``, the blocks
``[start_block, start_block + max_skip_blocks)`` are skipped and the cached
output is reused.

Step counting is driven externally by the sampler via ``step()``.  With CFG,
the second (uncond) call within the same step is forced to compute so the
cache always holds the cond-pass state.
"""

from __future__ import annotations

from typing import Optional

import torch

DEFAULT_START_BLOCK = 3
DEFAULT_MAX_SKIP_BLOCKS = 15
DEFAULT_REL_L1_THRESH = 0.08
DEFAULT_WARMUP_STEPS = 1   # skip first step (structure formation)
DEFAULT_COOLDOWN_STEPS = 2  # skip last 2 steps (detail refinement)


def _l1(x: torch.Tensor, y: torch.Tensor) -> float:
    return (x - y).abs().float().mean().item()


class TeaCache:
    """Attaches caching hooks to the MiniMaxH3Model transformer blocks.

    Usage::

        cache = TeaCache(model, start_block=3, max_skip_blocks=15, rel_l1_thresh=0.08)
        cache.reset(total_steps=30)
        # per denoising step: cache.step() once, then run the model forward(s)
        cache.detach()
    """

    def __init__(
        self,
        model,
        *,
        start_block: int = DEFAULT_START_BLOCK,
        max_skip_blocks: int = DEFAULT_MAX_SKIP_BLOCKS,
        rel_l1_thresh: float = DEFAULT_REL_L1_THRESH,
        warmup_steps: int = DEFAULT_WARMUP_STEPS,
        cooldown_steps: int = DEFAULT_COOLDOWN_STEPS,
    ):
        if not hasattr(model, "blocks"):
            raise RuntimeError("TeaCache: model has no .blocks (expected MiniMaxH3Model).")
        self._model = model
        n_blocks = len(model.blocks)
        self._start = max(0, min(start_block, n_blocks - 1))
        self._end = min(self._start + max_skip_blocks, n_blocks)
        self._thresh = rel_l1_thresh
        self._warmup = warmup_steps
        self._cooldown = cooldown_steps

        self._step: int = 0
        self._total_steps: int = 0
        self._skipping: bool = False
        self._cache_output_pending: bool = False
        self._consumed_step: int = -1
        self._cached_input: Optional[torch.Tensor] = None
        self._cached_output: Optional[torch.Tensor] = None
        self._orig_forwards: dict[int, callable] = {}
        self._patched: bool = False

        self._patch()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def reset(self, total_steps: int):
        """Call at the start of every sampling run."""
        self._step = 0
        self._total_steps = total_steps
        self._skipping = False
        self._cache_output_pending = False
        self._consumed_step = -1
        self._cached_input = None
        self._cached_output = None

    def step(self):
        """Advance the step counter (once per denoising step, NOT per model call)."""
        self._step += 1
        self._skipping = False

    def detach(self):
        """Restore original block forwards and release references (idempotent)."""
        if not self._patched:
            return
        for i, orig in self._orig_forwards.items():
            if i < len(self._model.blocks):
                self._model.blocks[i].forward = orig
        self._orig_forwards.clear()
        self._cached_input = None
        self._cached_output = None
        self._model = None
        self._patched = False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _active(self) -> bool:
        return self._warmup < self._step <= self._total_steps - self._cooldown

    def _patch(self):
        if self._patched:
            return
        for i in range(self._start, self._end):
            blk = self._model.blocks[i]
            self._orig_forwards[i] = blk.forward
            blk.forward = self._hook(blk, i)
        self._patched = True

    def _hook(self, block, idx: int):
        orig = self._orig_forwards[idx]

        def forward(x, t_emb, mod_segments, rope_freqs, precomputed=None):
            if idx == self._start:
                fresh = self._step != self._consumed_step
                if fresh:
                    self._consumed_step = self._step
                    if (
                        self._active()
                        and self._cached_input is not None
                        and _l1(x, self._cached_input) < self._thresh
                    ):
                        self._skipping = True
                    else:
                        self._skipping = False
                        self._cached_input = x.detach()
                        self._cache_output_pending = True
                else:
                    # Same step, second call (CFG uncond): force compute so the
                    # cache always holds the cond-pass state.
                    self._skipping = False

            if self._skipping:
                result = self._cached_output if idx == self._start else x
                if idx == self._end - 1:
                    self._skipping = False
                return result

            result = orig(x, t_emb, mod_segments, rope_freqs,
                          precomputed=precomputed)

            if idx == self._end - 1 and self._cache_output_pending:
                self._cached_output = result.detach()
                self._cache_output_pending = False

            return result

        # The data-dependent skip/compute decision cannot be traced by Dynamo.
        return torch._dynamo.disable(forward)
