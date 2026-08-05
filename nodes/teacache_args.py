"""TeaCache config node: frozen H3TeaCache payload for the KSampler."""

from __future__ import annotations

from ..utils.types import H3TeaCache


class MiniMaxH3TeaCacheArgs:
    """TeaCache: skip near-identical DiT block runs between adjacent steps."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_block": ("INT", {"default": 3, "min": 0, "max": 49,
                    "tooltip": "First block where cache checks are enabled."}),
                "max_skip_blocks": ("INT", {"default": 15, "min": 1, "max": 50,
                    "tooltip": "Maximum number of consecutive blocks skipped on a cache hit."}),
                "rel_l1_thresh": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Reuse the cache when mean L1 distance is below this threshold. "
                               "Higher values are faster but may reduce quality."}),
                "warmup_steps": ("INT", {"default": 1, "min": 0, "max": 100,
                    "tooltip": "Steps at the start that always compute fully."}),
                "cooldown_steps": ("INT", {"default": 2, "min": 0, "max": 100,
                    "tooltip": "Steps at the end that always compute fully."}),
            },
        }

    RETURN_TYPES = ("MINIMAX_H3_TEACACHE",)
    RETURN_NAMES = ("teacache_args",)
    FUNCTION = "build"
    CATEGORY = "MiniMax-H3/sampling"

    def build(self, start_block, max_skip_blocks, rel_l1_thresh, warmup_steps, cooldown_steps):
        return (H3TeaCache(start_block=start_block, max_skip_blocks=max_skip_blocks,
                           rel_l1_thresh=rel_l1_thresh, warmup_steps=warmup_steps,
                           cooldown_steps=cooldown_steps),)
