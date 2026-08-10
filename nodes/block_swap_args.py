"""BlockSwap config node: frozen H3BlockSwap payload for the KSampler."""

from __future__ import annotations

from ..utils.types import H3BlockSwap


class MiniMaxH3BlockSwapArgs:
    """Ring-buffer BlockSwap configuration for the KSampler."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "block_to_swap": ("INT", {"default": 47, "min": 0, "max": 50,
                    "tooltip": "Number of DiT blocks to swap off GPU (50 total; "
                               "resident blocks = 50 - block_to_swap). 0 keeps all blocks resident."}),
                "hot_blocks": ("INT", {"default": 0, "min": 0, "max": 50,
                    "tooltip": "Leading DiT blocks kept permanently on GPU; these avoid H2D/D2H every pass. "
                               "Effective value is capped at resident window - 1."}),
                "prefetch": ("BOOLEAN", {"default": True,
                    "tooltip": "Prefetch the next block window from disk into RAM in the background."}),
                "prefetch_count": ("INT", {"default": 2, "min": 1, "max": 8,
                    "tooltip": "Number of home slots reserved for disk prefetch."}),
                "pin_memory": ("BOOLEAN", {"default": True,
                    "tooltip": "Use pinned memory for staging/prefetch transfers; the home pool stays pageable."}),
                "disk_workers": ("INT", {"default": 2, "min": 1, "max": 16,
                    "tooltip": "Number of background disk read threads."}),
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16",
                    "tooltip": "DiT compute/storage dtype: bfloat16 recommended, float16 for older GPUs, float32 for debugging."}),
            },
        }

    RETURN_TYPES = ("MINIMAX_H3_SWAP",)
    RETURN_NAMES = ("block_swap_args",)
    FUNCTION = "build"
    CATEGORY = "MiniMax-H3/sampling"

    def build(self, block_to_swap, hot_blocks, prefetch, prefetch_count, pin_memory, disk_workers, dtype):
        return (H3BlockSwap(enabled=True, block_to_swap=block_to_swap, hot_blocks=hot_blocks,
                            prefetch=prefetch,
                            prefetch_count=prefetch_count, pin_memory=pin_memory,
                            disk_workers=disk_workers, dtype=dtype),)
