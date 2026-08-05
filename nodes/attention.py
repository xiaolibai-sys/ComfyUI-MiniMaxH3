"""Attention backend selection node (BerniniRWrapper-style)."""

from __future__ import annotations

from ..attention.backends import BACKEND_NAMES, available_backends, best_available
from ..utils.types import AttentionConfig


class MiniMaxH3AttentionConfig:
    """Select the attention backend with automatic fallback.

    Chain: SageAttention3 (Blackwell) -> SageAttention2 (sm_90) ->
    SageAttention1 -> FlashAttention -> SDPA-flash (torch precompiled kernel) ->
    xformers -> SDPA (auto) -> SDPA-math (eager, always works).
    Connect to MiniMaxH3Loader's `attn_backend` input.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "backend": (BACKEND_NAMES, {"default": "auto",
                    "tooltip": "Attention backend. auto selects the best available backend: "
                               "Sage3 -> Sage2/1 -> FlashAttn -> SDPA-flash(torch) -> xformers -> SDPA -> SDPA-math"}),
                "force_backend": ("BOOLEAN", {"default": False,
                    "tooltip": "Force the selected backend instead of falling back."}),
            },
        }

    RETURN_TYPES = ("MINIMAX_H3_ATTN",)
    RETURN_NAMES = ("attention_config",)
    FUNCTION = "configure"
    CATEGORY = "MiniMax-H3/loaders"

    def configure(self, backend="auto", force_backend=False):
        cfg = AttentionConfig(
            backend=backend,
            force_backend=force_backend,
            available=tuple(available_backends()),
            best=best_available(),
        )
        return (cfg,)
