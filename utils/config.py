"""Model configuration for the MiniMax H3 pipeline.

Defaults match the official ``MiniMaxH3DiTModel`` config.json (FL2VA/Ref2VA)
and the ComfyUI PR implementation.  When a checkpoint is loaded, the runtime
overrides these from the state-dict shapes where it can (see
``lifecycle.build_dit_from_checkpoint``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MiniMaxH3DiTConfig:
    """Packed audio-video DiT architecture (official values)."""
    hidden_size: int = 5376
    num_layers: int = 50
    token_refiner_num_layers: int = 2
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_hidden_size: int = 14336
    latents_dim: int = 24
    audio_latents_dim: int = 32
    patch_size: tuple = (1, 2, 2)
    text_dim: int = 5120
    timestep_input_dim: int = 256
    time_embed_hidden_size: int = 5376
    time_embed_dim: int = 2688
    rope_inv_freq_len: int = 16
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5
    sigma_shift_video: float = 12.0
    sigma_shift_audio: float = 3.0
    adaln_curve_grid: Optional[int] = None   # curve-form checkpoint (adaln_t_table)

    @property
    def video_patch_dim(self) -> int:
        pt, ph, pw = self.patch_size
        return self.latents_dim * pt * ph * pw

    @property
    def rope_rot_dim(self) -> int:
        """Rotated dims per head (3 axes x inv_freq_len x 2 halves)."""
        return 3 * self.rope_inv_freq_len * 2

    @property
    def adaln_out_features(self) -> int:
        """Expand=6, modalities=3 -> per-block AdaLN projection output width."""
        return 6 * self.hidden_size * 3
