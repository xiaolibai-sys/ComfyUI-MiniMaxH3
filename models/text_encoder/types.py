"""Typed data-transfer payloads for the MiniMax H3 Text Encoder.

This module centralises the structured data that flows between the streaming
text encoder and the rest of the MiniMax H3 pipeline.  The style follows
``ComfyUI-BerniniRWrapper/utils/types.py``: frozen dataclasses for immutable
payloads, enums for closed option sets, and small property helpers so the
consumer never repeats dtype/device arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class LoadingMode(str, Enum):
    """How transformer-layer weights are held during the (single) forward pass."""
    STREAMING = "Streaming"   # load group -> compute -> destroy, group by group
    FULL = "Full"             # load everything to GPU once and keep resident


class PoolMode(str, Enum):
    """How the variable-length token sequence is reduced to one conditioning vector."""
    LAST = "last"   # hidden state of the last non-padding token
    MEAN = "mean"   # masked mean over all non-padding tokens


# ---------------------------------------------------------------------------
# Streaming configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StreamConfig:
    """Disk-streaming knobs for the encoder forward pass.

    Mirrors ``BerniniBlockSwap`` from BerniniRWrapper (loading_mode,
    disk_workers, prefetch, pin_memory) with the naming/scope adapted to a
    one-shot text-encoder run.
    """
    group_size: int = 2                 # decoder layers loaded per group
    loading_mode: str = LoadingMode.STREAMING.value
    prefetch: bool = True               # read next group from disk while current computes
    prefetch_count: int = 1             # groups prefetched ahead (currently 1)
    disk_workers: int = 2               # background reader threads
    pin_memory: bool = True             # stage disk reads in pinned CPU buffers
    device: str = "cuda"
    dtype: str = "float32"              # bfloat16 | float16 | float32
    full_precision_mm: bool = True      # dequantize weights and use F.linear (official ComfyUI path)
    layer_prefix: Optional[str] = None  # safetensors key prefix, e.g. "model.language_model.model.layers."
    weight_path: Optional[str] = None   # optional standalone safetensors (ComfyUI model file)

    @property
    def torch_dtype(self) -> torch.dtype:
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.dtype]

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TextEncoderInput:
    """Prompt payload handed to :meth:`TextEncoder.encode`.

    ``input_ids`` may be passed directly (bypassing the tokenizer) — that is
    the path used when the H3 pipeline feeds pre-tokenized text.
    """
    text: str = ""
    input_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    max_length: int = 4096
    # raw multimodal inputs (processed by AutoProcessor when not preprocessed)
    images: Optional[list] = None
    videos: Optional[list] = None
    # ref2va presentation: [{"type": "image"|"video"|"audio", "data": ..., ...}]
    # in request order (official MiniMaxH3Tokenizer minimax_ref_items contract).
    # When set, the encoder builds the non-chat-template <Picture i>/<Video k>/
    # <Audio j> token sequence directly (bypasses AutoProcessor templates).
    minimax_ref_items: Optional[list] = None
    # pre-processed multimodal tensors (bypass processor)
    pixel_values: Optional[torch.Tensor] = None
    image_grid_thw: Optional[torch.Tensor] = None
    pixel_values_videos: Optional[torch.Tensor] = None
    video_grid_thw: Optional[torch.Tensor] = None
    mm_token_type_ids: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class TextEncoderOutput:
    """Encoder result consumed by the downstream H3 generation pipeline.

    ``last_hidden_state`` is the full ``(batch, seq, hidden)`` tensor after the
    final norm; ``pooled_embedding`` is the reduction requested via
    :class:`PoolMode` and is the natural conditioning vector for the DiT.
    """
    last_hidden_state: torch.Tensor
    pooled_embedding: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    token_tags: Optional[torch.Tensor] = None  # [1, L] int64: 1=text, 0=vision pad

    @property
    def embed_dim(self) -> int:
        return self.last_hidden_state.shape[-1]


@dataclass(frozen=True)
class DiskGroupSpec:
    """Static description of one layer-group on disk.

    ``keys`` are the safetensors tensor names that must be resident for
    ``layers[layer_start:layer_end]`` to run.
    """
    group_idx: int
    layer_start: int
    layer_end: int                      # exclusive
    keys: tuple[str, ...] = field(default_factory=tuple)

    @property
    def num_layers(self) -> int:
        return self.layer_end - self.layer_start
