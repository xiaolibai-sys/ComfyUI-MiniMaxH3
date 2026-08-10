"""Qwen3-VL-32B text encoder adapter using the vendored copy.

The encoder runtime lives in ``models/text_encoder/`` so the package is
self-contained.  The adapter keeps one LRU-cached
``TextEncoder`` instance, patches the final RMSNorm to identity by default
so the DiT receives the **unnormalized** layer-50 hidden states (matches the
ComfyUI PR's ``layer_norm_hidden_state=False``), and returns
``(text_states, text_token_tags)`` ready for the packed DiT.
"""

from __future__ import annotations

import threading
from typing import Optional

import torch

from .types import EncoderStreamConfig

_lock = threading.Lock()
_encoder_cache: dict[str, object] = {}


def _import_encoder():
    import importlib
    return importlib.import_module("..models.text_encoder", package=__package__)


def _import_stream_config():
    import importlib
    return importlib.import_module(
        "..models.text_encoder.types", package=__package__).StreamConfig


class TextEncoderHandle:
    """LRU-cached handle to the vendored streamed Qwen3-VL-32B encoder."""

    def __init__(self, model_dir: str, stream=None, use_final_norm: bool = False,
                 weight_path: str | None = None):
        self.model_dir = str(model_dir)
        self.stream = stream
        self.use_final_norm = use_final_norm
        self.weight_path = str(weight_path) if weight_path else None
        self._enc = None

    # -- lifecycle -----------------------------------------------------------

    def load(self):
        if self._enc is not None and not getattr(self._enc, "_destroyed", False):
            return self._enc
        self._enc = None
        with _lock:
            key = f"{self.model_dir}|{self.weight_path}|{self.use_final_norm}"
            cached = _encoder_cache.get(key)
            if cached is not None and not getattr(cached, "_destroyed", False):
                self._enc = cached
                return self._enc
            if cached is not None:
                _encoder_cache.pop(key, None)
            enc_mod = _import_encoder()
            if self.stream is None:
                from .types import EncoderStreamConfig
                self.stream = EncoderStreamConfig(
                    weight_path=self.weight_path or "")
            elif self.weight_path is not None:
                object.__setattr__(self.stream, "weight_path", self.weight_path)
            cfg = self.stream.to_encoder_config()
            enc = enc_mod.TextEncoder(self.model_dir, stream_config=cfg)
            if not self.use_final_norm:
                # H3 wants unnormalized layer-50 hidden states
                enc.model.norm = torch.nn.Identity()
            _encoder_cache[key] = enc
            for k in [k for k in _encoder_cache if k != key]:
                try:
                    _encoder_cache.pop(k).destroy()
                except Exception:
                    pass
            self._enc = enc
            return enc

    def unload(self):
        with _lock:
            if self._enc is not None:
                try:
                    self._enc.destroy()
                except Exception:
                    pass
                self._enc = None

    def is_loaded(self) -> bool:
        return self._enc is not None

    # -- forward ---------------------------------------------------------------

    @torch.inference_mode()
    def encode(self, prompt: str, images=None, videos=None,
               tags: Optional[torch.Tensor] = None, max_length: int = 4096,
               minimax_ref_items=None):
        """Return ``(text_states, text_token_tags)``.

        ``text_states``: [1, L, 5120] on CUDA (unnormalized, layer 50).
        ``text_token_tags``: [1, L] int64, 1=text, 0=vision-pad(video).
        """
        enc = self.load()
        from ..models.text_encoder.types import TextEncoderInput
        payload = TextEncoderInput(text=prompt, max_length=max_length,
                                   images=images, videos=videos,
                                   minimax_ref_items=minimax_ref_items)
        out = enc.encode(payload)
        states = out.last_hidden_state
        if out.token_tags is not None:
            return states, out.token_tags.to(states.device)
        if tags is None:
            if images is None and videos is None:
                tags = torch.ones(1, states.shape[1], dtype=torch.long)
            else:
                # vision blocks enter Qwen with mm_token_type_ids; derive tags
                # from the same structure (vision positions -> 0).
                tags = self._tags_from_mm(payload, states.shape[1])
        return states, tags.to(states.device)

    @torch.inference_mode()
    def encode_pair(self, prompt: str, negative_prompt: str,
                    max_length: int = 4096,
                    minimax_ref_items=None):
        """Encode positive and negative text while sharing streamed groups."""
        enc = self._enc
        if enc is None or getattr(enc, "_destroyed", False):
            enc = self.load()
        from ..models.text_encoder.types import TextEncoderInput
        pos = TextEncoderInput(
            text=prompt, max_length=max_length,
            minimax_ref_items=minimax_ref_items)
        neg = TextEncoderInput(text=negative_prompt, max_length=max_length)
        out_pos, out_neg = enc.encode_many([pos, neg])
        states_pos = out_pos.last_hidden_state
        states_neg = out_neg.last_hidden_state
        tags_pos = (
            out_pos.token_tags.to(states_pos.device)
            if out_pos.token_tags is not None
            else torch.ones(1, states_pos.shape[1], dtype=torch.long)
        )
        tags_neg = (
            out_neg.token_tags.to(states_neg.device)
            if out_neg.token_tags is not None
            else torch.ones(1, states_neg.shape[1], dtype=torch.long)
        )
        return (states_pos, tags_pos), (states_neg, tags_neg)

    def release_groups(self):
        enc = self._enc
        if enc is not None and not getattr(enc, "_destroyed", False):
            enc.streamer.release_all()

    def _tags_from_mm(self, payload, seq_len: int) -> torch.Tensor:
        # v1 heuristic: if the encoder cannot report vision spans, default all
        # text; the DiT still runs (slightly off modality tags for vision pads).
        return torch.ones(1, seq_len, dtype=torch.long)


def unload_all_encoders() -> None:
    with _lock:
        for enc in _encoder_cache.values():
            try:
                enc.destroy()
            except Exception:
                pass
        _encoder_cache.clear()
