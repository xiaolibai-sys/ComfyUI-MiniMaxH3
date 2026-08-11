"""Loader nodes: model handle, VAEs, text encoder."""

from __future__ import annotations

from dataclasses import dataclass
import os

import folder_paths

from ..utils.types import EncoderStreamConfig
from ..utils.lifecycle import load_model_handle
from ..utils.encoder_use import TextEncoderHandle


TEXT_ENCODER_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "text_encoder", "ModelsData", "Minimax EncModel")


@dataclass
class VAERef:
    """Paths only; the pack is loaded lazily (and cached) by decode/sample nodes."""
    video_path: str
    audio_path: str = ""


class MiniMaxH3Loader:
    """Create a lazy streaming DiT model handle (weights not loaded yet).

    The BlockSwap layout is chosen at sampling time (KSampler's
    ``block_swap_args`` socket), not here.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"),
                               {"tooltip": "MiniMax H3 DiT checkpoint (plain bf16/fp16 safetensors)"}),
            },
            "optional": {
                "attn_backend": ("MINIMAX_H3_ATTN",),
            },
        }

    RETURN_TYPES = ("MINIMAX_H3_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "MiniMax-H3/loaders"

    def load(self, model_name, attn_backend=None):
        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        override = attn_backend.make_override() if attn_backend is not None else None
        handle = load_model_handle(model_path=model_path, attn_backend=override)
        handle.attn_backend_name = (
            attn_backend.backend if attn_backend is not None else "sageattn2"
        )
        return (handle,)


class MiniMaxH3VAELoader:
    """Reference the video (+optional audio) VAE files. Loaded lazily by decode."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("vae"),
                             {"tooltip": "MiniMax H3 video VAE (.safetensors)"}),
            },
            "optional": {
                "audio_vae_name": (["none"] + folder_paths.get_filename_list("vae"), {"default": "none"}),
            },
        }

    RETURN_TYPES = ("MINIMAX_H3_AV_ENCODER",)
    RETURN_NAMES = ("av_encoder",)
    FUNCTION = "load"
    CATEGORY = "MiniMax-H3/loaders"

    def load(self, vae_name, audio_vae_name="none"):
        video_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        audio_path = ""
        if audio_vae_name != "none":
            audio_path = folder_paths.get_full_path_or_raise("vae", audio_vae_name)
        return (VAERef(video_path=video_path, audio_path=audio_path),)


class MiniMaxH3EncoderLoader:
    """Streamed Qwen3-VL-32B conditioning (vendored encoder).

    ``model_dir`` holds the config/tokenizer (``models/text_encoder/
    ModelsData/Minimax EncModel`` by default); the weights are the selected ComfyUI
    ``text_encoders`` model file (plain or NVFP4/INT8 comfy_quant).  The H3 DiT
    wants the *unnormalized* layer-50 hidden states, so the encoder's final
    RMSNorm is patched to identity by default.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("text_encoders"),
                               {"tooltip": "Qwen3-VL-32B weights (plain, int8 or NVFP4 comfy_quant)"}),
                "use_final_norm": ("BOOLEAN", {"default": False,
                    "tooltip": "False = unnormalized layer-50 hidden states (H3 convention)"}),
                "group_size": ("INT", {"default": 2, "min": 1, "max": 8}),
                "pin_memory": ("BOOLEAN", {"default": True}),
                "disk_workers": ("INT", {"default": 2, "min": 1, "max": 8}),
            },
        }

    RETURN_TYPES = ("MINIMAX_H3_TEXT_ENCODER",)
    RETURN_NAMES = ("text_encoder",)
    FUNCTION = "load"
    CATEGORY = "MiniMax-H3/loaders"

    def load(self, model_name, use_final_norm, group_size,
             pin_memory, disk_workers):
        weight_path = folder_paths.get_full_path_or_raise("text_encoders", model_name)
        stream = EncoderStreamConfig(group_size=group_size, pin_memory=pin_memory,
                                     disk_workers=disk_workers,
                                     weight_path=weight_path)
        return (TextEncoderHandle(model_dir=TEXT_ENCODER_MODEL_DIR, stream=stream,
                                  use_final_norm=use_final_norm),)
