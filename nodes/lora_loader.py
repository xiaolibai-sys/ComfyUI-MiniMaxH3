"""LoRA loader node for the MiniMax H3 streaming model handle."""

from __future__ import annotations

import folder_paths

from ..models.lora import parse_lora_h3
from ..utils.config import MiniMaxH3DiTConfig
from ..utils.lifecycle import detect_key_prefix, scan_dit_config
from ..utils.stream import BlockReader


class MiniMaxH3LoraLoader:
    """Attach a LoRA to the lazy H3 model handle before sampling."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MINIMAX_H3_MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0,
                                       "step": 0.01}),
            },
        }

    RETURN_TYPES = ("MINIMAX_H3_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "MiniMax-H3/loaders"

    def load(self, model, lora_name, strength=1.0):
        handle = model
        lora_path = folder_paths.get_full_path("loras", lora_name)

        reader = BlockReader(handle.model_path)
        try:
            config = scan_dit_config(reader, MiniMaxH3DiTConfig())
            prefix = detect_key_prefix(reader)
            pruned = config.adaln_curve_grid is not None or \
                reader.has(f"{prefix}adaln_t_table")
        finally:
            reader.close()

        lora = parse_lora_h3(lora_path, strength=strength,
                             silu_grid_path="")
        has_adaln = bool(lora.block_groups and any(
            e.target == "adaln_proj.linear"
            for entries in lora.block_groups.values() for e in entries
        )) or lora.final_adaln is not None
        if pruned and has_adaln and lora.adaln_override is None:
            raise ValueError(
                "MiniMax H3 LoRA loader: only complete pruned LoRA files are "
                "supported. Bake the original 2688-dim AdaLN LoRA into a "
                "complete pruned LoRA first."
            )

        if handle.is_loaded():
            handle.unload()
        handle.loras.add(lora)
        return (handle,)
