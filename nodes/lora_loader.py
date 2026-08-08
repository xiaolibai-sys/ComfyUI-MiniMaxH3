"""LoRA loader node for the MiniMax H3 streaming model handle."""

from __future__ import annotations

import os

import folder_paths

from ..models.lora import parse_lora_h3
from ..utils.config import MiniMaxH3DiTConfig
from ..utils.lifecycle import detect_key_prefix, scan_dit_config
from ..utils.stream import BlockReader

_NODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID_TYPES = ["Auto", "FL2VA", "REF2VA"]


def _bundled_grid_path(grid_type: str) -> str:
    name = grid_type.lower()
    path = os.path.join(
        _NODE_ROOT, "assets", "silu_grids",
        f"h3_silu_temb_grid_{name}.safetensors",
    )
    if not os.path.exists(path):
        raise ValueError(f"Missing bundled silu(t_emb) grid: {path}")
    return path


def _auto_grid_path(model_path: str) -> str:
    base = os.path.basename(model_path).lower()
    return _bundled_grid_path("ref2va" if "ref2va" in base else "fl2va")


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
                "silu_grid": (GRID_TYPES, {
                    "default": "Auto",
                    "tooltip": "Grid used by Turbo runtime AdaLN. Auto selects "
                               "REF2VA when the model filename contains ref2va.",
                }),
            },
        }

    RETURN_TYPES = ("MINIMAX_H3_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "MiniMax-H3/loaders"

    def load(self, model, lora_name, strength=1.0, silu_grid="Auto"):
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

        grid_path = ""
        runtime_adaln = pruned and has_adaln and lora.adaln_override is None
        if runtime_adaln:
            grid_path = (
                _auto_grid_path(handle.model_path)
                if silu_grid == "Auto"
                else _bundled_grid_path(silu_grid)
            )
            lora = parse_lora_h3(
                lora_path, strength=strength, silu_grid_path=grid_path)
        elif pruned and has_adaln and lora.adaln_override is None:
            raise ValueError(
                "MiniMax H3 LoRA loader: only complete pruned LoRA files are "
                "supported. Bake the original 2688-dim AdaLN LoRA into a "
                "complete pruned LoRA first."
            )

        if handle.is_loaded():
            handle.unload()
        handle.loras.add(lora)
        return (handle,)
