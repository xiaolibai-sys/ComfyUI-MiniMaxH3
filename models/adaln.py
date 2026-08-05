"""AdaLN modulation baking for inference-only deployments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.types import AdaLNCache, AdaLNCacheEntry, AdaLNCacheKey
from ..utils.stream import BlockReader
from . import quant


def _bind_linear(reader: BlockReader, key: str, in_features: int,
                 out_features: int, dtype: torch.dtype,
                 device: torch.device) -> nn.Linear:
    mod = nn.Linear(in_features, out_features, bias=True,
                    dtype=dtype, device=device)
    spec = quant.load_layer_spec(reader, key, read_weight=True)
    if spec.is_quant:
        qt = quant.make_quantized_tensor(
            spec, tuple(mod.weight.shape), mod.weight.dtype, qdata=spec.qdata
        )
        quant.bind_param(mod, "weight", qt, spec.extra_params)
    else:
        weight = reader.get_tensors([key])[key].to(device, dtype)
        mod.weight = nn.Parameter(weight, requires_grad=False)
    bias_key = key[:-len("weight")] + "bias"
    if reader.has(bias_key):
        bias = reader.get_tensors([bias_key])[bias_key].to(device, dtype)
        mod.bias = nn.Parameter(bias, requires_grad=False)
    else:
        mod.bias = None
    return mod


def _embed_adaln_input(reader, config, prefix, timesteps, dtype, device):
    from .model import TimeEmbedder

    timesteps = sorted({float(t) for t in timesteps})
    t_tensor = torch.tensor(timesteps, dtype=torch.float32, device=device)

    if config.adaln_curve_grid is not None:
        table = reader.get_tensors([f"{prefix}adaln_t_table"])[
            f"{prefix}adaln_t_table"
        ].to(device)
        pos = t_tensor.clamp(0.0, 1.0) * (table.shape[0] - 1)
        i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
        return torch.lerp(
            table[i0], table[i0 + 1], (pos - i0).unsqueeze(1)
        )
    embedder = TimeEmbedder(
        config.timestep_input_dim,
        config.time_embed_hidden_size,
        config.time_embed_dim,
    ).to(device)
    for name in ("proj_in", "proj_out"):
        mod = getattr(embedder, name)
        key = f"{prefix}time_embedder.{name}.weight"
        loaded = _bind_linear(
            reader, key, mod.in_features, mod.out_features,
            torch.float32, device
        )
        mod.weight = loaded.weight
        mod.bias = loaded.bias
    return F.silu(embedder(t_tensor).to(dtype))


def bake_adaln_entry(
    reader: BlockReader,
    config,
    prefix: str,
    timesteps,
    dtype: torch.dtype,
    device: torch.device,
    pbar=None,
    progress_offset: int = 0,
) -> AdaLNCacheEntry:
    """Bake one exact unique-timestep set into block/final modulation tuples."""
    timesteps = sorted({float(t) for t in timesteps})
    if not timesteps:
        return AdaLNCacheEntry(block_mods=(), final_mods=())
    adaln_input = _embed_adaln_input(
        reader, config, prefix, timesteps, dtype, device)

    block_mods = []
    final_mods = []
    total = config.num_layers + 1
    progress = 0
    for i in range(config.num_layers):
        key = f"{prefix}blocks.{i}.adaln_proj.linear.weight"
        if reader.has(key):
            linear = _bind_linear(
                reader,
                key,
                config.time_embed_dim,
                6 * config.hidden_size * 3,
                dtype if config.adaln_curve_grid is None else torch.float32,
                device,
            )
            out = linear(adaln_input)
            out = out.view(len(timesteps) * 3, 6 * config.hidden_size)
            chunks = out.chunk(6, dim=-1)
            block_mods.append(tuple(chunk.detach().cpu() for chunk in chunks))
            del linear
        progress += 1
        if pbar is not None:
            pbar.update_absolute(progress_offset + progress)

    final_key = f"{prefix}final_layer.adaln_proj.linear.weight"
    if reader.has(final_key):
        linear = _bind_linear(
            reader,
            final_key,
            config.time_embed_dim,
            2 * config.hidden_size,
            dtype if config.adaln_curve_grid is None else torch.float32,
            device,
        )
        out = linear(adaln_input)
        out = out.view(len(timesteps), 2 * config.hidden_size)
        chunks = out.chunk(2, dim=-1)
        final_mods = tuple(chunk.detach().cpu() for chunk in chunks)
        del linear
    progress += 1
    if pbar is not None:
        pbar.update_absolute(progress_offset + progress)

    return AdaLNCacheEntry(
        block_mods=tuple(block_mods),
        final_mods=final_mods,
    )


class AdaLNCachePlanner:
    """Build exact AdaLN bake plans from sigmas and packed layouts."""

    def __init__(self, shift_video: float, shift_audio: float):
        self.shift_video = shift_video
        self.shift_audio = shift_audio

    def build(self, sigmas, payload, layout, neg_payload=None, neg_layout=None):
        from .model import plan_timesteps

        plans = []
        seen = set()
        for sigma in sigmas:
            sigma_f = float(sigma)
            candidates = [(payload, layout)]
            if neg_payload is not None:
                candidates.append((neg_payload, neg_layout))
            for plan_payload, plan_layout in candidates:
                _, unique = plan_timesteps(
                    sigma_f,
                    plan_payload,
                    plan_layout,
                    self.shift_video,
                    self.shift_audio,
                )
                has_vis = any(
                    kind in ("cond", "ref_img")
                    for _, _, kind in plan_layout.segments
                )
                has_aud = any(
                    kind == "ref_audio"
                    for _, _, kind in plan_layout.segments
                )
                key = AdaLNCacheKey(
                    sigma=sigma_f,
                    unique_timesteps=tuple(float(t) for t in unique),
                    has_visual_cond=has_vis,
                    has_audio_cond=has_aud,
                )
                if key in seen:
                    continue
                seen.add(key)
                plans.append((key, unique))
        return plans


class AdaLNCacheBaker:
    """Stream AdaLN weights one block at a time and fill AdaLNCache."""

    def __init__(self, reader, config, prefix, dtype, device, pbar=None):
        self.reader = reader
        self.config = config
        self.prefix = prefix
        self.dtype = dtype
        self.device = device
        self.pbar = pbar

    def bake(self, plans) -> AdaLNCache:
        cache = AdaLNCache()
        prepared = []
        for key, timesteps in plans:
            adaln_input = _embed_adaln_input(
                self.reader,
                self.config,
                self.prefix,
                timesteps,
                self.dtype,
                self.device,
            )
            prepared.append((key, adaln_input))
        block_mods = {
            key: [None] * self.config.num_layers for key, _ in prepared
        }
        final_mods = {key: () for key, _ in prepared}
        prepared_keys = [key for key, _ in prepared]
        plan_sizes = [adaln_input.shape[0] for _, adaln_input in prepared]
        stacked_input = (
            torch.cat([adaln_input for _, adaln_input in prepared], dim=0)
            if prepared else None
        )

        total = len(prepared) * (self.config.num_layers + 1)
        if self.pbar is not None and total:
            self.pbar.update_absolute(0)

        linear_dtype = (
            torch.float32
            if self.config.adaln_curve_grid is not None
            else self.dtype
        )
        progress = 0
        for i in range(self.config.num_layers):
            key = f"{self.prefix}blocks.{i}.adaln_proj.linear.weight"
            linear = None
            if self.reader.has(key):
                linear = _bind_linear(
                    self.reader,
                    key,
                    self.config.time_embed_dim,
                    6 * self.config.hidden_size * 3,
                    linear_dtype,
                    self.device,
                )
            if linear is not None:
                out = linear(stacked_input)
                out = out.view(
                    stacked_input.shape[0] * 3,
                    6 * self.config.hidden_size,
                )
                plan_outs = torch.split(
                    out, [size * 3 for size in plan_sizes], dim=0)
                for plan_key, plan_out in zip(prepared_keys, plan_outs):
                    block_mods[plan_key][i] = tuple(
                        chunk.detach().cpu()
                        for chunk in plan_out.chunk(6, dim=-1)
                    )
            else:
                for plan_key in prepared_keys:
                    block_mods[plan_key][i] = ()
            del linear
            progress += len(prepared)
            if self.pbar is not None:
                self.pbar.update_absolute(progress)

        final_key = f"{self.prefix}final_layer.adaln_proj.linear.weight"
        linear = None
        if self.reader.has(final_key):
            linear = _bind_linear(
                self.reader,
                final_key,
                self.config.time_embed_dim,
                2 * self.config.hidden_size,
                linear_dtype,
                self.device,
            )
        if linear is not None:
            out = linear(stacked_input)
            out = out.view(stacked_input.shape[0], 2 * self.config.hidden_size)
            plan_outs = torch.split(out, plan_sizes, dim=0)
            for plan_key, plan_out in zip(prepared_keys, plan_outs):
                final_mods[plan_key] = tuple(
                    chunk.detach().cpu() for chunk in plan_out.chunk(2, dim=-1)
                )
        else:
            for plan_key in prepared_keys:
                final_mods[plan_key] = ()
        del linear
        progress += len(prepared)
        if self.pbar is not None:
            self.pbar.update_absolute(progress)
        for key, _ in prepared:
            cache.add(
                key,
                AdaLNCacheEntry(
                    block_mods=tuple(block_mods[key]),
                    final_mods=final_mods[key],
                ),
            )
        return cache
