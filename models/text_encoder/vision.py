"""Qwen3-VL vision tower, ported from
``transformers/models/qwen3_vl/modeling_qwen3_vl.py`` + ``vision_utils.py``
(Apache-2.0), self-contained on SDPA.

The vision tower is small (~0.5B) and stays **resident** (unlike the streamed
text backbone).  It converts image/video pixel patches into vision embeddings
(``pooler_output``) plus optional ``deepstack_features``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Qwen3VLVisionConfig


# ---------------------------------------------------------------------------
# vision helpers (from transformers/vision_utils.py)
# ---------------------------------------------------------------------------

def get_vision_cu_seqlens(grid_thw: torch.Tensor) -> torch.Tensor:
    """``(total_patches + 1,)`` cumulative sequence boundaries across images."""
    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(dim=0, dtype=torch.int32)
    return F.pad(cu_seqlens, (1, 0), value=0)


def get_vision_position_ids(grid_thw: torch.Tensor,
                            spatial_merge_size: int) -> torch.Tensor:
    """``(total_tokens, 2)`` (h, w) position ids, block-major over merge blocks."""
    device = grid_thw.device
    position_ids = []
    for (t, h, w) in grid_thw.tolist():
        hpos_ids, wpos_ids = torch.meshgrid(
            torch.arange(h, device=device), torch.arange(w, device=device),
            indexing="ij")
        block_shape = (h // spatial_merge_size, spatial_merge_size,
                       w // spatial_merge_size, spatial_merge_size)
        hpos_ids = hpos_ids.reshape(block_shape).transpose(1, 2).flatten()
        wpos_ids = wpos_ids.reshape(block_shape).transpose(1, 2).flatten()
        position_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
    return torch.cat(position_ids, dim=0)


def get_vision_bilinear_indices_and_weights(
        grid_thw: torch.Tensor, num_grid_per_side: int,
        spatial_merge_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear interpolation indices/weights into the positional-embedding table."""
    side = num_grid_per_side
    merge_size = spatial_merge_size
    device = grid_thw.device
    idx_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    weight_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        h_grid = torch.linspace(0, side - 1, h, device=device)
        w_grid = torch.linspace(0, side - 1, w, device=device)
        h_floor, w_floor = h_grid.int(), w_grid.int()
        h_ceil = (h_floor + 1).clamp(max=side - 1)
        w_ceil = (w_floor + 1).clamp(max=side - 1)
        h_frac, w_frac = h_grid - h_floor, w_grid - w_floor
        corners = [
            (h_floor * side)[:, None] + w_floor[None, :],
            (h_floor * side)[:, None] + w_ceil[None, :],
            (h_ceil * side)[:, None] + w_floor[None, :],
            (h_ceil * side)[:, None] + w_ceil[None, :],
        ]
        weights = [
            ((1 - h_frac)[:, None] * (1 - w_frac)[None, :]),
            ((1 - h_frac)[:, None] * w_frac[None, :]),
            (h_frac[:, None] * (1 - w_frac)[None, :]),
            (h_frac[:, None] * w_frac[None, :]),
        ]
        h_idx = torch.arange(h, device=device).view(h // merge_size, merge_size)
        w_idx = torch.arange(w, device=device).view(w // merge_size, merge_size)
        reorder = (h_idx[:, :, None, None] * w + w_idx[None, None, :, :]
                   ).transpose(1, 2).flatten().repeat(t)
        for i in range(4):
            idx_parts[i].append(corners[i].flatten()[reorder])
            weight_parts[i].append(weights[i].flatten()[reorder])
    return (torch.stack([torch.cat(p) for p in idx_parts]),
            torch.stack([torch.cat(p) for p in weight_parts]))


# ---------------------------------------------------------------------------
# layers
# ---------------------------------------------------------------------------

def apply_rotary_pos_emb_vision(q, k, cos, sin):
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half_vision(q) * sin)
    k_embed = (k * cos) + (rotate_half_vision(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


def rotate_half_vision(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def gelu_pytorch_tanh(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


class Qwen3VLVisionMLP(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig):
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(gelu_pytorch_tanh(self.linear_fc1(hidden_state)))


class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim,
                              kernel_size=kernel, stride=kernel, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size,
            self.patch_size, self.patch_size)
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return (position_ids.unsqueeze(-1) * self.inv_freq).flatten(1)


class Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig, use_postshuffle_norm: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size * config.spatial_merge_size ** 2
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(
            self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size)
                      if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class Qwen3VLVisionAttention(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig):
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim ** (-0.5)
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor,
                position_embeddings) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1)
        query_states, key_states, value_states = qkv.permute(1, 0, 2, 3).unbind(0)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(
            query_states, key_states, cos, sin)

        # Per-image attention (matches the non-flash path in transformers).
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        outputs = []
        for q, k, v in zip(
                torch.split(query_states, lengths, dim=0),
                torch.split(key_states, lengths, dim=0),
                torch.split(value_states, lengths, dim=0)):
            q = q.transpose(0, 1).unsqueeze(0)
            k = k.transpose(0, 1).unsqueeze(0)
            v = v.transpose(0, 1).unsqueeze(0)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False,
                                                 scale=self.scaling)
            outputs.append(out.squeeze(0).transpose(0, 1))
        attn_output = torch.cat(outputs, dim=0).reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)


class Qwen3VLVisionBlock(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(config)
        self.mlp = Qwen3VLVisionMLP(config)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor,
                position_embeddings) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


# ---------------------------------------------------------------------------
# vision model
# ---------------------------------------------------------------------------

@dataclass
class VisionOutput:
    last_hidden_state: torch.Tensor          # (total_patches, hidden)
    pooler_output: torch.Tensor              # (total_merged_patches, out_hidden)
    deepstack_features: list[torch.Tensor]   # one per deepstack index


class Qwen3VLVisionModel(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = config.spatial_merge_size ** 2
        self.patch_embed = Qwen3VLVisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings ** 0.5)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            [Qwen3VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=False)
        self.deepstack_visual_indexes = list(config.deepstack_visual_indexes)
        self.deepstack_merger_list = nn.ModuleList([
            Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=True)
            for _ in range(len(self.deepstack_visual_indexes))])

    def forward(self, pixel_values: torch.Tensor,
                grid_thw: torch.Tensor) -> VisionOutput:
        bilinear_indices, bilinear_weights = get_vision_bilinear_indices_and_weights(
            grid_thw, self.num_grid_per_side, self.spatial_merge_size)
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size)
        cu_seqlens = get_vision_cu_seqlens(grid_thw)

        hidden_states = self.patch_embed(pixel_values)
        pos_embeds = (self.pos_embed(bilinear_indices)
                      * bilinear_weights[:, :, None]).sum(0)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)

        rotary_pos_emb = self.rotary_pos_emb(position_ids)
        seq_len = hidden_states.size(0)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        deepstack_feature_lists: list[torch.Tensor] = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens,
                                position_embeddings=position_embeddings)
            if layer_num in self.deepstack_visual_indexes:
                feat = self.deepstack_merger_list[
                    self.deepstack_visual_indexes.index(layer_num)](hidden_states)
                deepstack_feature_lists.append(feat)
        merged = self.merger(hidden_states)
        return VisionOutput(
            last_hidden_state=hidden_states,
            pooler_output=merged,
            deepstack_features=deepstack_feature_lists)
