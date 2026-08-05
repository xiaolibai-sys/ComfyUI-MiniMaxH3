"""Self-contained Qwen3-VL *text* model, ported from
``transformers/models/qwen3_vl/modeling_qwen3_vl.py`` (Apache-2.0).

Only the language-model part is kept (embedding, decoder layers, final norm,
interleaved-MRoPE rotary embeddings).  Attention runs through
``torch.nn.functional.scaled_dot_product_attention`` (no flash-attn / no
transformers internals), so the module can be imported anywhere torch exists.

Weight *loading* is intentionally decoupled: the module graph is built on
``meta`` device and the streaming engine (``stream.py``) rebinds ``.data`` of
each parameter from disk on demand.  Nothing here allocates a 32B checkpoint.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Qwen3VLTextConfig


# ---------------------------------------------------------------------------
# Rotary helpers (identical to transformers)
# ---------------------------------------------------------------------------

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1):
    """Applies Rotary Position Embedding to query/key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads to match the query head count (GQA)."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# ---------------------------------------------------------------------------
# Text layers
# ---------------------------------------------------------------------------

class Qwen3VLTextRMSNorm(nn.Module):
    """RMSNorm equivalent to T5LayerNorm."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3VLTextRotaryEmbedding(nn.Module):
    """Interleaved MRoPE used by Qwen3-VL (text path uses identical t/h/w ids)."""

    def __init__(self, config: Qwen3VLTextConfig, device: Optional[torch.device] = None):
        super().__init__()
        self.config = config
        rope = config.rope_parameters or {}
        self.rope_theta = float(rope.get("rope_theta", 500000.0))
        self.mrope_section = list(rope.get("mrope_section", [24, 20, 20]))
        self.attention_scaling = 1.0
        head_dim = config.head_dim
        # Computed on CPU on purpose: this tensor is tiny and arange/meta do not mix.
        inv_freq = 1.0 / self.rope_theta ** (
            torch.arange(0, head_dim, 2, dtype=torch.int64, device="cpu").float() / head_dim
        )
        if device is not None:
            inv_freq = inv_freq.to(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float()
            .expand(3, position_ids.shape[1], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return (cos.to(dtype=x.dtype), sin.to(dtype=x.dtype))

    @staticmethod
    def apply_interleaved_mrope(freqs: torch.Tensor, mrope_section) -> torch.Tensor:
        """Reorganise [TTT..HHH..WWW] frequency layout into [THWTHW..TT]."""
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t


class Qwen3VLTextAttention(nn.Module):
    """Multi-head attention with per-head QK RMSNorm and GQA."""

    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** (-0.5)
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        hidden = config.hidden_size
        self.q_proj = nn.Linear(hidden, config.num_attention_heads * self.head_dim,
                                bias=config.attention_bias)
        self.k_proj = nn.Linear(hidden, config.num_key_value_heads * self.head_dim,
                                bias=config.attention_bias)
        self.v_proj = nn.Linear(hidden, config.num_key_value_heads * self.head_dim,
                                bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, hidden,
                                bias=config.attention_bias)
        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor,
                position_embeddings,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=attention_mask,
                dropout_p=0.0,
                scale=self.scaling,
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                is_causal=True,
                dropout_p=0.0,
                scale=self.scaling,
            )
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class Qwen3VLTextMLP(nn.Module):
    """SwiGLU feed-forward (gate/up/down projections)."""

    def __init__(self, config: Qwen3VLTextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_proj = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Qwen3VLTextDecoderLayer(nn.Module):
    """Standard pre-norm transformer decoder layer with residual branches."""

    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3VLTextAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3VLTextMLP(config)
        self.input_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor,
                position_embeddings,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# ---------------------------------------------------------------------------
# Text model
# ---------------------------------------------------------------------------

class Qwen3VLTextModel(nn.Module):
    """Qwen3-VL language backbone (embedding + decoder stack + final norm).

    The parameter tensors are placeholders until the streaming engine fills
    them; ``run_layer_group`` lets the encoder drive load/compute/destroy at
    group granularity (mirrors ``BlockSwapManager.prepare(i)`` + ``block(x)``).
    """

    def __init__(self, config: Qwen3VLTextConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3VLTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)

    # -- streaming hooks ------------------------------------------------------

    def run_layer_group(self, hidden_states: torch.Tensor,
                        layer_start: int, layer_end: int,
                        position_embeddings,
                        attention_mask: Optional[torch.Tensor] = None,
                        visual_pos_masks: Optional[torch.Tensor] = None,
                        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None
                        ) -> torch.Tensor:
        """Run ``layers[layer_start:layer_end]`` on already-loaded weights.

        ``deepstack_visual_embeds`` (list indexed by *global* layer index, as in
        transformers) are added onto the visual positions at early layers.
        """
        for gi, layer in enumerate(self.layers[layer_start:layer_end],
                                   start=layer_start):
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )
            if deepstack_visual_embeds is not None and gi < len(deepstack_visual_embeds):
                hidden_states = self._deepstack_process(
                    hidden_states, visual_pos_masks, deepstack_visual_embeds[gi])
        return hidden_states

    @staticmethod
    def _deepstack_process(hidden_states: torch.Tensor,
                           visual_pos_masks: torch.Tensor,
                           visual_embeds: torch.Tensor) -> torch.Tensor:
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        hidden_states[visual_pos_masks, :] = (
            hidden_states[visual_pos_masks, :] + visual_embeds)
        return hidden_states

    def run_all_layers(self, hidden_states: torch.Tensor,
                       position_embeddings,
                       attention_mask: Optional[torch.Tensor] = None,
                       visual_pos_masks: Optional[torch.Tensor] = None,
                       deepstack_visual_embeds: Optional[list[torch.Tensor]] = None
                       ) -> torch.Tensor:
        return self.run_layer_group(hidden_states, 0, len(self.layers),
                                    position_embeddings, attention_mask,
                                    visual_pos_masks, deepstack_visual_embeds)
