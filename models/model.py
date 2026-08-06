"""MiniMax H3 packed audio-video DiT — self-contained torch port.

Faithful port of ``comfy/ldm/minimax/model.py`` (ComfyUI PR #15224) with all
ComfyUI imports removed:

* ``operations.RMSNorm`` -> local ``RMSNorm`` (fp32 accumulation)
* ``optimized_attention`` -> ``torch.nn.functional.scaled_dot_product_attention``
* ``comfy.quant_ops.ck.rms_rope_split_half`` -> eager ``_rms_rope_split_half``
  (same split-half pairing semantics as the comfy_kitchen triton kernel)
* patcher_extension / model_prefetch wrappers removed (the block-swap engine
  drives block residency instead)

The forward returns the **flow velocity** on both streams
``(v_video, v_audio)``, i.e. exactly what the ComfyUI model returns
(``[-video_out, -slope_a * audio_out]``).  A sampler integrates
``dX/dsigma = v`` on the video sigma grid; the audio velocity already
includes the schedule-map derivative scaling.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.types import AdaLNCacheKey

from ..utils.config import MiniMaxH3DiTConfig
from ..utils.interrupt import check_interrupt

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
VISUAL_COND_TIMESTEP = 0.999
AUDIO_COND_TIMESTEP = 1.0


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

def time_shift_sigma(sigma, from_shift, to_shift):
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def time_shift_slope(sigma, from_shift, to_shift):
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return (to_shift * (1.0 + (from_shift - 1.0) * base) ** 2) / (from_shift * (1.0 + (to_shift - 1.0) * base) ** 2)


def flow_sigmas(steps: int, shift: float = 12.0, grid: int = 1000) -> torch.Tensor:
    """Official ``normal_scheduler`` sigma grid for the flow model.

    Matches ComfyUI's ``ModelSamplingDiscreteFlow`` + ``CONST`` path:
    sigma_min is the shifted sigma at timestep 1, timesteps are linearly
    spaced from 1000 down to sigma_min*1000, and zero is appended.
    """
    sigma_min = time_shift_sigma(1.0 / grid, 1.0, shift)
    end_t = 1000.0 * sigma_min
    t = torch.linspace(1000.0, end_t, steps)
    sigmas = time_shift_sigma(t / 1000.0, 1.0, shift)
    return torch.cat([sigmas, torch.zeros(1, dtype=sigmas.dtype)])


def plan_timesteps(sigma, payload, layout, shift_v, shift_a):
    if isinstance(sigma, torch.Tensor):
        sigma_v = sigma.flatten()[0].float().clamp(min=1e-6)
    else:
        sigma_v = float(sigma)
        sigma_v = max(sigma_v, 1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - time_shift_sigma(sigma_v, shift_v, shift_a))
    vis_aug = float(payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP))
    aud_aug = float(payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP))
    has_vis_cond = any(kind in ("cond", "ref_img") for _, _, kind in layout.segments)
    has_aud_cond = any(kind == "ref_audio" for _, _, kind in layout.segments)
    seg_t = {
        "text": t_v,
        "video": t_v,
        "audio": t_a,
        "cond": max(t_v, vis_aug),
        "ref_img": max(t_v, vis_aug),
        "ref_audio": max(t_a, aud_aug),
    }
    unique_t = sorted(
        {t_v, t_a}
        | ({seg_t["cond"]} if has_vis_cond else set())
        | ({seg_t["ref_audio"]} if has_aud_cond else set())
    )
    return seg_t, unique_t


# ---------------------------------------------------------------------------
# Patch / pack helpers
# ---------------------------------------------------------------------------

def patchify_video(latent, patch_size=(1, 2, 2)):
    b, c, t_full, h_full, w_full = latent.shape
    pt, ph, pw = patch_size
    t, h, w = t_full // pt, h_full // ph, w_full // pw
    x = latent.reshape(b, c, t, pt, h, ph, w, pw)
    x = torch.einsum("nctrhpwq->nthwcrpq", x)
    return x.reshape(b * t * h * w, c * pt * ph * pw)


def unpatchify_video(rows, t, h, w, c=24, patch_size=(1, 2, 2)):
    pt, ph, pw = patch_size
    x = rows.reshape(-1, t, h, w, c, pt, ph, pw)
    x = torch.einsum("nthwcrpq->nctrhpwq", x)
    return x.reshape(-1, c, t * pt, h * ph, w * pw)


def pack_audio(latent):
    b, c, ch, t = latent.shape
    return latent[0].permute(1, 2, 0).reshape(ch * t, c)


def unpack_audio(rows, ch=2):
    t = rows.shape[0] // ch
    return rows.reshape(ch, t, rows.shape[-1]).permute(2, 0, 1).unsqueeze(0)


def _axis_from_sqrt_area(dim, patch, sqrt_area):
    ratio = dim / sqrt_area
    n = dim // patch
    return (torch.arange(n, dtype=torch.float64) * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0


def _frame_grid(h, w):
    area = math.sqrt(h * w)
    hh, ww = torch.meshgrid(_axis_from_sqrt_area(h, 2, area), _axis_from_sqrt_area(w, 2, area), indexing="ij")
    return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1), _axis_from_sqrt_area(w, 2, area)


def _video_t_spans(n):
    return [FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] for k in range(n)]


def _video_t_grid(n, origin):
    spans = torch.tensor(_video_t_spans(n), dtype=torch.float64)
    return float(origin) + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _audio_grid(cursor, t, w_low, w_high):
    g = torch.zeros(t * 2, 3, dtype=torch.float64)
    g[:, 0] = (cursor + torch.arange(t, dtype=torch.float64)).repeat(2)
    g[:t, 2] = w_low
    g[t:, 2] = w_high
    return g


def _video_grid(vt, frame, cursor):
    g = torch.empty(vt, frame.shape[0], 3, dtype=torch.float64)
    g[:, :, 0] = _video_t_grid(vt, cursor)[:, None]
    g[:, :, 1:] = frame[None]
    return g.reshape(-1, 3)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, elementwise_affine=True, dtype=None):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, dtype=dtype or torch.float32))
        else:
            self.register_parameter("weight", None)

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


def _rms_rope_split_half(q, k, table, q_scale, k_scale, eps, rot_dim):
    """Fused RMSNorm + split-half RoPE (matches comfy_kitchen rms_rope_split_half).

    q/k: [S, heads, dim].  Rotation pairs first-half dim i with second-half
    dim i: out0 = c*x0 - s*x1, out1 = s*x0 + c*x1; dims beyond ``rot_dim``
    pass through.  table: [1, S, 1, rot/2, 2, 2] with rows [[c, -s], [s, c]].
    """
    dim = q.shape[-1]
    q = F.rms_norm(q, (dim,), q_scale, eps)
    k = F.rms_norm(k, (dim,), k_scale, eps)
    if rot_dim and rot_dim < dim:
        n_pairs = rot_dim // 2
        c = table[0, :, 0, :, 0, 0].unsqueeze(1)   # [S, 1, n_pairs]
        s = table[0, :, 0, :, 1, 0].unsqueeze(1)
        q0, q1 = q[..., :n_pairs], q[..., n_pairs:rot_dim]
        k0, k1 = k[..., :n_pairs], k[..., n_pairs:rot_dim]
        qr = torch.cat([c * q0 - s * q1, s * q0 + c * q1], dim=-1)
        kr = torch.cat([c * k0 - s * k1, s * k0 + c * k1], dim=-1)
        q = torch.cat([qr, q[..., rot_dim:]], dim=-1)
        k = torch.cat([kr, k[..., rot_dim:]], dim=-1)
    else:
        n_pairs = rot_dim // 2
        c = table[0, :, 0, :, 0, 0].unsqueeze(1)
        s = table[0, :, 0, :, 1, 0].unsqueeze(1)
        q0, q1 = q[..., :n_pairs], q[..., n_pairs:]
        k0, k1 = k[..., :n_pairs], k[..., n_pairs:]
        q = torch.cat([c * q0 - s * q1, s * q0 + c * q1], dim=-1)
        k = torch.cat([c * k0 - s * k1, s * k0 + c * k1], dim=-1)
    return q, k


class TimeEmbedder(nn.Module):
    def __init__(self, freq_dim, hidden, out):
        super().__init__()
        self.freq_dim = freq_dim
        self.proj_in = nn.Linear(freq_dim, hidden, bias=True)
        self.proj_out = nn.Linear(hidden, out, bias=True)

    def forward(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t.to(torch.float32)[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.proj_out(F.silu(self.proj_in(emb)))


def rope_rotation_table(angles, dtype):
    half = angles.shape[-1] // 2
    ang = angles[:, :half]
    c, s = torch.cos(ang), torch.sin(ang)
    table = torch.stack([c, -s, s, c], dim=-1).reshape(1, angles.shape[0], 1, half, 2, 2)
    return table.to(dtype)


class Attention(nn.Module):
    def __init__(self, hidden, heads, head_dim, eps, dtype=None):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.qkv_proj = nn.Linear(hidden, inner * 3, bias=False, dtype=dtype)
        self.q_norm = RMSNorm(head_dim, eps=eps, dtype=dtype)
        self.k_norm = RMSNorm(head_dim, eps=eps, dtype=dtype)
        self.out_proj = nn.Linear(inner, hidden, bias=False, dtype=dtype)

    def forward(self, x, rope_freqs=None):
        s = x.shape[0]
        q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
        v = v.view(s, self.heads, self.head_dim)
        if rope_freqs is not None:
            q = q.view(1, s, self.heads, self.head_dim)
            k = k.view(1, s, self.heads, self.head_dim)
            rot = rope_freqs.shape[-3] * 2
            q, k = _rms_rope_split_half(q[0], k[0], rope_freqs,
                                        self.q_norm.weight.to(q.dtype), self.k_norm.weight.to(k.dtype),
                                        self.q_norm.eps, rot)
        else:
            q = self.q_norm(q.view(s, self.heads, self.head_dim))
            k = self.k_norm(k.view(s, self.heads, self.head_dim))
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        backend = getattr(self, "_backend", None)
        if backend is not None:
            out = backend(q, k, v, self.heads)          # [1, S, heads*dim]
            return self.out_proj(out[0])
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(1, s, -1)   # [1, S, heads*dim]
        return self.out_proj(out.squeeze(0))


class MLP(nn.Module):
    def __init__(self, hidden, ffn, dtype=None):
        super().__init__()
        self.fc1 = nn.Linear(hidden, ffn * 2, bias=False, dtype=dtype)
        self.fc2 = nn.Linear(ffn, hidden, bias=False, dtype=dtype)

    def forward(self, x):
        gate, up = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * up)


class AdalnProj(nn.Module):
    def __init__(self, t_dim, hidden, expand, modalities, apply_silu=True, dtype=None):
        super().__init__()
        self.expand = expand
        self.modalities = modalities
        self.hidden = hidden
        self.apply_silu = apply_silu
        self.linear = nn.Linear(t_dim, expand * hidden * modalities, bias=True, dtype=dtype)

    def forward(self, t_emb):
        x = self.linear(F.silu(t_emb) if self.apply_silu else t_emb)
        x = x.view(x.shape[0] * self.modalities, self.expand * self.hidden)
        return x.chunk(self.expand, dim=-1)


def _mod_scale_shift(h, shift, scale, segments):
    for a, b, row in segments:
        h[a:b].mul_(1.0 + scale[row].to(h.dtype)).add_(shift[row].to(h.dtype))
    return h


def _mod_gate(x, gate, other, segments):
    for a, b, row in segments:
        x[a:b].addcmul_(other[a:b], gate[row].to(x.dtype))
    return x


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

class RefinerBlock(nn.Module):
    def __init__(self, hidden, heads, head_dim, ffn, eps, qk_eps, dtype=None):
        super().__init__()
        self.norm1 = RMSNorm(hidden, eps=eps, dtype=dtype)
        self.norm2 = RMSNorm(hidden, eps=eps, dtype=dtype)
        self.attn = Attention(hidden, heads, head_dim, qk_eps, dtype=dtype)
        self.mlp = MLP(hidden, ffn, dtype=dtype)

    def forward(self, x):
        x = self.attn(self.norm1(x)).add_(x)
        return self.mlp(self.norm2(x)).add_(x)


class TokenRefiner(nn.Module):
    def __init__(self, num_layers, hidden, heads, head_dim, ffn, eps, qk_eps, final_eps, dtype=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            RefinerBlock(hidden, heads, head_dim, ffn, eps, qk_eps, dtype=dtype)
            for _ in range(num_layers)])
        self.final_norm = RMSNorm(hidden, eps=final_eps, dtype=dtype)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


class DiTBlock(nn.Module):
    def __init__(self, hidden, heads, head_dim, ffn, t_dim, eps, qk_eps,
                 apply_silu=True, adaln_dtype=None, dtype=None,
                 include_adaln=True):
        super().__init__()
        self.norm1 = RMSNorm(hidden, eps=eps, dtype=dtype)
        self.norm2 = RMSNorm(hidden, eps=eps, dtype=dtype)
        self.attn = Attention(hidden, heads, head_dim, qk_eps, dtype=dtype)
        self.mlp = MLP(hidden, ffn, dtype=dtype)
        self.adaln_proj = (
            AdalnProj(t_dim, hidden, 6, 3, apply_silu=apply_silu,
                      dtype=adaln_dtype if adaln_dtype is not None else dtype)
            if include_adaln else None
        )

    def forward(self, x, t_emb, mod_segments, rope_freqs, precomputed=None):
        if precomputed is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = precomputed
        elif self.adaln_proj is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        else:
            raise ValueError("DiTBlock requires AdaLN weights or precomputed modulations.")
        h = _mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)
        x = _mod_gate(x, gate_msa, self.attn(h, rope_freqs=rope_freqs), mod_segments)
        h = _mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments)
        return _mod_gate(x, gate_mlp, self.mlp(h), mod_segments)


class FinalLayer(nn.Module):
    def __init__(self, hidden, t_dim, video_dim, audio_dim, eps, apply_silu=True,
                 adaln_dtype=None, dtype=None, include_adaln=True):
        super().__init__()
        self.norm = RMSNorm(hidden, eps=eps, dtype=dtype)
        self.adaln_proj = (
            AdalnProj(t_dim, hidden, 2, 1, apply_silu=apply_silu,
                      dtype=adaln_dtype if adaln_dtype is not None else dtype)
            if include_adaln else None
        )
        self.video_out = nn.Linear(hidden, video_dim, bias=True, dtype=torch.float32)
        self.audio_out = nn.Linear(hidden, audio_dim, bias=True, dtype=torch.float32)

    def forward(self, x, t_emb, video_seg, audio_seg, precomputed=None):
        if precomputed is not None:
            shift, scale = precomputed
        elif self.adaln_proj is not None:
            shift, scale = self.adaln_proj(t_emb)
        else:
            raise ValueError("FinalLayer requires AdaLN weights or precomputed modulations.")
        va, vb, vrow = video_seg
        aa, ab, arow = audio_seg
        hv = (self.norm(x[va:vb]) * (1.0 + scale[vrow]) + shift[vrow]).to(torch.float32)
        ha = (self.norm(x[aa:ab]) * (1.0 + scale[arow]) + shift[arow]).to(torch.float32)
        return self.video_out(hv), self.audio_out(ha)


# ---------------------------------------------------------------------------
# Packed layout
# ---------------------------------------------------------------------------

class PackedLayout:
    """Static packed-sequence structure for one shape/conditioning signature."""

    def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes=None, refs=None, frame_count=None):
        frame, w_grid = _frame_grid(latent_h, latent_w)
        frame_rows = frame.shape[0]

        segments = [("text", text_len)]
        g = torch.zeros(text_len, 3, dtype=torch.float64)
        g[:, 0] = torch.arange(text_len, dtype=torch.float64)
        pos = [g]
        img_pos, img_update = [], []
        audio_pos, audio_update = [], []
        cursor = text_len
        row = text_len

        if keyframes:
            for kf in keyframes:
                pixel_index = kf["resolved_frame_index"]
                if pixel_index == 0:
                    cond_t = float(text_len)
                elif frame_count is not None and pixel_index == frame_count - 1:
                    cond_t = float(text_len) + sum(_video_t_spans(latent_t)) - FRAME_RESCALE
                else:
                    raise ValueError("only first/last keyframe anchors are supported")
                g = torch.empty(frame_rows, 3, dtype=torch.float64)
                g[:, 0] = cond_t
                g[:, 1:] = frame
                segments.append(("cond", frame_rows))
                pos.append(g)
                img_pos.append(torch.arange(row, row + frame_rows))
                img_update.append(torch.zeros(frame_rows, dtype=torch.bool))
                row += frame_rows

        target_audio_w = (float(w_grid[0]), float(w_grid[-1]))
        if refs:
            cursor = float(text_len)
            for blk in refs:
                kind = blk["kind"]
                if kind == "image":
                    r_frame, _ = _frame_grid(blk["latent_h"], blk["latent_w"])
                    n = r_frame.shape[0]
                    g = torch.empty(n, 3, dtype=torch.float64)
                    g[:, 0] = cursor
                    g[:, 1:] = r_frame
                    segments.append(("ref_img", n))
                    pos.append(g)
                    img_pos.append(torch.arange(row, row + n))
                    img_update.append(torch.zeros(n, dtype=torch.bool))
                    row += n
                    cursor += 1.0
                elif kind == "audio":
                    rt = blk["ref_audio_t"]
                    if rt > 0:
                        segments.append(("ref_audio", rt * 2))
                        pos.append(_audio_grid(cursor, rt, *target_audio_w))
                        audio_pos.append(torch.arange(row, row + rt * 2))
                        audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                        row += rt * 2
                    cursor += float(rt)
                elif kind in ("video", "video_audio"):
                    rt = blk["ref_audio_t"]
                    vt = blk["latent_t"]
                    r_frame, r_w_grid = _frame_grid(blk["latent_h"], blk["latent_w"])
                    if rt > 0:
                        segments.append(("ref_audio", rt * 2))
                        pos.append(_audio_grid(cursor, rt, float(r_w_grid[0]), float(r_w_grid[-1])))
                        audio_pos.append(torch.arange(row, row + rt * 2))
                        audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                        row += rt * 2
                    n = vt * r_frame.shape[0]
                    segments.append(("ref_img", n))
                    pos.append(_video_grid(vt, r_frame, cursor))
                    img_pos.append(torch.arange(row, row + n))
                    img_update.append(torch.zeros(n, dtype=torch.bool))
                    row += n
                    cursor += max(float(rt), sum(_video_t_spans(vt)))

        segments.append(("audio", audio_t * 2))
        pos.append(_audio_grid(cursor, audio_t, *target_audio_w))
        audio_pos.append(torch.arange(row, row + audio_t * 2))
        audio_update.append(torch.ones(audio_t * 2, dtype=torch.bool))
        row += audio_t * 2

        n_video = latent_t * frame_rows
        segments.append(("video", n_video))
        pos.append(_video_grid(latent_t, frame, cursor))
        img_pos.append(torch.arange(row, row + n_video))
        img_update.append(torch.ones(n_video, dtype=torch.bool))
        row += n_video

        self.seq_len = row
        self.position_ids = torch.cat(pos)
        self.img_pos = torch.cat(img_pos)
        self.img_update = torch.cat(img_update)
        self.audio_pos = torch.cat(audio_pos)
        self.audio_update = torch.cat(audio_update)
        self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)
        seg_abs = []
        off = 0
        for kind, n in segments:
            seg_abs.append((off, off + n, kind))
            off += n
        self.segments = seg_abs


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MiniMaxH3Model(nn.Module):
    def __init__(self, config: MiniMaxH3DiTConfig, device=None, dtype=None,
                 include_adaln=True, **kwargs):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.device = device
        self.hidden_size = config.hidden_size
        self.patch_size = tuple(config.patch_size)
        self.latents_dim = config.latents_dim
        self.audio_latents_dim = config.audio_latents_dim
        self.sigma_shift_video = config.sigma_shift_video
        self.sigma_shift_audio = config.sigma_shift_audio
        self.use_adaln_curves = config.adaln_curve_grid is not None
        self.include_adaln = include_adaln
        self.adaln_cache = None

        curve = {"apply_silu": not self.use_adaln_curves,
                 "adaln_dtype": torch.float32 if self.use_adaln_curves else dtype}
        video_patch_dim = config.video_patch_dim
        t_dim = config.time_embed_dim

        self.video_patch_proj = nn.Linear(video_patch_dim, config.hidden_size, bias=True, dtype=torch.float32)
        self.audio_patch_proj = nn.Linear(config.audio_latents_dim, config.hidden_size, bias=True, dtype=torch.float32)
        self.condition_proj = nn.Linear(config.text_dim, config.hidden_size, bias=True, dtype=dtype)
        if self.use_adaln_curves:
            self.register_buffer("adaln_t_table", torch.empty(config.adaln_curve_grid, t_dim, dtype=torch.float32))
        else:
            self.time_embedder = TimeEmbedder(config.timestep_input_dim, config.time_embed_hidden_size, t_dim)
        self.rope = nn.Module()
        self.rope.register_buffer("inv_freq", torch.empty(config.rope_inv_freq_len, dtype=torch.float32))
        self.token_refiner = TokenRefiner(config.token_refiner_num_layers, config.hidden_size,
                                          config.num_attention_heads, config.attention_head_dim,
                                          config.ffn_hidden_size, config.norm_eps, config.qk_norm_eps,
                                          config.final_norm_eps, dtype=dtype)
        self.blocks = nn.ModuleList([
            DiTBlock(config.hidden_size, config.num_attention_heads, config.attention_head_dim,
                     config.ffn_hidden_size, t_dim, config.norm_eps, config.qk_norm_eps,
                     **curve, dtype=dtype, include_adaln=include_adaln)
            for _ in range(config.num_layers)])
        self.final_layer = FinalLayer(config.hidden_size, t_dim, video_patch_dim, config.audio_latents_dim,
                                      config.final_norm_eps, **curve, dtype=dtype,
                                      include_adaln=include_adaln)

        if device is not None:
            self.to(device)

    def set_attn_backend(self, fn) -> None:
        """Attach an attention override to every DiT/refiner block (or None)."""
        for blk in self.blocks:
            blk.attn._backend = fn
        for blk in self.token_refiner.blocks:
            blk.attn._backend = fn

    def _precomputed_adaln(self, sigma, unique_t, payload, layout, device, dtype):
        if self.adaln_cache is None:
            return None, None
        has_vis = any(kind in ("cond", "ref_img") for _, _, kind in layout.segments)
        has_aud = any(kind == "ref_audio" for _, _, kind in layout.segments)
        key = AdaLNCacheKey(
            sigma=float(sigma),
            unique_timesteps=tuple(float(t) for t in unique_t),
            has_visual_cond=has_vis,
            has_audio_cond=has_aud,
        )
        entry = self.adaln_cache.entries.get(key)
        if entry is None:
            for cached_key, cached_entry in self.adaln_cache.entries.items():
                if (
                    cached_key.has_visual_cond != has_vis
                    or cached_key.has_audio_cond != has_aud
                    or len(cached_key.unique_timesteps) != len(unique_t)
                ):
                    continue
                if abs(cached_key.sigma - float(sigma)) > 1e-4:
                    continue
                if max(
                    abs(a - b)
                    for a, b in zip(cached_key.unique_timesteps, unique_t)
                ) > 1e-4:
                    continue
                entry = cached_entry
                break
        if entry is None:
            fallback = getattr(self, "_adaln_bake_fallback", None)
            if fallback is not None:
                entry = fallback(key, unique_t)
                if entry is not None:
                    self.adaln_cache.entries[key] = entry
        if entry is None:
            raise ValueError(
                f"MiniMax H3 AdaLN cache missing signature {key}; "
                "re-bake with the current sampler/payload."
            )
        return entry.block_mods, entry.final_mods

    # -- text preprocessing ---------------------------------------------------

    def preprocess_text_embeds(self, text_states):
        if text_states.shape[-1] == self.hidden_size:
            return text_states
        return self.token_refiner(self.condition_proj(text_states[0])).unsqueeze(0)

    # -- rope ----------------------------------------------------------------

    def rope_freqs(self, position_ids, device):
        pos = position_ids.to(torch.float32).to(device)
        inv = self.rope.inv_freq.to(device)
        per_axis = pos.unsqueeze(-1) * inv.view(1, 1, -1)
        t_f, h_f, w_f = per_axis.unbind(dim=1)
        half = torch.cat((t_f, h_f, w_f), dim=-1)
        return torch.cat((half, half), dim=-1)

    # -- conditioning rows -----------------------------------------------------

    def _cond_video_rows(self, payload, device):
        rows = []
        aug = float(payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP))
        seed = int(payload.get("seed", 0))
        for z in payload.get("cond_video_latents", []):
            r = patchify_video(z.to(torch.float32), self.patch_size)
            if aug < 1.0:
                gen = torch.Generator("cpu").manual_seed(seed)
                noise = torch.randn(r.shape, generator=gen, dtype=torch.float32)
                r = aug * r + (1.0 - aug) * noise.to(r.device)
            rows.append(r.to(device))
        return torch.cat(rows, dim=0) if rows else None

    def _cond_audio_rows(self, payload, device):
        rows = []
        aug = float(payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP))
        seed = int(payload.get("seed", 0)) + 1
        for z in payload.get("cond_audio_latents", []):
            r = pack_audio(z.to(torch.float32))
            if aug < 1.0:
                gen = torch.Generator("cpu").manual_seed(seed)
                noise = torch.randn(r.shape, generator=gen, dtype=torch.float32)
                r = aug * r + (1.0 - aug) * noise.to(r.device)
            rows.append(r.to(device))
        return torch.cat(rows, dim=0) if rows else None

    # -- forward ----------------------------------------------------------------

    def velocity(self, video_x, audio_x, sigma, text_states, payload=None):
        """One denoising step: returns (v_video, v_audio) flow velocities.

        ``sigma`` is the *video* sigma in [0,1] (float or 0-d tensor); the
        audio stream's shifted schedule is derived internally.
        """
        payload = payload or {}
        device = video_x.device
        dtype = video_x.dtype if video_x.dtype != torch.float32 else (self.dtype or torch.bfloat16)
        orig_t, orig_h, orig_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
        h_pad, w_pad = (orig_h + self.patch_size[1] - 1) // self.patch_size[1] * self.patch_size[1], \
                       (orig_w + self.patch_size[2] - 1) // self.patch_size[2] * self.patch_size[2]
        if h_pad != orig_h or w_pad != orig_w:
            video_x = F.pad(video_x, (0, w_pad - orig_w, 0, h_pad - orig_h))
        if video_x.shape[0] != 1:
            raise ValueError("MiniMax H3 supports batch size 1")

        latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
        audio_t = audio_x.shape[-1]
        text_len = text_states.shape[1]

        layout = payload.get("layout")
        if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
            layout = PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t,
                                  keyframes=payload.get("keyframes"),
                                  refs=payload.get("refs"),
                                  frame_count=payload.get("frame_count"))
            payload = dict(payload)
            payload["layout"] = layout

        shift_v = float(self.sigma_shift_video)
        shift_a = float(self.sigma_shift_audio)
        if isinstance(sigma, torch.Tensor):
            sigma_v = float(sigma.flatten()[0])
        else:
            sigma_v = float(sigma)
        sigma_v = max(sigma_v, 1e-6)
        seg_t, unique_t = plan_timesteps(sigma_v, payload, layout, shift_v, shift_a)
        t_row = {t: i for i, t in enumerate(unique_t)}
        seg_tag = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0, "ref_audio": 2}

        text_tags = payload.get("text_token_tags")
        mod_segments = []
        for a, b, kind in layout.segments:
            row_base = t_row[seg_t[kind]] * 3
            if kind == "text" and text_tags is not None:
                tags = text_tags.view(-1).tolist()
                run_start = 0
                for i in range(1, b - a + 1):
                    if i == b - a or tags[i] != tags[run_start]:
                        mod_segments.append((a + run_start, a + i, row_base + int(tags[run_start])))
                        run_start = i
            else:
                mod_segments.append((a, b, row_base + seg_tag[kind]))

        img_update = layout.img_update.to(device)
        audio_update = layout.audio_update.to(device)
        video_rows = patchify_video(video_x.to(torch.float32), self.patch_size)
        audio_rows = pack_audio(audio_x.to(torch.float32))
        cond_video_rows = self._cond_video_rows(payload, device)
        cond_audio_rows = self._cond_audio_rows(payload, device)

        all_video_rows = video_rows
        if cond_video_rows is not None:
            all_video_rows = torch.empty(img_update.shape[0], video_rows.shape[1], dtype=torch.float32, device=device)
            all_video_rows[~img_update] = cond_video_rows
            all_video_rows[img_update] = video_rows
        all_audio_rows = audio_rows
        if cond_audio_rows is not None:
            all_audio_rows = torch.empty(audio_update.shape[0], audio_rows.shape[1], dtype=torch.float32, device=device)
            all_audio_rows[~audio_update] = cond_audio_rows
            all_audio_rows[audio_update] = audio_rows

        video_embed = self.video_patch_proj(all_video_rows).to(dtype)
        audio_embed = self.audio_patch_proj(all_audio_rows).to(dtype)
        text_states = text_states.to(dtype)
        if text_states.shape[-1] != self.hidden_size:
            # refiner runs on a single packed sequence [L, D] (mirrors the PR's
            # preprocess_text_embeds path); per-step refinement re-enters here
            text_states = self.token_refiner(
                self.condition_proj(text_states[0])).unsqueeze(0)

        h = torch.empty(layout.seq_len, self.hidden_size, dtype=dtype, device=device)
        voff = aoff = 0
        for a, b, kind in layout.segments:
            n = b - a
            if kind == "text":
                h[a:b] = text_states
            elif kind in ("cond", "ref_img", "video"):
                h[a:b] = video_embed[voff:voff + n]
                voff += n
            else:
                h[a:b] = audio_embed[aoff:aoff + n]
                aoff += n

        t_vals = torch.tensor(unique_t, dtype=torch.float32, device=device)
        if self.use_adaln_curves:
            table = self.adaln_t_table.to(device)
            pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
            i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
            t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
        else:
            t_emb = self.time_embedder(t_vals).to(dtype)

        rope_freqs = rope_rotation_table(self.rope_freqs(layout.position_ids, device), dtype)
        if not self.include_adaln and self.adaln_cache is None:
            raise ValueError(
                "MiniMax H3 model was built without AdaLN weights but no "
                "AdaLN cache was attached."
            )
        swap = getattr(self, "_swap_mgr", None)
        if swap is not None:
            swap.begin()
        precomputed_blocks, precomputed_final = self._precomputed_adaln(
            sigma_v, unique_t, payload, layout, device, dtype)
        if swap is not None:
            for i, block in enumerate(self.blocks):
                if i % 4 == 0:
                    check_interrupt()
                swap.prepare(i)
                block_mods = (
                    tuple(mod.to(device, dtype) for mod in precomputed_blocks[i])
                    if precomputed_blocks is not None else None
                )
                h = block(h, t_emb, mod_segments, rope_freqs,
                          precomputed=block_mods)
                swap.after_compute(i)
            swap.end()
        else:
            for i, block in enumerate(self.blocks):
                if i % 4 == 0:
                    check_interrupt()
                block_mods = (
                    tuple(mod.to(device, dtype) for mod in precomputed_blocks[i])
                    if precomputed_blocks is not None else None
                )
                h = block(h, t_emb, mod_segments, rope_freqs,
                          precomputed=block_mods)

        video_seg = next((a, b, t_row[seg_t["video"]]) for a, b, k in layout.segments if k == "video")
        audio_seg = next((a, b, t_row[seg_t["audio"]]) for a, b, k in layout.segments if k == "audio")
        final_mods = (
            tuple(mod.to(device, dtype) for mod in precomputed_final)
            if precomputed_final is not None else None
        )
        v, a = self.final_layer(h, t_emb, video_seg, audio_seg,
                                precomputed=final_mods)

        video_out = unpatchify_video(v, latent_t, lat_h // 2, lat_w // 2, self.latents_dim, self.patch_size)
        video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
        audio_out = unpack_audio(a)

        slope_a = time_shift_slope(sigma_v, shift_v, shift_a)
        slope_t = torch.tensor(slope_a, dtype=audio_out.dtype, device=audio_out.device)
        return [-video_out.to(video_x.dtype), -slope_t * audio_out.to(audio_x.dtype)]

