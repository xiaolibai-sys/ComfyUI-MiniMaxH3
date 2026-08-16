"""BS.1770 K-weighted gated loudness matching for rolling audio segments.

Each rolling segment is sampled independently, so decoded audio energy can
jump at segment boundaries.  This module measures per-segment integrated
loudness (ITU-R BS.1770, K-weighted + gated so pauses/silence do not skew
the measurement) and applies clamped, peak-limited, boundary-ramped gains.
"""

from __future__ import annotations

import logging
import math

import torch

logger = logging.getLogger("h3.loudness")

# ITU-R BS.1770 K-weighting analog prototypes (sample-rate independent).
_SHELF_F0 = 1681.974450955533
_SHELF_GAIN_DB = 3.999843853973347
_SHELF_Q = 0.7071752369554196
_HP_F0 = 38.13547087613982
_HP_Q = 0.5003270373238773

_BLOCK_SECONDS = 0.4          # 400 ms gating blocks
_BLOCK_OVERLAP = 0.75         # 75% overlap
_ABSOLUTE_GATE_LUFS = -70.0   # below this a block/segment counts as silence
_RELATIVE_GATE_LU = 10.0      # blocks this far below the mean are gated out
_LOUDNESS_OFFSET = -0.691


def _rbj_high_shelf(fs: float):
    a_lin = 10.0 ** (_SHELF_GAIN_DB / 40.0)
    w0 = 2.0 * math.pi * _SHELF_F0 / fs
    alpha = math.sin(w0) / (2.0 * _SHELF_Q)
    cos_w0 = math.cos(w0)
    shelf = 2.0 * math.sqrt(a_lin) * alpha
    b0 = a_lin * ((a_lin + 1) + (a_lin - 1) * cos_w0 + shelf)
    b1 = -2.0 * a_lin * ((a_lin - 1) + (a_lin + 1) * cos_w0)
    b2 = a_lin * ((a_lin + 1) + (a_lin - 1) * cos_w0 - shelf)
    a0 = (a_lin + 1) - (a_lin - 1) * cos_w0 + shelf
    a1 = 2.0 * ((a_lin - 1) - (a_lin + 1) * cos_w0)
    a2 = (a_lin + 1) - (a_lin - 1) * cos_w0 - shelf
    return [b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0]


def _rbj_high_pass(fs: float):
    w0 = 2.0 * math.pi * _HP_F0 / fs
    alpha = math.sin(w0) / (2.0 * _HP_Q)
    cos_w0 = math.cos(w0)
    b0 = (1.0 + cos_w0) / 2.0
    b1 = -(1.0 + cos_w0)
    b2 = b0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    return [b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0]


def _k_weight(x: torch.Tensor, sample_rate: int) -> torch.Tensor:
    from scipy.signal import lfilter

    b1, a1 = _rbj_high_shelf(sample_rate)
    b2, a2 = _rbj_high_pass(sample_rate)
    y = lfilter(b1, a1, x.numpy(), axis=-1)
    y = lfilter(b2, a2, y, axis=-1)
    return torch.from_numpy(y).float()


def integrated_loudness(waveform: torch.Tensor, sample_rate: int = 32000) -> float:
    """BS.1770-5 integrated (gated) loudness in LUFS.

    Accepts [T], [C, T] or [1, C, T]; channel weights are 1.0 (mono/stereo).
    Returns ``-inf`` for silence or clips shorter than one 400 ms block.
    """
    x = waveform.detach().float().cpu()
    if x.ndim == 3:
        x = x[0]
    if x.ndim == 1:
        x = x.unsqueeze(0)
    block = int(_BLOCK_SECONDS * sample_rate)
    if x.shape[-1] < block:
        return float("-inf")
    y = _k_weight(x, sample_rate)
    hop = int(block * (1.0 - _BLOCK_OVERLAP))
    blocks = y.unfold(-1, block, hop)                 # [C, n_blocks, block]
    z = blocks.pow(2).mean(dim=-1).sum(dim=0)         # per-block energy, ch-summed
    gate_abs = 10.0 ** ((_ABSOLUTE_GATE_LUFS - _LOUDNESS_OFFSET) / 10.0)
    z = z[z > gate_abs]
    if z.numel() == 0:
        return float("-inf")
    ungated = _LOUDNESS_OFFSET + 10.0 * math.log10(z.mean().item())
    gate_rel = 10.0 ** (
        (ungated - _RELATIVE_GATE_LU - _LOUDNESS_OFFSET) / 10.0)
    z = z[z > gate_rel]
    if z.numel() == 0:
        return float("-inf")
    return _LOUDNESS_OFFSET + 10.0 * math.log10(z.mean().item())


def _true_peak(waveform: torch.Tensor, sample_rate: int = 32000) -> float:
    """Approximate true peak via 4x polyphase oversampling.

    Sample-peak detection under-reads inter-sample peaks by up to ~1 dB on
    HF-heavy (sibilant) content; the gain path must not trust it.
    """
    from scipy.signal import resample_poly

    x = waveform.detach().float().cpu().numpy()
    y = resample_poly(x, 4, 1, axis=-1)
    return float(abs(y).max()) if y.size else 0.0


def _crossfade_join(chunks, sample_rate: int, fade_ms: float = 20.0):
    """Overlap-join chunks with short equal-power crossfades.

    Independently generated segments splice with arbitrary waveform
    phase/value at the cut, which a gain envelope cannot fix - the click
    is sample-level.  A ~20 ms cos/sin overlap removes it; the duration
    cost (one fade per join) is far below perceptual A/V sync thresholds.
    """
    fade = int(sample_rate * fade_ms / 1000.0)
    out = chunks[0]
    for nxt in chunks[1:]:
        f = min(fade, out.shape[-1], nxt.shape[-1])
        if f < 2:
            out = torch.cat([out, nxt], dim=-1)
            continue
        t = torch.linspace(0.0, math.pi / 2.0, f, dtype=out.dtype)
        mixed = (out[..., -f:] * torch.cos(t)
                 + nxt[..., :f] * torch.sin(t))
        out = torch.cat([out[..., :-f], mixed, nxt[..., f:]], dim=-1)
    return out


def match_segment_loudness(
    chunks,
    sample_rate: int = 32000,
    max_gain_db: float = 12.0,
    attack_ms: float = 100.0,
    release_ms: float = 1000.0,
    peak_ceiling: float = 0.99,
) -> torch.Tensor:
    """Gain-match per-segment loudness and concatenate.

    Gains are matched (BS.1770 integrated) to the first non-silent segment,
    clamped to +/-max_gain_db and true-peak limited against clipping.  The
    correction is applied as a leveler-style gain envelope: constant per
    segment with smoothstep transitions centered on each boundary - fast
    attack when turning down, slow release when turning up.  Every join
    additionally gets a short equal-power crossfade so splice clicks are
    removed even when no gain correction is needed.
    """
    chunks = [c for c in chunks if c is not None and c.shape[-1] > 0]
    if not chunks:
        return torch.zeros(1, 2, 0)
    if len(chunks) == 1:
        return chunks[0]
    loudness = [integrated_loudness(c, sample_rate) for c in chunks]
    reference = next((v for v in loudness if v > _ABSOLUTE_GATE_LUFS), None)
    if reference is None:
        return _crossfade_join(chunks, sample_rate)
    max_gain = 10.0 ** (max_gain_db / 20.0)
    gains = []
    caps = []
    for chunk, seg_lufs in zip(chunks, loudness):
        if seg_lufs <= _ABSOLUTE_GATE_LUFS:
            gains.append(1.0)
            caps.append(float("inf"))
            continue
        gain = 10.0 ** ((reference - seg_lufs) / 20.0)
        gain = min(max(gain, 1.0 / max_gain), max_gain)
        true_peak = _true_peak(chunk, sample_rate)
        if true_peak * gain > peak_ceiling:
            gain = peak_ceiling / true_peak
        gains.append(gain)
        caps.append(
            peak_ceiling / true_peak if true_peak > 0 else float("inf"))
    if all(abs(20.0 * math.log10(g)) < 0.2 for g in gains):
        return _crossfade_join(chunks, sample_rate)
    logger.info(
        "rolling FL2VA: segment loudness (LUFS) %s, applied gains (dB) %s",
        [round(v, 1) for v in loudness],
        [round(20.0 * math.log10(g), 1) for g in gains],
    )
    total = sum(c.shape[-1] for c in chunks)
    env = torch.empty(total, dtype=chunks[0].dtype)
    bounds = []
    offset = 0
    for chunk, gain in zip(chunks, gains):
        n = chunk.shape[-1]
        env[offset:offset + n] = gain
        offset += n
        bounds.append(offset)
    for i in range(1, len(chunks)):
        g0, g1 = gains[i - 1], gains[i]
        if g0 == g1:
            continue
        ms = attack_ms if g1 < g0 else release_ms
        half = int(sample_rate * ms / 2000.0)
        lo = max(0, bounds[i - 1] - half)
        hi = min(total, bounds[i - 1] + half)
        if hi - lo > 1:
            t = torch.linspace(0.0, 1.0, hi - lo, dtype=env.dtype)
            smooth = t * t * (3.0 - 2.0 * t)
            env[lo:hi] = g0 + (g1 - g0) * smooth
    out = []
    offset = 0
    for chunk, cap in zip(chunks, caps):
        n = chunk.shape[-1]
        e = env[offset:offset + n]
        if cap != float("inf"):
            # transitions toward a lower gain exceed the steady-state cap
            # near the boundary; clamp the envelope so nothing can clip
            e = e.clamp(max=cap)
        out.append(chunk * e.view(1, 1, n))
        offset += n
    return _crossfade_join(out, sample_rate)


__all__ = ["integrated_loudness", "match_segment_loudness"]
