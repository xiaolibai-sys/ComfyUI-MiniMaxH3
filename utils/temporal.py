"""MiniMax H3 temporal alignment helpers."""

from __future__ import annotations

FPS = 24
AUDIO_LATENT_FPS = 40


def align_frame_count(n: int) -> int:
    while n % 17 != 5:
        n += 1
    return n


def video_latent_t(frame_count: int) -> int:
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def temporal_shape(length: int):
    frame_count = align_frame_count(max(5, length))
    duration = frame_count / FPS
    return frame_count, video_latent_t(frame_count), round(duration * AUDIO_LATENT_FPS)


__all__ = [
    "FPS",
    "AUDIO_LATENT_FPS",
    "align_frame_count",
    "video_latent_t",
    "temporal_shape",
]
