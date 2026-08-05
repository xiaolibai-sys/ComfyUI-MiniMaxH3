"""MiniMax H3 video batch utility node."""

from __future__ import annotations

FPS = 24
MAX_VIDEOS = 3


class MiniMaxH3VideoBatch:
    """Pack multiple IMAGE video-frame inputs into one independent video batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                f"video_{i}": ("IMAGE",) for i in range(1, MAX_VIDEOS + 1)
            },
        }

    RETURN_TYPES = ("MINIMAX_H3_VIDEO_BATCH",)
    RETURN_NAMES = ("VideoBatch",)
    FUNCTION = "make"
    CATEGORY = "MiniMax-H3/data"

    def make(self, **kwargs):
        videos = []
        for i in range(1, MAX_VIDEOS + 1):
            frames = kwargs.get(f"video_{i}")
            if frames is None:
                continue
            if frames.ndim != 4 or frames.shape[-1] not in (3, 4):
                raise ValueError(
                    f"MiniMax H3 VideoBatch: video_{i} must be [T,H,W,C] IMAGE with 3/4 channels."
                )
            videos.append({
                "frames": frames[..., :3],
                "duration": frames.shape[0] / FPS,
                "label": f"<Video {i}>",
            })
        if not videos:
            raise ValueError("MiniMax H3 VideoBatch: at least one video input is required.")
        return (videos,)
