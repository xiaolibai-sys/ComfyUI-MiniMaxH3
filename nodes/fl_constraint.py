"""MiniMax H3 first/last-frame constraint node."""

from __future__ import annotations

class MiniMaxH3FLConstraint:
    """Group first/last reference frames for the conditioning encoder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("MINIMAX_H3_FL_CONSTRAINT",)
    RETURN_NAMES = ("FL_Constraint",)
    FUNCTION = "make"
    CATEGORY = "MiniMax-H3/data"

    def make(self, first_frame=None, last_frame=None):
        for name, frame in (("first_frame", first_frame), ("last_frame", last_frame)):
            if frame is not None and frame.ndim != 4:
                raise ValueError(f"MiniMax H3 FL Constraint: {name} must be [B,H,W,C] IMAGE.")
        if first_frame is None and last_frame is None:
            raise ValueError("MiniMax H3 FL Constraint: at least one of first_frame/last_frame is required.")
        return ({"first_frame": first_frame, "last_frame": last_frame},)
