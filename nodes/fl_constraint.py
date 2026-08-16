"""MiniMax H3 first/last-frame constraint node."""

from __future__ import annotations

import json

_FL_STORE = {}


def _set_fl(node_id: str, data) -> None:
    _FL_STORE[str(node_id)] = data


def _get_fl(node_id: str):
    return _FL_STORE.get(str(node_id))


def _register_route() -> None:
    try:
        from aiohttp import web
        from server import PromptServer

        async def handler(request):
            body = await request.json()
            node_id = str(body.get("node_id", ""))
            data = body.get("data", {})
            _set_fl(node_id, data)
            return web.json_response({"status": "ok"})

        PromptServer.instance.routes.post("/minimax-h3/fl_constraint")(handler)
    except Exception:
        pass


_register_route()


class MiniMaxH3FLConstraint:
    """Group first/last reference frames for the conditioning encoder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("MINIMAX_H3_FL_CONSTRAINT",)
    RETURN_NAMES = ("FL_Constraint",)
    FUNCTION = "make"
    CATEGORY = "MiniMax-H3/data"

    def make(self, first_frame=None, last_frame=None, unique_id=None):
        for name, frame in (("first_frame", first_frame), ("last_frame", last_frame)):
            if frame is not None and frame.ndim != 4:
                raise ValueError(f"MiniMax H3 FL Constraint: {name} must be [B,H,W,C] IMAGE.")
        fl_data = _get_fl(unique_id)
        data = fl_data if isinstance(fl_data, dict) else {}
        return ({
            "first_frame": first_frame,
            "last_frame": last_frame,
            "fl_data": json.dumps(fl_data) if fl_data is not None else "{}",
            # the toggles live in the front-end panel and travel in fl_data;
            # mirror them here for dict-level consumers (conditioning/refiner)
            "offload_dit": bool(data.get("offload_dit", True)),
            "audio_loudness_match": bool(data.get("audio_loudness_match", True)),
        },)
