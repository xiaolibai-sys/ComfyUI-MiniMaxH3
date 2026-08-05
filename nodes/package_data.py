"""MiniMax H3 reference package data node."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

import folder_paths

_PACKAGE_STORE = {}

MAX_IMAGES = 9
MAX_VIDEOS = 3
MAX_AUDIOS = 3

VIDEO_DURATION_MIN = 2.0
VIDEO_DURATION_MAX = 15.0
VIDEO_DURATION_TOTAL = 15.0
AUDIO_DURATION_MIN = 2.0
AUDIO_DURATION_MAX = 15.0
AUDIO_DURATION_TOTAL = 15.0
FPS = 24


def _set_package(node_id: str, data) -> None:
    _PACKAGE_STORE[str(node_id)] = data


def _get_package(node_id: str):
    return _PACKAGE_STORE.get(str(node_id), {})


def _package_summary(data) -> dict:
    images = data.get("images") or []
    videos = data.get("videos") or []
    audios = data.get("audios") or []
    return {
        "images": [
            {"label": item.get("label") or f"<Picture {index}>"}
            for index, item in enumerate(images, 1)
        ],
        "videos": [
            {
                "label": item.get("label") or f"<Video {index}>",
                "name": item.get("path") or item.get("name") or "",
            }
            for index, item in enumerate(videos, 1)
        ],
        "audios": [
            {
                "label": item.get("label") or f"<Audio {index}>",
                "name": item.get("path") or item.get("name") or "",
            }
            for index, item in enumerate(audios, 1)
        ],
    }


def _register_route() -> None:
    try:
        from aiohttp import web
        from server import PromptServer

        async def handler(request):
            body = await request.json()
            node_id = str(body.get("node_id", ""))
            data = body.get("data", {})
            _set_package(node_id, data)
            return web.json_response({"status": "ok"})

        PromptServer.instance.routes.post("/minimax-h3/package-data")(handler)

        async def get_handler(request):
            node_id = str(request.query.get("node_id", ""))
            return web.json_response(_package_summary(_get_package(node_id)))

        PromptServer.instance.routes.get("/minimax-h3/package-data")(get_handler)
    except Exception:
        pass


_register_route()


def _load_image(path):
    with Image.open(path) as image:
        arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _load_video_frames(path):
    import av

    with av.open(path) as container:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            raise ValueError(f"No video stream found in {path}")
        frames = []
        for frame in container.decode(video=stream.index):
            arr = frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0
            frames.append(torch.from_numpy(arr))
    if not frames:
        raise ValueError(f"No video frames decoded from {path}")
    return torch.stack(frames)


def _load_audio(path):
    import av

    with av.open(path) as container:
        stream = container.streams.audio[0]
        chunks = []
        for frame in container.decode(stream):
            array = frame.to_ndarray()
            if array.ndim == 1:
                array = array[None, :]
            chunks.append(array)
        if not chunks:
            raise ValueError(f"No audio decoded from {path}")
        pcm = np.concatenate(chunks, axis=1)
        sample_rate = int(stream.rate)

    if pcm.dtype == np.int16:
        waveform = pcm.astype(np.float32) / 32768.0
    elif pcm.dtype == np.int32:
        waveform = pcm.astype(np.float32) / 2147483648.0
    elif pcm.dtype == np.uint8:
        waveform = (pcm.astype(np.float32) - 128.0) / 128.0
    else:
        waveform = pcm.astype(np.float32)
    waveform = np.clip(waveform, -1.0, 1.0)
    waveform = torch.from_numpy(waveform)
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    return waveform, sample_rate


def _validate_image_tensor(name, tensor):
    if tensor.ndim != 4 or tensor.shape[-1] not in (3, 4):
        raise ValueError(f"MiniMax H3 PackageData: {name} must be [B,H,W,C] IMAGE with 3/4 channels.")
    return tensor[..., :3]


def _validate_video_tensor(tensor):
    if tensor.ndim != 4 or tensor.shape[-1] not in (3, 4):
        raise ValueError("MiniMax H3 PackageData: external video must be [T,H,W,C] IMAGE with 3/4 channels.")
    return tensor[..., :3]


class MiniMaxH3PackageData:
    """Merge UI-loaded references with external IMAGE/AUDIO inputs.

    The UI stores selected files in the backend store.  Any connected optional
    inputs are appended to the loaded data, validated, and returned as one
    unified PACKAGE_DATA object.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
            "optional": {
                "images": ("IMAGE",),
                "videos": ("*",),
                "audios": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("PACKAGE_DATA",)
    RETURN_NAMES = ("PackageData",)
    FUNCTION = "make"
    CATEGORY = "MiniMax-H3/data"

    def make(self, unique_id, images=None, videos=None, audios=None):
        data = _get_package(unique_id)

        loaded_images = []
        loaded_image_labels = []
        loaded_image_roles = []
        loaded_image_paths = []
        loaded_image_notes = []
        if images is not None:
            tensor = _validate_image_tensor("images", images)
            for index in range(tensor.shape[0]):
                loaded_images.append(tensor[index:index + 1])
                loaded_image_labels.append(f"<Picture {len(loaded_images)}>")
                loaded_image_roles.append("reference_image")
                loaded_image_paths.append(None)
                loaded_image_notes.append("")
        for item in data.get("images", []) or []:
            loaded_images.append(_load_image(folder_paths.get_annotated_filepath(
                item.get("path") or item.get("name"))))
            loaded_image_labels.append(
                f"<Picture {len(loaded_images)}>"
                if images is not None
                else str(item.get("label") or f"<Picture {len(loaded_images)}>")
            )
        loaded_image_roles.extend(
            str(item.get("role") or "reference_image")
            for item in data.get("images", []) or []
        )
        loaded_image_paths.extend(
            folder_paths.get_annotated_filepath(
                item.get("path") or item.get("name"))
            for item in data.get("images", []) or []
        )
        loaded_image_notes.extend(
            str(item.get("note") or "")
            for item in data.get("images", []) or []
        )
        if len(loaded_images) > MAX_IMAGES:
            raise ValueError(f"MiniMax H3 PackageData: at most {MAX_IMAGES} reference images.")
        if loaded_images:
            shape = loaded_images[0].shape[1:]
            for tensor in loaded_images[1:]:
                if tensor.shape[1:] != shape:
                    raise ValueError("MiniMax H3 PackageData: images must share the same H/W.")
            images_out = torch.cat(loaded_images, dim=0)
        else:
            images_out = None
        image_labels = loaded_image_labels[:len(loaded_images)]
        image_roles = loaded_image_roles[:len(loaded_images)]
        image_paths = loaded_image_paths[:len(loaded_images)]
        image_notes = loaded_image_notes[:len(loaded_images)]

        videos_out = []
        if videos is not None:
            if isinstance(videos, torch.Tensor):
                frames = _validate_video_tensor(videos)
                duration = frames.shape[0] / FPS
                videos_out.append({
                    "frames": frames,
                    "duration": duration,
                    "role": "reference_video",
                    "path": None,
                    "note": "",
                    "label": f"<Video {len(videos_out) + 1}>",
                })
            elif isinstance(videos, (list, tuple)):
                for item in videos:
                    item = dict(item)
                    item.setdefault("role", "reference_video")
                    item.setdefault("path", None)
                    item.setdefault("note", "")
                    item.setdefault("label", f"<Video {len(videos_out) + 1}>")
                    videos_out.append(item)
            else:
                raise ValueError("MiniMax H3 PackageData: videos must be IMAGE or MINIMAX_H3_VIDEO_BATCH.")
        for item in data.get("videos", []) or []:
            frames = _load_video_frames(folder_paths.get_annotated_filepath(
                item.get("path") or item.get("name")))
            duration = float(item.get("duration", 0.0)) or frames.shape[0] / FPS
            videos_out.append({
                "frames": frames,
                "duration": duration,
                "role": str(item.get("role") or "reference_video"),
                "path": folder_paths.get_annotated_filepath(
                    item.get("path") or item.get("name")),
                "note": str(item.get("note") or ""),
                "label": str(item.get("label") or f"<Video {len(videos_out) + 1}>"),
            })
        if len(videos_out) > MAX_VIDEOS:
            raise ValueError(f"MiniMax H3 PackageData: at most {MAX_VIDEOS} reference videos.")
        for index, item in enumerate(videos_out):
            if not (VIDEO_DURATION_MIN <= item["duration"] <= VIDEO_DURATION_MAX):
                raise ValueError(
                    f"MiniMax H3 PackageData: video {index + 1} must be "
                    f"{VIDEO_DURATION_MIN:.0f}-{VIDEO_DURATION_MAX:.0f}s."
                )
        if videos_out and sum(v["duration"] for v in videos_out) > VIDEO_DURATION_TOTAL:
            raise ValueError(f"MiniMax H3 PackageData: videos total {VIDEO_DURATION_TOTAL:.0f}s max.")

        audios_out = []
        if audios is not None:
            waveform = audios["waveform"]
            sample_rate = int(audios["sample_rate"])
            if waveform.ndim != 3:
                raise ValueError("MiniMax H3 PackageData: external audio waveform must be [B,C,L].")
            for batch in range(waveform.shape[0]):
                duration = waveform.shape[-1] / sample_rate
                audios_out.append({
                    "waveform": waveform[batch:batch + 1],
                    "sample_rate": sample_rate,
                    "duration": duration,
                    "role": "reference_audio",
                    "path": None,
                    "note": "",
                    "label": f"<Audio {len(audios_out) + 1}>",
                })
        for item in data.get("audios", []) or []:
            waveform, sample_rate = _load_audio(folder_paths.get_annotated_filepath(
                item.get("path") or item.get("name")))
            duration = float(item.get("duration", 0.0)) or waveform.shape[-1] / sample_rate
            audios_out.append({
                "waveform": waveform,
                "sample_rate": sample_rate,
                "duration": duration,
                "role": str(item.get("role") or "reference_audio"),
                "path": folder_paths.get_annotated_filepath(
                    item.get("path") or item.get("name")),
                "note": str(item.get("note") or ""),
                "label": str(item.get("label") or f"<Audio {len(audios_out) + 1}>"),
            })
        if len(audios_out) > MAX_AUDIOS:
            raise ValueError(f"MiniMax H3 PackageData: at most {MAX_AUDIOS} reference audio clips.")
        for index, item in enumerate(audios_out):
            if not (AUDIO_DURATION_MIN <= item["duration"] <= AUDIO_DURATION_MAX):
                raise ValueError(
                    f"MiniMax H3 PackageData: audio {index + 1} must be "
                    f"{AUDIO_DURATION_MIN:.0f}-{AUDIO_DURATION_MAX:.0f}s."
                )
        if audios_out and sum(a["duration"] for a in audios_out) > AUDIO_DURATION_TOTAL:
            raise ValueError(f"MiniMax H3 PackageData: audio total {AUDIO_DURATION_TOTAL:.0f}s max.")

        return ({
            "images": images_out,
            "image_labels": image_labels,
            "image_roles": image_roles,
            "image_paths": image_paths,
            "image_notes": image_notes,
            "videos": [{
                "frames": item["frames"],
                "duration": item["duration"],
                "role": item["role"],
                "path": item.get("path"),
                "note": item.get("note") or "",
                "label": item.get("label") or f"<Video {n}>",
            } for n, item in enumerate(videos_out, 1)],
            "audios": [{
                "waveform": item["waveform"],
                "sample_rate": item["sample_rate"],
                "duration": item["duration"],
                "role": item["role"],
                "path": item.get("path"),
                "note": item.get("note") or "",
                "label": item.get("label") or f"<Audio {n}>",
            } for n, item in enumerate(audios_out, 1)],
            "limits": {
                "images": MAX_IMAGES,
                "videos": MAX_VIDEOS,
                "audios": MAX_AUDIOS,
                "video_duration_min": VIDEO_DURATION_MIN,
                "video_duration_max": VIDEO_DURATION_MAX,
                "video_duration_total": VIDEO_DURATION_TOTAL,
                "audio_duration_min": AUDIO_DURATION_MIN,
                "audio_duration_max": AUDIO_DURATION_MAX,
                "audio_duration_total": AUDIO_DURATION_TOTAL,
            },
        },)
