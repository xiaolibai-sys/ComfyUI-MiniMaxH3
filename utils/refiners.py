"""Refiner adapters for MiniMax H3 prompt polishing."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import tempfile
import time
import urllib.error
import urllib.request

from .reference_roles import reference_role_text
from .prompt_refiner import (
    _OFFICIAL_FORMAT_CONTRACT,
    _SYSTEM_PROMPT,
    _trim_to_fields,
)


_API_BASE = "https://api.minimaxi.com"
_API_KEY_ENV = "IR_KEY"
_POLL_SECONDS = 3.0
_DEFAULT_TIMEOUT = 300.0

_MAX_MEDIA_BYTES = {
    "image": 30 * 1024 * 1024,
    "video": 50 * 1024 * 1024,
    "audio": 15 * 1024 * 1024,
}


def _check_interrupt() -> None:
    try:
        from comfy.model_management import (
            throw_exception_if_processing_interrupted,
        )
        throw_exception_if_processing_interrupted()
    except ImportError:
        pass


def _api_key(
    env_name: str = _API_KEY_ENV,
    service: str = "MiniMax H3-Context-IR Refiner",
) -> str:
    key = os.environ.get(env_name, "").strip()
    if not key:
        raise RuntimeError(
            f"{service}: set the {env_name} environment variable before "
            "running this node."
        )
    return key


def _request_json(url: str, key: str, payload=None):
    _check_interrupt()
    body = json.dumps(payload).encode() if payload is not None else None
    method = "POST" if payload is not None else "GET"
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
        _check_interrupt()
        return data
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"MiniMax H3-Context-IR API error {exc.code}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"MiniMax H3-Context-IR API network error: {exc.reason}"
        ) from exc


def _data_uri(path: str, kind: str) -> str:
    max_bytes = _MAX_MEDIA_BYTES[kind]
    size = os.path.getsize(path)
    if size > max_bytes:
        raise ValueError(
            f"MiniMax H3-Context-IR Refiner: {path} is {size} bytes; "
            f"{kind} files must be <= {max_bytes} bytes for local data-URI upload."
        )
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _image_tensor_uri(image) -> str:
    import numpy as np
    from PIL import Image

    if image.ndim == 4:
        image = image[0]
    array = image[..., :3].detach().cpu().numpy()
    if array.dtype != np.uint8:
        array = (array * 255.0).round().astype(np.uint8)
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        Image.fromarray(array).save(path, format="PNG")
        return _data_uri(path, "image")
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _video_tensor_uri(frames) -> str:
    import av
    import numpy as np

    array = frames[..., :3].detach().cpu().numpy()
    if array.dtype != np.uint8:
        array = (array * 255.0).round().astype(np.uint8)
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        container = av.open(path, "w")
        stream = container.add_stream("h264", rate=24)
        stream.width = array.shape[2]
        stream.height = array.shape[1]
        stream.pix_fmt = "yuv420p"
        for index in range(array.shape[0]):
            frame = av.VideoFrame.from_ndarray(array[index], format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        return _data_uri(path, "video")
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _audio_dict_uri(audio) -> str:
    import torchaudio

    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if waveform.ndim == 3:
        waveform = waveform[0]
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        torchaudio.save(path, waveform.detach().cpu(), sample_rate)
        return _data_uri(path, "audio")
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _audio_path_uri(path: str) -> str:
    """Return an audio data URI, converting non-WAV files for MiniMax upload."""
    if os.path.splitext(path)[1].lower() == ".wav":
        return _data_uri(path, "audio")

    import av
    import numpy as np
    import wave

    with av.open(path) as container:
        stream = container.streams.audio[0]
        chunks = []
        for frame in container.decode(stream):
            array = frame.to_ndarray()
            if array.ndim == 1:
                array = array[None, :]
            chunks.append(array)
        pcm = np.concatenate(chunks, axis=1)
        if pcm.dtype != np.int16:
            pcm = np.clip(pcm * 32768.0, -32768, 32767).astype(np.int16)
        sample_rate = stream.rate

    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        with wave.open(tmp_path, "wb") as handle:
            handle.setnchannels(pcm.shape[0])
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm.tobytes())
        return _data_uri(tmp_path, "audio")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _has_reference_media(package) -> bool:
    if not package:
        return False
    return bool(
        package.get("images") is not None
        or package.get("videos")
        or package.get("audios")
    )


def _add_media_content(content, package, fl_constraint=None) -> None:
    if fl_constraint is not None:
        first = fl_constraint.get("first_frame")
        last = fl_constraint.get("last_frame")
        if first is not None:
            content.append({
                "type": "image_url",
                "image_url": {"url": _image_tensor_uri(first)},
                "role": "first_frame",
            })
        if last is not None:
            content.append({
                "type": "image_url",
                "image_url": {"url": _image_tensor_uri(last)},
                "role": "last_frame",
            })
        return

    if not package:
        return

    image_tensor = package.get("images")
    image_paths = package.get("image_paths") or []
    image_count = image_tensor.shape[0] if image_tensor is not None else len(image_paths)
    for index in range(image_count):
        if index < len(image_paths) and image_paths[index]:
            url = _data_uri(image_paths[index], "image")
        elif image_tensor is not None:
            url = _image_tensor_uri(image_tensor[index:index + 1])
        else:
            continue
        content.append({
            "type": "image_url",
            "image_url": {"url": url},
            "role": "reference_image",
        })

    for video in package.get("videos") or []:
        path = video.get("path")
        url = _data_uri(path, "video") if path else _video_tensor_uri(video["frames"])
        content.append({
            "type": "video_url",
            "video_url": {"url": url},
            "role": "reference_video",
        })

    for audio in package.get("audios") or []:
        path = audio.get("path")
        url = _audio_path_uri(path) if path else _audio_dict_uri(audio)
        content.append({
            "type": "audio_url",
            "audio_url": {"url": url},
            "role": "reference_audio",
        })


def polish_with_context_ir(
    *,
    prompt: str,
    instruction: str = "",
    music_style: str = "",
    package=None,
    fl_constraint=None,
    duration: int | float = 5,
    ratio: str = "16:9",
    env_key: str = _API_KEY_ENV,
    timeout: float = _DEFAULT_TIMEOUT,
) -> str:
    if not prompt.strip():
        return prompt

    has_fl = fl_constraint is not None and bool(
        fl_constraint.get("first_frame") is not None
        or fl_constraint.get("last_frame") is not None
    )
    has_refs = _has_reference_media(package)
    if has_fl and has_refs:
        raise ValueError(
            "MiniMax H3-Context-IR Refiner: first/last frames cannot be mixed "
            "with reference images, videos, or audio."
        )
    has_visual_ref = package is not None and (
        package.get("images") is not None or bool(package.get("videos"))
    )
    has_audio_ref = bool(package and package.get("audios"))
    if has_audio_ref and not has_visual_ref:
        raise ValueError(
            "MiniMax H3-Context-IR Refiner: audio references require at least "
            "one reference image or video."
        )

    text_parts = []
    if instruction.strip():
        text_parts.append(f"User requirement: {instruction.strip()}")
    text_parts.append(prompt.strip())
    if music_style.strip():
        text_parts.append(f"Non-diegetic music requirement: {music_style.strip()}")
    content = [{"type": "text", "text": "\n\n".join(text_parts)}]
    _add_media_content(content, package, fl_constraint=fl_constraint)

    try:
        duration = max(4, min(15, int(round(float(duration)))))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "MiniMax H3-Context-IR Refiner: duration must be a number."
        ) from exc
    if timeout <= 0:
        raise ValueError(
            "MiniMax H3-Context-IR Refiner: timeout must be positive."
        )
    if has_fl:
        ratio = "adaptive"
    elif not has_refs and ratio == "adaptive":
        ratio = "16:9"

    key = _api_key(env_key)
    created = _request_json(
        f"{_API_BASE}/v2/h3_context_ir",
        key,
        {
            "model": "MiniMax-H3",
            "content": content,
            "duration": duration,
            "ratio": ratio,
        },
    )
    task_id = created.get("task_id")
    if not task_id:
        raise RuntimeError(
            f"MiniMax H3-Context-IR Refiner: no task_id in response: {created}"
        )

    deadline = time.monotonic() + timeout
    while True:
        _check_interrupt()
        result = _request_json(
            f"{_API_BASE}/v2/query/video_generation/{task_id}",
            key,
        )
        task = result.get("task", {})
        status = task.get("status")
        if status == "succeeded":
            prompt = task.get("content", {}).get("prompt", "")
            _validate_output_timeline(prompt, float(duration))
            return prompt
        if status in ("failed", "cancelled"):
            raise RuntimeError(
                f"MiniMax H3-Context-IR task {task_id} {status}: {result}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"MiniMax H3-Context-IR task {task_id} timed out after "
                f"{timeout:.0f}s."
            )
        time.sleep(_POLL_SECONDS)
        _check_interrupt()


def _normalize_chat_url(base_url: str) -> str:
    url = (base_url or "").strip().rstrip("/")
    if not url:
        raise ValueError("OpenAI-compatible Refiner: base_url is required.")
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


def _effective_duration(
    frame_count: int | None = None,
    total_duration: float | None = None,
) -> float | None:
    if frame_count is not None:
        return int(frame_count) / 24.0
    if total_duration is not None:
        return float(total_duration)
    return None


def _duration_constraint_text(
    frame_count: int | None,
    total_duration: float | None,
) -> str:
    if frame_count is not None:
        effective = int(frame_count) / 24.0
        detail = f"{effective:.3f} seconds ({int(frame_count)} frames at 24fps)"
        if total_duration is not None:
            detail += f" aligned from {float(total_duration):.3f}s"
        return (
            f"Target video duration: {detail}. All [Shot N] cut times must stay "
            f"within 0.000 and {effective:.3f} seconds; the final shot must end "
            "at the target duration."
        )
    if total_duration is not None:
        duration = float(total_duration)
        return (
            f"Target video duration: {duration:.3f} seconds. All [Shot N] cut "
            f"times must stay within 0.000 and {duration:.3f} seconds; the "
            "final shot must end at the target duration."
        )
    return ""


def _validate_output_timeline(text: str, duration: float | None) -> None:
    if duration is None:
        return
    pattern = re.compile(r"\[Shot \d+\] At (\d{2}):(\d{2})\.(\d{3})")
    for match in pattern.finditer(text):
        seconds = (
            int(match.group(1)) * 60
            + int(match.group(2))
            + int(match.group(3)) / 1000
        )
        if seconds > duration + 1e-3:
            raise RuntimeError(
                "MiniMax Refiner output shot time "
                f"{match.group(0)} exceeds target duration {duration:.3f}s."
            )


def _chat_request_json(url: str, api_key: str, payload: dict, timeout: float):
    _check_interrupt()
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        _check_interrupt()
        return data
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenAI-compatible Refiner API error {exc.code}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"OpenAI-compatible Refiner network error: {exc.reason}"
        ) from exc


def _image_size_text(image) -> str:
    if image is None:
        return ""
    return f"{int(image.shape[2])}x{int(image.shape[1])}"


def _reference_guide_line(label, details, description, note) -> str:
    line = f"{label} = reference metadata"
    if details:
        line += f" ({details})"
    line += f": {description}."
    if note:
        line += f" Note: {note}"
    return line


def _chat_media_payload(
    package,
    fl_constraint=None,
    *,
    supports_image: bool,
    supports_video: bool,
    supports_audio: bool,
) -> tuple[list, str]:
    parts = []
    guide = []
    if fl_constraint is not None:
        first = fl_constraint.get("first_frame")
        last = fl_constraint.get("last_frame")
        if first is not None:
            if supports_image:
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": _image_tensor_uri(first)},
                })
            guide.append(_reference_guide_line(
                "<Picture 1>",
                _image_size_text(first),
                "first frame of the target video",
                "",
            ))
        if last is not None:
            if supports_image:
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": _image_tensor_uri(last)},
                })
            guide.append(_reference_guide_line(
                "<Picture 2>",
                _image_size_text(last),
                "last frame of the target video",
                "",
            ))

    if package:
        image_tensor = package.get("images")
        image_paths = package.get("image_paths") or []
        image_labels = list(package.get("image_labels") or [])
        image_count = (
            image_tensor.shape[0]
            if image_tensor is not None
            else len(image_paths)
        )
        for index in range(image_count):
            role = (
                list(package.get("image_roles") or [])[index]
                if index < len(package.get("image_roles") or [])
                else "reference_image"
            )
            notes = list(package.get("image_notes") or [])
            note = notes[index] if index < len(notes) else ""
            label = (
                image_labels[index]
                if index < len(image_labels)
                else f"<Picture {index + 1}>"
            )
            details = _image_size_text(image_tensor[index]) if image_tensor is not None else ""
            guide.append(_reference_guide_line(
                label,
                details,
                reference_role_text("image", role),
                note,
            ))
            if supports_image:
                if index < len(image_paths) and image_paths[index]:
                    url = _data_uri(image_paths[index], "image")
                elif image_tensor is not None:
                    url = _image_tensor_uri(image_tensor[index:index + 1])
                else:
                    continue
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": url},
                })

        for index, video in enumerate(package.get("videos") or [], 1):
            role = video.get("role") or "reference_video"
            details = []
            if video.get("duration"):
                details.append(f"{float(video['duration']):.1f}s")
            frames = video.get("frames")
            if frames is not None:
                details.append(_image_size_text(frames))
                details.append("24fps")
            guide.append(_reference_guide_line(
                video.get("label") or f"<Video {index}>",
                ", ".join(details),
                reference_role_text("video", role),
                video.get("note") or "",
            ))
            if supports_video:
                path = video.get("path")
                url = (
                    _data_uri(path, "video")
                    if path
                    else _video_tensor_uri(video["frames"])
                )
                parts.append({
                    "type": "video_url",
                    "video_url": {"url": url},
                })

        for index, audio in enumerate(package.get("audios") or [], 1):
            role = audio.get("role") or "reference_audio"
            details = []
            if audio.get("duration"):
                details.append(f"{float(audio['duration']):.1f}s")
            if audio.get("sample_rate"):
                details.append(
                    f"{int(audio['sample_rate']) / 1000:.0f}kHz"
                )
            guide.append(_reference_guide_line(
                audio.get("label") or f"<Audio {index}>",
                ", ".join(details),
                reference_role_text("audio", role),
                audio.get("note") or "",
            ))
            if supports_audio:
                path = audio.get("path")
                url = (
                    _audio_path_uri(path)
                    if path
                    else _audio_dict_uri(audio)
                )
                parts.append({
                    "type": "audio_url",
                    "audio_url": {"url": url},
                })

    guide_text = (
        "Reference media order:\n" + "\n".join(guide)
        if guide else ""
    )
    return parts, guide_text


def _chat_user_text(
    prompt: str,
    instruction: str,
    music_style: str,
    output_mode: str,
    frame_count: int | None = None,
    total_duration: float | None = None,
    ratio: str = "",
    subjects=None,
) -> str:
    parts = []
    if instruction.strip():
        parts.append(f"User requirement: {instruction.strip()}")
    if output_mode and output_mode not in ("auto", "T2VA"):
        parts.append(f"Output mode: {output_mode}")
    if output_mode == "full_reference":
        parts.append(
            "Output exactly these six sections in this order: "
            "subject_definitions, summary, retention_analysis, "
            "detailed_description, overall_soundscape, non_diegetic_music.")
        parts.append(_OFFICIAL_FORMAT_CONTRACT)
        if subjects:
            mapping = ["Global subject mapping:"]
            for index, subject in enumerate(subjects, 1):
                name = str(subject.get("name") or "").strip()
                if not name:
                    continue
                label = str(subject.get("label") or f"<Subject {index}>")
                mapping.append(f"{label} = {name}")
            if len(mapping) > 1:
                mapping.append(
                    "After this mapping, use only these labels in "
                    "detailed_description and retention_analysis; do not "
                    "write natural names in those sections.")
                parts.append("\n".join(mapping))
    elif output_mode in ("I2VA", "FL2VA", "L2VA"):
        parts.append(
            f"Use the official {output_mode} alignment line before the "
            "three core fields.")
    duration_text = _duration_constraint_text(frame_count, total_duration)
    if duration_text:
        parts.append(duration_text)
    if ratio:
        parts.append(f"Aspect ratio: {ratio}")
    parts.append(prompt.strip())
    if music_style.strip():
        parts.append(f"Non-diegetic music requirement: {music_style.strip()}")
    return "\n\n".join(parts)


def polish_with_openai_compatible(
    *,
    prompt: str,
    base_url: str,
    model: str,
    instruction: str = "",
    music_style: str = "",
    package=None,
    fl_constraint=None,
    output_mode: str = "T2VA",
    frame_count: int | None = None,
    total_duration: float | None = None,
    ratio: str = "",
    subjects=None,
    supports_image: bool = False,
    supports_video: bool = False,
    supports_audio: bool = False,
    api_key_env: str = "",
    reasoning: str = "auto",
    reasoning_effort: str = "auto",
    extra_body_json: str = "",
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_tokens: int = 0,
    timeout: float = 120.0,
) -> str:
    if not prompt.strip():
        return prompt
    if not model.strip():
        raise ValueError("OpenAI-compatible Refiner: model is required.")
    if timeout <= 0:
        raise ValueError("OpenAI-compatible Refiner: timeout must be positive.")

    api_key = (
        _api_key(api_key_env, "OpenAI-compatible Refiner")
        if api_key_env.strip()
        else ""
    )
    user_text = _chat_user_text(
        prompt,
        instruction,
        music_style,
        output_mode,
        frame_count=frame_count,
        total_duration=total_duration,
        ratio=ratio,
        subjects=subjects,
    )
    media, media_guide = _chat_media_payload(
        package,
        fl_constraint=fl_constraint,
        supports_image=supports_image,
        supports_video=supports_video,
        supports_audio=supports_audio,
    )
    if media_guide:
        user_text = f"{media_guide}\n\n{user_text}"
    user_content = [{"type": "text", "text": user_text}, *media] if media else user_text
    payload = {
        "model": model.strip(),
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if reasoning != "auto":
        payload["thinking"] = {"type": reasoning}
    if reasoning_effort != "auto":
        payload["reasoning_effort"] = reasoning_effort
    if extra_body_json.strip():
        try:
            extra = json.loads(extra_body_json)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"OpenAI-compatible Refiner: extra_body_json is not valid JSON: {exc}"
            ) from exc
        if not isinstance(extra, dict):
            raise ValueError(
                "OpenAI-compatible Refiner: extra_body_json must be a JSON object."
            )
        payload.update(extra)
    print(
        "[MiniMax H3 OpenAI Refiner] request "
        f"base_url={base_url} model={model} output_mode={output_mode} "
        f"frame_count={frame_count} duration={total_duration} ratio={ratio} "
        f"media={len(media)} prompt={prompt!r}",
        flush=True,
    )
    data = _chat_request_json(
        _normalize_chat_url(base_url),
        api_key,
        payload,
        timeout,
    )
    print(
        "[MiniMax H3 OpenAI Refiner] raw_response "
        f"base_url={base_url} model={model} data={data!r}",
        flush=True,
    )
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"OpenAI-compatible Refiner: unexpected response: {data}"
        ) from exc
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict)
        )
    if (
        not content
        and data["choices"][0].get("finish_reason") == "length"
        and data["choices"][0]["message"].get("reasoning_content")
        and max_tokens
        and max_tokens < 16384
    ):
        max_tokens = min(16384, max(8192, max_tokens * 2))
        payload["max_tokens"] = max_tokens
        print(
            "[MiniMax H3 OpenAI Refiner] retry with larger max_tokens="
            f"{max_tokens} because reasoning consumed the whole budget",
            flush=True,
        )
        data = _chat_request_json(
            _normalize_chat_url(base_url),
            api_key,
            payload,
            timeout,
        )
        print(
            "[MiniMax H3 OpenAI Refiner] raw_response "
            f"base_url={base_url} model={model} data={data!r}",
            flush=True,
        )
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"OpenAI-compatible Refiner: unexpected response: {data}"
            ) from exc
        if isinstance(content, list):
            content = "".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict)
            )
    if not content:
        raise RuntimeError(
            "OpenAI-compatible Refiner returned an empty prompt; "
            f"raw response: {data}"
        )
    print(
        "[MiniMax H3 OpenAI Refiner] response "
        f"base_url={base_url} model={model} text={content!r}",
        flush=True,
    )
    text = _trim_to_fields(content)
    _validate_output_timeline(
        text, _effective_duration(frame_count, total_duration))
    return text
