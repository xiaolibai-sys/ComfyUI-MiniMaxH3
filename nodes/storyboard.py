"""Storyboard planning and deterministic prompt compilation."""

from __future__ import annotations

import re
import json

from ..utils.reference_roles import reference_role_text

_STORYBOARD_STORE = {}

STORYBOARD_MODES = ("T2VA", "full_reference")

DEFAULT_STORYBOARD = {
    "mode": "T2VA",
    "ratio": "16:9",
    "fps": 24,
    "total_duration": 5.0,
    "negative_prompt": "",
    "soundscape": "",
    "music_style": "",
    "subjects": [],
    "shots": [
        {
            "duration": 5.0,
            "prompt": "",
            "camera": "",
            "dialogue": "",
            "sound": "",
        }
    ],
}


def _set_story(node_id: str, data) -> None:
    _STORYBOARD_STORE[str(node_id)] = data


def _get_story(node_id: str):
    return _STORYBOARD_STORE.get(str(node_id))


def _register_route() -> None:
    try:
        from aiohttp import web
        from server import PromptServer

        async def handler(request):
            body = await request.json()
            node_id = str(body.get("node_id", ""))
            data = body.get("data", {})
            _set_story(node_id, data)
            return web.json_response({"status": "ok"})

        PromptServer.instance.routes.post("/minimax-h3/storyboard")(handler)
    except Exception:
        pass


_register_route()


def _normalize_storyboard(data) -> dict:
    data = dict(data or {})
    data.setdefault("mode", DEFAULT_STORYBOARD["mode"])
    if data.get("mode") not in STORYBOARD_MODES:
        data["mode"] = DEFAULT_STORYBOARD["mode"]
    data.setdefault("ratio", DEFAULT_STORYBOARD["ratio"])
    data.setdefault("fps", DEFAULT_STORYBOARD["fps"])
    data.setdefault("total_duration", DEFAULT_STORYBOARD["total_duration"])
    data.setdefault("negative_prompt", "")
    data.setdefault("soundscape", "")
    data.setdefault("music_style", "")
    data.setdefault("subjects", [])
    subjects = []
    for index, raw in enumerate(data["subjects"] or [], 1):
        item = dict(raw or {})
        item["name"] = str(item.get("name") or "").strip()
        item["definition"] = str(item.get("definition") or "").strip()
        item["label"] = f"<Subject {index}>"
        subjects.append(item)
    data["subjects"] = subjects
    if not data.get("shots"):
        data["shots"] = [dict(DEFAULT_STORYBOARD["shots"][0])]
    shots = []
    for index, shot in enumerate(data["shots"], 1):
        item = dict(shot or {})
        item.pop("index", None)
        item.pop("start_time", None)
        item.pop("music_style", None)
        item.pop("refs", None)
        item["duration"] = float(item.get("duration") or 0.0)
        item["prompt"] = str(item.get("prompt") or "")
        item.setdefault("camera", "")
        item.setdefault("dialogue", "")
        item.setdefault("sound", "")
        shots.append(item)
    data["shots"] = shots
    return data


def _format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    whole = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        millis = 0
        whole += 1
    if whole >= 60:
        minutes += 1
        whole -= 60
    return f"{minutes:02d}:{whole:02d}.{millis:03d}"


def _frame_count(total_duration: float, fps: int = 24) -> int:
    count = max(5, round(total_duration * fps))
    while count % 17 != 5:
        count += 1
    return count


def _validate_shots(storyboard: dict) -> None:
    shots = storyboard.get("shots") or []
    total_duration = float(storyboard.get("total_duration") or 0.0)
    if not shots:
        raise ValueError("MiniMax H3 Storyboard: at least one shot is required.")
    if total_duration <= 0:
        raise ValueError("MiniMax H3 Storyboard: total_duration must be positive.")
    last_index = len(shots)
    previous_end = 0.0
    for index, shot in enumerate(shots, 1):
        duration = float(shot.get("duration") or 0.0)
        if duration <= 0.0:
            if index == last_index:
                duration = max(0.0, total_duration - previous_end)
                shot["duration"] = duration
            else:
                raise ValueError(
                    f"MiniMax H3 Storyboard: Shot {index} needs a positive duration."
                )
        if previous_end + duration > total_duration + 1e-6:
            raise ValueError(
                f"MiniMax H3 Storyboard: Shot {index} ends after total_duration."
            )
        previous_end += duration


def _shot_starts(shots: list[dict]) -> list[float]:
    starts = []
    cursor = 0.0
    for shot in shots:
        starts.append(cursor)
        cursor += float(shot.get("duration") or 0.0)
    return starts


def _normalize_media_anchors(text: str) -> str:
    if not text:
        return text
    canonical = {
        "picture": "Picture",
        "video": "Video",
        "audio": "Audio",
        "subject": "Subject",
    }
    return re.sub(
        r"<(picture|video|audio|subject)\s*(\d+)>",
        lambda match: f"<{canonical[match.group(1).lower()]} {match.group(2)}>",
        text,
        flags=re.IGNORECASE,
    )


def _replace_subject_refs(
    text: str,
    subjects: list[dict],
    used: set[str],
    inline_definitions: bool = False,
) -> str:
    if not text:
        return text
    text = _normalize_media_anchors(text)
    if not subjects:
        return text
    refs = []
    for subject in subjects:
        name = str(subject.get("name") or "").strip()
        label = str(subject.get("label") or "")
        if not name and not label:
            continue
        if name:
            refs.append((name, label, subject))
        if label:
            refs.append((label, label, subject))
    if not refs:
        return text
    refs.sort(key=lambda pair: len(pair[0]), reverse=True)
    pattern = re.compile(
        "|".join(
            r"(?<!\w)" + re.escape(raw) + r"(?!\w)"
            for raw, _, _ in refs
        ),
        re.IGNORECASE,
    )

    def _replace(match):
        value = match.group(0)
        for raw, label, subject in refs:
            if value.lower() != raw.lower():
                continue
            name = str(subject.get("name") or "").strip()
            definition = str(subject.get("definition") or "").strip()
            first = subject["label"] not in used
            if first:
                used.add(subject["label"])
            if name and value.lower() == name.lower():
                anchored = f"{name} ({label})"
            else:
                anchored = label
            if inline_definitions and first and definition:
                return f"{anchored}, {definition}"
            return anchored
        return value

    return pattern.sub(_replace, text)


def _normalize_dialogue_tag(tag: str) -> str:
    match = re.match(
        r"<d>\s*(\[[A-Za-z\-]+\])\s*(.*?)</d>",
        tag,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return tag
    return f"<d>{match.group(1)}{match.group(2).strip()}</d>"


def _replace_subject_names_outside_dialogue(
    text: str,
    subjects: list[dict],
    used: set[str],
    inline_definitions: bool,
) -> str:
    if not text:
        return text
    text = re.sub(r"\bsays\s*[：:]\s*", "says: ", text, flags=re.IGNORECASE)
    pattern = re.compile(r"<d>.*?</d>", flags=re.IGNORECASE | re.DOTALL)
    if not pattern.search(text):
        return _replace_subject_refs(
            text, subjects, used, inline_definitions)
    parts = []
    cursor = 0
    for match in pattern.finditer(text):
        parts.append(_replace_subject_refs(
            text[cursor:match.start()], subjects, used, inline_definitions))
        parts.append(_normalize_dialogue_tag(match.group(0)))
        cursor = match.end()
    parts.append(_replace_subject_refs(
        text[cursor:], subjects, used, inline_definitions))
    return "".join(parts)


def _compile_body(storyboard: dict) -> str:
    shots = storyboard.get("shots") or []
    subjects = storyboard.get("subjects") or []
    used_subjects: set[str] = set()
    inline_definitions = str(storyboard.get("mode") or "T2VA") != "full_reference"
    starts = _shot_starts(shots)
    parts = []
    for index, shot in enumerate(shots, 1):
        prompt = _replace_subject_refs(
            str(shot.get("prompt") or "").strip(),
            subjects,
            used_subjects,
            inline_definitions,
        )
        extras = []
        for key in ("camera", "sound"):
            value = _replace_subject_refs(
                str(shot.get(key) or "").strip(),
                subjects,
                used_subjects,
                inline_definitions,
            )
            if value:
                extras.append(value)
        dialogue = _replace_subject_names_outside_dialogue(
            str(shot.get("dialogue") or "").strip(),
            subjects,
            used_subjects,
            inline_definitions,
        )
        if dialogue:
            extras.append(dialogue)
        if extras:
            prompt = " ".join([prompt, *extras]).strip() if prompt else " ".join(extras)
        if not prompt:
            raise ValueError(f"MiniMax H3 Storyboard: Shot {index} prompt is empty.")
        if index == 1:
            parts.append(f"[Shot 1] {prompt}")
        else:
            start = starts[index - 1]
            parts.append(f"[Shot {index}] At {_format_time(start)}, {prompt}")
    return " ".join(parts)


def _package_labels(package) -> dict:
    if not package:
        return {"images": [], "videos": [], "audios": []}
    image_labels = list(package.get("image_labels") or [])
    image_roles = list(package.get("image_roles") or [])
    for index in range(len(image_labels) - len(image_roles)):
        image_roles.append("reference_image")
    images = [
        {
            "label": label,
            "role": image_roles[index],
        }
        for index, label in enumerate(image_labels)
    ]
    videos = [
        {
            "label": item.get("label") or f"<Video {index}>",
            "role": item.get("role") or "reference_video",
        }
        for index, item in enumerate(package.get("videos") or [], 1)
    ]
    audios = [
        {
            "label": item.get("label") or f"<Audio {index}>",
            "role": item.get("role") or "reference_audio",
        }
        for index, item in enumerate(package.get("audios") or [], 1)
    ]
    return {"images": images, "videos": videos, "audios": audios}


def _cited_media_labels(subjects: list[dict]) -> set[str]:
    cited = set()
    for subject in subjects:
        text = _normalize_media_anchors(" ".join(
            str(subject.get(key) or "")
            for key in ("name", "definition")
        ))
        for prefix in ("<Picture ", "<Video ", "<Audio "):
            for match in re.finditer(re.escape(prefix) + r"\d+>", text):
                cited.add(match.group(0))
    return cited


def _subject_definition_lines(package, subjects: list[dict]) -> list[str]:
    lines = []
    for index, subject in enumerate(subjects, 1):
        name = _normalize_media_anchors(str(subject.get("name") or "").strip())
        definition = _normalize_media_anchors(str(subject.get("definition") or "").strip())
        if not name and not definition:
            continue
        label = f"<Subject {index}>"
        if name and definition:
            lines.append(f"{label} is {name}, {definition}.")
        elif name:
            lines.append(f"{label} is {name}.")
        else:
            lines.append(f"{label} is {definition}.")

    refs = _package_labels(package)
    cited = _cited_media_labels(subjects)
    for kind, label_key in (("images", "image"), ("videos", "video"), ("audios", "audio")):
        for item in refs[kind]:
            label = item["label"]
            if label in cited:
                continue
            lines.append(
                f"{label} is {reference_role_text(label_key, item['role'])}."
            )
    return lines


def _subject_definitions(package, subjects: list[dict]) -> str:
    lines = _subject_definition_lines(package, subjects)
    return "\n".join(lines) if lines else "N/A"


def _retention_analysis(package) -> str:
    refs = _package_labels(package)
    lines = []
    for item in refs["images"] + refs["videos"]:
        lines.append(
            f"{item['label']}: weak_reference - the target video follows broad "
            "visual guidance without claiming exact preservation."
        )
    for item in refs["audios"]:
        if item["role"] == "audio_copy":
            lines.append(
                f"{item['label']}: partially_copy - the target audio follows "
                "the reference signal without claiming exact copy semantics."
            )
        else:
            lines.append(
                f"{item['label']}: reference - the target audio follows the "
                "reference's style, rhythm, or timbre without copying the signal."
            )
    return "\n".join(lines) if lines else "N/A"


def compile_storyboard(storyboard, package=None) -> dict:
    storyboard = _normalize_storyboard(storyboard)
    _validate_shots(storyboard)
    mode = str(storyboard.get("mode") or "T2VA")
    total_duration = float(storyboard.get("total_duration") or 0.0)
    subjects = storyboard.get("subjects") or []
    body = _compile_body(storyboard)
    soundscape = str(storyboard.get("soundscape") or "").strip() or "N/A"
    music_style = str(storyboard.get("music_style") or "").strip() or "N/A"
    negative_prompt = str(storyboard.get("negative_prompt") or "").strip()

    if mode == "full_reference":
        text = (
            "subject_definitions:\n"
            f"{_subject_definitions(package, subjects)}\n\n"
            "summary:\n"
            "[reference generation] The target video follows the shot-by-shot "
            "storyboard and uses the provided references as conservative "
            "generation guidance.\n\n"
            "retention_analysis:\n"
            f"{_retention_analysis(package)}\n\n"
            "detailed_description:\n"
            f"{body}\n\n"
            "overall_soundscape: "
            f"{soundscape}\n\n"
            "non_diegetic_music: "
            f"{music_style}"
        )
    else:
        core = (
            "integrated_multimodal_description: "
            f"{body}\n\n"
            "overall_soundscape: "
            f"{soundscape}\n\n"
            "non_diegetic_music: "
            f"{music_style}"
        )
        text = core

    return {
        "text": text,
        "mode": mode,
        "frame_count": _frame_count(total_duration, int(storyboard.get("fps") or 24)),
        "negative_prompt": negative_prompt,
        "ratio": str(storyboard.get("ratio") or "16:9"),
        "total_duration": total_duration,
        "storyboard": storyboard,
        "subjects": subjects,
    }


class MiniMaxH3Storyboard:
    """Structured multi-shot storyboard edited from the node UI."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("MINIMAX_H3_PROMPT",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "make"
    CATEGORY = "MiniMax-H3/data"

    @classmethod
    def IS_CHANGED(cls, unique_id, **kwargs):
        node_id = unique_id[0] if isinstance(unique_id, (list, tuple)) else unique_id
        data = _get_story(node_id)
        if data is None:
            return "default"
        return json.dumps(data, sort_keys=True, ensure_ascii=False)

    def make(self, unique_id):
        data = _get_story(unique_id)
        return (compile_storyboard(
            data if data else DEFAULT_STORYBOARD),)
