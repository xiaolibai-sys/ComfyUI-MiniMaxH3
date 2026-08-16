"""MiniMax H3 refiner nodes."""

from __future__ import annotations

import json

import folder_paths


def _has_reference_media(package) -> bool:
    if not package:
        return False
    return bool(
        package.get("images") is not None
        or package.get("videos")
        or package.get("audios")
    )


def _load_fl_image_tensor(info):
    if info is None:
        return None
    try:
        import numpy as np
        import torch
        from PIL import Image

        path = folder_paths.get_annotated_filepath(info.get("name"))
        with Image.open(path) as image:
            array = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)
    except Exception:
        return None


def _fl_data(fl_constraint) -> dict:
    if not isinstance(fl_constraint, dict):
        return {}
    raw = fl_constraint.get("fl_data")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _frame_number_at(time: float, fps: int) -> int:
    from .conditioning import align_frame_count

    raw = round(float(time) * int(fps or 24))
    if raw <= 0:
        return 0
    return align_frame_count(raw)


def _build_rolling_segments(fl_constraint, base_prompt: str, fps: int = 24):
    """Return LLM-ready rolling segment descriptors and the global negative prompt."""
    data = _fl_data(fl_constraint)
    keyframes = data.get("keyframes") or []
    if len(keyframes) < 2:
        return [], str(data.get("global_negative_prompt") or "")
    fps = int(data.get("fps") or fps or 24)

    first_frame = fl_constraint.get("first_frame") if isinstance(fl_constraint, dict) else None
    last_frame = fl_constraint.get("last_frame") if isinstance(fl_constraint, dict) else None
    total_duration = float(keyframes[-1].get("time") or 0.0)
    base_prompt = str(base_prompt or "").strip()
    segments = []

    for index in range(len(keyframes) - 1):
        start = keyframes[index]
        end = keyframes[index + 1]
        start_time = float(start.get("time") or 0.0)
        end_time = float(end.get("time") or start_time + 1.0)

        start_image = _load_fl_image_tensor(start.get("image"))
        if start_image is None and abs(start_time) < 1e-6:
            start_image = first_frame
        end_image = _load_fl_image_tensor(end.get("image"))
        if end_image is None and abs(end_time - total_duration) < 1e-6:
            end_image = last_frame

        segment_prompt = " ".join(
            part for part in (base_prompt, str(start.get("prompt") or ""))
            if part.strip()
        )
        segments.append({
            "index": index + 1,
            "start_keyframe_index": index + 1,
            "end_keyframe_index": index + 2,
            "start_time": start_time,
            "end_time": end_time,
            "start_frame": _frame_number_at(start_time, fps),
            "end_frame": _frame_number_at(end_time, fps),
            "start_image": start_image,
            "end_image": end_image,
            "start_image_exists": start_image is not None,
            "end_image_exists": end_image is not None,
            "start_image_note": str(start.get("note") or ""),
            "end_image_note": str(end.get("note") or ""),
            "prompt": segment_prompt,
            "negative_prompt": str(start.get("negative_prompt") or ""),
        })

    return segments, str(data.get("global_negative_prompt") or "")


def _fl_timeline_metadata(fl_constraint):
    """Return (frame_count, total_duration) derived from FL keyframes."""
    data = _fl_data(fl_constraint)
    keyframes = data.get("keyframes") or []
    if len(keyframes) < 2:
        return None, None
    fps = int(data.get("fps") or 24)
    duration = float(keyframes[-1].get("time") or 0.0)
    if duration <= 0:
        return None, None
    return _frame_number_at(duration, fps), duration


def _refiner_preview(prompt_ref, text) -> str:
    return (
        "MiniMax H3 Refiner\n"
        f"mode: {prompt_ref.get('mode')}\n"
        f"frame_count: {prompt_ref.get('frame_count')}\n"
        f"total_duration: {prompt_ref.get('total_duration')}\n"
        f"ratio: {prompt_ref.get('ratio')}\n\n"
        f"prompt:\n{text}"
    )


def _refiner_result(prompt_ref, text):
    preview = _refiner_preview(prompt_ref, text)
    return {"ui": {"text": [preview]}, "result": (prompt_ref, text)}


def _mode_from_fl_constraint(fl_constraint) -> str | None:
    if fl_constraint is None:
        return None
    first = fl_constraint.get("first_frame") is not None
    last = fl_constraint.get("last_frame") is not None
    if first and last:
        return "FL2VA"
    if first:
        return "I2VA"
    if last:
        return "L2VA"
    return None


def _prepare_refiner_input(
    prompt,
    package,
    fl_constraint=None,
):
    if prompt is not None:
        prompt_ref = prompt if isinstance(prompt, dict) else {}
        text = str(prompt_ref.get("text") or "")
        mode = str(prompt_ref.get("mode") or "")
        if not mode:
            if fl_constraint is not None:
                first = fl_constraint.get("first_frame") is not None
                last = fl_constraint.get("last_frame") is not None
                mode = "FL2VA" if first and last else (
                    "I2VA" if first else "L2VA")
            elif _has_reference_media(package):
                mode = "full_reference"
            else:
                mode = "T2VA"
        ratio = str(prompt_ref.get("ratio") or "16:9")
        if mode in ("I2VA", "FL2VA", "L2VA"):
            ratio = "adaptive"
        elif mode == "T2VA" and ratio == "adaptive":
            ratio = "16:9"
        fl_mode = _mode_from_fl_constraint(fl_constraint)
        if fl_mode:
            mode = fl_mode
            ratio = "adaptive"
        return (
            text,
            mode,
            prompt_ref.get("frame_count"),
            prompt_ref.get("total_duration"),
            ratio,
        )

    return "", "T2VA", None, None, "16:9"


class MiniMaxH3SimplePrompt:
    """Simple single-text prompt for direct Refiner workflows."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "mode": (["T2VA", "I2VA", "FL2VA", "L2VA", "full_reference"],
                         {"default": "T2VA"}),
                "total_duration": ("FLOAT", {
                    "default": 5.0, "min": 1.0, "max": 15.0, "step": 0.1}),
                "ratio": (["adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"], {
                    "default": "16:9"}),
            },
            "optional": {
                "negative_text": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Optional negative prompt attached to this conditioning."}),
            },
        }

    RETURN_TYPES = ("MINIMAX_H3_PROMPT", "STRING")
    RETURN_NAMES = ("prompt", "text")
    FUNCTION = "make"
    CATEGORY = "MiniMax-H3/prompt"

    def make(self, text="", mode="T2VA", total_duration=5.0, ratio="16:9",
             negative_text=""):
        from .conditioning import align_frame_count

        text = str(text or "")
        total_duration = max(1.0, float(total_duration or 0.0))
        frame_count = align_frame_count(round(total_duration * 24))
        return ({
            "text": text,
            "mode": mode,
            "frame_count": frame_count,
            "total_duration": total_duration,
            "ratio": ratio,
            "negative_prompt": str(negative_text or ""),
        }, text)


class MiniMaxH3ContextIRRefiner:
    """Official MiniMax H3-Context-IR refiner.

    The API key is read from the ``IR_KEY`` environment variable at execution
    time, so it never appears in workflow JSON or node inputs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "instruction": ("STRING", {
                    "default": "Make it more cinematic, detailed and temporally clear.",
                    "multiline": True,
                    "tooltip": "Instruction for the official refiner."}),
                "music_style": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Non-diegetic music requirement. Empty = N/A."}),
            },
            "optional": {
                "prompt": ("MINIMAX_H3_PROMPT",),
                "package": ("PACKAGE_DATA",),
                "timeout": ("INT", {"default": 300, "min": 10, "max": 3600}),
            },
        }

    RETURN_TYPES = ("MINIMAX_H3_PROMPT", "STRING")
    RETURN_NAMES = ("prompt", "text")
    FUNCTION = "polish"
    CATEGORY = "MiniMax-H3/conditioning"
    OUTPUT_NODE = True

    def polish(self, instruction="", music_style="", prompt=None,
               package=None, timeout=300):
        prompt_obj = prompt
        source_negative = (
            prompt_obj.get("negative_prompt")
            if isinstance(prompt_obj, dict)
            else ""
        )
        prompt, output_mode, frame_count, duration, ratio = _prepare_refiner_input(
            prompt, package)
        if not prompt.strip():
            return _refiner_result({
                "text": prompt, "mode": output_mode,
                "frame_count": frame_count, "total_duration": duration,
                "ratio": ratio, "negative_prompt": source_negative}, prompt)

        from ..utils.refiners import polish_with_context_ir

        text = polish_with_context_ir(
            prompt=prompt,
            instruction=instruction,
            music_style=music_style,
            package=package,
            duration=duration or 5,
            ratio=ratio,
            timeout=timeout,
        )
        return _refiner_result({
            "text": text, "mode": output_mode,
            "frame_count": frame_count, "total_duration": duration,
            "ratio": ratio, "negative_prompt": source_negative}, text)


class MiniMaxH3OpenAICompatibleRefiner:
    """OpenAI-compatible refiner for local vLLM or third-party chat APIs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "instruction": ("STRING", {
                    "default": "Make it more cinematic, detailed and temporally clear.",
                    "multiline": True}),
                "music_style": ("STRING", {
                    "default": "", "multiline": True}),
                "base_url": ("STRING", {
                    "default": "http://127.0.0.1:8000/v1",
                    "tooltip": "OpenAI-compatible endpoint, e.g. http://host:8000/v1"}),
                "model": ("STRING", {
                    "default": "",
                    "tooltip": "Model or deployment name served by the endpoint."}),
            },
            "optional": {
                "prompt": ("MINIMAX_H3_PROMPT",),
                "package": ("PACKAGE_DATA",),
                "fl_constraint": ("MINIMAX_H3_FL_CONSTRAINT",),
                "api_key_env": ("STRING", {
                    "default": "",
                    "tooltip": "Environment variable name holding the API key. Empty = no auth."}),
                "supports_image": ("BOOLEAN", {"default": False}),
                "supports_video": ("BOOLEAN", {"default": False}),
                "supports_audio": ("BOOLEAN", {"default": False}),
                "reasoning": (["auto", "enabled", "disabled"], {
                    "default": "auto",
                    "tooltip": "DeepSeek thinking mode. Auto leaves the "
                               "request unchanged."}),
                "reasoning_effort": (["auto", "low", "high", "max"], {
                    "default": "auto",
                    "tooltip": "DeepSeek V4 reasoning effort. Auto omits it."}),
                "extra_body_json": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Optional provider-specific JSON object merged "
                               "into the request body, e.g. "
                               '{"enable_thinking": true}.'}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0,
                                          "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0,
                                    "step": 0.01}),
                "max_tokens": ("INT", {
                    "default": 0, "min": 0, "max": 16384,
                    "tooltip": "0 = no token cap. Positive values cap output tokens."}),
                "timeout": ("INT", {"default": 120, "min": 10, "max": 3600}),
            },
        }

    RETURN_TYPES = ("MINIMAX_H3_PROMPT", "STRING")
    RETURN_NAMES = ("prompt", "text")
    FUNCTION = "polish"
    CATEGORY = "MiniMax-H3/conditioning"
    OUTPUT_NODE = True

    def polish(self, instruction="", music_style="", base_url="", model="",
               prompt=None, package=None,
               fl_constraint=None, api_key_env="",
               supports_image=False, supports_video=False, supports_audio=False,
               reasoning="auto", reasoning_effort="auto", temperature=1.0,
               top_p=0.95, max_tokens=0, timeout=120, extra_body_json=""):
        prompt_obj = prompt
        source_negative = (
            prompt_obj.get("negative_prompt")
            if isinstance(prompt_obj, dict)
            else ""
        )
        prompt, output_mode, frame_count, duration, ratio = _prepare_refiner_input(
            prompt, package, fl_constraint)

        rolling_segments, global_negative = _build_rolling_segments(
            fl_constraint, prompt)
        has_rolling_segments = bool(
            rolling_segments
            and any(
                str(segment.get("prompt") or "").strip()
                or segment.get("start_image") is not None
                or segment.get("end_image") is not None
                for segment in rolling_segments
            )
        )
        if has_rolling_segments:
            rolling_frame_count, rolling_duration = _fl_timeline_metadata(
                fl_constraint)
            if rolling_frame_count is not None:
                frame_count = rolling_frame_count
            if rolling_duration is not None:
                duration = rolling_duration
            output_mode = "FL2VA"
            ratio = "adaptive"
        if not prompt.strip() and not has_rolling_segments:
            return _refiner_result({
                "text": prompt, "mode": output_mode,
                "frame_count": frame_count, "total_duration": duration,
                "ratio": ratio, "negative_prompt": source_negative}, prompt)

        from ..utils.refiners import (
            polish_with_openai_compatible,
            polish_with_openai_compatible_rolling,
        )

        subjects = (
            prompt_obj.get("subjects")
            if isinstance(prompt_obj, dict) else None
        )
        if has_rolling_segments:
            refined_prompts, refined_negatives = (
                polish_with_openai_compatible_rolling(
                    prompt=prompt,
                    base_url=base_url,
                    model=model,
                    instruction=instruction,
                    music_style=music_style,
                    rolling_segments=rolling_segments,
                    global_negative_prompt=global_negative,
                    base_negative_prompt=(
                        prompt_obj.get("negative_prompt")
                        if isinstance(prompt_obj, dict)
                        else ""
                    ),
                    output_mode=output_mode,
                    ratio=ratio,
                    supports_image=supports_image,
                    supports_video=supports_video,
                    supports_audio=supports_audio,
                    api_key_env=api_key_env,
                    reasoning=reasoning,
                    reasoning_effort=reasoning_effort,
                    extra_body_json=extra_body_json,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
            )
            joined = "\n\n".join(refined_prompts)
            return _refiner_result({
                "text": joined,
                "mode": output_mode,
                "frame_count": frame_count,
                "total_duration": duration,
                "ratio": ratio,
                "negative_prompt": source_negative,
                "segment_prompts": refined_prompts,
                "segment_negative_prompts": refined_negatives,
                "fl_data": (
                    fl_constraint.get("fl_data")
                    if isinstance(fl_constraint, dict)
                    else None
                ),
                "offload_dit": bool(
                    fl_constraint.get("offload_dit")
                    if isinstance(fl_constraint, dict)
                    else False
                ),
            }, joined)

        text = polish_with_openai_compatible(
            prompt=prompt,
            base_url=base_url,
            model=model,
            instruction=instruction,
            music_style=music_style,
            package=package,
            output_mode=output_mode,
            frame_count=frame_count,
            total_duration=duration,
            ratio=ratio,
            subjects=subjects,
            supports_image=supports_image,
            supports_video=supports_video,
            supports_audio=supports_audio,
            api_key_env=api_key_env,
            reasoning=reasoning,
            reasoning_effort=reasoning_effort,
            extra_body_json=extra_body_json,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return _refiner_result({
            "text": text, "mode": output_mode,
            "frame_count": frame_count, "total_duration": duration,
            "ratio": ratio, "negative_prompt": source_negative}, text)
