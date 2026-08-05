"""MiniMax H3 refiner nodes."""

from __future__ import annotations

def _has_reference_media(package) -> bool:
    if not package:
        return False
    return bool(
        package.get("images") is not None
        or package.get("videos")
        or package.get("audios")
    )


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
    storyboard,
    prompt_ref,
    package,
    fl_constraint,
):
    if storyboard is not None:
        from .storyboard import compile_storyboard

        plan = compile_storyboard(storyboard, package)
        mode = str(plan["mode"] or "T2VA")
        ratio = str(plan["ratio"] or "16:9")
        if mode in ("I2VA", "FL2VA", "L2VA"):
            ratio = "adaptive"
        elif mode == "T2VA" and ratio == "adaptive":
            ratio = "16:9"
        fl_mode = _mode_from_fl_constraint(fl_constraint)
        if fl_mode:
            mode = fl_mode
            ratio = "adaptive"
        return (
            plan["text"],
            mode,
            plan["frame_count"],
            plan["total_duration"],
            ratio,
        )

    if prompt_ref is not None:
        prompt = str(prompt_ref.get("text") or "")
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
            prompt,
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
        }

    RETURN_TYPES = ("MINIMAX_H3_PROMPT", "STRING")
    RETURN_NAMES = ("prompt", "text")
    FUNCTION = "make"
    CATEGORY = "MiniMax-H3/prompt"

    def make(self, text="", mode="T2VA", total_duration=5.0, ratio="16:9"):
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
                "storyboard": ("MINIMAX_H3_STORYBOARD",),
                "prompt_ref": ("MINIMAX_H3_PROMPT",),
                "package": ("PACKAGE_DATA",),
                "fl_constraint": ("MINIMAX_H3_FL_CONSTRAINT",),
                "timeout": ("INT", {"default": 300, "min": 10, "max": 3600}),
            },
        }

    RETURN_TYPES = ("MINIMAX_H3_PROMPT", "STRING")
    RETURN_NAMES = ("prompt", "text")
    FUNCTION = "polish"
    CATEGORY = "MiniMax-H3/conditioning"
    OUTPUT_NODE = True

    def polish(self, instruction="", music_style="", storyboard=None,
               prompt_ref=None, package=None, fl_constraint=None, timeout=300):
        prompt, output_mode, frame_count, duration, ratio = _prepare_refiner_input(
            storyboard, prompt_ref, package, fl_constraint)
        if not prompt.strip():
            return _refiner_result({
                "text": prompt, "mode": output_mode,
                "frame_count": frame_count, "total_duration": duration,
                "ratio": ratio}, prompt)

        from ..utils.refiners import polish_with_context_ir

        text = polish_with_context_ir(
            prompt=prompt,
            instruction=instruction,
            music_style=music_style,
            package=package,
            fl_constraint=fl_constraint,
            duration=duration or 5,
            ratio=ratio,
            timeout=timeout,
        )
        return _refiner_result({
            "text": text, "mode": output_mode,
            "frame_count": frame_count, "total_duration": duration,
            "ratio": ratio}, text)


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
                "storyboard": ("MINIMAX_H3_STORYBOARD",),
                "prompt_ref": ("MINIMAX_H3_PROMPT",),
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
               storyboard=None, prompt_ref=None, package=None,
               fl_constraint=None, api_key_env="",
               supports_image=False, supports_video=False, supports_audio=False,
               reasoning="auto", reasoning_effort="auto", temperature=1.0,
               top_p=0.95, max_tokens=0, timeout=120, extra_body_json=""):
        prompt, output_mode, frame_count, duration, ratio = _prepare_refiner_input(
            storyboard, prompt_ref, package, fl_constraint)
        if not prompt.strip():
            return _refiner_result({
                "text": prompt, "mode": output_mode,
                "frame_count": frame_count, "total_duration": duration,
                "ratio": ratio}, prompt)

        from ..utils.refiners import polish_with_openai_compatible

        subjects = storyboard.get("subjects") if storyboard is not None else None
        text = polish_with_openai_compatible(
            prompt=prompt,
            base_url=base_url,
            model=model,
            instruction=instruction,
            music_style=music_style,
            package=package,
            fl_constraint=fl_constraint,
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
            "ratio": ratio}, text)
