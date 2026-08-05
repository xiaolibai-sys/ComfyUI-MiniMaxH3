"""MiniMax H3 refiner adapter tests."""

import os
import sys

_COMFY_ROOT = r"D:\ComfyUI-installs\ComfyUI\ComfyUI"
sys.path.insert(0, _COMFY_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pkg_loader import load as _load_h3rt
_load_h3rt()

from h3rt.utils import refiners
from h3rt.utils.prompt_refiner import _OFFICIAL_FORMAT_CONTRACT, _SYSTEM_PROMPT
from h3rt.nodes.refiner import (
    MiniMaxH3ContextIRRefiner,
    MiniMaxH3OpenAICompatibleRefiner,
    MiniMaxH3SimplePrompt,
    _prepare_refiner_input,
)


assert "<Subject N> (Sx)" in _SYSTEM_PROMPT
assert 'never "Name (Sx)"' in _SYSTEM_PROMPT
assert "the shot cuts to" in _SYSTEM_PROMPT
assert "Full-reference mode: write 1-2 English style/lighting sentences" in _SYSTEM_PROMPT
assert "Short T2VA example:" in _SYSTEM_PROMPT
assert "Short full-reference example:" in _SYSTEM_PROMPT
assert "Official T2VA example (from MiniMax H3-Context-IR):" in _SYSTEM_PROMPT
assert "I get off at the next station" in _SYSTEM_PROMPT
assert "Official multi-image + voice example (from MiniMax H3-Context-IR):" in _SYSTEM_PROMPT
assert "voice timbre reference for <Subject 1>" in _SYSTEM_PROMPT
assert "Official video editing + clothing reference example (from MiniMax H3-Context-IR):" in _SYSTEM_PROMPT
assert "attribute_transfer - the blue and white gingham crop top" in _SYSTEM_PROMPT
assert "within 160-280 English words" in _SYSTEM_PROMPT
assert "Use 1-2 visible actions per shot" in _SYSTEM_PROMPT
assert "lips close into a warm smile" in _SYSTEM_PROMPT
assert 'summary must include "+ audio reference"' in _SYSTEM_PROMPT
assert "Every tracked label used in summary" in _SYSTEM_PROMPT
assert "Do not invent concrete source actions" in _SYSTEM_PROMPT
assert "mark <Subject N> as partially_preserved" in _SYSTEM_PROMPT
assert "attribute_transfer only when characteristics move to a different" in _OFFICIAL_FORMAT_CONTRACT
assert "do not call them weak_reference" in _OFFICIAL_FORMAT_CONTRACT
assert "Every tracked label used later must be defined here" in _OFFICIAL_FORMAT_CONTRACT
assert 'prefix must include "+ audio reference"' in _OFFICIAL_FORMAT_CONTRACT
assert "capped at 280 English words" in _OFFICIAL_FORMAT_CONTRACT


os.environ["IR_KEY"] = "test-key"
calls = []


def fake_request(url, key, payload=None):
    if payload is not None:
        calls.append(("create", payload))
        return {"task_id": "task-1"}
    return {
        "task": {
            "status": "succeeded",
            "content": {"prompt": "integrated_multimodal_description: [Shot 1] Test."},
        }
    }


refiners._request_json = fake_request
text = refiners.polish_with_context_ir(
    prompt="a cat in a courtyard",
    duration=5,
    ratio="16:9",
)
assert text == "integrated_multimodal_description: [Shot 1] Test."
assert calls[0][1]["duration"] == 5
assert calls[0][1]["ratio"] == "16:9"
assert calls[0][1]["content"][0]["type"] == "text"


ctx_prompt, ctx_mode, ctx_frames, ctx_duration, ctx_ratio = _prepare_refiner_input(
    {
        "mode": "full_reference",
        "ratio": "adaptive",
        "total_duration": 5.0,
        "shots": [
            {"duration": 5.0, "prompt": "A dancer moves through the room."},
        ],
    },
    None,
    None,
    None,
)
assert "A dancer moves through the room." in ctx_prompt
assert ctx_mode == "full_reference"
assert ctx_frames == 124
assert ctx_duration == 5.0
assert ctx_ratio == "adaptive"

ctx_node = MiniMaxH3ContextIRRefiner().polish(
    storyboard={
        "mode": "full_reference",
        "ratio": "adaptive",
        "total_duration": 5.0,
        "shots": [
            {"duration": 5.0, "prompt": "A dancer moves through the room."},
        ],
    }
)
assert ctx_node["result"][0]["frame_count"] == 124
assert ctx_node["result"][0]["total_duration"] == 5.0
assert ctx_node["result"][0]["ratio"] == "adaptive"

openai_prompt, openai_mode, openai_frames, _, _ = _prepare_refiner_input(
    None,
    {"text": "chat prompt", "mode": None},
    None,
    None,
)
assert openai_prompt == "chat prompt"
assert openai_mode == "T2VA"
assert openai_frames is None

fl_prompt, fl_mode, _, _, fl_ratio = _prepare_refiner_input(
    None,
    {"text": "fl2v prompt", "mode": "FL2VA"},
    None,
    {"first_frame": object(), "last_frame": object()},
)
assert fl_mode == "FL2VA"
assert fl_ratio == "adaptive"

first_only, first_mode, _, _, first_ratio = _prepare_refiner_input(
    None,
    {"text": "fl2v prompt", "mode": "FL2VA"},
    None,
    {"first_frame": object(), "last_frame": None},
)
assert first_mode == "I2VA"
assert first_ratio == "adaptive"

simple_prompt, simple_text = MiniMaxH3SimplePrompt().make("a simple prompt")
assert simple_prompt["text"] == "a simple prompt"
assert simple_text == "a simple prompt"
assert simple_prompt["mode"] == "T2VA"
assert simple_prompt["frame_count"] == 124
assert simple_prompt["total_duration"] == 5.0
assert simple_prompt["ratio"] == "16:9"

i2v_simple, _ = MiniMaxH3SimplePrompt().make(
    "image prompt", mode="I2VA", total_duration=3.0, ratio="16:9")
assert i2v_simple["mode"] == "I2VA"
assert i2v_simple["frame_count"] == 73

try:
    refiners.polish_with_context_ir(
        prompt="audio only",
        package={"audios": [{"waveform": object(), "sample_rate": 32000}]},
        duration=5,
        ratio="adaptive",
    )
except ValueError:
    pass
else:
    raise AssertionError("audio-only reference should be rejected")


chat_calls = []


def fake_chat_request(url, api_key, payload, timeout):
    chat_calls.append((url, api_key, payload, timeout))
    return {
        "choices": [
            {
                "message": {
                    "content": (
                        "integrated_multimodal_description: [Shot 1] A cat in a "
                        "courtyard.\n\noverall_soundscape: Soft breeze.\n\n"
                        "non_diegetic_music: N/A"
                    )
                }
            }
        ]
    }


refiners._chat_request_json = fake_chat_request
chat_text = refiners.polish_with_openai_compatible(
    prompt="a cat in a courtyard",
    base_url="http://127.0.0.1:8000/v1",
    model="local-model",
)
assert "integrated_multimodal_description:" in chat_text
assert chat_calls[-1][1] == ""
assert chat_calls[-1][2]["model"] == "local-model"
assert "max_tokens" not in chat_calls[-1][2]
assert chat_calls[-1][2]["temperature"] == 1.0
assert chat_calls[-1][2]["top_p"] == 0.95
assert "thinking" not in chat_calls[-1][2]
assert "reasoning_effort" not in chat_calls[-1][2]

openai_node = MiniMaxH3OpenAICompatibleRefiner().polish(
    base_url="http://127.0.0.1:8000/v1",
    model="local-model",
    prompt_ref=simple_prompt,
)
node_user_text = chat_calls[-1][2]["messages"][-1]["content"]
assert "Target video duration: 5.167 seconds (124 frames at 24fps)" in node_user_text
assert "within 0.000 and 5.167 seconds" in node_user_text
assert "Aspect ratio: 16:9" in node_user_text
assert openai_node["result"][0]["frame_count"] == 124
assert openai_node["result"][0]["total_duration"] == 5.0
assert openai_node["result"][0]["ratio"] == "16:9"

refiners.polish_with_openai_compatible(
    prompt="full-reference pool scene",
    base_url="http://127.0.0.1:8000/v1",
    model="local-model",
    output_mode="full_reference",
    frame_count=124,
    total_duration=5.0,
    ratio="9:16",
    subjects=[
        {"name": "Brey", "label": "<Subject 1>"},
        {"name": "Li", "label": "<Subject 2>"},
    ],
)
contract_user_text = chat_calls[-1][2]["messages"][-1]["content"]
assert "Official MiniMax H3 full-reference format contract" in contract_user_text
assert "Global subject mapping:" in contract_user_text
assert "<Subject 1> = Brey" in contract_user_text
assert "do not write natural names" in contract_user_text

chat_text = refiners.polish_with_openai_compatible(
    prompt="a cat in a courtyard",
    base_url="http://127.0.0.1:8000/v1",
    model="deepseek-v4-pro",
    reasoning="enabled",
    reasoning_effort="high",
)
assert chat_calls[-1][2]["thinking"] == {"type": "enabled"}
assert chat_calls[-1][2]["reasoning_effort"] == "high"

chat_text = refiners.polish_with_openai_compatible(
    prompt="a cat in a courtyard",
    base_url="http://127.0.0.1:8000/v1",
    model="siliconflow/deepseek-v3.2",
    extra_body_json='{"enable_thinking": false}',
)
assert chat_calls[-1][2]["enable_thinking"] is False

refiners._data_uri = lambda path, kind: f"data:{kind};base64,x"
chat_text = refiners.polish_with_openai_compatible(
    prompt="a cat in a courtyard",
    base_url="http://127.0.0.1:8000/v1",
    model="multimodal-model",
    package={
        "image_paths": ["ref.png"],
        "image_roles": ["style_reference"],
        "image_notes": ["Keep the red dress as the subject."],
        "videos": [{
            "path": "ref.mp4",
            "role": "motion_reference",
            "duration": 5.0,
            "note": "Follow the slow pan.",
        }],
        "audios": [{
            "path": "ref.wav",
            "role": "music_reference",
            "duration": 3.0,
            "sample_rate": 48000,
            "note": "Use as a soft music style.",
        }],
    },
    supports_image=True,
    supports_video=True,
    supports_audio=True,
)
user_content = chat_calls[-1][2]["messages"][-1]["content"]
user_text = user_content[0]["text"]
assert (
    "<Picture 1> = reference metadata: a style and mood reference. "
    "Note: Keep the red dress as the subject."
) in user_text
assert (
    "<Video 1> = reference metadata (5.0s): a motion reference. "
    "Note: Follow the slow pan."
) in user_text
assert (
    "<Audio 1> = reference metadata (3.0s, 48kHz): a background-music "
    "style reference. Note: Use as a soft music style."
) in user_text

try:
    refiners.polish_with_openai_compatible(
        prompt="a cat in a courtyard",
        base_url="http://127.0.0.1:8000/v1",
        model="local-model",
        extra_body_json="not json",
    )
except ValueError:
    pass
else:
    raise AssertionError("invalid extra_body_json should fail")

retry_state = {"count": 0}


text_only_calls = []


def fake_text_only_request(url, api_key, payload, timeout):
    text_only_calls.append(payload)
    return {
        "choices": [{
            "message": {
                "content": (
                    "integrated_multimodal_description: [Shot 1] Text-only "
                    "result.\n\noverall_soundscape: N/A\n\n"
                    "non_diegetic_music: N/A"
                )
            }
        }]
    }


refiners._chat_request_json = fake_text_only_request
text_only = refiners.polish_with_openai_compatible(
    prompt="a cat in a courtyard",
    base_url="http://127.0.0.1:8000/v1",
    model="deepseek-text",
    package={
        "image_paths": ["ref.png"],
        "image_roles": ["reference_image"],
        "image_notes": ["DeepSeek should see this note."],
        "videos": [],
        "audios": [],
    },
    supports_image=False,
    supports_video=False,
    supports_audio=False,
)
text_only_payload = text_only_calls[-1]["messages"][-1]["content"]
assert "DeepSeek should see this note." in text_only_payload
assert "Text-only" in text_only

fake_frame = type("FakeImage", (), {"shape": (1, 768, 1344, 3)})()
refiners.polish_with_openai_compatible(
    prompt="first and last frame task",
    base_url="http://127.0.0.1:8000/v1",
    model="deepseek-text",
    fl_constraint={
        "first_frame": fake_frame,
        "last_frame": fake_frame,
    },
    supports_image=False,
    supports_video=False,
    supports_audio=False,
)
fl_text_only_payload = text_only_calls[-1]["messages"][-1]["content"]
assert "first frame of the target video" in fl_text_only_payload
assert "last frame of the target video" in fl_text_only_payload


def fake_retry_request(url, api_key, payload, timeout):
    retry_state["count"] += 1
    if retry_state["count"] == 1:
        return {
            "choices": [{
                "message": {
                    "content": "",
                    "reasoning_content": "thinking uses all tokens",
                },
                "finish_reason": "length",
            }]
        }
    return {
        "choices": [{
            "message": {
                "content": (
                    "integrated_multimodal_description: [Shot 1] Retried "
                    "result.\n\noverall_soundscape: N/A\n\n"
                    "non_diegetic_music: N/A"
                )
            }
        }]
    }


refiners._chat_request_json = fake_retry_request
retried = refiners.polish_with_openai_compatible(
    prompt="a cat in a courtyard",
    base_url="http://127.0.0.1:8000/v1",
    model="deepseek-reasoning",
    max_tokens=2048,
)
assert retry_state["count"] == 2
assert "Retried result" in retried


bad_timeline_calls = []


def fake_bad_timeline_request(url, api_key, payload, timeout):
    bad_timeline_calls.append(payload)
    return {
        "choices": [{
            "message": {
                "content": (
                    "integrated_multimodal_description: [Shot 1] A cat in a "
                    "courtyard. [Shot 3] At 00:06.500, the shot cuts away.\n\n"
                    "overall_soundscape: N/A\n\nnon_diegetic_music: N/A"
                )
            }
        }]
    }


refiners._chat_request_json = fake_bad_timeline_request
try:
    refiners.polish_with_openai_compatible(
        prompt="a cat in a courtyard",
        base_url="http://127.0.0.1:8000/v1",
        model="deepseek-text",
        frame_count=124,
        total_duration=5.0,
    )
except RuntimeError as exc:
    assert "exceeds target duration 5.167s" in str(exc)
else:
    raise AssertionError("timeline beyond target duration should fail")
assert "within 0.000 and 5.167 seconds" in bad_timeline_calls[-1]["messages"][-1]["content"]

print("refiner tests ok")
