"""Deterministic storyboard compiler tests."""

import os
import sys

_COMFY_ROOT = r"D:\ComfyUI-installs\ComfyUI\ComfyUI"
sys.path.insert(0, _COMFY_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pkg_loader import load as _load_h3rt
_load_h3rt()

from h3rt.nodes.storyboard import compile_storyboard


plan = compile_storyboard({
    "mode": "T2VA",
    "total_duration": 5.0,
    "soundscape": "Breeze and paws on tile.",
    "music_style": "Solo acoustic guitar.",
    "subjects": [
        {"name": "Alice", "definition": "a young woman with long dark hair"},
    ],
    "shots": [
        {"duration": 2.0, "prompt": "Alice walks across the garden."},
        {"duration": 3.0, "prompt": "The camera reveals distant mountains.",
         "camera": "Alice turns around", "sound": "footsteps on gravel"},
    ],
})
assert "integrated_multimodal_description:" in plan["text"]
assert "Alice (<Subject 1>), a young woman with long dark hair walks across the garden." in plan["text"]
assert "Alice (<Subject 1>) turns around" in plan["text"]
assert "[Shot 2] At 00:02.000," in plan["text"]
assert plan["frame_count"] == 124
assert plan["negative_prompt"] == ""
assert plan["subjects"][0]["label"] == "<Subject 1>"
shot = plan["storyboard"]["shots"][0]
assert "start_time" not in shot
assert "music_style" not in shot
assert "refs" not in shot


dialogue_plan = compile_storyboard({
    "mode": "T2VA",
    "total_duration": 5.0,
    "subjects": [
        {"name": "Alice", "definition": "a young woman"},
        {"name": "Li", "definition": "another young woman"},
    ],
    "shots": [
        {"duration": 5.0,
         "prompt": "Alice and Li stand together.",
         "dialogue": "Alice says: <d> [Chinese] Alice loves Li.</d>"},
    ],
})
assert "Alice (<Subject 1>) says:" in dialogue_plan["text"]
assert "<d>[Chinese]Alice loves Li.</d>" in dialogue_plan["text"]
assert "Alice loves Li" in dialogue_plan["text"]
assert "Li (<Subject 2>)" not in dialogue_plan["text"].split("<d>")[1]


normalized_plan = compile_storyboard({
    "mode": "T2VA",
    "total_duration": 5.0,
    "shots": [
        {"duration": 5.0,
         "prompt": "<subject 1> walks past <picture 1> while <video 2> plays."},
    ],
})
assert "<Subject 1> walks past <Picture 1> while <Video 2> plays." in normalized_plan["text"]


mode_plan = compile_storyboard({
    "mode": "I2VA",
    "total_duration": 5.0,
    "subjects": [
        {"name": "Alice", "definition": "the woman in <Picture 1>"},
    ],
    "shots": [
        {"duration": 5.0,
         "prompt": "<Subject 1> opens the door from <Picture 1>."},
    ],
})
assert mode_plan["mode"] == "T2VA"
assert "<Subject 1>, the woman in <Picture 1> opens the door" in mode_plan["text"]
assert "from <Picture 1>" in mode_plan["text"]

fl_mode_plan = compile_storyboard({
    "mode": "FL2VA",
    "total_duration": 8.0,
    "shots": [
        {"duration": 8.0,
         "prompt": "A cyclist opens an umbrella."},
    ],
})
assert fl_mode_plan["mode"] == "T2VA"
assert "integrated_multimodal_description:" in fl_mode_plan["text"]


ref_plan = compile_storyboard({
    "mode": "full_reference",
    "total_duration": 5.0,
    "subjects": [
        {"name": "Alice", "definition": "a dancer in <Picture 1>"},
    ],
    "shots": [
        {"duration": 5.0,
         "prompt": "Alice follows the slow camera motion."},
    ],
}, {
    "image_labels": ["<Picture 1>"],
    "image_roles": ["reference_image"],
    "image_notes": ["User note must stay out of full_reference."],
    "videos": [],
    "audios": [],
})
assert "subject_definitions:" in ref_plan["text"]
assert "<Subject 1> is Alice, a dancer in <Picture 1>." in ref_plan["text"]
assert "Alice (<Subject 1>) follows the slow camera motion." in ref_plan["text"]
assert "User note must stay out" not in ref_plan["text"]


media_plan = compile_storyboard({
    "mode": "full_reference",
    "total_duration": 5.0,
    "shots": [
        {"duration": 5.0,
         "prompt": "A dancer follows the reference motion."},
    ],
}, {
    "image_labels": ["<Picture 1>"],
    "image_roles": ["reference_image"],
    "image_notes": ["Keep the red dress."],
    "videos": [{
        "label": "<Video 1>",
        "role": "motion_reference",
        "note": "Follow the slow pan.",
    }],
    "audios": [{
        "label": "<Audio 1>",
        "role": "music_reference",
        "note": "Use a soft string sound.",
    }],
})
assert "<Picture 1> is a reference image used as visual guidance." in media_plan["text"]
assert "<Video 1> is a motion reference." in media_plan["text"]
assert "<Audio 1> is a background-music style reference." in media_plan["text"]
assert "Keep the red dress." not in media_plan["text"]
assert "Follow the slow pan." not in media_plan["text"]
assert "Use a soft string sound." not in media_plan["text"]


print("storyboard compiler tests ok")
