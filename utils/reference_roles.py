"""Human-readable descriptions for MiniMax H3 reference media roles."""


_ROLE_TEXT = {
    "image": {
        "reference_image": "a reference image used as visual guidance",
        "subject_reference": "a subject or appearance reference",
        "scene_reference": "a scene and environment reference",
        "style_reference": "a style and mood reference",
        "storyboard_anchor": "a storyboard and composition anchor",
    },
    "video": {
        "reference_video": "a reference video used as temporal guidance",
        "motion_reference": "a motion reference",
        "camera_reference": "a camera-movement reference",
        "continuation_source": "a continuation source",
        "edit_source": "an edit source",
        "structure_reference": "a whole-video structure reference",
    },
    "audio": {
        "reference_audio": "a reference audio clip",
        "music_reference": "a background-music style reference",
        "sound_effect_reference": "a sound-effect reference",
        "voice_reference": "a voice or timbre reference",
        "audio_copy": "an audio signal intended for reuse",
    },
}


def reference_role_text(kind: str, role: str) -> str:
    """Return the conservative one-line role description for a reference."""
    return _ROLE_TEXT.get(kind, {}).get(role) or (
        f"a {kind} reference used conservatively"
    )
