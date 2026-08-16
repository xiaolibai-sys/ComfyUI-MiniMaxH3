"""Central dataclass contract tests for utils/types.py."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pkg_loader import load as _load_h3rt
_load_h3rt()

import torch

from h3rt.utils.blockswap import SwapBlock
from h3rt.utils.types import (
    FLConstraint,
    FLKeyframe,
    H3Conditioning,
    KeyframeCondition,
    MediaConditioning,
    ReferenceCondition,
    RuntimeOptions,
    SamplingAssets,
    SequenceSpec,
    TextConditioning,
    VAERef,
)


def test_typed_conditioning_constructor_round_trips_media():
    keyframe = KeyframeCondition(0, torch.zeros(1, 24, 1, 2, 2))
    ref = ReferenceCondition(
        kind="audio",
        ref_audio_t=4,
        audio_latent=torch.zeros(1, 32, 2, 4),
    )
    media = MediaConditioning(
        keyframes=(keyframe,),
        refs=(ref,),
        frame_count=5,
    )
    cond = H3Conditioning(
        text=TextConditioning(
            torch.zeros(1, 4, 8),
            torch.ones(1, 4, dtype=torch.long),
        ),
        media=media,
    )
    assert cond.media.keyframes[0] is keyframe
    assert cond.media.refs[0] is ref
    payload = cond.to_payload()
    assert payload["keyframes"][0]["resolved_frame_index"] == 0
    assert payload["refs"][0]["ref_audio_t"] == 4


def test_central_imports_resolve():
    assert SwapBlock("x", torch.nn.Linear(1, 1)).name == "x"
    assert VAERef("video.safetensors").video_path == "video.safetensors"
    assert RuntimeOptions().swap is None
    fl = FLConstraint.from_json(
        '{"fps":24,"global_negative_prompt":"ng",'
        '"keyframes":[{"time":0.0,"note":"first"},{"time":2.5}]}'
    )
    assert isinstance(fl.keyframes[0], FLKeyframe)
    assert fl.keyframes[1].time == 2.5
    assert fl.global_negative_prompt == "ng"
    assert fl.keyframes[0].note == "first"
    spec = SequenceSpec(text_len=1, latent_t=1, latent_h=2, latent_w=2, audio_t=1)
    assert spec.media.keyframes == ()
    assets = SamplingAssets(
        handle=object(),
        positive=H3Conditioning(
            text=TextConditioning(
                states=torch.zeros(1, 4, 8),
                tags=torch.ones(1, 4, dtype=torch.long),
            )
        ),
        negative=None,
        fl_constraint=fl,
        av_encoder=object(),
    )
    assert assets.fl_constraint.fps == 24


if __name__ == "__main__":
    test_typed_conditioning_constructor_round_trips_media()
    test_central_imports_resolve()
    print("types contract tests passed")
