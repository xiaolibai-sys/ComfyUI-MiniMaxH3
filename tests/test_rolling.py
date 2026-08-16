"""Rolling FL2VA pipeline smoke test."""

import os
import json
import sys
import tempfile
from unittest import mock

sys.path.insert(0, r"D:\ComfyUI-installs\ComfyUI\ComfyUI")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pkg_loader import load as _load_h3rt
_load_h3rt()

import torch
import zlib
from safetensors.torch import save_file

from h3rt.models.model import MiniMaxH3Model
from h3rt.utils.config import MiniMaxH3DiTConfig
from h3rt.utils.lifecycle import load_model_handle
from h3rt.utils.rolling import rolling_sample
from h3rt.utils.types import (
    AVLatent,
    FLConstraint,
    FLKeyframe,
    H3BlockSwap,
    H3Conditioning,
    H3SampleResult,
    RuntimeOptions,
    SamplingAssets,
    SamplingConfig,
    TextConditioning,
    VAERef,
)

torch.manual_seed(13)


class FakeVAE:
    def __init__(self):
        self.encodes = 0
        self.devices = []

    def encode_video(self, x):
        self.encodes += 1
        if x.ndim == 4:
            x = x.unsqueeze(2)
        return torch.zeros(
            x.shape[0],
            24,
            1,
            x.shape[3] // 16,
            x.shape[4] // 16,
            device=x.device,
            dtype=torch.float32,
        )

    def decode_video_streaming(self, z):
        b, _c, t, h, w = z.shape
        return torch.zeros(
            b,
            3,
            t * 4,
            h * 16,
            w * 16,
            device="cpu",
            dtype=torch.float32,
        )

    def decode_audio(self, z):
        # alternate loud/quiet segments so the loudness matcher stays active
        self.audio_calls = getattr(self, "audio_calls", 0) + 1
        amp = 0.4 if self.audio_calls % 2 else 0.05
        g = torch.Generator().manual_seed(self.audio_calls)
        return torch.randn(z.shape[0], 2, 32000, generator=g) * amp

    def unload(self):
        pass

    def to(self, device):
        self.devices.append(str(device))
        return self


def _build_checkpoint():
    cfg = MiniMaxH3DiTConfig(
        hidden_size=64,
        num_layers=3,
        token_refiner_num_layers=1,
        num_attention_heads=4,
        attention_head_dim=16,
        ffn_hidden_size=128,
        latents_dim=24,
        audio_latents_dim=32,
        patch_size=(1, 2, 2),
        text_dim=16,
        timestep_input_dim=16,
        time_embed_hidden_size=64,
        time_embed_dim=32,
        rope_inv_freq_len=2,
        norm_eps=1e-5,
        qk_norm_eps=1e-5,
        final_norm_eps=1e-5,
    )
    with torch.device("meta"):
        ref = MiniMaxH3Model(cfg, dtype=torch.float32)
    sd = {}
    for pname, p in ref.named_parameters():
        g = torch.Generator().manual_seed(
            zlib.crc32(pname.encode()) % (2 ** 32))
        sd[pname] = torch.randn(p.shape, dtype=torch.float32, generator=g)
    for bname, b in ref.named_buffers():
        sd[bname] = torch.randn(b.shape, dtype=torch.float32)
    path = os.path.join(tempfile.mkdtemp(prefix="h3_roll_"), "model.safetensors")
    save_file(sd, path)
    return path


def test_rolling_two_segments():
    path = _build_checkpoint()
    handle = load_model_handle(path)
    handle.attn_backend_name = "sageattn2"
    swap = H3BlockSwap(
        enabled=True,
        block_to_swap=1,
        prefetch=True,
        prefetch_count=1,
        pin_memory=False,
        disk_workers=2,
        dtype="float32",
    )
    device = torch.device("cuda")
    text = torch.randn(1, 8, 16, device=device, dtype=torch.float32)
    tags = torch.ones(1, 8, dtype=torch.long, device=device)
    positive = H3Conditioning(
        text=TextConditioning(states=text, tags=tags))
    first_img = torch.rand(1, 64, 64, 3, dtype=torch.float32)
    fl = FLConstraint(
        fps=24,
        keyframes=(
            FLKeyframe(time=0.0, image=first_img),
            FLKeyframe(time=0.5, image=None),
            FLKeyframe(time=1.0, image=None),
        ),
    )
    assets = SamplingAssets(
        handle=handle,
        positive=positive,
        negative=None,
        fl_constraint=fl,
        av_encoder=VAERef("video.safetensors"),
        runtime=RuntimeOptions(swap=swap),
        latent=AVLatent(
            video=torch.zeros(1, 24, 7, 4, 4, device=device),
            audio=torch.zeros(1, 32, 2, 20, device=device),
        ),
    )
    config = SamplingConfig(
        steps=2,
        cfg=1.0,
        seed=7,
        sampler_name="euler",
        scheduler_name="normal",
        shift_video=12.0,
        shift_audio=3.0,
        use_adaln_cache=False,
        adaln_prebake_batch=3,
        width=64,
        height=64,
        denoise=1.0,
    )
    try:
        with mock.patch(
            "h3rt.utils.session.load_vae_pack",
            return_value=FakeVAE(),
        ):
            out = rolling_sample(assets, config, disable_pbar=True)
        assert out.segment_count == 2
        assert out.video is not None
        assert torch.isfinite(out.video.float()).all()
        assert torch.isfinite(out.audio.float()).all()
    finally:
        handle.unload()
    print("ROLLING PIPELINE OK")


def test_t2va_rolling_without_frames():
    path = _build_checkpoint()
    handle = load_model_handle(path)
    handle.attn_backend_name = "sageattn2"
    swap = H3BlockSwap(
        enabled=True,
        block_to_swap=1,
        prefetch=True,
        prefetch_count=1,
        pin_memory=False,
        disk_workers=2,
        dtype="float32",
    )
    device = torch.device("cuda")
    text = torch.randn(1, 8, 16, device=device, dtype=torch.float32)
    tags = torch.ones(1, 8, dtype=torch.long, device=device)
    positive = H3Conditioning(
        text=TextConditioning(states=text, tags=tags))
    fl = {
        "first_frame": None,
        "last_frame": None,
        "fl_data": json.dumps({
            "keyframes": [
                {"time": 0.0},
                {"time": 0.5},
                {"time": 1.0},
            ],
        }),
    }
    assets = SamplingAssets(
        handle=handle,
        positive=positive,
        negative=None,
        fl_constraint=fl,
        av_encoder=VAERef("video.safetensors"),
        runtime=RuntimeOptions(swap=swap),
        latent=AVLatent(
            video=torch.zeros(1, 24, 7, 4, 4, device=device),
            audio=torch.zeros(1, 32, 2, 20, device=device),
        ),
    )
    config = SamplingConfig(
        steps=2,
        cfg=1.0,
        seed=7,
        sampler_name="euler",
        scheduler_name="normal",
        shift_video=12.0,
        shift_audio=3.0,
        use_adaln_cache=False,
        adaln_prebake_batch=3,
        width=64,
        height=64,
        denoise=1.0,
    )
    try:
        with mock.patch(
            "h3rt.utils.session.load_vae_pack",
            return_value=FakeVAE(),
        ):
            out = rolling_sample(assets, config, disable_pbar=True)
        assert out.segment_count == 2
        assert out.video is not None
        assert torch.isfinite(out.video.float()).all()
    finally:
        handle.unload()
    print("T2VA ROLLING OK")


def test_sampler_propagates_fl_constraint_and_vae():
    from h3rt.nodes.sampler import MiniMaxH3KSampler

    device = torch.device("cuda")
    text = torch.randn(1, 8, 16, device=device, dtype=torch.float32)
    tags = torch.ones(1, 8, dtype=torch.long, device=device)
    video = torch.zeros(1, 24, 7, 4, 4, device=device)
    audio = torch.zeros(1, 32, 2, 20, device=device)
    fl = {
        "first_frame": None,
        "last_frame": None,
        "fl_data": json.dumps({
            "keyframes": [
                {"time": 0.0},
                {"time": 0.5},
                {"time": 1.0},
            ],
        }),
    }
    positive = H3Conditioning(
        text=TextConditioning(states=text, tags=tags),
        fl_constraint=fl,
        av_encoder=VAERef("video.safetensors", "audio.safetensors"),
    )
    latent = AVLatent(video=video, audio=audio)
    captured = {}

    def fake_run(assets, config, **kwargs):
        captured["assets"] = assets
        return H3SampleResult(
            video=video,
            audio=audio,
            steps=1,
        )

    with mock.patch(
        "h3rt.nodes.sampler.run_sampling",
        side_effect=fake_run,
    ):
        import latent_preview
        with mock.patch.object(
            latent_preview,
            "prepare_callback",
            return_value=None,
        ):
            MiniMaxH3KSampler().sample(
                model=object(),
                positive=positive,
                seed=7,
                steps=1,
                cfg=1.0,
                sampler_name="euler",
                scheduler_name="normal",
                shift_video=12.0,
                shift_audio=3.0,
                denoise=1.0,
                use_adaln_cache=False,
                adaln_prebake_batch=3,
                latent=latent,
            )

    assets = captured["assets"]
    assert assets.fl_constraint is fl
    assert assets.av_encoder.video_path == "video.safetensors"
    print("SAMPLER FL PROPAGATION OK")


def test_fl_offload_dit_propagates_and_vae_phase():
    from h3rt.utils.memory import SamplingMemory
    from h3rt.utils.rolling import _coerce_fl_constraint
    from h3rt.nodes.fl_constraint import MiniMaxH3FLConstraint
    from dataclasses import dataclass

    fl = {
        "first_frame": None,
        "last_frame": None,
        "offload_dit": True,
        "fl_data": json.dumps({
            "keyframes": [
                {"time": 0.0},
                {"time": 0.5},
                {"time": 1.0},
            ],
        }),
    }
    parsed = _coerce_fl_constraint(fl, None)
    assert parsed.offload_dit is True

    from h3rt.nodes.fl_constraint import _set_fl

    _set_fl("offload-test", {
        "offload_dit": True,
        "keyframes": [{"time": 0.0}, {"time": 1.0}],
    })
    node_out = MiniMaxH3FLConstraint().make(unique_id="offload-test")
    assert node_out[0]["offload_dit"] is True
    assert json.loads(node_out[0]["fl_data"])["offload_dit"] is True

    fl_obj = FLConstraint.from_json(
        '{"offload_dit": true, '
        '"keyframes": [{"time": 0.0}, {"time": 1.0}]}'
    )
    assert fl_obj.offload_dit is True

    # offload_dit=True: full model unload/reload through the handle
    @dataclass
    class FakeContext:
        device: str = "cuda"
        model: object = None
        reader: object = None
        block_stats: object = None

    class FakeHandle:
        def __init__(self):
            self.swap = H3BlockSwap()
            self.unloads = 0
            self.loads = 0

        def unload(self):
            self.unloads += 1

        def load(self, **kwargs):
            self.loads += 1
            return type("Model", (), {"_swap_mgr": None})()

    class FakeVae:
        def __init__(self):
            self.devices = []

        def to(self, device):
            self.devices.append(str(device))

    class FakeSession:
        def __init__(self):
            self.handle = FakeHandle()
            self.assets = type("Assets", (), {
                "fl_constraint": FLConstraint(offload_dit=True),
                "handle": self.handle,
                "runtime": RuntimeOptions(),
                "vram_spec": None,
            })()
            self.model = type("Model", (), {"_swap_mgr": None})()
            self.context = FakeContext()
            self.config = type("Config", (), {"use_adaln_cache": False})()
            self.vae = FakeVae()

    session = FakeSession()
    memory = SamplingMemory(session)
    with memory.vae_phase():
        assert session.vae.devices[-1] == "cuda"
    assert session.vae.devices[-1] == "cpu"
    assert session.handle.unloads == 1 and session.handle.loads == 1

    # final VAE phase of a run: model must stay unloaded (finish() would
    # tear it down anyway), so no restore happens
    with memory.vae_phase(reload_after=False):
        pass
    assert session.handle.unloads == 2
    assert session.handle.loads == 1, "final phase must not reload the model"

    # offload_dit=False: blocks swap to RAM via the swap manager instead
    class FakeMgr:
        def __init__(self):
            self.offloaded = False
            self.restored = False

        def offload_all(self):
            self.offloaded = True

        def restore_initial(self):
            self.restored = True

    class FakeSession2:
        def __init__(self):
            self.assets = type("Assets", (), {
                "fl_constraint": FLConstraint(offload_dit=False),
            })()
            self._mgr = FakeMgr()
            self.model = type("Model", (), {"_swap_mgr": self._mgr})()
            self.context = type("Context", (), {"device": "cuda"})()
            self.vae = FakeVae()

    session2 = FakeSession2()
    memory = SamplingMemory(session2)
    with memory.vae_phase():
        assert session2.vae.devices[-1] == "cuda"
    assert session2.vae.devices[-1] == "cpu"
    assert session2._mgr.offloaded
    assert session2._mgr.restored
    print("FL OFFLOAD DIT OK")


def test_last_boundary_skips_start_encode():
    from h3rt.utils.memory import SamplingMemory

    class CountingVae:
        def __init__(self):
            self.encodes = 0

        def decode_video_streaming(self, v):
            return torch.zeros(1, 3, 2, 8, 8)

        def decode_audio(self, a):
            return torch.zeros(1, 2, 4)

        def encode_video(self, v):
            self.encodes += 1
            return torch.zeros(1, 24, 1, 1, 1)

    session = type("S", (), {
        "vae": CountingVae(),
        "assets": type("A", (), {
            "fl_constraint": FLConstraint(offload_dit=False),
        })(),
    })()
    memory = SamplingMemory(session)
    result = type("R", (), {"video": None, "audio": None})()
    _, _, start = memory.decode_and_encode(result, need_start=False)
    assert start is None
    assert session.vae.encodes == 0, "last boundary must not encode a start latent"
    _, _, start = memory.decode_and_encode(result, need_start=True)
    assert start is not None
    assert session.vae.encodes == 1
    print("LAST BOUNDARY ENCODE SKIP OK")


def test_rolling_offload_dit_reload_count():
    """End-to-end: upfront batch encode + no final reload cut model loads."""
    path = _build_checkpoint()
    handle = load_model_handle(path)
    handle.attn_backend_name = "sageattn2"
    swap = H3BlockSwap(
        enabled=True,
        block_to_swap=1,
        prefetch=True,
        prefetch_count=1,
        pin_memory=False,
        disk_workers=2,
        dtype="float32",
    )
    device = torch.device("cuda")
    text = torch.randn(1, 8, 16, device=device, dtype=torch.float32)
    tags = torch.ones(1, 8, dtype=torch.long, device=device)
    positive = H3Conditioning(
        text=TextConditioning(states=text, tags=tags))
    imgs = tuple(
        torch.rand(1, 64, 64, 3, dtype=torch.float32) for _ in range(3))
    fl = FLConstraint(
        fps=24,
        offload_dit=True,
        keyframes=(
            FLKeyframe(time=0.0, image=imgs[0]),
            FLKeyframe(time=0.5, image=imgs[1]),
            FLKeyframe(time=1.0, image=imgs[2]),
        ),
    )
    assets = SamplingAssets(
        handle=handle,
        positive=positive,
        negative=None,
        fl_constraint=fl,
        av_encoder=VAERef("video.safetensors"),
        runtime=RuntimeOptions(swap=swap),
        latent=AVLatent(
            video=torch.zeros(1, 24, 7, 4, 4, device=device),
            audio=torch.zeros(1, 32, 2, 20, device=device),
        ),
    )
    config = SamplingConfig(
        steps=2,
        cfg=1.0,
        seed=7,
        sampler_name="euler",
        scheduler_name="normal",
        shift_video=12.0,
        shift_audio=3.0,
        use_adaln_cache=False,
        adaln_prebake_batch=3,
        width=64,
        height=64,
        denoise=1.0,
    )
    loads = {"n": 0}
    orig_load = handle.load

    def counting_load(*args, **kwargs):
        loads["n"] += 1
        return orig_load(*args, **kwargs)

    handle.load = counting_load
    vae = FakeVAE()
    try:
        with mock.patch(
            "h3rt.utils.session.load_vae_pack",
            return_value=vae,
        ):
            out = rolling_sample(assets, config, disable_pbar=True)
        assert out.segment_count == 2
        # 3 loads: session prepare + upfront encode phase + seg0 boundary.
        # The final boundary decode leaves the model unloaded.
        assert loads["n"] == 3, f"expected 3 model loads, got {loads['n']}"
        # 3 keyframe images up front + 1 boundary start latent; the last
        # boundary encodes nothing.
        assert vae.encodes == 4, f"expected 4 vae encodes, got {vae.encodes}"
    finally:
        handle.load = orig_load
        handle.unload()
    print("ROLLING RELOAD COUNT OK")


def test_multi_keyframe_prompts_and_negative_prompts():
    from h3rt.utils.rolling import _coerce_fl_constraint, build_rolling_plan

    fl = {
        "first_frame": None,
        "last_frame": None,
        "fl_data": json.dumps({
            "keyframes": [
                {"time": 0.0, "prompt": "p0", "negative_prompt": "n0"},
                {"time": 2.5, "prompt": "p1", "negative_prompt": "n1"},
                {"time": 5.0, "prompt": "p2", "negative_prompt": "n2"},
            ],
        }),
    }
    parsed = _coerce_fl_constraint(fl, None)
    assert [k.prompt for k in parsed.keyframes] == ["p0", "p1", "p2"]
    assert [k.negative_prompt for k in parsed.keyframes] == ["n0", "n1", "n2"]
    plan = build_rolling_plan(parsed, 64, 64)
    assert len(plan.segments) == 2
    assert plan.segments[0].negative_prompt == "n0"
    assert plan.segments[1].negative_prompt == "n1"
    print("MULTI KEYFRAME PROMPT PLAN OK")


def test_loudness_measurement_and_matching():
    import math

    from h3rt.utils.loudness import integrated_loudness, match_segment_loudness

    sr = 32000
    t = torch.arange(sr * 2, dtype=torch.float32) / sr
    sine = torch.sin(2 * math.pi * 997.0 * t)

    def stereo(x):
        return x.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1)

    # energy x4 must read +6.02 LU regardless of absolute calibration
    l_full = integrated_loudness(stereo(sine * 0.5), sr)
    l_half = integrated_loudness(stereo(sine * 0.25), sr)
    assert abs((l_full - l_half) - 6.02) < 0.3, (
        f"amplitude halving should read -6.02 LU, got {l_full - l_half:.2f}")

    # gating: tone followed by 3 s silence must read ~= the tone alone
    tone = sine * 0.3
    padded = torch.cat([tone, torch.zeros(sr * 3)])
    l_padded = integrated_loudness(stereo(padded), sr)
    l_tone = integrated_loudness(stereo(tone), sr)
    assert abs(l_padded - l_tone) < 1.0, (
        f"gated loudness should ignore trailing silence: "
        f"{l_padded:.1f} vs {l_tone:.1f} LUFS")

    # matching: quieter second segment pulled up to the reference
    g1 = torch.Generator().manual_seed(1)
    g2 = torch.Generator().manual_seed(2)
    seg_ref = (torch.randn(1, 2, sr * 2, generator=g1) * 0.12).clamp(-1, 1)
    seg_loud = (torch.randn(1, 2, sr * 2, generator=g2) * 0.4).clamp(-1, 1)
    out = match_segment_loudness([seg_ref, seg_loud], sample_rate=sr)
    l0 = integrated_loudness(out[..., :sr * 2], sr)
    l1 = integrated_loudness(out[..., sr * 2:], sr)
    assert abs(l0 - l1) < 1.0, f"segments not matched: {l0:.1f} vs {l1:.1f}"
    assert out.abs().max() <= 0.99 + 1e-6

    # boost path: peak guard must prevent clipping
    out_boost = match_segment_loudness([seg_loud, seg_ref], sample_rate=sr)
    assert out_boost.abs().max() <= 0.99 + 1e-6, "boost clipped"

    # silence stays silent (outside the crossfade window)
    fade = int(sr * 0.02)
    silence = torch.zeros(1, 2, sr)
    out_sil = match_segment_loudness([silence, seg_ref], sample_rate=sr)
    assert out_sil[..., :sr - fade].abs().max() == 0

    # already matched: unity gain, only the crossfade window is blended
    matched = match_segment_loudness([seg_ref, seg_ref * 0.99], sample_rate=sr)
    assert matched.shape[-1] == 4 * sr - fade
    assert torch.equal(matched[..., :2 * sr - fade], seg_ref[..., :2 * sr - fade])
    assert torch.equal(matched[..., 2 * sr:], (seg_ref * 0.99)[..., fade:])

    # crossfade removes splice clicks: a hard sign flip at the cut must not
    # survive as an instantaneous jump
    joined = match_segment_loudness(
        [torch.full((1, 2, sr), 0.5), torch.full((1, 2, sr), -0.5)],
        sample_rate=sr)
    max_step = (joined[..., 1:] - joined[..., :-1]).abs().max().item()
    assert max_step < 0.1, f"splice step {max_step:.3f} would click"

    # true-peak detection: fs/4 sine at 45 deg phase peaks between samples
    from h3rt.utils.loudness import _true_peak

    inter = torch.sin(2 * math.pi * (sr / 4) * t + math.pi / 4) * 0.9
    sample_peak = inter.abs().max().item()
    true_peak = _true_peak(stereo(inter), sr)
    assert true_peak > sample_peak * 1.3, (
        f"true peak {true_peak:.3f} should far exceed "
        f"sample peak {sample_peak:.3f}")
    assert abs(true_peak - 0.9) < 0.03, (
        f"true peak should recover ~0.9, got {true_peak:.3f}")

    # inter-sample clip guard: a segment that measures quiet (gated silence
    # around a short blip) but has a hot true peak must not be boosted past
    # the ceiling - sample-peak detection would miss this
    seg_blip = torch.zeros(sr * 2)
    blip_t = torch.arange(sr // 20, dtype=torch.float32) / sr
    seg_blip[10000:10000 + sr // 20] = (
        torch.sin(2 * math.pi * 8000.0 * blip_t + math.pi / 4) * 0.9)
    out_tp = match_segment_loudness(
        [stereo(sine * 0.7), stereo(seg_blip)], sample_rate=sr)
    assert _true_peak(out_tp[..., sr * 2:], sr) <= 1.01, (
        "boosted segment clipped on inter-sample peaks")
    print("LOUDNESS MEASURE+MATCH OK")


def test_audio_loudness_match_toggle():
    from h3rt.utils.rolling import _coerce_fl_constraint
    from h3rt.nodes.fl_constraint import MiniMaxH3FLConstraint, _set_fl

    # the toggle travels inside fl_data; the node mirrors it in the dict
    _set_fl("lm-test", {
        "audio_loudness_match": False,
        "offload_dit": False,
        "keyframes": [{"time": 0.0}, {"time": 1.0}],
    })
    out = MiniMaxH3FLConstraint().make(unique_id="lm-test")
    assert out[0]["audio_loudness_match"] is False
    assert _coerce_fl_constraint(
        out[0], None).audio_loudness_match is False
    assert _coerce_fl_constraint(out[0], None).offload_dit is False

    # no stored data / missing keys default both toggles to on
    out_default = MiniMaxH3FLConstraint().make(unique_id="lm-unset")
    assert out_default[0]["audio_loudness_match"] is True
    assert out_default[0]["offload_dit"] is True
    fl_default = {
        "fl_data": json.dumps({"keyframes": [{"time": 0.0}, {"time": 1.0}]}),
    }
    assert _coerce_fl_constraint(fl_default, None).audio_loudness_match is True
    assert _coerce_fl_constraint(fl_default, None).offload_dit is True
    assert FLConstraint.from_json("{}").audio_loudness_match is True
    print("AUDIO LOUDNESS TOGGLE OK")


if __name__ == "__main__":
    test_rolling_two_segments()
    test_t2va_rolling_without_frames()
    test_sampler_propagates_fl_constraint_and_vae()
    test_fl_offload_dit_propagates_and_vae_phase()
    test_multi_keyframe_prompts_and_negative_prompts()
    test_last_boundary_skips_start_encode()
    test_rolling_offload_dit_reload_count()
    test_loudness_measurement_and_matching()
    test_audio_loudness_match_toggle()
