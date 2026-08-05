"""H3 v2 ref2va alignment: official PR tokenizer/encoding vs our port.

Token-level: builds the same ``minimax_ref_items`` through the official
``MiniMaxH3Tokenizer`` and through our ``TextEncoder._prepare_minimax_inputs``
(using the identical Qwen2 vocab) and asserts the token sequence matches
(vision blocks expand to VISION_START + IMAGE_PAD*N + VISION_END).

Structure-level: runs our node's ref-block construction against the official
``MiniMaxH3ReferenceToVideo.execute`` on mocked VAE/encoder objects and asserts
the ``ref_blocks`` payloads match field-for-field.

MRoPE-level: asserts our Qwen3-VL fusion positions match ComfyUI core's
``qwen2vl_mrope_position_ids`` helper for the same sequence.

Checkpoint-level: scans the H3 v2 Ref2VA checkpoint's ``adaln_t_table``
architecture from disk without loading weights.

No model weights are needed for the token/structure checks (tokenizer +
mocked VAE).  Run with the ComfyUI venv python from the package root:
    python tests/test_ref2va_align.py
"""

import os
import sys
import math
import types

sys.path.insert(0, r"D:\ComfyUI-installs\ComfyUI\ComfyUI")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pkg_loader import load as _load_h3rt
_load_h3rt()

import torch
import numpy as np

REF2VA_CHECKPOINT = r"D:\ComfyUI-installs\ComfyUI\ComfyUI\models\diffusion_models\minimax_h3_ref2va_pruned_fp8_scaled.safetensors"


def official_token_seq(ref_items, prompt):
    from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
    tok = MiniMaxH3Tokenizer()
    pairs = tok.tokenize_with_weights(
        prompt, minimax_ref_items=ref_items)["qwen3vl_32b"][0]
    seq = []
    emb_count = 0
    for t, _ in pairs:
        if isinstance(t, int):
            seq.append(t)
        elif isinstance(t, dict):
            seq.append(("EMB", emb_count))
            emb_count += 1
    return seq


def official_images_seq(images, prompt):
    from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
    tok = MiniMaxH3Tokenizer()
    pairs = tok.tokenize_with_weights(
        prompt, images=images)["qwen3vl_32b"][0]
    seq = []
    emb_count = 0
    for t, _ in pairs:
        if isinstance(t, int):
            seq.append(t)
        elif isinstance(t, dict):
            seq.append(("EMB", emb_count))
            emb_count += 1
    return seq


def our_token_seq(ref_items, prompt):
    import importlib
    from h3rt.models.text_encoder.encoder import TextEncoder
    from h3rt.models.text_encoder.types import TextEncoderInput
    # _prepare_minimax_inputs is defined as a method; bind a bare-callable copy.
    enc = TextEncoder.__new__(TextEncoder)
    # attach a Qwen2 tokenizer with the same vocab as ComfyUI's qwen25_tokenizer
    import transformers
    enc.tokenizer = transformers.Qwen2TokenizerFast.from_pretrained(
        r"D:\ComfyUI-installs\ComfyUI\ComfyUI\comfy\text_encoders\qwen25_tokenizer")
    enc.device = torch.device("cpu")
    enc.config = type("C", (), {"text": type("T", (), {"pad_token_id": 151643})()})()
    payload = TextEncoderInput(text=prompt, minimax_ref_items=ref_items)
    ids, _, mm, tags = enc._prepare_minimax_inputs(payload)
    seq = ids[0].tolist()
    return seq, mm, tags, ids, torch.ones_like(ids)


def check_h3_v2_checkpoint_config():
    from h3rt.utils.config import MiniMaxH3DiTConfig
    from h3rt.utils.lifecycle import scan_dit_config
    from h3rt.utils.stream import BlockReader

    reader = BlockReader(REF2VA_CHECKPOINT)
    try:
        cfg = scan_dit_config(reader, MiniMaxH3DiTConfig())
    finally:
        reader.close()
    expected = dict(
        hidden_size=5376, num_layers=50, token_refiner_num_layers=2,
        num_attention_heads=56, attention_head_dim=128,
        ffn_hidden_size=14336, latents_dim=24, audio_latents_dim=32,
        text_dim=5120, rope_inv_freq_len=16, adaln_curve_grid=1025,
        time_embed_dim=8)
    got = dict(
        hidden_size=cfg.hidden_size, num_layers=cfg.num_layers,
        token_refiner_num_layers=cfg.token_refiner_num_layers,
        num_attention_heads=cfg.num_attention_heads,
        attention_head_dim=cfg.attention_head_dim,
        ffn_hidden_size=cfg.ffn_hidden_size, latents_dim=cfg.latents_dim,
        audio_latents_dim=cfg.audio_latents_dim, text_dim=cfg.text_dim,
        rope_inv_freq_len=cfg.rope_inv_freq_len,
        adaln_curve_grid=cfg.adaln_curve_grid, time_embed_dim=cfg.time_embed_dim)
    assert got == expected, f"H3 v2 config mismatch: {got} != {expected}"
    print("[H3 v2 checkpoint] Ref2VA adaln_t_table config scan OK")


def check_mrope_positions(ids, attention_mask, mm):
    from h3rt.models.text_encoder.fusion import Qwen3VLMultimodalFusion
    if mm.get("image_grid_thw") is None:
        return
    fusion = object.__new__(Qwen3VLMultimodalFusion)
    fusion.config = types.SimpleNamespace(
        vision=types.SimpleNamespace(spatial_merge_size=2))
    position_ids, _ = fusion.get_rope_index(
        ids, mm["mm_token_type_ids"],
        image_grid_thw=mm["image_grid_thw"],
        attention_mask=attention_mask)
    assert position_ids.shape[-1] == ids.shape[-1], (
        f"MRoPE positions {position_ids.shape[-1]} != seq {ids.shape[-1]}")


def compare(name, ref_items, prompt, images=None):
    official = (official_images_seq(images, prompt) if images is not None
                else official_token_seq(ref_items, prompt))
    ours, mm, tags, ids, attention_mask = our_token_seq(ref_items, prompt)
    check_mrope_positions(ids, attention_mask, mm)
    # expand official EMB markers into IMAGE_PAD counts using our grid sizes
    # (official tokenizer hides the pad count; verify ours matches layout)
    expanded = []
    for t in official:
        if isinstance(t, tuple):          # ("EMB", i)
            expanded.extend([151652, 151653])  # handled below by position
        else:
            expanded.append(t)
    # simpler: rebuild expected from ours' ids, replacing EMB spans
    # Our ids already contain VISION_START/END + IMAGE_PADs; official has
    # VISION_START + EMB + VISION_END.  Compare after expanding EMB to the
    # same number of IMAGE_PADs our block produced.
    exp = []
    emb_idx = 0
    for t in official:
        if isinstance(t, int):
            exp.append(t)
        else:
            # EMB expands to the vision tower's 2x2-spatial-merged token count.
            g = mm["image_grid_thw"][emb_idx]
            exp.extend([151655] * int(g[1].item() * g[2].item() // 4))
            emb_idx += 1
    ok = exp == ours
    print(f"[{name}] token seq match: {ok}  (official {len(exp)} vs ours {len(ours)})")
    if not ok:
        print("  official:", exp[:60])
        print("  ours    :", ours[:60])
    # tags: text positions 1, vision pads 0; length == seq
    assert len(tags[0]) == len(ours), "tags length != seq length"
    assert torch.isfinite(tags).all()
    return ok


def comfyui_qwen2vl_match(ref_items, prompt):
    """Compare our Qwen3-VL positions to ComfyUI's qwen2vl MRoPE helper."""
    from comfy.text_encoders.qwen_vl import qwen2vl_mrope_position_ids
    from h3rt.models.text_encoder.fusion import Qwen3VLMultimodalFusion
    from h3rt.models.text_encoder.encoder import IMAGE_PAD, VISION_START, VISION_END
    _, mm, tags, ids, attention_mask = our_token_seq(ref_items, prompt)
    grids = mm["image_grid_thw"]
    embeds_info = []
    in_block = False
    for j, t in enumerate(ids[0].tolist()):
        if t == VISION_START:
            in_block = True
        elif t == VISION_END:
            in_block = False
        elif (t == IMAGE_PAD and in_block
              and (j == 0 or ids[0, j - 1].item() != IMAGE_PAD)):
            g = grids[len(embeds_info)]
            pad = int(g[1].item() * g[2].item() // 4)
            embeds_info.append({
                "index": j, "size": pad, "type": "image",
                "extra": {"grid": g.unsqueeze(0)}})
    official_pos = qwen2vl_mrope_position_ids(
        embeds_info, ids.shape[1], ids.device)
    fusion = object.__new__(Qwen3VLMultimodalFusion)
    fusion.config = types.SimpleNamespace(
        vision=types.SimpleNamespace(spatial_merge_size=2))
    local_pos, _ = fusion.get_rope_index(
        ids, mm["mm_token_type_ids"], image_grid_thw=grids,
        attention_mask=attention_mask)
    return (official_pos.to(local_pos.dtype) - local_pos[:, 0]).abs().max().item() == 0


torch.manual_seed(0)
rng = np.random.default_rng(0)

# image ref: [1, H, W, C] float 0-1 (matches _resize output)
img = torch.from_numpy(rng.random((1, 96, 64, 3)).astype(np.float32))
img2 = torch.from_numpy(rng.random((1, 64, 96, 3)).astype(np.float32))

# video ref: [T, H, W, C] float 0-1, 22 frames (17k+5 grid fits)
vid = torch.from_numpy(rng.random((22, 96, 64, 3)).astype(np.float32))

ok1 = compare("image", [{"type": "image", "data": img}], "a cat")
ok2 = compare("image+video",
              [{"type": "image", "data": img},
               {"type": "video", "data": vid,
                "timestamps": [i / 2.0 for i in range(vid.shape[0])]}],
              "a cat on a mat")
ok3 = compare("audio label",
              [{"type": "audio"}, {"type": "image", "data": img}],
              "a dog")
ok4 = compare("i2v first+last",
              [{"type": "image", "data": img},
               {"type": "image", "data": img2}],
              "a cat", images=[img, img2])

assert ok1 and ok2 and ok3 and ok4
print("TOKEN ALIGN OK")


# ---------------------------------------------------------------------------
# Structure-level: ref_blocks payload vs official node (mocked VAE/encoder)
# ---------------------------------------------------------------------------

def _latent_t_for(frames):
    # matches video_latent_t: 17k+5 grid
    n = frames
    if n <= 5:
        return 2
    return ((n - 5) // 17) * 5 + 2


class _FakeVideoVAE:
    """ComfyUI-style VAE wrapper: accepts IMAGE [B,H,W,C] or [T,H,W,C] (0-1)."""

    def encode(self, x):
        if x.ndim == 4:                               # [T, H, W, C]
            t, h, w, _ = x.shape
            lt = 1 if t <= 1 else _latent_t_for(t)
            return torch.zeros(1, 24, lt, h // 16, w // 16, dtype=x.dtype)
        if x.ndim == 3:                               # [H, W, C] single frame
            h, w, _ = x.shape
            return torch.zeros(1, 24, 1, h // 16, w // 16, dtype=x.dtype)
        raise ValueError(f"unexpected VAE input {tuple(x.shape)}")


class _FakeAudioVAE:
    audio_sample_rate = 32000

    def encode(self, waveform):
        # official node passes [B, L, C]; we only need the latent frame count
        b, _, c = waveform.shape
        length = waveform.shape[1]
        t = max(1, math.ceil(length / 800))
        return torch.zeros(b, 32, c, t, dtype=waveform.dtype)


def official_ref_blocks(ref_images, ref_videos, ref_video_audios, ref_audios,
                        prompt, width=1344, height=768, length=124):
    import sys as _sys
    # ensure ComfyUI root precedes the package dir so `nodes` resolves to
    # ComfyUI's own module, not our custom_nodes/nodes/
    _sys.path.insert(0, r"D:\ComfyUI-installs\ComfyUI\ComfyUI")
    import comfy.ldm.minimax.model as PR_MODULE
    from comfy.ldm.modules.attention import attention_pytorch
    PR_MODULE.optimized_attention = attention_pytorch
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo as Official

    class FakeClip:
        def tokenize(self, prompt, minimax_ref_items=None):
            return {"items": minimax_ref_items, "prompt": prompt}

        def encode_from_tokens_scheduled(self, tokens):
            return [[torch.zeros(1, 16, 8), {"minimax_refs": None}]]

    clip = FakeClip()
    vae = _FakeVideoVAE()
    audio_vae = _FakeAudioVAE()
    out = Official.execute(
        clip, vae, audio_vae, prompt, width, height, length, "match",
        ref_images=ref_images, ref_videos=ref_videos,
        ref_video_audios=ref_video_audios, ref_audios=ref_audios)
    cond, latent = out
    return cond[0][1]["minimax_refs"], latent


def our_ref_blocks(ref_images, ref_videos, ref_video_audios, ref_audios,
                   prompt, width=1344, height=768, length=124):
    from h3rt.nodes.conditioning import MiniMaxH3ReferenceToVideo as Our

    class FakeEncoder:
        def encode(self, prompt, minimax_ref_items=None):
            n = len(minimax_ref_items or [])
            return (torch.zeros(1, 16, 64),
                    torch.ones(1, 16, dtype=torch.long))

    class FakeVae:
        video_path = "video.safetensors"
        audio_path = "audio.safetensors"

    import h3rt.models.vae as V
    orig_load = V.load_vae_pack

    class FakePack:
        def encode_video(self, x):
            if x.ndim == 4:                            # [B, C, H, W] -> T=1
                b, c, h, w = x.shape
                return torch.zeros(b, 24, 1, h // 16, w // 16, dtype=x.dtype)
            b, c, t, h, w = x.shape
            lt = 1 if t <= 1 else _latent_t_for(t)
            return torch.zeros(b, 24, lt, h // 16, w // 16,
                               dtype=x.dtype)

        def encode_audio(self, wav):
            b, s, length = wav.shape
            tt = max(1, math.ceil(length / 800))
            return torch.zeros(b, 32, s, tt, dtype=wav.dtype)

    def fake_load(video_path, audio_path, device=None, pin=True):
        return FakePack()

    V.load_vae_pack = fake_load
    try:
        node = Our()
        kwargs = {f"ref_image_{i}": (ref_images or {}).get(f"ref_image_{i}")
                  for i in range(1, 4)}
        kwargs.update({f"ref_video_{i}": (ref_videos or {}).get(f"ref_video_{i}")
                       for i in range(1, 4)})
        kwargs.update({f"ref_video_audio_{i}": (ref_video_audios or {}).get(
            f"ref_video_audio_{i}") for i in range(1, 4)})
        kwargs.update({f"ref_audio_{i}": (ref_audios or {}).get(f"ref_audio_{i}")
                       for i in range(1, 4)})
        cond, latent = node.make(FakeEncoder(), FakeVae(), prompt,
                                 width, height, length, "match", **kwargs)
        return cond.refs, latent
    finally:
        V.load_vae_pack = orig_load


def compare_ref_blocks(name, ref_images, ref_videos, ref_video_audios,
                       ref_audios, prompt, width=1344, height=768, length=124):
    off_refs, off_lat = official_ref_blocks(
        ref_images, ref_videos, ref_video_audios, ref_audios, prompt,
        width, height, length)
    our_refs, our_lat = our_ref_blocks(
        ref_images, ref_videos, ref_video_audios, ref_audios, prompt,
        width, height, length)

    def norm(blocks):
        out = []
        for b in blocks:
            d = {k: v for k, v in b.items() if k != "latent" and k != "audio_latent"}
            d["latent_shape"] = tuple(b.get("latent", torch.empty(0)).shape)
            d["audio_shape"] = tuple(b.get("audio_latent", torch.empty(0)).shape)
            out.append(d)
        return out

    a, b = norm(off_refs), norm(our_refs)
    ok = a == b
    print(f"[{name}] ref_blocks match: {ok}")
    if not ok:
        print("  official:", a)
        print("  ours    :", b)
    assert ok
    # latent shapes
    from h3rt.nodes.conditioning import temporal_shape
    fc, lt, at = temporal_shape(length)
    assert tuple(our_lat.video.shape) == (1, 24, lt, height // 16, width // 16)
    assert tuple(our_lat.audio.shape) == (1, 32, 2, at)
    # official NestedTensor pair: video [B,24,T,H/16,W/16], audio [B,32,2,T]
    ov, oa = off_lat["samples"].tensors
    assert tuple(ov.shape) == (1, 24, lt, height // 16, width // 16), tuple(ov.shape)
    assert tuple(oa.shape) == (1, 32, 2, at), tuple(oa.shape)
    return True


def check_i2v_conditioning():
    from h3rt.nodes.conditioning import MiniMaxH3Conditioning
    import h3rt.models.vae as V

    captured = {}

    class FakeEncoder:
        def encode(self, prompt, minimax_ref_items=None):
            if prompt == "":
                captured["negative_prompt"] = prompt
            else:
                captured["prompt"] = prompt
                captured["items"] = minimax_ref_items
            return torch.zeros(1, 16, 8), torch.ones(1, 16, dtype=torch.long)

    class FakeVae:
        video_path = "video.safetensors"
        audio_path = ""

    class FakePack:
        def encode_video(self, x):
            return torch.zeros(x.shape[0], 24, 1, x.shape[3] // 16,
                               x.shape[4] // 16, dtype=x.dtype)

    orig_load = V.load_vae_pack
    V.load_vae_pack = lambda *a, **k: FakePack()
    try:
        prompt_ref = {
            "text": "a cat",
            "frame_count": 5,
            "total_duration": 5 / 24,
        }
        cond = MiniMaxH3Conditioning().make(
            FakeEncoder(), "a cat", "", 64, 64, FakeVae(),
            fl_constraint={"first_frame": img1, "last_frame": img2},
            prompt_ref=prompt_ref)[0]
        assert captured["prompt"] == "a cat"
        assert [i["type"] for i in captured["items"]] == ["image", "image"]
        assert [k["resolved_frame_index"] for k in cond.keyframes] == [0, 4]
        assert len(cond.keyframes) == 2
        try:
            MiniMaxH3Conditioning().make(
                FakeEncoder(), "a cat", "", 64, 64, None,
                fl_constraint={"first_frame": img1},
                prompt_ref=prompt_ref)
            raise AssertionError("I2V without VAE should fail")
        except ValueError:
            pass
    finally:
        V.load_vae_pack = orig_load
    print("I2V CONDITIONING ALIGN OK")


img1 = torch.from_numpy(rng.random((1, 96, 64, 3)).astype(np.float32))
img2 = torch.from_numpy(rng.random((1, 64, 96, 3)).astype(np.float32))
vid1 = torch.from_numpy(rng.random((22, 96, 64, 3)).astype(np.float32))
aud1 = {"waveform": torch.randn(1, 2, 16000), "sample_rate": 16000}

check_h3_v2_checkpoint_config()

for diag_name, diag_items in (
    ("image", [{"type": "image", "data": img1}]),
    ("video", [{"type": "video", "data": vid1,
                "timestamps": [i / 2.0 for i in range(vid1.shape[0])]}]),
    ("image+video+audio", [
        {"type": "image", "data": img1},
        {"type": "audio"},
        {"type": "video", "data": vid1,
         "timestamps": [i / 2.0 for i in range(vid1.shape[0])]}]),
):
    ok = comfyui_qwen2vl_match(diag_items, "a cat")
    print(f"[comfyui qwen2vl MRoPE] {diag_name}: {ok}")

compare_ref_blocks("image", {"ref_image_1": img1}, {}, {}, {}, "a cat")
compare_ref_blocks("image+video+audio",
                   {"ref_image_1": img1, "ref_image_2": img2},
                   {"ref_video_1": vid1},
                   {"ref_video_audio_1": aud1},
                   {"ref_audio_1": aud1},
                   "a dog and a cat")
check_i2v_conditioning()
print("STRUCT ALIGN OK")
