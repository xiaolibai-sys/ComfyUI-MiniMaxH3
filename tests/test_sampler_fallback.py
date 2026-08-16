"""SamplerRunner fallback when ModelSamplingAV is missing."""

import os
import sys
import tempfile
from unittest import mock

sys.path.insert(0, r"D:\ComfyUI-installs\ComfyUI\ComfyUI")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pkg_loader import load as _load_h3rt
_load_h3rt()

import torch
import zlib
import comfy.model_sampling
from safetensors.torch import save_file

from h3rt.models.model import MiniMaxH3Model
from h3_test_utils import h3_sample
from h3rt.utils.config import MiniMaxH3DiTConfig
from h3rt.utils.lifecycle import load_model_handle
from h3rt.utils.types import (
    AVLatent,
    H3BlockSwap,
    H3Conditioning,
    RuntimeOptions,
    TextConditioning,
)

torch.manual_seed(7)


def _build_checkpoint():
    cfg = MiniMaxH3DiTConfig(
        hidden_size=64,
        num_layers=3,
        token_refiner_num_layers=1,
        num_attention_heads=4,
        attention_head_dim=16,
        ffn_hidden_size=128,
        latents_dim=4,
        audio_latents_dim=8,
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
    path = os.path.join(tempfile.mkdtemp(prefix="h3_fallback_"), "model.safetensors")
    save_file(sd, path)
    return path


def test_fallback_when_model_sampling_av_missing():
    path = _build_checkpoint()
    handle = load_model_handle(path)
    handle.attn_backend_name = "sageattn2"
    swap = H3BlockSwap(
        enabled=True,
        block_to_swap=1,
        prefetch=True,
        prefetch_count=2,
        pin_memory=True,
        disk_workers=2,
        dtype="float32",
    )
    device = torch.device("cuda")
    video = torch.randn(1, 4, 2, 16, 16, device=device, dtype=torch.float32)
    audio = torch.randn(1, 8, 2, 8, device=device, dtype=torch.float32)
    text = torch.randn(1, 8, 64, device=device, dtype=torch.float32)
    tags = torch.ones(1, 8, dtype=torch.long, device=device)
    cond = H3Conditioning(text=TextConditioning(states=text, tags=tags))
    latent = AVLatent(video=video, audio=audio)

    try:
        with mock.patch.object(comfy.model_sampling, "ModelSamplingAV", None):
            result = h3_sample(
                handle,
                cond,
                latent,
                None,
                2,
                1.0,
                "euler",
                12.0,
                1.0,
                42,
                RuntimeOptions(swap=swap),
                disable_pbar=True,
            )
            result_cached = h3_sample(
                handle,
                cond,
                latent,
                None,
                2,
                1.0,
                "euler",
                12.0,
                1.0,
                42,
                RuntimeOptions(swap=swap),
                disable_pbar=True,
                use_adaln_cache=True,
            )
        assert torch.isfinite(result.video.float()).all()
        assert torch.isfinite(result.audio.float()).all()
        assert torch.isfinite(result_cached.video.float()).all()
        assert torch.isfinite(result_cached.audio.float()).all()
    finally:
        handle.unload()
    print("SAMPLER FALLBACK OK")


if __name__ == "__main__":
    test_fallback_when_model_sampling_av_missing()
