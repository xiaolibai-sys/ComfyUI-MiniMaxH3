"""Real H3 v2 Ref2VA VAE encode alignment.

Loads the official and local MiniMax H3 video/audio VAEs in separate
subprocesses (two video VAE copies do not fit in one 16 GB process), encodes
the same small inputs, and compares the normalized latents.

Run with the ComfyUI venv python:
    python tests/test_h3v2_vae_align.py
"""

import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, r"D:\ComfyUI-installs\ComfyUI\ComfyUI")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pkg_loader import load as _load_h3rt
_load_h3rt()

import torch

VIDEO_VAE = r"D:\ComfyUI-installs\ComfyUI\ComfyUI\models\vae\minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = r"D:\ComfyUI-installs\ComfyUI\ComfyUI\models\vae\minimax_h3_audio_vae_fp32.safetensors"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _free(*objs):
    for obj in objs:
        del obj
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _load_sd(path):
    import comfy.utils
    return comfy.utils.load_torch_file(path)


def _run_official(tmp):
    import comfy.ldm.minimax.audio_vae as PRA
    import comfy.ldm.minimax.vae as PRV
    from comfy.ldm.modules.attention import attention_pytorch
    PRV.optimized_attention = attention_pytorch
    PRA.optimized_attention = attention_pytorch

    torch.manual_seed(0)
    video_x = (torch.rand(1, 3, 35, 64, 64, device=DEVICE) * 2 - 1).half()
    sd = _load_sd(VIDEO_VAE)
    vae = PRV.MiniMaxH3VideoVAE().to(DEVICE, torch.float16)
    vae.load_state_dict(sd, strict=False)
    with torch.inference_mode():
        video_z = vae.encode(video_x)
    torch.save({"x": video_x.cpu(), "z": video_z.cpu()},
               os.path.join(tmp, "video_official.pt"))
    _free(vae, sd, video_z, video_x)

    torch.manual_seed(1)
    audio_x = (torch.rand(1, 2, 8000, device=DEVICE) * 2 - 1).float()
    sd = _load_sd(AUDIO_VAE)
    vae = PRA.MiniMaxH3AudioVAE().to(DEVICE, torch.float32)
    vae.load_state_dict(sd, strict=False)
    with torch.inference_mode():
        audio_z = vae.encode(audio_x)
    torch.save({"x": audio_x.cpu(), "z": audio_z.cpu()},
               os.path.join(tmp, "audio_official.pt"))
    _free(vae, sd, audio_z, audio_x)


def _rel_err(a, b, tol):
    a = a.float().to(DEVICE)
    b = b.float().to(DEVICE)
    assert tuple(a.shape) == tuple(b.shape), (
        f"shape {tuple(a.shape)} vs {tuple(b.shape)}")
    err = (a - b).abs().max().item() / max(1e-6, a.abs().max().item())
    assert err < tol, f"rel_err={err} > {tol}"
    return float(err)


def _run_local(tmp):
    from h3rt.models.vae import MiniMaxH3AudioVAE, MiniMaxH3VideoVAE

    ref_video = torch.load(os.path.join(tmp, "video_official.pt"),
                           weights_only=True)
    video_x = ref_video["x"].to(DEVICE).half()
    sd = _load_sd(VIDEO_VAE)
    vae = MiniMaxH3VideoVAE().to(DEVICE, torch.float16)
    vae.load_state_dict(sd, strict=False)
    with torch.inference_mode():
        video_z = vae.encode(video_x)
    video_err = _rel_err(ref_video["z"], video_z, 2e-3)
    print(f"video rel_err={video_err:.9f}", flush=True)
    _free(vae, sd, video_z, video_x, ref_video)

    ref_audio = torch.load(os.path.join(tmp, "audio_official.pt"),
                           weights_only=True)
    audio_x = ref_audio["x"].to(DEVICE)
    sd = _load_sd(AUDIO_VAE)
    vae = MiniMaxH3AudioVAE().to(DEVICE, torch.float32)
    vae.load_state_dict(sd, strict=False)
    with torch.inference_mode():
        audio_z = vae.encode(audio_x)
    audio_err = _rel_err(ref_audio["z"], audio_z, 2e-4)
    print(f"audio rel_err={audio_err:.9f}", flush=True)
    _free(vae, sd, audio_z, audio_x, ref_audio)

    with open(os.path.join(tmp, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"video_rel_err": video_err, "audio_rel_err": audio_err}, f)


def main():
    stage = os.environ.get("H3_VAE_STAGE")
    tmp = os.environ.get("H3_VAE_TMP")
    if stage == "official":
        _run_official(tmp)
        return
    if stage == "local":
        _run_local(tmp)
        return

    tmp = tempfile.mkdtemp(prefix="h3_vae_align_")
    env = dict(os.environ)
    try:
        env["H3_VAE_STAGE"] = "official"
        env["H3_VAE_TMP"] = tmp
        subprocess.check_call([sys.executable, os.path.abspath(__file__)], env=env)
        env["H3_VAE_STAGE"] = "local"
        subprocess.check_call([sys.executable, os.path.abspath(__file__)], env=env)
        with open(os.path.join(tmp, "results.json"), encoding="utf-8") as f:
            results = json.load(f)
        print(f"video rel_err={results['video_rel_err']:.9f}")
        print(f"audio rel_err={results['audio_rel_err']:.9f}")
        print("REAL H3 V2 VAE ALIGN OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
