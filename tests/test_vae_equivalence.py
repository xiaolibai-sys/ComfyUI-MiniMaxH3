"""VAE equivalence test: our ports vs the ComfyUI PR models (#15224).

Memory-bounded: fp16 (video VAE working dtype) and fp32 run in separate
subprocesses so the GPU allocator starts clean.  Intermediate tensors are
released between checks.

Run:
    python tests/test_vae_equivalence.py fp16
    python tests/test_vae_equivalence.py fp32
"""

import gc
import os
import sys

# ComfyUI root first, then our package root so `utils`/`models` resolve to ours
sys.path.insert(0, r"D:\ComfyUI-installs\ComfyUI\ComfyUI")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pkg_loader import load as _load_h3rt
_load_h3rt()

import torch
import zlib

torch.manual_seed(21)

import comfy.ldm.minimax.vae as PRV
import comfy.ldm.minimax.audio_vae as PRA
from comfy.ldm.modules.attention import attention_pytorch
PRV.optimized_attention = attention_pytorch
PRA.optimized_attention = attention_pytorch

from h3rt.models import vae as OUR


def rel_err(a, b):
    a, b = a.float(), b.float()
    return (a - b).abs().max().item() / max(1e-6, b.abs().max().item())


def check(name, a, b, tol):
    assert tuple(a.shape) == tuple(b.shape), f"{name}: shape {tuple(a.shape)} vs {tuple(b.shape)}"
    err = rel_err(a, b)
    assert err < tol, f"{name}: rel_err={err} > {tol}"
    print(f"  {name}: rel_err={err:.6f}  OK", flush=True)


def load_state_into(ours, pr_model):
    sd = pr_model.state_dict()
    missing, unexpected = ours.load_state_dict(sd, strict=False)
    assert not missing, f"missing keys: {missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"


def fill_realistic(pr_model):
    """Deterministic small-scale weights so fp16 does not overflow through
    the deep conv stacks."""
    with torch.no_grad():
        for n, p in pr_model.named_parameters():
            g = torch.Generator().manual_seed(zlib.crc32(n.encode()) % (2 ** 32))
            if p.ndim >= 2:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
            else:
                # 1D params include Snake's alpha (must stay > 0) and norm
                # weights / biases; positive [0.5, 1.0] works for all of them
                p.copy_(torch.rand(p.shape, generator=g) * 0.5 + 0.5)


def free(*tensors):
    for t in tensors:
        del t
    gc.collect()
    torch.cuda.empty_cache()


device = torch.device("cuda")
DT = sys.argv[1] if len(sys.argv) > 1 else "fp16"

if DT == "fp16":
    print("=== dtype float16 (tol=3e-2) ===", flush=True)
    pr_v = PRV.MiniMaxH3VideoVAE().to(device, torch.float16)
    fill_realistic(pr_v)
    our_v = OUR.MiniMaxH3VideoVAE().to(device, torch.float16)
    load_state_into(our_v, pr_v)
    for m in (pr_v, our_v):
        m.latents_mean.zero_()
        m.latents_std.fill_(1.0)

    video_pix = (torch.rand(1, 3, 35, 64, 64, device=device) * 2 - 1).half()
    video_pix_big = (torch.rand(1, 3, 17, 320, 288, device=device) * 2 - 1).half()

    with torch.inference_mode():
        z_pr = pr_v.encode(video_pix); z_our = our_v.encode(video_pix)
        check("video encode 35f", z_our, z_pr, 3e-2); free(z_pr, z_our)

        z1_pr = pr_v.encode(video_pix[:, :, :1]); z1_our = our_v.encode(video_pix[:, :, :1])
        check("video encode 1f", z1_our, z1_pr, 3e-2)
        z1_pack = OUR.VAEPack(video_vae=our_v).encode_video(video_pix[:, :, :1].cpu())
        check("video encode 1f CPU via VAEPack", z1_pack, z1_pr, 3e-2)
        free(z1_pr, z1_our, z1_pack)

        zb_pr = pr_v.encode(video_pix_big); zb_our = our_v.encode(video_pix_big)
        check("video encode tiled 320x288", zb_our, zb_pr, 3e-2)
        free(video_pix_big, zb_pr, zb_our, video_pix)

        lat = torch.randn(1, 24, 7, 4, 4, device=device, dtype=torch.float16)
        d_pr = pr_v.decode(lat); d_our = our_v.decode(lat)
        check("video decode 35f", d_our, d_pr, 6e-2); free(lat, d_pr, d_our)

        lat_big = torch.randn(1, 24, 5, 10, 9, device=device, dtype=torch.float16)
        db_pr = pr_v.decode(lat_big); db_our = our_v.decode(lat_big)
        check("video decode tiled", db_our, db_pr, 6e-2); free(lat_big, db_pr, db_our)

        free(pr_v, our_v)

elif DT == "fp32":
    # Audio VAE only: ComfyUI runs the audio VAE in fp32 (video VAE is fp16;
    # its ViT3D decoder is ~2.4B params = ~9.7 GB fp32 per copy, two copies do
    # not fit a 16 GB card).
    print("=== dtype float32 audio only (tol=2e-4) ===", flush=True)
    pr_a = PRA.MiniMaxH3AudioVAE().to(device, torch.float32)
    fill_realistic(pr_a)
    our_a = OUR.MiniMaxH3AudioVAE().to(device, torch.float32)
    load_state_into(our_a, pr_a)
    # latents_mean/std are torch.empty() until the real checkpoint fills them;
    # use identity values for the equivalence run
    for m in (pr_a, our_a):
        m.latents_mean.zero_()
        m.latents_std.fill_(1.0)
    wav32 = (torch.rand(1, 2, 8000, device=device) * 2 - 1).float()

    with torch.inference_mode():
        za_pr = pr_a.encode(wav32); za_our = our_a.encode(wav32)
        check("audio encode fp32", za_our, za_pr, 2e-4)
        za_pack = OUR.VAEPack(audio_vae=our_a).encode_audio(wav32.cpu())
        check("audio encode fp32 CPU via VAEPack", za_pack, za_pr, 2e-4)
        free(za_pr, za_our, za_pack, wav32)
        lat_a = torch.randn(1, 32, 2, 10, device=device, dtype=torch.float32)
        da_pr = pr_a.decode(lat_a); da_our = our_a.decode(lat_a)
        check("audio decode fp32", da_our, da_pr, 2e-4); free(lat_a, da_pr, da_our, pr_a, our_a)

else:
    raise SystemExit(f"unknown dtype {DT}")

print("VAE EQUIVALENCE TEST OK", flush=True)
