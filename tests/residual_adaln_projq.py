"""ProjQ-style residual experiment for the pruned AdaLN projection.

For selected DiT blocks, computes the full AdaLN target matrix M_i from the
BF16 model + original ckpt850 LoRA, reconstructs M_hat_i from the pruned
8-dimensional table/projection, and measures how much of the residual a
low-rank output-space correction can absorb.

Run with the ComfyUI venv python:
    python tests/residual_adaln_projq.py
"""

from __future__ import annotations

import gc
import os
import sys
import time

_NODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _NODE_ROOT)

from tests.pkg_loader import load as _load_h3rt  # noqa: E402
_load_h3rt()

_TOOL_ROOT = r"D:\MiniMax Lora Adapter for pruned model"
sys.path.insert(0, _TOOL_ROOT)

from adaln_builder.reader import SafetensorsReader  # noqa: E402
from adaln_builder.time_embed import compute_silu_grid  # noqa: E402

import torch  # noqa: E402

FULL = r"D:\AdaLN_t_table\minimax_h3_fl2va_bf16.safetensors"
ORIGINAL_LORA = r"C:\Users\Administrator\Downloads\minimax_h3_turbo_4step_ckpt850.safetensors"
PRUNED_LORA = r"D:\AdaLN_t_table\minimax_h3_pruned_turbo_ckpt850_loraWithALS.safetensors"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BLOCKS = [0, 10, 20, 30, 40, 49]
RANKS = [8, 16, 32, 64]


def randomized_svd(a, rank, oversample=16, n_iter=2):
    m, n = a.shape
    r = min(rank + oversample, min(m, n))
    omega = torch.randn(n, r, device=a.device, dtype=torch.float32)
    y = a @ omega
    for _ in range(max(0, n_iter)):
        y = a @ (a.T @ y)
        y = y / (y.norm(dim=0, keepdim=True).clamp_min(1e-12))
    q, _ = torch.linalg.qr(y)
    b = q.T @ a
    if n > r:
        ub, s, vh = torch.linalg.svd(b.T, full_matrices=False)
        vh, ub = ub.T, vh.T
    else:
        ub, s, vh = torch.linalg.svd(b, full_matrices=False)
    u = q @ ub
    return u[:, :rank].contiguous(), s[:rank].contiguous(), vh[:rank, :].contiguous()


def main():
    full_reader = SafetensorsReader(FULL)
    original_reader = SafetensorsReader(ORIGINAL_LORA)
    pruned_reader = SafetensorsReader(PRUNED_LORA)

    print("computing silu(t_emb) grid...", flush=True)
    e = compute_silu_grid(full_reader, DEVICE).float()  # [1025, 2688]
    table = pruned_reader.get_tensors(["adaln_t_table"])["adaln_t_table"].float().to(DEVICE)
    t_inv = torch.linalg.pinv(table.T @ table).to(DEVICE)

    t0 = time.time()
    for i in BLOCKS:
        prefix = f"blocks.{i}.adaln_proj.linear"
        w = full_reader.get_tensors([prefix + ".weight"])[prefix + ".weight"].to(DEVICE).float()
        bias = full_reader.get_tensors([prefix + ".bias"])[prefix + ".bias"].to(DEVICE).float()
        a = original_reader.get_tensors([prefix + ".lora_A.weight"])[prefix + ".lora_A.weight"].to(DEVICE).float()
        b = original_reader.get_tensors([prefix + ".lora_B.weight"])[prefix + ".lora_B.weight"].to(DEVICE).float()
        p = pruned_reader.get_tensors([prefix + ".weight"])[prefix + ".weight"].float().to(DEVICE)
        pb = pruned_reader.get_tensors([prefix + ".bias"])[prefix + ".bias"].float().to(DEVICE)

        m = e @ (w + b @ a).T + bias.unsqueeze(0)
        m_hat = table @ p.T + pb.unsqueeze(0)
        residual = m - m_hat
        m_norm = m.norm().item()
        rel_l2 = residual.norm().item() / max(m_norm, 1e-12)
        rel_max = residual.abs().max().item() / max(m.abs().max().item(), 1e-12)

        # Can a LoRA attached to the 8-input projection represent this residual?
        p_res = residual.T @ table @ t_inv
        p_norm = p.norm().item()
        p_res_norm = p_res.norm().item()
        projection_repair = p_res_norm / max(p_norm, 1e-12)

        print(
            f"{prefix}: rel_l2={rel_l2:.3e} rel_max={rel_max:.3e} "
            f"proj_repair_norm={projection_repair:.3e}",
            flush=True,
        )

        rank = max(RANKS)
        u, s, vh = randomized_svd(residual, rank)
        for r in RANKS:
            ur = u[:, :r]
            sr = torch.diag(s[:r])
            vr = vh[:r]
            approx = ur @ sr @ vr
            residual_after = (residual - approx).norm().item() / max(m_norm, 1e-12)
            print(
                f"  rank={r:3d}: residual_rel_l2={residual_after:.3e} "
                f"absorbed={1.0 - residual_after / max(rel_l2, 1e-12):.3f}",
                flush=True,
            )
            del approx, ur, sr, vr
            torch.cuda.empty_cache()

        del w, bias, a, b, p, pb, m, m_hat, residual, u, s, vh
        gc.collect()
        torch.cuda.empty_cache()
        print(f"elapsed={time.time()-t0:.1f}s", flush=True)

    full_reader.close()
    original_reader.close()
    pruned_reader.close()


if __name__ == "__main__":
    main()
