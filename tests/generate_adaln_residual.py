"""Generate per-block AdaLN residual rows for the t=0.5 prototype.

The pruned 8-dimensional table/projection cannot represent residuals that are
orthogonal to the table's column space. This script saves the exact residual
row and its rank-8 low-rank reconstruction at t=0.5 so the validation script
can test a direct modulation-output residual correction.

Run with the ComfyUI venv python:
    python tests/generate_adaln_residual.py
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

from tests.residual_adaln_projq import randomized_svd  # noqa: E402

_TOOL_ROOT = r"D:\MiniMax Lora Adapter for pruned model"
sys.path.insert(0, _TOOL_ROOT)

from adaln_builder.reader import SafetensorsReader  # noqa: E402
from adaln_builder.time_embed import compute_silu_grid  # noqa: E402

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

FULL = r"D:\AdaLN_t_table\minimax_h3_fl2va_bf16.safetensors"
ORIGINAL_LORA = r"C:\Users\Administrator\Downloads\minimax_h3_turbo_4step_ckpt850.safetensors"
PRUNED_LORA = r"D:\AdaLN_t_table\minimax_h3_pruned_turbo_ckpt850_loraWithALS.safetensors"
OUTPUT = r"D:\AdaLN_t_table\adaln_residual_rows.safetensors"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROW = 512  # t=0.5 on the 1025-row grid


def _target_matrix(full_reader, lora_reader, prefix):
    weight = full_reader.get_tensors([prefix + ".weight"])[prefix + ".weight"].to(DEVICE).float()
    bias = full_reader.get_tensors([prefix + ".bias"])[prefix + ".bias"].to(DEVICE).float()
    a = lora_reader.get_tensors([prefix + ".lora_A.weight"])[prefix + ".lora_A.weight"].to(DEVICE).float()
    b = lora_reader.get_tensors([prefix + ".lora_B.weight"])[prefix + ".lora_B.weight"].to(DEVICE).float()
    merged = weight + b @ a
    del weight, a, b
    return merged, bias


def main():
    full_reader = SafetensorsReader(FULL)
    original_reader = SafetensorsReader(ORIGINAL_LORA)
    pruned_reader = SafetensorsReader(PRUNED_LORA)

    print("computing silu(t_emb) grid...", flush=True)
    e = compute_silu_grid(full_reader, DEVICE).float()
    table = pruned_reader.get_tensors(["adaln_t_table"])["adaln_t_table"].float().to(DEVICE)

    out = {}
    t0 = time.time()
    for i in range(50):
        prefix = f"blocks.{i}.adaln_proj.linear"
        merged, bias = _target_matrix(full_reader, original_reader, prefix)
        p = pruned_reader.get_tensors([prefix + ".weight"])[prefix + ".weight"].float().to(DEVICE)
        pb = pruned_reader.get_tensors([prefix + ".bias"])[prefix + ".bias"].float().to(DEVICE)
        m = e @ merged.T + bias.unsqueeze(0)
        m_hat = table @ p.T + pb.unsqueeze(0)
        residual = m - m_hat

        exact_row = residual[ROW].detach().cpu()
        u, s, vh = randomized_svd(residual, 8)
        rank8_row = (u[ROW] @ (torch.diag(s) @ vh)).detach().cpu()
        out[f"blocks.{i}.adaln_residual_exact"] = exact_row
        out[f"blocks.{i}.adaln_residual_rank8"] = rank8_row
        del merged, bias, p, pb, m, m_hat, residual, exact_row, rank8_row, u, s, vh
        gc.collect()
        torch.cuda.empty_cache()
        print(f"blocks.{i} done  elapsed={time.time()-t0:.1f}s", flush=True)

    # Final-layer AdaLN residual.
    prefix = "final_layer.adaln_proj.linear"
    merged, bias = _target_matrix(full_reader, original_reader, prefix)
    p = pruned_reader.get_tensors([prefix + ".weight"])[prefix + ".weight"].float().to(DEVICE)
    pb = pruned_reader.get_tensors([prefix + ".bias"])[prefix + ".bias"].float().to(DEVICE)
    m = e @ merged.T + bias.unsqueeze(0)
    m_hat = table @ p.T + pb.unsqueeze(0)
    residual = m - m_hat
    out["final_layer.adaln_residual_exact"] = residual[ROW].detach().cpu()
    u, s, vh = randomized_svd(residual, 8)
    out["final_layer.adaln_residual_rank8"] = (u[ROW] @ (torch.diag(s) @ vh)).detach().cpu()
    del merged, bias, p, pb, m, m_hat, residual, u, s, vh

    full_reader.close()
    original_reader.close()
    pruned_reader.close()
    save_file(out, OUTPUT)
    print(f"saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
