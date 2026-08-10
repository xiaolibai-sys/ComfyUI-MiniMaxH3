"""Check whether the pruned AdaLN table attenuates timestep derivatives.

Motion is driven by how AdaLN modulation changes across timesteps. This script
compares full-model modulation and pruned table/projection modulation across
the 1025-row grid, focusing on derivative norms and direction alignment.

Run with the ComfyUI venv python:
    python tests/analyze_adaln_temporal.py
"""

from __future__ import annotations

import argparse
import csv
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
REPORT = os.path.join(_NODE_ROOT, "tests", "adaln_temporal.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _derivative(x):
    # central difference across the 1025-row grid
    return x[2:] - x[:-2]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", default="0,10,20,30,40,49")
    args = parser.parse_args(argv)
    block_ids = [int(s) for s in args.blocks.split(",")]

    full_reader = SafetensorsReader(FULL)
    original_reader = SafetensorsReader(ORIGINAL_LORA)
    pruned_reader = SafetensorsReader(PRUNED_LORA)

    print("computing silu(t_emb) grid...", flush=True)
    e = compute_silu_grid(full_reader, DEVICE).float()
    table = pruned_reader.get_tensors(["adaln_t_table"])["adaln_t_table"].float().to(DEVICE)
    rows = []
    t0 = time.time()

    for i in block_ids:
        prefix = f"blocks.{i}.adaln_proj.linear"
        w = full_reader.get_tensors([prefix + ".weight"])[prefix + ".weight"].to(DEVICE).float()
        bias = full_reader.get_tensors([prefix + ".bias"])[prefix + ".bias"].to(DEVICE).float()
        a = original_reader.get_tensors([prefix + ".lora_A.weight"])[prefix + ".lora_A.weight"].to(DEVICE).float()
        b = original_reader.get_tensors([prefix + ".lora_B.weight"])[prefix + ".lora_B.weight"].to(DEVICE).float()
        p = pruned_reader.get_tensors([prefix + ".weight"])[prefix + ".weight"].float().to(DEVICE)
        pb = pruned_reader.get_tensors([prefix + ".bias"])[prefix + ".bias"].float().to(DEVICE)

        m_full = e @ (w + b @ a).T + bias.unsqueeze(0)
        m_hat = table @ p.T + pb.unsqueeze(0)

        rel_l2 = (m_full - m_hat).norm().item() / max(m_full.norm().item(), 1e-12)
        d_full = _derivative(m_full)
        d_hat = _derivative(m_hat)
        d_norm_ratio = d_hat.norm().item() / max(d_full.norm().item(), 1e-12)
        d_cos = (
            (d_full.reshape(-1) @ d_hat.reshape(-1)).item()
            / max(d_full.norm().item() * d_hat.norm().item(), 1e-12)
        )
        rows.append({
            "layer": f"blocks.{i}",
            "mod_rel_l2": f"{rel_l2:.6e}",
            "derivative_norm_ratio": f"{d_norm_ratio:.4f}",
            "derivative_cosine": f"{d_cos:.4f}",
        })
        print(
            f"blocks.{i}: mod_rel_l2={rel_l2:.3e} "
            f"d_norm_ratio={d_norm_ratio:.4f} d_cos={d_cos:.4f} "
            f"elapsed={time.time()-t0:.1f}s",
            flush=True,
        )

        del w, bias, a, b, p, pb, m_full, m_hat, d_full, d_hat
        gc.collect()
        torch.cuda.empty_cache()

    with open(REPORT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    full_reader.close()
    original_reader.close()
    pruned_reader.close()
    print(f"saved {REPORT}", flush=True)


if __name__ == "__main__":
    main()
