"""Generate a rank-16 AdaLN table/projection for the ckpt850 turbo LoRA.

Uses the same PCA-init + least-squares math as the rank-8 ALS tool, with the
shared table fixed to rank 16. The output is a projection-only safetensors
file for numerical comparison.

Run with the ComfyUI venv python:
    python tests/generate_rank16_adaln.py
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
from safetensors.torch import save_file  # noqa: E402

FULL = r"D:\AdaLN_t_table\minimax_h3_fl2va_bf16.safetensors"
ORIGINAL_LORA = r"C:\Users\Administrator\Downloads\minimax_h3_turbo_4step_ckpt850.safetensors"
PRUNED_LORA = r"D:\AdaLN_t_table\minimax_h3_pruned_turbo_ckpt850_loraWithALS.safetensors"
OUTPUT = r"D:\AdaLN_t_table\adaln_turbo_rank16_ckpt850.safetensors"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANK = 16


def main():
    full_reader = SafetensorsReader(FULL)
    original_reader = SafetensorsReader(ORIGINAL_LORA)
    pruned_reader = SafetensorsReader(PRUNED_LORA)

    print("computing silu(t_emb) grid...", flush=True)
    e = compute_silu_grid(full_reader, DEVICE).float()
    table = pruned_reader.get_tensors(["adaln_t_table"])["adaln_t_table"].float().to(DEVICE)
    del table

    specs = [(f"blocks.{i}.adaln_proj.linear", 6 * 5376 * 3) for i in range(50)]
    specs.append(("final_layer.adaln_proj.linear", 2 * 5376))

    s = torch.zeros((1025, 1025), dtype=torch.float64, device=DEVICE)
    mean = torch.zeros(1025, dtype=torch.float64, device=DEVICE)
    total_features = 0

    def target(prefix):
        weight = full_reader.get_tensors([prefix + ".weight"])[prefix + ".weight"].to(DEVICE).float()
        bias = full_reader.get_tensors([prefix + ".bias"])[prefix + ".bias"].to(DEVICE).float()
        a = original_reader.get_tensors([prefix + ".lora_A.weight"])[prefix + ".lora_A.weight"].to(DEVICE).float()
        b = original_reader.get_tensors([prefix + ".lora_B.weight"])[prefix + ".lora_B.weight"].to(DEVICE).float()
        merged = weight + b @ a
        m = e @ merged.T + bias.unsqueeze(0)
        del weight, bias, a, b, merged
        return m

    print("building covariance...", flush=True)
    t0 = time.time()
    for prefix, out_dim in specs:
        m = target(prefix)
        s += m.double() @ m.T.double()
        mean += m.double().sum(dim=1)
        total_features += out_dim
        del m
        gc.collect()
        torch.cuda.empty_cache()

    mean = mean / total_features
    c = s / total_features - torch.outer(mean, mean)
    _, eigvecs = torch.linalg.eigh(c)
    t = eigvecs[:, -RANK:].to(torch.float32)
    for j in range(RANK):
        if t[0, j] < 0:
            t[:, j] = -t[:, j]
    t = t.contiguous()
    ones = torch.ones(1025, 1, dtype=torch.float32, device=DEVICE)
    x = torch.cat([t, ones], dim=1)
    xtx = x.T @ x

    out = {"adaln_t_table": t.cpu()}
    for idx, (prefix, _) in enumerate(specs):
        m = target(prefix)
        p = torch.linalg.solve(
            xtx.double(), x.double().T @ m.double()).float()
        out[prefix + ".weight"] = p[:RANK].T.contiguous().cpu()
        out[prefix + ".bias"] = p[RANK].contiguous().cpu()
        print(
            f"{idx + 1}/{len(specs)} {prefix} done  elapsed={time.time()-t0:.1f}s",
            flush=True,
        )
        del m, p
        gc.collect()
        torch.cuda.empty_cache()

    full_reader.close()
    original_reader.close()
    pruned_reader.close()
    save_file(out, OUTPUT)
    print(f"saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
