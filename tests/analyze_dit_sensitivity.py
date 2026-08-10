"""DiT activation sensitivity and error-structure analysis.

For each real block, this script computes:
* teacher-forcing single-block hidden error of the pruned AdaLN model;
* input perturbation gain (how hidden-state errors are amplified);
* modulation perturbation gain (how AdaLN errors affect the output);
* pairwise cosine similarity of single-block error directions across seeds.

Run with the ComfyUI venv python:
    python tests/analyze_dit_sensitivity.py
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

from tests.check_ckpt850_layer_error import (  # noqa: E402
    DEVICE,
    DTYPE,
    FFN,
    FULL,
    HEAD_DIM,
    HEADS,
    HIDDEN,
    NUM_LAYERS,
    ORIGINAL_LORA,
    PRUNED_LORA,
    _build_block,
    _load_many,
)
from tests.calibrate_ckpt850_closed_loop import (  # noqa: E402
    _build_samples,
    _forward_calib,
)

from adaln_builder.reader import SafetensorsReader  # noqa: E402

import torch  # noqa: E402
from h3rt.models.model import AdalnProj  # noqa: E402

REPORT = os.path.join(_NODE_ROOT, "tests", "dit_sensitivity.csv")
SEEDS = [0, 1, 2]


def _replace_pruned(block, pruned_reader, i):
    prefix = f"blocks.{i}.adaln_proj.linear"
    current = _load_many(
        pruned_reader,
        [prefix + ".weight", prefix + ".bias"],
        DEVICE,
    )
    adaln = AdalnProj(
        8, HIDDEN, 6, 3,
        apply_silu=False, dtype=torch.float32,
    ).to(DEVICE)
    with torch.no_grad():
        adaln.linear.weight.copy_(current[prefix + ".weight"])
        adaln.linear.bias.copy_(current[prefix + ".bias"])
    block.adaln_proj = adaln
    del current


def _rel_l2(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return (a - b).norm().item() / max(b.norm().item(), 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--timesteps", default="0.5")
    parser.add_argument("--blocks", default="0,10,20,30,40,49")
    parser.add_argument("--input-perts", type=int, default=1)
    parser.add_argument("--mod-perts", type=int, default=1)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    timesteps = [float(s) for s in args.timesteps.split(",")]
    block_ids = [int(s) for s in args.blocks.split(",")]

    full_reader = SafetensorsReader(FULL)
    original_reader = SafetensorsReader(ORIGINAL_LORA)
    pruned_reader = SafetensorsReader(PRUNED_LORA)
    samples, mod_segments, rope = _build_samples(
        full_reader, pruned_reader, seeds, timesteps, DEVICE)
    h0 = [s[0].clone() for s in samples]
    t_orig = [s[1] for s in samples]
    t_pruned = [s[2] for s in samples]

    rows = []
    t0 = time.time()
    for i in block_ids:
        block = _build_block(full_reader, original_reader, i, DEVICE, DTYPE)
        teacher_outs = []
        with torch.no_grad():
            for s in range(len(samples)):
                out = block(h0[s].clone(), t_orig[s], mod_segments, rope)
                teacher_outs.append(out)

        _replace_pruned(block, pruned_reader, i)
        student_outs = []
        iso_l2 = []
        input_gains = []
        mod_gains = []
        error_vecs = []

        for s in range(len(samples)):
            with torch.no_grad():
                student_out = _forward_calib(
                    block, h0[s], t_pruned[s], mod_segments, rope)
                student_outs.append(student_out)
                error = student_out.float() - teacher_outs[s].float()
                error_vecs.append(error)
                iso_l2.append(_rel_l2(student_out, teacher_outs[s]))

                # Input perturbation gain.
                for p in range(args.input_perts):
                    torch.manual_seed(1000 + i * 10 + s * 10 + p)
                    delta = torch.randn_like(student_out)
                    delta = delta / delta.norm() * student_out.float().norm().item() * 1e-3
                    out_pert = _forward_calib(
                        block, h0[s].clone() + delta.to(DTYPE),
                        t_pruned[s], mod_segments, rope)
                    input_gains.append(
                        (out_pert.float() - student_out.float()).norm().item()
                        / max(delta.float().norm().item(), 1e-12)
                    )

                # Modulation perturbation gain.
                for p in range(args.mod_perts):
                    torch.manual_seed(2000 + i * 10 + s * 10 + p)
                    mod_pert = torch.randn(6 * HIDDEN * 3, device=DEVICE, dtype=torch.float32)
                    mod_pert = mod_pert / mod_pert.norm() * 1e-2
                    block.adaln_proj._residual_row = mod_pert
                    out_mod = _forward_calib(
                        block, h0[s], t_pruned[s], mod_segments, rope)
                    mod_gains.append(
                        (out_mod.float() - student_out.float()).norm().item()
                        / max(mod_pert.norm().item(), 1e-12)
                    )
                    block.adaln_proj._residual_row = None

        cosines = []
        for a in range(len(error_vecs)):
            for b in range(a + 1, len(error_vecs)):
                ea = error_vecs[a].reshape(-1)
                eb = error_vecs[b].reshape(-1)
                denom = max(ea.norm().item() * eb.norm().item(), 1e-12)
                cosines.append((ea @ eb).item() / denom)

        mean_iso = sum(iso_l2) / len(iso_l2)
        mean_in = sum(input_gains) / len(input_gains)
        mean_mod = sum(mod_gains) / len(mod_gains)
        mean_cos = sum(cosines) / len(cosines) if cosines else 0.0
        rows.append({
            "layer": f"blocks.{i}",
            "iso_hidden_rel_l2": f"{mean_iso:.6e}",
            "input_gain": f"{mean_in:.6e}",
            "mod_gain": f"{mean_mod:.6e}",
            "mod_over_input_gain": f"{mean_mod / max(mean_in, 1e-12):.6e}",
            "error_cosine_seeds": f"{mean_cos:.4f}",
        })
        print(
            f"blocks.{i}: iso_l2={mean_iso:.3e} input_gain={mean_in:.3e} "
            f"mod_gain={mean_mod:.3e} cos={mean_cos:.3f} "
            f"elapsed={time.time()-t0:.1f}s",
            flush=True,
        )

        del block, teacher_outs, student_outs, error_vecs
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
