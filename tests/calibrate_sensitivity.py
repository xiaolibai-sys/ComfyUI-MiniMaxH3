"""Sensitivity-aware backprop calibration for AdaLN projection.

For each block, P_i/b_i are optimized with a loss that combines:
* clean-output tracking of the teacher hidden state;
* robustness to small input perturbations (input-noise sensitivity penalty).

Run with the ComfyUI venv python:
    python tests/calibrate_sensitivity.py
"""

from __future__ import annotations

import argparse
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
import torch.nn.functional as F  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from h3rt.models.model import AdalnProj  # noqa: E402

DEFAULT_OUTPUT = r"D:\AdaLN_t_table\adaln_sensitivity_ckpt850.safetensors"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", default=PRUNED_LORA)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--timesteps", default="0.5")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-sens", type=float, default=0.5)
    parser.add_argument("--noise-scale", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=NUM_LAYERS)
    args = parser.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",")]
    timesteps = [float(s) for s in args.timesteps.split(",")]
    full_reader = SafetensorsReader(FULL)
    original_reader = SafetensorsReader(ORIGINAL_LORA)
    pruned_reader = SafetensorsReader(args.lora)

    samples, mod_segments, rope = _build_samples(
        full_reader, pruned_reader, seeds, timesteps, DEVICE)
    h0 = [s[0].clone() for s in samples]
    t_orig = [s[1] for s in samples]
    t_pruned = [s[2] for s in samples]

    # Teacher trajectory.
    limit = min(args.limit, NUM_LAYERS)
    teacher = []
    for s in range(len(samples)):
        h = h0[s].clone()
        states = [h]
        for i in range(limit):
            block = _build_block(full_reader, original_reader, i, DEVICE, DTYPE)
            with torch.no_grad():
                h = block(h.clone(), t_orig[s], mod_segments, rope)
            states.append(h)
            del block
        teacher.append(states)

    out = {"adaln_t_table": pruned_reader.get_tensors(["adaln_t_table"])["adaln_t_table"]}
    t0 = time.time()
    for i in range(limit):
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
        del current

        block = _build_block(full_reader, original_reader, i, DEVICE, DTYPE)
        block.adaln_proj = adaln
        for p in block.parameters():
            p.requires_grad_(False)
        adaln.linear.weight.requires_grad_(True)
        adaln.linear.bias.requires_grad_(True)

        optimizer = torch.optim.Adam(
            [adaln.linear.weight, adaln.linear.bias], lr=args.lr)
        loss_sum = 0.0
        for step in range(args.steps):
            step_loss = 0.0
            for s in range(len(samples)):
                optimizer.zero_grad(set_to_none=True)
                clean = _forward_calib(
                    block, teacher[s][i], t_pruned[s], mod_segments, rope)
                clean_loss = F.mse_loss(
                    clean.float(), teacher[s][i + 1].float())

                h = teacher[s][i]
                noise = torch.randn_like(h, dtype=torch.float32)
                noise = noise / noise.norm() * h.float().norm() * args.noise_scale
                noisy = _forward_calib(
                    block, h.clone() + noise.to(DTYPE), t_pruned[s],
                    mod_segments, rope)
                sens_loss = F.mse_loss(
                    noisy.float(), clean.detach().float())
                loss = clean_loss + args.lambda_sens * sens_loss
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach().item())
                step_loss += float(loss.detach().item())
            print(
                f"  blocks.{i} step={step + 1}/{args.steps} "
                f"loss={step_loss / len(samples):.6e} "
                f"elapsed={time.time()-t0:.1f}s",
                flush=True,
            )

        out[prefix + ".weight"] = adaln.linear.weight.detach().cpu()
        out[prefix + ".bias"] = adaln.linear.bias.detach().cpu()
        print(
            f"blocks.{i}: loss={loss_sum / (args.steps * len(samples)):.6e} "
            f"elapsed={time.time()-t0:.1f}s",
            flush=True,
        )

        del block, adaln, optimizer
        if i % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    for suffix in (".weight", ".bias"):
        key = "final_layer.adaln_proj.linear" + suffix
        if pruned_reader.has(key):
            out[key] = pruned_reader.get_tensors([key])[key]

    full_reader.close()
    original_reader.close()
    pruned_reader.close()
    save_file(out, args.output)
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
