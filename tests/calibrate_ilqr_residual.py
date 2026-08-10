"""Simplified iLQR/BPTT control calibration for AdaLN residual vectors.

Each block's per-timestep modulation residual is treated as a control input.
The prototype simulates the compressed trajectory, computes teacher hidden
states, then back-propagates the costate through the blocks one at a time and
updates all control vectors with Adam. It uses gradient checkpointing at the
block level so the full 50-block model never has to reside in memory at once.

Run with the ComfyUI venv python:
    python tests/calibrate_ilqr_residual.py
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

DEFAULT_OUTPUT = r"D:\AdaLN_t_table\adaln_ilqr_residual_rows.safetensors"


def _student_block(full_reader, original_reader, pruned_reader, i, residual):
    block = _build_block(full_reader, original_reader, i, DEVICE, DTYPE)
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
    adaln._residual_row = residual
    block.adaln_proj = adaln
    for p in block.parameters():
        p.requires_grad_(False)
    del current
    return block


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", default=PRUNED_LORA)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--timesteps", default="0.5")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--init-residual", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",")]
    timesteps = [float(s) for s in args.timesteps.split(",")]
    full_reader = SafetensorsReader(FULL)
    original_reader = SafetensorsReader(ORIGINAL_LORA)
    pruned_reader = SafetensorsReader(args.lora)

    samples, mod_segments, rope = _build_samples(
        full_reader, pruned_reader, seeds, timesteps, DEVICE)
    h0 = [s[0].clone() for s in samples]
    t_pruned = [s[2] for s in samples]

    # Teacher hidden trajectory.
    teacher = []
    for s in range(len(samples)):
        h = h0[s].clone()
        states = [h]
        for i in range(NUM_LAYERS):
            block = _build_block(full_reader, original_reader, i, DEVICE, DTYPE)
            with torch.no_grad():
                h = block(h.clone(), samples[s][1], mod_segments, rope)
            states.append(h)
            del block
        teacher.append(states)

    # Controls.
    controls = []
    init_reader = SafetensorsReader(args.init_residual) if args.init_residual else None
    for i in range(NUM_LAYERS):
        if init_reader is not None:
            key = f"blocks.{i}.adaln_residual_exact"
            row = (
                init_reader.get_tensors([key])[key].to(DEVICE)
                if init_reader.has(key) else None
            )
        else:
            row = None
        u = (
            row.clone()
            if row is not None
            else torch.zeros(6 * HIDDEN * 3, device=DEVICE, dtype=torch.float32)
        )
        u.requires_grad_(True)
        controls.append(u)
    if init_reader is not None:
        init_reader.close()

    optimizer = torch.optim.Adam(controls, lr=args.lr)
    t0 = time.time()
    for it in range(args.iterations):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for s in range(len(samples)):
            fwd = [h0[s].clone()]
            # Forward simulate with current controls.
            for i in range(NUM_LAYERS):
                block = _student_block(
                    full_reader, original_reader, pruned_reader, i, controls[i])
                with torch.no_grad():
                    h = _forward_calib(
                        block, fwd[-1], t_pruned[s], mod_segments, rope)
                fwd.append(h.detach())
                del block

            # Backward costate through blocks, one block at a time.
            grad_h = None
            for i in reversed(range(NUM_LAYERS)):
                block = _student_block(
                    full_reader, original_reader, pruned_reader, i, controls[i])
                h_prev = fwd[i].clone().requires_grad_(True)
                h_i = _forward_calib(
                    block, h_prev, t_pruned[s], mod_segments, rope)
                local = F.mse_loss(h_i.float(), teacher[s][i + 1].float())
                future = (
                    (grad_h * h_i.float()).sum()
                    if grad_h is not None
                    else torch.tensor(0.0, device=DEVICE)
                )
                total = local + future
                g_h_prev, g_u = torch.autograd.grad(
                    total, [h_prev, controls[i]], retain_graph=False)
                grad_h = g_h_prev
                if controls[i].grad is None:
                    controls[i].grad = g_u.clone()
                else:
                    controls[i].grad.add_(g_u)
                total_loss += float(local.detach().item())
                del block, h_i, local, future, total, g_h_prev, g_u
            del fwd

        optimizer.step()
        print(
            f"iter={it + 1}/{args.iterations} "
            f"loss={total_loss / (len(samples) * NUM_LAYERS):.6e} "
            f"elapsed={time.time()-t0:.1f}s",
            flush=True,
        )

    out = {}
    for i, u in enumerate(controls):
        out[f"blocks.{i}.adaln_residual_exact"] = u.detach().cpu()
        out[f"blocks.{i}.adaln_residual_rank8"] = u.detach().cpu()

    full_reader.close()
    original_reader.close()
    pruned_reader.close()
    save_file(out, args.output)
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
