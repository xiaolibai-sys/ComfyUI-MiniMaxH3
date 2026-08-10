"""GPTAQ-style asymmetric calibration with a per-timestep residual vector.

Unlike projection-only calibration, this prototype optimizes a direct residual
vector on the modulation output for each block. The input is the actual
student trajectory and the target is the full model teacher output, which is
the same "asymmetric calibration" idea as GPTAQ.

Run with the ComfyUI venv python:
    python tests/calibrate_gptaq_residual.py
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

DEFAULT_OUTPUT = r"D:\AdaLN_t_table\adaln_gptaq_residual_rows.safetensors"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", default=PRUNED_LORA)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--timesteps", default="0.5")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--limit", type=int, default=NUM_LAYERS)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--init-residual", default="")
    args = parser.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",")]
    timesteps = [float(s) for s in args.timesteps.split(",")]
    full_reader = SafetensorsReader(FULL)
    original_reader = SafetensorsReader(ORIGINAL_LORA)
    pruned_reader = SafetensorsReader(args.lora)

    samples, mod_segments, rope = _build_samples(
        full_reader, pruned_reader, seeds, timesteps, DEVICE)
    h_teacher = [s[0].clone() for s in samples]
    h_student = [s[0].clone() for s in samples]

    out = {}
    t0 = time.time()
    for i in range(min(args.limit, NUM_LAYERS)):
        block = _build_block(full_reader, original_reader, i, DEVICE, DTYPE)
        teacher_next = []
        with torch.no_grad():
            for s in range(len(samples)):
                teacher_next.append(
                    block(h_teacher[s].clone(), samples[s][1], mod_segments, rope))

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

        if args.init_residual:
            init_reader = SafetensorsReader(args.init_residual)
            init_key = f"blocks.{i}.adaln_residual_exact"
            init_row = (
                init_reader.get_tensors([init_key])[init_key].to(DEVICE)
                if init_reader.has(init_key) else None
            )
            init_reader.close()
            if init_row is None:
                raise ValueError(f"missing init residual {init_key}")
        else:
            init_row = None
        residual = (
            init_row.clone()
            if init_row is not None
            else torch.zeros(6 * HIDDEN * 3, device=DEVICE, dtype=torch.float32)
        )
        residual.requires_grad_(True)
        adaln._residual_row = residual
        block.adaln_proj = adaln
        for p in block.parameters():
            p.requires_grad_(False)
        residual.requires_grad_(True)

        optimizer = torch.optim.Adam([residual], lr=args.lr)
        loss_sum = 0.0
        for step in range(args.steps):
            for s in range(len(samples)):
                optimizer.zero_grad(set_to_none=True)
                out_s = _forward_calib(
                    block,
                    h_student[s],
                    samples[s][2],
                    mod_segments,
                    rope,
                )
                loss = F.mse_loss(out_s.float(), teacher_next[s].float())
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach().item())
                if args.verbose:
                    print(
                        f"  step={step + 1}/{args.steps} sample={s} "
                        f"loss={float(loss.detach().item()):.6e}",
                        flush=True,
                    )

        with torch.no_grad():
            for s in range(len(samples)):
                h_teacher[s] = teacher_next[s]
                h_student[s] = _forward_calib(
                    block,
                    h_student[s],
                    samples[s][2],
                    mod_segments,
                    rope,
                )

        out[f"blocks.{i}.adaln_residual_exact"] = residual.detach().cpu()
        out[f"blocks.{i}.adaln_residual_rank8"] = residual.detach().cpu()
        print(
            f"blocks.{i}: loss={loss_sum / (args.steps * len(samples)):.6e} "
            f"elapsed={time.time()-t0:.1f}s",
            flush=True,
        )

        del block, adaln, optimizer, residual, teacher_next
        if i % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    full_reader.close()
    original_reader.close()
    pruned_reader.close()
    save_file(out, args.output)
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
