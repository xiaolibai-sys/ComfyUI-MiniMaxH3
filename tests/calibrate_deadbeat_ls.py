"""One-pass deadbeat least-squares calibration for per-timestep AdaLN residual.

For each block, the modulation residual is the control variable. The target is
the teacher hidden state. A small number of CGLS iterations solve the local
least-squares problem without materialising the full Jacobian.

Run with the ComfyUI venv python:
    python tests/calibrate_deadbeat_ls.py
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
from safetensors.torch import save_file  # noqa: E402
from h3rt.models.model import AdalnProj  # noqa: E402

DEFAULT_OUTPUT = r"D:\AdaLN_t_table\adaln_deadbeat_residual_rows.safetensors"


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


def _forward_stack(block, h_inputs, t_pruned, mod_segments, rope):
    outs = []
    for s in range(len(h_inputs)):
        out = _forward_calib(
            block, h_inputs[s], t_pruned[s], mod_segments, rope)
        outs.append(out.float().reshape(-1))
    return torch.cat(outs)


def _deadbeat_solve(
    block,
    h_inputs,
    t_pruned,
    mod_segments,
    rope,
    target,
    base_residual,
    iters,
    eps,
):
    block.adaln_proj._residual_row = base_residual
    y0 = _forward_stack(block, h_inputs, t_pruned, mod_segments, rope)
    e = target - y0
    delta = torch.zeros_like(base_residual)
    p = torch.zeros_like(base_residual)
    r = e.clone()

    def jvp(v):
        block.adaln_proj._residual_row = base_residual + eps * v
        yp = _forward_stack(block, h_inputs, t_pruned, mod_segments, rope)
        block.adaln_proj._residual_row = base_residual
        return (yp - y0) / eps

    def jt(y):
        u = base_residual.clone().requires_grad_(True)
        block.adaln_proj._residual_row = u
        grads = []
        off = 0
        for s in range(len(h_inputs)):
            n = h_inputs[s].numel()
            out = _forward_calib(
                block, h_inputs[s], t_pruned[s], mod_segments, rope)
            loss = (out.float().reshape(-1) * y[off:off + n]).sum()
            grads.append(torch.autograd.grad(loss, u)[0])
            off += n
        block.adaln_proj._residual_row = base_residual
        return sum(grads)

    s = jt(r)
    rs = s.dot(s).item()
    p = s.clone()
    for _ in range(iters):
        q = jvp(p)
        denom = q.dot(q).item()
        if denom < 1e-30:
            break
        alpha = rs / denom
        delta = delta + alpha * p
        r = r - alpha * q
        s_new = jt(r)
        rs_new = s_new.dot(s_new).item()
        if rs_new < 1e-30:
            break
        beta = rs_new / rs
        p = s_new + beta * p
        rs = rs_new

    block.adaln_proj._residual_row = base_residual
    return base_residual + delta


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", default=PRUNED_LORA)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--timesteps", default="0.5")
    parser.add_argument("--cg-iters", type=int, default=4)
    parser.add_argument("--eps", type=float, default=1e-2)
    parser.add_argument("--init-residual", default="")
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
    t_pruned = [s[2] for s in samples]

    # Teacher trajectory.
    teacher = []
    limit = min(args.limit, NUM_LAYERS)
    for s in range(len(samples)):
        h = h0[s].clone()
        states = [h]
        for i in range(limit):
            block = _build_block(full_reader, original_reader, i, DEVICE, DTYPE)
            with torch.no_grad():
                h = block(h.clone(), samples[s][1], mod_segments, rope)
            states.append(h)
            del block
        teacher.append(states)

    init_reader = SafetensorsReader(args.init_residual) if args.init_residual else None
    residuals = []
    for i in range(NUM_LAYERS):
        if init_reader is not None:
            key = f"blocks.{i}.adaln_residual_exact"
            row = (
                init_reader.get_tensors([key])[key].to(DEVICE)
                if init_reader.has(key) else None
            )
        else:
            row = None
        residuals.append(
            row.clone()
            if row is not None
            else torch.zeros(6 * HIDDEN * 3, device=DEVICE, dtype=torch.float32)
        )
    if init_reader is not None:
        init_reader.close()

    h_student = [h.clone() for h in h0]
    t0 = time.time()
    for i in range(limit):
        block = _student_block(
            full_reader, original_reader, pruned_reader, i, residuals[i])
        target_parts = []
        for s in range(len(samples)):
            target_parts.append(teacher[s][i + 1].float().reshape(-1))
        target = torch.cat(target_parts)

        residuals[i] = _deadbeat_solve(
            block,
            h_student,
            t_pruned,
            mod_segments,
            rope,
            target,
            residuals[i],
            args.cg_iters,
            args.eps,
        )

        with torch.no_grad():
            block.adaln_proj._residual_row = residuals[i]
            for s in range(len(samples)):
                h_student[s] = _forward_calib(
                    block, h_student[s], t_pruned[s], mod_segments, rope)

        print(
            f"blocks.{i} done  elapsed={time.time()-t0:.1f}s",
            flush=True,
        )
        del block
        if i % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    out = {}
    for i, u in enumerate(residuals):
        out[f"blocks.{i}.adaln_residual_exact"] = u.detach().cpu()
        out[f"blocks.{i}.adaln_residual_rank8"] = u.detach().cpu()

    full_reader.close()
    original_reader.close()
    pruned_reader.close()
    save_file(out, args.output)
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
