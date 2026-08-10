"""Closed-loop layerwise AdaLN calibration prototype.

Keeps the shared pruned table fixed and optimizes each block's fitted
projection P_i/b_i so the compressed block output is pulled back to the full
model teacher trajectory at the block boundary. This is intentionally a small
offline prototype: a few seeds and timesteps, a few Adam steps per block.

Run with the ComfyUI venv python:
    python tests/calibrate_ckpt850_closed_loop.py
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

from adaln_builder.reader import SafetensorsReader  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from h3rt.models.model import (  # noqa: E402
    AdalnProj,
    PackedLayout,
    TimeEmbedder,
    _mod_gate,
    _mod_scale_shift,
    pack_audio,
    patchify_video,
    rope_rotation_table,
)

DEFAULT_OUTPUT = r"D:\AdaLN_t_table\adaln_turbo_closedloop_ckpt850.safetensors"


def _forward_calib(block, x, t_emb, mod_segments, rope):
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)
    h = _mod_scale_shift(block.norm1(x.clone()), shift_msa, scale_msa, mod_segments)
    x = _mod_gate(x.clone(), gate_msa, block.attn(h, rope_freqs=rope), mod_segments)
    h = _mod_scale_shift(block.norm2(x.clone()), shift_mlp, scale_mlp, mod_segments)
    return _mod_gate(x.clone(), gate_mlp, block.mlp(h), mod_segments)


def _gauss_newton_calibrate(
    block,
    samples,
    h_teacher,
    h_student,
    teacher_next,
    mod_segments,
    rope,
    mode,
    iters,
    basis_size,
    eps,
    verbose,
):
    p = block.adaln_proj.linear.weight
    b = block.adaln_proj.linear.bias
    s_count = len(samples)

    def _residual(inp, s):
        out = _forward_calib(
            block,
            inp,
            samples[s][2],
            mod_segments,
            rope,
        )
        return (out.float() - teacher_next[s].float()).reshape(-1)

    for it in range(iters):
        residuals = []
        grads = []
        for s in range(s_count):
            inp = h_teacher[s] if mode == "teacher" else h_student[s]
            res = _residual(inp, s)
            residuals.append(res)
            loss = res.pow(2).mean()
            gp, gb = torch.autograd.grad(loss, [p, b])
            grads.append((gp.detach(), gb.detach()))

        r0 = torch.cat(residuals)
        k_count = min(basis_size, s_count)
        basis = []
        cols = []
        for k in range(k_count):
            gp, gb = grads[k]
            norm = gp.norm() + gb.norm()
            dp = gp / norm.clamp_min(1e-12)
            db = gb / norm.clamp_min(1e-12)
            basis.append((dp, db))

            p2 = p.detach() + eps * dp
            b2 = b.detach() + eps * db
            with torch.no_grad():
                p.copy_(p2)
                b.copy_(b2)
            r2_parts = [
                _residual(h_teacher[s] if mode == "teacher" else h_student[s], s)
                for s in range(s_count)
            ]
            r2 = torch.cat(r2_parts)
            cols.append(((r2 - r0) / eps).unsqueeze(1))
            with torch.no_grad():
                p.copy_(p2 - eps * dp)
                b.copy_(b2 - eps * db)

        jac = torch.cat(cols, dim=1)
        coef = torch.linalg.lstsq(
            jac.cpu(), (-r0).cpu().unsqueeze(1), driver="gelsy"
        ).solution.squeeze(1).to(p.device)

        step_p = torch.zeros_like(p)
        step_b = torch.zeros_like(b)
        for k in range(k_count):
            dp, db = basis[k]
            step_p = step_p + coef[k] * dp
            step_b = step_b + coef[k] * db
        base_norm = p.norm() + b.norm()
        step_norm = step_p.norm() + step_b.norm()
        scale = min(1.0, (0.1 * base_norm / step_norm.clamp_min(1e-12)).item())
        with torch.no_grad():
            p.add_(scale * step_p)
            b.add_(scale * step_b)

        loss_sum = 0.0
        for s in range(s_count):
            inp = h_teacher[s] if mode == "teacher" else h_student[s]
            res = _residual(inp, s)
            loss_sum += float(res.pow(2).mean().item())
        loss_value = loss_sum / s_count
        if verbose:
            print(
                f"  gn_iter={it + 1}/{iters} loss={loss_value:.6e} "
                f"step_norm={step_norm.item():.4e}",
                flush=True,
            )
    return loss_value


def _build_samples(full_reader, pruned_reader, seeds, timesteps, device):
    layout = PackedLayout(4, 1, 16, 16, 8)
    inv_freq = full_reader.get_tensors(["rope.inv_freq"])["rope.inv_freq"].to(device)
    position_ids = layout.position_ids.to(device, torch.float32)
    per_axis = position_ids.unsqueeze(-1) * inv_freq.view(1, 1, -1)
    t_f, h_f, w_f = per_axis.unbind(dim=1)
    half = torch.cat((t_f, h_f, w_f), dim=-1)
    rope = rope_rotation_table(torch.cat((half, half), dim=-1), DTYPE)
    del inv_freq

    te = TimeEmbedder(256, 5376, 2688).to(device)
    te_tensors = _load_many(
        full_reader,
        [
            "time_embedder.proj_in.weight",
            "time_embedder.proj_in.bias",
            "time_embedder.proj_out.weight",
            "time_embedder.proj_out.bias",
        ],
        device,
    )
    with torch.no_grad():
        te.proj_in.weight.copy_(te_tensors["time_embedder.proj_in.weight"])
        te.proj_in.bias.copy_(te_tensors["time_embedder.proj_in.bias"])
        te.proj_out.weight.copy_(te_tensors["time_embedder.proj_out.weight"])
        te.proj_out.bias.copy_(te_tensors["time_embedder.proj_out.bias"])
    del te_tensors

    proj_t = _load_many(
        full_reader,
        [
            "video_patch_proj.weight",
            "video_patch_proj.bias",
            "audio_patch_proj.weight",
            "audio_patch_proj.bias",
        ],
        device,
    )
    table = pruned_reader.get_tensors(["adaln_t_table"])["adaln_t_table"].float().to(device)

    mod_segments = []
    for a, b, kind in layout.segments:
        mod_segments.append((a, b, {"text": 1, "video": 0, "audio": 2}[kind]))

    samples = []
    for seed in seeds:
        torch.manual_seed(seed)
        video_latent = torch.randn(1, 24, 1, 16, 16, device=device, dtype=torch.float32)
        audio_latent = torch.randn(1, 32, 2, 8, device=device, dtype=torch.float32)
        text_states = torch.randn(1, 4, HIDDEN, device=device, dtype=DTYPE)
        video_rows = patchify_video(video_latent)
        audio_rows = pack_audio(audio_latent)
        video_embed = (
            video_rows @ proj_t["video_patch_proj.weight"].T
            + proj_t["video_patch_proj.bias"]
        ).to(DTYPE)
        audio_embed = (
            audio_rows @ proj_t["audio_patch_proj.weight"].T
            + proj_t["audio_patch_proj.bias"]
        ).to(DTYPE)

        h0 = torch.empty(layout.seq_len, HIDDEN, device=device, dtype=DTYPE)
        video_off = audio_off = 0
        for a, b, kind in layout.segments:
            if kind == "text":
                h0[a:b] = text_states[0]
            elif kind in ("cond", "ref_img", "video"):
                h0[a:b] = video_embed[video_off:video_off + (b - a)]
                video_off += b - a
            else:
                h0[a:b] = audio_embed[audio_off:audio_off + (b - a)]
                audio_off += b - a

        for t in timesteps:
            with torch.no_grad():
                t_orig = te(torch.tensor([t], device=device, dtype=torch.float32)).to(DTYPE)
            pos = t * (table.shape[0] - 1)
            i0 = int(pos)
            frac = pos - i0
            t_pruned = torch.lerp(table[i0], table[i0 + 1], frac).reshape(1, 8)
            samples.append((h0.clone(), t_orig, t_pruned))

    del proj_t, table, video_latent, audio_latent, text_states, video_rows, audio_rows
    del video_embed, audio_embed
    return samples, mod_segments, rope


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", default=PRUNED_LORA)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--timesteps", default="0.5")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--limit", type=int, default=NUM_LAYERS)
    parser.add_argument("--mode", choices=("closed", "teacher"), default="closed")
    parser.add_argument("--method", choices=("adam", "gn"), default="gn")
    parser.add_argument("--gn-iters", type=int, default=2)
    parser.add_argument("--basis", type=int, default=3)
    parser.add_argument("--eps", type=float, default=1e-2)
    parser.add_argument("--verbose", action="store_true")
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

    out_sd = {"adaln_t_table": pruned_reader.get_tensors(["adaln_t_table"])["adaln_t_table"]}
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

        for p in block.parameters():
            p.requires_grad_(False)
        adaln.linear.weight.requires_grad_(True)
        adaln.linear.bias.requires_grad_(True)
        block.adaln_proj = adaln

        if args.method == "adam":
            optimizer = torch.optim.Adam(
                [adaln.linear.weight, adaln.linear.bias], lr=args.lr)
            loss_sum = 0.0
            for step in range(args.steps):
                for s in range(len(samples)):
                    optimizer.zero_grad(set_to_none=True)
                    calib_input = (
                        h_teacher[s] if args.mode == "teacher" else h_student[s])
                    out = _forward_calib(
                        block,
                        calib_input,
                        samples[s][2],
                        mod_segments,
                        rope,
                    )
                    loss = F.mse_loss(out.float(), teacher_next[s].float())
                    loss.backward()
                    optimizer.step()
                    loss_sum += float(loss.detach().item())
                    if args.verbose:
                        print(
                            f"  step={step + 1}/{args.steps} sample={s} "
                            f"loss={float(loss.detach().item()):.6e}",
                            flush=True,
                        )
            loss_value = loss_sum / (args.steps * len(samples))
        else:
            loss_value = _gauss_newton_calibrate(
                block,
                samples,
                h_teacher,
                h_student,
                teacher_next,
                mod_segments,
                rope,
                args.mode,
                args.gn_iters,
                args.basis,
                args.eps,
                args.verbose,
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

        out_sd[prefix + ".weight"] = adaln.linear.weight.detach().cpu()
        out_sd[prefix + ".bias"] = adaln.linear.bias.detach().cpu()
        print(
            f"blocks.{i}: loss={loss_value:.6e}  elapsed={time.time()-t0:.1f}s",
            flush=True,
        )

        del block, adaln, teacher_next
        if args.method == "adam":
            del optimizer
        if i % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    full_reader.close()
    original_reader.close()
    for suffix in (".weight", ".bias"):
        key = "final_layer.adaln_proj.linear" + suffix
        if pruned_reader.has(key):
            out_sd[key] = pruned_reader.get_tensors([key])[key]
    pruned_reader.close()
    save_file(out_sd, args.output)
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
