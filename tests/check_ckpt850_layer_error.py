"""Layer-by-layer hidden-state error for ckpt850 pruned LoRA.

Loads real full BF16 MiniMax H3 weights and the original ckpt850 LoRA, runs
one DiT block at a time, then replaces only the AdaLN projection with the
complete pruned LoRA's fitted table/projection and runs the same block again.
The backbone LoRA tensors in the complete pruned LoRA are identical to the
original ckpt850 LoRA, so this isolates the compressed-AdaLN reconstruction
error.

Run with the ComfyUI venv python:
    python tests/check_ckpt850_layer_error.py
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

import torch  # noqa: E402
from h3rt.models.model import (  # noqa: E402
    AdalnProj,
    DiTBlock,
    FinalLayer,
    PackedLayout,
    TimeEmbedder,
    pack_audio,
    patchify_video,
    rope_rotation_table,
)

_ORIGINAL_ADALN_FORWARD = AdalnProj.forward


def _forward_with_residual(self, t_emb):
    chunks = _ORIGINAL_ADALN_FORWARD(self, t_emb)
    row = getattr(self, "_residual_row", None)
    if row is None:
        return chunks
    res = row.to(device=chunks[0].device, dtype=chunks[0].dtype).reshape(
        self.modalities, self.expand * self.hidden
    ).chunk(self.expand, dim=-1)
    return tuple(c + r for c, r in zip(chunks, res))


AdalnProj.forward = _forward_with_residual


def _attach_residual(adaln, residual_reader, key_prefix, kind):
    if residual_reader is None:
        return
    key = f"{key_prefix}.adaln_residual_{kind}"
    if not residual_reader.has(key):
        return
    adaln._residual_row = residual_reader.get_tensors([key])[key].to(DEVICE)

FULL = r"D:\AdaLN_t_table\minimax_h3_fl2va_bf16.safetensors"
ORIGINAL_LORA = r"C:\Users\Administrator\Downloads\minimax_h3_turbo_4step_ckpt850.safetensors"
PRUNED_LORA = r"D:\AdaLN_t_table\minimax_h3_pruned_turbo_ckpt850_loraWithALS.safetensors"
REPORT = os.path.join(_NODE_ROOT, "tests", "ckpt850_layer_error.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16
HIDDEN = 5376
HEADS = 56
HEAD_DIM = 128
FFN = 14336
NUM_LAYERS = 50
SEQ = 63
SIGMA = 0.5


def _load_many(reader: SafetensorsReader, names: list[str], device, dtype=None):
    out = {}
    for name in names:
        v = reader.get_tensors([name])[name]
        out[name] = v.to(device=device, dtype=dtype) if dtype is not None else v.to(device)
        del v
    return out


def _rel_max(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    denom = max(b.abs().max().item(), 1e-12)
    return (a - b).abs().max().item() / denom


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    denom = max(b.norm().item(), 1e-12)
    return (a - b).norm().item() / denom


def _mod_metric(a: tuple[torch.Tensor, ...], b: tuple[torch.Tensor, ...]):
    ca = torch.cat([x.float().reshape(-1) for x in a])
    cb = torch.cat([x.float().reshape(-1) for x in b])
    return _rel_max(ca, cb), _rel_l2(ca, cb)


def _apply_linear_lora(module, weight_key: str, reader: SafetensorsReader, device):
    a_key = weight_key[:-len("weight")] + "lora_A.weight"
    b_key = weight_key[:-len("weight")] + "lora_B.weight"
    if not reader.has(a_key) or not reader.has(b_key):
        return
    a = reader.get_tensors([a_key])[a_key].to(device, torch.float32)
    b = reader.get_tensors([b_key])[b_key].to(device, torch.float32)
    with torch.no_grad():
        module.weight.copy_((module.weight.float() + b @ a).to(module.weight.dtype))
    del a, b


def _load_original_adaln(full_reader: SafetensorsReader, lora_reader: SafetensorsReader,
                         i: int, device, dtype):
    prefix = f"blocks.{i}.adaln_proj.linear"
    weight_key, bias_key = prefix + ".weight", prefix + ".bias"
    a_key, b_key = prefix + ".lora_A.weight", prefix + ".lora_B.weight"
    weight = full_reader.get_tensors([weight_key])[weight_key].to(device, torch.float32)
    bias = full_reader.get_tensors([bias_key])[bias_key].to(device, dtype)
    a = lora_reader.get_tensors([a_key])[a_key].to(device, torch.float32)
    b = lora_reader.get_tensors([b_key])[b_key].to(device, torch.float32)
    with torch.no_grad():
        weight = (weight + b @ a).to(dtype)
    del a, b
    return weight, bias


def _load_pruned_adaln(reader: SafetensorsReader, i: int, device):
    prefix = f"blocks.{i}.adaln_proj.linear"
    return _load_many(
        reader,
        [prefix + ".weight", prefix + ".bias"],
        device,
    )


def _build_block(full_reader, original_reader, i, device, dtype):
    prefix = f"blocks.{i}."
    names = [
        prefix + "norm1.weight",
        prefix + "norm2.weight",
        prefix + "attn.q_norm.weight",
        prefix + "attn.k_norm.weight",
        prefix + "attn.qkv_proj.weight",
        prefix + "attn.out_proj.weight",
        prefix + "mlp.fc1.weight",
        prefix + "mlp.fc2.weight",
    ]
    tensors = _load_many(full_reader, names, device, dtype)
    block = DiTBlock(
        HIDDEN, HEADS, HEAD_DIM, FFN, 2688, 1e-5, 1e-5,
        apply_silu=True, adaln_dtype=dtype, dtype=dtype,
        include_adaln=True,
    ).to(device)
    with torch.no_grad():
        block.norm1.weight.copy_(tensors[prefix + "norm1.weight"])
        block.norm2.weight.copy_(tensors[prefix + "norm2.weight"])
        block.attn.q_norm.weight.copy_(tensors[prefix + "attn.q_norm.weight"])
        block.attn.k_norm.weight.copy_(tensors[prefix + "attn.k_norm.weight"])
        block.attn.qkv_proj.weight.copy_(tensors[prefix + "attn.qkv_proj.weight"])
        block.attn.out_proj.weight.copy_(tensors[prefix + "attn.out_proj.weight"])
        block.mlp.fc1.weight.copy_(tensors[prefix + "mlp.fc1.weight"])
        block.mlp.fc2.weight.copy_(tensors[prefix + "mlp.fc2.weight"])

    for target in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
        _apply_linear_lora(
            getattr(block.attn if target.startswith("attn.") else block.mlp, target.split(".")[-1]),
            prefix + target + ".weight",
            original_reader,
            device,
        )

    weight, bias = _load_original_adaln(
        full_reader, original_reader, i, device, dtype)
    with torch.no_grad():
        block.adaln_proj.linear.weight.copy_(weight)
        block.adaln_proj.linear.bias.copy_(bias)
    del tensors, weight, bias
    return block


def _replace_with_pruned_adaln(
    block, pruned_reader, i, device,
    residual_reader=None, residual_kind="exact",
):
    tensors = _load_pruned_adaln(pruned_reader, i, device)
    rank = tensors[f"blocks.{i}.adaln_proj.linear.weight"].shape[1]
    adaln = AdalnProj(
        rank, HIDDEN, 6, 3,
        apply_silu=False, dtype=torch.float32,
    ).to(device)
    with torch.no_grad():
        adaln.linear.weight.copy_(tensors[f"blocks.{i}.adaln_proj.linear.weight"])
        adaln.linear.bias.copy_(tensors[f"blocks.{i}.adaln_proj.linear.bias"])
    _attach_residual(adaln, residual_reader, f"blocks.{i}", residual_kind)
    block.adaln_proj = adaln
    del tensors


def _check_backbone_identical(orig_reader, pruned_reader):
    max_diff = 0.0
    count = 0
    for i in range(NUM_LAYERS):
        for target in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
            for suffix in (".lora_A.weight", ".lora_B.weight"):
                key = f"blocks.{i}.{target}{suffix}"
                if not orig_reader.has(key) or not pruned_reader.has(key):
                    continue
                a = orig_reader.get_tensors([key])[key].float()
                b = pruned_reader.get_tensors([key])[key].float()
                count += 1
                max_diff = max(max_diff, (a - b).abs().max().item())
                del a, b
    return max_diff, count


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", default=PRUNED_LORA)
    parser.add_argument("--report", default=REPORT)
    parser.add_argument("--residual", default="")
    parser.add_argument("--residual-kind", choices=("exact", "rank8"), default="exact")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("WARNING: CUDA unavailable; this will be much slower", flush=True)

    full_reader = SafetensorsReader(FULL)
    original_reader = SafetensorsReader(ORIGINAL_LORA)
    pruned_reader = SafetensorsReader(args.lora)
    residual_reader = (
        SafetensorsReader(args.residual) if args.residual else None
    )

    backbone_max_diff, backbone_count = _check_backbone_identical(
        original_reader, pruned_reader)
    print(
        f"backbone lora identical: {backbone_count} tensors, "
        f"max_abs_diff={backbone_max_diff:.3e}",
        flush=True,
    )

    te = TimeEmbedder(256, 5376, 2688).to(DEVICE)
    te_tensors = _load_many(
        full_reader,
        [
            "time_embedder.proj_in.weight",
            "time_embedder.proj_in.bias",
            "time_embedder.proj_out.weight",
            "time_embedder.proj_out.bias",
        ],
        DEVICE,
    )
    with torch.no_grad():
        te.proj_in.weight.copy_(te_tensors["time_embedder.proj_in.weight"])
        te.proj_in.bias.copy_(te_tensors["time_embedder.proj_in.bias"])
        te.proj_out.weight.copy_(te_tensors["time_embedder.proj_out.weight"])
        te.proj_out.bias.copy_(te_tensors["time_embedder.proj_out.bias"])
    del te_tensors

    t_orig = torch.tensor([SIGMA], device=DEVICE, dtype=torch.float32)
    with torch.no_grad():
        t_emb_orig = te(t_orig).to(DTYPE)
    table = pruned_reader.get_tensors(["adaln_t_table"])["adaln_t_table"].float().to(DEVICE)
    row = int(round(SIGMA * (table.shape[0] - 1)))
    t_emb_pruned = table[row].reshape(1, table.shape[1])
    del table

    inv_freq = full_reader.get_tensors(["rope.inv_freq"])["rope.inv_freq"].to(DEVICE)
    layout = PackedLayout(4, 1, 16, 16, 8)
    position_ids = layout.position_ids.to(DEVICE, torch.float32)
    per_axis = position_ids.unsqueeze(-1) * inv_freq.view(1, 1, -1)
    t_f, h_f, w_f = per_axis.unbind(dim=1)
    half = torch.cat((t_f, h_f, w_f), dim=-1)
    rope = rope_rotation_table(torch.cat((half, half), dim=-1), DTYPE)

    proj_t = _load_many(
        full_reader,
        [
            "video_patch_proj.weight",
            "video_patch_proj.bias",
            "audio_patch_proj.weight",
            "audio_patch_proj.bias",
        ],
        DEVICE,
    )
    torch.manual_seed(123)
    video_latent = torch.randn(1, 24, 1, 16, 16, device=DEVICE, dtype=torch.float32)
    audio_latent = torch.randn(1, 32, 2, 8, device=DEVICE, dtype=torch.float32)
    text_states = torch.randn(1, 4, HIDDEN, device=DEVICE, dtype=DTYPE)
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

    h_base = torch.empty(layout.seq_len, HIDDEN, device=DEVICE, dtype=DTYPE)
    video_off = audio_off = 0
    for a, b, kind in layout.segments:
        if kind == "text":
            h_base[a:b] = text_states[0]
        elif kind in ("cond", "ref_img", "video"):
            h_base[a:b] = video_embed[video_off:video_off + (b - a)]
            video_off += b - a
        else:
            h_base[a:b] = audio_embed[audio_off:audio_off + (b - a)]
            audio_off += b - a
    h_ours = h_base.clone()
    mod_segments = []
    for a, b, kind in layout.segments:
        row = {"text": 1, "video": 0, "audio": 2}[kind]
        mod_segments.append((a, b, row))
    del proj_t, video_latent, audio_latent, text_states, video_rows, audio_rows
    del video_embed, audio_embed, inv_freq

    rows = []
    t0 = time.time()
    for i in range(NUM_LAYERS):
        block = _build_block(full_reader, original_reader, i, DEVICE, DTYPE)
        h_shared = h_base.clone()
        with torch.inference_mode():
            base_mods = block.adaln_proj(t_emb_orig)
            h_base = block(h_shared.clone(), t_emb_orig, mod_segments, rope)
            torch.cuda.synchronize()
            _replace_with_pruned_adaln(
                block, pruned_reader, i, DEVICE,
                residual_reader, args.residual_kind,
            )
            pruned_mods = block.adaln_proj(t_emb_pruned)
            h_ours_iso = block(h_shared.clone(), t_emb_pruned, mod_segments, rope)
            h_ours = block(h_ours, t_emb_pruned, mod_segments, rope)
            torch.cuda.synchronize()

        mod_rel, mod_l2 = _mod_metric(base_mods, pruned_mods)
        iso_h_rel = _rel_max(h_ours_iso, h_base)
        iso_h_l2 = _rel_l2(h_ours_iso, h_base)
        h_rel = _rel_max(h_ours, h_base)
        h_l2 = _rel_l2(h_ours, h_base)
        rows.append({
            "layer": f"blocks.{i}",
            "adaln_rel_max": f"{mod_rel:.6e}",
            "adaln_rel_l2": f"{mod_l2:.6e}",
            "iso_hidden_rel_max": f"{iso_h_rel:.6e}",
            "iso_hidden_rel_l2": f"{iso_h_l2:.6e}",
            "hidden_rel_max": f"{h_rel:.6e}",
            "hidden_rel_l2": f"{h_l2:.6e}",
        })
        print(
            f"blocks.{i}: adaln_rel_max={mod_rel:.3e}  "
            f"iso_hidden_rel_max={iso_h_rel:.3e}  "
            f"cum_hidden_rel_l2={h_l2:.3e}",
            flush=True,
        )

        del block, base_mods, pruned_mods, h_shared, h_ours_iso
        if i % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Final-layer modulation and output comparison.
    final = FinalLayer(
        HIDDEN, 2688, 96, 32, 1e-5,
        apply_silu=True, adaln_dtype=DTYPE, dtype=DTYPE,
        include_adaln=True,
    ).to(DEVICE)
    final_names = [
        "final_layer.norm.weight",
        "final_layer.adaln_proj.linear.weight",
        "final_layer.adaln_proj.linear.bias",
        "final_layer.video_out.weight",
        "final_layer.video_out.bias",
        "final_layer.audio_out.weight",
        "final_layer.audio_out.bias",
    ]
    final_t = _load_many(full_reader, final_names, DEVICE)
    with torch.no_grad():
        final.norm.weight.copy_(final_t["final_layer.norm.weight"])
        a = original_reader.get_tensors([
            "final_layer.adaln_proj.linear.lora_A.weight",
            "final_layer.adaln_proj.linear.lora_B.weight",
        ])
        w = final_t["final_layer.adaln_proj.linear.weight"].float()
        w = (w + a["final_layer.adaln_proj.linear.lora_B.weight"].float().to(DEVICE)
             @ a["final_layer.adaln_proj.linear.lora_A.weight"].float().to(DEVICE)).to(DTYPE)
        final.adaln_proj.linear.weight.copy_(w)
        final.adaln_proj.linear.bias.copy_(final_t["final_layer.adaln_proj.linear.bias"])
        final.video_out.weight.copy_(final_t["final_layer.video_out.weight"])
        final.video_out.bias.copy_(final_t["final_layer.video_out.bias"])
        final.audio_out.weight.copy_(final_t["final_layer.audio_out.weight"])
        final.audio_out.bias.copy_(final_t["final_layer.audio_out.bias"])
    del final_t, a, w

    final_pruned = _load_many(
        pruned_reader,
        ["final_layer.adaln_proj.linear.weight", "final_layer.adaln_proj.linear.bias"],
        DEVICE,
    )
    final_rank = final_pruned["final_layer.adaln_proj.linear.weight"].shape[1]
    final_adaln = AdalnProj(
        final_rank, HIDDEN, 2, 1,
        apply_silu=False, dtype=torch.float32,
    ).to(DEVICE)
    with torch.no_grad():
        final_adaln.linear.weight.copy_(final_pruned["final_layer.adaln_proj.linear.weight"])
        final_adaln.linear.bias.copy_(final_pruned["final_layer.adaln_proj.linear.bias"])
    _attach_residual(final_adaln, residual_reader, "final_layer", args.residual_kind)
    del final_pruned

    video_seg = next((a, b, 0) for a, b, kind in layout.segments if kind == "video")
    audio_seg = next((a, b, 0) for a, b, kind in layout.segments if kind == "audio")
    with torch.inference_mode():
        base_shift, base_scale = final.adaln_proj(t_emb_orig)
        pruned_shift, pruned_scale = final_adaln(t_emb_pruned)
        final_mod_rel, final_mod_l2 = _mod_metric(
            (base_shift, base_scale), (pruned_shift, pruned_scale))
        base_v, base_a = final(h_base, t_emb_orig, video_seg, audio_seg)
        final.adaln_proj = final_adaln
        pruned_v, pruned_a = final(h_ours, t_emb_pruned, video_seg, audio_seg)
        torch.cuda.synchronize()

    final_v_rel = _rel_max(pruned_v, base_v)
    final_v_l2 = _rel_l2(pruned_v, base_v)
    final_a_rel = _rel_max(pruned_a, base_a)
    final_a_l2 = _rel_l2(pruned_a, base_a)
    rows.append({
        "layer": "final_layer",
        "adaln_rel_max": f"{final_mod_rel:.6e}",
        "adaln_rel_l2": f"{final_mod_l2:.6e}",
        "iso_hidden_rel_max": "",
        "iso_hidden_rel_l2": "",
        "hidden_rel_max": f"{h_rel:.6e}",
        "hidden_rel_l2": f"{h_l2:.6e}",
        "video_rel_max": f"{final_v_rel:.6e}",
        "video_rel_l2": f"{final_v_l2:.6e}",
        "audio_rel_max": f"{final_a_rel:.6e}",
        "audio_rel_l2": f"{final_a_l2:.6e}",
    })
    print(
        f"final_layer: adaln_rel_max={final_mod_rel:.3e}  "
        f"video_rel_max={final_v_rel:.3e}  audio_rel_max={final_a_rel:.3e}",
        flush=True,
    )

    with open(args.report, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        for row in rows[1:]:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    full_reader.close()
    original_reader.close()
    pruned_reader.close()
    print(f"elapsed={time.time()-t0:.1f}s report={args.report}", flush=True)


if __name__ == "__main__":
    main()
