import json, collections, sys
from safetensors import safe_open
p = r"D:\ComfyUI-installs\ComfyUI\ComfyUI\models\diffusion_models\minimax_h3_fl2va_pruned_int8_convrot.safetensors"
with safe_open(p, framework="pt") as f:
    keys = list(f.keys())
    qkeys = [k for k in keys if k.endswith(".comfy_quant")]
    fmts = collections.Counter()
    convrot = collections.Counter()
    examples = []
    for k in qkeys[:30]:
        raw = bytes(f.get_tensor(k).tolist()).decode("utf-8", "replace")
        try:
            conf = json.loads(raw)
        except Exception:
            conf = {"raw": raw[:80]}
        fmts[conf.get("format")] += 1
        convrot[(conf.get("format"), conf.get("convrot"), conf.get("convrot_groupsize"), conf.get("is_weight"))] += 1
        if len(examples) < 3:
            examples.append((k, conf))
    print("formats (first 30):", dict(fmts), flush=True)
    print("convrot:", dict(convrot), flush=True)
    for k, c in examples:
        print(k, "->", c, flush=True)
    # scales present?
    scale_keys = [k for k in keys if "weight_scale" in k]
    print("scale-like keys:", len(scale_keys), flush=True)
    for k in scale_keys[:5]:
        print("  ", k, flush=True)
