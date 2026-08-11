"""Self-managed lifecycle for the streamed MiniMax H3 DiT.

Mirrors ``ComfyUI-BerniniRWrapper/utils/model_manager.py``:

* ``ModelHandle`` is a tiny lazy object (paths + swap config); weights are
  built and resident only between ``load()`` / ``unload()``;
* a module-level LRU cache (max 1) avoids re-reading the checkpoint on
  re-runs; **eviction/unload removes the handle from the cache** so a second
  workflow run builds a fresh model (fixes the stale-corpse re-run bug);
* the DiT periphery (~150 MB) is loaded to VRAM once; the 50 DiT blocks live
  in the ring-buffer BlockSwap manager;
* weight formats are auto-detected per layer (plain bf16 / int8_tensorwise /
  convrot int8 / nvfp4 ...) and bound as comfy-kitchen QuantizedTensors;
* the 2 token-refiner blocks are streamed one-shot (disk -> GPU -> run -> free).
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
from collections import OrderedDict
from typing import Optional

import torch

from .config import MiniMaxH3DiTConfig
from .stream import BlockReader
from .blockswap import BlockSwapManager, SwapBlock, free_module_storage
from .types import H3BlockSwap, H3LoraSet, SlotEntry
from ..models import quant

_MAX_MODEL_CACHE = 1
_model_cache: "OrderedDict[str, object]" = OrderedDict()
_cache_lock = threading.Lock()


def collect_garbage(aggressive: bool = False) -> None:
    """GC + CUDA sync + allocator caches flushed (host AND device).

    On Windows the PyTorch CPU caching allocator never returns pages to the
    OS by itself; ``torch._C._host_emptyCache`` + a working-set trim actually
    hand freed RAM back (mirrors BerniniRWrapper's ``vram.collect_garbage``).
    """
    import gc
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    try:
        torch._C._host_emptyCache()
    except Exception:
        pass
    if aggressive and sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
            psapi.EmptyWorkingSet.restype = ctypes.c_bool
            psapi.EmptyWorkingSet(kernel32.GetCurrentProcess())
        except Exception:
            pass
        gc.collect()


# ---------------------------------------------------------------------------
# Checkpoint scanning
# ---------------------------------------------------------------------------

def detect_key_prefix(reader: BlockReader) -> str:
    keys = reader.all_keys()
    # 1) exact unprefixed key -> ""
    if "blocks.0.attn.qkv_proj.weight" in keys:
        return ""
    # 2) otherwise the shortest prefix ending right before "blocks.0...",
    #    excluding token_refiner-style keys (those match ".blocks.0." too)
    import re
    best = None
    for key in keys:
        m = re.search(r"(^|(?<=[^.]))blocks\.0\.attn\.qkv_proj\.weight$", key)
        if m:
            prefix = key[: m.start()]
            if best is None or len(prefix) < len(best):
                best = prefix
    return best or ""


def scan_dit_config(reader: BlockReader, fallback: MiniMaxH3DiTConfig) -> MiniMaxH3DiTConfig:
    keys = set(reader.all_keys())
    prefix = detect_key_prefix(reader)
    d = fallback
    cfg = dict(
        hidden_size=d.hidden_size, num_layers=d.num_layers,
        token_refiner_num_layers=d.token_refiner_num_layers,
        num_attention_heads=d.num_attention_heads,
        attention_head_dim=d.attention_head_dim,
        ffn_hidden_size=d.ffn_hidden_size, latents_dim=d.latents_dim,
        audio_latents_dim=d.audio_latents_dim, patch_size=d.patch_size,
        text_dim=d.text_dim, timestep_input_dim=d.timestep_input_dim,
        time_embed_hidden_size=d.time_embed_hidden_size,
        time_embed_dim=d.time_embed_dim, rope_inv_freq_len=d.rope_inv_freq_len,
        norm_eps=d.norm_eps, qk_norm_eps=d.qk_norm_eps,
        final_norm_eps=d.final_norm_eps, sigma_shift_video=d.sigma_shift_video,
        sigma_shift_audio=d.sigma_shift_audio, adaln_curve_grid=d.adaln_curve_grid)
    try:
        cfg["hidden_size"] = reader.get_tensor_info(f"{prefix}video_patch_proj.weight")[0][0]
        cfg["latents_dim"] = reader.get_tensor_info(f"{prefix}final_layer.video_out.weight")[0][0] // 4
        cfg["audio_latents_dim"] = reader.get_tensor_info(f"{prefix}final_layer.audio_out.weight")[0][0]
        cfg["attention_head_dim"] = reader.get_tensor_info(f"{prefix}blocks.0.attn.q_norm.weight")[0][0]
        qkv_shape, _ = reader.get_tensor_info(f"{prefix}blocks.0.attn.qkv_proj.weight")
        cfg["num_attention_heads"] = qkv_shape[0] // (3 * cfg["attention_head_dim"])
        ffn_shape, _ = reader.get_tensor_info(f"{prefix}blocks.0.mlp.fc1.weight")
        cfg["ffn_hidden_size"] = ffn_shape[0] // 2
        cfg["text_dim"] = reader.get_tensor_info(f"{prefix}condition_proj.weight")[0][1]
        cfg["num_layers"] = sum(1 for k in keys if k.startswith(f"{prefix}blocks.") and k.endswith(".attn.qkv_proj.weight"))
        cfg["token_refiner_num_layers"] = sum(1 for k in keys if k.startswith(f"{prefix}token_refiner.blocks.") and k.endswith(".attn.qkv_proj.weight"))
        cfg["rope_inv_freq_len"] = reader.get_tensor_info(f"{prefix}rope.inv_freq")[0][0]
        if f"{prefix}adaln_t_table" in keys:
            cfg["adaln_curve_grid"] = reader.get_tensor_info(f"{prefix}adaln_t_table")[0][0]
            cfg["time_embed_dim"] = reader.get_tensor_info(f"{prefix}adaln_t_table")[0][1]
        else:
            cfg["timestep_input_dim"] = reader.get_tensor_info(f"{prefix}time_embedder.proj_in.weight")[0][1]
            cfg["time_embed_hidden_size"] = reader.get_tensor_info(f"{prefix}time_embedder.proj_in.weight")[0][0]
            cfg["time_embed_dim"] = reader.get_tensor_info(f"{prefix}time_embedder.proj_out.weight")[0][0]
    except Exception:
        pass
    return MiniMaxH3DiTConfig(**cfg)


def _is_swappable(pname: str) -> bool:
    return pname.startswith("blocks.") or pname.startswith("token_refiner.blocks.")


def _bind_periphery(module, reader, prefix, device, dtype,
                    adaln_override=None):
    """Bind every non-swappable parameter/buffer from the checkpoint."""
    from ..models.quant import bind_param
    sd = reader.get_tensors([
        f"{prefix}{pname}" for pname, _ in module.named_parameters()
        if not _is_swappable(pname) and reader.has(f"{prefix}{pname}")
    ] + [
        f"{prefix}{bname}" for bname, _ in module.named_buffers()
        if not _is_swappable(bname) and reader.has(f"{prefix}{bname}")
    ])
    for pname, p in module.named_parameters():
        if _is_swappable(pname):
            continue
        key = f"{prefix}{pname}"
        if key not in sd:
            continue
        mod = module.get_submodule(pname.rsplit(".", 1)[0]) if "." in pname else module
        leaf = pname.rsplit(".", 1)[1] if "." in pname else pname
        t = sd[key]
        if isinstance(mod, torch.nn.Linear) and quant.is_quant_layer(reader, key):
            spec = quant.load_layer_spec(reader, key, read_weight=True)
            qt = quant.make_quantized_tensor(spec, tuple(p.shape), p.dtype, qdata=t)
            bind_param(mod, leaf, qt, spec.extra_params)
        else:
            bind_param(mod, leaf, t.to(device, p.dtype))
    for bname, b in module.named_buffers():
        if _is_swappable(bname):
            continue
        key = f"{prefix}{bname}"
        if key in sd:
            mod = module.get_submodule(bname.rsplit(".", 1)[0]) if "." in bname else module
            leaf = bname.rsplit(".", 1)[1] if "." in bname else bname
            mod._buffers[leaf] = sd[key].to(device, b.dtype if b.dtype != torch.float32 else b.dtype)

    if adaln_override is not None:
        table = adaln_override.table
        module.adaln_t_table.copy_(table.to(device, module.adaln_t_table.dtype))
        final = module.final_layer.adaln_proj.linear
        if adaln_override.final_weight is not None:
            final.weight = torch.nn.Parameter(
                adaln_override.final_weight.to(device, final.weight.dtype),
                requires_grad=False)
        if adaln_override.final_bias is not None:
            final.bias = torch.nn.Parameter(
                adaln_override.final_bias.to(device, final.bias.dtype),
                requires_grad=False)


def build_dit(reader: BlockReader, dtype: torch.dtype, device: torch.device,
              swap: H3BlockSwap, include_adaln: bool = True,
              adaln_override=None
              ) -> tuple["MiniMaxH3Model", BlockSwapManager, str]:
    from ..models.model import MiniMaxH3Model
    config = scan_dit_config(reader, MiniMaxH3DiTConfig())
    prefix = detect_key_prefix(reader)

    with torch.device("meta"):
        model = MiniMaxH3Model(config, dtype=dtype, include_adaln=include_adaln)
    model._config = config
    model._key_prefix = prefix
    device = torch.device(device)

    _bind_periphery(model, reader, prefix, device, dtype, adaln_override)

    # ---- block plan (metadata-only; weights stream later) ------------------
    blocks: list[SwapBlock] = []
    for i in range(config.num_layers):
        blk = model.blocks[i]
        keys, names, refs, templates = [], [], [], []
        overrides = {}
        for pname, p in blk.named_parameters():
            key = f"{prefix}blocks.{i}.{pname}"
            keys.append(key)
            names.append(pname)
            mod = blk.get_submodule(pname.rsplit(".", 1)[0]) if "." in pname else blk
            leaf = pname.rsplit(".", 1)[1] if "." in pname else pname
            refs.append((mod, leaf, "param"))
            if (adaln_override is not None and
                    pname in ("adaln_proj.linear.weight", "adaln_proj.linear.bias")):
                leaf_name = pname.split(".")[-1]
                src = (adaln_override.block_weights.get(i)
                       if leaf_name == "weight"
                       else adaln_override.block_biases.get(i))
                if src is not None:
                    tpl = SlotEntry(data=src.to(p.dtype))
                    overrides[pname] = tpl.data
                else:
                    spec = quant.load_layer_spec(reader, key, read_weight=False)
                    tpl = quant.slot_entry_template(
                        spec, tuple(p.shape), p.dtype)
            else:
                spec = quant.load_layer_spec(reader, key, read_weight=False)
                # use the MODULE param dtype (curve-form adaln is fp32 while the
                # rest of the block is bf16) so the bound weight matches forward()
                tpl = quant.slot_entry_template(spec, tuple(p.shape), p.dtype)
                if spec.is_quant and isinstance(mod, torch.nn.Linear):
                    # swap blocks are bound from slot entries (not bind_param), so
                    # the quant dispatch patch + per-layer extras must be attached
                    # once here, at plan time.
                    quant.patch_linear(mod)
                    for ename, et in spec.extra_params.items():
                        setattr(mod, f"_{ename}", et.to(device))
            templates.append(tpl)
        for bname, b in blk.named_buffers():
            key = f"{prefix}blocks.{i}.{bname}"
            keys.append(key)
            names.append(bname)
            mod = blk.get_submodule(bname.rsplit(".", 1)[0]) if "." in bname else blk
            leaf = bname.rsplit(".", 1)[1] if "." in bname else bname
            refs.append((mod, leaf, "buffer"))
            spec = quant.load_layer_spec(reader, key, read_weight=False)
            templates.append(quant.slot_entry_template(spec, tuple(b.shape), b.dtype))
        blocks.append(SwapBlock(name=f"blocks.{i}", module=blk, keys=keys, names=names,
                                refs=refs, templates=templates,
                                overrides=overrides))

    mgr = BlockSwapManager(
        blocks, reader, device,
        window_size=swap.window_size(config.num_layers),
        prefetch=swap.prefetch,
        prefetch_count=swap.prefetch_count,
        hot_blocks=swap.hot_blocks,
        pin_memory=swap.pin_memory,
        disk_workers=swap.disk_workers,
        vram_reserve_mb=swap.vram_reserve_mb,
        runtime_lora_total_mb=swap.runtime_lora_total_mb,
        runtime_lora_fixed_mb=swap.runtime_lora_fixed_mb,
        dtype=dtype,
    )
    model._swap_mgr = mgr
    return model, mgr, prefix


# ---------------------------------------------------------------------------
# One-shot token-refiner streaming (quantized-aware)
# ---------------------------------------------------------------------------

def run_token_refiner(model: MiniMaxH3Model, reader: BlockReader, text_states: torch.Tensor,
                      device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    prefix = model._key_prefix
    cfg = model._config
    if text_states.shape[-1] != cfg.hidden_size:
        x = model.condition_proj(text_states[0].to(dtype))
    else:
        x = text_states[0].to(dtype)
    for i, blk in enumerate(model.token_refiner.blocks):
        keys, refs = [], []
        for pname, p in blk.named_parameters():
            key = f"{prefix}token_refiner.blocks.{i}.{pname}"
            keys.append(key)
            mod = blk.get_submodule(pname.rsplit(".", 1)[0]) if "." in pname else blk
            leaf = pname.rsplit(".", 1)[1] if "." in pname else pname
            refs.append((mod, leaf))
        sd = reader.get_tensors(keys)
        for key, (mod, leaf) in zip(keys, refs):
            t = sd[key]
            if isinstance(mod, torch.nn.Linear) and quant.is_quant_layer(reader, key):
                spec = quant.load_layer_spec(reader, key, read_weight=True)
                qt = quant.make_quantized_tensor(spec, tuple(mod.weight.shape), dtype, qdata=t)
                from ..models.quant import bind_param
                bind_param(mod, leaf, qt, spec.extra_params)
            else:
                mod._parameters[leaf] = torch.nn.Parameter(t.to(device, dtype), requires_grad=False)
        lora_entries = getattr(model, "_lora_token_refiner_groups", {}).get(i, [])
        if lora_entries:
            from ..models.lora import fold_lora_into_module
            fold_lora_into_module(blk, lora_entries)
        x = blk(x)
        for mod, leaf in refs:
            mod._parameters[leaf] = torch.nn.Parameter(torch.empty((0,), dtype=dtype), requires_grad=False)
    norm_key = f"{prefix}token_refiner.final_norm.weight"
    if reader.has(norm_key):
        model.token_refiner.final_norm.weight = torch.nn.Parameter(
            reader.get_tensors([norm_key])[norm_key].to(device, dtype), requires_grad=False)
    return model.token_refiner.final_norm(x).unsqueeze(0)


# ---------------------------------------------------------------------------
# Model handle + LRU cache
# ---------------------------------------------------------------------------

def _cache_key(path: str, swap: H3BlockSwap, attn_backend=None,
               include_adaln: bool = True,
               loras: Optional[H3LoraSet] = None) -> str:
    lora_sig = loras.signature() if loras else None
    raw = json.dumps(dict(path=path, block_to_swap=swap.block_to_swap, prefetch=swap.prefetch,
                          prefetch_count=swap.prefetch_count, pin=swap.pin_memory,
                          hot_blocks=swap.hot_blocks, workers=swap.disk_workers,
                          vram_reserve=swap.vram_reserve_mb,
                          runtime_lora_total=swap.runtime_lora_total_mb,
                          runtime_lora_fixed=swap.runtime_lora_fixed_mb,
                          dtype=swap.dtype, enabled=swap.enabled,
                          include_adaln=include_adaln,
                          lora=lora_sig,
                          attn=getattr(attn_backend, "backend", "auto") if attn_backend is not None else "auto"),
                     sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class _LatentFormatProxy:
    """Minimal latent_format for ``latent_preview.get_previewer``: H3's
    24-channel video latent has no TAESD decoder or Latent2RGB factors, so
    the previewer resolves to None and the callback becomes a pure progress
    bar (mirrors BerniniRWrapper's handle proxy)."""
    latent_channels = 24
    latent_rgb_factors = None
    latent_rgb_factors_bias = None
    latent_rgb_factors_reshape = None
    taesd_decoder_name = None


class _PreviewProxy:
    def __init__(self):
        self.latent_format = _LatentFormatProxy()


class ModelHandle:
    """Lazy handle: builds + streams the DiT only on ``load()``.

    The block-swap layout is chosen per sampling run via
    ``load(swap_config=...)`` (the BlockSwap args node plugs into the
    KSampler, not the loader).  A config change tears the model down and
    rebuilds it with the new window.
    """

    def __init__(self, model_path: str, attn_backend=None):
        self.model_path = str(model_path)
        self.swap = H3BlockSwap()
        self.dtype = self.swap.torch_dtype
        self.include_adaln = True
        self.attn_backend = attn_backend
        self.loras: H3LoraSet = H3LoraSet()
        self._model: Optional[MiniMaxH3Model] = None
        self._swap_mgr: Optional[BlockSwapManager] = None
        self._reader: Optional[BlockReader] = None

        # latent_preview.prepare_callback(handle, steps) interface
        self.load_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _PreviewProxy()

    # -- cache ---------------------------------------------------------------

    def _cache_key(self) -> str:
        return _cache_key(self.model_path, self.swap, self.attn_backend,
                          self.include_adaln, self.loras)

    def _cache_evict_self(self) -> None:
        with _cache_lock:
            key = self._cache_key()
            entry = _model_cache.get(key)
            if entry is self:
                _model_cache.pop(key, None)

    def _cache_put(self) -> None:
        # evict the LRU victim OUTSIDE the lock: unload() -> _cache_evict_self()
        # re-acquires _cache_lock (a plain non-reentrant Lock) -> deadlock
        victims = []
        with _cache_lock:
            _model_cache[self._cache_key()] = self
            _model_cache.move_to_end(self._cache_key())
            while len(_model_cache) > _MAX_MODEL_CACHE:
                _, oldest = _model_cache.popitem(last=False)
                if oldest is not self:
                    victims.append(oldest)
        for v in victims:
            v.unload()

    def _cache_hit(self) -> bool:
        with _cache_lock:
            entry = _model_cache.get(self._cache_key())
        if entry is not None and entry is not self and entry._model is not None:
            self._model = entry._model
            self._swap_mgr = entry._swap_mgr
            self._reader = entry._reader
            return True
        return False

    # -- lifecycle -------------------------------------------------------------

    def load(self, swap_config: Optional[H3BlockSwap] = None,
             include_adaln: bool = True):
        swap = swap_config if swap_config is not None else self.swap
        if self._model is not None:
            if swap == self.swap and self.include_adaln == include_adaln:
                return self._model
            # swap layout changed: tear down and rebuild with the new window
            self.unload()
        self.swap = swap
        self.dtype = swap.torch_dtype
        self.include_adaln = include_adaln
        if self._cache_hit():
            return self._model  # type: ignore[return-value]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._reader = BlockReader(self.model_path)
        if self.swap.enabled and device.type == "cuda":
            model, mgr, _ = build_dit(
                self._reader, self.dtype, device, self.swap,
                include_adaln=include_adaln,
                adaln_override=self.loras.adaln_override if self.loras else None)
            if self.attn_backend is not None:
                model.set_attn_backend(self.attn_backend)
            self._model, self._swap_mgr = model, mgr
        else:
            from ..models.model import MiniMaxH3Model
            cfg = scan_dit_config(self._reader, MiniMaxH3DiTConfig())
            with torch.device("meta"):
                model = MiniMaxH3Model(cfg, dtype=self.dtype,
                                       include_adaln=include_adaln)
            _bind_periphery(
                model, self._reader, detect_key_prefix(self._reader),
                device, self.dtype,
                adaln_override=self.loras.adaln_override if self.loras else None)
            model.to(device)
            model._config = cfg
            self._model = model
        self._apply_lora(self._model, self._swap_mgr)
        self._cache_put()
        return self._model

    def preprocess_text(self, text_states: torch.Tensor,
                        include_adaln: bool = True) -> torch.Tensor:
        model = self.load(include_adaln=include_adaln)
        cfg = model._config
        if text_states.shape[-1] == cfg.hidden_size:
            return text_states
        assert self._reader is not None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return run_token_refiner(model, self._reader, text_states, device, self.dtype)

    def _apply_lora(self, model, mgr) -> None:
        loras = self.loras
        if not loras:
            return
        if mgr is None:
            raise ValueError(
                "MiniMax H3 LoRA loader currently requires BlockSwap enabled."
            )
        model._lora_token_refiner_groups = loras.token_refiner_groups
        use_curves = bool(getattr(model, "use_adaln_curves", False))
        if loras.adaln_override is not None and not use_curves:
            raise ValueError(
                "complete AdaLN table override requires a pruned/curve model"
            )
        block_groups = {
            i: list(entries) for i, entries in loras.block_groups.items()
        }
        runtime_adaln: dict[int, list] = {}

        if use_curves:
            for i, entries in list(block_groups.items()):
                folded = [e for e in entries if e.target != "adaln_proj.linear"]
                adaln = [e for e in entries if e.target == "adaln_proj.linear"]
                if adaln:
                    runtime_adaln[i] = adaln
                if folded:
                    block_groups[i] = folded
                else:
                    block_groups.pop(i, None)

        from ..models.lora import fold_lora_into_module
        for i, entries in block_groups.items():
            mgr.apply_lora(i, entries)

        runtime_final: list = []
        if loras.final_adaln_entries:
            if use_curves:
                runtime_final = loras.final_adaln_entries
            else:
                fold_lora_into_module(model.final_layer, loras.final_adaln_entries)

        if use_curves and (runtime_adaln or runtime_final):
            from ..models.lora import (
                AdalnLoraState,
                attach_adaln_lora,
                load_silu_grid,
            )
            state = AdalnLoraState(
                load_silu_grid(loras.silu_grid_path),
                runtime_adaln,
                runtime_final,
            )
            attach_adaln_lora(model, state)

    def unload(self) -> None:
        self._cache_evict_self()
        if self._swap_mgr is not None:
            try:
                self._swap_mgr.shutdown()
            except Exception:
                pass
            if self._model is not None:
                try:
                    self._model._swap_mgr = None
                except Exception:
                    pass
            self._swap_mgr = None
        if self._model is not None:
            for module in list(self._model.modules()):
                try:
                    free_module_storage(module)
                except Exception:
                    pass
            self._model = None
        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:
                pass
            self._reader = None
        collect_garbage()

    def is_loaded(self) -> bool:
        return self._model is not None

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass


def load_model_handle(model_path: str, attn_backend=None) -> ModelHandle:
    return ModelHandle(model_path=model_path, attn_backend=attn_backend)


def unload_all() -> None:
    with _cache_lock:
        handles = list(_model_cache.values())
        _model_cache.clear()
    for h in handles:
        h.unload()
