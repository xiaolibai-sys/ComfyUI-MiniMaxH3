"""Streaming disk load/destroy engine for the text encoder.

Design mirrors ``ComfyUI-BerniniRWrapper``'s ``utils/block_reader.py`` +
``utils/block_swap.py`` (``RandomAccessBlockReader`` / ``_DiskPrefetcher``),
simplified for a one-shot encoder forward:

* tensors stay on disk (safetensors) until a layer group is about to run;
* ``DiskGroupReader`` opens shards lazily and reads individual tensors on
  demand (safetensors ``safe_open`` is already lazy/random-access);
* ``GroupStreamer.load_group`` rebinds parameter ``.data`` on the GPU;
  ``release_group`` drops the storage so VRAM returns to the allocator;
* a background ``ThreadPoolExecutor`` prefetches the next group from disk
  while the current group computes (like ``_DiskPrefetcher``).
"""

from __future__ import annotations

import json
import re
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import torch

from ...utils.stream import _ShardStore as ShardStore

from .types import DiskGroupSpec

# ---------------------------------------------------------------------------
# Shard access
# ---------------------------------------------------------------------------


class DiskGroupReader:
    """Random-access reader over a sharded safetensors checkpoint.

    ``get_tensors`` returns a dict of tensor name -> CPU tensor, reading only
    the requested names (position-independent lazy reads, no full-file load).
    """

    def __init__(self, model_dir: str | Path, weight_path: str | Path | None = None):
        self.model_dir = Path(model_dir)
        self._weight_map: dict[str, str] = {}
        self._store: Optional[ShardStore] = None
        self._single_shard: Optional[str] = None

        if weight_path is not None:
            # Config/tokenizer live in model_dir; the weights are a separate
            # ComfyUI-model-file (e.g. models/text_encoders/*.safetensors).
            self._single_shard = Path(weight_path).name
            self._store = ShardStore([Path(weight_path)])
            return

        index_path = self.model_dir / "model.safetensors.index.json"
        if index_path.exists():
            data = json.loads(index_path.read_text(encoding="utf-8"))
            self._weight_map = data.get("weight_map", {})
            shard_files = sorted({self.model_dir / s for s in set(self._weight_map.values())})
            if not shard_files:
                raise FileNotFoundError(f"no shards referenced by {index_path}")
            self._store = ShardStore(shard_files)
        else:
            single = self.model_dir / "model.safetensors"
            if not single.exists():
                raise FileNotFoundError(
                    f"no model.safetensors(.index.json) found under {self.model_dir}")
            self._single_shard = single.name
            self._store = ShardStore([single])

    # -- queries ---------------------------------------------------------------

    def all_keys(self) -> list[str]:
        if self._weight_map:
            return list(self._weight_map)
        return self._store.keys(self._single_shard)  # type: ignore[union-attr]

    def has(self, name: str) -> bool:
        if self._weight_map:
            return name in self._weight_map
        return self._store.has(self._single_shard, name)  # type: ignore[arg-type]

    def get_tensors(self, names: list[str]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name in names:
            shard = self._weight_map.get(name) if self._weight_map else self._single_shard
            out[name] = self._store.get_tensor(shard, name)  # type: ignore[arg-type]
        return out

    def get_tensor_info(self, name: str) -> tuple[list, object]:
        """Header-only (shape, dtype); no data read."""
        shard = self._weight_map.get(name) if self._weight_map else self._single_shard
        return self._store.get_tensor_info(shard, name)  # type: ignore[arg-type]

    def detect_layer_prefix(self) -> str:
        """Find the prefix of ``layers.N.self_attn.q_proj.weight`` keys."""
        # Probe common prefixes without enumerating the full key list (avoids
        # materialising the whole safetensors header under low-RAM conditions).
        for prefix in ("", "model.", "model.language_model.model."):
            if self.has(f"{prefix}layers.0.self_attn.q_proj.weight"):
                return prefix
        raise RuntimeError(
            "cannot auto-detect layer prefix: no 'layers.N.self_attn.q_proj.weight' "
            "key found; pass StreamConfig.layer_prefix explicitly")

    def close(self):
        if self._store is not None:
            self._store.close()


# ---------------------------------------------------------------------------
# Group streaming
# ---------------------------------------------------------------------------

def _resolve_param(layer: torch.nn.Module, dotted: str):
    """Return ``(parent_module, leaf_name)`` for a dotted param name."""
    parts = dotted.split(".")
    if len(parts) == 1:
        return layer, parts[0]
    return layer.get_submodule(".".join(parts[:-1])), parts[-1]


def _set_param(module: torch.nn.Module, name: str, tensor: torch.Tensor) -> None:
    """Replace a parameter in-place via the module ``_parameters`` dict.

    Rebinding ``.data`` fails when the placeholder is a meta tensor (device
    type mismatch), so we swap in a fresh ``nn.Parameter``.  The module graph
    stays valid because ``named_parameters()`` / attribute access resolve
    through ``_parameters`` on every call.
    """
    module._parameters[name] = torch.nn.Parameter(tensor, requires_grad=False)


class _DictReader:
    """Reader shim over a prefetched ``{key: tensor}`` dict.

    Lets ``quant.load_layer_spec`` parse weights/scales from the prefetch
    buffer instead of hitting disk again during ``load_group``.
    """

    def __init__(self, tensors: dict[str, torch.Tensor]):
        self._tensors = tensors

    def has(self, name: str) -> bool:
        return name in self._tensors

    def get_tensors(self, names: list[str]) -> dict[str, torch.Tensor]:
        return {name: self._tensors[name] for name in names
                if name in self._tensors}

    def get_tensor_info(self, name: str):
        t = self._tensors[name]
        return list(t.shape), t.dtype


def _quant_metadata_keys(reader, full_key: str) -> list[str]:
    """Return quant metadata keys belonging to ``full_key``, if any."""
    if not full_key.endswith(".weight"):
        return []
    prefix = full_key[: -len("weight")]
    if not (reader.has(prefix + "comfy_quant")
            or reader.has(prefix + "weight_scale")):
        return []
    return [
        prefix + name
        for name in ("comfy_quant", "weight_scale", "weight_scale_2",
                     "input_scale", "pre_quant_scale")
        if reader.has(prefix + name)
    ]


class GroupStreamer:
    """Loads decoder-layer groups from disk onto the device, then destroys them.

    Parameter identity is preserved across load/destroy cycles: only
    ``param.data`` is rebound, so the ``nn.Module`` graph stays intact and the
    same parameter object is reused for every re-load.
    """

    def __init__(self, model, layer_prefix: str, num_layers: int,
                 reader: DiskGroupReader, device: torch.device,
                 dtype: torch.dtype, group_size: int = 2,
                 prefetch: bool = True, disk_workers: int = 2,
                 pin_memory: bool = False, full_precision_mm: bool = True):
        self.model = model
        self.layer_prefix = layer_prefix
        self.reader = reader
        self.device = device
        self.dtype = dtype
        self.full_precision_mm = full_precision_mm
        self.group_size = max(1, group_size)
        self.pin_memory = pin_memory
        self.prefetch_ahead = max(1, min(3, disk_workers))

        # The config may list more layers than the checkpoint actually holds
        # (ComfyUI's MiniMax-H3 encoder is a 50-layer truncation of the 64-layer
        # Qwen3-VL-32B).  Detect the real count from disk and clamp.
        probe = f"{layer_prefix}layers.0.self_attn.q_proj.weight"
        if reader.has(probe):
            real = 0
            while reader.has(f"{layer_prefix}layers.{real}.self_attn.q_proj.weight"):
                real += 1
                if real >= num_layers:
                    break
            num_layers = min(num_layers, real) if real else num_layers
        self.num_layers = num_layers

        # Group plan: (module, leaf-name) pairs so parameters can be *replaced*
        # on load/destroy (meta placeholders cannot be set_data'd to CPU/GPU).
        self.groups: list[DiskGroupSpec] = []
        self._param_refs: list[list[tuple[torch.nn.Module, str]]] = []
        for start in range(0, num_layers, self.group_size):
            end = min(start + self.group_size, num_layers)
            keys: list[str] = []
            refs: list[tuple[torch.nn.Module, str]] = []
            for idx in range(start, end):
                layer = model.layers[idx]
                for pname, _ in layer.named_parameters():
                    keys.append(f"{layer_prefix}layers.{idx}.{pname}")
                    mod, leaf = _resolve_param(layer, pname)
                    if isinstance(mod, torch.nn.Linear) and not hasattr(mod, "_orig_shape"):
                        mod._orig_shape = tuple(mod.weight.shape)
                    refs.append((mod, leaf))
            self.groups.append(DiskGroupSpec(
                group_idx=len(self.groups), layer_start=start, layer_end=end,
                keys=tuple(keys)))
            self._param_refs.append(refs)

        self._prefetch_buf: dict[int, dict[str, torch.Tensor]] = {}
        self._pending: dict[int, Future] = {}
        self._group_read_keys: list[tuple[str, ...]] = []
        for group in self.groups:
            read_keys: list[str] = []
            for key in group.keys:
                read_keys.append(key)
                read_keys.extend(_quant_metadata_keys(self.reader, key))
            self._group_read_keys.append(tuple(read_keys))
        self._executor = (
            ThreadPoolExecutor(max_workers=max(1, disk_workers),
                               thread_name_prefix="h3-encoder-prefetch")
            if prefetch else None
        )

    # -- public API -------------------------------------------------------------

    def load_group(self, group_idx: int) -> None:
        """Make ``groups[group_idx]`` resident on the compute device."""
        refs = self._param_refs[group_idx]
        if group_idx in self._prefetch_buf:
            tensors = self._prefetch_buf.pop(group_idx)
        elif group_idx in self._pending:
            fut = self._pending.pop(group_idx)
            tensors = fut.result()
        else:
            tensors = self.reader.get_tensors(
                list(self._group_read_keys[group_idx]))
        source = _DictReader(tensors)
        keys = list(self.groups[group_idx].keys)
        for (mod, pname), key in zip(refs, keys):
            self._bind_one(
                mod, pname, key, tensors.get(key), source=source)

    def _bind_one(self, mod, pname: str, key: str, tensor,
                  source=None):
        """Bind one weight: quantized (comfy_quant) via comfy-kitchen, else plain."""
        from ..quant import (is_quant_layer, load_layer_spec,
                             make_quantized_tensor, bind_param)
        if isinstance(mod, torch.nn.Linear) and is_quant_layer(self.reader, key):
            spec = load_layer_spec(
                source or self.reader, key, read_weight=True)
            # move qdata + scales to the compute device (disk reads are CPU)
            dev = self.device
            if isinstance(spec.qdata, torch.Tensor) and spec.qdata.ndim > 0:
                spec.qdata = spec.qdata.to(dev)
            spec.scales = {n: (t.to(dev) if isinstance(t, torch.Tensor) else t)
                           for n, t in spec.scales.items()}
            spec.extra_params = {n: (t.to(dev) if isinstance(t, torch.Tensor) else t)
                                 for n, t in spec.extra_params.items()}
            shape = tuple(getattr(mod, "_orig_shape", mod.weight.shape))
            orig_dtype = getattr(mod, "_orig_dtype", None) or self.dtype
            qt = make_quantized_tensor(spec, shape, orig_dtype)
            mod._h3_full_precision_mm = self.full_precision_mm
            bind_param(mod, pname, qt, spec.extra_params)
            return
        if tensor is None:
            tensor = self.reader.get_tensors([key])[key]
        _set_param(mod, pname,
                   tensor.to(device=self.device, dtype=self.dtype,
                             non_blocking=bool(self.pin_memory and self.device.type == "cuda")))

    def release_group(self, group_idx: int) -> None:
        """Drop the group's parameter storage (VRAM returns to the allocator)."""
        for mod, pname in self._param_refs[group_idx]:
            _set_param(mod, pname, torch.empty(0))

    def prefetch_next(self, group_idx: int) -> None:
        """Kick background reads for the next few groups (no-op if past end)."""
        if self._executor is None:
            return
        for idx in range(group_idx, min(
                len(self.groups), group_idx + self.prefetch_ahead)):
            if idx in self._prefetch_buf or idx in self._pending:
                continue
            while len(self._pending) >= self.prefetch_ahead:
                oldest = min(self._pending)
                fut = self._pending.pop(oldest)
                self._prefetch_buf[oldest] = fut.result()
            self._pending[idx] = self._executor.submit(
                self._read_group, idx)

    def load_all(self) -> None:
        """Full mode: make every group resident (requires VRAM >= checkpoint)."""
        for i in range(len(self.groups)):
            self.load_group(i)

    def release_all(self) -> None:
        for i in range(len(self.groups)):
            self.release_group(i)
        self._prefetch_buf.clear()

    def shutdown(self) -> None:
        # Drain any in-flight prefetch BEFORE the caller closes shard handles:
        # a worker touching a closed safetensors handle would raise off-thread.
        for idx, fut in list(self._pending.items()):
            try:
                self._prefetch_buf[idx] = fut.result(timeout=60)
            except Exception:
                pass
        self._pending.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        self._prefetch_buf.clear()

    # -- internals ---------------------------------------------------------------

    def _read_group(self, group_idx: int) -> dict[str, torch.Tensor]:
        tensors = self.reader.get_tensors(
            list(self._group_read_keys[group_idx]))
        if not (self.pin_memory and self.device.type == "cuda"):
            return tensors
        # Stage disk reads into pinned CPU buffers so the H2D copy in
        # ``load_group`` can be a real asynchronous ``cudaMemcpyAsync``
        # (non_blocking=True is silently ignored on pageable memory).
        pinned: dict[str, torch.Tensor] = {}
        for name, t in tensors.items():
            buf = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
            buf.copy_(t)
            pinned[name] = buf
        return pinned
