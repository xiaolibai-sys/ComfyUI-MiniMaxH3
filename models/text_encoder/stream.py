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
import os
import re
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import torch


from .types import DiskGroupSpec

# ---------------------------------------------------------------------------
# Shard access
# ---------------------------------------------------------------------------

class ShardStore:
    """Lazy non-mmap safetensors shard handles with per-shard read locks."""

    def __init__(self, shard_paths: list[Path]):
        self._paths = [Path(p) for p in shard_paths]
        self._readers: dict[str, _SingleFileReader] = {}
        self._locks: dict[str, threading.Lock] = {}

    def _reader(self, shard: str) -> _SingleFileReader:
        r = self._readers.get(shard)
        if r is None:
            r = _SingleFileReader(self._paths[self._shard_index(shard)])
            self._readers[shard] = r
            self._locks[shard] = threading.Lock()
        return r

    def _shard_index(self, shard: str) -> int:
        for i, p in enumerate(self._paths):
            if p.name == shard:
                return i
        raise KeyError(f"shard not registered: {shard}")

    def get_tensor(self, shard: str, name: str) -> torch.Tensor:
        r = self._reader(shard)
        with self._locks[shard]:
            return r.get_tensor(name)

    def get_tensor_info(self, shard: str, name: str) -> tuple[list, object]:
        r = self._reader(shard)
        with self._locks[shard]:
            return r.get_tensor_info(name)

    def keys(self, shard: str):
        r = self._reader(shard)
        with self._locks[shard]:
            return [k for k, v in r._header.items()
                    if isinstance(v, dict) and "data_offsets" in v]

    def has(self, shard: str, name: str) -> bool:
        r = self._reader(shard)
        with self._locks[shard]:
            return r.has(name)

    def close(self):
        for r in self._readers.values():
            r.close()
        self._readers.clear()
        self._locks.clear()


class _SingleFileReader:
    """Positioned reads over one safetensors file WITHOUT mmap (os.pread).

    ``safe_open`` memory-maps the whole 14 GB file, which exhausts the
    virtual-address space on machines with no pagefile (Windows os error
    1455).  This reader only touches the header and the requested tensors.
    """

    _DTYPE_MAP = None

    def __init__(self, path):
        self._path = str(path)
        self._fd = os.open(self._path, os.O_RDONLY | os.O_BINARY)
        if self._DTYPE_MAP is None:
            try:
                from safetensors.torch import _TYPES  # type: ignore
                type(self)._DTYPE_MAP = _TYPES
            except Exception:
                type(self)._DTYPE_MAP = {
                    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
                    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
                    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
                    "BOOL": torch.bool, "F8_E4M3": torch.float8_e4m3fn,
                    "F8_E5M2": torch.float8_e5m2, "F8_E8M0": torch.float8_e8m0fnu,
                }
        header_len = struct.unpack("<Q", os.read(self._fd, 8))[0]
        self._header = json.loads(os.read(self._fd, header_len))
        self._data_offset = 8 + header_len

    def _pread(self, nbytes: int, offset: int) -> bytes:
        from ...utils.stream import _pread as _posix_pread
        return _posix_pread(self._fd, nbytes, offset)

    def has(self, name: str) -> bool:
        info = self._header.get(name)
        return isinstance(info, dict) and "data_offsets" in info

    def get_tensor(self, name: str) -> torch.Tensor:
        shape, dt = self.get_tensor_info(name)
        info = self._header[name]
        begin, end = info["data_offsets"]
        raw = self._pread(end - begin, self._data_offset + begin)
        return torch.frombuffer(raw, dtype=dt).reshape(shape)

    def get_tensor_info(self, name: str):
        info = self._header.get(name)
        if info is None or not isinstance(info, dict):
            raise KeyError(f"{name} not in {self._path}")
        return list(info["shape"]), self._DTYPE_MAP[info["dtype"]]

    def close(self):
        try:
            os.close(self._fd)
        except OSError:
            pass


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
        self._pending = None
        self._executor = (
            ThreadPoolExecutor(max_workers=max(1, disk_workers),
                               thread_name_prefix="h3-encoder-prefetch")
            if prefetch else None
        )

    # -- public API -------------------------------------------------------------

    def load_group(self, group_idx: int) -> None:
        """Make ``groups[group_idx]`` resident on the compute device."""
        refs = self._param_refs[group_idx]
        keys = list(self.groups[group_idx].keys)
        non_blocking = bool(self.pin_memory and self.device.type == "cuda")
        if group_idx in self._prefetch_buf:
            src = self._prefetch_buf.pop(group_idx)
            for (mod, pname), key in zip(refs, keys):
                self._bind_one(mod, pname, key, src.get(key))
            return
        tensors = self.reader.get_tensors(keys)
        for (mod, pname), key in zip(refs, keys):
            self._bind_one(mod, pname, key, tensors.get(key))

    def _bind_one(self, mod, pname: str, key: str, tensor):
        """Bind one weight: quantized (comfy_quant) via comfy-kitchen, else plain."""
        from ..quant import (is_quant_layer, load_layer_spec,
                             make_quantized_tensor, bind_param)
        if isinstance(mod, torch.nn.Linear) and is_quant_layer(self.reader, key):
            # quant metadata (scales + comfy_quant) is not part of the prefetch
            # keys, so quantized layers read directly from disk here.
            spec = load_layer_spec(self.reader, key, read_weight=True)
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
        """Kick a background disk read of ``groups[group_idx]`` (no-op if past end)."""
        if self._executor is None:
            return
        if group_idx >= len(self.groups):
            return
        if self._pending is not None:
            # Only one in-flight prefetch at a time (prefetch_count == 1).
            self._pending.result()
        self._pending = self._executor.submit(self._read_group, group_idx)

    def load_all(self) -> None:
        """Full mode: make every group resident (requires VRAM >= checkpoint)."""
        for i in range(len(self.groups)):
            self.load_group(i)

    def release_all(self) -> None:
        for i in range(len(self.groups)):
            self.release_group(i)

    def shutdown(self) -> None:
        # Drain any in-flight prefetch BEFORE the caller closes shard handles:
        # a worker touching a closed safetensors handle would raise off-thread.
        if self._pending is not None:
            try:
                self._pending.result(timeout=60)
            except Exception:
                pass
            self._pending = None
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        self._prefetch_buf.clear()

    # -- internals ---------------------------------------------------------------

    def _read_group(self, group_idx: int) -> dict[str, torch.Tensor]:
        tensors = self.reader.get_tensors(list(self.groups[group_idx].keys))
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
