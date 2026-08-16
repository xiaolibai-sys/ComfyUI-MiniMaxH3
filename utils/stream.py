"""Random-access safetensors readers for streaming block loads.

Thread-safe positioned reads (``os.pread`` on POSIX, pywin32 OVERLAPPED on
Windows, per-fd locked ``lseek``+``read`` fallback) so multiple disk
prefetch workers can read different blocks concurrently without corrupting
each other's file position.

Supports both the official single-file Comfy-Org checkpoints and the sharded
HF layout (``model.safetensors.index.json`` + ``model-0000N-of-00013.safetensors``).
"""

from __future__ import annotations

import json
import os
import struct
import threading
import warnings
from pathlib import Path
from typing import Optional

import torch

_DTYPE_MAP = {
    "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
    "F64": torch.float64, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
    "U16": torch.uint16, "U32": torch.uint32, "U64": torch.uint64,
}

_PREAD_LOCKS: dict = {}


def _pread(fd: int, n: int, offset: int) -> bytes:
    if hasattr(os, "pread"):
        return os.pread(fd, n, offset)
    try:
        import msvcrt
        import pywintypes  # type: ignore
        import win32file  # type: ignore
        h = msvcrt.get_osfhandle(fd)
        ov = pywintypes.OVERLAPPED()
        ov.Offset = offset & 0xFFFFFFFF
        ov.OffsetHigh = (offset >> 32) & 0xFFFFFFFF
        try:
            _, data = win32file.ReadFile(h, n, ov)
        except pywintypes.error as e:
            if e.winerror != win32file.ERROR_IO_PENDING:
                raise
            _, data = win32file.GetOverlappedResult(h, ov, True)
        return data
    except ImportError:
        lock = _PREAD_LOCKS.get(fd)
        if lock is None:
            lock = _PREAD_LOCKS.setdefault(fd, threading.Lock())
        with lock:
            os.lseek(fd, offset, os.SEEK_SET)
            return os.read(fd, n)


def _pread_path(path: str, n: int, offset: int):
    """Read exactly ``n`` bytes at ``offset`` using a private file handle."""
    fd = os.open(path, os.O_RDONLY | os.O_BINARY)
    try:
        os.lseek(fd, offset, os.SEEK_SET)
        chunks = []
        remaining = n
        while remaining:
            chunk = os.read(fd, remaining)
            if not chunk:
                raise OSError(
                    f"short read from {path}: expected {n} bytes, "
                    f"got {n - remaining}"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _torch_dtype(dt: str) -> torch.dtype:
    return _DTYPE_MAP[dt]


def _tensor_from_buffer(raw, shape, dtype: torch.dtype) -> torch.Tensor:
    """Copy a raw buffer into an independently owned tensor.

    ``torch.frombuffer`` can leave a Python buffer exported while another
    disk/allocator thread is running on Windows, which raises
    ``SystemError: deallocated bytearray object has exported buffers``.
    Copying through numpy first detaches the tensor from the read buffer.
    """
    import numpy as np
    buf = np.frombuffer(raw, dtype=np.uint8).copy()
    return torch.from_numpy(buf).view(dtype).reshape(tuple(shape))


def _tensor_from_file(path: str, nbytes: int, offset: int,
                      shape: list, dtype: torch.dtype) -> torch.Tensor:
    """Read a safetensors tensor through a short-lived numpy memmap.

    ``os.read`` allocates a Python bytes object for every tensor and can raise
    ``MemoryError`` under BlockSwap/VAE memory pressure. A read-only memmap
    avoids that transient bytes allocation; ``clone()`` detaches the returned
    tensor before the mapping is closed.
    """
    import numpy as np

    mm = np.memmap(path, dtype=np.uint8, mode="r", offset=offset, shape=(nbytes,))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return (
                torch.from_numpy(mm)
                .clone()
                .view(dtype)
                .reshape(tuple(shape))
            )
    finally:
        mm._mmap.close()


class _SingleFileReader:
    """Positioned reads over one safetensors file."""

    def __init__(self, path: str | Path):
        self._path = str(path)
        self._fd = os.open(self._path, os.O_RDONLY | os.O_BINARY)
        try:
            from safetensors.torch import _TYPES  # type: ignore
            self._dtype_map = _TYPES
        except Exception:
            self._dtype_map = _DTYPE_MAP
        header_len = struct.unpack("<Q", os.read(self._fd, 8))[0]
        self._header = json.loads(os.read(self._fd, header_len))
        self._data_offset = 8 + header_len
        self._keys: list[str] = [k for k, v in self._header.items()
                                 if isinstance(v, dict) and "data_offsets" in v]

    def keys(self) -> list[str]:
        return list(self._keys)

    def has(self, name: str) -> bool:
        return name in self._header and isinstance(self._header[name], dict)

    def get_tensor(self, name: str) -> torch.Tensor:
        shape, dt = self.get_tensor_info(name)
        info = self._header[name]
        begin, end = info["data_offsets"]
        nbytes = end - begin
        return _tensor_from_file(
            self._path,
            nbytes,
            self._data_offset + begin,
            shape,
            dt,
        )

    def get_tensor_info(self, name: str) -> tuple[list, torch.dtype]:
        """Return (shape, dtype) from the header without reading data."""
        info = self._header.get(name)
        if info is None or not isinstance(info, dict):
            raise KeyError(f"{name} not in {self._path}")
        return list(info["shape"]), self._torch_dtype(info["dtype"])

    def get_tensor_offsets(self, name: str) -> tuple[int, int]:
        """Return (file_offset, nbytes) without reading tensor data."""
        info = self._header.get(name)
        if info is None or not isinstance(info, dict):
            raise KeyError(f"{name} not in {self._path}")
        begin, end = info["data_offsets"]
        return self._data_offset + begin, end - begin

    def _torch_dtype(self, dt: str) -> torch.dtype:
        return self._dtype_map[dt]

    def close(self):
        try:
            os.close(self._fd)
        except OSError:
            pass


class _ShardStore:
    """Lazy shard handles with per-read OVERLAPPED file handles."""

    def __init__(self, shard_paths: list[Path]):
        self._readers: dict[str, _SingleFileReader] = {}
        self._paths = {p.name: p for p in shard_paths}

    def _reader(self, shard: str) -> _SingleFileReader:
        r = self._readers.get(shard)
        if r is None:
            r = _SingleFileReader(self._paths[shard])
            self._readers[shard] = r
        return r

    def get_tensor(self, shard: str, name: str) -> torch.Tensor:
        return self._reader(shard).get_tensor(name)

    def get_tensor_info(self, shard: str, name: str):
        return self._reader(shard).get_tensor_info(name)

    def keys(self, shard: str) -> list[str]:
        return self._reader(shard).keys()

    def has(self, shard: str, name: str) -> bool:
        return self._reader(shard).has(name)

    def close(self):
        for r in self._readers.values():
            r.close()
        self._readers.clear()


class BlockReader:
    """Unified random-access reader over a single file or a sharded checkpoint.

    ``get_tensors(names)`` returns ``{name: CPU tensor}`` reading only the
    requested tensors — the basis of the streaming block loader.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._single: Optional[_SingleFileReader] = None
        self._shards: Optional[_ShardStore] = None
        self._weight_map: dict[str, str] = {}

        if self.path.is_dir():
            index_path = self.path / "model.safetensors.index.json"
            if not index_path.exists():
                raise FileNotFoundError(f"no model.safetensors(.index.json) under {self.path}")
            data = json.loads(index_path.read_text(encoding="utf-8"))
            self._weight_map = data.get("weight_map", {})
            shard_files = sorted({self.path / s for s in set(self._weight_map.values())})
            self._shards = _ShardStore(shard_files)
        else:
            self._single = _SingleFileReader(self.path)

    def all_keys(self) -> list[str]:
        if self._single is not None:
            return self._single.keys()
        return list(self._weight_map)

    def has(self, name: str) -> bool:
        if self._single is not None:
            return self._single.has(name)
        return name in self._weight_map

    def get_tensor_info(self, name: str) -> tuple[list, torch.dtype]:
        """Header-only (shape, dtype); no data read."""
        if self._single is not None:
            return self._single.get_tensor_info(name)
        shard = self._weight_map[name]
        return self._shards.get_tensor_info(shard, name)  # type: ignore[union-attr]

    def get_tensor_offsets(self, name: str) -> tuple[int, int]:
        """Header-only (file_offset, nbytes); single-file mmap support."""
        if self._single is not None:
            return self._single.get_tensor_offsets(name)
        raise NotImplementedError(
            "mmap tensor offsets are not implemented for sharded checkpoints"
        )

    def get_tensors(self, names: list[str]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name in names:
            if self._single is not None:
                out[name] = self._single.get_tensor(name)
            else:
                shard = self._weight_map[name]
                out[name] = self._shards.get_tensor(shard, name)  # type: ignore[union-attr]
        return out

    def get_tensor_group(self, names: list[str]) -> dict[str, torch.Tensor]:
        """Read one logical tensor group, failing if any member is missing."""
        missing = [name for name in names if not self.has(name)]
        if missing:
            raise KeyError(
                f"BlockReader tensor group missing members: {missing}"
            )
        return self.get_tensors(names)

    def close(self):
        if self._single is not None:
            self._single.close()
        if self._shards is not None:
            self._shards.close()
