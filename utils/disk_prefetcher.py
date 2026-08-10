"""Background safetensors -> home-slot loader."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor

from ..models import quant
from .stream import BlockReader
from .swap_types import SwapBlock, _entry_of, free_module_storage
from .types import SlotEntry


class _DiskPrefetcher:
    """Reads blocks from disk into fixed home slots.

    The worker never touches swap slots, so a slot can never be recycled
    under an in-flight read.
    """

    def __init__(self, reader: BlockReader, blocks: list[SwapBlock],
                 max_workers: int = 2):
        self._reader = reader
        self._blocks = blocks
        self._executor = ThreadPoolExecutor(max_workers=max(1, max_workers),
                                            thread_name_prefix="h3disk")
        self._pending: dict[int, Future] = {}
        self._home_pending: dict[int, tuple[int, Future, threading.Event]] = {}
        self._loaded: set[int] = set()
        self._lock = threading.Lock()
        self._shut_down = False
        self.disk_reads = 0

    def ensure_ram(self, block_idx: int) -> None:
        with self._lock:
            if block_idx in self._loaded:
                return
            fut = self._pending.pop(block_idx, None)
        if fut is not None:
            fut.result()
        else:
            self._load_immediate(block_idx)
        with self._lock:
            self._loaded.add(block_idx)

    def start_ram_load(self, block_idx: int) -> None:
        return

    def start_home_fill(self, block_idx: int, hslot: int, slot: dict,
                        block: SwapBlock) -> bool:
        with self._lock:
            if hslot in self._home_pending:
                return False
            ready = threading.Event()
            fut = self._executor.submit(
                self._home_fill_worker, block_idx, hslot, slot, block, ready)
            self._home_pending[hslot] = (block_idx, fut, ready)
            return True

    def wait_home_fill(self, hslot: int) -> None:
        with self._lock:
            item = self._home_pending.get(hslot)
        if item is None:
            return
        try:
            item[1].result()
        finally:
            with self._lock:
                if self._home_pending.get(hslot) is item:
                    del self._home_pending[hslot]

    def read_into(self, block_idx: int, slot: dict, block: SwapBlock) -> None:
        tensors = self._reader.get_tensor_group(block.keys)
        if len(tensors) != len(block.keys):
            raise RuntimeError(
                f"blockswap: tensor group for {block.name} returned "
                f"{len(tensors)}/{len(block.keys)} tensors"
            )
        with self._lock:
            self.disk_reads += 1
        for idx, (full, tpl) in enumerate(zip(block.keys, block.templates)):
            n = block.names[idx]
            t = tensors.get(full)
            if t is None:
                continue
            if n in block.overrides:
                t = block.overrides[n]
            e = slot[n]
            if e is None:
                continue
            if tpl.is_qt:
                entry = SlotEntry(
                    data=t.to(tpl.data.dtype), scale=tpl.scale.clone(),
                    layout_cls=tpl.layout_cls, orig_dtype=tpl.orig_dtype,
                    orig_shape=tpl.orig_shape,
                    extra={n: v.clone() for n, v in tpl.extra.items()},
                    meta=dict(tpl.meta))
                e.copy_from(entry)
            else:
                e.copy_from(_entry_of(t.to(tpl.data.dtype)))

    def _home_fill_worker(self, block_idx: int, hslot: int, slot: dict,
                          block: SwapBlock, ready: threading.Event) -> None:
        try:
            self.read_into(block_idx, slot, block)
        finally:
            ready.set()

    def _release_params(self, block_idx: int) -> None:
        try:
            block = self._blocks[block_idx]
        except IndexError:
            return
        for mod in block.module.modules():
            free_module_storage(mod)
        with self._lock:
            self._loaded.discard(block_idx)

    def release(self, block_idx: int) -> None:
        self._release_params(block_idx)
        with self._lock:
            self._loaded.discard(block_idx)

    def join(self, timeout: float = 30.0) -> None:
        with self._lock:
            futs = list(self._pending.values())
            home_futs = [item[1] for item in self._home_pending.values()]
        for f in home_futs:
            try:
                f.result(timeout=timeout)
            except Exception:
                pass
        for f in futs:
            try:
                f.result(timeout=timeout)
            except Exception:
                pass

    def cancel_all(self) -> None:
        with self._lock:
            for fut in self._pending.values():
                fut.cancel()
            self._pending.clear()

    def shutdown(self) -> None:
        try:
            self.cancel_all()
            try:
                self._executor.shutdown(wait=True, timeout=30)
            except Exception:
                self._executor.shutdown(wait=False, cancel_futures=True)
            self.join()
        finally:
            self._shut_down = True
            with self._lock:
                self._home_pending.clear()
                self._pending.clear()

    def _on_load_done(self, fut: Future) -> None:
        if fut.exception() is not None:
            return
        with self._lock:
            for bi in list(self._pending):
                if self._pending.get(bi) is fut:
                    self._loaded.add(bi)
                    break

    def _load_immediate(self, block_idx: int) -> None:
        block = self._blocks[block_idx]
        tensors = self._reader.get_tensors(block.keys)
        for full, tpl, (mod, leaf, kind) in zip(
                block.keys, block.templates, block.refs):
            t = tensors.get(full)
            if t is None:
                continue
            if kind != "param":
                mod._buffers[leaf] = t.to(tpl.data.dtype)
            elif tpl.is_qt:
                entry = SlotEntry(
                    data=t.to(tpl.data.dtype), scale=tpl.scale.clone(),
                    layout_cls=tpl.layout_cls, orig_dtype=tpl.orig_dtype,
                    orig_shape=tpl.orig_shape,
                    extra={n: v.clone() for n, v in tpl.extra.items()},
                    meta=dict(tpl.meta))
                quant.bind_param(mod, leaf, entry)
            else:
                quant.bind_param(mod, leaf, t.to(tpl.data.dtype))
        with self._lock:
            self.disk_reads += 1
