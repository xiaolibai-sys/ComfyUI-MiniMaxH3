"""Ring-buffer BlockSwap for the MiniMax H3 DiT - BerniniRWrapper-style.

Mirrors ``ComfyUI-BerniniRWrapper/utils/block_swap.py`` (lazy disk mode):

* ``_BlockWindow`` - pure residency tracking (which blocks are on GPU).
* ``_TransferEngine`` - fixed slot pools: a registered/global CPU home pool of
  ``total - window`` block buffers and a GPU ring of
  ``window + prefetch_count``.  Blocks move with ``copy_`` on a dedicated
  transfer stream; host RAM stays flat at ``(N-W)`` blocks with no per-step
  allocation churn.  Block params are rebound to whichever slot currently
  owns the block, so the params themselves point at the live weights.
* ``_DiskPrefetcher`` - background safetensors -> block-param loader.  Disk
  is only touched on a block's first use (and after rare forced evictions);
  the steady-state ring cycles purely RAM <-> GPU.
* ``BlockSwapManager`` - sliding window: load-first so the incoming block
  frees its home slot before the outgoing one needs it, transfer-stream
  prefetch of the next window, LoRA folded on the GPU ring right after the
  block lands (``block.lora`` consumed to ``None`` marks "folded").

Public API (``begin/prepare/end/shutdown/stats`` + ``apply_lora/clear_lora``)
is unchanged, so ``models/model.py`` and ``utils/lifecycle.py`` keep working.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import torch

from ..models.lora import fold_lora_into_slot
from .block_window import _BlockWindow, _VRAMBudget
from .disk_prefetcher import _DiskPrefetcher
from .interrupt import check_interrupt
from .stream import BlockReader
from .types import BlockSwapStats, SwapAllocation, SwapBlock
from .transfer_scheduler import TransferScheduler
from .types import SlotEntry

logger = logging.getLogger("h3.blockswap")


def _align_offset(offset: int, alignment: int) -> int:
    return (offset + alignment - 1) // alignment * alignment


_POOL_COMPONENT_ALIGNMENT = 128
# cuBLASLt NVFP4 GEMM requires aligned scale/qdata pointers; keeping every
# component 128-byte aligned also makes the CPU/GPU pool layouts identical.


def _home_slot_layout(block: SwapBlock):
    """Return (plan, total_bytes) for one block-shaped home slot."""
    plan: list[tuple[str, list[tuple[str, torch.dtype, tuple]]]] = []
    total = 0
    for name, tpl in zip(block.names, block.templates):
        comps: list[tuple[str, torch.dtype, tuple]] = [
            ("data", tpl.data.dtype, tuple(tpl.data.shape))]
        if tpl.scale is not None:
            comps.append(("scale", tpl.scale.dtype, tuple(tpl.scale.shape)))
        for extra_name, t in tpl.extra.items():
            comps.append((extra_name, t.dtype, tuple(t.shape)))
        plan.append((name, comps))
        for _kind, dtype, shape in comps:
            nbytes = int(torch.tensor(shape, dtype=torch.int64).prod()) * dtype.itemsize
            total = _align_offset(total, _POOL_COMPONENT_ALIGNMENT) + nbytes
    return plan, _align_offset(total, _POOL_COMPONENT_ALIGNMENT)


def _build_contiguous_home_slot(block: SwapBlock):
    """Build one pageable home slot backed by a single CPU allocation."""
    plan, total = _home_slot_layout(block)
    raw = torch.empty(max(1, total), dtype=torch.uint8)
    return _home_slot_views(block, raw, 0), raw


def _build_global_pool(block: SwapBlock, n_slots: int, device="cpu"):
    """Build ``n_slots`` home slots inside one contiguous pageable buffer."""
    plan, slot_total = _home_slot_layout(block)
    raw = torch.empty(
        max(1, n_slots * slot_total), dtype=torch.uint8, device=device)
    slots = []
    for i in range(n_slots):
        slots.append(_home_slot_views(block, raw, i * slot_total))
    return slots, raw, slot_total


def _home_slot_views(block: SwapBlock, raw: torch.Tensor, base: int):
    offset = 0
    slot = {}
    plan, _total = _home_slot_layout(block)
    for name, comps in plan:
        kwargs = {}
        extras = {}
        for kind, dtype, shape in comps:
            nbytes = int(torch.tensor(shape, dtype=torch.int64).prod()) * dtype.itemsize
            offset = _align_offset(offset, _POOL_COMPONENT_ALIGNMENT)
            view = raw[base + offset:base + offset + nbytes].view(dtype).reshape(shape)
            offset += nbytes
            if kind == "data":
                kwargs["data"] = view
            elif kind == "scale":
                kwargs["scale"] = view
            else:
                extras[kind] = view
        tpl = block.templates[block.names.index(name)]
        kwargs["layout_cls"] = tpl.layout_cls
        kwargs["orig_dtype"] = tpl.orig_dtype
        kwargs["orig_shape"] = tpl.orig_shape
        kwargs["extra"] = extras
        kwargs["meta"] = dict(tpl.meta)
        slot[name] = SlotEntry(**kwargs)
    return slot


def _register_cpu_entries(entries: dict,
                          storage: Optional[torch.Tensor] = None
                          ) -> list[tuple[int, int]]:
    """Persistently register pageable home tensors as CUDA-pinned."""
    cudart = torch.cuda.cudart()
    tokens: list[tuple[int, int]] = []
    try:
        if storage is not None:
            ptr = storage.data_ptr()
            nbytes = storage.numel() * storage.element_size()
            err = cudart.cudaHostRegister(ptr, nbytes, 0)
            if err != 0:
                raise RuntimeError(
                    f"cudaHostRegister failed: {_cuda_error_str(cudart, err)}")
            return [(ptr, nbytes)]
        for entry in entries.values():
            tensors = [entry.data]
            if entry.scale is not None:
                tensors.append(entry.scale)
            tensors.extend(entry.extra.values())
            for t in tensors:
                if t.numel() == 0:
                    continue
                ptr = t.data_ptr()
                nbytes = t.numel() * t.element_size()
                err = cudart.cudaHostRegister(ptr, nbytes, 0)
                if err != 0:
                    raise RuntimeError(
                        f"cudaHostRegister failed: "
                        f"{_cuda_error_str(cudart, err)}")
                tokens.append((ptr, nbytes))
    except Exception:
        for ptr, _nbytes in tokens:
            if cudart.cudaHostUnregister(ptr) != 0:
                _discard_cuda_async_error()
        _discard_cuda_async_error()
        raise
    return tokens


def _cuda_error_str(cudart, err) -> str:
    try:
        return cudart.cudaGetErrorString(cudart.cudaError(err))
    except Exception:
        return str(err)


def _discard_cuda_async_error() -> None:
    """Drop a CUDA async error queued by a failed host-register call.

    ComfyUI's ``model_management.discard_cuda_async_error`` does the same
    thing: ``cudaHostRegister`` can return nonzero and still leave a pending
    CUDA error.  If we do not flush it here, the next innocuous CUDA call can
    surface as ``CUDA error: out of memory`` long after the fallback.
    """
    if not torch.cuda.is_available():
        return
    try:
        a = torch.empty(1, device="cuda")
        b = torch.empty(1, device="cuda")
        _ = a + b
        torch.cuda.synchronize()
    except Exception:
        pass


def _unregister_cpu_entries(tokens: list[tuple[int, int]]) -> None:
    cudart = torch.cuda.cudart()
    for ptr, _nbytes in tokens:
        if cudart.cudaHostUnregister(ptr) != 0:
            _discard_cuda_async_error()


# ---------------------------------------------------------------------------
# _TransferEngine - CUDA stream + fixed slot pools (churn-free swaps)
# ---------------------------------------------------------------------------

class _TransferEngine:
    """Dedicated transfer stream + home/GPU slot pools.

    Both pools are ``dict[name, SlotEntry]`` lists built lazily from each
    block's templates and reused across blocks afterwards; every fill path
    copies data + scale + extras from the source, so a recycled slot never
    carries stale quant metadata.  D2H offloads and H2D loads share one
    transfer stream, which is what makes GPU-ring reuse safe: a slot's new
    H2D is stream-ordered after the previous occupant's D2H.
    """

    def __init__(self, device, prefetch: bool, prefetch_count: int,
                 pin_memory: bool, n_home_slots: int, n_gpu_slots: int,
                 home: "_DiskPrefetcher"):
        self.device = torch.device(device)
        self.prefetch = bool(prefetch) and torch.cuda.is_available()
        self.prefetch_count = max(1, prefetch_count)
        self.pin_memory = bool(pin_memory) and torch.cuda.is_available()
        self._home = home

        self._stream: Optional[torch.cuda.Stream] = None
        self._d2h_stream: Optional[torch.cuda.Stream] = None
        if self.prefetch:
            try:
                self._stream = torch.cuda.Stream(device=self.device)
            except Exception:
                self._stream = None
            try:
                self._d2h_stream = torch.cuda.Stream(device=self.device)
            except Exception:
                self._d2h_stream = None
        self._scheduler: Optional[TransferScheduler] = None

        nh = max(0, n_home_slots)
        ng = max(0, n_gpu_slots)
        self._home_pool: list[dict | None] = [None] * nh
        self._home_built: list[bool] = [False] * nh
        self._home_storage: Optional[torch.Tensor] = None
        self._home_storage_tokens: Optional[list[tuple[int, int]]] = None
        self._home_slot_size: int = 0
        self._home_free: list[int] = list(range(nh))
        self._home_free_lock = threading.Lock()
        self._home_free_cond = threading.Condition(self._home_free_lock)
        self._gpu_pool: list[dict | None] = [None] * ng
        self._gpu_built: list[bool] = [False] * ng
        self._gpu_storage: Optional[torch.Tensor] = None
        self._gpu_slot_size: int = 0
        # hslot -> event set once the background pinned->home copy completed
        self._home_ready: dict[int, threading.Event] = {}
        self._home_registered: dict[int, list[tuple[int, int]]] = {}
        self._cleanup_exec = ThreadPoolExecutor(
            max_workers=max(2, self.prefetch_count),
            thread_name_prefix="h3clean")
        self._transfer_lock = threading.RLock()

        self._gpu_cursor = 0
        self._block_home: dict[int, int] = {}
        self._block_gpu: dict[int, int] = {}
        # block_idx -> event recorded after its async H2D was enqueued
        self._events: dict[int, torch.cuda.Event] = {}
        # gpu slot -> event recorded after its latest H2D was enqueued
        self._gpu_h2d_events: dict[int, torch.cuda.Event] = {}
        # gpu slot -> event recorded after its latest D2H was enqueued
        self._gpu_d2h_events: dict[int, torch.cuda.Event] = {}
        # hslot -> event recorded after an async D2H into that home slot
        self._d2h_events: dict[int, torch.cuda.Event] = {}
        self.d2h_stage_hits = 0
        self.d2h_sync_fallback = 0

    # -- slot bookkeeping ----------------------------------------------------

    def _ring_acquire(self) -> int:
        ng = len(self._gpu_pool)
        occupied = set(self._block_gpu.values())
        for _ in range(ng):
            idx = self._gpu_cursor % ng
            self._gpu_cursor = (self._gpu_cursor + 1) % ng
            if idx not in occupied:
                return idx
        raise RuntimeError("blockswap: no free GPU ring slot")

    def _ensure_entries(self, pool, built, idx, block: SwapBlock, device) -> dict:
        """Build the ``name -> SlotEntry`` buffers for a pool slot once."""
        slot = pool[idx]
        if slot is not None and built[idx]:
            return slot
        if pool is self._home_pool and self._home_storage is not None:
            return pool[idx]
        if pool is self._gpu_pool and self._gpu_storage is not None:
            return pool[idx]
        with torch.inference_mode(False):
            pool[idx] = {
                n: SlotEntry.empty_like_entry(
                    tpl, device=device, pin_memory=False)
                for n, tpl in zip(block.names, block.templates)
            }
        if pool is self._home_pool and self.pin_memory:
            if self._home_registered.get(idx) is None:
                self._home_registered[idx] = _register_cpu_entries(pool[idx])
        built[idx] = True
        return pool[idx]

    def _acquire_home(self) -> int:
        with self._home_free_cond:
            if not self._home_free:
                raise RuntimeError("blockswap: out of home slots")
            return self._home_free.pop()

    def _release_home(self, hslot: int) -> None:
        with self._home_free_cond:
            self._home_free.append(hslot)
            self._home_free_cond.notify()

    def home_free_count(self) -> int:
        with self._home_free_cond:
            return len(self._home_free)

    def _wait_home_free(self, timeout: float = 30.0) -> int:
        with self._home_free_cond:
            if self._home_free:
                return self._home_free.pop()
            raise RuntimeError("blockswap: no free home slot")

    def preallocate_home_pool(self, block: SwapBlock) -> None:
        """Build and register the whole home pool as one contiguous buffer."""
        if self._home_storage is not None or not self.pin_memory:
            return
        if not self._home_pool:
            return
        _plan, slot_total = _home_slot_layout(block)
        with torch.inference_mode(False):
            slots, raw, slot_total = _build_global_pool(
                block, len(self._home_pool))
        self._home_storage = raw
        self._home_slot_size = slot_total
        self._home_pool = slots
        self._home_built = [True] * len(self._home_pool)
        try:
            tokens = _register_cpu_entries({}, storage=raw)
        except Exception as e:
            logger.warning(
                "blockswap: cudaHostRegister failed, using pageable home "
                "fallback: %s", e)
            self.pin_memory = False
            self._home_registered = {}
            self._home_storage_tokens = None
            self._home_storage = None
            self._home_slot_size = 0
            nh = len(self._home_pool)
            self._home_pool = [None] * nh
            self._home_built = [False] * nh
            with self._home_free_cond:
                self._home_free = list(range(nh))
            self._scheduler = None
            return
        self._home_storage_tokens = tokens
        self._home_registered = {
            i: tokens for i in range(len(self._home_pool))
        }

    def preallocate_gpu_pool(self, block: SwapBlock) -> None:
        """Build all GPU ring slots inside one contiguous CUDA buffer."""
        if self._gpu_storage is not None or not self._gpu_pool:
            return
        with torch.inference_mode(False):
            slots, raw, slot_total = _build_global_pool(
                block, len(self._gpu_pool), device=self.device)
        self._gpu_pool = slots
        self._gpu_built = [True] * len(self._gpu_pool)
        self._gpu_storage = raw
        self._gpu_slot_size = slot_total

    def release_gpu_pool(self) -> None:
        """Drop every GPU ring allocation after a full D2H offload.

        Used before a non-DiT compute phase (VAE encode/decode) so the GPU
        ring memory is actually returned to the allocator instead of merely
        being unassigned.  The next ``load_block`` rebuilds the ring slots.
        """
        self.sync_all()
        with self._transfer_lock:
            if self._scheduler is not None:
                self._scheduler.close()
                self._scheduler = None
            self._gpu_pool = [None] * len(self._gpu_pool)
            self._gpu_built = [False] * len(self._gpu_built)
            self._gpu_storage = None
            self._gpu_slot_size = 0
            self._block_gpu.clear()
            self._events.clear()
            self._gpu_h2d_events.clear()
            self._gpu_d2h_events.clear()
            self._gpu_cursor = 0
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _wait_home_slot(self, hslot: int) -> None:
        """Host-wait for any in-flight async D2H into home slot *hslot*."""
        ev = self._d2h_events.pop(hslot, None)
        ready = self._home_ready.get(hslot)
        if ready is not None:
            ready.wait()
        elif ev is not None:
            ev.synchronize()

    @staticmethod
    def _assign(block: SwapBlock, entries: dict) -> None:
        """Rebind the block's params/buffers to *entries* (host-side only)."""
        missing = [name for name in block.names if name not in entries]
        if missing:
            raise RuntimeError(
                f"blockswap: {block.name} missing slot entries: {missing}"
            )
        for n, (mod, leaf, kind) in zip(block.names, block.refs):
            e = entries.get(n)
            if e is None:
                continue
            if kind == "param":
                e.assign_to(mod, leaf)
            else:
                mod._buffers[leaf] = e.data

    # -- transfers -----------------------------------------------------------

    def ensure_home(self, block_idx: int, block: SwapBlock,
                    non_blocking: bool = False) -> int:
        """Ensure *block_idx* has a home slot; return its index.

        With ``non_blocking=True`` the disk read is started on a worker and
        the caller is responsible for ``wait_home_fill(hslot)`` before using
        the slot.  ``non_blocking=False`` preserves the old synchronous
        contract used by the compute-critical load path.
        """
        if block_idx in self._block_home:
            hslot = self._block_home[block_idx]
            if not non_blocking:
                self._home.wait_home_fill(hslot)
            return hslot
        try:
            hslot = self._acquire_home()
        except RuntimeError:
            if not self.pin_memory and self._block_home:
                victim = min(self._block_home)
                hslot = self._block_home.pop(victim)
                self._wait_home_slot(hslot)
                self._home.wait_home_fill(hslot)
                self._home.release(victim)
            else:
                hslot = self._wait_home_free()
        # The acquired slot may still be receiving an async D2H from a
        # previous offload - read_into overwrites it from the host.
        self._wait_home_slot(hslot)
        entries = self._ensure_entries(self._home_pool, self._home_built,
                                       hslot, block, "cpu")
        if non_blocking:
            self._home.start_home_fill(block_idx, hslot, entries, block)
        else:
            self._home.read_into(block_idx, entries, block)
        self._block_home[block_idx] = hslot
        return hslot

    def load_block(self, block_idx: int, block: SwapBlock,
                   non_blocking: bool = True) -> None:
        """Copy *block*'s weights onto a GPU ring slot."""
        with self._transfer_lock:
            if block_idx in self._block_gpu:
                return
            hslot = self.ensure_home(block_idx, block, non_blocking=False)
            # The reload source is the home slot itself, so any in-flight async
            # D2H into it must finish before we read it.
            home_registered = (
                self.pin_memory
                and self._home_registered.get(hslot) is not None
            )
            d2h_ev = None
            if home_registered:
                d2h_ev = self._d2h_events.get(hslot)
            else:
                self._wait_home_slot(hslot)
            self._home.wait_home_fill(hslot)
            ridx = self._ring_acquire()
            gpu = self._ensure_entries(self._gpu_pool, self._gpu_built,
                                       ridx, block, self.device)
            prev_d2h = self._gpu_d2h_events.get(ridx)
            if prev_d2h is not None:
                if self._stream is not None:
                    self._stream.wait_event(prev_d2h)
                else:
                    torch.cuda.current_stream().wait_event(prev_d2h)
            nb = self._stream is not None and non_blocking

            home_entries = self._home_pool[hslot]
            src = {}
            for n, (mod, leaf, kind) in zip(block.names, block.refs):
                e = home_entries.get(n)
                if e is not None:
                    src[n] = e

            if (self.pin_memory and self._stream is not None
                    and self._home_registered.get(hslot) is not None):
                if d2h_ev is not None:
                    self._stream.wait_event(d2h_ev)
                with torch.cuda.stream(self._stream):
                    if (self._gpu_storage is not None
                            and self._home_storage is not None
                            and self._home_slot_size
                            and self._gpu_slot_size == self._home_slot_size):
                        hoff = hslot * self._home_slot_size
                        goff = ridx * self._gpu_slot_size
                        nbytes = self._home_slot_size
                        self._gpu_storage[
                            goff:goff + nbytes
                        ].copy_(
                            self._home_storage[hoff:hoff + nbytes],
                            non_blocking=True,
                        )
                    else:
                        for n, e in src.items():
                            ge = gpu.get(n)
                            if ge is None:
                                continue
                            ge.copy_from(e, non_blocking=True)
                compute = torch.cuda.current_stream()
                ev = torch.cuda.Event()
                ev.record(self._stream)
                compute.wait_event(ev)
                self._gpu_h2d_events[ridx] = ev
                self._events[block_idx] = ev
                self._assign(block, gpu)
                hslot = self._block_home.pop(block_idx, None)
                if hslot is not None:
                    self._release_home(hslot)
                self._block_gpu[block_idx] = ridx
                return
            elif (self._scheduler is not None and non_blocking
                    and self._stream is not None):
                wait_events = []
                if prev_d2h is not None:
                    wait_events.append(prev_d2h)
                self._scheduler.schedule_h2d_entries(
                    ridx, src, gpu, tuple(wait_events))
                ev = self._scheduler.slot_event(ridx, "h2d")
                self._gpu_h2d_events[ridx] = ev
                self._events[block_idx] = ev
                self._assign(block, gpu)
                hslot_pop = self._block_home.pop(block_idx, None)
                if hslot_pop is not None:
                    self._cleanup_exec.submit(
                        self._release_home_after_h2d, hslot_pop, ev)
                self._block_gpu[block_idx] = ridx
                return
            else:
                for n, e in src.items():
                    ge = gpu.get(n)
                    if ge is None:
                        continue
                    ge.copy_from(e, non_blocking=False)
                nb = False

            self._assign(block, gpu)

            hslot = self._block_home.pop(block_idx, None)
            if hslot is not None:
                self._release_home(hslot)
            self._block_gpu[block_idx] = ridx
            if nb:
                ev = torch.cuda.Event()
                ev.record(self._stream)
                self._gpu_h2d_events[ridx] = ev
                self._events[block_idx] = ev

    def offload_block(self, block_idx: int, block: SwapBlock,
                      force: bool = False) -> bool:
        """Copy *block* back to a home slot; returns True if moved.

        Copies GPU slot entry -> home slot entry directly (NOT through the
        block params, which may already be freed).  D2H runs on the transfer
        stream so it overlaps subsequent compute; ``_wait_home_slot`` guards
        host-side readers of the slot.  When no home slot is free the offload
        is deferred (block stays on GPU) unless *force* steals the lowest
        block's slot - the victim's RAM copy is released and will be re-read
        from disk on its next use.
        """
        with self._transfer_lock:
            event = self._events.pop(block_idx, None)
            if event is not None:
                torch.cuda.current_stream().wait_event(event)
            gslot = self._block_gpu.pop(block_idx, None)
            if gslot is None:
                return False
            hslot = self._block_home.get(block_idx)
            if hslot is None:
                try:
                    hslot = self._acquire_home()
                except RuntimeError:
                    if not force:
                        try:
                            hslot = self._wait_home_free()
                        except RuntimeError:
                            self._block_gpu[block_idx] = gslot
                            return False
                    else:
                        victims = list(self._block_home)
                        if not victims:
                            self._block_gpu[block_idx] = gslot
                            return False
                        victim = min(victims)
                        hslot = self._block_home.pop(victim)
                        self._wait_home_slot(hslot)
                        self._home.wait_home_fill(hslot)
                        self._home.release(victim)
            self._wait_home_slot(hslot)
            self._home.wait_home_fill(hslot)
            gpu = self._gpu_pool[gslot]
            home = self._ensure_entries(self._home_pool, self._home_built,
                                        hslot, block, "cpu")

            prev_h2d = self._gpu_h2d_events.get(gslot)
            if (self.pin_memory and self._d2h_stream is not None
                    and self._home_registered.get(hslot) is not None):
                if prev_h2d is not None:
                    self._d2h_stream.wait_event(prev_h2d)
                self._d2h_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._d2h_stream):
                    if (self._gpu_storage is not None
                            and self._home_storage is not None
                            and self._home_slot_size
                            and self._gpu_slot_size == self._home_slot_size):
                        hoff = hslot * self._home_slot_size
                        goff = gslot * self._gpu_slot_size
                        nbytes = self._home_slot_size
                        self._home_storage[
                            hoff:hoff + nbytes
                        ].copy_(
                            self._gpu_storage[goff:goff + nbytes],
                            non_blocking=True,
                        )
                    else:
                        for n in list(gpu.keys() & home.keys()):
                            home[n].copy_from(gpu[n], non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(self._d2h_stream)
                self._gpu_d2h_events[gslot] = ev
                self._d2h_events[hslot] = ev
                self._block_home[block_idx] = hslot
                return True
            # Last-resort synchronous fallback.
            self.d2h_sync_fallback += 1
            if prev_h2d is not None:
                torch.cuda.current_stream().wait_event(prev_h2d)
            for n in list(gpu.keys() & home.keys()):
                home[n].copy_from(gpu[n])
            self._gpu_d2h_events.pop(gslot, None)
            self._assign(block, home)
            self._block_home[block_idx] = hslot
            return True

    def _release_home_after_h2d(self, hslot: int,
                                ev: torch.cuda.Event) -> None:
        """Release a pageable home slot only after its async H2D completes."""
        try:
            ev.synchronize()
        finally:
            self._release_home(hslot)

    def discard_block(self, block_idx: int) -> None:
        """Drop *block_idx* from the GPU ring without writing weights home.

        Only used at shutdown (lazy disk mode): disk is the source of truth
        and the pools are about to be cleared, so the write-back is waste.
        """
        self._events.pop(block_idx, None)
        self._block_gpu.pop(block_idx, None)

    # -- prefetch ------------------------------------------------------------

    def start_prefetch(self, block_idx: int, block: SwapBlock) -> None:
        if block_idx in self._block_gpu or block_idx in self._events:
            return
        if block_idx not in self._block_home and self.home_free_count() <= 0:
            return
        # Pageable home slots cannot DMA asynchronously.  Background reads
        # would only add a second full block copy during compute and can
        # exhaust host memory alongside D2H home-slot allocation.
        if not self.pin_memory:
            return
        try:
            hslot = self.ensure_home(block_idx, block, non_blocking=True)
        except RuntimeError:
            return
        if self._home_registered.get(hslot) is not None:
            self.load_block(block_idx, block, non_blocking=True)

    def flush_prefetch(self) -> None:
        return

    def sync_prefetch(self, block_idx: int) -> bool:
        """Wait for *block_idx*'s async H2D event if it was scheduled."""
        event = self._events.pop(block_idx, None)
        if event is not None:
            torch.cuda.current_stream().wait_event(event)
            return True
        return False

    def cancel_all(self) -> None:
        if not self._events and not self._gpu_h2d_events and not self._gpu_d2h_events:
            return
        if self._stream is not None:
            self._stream.synchronize()
        if self._d2h_stream is not None:
            self._d2h_stream.synchronize()
        self._events.clear()
        self._gpu_h2d_events.clear()
        self._gpu_d2h_events.clear()

    def sync_all(self) -> None:
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

    def reset_step(self) -> None:
        """Clear per-step block events without synchronizing transfer streams."""
        self._events.clear()

    def clear_pools(self) -> None:
        if self._scheduler is not None:
            self._scheduler.close()
            self._scheduler = None
        try:
            self._cleanup_exec.shutdown(wait=True, timeout=30)
        except Exception:
            pass
        if self._home_storage_tokens is not None:
            try:
                _unregister_cpu_entries(self._home_storage_tokens)
            except Exception:
                pass
        self._home_storage_tokens = None
        self._home_registered.clear()
        self._home_storage = None
        self._home_slot_size = 0
        nh, ng = len(self._home_pool), len(self._gpu_pool)
        self._home_pool = [None] * nh
        self._gpu_pool = [None] * ng
        self._gpu_storage = None
        self._gpu_slot_size = 0
        self._home_ready = {}
        self._home_built = [False] * nh
        self._gpu_built = [False] * ng
        with self._home_free_lock:
            self._home_free = list(range(nh))
        self._gpu_cursor = 0
        self._block_home.clear()
        self._block_gpu.clear()
        self._events.clear()
        self._gpu_h2d_events.clear()
        self._gpu_d2h_events.clear()
        self._d2h_events.clear()




# ---------------------------------------------------------------------------
# BlockSwapManager - sliding-window orchestration
# ---------------------------------------------------------------------------

class BlockSwapManager:
    def __init__(self, blocks: list[SwapBlock], reader: BlockReader, device,
                 allocation: SwapAllocation,
                 prefetch: bool = True,
                 pin_memory: bool = True,
                 disk_workers: int = 2,
                 dtype=torch.bfloat16):
        if allocation is None:
            raise ValueError("BlockSwapManager requires a SwapAllocation.")
        self.blocks = blocks
        self.total = len(blocks)
        self.device = torch.device(device)
        self.dtype = dtype
        self.prefetch = bool(prefetch)
        pool = allocation.pool
        self.window_size = pool.window_size
        self.prefetch_count = pool.prefetch_count
        self.hot_blocks = pool.hot_blocks
        self.home_size = pool.home_slots
        assert self.window_size >= 1, "window_size must be >= 1"
        assert self.window_size <= self.total, "window_size exceeds block count"
        assert self.prefetch_count >= 0, "prefetch_count must be >= 0"
        assert (
            self.hot_blocks <= max(0, self.window_size - 1)
        ), "hot_blocks must fit inside window"
        assert (
            pool.gpu_slots >= self.window_size
        ), "gpu_slots must cover the window"
        assert (
            0 <= self.home_size <= self.total
        ), "home_slots must stay within the model block count"
        self.swap_hits = 0
        self.swap_loads = 0
        self._flushed_this_step = False
        block_mb = pool.block_mb
        self._window = _BlockWindow(
            self.total, self.window_size, hot_blocks=self.hot_blocks)
        self.hot_blocks = len(self._window.hot)

        self._disk = _DiskPrefetcher(reader, blocks, max_workers=disk_workers)
        self._xfer = _TransferEngine(
            device, prefetch, self.prefetch_count, pin_memory,
            n_home_slots=max(self.home_size, 1),
            n_gpu_slots=self.window_size + self.prefetch_count,
            home=self._disk,
        )
        block_mb = (blocks[0].bytes_per_block() / 2 ** 20) if blocks else 0.0
        self._budget = _VRAMBudget(block_mb)
        if blocks and self._xfer.pin_memory:
            self._xfer.preallocate_home_pool(blocks[0])
        if blocks and self._xfer.pin_memory:
            self._xfer.preallocate_gpu_pool(blocks[0])
        pin_label = (
            "True"
            if self._xfer.pin_memory
            else "False (pageable fallback)"
        )
        logger.info("%d blocks, window=%d, hot=%d, ~%.0f MB/block, prefetch=%s x%d, pin=%s",
                    self.total, self.window_size, self.hot_blocks, block_mb,
                    self.prefetch, self.prefetch_count, pin_label)

    # -- public API ----------------------------------------------------------

    def apply_lora(self, block_idx: int, entries: list) -> None:
        self.blocks[block_idx].lora = list(entries)

    def clear_lora(self, block_idx: int) -> None:
        self.blocks[block_idx].lora = None

    def begin(self) -> None:
        self._flushed_this_step = False

    def prepare(self, block_idx: int) -> None:
        """Ensure block *block_idx* and its window are on GPU.

        Load and offload are interleaved *load-first*, so the incoming block
        frees its home slot before the outgoing one needs it - the home pool
        stays balanced at (N-W) and host RAM never spikes.  A block whose
        offload finds no free home slot is deferred (left on GPU) and retired
        at shutdown.
        """
        check_interrupt()
        if block_idx in self._window.on_gpu:
            self.swap_hits += 1
        to_offload = sorted(self._window.to_offload(block_idx))
        to_load_now = sorted(self._window.to_load(block_idx))
        did_offload = bool(to_offload)
        self._start_parallel_home_fills(to_load_now)

        n = max(len(to_offload), len(to_load_now))
        for k in range(n):
            home_freed = False
            if (not self._xfer.pin_memory
                    or self._xfer._home_registered):
                if k < len(to_load_now):
                    j = to_load_now[k]
                    if not self._xfer.sync_prefetch(j):
                        self._xfer.load_block(
                            j, self.blocks[j], non_blocking=True)
                        self._xfer.sync_prefetch(j)
                    self._window.mark_loaded(j)
                    self.swap_loads += 1
                    self._fold_lora_on_gpu(j)
                if k < len(to_offload):
                    i = to_offload[k]
                    if (i in self._xfer._block_home
                            or self._xfer.home_free_count() > 0):
                        if self._xfer.offload_block(i, self.blocks[i]):
                            self._window.mark_offloaded(i)
            else:
                if k < len(to_load_now):
                    j = to_load_now[k]
                    home_before = self._xfer.home_free_count()
                    if not self._xfer.sync_prefetch(j):
                        self._xfer.load_block(
                            j, self.blocks[j], non_blocking=True)
                        self._xfer.sync_prefetch(j)
                    self._window.mark_loaded(j)
                    self.swap_loads += 1
                    self._fold_lora_on_gpu(j)
                    home_freed = self._xfer.home_free_count() > home_before
                if k < len(to_offload):
                    i = to_offload[k]
                    if home_freed or self._xfer.home_free_count() > 0:
                        if self._xfer.offload_block(i, self.blocks[i]):
                            self._window.mark_offloaded(i)

        # bulletproof residency guarantee for the window
        for j in sorted(self._window.needed(block_idx)):
            if self._xfer._block_gpu.get(j) is None:
                if not self._xfer.sync_prefetch(j):
                    self._xfer.load_block(j, self.blocks[j], non_blocking=True)
                    self._xfer.sync_prefetch(j)
                self._window.mark_loaded(j)
                self._fold_lora_on_gpu(j)

        self._fold_lora_on_gpu(block_idx)

        if did_offload and not self._flushed_this_step:
            self._budget.maybe_flush(self._xfer.device, reserve_blocks=2)
            self._flushed_this_step = True
        if self.prefetch:
            self._xfer.flush_prefetch()
            self.prefetch_next(block_idx)

    def after_compute(self, block_idx: int) -> None:
        """Enqueue D2H immediately after a non-hot block's forward returns.

        The D2H transfer stream waits on the current compute stream, so the
        copy runs after this block's kernels and can overlap the next block's
        compute.  Blocks kept in the hot set stay on GPU.
        """
        if block_idx in self._window.hot:
            return
        if block_idx not in self._window.on_gpu:
            return
        if self._xfer._block_gpu.get(block_idx) is None:
            return
        if self._xfer.offload_block(block_idx, self.blocks[block_idx]):
            self._window.mark_offloaded(block_idx)

    def offload_all(self) -> None:
        """Move every resident GPU block home and release the GPU ring.

        This is the DIT side of a rolling VAE phase: the window/hot blocks are
        written back to their home slots and the preallocated CUDA ring storage
        is freed so VAE weights can occupy the GPU alone.
        """
        if self.total == 0:
            return
        self._xfer.sync_all()
        resident = sorted(self._window.on_gpu)
        prefetch = sorted(set(self._xfer._block_gpu) - set(resident))
        if not resident and not prefetch:
            self._xfer.release_gpu_pool()
            return
        for idx in prefetch:
            self._xfer.discard_block(idx)
        moved = 0
        for idx in resident:
            if self._xfer.offload_block(idx, self.blocks[idx], force=True):
                self._window.mark_offloaded(idx)
                moved += 1
        if moved < len(resident):
            logger.warning(
                "blockswap: partial DIT offload for VAE phase "
                "(%d/%d moved); keeping GPU ring allocated",
                moved, len(resident),
            )
            self._xfer.sync_all()
            return
        self._xfer.sync_all()
        self._window.clear()
        self._xfer.release_gpu_pool()

    def restore_initial(self) -> None:
        """Reload the first block window after a VAE phase."""
        if self.total == 0:
            return
        needed = set(range(0, min(self.total, self.window_size)))
        needed |= self._window.hot
        for idx in sorted(set(self._xfer._block_home) - needed):
            hslot = self._xfer._block_home.pop(idx, None)
            if hslot is None:
                continue
            self._xfer._wait_home_slot(hslot)
            self._xfer._home.wait_home_fill(hslot)
            self._xfer._home.release(idx)
            self._xfer._release_home(hslot)
        self.prepare(0)

    def prefetch_next(self, block_idx: int) -> None:
        """Prefetch blocks just beyond the window.

        Disk -> RAM reads for the range after the H2D prefetch range run on
        the background workers, so ``start_prefetch`` rarely blocks on disk;
        the nearest blocks are async-loaded onto the GPU ring on the transfer
        stream, overlapping the current block's compute.
        """
        pending_offload = self._window.to_offload(block_idx)
        if pending_offload and self._xfer.home_free_count() <= 1:
            return
        start = block_idx + self._window.window_size
        if start >= self.total:
            return
        end = min(self.total, start + self.prefetch_count)
        for i in range(end, min(self.total, end + self.prefetch_count)):
            self._disk.start_ram_load(i)
        for i in range(start, end):
            if self._window.is_on_gpu(i):
                continue
            if (
                i not in self._xfer._block_home
                and self._xfer.home_free_count() <= 0
            ):
                break
            self._xfer.start_prefetch(i, self.blocks[i])

    def end(self) -> None:
        self._flushed_this_step = False

    def shutdown(self) -> None:
        """Full teardown; safe to call on model unload/switch.

        Lazy disk mode: disk is the source of truth, so GPU ring blocks are
        discarded without write-back and every block's CPU RAM is released.
        """
        try:
            self._xfer.cancel_all()
        except Exception:
            pass
        try:
            self._disk.join()
        except Exception:
            pass
        self._xfer.sync_all()
        for idx in list(self._window.on_gpu):
            self._xfer.discard_block(idx)
        self._window.clear()
        self._xfer.sync_all()
        try:
            self._disk.shutdown()
        except Exception:
            pass
        for idx in range(self.total):
            self._disk.release(idx)
        self._xfer.clear_pools()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    def stats(self) -> BlockSwapStats:
        return BlockSwapStats(
            swap_hits=self.swap_hits,
            swap_loads=self.swap_loads,
            home_size=self.home_size,
            total=self.total,
            window=self.window_size,
            hot=self.hot_blocks,
            disk_reads=self._disk.disk_reads,
            d2h_stage=self._xfer.d2h_stage_hits,
            d2h_direct=0,
            d2h_host_register=0,
            d2h_sync=self._xfer.d2h_sync_fallback,
        )

    # -- internal ------------------------------------------------------------

    def _fold_lora_on_gpu(self, block_idx: int) -> None:
        """Fold *block_idx*'s pending LoRA payload into its GPU ring slot.

        The payload (``block.lora``) is consumed to ``None`` afterwards, so
        "folded or not" is inferred from the payload itself and re-loads
        (whose slots already hold merged weights) skip folding automatically.
        """
        block = self.blocks[block_idx]
        if not block.lora:
            return
        gslot = self._xfer._block_gpu.get(block_idx)
        if gslot is None:
            return
        fold_lora_into_slot(block, self._xfer._gpu_pool[gslot])
        block.lora = None

    def _start_parallel_home_fills(self, to_load: list[int]) -> None:
        """Submit cold disk reads to the worker pool before the swap loop.

        ``load_block`` normally calls ``ensure_home(non_blocking=False)``,
        which reads one block synchronously.  For the first pass this makes
        ``disk_workers`` useless.  Starting the home fills ahead of time lets
        the disk pool read multiple blocks while the swap loop waits on them
        one by one.
        """
        xfer = self._xfer
        for block_idx in to_load:
            if xfer._block_gpu.get(block_idx) is not None:
                continue
            if block_idx in xfer._block_home:
                continue
            if xfer.home_free_count() <= 0:
                break
            try:
                xfer.ensure_home(
                    block_idx, self.blocks[block_idx], non_blocking=True)
            except RuntimeError:
                break
