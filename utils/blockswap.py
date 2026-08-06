"""Ring-buffer BlockSwap for the MiniMax H3 DiT - BerniniRWrapper-style.

Mirrors ``ComfyUI-BerniniRWrapper/utils/block_swap.py`` (lazy disk mode):

* ``_BlockWindow`` - pure residency tracking (which blocks are on GPU).
* ``PinStage`` - small pinned CPU staging ring so H2D copies on the transfer
  stream are true async DMA (the big home pool stays pageable).
* ``_TransferEngine`` - fixed slot pools: a CPU "home" pool of
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
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import torch

from ..models import quant
from ..models.lora import fold_lora_into_slot
from .interrupt import check_interrupt
from .stream import BlockReader
from .types import SlotEntry

logger = logging.getLogger("h3.blockswap")


@dataclass
class SwapBlock:
    """One swappable weight group (a DiT block)."""
    name: str
    module: torch.nn.Module
    keys: list[str] = field(default_factory=list)            # full safetensors names
    names: list[str] = field(default_factory=list)           # relative param names
    refs: list[tuple[torch.nn.Module, str, str]] = field(default_factory=list)
    templates: list[SlotEntry] = field(default_factory=list)  # slot templates
    lora: Optional[list] = None                                # pending LoRA payload
    overrides: dict = field(default_factory=dict)              # full weight overrides

    def bytes_per_block(self) -> int:
        total = 0
        for t in self.templates:
            total += t.data.numel() * t.data.element_size()
            if t.scale is not None:
                total += t.scale.numel() * t.scale.element_size()
            for e in t.extra.values():
                total += e.numel() * e.element_size()
        return total


def _dtype_of(t) -> torch.dtype:
    """Safe dtype extraction (QuantizedTensor hides .dtype behind a property)."""
    try:
        return t.dtype
    except Exception:
        return torch.float32


def free_module_storage(module: torch.nn.Module) -> None:
    """Replace every parameter/buffer with a zero-size tensor.

    Safe for QuantizedTensor and any tensor subclass: the old object is
    dereferenced via ``_parameters[key] = ...`` replacement.  ``param.data =
    empty(0)`` would silently keep QuantizedTensor storage alive because its
    ``__torch_dispatch__`` intercepts the assignment.
    """
    for key in list(module._parameters.keys()):
        p = module._parameters.get(key)
        if p is None:
            continue
        try:
            module._parameters[key] = torch.nn.Parameter(
                torch.empty((0,), dtype=_dtype_of(p)), requires_grad=False)
        except Exception:
            pass
    for key in list(module._buffers.keys()):
        b = module._buffers.get(key)
        if b is None:
            continue
        try:
            module._buffers[key] = torch.empty((0,), dtype=_dtype_of(b))
        except Exception:
            pass


def _entry_of(t) -> SlotEntry:
    """Transient SlotEntry view over a live param/buffer tensor."""
    if hasattr(t, "_qdata") and hasattr(t, "_params"):
        return SlotEntry.from_qt(t)
    return SlotEntry(data=t)


def _slot_tensors(entry: SlotEntry):
    yield entry.data
    if entry.scale is not None:
        yield entry.scale
    for extra in entry.extra.values():
        yield extra


def _register_cpu_entries(entries: dict) -> list[tuple[int, int]]:
    """Temporarily pin pageable home tensors with CUDA host registration.

    This lets a pageable home slot be used directly as the source/destination
    of an async transfer, avoiding a full block-sized CPU copy through the
    pinned staging ring.  The returned tokens must be unregistered after the
    corresponding CUDA event completes.
    """
    cudart = torch.cuda.cudart()
    tokens: list[tuple[int, int]] = []
    try:
        for entry in entries.values():
            for t in _slot_tensors(entry):
                if t.numel() == 0:
                    continue
                ptr = t.data_ptr()
                nbytes = t.numel() * t.element_size()
                err = cudart.cudaHostRegister(ptr, nbytes, 0)
                if err != 0:
                    raise RuntimeError(
                        f"cudaHostRegister failed: {torch.cuda.cudaError(err)}")
                tokens.append((ptr, nbytes))
    except Exception:
        for ptr, _nbytes in tokens:
            cudart.cudaHostUnregister(ptr)
        raise
    return tokens


def _unregister_cpu_entries(tokens: list[tuple[int, int]]) -> None:
    cudart = torch.cuda.cudart()
    for ptr, _nbytes in tokens:
        cudart.cudaHostUnregister(ptr)


# ---------------------------------------------------------------------------
# _BlockWindow - pure residency tracking (no CUDA, no tensors)
# ---------------------------------------------------------------------------

class _BlockWindow:
    __slots__ = ("on_gpu", "total", "window_size", "hot")

    def __init__(self, total: int, window_size: int, hot_blocks: int = 0):
        self.total = total
        max_hot = max(0, min(window_size, total) - 1)
        self.hot = set(range(max(0, min(hot_blocks, max_hot))))
        self.window_size = max(
            1, min(window_size - len(self.hot),
                   max(1, total - len(self.hot))))
        self.on_gpu: set[int] = set()

    def needed(self, block_idx: int) -> set[int]:
        return set(range(block_idx, min(self.total, block_idx + self.window_size))) | self.hot

    def to_offload(self, block_idx: int) -> set[int]:
        return self.on_gpu - self.needed(block_idx)

    def to_load(self, block_idx: int) -> set[int]:
        return self.needed(block_idx) - self.on_gpu

    def is_on_gpu(self, block_idx: int) -> bool:
        return block_idx in self.on_gpu

    def mark_loaded(self, block_idx: int) -> None:
        self.on_gpu.add(block_idx)

    def mark_offloaded(self, block_idx: int) -> None:
        self.on_gpu.discard(block_idx)

    def clear(self) -> None:
        self.on_gpu.clear()


# ---------------------------------------------------------------------------
# PinStage - pinned CPU staging ring, indexed by the GPU ring cursor
# ---------------------------------------------------------------------------

@dataclass
class PinStage:
    slots: list[dict | None] = field(default_factory=list)
    built: list[bool] = field(default_factory=list)

    @classmethod
    def new(cls, ring_size: int) -> "PinStage":
        return cls(slots=[None] * ring_size, built=[False] * ring_size)

    @property
    def size(self) -> int:
        return len(self.slots)

    def ensure(self, ring_idx: int, template: dict) -> dict:
        idx = ring_idx % self.size if self.size else 0
        slot = self.slots[idx]
        if slot is not None and self.built[idx]:
            return slot
        with torch.inference_mode(False):
            self.slots[idx] = {
                n: SlotEntry.empty_like_entry(e, "cpu", pin_memory=True)
                for n, e in template.items()
            }
        self.built[idx] = True
        return self.slots[idx]


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

        nh = max(0, n_home_slots)
        ng = max(0, n_gpu_slots)
        self._home_pool: list[dict | None] = [None] * nh
        self._home_built: list[bool] = [False] * nh
        self._home_free: list[int] = list(range(nh))
        self._home_free_lock = threading.Lock()
        self._home_free_cond = threading.Condition(self._home_free_lock)
        self._home_h2d_release: set[threading.Event] = set()
        self._d2h_cleanup_pending: set[threading.Event] = set()
        self._d2h_cleanup_lock = threading.Lock()
        self._gpu_pool: list[dict | None] = [None] * ng
        self._gpu_built: list[bool] = [False] * ng
        self._pin: PinStage = PinStage.new(max(4, self.prefetch_count + 2))
        # pinned D2H staging ring: offloads land here as true async DMA, then a
        # background thread copies into the pageable home slot.  The ring keeps
        # the pinned footprint small and every slot is guarded by an in-use
        # event so a slot is never reused under an in-flight copy.
        self._d2h_stage: PinStage = PinStage.new(max(4, self.prefetch_count + 2))
        self._stage_inuse: list[threading.Event] = [
            threading.Event() for _ in range(self._d2h_stage.size)]
        for _e in self._stage_inuse:
            _e.set()
        # hslot -> event set once the background pinned->home copy completed
        self._home_ready: dict[int, threading.Event] = {}
        # H2D staging slot -> event of its last async H2D (guards refill races)
        self._pin_events: dict[int, torch.cuda.Event] = {}
        self._home_exec = ThreadPoolExecutor(max_workers=max(2, self.prefetch_count),
                                             thread_name_prefix="h3d2h")
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
        # block_idx -> (slot id, ready event, future, block, mode)
        self._pin_stage: dict[int, tuple[int, threading.Event, Future, SwapBlock, str]] = {}
        # hslot -> event recorded after an async D2H into that home slot
        self._d2h_events: dict[int, torch.cuda.Event] = {}
        self._use_host_register = (
            os.environ.get("H3_BLOCKSWAP_HOST_REGISTER", "0") == "1"
            and torch.cuda.is_available()
        )
        self._use_host_register_d2h = (
            os.environ.get("H3_BLOCKSWAP_D2H_SYNC", "0") != "1"
            and torch.cuda.is_available()
        )
        self._home_registered: dict[int, list[tuple[int, int]]] = {}

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
        with torch.inference_mode(False):
            pool[idx] = {
                n: SlotEntry.empty_like_entry(tpl, device=device, pin_memory=False)
                for n, tpl in zip(block.names, block.templates)
            }
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
        deadline = time.monotonic() + timeout
        while True:
            with self._home_free_cond:
                if self._home_free:
                    return self._home_free.pop()
                pending = list(self._home_h2d_release)
            if not pending:
                raise RuntimeError("blockswap: no free home slot")
            for ev in pending:
                ev.wait(0.05)
            if time.monotonic() >= deadline:
                raise RuntimeError("blockswap: no free home slot")

    def _pin_acquire(self) -> Optional[int]:
        used = {item[0] for item in self._pin_stage.values()
                if item[4] == "pin"}
        for pidx in range(self._pin.size):
            if pidx not in used:
                return pidx
        return None

    def _wait_home_slot(self, hslot: int) -> None:
        """Host-wait for any in-flight async D2H into home slot *hslot*."""
        ev = self._d2h_events.pop(hslot, None)
        if ev is not None:
            ev.synchronize()
        ready = self._home_ready.get(hslot)
        if ready is not None:
            ready.wait()

    def _stage_to_home(self, hslot: int, stage: dict, home: dict,
                       ev: torch.cuda.Event, ready: threading.Event,
                       sidx: int) -> None:
        try:
            ev.synchronize()
            with torch.inference_mode():
                for n, se in stage.items():
                    he = home.get(n)
                    if he is not None:
                        he.copy_from(se)
        except Exception as e:
            logger.error("d2h stage->home copy failed (hslot=%d): %s", hslot, e)
        finally:
            ready.set()
            self._complete_h2d(block_idx, block, wait_ready=False)
            self._stage_inuse[sidx].set()

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

            if non_blocking and self._stream is not None and self.pin_memory:
                pidx = self._pin_acquire()
                if pidx is not None:
                    prev = self._pin_events.pop(pidx, None)
                    if prev is not None:
                        prev.synchronize()
                    pin = self._pin.ensure(pidx, gpu)
                    for n, e in src.items():
                        pe = pin.get(n)
                        if pe is not None:
                            pe.copy_from(e)
                    self._stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(self._stream):
                        for n in pin:
                            pe, ge = pin[n], gpu.get(n)
                            if ge is None:
                                continue
                            ge.copy_from(pe, non_blocking=True)
                    compute = torch.cuda.current_stream()
                    ev = torch.cuda.Event()
                    ev.record(self._stream)
                    compute.wait_event(ev)
                    self._gpu_h2d_events[ridx] = ev
                    self._pin_events[pidx] = ev
                    self._events[block_idx] = ev
                    nb = False
                else:
                    for n, e in src.items():
                        ge = gpu.get(n)
                        if ge is None:
                            continue
                        ge.copy_from(e, non_blocking=False)
                    nb = False
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
                        victims = [idx for idx in self._block_home
                                   if idx not in self._pin_stage]
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
            if self._use_host_register_d2h and self._d2h_stream is not None:
                try:
                    tokens = _register_cpu_entries(home)
                except Exception as e:
                    self._use_host_register_d2h = False
                    logger.error(
                        "blockswap: cudaHostRegister for D2H failed for %s: %s",
                        block.name, e)
                    tokens = None
                if tokens is not None:
                    self._home_registered[hslot] = tokens
                    if prev_h2d is not None:
                        self._d2h_stream.wait_event(prev_h2d)
                    self._d2h_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(self._d2h_stream):
                        for n in list(gpu.keys() & home.keys()):
                            home[n].copy_from(gpu[n], non_blocking=True)
                    ev = torch.cuda.Event()
                    ev.record(self._d2h_stream)
                    self._gpu_d2h_events[gslot] = ev
                    ready = threading.Event()
                    self._d2h_events[hslot] = ev
                    self._home_ready[hslot] = ready
                    self._block_home[block_idx] = hslot
                    with self._d2h_cleanup_lock:
                        self._d2h_cleanup_pending.add(ready)
                    try:
                        self._cleanup_exec.submit(
                            self._unregister_home_after_d2h,
                            hslot, ev, tokens, ready)
                    except Exception:
                        ev.synchronize()
                        _unregister_cpu_entries(tokens)
                        self._home_registered.pop(hslot, None)
                        ready.set()
                        with self._d2h_cleanup_lock:
                            self._d2h_cleanup_pending.discard(ready)
                    return True

            # Synchronous fallback when host registration is unavailable.
            if prev_h2d is not None:
                torch.cuda.current_stream().wait_event(prev_h2d)
            for n in list(gpu.keys() & home.keys()):
                home[n].copy_from(gpu[n])
            self._gpu_d2h_events.pop(gslot, None)
            self._assign(block, home)
            self._block_home[block_idx] = hslot
            return True

    def _unregister_home_after_d2h(self, hslot: int, ev: torch.cuda.Event,
                                   tokens: list[tuple[int, int]],
                                   ready: threading.Event) -> None:
        """Worker: release temporary host registration after async D2H."""
        try:
            ev.synchronize()
        finally:
            _unregister_cpu_entries(tokens)
            self._home_registered.pop(hslot, None)
            ready.set()
            with self._d2h_cleanup_lock:
                self._d2h_cleanup_pending.discard(ready)

    def wait_d2h_cleanup(self) -> None:
        """Wait for async D2H cleanup tasks before a sampling pass ends."""
        while True:
            with self._d2h_cleanup_lock:
                pending = list(self._d2h_cleanup_pending)
            if not pending:
                return
            for ev in pending:
                ev.wait(0.1)

    def _unregister_home_after_h2d(self, hslot: int, ev: torch.cuda.Event,
                                   tokens: list[tuple[int, int]],
                                   release_ready: threading.Event) -> None:
        """Worker: release a temporary H2D home registration after DMA."""
        try:
            ev.synchronize()
        finally:
            _unregister_cpu_entries(tokens)
            self._home_registered.pop(hslot, None)
            self._release_home(hslot)
            with self._home_free_cond:
                self._home_h2d_release.discard(release_ready)
            release_ready.set()

    def discard_block(self, block_idx: int) -> None:
        """Drop *block_idx* from the GPU ring without writing weights home.

        Only used at shutdown (lazy disk mode): disk is the source of truth
        and the pools are about to be cleared, so the write-back is waste.
        """
        self._events.pop(block_idx, None)
        self._block_gpu.pop(block_idx, None)

    # -- prefetch ------------------------------------------------------------

    def start_prefetch(self, block_idx: int, block: SwapBlock) -> None:
        if (block_idx in self._block_gpu or block_idx in self._events
                or block_idx in self._pin_stage):
            return
        had_home = block_idx in self._block_home
        hslot = self.ensure_home(block_idx, block, non_blocking=True)
        self._wait_home_slot(hslot)
        if not self.pin_memory:
            return
        if self._use_host_register and had_home:
            ready = threading.Event()
            fut = self._home_exec.submit(
                self._register_home_for_h2d, block_idx, hslot, ready, block)
            self._pin_stage[block_idx] = (hslot, ready, fut, block, "register")
        else:
            pidx = self._pin_acquire()
            if pidx is None:
                return
            template = {n: tpl for n, tpl in zip(block.names, block.templates)}
            self._pin.ensure(pidx, template)
            ready = threading.Event()
            fut = self._home_exec.submit(
                self._copy_home_to_pin, block_idx, hslot, pidx, ready, block)
            self._pin_stage[block_idx] = (pidx, ready, fut, block, "pin")

    def flush_prefetch(self) -> None:
        """Enqueue H2D for lookahead blocks whose staging has finished."""
        for block_idx in list(self._pin_stage.keys()):
            item = self._pin_stage.get(block_idx)
            if item is None or not item[1].is_set():
                continue
            self._complete_h2d(block_idx, wait_ready=False)

    def _register_home_for_h2d(self, block_idx: int, hslot: int,
                               ready: threading.Event,
                               block: SwapBlock) -> None:
        """Worker: temporarily pin the home slot used as an H2D source."""
        try:
            self._wait_home_slot(hslot)
            self._home.wait_home_fill(hslot)
            home = self._home_pool[hslot]
            if home is None:
                raise RuntimeError("blockswap: missing home slot")
            tokens = _register_cpu_entries(home)
            self._home_registered[hslot] = tokens
        except Exception as e:
            self._use_host_register = False
            logger.error("blockswap: cudaHostRegister for H2D failed for %s: %s",
                         block.name, e)
            raise
        finally:
            ready.set()
            self._complete_h2d(block_idx, block, wait_ready=False)

    def _copy_home_to_pin(self, block_idx: int, hslot: int, pidx: int,
                          ready: threading.Event, block: SwapBlock) -> None:
        """Worker: pageable home -> pinned H2D staging (off critical path)."""
        try:
            self._wait_home_slot(hslot)
            self._home.wait_home_fill(hslot)
            prev = self._pin_events.pop(pidx, None)
            if prev is not None:
                prev.synchronize()
            pin = self._pin.slots[pidx]
            home = self._home_pool[hslot]
            if pin is None or home is None:
                raise RuntimeError("blockswap: missing staging slot")
            with torch.inference_mode():
                for n, e in home.items():
                    pe = pin.get(n)
                    if pe is not None:
                        pe.copy_from(e)
        except Exception as e:
            logger.error("blockswap: home->pin staging failed for block %s: %s",
                         block.name, e)
            raise
        finally:
            ready.set()

    def _complete_h2d(self, block_idx: int, block: Optional[SwapBlock] = None,
                      wait_ready: bool = False) -> bool:
        """Enqueue an H2D for a staged/prefetched block.

        Called from either the main thread or a staging worker.  The transfer
        lock makes the ring-slot acquisition, CUDA stream enqueue, and state
        update atomic, so a worker can enqueue lookahead H2D as soon as its
        staging work is done.
        """
        fallback_load = False
        with self._transfer_lock:
            if block_idx in self._events or block_idx in self._block_gpu:
                return False
            item = self._pin_stage.get(block_idx)
            if item is None:
                return False
            slot_id, ready, fut, stored_block, mode = item
            if block is None:
                block = stored_block
            if not ready.is_set():
                if not wait_ready:
                    return False
                ready.wait()
            if fut.exception() is not None:
                self._pin_stage.pop(block_idx, None)
                if mode == "register":
                    self._use_host_register = False
                    self._home_registered.pop(slot_id, None)
                fallback_load = True
            elif mode == "register":
                hslot = slot_id
                home = self._home_pool[hslot]
                tokens = self._home_registered.get(hslot)
                if home is None or tokens is None:
                    raise RuntimeError("blockswap: missing registered home slot")
                self._pin_stage.pop(block_idx, None)
                ridx = self._ring_acquire()
                gpu = self._ensure_entries(self._gpu_pool, self._gpu_built,
                                           ridx, block, self.device)
                prev_d2h = self._gpu_d2h_events.get(ridx)
                if prev_d2h is not None:
                    self._stream.wait_event(prev_d2h)
                self._stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._stream):
                    for n, e in home.items():
                        ge = gpu.get(n)
                        if ge is None:
                            continue
                        ge.copy_from(e, non_blocking=True)
                compute = torch.cuda.current_stream()
                ev = torch.cuda.Event()
                ev.record(self._stream)
                compute.wait_event(ev)
                self._gpu_h2d_events[ridx] = ev
                release_ready = threading.Event()
                with self._home_free_cond:
                    self._home_h2d_release.add(release_ready)
                try:
                    self._cleanup_exec.submit(
                        self._unregister_home_after_h2d,
                        hslot, ev, tokens, release_ready)
                except Exception:
                    ev.synchronize()
                    _unregister_cpu_entries(tokens)
                    self._home_registered.pop(hslot, None)
                    self._release_home(hslot)
                    with self._home_free_cond:
                        self._home_h2d_release.discard(release_ready)
                    release_ready.set()
                self._events[block_idx] = ev
                self._assign(block, gpu)
                hslot_pop = self._block_home.pop(block_idx, None)
                if hslot_pop is not None and hslot_pop != hslot:
                    self._release_home(hslot_pop)
                self._block_gpu[block_idx] = ridx
            else:
                pidx = slot_id
                self._pin_stage.pop(block_idx, None)
                ridx = self._ring_acquire()
                gpu = self._ensure_entries(self._gpu_pool, self._gpu_built,
                                           ridx, block, self.device)
                prev_d2h = self._gpu_d2h_events.get(ridx)
                if prev_d2h is not None:
                    self._stream.wait_event(prev_d2h)
                pin = self._pin.slots[pidx]
                if pin is None:
                    raise RuntimeError("blockswap: missing pin stage")
                self._stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._stream):
                    for n in pin:
                        pe, ge = pin[n], gpu.get(n)
                        if ge is None:
                            continue
                        ge.copy_from(pe, non_blocking=True)
                compute = torch.cuda.current_stream()
                ev = torch.cuda.Event()
                ev.record(self._stream)
                compute.wait_event(ev)
                self._gpu_h2d_events[ridx] = ev
                self._pin_events[pidx] = ev
                self._events[block_idx] = ev
                self._assign(block, gpu)
                hslot = self._block_home.pop(block_idx, None)
                if hslot is not None:
                    self._release_home(hslot)
                self._block_gpu[block_idx] = ridx
        if fallback_load:
            self.load_block(block_idx, block, non_blocking=True)
            return True
        return True

    def sync_prefetch(self, block_idx: int) -> bool:
        """Wait for *block_idx*'s async H2D, enqueuing it if still pending."""
        event = self._events.pop(block_idx, None)
        if event is not None:
            torch.cuda.current_stream().wait_event(event)
            return True
        return self._complete_h2d(block_idx, wait_ready=True)

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

    def clear_pools(self) -> None:
        try:
            self._home_exec.shutdown(wait=True, timeout=30)
        except Exception:
            pass
        try:
            self._cleanup_exec.shutdown(wait=True, timeout=30)
        except Exception:
            pass
        for tokens in self._home_registered.values():
            try:
                _unregister_cpu_entries(tokens)
            except Exception:
                pass
        self._home_registered.clear()
        with self._home_free_cond:
            self._home_h2d_release.clear()
        with self._d2h_cleanup_lock:
            self._d2h_cleanup_pending.clear()
        nh, ng = len(self._home_pool), len(self._gpu_pool)
        self._home_pool = [None] * nh
        self._gpu_pool = [None] * ng
        self._pin = PinStage.new(self._pin.size)
        self._d2h_stage = PinStage.new(self._d2h_stage.size)
        self._stage_inuse = [threading.Event() for _ in range(self._d2h_stage.size)]
        for _e in self._stage_inuse:
            _e.set()
        self._home_ready = {}
        self._pin_events = {}
        self._pin_stage.clear()
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
# _VRAMBudget - CUDA cache defrag
# ---------------------------------------------------------------------------

class _VRAMBudget:
    def __init__(self, block_mb: float):
        self.block_mb = block_mb

    def free_mb(self, device=None) -> float:
        if not torch.cuda.is_available():
            return float("inf")
        try:
            free, _ = torch.cuda.mem_get_info(device)
            return free / (1024 * 1024)
        except Exception:
            return float("inf")

    def maybe_flush(self, device=None, reserve_blocks: int = 2) -> None:
        if not torch.cuda.is_available():
            return
        needed_mb = self.block_mb * max(1, reserve_blocks)
        if self.free_mb(device) < needed_mb:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# _DiskPrefetcher - background safetensors -> block-param loader
# ---------------------------------------------------------------------------

class _DiskPrefetcher:
    """Reads blocks from disk into the block's own CPU parameters.

    The worker never touches swap slots, so a slot can never be recycled
    under an in-flight read.  ``_loaded`` tracks blocks whose params are
    valid (fresh from disk or rebound to a home slot), making
    ``ensure_ram`` a no-op for warm blocks.
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

    # -- public API ----------------------------------------------------------

    def ensure_ram(self, block_idx: int) -> None:
        """Guarantee *block_idx*'s params are in CPU RAM (block until ready)."""
        with self._lock:
            if block_idx in self._loaded:
                return
            fut = self._pending.pop(block_idx, None)
        if fut is not None:
            fut.result()  # wait for the disk read (re-raises worker errors)
        else:
            self._load_immediate(block_idx)
        with self._lock:
            self._loaded.add(block_idx)

    def start_ram_load(self, block_idx: int) -> None:
        """Disk-to-RAM prefetch is not needed; home slots read directly."""
        return

    def start_home_fill(self, block_idx: int, hslot: int, slot: dict,
                        block: SwapBlock) -> bool:
        """Fill *hslot* from disk on a worker thread.

        Returns False if *hslot* already has an in-flight fill.  The caller
        uses ``wait_home_fill(hslot)`` before reading the slot, so the home
        slot is never observed in a partially written state.
        """
        with self._lock:
            if hslot in self._home_pending:
                return False
            ready = threading.Event()
            fut = self._executor.submit(
                self._home_fill_worker, block_idx, hslot, slot, block, ready)
            self._home_pending[hslot] = (block_idx, fut, ready)
            return True

    def wait_home_fill(self, hslot: int) -> None:
        """Wait for any in-flight disk fill targeting *hslot*."""
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
        """Fill *slot* directly from disk; no block-params copy remains."""
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
        """Free *block_idx*'s CPU RAM (disk remains the source of truth)."""
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

    # -- internal ------------------------------------------------------------

    def _on_load_done(self, fut: Future) -> None:
        if fut.exception() is not None:
            return
        with self._lock:
            for bi in list(self._pending):
                if self._pending.get(bi) is fut:
                    self._loaded.add(bi)
                    break

    def _load_immediate(self, block_idx: int) -> None:
        """Worker: read block weights from disk into the block's params."""
        block = self._blocks[block_idx]
        tensors = self._reader.get_tensors(block.keys)
        for full, tpl, (mod, leaf, kind) in zip(block.keys, block.templates, block.refs):
            t = tensors.get(full)
            if t is None:
                continue
            if kind != "param":
                mod._buffers[leaf] = t.to(tpl.data.dtype)
            elif tpl.is_qt:
                # The template (built from the checkpoint's quant metadata at
                # build_dit time) already carries layout/scale/extras; only
                # the qdata streams in here.
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


# ---------------------------------------------------------------------------
# BlockSwapManager - sliding-window orchestration
# ---------------------------------------------------------------------------

class BlockSwapManager:
    def __init__(self, blocks: list[SwapBlock], reader: BlockReader, device,
                 window_size: int = 2, prefetch: bool = True,
                 prefetch_count: int = 2, pin_memory: bool = True,
                 disk_workers: int = 2, hot_blocks: int = 0,
                 dtype=torch.bfloat16):
        self.blocks = blocks
        self.total = len(blocks)
        self.device = torch.device(device)
        self.dtype = dtype
        self.prefetch = bool(prefetch)
        self.window_size = max(1, min(window_size, self.total))
        self.prefetch_count = max(1, prefetch_count)
        self.swap_hits = 0
        self.swap_loads = 0
        self.home_size = max(0, self.total - self.window_size)
        self._flushed_this_step = False

        self.hot_blocks = max(0, min(hot_blocks, self.total))
        self._window = _BlockWindow(
            self.total, self.window_size, hot_blocks=self.hot_blocks)
        self.hot_blocks = len(self._window.hot)
        self._disk = _DiskPrefetcher(reader, blocks, max_workers=disk_workers)
        self._xfer = _TransferEngine(
            device, prefetch, prefetch_count, pin_memory,
            n_home_slots=max(self.home_size, 1),
            n_gpu_slots=self.window_size + self.prefetch_count,
            home=self._disk,
        )
        block_mb = (blocks[0].bytes_per_block() / 2 ** 20) if blocks else 0.0
        self._budget = _VRAMBudget(block_mb)
        logger.info("%d blocks, window=%d, ~%.0f MB/block, prefetch=%s x%d, pin=%s",
                    self.total, self.window_size, block_mb,
                    self.prefetch, self.prefetch_count, pin_memory)

    # -- public API ----------------------------------------------------------

    def apply_lora(self, block_idx: int, entries: list) -> None:
        self.blocks[block_idx].lora = list(entries)

    def clear_lora(self, block_idx: int) -> None:
        self.blocks[block_idx].lora = None

    def begin(self) -> None:
        self._xfer.cancel_all()
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

        n = max(len(to_offload), len(to_load_now))
        for k in range(n):
            home_freed = False
            if k < len(to_load_now):
                j = to_load_now[k]
                home_before = self._xfer.home_free_count()
                if not self._xfer.sync_prefetch(j):
                    self._xfer.load_block(j, self.blocks[j], non_blocking=True)
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
            self._xfer.start_prefetch(i, self.blocks[i])

    def end(self) -> None:
        if self._xfer._stream is not None:
            self._xfer._stream.synchronize()
        if self._xfer._d2h_stream is not None:
            self._xfer._d2h_stream.synchronize()
        self._xfer.wait_d2h_cleanup()

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

    def stats(self) -> dict:
        return {"hits": self.swap_hits, "loads": self.swap_loads,
                "home_size": self.home_size, "total": self.total,
                "window": self.window_size, "hot": self.hot_blocks,
                "pin": self._xfer._pin.size, "stage": 0,
                "disk_reads": self._disk.disk_reads}

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
