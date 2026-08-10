"""Explicit async transfer scheduler for BlockSwap.

The scheduler keeps the compute stream free of implicit transfer waits.  H2D
and D2H run on dedicated streams, and every dependency is expressed with
``torch.cuda.Event``.  Callers must:

1. ``schedule_h2d`` / ``schedule_d2h`` to enqueue a transfer;
2. ``wait_event`` on the compute stream only when the block is about to be
   consumed;
3. ``record_compute`` immediately after the block forward.

The scheduler never calls ``synchronize`` in the enqueue path.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import torch

from .types import SlotEntry


class TransferScheduler:
    def __init__(self, device, max_workers: int = 2):
        self.device = torch.device(device)
        self.max_workers = max(1, max_workers)
        self._h2d_stream = torch.cuda.Stream(device=self.device)
        self._d2h_stream = torch.cuda.Stream(device=self.device)
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="h3sched")
        self._lock = threading.RLock()
        self._events: dict[tuple[int, str], torch.cuda.Event] = {}
        self._slot_events: dict[int, dict[str, torch.cuda.Event]] = {}
        self._closed = False

    # -- public API ----------------------------------------------------------

    def schedule_h2d(self, slot_id: int, src: torch.Tensor,
                     dst: torch.Tensor,
                     wait_events: tuple[torch.cuda.Event, ...] = ()) -> None:
        self._submit(slot_id, "h2d", src, dst, self._h2d_stream, wait_events)

    def schedule_d2h(self, slot_id: int, src: torch.Tensor,
                     dst: torch.Tensor,
                     wait_events: tuple[torch.cuda.Event, ...] = ()) -> None:
        self._submit(slot_id, "d2h", src, dst, self._d2h_stream, wait_events)

    def schedule_h2d_entries(self, slot_id: int, src: dict,
                             dst: dict,
                             wait_events: tuple[torch.cuda.Event, ...] = ()) -> None:
        self._submit_entries(
            slot_id, "h2d", src, dst, self._h2d_stream, wait_events)

    def schedule_d2h_entries(self, slot_id: int, src: dict,
                             dst: dict,
                             wait_events: tuple[torch.cuda.Event, ...] = ()) -> None:
        self._submit_entries(
            slot_id, "d2h", src, dst, self._d2h_stream, wait_events)

    def record_compute(self, block_id: int) -> torch.cuda.Event:
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream())
        with self._lock:
            self._events[(block_id, "compute")] = ev
        return ev

    def event(self, block_id: int, kind: str) -> Optional[torch.cuda.Event]:
        with self._lock:
            return self._events.get((block_id, kind))

    def slot_event(self, slot_id: int, kind: str) -> Optional[torch.cuda.Event]:
        with self._lock:
            slot = self._slot_events.get(slot_id)
            return None if slot is None else slot.get(kind)

    def wait_on_compute(self, event: torch.cuda.Event) -> None:
        torch.cuda.current_stream().wait_event(event)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True)

    # -- internal ------------------------------------------------------------

    def _submit(self, slot_id: int, kind: str, src: torch.Tensor,
                dst: torch.Tensor, stream: torch.cuda.Stream,
                wait_events: tuple[torch.cuda.Event, ...]) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("TransferScheduler is closed")
        ev = torch.cuda.Event()
        with self._lock:
            self._events[(slot_id, kind)] = ev
            self._slot_events.setdefault(slot_id, {})[kind] = ev

        def run() -> None:
            with torch.cuda.stream(stream):
                for wait_ev in wait_events:
                    stream.wait_event(wait_ev)
                host_pinned = (
                    (src.device.type == "cpu" and src.is_pinned())
                    or (dst.device.type == "cpu" and dst.is_pinned())
                )
                dst.copy_(src, non_blocking=host_pinned)
            ev.record(stream)

        self._executor.submit(run)

    def _submit_entries(self, slot_id: int, kind: str, src: dict,
                        dst: dict, stream: torch.cuda.Stream,
                        wait_events: tuple[torch.cuda.Event, ...]) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("TransferScheduler is closed")
        ev = torch.cuda.Event()
        with self._lock:
            self._events[(slot_id, kind)] = ev
            self._slot_events.setdefault(slot_id, {})[kind] = ev

        def run() -> None:
            with torch.cuda.stream(stream):
                for wait_ev in wait_events:
                    stream.wait_event(wait_ev)
                host_pinned = False
                for name, src_entry in src.items():
                    dst_entry = dst.get(name)
                    if dst_entry is None:
                        continue
                    src_t = src_entry.data
                    dst_t = dst_entry.data
                    host_pinned = host_pinned or (
                        (src_t.device.type == "cpu" and src_t.is_pinned())
                        or (dst_t.device.type == "cpu" and dst_t.is_pinned())
                    )
                    dst_entry.copy_from(src_entry, non_blocking=host_pinned)
            ev.record(stream)

        self._executor.submit(run)
