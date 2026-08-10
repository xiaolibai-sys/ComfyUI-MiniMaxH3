"""Minimal TransferScheduler lifecycle test: H2D -> compute -> D2H."""

import os
import sys
import time

sys.path.insert(0, os.path.abspath("tests"))
sys.path.insert(0, os.path.abspath("."))

from pkg_loader import load as _load_h3rt
_load_h3rt()

import torch

from h3rt.utils.transfer_scheduler import TransferScheduler


def _wait_event(ev, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ev.query():
            return True
        time.sleep(0.01)
    return False


def main():
    torch.cuda.synchronize()
    sched = TransferScheduler("cuda", max_workers=2)
    cpu = torch.randn(1 << 20, device="cpu", pin_memory=True)
    gpu = torch.empty_like(cpu, device="cuda")
    cpu_out = torch.empty_like(cpu, pin_memory=True)

    sched.schedule_h2d(0, cpu, gpu)
    assert _wait_event(sched.slot_event(0, "h2d")), "H2D event missing"
    sched.wait_on_compute(sched.slot_event(0, "h2d"))
    gpu_computed = gpu * 2.0
    compute_ev = sched.record_compute(0)

    sched.schedule_d2h(
        0, gpu_computed, cpu_out, wait_events=(compute_ev,))
    assert _wait_event(sched.slot_event(0, "d2h")), "D2H event missing"
    torch.cuda.synchronize()
    assert torch.allclose(cpu_out, cpu * 2.0, atol=1e-5), "copy mismatch"
    print("TRANSFER SCHEDULER OK", flush=True)
    sched.close()


if __name__ == "__main__":
    main()
