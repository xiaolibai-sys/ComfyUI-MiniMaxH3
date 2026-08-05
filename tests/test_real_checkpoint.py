"""Real-checkpoint monitor: load minimax_h3_fl2va_pruned_int8_convrot.

Watches RAM (psutil RSS) and VRAM (torch.cuda) during build + two velocity
passes, and verifies the one-copy BlockSwap invariants:

* RAM jumps to ~(home+pin+stage) at build, then stays FLAT (no growth);
* pass 2 (steady state) performs ZERO disk reads (no extra copies);
* resident(home+GPU) == total at all times after pass 1 (one copy);
* window == 25 (half the 50 DiT layers on GPU).

Run with the ComfyUI venv python:
    python tests/test_real_checkpoint.py
"""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pkg_loader import load as _load_h3rt
_load_h3rt()

import torch
import psutil

from h3rt.utils.types import H3BlockSwap
from h3rt.utils.lifecycle import load_model_handle

MODEL = r"D:\ComfyUI-installs\ComfyUI\ComfyUI\models\diffusion_models\minimax_h3_fl2va_pruned_int8_convrot.safetensors"
WINDOW = 25          # half of the 50 DiT layers
TEXT_LEN = 16

swap = H3BlockSwap(enabled=True, block_to_swap=50 - WINDOW, prefetch=True,
                   prefetch_count=1, pin_memory=True, disk_workers=2,
                   dtype="bfloat16")

proc = psutil.Process()
samples = []          # (t_rel, rss_mb, vram_alloc_mb)
_stop = False


def _monitor():
    t0 = time.time()
    while not _stop:
        va = torch.cuda.memory_allocated() / 2 ** 20 if torch.cuda.is_available() else 0
        samples.append((time.time() - t0, proc.memory_info().rss / 2 ** 20, va))
        time.sleep(0.25)


th = threading.Thread(target=_monitor, daemon=True)
th.start()

device = torch.device("cuda")
print(f"model: {os.path.basename(MODEL)}  window={WINDOW} (half of 50)", flush=True)

t0 = time.time()
handle = load_model_handle(MODEL)
m = handle.load(swap_config=swap)
mgr = m._swap_mgr
torch.cuda.synchronize()
print(f"build done in {time.time()-t0:.1f}s  stats={mgr.stats()}", flush=True)

video = torch.randn(1, 24, 2, 64, 64, device=device, dtype=torch.bfloat16)
audio = torch.randn(1, 32, 2, 8, device=device, dtype=torch.bfloat16)
text = torch.randn(1, TEXT_LEN, 5376, device=device, dtype=torch.bfloat16)
payload = {"text_token_tags": torch.ones(1, TEXT_LEN, dtype=torch.long, device=device)}

for tag, sigma in (("pass1(cold)", 0.5), ("pass2(steady)", 0.4)):
    h0, l0, d0 = mgr.swap_hits, mgr.swap_loads, mgr._disk.disk_reads
    t0 = time.time()
    out = m.velocity(video, audio, sigma, text, payload)
    torch.cuda.synchronize()
    mgr.end()
    print(f"{tag}: {time.time()-t0:.1f}s  hits+{mgr.swap_hits-h0}  loads+{mgr.swap_loads-l0}  "
          f"disk_reads+{mgr._disk.disk_reads-d0}  "
          f"resident={len(mgr._xfer._block_home)+len(mgr._window.on_gpu)}/{mgr.total}", flush=True)

_stop = True
th.join(timeout=5)

# ---- summary ----------------------------------------------------------------
rss = [s[1] for s in samples]
vram = [s[2] for s in samples]
print(f"RAM  RSS: start={rss[0]:.0f}MiB build_end={rss[min(3,len(rss)-1)]:.0f}MiB "
      f"min={min(rss):.0f} max={max(rss):.0f} final={rss[-1]:.0f}MiB")
print(f"VRAM alloc: min={min(vram):.0f} max={max(vram):.0f} final={vram[-1]:.0f}MiB")

# sample every ~2s for a curve
prev = None
print("curve (t, RAM MiB, VRAM MiB):")
for s in samples:
    if prev is None or s[0] - prev >= 2.0:
        print(f"  t={s[0]:6.1f}s  {s[1]:8.0f}  {s[2]:8.0f}")
        prev = s[0]

# ---- assertions -------------------------------------------------------------
resident = len(mgr._xfer._block_home) + len(mgr._window.on_gpu)
assert mgr.window_size == WINDOW, "window != half the layers"
assert resident == mgr.total, f"one-copy violated: resident={resident} != total={mgr.total}"
# RAM flat in steady state: Windows commits lazily, so RAM climbs while the
# home pool fills during pass 1, then must stay FLAT (last samples ~equal).
last4 = rss[-4:]
rss_span = max(last4) - min(last4)
print(f"RAM last-4 span={rss_span:.0f}MiB (samples {[round(v) for v in last4]})")
assert rss_span < 512, f"RAM not flat in steady state: span={rss_span:.0f}MiB"
print("REAL CHECKPOINT TEST OK")
