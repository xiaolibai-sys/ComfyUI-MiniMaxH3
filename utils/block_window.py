"""Pure residency and VRAM budget helpers for BlockSwap."""

from __future__ import annotations

import torch


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
        return set(
            range(block_idx, min(self.total, block_idx + self.window_size))
        ) | self.hot

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
