"""VRAMPlanner / PoolPlan contract tests."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pkg_loader import load as _load_h3rt
_load_h3rt()

from h3rt.utils.types import H3BlockSwap, MediaConditioning, SequenceSpec
from h3rt.utils.vram_planner import VRAMPlanner


def _spec():
    return SequenceSpec(
        text_len=16,
        latent_t=2,
        latent_h=4,
        latent_w=4,
        audio_t=8,
        media=MediaConditioning(),
        cfg=1.0,
    )


def test_manual_mode_preserves_request():
    swap = H3BlockSwap(
        block_to_swap=30,
        hot_blocks=2,
        prefetch_count=2,
        auto_vram=False,
    )
    planner = VRAMPlanner("sageattn2", "cpu")
    planner.measure_free_mb = lambda: 0.0
    allocation = planner.plan(swap, 371.0, _spec(), 50)
    assert allocation.config is swap
    assert allocation.pool.window_size == 20
    assert allocation.pool.hot_blocks == 2
    assert allocation.pool.prefetch_count == 2


def test_auto_mode_reduces_prefetch_before_window():
    swap = H3BlockSwap(
        block_to_swap=30,
        hot_blocks=2,
        prefetch_count=2,
        auto_vram=True,
    )
    planner = VRAMPlanner("sageattn2", "cpu")
    planner.measure_free_mb = lambda: 9000.0
    allocation = planner.plan(swap, 371.0, _spec(), 50)
    pool = allocation.pool
    assert pool.max_slots >= 1
    assert pool.window_size <= 20
    assert pool.prefetch_count <= 2
    assert pool.home_slots == 50 - pool.window_size


if __name__ == "__main__":
    test_manual_mode_preserves_request()
    test_auto_mode_reduces_prefetch_before_window()
    print("vram planner tests passed")
