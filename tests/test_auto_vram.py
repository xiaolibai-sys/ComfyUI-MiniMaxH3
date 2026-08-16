"""Static VRAM auto-planning switch regression test."""

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
        latent_h=8,
        latent_w=8,
        audio_t=8,
        media=MediaConditioning(),
    )


def test_manual_plan_is_preserved_when_auto_disabled():
    swap = H3BlockSwap(
        block_to_swap=30,
        hot_blocks=5,
        prefetch_count=4,
        auto_vram=False,
    )
    planner = VRAMPlanner("sageattn2", "cpu")
    planner.measure_free_mb = lambda: 0.0
    out = planner.plan(swap, 371.0, _spec(), 50)
    assert out.config is swap
    assert out.pool.window_size == 20
    assert out.pool.hot_blocks == 5
    assert out.pool.prefetch_count == 4
    assert out.pool.effective_reserve_mb == 0.0


def test_auto_plan_is_enabled_by_default():
    swap = H3BlockSwap(block_to_swap=30, hot_blocks=5, prefetch_count=4)
    planner = VRAMPlanner("sageattn2", "cpu")
    planner.measure_free_mb = lambda: 9000.0
    out = planner.plan(swap, 371.0, _spec(), 50)
    assert out.config.auto_vram is True
    assert out.config.vram_reserve_mb > 0.0
    assert out.pool.window_size <= 20


def test_offload_dit_uses_ram_small_home_pool():
    swap = H3BlockSwap(
        block_to_swap=30,
        hot_blocks=5,
        prefetch_count=4,
        offload_dit=True,
    )
    planner = VRAMPlanner("sageattn2", "cpu")
    planner.measure_free_mb = lambda: 9000.0
    out = planner.plan(swap, 371.0, _spec(), 50)
    assert out.pool.home_slots == min(50, max(1, out.pool.window_size))
    assert out.pool.home_slots <= out.pool.window_size
    assert out.config.offload_dit is True


def test_offload_dit_expands_window_to_freed_vram():
    swap = H3BlockSwap(
        block_to_swap=47,
        prefetch_count=1,
        offload_dit=True,
    )
    planner = VRAMPlanner("sageattn2", "cpu")
    planner.measure_free_mb = lambda: 15000.0
    out = planner.plan(swap, 371.0, _spec(), 50)
    assert out.pool.window_size > 3
    assert out.pool.home_slots == out.pool.window_size


if __name__ == "__main__":
    test_manual_plan_is_preserved_when_auto_disabled()
    test_auto_plan_is_enabled_by_default()
    test_offload_dit_uses_ram_small_home_pool()
    test_offload_dit_expands_window_to_freed_vram()
    print("auto_vram tests passed")
