"""Static VRAM auto-planning switch regression test."""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pkg_loader import load as _load_h3rt
_load_h3rt()

from h3rt.utils.types import H3BlockSwap
from h3rt.utils.vram_models import make_static_reserved_swap


def _latent():
    return SimpleNamespace(
        video=SimpleNamespace(shape=(1, 24, 2, 64, 64)),
        audio=SimpleNamespace(shape=(1, 32, 2, 8)),
    )


def test_manual_plan_is_preserved_when_auto_disabled():
    swap = H3BlockSwap(
        block_to_swap=30,
        hot_blocks=5,
        prefetch_count=4,
        auto_vram=False,
    )
    out = make_static_reserved_swap(
        swap, "sageattn2", 16, _latent(), {})
    assert out is swap
    assert out.block_to_swap == 30
    assert out.hot_blocks == 5
    assert out.prefetch_count == 4
    assert out.vram_reserve_mb == 0.0
    assert out.runtime_lora_total_mb == 0.0


def test_auto_plan_is_enabled_by_default():
    swap = H3BlockSwap(block_to_swap=30, hot_blocks=5, prefetch_count=4)
    out = make_static_reserved_swap(
        swap, "sageattn2", 16, _latent(), {})
    assert out.auto_vram is True
    assert out.vram_reserve_mb > 0.0
    assert out.block_to_swap == 30
    assert out.hot_blocks == 5
    assert out.prefetch_count == 4


if __name__ == "__main__":
    test_manual_plan_is_preserved_when_auto_disabled()
    test_auto_plan_is_enabled_by_default()
    print("auto_vram tests passed")
