"""Native dual-clock sampler fallback mapping tests."""

import os
import sys

sys.path.insert(0, r"D:\ComfyUI-installs\ComfyUI\ComfyUI")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pkg_loader import load as _load_h3rt
_load_h3rt()

from h3rt.utils.native_samplers import (
    H3_SAMPLER_FUNCTIONS,
    _dual_euler,
    _h3_sampler,
)


def test_supported_mapping():
    assert _h3_sampler("euler") is H3_SAMPLER_FUNCTIONS["euler"]
    assert _h3_sampler("euler_ancestral") is H3_SAMPLER_FUNCTIONS["euler_ancestral"]
    assert _h3_sampler("heun") is H3_SAMPLER_FUNCTIONS["heun"]
    assert _h3_sampler("dpmpp_2m") is H3_SAMPLER_FUNCTIONS["dpmpp_2m"]


def test_unknown_sampler_falls_back_to_euler():
    assert _h3_sampler("not_a_real_sampler") is _dual_euler


if __name__ == "__main__":
    test_supported_mapping()
    test_unknown_sampler_falls_back_to_euler()
    print("native sampler tests passed")
