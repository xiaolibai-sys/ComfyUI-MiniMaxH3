"""Static VRAM planning for BlockSwap pool allocation."""

from __future__ import annotations

from dataclasses import replace

import torch

from .types import (
    H3BlockSwap,
    H3LoraSet,
    PoolPlan,
    SequenceSpec,
    SwapAllocation,
    VRAMEstimate,
)
from .vram_models import (
    COMFY_RESERVE_MB,
    PERF_HEADROOM_MB,
    estimate_runtime_lora_mb,
    full_forward_activation_mb,
    sequence_token_count,
)


class VRAMPlanner:
    """Convert typed sequence specs into final BlockSwap allocations."""

    def __init__(
        self,
        backend: str,
        device,
        loras: H3LoraSet | None = None,
    ):
        self.backend = backend
        self.device = torch.device(device)
        self.loras = loras

    def measure_free_mb(self) -> float:
        if not torch.cuda.is_available():
            return float("inf")
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            free, _ = torch.cuda.mem_get_info(self.device)
            return free / (1024 * 1024)
        except Exception:
            return float("inf")

    def estimate(self, spec: SequenceSpec) -> VRAMEstimate:
        tokens = sequence_token_count(spec)
        activation_mb = full_forward_activation_mb(
            self.backend, tokens, cfg=spec.cfg)
        runtime_total, runtime_fixed = estimate_runtime_lora_mb(
            self.loras or H3LoraSet())
        return VRAMEstimate(
            sequence_tokens=tokens,
            activation_mb=activation_mb,
            comfy_reserve_mb=COMFY_RESERVE_MB,
            perf_headroom_mb=PERF_HEADROOM_MB,
            runtime_lora_total_mb=runtime_total,
            runtime_lora_fixed_mb=runtime_fixed,
        )

    def plan(
        self,
        swap: H3BlockSwap,
        block_mb: float,
        spec: SequenceSpec,
        total_blocks: int,
    ) -> SwapAllocation:
        assert block_mb > 0, "block_mb must be > 0"
        assert total_blocks > 0, "total_blocks must be > 0"
        assert spec.text_len > 0, "text_len must be > 0"
        assert spec.latent_t > 0, "latent_t must be > 0"
        assert spec.latent_h > 0 and spec.latent_w > 0, "latent H/W must be > 0"
        assert spec.audio_t > 0, "audio_t must be > 0"
        assert 0 <= swap.block_to_swap <= total_blocks, "block_to_swap out of range"
        requested_window = swap.window_size(total_blocks)
        requested_slots = requested_window + swap.prefetch_count
        home_slots = (
            min(total_blocks, max(1, requested_window))
            if swap.offload_dit
            else max(0, total_blocks - requested_window)
        )

        if not swap.auto_vram and swap.vram_reserve_mb <= 0:
            pool = PoolPlan(
                block_mb=block_mb,
                free_mb=self.measure_free_mb(),
                effective_reserve_mb=0.0,
                lora_per_slot_mb=0.0,
                max_slots=requested_slots,
                requested_slots=requested_slots,
                window_size=requested_window,
                hot_blocks=min(swap.hot_blocks, requested_window - 1),
                prefetch_count=swap.prefetch_count,
                home_slots=home_slots,
                gpu_slots=requested_slots,
            )
            return SwapAllocation(config=swap, pool=pool)

        estimate = self.estimate(spec)
        free_mb = self.measure_free_mb()
        effective_reserve = (
            estimate.reserve_mb
            if swap.auto_vram
            else swap.vram_reserve_mb
        ) + estimate.runtime_lora_fixed_mb
        lora_total = (
            estimate.runtime_lora_total_mb
            if swap.auto_vram
            else swap.runtime_lora_total_mb
        )
        lora_per_slot = lora_total / max(1, total_blocks)
        max_slots = max(
            1,
            int((free_mb - effective_reserve)
                // (block_mb + lora_per_slot)),
        )

        window = requested_window
        prefetch_count = swap.prefetch_count
        hot_blocks = swap.hot_blocks
        if requested_slots > max_slots:
            prefetch_count = min(prefetch_count, max(0, max_slots - 1))
            window = max(
                1,
                min(window, max_slots - prefetch_count),
            )
            hot_blocks = min(hot_blocks, max(0, window - 1))
        elif swap.offload_dit and swap.auto_vram:
            # VAE is kept off GPU during DiT sampling, so the freed VRAM can
            # enlarge the requested window instead of being reserved for VAE.
            window = min(
                total_blocks,
                max(window, max(1, max_slots - prefetch_count)),
            )
            hot_blocks = min(hot_blocks, max(0, window - 1))

        pool = PoolPlan(
            block_mb=block_mb,
            free_mb=free_mb,
            effective_reserve_mb=effective_reserve,
            lora_per_slot_mb=lora_per_slot,
            max_slots=max_slots,
            requested_slots=requested_slots,
            window_size=window,
            hot_blocks=hot_blocks,
            prefetch_count=prefetch_count,
            home_slots=(
                min(total_blocks, max(1, window))
                if swap.offload_dit
                else max(0, total_blocks - window)
            ),
            gpu_slots=window + prefetch_count,
        )
        config = replace(
            swap,
            vram_reserve_mb=(
                estimate.reserve_mb if swap.auto_vram else swap.vram_reserve_mb
            ),
            runtime_lora_total_mb=lora_total,
            runtime_lora_fixed_mb=estimate.runtime_lora_fixed_mb,
        )
        return SwapAllocation(config=config, pool=pool)


__all__ = ["VRAMPlanner"]
