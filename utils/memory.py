"""Memory coordination between DIT BlockSwap and VAE during rolling."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import replace
from typing import Iterator

import torch

from .lifecycle import collect_garbage

logger = logging.getLogger("h3.memory")


class SamplingMemory:
    """Owns VAE/BlockSwap residency transitions around rolling segments."""

    def __init__(self, session):
        self.session = session

    @property
    def offload_dit(self) -> bool:
        return bool(getattr(
            getattr(self.session.assets, "fl_constraint", None),
            "offload_dit",
            False,
        ))

    @contextmanager
    def vae_phase(self, reload_after: bool = True) -> Iterator[None]:
        """Run VAE work with model memory released, then restore.

        ``reload_after=False`` skips the model restore for the final VAE
        phase of a rolling run: nothing samples afterwards and
        ``session.finish()`` tears the model down regardless, so reloading
        would be pure waste.
        """
        if self.offload_dit:
            logger.info("blockswap: unloading full model before rolling VAE phase")
            self._unload_model()
        else:
            logger.info("blockswap: swapping DIT to RAM before rolling VAE phase")
            self._offload_dit_blocks()
        if self.session.vae is not None:
            self.session.vae.to(self.session.context.device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        try:
            yield
        finally:
            if self.session.vae is not None:
                self.session.vae.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            collect_garbage()
            if not reload_after:
                logger.info("blockswap: model stays unloaded after final VAE phase")
            elif self.offload_dit:
                logger.info("blockswap: reloading full model after rolling VAE phase")
                self._restore_model()
            else:
                logger.info("blockswap: restoring DIT after rolling VAE phase")
                self._restore_dit_blocks()

    def encode_images_phase(self, images, width: int, height: int):
        """Encode keyframe images in one shared VAE residency window.

        Per-image phases each cost a full model unload/reload when
        ``offload_dit`` is on, so callers should batch every image.
        """
        if not images:
            return []
        if self.offload_dit:
            with self.vae_phase():
                return [
                    _encode_image(self.session.vae, image, width, height)
                    for image in images
                ]
        return [
            _encode_image(self.session.vae, image, width, height)
            for image in images
        ]

    def encode_image_phase(self, image, width: int, height: int):
        return self.encode_images_phase([image], width, height)[0]

    def decode_and_encode_boundary(self, result, need_start: bool = True,
                                   reload_after: bool = True):
        if self.offload_dit:
            with self.vae_phase(reload_after=reload_after):
                return self.decode_and_encode(result, need_start=need_start)
        return self.decode_and_encode(result, need_start=need_start)

    def decode_and_encode(self, result, need_start: bool = True):
        decoded_video = self.session.vae.decode_video_streaming(result.video)
        decoded_audio = self.session.vae.decode_audio(result.audio)
        prev_start_latent = (
            self.session.vae.encode_video(decoded_video[:, :, -1:, :, :])
            if need_start
            else None
        )
        return decoded_video, decoded_audio, prev_start_latent

    def _offload_dit_blocks(self) -> None:
        if self.session.model is None:
            return
        mgr = getattr(self.session.model, "_swap_mgr", None)
        if mgr is not None:
            mgr.offload_all()

    def _unload_model(self) -> None:
        handle = self.session.assets.handle
        handle.unload()
        self.session.model = None
        collect_garbage()

    def _restore_model(self) -> None:
        handle = self.session.assets.handle
        swap = self.session.assets.runtime.swap or handle.swap
        swap = replace(swap, offload_dit=True)
        model = handle.load(
            swap_config=swap,
            include_adaln=not self.session.config.use_adaln_cache,
            vram_spec=self.session.assets.vram_spec,
        )
        self.session.model = model
        block_stats = None
        mgr = getattr(model, "_swap_mgr", None)
        if mgr is not None:
            block_stats = mgr.stats()
        self.session.context = replace(
            self.session.context,
            model=model,
            reader=None,
            block_stats=block_stats,
        )

    def _restore_dit_blocks(self) -> None:
        if self.session.model is None:
            return
        mgr = getattr(self.session.model, "_swap_mgr", None)
        if mgr is not None:
            mgr.restore_initial()


def _encode_image(vae, image: torch.Tensor, width: int, height: int):
    import comfy.utils

    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(
        samples, width, height, "lanczos", "disabled")
    pixels = (samples * 2 - 1).to(torch.float16)
    return vae.encode_video(pixels)


__all__ = ["SamplingMemory"]
