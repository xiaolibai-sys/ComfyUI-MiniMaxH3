"""Sampling session lifecycle: model/VAE/prebake/text resources."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional

import torch
import comfy.model_sampling
import comfy.samplers

from ..models import adaln
from ..models.vae import load_vae_pack, unload_all_vaes
from .config import MiniMaxH3DiTConfig
from .encoder_use import unload_all_encoders
from .lifecycle import (
    collect_garbage,
    detect_key_prefix,
    scan_dit_config,
)
from .native_samplers import _h3_sampler
from .stream import BlockReader
from .types import (
    AdaLNCache,
    SamplingAssets,
    SamplingConfig,
    SessionContext,
    TextConditioning,
)
from .wrapper import _H3ModelSampling, _SigmaCaptureModel


@dataclass
class H3Session:
    """Owns all sampling resources and produces a reusable SessionContext."""

    assets: SamplingAssets
    config: SamplingConfig

    model: Optional[Any] = None
    reader: Optional[BlockReader] = None
    vae: Optional[Any] = None
    adaln_cache: Optional[AdaLNCache] = None
    teacache: Optional[Any] = None
    context: Optional[SessionContext] = None
    prebake_seconds: float = 0.0

    def prepare(self, sigmas: torch.Tensor) -> SessionContext:
        import time

        unload_all_vaes()
        unload_all_encoders()
        collect_garbage()

        handle = self.assets.handle
        device = handle.load_device
        dtype = (self.assets.runtime.swap or handle.swap).torch_dtype
        cfg = self.config
        conditioning = self.assets.positive
        negative = self.assets.negative
        latent = self.assets.latent
        if latent is None:
            raise ValueError("SamplingAssets.latent is required.")
        assert latent.video.ndim == 5, "video latent must be [B,C,T,H,W]"
        assert latent.audio.ndim == 4, "audio latent must be [B,C,Ch,T]"
        assert latent.video.shape[0] == 1, "batch size must be 1"
        assert latent.audio.shape[0] == 1, "audio batch size must be 1"
        assert conditioning.text.states.ndim == 3, "text states must be [B,L,D]"
        assert (
            conditioning.text.states.shape[0] == 1
        ), "text batch size must be 1"

        from ..models.model import PackedLayout

        text_len = conditioning.text.states.shape[1]
        latent_t, lat_h, lat_w = (
            latent.video.shape[2],
            latent.video.shape[3],
            latent.video.shape[4],
        )
        audio_t = latent.audio.shape[-1]
        payload = conditioning.to_payload()
        layout = PackedLayout(
            text_len,
            latent_t,
            lat_h,
            lat_w,
            audio_t,
            keyframes=payload.get("keyframes"),
            refs=payload.get("refs"),
            frame_count=payload.get("frame_count"),
        )
        payload["layout"] = layout
        neg_payload = None
        if negative is not None:
            neg_payload = negative.to_payload()
            neg_layout = PackedLayout(
                text_len,
                latent_t,
                lat_h,
                lat_w,
                audio_t,
                keyframes=neg_payload.get("keyframes"),
                refs=neg_payload.get("refs"),
                frame_count=neg_payload.get("frame_count"),
            )
            neg_payload["layout"] = neg_layout
        payload["text_token_tags"] = conditioning.text.tags.to(device)
        if neg_payload is not None:
            neg_payload["text_token_tags"] = negative.text.tags.to(device)

        offload_dit = bool(getattr(
            getattr(self.assets, "fl_constraint", None),
            "offload_dit",
            False,
        ))
        rolling_path = (
            self.assets.av_encoder is not None
            and bool(getattr(
                self.assets.fl_constraint,
                "keyframes",
                None,
            ))
        )
        vae_device = "cpu" if (offload_dit or not rolling_path) else None
        if self.assets.av_encoder is not None and rolling_path:
            self.vae = load_vae_pack(
                self.assets.av_encoder.video_path,
                self.assets.av_encoder.audio_path,
                device=vae_device,
            )

        adaln_reader = None
        adaln_bake_entries = {}
        final_bake_entries = []
        config = None
        prefix = ""

        if cfg.use_adaln_cache and len(sigmas) > 1:
            prebake_start = time.perf_counter()
            capture = _SigmaCaptureModel(
                _H3ModelSampling(),
                shift_video=cfg.shift_video,
                shift_audio=cfg.shift_audio,
            )
            tiny = torch.zeros((1, 16), dtype=dtype)
            if getattr(comfy.model_sampling, "ModelSamplingAV", None) is None:
                _h3_sampler(cfg.sampler_name)(
                    capture,
                    tiny,
                    sigmas.cpu(),
                    extra_args={"model_options": {}, "seed": cfg.seed},
                    callback=lambda *args: None,
                    disable=True,
                )
            else:
                samp = comfy.samplers.sampler_object(cfg.sampler_name)
                samp.sample(
                    capture,
                    sigmas.cpu(),
                    extra_args={"model_options": {}, "seed": cfg.seed},
                    callback=lambda *args: None,
                    noise=torch.zeros_like(tiny),
                    latent_image=tiny,
                    denoise_mask=None,
                    disable_pbar=True,
                )
            bake_sigmas = sorted(
                set(capture.captured) | {float(s) for s in sigmas}
            )

            adaln_reader = BlockReader(handle.model_path)
            config = scan_dit_config(adaln_reader, MiniMaxH3DiTConfig())
            prefix = detect_key_prefix(adaln_reader)
            if handle.loras and config.adaln_curve_grid is None:
                for idx, entries in handle.loras.block_groups.items():
                    adaln_entries = [
                        e for e in entries
                        if e.target == "adaln_proj.linear"
                    ]
                    if adaln_entries:
                        adaln_bake_entries[idx] = adaln_entries
                final_bake_entries = handle.loras.final_adaln_entries

            planner = adaln.AdaLNCachePlanner(
                cfg.shift_video, cfg.shift_audio)
            bake_plans = planner.build(
                bake_sigmas,
                payload,
                payload["layout"],
                neg_payload,
                neg_payload["layout"] if neg_payload is not None else None,
            )
            baker = adaln.AdaLNCacheBaker(
                adaln_reader,
                config,
                prefix,
                dtype,
                device,
                adaln_entries=adaln_bake_entries,
                final_adaln_entries=final_bake_entries,
                batch_blocks=cfg.adaln_prebake_batch,
            )
            self.adaln_cache = baker.bake(bake_plans)
            self.prebake_seconds = time.perf_counter() - prebake_start
            collect_garbage()

        swap = self.assets.runtime.swap or handle.swap
        if offload_dit:
            swap = replace(swap, offload_dit=True)

        self.reader = adaln_reader
        self.model = handle.load(
            swap_config=swap,
            include_adaln=not cfg.use_adaln_cache,
            vram_spec=self.assets.vram_spec,
        )
        if self.adaln_cache is not None:
            self.model.adaln_cache = self.adaln_cache
            if adaln_reader is not None:
                def bake_missing(key, unique_t):
                    entry = adaln.bake_adaln_entry(
                        adaln_reader,
                        config,
                        prefix,
                        unique_t,
                        dtype,
                        device,
                        adaln_entries=adaln_bake_entries,
                        final_adaln_entries=final_bake_entries,
                    )
                    self.adaln_cache.entries[key] = entry
                    return entry
                self.model._adaln_bake_fallback = bake_missing

        self.teacache = self.assets.runtime.make_teacache(self.model)
        if offload_dit and self.vae is not None:
            self.vae.to("cpu")

        block_stats = None
        swap_mgr = getattr(self.model, "_swap_mgr", None)
        if swap_mgr is not None:
            block_stats = swap_mgr.stats()

        positive_text = TextConditioning(
            states=handle.preprocess_text(
                self.assets.positive.text.states.to(device, dtype),
                include_adaln=not cfg.use_adaln_cache,
            ),
            tags=self.assets.positive.text.tags,
        )
        negative_text = None
        if self.assets.negative is not None:
            negative_text = TextConditioning(
                states=handle.preprocess_text(
                    self.assets.negative.text.states.to(device, dtype),
                    include_adaln=not cfg.use_adaln_cache,
                ),
                tags=self.assets.negative.text.tags,
            )

        self.context = SessionContext(
            model=self.model,
            reader=self.reader,
            vae=self.vae,
            positive_text=positive_text,
            negative_text=negative_text,
            sigmas=sigmas,
            teacache=self.teacache,
            adaln_cache=self.adaln_cache,
            device=device,
            dtype=dtype,
            positive_payload=payload,
            negative_payload=neg_payload,
            block_stats=block_stats,
        )
        return self.context

    def finish(self) -> None:
        if self.teacache is not None:
            try:
                self.teacache.detach()
            except Exception:
                pass
        if self.reader is not None:
            try:
                self.reader.close()
            except Exception:
                pass
            if self.model is not None:
                try:
                    self.model._adaln_bake_fallback = None
                except Exception:
                    pass
        try:
            self.assets.handle.unload()
        except Exception:
            pass
        if self.vae is not None:
            try:
                self.vae.unload()
            except Exception:
                pass
        unload_all_vaes()
        unload_all_encoders()
        collect_garbage(aggressive=True)


__all__ = ["H3Session"]
