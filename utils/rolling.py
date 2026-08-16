"""Rolling FL2VA pipeline built on H3Session + SamplerRunner."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Optional

import torch

from .memory import SamplingMemory
from .loudness import match_segment_loudness
from .session import H3Session
from .runner import SamplerRunner
from .temporal import AUDIO_LATENT_FPS, align_frame_count, video_latent_t
from .types import (
    AVLatent,
    FLConstraint,
    FLKeyframe,
    H3Conditioning,
    H3SampleResult,
    KeyframeCondition,
    MediaConditioning,
    RollingOutput,
    RollingPlan,
    RollingSegment,
    SamplerRequest,
    SamplingAssets,
    SamplingConfig,
    TextConditioning,
)
from .wrapper import h3_sigmas

logger = logging.getLogger("h3.rolling")


def _coerce_fl_constraint(fl, latent) -> FLConstraint:
    if isinstance(fl, FLConstraint):
        return fl
    if isinstance(fl, dict):
        fl_data = fl.get("fl_data")
        data = {}
        if fl_data:
            try:
                data = json.loads(fl_data)
            except Exception:
                data = {}
            raw_keyframes = data.get("keyframes") or []
            if raw_keyframes:
                times = sorted(
                    float(kf.get("time") or 0.0)
                    for kf in raw_keyframes
                )
                raw_by_time = {
                    float(kf.get("time") or 0.0): kf
                    for kf in raw_keyframes
                }
                first = fl.get("first_frame")
                last = fl.get("last_frame")
                duration = times[-1]
                offload_dit = fl.get("offload_dit")
                if offload_dit is None:
                    offload_dit = data.get("offload_dit", True)
                audio_loudness_match = fl.get("audio_loudness_match")
                if audio_loudness_match is None:
                    audio_loudness_match = data.get(
                        "audio_loudness_match", True)
                keyframes = tuple(
                    _fl_keyframe_from_time(
                        t,
                        duration,
                        first,
                        last,
                        raw_by_time.get(t),
                    )
                    for t in times
                )
                return FLConstraint(
                    fps=24,
                    keyframes=keyframes,
                    offload_dit=offload_dit,
                    audio_loudness_match=bool(audio_loudness_match),
                    global_negative_prompt=str(
                        data.get("global_negative_prompt") or ""
                    ),
                )
        duration = fl.get("duration")
        if duration is None and latent is not None:
            duration = latent.audio.shape[-1] / AUDIO_LATENT_FPS
        duration = float(duration or 5.0)
        first = fl.get("first_frame")
        last = fl.get("last_frame")
        offload_dit = fl.get("offload_dit")
        if offload_dit is None:
            offload_dit = data.get("offload_dit", True)
        audio_loudness_match = fl.get("audio_loudness_match")
        if audio_loudness_match is None:
            audio_loudness_match = data.get("audio_loudness_match", True)
        global_negative = str(
            data.get("global_negative_prompt")
            or fl.get("global_negative_prompt")
            or ""
        )
        if first is None and last is None:
            keyframes = (
                FLKeyframe(time=0.0, image=None),
                FLKeyframe(time=duration, image=None),
            )
        else:
            keyframes = (
                FLKeyframe(time=0.0, image=first),
                FLKeyframe(time=duration, image=last),
            )
        return FLConstraint(
            fps=24,
            keyframes=keyframes,
            offload_dit=offload_dit,
            audio_loudness_match=bool(audio_loudness_match),
            global_negative_prompt=global_negative,
        )
    raise TypeError(f"Unsupported FL constraint: {type(fl)!r}")


def _fl_keyframe_from_time(time, duration, first, last, raw):
    image = (
        first
        if abs(time - 0.0) < 1e-6
        else last
        if abs(time - duration) < 1e-6
        else None
    )
    raw_image = raw.get("image") if isinstance(raw, dict) else None
    if isinstance(raw_image, torch.Tensor):
        image = raw_image
    elif isinstance(raw_image, dict) and raw_image.get("name"):
        image = _load_fl_image(raw_image)
    return FLKeyframe(
        time=time,
        image=image,
        prompt=str(raw.get("prompt") or "") if isinstance(raw, dict) else "",
        negative_prompt=(
            str(raw.get("negative_prompt") or "")
            if isinstance(raw, dict)
            else ""
        ),
        note=str(raw.get("note") or "") if isinstance(raw, dict) else "",
    )


def _load_fl_image(info: dict):
    try:
        import numpy as np
        from PIL import Image
        import folder_paths

        path = folder_paths.get_annotated_filepath(info.get("name"))
        with Image.open(path) as pil:
            arr = np.array(pil.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
    except Exception:
        return None


def build_rolling_plan(
    fl: FLConstraint,
    width: int,
    height: int,
) -> RollingPlan:
    keyframes = sorted(fl.keyframes, key=lambda kf: kf.time)
    if not keyframes:
        raise ValueError("Rolling FL2VA requires at least one keyframe.")

    total_duration = float(keyframes[-1].time)
    segments = []
    for i, kf in enumerate(keyframes):
        start_time = float(kf.time)
        end_time = (
            float(keyframes[i + 1].time)
            if i + 1 < len(keyframes)
            else total_duration
        )
        if end_time <= start_time:
            continue
        frame_count = align_frame_count(round((end_time - start_time) * fl.fps))
        segments.append(RollingSegment(
            start_time=start_time,
            end_time=end_time,
            frame_count=frame_count,
            latent_t=video_latent_t(frame_count),
            audio_t=round((end_time - start_time) * AUDIO_LATENT_FPS),
            start_image=kf.image if i == 0 else None,
            end_image=(
                keyframes[i + 1].image
                if i + 1 < len(keyframes)
                else None
            ),
            prompt=kf.prompt,
            negative_prompt=kf.negative_prompt,
            note=kf.note,
        ))
    return RollingPlan(
        segments=tuple(segments),
        width=width,
        height=height,
        fps=fl.fps,
    )


def build_segment_latent(
    segment: RollingSegment,
    width: int,
    height: int,
    device,
    dtype,
) -> AVLatent:
    return AVLatent(
        video=torch.zeros(
            [1, 24, segment.latent_t, height // 16, width // 16],
            device=device,
            dtype=dtype,
        ),
        audio=torch.zeros(
            [1, 32, 2, segment.audio_t],
            device=device,
            dtype=dtype,
        ),
    )


def build_segment_conditioning(
    base: H3Conditioning,
    segment: RollingSegment,
    start_latent: Optional[torch.Tensor],
    end_latent: Optional[torch.Tensor],
) -> H3Conditioning:
    keyframes = []
    if start_latent is not None:
        keyframes.append(KeyframeCondition(0, start_latent))
    if end_latent is not None:
        keyframes.append(KeyframeCondition(
            segment.frame_count - 1, end_latent))
    return H3Conditioning(
        text=base.text,
        media=MediaConditioning(
            keyframes=tuple(keyframes),
            frame_count=segment.frame_count,
        ),
    )


def _encode_image(vae, image: torch.Tensor, width: int, height: int):
    import comfy.utils

    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(
        samples, width, height, "lanczos", "disabled")
    pixels = (samples * 2 - 1).to(torch.float16)
    return vae.encode_video(pixels)


def _build_segment_payload(
    cond: H3Conditioning,
    latent: AVLatent,
    device,
) -> dict:
    from ..models.model import PackedLayout

    payload = cond.to_payload()
    layout = PackedLayout(
        cond.text.states.shape[1],
        latent.video.shape[2],
        latent.video.shape[3],
        latent.video.shape[4],
        latent.audio.shape[-1],
        keyframes=payload.get("keyframes"),
        refs=payload.get("refs"),
        frame_count=payload.get("frame_count"),
    )
    payload["layout"] = layout
    payload["text_token_tags"] = cond.text.tags.to(device)
    return payload


def _decode_to_image(decoded: torch.Tensor) -> torch.Tensor:
    return (
        decoded.permute(0, 2, 3, 4, 1)
        .flatten(0, 1)
        .mul_(0.5)
        .add_(0.5)
        .clamp_(0, 1)
        .cpu()
    )


def _sample_single(
    assets: SamplingAssets,
    config: SamplingConfig,
    preview_callback=None,
    disable_pbar: bool = False,
) -> H3SampleResult:
    latent = assets.latent
    if latent is None:
        raise ValueError("SamplingAssets.latent is required.")
    if assets.negative is None:
        config = replace(config, cfg=1.0)
    if config.sampler_name in {"dpm_adaptive"}:
        config = replace(config, use_adaln_cache=False)
    device = assets.handle.load_device
    if config.denoise <= 0.0:
        return H3SampleResult(video=latent.video, audio=latent.audio, steps=0)
    if config.denoise >= 1.0:
        sigmas = h3_sigmas(
            config.scheduler_name, config.steps, config.shift_video)
    else:
        sigmas = h3_sigmas(
            config.scheduler_name,
            int(config.steps / config.denoise),
            config.shift_video,
        )[-(config.steps + 1):]
    sigmas = sigmas.to(device)

    session = H3Session(assets, config)
    try:
        context = session.prepare(sigmas)
        assets.handle.release_token_refiner()
        request = SamplerRequest(latent=latent, seed=config.seed)
        result = SamplerRunner(context, config).run(
            request,
            preview_callback=preview_callback,
            disable_pbar=disable_pbar,
        )
        result.prebake_seconds = session.prebake_seconds
        return result
    finally:
        session.finish()


def rolling_sample(
    assets: SamplingAssets,
    config: SamplingConfig,
    preview_callback=None,
    disable_pbar: bool = False,
) -> H3SampleResult | RollingOutput:
    fl = _coerce_fl_constraint(assets.fl_constraint, assets.latent)
    if assets.av_encoder is None or not fl.keyframes:
        return _sample_single(
            assets,
            config,
            preview_callback=preview_callback,
            disable_pbar=disable_pbar,
        )

    plan = build_rolling_plan(fl, config.width, config.height)
    if not plan.segments:
        raise ValueError("Rolling FL2VA produced no segments.")
    logger.info(
        "rolling FL2VA keyframes=%d segments=%d keyframe_images=%s",
        len(fl.keyframes),
        len(plan.segments),
        [kf.image is not None for kf in fl.keyframes],
    )
    first = plan.segments[0]
    source_positive = assets.positive

    device = assets.handle.load_device
    dtype = (assets.runtime.swap or assets.handle.swap).torch_dtype
    first_latent = build_segment_latent(
        first, config.width, config.height, device, dtype)
    representative = build_segment_conditioning(
        source_positive, first, first_latent, None)
    session_assets = replace(
        assets,
        fl_constraint=fl,
        positive=representative,
        latent=first_latent,
    )

    sigmas = h3_sigmas(
        config.scheduler_name, config.steps, config.shift_video).to(device)
    session = H3Session(session_assets, config)
    try:
        session.prepare(sigmas)
    except Exception:
        session.finish()
        raise

    memory = SamplingMemory(session)
    assets.handle.release_token_refiner()
    prev_start_latent = None
    final_video = None
    audio_chunks = []

    try:
        # Encode every keyframe image up front in one VAE window: with
        # offload_dit each per-image phase inside the loop would cost a
        # full model unload/reload.  Chained starts are unaffected -
        # segments after the first still begin from prev_start_latent.
        plan_images = []
        for seg in plan.segments:
            for img in (seg.start_image, seg.end_image):
                if img is not None and not any(
                        img is seen for seen in plan_images):
                    plan_images.append(img)
        image_latents = {}
        if plan_images:
            logger.info(
                "rolling FL2VA: encoding %d keyframe image(s) in one VAE phase",
                len(plan_images),
            )
            encoded = memory.encode_images_phase(
                plan_images, config.width, config.height)
            image_latents = {
                id(img): lat for img, lat in zip(plan_images, encoded)
            }

        for i, segment in enumerate(plan.segments):
            start_latent = prev_start_latent
            if segment.start_image is not None:
                start_latent = image_latents.get(id(segment.start_image))
            end_latent = None
            if segment.end_image is not None:
                end_latent = image_latents.get(id(segment.end_image))

            base_text = assets.positive.text
            segment_text = None
            negative_text = None

            if i < len(source_positive.segment_texts):
                segment_text = source_positive.segment_texts[i]
                states = assets.handle.preprocess_text(
                    segment_text.states,
                    include_adaln=not config.use_adaln_cache,
                )
                base_text = TextConditioning(
                    states=states,
                    tags=segment_text.tags,
                )
                segment_text = base_text

            if negative_text is None and i < len(
                source_positive.segment_negative_texts
            ):
                negative_text = source_positive.segment_negative_texts[i]
                states = assets.handle.preprocess_text(
                    negative_text.states,
                    include_adaln=not config.use_adaln_cache,
                )
                negative_text = TextConditioning(
                    states=states,
                    tags=negative_text.tags,
                )
            elif negative_text is None and assets.negative is not None:
                negative_text = assets.negative.text

            assets.handle.release_token_refiner()

            segment_cond = build_segment_conditioning(
                H3Conditioning(text=base_text or assets.positive.text),
                segment,
                start_latent,
                end_latent,
            )
            segment_latent = build_segment_latent(
                segment, config.width, config.height, device, dtype)
            payload = _build_segment_payload(
                segment_cond, segment_latent, device)
            neg_payload = None
            if negative_text is not None:
                neg_cond = H3Conditioning(
                    text=negative_text,
                    media=segment_cond.media,
                )
                neg_payload = _build_segment_payload(
                    neg_cond, segment_latent, device)

            request = SamplerRequest(
                latent=segment_latent,
                seed=config.seed + i,
                payload=payload,
                negative_payload=neg_payload,
                positive_text=segment_text,
                negative_text=negative_text,
            )
            result = SamplerRunner(session.context, config).run(
                request,
                disable_pbar=disable_pbar,
            )
            is_last = i == len(plan.segments) - 1
            decoded_video, decoded_audio, prev_start_latent = (
                memory.decode_and_encode_boundary(
                    result,
                    need_start=not is_last,
                    reload_after=not is_last,
                )
            )

            frames = _decode_to_image(decoded_video)
            if i > 0 and frames.shape[0] > 1:
                frames = frames[1:]
            final_video = (
                frames
                if final_video is None
                else torch.cat([final_video, frames], dim=0)
            )
            audio_chunks.append(decoded_audio.cpu())
            del decoded_video, decoded_audio, result, segment_latent
    finally:
        try:
            assets.handle.release_token_refiner()
        finally:
            session.finish()

    if fl.audio_loudness_match:
        final_audio = match_segment_loudness(audio_chunks)
    elif audio_chunks:
        final_audio = torch.cat(audio_chunks, dim=-1)
    else:
        final_audio = torch.zeros(1, 2, 0)

    return RollingOutput(
        video=final_video,
        audio=final_audio,
        stats=f"segments={len(plan.segments)}",
        segment_count=len(plan.segments),
        peak_vram_mb=0.0,
    )


__all__ = [
    "build_rolling_plan",
    "rolling_sample",
]
