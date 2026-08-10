"""Conditioning + latent nodes for the MiniMax H3 runner."""

from __future__ import annotations

import math

import torch

import comfy.utils

from ..utils.types import AVLatent, H3Conditioning

FPS = 24
AUDIO_LATENT_FPS = 40
MAX_RESOLUTION = 8192
CANVAS_MULTIPLE = 32
REF_IMAGE_SHORT_EDGE = 2048


def align_frame_count(n: int) -> int:
    while n % 17 != 5:
        n += 1
    return n


def video_latent_t(frame_count: int) -> int:
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def temporal_shape(length: int):
    frame_count = align_frame_count(max(5, length))
    duration = frame_count / FPS
    return frame_count, video_latent_t(frame_count), round(duration * AUDIO_LATENT_FPS)


def _resize(image: torch.Tensor, width: int, height: int, crop: str) -> torch.Tensor:
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def adapt_canvas(width: int, height: int):
    """768-short-edge canvas with 768*1344 area cap (official node logic)."""
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = 768.0 * ratio, 768.0
    else:
        nom_w, nom_h = 768.0, 768.0 / ratio
    if nom_w * nom_h > 768 * 1344:
        s = math.sqrt(768 * 1344 / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))


def _make_av_latent(width, height, length):
    frame_count, latent_t, audio_t = temporal_shape(length)
    video = torch.zeros([1, 24, latent_t, height // 16, width // 16])
    audio = torch.zeros([1, 32, 2, audio_t])
    return AVLatent(video=video, audio=audio)


def _resolve_duration(prompt_ref, package):
    if prompt_ref is not None:
        frame_count = prompt_ref.get("frame_count")
        if not frame_count:
            total_duration = float(prompt_ref.get("total_duration") or 5.0)
            frame_count = align_frame_count(round(total_duration * FPS))
        total_duration = float(
            prompt_ref.get("total_duration") or frame_count / FPS
        )
        return (
            int(frame_count),
            total_duration,
            str(prompt_ref.get("text") or ""),
            str(prompt_ref.get("negative_prompt") or ""),
        )
    return 124, 124 / FPS, "", ""


class MiniMaxH3Conditioning:
    """Prompt -> Qwen3-VL-32B hidden states (+ optional FL constraint)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_encoder": ("MINIMAX_H3_TEXT_ENCODER",),
                "width": ("INT", {"default": 1344, "min": 32, "max": MAX_RESOLUTION, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": MAX_RESOLUTION, "step": 32}),
            },
            "optional": {
                "av_encoder": ("MINIMAX_H3_AV_ENCODER",),
                "fl_constraint": ("MINIMAX_H3_FL_CONSTRAINT",),
                "package": ("PACKAGE_DATA",),
                "prompt": ("MINIMAX_H3_PROMPT",),
                "ref_max": ("INT", {"default": 1280, "min": 32, "max": 4096,
                    "step": 32, "display": "slider",
                    "tooltip": "Reference longest-edge pixel limit before encoding."}),
            },
        }

    RETURN_TYPES = ("MINIMAX_H3_COND", "MINIMAX_H3_COND", "MINIMAX_H3_LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "make"
    CATEGORY = "MiniMax-H3/conditioning"

    def make(self, text_encoder, prompt="", negative_prompt="", width=1344, height=768,
             av_encoder=None, fl_constraint=None, package=None, ref_max=1280):
        if isinstance(prompt, dict):
            prompt_ref = prompt
            prompt = str(prompt.get("text") or "")
        else:
            prompt_ref = None
        if prompt_ref is not None:
            frame_count, total_duration, ref_text, ref_negative = _resolve_duration(
                prompt_ref, package)
            if ref_text:
                prompt = ref_text
            if ref_negative:
                negative_prompt = ref_negative
        else:
            frame_count, total_duration, _, _ = _resolve_duration(
                prompt_ref, package)
        if not prompt.strip():
            raise ValueError(
                "MiniMax H3 Conditioning: prompt did not provide a prompt."
            )
        length = frame_count
        if package is not None and fl_constraint is not None:
            raise ValueError(
                "MiniMax H3 Conditioning: PackageData references and FL Constraint "
                "cannot be mixed; use separate first/last-frame and reference paths."
            )
        if package is not None:
            return self._from_package(
                text_encoder, av_encoder, prompt, negative_prompt,
                width, height, length, package, ref_max)
        keyframes = []
        images = []
        frame_count, _, _ = temporal_shape(length)
        first_frame = last_frame = None
        if fl_constraint is not None:
            first_frame = fl_constraint.get("first_frame")
            last_frame = fl_constraint.get("last_frame")
            for name, frame in (("first_frame", first_frame), ("last_frame", last_frame)):
                if frame is not None and frame.shape[0] != 1:
                    raise ValueError(
                        f"MiniMax H3 Conditioning: {name} must contain exactly one image."
                    )
        if (first_frame is not None or last_frame is not None) and av_encoder is None:
            raise ValueError("MiniMax H3 I2V keyframes require the AV encoder input.")
        if av_encoder is not None and (first_frame is not None or last_frame is not None):
            from ..models.vae import load_vae_pack
            pack = load_vae_pack(av_encoder.video_path, av_encoder.audio_path)
            if first_frame is not None:
                img = _resize(first_frame[:1], width, height, "disabled")
                latent = pack.encode_video(img.movedim(-1, 1).unsqueeze(0).to(torch.float16))
                images.append(img)
                keyframes.append({"resolved_frame_index": 0, "latent": latent})
            if last_frame is not None:
                img = _resize(last_frame[:1], width, height, "center")
                latent = pack.encode_video(img.movedim(-1, 1).unsqueeze(0).to(torch.float16))
                images.append(img)
                keyframes.append({"resolved_frame_index": frame_count - 1, "latent": latent})

        ref_items = [{"type": "image", "data": img} for img in images] or None
        if negative_prompt.strip():
            (text_states, tags), (neg_states, neg_tags) = (
                text_encoder.encode_pair(
                    prompt, negative_prompt,
                    minimax_ref_items=ref_items))
            neg_cond = H3Conditioning(
                text_states=neg_states,
                text_token_tags=neg_tags,
                frame_count=frame_count,
            )
        else:
            text_states, tags = text_encoder.encode(
                prompt, minimax_ref_items=ref_items)
            neg_cond = None

        cond = H3Conditioning(
            text_states=text_states,
            text_token_tags=tags,
            keyframes=keyframes,
            frame_count=frame_count,
        )
        latent = _make_av_latent(width, height, length)
        return (cond, neg_cond, latent)

    @staticmethod
    def _from_package(text_encoder, av_encoder, prompt, negative_prompt,
                      width, height, length, package, ref_max=1280):
        kwargs = {}
        images = package.get("images")
        if images is not None:
            for i, frame in enumerate(images):
                kwargs[f"ref_image_{i + 1}"] = frame.unsqueeze(0)
        for i, video in enumerate(package.get("videos", []) or []):
            kwargs[f"ref_video_{i + 1}"] = video["frames"]
        for i, audio in enumerate(package.get("audios", []) or []):
            kwargs[f"ref_audio_{i + 1}"] = audio

        if not kwargs:
            return MiniMaxH3Conditioning().make(
                text_encoder, "", "", width, height,
                prompt={
                    "text": prompt,
                    "negative_prompt": negative_prompt,
                    "frame_count": length,
                    "total_duration": length / FPS,
                })
        if av_encoder is None:
            raise ValueError("MiniMax H3 Conditioning: PackageData requires AV encoder.")

        cond, latent = MiniMaxH3ReferenceToVideo().make(
            text_encoder, av_encoder, prompt, width, height, length, "match",
            ref_max=ref_max, **kwargs)
        _, neg_cond, _ = MiniMaxH3Conditioning().make(
            text_encoder, "", "", width, height,
            prompt={
                "text": prompt,
                "negative_prompt": negative_prompt,
                "frame_count": length,
                "total_duration": length / FPS,
            })
        return (cond, neg_cond, latent)


class MiniMaxH3ReferenceToVideo:
    """ref2va: prompt + reference images / videos / audio -> conditioning + AV latent.

    Mirrors the official ``MiniMaxH3ReferenceToVideo`` node (#15224).  References
    enter the presentation in fixed order: images, then videos (each soundtrack's
    <Audio j> label right before its <Video k>), then standalone audio.  Ordinals
    are 1-based per type (<Picture i> / <Video k> / <Audio j>).
    """

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, 10):
            optional[f"ref_image_{i}"] = ("IMAGE", {"tooltip": "Reference image"})
        for i in range(1, 4):
            optional[f"ref_video_{i}"] = ("IMAGE", {"tooltip": "Reference video frames at 24 fps [T,H,W,C]"})
        for i in range(1, 4):
            optional[f"ref_video_audio_{i}"] = ("AUDIO", {"tooltip": "Soundtrack of the same-numbered reference video"})
        for i in range(1, 4):
            optional[f"ref_audio_{i}"] = ("AUDIO", {"tooltip": "Standalone reference audio"})
        return {
            "required": {
                "text_encoder": ("MINIMAX_H3_TEXT_ENCODER",),
                "av_encoder": ("MINIMAX_H3_AV_ENCODER",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "width": ("INT", {"default": 1344, "min": 32, "max": MAX_RESOLUTION, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": MAX_RESOLUTION, "step": 32}),
                "length": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17,
                    "tooltip": "Frames at 24fps, snapped to the 17k+5 grid (124 = ~5s)"}),
                "ref_image_size": (["match", "max"], {"default": "match"}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("MINIMAX_H3_COND", "MINIMAX_H3_LATENT")
    RETURN_NAMES = ("conditioning", "latent")
    FUNCTION = "make"
    CATEGORY = "MiniMax-H3/conditioning"

    @staticmethod
    def _encode_ref_audio(pack, audio):
        import torchaudio
        waveform = audio["waveform"]            # [B, C, L]
        sr = int(audio["sample_rate"])
        vae_sr = 32000
        if sr != vae_sr:
            waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
        wav = waveform[:1]                       # [1, C, L]
        if wav.shape[1] == 1:                    # mono -> stereo
            wav = wav.repeat(1, 2, 1)
        z = pack.encode_audio(wav.to(torch.float32))
        return z, z.shape[-1]

    def make(self, text_encoder, av_encoder, prompt, width, height, length,
             ref_image_size="match", ref_max=1280,
             **kwargs):
        from ..models.vae import load_vae_pack

        frame_count, latent_t, audio_t = temporal_shape(length)
        latent = AVLatent(
            video=torch.zeros([1, 24, latent_t, height // 16, width // 16]),
            audio=torch.zeros([1, 32, 2, audio_t]),
        )
        pack = load_vae_pack(av_encoder.video_path, av_encoder.audio_path)

        ref_items = []     # tokenizer presentation, in request order
        ref_blocks = []    # DiT payload, same order

        images = [kwargs.get(f"ref_image_{i}") for i in range(1, 10)]
        videos = [kwargs.get(f"ref_video_{i}") for i in range(1, 4)]
        video_audios = {f"ref_video_audio_{i}": kwargs.get(f"ref_video_audio_{i}")
                        for i in range(1, 4)}
        audios = [kwargs.get(f"ref_audio_{i}") for i in range(1, 4)]

        for img in images:
            if img is None:
                continue
            h, w = img.shape[1], img.shape[2]
            if ref_max:
                max_edge = max(1, int(ref_max))
                scale = min(1.0, max_edge / max(w, h))
            elif ref_image_size == "match":
                scale = min(1.0, math.sqrt((width * height) / (w * h)))
            else:
                scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
            tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            resized = _resize(img[:1], tw, th, "disabled")
            # [1, H, W, 3] (0-1) -> [1, 3, 1, H, W] ([-1, 1]) for the video VAE
            pixels = (resized.movedim(-1, 1) * 2 - 1).to(torch.float16)
            z = pack.encode_video(pixels)
            ref_items.append({"type": "image", "data": resized})
            ref_blocks.append({"kind": "image", "latent_h": th // 16,
                               "latent_w": tw // 16, "latent": z})

        for name, video_frames in ((f"ref_video_{i}", videos[i - 1]) for i in range(1, 4)):
            if video_frames is None:
                continue
            soundtrack = video_audios.get(
                "ref_video_audio_" + name.rsplit("_", 1)[-1])
            vh, vw = video_frames.shape[1], video_frames.shape[2]
            if ref_max:
                max_edge = max(1, int(ref_max))
                scale = min(1.0, max_edge / max(vw, vh))
                cw = max(CANVAS_MULTIPLE, round(vw * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                ch = max(CANVAS_MULTIPLE, round(vh * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            else:
                cw, ch = adapt_canvas(vw, vh)
            if vw * vh < cw * ch:
                cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            frames = _resize(video_frames, cw, ch, "disabled")
            if frames.shape[0] > frame_count:
                frames = frames[:frame_count]
            n = frames.shape[0]
            if n < 5:
                raise ValueError("MiniMax H3 reference videos need >= 5 frames")
            while n % 17 != 5:
                n -= 1
            frames = frames[:n]
            # [T, H, W, C] -> [1, C, T, H, W] ([-1, 1])
            pixels = (frames.movedim(-1, 1).permute(1, 0, 2, 3).unsqueeze(0)
                      * 2 - 1).to(torch.float16)
            z = pack.encode_video(pixels)
            audio_latent, ref_audio_t = (None, 0)
            if soundtrack is not None:
                audio_latent, ref_audio_t = self._encode_ref_audio(pack, soundtrack)
                ref_items.append({"type": "audio"})
            sample_idx = list(range(0, frames.shape[0], FPS // 2))
            qwen_frames = frames[sample_idx]
            ref_items.append({"type": "video", "data": qwen_frames,
                              "timestamps": [i / 2.0 for i in range(len(sample_idx))]})
            ref_blocks.append({"kind": "video_audio" if ref_audio_t else "video",
                               "latent_t": z.shape[2], "latent_h": ch // 16,
                               "latent_w": cw // 16, "ref_audio_t": ref_audio_t,
                               "latent": z, "audio_latent": audio_latent})

        for audio in audios:
            if audio is None:
                continue
            audio_latent, ref_audio_t = self._encode_ref_audio(pack, audio)
            ref_items.append({"type": "audio"})
            ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t,
                               "audio_latent": audio_latent})

        text_states, tags = text_encoder.encode(prompt, minimax_ref_items=ref_items)
        cond = H3Conditioning(
            text_states=text_states,
            text_token_tags=tags,
            refs=ref_blocks,
            frame_count=frame_count,
        )
        return (cond, latent)
