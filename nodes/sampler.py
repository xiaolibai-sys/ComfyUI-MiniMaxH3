"""KSampler + decode + unload nodes for the MiniMax H3 runner."""

from __future__ import annotations

import torch


class MiniMaxH3KSampler:
    """Native MiniMax-H3 dual-schedule sampler over the packed AV latent.

    Optional config sockets follow the BerniniRWrapper pattern: TeaCache and
    BlockSwap are frozen dataclass payloads built by their own args nodes and
    injected by the sampling core.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MINIMAX_H3_MODEL",),
                "positive": ("MINIMAX_H3_COND",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 30.0, "step": 0.1,
                    "tooltip": "Classifier-free guidance scale. 1.0 disables negative guidance; "
                               "values above 1.0 require a negative conditioning input."}),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 1.0, "max": 100.0}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 100.0}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "use_adaln_cache": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Pre-bake AdaLN modulations and skip AdaLN weights during sampling."}),
            },
            "optional": {
                "negative": ("MINIMAX_H3_COND",),
                "latent": ("MINIMAX_H3_LATENT",),
                "teacache_args": ("MINIMAX_H3_TEACACHE",),
                "block_swap_args": ("MINIMAX_H3_SWAP",),
            },
        }

    RETURN_TYPES = ("MINIMAX_H3_LATENT", "STRING")
    RETURN_NAMES = ("latent", "stats")
    FUNCTION = "sample"
    CATEGORY = "MiniMax-H3/sampling"

    def sample(self, model, positive, seed, steps, cfg,
               shift_video, shift_audio, denoise, use_adaln_cache,
               negative=None, latent=None,
               teacache_args=None, block_swap_args=None):
        import comfy.utils
        import latent_preview
        from ..utils.injection import InjectionContext
        from .h3_sampling import h3_sample

        if latent is None:
            raise ValueError(
                "MiniMax H3 KSampler: latent input is required; connect "
                "MiniMax H3 Conditioning.latent."
            )

        injection = InjectionContext.build(block_swap_args=block_swap_args,
                                           teacache_args=teacache_args)
        callback = latent_preview.prepare_callback(model, steps)

        result = h3_sample(
            model, positive, latent, negative, steps, cfg,
            sampler_name="euler",
            shift_video=shift_video,
            shift_audio=shift_audio,
            denoise=denoise,
            seed=seed,
            injection=injection,
            preview_callback=callback,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
            use_adaln_cache=use_adaln_cache)

        stats = (f"steps={result.steps} swap_hits={result.swap_hits} "
                 f"swap_loads={result.swap_loads} peak_vram={result.peak_vram_mb:.0f}MiB")
        return (result.av, stats)


class MiniMaxH3Decode:
    """Decode the joint AV latent to video frames + stereo audio."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("MINIMAX_H3_LATENT",),
                "av_encoder": ("MINIMAX_H3_AV_ENCODER",),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("video", "audio")
    FUNCTION = "decode"
    CATEGORY = "MiniMax-H3/decode"

    def decode(self, latent, av_encoder):
        from ..models.vae import load_vae_pack
        pack = load_vae_pack(av_encoder.video_path, av_encoder.audio_path)
        video = pack.decode_video(latent.video)
        # [B, 3, T, H, W] (-1..1) -> [T, H, W, 3] (0..1)
        video = video.permute(0, 2, 3, 4, 1).flatten(0, 1)
        video = video.mul_(0.5).add_(0.5).clamp_(0, 1).cpu()
        audio = pack.decode_audio(latent.audio) if pack.audio_vae is not None else None
        if audio is None:
            audio = torch.zeros(latent.audio.shape[0], 2, 0)
        else:
            audio = audio.cpu()
        audio_dict = {"waveform": audio, "sample_rate": 32000}
        return (video, audio_dict)


class MiniMaxH3UnloadAll:
    """Free every cached model / VAE / encoder (self-managed lifecycle)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ()
    FUNCTION = "unload"
    CATEGORY = "MiniMax-H3/utils"
    OUTPUT_NODE = True

    def unload(self):
        from ..utils.lifecycle import unload_all
        from ..models.vae import unload_all_vaes
        from ..utils.encoder_use import unload_all_encoders
        unload_all()
        unload_all_vaes()
        unload_all_encoders()
        return ()
