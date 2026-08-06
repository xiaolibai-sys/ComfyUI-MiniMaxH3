"""ComfyUI node registry for the MiniMax H3 runner."""

from .loaders import MiniMaxH3Loader, MiniMaxH3VAELoader, MiniMaxH3EncoderLoader
from .lora_loader import MiniMaxH3LoraLoader
from .attention import MiniMaxH3AttentionConfig
from .conditioning import MiniMaxH3Conditioning
from .fl_constraint import MiniMaxH3FLConstraint
from .package_data import MiniMaxH3PackageData
from .storyboard import MiniMaxH3Storyboard
from .sampler import MiniMaxH3KSampler, MiniMaxH3Decode, MiniMaxH3UnloadAll
from .teacache_args import MiniMaxH3TeaCacheArgs
from .block_swap_args import MiniMaxH3BlockSwapArgs
from .video_batch import MiniMaxH3VideoBatch
from .refiner import (
    MiniMaxH3ContextIRRefiner,
    MiniMaxH3OpenAICompatibleRefiner,
    MiniMaxH3SimplePrompt,
)

NODE_CLASS_MAPPINGS = {
    "MiniMaxH3Loader": MiniMaxH3Loader,
    "MiniMaxH3VAELoader": MiniMaxH3VAELoader,
    "MiniMaxH3EncoderLoader": MiniMaxH3EncoderLoader,
    "MiniMaxH3LoraLoader": MiniMaxH3LoraLoader,
    "MiniMaxH3AttentionConfig": MiniMaxH3AttentionConfig,
    "MiniMaxH3Conditioning": MiniMaxH3Conditioning,
    "MiniMaxH3FLConstraint": MiniMaxH3FLConstraint,
    "MiniMaxH3PackageData": MiniMaxH3PackageData,
    "MiniMaxH3Storyboard": MiniMaxH3Storyboard,
    "MiniMaxH3SimplePrompt": MiniMaxH3SimplePrompt,
    "MiniMaxH3VideoBatch": MiniMaxH3VideoBatch,
    "MiniMaxH3ContextIRRefiner": MiniMaxH3ContextIRRefiner,
    "MiniMaxH3OpenAICompatibleRefiner": MiniMaxH3OpenAICompatibleRefiner,
    "MiniMaxH3KSampler": MiniMaxH3KSampler,
    "MiniMaxH3TeaCacheArgs": MiniMaxH3TeaCacheArgs,
    "MiniMaxH3BlockSwapArgs": MiniMaxH3BlockSwapArgs,
    "MiniMaxH3Decode": MiniMaxH3Decode,
    "MiniMaxH3UnloadAll": MiniMaxH3UnloadAll,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3Loader": "MiniMax H3 Model Loader (Streaming)",
    "MiniMaxH3VAELoader": "MiniMax H3 VAE Loader",
    "MiniMaxH3EncoderLoader": "MiniMax H3 Text Encoder Loader",
    "MiniMaxH3LoraLoader": "MiniMax H3 LoRA Loader",
    "MiniMaxH3AttentionConfig": "MiniMax H3 Attention Config",
    "MiniMaxH3Conditioning": "MiniMax H3 Conditioning",
    "MiniMaxH3FLConstraint": "MiniMax H3 FL Constraint",
    "MiniMaxH3PackageData": "MiniMax H3 PackageData",
    "MiniMaxH3Storyboard": "MiniMax H3 Storyboard",
    "MiniMaxH3SimplePrompt": "MiniMax H3 Simple Prompt",
    "MiniMaxH3VideoBatch": "MiniMax H3 VideoBatch",
    "MiniMaxH3ContextIRRefiner": "MiniMax H3 Context IR Refiner",
    "MiniMaxH3OpenAICompatibleRefiner": "MiniMax H3 OpenAI-Compatible Refiner",
    "MiniMaxH3KSampler": "MiniMax H3 KSampler",
    "MiniMaxH3TeaCacheArgs": "MiniMax H3 TeaCache Args",
    "MiniMaxH3BlockSwapArgs": "MiniMax H3 BlockSwap Args",
    "MiniMaxH3Decode": "MiniMax H3 Decode AV",
    "MiniMaxH3UnloadAll": "MiniMax H3 Unload All",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
