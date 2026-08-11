"""High-level streaming text-encoder API for MiniMax H3.

Loads the Qwen3-VL checkpoint from disk.  The **text backbone** (32B class)
runs in streaming mode (load group -> compute -> destroy); the small **vision
tower** stays resident.  Images/videos are preprocessed with
``transformers.AutoProcessor`` and fused into the text sequence with 3D MRoPE
position ids, exactly as ``transformers`` Qwen3VLModel does.

Example
-------
>>> from text_encoder import TextEncoder, StreamConfig
>>> enc = TextEncoder(r"D:\\MiniMax H3\\weights", StreamConfig(group_size=4))
>>> out = enc.encode("a cat riding a skateboard")            # text only
>>> out = enc.encode(TextEncoderInput(
...     text="what is in this image?", images=[r"C:\\img.png"]))  # image+text
>>> print(out.pooled_embedding.shape)   # (1, hidden_size)
>>> enc.destroy()
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .config import Qwen3VLConfig
from .fusion import Qwen3VLMultimodalFusion
from .modeling import Qwen3VLTextModel
from .stream import DiskGroupReader, GroupStreamer
from .types import (LoadingMode, PoolMode, StreamConfig,
                    TextEncoderInput, TextEncoderOutput)
from .vision import Qwen3VLVisionModel

# Official MiniMax H3 presentation tokens (comfy/text_encoders/minimax.py).
VISION_START = 151652
VISION_END = 151653
IMAGE_PAD = 151655
SPATIAL_MERGE_SIZE = 2
QWEN_IMAGE_MEAN = [0.5, 0.5, 0.5]
QWEN_IMAGE_STD = [0.5, 0.5, 0.5]


def _resize_patches(imgs, factor, min_pixels, max_pixels):
    """Resize [T, C, H, W] frames to the Qwen3-VL grid (bilinear, aligned)."""
    _, channel, height, width = imgs.shape
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    out = torch.nn.functional.interpolate(
        imgs, size=(h_bar, w_bar), mode="bilinear", align_corners=False)
    mean = torch.tensor(QWEN_IMAGE_MEAN, device=imgs.device).view(1, 3, 1, 1)
    std = torch.tensor(QWEN_IMAGE_STD, device=imgs.device).view(1, 3, 1, 1)
    return (out - mean) / std


def process_image_block(image, patch_size=16, temporal_patch_size=2,
                        merge_size=2, min_pixels=3136, max_pixels=12845056):
    """[H, W, C] (float 0-1) -> (flatten_patches [N, C*T*P*P], grid_thw [1, 3])."""
    if image.ndim == 4:                              # [B, H, W, C] -> single frame
        image = image[0]
    height, width, _ = image.shape
    imgs = image.permute(2, 0, 1).unsqueeze(0)          # [1, C, H, W]
    factor = patch_size * merge_size
    normed = _resize_patches(imgs, factor, min_pixels, max_pixels)
    grid_h = normed.shape[2] // patch_size
    grid_w = normed.shape[3] // patch_size
    pixel_values = normed.repeat(temporal_patch_size, 1, 1, 1)
    patches = pixel_values.reshape(
        1, temporal_patch_size, 3, grid_h // merge_size, merge_size,
        patch_size, grid_w // merge_size, merge_size, patch_size)
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten = patches.reshape(grid_h * grid_w,
                              3 * temporal_patch_size * patch_size * patch_size)
    grid_thw = torch.tensor([1, grid_h, grid_w], device=image.device,
                            dtype=torch.long).unsqueeze(0)
    return flatten, grid_thw


def process_video_block(frames, patch_size=16, temporal_patch_size=2,
                        merge_size=2, min_pixels=3136, max_pixels=12845056):
    """[2, H, W, C] frame pair -> (flatten_patches, grid_thw) with grid_t=1."""
    t, height, width, _ = frames.shape
    imgs = frames.permute(0, 3, 1, 2)                   # [2, C, H, W]
    factor = patch_size * merge_size
    normed = _resize_patches(imgs, factor, min_pixels, max_pixels)
    grid_h = normed.shape[2] // patch_size
    grid_w = normed.shape[3] // patch_size
    patches = normed.reshape(
        t, 3, grid_h // merge_size, merge_size, patch_size,
        grid_w // merge_size, merge_size, patch_size)
    patches = patches.permute(0, 3, 6, 2, 4, 1, 5, 7)
    flatten = patches.reshape(grid_h * grid_w,
                              3 * temporal_patch_size * patch_size * patch_size)
    grid_thw = torch.tensor([1, grid_h, grid_w], device=frames.device,
                            dtype=torch.long).unsqueeze(0)
    return flatten, grid_thw



# ---------------------------------------------------------------------------
# Mask / position helpers
# ---------------------------------------------------------------------------

def build_attention_mask(attention_mask: Optional[torch.Tensor],
                         seq_len: int, device: torch.device,
                         dtype: torch.dtype) -> Optional[torch.Tensor]:
    """Additive 4D causal+padding mask (transformers semantics, key-side only)."""
    if attention_mask is None:
        return None
    attention_mask = attention_mask.to(device=device).bool()
    causal = torch.tril(torch.ones(1, 1, seq_len, seq_len, device=device, dtype=torch.bool))
    valid = causal & attention_mask[:, None, None, :]
    mask = torch.zeros(attention_mask.shape[0], 1, seq_len, seq_len,
                       device=device, dtype=dtype)
    return mask.masked_fill(~valid, torch.finfo(dtype).min)


def build_position_ids(input_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Text-only MRoPE position ids: ``(3, batch, seq)`` with identical t/h/w."""
    seq_len = input_ids.shape[1]
    batch = input_ids.shape[0]
    return torch.arange(seq_len, device=device).view(1, 1, -1).expand(3, batch, -1)


def pool_hidden_states(hidden: torch.Tensor,
                       attention_mask: Optional[torch.Tensor],
                       mode: PoolMode) -> torch.Tensor:
    """Reduce ``(batch, seq, hidden)`` to ``(batch, hidden)``."""
    if attention_mask is not None:
        attention_mask = attention_mask.to(hidden.device)
    if mode == PoolMode.LAST:
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=-1).clamp(min=1)
            idx = (lengths - 1).unsqueeze(-1).expand(-1, hidden.shape[-1]).unsqueeze(1)
            return hidden.gather(1, idx).squeeze(1)
        return hidden[:, -1]
    if attention_mask is not None:
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return hidden.mean(dim=1)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class TextEncoder:
    """Streaming Qwen3-VL encoder (text + optional vision tower).

    Parameters
    ----------
    model_dir:
        Directory containing ``config.json`` and ``model.safetensors``
        (or ``model.safetensors.index.json`` + shards).
    stream_config:
        :class:`StreamConfig` controlling group size / prefetch / device.
    tokenizer:
        Optional tokenizer.  If ``None``, ``AutoTokenizer`` is loaded from
        ``model_dir`` when available.
    processor:
        Optional multimodal processor (``transformers.AutoProcessor``).  If
        ``None``, one is loaded from ``model_dir`` when available.
    """

    def __init__(self, model_dir: str | Path,
                 stream_config: Optional[StreamConfig] = None,
                 tokenizer=None, processor=None):
        self.model_dir = Path(model_dir)
        self.stream_config = stream_config or StreamConfig()
        self.device = self.stream_config.torch_device
        self.dtype = self.stream_config.torch_dtype
        self._destroyed = False

        self.config = Qwen3VLConfig.from_pretrained(self.model_dir)
        text_config = self.config.text
        self.reader = DiskGroupReader(
            self.model_dir, weight_path=self.stream_config.weight_path)
        self.layer_prefix = (
            self.stream_config.layer_prefix or self.reader.detect_layer_prefix()
        )

        # Text backbone skeleton on meta device (streamed).
        with torch.device("meta"):
            self.model = Qwen3VLTextModel(text_config)

        self._load_resident()
        self.streamer = GroupStreamer(
            model=self.model,
            layer_prefix=self.layer_prefix,
            num_layers=text_config.num_hidden_layers,
            reader=self.reader,
            device=self.device,
            dtype=self.dtype,
            group_size=self.stream_config.group_size,
            prefetch=self.stream_config.prefetch,
            disk_workers=self.stream_config.disk_workers,
            pin_memory=self.stream_config.pin_memory,
            full_precision_mm=self.stream_config.full_precision_mm,
        )

        # Vision tower (optional; stays resident).
        self.vision_model: Optional[Qwen3VLVisionModel] = None
        self.fusion: Optional[Qwen3VLMultimodalFusion] = None
        self.vision_prefix: Optional[str] = None
        if self.config.vision is not None and self._detect_vision_prefix() is not None:
            with torch.device("meta"):
                self.vision_model = Qwen3VLVisionModel(self.config.vision)
            self._load_vision_resident()
            self.fusion = Qwen3VLMultimodalFusion(
                self.config, self.vision_model, self.model)

        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = self._try_load("AutoTokenizer")
        self.processor = processor
        if self.processor is None:
            self.processor = self._try_load("AutoProcessor")

    # -- lifecycle -------------------------------------------------------------

    def _try_load(self, cls_name: str):
        try:
            import transformers
            return getattr(transformers, cls_name).from_pretrained(str(self.model_dir))
        except Exception:
            return None

    def _detect_vision_prefix(self) -> Optional[str]:
        pattern = re.compile(r"^(?P<prefix>.*)visual\.patch_embed\.proj\.weight$")
        for key in self.reader.all_keys():
            m = pattern.match(key)
            if m:
                return m.group("prefix")
        return None

    def _load_resident(self) -> None:
        """Text: embedding table, final norm, rotary buffer (small, keep on GPU)."""
        embed_key = f"{self.layer_prefix}embed_tokens.weight"
        norm_key = f"{self.layer_prefix}norm.weight"
        names = [embed_key, norm_key]
        tensors = self.reader.get_tensors([n for n in names if self.reader.has(n)])
        if embed_key in tensors:
            self._bind_embed(embed_key, tensors[embed_key])
        if norm_key in tensors:
            self.model.norm.weight = nn.Parameter(
                tensors[norm_key].to(self.device, self.dtype), requires_grad=False)
        rope = self.config.text.rope_parameters or {}
        theta = float(rope.get("rope_theta", 500000.0))
        head_dim = self.config.text.head_dim
        inv_freq = 1.0 / theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        self.model.rotary_emb.inv_freq.data = inv_freq.to(self.device)

    def _bind_embed(self, embed_key: str, weight: torch.Tensor) -> None:
        """Embedding table: dequantize int8_tensorwise (per-row scale) to
        compute dtype; plain tensors pass through unchanged."""
        scale_key = f"{self.layer_prefix}embed_tokens.weight_scale"
        if self.reader.has(scale_key):
            scale = self.reader.get_tensors([scale_key])[scale_key].to(
                self.device, torch.float32)
            try:
                from comfy_kitchen.tensor import (QuantizedTensor,
                                                  get_layout_class)
                cls = get_layout_class("TensorWiseINT8Layout")
                p = cls.Params(scale=scale, orig_dtype=self.dtype,
                               orig_shape=tuple(weight.shape))
                qt = QuantizedTensor(weight.to(self.device, torch.int8),
                                     "TensorWiseINT8Layout", p)
                weight = qt.dequantize().to(self.dtype)
            except Exception:
                # fallback: plain per-row dequantization
                w = weight.to(self.device, torch.float32)
                if weight.ndim == 2 and scale.numel() == weight.shape[0]:
                    w = w * scale.to(w.device).view(-1, 1)
                weight = w.to(self.dtype)
        self.model.embed_tokens.weight = nn.Parameter(
            weight, requires_grad=False)

    def _load_vision_resident(self) -> None:
        """Vision tower: load all weights to GPU once (small, ~1 GB for 32B)."""
        prefix = self._detect_vision_prefix()
        assert prefix is not None
        self.vision_prefix = prefix
        for name, p in self.vision_model.named_parameters():
            key = f"{prefix}visual.{name}"
            t = self.reader.get_tensors([key])[key].to(self.device, self.dtype)
            mod = self.vision_model.get_submodule(name.rsplit(".", 1)[0])
            mod._parameters[name.rsplit(".", 1)[1]] = nn.Parameter(
                t, requires_grad=False)
        for name, buf in self.vision_model.named_buffers():
            key = f"{prefix}visual.{name}"
            mod = self.vision_model.get_submodule(name.rsplit(".", 1)[0])
            leaf = name.rsplit(".", 1)[1]
            if self.reader.has(key):
                mod._buffers[leaf] = self.reader.get_tensors([key])[key].to(
                    self.device, self.dtype)
            elif name == "rotary_pos_emb.inv_freq":
                # non-persistent buffer: recompute from config (theta=10000.0)
                head_dim = (self.config.vision.hidden_size
                            // self.config.vision.num_heads)
                dim = head_dim // 2
                inv = 1.0 / 10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
                mod._buffers[leaf] = inv.to(self.device)

    # -- forward ---------------------------------------------------------------

    @torch.inference_mode()
    def encode(self, payload: TextEncoderInput | str,
               pool_mode: PoolMode = PoolMode.LAST) -> TextEncoderOutput:
        """Run one streaming forward pass (text, or image/video + text)."""
        if isinstance(payload, str):
            payload = TextEncoderInput(text=payload)
        has_vision = (payload.images is not None or payload.videos is not None
                      or payload.pixel_values is not None
                      or payload.pixel_values_videos is not None
                      or payload.minimax_ref_items is not None)
        if has_vision:
            return self._encode_multimodal(payload, pool_mode)
        return self._encode_text(payload, pool_mode)

    @torch.inference_mode()
    def encode_many(self, payloads, pool_modes=None) -> list[TextEncoderOutput]:
        """Run several payloads through the same streamed groups.

        Each decoder group is loaded once, applied to every payload, then
        released before the next group.  This is the memory-safe way to share
        one streaming pass between positive and negative conditioning.
        """
        if pool_modes is None:
            pool_modes = [PoolMode.LAST] * len(payloads)
        if len(payloads) == 1:
            return [self.encode(payloads[0], pool_modes[0])]

        prepared = [
            self._prepare_encode(payload, pool_mode)
            for payload, pool_mode in zip(payloads, pool_modes)
        ]
        hiddens = [
            self._run_streamed_many(
                [p["hidden"] for p in prepared],
                [p["position_embeddings"] for p in prepared],
                [p["attn_mask"] for p in prepared],
                [p["visual_pos_masks"] for p in prepared],
                [p["deepstack_visual_embeds"] for p in prepared],
            )
        ]
        outputs = []
        for hidden, p in zip(hiddens[0], prepared):
            hidden = self.model.norm(hidden)
            pooled = pool_hidden_states(
                hidden, p["pool_mask"], p["pool_mode"])
            outputs.append(TextEncoderOutput(
                last_hidden_state=hidden,
                pooled_embedding=pooled,
                input_ids=p["input_ids"],
                attention_mask=p["attention_mask"],
                token_tags=p.get("token_tags"),
            ))
        return outputs

    def _prepare_encode(self, payload, pool_mode: PoolMode) -> dict:
        """Build input embeds/masks for one text or multimodal payload."""
        has_vision = (payload.images is not None or payload.videos is not None
                      or payload.pixel_values is not None
                      or payload.pixel_values_videos is not None
                      or payload.minimax_ref_items is not None)
        if has_vision:
            if self.fusion is None:
                raise RuntimeError(
                    "checkpoint has no vision tower (no 'model.visual.*' "
                    "tensors); cannot process images/videos")
            input_ids, attention_mask, mm, tags = (
                self._prepare_multimodal_inputs(payload))
            fused = self.fusion.forward(
                input_ids=input_ids, attention_mask=attention_mask,
                pixel_values=mm.get("pixel_values"),
                image_grid_thw=mm.get("image_grid_thw"),
                pixel_values_videos=mm.get("pixel_values_videos"),
                video_grid_thw=mm.get("video_grid_thw"),
                mm_token_type_ids=mm.get("mm_token_type_ids"))
            hidden = fused.inputs_embeds
            position_embeddings = self.model.rotary_emb(
                hidden, fused.position_ids)
            attn_mask = build_attention_mask(
                attention_mask, input_ids.shape[1], self.device, self.dtype)
            pool_mask = attention_mask
            if pool_mask is not None and mm.get("mm_token_type_ids") is not None:
                pool_mask = (attention_mask.bool()
                             & (mm["mm_token_type_ids"] == 0))
            return dict(
                hidden=hidden,
                position_embeddings=position_embeddings,
                attn_mask=attn_mask,
                visual_pos_masks=fused.visual_pos_masks,
                deepstack_visual_embeds=fused.deepstack_visual_embeds,
                pool_mask=pool_mask,
                pool_mode=pool_mode,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_tags=tags,
            )
        input_ids, attention_mask = self._prepare_text_inputs(payload)
        hidden = self.model.embed_tokens(input_ids.to(self.device))
        position_embeddings = self.model.rotary_emb(
            hidden, build_position_ids(input_ids, self.device))
        attn_mask = build_attention_mask(
            attention_mask, input_ids.shape[1], self.device, self.dtype)
        return dict(
            hidden=hidden,
            position_embeddings=position_embeddings,
            attn_mask=attn_mask,
            visual_pos_masks=None,
            deepstack_visual_embeds=None,
            pool_mask=attention_mask,
            pool_mode=pool_mode,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_tags=None,
        )

    def _run_streamed_many(self, hiddens, position_embeddings, attn_masks,
                           visual_pos_masks, deepstack_visual_embeds):
        """Run one streamed pass over multiple hidden states group by group."""
        num_groups = len(self.streamer.groups)
        for g in range(num_groups):
            spec = self.streamer.groups[g]
            self.streamer.load_group(g)
            if self.stream_config.prefetch:
                self.streamer.prefetch_next(g + 1)
            hiddens = [
                self.model.run_layer_group(
                    h, spec.layer_start, spec.layer_end,
                    pe, am, vpm, ds)
                for h, pe, am, vpm, ds in zip(
                    hiddens, position_embeddings, attn_masks,
                    visual_pos_masks, deepstack_visual_embeds)
            ]
            self.streamer.release_group(g)
        return hiddens

    def _encode_text(self, payload: TextEncoderInput,
                     pool_mode: PoolMode) -> TextEncoderOutput:
        input_ids, attention_mask = self._prepare_text_inputs(payload)
        hidden = self.model.embed_tokens(input_ids.to(self.device))
        position_embeddings = self.model.rotary_emb(
            hidden, build_position_ids(input_ids, self.device))
        attn_mask = build_attention_mask(attention_mask, input_ids.shape[1],
                                         self.device, self.dtype)
        hidden = self._run_streamed(hidden, position_embeddings, attn_mask)
        hidden = self.model.norm(hidden)
        pooled = pool_hidden_states(hidden, attention_mask, pool_mode)
        return TextEncoderOutput(
            last_hidden_state=hidden, pooled_embedding=pooled,
            input_ids=input_ids, attention_mask=attention_mask)

    def _encode_multimodal(self, payload: TextEncoderInput,
                           pool_mode: PoolMode) -> TextEncoderOutput:
        if self.fusion is None:
            raise RuntimeError(
                "checkpoint has no vision tower (no 'model.visual.*' tensors); "
                "cannot process images/videos")
        input_ids, attention_mask, mm, tags = self._prepare_multimodal_inputs(payload)
        fused = self.fusion.forward(
            input_ids=input_ids, attention_mask=attention_mask,
            pixel_values=mm.get("pixel_values"),
            image_grid_thw=mm.get("image_grid_thw"),
            pixel_values_videos=mm.get("pixel_values_videos"),
            video_grid_thw=mm.get("video_grid_thw"),
            mm_token_type_ids=mm.get("mm_token_type_ids"))
        position_embeddings = self.model.rotary_emb(fused.inputs_embeds,
                                                    fused.position_ids)
        attn_mask = build_attention_mask(attention_mask, input_ids.shape[1],
                                         self.device, self.dtype)
        hidden = self._run_streamed(
            fused.inputs_embeds, position_embeddings, attn_mask,
            visual_pos_masks=fused.visual_pos_masks,
            deepstack_visual_embeds=fused.deepstack_visual_embeds)
        hidden = self.model.norm(hidden)
        text_mask = attention_mask
        if text_mask is not None and mm.get("mm_token_type_ids") is not None:
            text_mask = (attention_mask.bool()
                         & (mm["mm_token_type_ids"] == 0))
        pooled = pool_hidden_states(hidden, text_mask, pool_mode)
        return TextEncoderOutput(
            last_hidden_state=hidden, pooled_embedding=pooled,
            input_ids=input_ids, attention_mask=attention_mask,
            token_tags=tags)

    def _run_streamed(self, hidden: torch.Tensor,
                      position_embeddings, attn_mask,
                      visual_pos_masks=None,
                      deepstack_visual_embeds=None) -> torch.Tensor:
        num_groups = len(self.streamer.groups)
        if self.stream_config.loading_mode == LoadingMode.FULL.value:
            self.streamer.load_all()
            hidden = self.model.run_all_layers(
                hidden, position_embeddings, attn_mask,
                visual_pos_masks, deepstack_visual_embeds)
            self.streamer.release_all()
        else:
            for g in range(num_groups):
                spec = self.streamer.groups[g]
                self.streamer.load_group(g)
                if self.stream_config.prefetch:
                    self.streamer.prefetch_next(g + 1)
                hidden = self.model.run_layer_group(
                    hidden, spec.layer_start, spec.layer_end,
                    position_embeddings, attn_mask,
                    visual_pos_masks, deepstack_visual_embeds)
                self.streamer.release_group(g)
        return hidden

    def _prepare_text_inputs(self, payload: TextEncoderInput):
        if payload.input_ids is not None:
            input_ids = payload.input_ids
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            attention_mask = payload.attention_mask
            if attention_mask is not None and attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)
            return input_ids.long(), attention_mask
        if self.tokenizer is None:
            raise RuntimeError("no tokenizer available and input_ids not provided")
        enc = self.tokenizer(payload.text, return_tensors="pt",
                             truncation=True, max_length=payload.max_length,
                             padding=True)
        return enc["input_ids"], enc.get("attention_mask")

    def _prepare_multimodal_inputs(self, payload: TextEncoderInput):
        if payload.input_ids is not None:
            attention_mask = payload.attention_mask
            if attention_mask is not None and attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)
            mm = dict(pixel_values=payload.pixel_values,
                      image_grid_thw=payload.image_grid_thw,
                      pixel_values_videos=payload.pixel_values_videos,
                      video_grid_thw=payload.video_grid_thw,
                      mm_token_type_ids=payload.mm_token_type_ids)
            tags = getattr(payload, "token_tags", None)
            if tags is not None and tags.ndim == 1:
                tags = tags.unsqueeze(0)
            return payload.input_ids.long(), attention_mask, mm, tags
        if self.processor is None:
            raise RuntimeError(
                "no processor available; pass preprocessed tensors or an "
                "AutoProcessor to TextEncoder")
        if payload.minimax_ref_items is not None:
            return self._prepare_minimax_inputs(payload)
        kwargs = {}
        if payload.text:
            kwargs["text"] = payload.text
        if payload.images is not None:
            kwargs["images"] = payload.images
        if payload.videos is not None:
            kwargs["videos"] = payload.videos
        enc = self.processor(return_tensors="pt", truncation=True,
                             max_length=payload.max_length, **kwargs)
        mm = dict(pixel_values=enc.get("pixel_values"),
                  image_grid_thw=enc.get("image_grid_thw"),
                  pixel_values_videos=enc.get("pixel_values_videos"),
                  video_grid_thw=enc.get("video_grid_thw"),
                  mm_token_type_ids=enc.get("mm_token_type_ids"))
        return enc["input_ids"], enc.get("attention_mask"), mm, None

    def _prepare_minimax_inputs(self, payload: TextEncoderInput):
        """ref2va presentation (official minimax tokenizer contract).

        Builds the non-chat-template token sequence ``<Picture i>: `` +
        [VISION_START, IMAGE_PAD*N, VISION_END] (+ ``<Video k>: `` + per-2-frame
        ``<T.T seconds>`` blocks; ``<Audio j>: `` is text-only) then the prompt.
        Vision blocks are encoded by the vision tower (images repeat to fill the
        2-frame temporal patch, videos use real frame pairs) and merged into
        the sequence exactly like the ComfyUI PR's MiniMaxQwen3VL.
        """
        if self.tokenizer is None:
            raise RuntimeError("no tokenizer available for minimax ref items")
        tokenizer = self.tokenizer

        def _text_ids(s: str) -> list[int]:
            return tokenizer(s, add_special_tokens=False)["input_ids"]

        ids: list = []
        tags: list[int] = []          # 1=text, 0=vision pad (DiT modality)
        pixel_blocks = []             # flatten patches per vision block
        grids = []                    # grid_thw per block
        mm_types: list[int] = []      # 0=text, 1=image
        counters = {"image": 0, "audio": 0, "video": 0}

        def add_text(s: str) -> None:
            t = _text_ids(s)
            ids.extend(t)
            tags.extend([1] * len(t))
            mm_types.extend([0] * len(t))

        def add_vision_block(flatten, grid) -> None:
            # Vision tower merges SPATIAL_MERGE_SIZE x SPATIAL_MERGE_SIZE
            # patches before the embeddings enter the text sequence.
            pad_count = (int(grid[0, 1].item() * grid[0, 2].item())
                         // (SPATIAL_MERGE_SIZE ** 2))
            ids.append(VISION_START)
            ids.extend([IMAGE_PAD] * pad_count)
            ids.append(VISION_END)
            tags.extend([0] * (pad_count + 2))
            mm_types.extend([0] + [1] * pad_count + [0])
            pixel_blocks.append(flatten)
            grids.append(grid)

        for item in payload.minimax_ref_items:
            kind = item["type"]
            counters[kind] += 1
            if kind == "image":
                add_text("<Picture %d>: " % counters["image"])
                flatten, grid = process_image_block(item["data"])
                add_vision_block(flatten, grid)
            elif kind == "audio":
                add_text("<Audio %d>: " % counters["audio"])
            elif kind == "video":
                frames = item["data"]                    # [T, H, W, C]
                timestamps = item.get("timestamps")
                if timestamps is None:
                    timestamps = [i / 2.0 for i in range(frames.shape[0])]
                if frames.shape[0] % 2 == 1:
                    frames = torch.cat([frames, frames[-1:]], dim=0)
                    timestamps = list(timestamps) + [timestamps[-1]]
                add_text("<Video %d>: " % counters["video"])
                for i in range(0, frames.shape[0], 2):
                    block_ts = (timestamps[i] + timestamps[i + 1]) / 2.0
                    add_text("<%.1f seconds>" % block_ts)
                    flatten, grid = process_video_block(frames[i:i + 2])
                    add_vision_block(flatten, grid)

        add_text(payload.text)
        if not ids:
            ids.append(self.config.text.pad_token_id or 0)
            tags.append(1)
            mm_types.append(0)

        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        mm_token_type_ids = torch.tensor([mm_types], dtype=torch.long,
                                         device=self.device)
        token_tags = torch.tensor([tags], dtype=torch.long, device=self.device)
        pixel_values = (torch.cat(pixel_blocks, dim=0) if pixel_blocks else None)
        image_grid_thw = (torch.cat(grids, dim=0) if grids else None)
        mm = dict(pixel_values=pixel_values,
                  image_grid_thw=image_grid_thw,
                  pixel_values_videos=None,
                  video_grid_thw=None,
                  mm_token_type_ids=mm_token_type_ids)
        return input_ids, attention_mask, mm, token_tags

    # -- teardown --------------------------------------------------------------

    def destroy(self) -> None:
        """Free streamer threads, prefetch buffers, shard handles, all weights."""
        self._destroyed = True
        self.streamer.shutdown()
        self.reader.close()
        if hasattr(self, "model") and self.model is not None:
            for mod in self.model.modules():
                for name in list(mod._parameters.keys()):
                    mod._parameters[name] = nn.Parameter(torch.empty(0),
                                                         requires_grad=False)
                for name in list(mod._buffers.keys()):
                    mod._buffers[name] = torch.empty(0)
        if self.vision_model is not None:
            for mod in self.vision_model.modules():
                for name in list(mod._parameters.keys()):
                    mod._parameters[name] = nn.Parameter(torch.empty(0),
                                                         requires_grad=False)
                for name in list(mod._buffers.keys()):
                    mod._buffers[name] = torch.empty(0)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.destroy()
        return False
