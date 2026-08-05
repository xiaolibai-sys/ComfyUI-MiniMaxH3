"""Image/video <-> text fusion for the streaming Qwen3-VL encoder.

Ported from ``transformers`` Qwen3VLModel (get_image_features /
get_placeholder_mask / get_rope_index / compute_3d_position_ids).  This module
turns processor outputs (input_ids + pixel_values + grid_thw +
mm_token_type_ids) into a merged embedding sequence with 3D MRoPE position
ids, which the streaming text decoder then consumes.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Optional

import torch

from .config import Qwen3VLConfig
from .modeling import Qwen3VLTextModel
from .vision import Qwen3VLVisionModel


@dataclass
class FusedInputs:
    """Everything the streaming text decoder needs for one multimodal forward."""
    inputs_embeds: torch.Tensor            # (B, L, hidden) merged text+vision
    position_ids: torch.Tensor             # (3, B, L) MRoPE ids
    visual_pos_masks: Optional[torch.Tensor] = None   # (B, L) bool
    deepstack_visual_embeds: Optional[list[torch.Tensor]] = None


class Qwen3VLMultimodalFusion:
    def __init__(self, config: Qwen3VLConfig,
                 vision_model: Qwen3VLVisionModel,
                 text_model: Qwen3VLTextModel):
        self.config = config
        self.vision = vision_model
        self.text = text_model

    # -- vision features -------------------------------------------------------

    def get_image_features(self, pixel_values: torch.Tensor,
                           image_grid_thw: torch.Tensor):
        vision_output = self.vision(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1)
                       // self.vision.spatial_merge_size ** 2).tolist()
        image_embeds = torch.split(vision_output.pooler_output, split_sizes)
        return image_embeds, vision_output.deepstack_features

    def get_video_features(self, pixel_values_videos: torch.Tensor,
                           video_grid_thw: torch.Tensor):
        return self.get_image_features(pixel_values_videos, video_grid_thw)

    # -- placeholder masks ------------------------------------------------------

    def get_placeholder_mask(self, input_ids: torch.Tensor,
                             image_features=None, video_features=None):
        special_image_mask = input_ids == self.config.image_token_id
        special_video_mask = input_ids == self.config.video_token_id
        n_image = special_image_mask.sum().item()
        n_video = special_video_mask.sum().item()
        if image_features is not None and n_image != image_features.shape[0]:
            raise ValueError(
                f"image tokens {n_image} vs features {image_features.shape[0]} mismatch")
        if video_features is not None and n_video != video_features.shape[0]:
            raise ValueError(
                f"video tokens {n_video} vs features {video_features.shape[0]} mismatch")
        return (special_image_mask.unsqueeze(-1).to(input_ids.device),
                special_video_mask.unsqueeze(-1).to(input_ids.device))

    # -- 3D MRoPE positions -----------------------------------------------------

    def get_vision_position_ids(self, start_position: int, grid_thw: torch.Tensor,
                                temp_merge_size: int = 1,
                                spatial_merge_size: int = 1,
                                time_interval: int = 1,
                                device=None) -> torch.Tensor:
        """(3, n) t/h/w positions for one image/video, offset by start_position."""
        llm_grid_t = int(grid_thw[0].item() // temp_merge_size)
        llm_grid_h = int(grid_thw[1].item() // spatial_merge_size)
        llm_grid_w = int(grid_thw[2].item() // spatial_merge_size)
        position_temporal = torch.arange(llm_grid_t, device=device) * time_interval
        position_height = torch.arange(llm_grid_h, device=device) + start_position
        position_width = torch.arange(llm_grid_w, device=device) + start_position
        t_grid, h_grid, w_grid = torch.meshgrid(
            position_temporal, position_height, position_width, indexing="ij")
        vision_position_ids = torch.stack(
            [t_grid, h_grid, w_grid], dim=0).reshape(3, -1)
        vision_position_ids[0] += start_position
        return vision_position_ids

    def get_rope_index(self, input_ids: torch.Tensor,
                       mm_token_type_ids: torch.Tensor,
                       image_grid_thw: Optional[torch.Tensor] = None,
                       video_grid_thw: Optional[torch.Tensor] = None,
                       attention_mask: Optional[torch.Tensor] = None):
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(
                video_grid_thw, video_grid_thw[:, 0], dim=0).clone()
            video_grid_thw[:, 0] = 1
        spatial_merge_size = self.config.vision.spatial_merge_size

        deltas = []
        position_ids = torch.zeros(3, input_ids.shape[0], input_ids.shape[1],
                                   dtype=input_ids.dtype, device=input_ids.device)
        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }
        for batch_idx, current_input_ids in enumerate(input_ids):
            input_token_type = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                mask = attention_mask[batch_idx].bool()
                current_input_ids = current_input_ids[mask]
                input_token_type = input_token_type[mask]

            groups = []
            for key, group in itertools.groupby(
                    enumerate(input_token_type.tolist()), lambda x: x[1]):
                group = list(group)
                groups.append((key, group[0][0], group[-1][0] + 1))

            current_pos = 0
            parts = []
            for modality, start_idx, end_idx in groups:
                if modality == 0:  # text
                    text_len = end_idx - start_idx
                    parts.append(torch.arange(
                        text_len, device=input_ids.device).view(1, -1).expand(3, -1)
                        + current_pos)
                    current_pos += text_len
                else:  # image(1) / video(2)
                    grid_thw = next(grid_iters[modality])
                    parts.append(self.get_vision_position_ids(
                        current_pos, grid_thw, 1, spatial_merge_size,
                        device=input_ids.device))
                    current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size

            llm_positions = torch.cat(parts, dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[:, batch_idx, mask] = llm_positions.to(position_ids.device)
            else:
                position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
            deltas.append(llm_positions.max() + 1 - current_input_ids.shape[0])
        deltas = torch.tensor(deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, deltas

    def compute_3d_position_ids(self, input_ids, inputs_embeds,
                                image_grid_thw=None, video_grid_thw=None,
                                attention_mask=None,
                                mm_token_type_ids=None) -> Optional[torch.Tensor]:
        has_multimodal = image_grid_thw is not None or video_grid_thw is not None
        if has_multimodal and mm_token_type_ids is None:
            raise ValueError("mm_token_type_ids is required for MRoPE")
        if input_ids is not None and mm_token_type_ids is not None and has_multimodal:
            position_ids, _ = self.get_rope_index(
                input_ids, mm_token_type_ids, image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw, attention_mask=attention_mask)
            return position_ids
        return None

    # -- combined forward --------------------------------------------------------

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                pixel_values=None, image_grid_thw=None,
                pixel_values_videos=None, video_grid_thw=None,
                mm_token_type_ids=None) -> FusedInputs:
        device = self.text.embed_tokens.weight.device
        dtype = self.text.embed_tokens.weight.dtype
        input_ids = input_ids.to(device)
        inputs_embeds = self.text.embed_tokens(input_ids)

        image_embeds = deep_image = None
        video_embeds = deep_video = None
        image_mask = video_mask = None

        if pixel_values is not None:
            image_embeds, deep_image = self.get_image_features(
                pixel_values.to(device=device, dtype=dtype),
                image_grid_thw.to(device))
            image_embeds = torch.cat(image_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deep_video = self.get_video_features(
                pixel_values_videos.to(device=device, dtype=dtype),
                video_grid_thw.to(device))
            video_embeds = torch.cat(video_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, video_features=video_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # deepstack visual embeddings + masks (exactly as transformers)
        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deep_image, deep_video):
                embed_joint = img_embed.new_zeros(
                    int(visual_pos_masks.sum()), img_embed.shape[-1])
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deep_image
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deep_video

        position_ids = self.compute_3d_position_ids(
            input_ids, inputs_embeds, image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw, attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids)

        return FusedInputs(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds)
