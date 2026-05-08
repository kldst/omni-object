# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Callable, Any

from omnivggt.layers import PatchEmbed
from omnivggt.layers.block import Block
from omnivggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from omnivggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in OniVGGT: Omni-Modality Driven Visual Geometry Grounded Transformer.


    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        patch_embed_pretrained_path=None,
        load_patch_embed_from_hub=True,
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        enable_checkpoint=True
    ):
        super().__init__()

        self.__build_patch_embed__(
            patch_embed,
            img_size,
            patch_size,
            num_register_tokens,
            embed_dim=embed_dim,
            patch_embed_pretrained_path=patch_embed_pretrained_path,
            load_patch_embed_from_hub=load_patch_embed_from_hub,
        )

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.use_checkpoint = enable_checkpoint
        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name,
                torch.FloatTensor(value).view(1, 1, 3, 1, 1),
                persistent=False,
            )

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
        patch_embed_pretrained_path=None,
        load_patch_embed_from_hub=True,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            # if hasattr(self.patch_embed, "mask_token"):
            #     self.patch_embed.mask_token.requires_grad_(False)
            
            if patch_embed_pretrained_path:
                if not os.path.isfile(patch_embed_pretrained_path):
                    raise FileNotFoundError(
                        f"patch_embed_pretrained_path does not exist: {patch_embed_pretrained_path}"
                    )
                logger.info("Loading patch embed weights from local file: %s", patch_embed_pretrained_path)
                state_dict = torch.load(patch_embed_pretrained_path, map_location="cpu")
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                self.patch_embed.load_state_dict(state_dict, strict=False)
            elif load_patch_embed_from_hub:
                if patch_embed == "dinov2_vitl14_reg":
                    dinov2_pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
                    self.patch_embed.load_state_dict(dinov2_pretrained.state_dict(), strict=False)
                elif patch_embed == "dinov2_vitb14_reg":
                    dinov2_pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
                    self.patch_embed.load_state_dict(dinov2_pretrained.state_dict(), strict=True)
                elif patch_embed == "dinov2_vits14_reg":
                    dinov2_pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
                    self.patch_embed.load_state_dict(dinov2_pretrained.state_dict(), strict=True)
                elif patch_embed == "dinov2_vitg2_reg":
                    dinov2_pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg2_reg')
                    self.patch_embed.load_state_dict(dinov2_pretrained.state_dict(), strict=True)
            else:
                logger.info("Skipping torch.hub patch embed preload for %s", patch_embed)
                

    def forward(
        self,
        images: torch.Tensor,
        *,
        layer_postprocessor: Optional[Callable[[int, torch.Tensor, int], torch.Tensor]] = None,
        return_layer_tokens: bool = False,
        layer_token_indices: Optional[List[int]] = None,
        collect_output_list: bool = True,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, _, H, W = images.shape
        patch_tokens = self.embed_images(images)
        return self.forward_from_patch_tokens(
            patch_tokens,
            batch_size=B,
            seq_len=S,
            height=H,
            width=W,
            layer_postprocessor=layer_postprocessor,
            return_layer_tokens=return_layer_tokens,
            layer_token_indices=layer_token_indices,
            collect_output_list=collect_output_list,
        )

    def embed_images(self, images: torch.Tensor) -> torch.Tensor:
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        return patch_tokens

    def prepare_tokens_from_patch_tokens(
        self,
        patch_tokens: torch.Tensor,
        *,
        batch_size: int,
        seq_len: int,
        height: int,
        width: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B = batch_size
        S = seq_len

        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, height // self.patch_size, width // self.patch_size, device=patch_tokens.device)

        if self.patch_start_idx > 0 and pos is not None:
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2, device=patch_tokens.device, dtype=pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        return tokens, pos

    def forward_from_patch_tokens(
        self,
        patch_tokens: torch.Tensor,
        *,
        batch_size: int,
        seq_len: int,
        height: int,
        width: int,
        layer_postprocessor: Optional[Callable[[int, torch.Tensor, int], torch.Tensor]] = None,
        return_layer_tokens: bool = False,
        layer_token_indices: Optional[List[int]] = None,
        collect_output_list: bool = True,
    ) -> Tuple[List[torch.Tensor], int]:
        B = batch_size
        S = seq_len
        tokens, pos = self.prepare_tokens_from_patch_tokens(
            patch_tokens,
            batch_size=B,
            seq_len=S,
            height=height,
            width=width,
        )

        _, P, C = tokens.shape
        frame_idx = 0
        global_idx = 0
        output_list = [] if collect_output_list else None
        selected_layer_indices = None if layer_token_indices is None else set(int(idx) for idx in layer_token_indices)
        layer_tokens: Any = None
        if return_layer_tokens:
            layer_tokens = [] if selected_layer_indices is None else {}
        logical_layer_idx = 0
        last_attn_type = self.aa_order[-1]

        for _ in range(self.aa_block_num):
            for _ in range(self.aa_block_size):
                frame_tokens = None
                global_tokens = None

                for attn_type in self.aa_order:
                    if attn_type == "frame":
                        tokens, frame_idx, frame_tokens = self._process_frame_attention(tokens, B, S, P, C, frame_idx, pos=pos)
                    elif attn_type == "global":
                        tokens, global_idx, global_tokens = self._process_global_attention(tokens, B, S, P, C, global_idx, pos=pos)
                    else:
                        raise ValueError(f"Unknown attention type: {attn_type}")

                if frame_tokens is None and global_tokens is None:
                    raise RuntimeError("Aggregator step produced no tokens")
                if frame_tokens is None:
                    frame_tokens = global_tokens
                if global_tokens is None:
                    global_tokens = frame_tokens

                current_tokens = global_tokens if last_attn_type == "global" else frame_tokens
                if layer_postprocessor is not None:
                    current_tokens = layer_postprocessor(logical_layer_idx, current_tokens, self.patch_start_idx)
                    if current_tokens.shape != (B, S, P, C):
                        raise ValueError(
                            f"layer_postprocessor must return shape {(B, S, P, C)}, got {tuple(current_tokens.shape)}"
                        )
                    if last_attn_type == "global":
                        global_tokens = current_tokens
                        tokens = current_tokens.reshape(B, S * P, C)
                    else:
                        frame_tokens = current_tokens
                        tokens = current_tokens.reshape(B * S, P, C)

                if collect_output_list:
                    output_list.append(torch.cat([frame_tokens, global_tokens], dim=-1))

                if return_layer_tokens and (selected_layer_indices is None or logical_layer_idx in selected_layer_indices):
                    if selected_layer_indices is None:
                        layer_tokens.append(current_tokens)
                    else:
                        layer_tokens[logical_layer_idx] = current_tokens

                logical_layer_idx += 1

        if return_layer_tokens:
            return output_list, self.patch_start_idx, layer_tokens
        return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        blk = self.frame_blocks[frame_idx]
        if self.use_checkpoint and self.training:
            tokens = checkpoint(
                lambda inp, p: blk(inp, pos=p,),
                tokens,
                pos,
                use_reentrant=False
            )
        else:
            tokens = blk(tokens, pos=pos,)
        frame_idx += 1

        return tokens, frame_idx, tokens.view(B, S, P, C)

    def _process_global_attention(self, tokens, B, S, P, C, global_idx,
                                  pos=None, pose_encoding=None, depth_encoding=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        blk = self.global_blocks[global_idx]   
        if self.use_checkpoint and self.training:
            tokens = checkpoint(
                lambda inp, p: blk(inp, pos=p, ),        
                tokens,
                pos,
                use_reentrant=False
            )
        else:
            tokens = blk(tokens, pos=pos, )

        global_idx += 1

        return tokens, global_idx, tokens.view(B, S, P, C)
    
def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
