import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional, Callable, Any

from omnivggt.layers import PatchEmbed
from omnivggt.layers.block import Block
from omnivggt.utils.pose_enc import extri_intri_to_pose_encoding
from torch.utils.checkpoint import checkpoint
from omnivggt.utils.geometry import closed_form_inverse_se3
from omnivggt.models.aggregator import Aggregator, slice_expand_and_flatten

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

class ZeroAggregator(Aggregator):
    def __init__(self, img_size=518, 
                 patch_size=14, 
                 embed_dim=1024, 
                 depth=24, 
                 num_heads=16, 
                 mlp_ratio=4, 
                 num_register_tokens=4, 
                 block_fn=Block, 
                 pose_hidden_dim=9,
                 cam_drop_prob=0.1,
                 depth_drop_prob=0.1,
                 always_use_depth_gt=False,
                 patch_embed_pretrained_path=None,
                 load_patch_embed_from_hub=True,
                 qkv_bias=True, 
                 proj_bias=True, 
                 ffn_bias=True, 
                 patch_embed="dinov2_vitl14_reg", 
                 aa_order=["frame", "global"], 
                 aa_block_size=1, 
                 qk_norm=True, 
                 rope_freq=100, 
                 init_values=0.01,
                 enable_checkpoint=True):
        super().__init__(img_size, 
                         patch_size, 
                         embed_dim, 
                         depth, 
                         num_heads, 
                         mlp_ratio, 
                         num_register_tokens, 
                         block_fn,
                         qkv_bias, 
                         proj_bias, 
                         ffn_bias, 
                         patch_embed,
                         patch_embed_pretrained_path,
                         load_patch_embed_from_hub,
                         aa_order, 
                         aa_block_size, 
                         qk_norm, 
                         rope_freq, 
                         init_values)
        
        
        self.cam_drop_prob = cam_drop_prob
        self.depth_drop_prob = depth_drop_prob
        self.always_use_depth_gt = bool(always_use_depth_gt)
        self.patch_start_idx = 1 + num_register_tokens
        self.depth_placeholder = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.use_checkpoint = enable_checkpoint
        self.num_groups = self.aa_block_num + 1
        self.pose_embeddings   = nn.ModuleList()
        self.camera_adapters   = nn.ModuleList()

        for _ in range(self.num_groups):
            # pose_embedding
            pose_emb = nn.Linear(pose_hidden_dim, embed_dim)
            
            # camera adapter (zero init)
            cam_adapt = nn.Linear(embed_dim, embed_dim, bias=True)
            nn.init.zeros_(cam_adapt.weight)
            nn.init.zeros_(cam_adapt.bias)

            self.pose_embeddings.append(pose_emb)
            self.camera_adapters.append(cam_adapt)
            
        self.depth_patch_embed = PatchEmbed(img_size=img_size,
                                            patch_size=patch_size,
                                            in_chans=2,
                                            embed_dim=embed_dim)
        
    def _match_dtype(self, x, reference):
        return x.to(dtype=reference.dtype, device=reference.device)
    
    def normalize_extrinsics(self, extrinsics):
        B, S, _, _ = extrinsics.shape
        device = extrinsics.device
        extrinsics_homog = torch.cat(
            [
                extrinsics,
                torch.zeros((B, S, 1, 4), device=device),
            ],
            dim=-2,
        )
        extrinsics_homog[:, :, -1, -1] = 1.0
        first_cam_extrinsic_inv = closed_form_inverse_se3(extrinsics_homog[:, 0])
        new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv.unsqueeze(1))  # (B,N,4,4)
        
        if S > 1:
            cam_centers = new_extrinsics[:, :, :3, 3]  # (B, S, 3)
            ref_cam = cam_centers[:, 0:1, :]  # (B,1,3)
            rel_distances = torch.norm(cam_centers - ref_cam, dim=-1)[:,1:]  # (B, S)
            scale = rel_distances.mean(dim=1, keepdim=True).clamp(min=1e-6)  # (B, 1)
            new_extrinsics[:, :, :3, 3] /= scale.unsqueeze(-1)
        return new_extrinsics[:, :, :3]
    
    def normalize_depth(self, depth, mask, eps=1e-8):
        """
        depth: [B, V, H, W, 1]
        mask:  [B, V, H, W]
        """
        assert depth.shape[:4] == mask.shape, "mask and depth must have the same first four dimensions"

        B, V, H, W, _ = depth.shape
        depth_squeezed = depth.squeeze(-1)
        norm = torch.zeros_like(depth_squeezed)

        for b in range(B):
            valid = depth_squeezed[b][mask[b] > 0]
            if valid.numel() == 0:
                continue

            mean = valid.mean()
            norm_b = depth_squeezed[b] / (mean + eps)

            norm[b] = norm_b * mask[b]

        return norm.unsqueeze(-1)
    
    def select_camera_gt(self, S, cam_drop_prob=0.1, rng=None):
        rng = rng or np.random.default_rng()

        if rng.random() < cam_drop_prob:
            return []

        k = rng.integers(0, S + 1)
        if k == 0:
            return []

        # 按顺序从 0 开始选取 k 个
        idx = list(range(k))

        return idx
    
    def select_depth_gt(self, S, depth_drop_prob=0.1, rng=None):
        rng = rng or np.random.default_rng()

        if self.always_use_depth_gt:
            return list(range(S))

        if rng.random() < depth_drop_prob:
            return []

        k = rng.integers(0, S + 1)
        if k == 0:
            return []

        idx = rng.choice(S, size=k, replace=False)

        return sorted(idx.tolist())

    def _normalize_index_list(self, indices: Optional[List[int]]) -> List[int]:
        if indices is None:
            return []
        return [int(idx) for idx in indices]

    def _build_camera_condition_tokens(
        self,
        *,
        batch_size: int,
        seq_len: int,
        token_count: int,
        token_dim: int,
        height: int,
        width: int,
        extrinsics: Optional[torch.Tensor],
        intrinsics: Optional[torch.Tensor],
        camera_gt_index: List[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        if len(camera_gt_index) == 0:
            return None, torch.zeros(token_count, 1, token_dim, device=device, dtype=dtype)

        if extrinsics is None or intrinsics is None:
            raise ValueError("extrinsics and intrinsics must be provided when camera_gt_index is not empty")

        camera_gt_length = len(camera_gt_index)
        camera_idx_tensor = torch.tensor(camera_gt_index, device=device, dtype=torch.long)
        extrinsics_selected = torch.index_select(extrinsics, dim=1, index=camera_idx_tensor)
        intrinsics_selected = torch.index_select(intrinsics, dim=1, index=camera_idx_tensor)

        extrinsics_gt_normalized = self.normalize_extrinsics(extrinsics_selected)
        pose_encoding = extri_intri_to_pose_encoding(
            extrinsics=extrinsics_gt_normalized,
            intrinsics=intrinsics_selected,
            image_size_hw=(height, width),
            pose_encoding_type="absT_quaR_FoV",
        )
        gt_camera_token = self.pose_embeddings[0](pose_encoding).view(batch_size * camera_gt_length, token_dim).unsqueeze(1)

        camera_full = torch.zeros(token_count, 1, token_dim, device=device, dtype=dtype)
        camera_rows = (
            torch.arange(batch_size, device=device).unsqueeze(1) * seq_len + camera_idx_tensor.unsqueeze(0)
        ).reshape(-1)
        camera_full[camera_rows] = gt_camera_token.to(dtype=dtype)
        return pose_encoding, camera_full

    def _build_depth_condition_tokens(
        self,
        *,
        batch_size: int,
        seq_len: int,
        token_count: int,
        patch_count: int,
        token_dim: int,
        height: int,
        width: int,
        patch_tokens: torch.Tensor,
        depth: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        depth_gt_index: List[int],
        device: torch.device,
    ) -> torch.Tensor:
        if len(depth_gt_index) == 0:
            return self.depth_placeholder.expand(token_count, patch_count, token_dim)

        if depth is None or mask is None:
            raise ValueError("depth and mask must be provided when depth_gt_index is not empty")

        depth_gt_length = len(depth_gt_index)
        idx_tensor = torch.tensor(depth_gt_index, device=device, dtype=torch.long)

        depth_selected = torch.index_select(depth, dim=1, index=idx_tensor)
        mask_selected = torch.index_select(mask, dim=1, index=idx_tensor)

        depth_gt_normalized = self.normalize_depth(depth_selected, mask_selected)
        depth_gt_normalized = depth_gt_normalized.view(batch_size * depth_gt_length, 1, height, width)
        mask_selected = mask_selected.view(batch_size * depth_gt_length, 1, height, width)

        depthmaps = torch.cat([depth_gt_normalized, mask_selected], dim=1)
        depthmaps = self._match_dtype(depthmaps, self.depth_patch_embed.proj.weight)
        gt_depth_token = self.depth_patch_embed(depthmaps)

        depth_full = self.depth_placeholder.expand(token_count, patch_count, token_dim).clone()
        rows = (torch.arange(batch_size, device=device).unsqueeze(1) * seq_len + idx_tensor.unsqueeze(0)).reshape(-1)
        depth_full[rows] = gt_depth_token.to(dtype=patch_tokens.dtype)
        return depth_full

    def prepare_tokens_from_patch_tokens(
        self,
        patch_tokens: torch.Tensor,
        *,
        batch_size: int,
        seq_len: int,
        height: int,
        width: int,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        camera_gt_index: Optional[List[int]] = None,
        depth_gt_index: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        B = batch_size
        S = seq_len
        K, P, C = patch_tokens.shape
        device = patch_tokens.device

        camera_gt_index = self._normalize_index_list(camera_gt_index)
        depth_gt_index = self._normalize_index_list(depth_gt_index)

        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        pose_encoding, gt_camera_token = self._build_camera_condition_tokens(
            batch_size=B,
            seq_len=S,
            token_count=K,
            token_dim=C,
            height=height,
            width=width,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            camera_gt_index=camera_gt_index,
            device=device,
            dtype=camera_token.dtype,
        )
        gt_depth_token = self._build_depth_condition_tokens(
            batch_size=B,
            seq_len=S,
            token_count=K,
            patch_count=P,
            token_dim=C,
            height=height,
            width=width,
            patch_tokens=patch_tokens,
            depth=depth,
            mask=mask,
            depth_gt_index=depth_gt_index,
            device=device,
        )

        camera_token = camera_token + self.camera_adapters[0](gt_camera_token)
        patch_tokens = patch_tokens + gt_depth_token
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, height // self.patch_size, width // self.patch_size, device=device)

        if self.patch_start_idx > 0 and pos is not None:
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2, device=device, dtype=pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        rollout_state = {
            "pose_encoding": pose_encoding,
            "camera_gt_index": camera_gt_index,
            "register_shape": register_token.shape,
            "patch_token_count": P,
        }
        return tokens, pos, rollout_state

    def _build_frame_injection_tokens(
        self,
        *,
        batch_size: int,
        seq_len: int,
        token_dim: int,
        index: int,
        camera_gt_index: List[int],
        pose_encoding: Optional[torch.Tensor],
        register_shape: torch.Size,
        patch_token_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        token_count = batch_size * seq_len
        register_token = torch.zeros(register_shape, device=device, dtype=dtype).expand(token_count, -1, -1)

        if len(camera_gt_index) != 0:
            camera_gt_length = len(camera_gt_index)
            camera_idx_tensor = torch.tensor(camera_gt_index, device=device, dtype=torch.long)
            gt_camera_token = self.pose_embeddings[index](pose_encoding).view(batch_size * camera_gt_length, token_dim).unsqueeze(1)
            camera_full = torch.zeros(token_count, 1, token_dim, device=device, dtype=gt_camera_token.dtype)
            camera_rows = (
                torch.arange(batch_size, device=device).unsqueeze(1) * seq_len + camera_idx_tensor.unsqueeze(0)
            ).reshape(-1)
            camera_full[camera_rows] = gt_camera_token.to(dtype=camera_full.dtype)
        else:
            camera_full = torch.zeros(token_count, 1, token_dim, device=device, dtype=dtype)

        depth_injection = torch.zeros(token_count, patch_token_count, token_dim, device=device, dtype=dtype)
        camera_injection = self.camera_adapters[index](camera_full).to(dtype=dtype)
        return torch.cat([camera_injection, register_token, depth_injection], dim=1)

    def forward_from_patch_tokens(
        self,
        patch_tokens: torch.Tensor,
        *,
        batch_size: int,
        seq_len: int,
        height: int,
        width: int,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        camera_gt_index: Optional[List[int]] = None,
        depth_gt_index: Optional[List[int]] = None,
        layer_postprocessor: Optional[Callable[[int, torch.Tensor, int], torch.Tensor]] = None,
        return_layer_tokens: bool = False,
        layer_token_indices: Optional[List[int]] = None,
        collect_output_list: bool = True,
    ) -> Tuple[List[torch.Tensor], int]:
        B = batch_size
        S = seq_len
        tokens, pos, rollout_state = self.prepare_tokens_from_patch_tokens(
            patch_tokens,
            batch_size=B,
            seq_len=S,
            height=height,
            width=width,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            depth=depth,
            mask=mask,
            camera_gt_index=camera_gt_index,
            depth_gt_index=depth_gt_index,
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

        for group_idx in range(self.aa_block_num):
            frame_injection_tokens = self._build_frame_injection_tokens(
                batch_size=B,
                seq_len=S,
                token_dim=C,
                index=group_idx + 1,
                camera_gt_index=rollout_state["camera_gt_index"],
                pose_encoding=rollout_state["pose_encoding"],
                register_shape=rollout_state["register_shape"],
                patch_token_count=rollout_state["patch_token_count"],
                device=tokens.device,
                dtype=tokens.dtype,
            )

            for _ in range(self.aa_block_size):
                frame_tokens = None
                global_tokens = None

                for attn_type in self.aa_order:
                    if attn_type == "frame":
                        tokens, frame_idx, frame_tokens = self._process_frame_attention(
                            tokens,
                            B,
                            S,
                            P,
                            C,
                            frame_idx,
                            pos=pos,
                            injection_tokens=frame_injection_tokens,
                        )
                    elif attn_type == "global":
                        tokens, global_idx, global_tokens = self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos
                        )
                    else:
                        raise ValueError(f"Unknown attention type: {attn_type}")

                if frame_tokens is None and global_tokens is None:
                    raise RuntimeError("ZeroAggregator step produced no tokens")
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
    
    def forward(self, images: torch.Tensor, 
                extrinsics: torch.Tensor, 
                intrinsics: torch.Tensor,
                depth: torch.Tensor,
                mask: torch.Tensor,
                *,
                layer_postprocessor: Optional[Callable[[int, torch.Tensor, int], torch.Tensor]] = None,
                return_layer_tokens: bool = False,
                layer_token_indices: Optional[List[int]] = None,
                collect_output_list: bool = True) -> Tuple[List[torch.Tensor], int]:
        B, S, _, H, W = images.shape
        patch_tokens = self.embed_images(images)
        camera_gt_index = self.select_camera_gt(S, self.cam_drop_prob)
        depth_gt_index = self.select_depth_gt(S, self.depth_drop_prob)
        return self.forward_from_patch_tokens(
            patch_tokens,
            batch_size=B,
            seq_len=S,
            height=H,
            width=W,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            depth=depth,
            mask=mask,
            camera_gt_index=camera_gt_index,
            depth_gt_index=depth_gt_index,
            layer_postprocessor=layer_postprocessor,
            return_layer_tokens=return_layer_tokens,
            layer_token_indices=layer_token_indices,
            collect_output_list=collect_output_list,
        )


    def inference(self, images: torch.Tensor, 
                extrinsics: torch.Tensor, 
                intrinsics: torch.Tensor,
                depth: torch.Tensor,
                mask: torch.Tensor,
                depth_gt_index: List[int],
                camera_gt_index: List[int],
                *,
                layer_postprocessor: Optional[Callable[[int, torch.Tensor, int], torch.Tensor]] = None,
                return_layer_tokens: bool = False,
                layer_token_indices: Optional[List[int]] = None,
                collect_output_list: bool = True) -> Tuple[List[torch.Tensor], int]:
        B, S, _, H, W = images.shape
        patch_tokens = self.embed_images(images)
        return self.forward_from_patch_tokens(
            patch_tokens,
            batch_size=B,
            seq_len=S,
            height=H,
            width=W,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            depth=depth,
            mask=mask,
            camera_gt_index=self._normalize_index_list(camera_gt_index),
            depth_gt_index=self._normalize_index_list(depth_gt_index),
            layer_postprocessor=layer_postprocessor,
            return_layer_tokens=return_layer_tokens,
            layer_token_indices=layer_token_indices,
            collect_output_list=collect_output_list,
        )

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None, injection_tokens=None):
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

        if injection_tokens is not None:
            tokens = tokens + injection_tokens

        frame_idx += 1
        return tokens, frame_idx, tokens.view(B, S, P, C)
