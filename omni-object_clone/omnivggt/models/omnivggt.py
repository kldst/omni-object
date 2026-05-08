import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from omnivggt.heads.object_pose_head import ObjectPoseHead, ObjectPoseHeadConfig
from omnivggt.models.omnivggt_aggregator import ZeroAggregator
from omnivggt.heads.camera_head import CameraHead
from omnivggt.heads.dpt_head import DPTHead


class ObjectTokenCrossAttentionBlock(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.query_norm = nn.LayerNorm(query_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            kdim=context_dim,
            vdim=context_dim,
            batch_first=True,
        )
        hidden_dim = int(query_dim * mlp_ratio)
        self.mlp_norm = nn.LayerNorm(query_dim)
        self.mlp = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, query_dim),
        )

    def forward(self, query_tokens: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        context_tokens = self.context_norm(context_tokens)
        attn_out, _ = self.cross_attn(
            self.query_norm(query_tokens),
            context_tokens,
            context_tokens,
            need_weights=False,
        )
        query_tokens = query_tokens + attn_out
        query_tokens = query_tokens + self.mlp(self.mlp_norm(query_tokens))
        return query_tokens


class ObjectPrototypePool(nn.Module):
    def __init__(self, dim: int, num_prototypes: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.prototype_queries = nn.Parameter(torch.randn(1, num_prototypes, dim))
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        nn.init.normal_(self.prototype_queries, std=1e-6)

    def forward(self, object_patch_tokens: torch.Tensor) -> torch.Tensor:
        B, S_obj, P_obj, C = object_patch_tokens.shape
        context_tokens = self.context_norm(object_patch_tokens.reshape(B, S_obj * P_obj, C))
        prototype_queries = self.prototype_queries.expand(B, -1, -1)
        attn_out, _ = self.cross_attn(
            self.query_norm(prototype_queries),
            context_tokens,
            context_tokens,
            need_weights=False,
        )
        prototype_queries = prototype_queries + attn_out
        prototype_queries = prototype_queries + self.mlp(self.mlp_norm(prototype_queries))
        return prototype_queries


class OmniVGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, cam_drop_prob=0.1, depth_drop_prob=0.1,
                 always_use_depth_gt=False,
                 patch_embed_pretrained_path=None,
                 load_patch_embed_from_hub=True,
                 enable_camera=True, enable_depth=True, enable_point=True,
                 enable_object_srt=False,
                 object_pose_context_pool="flatten",
                 object_pose_use_global_scene_object_concat=False,
                 object_pose_transformer_depth=6,
                 object_pose_transformer_heads=8,
                 object_pose_transformer_mlp_dim=1024,
                 object_pose_transformer_dim_head=64,
                 object_pose_transformer_dropout=0.0,
                 object_pose_transformer_emb_dropout=0.0,
                 object_pose_transformer_norm="layer",
                 object_pose_transformer_dim=1024,
                 object_pose_ief_iters=1,
                 object_pose_init_params_path=None,
                 enable_multi_layer_object_prototype_cross_attn=False,
                 object_prototype_layer_indices=(4, 11, 17, 23),
                 object_prototype_num_tokens=4,
                 object_prototype_object_encoder_no_grad=False,
                 object_cross_attn_heads=16):
        super().__init__()

        self.aggregator = ZeroAggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, 
                                         pose_hidden_dim = 9, cam_drop_prob=cam_drop_prob, depth_drop_prob=depth_drop_prob,
                                         always_use_depth_gt=always_use_depth_gt,
                                         patch_embed_pretrained_path=patch_embed_pretrained_path,
                                         load_patch_embed_from_hub=load_patch_embed_from_hub)
        self.enable_multi_layer_object_prototype_cross_attn = bool(enable_multi_layer_object_prototype_cross_attn)
        self.object_prototype_layer_indices = tuple(int(idx) for idx in object_prototype_layer_indices)
        self.object_prototype_num_tokens = int(object_prototype_num_tokens)
        self.object_prototype_object_encoder_no_grad = bool(object_prototype_object_encoder_no_grad)
        self.object_token_cross_attn_blocks = None
        self.object_prototype_poolers = None
        if self.enable_multi_layer_object_prototype_cross_attn:
            progressive_layer_indices = self._resolve_object_prototype_layer_indices(self.aggregator.depth)
            self.object_token_cross_attn_blocks = nn.ModuleDict(
                {
                    str(layer_idx): ObjectTokenCrossAttentionBlock(
                        query_dim=embed_dim,
                        context_dim=embed_dim,
                        num_heads=object_cross_attn_heads,
                        mlp_ratio=4.0,
                    )
                    for layer_idx in progressive_layer_indices
                }
            )
            self.object_prototype_poolers = nn.ModuleDict(
                {
                    str(layer_idx): ObjectPrototypePool(
                        dim=embed_dim,
                        num_prototypes=self.object_prototype_num_tokens,
                        num_heads=object_cross_attn_heads,
                        mlp_ratio=4.0,
                    )
                    for layer_idx in progressive_layer_indices
                }
            )
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.object_srt_head = None
        if enable_object_srt:
            object_pose_cfg = ObjectPoseHeadConfig(
                transformer_depth=object_pose_transformer_depth,
                transformer_heads=object_pose_transformer_heads,
                transformer_mlp_dim=object_pose_transformer_mlp_dim,
                transformer_dim_head=object_pose_transformer_dim_head,
                transformer_dropout=object_pose_transformer_dropout,
                transformer_emb_dropout=object_pose_transformer_emb_dropout,
                transformer_norm=object_pose_transformer_norm,
                transformer_dim=object_pose_transformer_dim,
                ief_iters=object_pose_ief_iters,
                init_params_path=object_pose_init_params_path,
            )
            self.object_srt_head = ObjectPoseHead(
                dim_in=2 * embed_dim,
                object_pose_cfg=object_pose_cfg,
                context_pool=object_pose_context_pool,
                use_global_scene_object_concat=object_pose_use_global_scene_object_concat,
            )

    def _ensure_batched_images(self, images: torch.Tensor):
        if images is None:
            return None
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        return images

    def _resolve_object_prototype_layer_indices(self, num_layers: int):
        resolved_indices = []
        for layer_idx in self.object_prototype_layer_indices:
            resolved_idx = layer_idx if layer_idx >= 0 else num_layers + layer_idx
            if resolved_idx < 0 or resolved_idx >= num_layers:
                raise ValueError(
                    f"object_prototype_layer_indices contains invalid layer index {layer_idx} for {num_layers} layers"
                )
            resolved_indices.append(resolved_idx)
        return tuple(dict.fromkeys(resolved_indices))

    def _build_object_prototypes(self, object_layer_tokens: torch.Tensor, object_patch_start_idx: int, layer_idx: int):
        if self.object_prototype_poolers is None:
            raise RuntimeError("object_prototype_poolers is not initialized")
        object_patch_tokens = object_layer_tokens[:, :, object_patch_start_idx:, :]
        if object_patch_tokens.numel() == 0:
            raise ValueError("Object patch tokens are empty; cannot build object prototypes")
        return self.object_prototype_poolers[str(layer_idx)](object_patch_tokens)

    def _encode_object_prototypes(self, object_images: torch.Tensor):
        B, S_obj, _, H_obj, W_obj = object_images.shape
        object_patch_tokens = self.aggregator.embed_images(object_images)

        selected_layers = self._resolve_object_prototype_layer_indices(self.aggregator.depth)
        requested_layers = tuple(dict.fromkeys((*selected_layers, self.aggregator.depth - 1)))
        if self.object_prototype_object_encoder_no_grad:
            with torch.no_grad():
                _, object_patch_start_idx, object_layer_tokens = self.aggregator.forward_from_patch_tokens(
                    object_patch_tokens,
                    batch_size=B,
                    seq_len=S_obj,
                    height=H_obj,
                    width=W_obj,
                    return_layer_tokens=True,
                    layer_token_indices=requested_layers,
                    collect_output_list=False,
                )
        else:
            _, object_patch_start_idx, object_layer_tokens = self.aggregator.forward_from_patch_tokens(
                object_patch_tokens,
                batch_size=B,
                seq_len=S_obj,
                height=H_obj,
                width=W_obj,
                return_layer_tokens=True,
                layer_token_indices=requested_layers,
                collect_output_list=False,
            )

        prototypes_by_idx = {
            layer_idx: self._build_object_prototypes(object_layer_tokens[layer_idx], object_patch_start_idx, layer_idx)
            for layer_idx in selected_layers
        }
        final_object_tokens = object_layer_tokens[self.aggregator.depth - 1][:, :, object_patch_start_idx:, :]
        return prototypes_by_idx, final_object_tokens

    def _apply_progressive_object_prototype_cross_attention(
        self,
        layer_idx: int,
        scene_layer_tokens: torch.Tensor,
        scene_patch_start_idx: int,
        object_prototypes_by_idx,
    ):
        if self.object_token_cross_attn_blocks is None or layer_idx not in object_prototypes_by_idx:
            return scene_layer_tokens

        scene_special_tokens = scene_layer_tokens[:, :, :scene_patch_start_idx, :]
        scene_patch_tokens = scene_layer_tokens[:, :, scene_patch_start_idx:, :]
        if scene_patch_tokens.numel() == 0:
            return scene_layer_tokens

        object_prototypes = object_prototypes_by_idx[layer_idx]
        if scene_layer_tokens.shape[0] != object_prototypes.shape[0]:
            raise ValueError(
                f"Scene/object batch size mismatch at layer {layer_idx}: "
                f"{scene_layer_tokens.shape[0]} vs {object_prototypes.shape[0]}"
            )
        if scene_layer_tokens.shape[-1] != object_prototypes.shape[-1]:
            raise ValueError(
                f"Scene/object channel mismatch at layer {layer_idx}: "
                f"{scene_layer_tokens.shape[-1]} vs {object_prototypes.shape[-1]}"
            )

        B_scene, S_scene, P_scene, C_scene = scene_patch_tokens.shape
        scene_query = scene_patch_tokens.reshape(B_scene, S_scene * P_scene, C_scene)
        fused_scene_query = self.object_token_cross_attn_blocks[str(layer_idx)](scene_query, object_prototypes)
        fused_scene_patch_tokens = fused_scene_query.view(B_scene, S_scene, P_scene, C_scene)
        return torch.cat([scene_special_tokens, fused_scene_patch_tokens], dim=2)

    def forward(
        self,
        images: torch.Tensor,
        object_images: torch.Tensor = None,
        extrinsics: torch.Tensor = None,
        intrinsics: torch.Tensor = None,
        depth: torch.Tensor = None,
        mask: torch.Tensor = None,
    ):
        images = self._ensure_batched_images(images)
        object_images = self._ensure_batched_images(object_images)

        object_prototypes_by_idx = None
        object_patch_tokens = None
        need_object_encoder = object_images is not None and (
            self.enable_multi_layer_object_prototype_cross_attn or self.object_srt_head is not None
        )
        if need_object_encoder:
            object_prototypes_by_idx, object_patch_tokens = self._encode_object_prototypes(object_images)

        if self.enable_multi_layer_object_prototype_cross_attn and object_prototypes_by_idx is not None:
            def progressive_object_fusion(layer_idx, scene_layer_tokens, scene_patch_start_idx):
                return self._apply_progressive_object_prototype_cross_attention(
                    layer_idx,
                    scene_layer_tokens,
                    scene_patch_start_idx,
                    object_prototypes_by_idx,
                )

            aggregated_tokens_list, patch_start_idx = self.aggregator(
                images=images,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                depth=depth,
                mask=mask,
                layer_postprocessor=progressive_object_fusion,
            )
        else:
            aggregated_tokens_list, patch_start_idx = self.aggregator(images = images, 
                                                                      extrinsics = extrinsics, 
                                                                      intrinsics = intrinsics,
                                                                      depth = depth,
                                                                      mask = mask,)
                            
        B, S, C_in, H, W = images.shape
        predictions = {}
        
        with torch.amp.autocast('cuda', enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            if self.object_srt_head is not None:
                predictions.update(
                    self.object_srt_head(
                        aggregated_tokens_list,
                        patch_start_idx=patch_start_idx,
                        object_tokens=object_patch_tokens,
                    )
                )
                if "object_size_log" in predictions:
                    predictions["object_size"] = torch.exp(predictions["object_size_log"])


        predictions["images"] = images  # store the images for visualization during inference
        if object_patch_tokens is not None:
            predictions["object_patch_tokens"] = object_patch_tokens

        return predictions
    
    
    def inference(self,
        images: torch.Tensor,
        object_images: torch.Tensor = None,
        extrinsics: torch.Tensor = None,
        intrinsics: torch.Tensor = None,
        depth: torch.Tensor = None,
        mask: torch.Tensor = None,
        depth_gt_index: list = None,
        camera_gt_index: list = None,
    ):
        images = self._ensure_batched_images(images)
        object_images = self._ensure_batched_images(object_images)

        object_prototypes_by_idx = None
        object_patch_tokens = None
        need_object_encoder = object_images is not None and (
            self.enable_multi_layer_object_prototype_cross_attn or self.object_srt_head is not None
        )
        if need_object_encoder:
            object_prototypes_by_idx, object_patch_tokens = self._encode_object_prototypes(object_images)

        if self.enable_multi_layer_object_prototype_cross_attn and object_prototypes_by_idx is not None:
            def progressive_object_fusion(layer_idx, scene_layer_tokens, scene_patch_start_idx):
                return self._apply_progressive_object_prototype_cross_attention(
                    layer_idx,
                    scene_layer_tokens,
                    scene_patch_start_idx,
                    object_prototypes_by_idx,
                )

            aggregated_tokens_list, patch_start_idx = self.aggregator.inference(images = images, 
                                                                                extrinsics = extrinsics, 
                                                                                intrinsics = intrinsics,
                                                                                depth = depth,
                                                                                mask = mask,
                                                                                depth_gt_index = depth_gt_index,
                                                                                camera_gt_index = camera_gt_index,
                                                                                layer_postprocessor=progressive_object_fusion)
        else:
            aggregated_tokens_list, patch_start_idx = self.aggregator.inference(images = images, 
                                                                                extrinsics = extrinsics, 
                                                                                intrinsics = intrinsics,
                                                                                depth = depth,
                                                                                mask = mask,
                                                                                depth_gt_index = depth_gt_index,
                                                                                camera_gt_index = camera_gt_index)
        
        B, S, C_in, H, W = images.shape
        predictions = {}

        with torch.amp.autocast('cuda', enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            if self.object_srt_head is not None:
                predictions.update(
                    self.object_srt_head(
                        aggregated_tokens_list,
                        patch_start_idx=patch_start_idx,
                        object_tokens=object_patch_tokens,
                    )
                )
                if "object_size_log" in predictions:
                    predictions["object_size"] = torch.exp(predictions["object_size_log"])


        predictions["images"] = images  # store the images for visualization during inference
        if object_patch_tokens is not None:
            predictions["object_patch_tokens"] = object_patch_tokens

        return predictions
