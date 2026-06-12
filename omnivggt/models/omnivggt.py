from collections import OrderedDict
from contextlib import nullcontext

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from omnivggt.heads.object_pose_head import ObjectPoseHead, ObjectPoseHeadConfig
from omnivggt.heads.object_mask_head import ObjectMaskHead
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
                 enable_object_mask=False,
                 enable_object_srt=False,
                 enable_object_size=True,
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
                 disable_object_prototype_pooler=False,
                 freeze_object_encoder=False,
                 object_prototype_object_encoder_no_grad=False,
                 object_cross_attn_heads=16,
                 object_encode_cache=False,
                 object_encode_cache_max=256):
        super().__init__()
        # Object-encoder token cache: the reference-image ViT forward is the most
        # expensive part of object encoding and only depends on the (frozen) shared
        # encoder, so its per-layer tokens can be cached and reused across batches
        # and epochs. Keyed by object id. Requires object_prototype_object_encoder_no_grad
        # and a frozen object encoder, and deterministic reference images
        # (object_ref_color_jitter=False). The trainable prototype poolers still run
        # live on the cached tokens, so their gradients are unaffected.
        self.object_encode_cache = bool(object_encode_cache)
        self.object_encode_cache_max = int(object_encode_cache_max)
        self._object_token_cache = OrderedDict()

        self.aggregator = ZeroAggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
                                         pose_hidden_dim = 9, cam_drop_prob=cam_drop_prob, depth_drop_prob=depth_drop_prob,
                                         always_use_depth_gt=always_use_depth_gt,
                                         patch_embed_pretrained_path=patch_embed_pretrained_path,
                                         load_patch_embed_from_hub=load_patch_embed_from_hub)
        # Optional separate, frozen encoder used ONLY for object reference images. The
        # scene aggregator above is trained (object/pose/mask losses), which would
        # otherwise drift the object-reference encoding step over step, since object and
        # scene share the same weights. With a dedicated frozen copy, object encodings
        # stay fixed throughout training. Its weights must be copied from `aggregator`
        # after the checkpoint load and then frozen (see build_model in train_utils.py).
        # NOTE: this roughly doubles backbone parameter memory.
        self.freeze_object_encoder = bool(freeze_object_encoder)
        self.object_aggregator = None
        if self.freeze_object_encoder:
            self.object_aggregator = ZeroAggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
                                                    pose_hidden_dim=9, cam_drop_prob=cam_drop_prob, depth_drop_prob=depth_drop_prob,
                                                    always_use_depth_gt=always_use_depth_gt,
                                                    patch_embed_pretrained_path=patch_embed_pretrained_path,
                                                    load_patch_embed_from_hub=load_patch_embed_from_hub)
        self.enable_multi_layer_object_prototype_cross_attn = bool(enable_multi_layer_object_prototype_cross_attn)
        self.object_prototype_layer_indices = tuple(int(idx) for idx in object_prototype_layer_indices)
        self.object_prototype_num_tokens = int(object_prototype_num_tokens)
        # When True, skip the ObjectPrototypePool entirely: the cross-attention blocks
        # use the raw (flattened) object patch tokens as context instead of the 32
        # pooled prototypes. Heavier (context grows from num_tokens to S_obj*P_obj per
        # layer) but no learned compression in between. object_prototype_num_tokens is
        # ignored in this mode.
        self.disable_object_prototype_pooler = bool(disable_object_prototype_pooler)
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
            if not self.disable_object_prototype_pooler:
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
        self.object_mask_head = (
            ObjectMaskHead(dim_in=2 * embed_dim, patch_size=patch_size)
            if enable_object_mask
            else None
        )
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
                predict_size=enable_object_size,
            )
            self.object_srt_head = ObjectPoseHead(
                dim_in=2 * embed_dim,
                object_pose_cfg=object_pose_cfg,
                context_pool=object_pose_context_pool,
                use_global_scene_object_concat=object_pose_use_global_scene_object_concat,
                predict_size=enable_object_size,
            )

    def _ensure_batched_images(self, images: torch.Tensor): # 確保 batch 維度存在
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
        object_patch_tokens = object_layer_tokens[:, :, object_patch_start_idx:, :]
        if object_patch_tokens.numel() == 0:
            raise ValueError("Object patch tokens are empty; cannot build object prototypes")
        if self.disable_object_prototype_pooler:
            # No pooling: use the raw object patch tokens (all views concatenated) as
            # the cross-attention context. (B, S_obj, P_obj, C) -> (B, S_obj*P_obj, C).
            B, S_obj, P_obj, C = object_patch_tokens.shape
            return object_patch_tokens.reshape(B, S_obj * P_obj, C)
        if self.object_prototype_poolers is None:
            raise RuntimeError("object_prototype_poolers is not initialized")
        return self.object_prototype_poolers[str(layer_idx)](object_patch_tokens)

    def _compute_object_layer_tokens(self, object_images: torch.Tensor, requested_layers):
        """Run the (frozen) shared encoder over reference images and return
        ``(patch_start_idx, {layer_idx: tokens (B, S_obj, P, C)})``. This is the
        expensive part that the cache stores."""
        B, S_obj, _, H_obj, W_obj = object_images.shape
        # Use the dedicated frozen object encoder when enabled, otherwise the shared
        # (trainable) scene aggregator.
        encoder = self.object_aggregator if self.object_aggregator is not None else self.aggregator
        object_patch_tokens = encoder.embed_images(object_images)
        ctx = torch.no_grad() if self.object_prototype_object_encoder_no_grad else nullcontext()
        with ctx:
            _, object_patch_start_idx, object_layer_tokens = encoder.forward_from_patch_tokens(
                object_patch_tokens,
                batch_size=B,
                seq_len=S_obj,
                height=H_obj,
                width=W_obj,
                return_layer_tokens=True,
                layer_token_indices=requested_layers,
                collect_output_list=False,
            )
        return object_patch_start_idx, object_layer_tokens

    def _cache_put(self, key, entry):
        cache = self._object_token_cache
        if key in cache:
            cache.move_to_end(key)
            return
        cache[key] = entry
        while len(cache) > self.object_encode_cache_max:
            cache.popitem(last=False)  # evict least-recently-used

    def clear_object_cache(self):
        self._object_token_cache.clear()

    def _gather_object_layer_tokens_cached(self, object_images, keys, requested_layers):
        """Return per-batch-row layer tokens, encoding only object ids missing
        from the cache (deduplicated within the batch)."""
        cache = self._object_token_cache
        missing_unique = OrderedDict()
        for i, key in enumerate(keys):
            if key not in cache and key not in missing_unique:
                missing_unique[key] = i
        if missing_unique:
            rows = list(missing_unique.values())
            sub_images = object_images[rows]
            patch_start_idx, sub_layer_tokens = self._compute_object_layer_tokens(sub_images, requested_layers)
            for pos, key in enumerate(missing_unique.keys()):
                self._cache_put(
                    key,
                    {
                        "patch_start_idx": int(patch_start_idx),
                        "layers": {L: sub_layer_tokens[L][pos].detach() for L in requested_layers},
                    },
                )
        for key in keys:
            cache.move_to_end(key)

        patch_start_idx = cache[keys[0]]["patch_start_idx"]
        layer_tokens = {
            L: torch.stack([cache[k]["layers"][L].to(object_images.device) for k in keys], dim=0)
            for L in requested_layers
        }
        return patch_start_idx, layer_tokens

    def _encode_object_prototypes(self, object_images: torch.Tensor, object_ids=None):
        selected_layers = self._resolve_object_prototype_layer_indices(self.aggregator.depth)
        requested_layers = tuple(dict.fromkeys((*selected_layers, self.aggregator.depth - 1)))

        use_cache = (
            self.object_encode_cache
            and self.object_prototype_object_encoder_no_grad
            and object_ids is not None
        )
        if use_cache:
            keys = self._object_ids_to_keys(object_ids)
            object_patch_start_idx, object_layer_tokens = self._gather_object_layer_tokens_cached(
                object_images, keys, requested_layers
            )
        else:
            object_patch_start_idx, object_layer_tokens = self._compute_object_layer_tokens(
                object_images, requested_layers
            )

        prototypes_by_idx = {
            layer_idx: self._build_object_prototypes(object_layer_tokens[layer_idx], object_patch_start_idx, layer_idx)
            for layer_idx in selected_layers
        }
        final_object_tokens = object_layer_tokens[self.aggregator.depth - 1][:, :, object_patch_start_idx:, :]
        return prototypes_by_idx, final_object_tokens

    @staticmethod
    def _object_ids_to_keys(object_ids):
        if object_ids is None:
            return None
        if torch.is_tensor(object_ids):
            return [str(int(x)) for x in object_ids.reshape(-1).tolist()]
        return [str(int(x)) if not isinstance(x, str) else x for x in object_ids]

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
        object_ids=None,
    ):
        images = self._ensure_batched_images(images)
        object_images = self._ensure_batched_images(object_images)

        object_prototypes_by_idx = None
        object_patch_tokens = None
        need_object_encoder = object_images is not None and (
            self.enable_multi_layer_object_prototype_cross_attn or self.object_srt_head is not None
        )
        if need_object_encoder:
            object_prototypes_by_idx, object_patch_tokens = self._encode_object_prototypes(object_images, object_ids)

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

            if self.object_mask_head is not None:
                predictions.update(
                    self.object_mask_head(
                        aggregated_tokens_list,
                        images=images,
                        patch_start_idx=patch_start_idx,
                    )
                )

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
        object_ids=None,
    ):
        images = self._ensure_batched_images(images)
        object_images = self._ensure_batched_images(object_images)

        object_prototypes_by_idx = None
        object_patch_tokens = None
        need_object_encoder = object_images is not None and (
            self.enable_multi_layer_object_prototype_cross_attn or self.object_srt_head is not None
        )
        if need_object_encoder:
            object_prototypes_by_idx, object_patch_tokens = self._encode_object_prototypes(object_images, object_ids)

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

            if self.object_mask_head is not None:
                predictions.update(
                    self.object_mask_head(
                        aggregated_tokens_list,
                        images=images,
                        patch_start_idx=patch_start_idx,
                    )
                )

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
