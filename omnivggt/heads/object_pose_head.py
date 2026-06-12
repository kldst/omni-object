from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from omnivggt.heads.pose_transformer import TransformerDecoder


@dataclass(frozen=True)
class ObjectPoseHeadConfig:
    transformer_depth: int = 6
    transformer_heads: int = 8
    transformer_mlp_dim: int = 1024
    transformer_dim_head: int = 64
    transformer_dropout: float = 0.0
    transformer_emb_dropout: float = 0.0
    transformer_norm: str = "layer"
    transformer_dim: int = 1024
    ief_iters: int = 1
    init_params_path: Optional[str] = None
    predict_size: bool = True


def _default_init_params_path() -> Optional[str]:
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "init_6dpose" / "init_6dpose_params_identity_zero_translate.npz"
    return str(candidate) if candidate.is_file() else None


def _load_global_init_params(init_params_path: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    if init_params_path is None:
        init_params_path = _default_init_params_path()
    if init_params_path is None:
        return (
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        )

    data = np.load(init_params_path)
    init_translate = data.get("global_init_translate")
    init_rot6d = data.get("global_init_rot6d")
    if init_translate is None or init_rot6d is None:
        return (
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        )

    return (
        init_translate.astype(np.float32).reshape(-1)[:3],
        init_rot6d.astype(np.float32).reshape(-1)[:6],
    )


class ObjectQueryPooler(nn.Module):
    """Attention-pool the frozen-encoder object patch tokens into a fixed set of
    object-conditioned query vectors for the pose decoder.

    Single-layer cross-attention + MLP (mirrors ObjectPrototypePool in
    omnivggt.models.omnivggt, but kept local here to avoid a circular import and so
    its parameters ride the pose head's optimizer group). The K/V side of the
    aggregator cross-attention is unaffected -- this only produces the decoder Q.
    """

    def __init__(self, token_dim: int, query_dim: int, num_queries: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, query_dim) * 0.02)
        self.query_norm = nn.LayerNorm(query_dim)
        self.context_norm = nn.LayerNorm(token_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            kdim=token_dim,
            vdim=token_dim,
            batch_first=True,
        )
        hidden_dim = int(query_dim * mlp_ratio)
        self.mlp_norm = nn.LayerNorm(query_dim)
        self.mlp = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, query_dim),
        )

    def forward(self, object_tokens: torch.Tensor) -> torch.Tensor:
        # object_tokens: (B, S_obj, P_obj, token_dim) -> (B, num_queries, query_dim)
        batch_size, num_views, num_patches, channels = object_tokens.shape
        context = self.context_norm(object_tokens.reshape(batch_size, num_views * num_patches, channels))
        queries = self.queries.expand(batch_size, -1, -1)
        attn_out, _ = self.cross_attn(self.query_norm(queries), context, context, need_weights=False)
        queries = queries + attn_out
        queries = queries + self.mlp(self.mlp_norm(queries))
        return queries


class AttentionPool(nn.Module):
    """Aggregate N decoder output tokens into a single vector via a learnable query."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size = tokens.shape[0]
        query = self.query.expand(batch_size, -1, -1)
        context = self.norm(tokens)
        out, _ = self.attn(query, context, context, need_weights=False)
        return out.squeeze(1)


class ObjectPoseTransformerDecoderHead(nn.Module):
    """Predict object rotation-6D, camera-frame translation, and log metric size."""

    def __init__(
        self,
        *,
        context_dim: int,
        cfg: Optional[ObjectPoseHeadConfig] = None,
        use_object_queries: bool = False,
        num_object_queries: int = 1,
        query_aggregation: str = "attention_pool",
        attn_supervise_layers: Tuple[int, ...] = (),
    ):
        super().__init__()
        self.cfg = cfg or ObjectPoseHeadConfig()
        self.predict_size = bool(self.cfg.predict_size)
        self.use_object_queries = bool(use_object_queries)
        self.num_object_queries = int(num_object_queries)
        self.query_aggregation = query_aggregation
        self.attn_supervise_layers = tuple(int(layer_idx) for layer_idx in attn_supervise_layers)

        init_translate, init_rot6d = _load_global_init_params(self.cfg.init_params_path)
        self.register_buffer("init_translate", torch.from_numpy(init_translate).unsqueeze(0))
        self.register_buffer("init_pose", torch.from_numpy(init_rot6d).unsqueeze(0))
        if self.predict_size:
            self.register_buffer("init_size_log", torch.zeros(1, 3, dtype=torch.float32))

        if self.use_object_queries:
            # Object queries (B, num_object_queries, transformer_dim) are fed directly as
            # decoder tokens, so the token embedding is bypassed (token_dim == dim).
            num_decoder_tokens = self.num_object_queries
            self.pose_token = None
            if self.query_aggregation == "pose_token":
                self.pose_token = nn.Parameter(torch.randn(1, 1, self.cfg.transformer_dim) * 0.02)
                num_decoder_tokens = self.num_object_queries + 1
            self.transformer = TransformerDecoder(
                num_tokens=num_decoder_tokens,
                token_dim=self.cfg.transformer_dim,
                dim=self.cfg.transformer_dim,
                depth=self.cfg.transformer_depth,
                heads=self.cfg.transformer_heads,
                mlp_dim=self.cfg.transformer_mlp_dim,
                dim_head=self.cfg.transformer_dim_head,
                dropout=self.cfg.transformer_dropout,
                emb_dropout=self.cfg.transformer_emb_dropout,
                norm=self.cfg.transformer_norm,
                context_dim=context_dim,
                skip_token_embedding=True,
            )
            self.attn_pool = (
                AttentionPool(self.cfg.transformer_dim, num_heads=self.cfg.transformer_heads)
                if self.query_aggregation == "attention_pool"
                else None
            )
        else:
            self.pose_token = None
            self.attn_pool = None
            self.transformer = TransformerDecoder(
                num_tokens=1,
                token_dim=1,
                dim=self.cfg.transformer_dim,
                depth=self.cfg.transformer_depth,
                heads=self.cfg.transformer_heads,
                mlp_dim=self.cfg.transformer_mlp_dim,
                dim_head=self.cfg.transformer_dim_head,
                dropout=self.cfg.transformer_dropout,
                emb_dropout=self.cfg.transformer_emb_dropout,
                norm=self.cfg.transformer_norm,
                context_dim=context_dim,
            )

        self.decpose = nn.Linear(self.cfg.transformer_dim, 6)
        self.dectranslate = nn.Linear(self.cfg.transformer_dim, 3)
        self.decsize = nn.Linear(self.cfg.transformer_dim, 3) if self.predict_size else None
        self.presence_branch = nn.Linear(self.cfg.transformer_dim, 1)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.dectranslate.weight, gain=0.01)
        if self.decsize is not None:
            nn.init.xavier_uniform_(self.decsize.weight, gain=0.01)
        nn.init.xavier_uniform_(self.presence_branch.weight, gain=0.01)

    def _aggregate_query_outputs(self, tokens_out: torch.Tensor) -> torch.Tensor:
        if self.query_aggregation == "pose_token":
            return tokens_out[:, -1, :]
        query_outputs = tokens_out[:, : self.num_object_queries, :]
        if self.attn_pool is not None:
            return self.attn_pool(query_outputs)
        return query_outputs.mean(dim=1)

    def forward(self, context_tokens: torch.Tensor, object_queries: Optional[torch.Tensor] = None):
        batch_size = context_tokens.shape[0]
        pred_pose = self.init_pose.expand(batch_size, -1)
        pred_translate = self.init_translate.expand(batch_size, -1)
        pred_size_log = self.init_size_log.expand(batch_size, -1) if self.predict_size else None
        presence_logits = None
        attn_maps = None

        # Only materialize attention maps when they will actually be supervised, to
        # avoid holding the (B, L, num_queries, S*P) tensor at eval / when disabled.
        need_attn = self.use_object_queries and len(self.attn_supervise_layers) > 0 and self.training

        for _ in range(int(self.cfg.ief_iters)):
            if self.use_object_queries:
                if object_queries is None:
                    raise ValueError("object_queries must be provided when use_object_queries=True")
                tokens_in = object_queries
                if self.pose_token is not None:
                    tokens_in = torch.cat([object_queries, self.pose_token.expand(batch_size, -1, -1)], dim=1)
                if need_attn:
                    tokens_out, layer_attn = self.transformer(tokens_in, context=context_tokens, return_attn=True)
                    selected = torch.stack([layer_attn[i] for i in self.attn_supervise_layers], dim=1)
                    # Keep only the object-query rows (drop the pose token row if present).
                    attn_maps = selected[:, :, : self.num_object_queries, :]
                else:
                    tokens_out = self.transformer(tokens_in, context=context_tokens)
                token_out = self._aggregate_query_outputs(tokens_out)
            else:
                token = torch.zeros((batch_size, 1, 1), device=context_tokens.device, dtype=context_tokens.dtype)
                token_out = self.transformer(token, context=context_tokens).squeeze(1)
            pred_pose = self.decpose(token_out) + pred_pose
            pred_translate = self.dectranslate(token_out) + pred_translate
            if self.decsize is not None:
                pred_size_log = self.decsize(token_out) + pred_size_log
            presence_logits = self.presence_branch(token_out).squeeze(-1)

        return pred_pose, pred_translate, pred_size_log, presence_logits, attn_maps


class ObjectPoseHead(nn.Module):
    def __init__(
        self,
        *,
        dim_in: int,
        object_pose_cfg: Optional[ObjectPoseHeadConfig] = None,
        context_pool: str = "flatten",
        use_global_scene_object_concat: bool = False,
        predict_size: bool = True,
        enable_query_pooler: bool = False,
        num_object_queries: int = 32,
        query_aggregation: str = "attention_pool",
        attn_supervise_layers: Tuple[int, ...] = (),
        object_token_dim: Optional[int] = None,
    ):
        super().__init__()
        if object_pose_cfg is not None and object_pose_cfg.predict_size != bool(predict_size):
            object_pose_cfg = ObjectPoseHeadConfig(
                transformer_depth=object_pose_cfg.transformer_depth,
                transformer_heads=object_pose_cfg.transformer_heads,
                transformer_mlp_dim=object_pose_cfg.transformer_mlp_dim,
                transformer_dim_head=object_pose_cfg.transformer_dim_head,
                transformer_dropout=object_pose_cfg.transformer_dropout,
                transformer_emb_dropout=object_pose_cfg.transformer_emb_dropout,
                transformer_norm=object_pose_cfg.transformer_norm,
                transformer_dim=object_pose_cfg.transformer_dim,
                ief_iters=object_pose_cfg.ief_iters,
                init_params_path=object_pose_cfg.init_params_path,
                predict_size=bool(predict_size),
            )
        self.context_pool = context_pool
        self.use_global_scene_object_concat = bool(use_global_scene_object_concat)
        self.enable_query_pooler = bool(enable_query_pooler)
        decoder_context_dim = 2 * dim_in if self.use_global_scene_object_concat else dim_in
        cfg_for_decoder = object_pose_cfg or ObjectPoseHeadConfig()

        # Query pooler: layer-23 object tokens (token_dim, default = embed_dim = dim_in//2)
        # -> num_object_queries object-conditioned queries in the decoder's dim.
        self.query_pooler = None
        if self.enable_query_pooler:
            token_dim = int(object_token_dim) if object_token_dim is not None else dim_in // 2
            self.query_pooler = ObjectQueryPooler(
                token_dim=token_dim,
                query_dim=cfg_for_decoder.transformer_dim,
                num_queries=int(num_object_queries),
                num_heads=cfg_for_decoder.transformer_heads,
            )

        self.decoder = ObjectPoseTransformerDecoderHead(
            context_dim=decoder_context_dim,
            cfg=object_pose_cfg,
            use_object_queries=self.enable_query_pooler,
            num_object_queries=int(num_object_queries),
            query_aggregation=query_aggregation,
            attn_supervise_layers=attn_supervise_layers,
        )

    def forward(
        self,
        aggregated_tokens_list,
        patch_start_idx: int,
        object_latent: Optional[torch.Tensor] = None,
        object_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        tokens = aggregated_tokens_list[-1]
        patch_tokens = tokens[:, :, patch_start_idx:, :]

        # Pool layer-23 object tokens into object-conditioned decoder queries. Cast to
        # the scene-token dtype (fp32 in the head's autocast-disabled region) so the
        # bf16 frozen-encoder tokens match the fp32 pooler weights.
        object_queries = None
        if self.query_pooler is not None:
            if object_tokens is None:
                raise ValueError("object_tokens must be provided when enable_query_pooler=True")
            object_queries = self.query_pooler(object_tokens.to(patch_tokens.dtype))

        if self.use_global_scene_object_concat:
            if object_tokens is None:
                raise ValueError("object_tokens must be provided when use_global_scene_object_concat=True")
            scene_global = patch_tokens.mean(dim=(1, 2))
            object_global = object_tokens.mean(dim=(1, 2))
            context_tokens = torch.cat([scene_global, object_global], dim=-1).unsqueeze(1)
            object_pose, object_translation, object_size_log, object_presence_logits, attn_maps = self.decoder(
                context_tokens, object_queries=object_queries
            )
            outputs = {
                "object_pose": object_pose,
                "object_translation": object_translation,
                "object_presence_logits": object_presence_logits,
            }
            if object_size_log is not None:
                outputs["object_size_log"] = object_size_log
            if attn_maps is not None:
                outputs["object_pose_attn"] = attn_maps
            return outputs

        if self.context_pool == "mean":
            context = patch_tokens.mean(dim=2)
        elif self.context_pool == "flatten":
            batch_size, seq_len, num_patches, channels = patch_tokens.shape
            context = patch_tokens.reshape(batch_size, seq_len * num_patches, channels)
        else:
            raise ValueError(f"Unknown context_pool: {self.context_pool}")

        context_tokens = context
        if object_latent is not None:
            if object_latent.dim() != 3:
                raise ValueError(f"object_latent should be (B,S,C), got {tuple(object_latent.shape)}")
            context_tokens = torch.cat([object_latent, context_tokens], dim=1)

        object_pose, object_translation, object_size_log, object_presence_logits, attn_maps = self.decoder(
            context_tokens, object_queries=object_queries
        )
        outputs = {
            "object_pose": object_pose,
            "object_translation": object_translation,
            "object_presence_logits": object_presence_logits,
        }
        if object_size_log is not None:
            outputs["object_size_log"] = object_size_log
        if attn_maps is not None:
            # (B, num_supervised_layers, num_object_queries, S*P) -- consumed by the
            # coverage attn-mask loss. context_pool must be "flatten" for the S*P key
            # axis to map back to per-frame patch grids.
            outputs["object_pose_attn"] = attn_maps
        return outputs


__all__ = [
    "ObjectPoseHead",
    "ObjectPoseTransformerDecoderHead",
    "ObjectPoseHeadConfig",
]
