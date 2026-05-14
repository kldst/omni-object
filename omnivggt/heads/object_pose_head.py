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


class ObjectPoseTransformerDecoderHead(nn.Module):
    """Predict object rotation-6D, camera-frame translation, and log metric size."""

    def __init__(self, *, context_dim: int, cfg: Optional[ObjectPoseHeadConfig] = None):
        super().__init__()
        self.cfg = cfg or ObjectPoseHeadConfig()
        self.predict_size = bool(self.cfg.predict_size)

        init_translate, init_rot6d = _load_global_init_params(self.cfg.init_params_path)
        self.register_buffer("init_translate", torch.from_numpy(init_translate).unsqueeze(0))
        self.register_buffer("init_pose", torch.from_numpy(init_rot6d).unsqueeze(0))
        if self.predict_size:
            self.register_buffer("init_size_log", torch.zeros(1, 3, dtype=torch.float32))

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

    def forward(self, context_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = context_tokens.shape[0]
        pred_pose = self.init_pose.expand(batch_size, -1)
        pred_translate = self.init_translate.expand(batch_size, -1)
        pred_size_log = self.init_size_log.expand(batch_size, -1) if self.predict_size else None
        presence_logits = None

        for _ in range(int(self.cfg.ief_iters)):
            token = torch.zeros((batch_size, 1, 1), device=context_tokens.device, dtype=context_tokens.dtype)
            token_out = self.transformer(token, context=context_tokens).squeeze(1)
            pred_pose = self.decpose(token_out) + pred_pose
            pred_translate = self.dectranslate(token_out) + pred_translate
            if self.decsize is not None:
                pred_size_log = self.decsize(token_out) + pred_size_log
            presence_logits = self.presence_branch(token_out).squeeze(-1)

        return pred_pose, pred_translate, pred_size_log, presence_logits


class ObjectPoseHead(nn.Module):
    def __init__(
        self,
        *,
        dim_in: int,
        object_pose_cfg: Optional[ObjectPoseHeadConfig] = None,
        context_pool: str = "flatten",
        use_global_scene_object_concat: bool = False,
        predict_size: bool = True,
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
        decoder_context_dim = 2 * dim_in if self.use_global_scene_object_concat else dim_in
        self.decoder = ObjectPoseTransformerDecoderHead(context_dim=decoder_context_dim, cfg=object_pose_cfg)

    def forward(
        self,
        aggregated_tokens_list,
        patch_start_idx: int,
        object_latent: Optional[torch.Tensor] = None,
        object_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        tokens = aggregated_tokens_list[-1]
        patch_tokens = tokens[:, :, patch_start_idx:, :]

        if self.use_global_scene_object_concat:
            if object_tokens is None:
                raise ValueError("object_tokens must be provided when use_global_scene_object_concat=True")
            scene_global = patch_tokens.mean(dim=(1, 2))
            object_global = object_tokens.mean(dim=(1, 2))
            context_tokens = torch.cat([scene_global, object_global], dim=-1).unsqueeze(1)
            object_pose, object_translation, object_size_log, object_presence_logits = self.decoder(context_tokens)
            outputs = {
                "object_pose": object_pose,
                "object_translation": object_translation,
                "object_presence_logits": object_presence_logits,
            }
            if object_size_log is not None:
                outputs["object_size_log"] = object_size_log
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

        object_pose, object_translation, object_size_log, object_presence_logits = self.decoder(context_tokens)
        outputs = {
            "object_pose": object_pose,
            "object_translation": object_translation,
            "object_presence_logits": object_presence_logits,
        }
        if object_size_log is not None:
            outputs["object_size_log"] = object_size_log
        return outputs


__all__ = [
    "ObjectPoseHead",
    "ObjectPoseTransformerDecoderHead",
    "ObjectPoseHeadConfig",
]
