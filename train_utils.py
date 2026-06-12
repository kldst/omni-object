"""
Training utility functions for OmniVGGT

This module contains helper functions for training setup, including:
- Dataset building
- Model loading
- Optimizer and scheduler setup
- Loss criterion setup
- Logging configuration

License: MIT
"""

import os
import math
import logging
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import wandb
import numpy as np
import accelerate
from safetensors.torch import load_file as load_safetensors_file
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from accelerate.logging import get_logger

from omnivggt.loss import MultitaskLoss
from omnivggt.models.omnivggt import OmniVGGT
from omnivggt.datasets import get_data_loader

logger = get_logger(__name__, log_level="INFO")

try:
    import transformers
except ImportError as exc:
    transformers = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None


def build_dataset(
    dataset: str,
    batch_size: int,
    num_workers: int,
    test: bool = False
) -> torch.utils.data.DataLoader:
    """
    Build data loader for training or testing.
    
    Args:
        dataset: Dataset configuration string
        batch_size: Batch size
        num_workers: Number of data loading workers
        test: Whether this is a test dataset
        
    Returns:
        DataLoader instance
    """
    split = 'Test' if test else 'Train'
    logger.info(f'Building {split} DataLoader for dataset: {dataset}')
    
    loader = get_data_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=True,
        shuffle=not test,
        drop_last=not test
    )
    
    logger.info(f"{split} dataset length: {len(loader)}")
    return loader


def build_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    eta_min_factor: float = 0.05
) -> LambdaLR:
    """
    Build a learning rate scheduler with linear warmup and cosine decay.
    
    Args:
        optimizer: Optimizer instance
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        base_lr: Base learning rate
        eta_min_factor: Minimum learning rate factor (eta_min = eta_min_factor * base_lr)
        
    Returns:
        LambdaLR scheduler instance
    """
    def lr_lambda(current_step: int) -> float:
        # Linear warmup
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # Cosine decay
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_factor + (1.0 - eta_min_factor) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda)


def setup_logging(accelerator: accelerate.Accelerator) -> None:
    """Setup logging configuration for all processes."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if transformers is None:
        logger.warning("Transformers logging setup skipped: %s", _TRANSFORMERS_IMPORT_ERROR)
    elif accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()


def setup_directories(cfg: Any) -> Tuple[str, str]:
    """
    Setup output and logging directories.
    
    Args:
        cfg: Configuration object
        
    Returns:
        Tuple of (save_dir, logging_dir)
    """
    save_dir = os.path.join(cfg.get("output_dir"), cfg.get("exp_name"))
    logging_dir = os.path.join(save_dir, cfg.get("logging_dir"))
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    return save_dir, logging_dir


def setup_wandb(cfg: Any, save_dir: str) -> None:
    """Setup Weights & Biases logging."""
    if cfg.get("wandb", False):
        wandb_dir = os.path.join(save_dir, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        
        wandb.init(
            project="OmniVGGT",
            name=cfg.get("exp_name"),
            config=cfg.to_dict(),
            dir=wandb_dir,
            settings=wandb.Settings(code_dir=".")
        )
        wandb.define_metric("val_step/global_step")
        wandb.define_metric("val_step/batch_index")
        wandb.define_metric("val_step/*", step_metric="val_step/batch_index")
        wandb.run.log_code(".")
        logger.info("WandB logging initialized")


def setup_tensorboard(cfg: Any, save_dir: str) -> Optional[SummaryWriter]:
    """
    Setup TensorBoard logging.
    
    Args:
        cfg: Configuration object
        save_dir: Output directory
        
    Returns:
        SummaryWriter instance or None
    """
    if cfg.get("tensorboard", True):
        tensorboard_log_dir = os.path.join(save_dir, "tensorboard")
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_log_dir)
        logger.info(f"TensorBoard logging initialized at {tensorboard_log_dir}")
        return writer
    return None


def summarize_and_dump_model(
    model: torch.nn.Module,
    save_dir: str,
    logger_obj=None,
) -> None:
    """
    Save model architecture and trainable/frozen parameter lists.

    Files written under ``save_dir``:
    - model.txt
    - trainable.txt
    - frozen.txt
    """
    named_parameters = dict(model.named_parameters())
    total_params = sum(param.numel() for param in named_parameters.values())
    trainable_params = sum(param.numel() for param in named_parameters.values() if param.requires_grad)
    frozen_params = total_params - trainable_params

    if logger_obj is not None:
        logger_obj.info("=" * 60)
        logger_obj.info(f"Model type: {model.__class__.__name__}")
        logger_obj.info(f"Total params: {total_params:,}")
        logger_obj.info(f"Trainable params: {trainable_params:,}")
        logger_obj.info(f"Frozen params: {frozen_params:,}")
        logger_obj.info("=" * 60)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    model_txt = save_path / "model.txt"
    model_txt.write_text(str(model), encoding="utf-8")

    def _dump(param_names, filename: str):
        output_path = save_path / filename
        with output_path.open("w", encoding="utf-8") as handle:
            for param_name in param_names:
                param = named_parameters[param_name]
                handle.write(f"{param_name:<80s} {str(tuple(param.shape)):<24s} {param.numel()}\n")

    _dump([name for name, param in named_parameters.items() if param.requires_grad], "trainable.txt")
    _dump([name for name, param in named_parameters.items() if not param.requires_grad], "frozen.txt")


def load_model(cfg: Any, device: torch.device) -> Tuple[OmniVGGT, torch.dtype]:
    """
    Load and initialize the OmniVGGT model.
    
    Args:
        cfg: Configuration object
        device: Target device
        
    Returns:
        Tuple of (model, weight_dtype)
    """
    logger.info("Initializing OmniVGGT model...")
    model = OmniVGGT(
        enable_camera=cfg.get("enable_camera", True),
        enable_point=cfg.get("enable_point", True),
        enable_depth=cfg.get("enable_depth", True),
        enable_object_mask=cfg.get("enable_object_mask", False),
        enable_object_srt=cfg.get("enable_object_srt", False),
        enable_object_size=cfg.get("enable_object_size", True),
        cam_drop_prob=cfg.get("cam_drop_prob", 0.1),
        depth_drop_prob=cfg.get("depth_drop_prob", 0.1),
        always_use_depth_gt=cfg.get("always_use_depth_gt", False),
        patch_embed_pretrained_path=cfg.get("patch_embed_pretrained_path", None),
        load_patch_embed_from_hub=cfg.get("load_patch_embed_from_hub", True),
        object_pose_context_pool=cfg.get("object_pose_context_pool", "flatten"),
        object_pose_use_global_scene_object_concat=cfg.get("object_pose_use_global_scene_object_concat", False),
        object_pose_transformer_depth=cfg.get("object_pose_transformer_depth", 6),
        object_pose_transformer_heads=cfg.get("object_pose_transformer_heads", 8),
        object_pose_transformer_mlp_dim=cfg.get("object_pose_transformer_mlp_dim", 1024),
        object_pose_transformer_dim_head=cfg.get("object_pose_transformer_dim_head", 64),
        object_pose_transformer_dropout=cfg.get("object_pose_transformer_dropout", 0.0),
        object_pose_transformer_emb_dropout=cfg.get("object_pose_transformer_emb_dropout", 0.0),
        object_pose_transformer_norm=cfg.get("object_pose_transformer_norm", "layer"),
        object_pose_transformer_dim=cfg.get("object_pose_transformer_dim", 1024),
        object_pose_ief_iters=cfg.get("object_pose_ief_iters", 1),
        object_pose_init_params_path=cfg.get("object_pose_init_params_path", None),
        enable_multi_layer_object_prototype_cross_attn=cfg.get("enable_multi_layer_object_prototype_cross_attn", False),
        object_prototype_layer_indices=cfg.get("object_prototype_layer_indices", (4, 11, 17, 23)),
        object_prototype_num_tokens=cfg.get("object_prototype_num_tokens", 4),
        disable_object_prototype_pooler=cfg.get("disable_object_prototype_pooler", False),
        freeze_object_encoder=cfg.get("freeze_object_encoder", False),
        object_prototype_object_encoder_no_grad=cfg.get("object_prototype_object_encoder_no_grad", False),
        object_cross_attn_heads=cfg.get("object_cross_attn_heads", 16),
        object_encode_cache=cfg.get("object_encode_cache", False),
        object_encode_cache_max=cfg.get("object_encode_cache_max", 256),
    )

    # Print network parameters and their indices
    # logger.info("Network parameters and their indices:")
    # for idx, (name, param) in enumerate(model.named_parameters()):
    #     logger.info(f"Parameter {idx}: {name} - Shape: {param.shape}")

    # Load pretrained weights
    model_url = cfg.get("model_url", "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt")
    logger.info(f"Loading pretrained weights from {model_url}")
    
    try:
        load_source = "remote"
        if os.path.isfile(model_url):
            load_source = "local"
            logger.info(f"Detected local checkpoint file: {model_url}")
            if model_url.endswith(".safetensors"):
                state_dict = load_safetensors_file(model_url, device="cpu")
            else:
                state_dict = torch.load(model_url, map_location="cpu")
        else:
            state_dict = torch.hub.load_state_dict_from_url(model_url, map_location="cpu")
        strict = cfg.get("model_load_strict", False)
        incompatible_keys = model.load_state_dict(state_dict, strict=strict)
        missing_keys = list(incompatible_keys.missing_keys)
        unexpected_keys = list(incompatible_keys.unexpected_keys)

        logger.info(
            "Pretrained weights loaded successfully from %s checkpoint with strict=%s",
            load_source,
            strict,
        )
        logger.info(
            "Checkpoint compatibility summary: missing_keys=%d, unexpected_keys=%d",
            len(missing_keys),
            len(unexpected_keys),
        )
        if missing_keys:
            logger.warning("Missing keys (first 20): %s", missing_keys[:20])
        if unexpected_keys:
            logger.warning("Unexpected keys (first 20): %s", unexpected_keys[:20])
        if not missing_keys and not unexpected_keys:
            logger.info("Checkpoint fully matched the current model.")
        else:
            logger.warning("Checkpoint was only partially matched to the current model.")
    except Exception as e:
        logger.warning(f"Failed to load pretrained weights: {e}")
        logger.warning("Training from scratch...")
    
    # Set requires_grad
    model.requires_grad_(cfg.get("model_requires_grad", True))

    # Separate frozen object encoder: copy the just-loaded (trainable) aggregator
    # weights into object_aggregator, then freeze it so object reference encodings stay
    # fixed while the scene aggregator trains. Done after requires_grad_ so the freeze
    # sticks; build_optimizer only collects requires_grad=True params, so these stay out
    # of the optimizer.
    if getattr(model, "object_aggregator", None) is not None:
        model.object_aggregator.load_state_dict(model.aggregator.state_dict())
        # Frozen -> never updated, so storing it in bf16 is safe and halves its resident
        # weight memory (~3.74GB fp32 -> ~1.87GB bf16 for the 936M-param aggregator).
        if cfg.get("freeze_object_encoder_bf16", False):
            model.object_aggregator.to(torch.bfloat16)
        for param in model.object_aggregator.parameters():
            param.requires_grad = False
        model.object_aggregator.eval()
        logger.info(
            "Initialized separate frozen object encoder from aggregator weights (dtype=%s).",
            "bfloat16" if cfg.get("freeze_object_encoder_bf16", False) else "float32",
        )

    # Determine weight dtype
    weight_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    logger.info(f"Using weight dtype: {weight_dtype}")
    
    model.to(device)
    return model, weight_dtype


def build_optimizer(model: torch.nn.Module, cfg: Any) -> torch.optim.Optimizer:
    """
    Build optimizer with parameter groups for different learning rates.
    
    Args:
        model: Model instance
        cfg: Configuration object
        
    Returns:
        Optimizer instance
    """
    param_groups = []
    exclude_keys = ["aggregator.patch_embed"]

    def _set_module_trainable(module: Optional[torch.nn.Module], trainable: bool) -> list[torch.nn.Parameter]:
        if module is None:
            return []
        params = list(module.parameters())
        for param in params:
            param.requires_grad = trainable
        return params
    
    if cfg.get("patch_embed_freeze", False):
        _set_module_trainable(model.aggregator.patch_embed, False)
        logger.info("patch_embed parameters are frozen.")
    else:
        patch_embed_params = _set_module_trainable(model.aggregator.patch_embed, True)
        param_groups.append({
            "params": patch_embed_params,
            "lr": cfg.get("lr_patch_embed", cfg.get("lr")),
            "name": "patch_embed"
        })
        logger.info(f"patch_embed lr set to {cfg.get('lr_patch_embed', cfg.get('lr'))}")
        
    if cfg.get("enable_camera", False):
        exclude_keys.append("camera_head")
        if cfg.get("camera_head_freeze", False):
            _set_module_trainable(model.camera_head, False)
            logger.info("camera_head parameters are frozen.")
        else:
            camera_head_params = _set_module_trainable(model.camera_head, True)
            param_groups.append({
                "params": camera_head_params,
                "lr": cfg.get("lr_camera_head", cfg.get("lr_head", cfg.get("lr"))),
                "name": "camera_head"
            })
            logger.info(f"camera_head lr set to {cfg.get('lr_camera_head', cfg.get('lr'))}")
    
    if cfg.get("enable_depth", False):
        exclude_keys.append("depth_head")
        if cfg.get("depth_head_freeze", False):
            _set_module_trainable(model.depth_head, False)
            logger.info("depth_head parameters are frozen.")
        else:
            depth_head_params = _set_module_trainable(model.depth_head, True)
            param_groups.append({
                "params": depth_head_params,
                "lr": cfg.get("lr_depth_head", cfg.get("lr_head", cfg.get("lr"))),
                "name": "depth_head"
            })
            logger.info(f"depth_head lr set to {cfg.get('lr_depth_head', cfg.get('lr'))}")
            
    if cfg.get("enable_point", False):
        exclude_keys.append("point_head")
        if cfg.get("point_head_freeze", False):
            _set_module_trainable(model.point_head, False)
            logger.info("point_head parameters are frozen.")
        else:
            point_head_params = _set_module_trainable(model.point_head, True)
            param_groups.append({
                "params": point_head_params,
                "lr": cfg.get("lr_point_head", cfg.get("lr_head", cfg.get("lr"))),
                "name": "point_head"
            })
            logger.info(f"point_head lr set to {cfg.get('lr_point_head', cfg.get('lr'))}")

    if cfg.get("enable_object_srt", False) and model.object_srt_head is not None:
        exclude_keys.append("object_srt_head")
        if cfg.get("object_srt_head_freeze", False):
            _set_module_trainable(model.object_srt_head, False)
            logger.info("object_srt_head parameters are frozen.")
        else:
            object_srt_params = _set_module_trainable(model.object_srt_head, True)
            param_groups.append({
                "params": object_srt_params,
                "lr": cfg.get("lr_object_srt_head", cfg.get("lr_head", cfg.get("lr"))),
                "name": "object_srt_head"
            })
            logger.info(f"object_srt_head lr set to {cfg.get('lr_object_srt_head', cfg.get('lr'))}")

    if cfg.get("enable_object_mask", False) and getattr(model, "object_mask_head", None) is not None:
        exclude_keys.append("object_mask_head")
        if cfg.get("object_mask_head_freeze", False):
            _set_module_trainable(model.object_mask_head, False)
            logger.info("object_mask_head parameters are frozen.")
        else:
            object_mask_params = _set_module_trainable(model.object_mask_head, True)
            param_groups.append({
                "params": object_mask_params,
                "lr": cfg.get("lr_object_mask_head", cfg.get("lr_head", cfg.get("lr"))),
                "name": "object_mask_head"
            })
            logger.info(f"object_mask_head lr set to {cfg.get('lr_object_mask_head', cfg.get('lr'))}")

    if getattr(model, "object_token_cross_attn_blocks", None) is not None:
        exclude_keys.append("object_token_cross_attn_blocks")
        if cfg.get("object_cross_attn_freeze", False):
            _set_module_trainable(model.object_token_cross_attn_blocks, False)
            logger.info("object_token_cross_attn_blocks parameters are frozen.")
        else:
            object_cross_attn_params = _set_module_trainable(model.object_token_cross_attn_blocks, True)
            param_groups.append({
                "params": object_cross_attn_params,
                "lr": cfg.get("lr_object_cross_attn", cfg.get("lr")),
                "name": "object_cross_attn"
            })
            logger.info(f"object_cross_attn lr set to {cfg.get('lr_object_cross_attn', cfg.get('lr'))}")

    if getattr(model, "object_prototype_poolers", None) is not None:
        exclude_keys.append("object_prototype_poolers")
        if cfg.get("object_prototype_poolers_freeze", False):
            _set_module_trainable(model.object_prototype_poolers, False)
            logger.info("object_prototype_poolers parameters are frozen.")
        else:
            object_prototype_pooler_params = _set_module_trainable(model.object_prototype_poolers, True)
            param_groups.append({
                "params": object_prototype_pooler_params,
                "lr": cfg.get("lr_object_prototype_poolers", cfg.get("lr")),
                "name": "object_prototype_poolers"
            })
            logger.info(
                f"object_prototype_poolers lr set to {cfg.get('lr_object_prototype_poolers', cfg.get('lr'))}"
            )

    other_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and not any(k in n for k in exclude_keys)
    ]
    if other_params:
        param_groups.append({
            "params": other_params,
            "lr": cfg.get("lr"),
            "name": "other"
        })
    
    optimizer_type = cfg.get("optimizer_type", "adamw").lower()
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(cfg.get("adam_beta1", 0.9), cfg.get("adam_beta2", 0.95)),
            eps=cfg.get("adam_epsilon", 1e-8),
            weight_decay=cfg.get("adam_weight_decay", 0.01)
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    logger.info(f"Optimizer created: {optimizer_type}")
    for i, pg in enumerate(param_groups):
        logger.info(f"  Group {i} ({pg['name']}): lr={pg['lr']}")
    
    return optimizer


def build_loss_criterion(cfg: Any) -> MultitaskLoss:
    """
    Build multi-task loss criterion.
    
    Args:
        cfg: Configuration object
        
    Returns:
        MultitaskLoss instance
    """
    criterion = MultitaskLoss(
        camera={
            "weight": cfg.get("camera_loss_weight", 5.0),
            "loss_type": cfg.get("camera_loss_type", "l1")
        },
        depth={
            "weight": cfg.get("depth_loss_weight", 1.0),
            "gradient_loss_fn": cfg.get("depth_gradient_loss_fn", "grad"),
            "valid_range": cfg.get("depth_valid_range", 0.98)
        },
        point={
            "weight": cfg.get("point_loss_weight", 1.0),
            "gradient_loss_fn": cfg.get("point_gradient_loss_fn", "normal"),
            "valid_range": cfg.get("point_valid_range", 0.98)
        },
        object_srt={
            "weight": cfg.get("object_srt_loss_weight", 1.0),
            "loss_type": cfg.get("object_srt_loss_type", "l1"),
            "pose_rep": cfg.get("object_srt_pose_rep", "rot6d"),
            "weight_pose": cfg.get("object_srt_weight_pose", 1.0),
            "weight_translation": cfg.get("object_srt_weight_translation", 1.0),
            "weight_size": cfg.get("object_srt_weight_size", 0.0) if cfg.get("enable_object_size", True) else 0.0,
            "symmetry_info_path": cfg.get("object_srt_symmetry_info_path", ""),
            "symmetry_continuous_steps": cfg.get("object_srt_symmetry_continuous_steps", 72),
        } if cfg.get("enable_object_srt", False) else None,
        object_mask={
            "weight": cfg.get("object_mask_loss_weight", 1.0),
            "bce_weight": cfg.get("object_mask_bce_weight", 1.0),
            "dice_weight": cfg.get("object_mask_dice_weight", 1.0),
            "pos_weight": cfg.get("object_mask_pos_weight", 1.0),
        } if cfg.get("enable_object_mask", False) else None,
        object_presence={
            "weight": cfg.get("object_presence_loss_weight", 1.0),
            "pos_weight": cfg.get("object_presence_pos_weight", None),
        } if cfg.get("enable_object_presence", cfg.get("enable_object_srt", False)) else None,
        relative_pose={
            "weight": cfg.get("relative_pose_loss_weight", 0.0),
            "weight_rot": cfg.get("relative_pose_weight_rot", 1.0),
            "weight_trans": cfg.get("relative_pose_weight_trans", 0.0),
            "loss_type": cfg.get("relative_pose_loss_type", "l1"),
            "symmetry_info_path": cfg.get("object_srt_symmetry_info_path", ""),
            "symmetry_continuous_steps": cfg.get("object_srt_symmetry_continuous_steps", 72),
        } if cfg.get("enable_object_srt", False) and cfg.get("relative_pose_loss_weight", 0.0) > 0 else None,
    )
    
    logger.info("Loss criterion initialized:")
    logger.info(f"  Camera loss weight: {cfg.get('camera_loss_weight', 5.0)}")
    logger.info(f"  Depth loss weight: {cfg.get('depth_loss_weight', 1.0)}")
    logger.info(f"  Point loss weight: {cfg.get('point_loss_weight', 1.0)}")
    if cfg.get("enable_object_srt", False):
        logger.info(f"  Object SRT loss weight: {cfg.get('object_srt_loss_weight', 1.0)}")
        logger.info(f"  Object size prediction: {cfg.get('enable_object_size', True)}")
        logger.info(
            f"  Object size loss weight: "
            f"{cfg.get('object_srt_weight_size', 0.0) if cfg.get('enable_object_size', True) else 0.0}"
        )
    if cfg.get("enable_object_srt", False) and cfg.get("relative_pose_loss_weight", 0.0) > 0:
        logger.info(f"  Relative-pose loss weight: {cfg.get('relative_pose_loss_weight', 0.0)} "
                    f"(rot={cfg.get('relative_pose_weight_rot', 1.0)}, "
                    f"trans={cfg.get('relative_pose_weight_trans', 0.0)})")
    if cfg.get("enable_object_mask", False):
        logger.info(f"  Object mask loss weight: {cfg.get('object_mask_loss_weight', 1.0)}")
    if cfg.get("enable_object_presence", cfg.get("enable_object_srt", False)):
        logger.info(f"  Object presence loss weight: {cfg.get('object_presence_loss_weight', 1.0)}")
    
    return criterion
