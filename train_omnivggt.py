#!/usr/bin/env python3
"""OmniVGGT Training Script"""

import os
import gc
from collections import Counter
from pathlib import Path

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import wandb
import accelerate
import numpy as np
from tqdm import tqdm
import itertools

from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, DistributedDataParallelKwargs

from omnivggt.utils.configs import parse_configs
from omnivggt.datasets.utils.misc import merge_dicts
from omnivggt.utils.misc import select_first_batch
from omnivggt.utils.normalization import normalize_camera_extrinsics_and_points_batch
from visual_util import (
    predictions_to_glb,
    get_world_points_from_depth,
)
from train_utils import (
    build_dataset,
    build_cosine_warmup_scheduler,
    setup_logging,
    setup_directories,
    setup_wandb,
    setup_tensorboard,
    load_model,
    build_optimizer,
    build_loss_criterion,
    summarize_and_dump_model,
)

logger = get_logger(__name__, log_level="INFO")


def _is_file_checkpoint(path):
    if not path:
        return False
    expanded = os.path.expanduser(str(path))
    return os.path.isfile(expanded)


def _unwrap_dataset(dataset):
    current = dataset
    wrappers = []
    while hasattr(current, "dataset"):
        wrappers.append(type(current).__name__)
        current = current.dataset
    return current, wrappers


def _record_run_name(record):
    return record.get("run_name", record.get("scene_name", record.get("seq_name", "unknown")))


def _to_python_list(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


def _format_debug_value(value, precision=6):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return f"{float(value.item()):.{precision}f}"
        value = value.tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        formatted = []
        for item in value:
            if isinstance(item, float):
                formatted.append(round(item, precision))
            else:
                formatted.append(item)
        return str(formatted)
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def _sample_batch_item(value, sample_idx):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value
        return value[sample_idx]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value
        return value[sample_idx]
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        if not value:
            return None
        if sample_idx >= len(value):
            return None
        return value[sample_idx]
    return value


def _print_paths_from_batch(logger_obj, batch, sample_idx):
    path_keys = (
        "scene_rgb_path",
        "scene_depth_path",
        "scene_camera_path",
        "scene_mask_path",
        "object_rgb_paths",
        "object_mask_paths",
    )
    printed = False
    for key in path_keys:
        raw_value = batch.get(key)
        if key in {"object_rgb_paths", "object_mask_paths"} and isinstance(raw_value, list) and raw_value:
            if isinstance(raw_value[0], (list, tuple)):
                value = [view_paths[sample_idx] for view_paths in raw_value if sample_idx < len(view_paths)]
            else:
                value = _sample_batch_item(raw_value, sample_idx)
        else:
            value = _sample_batch_item(raw_value, sample_idx)
        if value is not None:
            logger_obj.info("  %s: %s", key, _format_debug_value(value))
            printed = True
    return printed


def _print_object_target_values(logger_obj, batch, sample_idx, include_depth_stats=True):
    target_keys = (
        "has_object",
        "object_id",
        "object_rotation",
        "object_translation",
        "object_size",
        "object_size_log",
        "normalization_scale",
        "object_translation_scale",
    )
    for key in target_keys:
        value = _sample_batch_item(batch.get(key), sample_idx)
        if value is not None:
            logger_obj.info("  %s: %s", key, _format_debug_value(value))

    object_mask = _sample_batch_item(batch.get("object_masks"), sample_idx)
    if object_mask is not None:
        mask_t = (
            object_mask.detach().float().cpu()
            if isinstance(object_mask, torch.Tensor)
            else torch.as_tensor(object_mask).float()
        )
        logger_obj.info(
            "  object_mask_stats: shape=%s mean=%.6f pixels=%d",
            tuple(mask_t.shape),
            float(mask_t.mean()) if mask_t.numel() > 0 else 0.0,
            int((mask_t > 0).sum().item()) if mask_t.numel() > 0 else 0,
        )

    if include_depth_stats and "depth" in batch:
        depth = _sample_batch_item(batch.get("depth"), sample_idx)
        valid_mask = _sample_batch_item(batch.get("valid_mask"), sample_idx)
        if depth is not None:
            depth_t = depth.detach().float().cpu() if isinstance(depth, torch.Tensor) else torch.as_tensor(depth).float()
            if depth_t.ndim == 3 and depth_t.shape[-1] == 1:
                depth_t = depth_t[..., 0]
            if valid_mask is not None:
                mask_t = valid_mask.detach().bool().cpu() if isinstance(valid_mask, torch.Tensor) else torch.as_tensor(valid_mask).bool()
                valid_depth = depth_t[mask_t]
            else:
                valid_depth = depth_t[depth_t > 0]
            if valid_depth.numel() > 0:
                logger_obj.info(
                    "  depth_valid_stats_m: mean=%.6f min=%.6f max=%.6f count=%d",
                    float(valid_depth.mean()),
                    float(valid_depth.min()),
                    float(valid_depth.max()),
                    int(valid_depth.numel()),
                )
            else:
                logger_obj.info("  depth_valid_stats_m: no valid depth")


def _print_debug_object_paths(logger_obj, dataset, batch, epoch, step, max_samples=2):
    run_names = [str(x) for x in _to_python_list(batch.get("run_name", []))]
    object_names = [str(x) for x in _to_python_list(batch.get("object_name", []))]
    sample_count = min(max_samples, len(run_names), len(object_names))
    if sample_count > 0:
        logger_obj.info(
            "[debug_object_paths] epoch=%s step=%s printing %s sample(s)",
            epoch + 1,
            step,
            sample_count,
        )
        any_batch_paths = False
        for sample_idx in range(sample_count):
            logger_obj.info(
                "[debug_object_paths][sample %s] run=%s object=%s",
                sample_idx,
                run_names[sample_idx],
                object_names[sample_idx],
            )
            any_batch_paths = _print_paths_from_batch(logger_obj, batch, sample_idx) or any_batch_paths
        if any_batch_paths:
            return

    if dataset is None:
        logger_obj.warning("debug_print_object_paths is enabled, but base dataset is unavailable.")
        return

    required_methods = (
        "_resolve_scene_image_path",
        "_resolve_depth_path",
        "_resolve_camera_path",
        "_resolve_object_image_path",
    )
    if not all(hasattr(dataset, name) for name in required_methods):
        logger_obj.warning(
            "debug_print_object_paths is enabled, but dataset %s does not expose path resolvers.",
            type(dataset).__name__,
        )
        return

    run_names = [str(x) for x in _to_python_list(batch.get("run_name", []))]
    object_names = [str(x) for x in _to_python_list(batch.get("object_name", []))]
    camera_indices = _to_python_list(batch.get("camera_indices", [])) or []
    object_cam_indices = _to_python_list(batch.get("object_cam_indices", [])) or []

    sample_count = min(max_samples, len(run_names), len(object_names))
    logger_obj.info(
        "[debug_object_paths] epoch=%s step=%s printing %s sample(s)",
        epoch + 1,
        step,
        sample_count,
    )

    for sample_idx in range(sample_count):
        run_name = run_names[sample_idx]
        object_name = object_names[sample_idx]
        scene_views = _to_python_list(camera_indices[sample_idx]) if sample_idx < len(camera_indices) else []
        object_views = _to_python_list(object_cam_indices[sample_idx]) if sample_idx < len(object_cam_indices) else []

        logger_obj.info(
            "[debug_object_paths][sample %s] run=%s object=%s",
            sample_idx,
            run_name,
            object_name,
        )
        for cam_idx in scene_views:
            logger_obj.info("  scene_rgb: %s", dataset._resolve_scene_image_path(run_name, int(cam_idx)))
            logger_obj.info("  scene_depth: %s", dataset._resolve_depth_path(run_name, int(cam_idx)))
            logger_obj.info("  scene_camera: %s", dataset._resolve_camera_path(run_name, int(cam_idx)))
        for cam_idx in object_views:
            logger_obj.info("  object_rgb: %s", dataset._resolve_object_image_path(object_name, int(cam_idx)))


def _print_debug_object_batch(
    logger_obj,
    dataset,
    batch,
    epoch,
    step,
    max_samples=2,
    include_depth_stats=True,
):
    del dataset
    run_names = [str(x) for x in _to_python_list(batch.get("run_name", []))]
    object_names = [str(x) for x in _to_python_list(batch.get("object_name", []))]
    object_ids = batch.get("object_id")
    if not run_names and object_ids is not None:
        first_dim = object_ids.shape[0] if isinstance(object_ids, torch.Tensor) and object_ids.ndim > 0 else 1
        run_names = [""] * first_dim
        object_names = [""] * first_dim
    sample_count = min(max_samples, len(run_names))
    logger_obj.info(
        "[debug_object_batch] epoch=%s step=%s printing %s sample(s)",
        epoch + 1,
        step,
        sample_count,
    )
    for sample_idx in range(sample_count):
        logger_obj.info(
            "[debug_object_batch][sample %s] run=%s object=%s",
            sample_idx,
            run_names[sample_idx] if sample_idx < len(run_names) else "",
            object_names[sample_idx] if sample_idx < len(object_names) else "",
        )
        _print_paths_from_batch(logger_obj, batch, sample_idx)
        _print_object_target_values(logger_obj, batch, sample_idx, include_depth_stats=include_depth_stats)


def _rot6d_to_matrix_torch(rot6d: torch.Tensor) -> torch.Tensor:
    rot6d = rot6d.reshape(-1, 3, 2)
    x_raw = rot6d[:, :, 0]
    y_raw = rot6d[:, :, 1]

    x = torch.nn.functional.normalize(x_raw, dim=-1)
    z = torch.nn.functional.normalize(torch.cross(x, y_raw, dim=-1), dim=-1)
    y = torch.cross(z, x, dim=-1)
    return torch.stack((x, y, z), dim=-1)


def _compute_object_pose_metrics(predictions, batch):
    if "object_pose" not in predictions or "object_rotation" not in batch:
        return {}

    pred_pose = predictions["object_pose"]
    gt_rot = batch["object_rotation"]
    has_object = batch.get("has_object", None)

    with torch.no_grad():
        pred_pose = pred_pose.detach()
        gt_rot = gt_rot.detach()
        pred_translation = predictions.get("object_translation", None)
        gt_translation = batch.get("object_translation", None)
        pred_size_log = predictions.get("object_size_log", None)
        gt_size_log = batch.get("object_size_log", None)

        if has_object is not None:
            valid_mask = has_object.bool().detach()
            if valid_mask.sum() == 0:
                return {}
            pred_pose = pred_pose[valid_mask]
            gt_rot = gt_rot[valid_mask]
            if pred_translation is not None and gt_translation is not None:
                pred_translation = pred_translation.detach()[valid_mask]
                gt_translation = gt_translation.detach()[valid_mask]
            if pred_size_log is not None and gt_size_log is not None:
                pred_size_log = pred_size_log.detach()[valid_mask]
                gt_size_log = gt_size_log.detach()[valid_mask]

        pred_rot = _rot6d_to_matrix_torch(pred_pose)
        rel_rot = torch.matmul(pred_rot.transpose(-1, -2), gt_rot)
        trace = rel_rot[..., 0, 0] + rel_rot[..., 1, 1] + rel_rot[..., 2, 2]
        cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
        angle_deg = torch.rad2deg(torch.acos(cos_theta))

        pred_trace = pred_rot[..., 0, 0] + pred_rot[..., 1, 1] + pred_rot[..., 2, 2]
        pred_cos_theta = ((pred_trace - 1.0) * 0.5).clamp(-1.0, 1.0)
        pred_angle_deg = torch.rad2deg(torch.acos(pred_cos_theta))

        metrics = {
            "rot_err_deg": angle_deg.mean().item(),
            "rot_pred_deg": pred_angle_deg.mean().item(),
        }
        if pred_translation is not None and gt_translation is not None:
            trans_l2_m = torch.norm(pred_translation - gt_translation, dim=-1)
            metrics["translation_err_m"] = trans_l2_m.mean().item()
            metrics["translation_err_cm"] = (trans_l2_m * 100.0).mean().item()
        if pred_size_log is not None and gt_size_log is not None:
            pred_size = torch.exp(pred_size_log)
            gt_size = torch.exp(gt_size_log)
            metrics["size_err_m"] = torch.abs(pred_size - gt_size).mean().item()
            metrics["size_rel_err"] = (torch.abs(pred_size - gt_size) / gt_size.clamp(min=1e-6)).mean().item()
        return metrics


def _prepare_batch_and_compute_loss(batch, model, criterion):
    if isinstance(batch, list):
        batch = merge_dicts(batch)

    input_extrinsics = batch['extrinsic'].clone()
    input_depths = batch['depth'].clone()
    input_mask = batch['valid_mask'].clone()

    if 'world_points' in batch:
        new_extrinsics, _, new_world_points, new_depths = normalize_camera_extrinsics_and_points_batch(
            extrinsics=batch['extrinsic'],
            cam_points=None,
            world_points=batch['world_points'],
            depths=batch['depth'],
            point_masks=batch['valid_mask'],
        )
        batch['extrinsic'] = new_extrinsics
        batch['world_points'] = new_world_points
        batch['depth'] = new_depths

    inputs = {
        'images': batch['images'],
        'extrinsics': input_extrinsics,
        'intrinsics': batch['intrinsic'],
        'depth': input_depths,
        'mask': input_mask
    }
    if 'object_images' in batch:
        inputs['object_images'] = batch['object_images']

    predictions = model(**inputs)

    loss_details = {}
    with torch.amp.autocast('cuda', enabled=False):
        loss_dict = criterion(predictions, batch)
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                loss_details[key] = value.detach().item()
            else:
                loss_details[key] = value
        loss_details.update(_compute_object_pose_metrics(predictions, batch))

    return batch, predictions, loss_dict, loss_details


def _ordered_loss_postfix(loss_details, preferred_keys):
    ordered = {
        key: loss_details[key]
        for key in preferred_keys
        if key in loss_details
    }
    ordered.update({
        key: value
        for key, value in loss_details.items()
        if key not in ordered
    })
    return ordered


def run_validation(model, val_dataloader, criterion, accelerator, cfg, epoch, global_step, writer):
    if val_dataloader is None:
        return None

    logger.info("=" * 60)
    logger.info(f"Running validation at epoch {epoch + 1}")
    logger.info("=" * 60)

    model.eval()
    aggregated_metrics = {}

    progress_bar = tqdm(
        total=len(val_dataloader),
        desc=f"Val {epoch + 1}",
        disable=not accelerator.is_local_main_process,
    )

    for batch_idx, batch in enumerate(val_dataloader):
        with torch.no_grad():
            _, _, loss_dict, loss_details = _prepare_batch_and_compute_loss(batch, model, criterion)

        batch_losses = {}
        for key, value in loss_details.items():
            if torch.is_tensor(value):
                reduced_value = accelerator.gather_for_metrics(value.detach().reshape(1)).mean().item()
            else:
                reduced_value = float(value)
            aggregated_metrics.setdefault(key, []).append(reduced_value)
            batch_losses[key] = reduced_value

        if accelerator.is_main_process and cfg.get("wandb", False) and batch_losses:
            wandb.log(
                {
                    "val_step/global_step": global_step,
                    "val_step/batch_index": batch_idx,
                    **{f"val_step/{k}": v for k, v in batch_losses.items()},
                },
                step=global_step,
            )

        preferred_postfix_keys = (
            "loss_object_pose",
            "loss_object_translation",
            "loss_object_srt",
            "loss_object_size",
            "rot_err_deg",
            "translation_err_cm",
            "size_rel_err",
            "rot_pred_deg",
        )
        running_means = {
            key: f"{sum(aggregated_metrics[key]) / len(aggregated_metrics[key]):.4f}"
            for key in preferred_postfix_keys
            if key in aggregated_metrics and aggregated_metrics[key]
        }
        if not running_means:
            running_means = {
                key: f"{sum(values) / len(values):.4f}"
                for key, values in aggregated_metrics.items()
                if values
            }
        if running_means:
            progress_bar.set_postfix(running_means)

        progress_bar.update(1)

    progress_bar.close()

    mean_losses = {
        key: sum(values) / len(values)
        for key, values in aggregated_metrics.items()
        if values
    }

    if accelerator.is_main_process and mean_losses:
        accelerator.log({f"val/{k}": v for k, v in mean_losses.items()}, step=global_step)

        if cfg.get("wandb", False):
            wandb.log({f"val/{k}": v for k, v in mean_losses.items()}, step=global_step)

        if writer is not None:
            for key, value in mean_losses.items():
                writer.add_scalar(f"val/{key}", value, global_step)
            writer.add_scalar("val/epoch", epoch, global_step)

    if mean_losses:
        logger.info(
            "Validation summary: %s",
            ", ".join(f"{key}={value:.6f}" for key, value in mean_losses.items())
        )

    model.train()
    return mean_losses


if __name__ == '__main__':
    # ======================================================
    # 1. Configuration and Initialization
    # ======================================================
    # Parse configuration
    cfg = parse_configs()
    val_only = bool(cfg.get("val_only", False))
    resume_path = cfg.get("resume_model_path")
    if val_only and _is_file_checkpoint(resume_path):
        print(
            "val_only received a file checkpoint via resume_model_path; "
            "using it as model_url for weight loading instead of accelerator state restore."
        )
        cfg.model_url = resume_path
        cfg.resume_model_path = None
    save_dir, logging_dir = setup_directories(cfg)
    
    accelerator_project_config = ProjectConfiguration(
        project_dir=save_dir,
        logging_dir=logging_dir
    )
    
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True,
        gradient_as_bucket_view=False,
    )

    accelerator = accelerate.Accelerator(
        mixed_precision=cfg.get("mixed_precision", "no "),
        log_with=cfg.get("report_to", "tensorboard"),
        project_config=accelerator_project_config,
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        kwargs_handlers=[ddp_kwargs],
    )
    
    setup_logging(accelerator)
    set_seed(cfg.get("seed", 42))
    logger.info(f"Random seed set to {cfg.get('seed', 42)}")
    
    writer = None
    if accelerator.is_main_process:
        setup_wandb(cfg, save_dir)
        writer = setup_tensorboard(cfg, save_dir)
    
    # Load model
    model, weight_dtype = load_model(cfg, accelerator.device)

    # ======================================================
    # 2. Dataset and DataLoader
    # ======================================================
    logger.info("Building datasets...")

    train_dataloader = None
    if not val_only:
        train_dataloader = build_dataset(
            dataset=cfg.train_dataset,
            batch_size=cfg.get("train_batch_images", 24),
            num_workers=cfg.get("num_workers", 8),
            test=False
        )
    val_dataloader = None
    if cfg.get("val_dataset", None):
        val_dataloader = build_dataset(
            dataset=cfg.val_dataset,
            batch_size=cfg.get("val_batch_images", cfg.get("train_batch_images", 24)),
            num_workers=cfg.get("num_workers", 8),
            test=True
        )
    raw_dataset = getattr(train_dataloader, "dataset", None) if train_dataloader is not None else None
    raw_sampler = getattr(train_dataloader, "sampler", None) if train_dataloader is not None else None
    dataset_num_samples = len(raw_dataset) if raw_dataset is not None else None
    sampler_num_samples = len(raw_sampler) if raw_sampler is not None else None
    pre_prepare_batches = len(train_dataloader) if train_dataloader is not None else 0
    base_dataset, dataset_wrappers = _unwrap_dataset(raw_dataset) if raw_dataset is not None else (None, [])
    base_records = getattr(base_dataset, "records", None) if base_dataset is not None else None
    run_counter = Counter(_record_run_name(record) for record in base_records) if base_records is not None else Counter()
    
    # ======================================================
    # 3. Optimizer and Loss
    # ======================================================
    
    # Build loss criterion
    train_criterion = build_loss_criterion(cfg)

    optimizer = None
    lr_scheduler = None
    local_steps_per_epoch = 0
    total_training_steps = 0
    actual_local_batches = 0
    gradient_accumulation_steps = cfg.get("gradient_accumulation_steps", 2)
    world_size = accelerator.num_processes

    if not val_only:
        optimizer = build_optimizer(model, cfg)

    if accelerator.is_main_process:
        summarize_and_dump_model(model, save_dir=save_dir, logger_obj=logger)
    
    # ======================================================
    # 4. Prepare for Distributed Training
    # ======================================================
    logger.info("Preparing model, optimizer, and dataloaders for distributed training...")
    logger.info("Made all model parameters and buffers contiguous for DDP compatibility")
    
    if val_only:
        if val_dataloader is None:
            raise ValueError("val_only=True requires val_dataset to be configured.")
        model, val_dataloader = accelerator.prepare(model, val_dataloader)
    else:
        if val_dataloader is not None:
            model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader, val_dataloader
            )
        else:
            model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
        
        # NOW calculate training steps based on ACTUAL sharded dataloader
        # After prepare(), len(train_dataloader) returns the LOCAL length for this process
        actual_local_batches = len(train_dataloader)
        local_steps_per_epoch = actual_local_batches // gradient_accumulation_steps
        total_training_steps = cfg.get('num_train_epochs') * local_steps_per_epoch
        
        logger.info("Training steps calculation (AFTER Accelerate sharding):")
        logger.info(f"  World size: {world_size}")
        if dataset_wrappers:
            logger.info(f"  Dataset wrappers: {dataset_wrappers}")
        if base_dataset is not None:
            logger.info(f"  Base dataset type: {type(base_dataset).__name__}")
        if base_records is not None:
            logger.info(f"  Base dataset raw samples: {len(base_records)}")
            logger.info(f"  Base dataset unique runs: {len(run_counter)}")
            if run_counter:
                run_object_counts = list(run_counter.values())
                logger.info(
                    "  Objects per run stats: "
                    f"min={min(run_object_counts)} "
                    f"max={max(run_object_counts)} "
                    f"mean={sum(run_object_counts) / len(run_object_counts):.2f}"
                )
        if dataset_num_samples is not None:
            logger.info(f"  Dataset total samples: {dataset_num_samples}")
        if sampler_num_samples is not None:
            logger.info(f"  Sampler samples per epoch (before Accelerate sharding): {sampler_num_samples}")
        logger.info(f"  Batches per epoch (before Accelerate sharding): {pre_prepare_batches}")
        logger.info(f"  Actual local batches: {actual_local_batches}")
        logger.info(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
        logger.info(f"  Steps per epoch (per process): {local_steps_per_epoch}")
        logger.info(f"  Total training steps: {total_training_steps}")
        
        lr_scheduler = build_cosine_warmup_scheduler(
            optimizer=optimizer,
            warmup_steps=cfg.get("warmup_steps", 5000),
            total_steps=total_training_steps,
            eta_min_factor=cfg.get("eta_min_factor", 0.1)
        )
        accelerator.register_for_checkpointing(lr_scheduler)
    
    # ======================================================
    # 5. Resume from Checkpoint (if specified)
    # ======================================================
    initial_step = 0
    initial_epoch = 0
    
    if cfg.get("resume_model_path"):
        resume_path = cfg.get("resume_model_path")
        if os.path.exists(resume_path):
            logger.info(f"Resuming from checkpoint: {resume_path}")
            accelerator.load_state(resume_path)
            
            checkpoint_dir = resume_path.rstrip('/')
            if os.path.isdir(checkpoint_dir):
                checkpoint_name = os.path.basename(checkpoint_dir)
            else:
                checkpoint_name = os.path.basename(os.path.dirname(checkpoint_dir))
            
            if checkpoint_name.startswith('checkpoint-'):
                parts = checkpoint_name.replace('checkpoint-', '').split('-')
                initial_epoch = int(parts[0])
                initial_step = int(parts[1])
                logger.info(f"Resumed at epoch {initial_epoch}, step {initial_step}")
            else:
                logger.warning(f"Checkpoint name does not match expected format: {checkpoint_name}")
        else:
            logger.warning(f"Resume path does not exist: {resume_path}")
            logger.warning("Starting training from scratch...")
    
    # ======================================================
    # 6. Training Information
    # ======================================================
    logger.info("=" * 60)
    logger.info("Training Configuration Summary")
    logger.info("=" * 60)
    logger.info(f"  Validation-only mode: {val_only}")
    logger.info(f"  Number of epochs: {cfg.get('num_train_epochs')}")
    logger.info(f"  Validation enabled: {val_dataloader is not None}")
    if val_dataloader is not None:
        logger.info(f"  Validation frequency (epochs): {cfg.get('val_epoch_freq', 1)}")
        logger.info(f"  Validation batches (this process): {len(val_dataloader)}")
    if dataset_wrappers:
        logger.info(f"  Dataset wrappers: {dataset_wrappers}")
    if base_dataset is not None:
        logger.info(f"  Base dataset type: {type(base_dataset).__name__}")
    if base_records is not None:
        logger.info(f"  Base dataset raw samples: {len(base_records)}")
        logger.info(f"  Base dataset unique runs: {len(run_counter)}")
        if run_counter:
            run_object_counts = list(run_counter.values())
            logger.info(
                "  Objects per run stats: "
                f"min={min(run_object_counts)} "
                f"max={max(run_object_counts)} "
                f"mean={sum(run_object_counts) / len(run_object_counts):.2f}"
            )
    if dataset_num_samples is not None:
        logger.info(f"  Dataset total samples: {dataset_num_samples}")
    if sampler_num_samples is not None:
        logger.info(f"  Sampler samples per epoch: {sampler_num_samples}")
    logger.info(f"  Batches per epoch (before Accelerate sharding): {pre_prepare_batches}")
    logger.info(
        f"  Local batches per epoch (this process): "
        f"{len(train_dataloader) if train_dataloader is not None else 0}"
    )
    logger.info(f"  Steps per epoch (this process): {local_steps_per_epoch}")
    logger.info(f"  Total training steps (this process): {total_training_steps}")
    logger.info("---")
    logger.info(f"  Batch size per device: {cfg.get('train_batch_images')}")
    logger.info(f"  Number of GPUs: {world_size}")
    logger.info(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"  Effective global batch size: {cfg.get('train_batch_images') * world_size * gradient_accumulation_steps}")
    logger.info("---")
    logger.info(f"  Max gradient norm: {cfg.get('max_grad_norm', 1.0)}")
    logger.info(f"  Mixed precision: {cfg.get('mixed_precision', 'no')}")
    logger.info(f"  Checkpointing frequency: every {cfg.get('checkpointing_steps', 10000)} steps")
    logger.info(f"  Logging frequency: every {cfg.get('num_save_log', 10)} steps")
    logger.info("=" * 60)

    if val_only:
        logger.info("=" * 60)
        logger.info("Validation-only run")
        logger.info("=" * 60)
        run_validation(
            model=model,
            val_dataloader=val_dataloader,
            criterion=train_criterion,
            accelerator=accelerator,
            cfg=cfg,
            epoch=0,
            global_step=initial_step,
            writer=writer,
        )
        if accelerator.is_main_process:
            if cfg.get("wandb", False):
                wandb.finish()
                logger.info("WandB logging finished")

            if writer is not None:
                writer.close()
                logger.info("TensorBoard logging finished")

        logger.info("Validation-only run completed")
        raise SystemExit(0)

    # ======================================================
    # 7. Training Loop
    # ======================================================
    global_step = initial_step
    accumulation_steps = cfg.get("gradient_accumulation_steps", 2)
    debug_print_object_paths = bool(cfg.get("debug_print_object_paths", False))
    debug_print_object_paths_steps = int(cfg.get("debug_print_object_paths_steps", 1))
    debug_print_object_paths_max_samples = int(cfg.get("debug_print_object_paths_max_samples", 2))
    debug_print_object_batch = bool(cfg.get("debug_print_object_batch", False))
    debug_print_object_batch_steps = int(cfg.get("debug_print_object_batch_steps", 1))
    debug_print_object_batch_max_samples = int(cfg.get("debug_print_object_batch_max_samples", 2))
    debug_print_object_batch_depth_stats = bool(cfg.get("debug_print_object_batch_depth_stats", True))
    
    for epoch in range(initial_epoch, cfg.get('num_train_epochs')):
        logger.info("=" * 60)
        logger.info(f"Starting Epoch {epoch + 1}/{cfg.get('num_train_epochs')}")
        logger.info("=" * 60)
        
        model.train()
        
        # Set epoch for proper shuffling in distributed training
        if hasattr(train_dataloader, 'dataset') and hasattr(train_dataloader.dataset, 'set_epoch'):
            train_dataloader.dataset.set_epoch(epoch)
        if hasattr(train_dataloader, 'sampler') and hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)
        
        if epoch == initial_epoch:
            step_in_epoch = global_step % local_steps_per_epoch
        else:
            step_in_epoch = 0
        
        progress_bar = tqdm(
            total=local_steps_per_epoch,
            initial=step_in_epoch,
            desc=f"Epoch {epoch + 1}",
            disable=not accelerator.is_local_main_process,
        )
        
        # Build iterator so we can skip already completed batches when resuming
        if step_in_epoch > 0:
            logger.info(f"Skipping {step_in_epoch} batches to resume within epoch {epoch + 1}")
        train_iter = itertools.islice(train_dataloader, step_in_epoch, None)
        
        # Training loop for this epoch
        for step, batch in enumerate(train_iter, start=step_in_epoch):
            if (
                debug_print_object_paths
                and accelerator.is_main_process
                and step < debug_print_object_paths_steps
            ):
                _print_debug_object_paths(
                    logger,
                    base_dataset,
                    batch,
                    epoch,
                    step,
                    max_samples=debug_print_object_paths_max_samples,
                )
            if (
                debug_print_object_batch
                and accelerator.is_main_process
                and step < debug_print_object_batch_steps
            ):
                _print_debug_object_batch(
                    logger,
                    base_dataset,
                    batch,
                    epoch,
                    step,
                    max_samples=debug_print_object_batch_max_samples,
                    include_depth_stats=debug_print_object_batch_depth_stats,
                )

            batch, predictions, loss_dict, loss_details = _prepare_batch_and_compute_loss(
                batch,
                model,
                train_criterion,
            )
            
            accelerator.backward(loss_dict['objective'])
            train_postfix_keys = (
                "loss_object_pose",
                "loss_object_translation",
                "loss_object_srt",
                "rot_err_deg",
                "translation_err_cm",
                "loss_object_size",
                "loss_object_mask",
                "loss_object_presence",
                "acc_object_presence",
            )
            progress_bar.set_postfix(_ordered_loss_postfix(loss_details, train_postfix_keys))
            
            # Optimizer step with gradient accumulation
            if (step + 1) % accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), cfg.get('max_grad_norm', 1.0))
                
                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                progress_bar.update(1)
                global_step += 1
                accelerator.log(loss_details, step=global_step)
                
                # Logging
                if accelerator.is_main_process and global_step % cfg.get("num_save_log", 10) == 0:
                    if cfg.get("wandb", False):
                        wandb_dict = {**loss_details, "epoch": epoch}
                        for i, param_group in enumerate(optimizer.param_groups):
                            wandb_dict[f"lr/group_{i}"] = param_group['lr']
                        wandb.log(wandb_dict, step=global_step)
                    
                    if writer is not None:
                        for k, v in loss_details.items():
                            if isinstance(v, (int, float)):
                                writer.add_scalar(f"train/{k}", v, global_step)
                        for i, param_group in enumerate(optimizer.param_groups):
                            writer.add_scalar(f"lr/group_{i}", param_group['lr'], global_step)
                        writer.add_scalar("train/epoch", epoch, global_step)

                # Visualization
                if accelerator.is_main_process and cfg.get("save_glb_visualization", False) and global_step % cfg.get("num_save_visual", 5000) == 0:
                    logger.info(f"Generating visualization at step {global_step}...")
                    save_pts_dir = os.path.join(save_dir, f'epoch-{epoch}')
                    Path(save_pts_dir).mkdir(parents=True, exist_ok=True)
                    
                    try:
                        with torch.no_grad():
                            predictions_0 = select_first_batch(predictions)
                            get_world_points_from_depth(predictions_0)
                            
                            glbscene = predictions_to_glb(
                                predictions_0,
                                conf_thres=cfg.get("vis_conf_threshold", 0.2),
                                filter_by_frames=cfg.get("vis_filter_by_frames", "All"),
                                mask_black_bg=cfg.get("vis_mask_black_bg", False),
                                mask_white_bg=cfg.get("vis_mask_white_bg", False),
                                show_cam=cfg.get("vis_show_cam", True),
                                mask_sky=cfg.get("vis_mask_sky", False),
                                target_dir=save_pts_dir,
                                prediction_mode=cfg.get("vis_prediction_mode", "Predicted Depth"),
                            )
                            
                            glb_path = os.path.join(save_pts_dir, f'glbscene_{global_step}.glb')
                            glbscene.export(file_obj=glb_path)
                            logger.info(f"Visualization saved to {glb_path}")
                            
                            if cfg.get("wandb", False):
                                wandb.log({"visualization": wandb.Object3D(glb_path)}, step=global_step)
                            
                            del glbscene, predictions_0
                            gc.collect()
                    except Exception as e:
                        logger.warning(f"Failed to generate visualization: {e}")
                
                # Checkpointing
                if accelerator.is_main_process and global_step % cfg.get("checkpointing_steps", 10000) == 0:
                    # Calculate completed epochs based on global_step
                    completed_epochs = global_step // local_steps_per_epoch
                    save_path = os.path.join(save_dir, f"checkpoint-{completed_epochs}-{global_step}")
                    logger.info(f"Saving checkpoint to {save_path}...")
                    accelerator.save_state(save_path)
                    logger.info(f"Checkpoint saved successfully")
        
        progress_bar.close()
        
        # Apply remaining gradients at epoch end
        if (step + 1) % accumulation_steps != 0:
            logger.info(f"Applying remaining gradients at end of epoch {epoch + 1}")
            accelerator.clip_grad_norm_(model.parameters(), cfg.get('max_grad_norm', 1.0))
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1
        
        logger.info("=" * 60)
        logger.info(f"Epoch {epoch + 1} completed - Global step: {global_step}")
        logger.info("=" * 60)
        
        if accelerator.is_main_process and cfg.get("save_each_epoch", True):
            epoch_save_path = os.path.join(save_dir, f"checkpoint-epoch-{epoch + 1}")
            logger.info(f"Saving end-of-epoch checkpoint to {epoch_save_path}...")
            accelerator.save_state(epoch_save_path)

        if (
            val_dataloader is not None
            and (epoch + 1) % cfg.get("val_epoch_freq", 1) == 0
        ):
            run_validation(
                model=model,
                val_dataloader=val_dataloader,
                criterion=train_criterion,
                accelerator=accelerator,
                cfg=cfg,
                epoch=epoch,
                global_step=global_step,
                writer=writer,
            )
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # Training completed
    logger.info("=" * 60)
    logger.info("Training Completed!")
    logger.info("=" * 60)

    if val_dataloader is not None and cfg.get('num_train_epochs', 0) > 0:
        run_validation(
            model=model,
            val_dataloader=val_dataloader,
            criterion=train_criterion,
            accelerator=accelerator,
            cfg=cfg,
            epoch=cfg.get('num_train_epochs') - 1,
            global_step=global_step,
            writer=writer,
        )
    
    if accelerator.is_main_process:
        final_save_path = os.path.join(save_dir, "final_checkpoint")
        logger.info(f"Saving final checkpoint to {final_save_path}...")
        accelerator.save_state(final_save_path)
        logger.info("Final checkpoint saved successfully")
        
        if cfg.get("wandb", False):
            wandb.finish()
            logger.info("WandB logging finished")
        
        if writer is not None:
            writer.close()
            logger.info("TensorBoard logging finished")
    
    logger.info("All done!")
