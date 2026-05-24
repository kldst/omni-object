#!/usr/bin/env python3
import argparse
import json
import os
import random
import runpy
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.INTER_NEAREST = 0
    cv2_stub.INTER_LINEAR = 1
    cv2_stub.INTER_CUBIC = 2

    def _resize(image: Any, dsize: tuple[int, int], fx: float = 0.0, fy: float = 0.0, interpolation: int = 0) -> Any:
        resampling = getattr(Image, "Resampling", Image)
        mode = resampling.NEAREST if interpolation == cv2_stub.INTER_NEAREST else resampling.BICUBIC
        return np.asarray(Image.fromarray(np.asarray(image)).resize(tuple(dsize), mode))

    cv2_stub.resize = _resize
    sys.modules["cv2"] = cv2_stub


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_oo9d_real275_ycbv_hc.py"
DEFAULT_SAVE_DIR = PROJECT_ROOT / "outputs" / "dataset_config_verification" / "train_oo9d_real275_ycbv_hc"

BBOX_EDGES = (
    (0, 1),
    (1, 3),
    (3, 2),
    (2, 0),
    (4, 5),
    (5, 7),
    (7, 6),
    (6, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def scalar(value: Any, default: float = 1.0) -> float:
    if value is None:
        return float(default)
    arr = to_numpy(value).reshape(-1)
    if arr.size == 0:
        return float(default)
    return float(arr[0])


def tensor_image_to_pil(value: Any) -> Image.Image:
    arr = to_numpy(value).astype(np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    arr = np.nan_to_num(arr)
    if arr.max(initial=0.0) <= 1.5:
        arr = arr * 255.0
    arr = np.clip(arr, 0.0, 255.0).round().astype(np.uint8)
    return Image.fromarray(arr[..., :3], mode="RGB")


def depth_to_pil(value: Any, valid_mask: Any | None = None) -> Image.Image:
    depth = to_numpy(value).astype(np.float32)
    if depth.ndim == 4:
        depth = depth[0]
    if depth.ndim == 3:
        depth = depth[0] if depth.shape[0] == 1 else depth[..., 0]
    valid = np.isfinite(depth) & (depth > 0)
    if valid_mask is not None:
        mask = to_numpy(valid_mask).astype(bool)
        if mask.ndim == 3:
            mask = mask[0]
        valid &= mask
    if not np.any(valid):
        gray = np.zeros(depth.shape, dtype=np.uint8)
    else:
        lo, hi = np.percentile(depth[valid], [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
        gray = (norm * 255.0).round().astype(np.uint8)
        gray[~valid] = 0
    return Image.fromarray(gray, mode="L").convert("RGB")


def mask_to_pil(value: Any) -> Image.Image:
    mask = to_numpy(value)
    if mask.ndim == 4:
        mask = mask[0]
    if mask.ndim == 3:
        mask = mask[0]
    mask = (mask.astype(np.float32) > 0).astype(np.uint8) * 255
    return Image.fromarray(mask, mode="L").convert("RGB")


def project_points(points_cam: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = points_cam[:, 2]
    valid = np.isfinite(points_cam).all(axis=1) & (z > 1e-6)
    pixels = np.full((points_cam.shape[0], 2), np.nan, dtype=np.float32)
    if np.any(valid):
        projected = (intrinsic @ points_cam[valid].T).T
        pixels[valid] = projected[:, :2] / projected[:, 2:3]
    return pixels, valid


def object_bbox_corners(object_size: np.ndarray) -> np.ndarray:
    sx, sy, sz = (np.asarray(object_size, dtype=np.float32).reshape(3) / 2.0).tolist()
    return np.asarray(
        [
            [-sx, -sy, -sz],
            [sx, -sy, -sz],
            [-sx, sy, -sz],
            [sx, sy, -sz],
            [-sx, -sy, sz],
            [sx, -sy, sz],
            [-sx, sy, sz],
            [sx, sy, sz],
        ],
        dtype=np.float32,
    )


def draw_line_if_visible(draw: ImageDraw.ImageDraw, pixels: np.ndarray, valid: np.ndarray, i: int, j: int, color, width: int) -> None:
    if valid[i] and valid[j] and np.isfinite(pixels[[i, j]]).all():
        draw.line([tuple(pixels[i]), tuple(pixels[j])], fill=color, width=width)


def draw_label(draw: ImageDraw.ImageDraw, xy: np.ndarray, label: str, color) -> None:
    if not np.isfinite(xy).all():
        return
    x, y = float(xy[0]) + 6.0, float(xy[1]) - 16.0
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        draw.text((x + dx, y + dy), label, fill=(255, 255, 255))
    draw.text((x, y), label, fill=color)


def draw_axes_and_bbox(sample: dict[str, Any], axis_length_scale: float) -> tuple[Image.Image, dict[str, Any]]:
    image = tensor_image_to_pil(sample["images"][0])
    draw = ImageDraw.Draw(image)
    has_object = bool(to_numpy(sample.get("has_object", True)).reshape(-1)[0])

    intrinsic = to_numpy(sample["intrinsic"][0]).astype(np.float32).reshape(3, 3)
    rotation = to_numpy(sample["object_rotation"]).astype(np.float32).reshape(3, 3)
    object_size = to_numpy(sample["object_size"]).astype(np.float32).reshape(3)

    depth_mean = scalar(sample.get("depth_mean_scale", sample.get("object_translation_scale")), 1.0)
    translation_norm = to_numpy(sample["object_translation"]).astype(np.float32).reshape(3)
    translation_restored = translation_norm * np.float32(depth_mean)
    translation_metric = (
        to_numpy(sample["object_translation_metric"]).astype(np.float32).reshape(3)
        if "object_translation_metric" in sample
        else translation_restored
    )

    if has_object:
        corners_obj = object_bbox_corners(object_size)
        corners_cam = (rotation @ corners_obj.T).T + translation_restored[None]
        corner_pixels, corner_valid = project_points(corners_cam, intrinsic)
        for i, j in BBOX_EDGES:
            draw_line_if_visible(draw, corner_pixels, corner_valid, i, j, (255, 220, 0), 2)

        axis_length = max(float(np.max(object_size)) * axis_length_scale, 1e-4)
        axes_obj = np.asarray(
            [[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]],
            dtype=np.float32,
        )
        axes_cam = (rotation @ axes_obj.T).T + translation_restored[None]
        axis_pixels, axis_valid = project_points(axes_cam, intrinsic)
        draw_line_if_visible(draw, axis_pixels, axis_valid, 0, 1, (255, 0, 0), 4)
        draw_line_if_visible(draw, axis_pixels, axis_valid, 0, 2, (0, 220, 0), 4)
        draw_line_if_visible(draw, axis_pixels, axis_valid, 0, 3, (0, 90, 255), 4)
        if axis_valid[1]:
            draw_label(draw, axis_pixels[1], "X", (255, 0, 0))
        if axis_valid[2]:
            draw_label(draw, axis_pixels[2], "Y", (0, 160, 0))
        if axis_valid[3]:
            draw_label(draw, axis_pixels[3], "Z", (0, 90, 255))
        if axis_valid[0]:
            x, y = axis_pixels[0]
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 255, 255), outline=(0, 0, 0))
    else:
        corner_pixels = np.full((8, 2), np.nan, dtype=np.float32)
        corner_valid = np.zeros(8, dtype=bool)
        axis_pixels = np.full((4, 2), np.nan, dtype=np.float32)
        axis_valid = np.zeros(4, dtype=bool)
        draw.rectangle((8, 8, 250, 36), fill=(180, 0, 0))
        draw.text((14, 15), "ABSENT TARGET OBJECT", fill=(255, 255, 255))

    meta = {
        "has_object": has_object,
        "intrinsic": intrinsic.tolist(),
        "object_rotation": rotation.tolist(),
        "object_size": object_size.tolist(),
        "depth_mean_scale": depth_mean,
        "object_translation_normalized_used": translation_norm.tolist(),
        "object_translation_restored_norm_times_depth_mean": translation_restored.tolist(),
        "object_translation_metric_from_sample": translation_metric.tolist(),
        "object_translation_restore_abs_err_max": float(np.max(np.abs(translation_restored - translation_metric))),
        "bbox_corner_pixels": corner_pixels.tolist(),
        "bbox_corner_valid": corner_valid.tolist(),
        "axis_pixels": axis_pixels.tolist(),
        "axis_valid": axis_valid.tolist(),
    }
    return image, meta


def safe_name(value: Any) -> str:
    text = str(value) if value is not None else "none"
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)[:120]


def concat_source(dataset: Any, global_idx: int) -> tuple[str, int]:
    if not hasattr(dataset, "datasets") or not hasattr(dataset, "cumulative_sizes"):
        return type(dataset).__name__, global_idx
    prev = 0
    for child, cumulative in zip(dataset.datasets, dataset.cumulative_sizes):
        if global_idx < cumulative:
            return type(child).__name__, global_idx - prev
        prev = cumulative
    return type(dataset).__name__, global_idx


def sample_indices(length: int, num_samples: int, seed: int, sequential: bool) -> list[int]:
    if sequential:
        return list(range(min(num_samples, length)))
    rng = random.Random(seed)
    if num_samples >= length:
        indices = list(range(length))
        rng.shuffle(indices)
        return indices
    return rng.sample(range(length), num_samples)


def load_dataset_from_config(config_path: Path, dataset_name: str) -> tuple[Any, dict[str, Any]]:
    import torch
    import omnivggt.datasets as dataset_module

    dataset_module.__dict__["torch"] = torch
    cfg = runpy.run_path(str(config_path))
    key = "train_dataset" if dataset_name == "train" else "val_dataset"
    if key not in cfg:
        raise KeyError(f"{key} not found in config: {config_path}")
    dataset = eval(cfg[key], dataset_module.__dict__)
    return dataset, cfg


def save_sample_outputs(sample: dict[str, Any], dataset_label: str, global_idx: int, local_idx: int, save_dir: Path, axis_length_scale: float) -> dict[str, Any]:
    scene = safe_name(sample.get("scene_name", sample.get("run_name", "scene")))
    obj = safe_name(sample.get("object_name", sample.get("object_id", "object")))
    image_id = safe_name(to_numpy(sample.get("ids", [0])).reshape(-1)[0])
    prefix = f"{global_idx:06d}_{safe_name(dataset_label)}_{local_idx:06d}_{scene}_{obj}_{image_id}"

    overlay, overlay_meta = draw_axes_and_bbox(sample, axis_length_scale)
    rgb = tensor_image_to_pil(sample["images"][0])
    depth = depth_to_pil(sample["depth"][0], sample.get("valid_mask", None))
    mask = mask_to_pil(sample["object_masks"][0]) if "object_masks" in sample else None

    sample_dir = save_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = sample_dir / f"{prefix}_axes_bbox.png"
    rgb_path = sample_dir / f"{prefix}_rgb.png"
    depth_path = sample_dir / f"{prefix}_depth.png"
    mask_path = sample_dir / f"{prefix}_object_mask.png"
    json_path = sample_dir / f"{prefix}.json"

    overlay.save(overlay_path)
    rgb.save(rgb_path)
    depth.save(depth_path)
    if mask is not None:
        mask.save(mask_path)

    object_ref_paths = []
    if "object_images" in sample:
        object_images = sample["object_images"]
        for ref_idx in range(int(object_images.shape[0])):
            ref_path = sample_dir / f"{prefix}_object_ref_{ref_idx:02d}.png"
            tensor_image_to_pil(object_images[ref_idx]).save(ref_path)
            object_ref_paths.append(str(ref_path))

    meta = {
        "global_index": int(global_idx),
        "local_index": int(local_idx),
        "dataset_label": dataset_label,
        "scene_name": str(sample.get("scene_name", "")),
        "run_name": str(sample.get("run_name", "")),
        "image_ids": to_numpy(sample.get("ids", [])).reshape(-1).astype(int).tolist()
        if "ids" in sample
        else [],
        "object_name": str(sample.get("object_name", "")),
        "object_id": int(to_numpy(sample.get("object_id", -1)).reshape(-1)[0]) if "object_id" in sample else -1,
        "inst_id": int(to_numpy(sample.get("inst_id", -1)).reshape(-1)[0]) if "inst_id" in sample else -1,
        "class_id": int(to_numpy(sample.get("class_id", -1)).reshape(-1)[0]) if "class_id" in sample else -1,
        "category": str(sample.get("category", "")),
        "scene_rgb_path": str(sample.get("scene_rgb_path", "")),
        "scene_depth_path": str(sample.get("scene_depth_path", "")),
        "scene_gt_path": str(sample.get("scene_gt_path", "")),
        "scene_mask_path": str(sample.get("scene_mask_path", "")),
        "object_rgb_paths": [str(path) for path in sample.get("object_rgb_paths", [])],
        "output_rgb": str(rgb_path),
        "output_depth": str(depth_path),
        "output_axes_bbox": str(overlay_path),
        "output_object_mask": str(mask_path) if mask is not None else "",
        "output_object_refs": object_ref_paths,
    }
    meta.update(overlay_meta)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    return meta


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample a train/val dataset from a config and save cropped-resolution RGB, "
            "visualized depth, and GT axes/bbox overlays."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", choices=("train", "val"), default="train")
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--positive-only", action="store_true")
    parser.add_argument("--max-positive-attempts", type=int, default=10000)
    parser.add_argument("--axis-length-scale", type=float, default=0.65)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    dataset, cfg = load_dataset_from_config(args.config, args.dataset)
    indices = args.indices if args.indices is not None else sample_indices(len(dataset), args.num_samples, args.seed, args.sequential)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    print(f"config: {args.config}")
    print(f"dataset: {args.dataset}")
    print(f"dataset_type: {type(dataset).__name__}")
    print(f"dataset_len: {len(dataset)}")
    if hasattr(dataset, "datasets"):
        print("children:")
        for child in dataset.datasets:
            print(f"  {type(child).__name__}: {len(child)}")
    print(f"num_samples: {len(indices)}")
    print(f"save_dir: {args.save_dir}")

    rows = []
    attempts = 0
    order = 0
    for global_idx in indices:
        dataset_label, local_idx = concat_source(dataset, global_idx)
        sample = dataset[global_idx]
        attempts += 1
        has_object = bool(to_numpy(sample.get("has_object", True)).reshape(-1)[0])
        if args.positive_only and not has_object:
            continue
        meta = save_sample_outputs(sample, dataset_label, global_idx, local_idx, args.save_dir, args.axis_length_scale)
        rows.append(meta)
        order += 1
        print(
            f"[{order:03d}/{args.num_samples if args.indices is None else len(indices):03d}] "
            f"idx={global_idx} {dataset_label}[{local_idx}] "
            f"has_object={meta['has_object']} "
            f"obj={meta['object_name'] or meta['object_id']} "
            f"restore_err={meta['object_translation_restore_abs_err_max']:.9f}"
        )
        if args.positive_only and args.indices is None and order >= args.num_samples:
            break

    while args.positive_only and args.indices is None and order < args.num_samples and attempts < args.max_positive_attempts:
        global_idx = random.Random(args.seed + attempts).randrange(len(dataset))
        dataset_label, local_idx = concat_source(dataset, global_idx)
        sample = dataset[global_idx]
        attempts += 1
        if not bool(to_numpy(sample.get("has_object", True)).reshape(-1)[0]):
            continue
        meta = save_sample_outputs(sample, dataset_label, global_idx, local_idx, args.save_dir, args.axis_length_scale)
        rows.append(meta)
        order += 1
        print(
            f"[{order:03d}/{args.num_samples:03d}] "
            f"idx={global_idx} {dataset_label}[{local_idx}] "
            f"has_object={meta['has_object']} "
            f"obj={meta['object_name'] or meta['object_id']} "
            f"restore_err={meta['object_translation_restore_abs_err_max']:.9f}"
        )

    summary = {
        "config": str(args.config),
        "dataset": args.dataset,
        "dataset_type": type(dataset).__name__,
        "dataset_len": len(dataset),
        "seed": args.seed,
        "num_samples": len(indices),
        "indices": indices,
        "resolution": cfg.get("resolution"),
        "fixed_object_view_ids": cfg.get("fixed_object_view_ids"),
        "samples": rows,
    }
    summary_path = args.save_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
