#!/usr/bin/env python3
"""Visualize symmetry-equivalent object axes on real RGB samples."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache")

from omnivggt.datasets.housecat6d.housecat6d_camera_pose import HouseCat6DCameraPose
from omnivggt.datasets.real275.real275_camera_pose import Real275CameraPose
from omnivggt.datasets.ycbv.ycbv_camera_pose import YCBVCameraPose
from omnivggt.loss import _load_symmetry_info, _rotation_matrix_to_rot6d, compute_object_srt_loss


DEFAULT_FREEPOSE_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose")
DEFAULT_SYM_PATH = PROJECT_ROOT / "mixed_symmetry_info.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "symmetry_loss_axes"

AXIS_COLORS = {
    "x": (255, 35, 35),
    "y": (20, 220, 40),
    "z": (30, 120, 255),
}


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


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


def scalar(value: Any, default: float = 1.0) -> float:
    if value is None:
        return float(default)
    arr = to_numpy(value).reshape(-1)
    return float(arr[0]) if arr.size else float(default)


def project_points(points_cam: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = points_cam[:, 2]
    valid = np.isfinite(points_cam).all(axis=1) & (z > 1e-6)
    pixels = np.full((points_cam.shape[0], 2), np.nan, dtype=np.float32)
    if np.any(valid):
        projected = (intrinsic @ points_cam[valid].T).T
        pixels[valid] = projected[:, :2] / projected[:, 2:3]
    return pixels, valid


def line(draw: ImageDraw.ImageDraw, pixels: np.ndarray, valid: np.ndarray, i: int, j: int, color, width: int) -> None:
    if valid[i] and valid[j] and np.isfinite(pixels[[i, j]]).all():
        draw.line([tuple(pixels[i]), tuple(pixels[j])], fill=color, width=width)


def label(draw: ImageDraw.ImageDraw, xy: np.ndarray, text: str, color) -> None:
    if not np.isfinite(xy).all():
        return
    x, y = float(xy[0]) + 5.0, float(xy[1]) - 14.0
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        draw.text((x + dx, y + dy), text, fill=(255, 255, 255))
    draw.text((x, y), text, fill=color)


def draw_axes(image: Image.Image, sample: dict[str, Any], rotation: np.ndarray, title: str) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    intrinsic = to_numpy(sample["intrinsic"][0]).astype(np.float32).reshape(3, 3)
    object_size = to_numpy(sample["object_size"]).astype(np.float32).reshape(3)
    depth_mean = scalar(sample.get("depth_mean_scale", sample.get("object_translation_scale")), 1.0)
    translation = to_numpy(sample["object_translation"]).astype(np.float32).reshape(3) * np.float32(depth_mean)

    axis_length = max(float(np.max(object_size)) * 0.75, 1e-4)
    axes_obj = np.asarray(
        [[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]],
        dtype=np.float32,
    )
    axes_cam = (rotation.astype(np.float32) @ axes_obj.T).T + translation[None]
    pixels, valid = project_points(axes_cam, intrinsic)
    line(draw, pixels, valid, 0, 1, AXIS_COLORS["x"], 4)
    line(draw, pixels, valid, 0, 2, AXIS_COLORS["y"], 4)
    line(draw, pixels, valid, 0, 3, AXIS_COLORS["z"], 4)
    label(draw, pixels[1], "X", AXIS_COLORS["x"])
    label(draw, pixels[2], "Y", AXIS_COLORS["y"])
    label(draw, pixels[3], "Z", AXIS_COLORS["z"])
    if valid[0]:
        x, y = pixels[0]
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 255, 255), outline=(0, 0, 0))

    draw.rectangle((0, 0, canvas.width, 30), fill=(0, 0, 0))
    draw.text((8, 8), title, fill=(255, 255, 255))
    return canvas


def make_dataset(name: str, root: Path, resolution: tuple[int, int]):
    common = dict(
        num_object_views=1,
        fixed_object_view_ids=(1,),
        strict_fixed_object_view_ids=True,
        verify_files=True,
        max_records=1,
        resolution=resolution,
    )
    align_json = PROJECT_ROOT / "dataset_align.json"
    if name == "real275":
        return Real275CameraPose(
            dataset_location=str(root / "real275"),
            dset="train",
            split_root=str(root / "real275" / "real_train"),
            gt_root=str(root / "real275" / "gts" / "real_train_umeyama"),
            object_image_root=str(root / "real275" / "real275_aligned_object_refs"),
            align_json=str(align_json),
            only_category="bottle",
            **common,
        )
    if name == "ycbv":
        return YCBVCameraPose(
            dataset_location=str(root / "datasets_real" / "ycbv"),
            dset="train_real",
            split_root=str(root / "datasets_real" / "ycbv" / "train_real"),
            object_image_root=str(root / "datasets_real" / "ycbv" / "ycbv_aligned_object_refs"),
            align_json=str(align_json),
            only_object_id=13,
            **common,
        )
    if name == "housecat6d":
        return HouseCat6DCameraPose(
            dataset_location=str(root / "housecat6d"),
            dset="train",
            object_image_root=str(root / "housecat6d" / "housecat6d_aligned_object_refs"),
            align_json=str(align_json),
            only_category="bottle",
            **common,
        )
    raise ValueError(f"Unknown dataset: {name}")


def loss_for_rotation(sample: dict[str, Any], rotation: np.ndarray, sym_path: Path, use_symmetry: bool) -> float:
    gt_rot = torch.from_numpy(to_numpy(sample["object_rotation"]).astype(np.float32)).unsqueeze(0)
    pred_rot = torch.from_numpy(rotation.astype(np.float32)).unsqueeze(0)
    translation = torch.from_numpy(to_numpy(sample["object_translation"]).astype(np.float32)).unsqueeze(0)
    size_log = torch.from_numpy(to_numpy(sample["object_size_log"]).astype(np.float32)).unsqueeze(0)
    batch = {
        "object_rotation": gt_rot,
        "object_translation": translation,
        "object_size_log": size_log,
        "object_id": torch.tensor([int(to_numpy(sample["object_id"]))], dtype=torch.long),
        "dataset": [str(sample["dataset"])],
        "has_object": torch.tensor([True]),
    }
    pred = {
        "object_pose": _rotation_matrix_to_rot6d(pred_rot),
        "object_translation": translation.clone(),
        "object_size_log": size_log.clone(),
    }
    losses = compute_object_srt_loss(
        pred,
        batch,
        pose_rep="symmetric_rot6d",
        symmetry_info_path=str(sym_path) if use_symmetry else "",
        symmetry_continuous_steps=72,
        weight_size=1.0,
    )
    return float(losses["loss_object_pose"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("real275", "ycbv", "housecat6d"), default="ycbv")
    parser.add_argument("--freepose-root", type=Path, default=DEFAULT_FREEPOSE_ROOT)
    parser.add_argument("--symmetry-info", type=Path, default=DEFAULT_SYM_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--resolution", nargs=2, type=int, default=(518, 476), metavar=("W", "H"))
    parser.add_argument("--steps", nargs="+", type=int, default=(0, 6, 12, 18))
    args = parser.parse_args()

    dataset = make_dataset(args.dataset, args.freepose_root, tuple(args.resolution))
    sample = dataset[0]
    key = f"{sample['dataset']}:{int(to_numpy(sample['object_id']))}"
    symmetry_info = _load_symmetry_info(str(args.symmetry_info), 72)
    if key not in symmetry_info:
        raise KeyError(f"No symmetry rotations for {key} in {args.symmetry_info}")

    image = tensor_image_to_pil(sample["images"][0])
    gt_rot = to_numpy(sample["object_rotation"]).astype(np.float32).reshape(3, 3)
    sym_rots = symmetry_info[key]

    panels = []
    metadata = {
        "dataset": sample["dataset"],
        "object_id": int(to_numpy(sample["object_id"])),
        "object_name": sample.get("object_name", ""),
        "category": sample.get("category", ""),
        "symmetry_key": key,
        "steps": [],
    }
    for step in args.steps:
        sym = sym_rots[int(step) % sym_rots.shape[0]].detach().cpu().numpy().astype(np.float32)
        rotation = gt_rot @ sym
        sym_loss = loss_for_rotation(sample, rotation, args.symmetry_info, use_symmetry=True)
        plain_loss = loss_for_rotation(sample, rotation, args.symmetry_info, use_symmetry=False)
        title = f"step {step:02d}  sym={sym_loss:.6f}  plain={plain_loss:.6f}"
        panels.append(draw_axes(image, sample, rotation, title))
        metadata["steps"].append({"step": int(step), "sym_loss": sym_loss, "plain_loss": plain_loss})

    pad = 8
    cols = min(2, len(panels))
    rows = int(np.ceil(len(panels) / cols))
    grid = Image.new("RGB", (cols * image.width + (cols - 1) * pad, rows * image.height + (rows - 1) * pad), (30, 30, 30))
    for i, panel in enumerate(panels):
        x = (i % cols) * (image.width + pad)
        y = (i // cols) * (image.height + pad)
        grid.paste(panel, (x, y))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset}_{key.replace(':', '_')}_{metadata['object_name'] or metadata['category']}"
    image_path = args.output_dir / f"{stem}_sym_equiv_axes.png"
    meta_path = args.output_dir / f"{stem}_sym_equiv_axes.json"
    grid.save(image_path)
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(image_path)
    print(meta_path)


if __name__ == "__main__":
    main()
