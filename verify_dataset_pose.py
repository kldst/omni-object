"""Sanity-check dataset GT by projecting the object 3-axis onto sampled images.

Usage:
    python verify_dataset_pose.py \
        --config configs/train_oo9d_real275_ycbv_hc.py \
        --dataset ycbv \
        --split train \
        --num-samples 16 \
        --output-dir verify_out

For each sample, draws X(red)/Y(green)/Z(blue) axes at the object center and the
3D bbox, using object_rotation / object_translation_metric / object_size from the
dataset (i.e. the aligned object frame -> camera frame GT, in meters).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from mmengine.config import Config

# Make `omnivggt` importable when running from anywhere.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from omnivggt.datasets import (  # noqa: E402
    OO9DSingleCameraPose,
    Real275CameraPose,
    YCBVCameraPose,
    HouseCat6DCameraPose,
    ColorJitter,
)


DATASET_BUILDERS = {
    "oo9d": "OO9DSingleCameraPose",
    "real275": "Real275CameraPose",
    "ycbv": "YCBVCameraPose",
    "housecat6d": "HouseCat6DCameraPose",
}


AXIS_COLORS = (
    (255, 64, 64),   # X red
    (64, 220, 64),   # Y green
    (64, 128, 255),  # Z blue
)
BBOX_COLOR = (255, 220, 0)


def extract_first_view(value):
    """Slice the leading view dimension from a dataset output field if present."""
    if isinstance(value, torch.Tensor):
        if value.ndim >= 1 and value.shape[0] >= 1 and value.ndim > 1:
            return value[0]
        return value
    arr = np.asarray(value)
    if arr.ndim >= 1 and arr.shape[0] >= 1 and arr.ndim > 1:
        return arr[0]
    return arr


def tensor_image_to_pil(img):
    """images[0] is a [3, H, W] float tensor in [0, 1] (ImgNorm = ToTensor)."""
    t = img.detach().cpu() if isinstance(img, torch.Tensor) else torch.as_tensor(img)
    if t.ndim == 4:
        t = t[0]
    t = t.float().clamp(0.0, 1.0) * 255.0
    arr = t.byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def project_points(points_cam, K):
    """Project Nx3 camera-frame points through a 3x3 intrinsic."""
    points_cam = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    z = points_cam[:, 2]
    valid = z > 1e-6
    uv = np.full((points_cam.shape[0], 2), np.nan, dtype=np.float64)
    if np.any(valid):
        uv[valid] = (K @ points_cam[valid].T).T[:, :2] / z[valid, None]
    return uv, valid


def draw_arrow(draw, start_xy, end_xy, color, width=2):
    draw.line([tuple(start_xy), tuple(end_xy)], fill=color, width=width)
    direction = np.asarray(end_xy, dtype=np.float64) - np.asarray(start_xy, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return
    direction /= norm
    left = np.array([-direction[1], direction[0]])
    tip = np.asarray(end_xy, dtype=np.float64)
    back = tip - direction * 9.0
    p_left = tuple(np.round(back + left * 5.0).astype(int))
    p_right = tuple(np.round(back - left * 5.0).astype(int))
    draw.polygon([tuple(np.round(tip).astype(int)), p_left, p_right], fill=color)


def draw_pose_on_image(image, K, R_obj_to_cam, t_obj_to_cam, object_size,
                       draw_bbox=True, axis_scale=0.6, upscale=2):
    """Render XYZ axes (and optionally 3D bbox with face shading) on the image.

    upscale > 1 enlarges the image (and K) so thin geometry is easy to read.
    """
    if upscale != 1:
        image = image.resize((image.size[0] * upscale, image.size[1] * upscale))
        K = np.asarray(K, dtype=np.float64).copy().reshape(3, 3)
        K[:2] *= upscale
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")

    half = np.asarray(object_size, dtype=np.float64) * 0.5
    axis_length = float(max(np.mean(half), 1e-3) * axis_scale * 2.0)

    R = np.asarray(R_obj_to_cam, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t_obj_to_cam, dtype=np.float64).reshape(3)
    K3 = np.asarray(K, dtype=np.float64).reshape(3, 3)

    origin_uv, origin_valid = project_points(t[None, :], K3)
    if not bool(origin_valid[0]):
        return image
    origin_xy = tuple(np.round(origin_uv[0]).astype(int))

    if draw_bbox:
        sx, sy, sz = half
        corners_obj = np.array([
            [-sx, -sy, -sz], [+sx, -sy, -sz], [+sx, +sy, -sz], [-sx, +sy, -sz],
            [-sx, -sy, +sz], [+sx, -sy, +sz], [+sx, +sy, +sz], [-sx, +sy, +sz],
        ], dtype=np.float64)
        corners_cam = corners_obj @ R.T + t[None, :]
        corners_uv, corners_valid = project_points(corners_cam, K3)

        front = [4, 5, 6, 7]   # +Z face
        back = [0, 1, 2, 3]    # -Z face
        if all(corners_valid[i] for i in back):
            draw.polygon([tuple(corners_uv[i]) for i in back],
                         fill=(255, 60, 60, 70), outline=(255, 90, 0))
        if all(corners_valid[i] for i in front):
            draw.polygon([tuple(corners_uv[i]) for i in front],
                         fill=(60, 255, 60, 95), outline=(0, 200, 0))

        edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        for a, b in edges:
            if corners_valid[a] and corners_valid[b]:
                p1 = tuple(np.round(corners_uv[a]).astype(int))
                p2 = tuple(np.round(corners_uv[b]).astype(int))
                draw.line([p1, p2], fill=BBOX_COLOR, width=2)

    # crosshair at center
    r = 10
    draw.ellipse((origin_xy[0] - r, origin_xy[1] - r, origin_xy[0] + r, origin_xy[1] + r),
                 outline=(255, 255, 255), width=3)
    draw.line([(origin_xy[0] - r - 6, origin_xy[1]), (origin_xy[0] + r + 6, origin_xy[1])],
              fill=(255, 255, 255), width=2)
    draw.line([(origin_xy[0], origin_xy[1] - r - 6), (origin_xy[0], origin_xy[1] + r + 6)],
              fill=(255, 255, 255), width=2)

    # axes
    axis_pts_obj = np.eye(3, dtype=np.float64) * axis_length
    axis_cam = axis_pts_obj @ R.T + t[None, :]
    axis_uv, axis_valid = project_points(axis_cam, K3)
    for i, color in enumerate(AXIS_COLORS):
        if axis_valid[i]:
            end_xy = tuple(np.round(axis_uv[i]).astype(int))
            draw_arrow(draw, origin_xy, end_xy, color, width=5)

    return image


def draw_caption(image, lines):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    y = 4
    for line in lines:
        draw.rectangle((2, y - 1, 2 + 7 * len(line), y + 12), fill=(0, 0, 0))
        draw.text((4, y), line, fill=(255, 255, 255), font=font)
        y += 14
    return image


def build_dataset(cfg, dataset_key, split):
    """Build a single dataset instance using the same construction args as the config."""
    cls_name = DATASET_BUILDERS[dataset_key]
    cls = globals()[cls_name]
    res = tuple(cfg.get("resolution", (518, 476)))
    fixed_views = tuple(cfg.get("fixed_object_view_ids", (1, 5, 10, 15)))

    common = dict(
        num_object_views=4,
        fixed_object_view_ids=fixed_views,
        strict_fixed_object_view_ids=cfg.get("strict_fixed_object_view_ids", True),
        normalize_object_translation_by_depth_mean=True,
        verify_files=True,
        object_presence_prob=1.0,
        z_far=20,
        resolution=res,
        seed=42,
    )

    if dataset_key == "oo9d":
        return cls(
            dataset_location=cfg.oo9d_root,
            dset="train" if split == "train" else "val",
            single_root=cfg.oo9d_single_root,
            single_split_json=(f"{cfg.oo9d_split_root}/single/train.json"
                               if split == "train"
                               else f"{cfg.oo9d_split_root}/single/test_unseen_category_unseen_object.json"),
            object_image_root=cfg.oo9d_object_image_root,
            expand_records_by_view=True,
            **common,
        )
    if dataset_key == "real275":
        return cls(
            dataset_location=cfg.real275_root,
            dset="train" if split == "train" else "val",
            split_root=cfg.real275_split_root,
            gt_root=cfg.real275_gt_root,
            object_image_root=cfg.real275_object_image_root,
            align_json=cfg.align_json,
            expand_records_by_object=True,
            **common,
        )
    if dataset_key == "ycbv":
        return cls(
            dataset_location=cfg.ycbv_root,
            dset="train_real" if split == "train" else "test",
            split_root=(cfg.ycbv_train_split_root if split == "train" else cfg.ycbv_val_split_root),
            object_image_root=cfg.ycbv_object_image_root,
            align_json=cfg.align_json,
            expand_records_by_object=True,
            **common,
        )
    if dataset_key == "housecat6d":
        return cls(
            dataset_location=cfg.housecat6d_root,
            dset="train" if split == "train" else "val",
            object_image_root=cfg.housecat6d_object_image_root,
            align_json=cfg.align_json,
            expand_records_by_object=True,
            **common,
        )
    raise ValueError(f"Unknown dataset key: {dataset_key}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Training config .py path")
    p.add_argument("--dataset", required=True, choices=list(DATASET_BUILDERS.keys()))
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--num-samples", type=int, default=16)
    p.add_argument("--output-dir", default="verify_out")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stride", type=int, default=0,
                   help="If >0, step linearly through the dataset; else random.")
    p.add_argument("--no-bbox", action="store_true", help="Skip the 3D bbox overlay.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg, args.dataset, args.split)
    n_total = len(dataset)
    print(f"[verify] dataset={args.dataset} split={args.split} total={n_total}", flush=True)
    if n_total == 0:
        print("Empty dataset", flush=True)
        return

    out_dir = Path(args.output_dir) / f"{args.dataset}_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    if args.stride > 0:
        indices = list(range(0, n_total, args.stride))[: args.num_samples]
    else:
        indices = rng.choice(n_total, size=min(args.num_samples, n_total), replace=False).tolist()

    n_saved = 0
    n_skipped = 0
    for idx in indices:
        try:
            sample = dataset[idx]
        except Exception as exc:
            print(f"[skip] idx={idx} load error: {exc}", flush=True)
            n_skipped += 1
            continue

        has_object = bool(np.asarray(sample.get("has_object", True)).reshape(-1)[0])
        if not has_object:
            n_skipped += 1
            continue

        img_tensor = extract_first_view(sample["images"])
        pil_img = tensor_image_to_pil(img_tensor)

        K = np.asarray(extract_first_view(sample["intrinsic"]), dtype=np.float64).reshape(3, 3)
        R = np.asarray(sample["object_rotation"], dtype=np.float64).reshape(3, 3)
        t_metric = np.asarray(
            sample.get("object_translation_metric", sample["object_translation"]),
            dtype=np.float64,
        ).reshape(3)
        size = np.asarray(sample["object_size"], dtype=np.float64).reshape(3)

        rendered = draw_pose_on_image(
            pil_img, K, R, t_metric, size, draw_bbox=not args.no_bbox,
        )

        obj_id = int(np.asarray(sample.get("object_id", -1)).reshape(-1)[0])
        category = str(sample.get("category", ""))
        scene_name = str(sample.get("scene_name", ""))
        seq = str(sample.get("seq_name", ""))
        depth_scale = float(np.asarray(sample.get("depth_mean_scale", 1.0)).reshape(-1)[0])

        caption = [
            f"{args.dataset} idx={idx} obj={obj_id} cat={category[:20]}",
            f"scene={scene_name[:30]} depth_mean={depth_scale:.3f}m",
            f"t_metric=({t_metric[0]:+.2f},{t_metric[1]:+.2f},{t_metric[2]:+.2f})m"
            f" size=({size[0]:.2f},{size[1]:.2f},{size[2]:.2f})m",
        ]
        rendered = draw_caption(rendered, caption)

        safe_seq = seq.replace("/", "_") if seq else f"idx{idx}"
        out_path = out_dir / f"{n_saved:04d}_{safe_seq[-80:]}.png"
        rendered.save(out_path)
        n_saved += 1
        print(f"[ok] {out_path}", flush=True)

    print(f"[done] saved={n_saved} skipped={n_skipped} dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()
