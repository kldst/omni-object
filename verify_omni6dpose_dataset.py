#!/usr/bin/env python3
"""Verify the Omni6DPose SOPE training dataset.

Builds Omni6DPoseCameraPose using the parameters from configs/train_omnipose.py,
then:
  1. prints the configured paths (scene root, object-image root, patches, view ids);
  2. prints the total dataset size (#records) and #reference objects;
  3. prints the resolved file paths for a few sample records (scene rgb / depth /
     meta and the object reference rgb paths);
  4. draws those samples' GT pose (3D bbox + X/Y/Z axes) onto the ORIGINAL scene
     RGB using the per-sample GT rotation / metric translation / size, saving a
     PNG per drawn sample so you can eyeball that the dataset poses are correct.

Run in the training env (torch / cv2):
  python verify_omni6dpose_dataset.py --num-draw 6 --out-dir verify_omni6dpose_out
  # quick smoke (one patch, capped records):
  python verify_omni6dpose_dataset.py --patches 00 --max-records 500 --num-draw 4
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np

from omnivggt.datasets.omni6dpose.omni6dpose_camera_pose import Omni6DPoseCameraPose
from omnivggt.datasets.utils.transforms import ImgNorm

PROJECT_ROOT = Path(__file__).resolve().parent
AXIS_COLORS_RGB = ((255, 60, 60), (60, 220, 60), (80, 130, 255))  # X / Y / Z
AXIS_LABELS = ("X", "Y", "Z")
BBOX_COLOR_RGB = (255, 215, 0)
BBOX_EDGES = ((0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
              (0, 4), (1, 5), (2, 6), (3, 7))


def load_config(path: Path):
    spec = importlib.util.spec_from_file_location("train_omnipose_cfg", str(path))
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


def bbox_corners(size):
    hx, hy, hz = np.asarray(size, dtype=np.float64) / 2.0
    return np.array([[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)], dtype=np.float64)


def scaled_K(intr, W, H):
    sx, sy = W / float(intr["width"]), H / float(intr["height"])
    return np.array([[intr["fx"] * sx, 0, intr["cx"] * sx],
                     [0, intr["fy"] * sy, intr["cy"] * sy], [0, 0, 1.0]], dtype=np.float64)


def project(pts_cam, K):
    z = np.clip(pts_cam[:, 2:3], 1e-6, None)
    uv = (pts_cam @ K.T)[:, :2] / z
    return uv, (pts_cam[:, 2] > 1e-6)


def draw_pose(img_bgr, K, R, t, size):
    R = np.asarray(R, np.float64); t = np.asarray(t, np.float64).reshape(3)
    corners = bbox_corners(size) @ R.T + t[None, :]
    uv, valid = project(corners, K)
    for a, b in BBOX_EDGES:
        if valid[a] and valid[b]:
            cv2.line(img_bgr, tuple(np.round(uv[a]).astype(int)), tuple(np.round(uv[b]).astype(int)),
                     BBOX_COLOR_RGB[::-1], 2, cv2.LINE_AA)
    L = float(np.max(size)) * 0.6 + 1e-6
    axp = np.array([[0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, L]], np.float64) @ R.T + t[None, :]
    uvp, vp = project(axp, K)
    if vp[0]:
        o = tuple(np.round(uvp[0]).astype(int))
        cv2.circle(img_bgr, o, 4, (255, 255, 255), -1)
        for i in range(3):
            if vp[i + 1]:
                tip = tuple(np.round(uvp[i + 1]).astype(int))
                cv2.arrowedLine(img_bgr, o, tip, AXIS_COLORS_RGB[i][::-1], 3, tipLength=0.18, line_type=cv2.LINE_AA)
                cv2.putText(img_bgr, AXIS_LABELS[i], (tip[0] + 3, tip[1] + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, AXIS_COLORS_RGB[i][::-1], 2, cv2.LINE_AA)


def build_dataset(cfg, split, patches_override, max_records_override):
    patches = patches_override or (cfg.sope_val_patches if split == "val" else cfg.sope_train_patches)
    dset = "val" if split == "val" else "train"
    sope_split = getattr(cfg, "sope_val_split", "test") if split == "val" else "train"
    return Omni6DPoseCameraPose(
        dataset_location=cfg.sope_root,
        dset=dset,
        layout="sope",
        patches=patches,
        split=sope_split,
        object_image_root=cfg.object_image_root,
        oid_to_pam_json=None,
        num_object_views=4,
        fixed_object_view_ids=cfg.fixed_object_view_ids,
        strict_fixed_object_view_ids=cfg.strict_fixed_object_view_ids,
        normalize_object_translation_by_depth_mean=True,
        expand_records_by_object=True,
        verify_files=True,
        object_presence_prob=1.0,  # verify: always present so GT pose is drawn
        max_records=max_records_override,
        z_far=20,
        resolution=cfg.resolution,
        transform=ImgNorm,
        seed=42,
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "train_omnipose.py")
    p.add_argument("--split", choices=("train", "val"), default="train")
    p.add_argument("--patches", nargs="+", default=None, help="Override patches (e.g. 00) for a quick check.")
    p.add_argument("--max-records", type=int, default=None, help="Cap records (faster build).")
    p.add_argument("--num-draw", type=int, default=6)
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "verify_omni6dpose_out")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    print("=" * 70)
    print(f"[config] {args.config}")
    print(f"[paths] sope_root         = {cfg.sope_root}")
    print(f"[paths] object_image_root = {cfg.object_image_root}")
    print(f"[paths] fixed_view_ids    = {cfg.fixed_object_view_ids}")
    print(f"[paths] resolution        = {cfg.resolution}")
    patches = args.patches or (cfg.sope_val_patches if args.split == "val" else cfg.sope_train_patches)
    print(f"[build] split={args.split} patches={patches} max_records={args.max_records}")
    print("=" * 70)

    ds = build_dataset(cfg, args.split, args.patches, args.max_records)

    print(f"\n>>> TOTAL DATASET SIZE: {len(ds)} records "
          f"(unique reference objects available: {len(ds.object_records_by_name)})\n")

    n = min(args.num_draw, len(ds))
    step = max(1, len(ds) // max(n, 1))
    sample_indices = [i * step for i in range(n)]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for k, idx in enumerate(sample_indices):
        s = ds[idx]
        print(f"--- sample #{k} (dataset idx {idx}) ---")
        print(f"  seq_name        : {s['seq_name']}")
        print(f"  scene_name      : {s['scene_name']}")
        print(f"  object oid      : {s['oid']}  (category={s['category']})")
        print(f"  scene_rgb_path  : {s['scene_rgb_path']}")
        print(f"  scene_depth_path: {s['scene_depth_path']}")
        print(f"  scene_meta_path : {s['scene_meta_path']}")
        print(f"  object_rgb_paths: {s['object_rgb_paths']}")
        t_metric = np.asarray(s["object_translation_metric"], np.float64)
        size = np.asarray(s["object_size"], np.float64)
        print(f"  depth_mean_scale: {float(np.asarray(s['depth_mean_scale'])):.4f}  "
              f"t_metric={np.round(t_metric,3)}  t/dm={np.round(t_metric/float(np.asarray(s['depth_mean_scale'])),3)}  "
              f"size={np.round(size,3)}")

        # Draw GT pose on the ORIGINAL scene image (full res) using meta intrinsics.
        img = cv2.imread(s["scene_rgb_path"], cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [warn] cannot read scene rgb; skip drawing")
            continue
        H, W = img.shape[:2]
        meta = json.loads(Path(s["scene_meta_path"]).read_text(encoding="utf-8"))
        K = scaled_K(meta["camera"]["intrinsics"], W, H)
        draw_pose(img, K, s["object_rotation"], t_metric, size)
        out = args.out_dir / f"sample{k:02d}_{s['scene_name'].replace('/', '_')}_{s['oid']}.png"
        cv2.imwrite(str(out), img)
        print(f"  [drawn] {out}")

    print(f"\n[done] drew {n} samples into {args.out_dir}")


if __name__ == "__main__":
    main()
