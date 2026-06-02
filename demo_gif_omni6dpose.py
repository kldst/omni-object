"""Render per-object GIFs for Omni6DPose (ROPE) eval predictions.

Reuses the predictions written by ``eval_omni6dpose.py`` (no model re-run): each
sample already carries ``pred_R/pred_t``, ``gt_R/gt_t`` and ``gt_size`` in the
camera metric frame. For every (scene, oid) we sort the frames by id and draw,
on the original color image:

* GT 3D bbox in red   (gt_size + GT pose)         -> reference
* pred 3D bbox in green (gt_size + predicted pose) -> isolates rotation/translation
* predicted object axes (X red / Y green / Z blue)

Poses live in the true camera frame, so we draw on the full-resolution color
image using the meta intrinsics scaled to that image size (independent of the
crop used for inference).

Example:
  python demo_gif_omni6dpose.py \
      --predictions outputs/eval_omni6dpose_0531_REFER/predictions_merged.pkl \
      --output-dir gif_outputs/omni6dpose_0531_REFER \
      --max-objects 20 --max-frames 60
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from demo_gradio_6dpose_real import (
    AXIS_COLORS,
    BBOX_EDGES,
    centered_axis_bbox_corners,
    draw_axes_overlay_on_image,
    project_camera_points,
)

try:
    import cv2  # noqa: F401  (drawing helpers use it)
    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PREDICTIONS = PROJECT_ROOT / "outputs" / "eval_omni6dpose_0531_REFER" / "predictions_merged.pkl"
DEFAULT_DATA_ROOT = Path(
    "/mnt/train-data-4-hdd/yian/freepose/Omni6dpose/Omni6DPoseAPI/data/Omni6DPose/ROPE"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "gif_outputs" / "omni6dpose"

PRED_BBOX_COLOR = (0, 255, 0)
GT_BBOX_COLOR = (255, 0, 0)
PREDFULL_BBOX_COLOR = (255, 200, 0)  # pred size + pred pose

# box mode -> (size_key, R_key, t_key, color, label)
BOX_MODES = {
    "gt": ("gt_size", "gt_R", "gt_t", GT_BBOX_COLOR, "GT bbox @ GT pose"),
    "pose": ("gt_size", "pred_R", "pred_t", PRED_BBOX_COLOR, "GT bbox @ pred pose"),
    "pred": ("pred_size", "pred_R", "pred_t", PREDFULL_BBOX_COLOR, "pred bbox @ pred pose"),
}


def _draw_bbox_lines(overlay, intrinsic, rotation_cam, translation_cam, bbox_obj, color, thickness=2):
    import cv2
    bbox_cam = np.asarray(bbox_obj, dtype=np.float32) @ np.asarray(rotation_cam, dtype=np.float32).T + \
        np.asarray(translation_cam, dtype=np.float32)[None, :]
    uv, valid = project_camera_points(bbox_cam, intrinsic)
    for a, b in BBOX_EDGES:
        if not (bool(valid[a]) and bool(valid[b])):
            continue
        pa = tuple(np.round(uv[a]).astype(np.int32))
        pb = tuple(np.round(uv[b]).astype(np.int32))
        cv2.line(overlay, pa, pb, color, thickness)
    return overlay


def _intrinsic_for_image(meta_path: Path, width: int, height: int) -> np.ndarray:
    intr = json.loads(meta_path.read_text(encoding="utf-8"))["camera"]["intrinsics"]
    sx = width / float(intr["width"])
    sy = height / float(intr["height"])
    return np.array(
        [[float(intr["fx"]) * sx, 0.0, float(intr["cx"]) * sx],
         [0.0, float(intr["fy"]) * sy, float(intr["cy"]) * sy],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def render_object_gif(
    scene: str, oid: str, samples: List[Dict[str, Any]],
    data_root: Path, out_path: Path, max_frames: int, fps: float, upscale: int,
    box_modes: List[str], draw_axes: bool,
) -> bool:
    samples = sorted(samples, key=lambda s: int(s["image_id"]))
    if max_frames and len(samples) > max_frames:
        step = len(samples) / float(max_frames)
        samples = [samples[int(i * step)] for i in range(max_frames)]

    scene_dir = data_root / scene
    frames = []
    for s in samples:
        fid = int(s["image_id"])
        color_path = scene_dir / f"{fid:06d}_color.png"
        meta_path = scene_dir / f"{fid:06d}_meta.json"
        if not color_path.is_file() or not meta_path.is_file():
            continue
        image = np.asarray(Image.open(color_path).convert("RGB"), dtype=np.uint8)
        h, w = image.shape[:2]
        K = _intrinsic_for_image(meta_path, w, h)

        gt_size = np.asarray(s["gt_size"], dtype=np.float32).reshape(3)
        axis_len = float(max(gt_size) * 0.6 + 1e-6)

        overlay = image.copy()
        for mode in box_modes:
            size_key, r_key, t_key, color, _ = BOX_MODES[mode]
            size = s.get(size_key)
            if size is None:
                continue
            bbox_obj = centered_axis_bbox_corners(np.asarray(size, dtype=np.float32).reshape(3))
            overlay = _draw_bbox_lines(overlay, K, s[r_key], s[t_key], bbox_obj, color, 2)
        if draw_axes:
            overlay = draw_axes_overlay_on_image(overlay, K, s["pred_R"], np.asarray(s["pred_t"], np.float32),
                                                 axis_len, AXIS_COLORS)
        if upscale and upscale > 1:
            overlay = np.asarray(
                Image.fromarray(overlay).resize((w * upscale, h * upscale), Image.NEAREST), dtype=np.uint8
            )
        frames.append(overlay)

    if not frames:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, duration=1.0 / max(fps, 1e-3), loop=0)
    return True


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--scenes", nargs="+", default=None, help="Restrict to these scene ids.")
    p.add_argument("--oids", nargs="+", default=None, help="Restrict to these object oids.")
    p.add_argument("--max-objects", type=int, default=None, help="Cap number of (scene,oid) GIFs.")
    p.add_argument("--max-frames", type=int, default=60, help="Frames per GIF (subsampled).")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--upscale", type=int, default=2, help="Integer upscale of the small 640x360 frames.")
    p.add_argument("--box-modes", nargs="+", default=["gt", "pose"], choices=list(BOX_MODES.keys()),
                   help="Which boxes to draw: gt=GT bbox@GT pose, pose=GT bbox@pred pose, pred=pred bbox@pred pose.")
    p.add_argument("--no-axes", action="store_true", help="Do not draw the predicted object axes.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    with args.predictions.open("rb") as h:
        preds = pickle.load(h)
    groups: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for s in preds:
        if args.scenes and s["scene_name"] not in set(args.scenes):
            continue
        if args.oids and s["oid"] not in set(args.oids):
            continue
        groups[(s["scene_name"], s["oid"])].append(s)

    keys = sorted(groups.keys())
    if args.max_objects:
        keys = keys[: args.max_objects]
    print(f"[gif] {len(keys)} (scene,oid) groups -> {args.output_dir}")

    made = 0
    for scene, oid in keys:
        out_path = args.output_dir / scene / f"{oid}.gif"
        ok = render_object_gif(
            scene, oid, groups[(scene, oid)], args.data_root, out_path,
            args.max_frames, args.fps, args.upscale,
            args.box_modes, not args.no_axes,
        )
        if ok:
            made += 1
            print(f"  [{made}] {scene}/{oid}.gif  ({len(groups[(scene, oid)])} frames)")
    print(f"[gif] done: {made}/{len(keys)} gifs written under {args.output_dir}")


if __name__ == "__main__":
    main()
