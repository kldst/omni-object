"""Visualize each Omni6DPose (PAM) object's canonical coordinate frame.

The PAM ``Aligned.obj`` frame IS the canonical object frame the model predicts
poses in. ``render_omni6dpose_object_refs_bpy.py`` already saved, per object,
``scene_camera.json`` (cam_K) and ``scene_gt.json`` (cam_R_m2c / cam_t_m2c) for
every rendered view. Here we re-use those to draw the object's local X (red),
Y (green), Z (blue) axes at the object origin on top of the rendered references
-- no re-render needed.

Outputs per object (under ``--output-dir/<NAME>/``):
  * ``axes.gif``       : axes overlaid on all rendered views (orbit) -- shows the
                         canonical frame from every angle.
  * ``axes_grid.png``  : a static montage of the fixed eval views (0/5/8/19),
                         each labelled with X/Y/Z.

Example:
  python draw_object_axes_omni6dpose.py \
      --refs-root outputs/omni6dpose_refs/diverse24 \
      --output-dir gif_outputs/omni6dpose_axes \
      --grid-view-ids 0 5 8 19
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image

from demo_gradio_6dpose_real import project_camera_points

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REFS_ROOT = PROJECT_ROOT / "outputs" / "omni6dpose_refs" / "diverse24"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "gif_outputs" / "omni6dpose_axes"

# X red, Y green, Z blue (RGB).
AXIS_COLORS = ((255, 48, 48), (48, 220, 48), (64, 120, 255))
AXIS_LABELS = ("X", "Y", "Z")


def _draw_axes(image: np.ndarray, K: np.ndarray, R_m2c: np.ndarray, t_m2c: np.ndarray,
               axis_len: float, thickness: int = 3) -> np.ndarray:
    pts_obj = np.array([[0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]], dtype=np.float32)
    pts_cam = pts_obj @ np.asarray(R_m2c, np.float32).T + np.asarray(t_m2c, np.float32)[None, :]
    uv, valid = project_camera_points(pts_cam, K)
    out = np.ascontiguousarray(image).copy()
    if not bool(valid[0]):
        return out
    origin = tuple(np.round(uv[0]).astype(np.int32))
    cv2.circle(out, origin, 4, (255, 255, 255), -1)
    for i in range(3):
        if not bool(valid[i + 1]):
            continue
        tip = tuple(np.round(uv[i + 1]).astype(np.int32))
        cv2.arrowedLine(out, origin, tip, AXIS_COLORS[i], thickness, tipLength=0.15)
        cv2.putText(out, AXIS_LABELS[i], (tip[0] + 3, tip[1] + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, AXIS_COLORS[i], 2, cv2.LINE_AA)
    return out


def _axis_len_for(meta_path: Path, default: float = 0.45) -> float:
    if not meta_path.is_file():
        return default
    try:
        bounds = json.loads(meta_path.read_text(encoding="utf-8"))["mesh"]["centered_bounds"]
        half = np.abs(np.asarray(bounds, dtype=np.float32)).max()
        return float(half * 1.35)
    except Exception:
        return default


def make_grid(images: List[np.ndarray], cols: int = 2, pad: int = 4) -> np.ndarray:
    if not images:
        return np.zeros((10, 10, 3), np.uint8)
    h, w = images[0].shape[:2]
    rows = (len(images) + cols - 1) // cols
    grid = np.full((rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad, 3), 255, np.uint8)
    for idx, im in enumerate(images):
        r, c = divmod(idx, cols)
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        grid[y:y + h, x:x + w] = im
    return grid


def process_object(obj_dir: Path, out_dir: Path, grid_view_ids: List[int], fps: float) -> bool:
    sc_path = obj_dir / "scene_camera.json"
    sg_path = obj_dir / "scene_gt.json"
    rgb_dir = obj_dir / "rgb"
    if not (sc_path.is_file() and sg_path.is_file() and rgb_dir.is_dir()):
        return False
    scene_camera = json.loads(sc_path.read_text(encoding="utf-8"))
    scene_gt = json.loads(sg_path.read_text(encoding="utf-8"))
    axis_len = _axis_len_for(obj_dir / "metadata.json")

    view_ids = sorted(int(k) for k in scene_camera.keys())
    gif_frames, grid_imgs = [], []
    for vid in view_ids:
        img_path = rgb_dir / f"{vid:06d}.png"
        if not img_path.is_file():
            continue
        image = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        K = np.asarray(scene_camera[str(vid)]["cam_K"], np.float32).reshape(3, 3)
        gt = scene_gt[str(vid)][0]
        R = np.asarray(gt["cam_R_m2c"], np.float32).reshape(3, 3)
        t = np.asarray(gt["cam_t_m2c"], np.float32).reshape(3)
        overlay = _draw_axes(image, K, R, t, axis_len)
        gif_frames.append(overlay)
        if vid in set(grid_view_ids):
            tagged = overlay.copy()
            cv2.putText(tagged, f"view {vid}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
            grid_imgs.append(tagged)

    if not gif_frames:
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_dir / "axes.gif", gif_frames, duration=1.0 / max(fps, 1e-3), loop=0)
    if grid_imgs:
        Image.fromarray(make_grid(grid_imgs, cols=2)).save(out_dir / "axes_grid.png")
    return True


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--refs-root", type=Path, default=DEFAULT_REFS_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--objects", nargs="+", default=None, help="Restrict to these PAM dir names.")
    p.add_argument("--grid-view-ids", nargs="+", type=int, default=[0, 5, 8, 19])
    p.add_argument("--fps", type=float, default=10.0)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.objects:
        obj_dirs = [args.refs_root / name for name in args.objects]
    else:
        obj_dirs = sorted(d for d in args.refs_root.iterdir() if d.is_dir())
    print(f"[axes] {len(obj_dirs)} objects -> {args.output_dir}")
    made = 0
    for obj_dir in obj_dirs:
        ok = process_object(obj_dir, args.output_dir / obj_dir.name, args.grid_view_ids, args.fps)
        if ok:
            made += 1
            print(f"  [{made}] {obj_dir.name}")
    print(f"[axes] done: {made}/{len(obj_dirs)} objects under {args.output_dir}")


if __name__ == "__main__":
    main()
