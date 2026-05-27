"""Render a GIF of a single OV9D object across all its frames.

For the chosen scene in
`splits_ov9d_unseen_category_generalization/single/test_unseen_category_unseen_object.json`,
this iterates every frame, runs the model in
`outputs/0521/model.safetensors`, then draws predicted axes and GT 3D bbox on
the cropped scene image. Output is an animated GIF (one per scene).

Translation predictions are multiplied by the per-frame depth mean (GT depth
when `--use-depth-input` is on, otherwise the predicted depth mean), matching
the demo's `use_depth_scale=True` path.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
_LOCAL_TMP = PROJECT_ROOT / "tmp"
_LOCAL_TMP.mkdir(parents=True, exist_ok=True)
(_LOCAL_TMP / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(_LOCAL_TMP))
os.environ.setdefault("MPLCONFIGDIR", str(_LOCAL_TMP / "matplotlib"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import argparse
import json
from typing import Dict, List, Sequence, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from demo_gradio_6dpose_multi_single_0521 import (
    AXIS_COLORS,
    BBOX_EDGES,
    DEFAULT_PRETRAIN_MODEL,
    DEFAULT_TEST_UNSEEN_CATEGORY_UNSEEN_OBJECT_SPLIT_JSON,
    _axis_object_points,
    build_model_from_config,
    centered_axis_bbox_corners,
    clamp_predicted_size_for_bbox,
    draw_axes_overlay_on_image,
    load_config,
    load_ov9d_object_tensor,
    load_ov9d_scene_frame_inputs,
    ov9d_object_id_from_key,
    project_camera_points,
    read_json,
    resolve_local_path,
    resolve_runtime_settings,
    rot6d_to_matrix,
)

PRED_BBOX_COLOR = (0, 255, 0)
GT_BBOX_COLOR = (255, 0, 0)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train_oo9d.py"
DEFAULT_DATASET_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d")
DEFAULT_OBJECT_IMAGE_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d_around_image")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "gif_outputs" / "0521_test_unseen"


def _draw_bbox_lines(
    overlay: np.ndarray,
    intrinsic: np.ndarray,
    rotation_cam: np.ndarray,
    translation_cam: np.ndarray,
    bbox_obj: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    bbox_cam = (
        np.asarray(bbox_obj, dtype=np.float32) @ np.asarray(rotation_cam, dtype=np.float32).T
        + np.asarray(translation_cam, dtype=np.float32)[None, :]
    )
    uv, valid = project_camera_points(bbox_cam, intrinsic)
    height, width = overlay.shape[:2]
    rect = (0, 0, int(width), int(height))
    for start_idx, end_idx in BBOX_EDGES:
        if not (bool(valid[start_idx]) and bool(valid[end_idx])):
            continue
        p1 = tuple(np.round(uv[start_idx]).astype(np.int32))
        p2 = tuple(np.round(uv[end_idx]).astype(np.int32))
        ok, cp1, cp2 = cv2.clipLine(rect, p1, p2)
        if ok:
            cv2.line(overlay, cp1, cp2, color, int(thickness), lineType=cv2.LINE_AA)
    return overlay


def draw_pred_axes_pred_gt_bbox(
    image_rgb: np.ndarray,
    intrinsic: np.ndarray,
    pred_rotation_cam: np.ndarray,
    pred_translation_cam: np.ndarray,
    pred_bbox_obj: np.ndarray,
    gt_rotation_cam: np.ndarray,
    gt_translation_cam: np.ndarray,
    gt_bbox_obj: np.ndarray,
    axis_length: float,
) -> np.ndarray:
    overlay = draw_axes_overlay_on_image(
        image_rgb,
        intrinsic,
        pred_rotation_cam,
        pred_translation_cam,
        axis_length,
        AXIS_COLORS,
    )
    overlay = _draw_bbox_lines(
        overlay, intrinsic, gt_rotation_cam, gt_translation_cam, gt_bbox_obj, GT_BBOX_COLOR,
    )
    overlay = _draw_bbox_lines(
        overlay, intrinsic, pred_rotation_cam, pred_translation_cam, pred_bbox_obj, PRED_BBOX_COLOR,
    )
    return overlay


def compute_depth_mean_scale(
    gt_depth: np.ndarray,
    pred_depth: torch.Tensor | None,
    use_depth_input: bool,
    eps: float = 1e-6,
) -> float:
    if use_depth_input:
        valid = np.asarray(gt_depth, dtype=np.float32)
        valid = valid[valid > 0]
        if valid.size > 0:
            return float(max(float(valid.mean()), eps))
    if pred_depth is not None:
        depth_np = pred_depth.detach().float().cpu().numpy().reshape(-1)
        depth_np = depth_np[np.isfinite(depth_np) & (depth_np > 0)]
        if depth_np.size > 0:
            return float(max(float(depth_np.mean()), eps))
    return 1.0


def build_single_object_records(object_image_root: Path, object_id: int, object_views: Sequence[int]) -> Dict[int, List[Dict]]:
    object_dir = object_image_root / f"obj_{int(object_id):06d}"
    if not object_dir.is_dir():
        raise FileNotFoundError(f"Object reference dir not found: {object_dir}")
    image_ids = [int(p.stem) for p in sorted((object_dir / "rgb").glob("*.png"))]
    missing = [v for v in object_views if v not in image_ids]
    if missing:
        raise FileNotFoundError(f"Object {object_id} missing reference views {missing} in {object_dir}")
    return {
        int(object_id): [
            {
                "scene_dir": object_dir,
                "scene_name": object_dir.name,
                "image_ids": image_ids,
                "object_instance": object_dir.name,
            }
        ]
    }


def render_scene_gif(
    model,
    cfg: Dict,
    scene_record: Dict,
    dataset_root: Path,
    object_image_root: Path,
    object_views: Sequence[int],
    resolution: Tuple[int, int],
    device: torch.device,
    output_path: Path,
    use_depth_input: bool,
    use_depth_scale: bool,
    fps: float,
    max_frames: int | None,
) -> Path | None:
    scene_dir = Path(scene_record.get("scene_dir") or (dataset_root / scene_record["relative_path"]))
    if not scene_dir.is_dir():
        print(f"[skip] scene dir missing: {scene_dir}")
        return None
    scene_gt_path = scene_dir / "scene_gt.json"
    if not scene_gt_path.is_file():
        print(f"[skip] scene_gt.json missing: {scene_gt_path}")
        return None

    object_id = int(scene_record["object_id"])
    single_records = build_single_object_records(object_image_root, object_id, object_views)
    object_tensor, _ = load_ov9d_object_tensor(single_records, object_id, object_views, resolution, device)

    scene_gt = read_json(scene_gt_path)
    frame_ids = sorted(int(k) for k in scene_gt.keys())
    if max_frames is not None:
        frame_ids = frame_ids[: int(max_frames)]

    models_info = read_json(dataset_root / "models_info.json")
    info = models_info.get(str(object_id), {})
    size_mm = np.asarray(
        [info.get("size_x", 100.0), info.get("size_y", 100.0), info.get("size_z", 100.0)],
        dtype=np.float32,
    )

    frames: List[np.ndarray] = []
    for frame_id in frame_ids:
        gts = scene_gt[str(frame_id)]
        object_index = next(
            (idx for idx, gt in enumerate(gts) if int(gt.get("obj_id", -1)) == object_id),
            None,
        )
        if object_index is None:
            continue

        (
            scene_tensor,
            depth_tensor,
            mask_tensor,
            display_image,
            display_depth,
            _gt_mask,
            intrinsic,
        ) = load_ov9d_scene_frame_inputs(
            scene_dir,
            frame_id,
            object_id,
            resolution,
            device,
            target_crop=False,
        )

        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                outputs = model.inference(
                    images=scene_tensor,
                    object_images=object_tensor,
                    extrinsics=None,
                    intrinsics=None,
                    depth=depth_tensor if use_depth_input else None,
                    mask=mask_tensor if use_depth_input else None,
                    camera_gt_index=[],
                    depth_gt_index=[0] if use_depth_input else [],
                )

        if "object_pose" not in outputs or "object_translation" not in outputs:
            print(f"[skip] frame {frame_id:06d}: missing pose outputs")
            continue

        pred_rot6d_cam = outputs["object_pose"][0].detach().float().cpu().numpy()
        pred_rotation_cam = rot6d_to_matrix(pred_rot6d_cam).astype(np.float32)
        pred_translation_cam = outputs["object_translation"][0].detach().float().cpu().numpy().astype(np.float32)
        if use_depth_scale:
            depth_mean = compute_depth_mean_scale(display_depth, outputs.get("depth"), use_depth_input)
            pred_translation_cam = pred_translation_cam * np.float32(depth_mean)

        pred_size = None
        if "object_size" in outputs:
            pred_size = outputs["object_size"].reshape(-1, 3)[0].detach().float().cpu().numpy().astype(np.float32)
        elif "object_size_log" in outputs:
            pred_size = np.exp(outputs["object_size_log"].reshape(-1, 3)[0].detach().float().cpu().numpy()).astype(np.float32)

        gt = gts[object_index]
        gt_rotation_cam = np.asarray(gt["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
        gt_translation_cam = np.asarray(gt["cam_t_m2c"], dtype=np.float32).reshape(3) / 1000.0
        effective_scale = float(gt["scale"]) if "scale" in gt else (1.0 / 1000.0)
        gt_size_m = size_mm * np.float32(effective_scale)
        gt_bbox_obj = centered_axis_bbox_corners(gt_size_m)
        axis_length = max(float(np.linalg.norm(gt_size_m)) * 0.25, 1e-3)

        raw_pred_size = pred_size if pred_size is not None else gt_size_m
        size_for_pred_box, _ = clamp_predicted_size_for_bbox(raw_pred_size, gt_size_m)
        pred_bbox_obj = centered_axis_bbox_corners(size_for_pred_box)

        frame_image = draw_pred_axes_pred_gt_bbox(
            display_image,
            intrinsic,
            pred_rotation_cam,
            pred_translation_cam,
            pred_bbox_obj,
            gt_rotation_cam,
            gt_translation_cam,
            gt_bbox_obj,
            axis_length,
        )

        label = f"obj {object_id:06d} frame {frame_id:06d}"
        cv2.putText(frame_image, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame_image, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        frames.append(frame_image)

    if not frames:
        print(f"[skip] scene {scene_record['scene_name']}: no frames rendered")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = 1000.0 / max(float(fps), 1e-3)
    imageio.mimsave(str(output_path), frames, format="GIF", duration=duration_ms, loop=0)
    print(f"[ok] {scene_record['scene_name']}: {len(frames)} frames → {output_path}")
    return output_path


def select_scenes(payload: Dict, args) -> List[Dict]:
    scenes: List[Dict] = list(payload.get("scenes", []))
    if args.scene_name:
        matched = [s for s in scenes if s["scene_name"] == args.scene_name]
        if not matched:
            raise SystemExit(f"Scene not found in split: {args.scene_name}")
        return matched
    if args.object_id is not None:
        matched = [s for s in scenes if int(s.get("object_id", -1)) == int(args.object_id)]
        if not matched:
            raise SystemExit(f"Object id not found in split: {args.object_id}")
        return matched
    if args.per_category:
        seen: set[str] = set()
        picked: List[Dict] = []
        for s in scenes:
            cat = str(s.get("category", s["scene_name"]))
            if cat in seen:
                continue
            seen.add(cat)
            picked.append(s)
        return picked
    if args.all:
        return scenes
    return scenes[: max(1, int(args.limit))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_PRETRAIN_MODEL)
    parser.add_argument("--split-json", type=Path, default=DEFAULT_TEST_UNSEEN_CATEGORY_UNSEEN_OBJECT_SPLIT_JSON)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--object-image-root", type=Path, default=DEFAULT_OBJECT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scene-name", type=str, default=None, help="Pick a single scene by name from the split.")
    parser.add_argument("--object-id", type=int, default=None, help="Pick scenes by OV9D object id.")
    parser.add_argument("--all", action="store_true", help="Render every scene in the split.")
    parser.add_argument("--per-category", action="store_true", help="Pick one scene per unique `category` field in the split.")
    parser.add_argument("--limit", type=int, default=1, help="When neither --scene-name/--object-id/--all/--per-category is given, render the first N scenes.")
    parser.add_argument("--max-frames", type=int, default=None, help="Cap frames per scene (for quick previews).")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--no-depth-input", action="store_true", help="Disable depth input (use predicted depth mean for scaling).")
    parser.add_argument("--no-depth-scale", action="store_true", help="Skip depth-mean translation scaling.")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    runtime = resolve_runtime_settings(cfg)
    resolution = tuple(int(v) for v in runtime["resolution"])
    object_views = tuple(int(v) for v in runtime["object_input_views"])

    object_image_root = resolve_local_path(runtime.get("object_image_root"), default=args.object_image_root) or args.object_image_root

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    print(f"[info] device={device}, checkpoint={checkpoint_path}")
    model = build_model_from_config(cfg, checkpoint_path, device)

    payload = read_json(Path(args.split_json))
    scenes = select_scenes(payload, args)
    print(f"[info] rendering {len(scenes)} scene(s) from split {Path(args.split_json).name}")

    use_depth_input = not bool(args.no_depth_input)
    use_depth_scale = not bool(args.no_depth_scale)

    for scene_record in scenes:
        output_path = Path(args.output_dir) / f"{scene_record['scene_name']}.gif"
        render_scene_gif(
            model=model,
            cfg=cfg,
            scene_record=scene_record,
            dataset_root=Path(args.dataset_root),
            object_image_root=Path(object_image_root),
            object_views=object_views,
            resolution=resolution,
            device=device,
            output_path=output_path,
            use_depth_input=use_depth_input,
            use_depth_scale=use_depth_scale,
            fps=args.fps,
            max_frames=args.max_frames,
        )


if __name__ == "__main__":
    main()
