#!/usr/bin/env python3
"""Evaluate OV9D object pose checkpoints on train and eval splits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import runpy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file as load_safetensors_file
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from omnivggt.datasets import OV9DCameraPose
from omnivggt.datasets.utils.transforms import ImgNorm
from omnivggt.loss import _load_symmetry_info
from omnivggt.models.omnivggt import OmniVGGT


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train_ov9d_camera_pose.py"
DEFAULT_PRETRAIN_MODEL = PROJECT_ROOT / "outputs" / "0511" / "12000" / "model.safetensors"
METRIC_COLUMNS = (
    "Abs IoU@50",
    "Abs 5°5cm",
    "Abs 10°5cm",
    "Abs 10°10cm",
    "Abs 20°20cm",
)
ABS_METRIC_COLUMNS = METRIC_COLUMNS
BOX_EDGES = (
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
PRED_COLOR = (0, 255, 0)
GT_COLOR = (255, 160, 0)


def load_config(config_path: Path) -> Dict:
    cfg = runpy.run_path(str(config_path))
    return {key: value for key, value in cfg.items() if not key.startswith("__")}


def parse_dataset_ctor_arg(dataset_expr: str, arg_name: str, default=None):
    pattern = rf"{re.escape(arg_name)}\s*=\s*([^,()]+|\([^)]*\))"
    match = re.search(pattern, dataset_expr)
    if not match:
        return default
    value_str = match.group(1).strip()
    try:
        return eval(value_str, {"__builtins__": {}}, {})
    except Exception:
        return default


def latest_checkpoint_for_experiment(cfg: Dict) -> Path | None:
    output_dir = Path(cfg.get("output_dir", PROJECT_ROOT / "outputs"))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_root = output_dir / str(cfg.get("exp_name", ""))
    if not output_root.is_dir():
        return None
    candidates = []
    for checkpoint_dir in output_root.glob("checkpoint-*"):
        model_path = checkpoint_dir / "model.safetensors"
        if not model_path.is_file():
            continue
        match = re.search(r"checkpoint-(\d+)-(\d+)$", checkpoint_dir.name)
        sort_key = tuple(int(x) for x in match.groups()) if match else (0, 0)
        candidates.append((sort_key, model_path))
    candidates.sort()
    return candidates[-1][1] if candidates else None


def resolve_checkpoint_path(cfg: Dict, checkpoint_path: str | None) -> Path:
    if checkpoint_path:
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    if DEFAULT_PRETRAIN_MODEL.is_file():
        return DEFAULT_PRETRAIN_MODEL
    latest = latest_checkpoint_for_experiment(cfg)
    if latest is not None:
        return latest
    raise FileNotFoundError("Unable to locate checkpoint. Pass --checkpoint explicitly.")


def build_model_from_config(cfg: Dict, checkpoint_path: Path, device: torch.device) -> OmniVGGT:
    model = OmniVGGT(
        enable_camera=cfg.get("enable_camera", True),
        enable_point=cfg.get("enable_point", True),
        enable_depth=cfg.get("enable_depth", True),
        enable_object_mask=cfg.get("enable_object_mask", False),
        enable_object_srt=cfg.get("enable_object_srt", False),
        always_use_depth_gt=cfg.get("always_use_depth_gt", False),
        patch_embed_pretrained_path=cfg.get("patch_embed_pretrained_path", None),
        load_patch_embed_from_hub=cfg.get("load_patch_embed_from_hub", True),
        cam_drop_prob=cfg.get("cam_drop_prob", 0.1),
        depth_drop_prob=cfg.get("depth_drop_prob", 0.1),
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
        object_prototype_object_encoder_no_grad=cfg.get("object_prototype_object_encoder_no_grad", False),
        object_cross_attn_heads=cfg.get("object_cross_attn_heads", 16),
    )
    if checkpoint_path.suffix == ".safetensors":
        state_dict = load_safetensors_file(str(checkpoint_path), device="cpu")
    else:
        raw = torch.load(checkpoint_path, map_location="cpu")
        state_dict = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    missing, unexpected = model.load_state_dict(state_dict, strict=bool(cfg.get("model_load_strict", False)))
    if missing:
        print(f"[eval] Missing keys: {missing}")
    if unexpected:
        print(f"[eval] Unexpected keys: {unexpected}")
    return model.eval().to(device)


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float64).reshape(3, 2)
    x_raw = rot6d[:, 0]
    y_raw = rot6d[:, 1]
    x = x_raw / max(np.linalg.norm(x_raw), 1e-12)
    z = np.cross(x, y_raw)
    z = z / max(np.linalg.norm(z), 1e-12)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def rotation_error_degrees(pred_rot: np.ndarray, gt_rot: np.ndarray) -> float:
    rel = np.asarray(pred_rot, dtype=np.float64).T @ np.asarray(gt_rot, dtype=np.float64)
    cos_theta = np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def translation_l2_cm(pred_t: np.ndarray, gt_t: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(pred_t, dtype=np.float64) - np.asarray(gt_t, dtype=np.float64)) * 100.0)


def centered_bbox_corners(size_xyz: np.ndarray) -> np.ndarray:
    size = np.asarray(size_xyz, dtype=np.float64).reshape(3)
    half = size * 0.5
    xs = [-half[0], half[0]]
    ys = [-half[1], half[1]]
    zs = [-half[2], half[2]]
    return np.asarray([[x, y, z] for z in zs for y in ys for x in xs], dtype=np.float64)


def transform_bbox_corners(size_xyz: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray | None:
    size = np.asarray(size_xyz, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(size)) or np.any(size <= 0.0):
        return None
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    return centered_bbox_corners(size) @ rotation.T + translation[None, :]


def box_volume(size_xyz: np.ndarray) -> float:
    size = np.asarray(size_xyz, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(size)) or np.any(size <= 0.0):
        return 0.0
    return float(np.prod(size))


def point_inside_obb(point: np.ndarray, center: np.ndarray, rotation: np.ndarray, half_size: np.ndarray, eps: float = 1e-8) -> bool:
    local = np.asarray(rotation, dtype=np.float64).T @ (np.asarray(point, dtype=np.float64) - center)
    return bool(np.all(np.abs(local) <= half_size + eps))


def unique_points(points: List[np.ndarray], decimals: int = 9) -> np.ndarray:
    if not points:
        return np.empty((0, 3), dtype=np.float64)
    rounded = np.round(np.asarray(points, dtype=np.float64), decimals=decimals)
    _, keep = np.unique(rounded, axis=0, return_index=True)
    return np.asarray(points, dtype=np.float64)[np.sort(keep)]


def segment_plane_intersections(
    p0: np.ndarray,
    p1: np.ndarray,
    center: np.ndarray,
    rotation: np.ndarray,
    half_size: np.ndarray,
) -> Iterable[np.ndarray]:
    direction = p1 - p0
    for axis_idx in range(3):
        axis = rotation[:, axis_idx]
        for signed_half in (-half_size[axis_idx], half_size[axis_idx]):
            d0 = float(np.dot(p0 - center, axis) - signed_half)
            d1 = float(np.dot(p1 - center, axis) - signed_half)
            denom = d0 - d1
            if abs(denom) < 1e-12:
                continue
            alpha = d0 / denom
            if -1e-8 <= alpha <= 1.0 + 1e-8:
                yield p0 + np.clip(alpha, 0.0, 1.0) * direction


def intersection_volume_obb(
    pred_size: np.ndarray,
    pred_rotation: np.ndarray,
    pred_translation: np.ndarray,
    gt_size: np.ndarray,
    gt_rotation: np.ndarray,
    gt_translation: np.ndarray,
) -> float:
    pred_size = np.asarray(pred_size, dtype=np.float64).reshape(3)
    gt_size = np.asarray(gt_size, dtype=np.float64).reshape(3)
    pred_rotation = np.asarray(pred_rotation, dtype=np.float64).reshape(3, 3)
    gt_rotation = np.asarray(gt_rotation, dtype=np.float64).reshape(3, 3)
    pred_translation = np.asarray(pred_translation, dtype=np.float64).reshape(3)
    gt_translation = np.asarray(gt_translation, dtype=np.float64).reshape(3)
    pred_corners = transform_bbox_corners(pred_size, pred_rotation, pred_translation)
    gt_corners = transform_bbox_corners(gt_size, gt_rotation, gt_translation)
    if pred_corners is None or gt_corners is None:
        return 0.0

    pred_half = pred_size * 0.5
    gt_half = gt_size * 0.5
    points: List[np.ndarray] = []

    for point in pred_corners:
        if point_inside_obb(point, gt_translation, gt_rotation, gt_half):
            points.append(point)
    for point in gt_corners:
        if point_inside_obb(point, pred_translation, pred_rotation, pred_half):
            points.append(point)

    for start_idx, end_idx in BOX_EDGES:
        p0, p1 = pred_corners[start_idx], pred_corners[end_idx]
        for point in segment_plane_intersections(p0, p1, gt_translation, gt_rotation, gt_half):
            if point_inside_obb(point, pred_translation, pred_rotation, pred_half) and point_inside_obb(
                point, gt_translation, gt_rotation, gt_half
            ):
                points.append(point)
        p0, p1 = gt_corners[start_idx], gt_corners[end_idx]
        for point in segment_plane_intersections(p0, p1, pred_translation, pred_rotation, pred_half):
            if point_inside_obb(point, pred_translation, pred_rotation, pred_half) and point_inside_obb(
                point, gt_translation, gt_rotation, gt_half
            ):
                points.append(point)

    hull_points = unique_points(points)
    if len(hull_points) < 4:
        return 0.0
    try:
        from scipy.spatial import ConvexHull
    except ImportError as exc:
        raise RuntimeError("3D bbox IoU requires scipy. Please install scipy in the eval environment.") from exc
    try:
        hull = ConvexHull(hull_points)
        return float(max(hull.volume, 0.0))
    except Exception:
        return 0.0


def bbox_iou_3d(
    pred_size: np.ndarray | None,
    pred_rotation: np.ndarray,
    pred_translation: np.ndarray,
    gt_size: np.ndarray | None,
    gt_rotation: np.ndarray,
    gt_translation: np.ndarray,
) -> float | None:
    if pred_size is None or gt_size is None:
        return None
    pred_volume = box_volume(pred_size)
    gt_volume = box_volume(gt_size)
    if pred_volume <= 0.0 or gt_volume <= 0.0:
        return None
    inter_volume = intersection_volume_obb(
        pred_size,
        pred_rotation,
        pred_translation,
        gt_size,
        gt_rotation,
        gt_translation,
    )
    union = pred_volume + gt_volume - inter_volume
    if union <= 0.0:
        return None
    return float(np.clip(inter_volume / union, 0.0, 1.0))


def project_points(points_cam: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_cam, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    z = points[:, 2]
    valid = z > 1e-8
    uv = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    if np.any(valid):
        projected = points[valid] @ intrinsic.T
        uv[valid] = projected[:, :2] / projected[:, 2:3]
    return uv, valid


def tensor_image_to_uint8(image_tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(image_tensor):
        image = image_tensor.detach().float().cpu().numpy()
    else:
        image = np.asarray(image_tensor, dtype=np.float32)
    image = np.squeeze(image)
    if image.ndim == 3 and image.shape[0] in {1, 3}:
        image = np.transpose(image, (1, 2, 0))
    image = np.clip(image, 0.0, 1.0)
    return np.uint8(np.round(image * 255.0))


def draw_projected_box(
    image: Image.Image,
    corners_cam: np.ndarray | None,
    intrinsic: np.ndarray,
    color: tuple[int, int, int],
    width: int = 3,
):
    if corners_cam is None:
        return
    uv, valid = project_points(corners_cam, intrinsic)
    draw = ImageDraw.Draw(image)
    for start_idx, end_idx in BOX_EDGES:
        if not (valid[start_idx] and valid[end_idx]):
            continue
        p0 = tuple(float(x) for x in uv[start_idx])
        p1 = tuple(float(x) for x in uv[end_idx])
        draw.line([p0, p1], fill=color, width=width)


def depth_to_points_for_vis(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    rgb: np.ndarray,
    stride: int,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.squeeze(np.asarray(depth, dtype=np.float32))
    intrinsic = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    height, width = depth.shape[:2]
    ys = np.arange(0, height, max(1, int(stride)), dtype=np.int32)
    xs = np.arange(0, width, max(1, int(stride)), dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    z = depth[grid_y, grid_x]
    valid = np.isfinite(z) & (z > 1e-6)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    px = grid_x[valid].astype(np.float64)
    py = grid_y[valid].astype(np.float64)
    z = z[valid].astype(np.float64)
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    points = np.stack([(px - cx) * z / fx, (py - cy) * z / fy, z], axis=1).astype(np.float32)
    colors = np.asarray(rgb, dtype=np.uint8)[grid_y[valid], grid_x[valid]]
    if len(points) > max_points:
        keep = np.linspace(0, len(points) - 1, max_points, dtype=np.int32)
        points = points[keep]
        colors = colors[keep]
    return points, colors


def write_ply_with_boxes(
    path: Path,
    points: np.ndarray,
    point_colors: np.ndarray,
    pred_corners: np.ndarray | None,
    gt_corners: np.ndarray | None,
):
    vertices = []
    edges = []
    for point, color in zip(points, point_colors):
        vertices.append((float(point[0]), float(point[1]), float(point[2]), int(color[0]), int(color[1]), int(color[2])))

    def add_box(corners: np.ndarray | None, color: tuple[int, int, int]):
        if corners is None:
            return
        offset = len(vertices)
        for point in np.asarray(corners, dtype=np.float64):
            vertices.append((float(point[0]), float(point[1]), float(point[2]), color[0], color[1], color[2]))
        for start_idx, end_idx in BOX_EDGES:
            edges.append((offset + start_idx, offset + end_idx, color[0], color[1], color[2]))

    add_box(pred_corners, PRED_COLOR)
    add_box(gt_corners, GT_COLOR)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write(f"element edge {len(edges)}\n")
        handle.write("property int vertex1\nproperty int vertex2\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for vertex in vertices:
            handle.write("%.8f %.8f %.8f %d %d %d\n" % vertex)
        for edge in edges:
            handle.write("%d %d %d %d %d\n" % edge)


def save_iou_visualization(
    vis_dir: Path,
    split_name: str,
    sample_index: int,
    batch: Dict,
    batch_index: int,
    object_id: int,
    pred_rotation: np.ndarray,
    pred_translation: np.ndarray,
    pred_size: np.ndarray | None,
    gt_rotation: np.ndarray,
    gt_translation: np.ndarray,
    gt_size: np.ndarray,
    iou: float | None,
    rot_err_deg: float,
    trans_err_cm: float,
    point_stride: int,
):
    vis_dir.mkdir(parents=True, exist_ok=True)
    scene_name = str(batch_item(batch["scene_name"], batch_index))
    image_id = int(np.asarray(batch_item(batch["ids"], batch_index)).reshape(-1)[0])
    safe_scene = re.sub(r"[^a-zA-Z0-9_.-]+", "_", scene_name)[:80]
    prefix = vis_dir / f"{split_name}_{sample_index:05d}_{safe_scene}_obj{object_id:06d}_{image_id:06d}"

    image_rgb = tensor_image_to_uint8(batch_item(batch["images"], batch_index))
    intrinsic = np.asarray(batch_item(batch["intrinsic"], batch_index), dtype=np.float64).reshape(-1, 3, 3)[0]
    pred_corners = transform_bbox_corners(pred_size, pred_rotation, pred_translation) if pred_size is not None else None
    gt_corners = transform_bbox_corners(gt_size, gt_rotation, gt_translation)

    overlay = Image.fromarray(image_rgb.copy())
    draw_projected_box(overlay, gt_corners, intrinsic, GT_COLOR, width=3)
    draw_projected_box(overlay, pred_corners, intrinsic, PRED_COLOR, width=3)
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((4, 4, 360, 68), fill=(0, 0, 0))
    draw.text((10, 10), f"pred green / gt orange", fill=(255, 255, 255))
    draw.text((10, 28), f"IoU={iou if iou is not None else float('nan'):.4f} rot={rot_err_deg:.2f}deg trans={trans_err_cm:.2f}cm", fill=(255, 255, 255))
    draw.text((10, 46), f"pred_size={np.round(pred_size, 4).tolist() if pred_size is not None else None}", fill=(255, 255, 255))
    overlay.save(prefix.with_suffix(".overlay.png"))

    depth = batch_item(batch["depth"], batch_index)
    points, colors = depth_to_points_for_vis(depth, intrinsic, image_rgb, stride=point_stride, max_points=40000)
    write_ply_with_boxes(prefix.with_suffix(".cloud_boxes.ply"), points, colors, pred_corners, gt_corners)
    meta = {
        "split": split_name,
        "scene_name": scene_name,
        "image_id": image_id,
        "object_id": object_id,
        "iou_3d": iou,
        "rotation_error_deg": rot_err_deg,
        "translation_error_cm": trans_err_cm,
        "pred_translation": np.asarray(pred_translation, dtype=float).tolist(),
        "gt_translation": np.asarray(gt_translation, dtype=float).tolist(),
        "pred_size": np.asarray(pred_size, dtype=float).tolist() if pred_size is not None else None,
        "gt_size": np.asarray(gt_size, dtype=float).tolist(),
        "overlay_png": str(prefix.with_suffix(".overlay.png")),
        "ply": str(prefix.with_suffix(".cloud_boxes.ply")),
    }
    prefix.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


class SymmetryRotationScorer:
    def __init__(self, symmetry_info_path: str, continuous_steps: int):
        self.symmetry_info_path = str(symmetry_info_path or "")
        self.continuous_steps = int(continuous_steps)
        self._symmetry_info = None

    @property
    def symmetry_info(self):
        if self._symmetry_info is None and self.symmetry_info_path:
            self._symmetry_info = _load_symmetry_info(self.symmetry_info_path, self.continuous_steps)
        return self._symmetry_info

    def __call__(self, pred_rot: np.ndarray, gt_rot: np.ndarray, object_id: int) -> float:
        if not self.symmetry_info_path:
            return rotation_error_degrees(pred_rot, gt_rot)
        sym_rots = self.symmetry_info.get(int(object_id))
        if sym_rots is None:
            return rotation_error_degrees(pred_rot, gt_rot)
        gt = np.asarray(gt_rot, dtype=np.float64)
        candidates = sym_rots.detach().cpu().numpy().astype(np.float64)
        return float(min(rotation_error_degrees(pred_rot, gt @ sym) for sym in candidates))


@dataclass
class PoseRecord:
    sample_index: int
    scene_name: str
    object_id: int
    image_id: int
    pred_rotation: np.ndarray
    pred_translation: np.ndarray
    gt_rotation: np.ndarray
    gt_translation: np.ndarray
    pred_size: np.ndarray | None = None
    gt_size: np.ndarray | None = None
    iou_3d: float | None = None
    rot_err_deg: float | None = None
    trans_err_cm: float | None = None


def pose_record_to_case(record: PoseRecord, metric: str | None = None) -> Dict:
    case = {
        "case_id": f"abs:{int(record.sample_index)}",
        "case_kind": "absolute",
        "metric": metric,
        "sample_index": int(record.sample_index),
        "scene_name": str(record.scene_name),
        "object_id": int(record.object_id),
        "object_name": f"obj_{int(record.object_id):06d}",
        "image_id": int(record.image_id),
        "frame_name": f"{int(record.image_id):06d}",
        "iou_3d": None if record.iou_3d is None else float(record.iou_3d),
        "rotation_error_deg": None if record.rot_err_deg is None else float(record.rot_err_deg),
        "translation_error_cm": None if record.trans_err_cm is None else float(record.trans_err_cm),
        "pred_translation": np.asarray(record.pred_translation, dtype=float).tolist(),
        "gt_translation": np.asarray(record.gt_translation, dtype=float).tolist(),
        "pred_size": np.asarray(record.pred_size, dtype=float).tolist() if record.pred_size is not None else None,
        "gt_size": np.asarray(record.gt_size, dtype=float).tolist() if record.gt_size is not None else None,
    }
    return case


@dataclass
class MetricAccumulator:
    abs_iou_hits: List[bool] = field(default_factory=list)
    abs_pose_errors: List[tuple[float, float]] = field(default_factory=list)
    pose_records: List[PoseRecord] = field(default_factory=list)

    def add_iou(self, iou: float | None):
        if iou is not None and math.isfinite(iou):
            self.abs_iou_hits.append(iou >= 0.5)

    def add_pose(self, rot_err_deg: float, trans_err_cm: float, record: PoseRecord):
        self.abs_pose_errors.append((float(rot_err_deg), float(trans_err_cm)))
        self.pose_records.append(record)

    @staticmethod
    def _success_rate(errors: Iterable[tuple[float, float]], rot_deg: float, trans_cm: float) -> float | None:
        values = list(errors)
        if not values:
            return None
        return 100.0 * float(np.mean([(r <= rot_deg and t <= trans_cm) for r, t in values]))

    def all_cases_by_metric(self, scorer: SymmetryRotationScorer) -> Dict[str, Dict[str, List[Dict]]]:
        cases: Dict[str, Dict[str, List[Dict]]] = {
            metric: {"matched": [], "missed": []} for metric in METRIC_COLUMNS
        }
        for record in self.pose_records:
            base = pose_record_to_case(record)
            abs_hits = {
                "Abs IoU@50": record.iou_3d is not None and math.isfinite(record.iou_3d) and record.iou_3d >= 0.5,
                "Abs 5°5cm": record.rot_err_deg is not None and record.trans_err_cm is not None and record.rot_err_deg <= 5.0 and record.trans_err_cm <= 5.0,
                "Abs 10°5cm": record.rot_err_deg is not None and record.trans_err_cm is not None and record.rot_err_deg <= 10.0 and record.trans_err_cm <= 5.0,
                "Abs 10°10cm": record.rot_err_deg is not None and record.trans_err_cm is not None and record.rot_err_deg <= 10.0 and record.trans_err_cm <= 10.0,
                "Abs 20°20cm": record.rot_err_deg is not None and record.trans_err_cm is not None and record.rot_err_deg <= 20.0 and record.trans_err_cm <= 20.0,
            }
            for metric, hit in abs_hits.items():
                bucket = "matched" if hit else "missed"
                cases[metric][bucket].append(
                    {
                        **base,
                        "metric": metric,
                        "matched": bool(hit),
                        "status": bucket,
                    }
                )

        return cases

    def matching_cases(self, scorer: SymmetryRotationScorer) -> Dict[str, List[Dict]]:
        all_cases = self.all_cases_by_metric(scorer)
        return {metric: payload["matched"] for metric, payload in all_cases.items()}

    def summary(self, scorer: SymmetryRotationScorer) -> Dict[str, float | None]:
        return {
            "Abs IoU@50": 100.0 * float(np.mean(self.abs_iou_hits)) if self.abs_iou_hits else None,
            "Abs 5°5cm": self._success_rate(self.abs_pose_errors, 5.0, 5.0),
            "Abs 10°5cm": self._success_rate(self.abs_pose_errors, 10.0, 5.0),
            "Abs 10°10cm": self._success_rate(self.abs_pose_errors, 10.0, 10.0),
            "Abs 20°20cm": self._success_rate(self.abs_pose_errors, 20.0, 20.0),
            "num_abs_pose": len(self.abs_pose_errors),
            "num_iou": len(self.abs_iou_hits),
        }


def dataset_kwargs_from_expr(dataset_expr: str, cfg: Dict, split_kind: str, args) -> Dict:
    dset_default = "train" if split_kind == "train" else "test1"
    split_default = cfg.get("train_split_json" if split_kind == "train" else "split_json", None)
    return {
        "dataset_location": parse_dataset_ctor_arg(dataset_expr, "dataset_location", cfg.get("ov9d_root")),
        "dset": parse_dataset_ctor_arg(dataset_expr, "dset", dset_default),
        "split_json": parse_dataset_ctor_arg(dataset_expr, "split_json", split_default),
        "num_object_views": int(parse_dataset_ctor_arg(dataset_expr, "num_object_views", 4)),
        "fixed_object_view_ids": tuple(
            parse_dataset_ctor_arg(
                dataset_expr,
                "fixed_object_view_ids",
                parse_dataset_ctor_arg(dataset_expr, "object_input_views", cfg.get("fixed_object_view_ids", (10, 20, 30, 40))),
            )
        ),
        "strict_fixed_object_view_ids": bool(parse_dataset_ctor_arg(dataset_expr, "strict_fixed_object_view_ids", True)),
        "verify_files": not bool(args.no_verify_files),
        "object_presence_prob": 1.0,
        "z_far": parse_dataset_ctor_arg(dataset_expr, "z_far", 20),
        "resolution": tuple(int(v) for v in cfg.get("resolution", (518, 518))),
        "transform": ImgNorm,
        "seed": int(args.seed),
    }


def build_dataset(dataset_expr: str, cfg: Dict, split_kind: str, args) -> OV9DCameraPose:
    kwargs = dataset_kwargs_from_expr(dataset_expr, cfg, split_kind, args)
    if args.max_samples is not None:
        kwargs["max_records"] = int(args.max_samples)
    return OV9DCameraPose(**kwargs)


def to_device_batch(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def batch_item(value, index: int):
    if torch.is_tensor(value):
        return value[index].detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value[index]
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def evaluate_loader(
    name: str,
    model: OmniVGGT,
    dataloader: DataLoader,
    device: torch.device,
    scorer: SymmetryRotationScorer,
    use_depth_input: bool,
    amp: bool,
    vis_dir: Path | None = None,
    vis_count: int = 0,
    vis_point_stride: int = 4,
) -> MetricAccumulator:
    metrics = MetricAccumulator()
    saved_vis = 0
    sample_index = 0
    for batch in tqdm(dataloader, desc=f"eval {name}", dynamic_ncols=True):
        batch = to_device_batch(batch, device)
        depth = batch["depth"] if use_depth_input else None
        mask = batch["valid_mask"] if use_depth_input else None
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                outputs = model.inference(
                    images=batch["images"],
                    object_images=batch["object_images"],
                    extrinsics=None,
                    intrinsics=None,
                    depth=depth,
                    mask=mask,
                    camera_gt_index=[],
                    depth_gt_index=[0] if use_depth_input else [],
                )

        if "object_pose" not in outputs or "object_translation" not in outputs:
            raise RuntimeError(f"Model output does not contain object pose keys: {sorted(outputs.keys())}")
        pred_pose = outputs["object_pose"].detach().float().cpu().numpy()
        pred_translation = outputs["object_translation"].detach().float().cpu().numpy()
        if "object_size" in outputs:
            pred_size = outputs["object_size"].detach().float().cpu().numpy()
        elif "object_size_log" in outputs:
            pred_size = np.exp(outputs["object_size_log"].detach().float().cpu().numpy())
        else:
            pred_size = None

        batch_size = int(pred_pose.shape[0])
        for i in range(batch_size):
            has_object = bool(np.asarray(batch_item(batch["has_object"], i)).reshape(-1)[0])
            if not has_object:
                continue
            object_id = int(np.asarray(batch_item(batch["object_id"], i)).reshape(-1)[0])
            gt_rotation = np.asarray(batch_item(batch["object_rotation"], i), dtype=np.float32).reshape(3, 3)
            gt_translation = np.asarray(batch_item(batch["object_translation"], i), dtype=np.float32).reshape(3)
            pred_rotation = rot6d_to_matrix(pred_pose[i])
            pred_t = np.asarray(pred_translation[i], dtype=np.float32).reshape(3)
            gt_size = np.asarray(batch_item(batch["object_size"], i), dtype=np.float32).reshape(3)
            one_pred_size = np.asarray(pred_size[i], dtype=np.float32).reshape(3) if pred_size is not None else None
            iou_3d = bbox_iou_3d(
                one_pred_size,
                pred_rotation,
                pred_t,
                gt_size,
                gt_rotation,
                gt_translation,
            )
            metrics.add_iou(iou_3d)

            rot_err = scorer(pred_rotation, gt_rotation, object_id)
            trans_err = translation_l2_cm(pred_t, gt_translation)
            if vis_dir is not None and saved_vis < int(vis_count):
                save_iou_visualization(
                    vis_dir,
                    name,
                    sample_index,
                    batch,
                    i,
                    object_id,
                    pred_rotation,
                    pred_t,
                    one_pred_size,
                    gt_rotation,
                    gt_translation,
                    gt_size,
                    iou_3d,
                    rot_err,
                    trans_err,
                    point_stride=vis_point_stride,
                )
                saved_vis += 1
            scene_name = str(batch_item(batch["scene_name"], i))
            image_id = int(np.asarray(batch_item(batch["ids"], i)).reshape(-1)[0])
            metrics.add_pose(
                rot_err,
                trans_err,
                PoseRecord(
                    sample_index=sample_index,
                    scene_name=scene_name,
                    object_id=object_id,
                    image_id=image_id,
                    pred_rotation=pred_rotation,
                    pred_translation=pred_t,
                    gt_rotation=gt_rotation,
                    gt_translation=gt_translation,
                    pred_size=one_pred_size,
                    gt_size=gt_size,
                    iou_3d=iou_3d,
                    rot_err_deg=rot_err,
                    trans_err_cm=trans_err,
                ),
            )
            sample_index += 1
    return metrics


def fmt_metric(value) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.2f}"


def print_table(results: Dict[str, Dict[str, float | None]]):
    header = ["split", *METRIC_COLUMNS, "num_abs_pose"]
    widths = [max(len(col), 10) for col in header]
    rows = []
    for split, values in results.items():
        row = [split] + [fmt_metric(values.get(col)) for col in METRIC_COLUMNS]
        row += [str(values.get("num_abs_pose", 0))]
        rows.append(row)
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    print(" | ".join(col.ljust(width) for col, width in zip(header, widths)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(cell.ljust(width) for cell, width in zip(row, widths)))


def write_outputs(output_dir: Path, results: Dict[str, Dict[str, float | None]]):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", *METRIC_COLUMNS, "num_abs_pose", "num_iou"])
        writer.writeheader()
        for split, values in results.items():
            writer.writerow({"split": split, **values})


def safe_metric_filename(metric_name: str) -> str:
    return (
        metric_name.replace("°", "deg")
        .replace("@", "at")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("cm", "cm")
    )


def write_matching_cases(output_dir: Path, split: str, cases_by_metric: Dict[str, Dict[str, List[Dict]]]):
    split_dir = output_dir / "matching_cases" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    summary = {metric: len(payload["matched"]) for metric, payload in cases_by_metric.items()}
    (split_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    case_browser_index = {"split": split, "metrics": {}}
    all_matched_cases = []
    all_cases = []
    for metric, payload in cases_by_metric.items():
        matched_cases = payload["matched"]
        missed_cases = payload["missed"]
        metric_path = split_dir / f"{safe_metric_filename(metric)}.jsonl"
        with metric_path.open("w", encoding="utf-8") as handle:
            for case in matched_cases:
                handle.write(json.dumps(case, ensure_ascii=False) + "\n")
                all_matched_cases.append(case)

        metric_all_cases = [*matched_cases, *missed_cases]
        metric_all_path = split_dir / f"{safe_metric_filename(metric)}_all.jsonl"
        with metric_all_path.open("w", encoding="utf-8") as handle:
            for case in metric_all_cases:
                handle.write(json.dumps(case, ensure_ascii=False) + "\n")
                all_cases.append(case)

        case_browser_index["metrics"][metric] = {
            "matched": len(matched_cases),
            "missed": len(missed_cases),
            "total": len(metric_all_cases),
            "case_file": metric_all_path.name,
            "case_kind": "absolute",
        }

    all_path = split_dir / "all_matching_cases.jsonl"
    with all_path.open("w", encoding="utf-8") as handle:
        for case in all_matched_cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")

    all_cases_path = split_dir / "all_cases.jsonl"
    with all_cases_path.open("w", encoding="utf-8") as handle:
        for case in all_cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")

    csv_fields = [
        "case_id",
        "case_kind",
        "metric",
        "matched",
        "status",
        "sample_index",
        "left_sample_index",
        "right_sample_index",
        "scene_name",
        "object_id",
        "object_name",
        "image_id",
        "frame_name",
        "left_image_id",
        "right_image_id",
        "left_frame_name",
        "right_frame_name",
        "iou_3d",
        "rotation_error_deg",
        "translation_error_cm",
    ]
    with (split_dir / "all_matching_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for case in all_matched_cases:
            writer.writerow(case)
    with (split_dir / "all_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for case in all_cases:
            writer.writerow(case)

    (split_dir / "case_browser_index.json").write_text(json.dumps(case_browser_index, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OmniVGGT OV9D object pose metrics on train/eval splits.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config Python file.")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint. Defaults to outputs/0511/model.safetensors.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--splits", nargs="+", default=["eval", "train"], choices=["eval", "train"])
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap per split for quick checks.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "eval_metrics"))
    parser.add_argument("--no-depth-input", action="store_true", help="Match demo inference with depth input disabled.")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA bfloat16 autocast.")
    parser.add_argument("--no-verify-files", action="store_true", help="Skip dataset file existence checks at construction.")
    parser.add_argument("--vis-dir", default=None, help="Optional directory for IoU debug visualizations.")
    parser.add_argument("--vis-count", type=int, default=0, help="Number of positive samples per split to visualize.")
    parser.add_argument("--vis-point-stride", type=int, default=4, help="Stride for depth point cloud in visualization PLY files.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(Path(args.config))
    checkpoint_path = resolve_checkpoint_path(cfg, args.checkpoint)
    device = torch.device(args.device)
    print(f"[eval] config={args.config}")
    print(f"[eval] checkpoint={checkpoint_path}")
    print(f"[eval] device={device} use_depth_input={not args.no_depth_input}")

    model = build_model_from_config(cfg, checkpoint_path, device)
    scorer = SymmetryRotationScorer(
        symmetry_info_path=str(cfg.get("object_srt_symmetry_info_path", "")),
        continuous_steps=int(cfg.get("object_srt_symmetry_continuous_steps", 72)),
    )

    split_exprs = {
        "eval": str(cfg.get("val_dataset", "")),
        "train": str(cfg.get("train_dataset", "")),
    }
    results = {}
    for split in args.splits:
        dataset = build_dataset(split_exprs[split], cfg, split, args)
        if args.max_samples is not None and len(dataset) > int(args.max_samples):
            dataset = Subset(dataset, range(int(args.max_samples)))
        loader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=device.type == "cuda",
            drop_last=False,
            persistent_workers=False,
        )
        metrics = evaluate_loader(
            split,
            model,
            loader,
            device,
            scorer,
            use_depth_input=not args.no_depth_input,
            amp=not args.no_amp,
            vis_dir=(Path(args.vis_dir) / split) if args.vis_dir and int(args.vis_count) > 0 else None,
            vis_count=int(args.vis_count),
            vis_point_stride=int(args.vis_point_stride),
        )
        results[split] = metrics.summary(scorer)
        write_matching_cases(Path(args.output_dir), split, metrics.all_cases_by_metric(scorer))

    print_table(results)
    write_outputs(Path(args.output_dir), results)
    print(f"[eval] wrote {Path(args.output_dir) / 'metrics.json'}")
    print(f"[eval] wrote {Path(args.output_dir) / 'metrics.csv'}")


if __name__ == "__main__":
    main()
