import os
from pathlib import Path

# Must be set before `import gradio` so Gradio cache files land in a writable
# location inside the repo.
PROJECT_ROOT = Path(__file__).resolve().parent
_LOCAL_TMP = PROJECT_ROOT / "tmp"
_LOCAL_TMP.mkdir(parents=True, exist_ok=True)
(_LOCAL_TMP / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", str(_LOCAL_TMP))
os.environ.setdefault("GRADIO_CACHE_DIR", str(_LOCAL_TMP))
os.environ.setdefault("TMPDIR", str(_LOCAL_TMP))
os.environ.setdefault("MPLCONFIGDIR", str(_LOCAL_TMP / "matplotlib"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import inspect
import json
import re
import runpy
import time
from typing import Dict, List, Sequence, Tuple

import cv2
import gradio as gr
import numpy as np
import torch
import trimesh
from PIL import Image
from safetensors.torch import load_file as load_safetensors_file

from omnivggt.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from omnivggt.loss import _load_symmetry_info
from omnivggt.models.omnivggt import OmniVGGT
from omnivggt.utils.image import ImgNorm


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train_oo9d.py"
DEFAULT_PRETRAIN_MODEL = Path(
    "/mnt/train-data-4-hdd/yian/freepose/omni-object_clone/outputs/0515_LARGE/model.safetensors"
)
DEFAULT_GOOGLE_DEMO_ROOT = Path(
    "/mnt/train-data-4-hdd/yian/freepose/ov9d/google_random_template_batch_upright_v2"
)

AXIS_COLORS = ((255, 64, 64), (0, 255, 255), (255, 215, 0))
PRED_BBOX_COLOR = (0, 255, 0)
GT_BBOX_COLOR = (255, 160, 0)

# Visualization-only post-rotation: after the model / GT rotation R is computed,
# the bbox and axes are rotated +90 deg about the object's up axis (+Y / cyan)
# in world space via R_vis = R @ R_VIS_POST_ROT. size_xyz is NOT permuted, so
# the bbox physically turns 90 deg about up rather than just relabeling axes.
R_VIS_POST_ROT = np.array(
    [[ 0.0, 0.0, 1.0],
     [ 0.0, 1.0, 0.0],
     [-1.0, 0.0, 0.0]],
    dtype=np.float64,
)
BBOX_EDGES = (
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
_GRADIO_IMAGE_PARAMS = inspect.signature(gr.Image.__init__).parameters


def compat_image(**kwargs):
    show_download_button = kwargs.pop("show_download_button", None)
    if "show_download_button" in _GRADIO_IMAGE_PARAMS:
        if show_download_button is not None:
            kwargs["show_download_button"] = show_download_button
        return gr.Image(**kwargs)
    if show_download_button is False and "buttons" not in kwargs:
        kwargs["buttons"] = ["share", "fullscreen"]
    return gr.Image(**kwargs)


def load_config(config_path: Path) -> Dict:
    cfg = runpy.run_path(str(config_path))
    return {key: value for key, value in cfg.items() if not key.startswith("__")}


def parse_dataset_ctor_arg(dataset_expr: str, arg_name: str, default=None):
    pattern = rf"{re.escape(arg_name)}\s*=\s*([^,()]+|\([^)]*\))"
    match = re.search(pattern, dataset_expr)
    if not match:
        return default
    try:
        return eval(match.group(1).strip(), {"__builtins__": {}}, {})
    except Exception:
        return default


def resolve_runtime_settings(cfg: Dict) -> Dict:
    dataset_expr = str(cfg.get("val_dataset", cfg.get("train_dataset", "")))
    object_input_views = tuple(
        parse_dataset_ctor_arg(
            dataset_expr,
            "object_input_views",
            default=parse_dataset_ctor_arg(dataset_expr, "fixed_object_view_ids", default=(5, 10, 15, 1)),
        )
    )
    resolution = tuple(int(v) for v in cfg.get("resolution", (518, 518)))
    return {
        "object_input_views": tuple(int(v) for v in object_input_views),
        "resolution": resolution,
    }


def latest_checkpoint_for_experiment(exp_name: str) -> Path | None:
    output_root = PROJECT_ROOT / "outputs" / exp_name
    if not output_root.is_dir():
        return None
    candidates = []
    for checkpoint_dir in output_root.glob("checkpoint-*"):
        model_path = checkpoint_dir / "model.safetensors"
        if model_path.is_file():
            match = re.search(r"checkpoint-(\d+)-(\d+)$", checkpoint_dir.name)
            sort_key = tuple(int(x) for x in match.groups()) if match else (0, 0)
            candidates.append((sort_key, model_path))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def resolve_checkpoint_path(cfg: Dict, checkpoint_path: str | None) -> Path:
    if checkpoint_path:
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    if DEFAULT_PRETRAIN_MODEL.is_file():
        return DEFAULT_PRETRAIN_MODEL
    latest_ckpt = latest_checkpoint_for_experiment(str(cfg.get("exp_name", "")))
    if latest_ckpt is not None:
        return latest_ckpt
    raise FileNotFoundError("Unable to locate a local checkpoint from defaults or outputs directory.")


def build_model_from_config(cfg: Dict, checkpoint_path: Path, device: torch.device) -> OmniVGGT:
    model = OmniVGGT(
        enable_camera=cfg.get("enable_camera", True),
        enable_point=cfg.get("enable_point", True),
        enable_depth=cfg.get("enable_depth", True),
        enable_object_mask=cfg.get("enable_object_mask", False),
        enable_object_srt=cfg.get("enable_object_srt", False),
        enable_object_size=cfg.get("enable_object_size", True),
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
        print(f"[demo] Missing keys: {missing}")
    if unexpected:
        print(f"[demo] Unexpected keys: {unexpected}")
    model.eval().to(device)
    return model


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float64).reshape(3, 2)
    x_raw = rot6d[:, 0]
    y_raw = rot6d[:, 1]
    x = x_raw / max(np.linalg.norm(x_raw), 1e-12)
    y = y_raw - np.dot(x, y_raw) * x
    if np.linalg.norm(y) < 1e-12:
        fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        y = fallback - np.dot(x, fallback) * x
    y = y / max(np.linalg.norm(y), 1e-12)
    z = np.cross(x, y)
    z = z / max(np.linalg.norm(z), 1e-12)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def rotation_error_degrees(pred_rot: np.ndarray, gt_rot: np.ndarray) -> float:
    rel = np.asarray(pred_rot, dtype=np.float64) @ np.asarray(gt_rot, dtype=np.float64).T
    cos_theta = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def translation_error(pred_t: np.ndarray, gt_t: np.ndarray) -> Dict[str, List[float] | float]:
    diff = np.asarray(pred_t, dtype=np.float64) - np.asarray(gt_t, dtype=np.float64)
    return {
        "l2": float(np.linalg.norm(diff)),
        "abs_xyz": np.abs(diff).tolist(),
        "signed_xyz": diff.tolist(),
    }


def symmetric_rotation_error_degrees(
    pred_rot: np.ndarray,
    gt_rot: np.ndarray,
    object_id: int,
    symmetry_info_path: Path | None,
    continuous_steps: int,
) -> Tuple[float, int]:
    if symmetry_info_path is None or not symmetry_info_path.is_file():
        return rotation_error_degrees(pred_rot, gt_rot), 1
    symmetry_info = _load_symmetry_info(str(symmetry_info_path), int(continuous_steps))
    sym_rots = symmetry_info.get(int(object_id))
    if sym_rots is None:
        return rotation_error_degrees(pred_rot, gt_rot), 1
    candidates = sym_rots.detach().cpu().numpy().astype(np.float64)
    errors = [rotation_error_degrees(pred_rot, np.asarray(gt_rot, dtype=np.float64) @ sym) for sym in candidates]
    return float(min(errors)), len(errors)


def read_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_optional_float(value, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def ov9d_read_depth_m(depth_path: Path, camera_entry: Dict) -> np.ndarray:
    depth_raw = np.asarray(Image.open(depth_path), dtype=np.float32)
    depth_m = depth_raw * float(camera_entry.get("depth_scale", 1.0)) / 1000.0
    depth_m[~np.isfinite(depth_m)] = 0.0
    depth_m[depth_m < 0.0] = 0.0
    return depth_m.astype(np.float32)


def ov9d_read_binary_mask(mask_path: Path) -> np.ndarray:
    return (np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0).astype(np.float32)


class DemoScenePreprocessor(BaseStereoViewDataset):
    def __init__(self, resolution):
        super().__init__(dset="google_demo", resolution=resolution, transform=ImgNorm, seed=0)


def crop_resize_image_depth_mask(
    processor: DemoScenePreprocessor,
    image: Image.Image,
    depthmap: np.ndarray,
    object_mask: np.ndarray | None,
    intrinsics: np.ndarray,
    resolution,
    info: str,
):
    if object_mask is None:
        image, depthmap, intrinsics = processor._crop_resize_if_necessary(
            image=image,
            depthmap=depthmap,
            intrinsics=intrinsics.copy(),
            resolution=resolution,
            rng=np.random.default_rng(seed=0),
            info=info,
        )
        return image, depthmap, None, intrinsics

    from importlib import import_module

    ov9d_cls = import_module("omnivggt.datasets.6Dpose.ov9d_camera_pose").OV9DCameraPose
    return ov9d_cls._crop_resize_if_necessary_with_mask(
        processor,
        image=image,
        depthmap=depthmap,
        object_mask=object_mask,
        intrinsics=intrinsics.copy(),
        resolution=resolution,
        rng=np.random.default_rng(seed=0),
        info=info,
    )


def load_object_tensor(
    object_record: Dict,
    object_views: Sequence[int],
    resolution,
    device,
) -> Tuple[torch.Tensor, List[Tuple[np.ndarray, str]]]:
    processor = DemoScenePreprocessor(resolution=resolution)
    resampling = getattr(Image, "Resampling", Image)
    tensors: List[torch.Tensor] = []
    gallery: List[Tuple[np.ndarray, str]] = []
    for image_id in object_views:
        image_path = object_record["object_dir"] / "rgb" / f"{int(image_id):06d}.png"
        mask_path = object_record["object_dir"] / "mask_visib" / f"{int(image_id):06d}_000000.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing object reference view: {image_path}")
        rgb_arr = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        if mask_path.is_file():
            mask_arr = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
            white_bg = np.full_like(rgb_arr, 255)
            white_bg[mask_arr > 0] = rgb_arr[mask_arr > 0]
            image = Image.fromarray(white_bg, mode="RGB")
        else:
            image = Image.fromarray(rgb_arr, mode="RGB")
        image = image.resize(tuple(resolution), resampling.LANCZOS)
        tensors.append(processor.transform(image).to(device))
        gallery.append((np.asarray(image), f"Object view {int(image_id):06d}"))
    return torch.stack(tensors, dim=0).unsqueeze(0), gallery


def load_scene_frame_inputs(
    scene_record: Dict,
    image_id: int,
    resolution,
    device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    scene_dir = scene_record["scene_dir"]
    camera_entry = scene_record["scene_camera"][str(image_id)]
    gts = scene_record["scene_gt"][str(image_id)]
    object_index = next(
        (idx for idx, gt in enumerate(gts) if int(gt.get("obj_id", -1)) == int(scene_record["object_id"])),
        None,
    )

    image = Image.open(scene_dir / "rgb" / f"{image_id:06d}.png").convert("RGB")
    depthmap = ov9d_read_depth_m(scene_dir / "depth" / f"{image_id:06d}.png", camera_entry)
    intrinsics = np.asarray(camera_entry["cam_K"], dtype=np.float32).reshape(3, 3)
    object_mask = None
    if object_index is not None:
        mask_path = scene_dir / "mask_visib" / f"{image_id:06d}_{object_index:06d}.png"
        if mask_path.is_file():
            object_mask = ov9d_read_binary_mask(mask_path)

    processor = DemoScenePreprocessor(resolution=resolution)
    image, depthmap, gt_mask, intrinsics = crop_resize_image_depth_mask(
        processor,
        image,
        depthmap,
        object_mask,
        intrinsics,
        resolution,
        info=str(scene_dir / "rgb" / f"{image_id:06d}.png"),
    )
    image_tensor = processor.transform(image).unsqueeze(0).unsqueeze(0).to(device)
    depth_tensor = torch.from_numpy(np.ascontiguousarray(depthmap.astype(np.float32)))[None, None, :, :, None].to(device)
    mask_tensor = torch.from_numpy((depthmap > 0).astype(np.float32))[None, None, :, :].to(device)
    return image_tensor, depth_tensor, mask_tensor, np.asarray(image.convert("RGB")), depthmap, gt_mask, intrinsics


def project_camera_points(points_cam: np.ndarray, intrinsic: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    z = points_cam[:, 2]
    valid = z > 1e-6
    uv = np.full((points_cam.shape[0], 2), np.nan, dtype=np.float32)
    if np.any(valid):
        uvw = points_cam[valid] @ intrinsic.T
        uv[valid] = (uvw[:, :2] / uvw[:, 2:3]).astype(np.float32)
    return uv, valid


def _axis_object_points(axis_length: float) -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, axis_length],
        ],
        dtype=np.float32,
    )


def centered_axis_bbox_corners(size_xyz: np.ndarray) -> np.ndarray:
    size = np.asarray(size_xyz, dtype=np.float32).reshape(3)
    half = size * 0.5
    xs = [-half[0], half[0]]
    ys = [-half[1], half[1]]
    zs = [-half[2], half[2]]
    return np.array([[x, y, z] for z in zs for y in ys for x in xs], dtype=np.float32)


def transform_bbox_corners(size_xyz: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray | None:
    size = np.asarray(size_xyz, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(size)) or np.any(size <= 0.0):
        return None
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    return centered_axis_bbox_corners(size) @ rotation.T + translation[None, :]


def box_volume(size_xyz: np.ndarray) -> float:
    size = np.asarray(size_xyz, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(size)) or np.any(size <= 0.0):
        return 0.0
    return float(np.prod(size))


def point_inside_obb(
    point: np.ndarray,
    center: np.ndarray,
    rotation: np.ndarray,
    half_size: np.ndarray,
    eps: float = 1e-8,
) -> bool:
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
):
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

    for start_idx, end_idx in BBOX_EDGES:
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
    except ImportError:
        return 0.0
    try:
        hull = ConvexHull(hull_points)
        return float(max(hull.volume, 0.0))
    except Exception:
        return 0.0


def bbox_iou_3d_details(
    pred_size: np.ndarray | None,
    pred_rotation: np.ndarray,
    pred_translation: np.ndarray,
    gt_size: np.ndarray | None,
    gt_rotation: np.ndarray,
    gt_translation: np.ndarray,
) -> Dict[str, float] | None:
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
    return {
        "iou": float(np.clip(inter_volume / union, 0.0, 1.0)),
        "intersection_volume": float(inter_volume),
        "union_volume": float(union),
        "pred_volume": float(pred_volume),
        "gt_volume": float(gt_volume),
    }


def clamp_predicted_size_for_bbox(pred_size: np.ndarray, gt_size: np.ndarray) -> Tuple[np.ndarray, bool]:
    pred = np.asarray(pred_size, dtype=np.float32).reshape(3)
    gt = np.asarray(gt_size, dtype=np.float32).reshape(3)
    finite = np.isfinite(pred) & (pred > 0)
    safe = np.where(finite, pred, gt)
    lower = np.maximum(gt * 0.10, 0.005)
    upper = np.maximum(gt * 4.00, lower + 1e-6)
    clipped = np.clip(safe, lower, upper).astype(np.float32)
    changed = bool(not np.allclose(clipped, pred, rtol=1e-4, atol=1e-6))
    return clipped, changed


def draw_axes_overlay_on_image(
    image_rgb: np.ndarray,
    intrinsic: np.ndarray,
    rotation_cam: np.ndarray,
    translation_cam: np.ndarray,
    axis_length: float,
    axis_colors: Sequence[Tuple[int, int, int]],
) -> np.ndarray:
    pts_cam = _axis_object_points(axis_length) @ rotation_cam.T + translation_cam[None, :]
    uv, valid = project_camera_points(pts_cam, intrinsic)

    overlay = np.asarray(image_rgb, dtype=np.uint8).copy()
    if not bool(valid[0]):
        return overlay
    center = tuple(np.round(uv[0]).astype(np.int32))
    cv2.circle(overlay, center, 4, (255, 255, 255), -1)
    for idx, color in enumerate(axis_colors):
        if not bool(valid[idx + 1]):
            continue
        end = tuple(np.round(uv[idx + 1]).astype(np.int32))
        cv2.arrowedLine(overlay, center, end, color, 3, tipLength=0.18)
    return overlay


def draw_bbox_axes_overlay_on_image(
    image_rgb: np.ndarray,
    intrinsic: np.ndarray,
    rotation_cam: np.ndarray,
    translation_cam: np.ndarray,
    bbox_obj: np.ndarray,
    axis_length: float,
    axis_colors: Sequence[Tuple[int, int, int]],
    bbox_color: Tuple[int, int, int],
) -> np.ndarray:
    overlay = draw_axes_overlay_on_image(
        image_rgb, intrinsic, rotation_cam, translation_cam, axis_length, axis_colors,
    )
    bbox_cam = np.asarray(bbox_obj, dtype=np.float32) @ rotation_cam.T + translation_cam[None, :]
    uv, valid = project_camera_points(bbox_cam, intrinsic)
    height, width = overlay.shape[:2]
    rect = (0, 0, int(width), int(height))
    for start_idx, end_idx in BBOX_EDGES:
        if not (valid[start_idx] and valid[end_idx]):
            continue
        p1 = tuple(np.round(uv[start_idx]).astype(np.int32))
        p2 = tuple(np.round(uv[end_idx]).astype(np.int32))
        ok, cp1, cp2 = cv2.clipLine(rect, p1, p2)
        if ok:
            cv2.line(overlay, cp1, cp2, bbox_color, 2, lineType=cv2.LINE_AA)
    return overlay


def mask_overlay_image(
    image_rgb: np.ndarray,
    mask: np.ndarray | None,
    color: Tuple[int, int, int] = (255, 64, 64),
    threshold: float = 0.5,
) -> np.ndarray | None:
    if mask is None:
        return None
    base = np.asarray(image_rgb, dtype=np.uint8).copy()
    mask_arr = np.asarray(mask, dtype=np.float32)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr.squeeze()
    if mask_arr.shape[:2] != base.shape[:2]:
        mask_arr = cv2.resize(mask_arr, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
    alpha = np.clip(mask_arr, 0.0, 1.0)
    hard = alpha >= float(threshold)
    tint = np.asarray(color, dtype=np.float32)
    out = base.astype(np.float32)
    out[hard] = out[hard] * 0.45 + tint * 0.55
    heat = cv2.applyColorMap(np.uint8(np.clip(alpha, 0, 1) * 255), cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32)
    soft = (alpha > 0.05) & ~hard
    out[soft] = out[soft] * 0.70 + heat[soft] * 0.30
    return np.clip(out, 0, 255).astype(np.uint8)


def mask_iou(pred_mask: np.ndarray | None, gt_mask: np.ndarray | None, threshold: float = 0.5) -> float | None:
    if pred_mask is None or gt_mask is None:
        return None
    pred = np.asarray(pred_mask) >= float(threshold)
    gt = np.asarray(gt_mask).astype(bool)
    if pred.shape != gt.shape:
        pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def depth_to_camera_points(
    depthmap: np.ndarray,
    intrinsic: np.ndarray,
    rgb_image: np.ndarray | None = None,
    *,
    stride: int = 4,
    max_points: int = 25000,
) -> Tuple[np.ndarray, np.ndarray | None]:
    depthmap = np.asarray(depthmap, dtype=np.float32)
    height, width = depthmap.shape[:2]
    ys = np.arange(0, height, max(1, int(stride)), dtype=np.int32)
    xs = np.arange(0, width, max(1, int(stride)), dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    z = depthmap[grid_y, grid_x]
    valid = np.isfinite(z) & (z > 1e-6)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), None

    px = grid_x[valid].astype(np.float32)
    py = grid_y[valid].astype(np.float32)
    z = z[valid].astype(np.float32)
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    x = (px - cx) * z / max(fx, 1e-6)
    y = (py - cy) * z / max(fy, 1e-6)
    points = np.stack([x, y, z], axis=1).astype(np.float32)

    colors = None
    if rgb_image is not None:
        colors = np.asarray(rgb_image, dtype=np.uint8)[grid_y[valid], grid_x[valid]]

    if len(points) > max_points:
        keep = np.linspace(0, len(points) - 1, max_points, dtype=np.int32)
        points = points[keep]
        if colors is not None:
            colors = colors[keep]
    return points, colors


def export_point_cloud_pose_glb(
    scene_name: str,
    frame_name: str,
    rgb_image: np.ndarray,
    depthmap: np.ndarray,
    intrinsic: np.ndarray,
    pred_rotation_cam: np.ndarray | None = None,
    pred_translation_cam: np.ndarray | None = None,
    pred_bbox_obj: np.ndarray | None = None,
    gt_rotation_cam: np.ndarray | None = None,
    gt_translation_cam: np.ndarray | None = None,
    gt_bbox_obj: np.ndarray | None = None,
    axis_length: float = 0.1,
    point_cloud_stride: int = 2,
) -> str:
    stride = max(1, int(point_cloud_stride))
    max_points = 80000 if stride <= 2 else 50000
    points, colors = depth_to_camera_points(depthmap, intrinsic, rgb_image, stride=stride, max_points=max_points)

    safe_scene = re.sub(r"[^a-zA-Z0-9_]+", "_", scene_name)[:120]
    safe_frame = re.sub(r"[^a-zA-Z0-9_]+", "_", str(frame_name))[:120]
    out_dir = _LOCAL_TMP / "google_pose_glb" / safe_scene
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe_frame}_stride{stride}_{int(time.time_ns())}.glb"

    scene_3d = trimesh.Scene()
    if len(points):
        scene_3d.add_geometry(trimesh.PointCloud(vertices=points, colors=colors))

    viewer_axis_length = float(axis_length) * 1.8
    radius = max(viewer_axis_length * 0.025, 2.0e-4)

    def add_segment(start: np.ndarray, end: np.ndarray, color: Tuple[int, int, int]):
        start = np.asarray(start, dtype=np.float32)
        end = np.asarray(end, dtype=np.float32)
        if float(np.linalg.norm(end - start)) < 1e-8:
            return
        mesh = trimesh.creation.cylinder(radius=radius, segment=np.stack([start, end], axis=0))
        rgba = np.array([[color[0], color[1], color[2], 255]], dtype=np.uint8)
        mesh.visual.face_colors = np.tile(rgba, (len(mesh.faces), 1))
        scene_3d.add_geometry(mesh)

    def add_pose(rotation_cam, translation_cam, bbox_obj, bbox_color, axis_colors):
        center = np.asarray(translation_cam, dtype=np.float32)
        axis_pts = _axis_object_points(viewer_axis_length) @ rotation_cam.T + translation_cam[None, :]
        for idx, axis_color in enumerate(axis_colors):
            add_segment(center, axis_pts[idx + 1], axis_color)
        if bbox_obj is not None:
            bbox_cam = np.asarray(bbox_obj, dtype=np.float32) @ rotation_cam.T + translation_cam[None, :]
            for start_idx, end_idx in BBOX_EDGES:
                add_segment(bbox_cam[start_idx], bbox_cam[end_idx], bbox_color)

    if pred_rotation_cam is not None and pred_translation_cam is not None:
        add_pose(pred_rotation_cam, pred_translation_cam, pred_bbox_obj, PRED_BBOX_COLOR, AXIS_COLORS)
    if gt_rotation_cam is not None and gt_translation_cam is not None:
        add_pose(gt_rotation_cam, gt_translation_cam, gt_bbox_obj, GT_BBOX_COLOR, AXIS_COLORS)

    scene_3d.export(out_path)
    return str(out_path)


class GoogleDemoApp:
    def __init__(
        self,
        config_path: Path,
        checkpoint_path: str | None,
        google_demo_root: Path,
        use_gt_pose_for_prediction: bool = False,
    ):
        self.config_path = Path(config_path)
        self.cfg = load_config(self.config_path)
        self.runtime = resolve_runtime_settings(self.cfg)
        self.object_views = tuple(int(v) for v in self.runtime["object_input_views"])
        self.resolution = tuple(int(v) for v in self.runtime["resolution"])
        self.google_demo_root = Path(google_demo_root).expanduser()
        self.data_root, self.object_image_root = self._resolve_google_demo_layout()
        self.use_gt_pose_for_prediction = bool(use_gt_pose_for_prediction)
        self.symmetry_info_path = self._resolve_symmetry_info_path()

        self.scene_records = self._load_scene_records()
        self.scene_choices = sorted(self.scene_records)
        if not self.scene_choices:
            raise RuntimeError(f"No valid Google demo scenes found under {self.data_root}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = resolve_checkpoint_path(self.cfg, checkpoint_path)
        self.model = None
        self.available_checkpoints = self.discover_checkpoints()
        self.load_checkpoint(self.checkpoint_path)

    def _resolve_google_demo_layout(self) -> Tuple[Path, Path]:
        nested_data_root = self.google_demo_root / "data"
        nested_object_root = nested_data_root / "_object_images"
        if nested_data_root.is_dir() and nested_object_root.is_dir():
            return nested_data_root, nested_object_root

        flat_object_root = self.google_demo_root / "_object_images"
        if self.google_demo_root.is_dir() and flat_object_root.is_dir():
            return self.google_demo_root, flat_object_root

        raise FileNotFoundError(
            "Unable to resolve Google demo layout. Expected either "
            f"`{nested_object_root}` or `{flat_object_root}`."
        )

    def _resolve_symmetry_info_path(self) -> Path | None:
        value = self.cfg.get("object_srt_symmetry_info_path")
        if not value:
            return None
        path = Path(str(value)).expanduser()
        if path.is_file():
            return path
        candidate = PROJECT_ROOT / str(value)
        return candidate if candidate.is_file() else None

    def _load_scene_records(self) -> Dict[str, Dict]:
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"Google demo data root not found: {self.data_root}")
        if not self.object_image_root.is_dir():
            raise FileNotFoundError(f"Google demo object image root not found: {self.object_image_root}")

        records: Dict[str, Dict] = {}
        for scene_dir in sorted(p for p in self.data_root.iterdir() if p.is_dir() and p.name != "_object_images"):
            rgb_dir = scene_dir / "rgb"
            depth_dir = scene_dir / "depth"
            mask_dir = scene_dir / "mask_visib"
            scene_gt_path = scene_dir / "scene_gt.json"
            scene_camera_path = scene_dir / "scene_camera.json"
            scene_meta_path = scene_dir / "scene_meta.json"
            object_dir = self.object_image_root / scene_dir.name
            if not all(
                [
                    rgb_dir.is_dir(),
                    depth_dir.is_dir(),
                    mask_dir.is_dir(),
                    scene_gt_path.is_file(),
                    scene_camera_path.is_file(),
                    scene_meta_path.is_file(),
                    object_dir.is_dir(),
                    (object_dir / "rgb").is_dir(),
                ]
            ):
                continue

            scene_gt = read_json(scene_gt_path)
            scene_camera = read_json(scene_camera_path)
            scene_meta = read_json(scene_meta_path)
            metadata = read_json(scene_dir / "metadata.json") if (scene_dir / "metadata.json").is_file() else {}
            frame_ids = sorted(int(path.stem) for path in rgb_dir.glob("*.png"))
            object_view_ids = sorted(int(path.stem) for path in (object_dir / "rgb").glob("*.png"))
            if not frame_ids or not object_view_ids:
                continue

            first_key = str(frame_ids[0]) if str(frame_ids[0]) in scene_gt else next(iter(scene_gt))
            first_gt = scene_gt[first_key][0]
            first_meta = scene_meta[first_key][0] if isinstance(scene_meta[first_key], list) else scene_meta[first_key]
            object_id = int(first_gt.get("obj_id", 1))
            object_name = str(metadata.get("obj_name") or scene_dir.name)
            default_object_views = [view_id for view_id in self.object_views if view_id in object_view_ids]
            if len(default_object_views) != len(self.object_views):
                default_object_views = object_view_ids[: min(len(object_view_ids), max(1, len(self.object_views)))]

            records[scene_dir.name] = {
                "scene_name": scene_dir.name,
                "object_name": object_name,
                "scene_dir": scene_dir,
                "object_dir": object_dir,
                "scene_gt": scene_gt,
                "scene_camera": scene_camera,
                "scene_meta": scene_meta,
                "frame_ids": frame_ids,
                "object_view_ids": object_view_ids,
                "default_object_views": default_object_views,
                "object_id": object_id,
                "size_xyz": np.asarray(
                    [first_meta.get("size_x", 100.0), first_meta.get("size_y", 100.0), first_meta.get("size_z", 100.0)],
                    dtype=np.float32,
                )
                / 1000.0,
            }
        return records

    def discover_checkpoints(self) -> List[str]:
        candidates = set()
        if DEFAULT_PRETRAIN_MODEL.is_file():
            candidates.add(str(DEFAULT_PRETRAIN_MODEL))
        latest_ckpt = latest_checkpoint_for_experiment(str(self.cfg.get("exp_name", "")))
        if latest_ckpt is not None:
            candidates.add(str(latest_ckpt))
        for path in sorted((PROJECT_ROOT / "outputs").glob("**/model.safetensors")):
            candidates.add(str(path))
        return sorted(candidates)

    def load_checkpoint(self, checkpoint_path: str | Path):
        resolved = Path(checkpoint_path).expanduser()
        if not resolved.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resolved}")
        self.model = build_model_from_config(self.cfg, resolved, self.device)
        self.checkpoint_path = resolved
        if str(resolved) not in self.available_checkpoints:
            self.available_checkpoints = sorted([*self.available_checkpoints, str(resolved)])
        return (
            f"Loaded checkpoint: `{self.checkpoint_path}`",
            gr.update(choices=self.available_checkpoints, value=str(self.checkpoint_path)),
            str(self.checkpoint_path),
        )

    def scene_record(self, scene_name: str) -> Dict:
        if scene_name not in self.scene_records:
            raise KeyError(f"Unknown scene: {scene_name}")
        return self.scene_records[scene_name]

    def frame_choices(self, scene_name: str) -> List[str]:
        return [f"{frame_id:06d}" for frame_id in self.scene_record(scene_name)["frame_ids"]]

    def scene_preview(self, scene_name: str, frame_name: str) -> str | None:
        if not scene_name or not frame_name:
            return None
        return str(self.scene_record(scene_name)["scene_dir"] / "rgb" / f"{int(frame_name):06d}.png")

    def object_gallery(self, scene_name: str):
        record = self.scene_record(scene_name)
        _, gallery = load_object_tensor(
            record,
            record["default_object_views"],
            self.resolution,
            torch.device("cpu"),
        )
        return gallery

    def object_summary(self, scene_name: str) -> str:
        record = self.scene_record(scene_name)
        return "\n".join(
            [
                "### Object Reference",
                f"- object: `{record['object_name']}`",
                f"- object id: `{record['object_id']}`",
                f"- object image dir: `{record['object_dir']}`",
                f"- default object views: `{record['default_object_views']}`",
            ]
        )

    def axis_length(self, scene_name: str) -> float:
        size_xyz = self.scene_record(scene_name)["size_xyz"]
        return max(float(np.linalg.norm(size_xyz)) * 0.25, 1e-3)

    def run_inference(
        self,
        scene_name: str,
        frame_name: str,
        use_depth_input: bool,
        show_point_cloud_pose: bool,
        point_cloud_stride: int,
        apply_pred_up_axis_90deg: bool = False,
    ):
        record = self.scene_record(scene_name)
        object_id = int(record["object_id"])
        image_id = int(frame_name)
        use_depth_input = bool(use_depth_input)
        apply_pred_up_axis_90deg = bool(apply_pred_up_axis_90deg)

        gts = record["scene_gt"][str(image_id)]
        object_index = next(
            (idx for idx, gt in enumerate(gts) if int(gt.get("obj_id", -1)) == object_id),
            None,
        )
        has_object = object_index is not None

        scene_tensor, depth_tensor, mask_tensor, display_image, display_depth, gt_mask, intrinsic = load_scene_frame_inputs(
            record,
            image_id,
            self.resolution,
            self.device,
        )
        object_tensor, _ = load_object_tensor(
            record,
            record["default_object_views"],
            self.resolution,
            self.device,
        )

        with torch.inference_mode():
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                outputs = self.model.inference(
                    images=scene_tensor,
                    object_images=object_tensor,
                    extrinsics=None,
                    intrinsics=None,
                    depth=depth_tensor if use_depth_input else None,
                    mask=mask_tensor if use_depth_input else None,
                    camera_gt_index=[],
                    depth_gt_index=[0] if use_depth_input else [],
                )

        if "object_presence_logits" not in outputs:
            raise RuntimeError(f"Model output does not contain object_presence_logits: {sorted(outputs.keys())}")

        presence_logit = float(outputs["object_presence_logits"].reshape(-1)[0].detach().float().cpu())
        presence_prob = float(torch.sigmoid(torch.tensor(presence_logit)).item())
        pred_present = presence_prob >= 0.5

        pred_mask = None
        pred_mask_image = None
        if "object_mask" in outputs:
            pred_mask = outputs["object_mask"][0, 0].detach().float().cpu().numpy()
            pred_mask_image = mask_overlay_image(display_image, pred_mask, color=(255, 64, 64))
        gt_mask_image = mask_overlay_image(display_image, gt_mask, color=(80, 255, 120)) if gt_mask is not None else None

        pred_size = None
        if "object_size" in outputs:
            pred_size = outputs["object_size"].reshape(-1, 3)[0].detach().float().cpu().numpy().astype(np.float32)
        elif "object_size_log" in outputs:
            pred_size = np.exp(outputs["object_size_log"].reshape(-1, 3)[0].detach().float().cpu().numpy()).astype(np.float32)

        lines = [
            "### Prediction Summary",
            f"- checkpoint: `{self.checkpoint_path}`",
            f"- scene: `{scene_name}`",
            f"- frame: `{image_id:06d}`",
            f"- object: `{record['object_name']}`",
            f"- GT presence in frame: `{bool(has_object)}`",
            f"- predicted presence probability: `{presence_prob:.6f}`",
            f"- predicted present @0.5: `{bool(pred_present)}`",
            f"- use depth input: `{use_depth_input}`",
        ]

        if not has_object:
            point_cloud_glb = None
            if show_point_cloud_pose:
                point_cloud_glb = export_point_cloud_pose_glb(
                    scene_name,
                    f"{image_id:06d}",
                    display_image,
                    display_depth,
                    intrinsic,
                    axis_length=self.axis_length(scene_name),
                    point_cloud_stride=point_cloud_stride,
                )
            lines += [
                "",
                "### Presence-Only Result",
                "- This frame does not contain the selected object annotation, so pose metrics are skipped.",
            ]
            return "\n".join(lines), None, None, point_cloud_glb, pred_mask_image, gt_mask_image

        if "object_pose" not in outputs or "object_translation" not in outputs:
            raise RuntimeError(f"Model output does not contain object pose keys: {sorted(outputs.keys())}")

        pred_rot6d_cam = outputs["object_pose"][0].detach().float().cpu().numpy()
        pred_translation_cam = outputs["object_translation"][0].detach().float().cpu().numpy().astype(np.float32)
        pred_rotation_cam = rot6d_to_matrix(pred_rot6d_cam).astype(np.float32)

        gt = gts[object_index]
        gt_rotation_cam = np.asarray(gt["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
        gt_translation_cam = np.asarray(gt["cam_t_m2c"], dtype=np.float32).reshape(3) / 1000.0
        if self.use_gt_pose_for_prediction:
            pred_rotation_cam = gt_rotation_cam
            pred_translation_cam = gt_translation_cam

        # Optionally apply +90 deg post-rotation about the object's up axis
        # (+Y) to the predicted rotation. When enabled this affects rotation
        # error and the predicted bbox + axes drawing. GT is never rotated.
        if apply_pred_up_axis_90deg:
            post_rot = R_VIS_POST_ROT.astype(pred_rotation_cam.dtype)
            pred_rotation_vis = pred_rotation_cam @ post_rot
        else:
            pred_rotation_vis = pred_rotation_cam

        rot_error_deg, sym_count = symmetric_rotation_error_degrees(
            pred_rotation_vis,
            gt_rotation_cam,
            object_id,
            self.symmetry_info_path,
            int(self.cfg.get("object_srt_symmetry_continuous_steps", 72)),
        )
        trans_error = translation_error(pred_translation_cam, gt_translation_cam)
        mask_score = mask_iou(pred_mask, gt_mask)

        axis_length = self.axis_length(scene_name)
        gt_size = record["size_xyz"]
        raw_pred_size = pred_size if pred_size is not None else gt_size
        size_for_pred_box, size_was_clipped = clamp_predicted_size_for_bbox(raw_pred_size, gt_size)
        pred_bbox_obj = centered_axis_bbox_corners(size_for_pred_box)
        gt_bbox_obj = centered_axis_bbox_corners(gt_size)
        raw_bbox_iou_details = bbox_iou_3d_details(
            raw_pred_size,
            pred_rotation_cam,
            pred_translation_cam,
            gt_size,
            gt_rotation_cam,
            gt_translation_cam,
        )
        visualized_bbox_iou_details = bbox_iou_3d_details(
            size_for_pred_box,
            pred_rotation_cam,
            pred_translation_cam,
            gt_size,
            gt_rotation_cam,
            gt_translation_cam,
        )

        # The predicted bbox + axes are drawn using pred_rotation_vis (computed
        # above), which adds a +90 deg rotation about the object's up axis on
        # top of the model output. GT is drawn with its native rotation.
        pred_image = draw_bbox_axes_overlay_on_image(
            display_image,
            intrinsic,
            pred_rotation_vis,
            pred_translation_cam,
            pred_bbox_obj,
            axis_length,
            AXIS_COLORS,
            PRED_BBOX_COLOR,
        )
        gt_image = draw_bbox_axes_overlay_on_image(
            display_image,
            intrinsic,
            gt_rotation_cam,
            gt_translation_cam,
            gt_bbox_obj,
            axis_length,
            AXIS_COLORS,
            GT_BBOX_COLOR,
        )

        point_cloud_glb = None
        if show_point_cloud_pose:
            point_cloud_glb = export_point_cloud_pose_glb(
                scene_name,
                f"{image_id:06d}",
                display_image,
                display_depth,
                intrinsic,
                pred_rotation_cam=pred_rotation_vis,
                pred_translation_cam=pred_translation_cam,
                pred_bbox_obj=pred_bbox_obj,
                gt_rotation_cam=gt_rotation_cam,
                gt_translation_cam=gt_translation_cam,
                gt_bbox_obj=gt_bbox_obj,
                axis_length=axis_length,
                point_cloud_stride=point_cloud_stride,
            )

        size_error = translation_error(raw_pred_size, gt_size)
        lines += [
            "",
            "### Predicted Pose",
            f"- camera-frame translation (m): `{np.round(pred_translation_cam, 6).tolist()}`",
            f"- camera-frame rotation matrix: `{np.round(pred_rotation_cam, 6).tolist()}`",
            f"- raw predicted size xyz (m): `{np.round(raw_pred_size, 6).tolist()}`",
            f"- visualized predicted bbox size xyz (m): `{np.round(size_for_pred_box, 6).tolist()}`"
            + (" clipped for display" if size_was_clipped else ""),
            "",
            "### Ground Truth Pose",
            f"- camera-frame translation (m): `{np.round(gt_translation_cam, 6).tolist()}`",
            f"- camera-frame rotation matrix: `{np.round(gt_rotation_cam, 6).tolist()}`",
            f"- GT size xyz (m): `{np.round(gt_size, 6).tolist()}`",
            "",
            "### Errors",
            f"- translation L2 (m): `{trans_error['l2']:.6f}`",
            f"- translation abs xyz (m): `{np.round(trans_error['abs_xyz'], 6).tolist()}`",
            f"- raw predicted size L2 (m): `{size_error['l2']:.6f}`",
            f"- raw predicted size abs xyz (m): `{np.round(size_error['abs_xyz'], 6).tolist()}`",
            f"- symmetric rotation error (deg): `{rot_error_deg:.6f}` using `{sym_count}` symmetry candidates",
            (
                f"- 3D bbox IoU, raw size: `{raw_bbox_iou_details['iou']:.6f}` "
                f"(intersection `{raw_bbox_iou_details['intersection_volume']:.8f}`, "
                f"union `{raw_bbox_iou_details['union_volume']:.8f}`)"
                if raw_bbox_iou_details is not None else "- 3D bbox IoU, raw size: `N/A`"
            ),
            (
                f"- 3D bbox IoU, visualized size: `{visualized_bbox_iou_details['iou']:.6f}` "
                f"(intersection `{visualized_bbox_iou_details['intersection_volume']:.8f}`, "
                f"union `{visualized_bbox_iou_details['union_volume']:.8f}`)"
                if visualized_bbox_iou_details is not None else "- 3D bbox IoU, visualized size: `N/A`"
            ),
            f"- mask IoU @0.5: `{mask_score:.6f}`" if mask_score is not None else "- mask IoU @0.5: `N/A`",
        ]
        return "\n".join(lines), pred_image, gt_image, point_cloud_glb, pred_mask_image, gt_mask_image


def build_demo(app: GoogleDemoApp):
    default_scene = app.scene_choices[0]
    default_frames = app.frame_choices(default_scene)
    default_frame = default_frames[0] if default_frames else None

    demo_css = """
    .gradio-container { max-width: 100% !important; }
    #scene_inputs, #pred_projection, #gt_projection { background: #1c1c1c; }
    #scene_inputs img, #pred_projection img, #gt_projection img,
    #scene_inputs button img, #pred_projection button img, #gt_projection button img {
        object-fit: contain !important;
        max-height: 100% !important;
        max-width: 100% !important;
        width: auto !important;
        height: auto !important;
    }
    #scene_inputs > div, #pred_projection > div, #gt_projection > div { overflow: hidden !important; }
    #object_inputs img { object-fit: contain !important; background: #1c1c1c; }
    """

    info_markdown = "\n".join(
        [
            "# OmniVGGT 6D Pose Demo (Google Only)",
            f"- config: `{app.config_path}`",
            f"- google demo root: `{app.google_demo_root}`",
            f"- data root: `{app.data_root}`",
            f"- object image root: `{app.object_image_root}`",
            f"- loaded checkpoint: `{app.checkpoint_path}`",
            f"- object views: `{app.object_views}`",
            f"- inference resolution: `{app.resolution}`",
            "",
            "這個頁面只保留 Google demo。",
            "支援兩種目錄格式：`<root>/data/<scene>/rgb` 或 `<root>/<scene>/rgb`。",
            "object reference 對應 `<data_root>/_object_images/<scene>/rgb`。",
        ]
    )

    with gr.Blocks(title="OmniVGGT Google Demo", css=demo_css) as demo:
        gr.Markdown(info_markdown)

        checkpoint_status = gr.Markdown(f"Loaded checkpoint: `{app.checkpoint_path}`")
        with gr.Row():
            checkpoint_dropdown = gr.Dropdown(
                choices=app.available_checkpoints,
                value=str(app.checkpoint_path),
                label="Checkpoint Presets",
                allow_custom_value=True,
            )
            checkpoint_textbox = gr.Textbox(
                value=str(app.checkpoint_path),
                label="Checkpoint Path",
                placeholder="/path/to/model.safetensors",
            )
            load_model_button = gr.Button("Load Model")

        with gr.Row():
            scene_dropdown = gr.Dropdown(choices=app.scene_choices, value=default_scene, label="Scene")
            frame_dropdown = gr.Dropdown(choices=default_frames, value=default_frame, label="Frame")
            use_depth_checkbox = gr.Checkbox(value=True, label="Use Depth Input")
            show_point_cloud_checkbox = gr.Checkbox(value=False, label="Show Point Cloud Pose")
            apply_pred_up_axis_90deg_checkbox = gr.Checkbox(
                value=False,
                label="Apply +90 deg about up axis to pred",
            )
            point_cloud_stride_slider = gr.Slider(
                minimum=1,
                maximum=8,
                value=2,
                step=1,
                label="Point Cloud Density",
                info="Smaller = denser RGB point cloud",
            )
            infer_button = gr.Button("Run Inference", variant="primary")

        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                object_summary_markdown = gr.Markdown(app.object_summary(default_scene))
                scene_input_image = compat_image(
                    label="Scene Input (RGB)",
                    height=360,
                    elem_id="scene_inputs",
                    interactive=False,
                )
                object_input_gallery = gr.Gallery(
                    label="Object Inputs",
                    columns=max(len(app.scene_record(default_scene)["default_object_views"]), 1),
                    height=180,
                    elem_id="object_inputs",
                    preview=False,
                    object_fit="contain",
                    show_label=True,
                    allow_preview=True,
                )
            with gr.Column(scale=1):
                summary_markdown = gr.Markdown()
                pred_image = compat_image(
                    label="Predicted axes (camera frame)",
                    height=360,
                    elem_id="pred_projection",
                    interactive=False,
                    show_download_button=False,
                )
                gt_image = compat_image(
                    label="GT axes (camera frame)",
                    height=360,
                    elem_id="gt_projection",
                    interactive=False,
                    show_download_button=False,
                )
                point_cloud_model = gr.Model3D(
                    label="Depth Point Cloud + Pose",
                    height=520,
                    zoom_speed=0.6,
                    pan_speed=0.6,
                    display_mode="solid",
                    clear_color=(0.0, 0.0, 0.0, 0.0),
                )
                with gr.Row():
                    pred_mask_image = compat_image(
                        label="Predicted object mask",
                        height=180,
                        interactive=False,
                        show_download_button=False,
                    )
                    gt_mask_image = compat_image(
                        label="GT object mask",
                        height=180,
                        interactive=False,
                        show_download_button=False,
                    )

        def sync_checkpoint_path(selected_value: str):
            return selected_value

        def refresh_scene(scene_name: str):
            frames = app.frame_choices(scene_name)
            frame_value = frames[0] if frames else None
            return (
                gr.update(choices=frames, value=frame_value),
                app.scene_preview(scene_name, frame_value) if frame_value else None,
                app.object_gallery(scene_name),
                app.object_summary(scene_name),
            )

        def refresh_frame(scene_name: str, frame_name: str):
            return app.scene_preview(scene_name, frame_name)

        scene_dropdown.change(
            refresh_scene,
            inputs=scene_dropdown,
            outputs=[frame_dropdown, scene_input_image, object_input_gallery, object_summary_markdown],
        )
        frame_dropdown.change(refresh_frame, inputs=[scene_dropdown, frame_dropdown], outputs=scene_input_image)
        checkpoint_dropdown.change(sync_checkpoint_path, inputs=checkpoint_dropdown, outputs=checkpoint_textbox)
        load_model_button.click(
            app.load_checkpoint,
            inputs=checkpoint_textbox,
            outputs=[checkpoint_status, checkpoint_dropdown, checkpoint_textbox],
        )
        infer_button.click(
            app.run_inference,
            inputs=[
                scene_dropdown,
                frame_dropdown,
                use_depth_checkbox,
                show_point_cloud_checkbox,
                point_cloud_stride_slider,
                apply_pred_up_axis_90deg_checkbox,
            ],
            outputs=[summary_markdown, pred_image, gt_image, point_cloud_model, pred_mask_image, gt_mask_image],
        )
        if default_frame is not None:
            demo.load(
                lambda: (
                    app.scene_preview(default_scene, default_frame),
                    app.object_gallery(default_scene),
                    app.object_summary(default_scene),
                ),
                outputs=[scene_input_image, object_input_gallery, object_summary_markdown],
            )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Gradio demo for OmniVGGT 6D pose inference on Google demo data")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--google-demo-root", type=Path, default=DEFAULT_GOOGLE_DEMO_ROOT)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--use-gt-pose-for-prediction", action="store_true")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    app = GoogleDemoApp(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        google_demo_root=args.google_demo_root,
        use_gt_pose_for_prediction=args.use_gt_pose_for_prediction,
    )
    demo = build_demo(app)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[str(app.google_demo_root), str(app.object_image_root)],
    )


if __name__ == "__main__":
    main()
