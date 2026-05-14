import os
from pathlib import Path

# Must be set before `import gradio` so Gradio's cache files land in a writable
# location instead of /tmp/gradio.
PROJECT_ROOT = Path(__file__).resolve().parent
_LOCAL_TMP = PROJECT_ROOT / "tmp"
_LOCAL_TMP.mkdir(parents=True, exist_ok=True)
(_LOCAL_TMP / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", str(_LOCAL_TMP))
os.environ.setdefault("GRADIO_CACHE_DIR", str(_LOCAL_TMP))
os.environ.setdefault("TMPDIR", str(_LOCAL_TMP))
os.environ.setdefault("MPLCONFIGDIR", str(_LOCAL_TMP / "matplotlib"))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# cd /mnt/train-data-4-hdd/yian/freepose/omni-object_clone
# python3 demo_gradio_6dpose_single.py --port 7860

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


# ============================================================
# Constants
# ============================================================
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train_ov9d_camera_pose.py"
DEFAULT_DATASET_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d")
DEFAULT_SCENE_SUBDIR = "oo3d9dsingle"
DEFAULT_PRETRAIN_MODEL = Path(
    "/mnt/train-data-4-hdd/yian/freepose/omni-object_clone/outputs/0511/12000/model.safetensors"
)
DEFAULT_MATCHING_CASES_ROOT = PROJECT_ROOT / "outputs" / "0511" / "12000" / "metric" / "matching_cases"

AXIS_COLORS = ((255, 64, 64), (0, 255, 255), (255, 215, 0))
PRED_AXIS_COLORS = AXIS_COLORS
GT_AXIS_COLORS = AXIS_COLORS
PRED_BBOX_COLOR = (0, 255, 0)
GT_BBOX_COLOR = (255, 160, 0)
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


# ============================================================
# Config & model loading
# ============================================================
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
    train_dataset_expr = str(cfg.get("train_dataset", ""))
    object_input_views = tuple(
        parse_dataset_ctor_arg(
            dataset_expr,
            "object_input_views",
            default=parse_dataset_ctor_arg(dataset_expr, "fixed_object_view_ids", default=(1, 5, 10, 15)),
        )
    )
    resolution = tuple(int(v) for v in cfg.get("resolution", (518, 518)))
    return {
        "object_input_views": tuple(int(v) for v in object_input_views),
        "resolution": resolution,
        "dataset_location": parse_dataset_ctor_arg(dataset_expr, "dataset_location", default=None),
        "split_json": parse_dataset_ctor_arg(dataset_expr, "split_json", default=None),
        "dset": parse_dataset_ctor_arg(dataset_expr, "dset", default="test1"),
        "train_split_json": parse_dataset_ctor_arg(train_dataset_expr, "split_json", default=None),
        "train_dset": parse_dataset_ctor_arg(train_dataset_expr, "dset", default="train"),
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


# ============================================================
# Pose math
# ============================================================
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
    symmetry_info_path: str,
    continuous_steps: int,
) -> Tuple[float, int]:
    if not symmetry_info_path:
        return rotation_error_degrees(pred_rot, gt_rot), 1
    symmetry_info = _load_symmetry_info(str(symmetry_info_path), int(continuous_steps))
    sym_rots = symmetry_info.get(int(object_id))
    if sym_rots is None:
        return rotation_error_degrees(pred_rot, gt_rot), 1
    candidates = sym_rots.detach().cpu().numpy().astype(np.float64)
    errors = [rotation_error_degrees(pred_rot, np.asarray(gt_rot, dtype=np.float64) @ sym) for sym in candidates]
    return float(min(errors)), len(errors)


# ============================================================
# OV9D file IO helpers
# ============================================================
def read_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ov9d_object_key(object_id: int) -> str:
    return f"obj_{int(object_id):06d}"


def ov9d_object_id_from_key(value) -> int:
    match = re.search(r"(\d+)$", str(value))
    if not match:
        raise ValueError(f"Could not parse OV9D object id from: {value}")
    return int(match.group(1))


def ov9d_object_display_name(object_id: int, oid_to_name: Dict[int, str]) -> str:
    object_id = int(object_id)
    return f"{ov9d_object_key(object_id)} · {oid_to_name.get(object_id, 'unknown')}"


def ov9d_single_object_instance_from_scene(scene_name: str) -> str:
    name_parts = str(scene_name).split("_")
    return "_".join(name_parts[:-1]) if len(name_parts) > 2 else str(scene_name)


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


def safe_metric_filename(metric_name: str) -> str:
    return (
        str(metric_name).replace("°", "deg")
        .replace("@", "at")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("cm", "cm")
    )


def ov9d_read_depth_m(depth_path: Path, camera_entry: Dict) -> np.ndarray:
    depth_raw = np.asarray(Image.open(depth_path), dtype=np.float32)
    depth_m = depth_raw * float(camera_entry.get("depth_scale", 1.0)) / 1000.0
    depth_m[~np.isfinite(depth_m)] = 0.0
    depth_m[depth_m < 0.0] = 0.0
    return depth_m.astype(np.float32)


def ov9d_read_binary_mask(mask_path: Path) -> np.ndarray:
    return (np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0).astype(np.float32)


# ============================================================
# OV9D tensor loaders (scene frame + object reference views)
# ============================================================
class DemoScenePreprocessor(BaseStereoViewDataset):
    def __init__(self, resolution):
        super().__init__(dset="demo", resolution=resolution, transform=ImgNorm, seed=0)


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

    # Use the OV9D mask-aware crop/resize so GT masks stay aligned to input pixels.
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


def load_ov9d_object_tensor(
    single_records_by_object_id: Dict[int, List[Dict]],
    object_id: int,
    object_views: Sequence[int],
    resolution,
    device,
) -> Tuple[torch.Tensor, List[Tuple[np.ndarray, str]]]:
    object_id = int(object_id)
    if object_id not in single_records_by_object_id:
        raise FileNotFoundError(f"No OV9D single-object reference renders for object id {object_id}")
    single_rec = single_records_by_object_id[object_id][0]
    processor = DemoScenePreprocessor(resolution=resolution)
    resampling = getattr(Image, "Resampling", Image)
    tensors: List[torch.Tensor] = []
    gallery: List[Tuple[np.ndarray, str]] = []
    for image_id in object_views:
        image_path = single_rec["scene_dir"] / "rgb" / f"{int(image_id):06d}.png"
        mask_path = single_rec["scene_dir"] / "mask_visib" / f"{int(image_id):06d}_000000.png"
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Missing OV9D object reference view: {image_path} / {mask_path}")
        rgb_arr = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        mask_arr = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        white_bg = np.full_like(rgb_arr, 255)
        white_bg[mask_arr > 0] = rgb_arr[mask_arr > 0]
        image = Image.fromarray(white_bg, mode="RGB").resize(tuple(resolution), resampling.LANCZOS)
        tensors.append(processor.transform(image).to(device))
        gallery.append((np.asarray(image), f"Object view {int(image_id)}"))
    return torch.stack(tensors, dim=0).unsqueeze(0), gallery


def load_ov9d_scene_frame_inputs(
    multi_root: Path,
    scene_name: str,
    image_id: int,
    object_id: int,
    resolution,
    device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    scene_dir = Path(multi_root) / scene_name
    camera_entry = read_json(scene_dir / "scene_camera.json")[str(image_id)]
    gts = read_json(scene_dir / "scene_gt.json")[str(image_id)]
    object_index = next(
        (idx for idx, gt in enumerate(gts) if int(gt.get("obj_id", -1)) == int(object_id)),
        None,
    )

    image = Image.open(scene_dir / "rgb" / f"{image_id:06d}.png").convert("RGB")
    depthmap = ov9d_read_depth_m(scene_dir / "depth" / f"{image_id:06d}.png", camera_entry)
    intrinsics = np.asarray(camera_entry["cam_K"], dtype=np.float32).reshape(3, 3)
    object_mask = None
    if object_index is not None:
        mask_path = scene_dir / "mask_visib" / f"{image_id:06d}_{object_index:06d}.png"
        object_mask = ov9d_read_binary_mask(mask_path)

    processor = DemoScenePreprocessor(resolution=resolution)
    image, depthmap, gt_mask, intrinsics = crop_resize_image_depth_mask(
        processor, image, depthmap, object_mask, intrinsics, resolution,
        info=str(scene_dir / "rgb" / f"{image_id:06d}.png"),
    )
    image_tensor = processor.transform(image).unsqueeze(0).unsqueeze(0).to(device)
    depth_tensor = torch.from_numpy(np.ascontiguousarray(depthmap.astype(np.float32)))[None, None, :, :, None].to(device)
    mask_tensor = torch.from_numpy((depthmap > 0).astype(np.float32))[None, None, :, :].to(device)
    return image_tensor, depth_tensor, mask_tensor, np.asarray(image.convert("RGB")), depthmap, gt_mask, intrinsics


# ============================================================
# 2D visualization helpers (axes / bbox / mask overlays)
# ============================================================
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


def bbox_iou_3d(
    pred_size: np.ndarray | None,
    pred_rotation: np.ndarray,
    pred_translation: np.ndarray,
    gt_size: np.ndarray | None,
    gt_rotation: np.ndarray,
    gt_translation: np.ndarray,
) -> float | None:
    details = bbox_iou_3d_details(
        pred_size,
        pred_rotation,
        pred_translation,
        gt_size,
        gt_rotation,
        gt_translation,
    )
    return None if details is None else float(details["iou"])


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


# ============================================================
# 3D point cloud + pose export (.glb for the Gradio Model3D viewer)
# ============================================================
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


def export_ov9d_point_cloud_pose_glb(
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
    out_dir = _LOCAL_TMP / "ov9d_point_cloud_pose_glb" / safe_scene
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time_ns())
    out_path = out_dir / f"{safe_frame}_stride{stride}_{stamp}.glb"

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
        add_pose(pred_rotation_cam, pred_translation_cam, pred_bbox_obj, PRED_BBOX_COLOR, PRED_AXIS_COLORS)
    if gt_rotation_cam is not None and gt_translation_cam is not None:
        add_pose(gt_rotation_cam, gt_translation_cam, gt_bbox_obj, GT_BBOX_COLOR, GT_AXIS_COLORS)

    scene_3d.export(out_path)
    return str(out_path)


# ============================================================
# DemoApp: OV9D metadata, choices, and inference
# ============================================================
class DemoApp:
    def __init__(
        self,
        config_path: Path,
        checkpoint_path: str | None,
        dataset_root: Path,
        train_split_json: Path | None = None,
        use_gt_pose_for_prediction: bool = False,
    ):
        self.cfg = load_config(config_path)
        self.runtime = resolve_runtime_settings(self.cfg)
        runtime_dataset_root = self.runtime.get("dataset_location")
        self.dataset_root = Path(runtime_dataset_root) if runtime_dataset_root else Path(dataset_root)
        self.object_views = tuple(int(v) for v in self.runtime["object_input_views"])
        self.resolution = tuple(int(v) for v in self.runtime["resolution"])
        self.use_gt_pose_for_prediction = bool(use_gt_pose_for_prediction)

        self.ov9d_multi_root = self.dataset_root / "oo3d9dmulti"
        self.ov9d_single_root = self.dataset_root / "oo3d9dsingle"
        self.ov9d_scene_root = self.dataset_root / DEFAULT_SCENE_SUBDIR
        self.ov9d_split_json = Path(self.runtime["split_json"]) if self.runtime.get("split_json") else None
        self.ov9d_train_split_json = (
            Path(train_split_json)
            if train_split_json is not None
            else Path(self.runtime["train_split_json"]) if self.runtime.get("train_split_json") else None
        )

        self._init_ov9d_metadata()
        self.scene_choices = self.get_scene_choices(self.default_split_name)
        if not self.scene_choices:
            raise RuntimeError(f"No OV9D scenes found under {self.ov9d_scene_root}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = resolve_checkpoint_path(self.cfg, checkpoint_path)
        self.model = None
        self.available_checkpoints = self.discover_checkpoints()
        self.load_checkpoint(self.checkpoint_path)

    # ----- OV9D metadata -----
    def _init_ov9d_metadata(self):
        self.ov9d_models_info = read_json(self.dataset_root / "models_info.json")
        self.ov9d_name_to_oid = {str(k): int(v) for k, v in read_json(self.dataset_root / "name2oid.json").items()}
        self.ov9d_oid_to_name = {oid: name for name, oid in self.ov9d_name_to_oid.items()}

        if not self.ov9d_scene_root.is_dir():
            raise FileNotFoundError(f"OV9D single scene root not found: {self.ov9d_scene_root}")

        self.ov9d_split_records = {"eval": self._load_ov9d_single_scene_records()}
        if self.ov9d_train_split_json is not None and not self.ov9d_train_split_json.is_file():
            print(f"[demo] OV9D train split not found: {self.ov9d_train_split_json}")

        self.default_split_name = "eval"
        self.ov9d_all_scene_records: Dict[str, Dict] = {}
        for records in self.ov9d_split_records.values():
            self.ov9d_all_scene_records.update(records)

        self.ov9d_single_records_by_object_id = self._build_ov9d_single_records_by_object_id()
        self.ov9d_train_object_ids = self._collect_ov9d_train_object_ids()
        self._init_metric_case_browser()

        print(
            f"[demo] OV9D single metadata: eval_scenes={len(self.ov9d_split_records.get('eval', {}))} "
            f"train_scenes={len(self.ov9d_split_records.get('train', {}))} "
            f"train_objects={len(self.ov9d_train_object_ids)} "
            f"single_objects={len(self.ov9d_single_records_by_object_id)} "
            f"scene_root={self.ov9d_scene_root} train_split={self.ov9d_train_split_json}"
        )

    def _load_ov9d_split_records(self, split_json: Path) -> Dict[str, Dict]:
        payload = read_json(split_json)
        return {
            str(item["scene_name"]): item
            for item in payload.get("scenes", [])
            if (self.ov9d_multi_root / str(item["scene_name"])).is_dir()
        }

    def _load_ov9d_single_scene_records(self) -> Dict[str, Dict]:
        records: Dict[str, Dict] = {}
        for scene_dir in sorted(p for p in self.ov9d_scene_root.iterdir() if p.is_dir()):
            if not (scene_dir / "scene_gt.json").is_file() or not (scene_dir / "rgb").is_dir():
                continue
            object_instance = ov9d_single_object_instance_from_scene(scene_dir.name)
            object_id = self.ov9d_name_to_oid.get(object_instance)
            if object_id is None:
                continue
            records[scene_dir.name] = {
                "scene_name": scene_dir.name,
                "object_ids": [int(object_id)],
                "object_instance": object_instance,
            }
        return records

    def _collect_ov9d_train_object_ids(self) -> set:
        if self.ov9d_train_split_json is None or not self.ov9d_train_split_json.is_file():
            return set()
        payload = read_json(self.ov9d_train_split_json)
        object_ids = {int(x) for x in payload.get("anchor_object_ids", [])}
        return {oid for oid in object_ids if oid in self.ov9d_single_records_by_object_id}

    def _build_ov9d_single_records_by_object_id(self) -> Dict[int, List[Dict]]:
        records: Dict[int, List[Dict]] = {}
        if not self.ov9d_single_root.is_dir():
            return records
        for scene_dir in sorted(p for p in self.ov9d_single_root.iterdir() if p.is_dir()):
            object_instance = ov9d_single_object_instance_from_scene(scene_dir.name)
            object_id = self.ov9d_name_to_oid.get(object_instance)
            if object_id is None:
                continue
            image_ids = []
            for rgb_path in sorted((scene_dir / "rgb").glob("*.png")):
                image_id = int(rgb_path.stem)
                if (scene_dir / "mask_visib" / f"{image_id:06d}_000000.png").is_file():
                    image_ids.append(image_id)
            if all(view_id in image_ids for view_id in self.object_views):
                records.setdefault(int(object_id), []).append(
                    {
                        "scene_dir": scene_dir,
                        "scene_name": scene_dir.name,
                        "image_ids": image_ids,
                        "object_instance": object_instance,
                    }
                )
        return records

    # ----- Per-object size / axis-length helpers -----
    def _ov9d_size(self, object_id: int) -> np.ndarray:
        info = self.ov9d_models_info.get(str(int(object_id)), {})
        return np.asarray(
            [info.get("size_x", 100.0), info.get("size_y", 100.0), info.get("size_z", 100.0)],
            dtype=np.float32,
        ) / 1000.0

    def _ov9d_axis_length(self, object_id: int) -> float:
        return max(float(np.linalg.norm(self._ov9d_size(object_id))) * 0.25, 1e-3)

    # ----- Metric case browser -----
    def _init_metric_case_browser(self):
        self.metric_case_root = DEFAULT_MATCHING_CASES_ROOT
        self.metric_case_index: Dict[str, Dict] = {}
        self.metric_case_cache: Dict[Tuple[str, str], List[Dict]] = {}
        if not self.metric_case_root.is_dir():
            return
        if (self.metric_case_root / "case_browser_index.json").is_file() or (self.metric_case_root / "summary.json").is_file():
            split_dirs = [self.metric_case_root]
            self.metric_case_root = self.metric_case_root.parent
        else:
            split_dirs = sorted(p for p in self.metric_case_root.iterdir() if p.is_dir())
        for split_dir in split_dirs:
            index_path = split_dir / "case_browser_index.json"
            if not index_path.is_file():
                summary_path = split_dir / "summary.json"
                if not summary_path.is_file():
                    continue
                summary = read_json(summary_path)
                metrics = {}
                for metric_name, matched_count in summary.items():
                    legacy_file = f"{safe_metric_filename(metric_name)}.jsonl"
                    if not (split_dir / legacy_file).is_file():
                        continue
                    metrics[str(metric_name)] = {
                        "matched": int(matched_count),
                        "missed": 0,
                        "total": int(matched_count),
                        "case_file": legacy_file,
                        "case_kind": "relative" if str(metric_name).startswith("Rel ") else "absolute",
                    }
                if metrics:
                    self.metric_case_index[str(split_dir.name)] = metrics
                continue
            payload = read_json(index_path)
            self.metric_case_index[str(split_dir.name)] = payload.get("metrics", {})

    @property
    def metric_case_split_choices(self) -> List[str]:
        return sorted(self.metric_case_index)

    def get_metric_case_metric_choices(self, split_name: str) -> List[Tuple[str, str]]:
        metrics = self.metric_case_index.get(split_name, {})
        choices = []
        for metric in sorted(metrics, key=lambda item: (item.startswith("Rel "), item)):
            meta = metrics[metric]
            label = f"{metric} | matched {int(meta.get('matched', 0))} | missed {int(meta.get('missed', 0))}"
            choices.append((label, metric))
        return choices

    def get_metric_case_status_choices(self, split_name: str, metric_name: str) -> List[Tuple[str, str]]:
        meta = self.metric_case_index.get(split_name, {}).get(metric_name, {})
        choices = []
        matched_count = int(meta.get("matched", 0))
        missed_count = int(meta.get("missed", 0))
        if matched_count > 0:
            choices.append((f"達成 ({matched_count})", "matched"))
        if missed_count > 0:
            choices.append((f"沒達成 ({missed_count})", "missed"))
        if not choices:
            choices.append(("沒有案例", "matched"))
        return choices

    def _load_metric_cases(self, split_name: str, metric_name: str) -> List[Dict]:
        cache_key = (str(split_name), str(metric_name))
        if cache_key in self.metric_case_cache:
            return self.metric_case_cache[cache_key]
        meta = self.metric_case_index.get(split_name, {}).get(metric_name, {})
        case_file = meta.get("case_file")
        if not case_file:
            self.metric_case_cache[cache_key] = []
            return []
        path = self.metric_case_root / split_name / str(case_file)
        cases = []
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        case = json.loads(line)
                        case.setdefault("metric", metric_name)
                        case.setdefault("case_kind", "relative" if str(metric_name).startswith("Rel ") else "absolute")
                        case.setdefault("matched", True)
                        case.setdefault("status", "matched" if bool(case.get("matched")) else "missed")
                        if "case_id" not in case:
                            if str(case.get("case_kind")) == "relative":
                                case["case_id"] = f"rel:{case.get('left_sample_index', -1)}:{case.get('right_sample_index', -1)}"
                            else:
                                case["case_id"] = f"abs:{case.get('sample_index', -1)}"
                        cases.append(case)
        self.metric_case_cache[cache_key] = cases
        return cases

    def get_metric_case_entries(self, split_name: str, metric_name: str, status: str) -> List[Dict]:
        want_matched = str(status) == "matched"
        return [case for case in self._load_metric_cases(split_name, metric_name) if bool(case.get("matched")) == want_matched]

    def _metric_case_label(self, case: Dict) -> str:
        object_name = str(case.get("object_name", ov9d_object_key(case.get("object_id", -1))))
        scene_name = str(case.get("scene_name", ""))
        if str(case.get("case_kind")) == "relative":
            return (
                f"{scene_name} | {object_name} | "
                f"{case.get('left_frame_name', '------')} -> {case.get('right_frame_name', '------')} | "
                f"r={format_optional_float(case.get('relative_rotation_error_deg'), 2)} deg | "
                f"t={format_optional_float(case.get('relative_translation_error_cm'), 2)} cm"
            )
        return (
            f"{scene_name} | {object_name} | frame {case.get('frame_name', '------')} | "
            f"IoU={format_optional_float(case.get('iou_3d'), 3)} | "
            f"r={format_optional_float(case.get('rotation_error_deg'), 2)} deg | "
            f"t={format_optional_float(case.get('translation_error_cm'), 2)} cm"
        )

    def get_metric_case_dropdown_choices(self, split_name: str, metric_name: str, status: str) -> List[Tuple[str, str]]:
        return [(self._metric_case_label(case), str(case.get("case_id"))) for case in self.get_metric_case_entries(split_name, metric_name, status)]

    def find_metric_case(self, split_name: str, metric_name: str, status: str, case_id: str | None) -> Dict | None:
        for case in self.get_metric_case_entries(split_name, metric_name, status):
            if str(case.get("case_id")) == str(case_id):
                return case
        return None

    def metric_case_selection_to_target(
        self,
        split_name: str,
        metric_name: str,
        status: str,
        case_id: str | None,
    ) -> Tuple[str | None, str | None, str | None, str | None, str]:
        case = self.find_metric_case(split_name, metric_name, status, case_id)
        if case is None:
            return split_name, None, None, None, "### Metric Case Browser\n- 沒有可用案例。"
        scene_name = str(case.get("scene_name", ""))
        object_key = ov9d_object_key(int(case.get("object_id", -1)))
        if str(case.get("case_kind")) == "relative":
            frame_name = str(case.get("left_frame_name", ""))
            details = [
                "### Metric Case Browser",
                f"- metric: `{metric_name}`",
                f"- status: `{status}`",
                f"- scene: `{scene_name}`",
                f"- object: `{object_key}`",
                f"- frames: `{case.get('left_frame_name')}` -> `{case.get('right_frame_name')}`",
                f"- relative rotation error (deg): `{format_optional_float(case.get('relative_rotation_error_deg'), 6)}`",
                f"- relative translation error (cm): `{format_optional_float(case.get('relative_translation_error_cm'), 6)}`",
                "- note: relative metrics 會自動跳到 left frame。",
            ]
        else:
            frame_name = str(case.get("frame_name", ""))
            details = [
                "### Metric Case Browser",
                f"- metric: `{metric_name}`",
                f"- status: `{status}`",
                f"- scene: `{scene_name}`",
                f"- object: `{object_key}`",
                f"- frame: `{frame_name}`",
                f"- 3D IoU: `{format_optional_float(case.get('iou_3d'), 6)}`",
                f"- rotation error (deg): `{format_optional_float(case.get('rotation_error_deg'), 6)}`",
                f"- translation error (cm): `{format_optional_float(case.get('translation_error_cm'), 6)}`",
            ]
        return split_name, scene_name, frame_name, object_key, "\n".join(details)

    # ----- Dropdown choices -----
    @property
    def ov9d_split_choices(self) -> List[str]:
        return list(self.ov9d_split_records.keys())

    def get_scene_choices(self, split_name: str | None = None) -> List[str]:
        split = split_name if split_name in self.ov9d_split_records else self.default_split_name
        return sorted(self.ov9d_split_records.get(split, {}))

    def get_frame_choices(self, scene_name: str) -> List[str]:
        scene_dir = self.ov9d_scene_root / scene_name
        if not (scene_dir / "scene_gt.json").is_file():
            return []
        return [f"{int(x):06d}" for x in sorted(int(k) for k in read_json(scene_dir / "scene_gt.json").keys())]

    def _ov9d_scene_object_ids(self, scene_name: str) -> List[int]:
        item = self.ov9d_all_scene_records.get(scene_name, {})
        ids = [int(x) for x in item.get("object_ids", item.get("eligible_object_ids", []))]
        return [oid for oid in ids if oid in self.ov9d_single_records_by_object_id]

    def _ov9d_present_ids_for_frame(self, scene_name: str, frame_name: str) -> List[int]:
        scene_gt = read_json(self.ov9d_scene_root / scene_name / "scene_gt.json")
        return [int(gt.get("obj_id", -1)) for gt in scene_gt[str(int(frame_name))]]

    def _ov9d_absent_object_ids(self, scene_name: str, frame_name: str, count: int = 6) -> List[int]:
        present = set(self._ov9d_present_ids_for_frame(scene_name, frame_name))
        scene_ids = set(self._ov9d_scene_object_ids(scene_name))
        candidates = sorted(
            oid
            for oid in self.ov9d_single_records_by_object_id
            if oid not in present and oid not in scene_ids
        )
        rng = np.random.default_rng(abs(hash((scene_name, str(frame_name)))) % (2**32))
        if len(candidates) > count:
            candidates = [int(x) for x in rng.choice(np.asarray(candidates, dtype=np.int64), size=count, replace=False)]
            candidates.sort()
        return candidates

    def ov9d_object_choices_for_frame(self, scene_name: str, frame_name: str):
        if not scene_name or not frame_name:
            return []
        present = set(self._ov9d_present_ids_for_frame(scene_name, frame_name))
        scene_ids = self._ov9d_scene_object_ids(scene_name)

        def make_label(oid: int, tag: str) -> str:
            anchor_marker = "🔴 " if oid in self.ov9d_train_object_ids else ""
            return f"{anchor_marker}[{tag}] {ov9d_object_display_name(oid, self.ov9d_oid_to_name)}"

        choices = []
        for oid in scene_ids:
            tag = "present" if oid in present else "scene object, not this frame"
            choices.append((make_label(oid, tag), ov9d_object_key(oid)))
        return choices

    # ----- Checkpoint discovery / loading -----
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

    # ----- Input gallery preview (shown before clicking Run) -----
    def input_gallery(self, scene_name: str, frame_name: str, object_name: str):
        image_id = int(frame_name)
        object_id = ov9d_object_id_from_key(object_name)
        scene_image = str(self.ov9d_scene_root / scene_name / "rgb" / f"{image_id:06d}.png")
        _, object_gallery = load_ov9d_object_tensor(
            self.ov9d_single_records_by_object_id, object_id,
            self.object_views, self.resolution, torch.device("cpu"),
        )
        return scene_image, object_gallery

    # ----- Inference + visualization -----
    def run_inference(
        self,
        scene_name: str,
        frame_name: str,
        object_name: str,
        use_depth_input: bool,
        show_point_cloud_pose: bool,
        point_cloud_stride: int,
    ):
        object_id = ov9d_object_id_from_key(object_name)
        image_id = int(frame_name)
        use_depth_input = bool(use_depth_input)
        scene_dir = self.ov9d_scene_root / scene_name

        # ---- GT lookup for this (scene, frame, object) ----
        gts = read_json(scene_dir / "scene_gt.json")[str(image_id)]
        object_index = next(
            (idx for idx, gt in enumerate(gts) if int(gt.get("obj_id", -1)) == int(object_id)),
            None,
        )
        has_object = object_index is not None

        # ---- Inputs ----
        (scene_tensor, depth_tensor, mask_tensor,
         display_image, display_depth, gt_mask, intrinsic) = load_ov9d_scene_frame_inputs(
            self.ov9d_scene_root, scene_name, image_id, object_id, self.resolution, self.device,
        )
        object_tensor, _ = load_ov9d_object_tensor(
            self.ov9d_single_records_by_object_id, object_id,
            self.object_views, self.resolution, self.device,
        )

        # ---- Model forward ----
        with torch.inference_mode():
            with torch.autocast(
                device_type=self.device.type, dtype=torch.bfloat16,
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

        # ---- Presence + auxiliary outputs (mask, size) ----
        presence_logit = float(outputs["object_presence_logits"].reshape(-1)[0].detach().float().cpu())
        presence_prob = float(torch.sigmoid(torch.tensor(presence_logit)).item())
        pred_present = presence_prob >= 0.5

        pred_mask = None
        pred_mask_image = None
        if "object_mask" in outputs:
            pred_mask = outputs["object_mask"][0, 0].detach().float().cpu().numpy()
            pred_mask_image = mask_overlay_image(display_image, pred_mask, color=(255, 64, 64))
        gt_mask_image = (
            mask_overlay_image(display_image, gt_mask, color=(80, 255, 120))
            if gt_mask is not None else None
        )
        pred_size = None
        if "object_size" in outputs:
            pred_size = outputs["object_size"].reshape(-1, 3)[0].detach().float().cpu().numpy().astype(np.float32)
        elif "object_size_log" in outputs:
            pred_size = np.exp(
                outputs["object_size_log"].reshape(-1, 3)[0].detach().float().cpu().numpy()
            ).astype(np.float32)

        pose_lines = [
            "### Prediction Summary",
            f"- checkpoint: `{self.checkpoint_path}`",
            f"- scene: `{scene_name}`",
            f"- frame: `{image_id:06d}`",
            f"- object: `{ov9d_object_display_name(object_id, self.ov9d_oid_to_name)}`",
            f"- GT presence in this frame: `{bool(has_object)}`",
            f"- predicted presence probability: `{presence_prob:.6f}` (logit `{presence_logit:.6f}`)",
            f"- predicted present @0.5: `{bool(pred_present)}`",
            f"- use depth input: `{use_depth_input}`",
        ]

        # ---- Absent object: presence-only output ----
        if not has_object:
            point_cloud_glb = None
            if show_point_cloud_pose:
                point_cloud_glb = export_ov9d_point_cloud_pose_glb(
                    scene_name, f"{image_id:06d}", display_image, display_depth, intrinsic,
                    axis_length=self._ov9d_axis_length(object_id),
                    point_cloud_stride=point_cloud_stride,
                )
            pose_lines += [
                "",
                "### Presence-Only Result",
                "- This object is not annotated in the selected frame, so rotation, translation, and mask metrics are skipped.",
            ]
            return "\n".join(pose_lines), None, None, point_cloud_glb, pred_mask_image, gt_mask_image

        # ---- Present object: decode pose ----
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

        # ---- Errors ----
        rot_error_deg, sym_count = symmetric_rotation_error_degrees(
            pred_rotation_cam, gt_rotation_cam, object_id,
            str(self.cfg.get("object_srt_symmetry_info_path", "")),
            int(self.cfg.get("object_srt_symmetry_continuous_steps", 72)),
        )
        trans_error = translation_error(pred_translation_cam, gt_translation_cam)
        mask_score = mask_iou(pred_mask, gt_mask)

        # ---- Bbox sizes (pred size is clamped vs GT for stable visualization) ----
        axis_length = self._ov9d_axis_length(object_id)
        gt_size = self._ov9d_size(object_id)
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

        # ---- 2D overlays ----
        pred_image = draw_bbox_axes_overlay_on_image(
            display_image, intrinsic, pred_rotation_cam, pred_translation_cam,
            pred_bbox_obj, axis_length, PRED_AXIS_COLORS, PRED_BBOX_COLOR,
        )
        gt_image = draw_bbox_axes_overlay_on_image(
            display_image, intrinsic, gt_rotation_cam, gt_translation_cam,
            gt_bbox_obj, axis_length, GT_AXIS_COLORS, GT_BBOX_COLOR,
        )

        # ---- Optional 3D point cloud export ----
        point_cloud_glb = None
        if show_point_cloud_pose:
            point_cloud_glb = export_ov9d_point_cloud_pose_glb(
                scene_name, f"{image_id:06d}",
                display_image, display_depth, intrinsic,
                pred_rotation_cam=pred_rotation_cam,
                pred_translation_cam=pred_translation_cam,
                pred_bbox_obj=pred_bbox_obj,
                gt_rotation_cam=gt_rotation_cam,
                gt_translation_cam=gt_translation_cam,
                gt_bbox_obj=gt_bbox_obj,
                axis_length=axis_length,
                point_cloud_stride=point_cloud_stride,
            )

        # ---- Markdown summary ----
        size_error = translation_error(raw_pred_size, gt_size)
        pose_lines += [
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
        return "\n".join(pose_lines), pred_image, gt_image, point_cloud_glb, pred_mask_image, gt_mask_image


# ============================================================
# Gradio UI
# ============================================================
def build_demo(app: DemoApp, image_focused_layout: bool = False):
    default_split = app.default_split_name
    default_scene = app.get_scene_choices(default_split)[0]
    default_frames = app.get_frame_choices(default_scene)
    default_frame = default_frames[0] if default_frames else None
    default_object_choices = app.ov9d_object_choices_for_frame(default_scene, default_frame)
    default_object = default_object_choices[0][1] if default_object_choices else None
    default_case_split = app.metric_case_split_choices[0] if app.metric_case_split_choices else None
    default_case_metric_choices = app.get_metric_case_metric_choices(default_case_split) if default_case_split else []
    default_case_metric = default_case_metric_choices[0][1] if default_case_metric_choices else None
    default_case_status_choices = (
        app.get_metric_case_status_choices(default_case_split, default_case_metric)
        if default_case_split and default_case_metric else []
    )
    default_case_status = default_case_status_choices[0][1] if default_case_status_choices else None
    default_case_choices = (
        app.get_metric_case_dropdown_choices(default_case_split, default_case_metric, default_case_status)
        if default_case_split and default_case_metric and default_case_status else []
    )
    default_case_id = default_case_choices[0][1] if default_case_choices else None
    _, _, _, _, default_case_summary = app.metric_case_selection_to_target(
        default_case_split or default_split,
        default_case_metric or "",
        default_case_status or "matched",
        default_case_id,
    )

    demo_css = """
    .gradio-container { max-width: 100% !important; }
    /* Single-image panels: scale to fit, no cropping, no scrollbars. */
    #scene_inputs, #pred_projection, #gt_projection {
        background: #1c1c1c;
    }
    #scene_inputs img, #pred_projection img, #gt_projection img,
    #scene_inputs button img, #pred_projection button img, #gt_projection button img {
        object-fit: contain !important;
        max-height: 100% !important;
        max-width: 100% !important;
        width: auto !important;
        height: auto !important;
    }
    #scene_inputs > div, #pred_projection > div, #gt_projection > div {
        overflow: hidden !important;
    }
    /* Object thumbnails strip: also contain. */
    #object_inputs img { object-fit: contain !important; background: #1c1c1c; }
    """
    if image_focused_layout:
        demo_css += """
        .gradio-container { padding: 8px 10px !important; }
        #demo_info { margin-bottom: 6px !important; }
        #demo_info p { margin: 0 !important; }
        #main_row { gap: 12px !important; }
        #inputs_col, #results_col { min-height: 80vh !important; }
        #scene_inputs, #pred_projection, #gt_projection { min-height: 38vh !important; }
        #object_inputs { min-height: 22vh !important; }
        """

    anchor_note = (
        f"(🔴 marks {len(app.ov9d_train_object_ids)} anchor objects)"
        if app.ov9d_train_object_ids else ""
    )
    info_markdown = "\n".join(
        [
            "# OmniVGGT 6D Pose Demo (OV9D, single-view)",
            f"- config: `{DEFAULT_CONFIG_PATH}`",
            f"- dataset: `{app.dataset_root}`",
            f"- default checkpoint: `{DEFAULT_PRETRAIN_MODEL}`",
            f"- loaded checkpoint: `{app.checkpoint_path}`",
            f"- object views: `{app.object_views}`",
            f"- inference resolution: `{app.resolution}`",
            f"- OV9D scene root: `{app.ov9d_scene_root}`",
            f"- OV9D train split: `{app.ov9d_train_split_json}` {anchor_note}",
            f"- use_gt_pose_for_prediction: `{app.use_gt_pose_for_prediction}`",
            "",
            "選擇 `scene` → `frame` → `object`,只用單一 frame 的 RGB+Depth 做 pose 預測;"
            "會用 `scene_gt.json` 的 camera-frame GT 比較 translation、sym rotation 與 mask IoU。"
            "勾選點雲選項後,會再把原始深度投影成相機座標系點雲,並畫出 Pred / GT pose 軸。"
            + (
                " 物件名稱前方的 🔴 表示該物件是 train split 的 anchor object。"
                if app.ov9d_train_object_ids else ""
            ),
        ]
    )

    with gr.Blocks(title="OmniVGGT 6D Pose Demo (OV9D)", css=demo_css) as demo:
        if image_focused_layout:
            with gr.Accordion("Demo Info", open=False, elem_id="demo_info"):
                gr.Markdown(info_markdown)
        else:
            gr.Markdown(info_markdown, elem_id="demo_info")

        # ---- Checkpoint controls ----
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

        # ---- Scene / frame / object selectors ----
        with gr.Row():
            split_dropdown = gr.Dropdown(
                choices=app.ov9d_split_choices, value=default_split, label="Split",
            )
            scene_dropdown = gr.Dropdown(choices=app.scene_choices, value=default_scene, label="Scene")
            frame_dropdown = gr.Dropdown(choices=default_frames, value=default_frame, label="Frame")
            object_dropdown = gr.Dropdown(choices=default_object_choices, value=default_object, label="Object")
            use_depth_checkbox = gr.Checkbox(value=True, label="Use Depth Input")
            show_point_cloud_checkbox = gr.Checkbox(value=False, label="Show Point Cloud Pose")
            point_cloud_stride_slider = gr.Slider(
                minimum=1, maximum=8, value=2, step=1,
                label="Point Cloud Density", info="Smaller = denser RGB point cloud",
            )
            infer_button = gr.Button("Run Inference", variant="primary")

        with gr.Accordion("Metric Case Browser", open=False):
            with gr.Row():
                case_split_dropdown = gr.Dropdown(
                    choices=app.metric_case_split_choices,
                    value=default_case_split,
                    label="Metric Split",
                )
                case_metric_dropdown = gr.Dropdown(
                    choices=default_case_metric_choices,
                    value=default_case_metric,
                    label="Metric",
                )
                case_status_dropdown = gr.Dropdown(
                    choices=default_case_status_choices,
                    value=default_case_status,
                    label="Case Status",
                )
                case_dropdown = gr.Dropdown(
                    choices=default_case_choices,
                    value=default_case_id,
                    label="Case",
                )
            case_browser_summary = gr.Markdown(default_case_summary)

        # ---- Image panels ----
        big_height = "38vh" if image_focused_layout else 360
        thumb_height = "18vh" if image_focused_layout else 180

        with gr.Row(elem_id="main_row", equal_height=False):
            with gr.Column(scale=1, elem_id="inputs_col"):
                gr.Markdown("#### Inputs")
                scene_input_image = compat_image(
                    label="Scene Input (RGB)", height=big_height, elem_id="scene_inputs",
                    interactive=False,
                )
                object_input_gallery = gr.Gallery(
                    label="Object Inputs", columns=max(len(app.object_views), 1), height=thumb_height,
                    elem_id="object_inputs", preview=False, object_fit="contain",
                    show_label=True, allow_preview=True,
                )
            with gr.Column(scale=1, elem_id="results_col"):
                gr.Markdown("#### Results (axes overlay)")
                pred_image = compat_image(
                    label="Predicted axes (camera frame)", height=big_height, elem_id="pred_projection",
                    interactive=False, show_download_button=False, container=True,
                )
                gt_image = compat_image(
                    label="GT axes (camera frame)", height=big_height, elem_id="gt_projection",
                    interactive=False, show_download_button=False, container=True,
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
                        label="Predicted object mask", height=thumb_height, interactive=False,
                        show_download_button=False, container=True,
                    )
                    gt_mask_image = compat_image(
                        label="GT object mask", height=thumb_height, interactive=False,
                        show_download_button=False, container=True,
                    )

        summary_markdown = gr.Markdown()

        # ---- Event handlers ----
        def refresh_inputs(scene_name, frame_name, object_name):
            if not scene_name or not frame_name or not object_name:
                return None, []
            return app.input_gallery(scene_name, frame_name, object_name)

        def refresh_split_controls(
            split_name: str,
            preferred_scene: str | None = None,
            preferred_frame: str | None = None,
            preferred_object: str | None = None,
        ):
            scenes = app.get_scene_choices(split_name)
            scene_value = preferred_scene if preferred_scene in scenes else (scenes[0] if scenes else None)
            frames = app.get_frame_choices(scene_value) if scene_value else []
            frame_value = preferred_frame if preferred_frame in frames else (frames[0] if frames else None)
            object_choices = (
                app.ov9d_object_choices_for_frame(scene_value, frame_value)
                if scene_value and frame_value else []
            )
            object_values = [value for _, value in object_choices]
            object_value = preferred_object if preferred_object in object_values else (object_values[0] if object_values else None)
            scene_image, object_gallery = refresh_inputs(scene_value, frame_value, object_value)
            return (
                gr.update(choices=scenes, value=scene_value),
                gr.update(choices=frames, value=frame_value),
                gr.update(choices=object_choices, value=object_value),
                scene_image,
                object_gallery,
            )

        def refresh_scene_controls(scene_name: str, preferred_frame: str | None = None, preferred_object: str | None = None):
            frames = app.get_frame_choices(scene_name)
            frame_value = preferred_frame if preferred_frame in frames else (frames[0] if frames else None)
            object_choices = (
                app.ov9d_object_choices_for_frame(scene_name, frame_value)
                if frame_value else []
            )
            object_values = [value for _, value in object_choices]
            object_value = preferred_object if preferred_object in object_values else (object_values[0] if object_values else None)
            return (
                gr.update(choices=frames, value=frame_value),
                gr.update(choices=object_choices, value=object_value),
            )

        def refresh_frame_controls(scene_name: str, frame_name: str, preferred_object: str | None = None):
            if not (scene_name and frame_name):
                return gr.update()
            object_choices = app.ov9d_object_choices_for_frame(scene_name, frame_name)
            object_values = [value for _, value in object_choices]
            object_value = preferred_object if preferred_object in object_values else (object_values[0] if object_values else None)
            return gr.update(choices=object_choices, value=object_value)

        def sync_checkpoint_path(selected_value: str):
            return selected_value

        def _main_controls_for_target(split_name: str | None, scene_name: str | None, frame_name: str | None, object_name: str | None):
            split_value = split_name if split_name in app.ov9d_split_choices else app.default_split_name
            scene_choices = app.get_scene_choices(split_value)
            if scene_name not in scene_choices:
                scene_name = scene_choices[0] if scene_choices else None
            frame_choices = app.get_frame_choices(scene_name) if scene_name else []
            if frame_name not in frame_choices:
                frame_name = frame_choices[0] if frame_choices else None
            object_choices = (
                app.ov9d_object_choices_for_frame(scene_name, frame_name)
                if scene_name and frame_name else []
            )
            object_values = [value for _, value in object_choices]
            if object_name not in object_values:
                object_name = object_values[0] if object_values else None
            scene_image, object_gallery = refresh_inputs(scene_name, frame_name, object_name)
            return (
                gr.update(choices=app.ov9d_split_choices, value=split_value),
                gr.update(choices=scene_choices, value=scene_name),
                gr.update(choices=frame_choices, value=frame_name),
                gr.update(choices=object_choices, value=object_name),
                scene_image,
                object_gallery,
            )

        def refresh_case_split_controls(case_split_name: str):
            metric_choices = app.get_metric_case_metric_choices(case_split_name)
            metric_value = metric_choices[0][1] if metric_choices else None
            status_choices = app.get_metric_case_status_choices(case_split_name, metric_value) if metric_value else []
            status_value = status_choices[0][1] if status_choices else None
            case_choices = app.get_metric_case_dropdown_choices(case_split_name, metric_value, status_value) if status_value else []
            case_value = case_choices[0][1] if case_choices else None
            target_split, target_scene, target_frame, target_object, summary = app.metric_case_selection_to_target(
                case_split_name, metric_value or "", status_value or "matched", case_value,
            )
            main_updates = _main_controls_for_target(target_split, target_scene, target_frame, target_object)
            return (
                gr.update(choices=metric_choices, value=metric_value),
                gr.update(choices=status_choices, value=status_value),
                gr.update(choices=case_choices, value=case_value),
                summary,
                *main_updates,
            )

        def refresh_case_metric_controls(case_split_name: str, metric_name: str):
            status_choices = app.get_metric_case_status_choices(case_split_name, metric_name)
            status_value = status_choices[0][1] if status_choices else None
            case_choices = app.get_metric_case_dropdown_choices(case_split_name, metric_name, status_value) if status_value else []
            case_value = case_choices[0][1] if case_choices else None
            target_split, target_scene, target_frame, target_object, summary = app.metric_case_selection_to_target(
                case_split_name, metric_name or "", status_value or "matched", case_value,
            )
            main_updates = _main_controls_for_target(target_split, target_scene, target_frame, target_object)
            return (
                gr.update(choices=status_choices, value=status_value),
                gr.update(choices=case_choices, value=case_value),
                summary,
                *main_updates,
            )

        def refresh_case_status_controls(case_split_name: str, metric_name: str, status: str):
            case_choices = app.get_metric_case_dropdown_choices(case_split_name, metric_name, status)
            case_value = case_choices[0][1] if case_choices else None
            target_split, target_scene, target_frame, target_object, summary = app.metric_case_selection_to_target(
                case_split_name, metric_name or "", status or "matched", case_value,
            )
            main_updates = _main_controls_for_target(target_split, target_scene, target_frame, target_object)
            return (
                gr.update(choices=case_choices, value=case_value),
                summary,
                *main_updates,
            )

        def apply_selected_case(case_split_name: str, metric_name: str, status: str, case_id: str):
            target_split, target_scene, target_frame, target_object, summary = app.metric_case_selection_to_target(
                case_split_name, metric_name or "", status or "matched", case_id,
            )
            main_updates = _main_controls_for_target(target_split, target_scene, target_frame, target_object)
            return (summary, *main_updates)

        split_dropdown.change(
            refresh_split_controls,
            inputs=[split_dropdown, scene_dropdown, frame_dropdown, object_dropdown],
            outputs=[scene_dropdown, frame_dropdown, object_dropdown, scene_input_image, object_input_gallery],
        )
        scene_dropdown.change(
            refresh_scene_controls, inputs=[scene_dropdown, frame_dropdown, object_dropdown], outputs=[frame_dropdown, object_dropdown],
        )
        scene_dropdown.change(
            refresh_inputs,
            inputs=[scene_dropdown, frame_dropdown, object_dropdown],
            outputs=[scene_input_image, object_input_gallery],
        )
        frame_dropdown.change(
            refresh_frame_controls,
            inputs=[scene_dropdown, frame_dropdown, object_dropdown],
            outputs=object_dropdown,
        )
        frame_dropdown.change(
            refresh_inputs,
            inputs=[scene_dropdown, frame_dropdown, object_dropdown],
            outputs=[scene_input_image, object_input_gallery],
        )
        object_dropdown.change(
            refresh_inputs,
            inputs=[scene_dropdown, frame_dropdown, object_dropdown],
            outputs=[scene_input_image, object_input_gallery],
        )
        checkpoint_dropdown.change(sync_checkpoint_path, inputs=checkpoint_dropdown, outputs=checkpoint_textbox)
        load_model_button.click(
            app.load_checkpoint,
            inputs=checkpoint_textbox,
            outputs=[checkpoint_status, checkpoint_dropdown, checkpoint_textbox],
        )
        case_split_dropdown.change(
            refresh_case_split_controls,
            inputs=case_split_dropdown,
            outputs=[
                case_metric_dropdown,
                case_status_dropdown,
                case_dropdown,
                case_browser_summary,
                split_dropdown,
                scene_dropdown,
                frame_dropdown,
                object_dropdown,
                scene_input_image,
                object_input_gallery,
            ],
        )
        case_metric_dropdown.change(
            refresh_case_metric_controls,
            inputs=[case_split_dropdown, case_metric_dropdown],
            outputs=[
                case_status_dropdown,
                case_dropdown,
                case_browser_summary,
                split_dropdown,
                scene_dropdown,
                frame_dropdown,
                object_dropdown,
                scene_input_image,
                object_input_gallery,
            ],
        )
        case_status_dropdown.change(
            refresh_case_status_controls,
            inputs=[case_split_dropdown, case_metric_dropdown, case_status_dropdown],
            outputs=[
                case_dropdown,
                case_browser_summary,
                split_dropdown,
                scene_dropdown,
                frame_dropdown,
                object_dropdown,
                scene_input_image,
                object_input_gallery,
            ],
        )
        case_dropdown.change(
            apply_selected_case,
            inputs=[case_split_dropdown, case_metric_dropdown, case_status_dropdown, case_dropdown],
            outputs=[
                case_browser_summary,
                split_dropdown,
                scene_dropdown,
                frame_dropdown,
                object_dropdown,
                scene_input_image,
                object_input_gallery,
            ],
        )
        infer_button.click(
            app.run_inference,
            inputs=[
                scene_dropdown,
                frame_dropdown,
                object_dropdown,
                use_depth_checkbox,
                show_point_cloud_checkbox,
                point_cloud_stride_slider,
            ],
            outputs=[summary_markdown, pred_image, gt_image, point_cloud_model, pred_mask_image, gt_mask_image],
        )

        if default_object is not None and default_frame is not None:
            demo.load(
                refresh_inputs,
                inputs=[scene_dropdown, frame_dropdown, object_dropdown],
                outputs=[scene_input_image, object_input_gallery],
            )
    return demo


# ============================================================
# Entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Gradio demo for OmniVGGT 6D pose inference on OV9D")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--train-split-json",
        type=Path,
        default=None,
        help="Optional OV9D train split JSON override. Defaults to the train_dataset split_json from the config.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--image-focused-layout", action="store_true",
        help="Use a comparison-oriented layout that gives most of the page to images.",
    )
    parser.add_argument(
        "--use-gt-pose-for-prediction", action="store_true",
        help="Replace the predicted pose with ground-truth pose before drawing the prediction overlay.",
    )
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    app = DemoApp(
        args.config,
        args.checkpoint,
        dataset_root=args.dataset_root,
        train_split_json=args.train_split_json,
        use_gt_pose_for_prediction=args.use_gt_pose_for_prediction,
    )
    demo = build_demo(app, image_focused_layout=args.image_focused_layout)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[str(app.dataset_root)],
    )


if __name__ == "__main__":
    main()
