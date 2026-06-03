import os
from pathlib import Path

# Must be set before `import gradio` / `import cv2` so Gradio cache files land in a
# writable location inside the repo and OpenCV can read Omni6DPose depth EXRs.
PROJECT_ROOT = Path(__file__).resolve().parent
_LOCAL_TMP = PROJECT_ROOT / "tmp"
_LOCAL_TMP.mkdir(parents=True, exist_ok=True)
(_LOCAL_TMP / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", str(_LOCAL_TMP))
os.environ.setdefault("GRADIO_CACHE_DIR", str(_LOCAL_TMP))
os.environ.setdefault("TMPDIR", str(_LOCAL_TMP))
os.environ.setdefault("MPLCONFIGDIR", str(_LOCAL_TMP / "matplotlib"))
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import argparse
import inspect
import json
import re
import runpy
import time
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import gradio as gr
import numpy as np
import torch
import trimesh
from PIL import Image
from safetensors.torch import load_file as load_safetensors_file

from omnivggt.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from omnivggt.datasets.omni6dpose.omni6dpose_camera_pose import quaternion_wxyz_to_matrix
from omnivggt.loss import _load_symmetry_info
from omnivggt.models.omnivggt import OmniVGGT
from omnivggt.utils.image import ImgNorm


# --------------------------------------------------------------------------- paths
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train_omnipose.py"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "0531_REFER" / "lr_1e5_1000" / "model.safetensors"

DEFAULT_OMNI6DPOSE_ROOT = Path(
    "/mnt/train-data-4-hdd/yian/freepose/Omni6dpose/Omni6DPoseAPI/data/Omni6DPose"
)
DEFAULT_ROPE_ROOT = DEFAULT_OMNI6DPOSE_ROOT / "ROPE"
DEFAULT_SOPE_ROOT = DEFAULT_OMNI6DPOSE_ROOT / "SOPE"
DEFAULT_OBJECT_IMAGE_ROOT = DEFAULT_OMNI6DPOSE_ROOT / "omni6dpose_ref" / "diverse24"
DEFAULT_ROPE_OID_TO_PAM = (
    PROJECT_ROOT / "outputs" / "omni6dpose_refs" / "rope_oid_to_pam.json"
)

DATASET_LABEL = "Omni6DPoseCameraPose"

# Data-source keys exposed in the UI. ROPE is the real eval set (flat layout, no
# train/test split); SOPE is the synthetic set with nested train/test splits.
SOURCE_ROPE = "ROPE"
SOURCE_SOPE_TRAIN = "SOPE/train"
SOURCE_SOPE_TEST = "SOPE/test"
SOURCE_SOPE_TEST_NOVEL_SRC = "SOPE/test (novel source-class)"
SOURCE_SOPE_TEST_NOVEL_CLS = "SOPE/test (novel class)"
DEFAULT_TRAIN_CATEGORIES = (
    PROJECT_ROOT / "outputs" / "omni6dpose_refs" / "sope_train_categories.json"
)

AXIS_COLORS = ((255, 64, 64), (0, 255, 255), (255, 215, 0))
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
            "fixed_object_view_ids",
            default=cfg.get("fixed_object_view_ids", (0, 5, 8, 19)),
        )
    )
    resolution = tuple(int(v) for v in cfg.get("resolution", (518, 476)))
    z_far = parse_dataset_ctor_arg(dataset_expr, "z_far", default=cfg.get("z_far", 20))
    return {
        "object_input_views": tuple(sorted(int(v) for v in object_input_views)),
        "resolution": resolution,
        "z_far": float(z_far) if z_far else 0.0,
    }


def resolve_checkpoint_path(checkpoint_path: str | None) -> Path:
    if checkpoint_path:
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    if DEFAULT_CHECKPOINT.is_file():
        return DEFAULT_CHECKPOINT
    raise FileNotFoundError("Unable to locate a local checkpoint from defaults.")


def build_model_from_config(cfg: Dict, checkpoint_path: Path, device: torch.device) -> OmniVGGT:
    model = OmniVGGT(
        enable_camera=cfg.get("enable_camera", False),
        enable_point=cfg.get("enable_point", False),
        enable_depth=cfg.get("enable_depth", False),
        enable_object_mask=cfg.get("enable_object_mask", True),
        enable_object_srt=cfg.get("enable_object_srt", True),
        enable_object_size=cfg.get("enable_object_size", True),
        always_use_depth_gt=cfg.get("always_use_depth_gt", True),
        patch_embed_pretrained_path=cfg.get("patch_embed_pretrained_path", None),
        load_patch_embed_from_hub=cfg.get("load_patch_embed_from_hub", False),
        cam_drop_prob=cfg.get("cam_drop_prob", 1.0),
        depth_drop_prob=cfg.get("depth_drop_prob", 0.0),
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
        enable_multi_layer_object_prototype_cross_attn=cfg.get("enable_multi_layer_object_prototype_cross_attn", True),
        object_prototype_layer_indices=cfg.get("object_prototype_layer_indices", (4, 11, 17, 23)),
        object_prototype_num_tokens=cfg.get("object_prototype_num_tokens", 32),
        object_prototype_object_encoder_no_grad=cfg.get("object_prototype_object_encoder_no_grad", True),
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


# ----------------------------------------------------------------- pose / metrics
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


def translation_error(pred_t: np.ndarray, gt_t: np.ndarray) -> Dict[str, Any]:
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
    dataset_label: str = DATASET_LABEL,
) -> Tuple[float, int]:
    if symmetry_info_path is None or not Path(symmetry_info_path).is_file():
        return rotation_error_degrees(pred_rot, gt_rot), 1
    symmetry_info = _load_symmetry_info(str(symmetry_info_path), int(continuous_steps))
    sym_rots = symmetry_info.get(f"{dataset_label}:{int(object_id)}")
    if sym_rots is None:
        sym_rots = symmetry_info.get(int(object_id))
    if sym_rots is None:
        return rotation_error_degrees(pred_rot, gt_rot), 1
    candidates = sym_rots.detach().cpu().numpy().astype(np.float64)
    errors = [rotation_error_degrees(pred_rot, np.asarray(gt_rot, dtype=np.float64) @ sym) for sym in candidates]
    return float(min(errors)), len(errors)


def read_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


# ----------------------------------------------------------------- Omni6DPose io
def omni6dpose_read_depth_m(depth_path: Path) -> np.ndarray:
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Failed to read depth EXR: {depth_path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth.astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0.0] = 0.0
    return depth


def omni6dpose_scaled_intrinsic(intr: Dict[str, Any], width: int, height: int) -> np.ndarray:
    scale_x = width / float(intr["width"])
    scale_y = height / float(intr["height"])
    return np.array(
        [
            [float(intr["fx"]) * scale_x, 0.0, float(intr["cx"]) * scale_x],
            [0.0, float(intr["fy"]) * scale_y, float(intr["cy"]) * scale_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def omni6dpose_read_object_mask(mask_path: Path, mask_id: int) -> np.ndarray | None:
    """Decode the per-instance mask EXR for `mask_id`.

    Follows the official Omni6DPose `Dataset.load_mask`: read EXR, take channel 2
    (BGR blue), scale by 255 to recover the integer instance id per pixel.
    """
    if not Path(mask_path).is_file():
        return None
    img = cv2.imread(str(mask_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        return None
    channel = img[:, :, 2] if img.ndim == 3 else img
    ids = np.asarray(np.asarray(channel, dtype=np.float64) * 255.0, dtype=np.uint8)
    return (ids == int(mask_id)).astype(np.float32)


class DemoScenePreprocessor(BaseStereoViewDataset):
    """Throwaway dataset used only for its `_crop_resize_if_necessary` + transform."""

    def __init__(self, resolution, z_far: float = 0.0):
        super().__init__(dset="omni6dpose_demo", resolution=resolution, transform=ImgNorm, seed=0, z_far=z_far)


def _target_crop_image_depth_mask(
    image: Image.Image,
    depthmap: np.ndarray,
    object_mask: np.ndarray,
    intrinsics: np.ndarray,
    resolution,
    margin: float = 0.45,
):
    """Crop tightly around the object mask, then resize to `resolution`."""
    width, height = image.size
    out_w, out_h = int(resolution[0]), int(resolution[1])
    ys, xs = np.where(object_mask > 0)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    box_w = max(float(x1 - x0), 1.0)
    box_h = max(float(y1 - y0), 1.0)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)

    crop_w = max(box_w * (1.0 + 2.0 * float(margin)), 32.0)
    crop_h = max(box_h * (1.0 + 2.0 * float(margin)), 32.0)
    target_aspect = out_w / max(float(out_h), 1.0)
    if crop_w / crop_h < target_aspect:
        crop_w = crop_h * target_aspect
    else:
        crop_h = crop_w / target_aspect

    left = int(np.floor(cx - crop_w * 0.5))
    top = int(np.floor(cy - crop_h * 0.5))
    right = int(np.ceil(cx + crop_w * 0.5))
    bottom = int(np.ceil(cy + crop_h * 0.5))

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > width:
        left -= right - width
        right = width
    if bottom > height:
        top -= bottom - height
        bottom = height
    left = max(left, 0)
    top = max(top, 0)
    right = min(max(right, left + 1), width)
    bottom = min(max(bottom, top + 1), height)

    crop_w = float(right - left)
    crop_h = float(bottom - top)
    image = image.crop((left, top, right, bottom))
    depthmap = depthmap[top:bottom, left:right]
    object_mask = object_mask[top:bottom, left:right]

    resampling = getattr(Image, "Resampling", Image)
    image = image.resize((out_w, out_h), resampling.LANCZOS)
    depthmap = cv2.resize(depthmap.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    object_mask = cv2.resize(
        (object_mask > 0).astype(np.uint8),
        (out_w, out_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    intrinsics = intrinsics.copy().astype(np.float32)
    intrinsics[0, 2] -= float(left)
    intrinsics[1, 2] -= float(top)
    sx = out_w / max(crop_w, 1.0)
    sy = out_h / max(crop_h, 1.0)
    intrinsics[0, :] *= np.float32(sx)
    intrinsics[1, :] *= np.float32(sy)
    return image, depthmap.astype(np.float32), object_mask, intrinsics


def load_omni6dpose_scene_frame_inputs(
    processor: DemoScenePreprocessor,
    color_path: Path,
    depth_path: Path,
    intr: Dict[str, Any],
    object_mask: np.ndarray | None,
    resolution,
    device,
    target_crop: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    image = Image.open(color_path).convert("RGB")
    depthmap = omni6dpose_read_depth_m(depth_path)
    width, height = image.size
    if depthmap.shape[:2] != (height, width):
        depthmap = cv2.resize(depthmap, (width, height), interpolation=cv2.INTER_NEAREST)
    intrinsics = omni6dpose_scaled_intrinsic(intr, width, height)
    if object_mask is not None and object_mask.shape[:2] != (height, width):
        object_mask = cv2.resize(
            (object_mask > 0).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
        ).astype(np.float32)

    if target_crop and object_mask is not None and np.any(object_mask > 0):
        image, depthmap, gt_mask, intrinsics = _target_crop_image_depth_mask(
            image, depthmap, object_mask, intrinsics, resolution
        )
    else:
        # Centered crop/resize identical to training (Omni6DPoseCameraPose._load_scene_view).
        image, depthmap, intrinsics = processor._crop_resize_if_necessary(
            image=image,
            depthmap=depthmap,
            intrinsics=intrinsics.copy(),
            resolution=resolution,
            rng=np.random.default_rng(seed=0),
            info=str(color_path),
        )
        gt_mask = None

    image_tensor = processor.transform(image).unsqueeze(0).unsqueeze(0).to(device)
    depth_tensor = torch.from_numpy(np.ascontiguousarray(depthmap.astype(np.float32)))[None, None, :, :, None].to(device)
    mask_tensor = torch.from_numpy((depthmap > 0).astype(np.float32))[None, None, :, :].to(device)
    return image_tensor, depth_tensor, mask_tensor, np.asarray(image.convert("RGB")), depthmap, gt_mask, intrinsics


def load_omni6dpose_object_tensor(
    object_dir: Path,
    object_views: Sequence[int],
    resolution,
    device,
) -> Tuple[torch.Tensor, List[Tuple[np.ndarray, str]]]:
    processor = DemoScenePreprocessor(resolution=resolution)
    resampling = getattr(Image, "Resampling", Image)
    tensors: List[torch.Tensor] = []
    gallery: List[Tuple[np.ndarray, str]] = []
    for image_id in object_views:
        image_path = object_dir / "rgb" / f"{int(image_id):06d}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing Omni6DPose object reference view: {image_path}")
        image = Image.open(image_path).convert("RGB")
        image = image.resize(tuple(resolution), resampling.LANCZOS)
        tensors.append(processor.transform(image).to(device))
        gallery.append((np.asarray(image), f"Object view {int(image_id):06d}"))
    return torch.stack(tensors, dim=0).unsqueeze(0), gallery


# ----------------------------------------------------------------- geometry / viz
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
    out_dir = _LOCAL_TMP / "omni6dpose_point_cloud_pose_glb" / safe_scene
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

    def add_pose(rotation_cam, translation_cam, bbox_obj, bbox_color):
        center = np.asarray(translation_cam, dtype=np.float32)
        axis_pts = _axis_object_points(viewer_axis_length) @ rotation_cam.T + translation_cam[None, :]
        for idx, axis_color in enumerate(AXIS_COLORS):
            add_segment(center, axis_pts[idx + 1], axis_color)
        if bbox_obj is not None:
            bbox_cam = np.asarray(bbox_obj, dtype=np.float32) @ rotation_cam.T + translation_cam[None, :]
            for start_idx, end_idx in BBOX_EDGES:
                add_segment(bbox_cam[start_idx], bbox_cam[end_idx], bbox_color)

    if pred_rotation_cam is not None and pred_translation_cam is not None:
        add_pose(pred_rotation_cam, pred_translation_cam, pred_bbox_obj, PRED_BBOX_COLOR)
    if gt_rotation_cam is not None and gt_translation_cam is not None:
        add_pose(gt_rotation_cam, gt_translation_cam, gt_bbox_obj, GT_BBOX_COLOR)

    scene_3d.export(out_path)
    return str(out_path)


# --------------------------------------------------------------------------- app
class DemoApp:
    def __init__(
        self,
        config_path: Path,
        checkpoint_path: str | None,
        omni6dpose_root: Path,
        rope_root: Path | None = None,
        sope_root: Path | None = None,
        object_image_root: Path | None = None,
        rope_oid_to_pam: Path | None = None,
    ):
        self.config_path = Path(config_path)
        self.cfg = load_config(self.config_path)
        self.runtime = resolve_runtime_settings(self.cfg)
        self.object_views = tuple(int(v) for v in self.runtime["object_input_views"])
        self.resolution = tuple(int(v) for v in self.runtime["resolution"])
        self.z_far = float(self.runtime.get("z_far", 0.0))
        self.symmetry_info_path = self._resolve_symmetry_info_path()
        self.symmetry_continuous_steps = int(self.cfg.get("object_srt_symmetry_continuous_steps", 72))

        self.omni6dpose_root = Path(omni6dpose_root).expanduser()
        self.rope_root = Path(rope_root or (self.omni6dpose_root / "ROPE")).expanduser()
        self.sope_root = Path(sope_root or (self.omni6dpose_root / "SOPE")).expanduser()
        self.object_image_root = Path(
            object_image_root or (self.omni6dpose_root / "omni6dpose_ref" / "diverse24")
        ).expanduser()

        self.rope_oid_to_pam_path = Path(rope_oid_to_pam or DEFAULT_ROPE_OID_TO_PAM).expanduser()

        # diverse24 object reference renders (shared across ROPE/SOPE).
        self.object_records_by_name = self._build_object_records_by_name()
        if not self.object_records_by_name:
            raise RuntimeError(f"No Omni6DPose object refs found under {self.object_image_root}")
        self.object_name_to_id = {
            name: idx + 1 for idx, name in enumerate(sorted(self.object_records_by_name.keys()))
        }

        # ROPE objects have no PAM mesh / rendered reference of their own, so they
        # borrow a *synthetic* diverse24 reference of the same class:
        #   1. exact instance — `rope_oid_to_pam.json` maps a few ROPE oids to the
        #      matching synthetic render (e.g. real-chess_001 -> omniobject3d-chess_001);
        #   2. same-class approx — otherwise reuse the nearest-instance diverse24 ref
        #      whose semantic class matches (most ROPE objects land here).
        # Build class_name -> sorted ref names over ALL diverse24 records.
        self.refs_by_category: Dict[str, List[str]] = {}
        for name in self.object_records_by_name:
            self.refs_by_category.setdefault(self._oid_class_name(name), []).append(name)
        for category in self.refs_by_category:
            self.refs_by_category[category].sort()
        # Kept for reporting: which classes have a real-* (real-world scanned) ref.
        self.real_refs_by_category: Dict[str, List[str]] = {
            cat: refs
            for cat, names in self.refs_by_category.items()
            if (refs := [n for n in names if n.startswith("real-")])
        }

        # Precomputed exact ROPE-oid -> synthetic-ref map (same instance), if present.
        self.rope_oid_to_pam: Dict[str, str] = {}
        if self.rope_oid_to_pam_path.is_file():
            try:
                self.rope_oid_to_pam = dict(read_json(self.rope_oid_to_pam_path).get("oid_to_pam", {}))
            except Exception:
                self.rope_oid_to_pam = {}

        # SOPE train categories at two granularities, used to flag SOPE/test objects
        # whose category never appears in train:
        #   source_class — e.g. "omniobject3d-cup" (mesh source + class)
        #   class_name   — e.g. "cup" (semantic class, source-agnostic)
        self.train_source_classes: set | None = None
        self.train_class_names: set | None = None
        if DEFAULT_TRAIN_CATEGORIES.is_file():
            _cats = read_json(DEFAULT_TRAIN_CATEGORIES)
            self.train_source_classes = set(_cats.get("source_class", []))
            self.train_class_names = set(_cats.get("class_name", []))

        # Per-source config. Only sources whose root exists are exposed.
        self.sources: Dict[str, Dict[str, Any]] = {}
        if self.rope_root.is_dir():
            self.sources[SOURCE_ROPE] = {
                "root": self.rope_root,
                "layout": "flat",
                "split": None,
                "oid_to_pam": self.rope_oid_to_pam,
                "match": "category",  # synthetic ref: exact (oid_to_pam) else nearest same-class
            }
        if self.sope_root.is_dir():
            self.sources[SOURCE_SOPE_TRAIN] = {
                "root": self.sope_root,
                "layout": "sope",
                "split": "train",
                "oid_to_pam": {},
            }
            self.sources[SOURCE_SOPE_TEST] = {
                "root": self.sope_root,
                "layout": "sope",
                "split": "test",
                "oid_to_pam": {},
            }
            # Filtered views: only SOPE/test objects+scenes whose category does NOT
            # appear in SOPE/train, at two granularities (novel-generalization subsets).
            if self.train_source_classes:
                self.sources[SOURCE_SOPE_TEST_NOVEL_SRC] = {
                    "root": self.sope_root, "layout": "sope", "split": "test",
                    "oid_to_pam": {}, "novel_level": "source_class",
                }
            if self.train_class_names:
                self.sources[SOURCE_SOPE_TEST_NOVEL_CLS] = {
                    "root": self.sope_root, "layout": "sope", "split": "test",
                    "oid_to_pam": {}, "novel_level": "class_name",
                }
        if not self.sources:
            raise RuntimeError(
                f"No Omni6DPose ROPE/SOPE roots found. rope={self.rope_root} sope={self.sope_root}"
            )

        self._scene_cache: Dict[str, List[str]] = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.checkpoint_path = resolve_checkpoint_path(checkpoint_path)
        self.available_checkpoints = self.discover_checkpoints()
        self.load_checkpoint(self.checkpoint_path)

    # ------------------------------------------------------------ setup helpers
    def _resolve_symmetry_info_path(self) -> Path | None:
        value = self.cfg.get("object_srt_symmetry_info_path")
        if value:
            path = Path(str(value)).expanduser()
            if path.is_file():
                return path
        local = PROJECT_ROOT / "omni6dpose_symmetry_info.json"
        return local if local.is_file() else None

    def _build_object_records_by_name(self) -> Dict[str, Dict[str, Any]]:
        if not self.object_image_root.is_dir():
            return {}
        records: Dict[str, Dict[str, Any]] = {}
        for object_dir in sorted(self.object_image_root.iterdir()):
            if not object_dir.is_dir():
                continue
            rgb_dir = object_dir / "rgb"
            if not rgb_dir.is_dir():
                continue
            available_ids = sorted(int(p.stem) for p in rgb_dir.glob("*.png") if p.stem.isdigit())
            if any(view_id not in available_ids for view_id in self.object_views):
                continue
            metadata_path = object_dir / "metadata.json"
            metadata = read_json(metadata_path) if metadata_path.is_file() else {}
            records[object_dir.name] = {
                "object_name": object_dir.name,
                "object_dir": object_dir,
                "image_ids": available_ids,
                "metadata": metadata,
            }
        return records

    def discover_checkpoints(self) -> List[str]:
        candidates = set()
        if DEFAULT_CHECKPOINT.is_file():
            candidates.add(str(DEFAULT_CHECKPOINT))
        candidates.add(str(self.checkpoint_path))
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

    # ------------------------------------------------------ dataset enumeration
    @property
    def dataset_choices(self) -> List[str]:
        return list(self.sources.keys())

    @property
    def default_dataset(self) -> str:
        return self.dataset_choices[0]

    @staticmethod
    def _oid_category(oid: str) -> str:
        stem = str(oid)
        if stem.startswith("real-"):
            stem = stem[len("real-"):]
        return stem.rsplit("_", 1)[0]

    @staticmethod
    def _oid_instance_id(oid: str) -> int:
        try:
            return int(str(oid).rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return -1

    @staticmethod
    def _oid_source_class(oid: str) -> str:
        """Source-qualified class, e.g. 'omniobject3d-cup_004' -> 'omniobject3d-cup'."""
        return str(oid).rsplit("_", 1)[0]

    @staticmethod
    def _oid_class_name(oid: str) -> str:
        """Source-agnostic class, e.g. 'omniobject3d-laundry_detergent_004' -> 'laundry_detergent'."""
        stem = str(oid).split("-", 1)[1] if "-" in str(oid) else str(oid)
        return stem.rsplit("_", 1)[0]

    def _is_novel(self, oid: str, level: str) -> bool:
        """True if this object's category (at `level`) never appears in SOPE/train."""
        if level == "class_name":
            return bool(self.train_class_names) and self._oid_class_name(oid) not in self.train_class_names
        # default: source_class
        return bool(self.train_source_classes) and self._oid_source_class(oid) not in self.train_source_classes

    def _resolve_pam_name(self, source_key: str, oid: str) -> str:
        source = self.sources[source_key]
        if source.get("match") == "category":
            # ROPE: substitute a synthetic diverse24 ref of the same class.
            # 1. exact instance, if a render exists (rope_oid_to_pam).
            mapped = source.get("oid_to_pam", {}).get(oid)
            if mapped and mapped in self.object_records_by_name:
                return str(mapped)
            # 2. nearest-instance diverse24 ref of the same class (approx).
            refs = self.refs_by_category.get(self._oid_class_name(oid))
            if refs:
                qid = self._oid_instance_id(oid)
                return min(refs, key=lambda n: (abs(self._oid_instance_id(n) - qid), self._oid_instance_id(n)))
            return str(oid)  # no same-class ref -> not referenceable
        return str(source["oid_to_pam"].get(oid, oid))

    def _is_category_approx(self, source_key: str, oid: str) -> bool:
        return (
            self.sources[source_key].get("match") == "category"
            and self._resolve_pam_name(source_key, oid) != str(oid)
        )

    def _ref_note(self, source_key: str, oid: str) -> str:
        """Concise label describing how a ROPE object's reference was substituted."""
        if not self._is_category_approx(source_key, oid):
            return ""
        resolved = self._resolve_pam_name(source_key, oid)
        if self.sources[source_key].get("oid_to_pam", {}).get(oid) == resolved:
            return " · synthetic ref (same instance)"
        if resolved.startswith("real-"):
            return " · category-approx (same-class real ref)"
        return " · synthetic category-approx (same class)"

    def get_scene_choices(self, source_key: str) -> List[str]:
        if source_key not in self.sources:
            return []
        if source_key in self._scene_cache:
            return self._scene_cache[source_key]
        source = self.sources[source_key]
        root = source["root"]
        scenes: List[str] = []
        if source["layout"] == "flat":
            # ROPE: only list scenes that contain at least one object with a
            # rendered reference (i.e. resolvable via rope_oid_to_pam → diverse24),
            # so the dropdown isn't filled with scenes nothing can be inferred on.
            for scene_dir in sorted(root.iterdir()):
                if scene_dir.is_dir() and scene_dir.name.isdigit() and \
                        self._flat_scene_has_referenceable_object(source_key, scene_dir):
                    scenes.append(scene_dir.name)
        else:  # sope: <patch>/<split>/<source>/<scene>
            split = source["split"]
            novel_level = source.get("novel_level")
            for patch_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                split_dir = patch_dir / split
                if not split_dir.is_dir():
                    continue
                for source_dir in sorted(s for s in split_dir.iterdir() if s.is_dir()):
                    for scene_dir in sorted(c for c in source_dir.iterdir() if c.is_dir()):
                        if novel_level and not self._sope_scene_has_novel(scene_dir, novel_level):
                            continue
                        scenes.append(str(scene_dir.relative_to(root)))
        self._scene_cache[source_key] = scenes
        return scenes

    def _sope_scene_has_novel(self, scene_dir: Path, level: str) -> bool:
        """True iff this SOPE scene contains ≥1 object whose category (at `level`)
        does not appear in SOPE/train (object set is fixed per scene)."""
        metas = sorted(scene_dir.glob("*_meta.json"))
        if not metas:
            return False
        try:
            meta = read_json(metas[0])
        except Exception:
            return False
        for obj in meta.get("objects", {}).values():
            if not obj.get("is_valid", True):
                continue
            om = obj.get("meta", {})
            if om.get("is_background", False):
                continue
            if self._is_novel(str(om.get("oid", "")), level):
                return True
        return False

    def _flat_scene_has_referenceable_object(self, source_key: str, scene_dir: Path) -> bool:
        """True iff this flat-layout (ROPE) scene has ≥1 object whose canonical
        mesh has a rendered reference. A scene's object set is fixed across frames,
        so inspecting the first frame's meta is sufficient."""
        metas = sorted(scene_dir.glob("*_meta.json"))
        if not metas:
            return False
        try:
            meta = read_json(metas[0])
        except Exception:
            return False
        for obj in meta.get("objects", {}).values():
            if not obj.get("is_valid", True):
                continue
            om = obj.get("meta", {})
            if om.get("is_background", False):
                continue
            if self._resolve_pam_name(source_key, str(om.get("oid", ""))) in self.object_records_by_name:
                return True
        return False

    def _scene_dir(self, source_key: str, scene_name: str) -> Path:
        return self.sources[source_key]["root"] / scene_name

    def get_frame_choices(self, source_key: str, scene_name: str) -> List[str]:
        if not (source_key and scene_name) or source_key not in self.sources:
            return []
        scene_dir = self._scene_dir(source_key, scene_name)
        if not scene_dir.is_dir():
            return []
        return [p.name[: -len("_meta.json")] for p in sorted(scene_dir.glob("*_meta.json"))]

    def _frame_paths(self, source_key: str, scene_name: str, frame_name: str):
        scene_dir = self._scene_dir(source_key, scene_name)
        meta_path = scene_dir / f"{frame_name}_meta.json"
        color_path = scene_dir / f"{frame_name}_color.png"
        depth_path = scene_dir / f"{frame_name}_depth.exr"
        mask_path = scene_dir / f"{frame_name}_mask.exr"
        return meta_path, color_path, depth_path, mask_path

    def object_choices_for_frame(self, source_key: str, scene_name: str, frame_name: str):
        """List every valid, non-background object in the frame.

        Objects whose canonical mesh has a rendered diverse24 reference are
        marked with the ref name (the model can predict pose for them); objects
        without a reference are still listed (labelled "no ref") so the whole
        scene is visible — selecting them yields a GT-only visualization.
        """
        if not (source_key and scene_name and frame_name) or source_key not in self.sources:
            return []
        meta_path, _, _, _ = self._frame_paths(source_key, scene_name, frame_name)
        if not meta_path.is_file():
            return []
        meta = read_json(meta_path)
        novel_level = self.sources[source_key].get("novel_level")
        with_ref, without_ref = [], []
        for obj_key, obj in meta.get("objects", {}).items():
            if not obj.get("is_valid", True):
                continue
            om = obj.get("meta", {})
            if om.get("is_background", False):
                continue
            oid = str(om.get("oid", ""))
            # Novel view: only list objects whose category (at this level) is absent from train.
            if novel_level and not self._is_novel(oid, novel_level):
                continue
            category = str(om.get("class_name", ""))
            pam_name = self._resolve_pam_name(source_key, oid)
            if pam_name in self.object_records_by_name:
                approx = self._ref_note(source_key, oid)
                with_ref.append((f"[{category}] {oid} → ref {pam_name}{approx}", str(obj_key)))
            else:
                without_ref.append((f"[{category}] {oid} (no ref · GT only)", str(obj_key)))
        # Objects that can actually be inferred come first.
        return with_ref + without_ref

    def has_reference(self, source_key: str, oid: str) -> bool:
        return self._resolve_pam_name(source_key, oid) in self.object_records_by_name

    def scene_preview(self, source_key: str, scene_name: str, frame_name: str):
        """Scene RGB path — independent of which object is selected."""
        if not (source_key and scene_name and frame_name) or source_key not in self.sources:
            return None
        _, color_path, _, _ = self._frame_paths(source_key, scene_name, frame_name)
        return str(color_path) if color_path.is_file() else None

    def input_gallery(self, source_key: str, scene_name: str, frame_name: str, obj_key: str):
        scene_image = self.scene_preview(source_key, scene_name, frame_name)
        if not (scene_image and obj_key):
            return scene_image, []
        meta_path, _, _, _ = self._frame_paths(source_key, scene_name, frame_name)
        if not meta_path.is_file():
            return scene_image, []
        meta = read_json(meta_path)
        obj = meta.get("objects", {}).get(obj_key)
        if obj is None:
            return scene_image, []
        pam_name = self._resolve_pam_name(source_key, str(obj.get("meta", {}).get("oid", "")))
        if pam_name not in self.object_records_by_name:
            return scene_image, []
        _, gallery = load_omni6dpose_object_tensor(
            self.object_records_by_name[pam_name]["object_dir"],
            self.object_views,
            self.resolution,
            torch.device("cpu"),
        )
        return scene_image, gallery

    @staticmethod
    def _compute_depth_mean_scale(
        gt_depth: np.ndarray,
        pred_depth: torch.Tensor | None,
        use_depth_input: bool,
        z_far: float = 0.0,
        eps: float = 1e-6,
    ) -> float:
        # Match training's normalization: depth.mean() over (depth > 0) & (depth < z_far).
        def _masked_mean(arr: np.ndarray) -> float | None:
            arr = np.asarray(arr, dtype=np.float32).reshape(-1)
            valid = np.isfinite(arr) & (arr > 0)
            if z_far and z_far > 0:
                valid &= arr < float(z_far)
            arr = arr[valid]
            return float(max(float(arr.mean()), eps)) if arr.size > 0 else None

        if use_depth_input:
            mean = _masked_mean(gt_depth)
            if mean is not None:
                return mean
        if pred_depth is not None:
            mean = _masked_mean(pred_depth.detach().float().cpu().numpy())
            if mean is not None:
                return mean
        return 1.0

    # ------------------------------------------------------------- inference
    def run_inference(
        self,
        source_key: str,
        scene_name: str,
        frame_name: str,
        obj_key: str,
        use_depth_input: bool,
        use_depth_scale: bool,
        show_point_cloud_pose: bool,
        point_cloud_stride: int,
        target_crop: bool,
        pred_pose_gt_bbox: bool,
    ):
        if source_key not in self.sources:
            raise RuntimeError(f"Unknown dataset source: {source_key}")
        meta_path, color_path, depth_path, mask_path = self._frame_paths(source_key, scene_name, frame_name)
        if not meta_path.is_file():
            raise RuntimeError(f"Meta file not found: {meta_path}")
        meta = read_json(meta_path)
        obj = meta.get("objects", {}).get(obj_key)
        if obj is None:
            raise RuntimeError(f"Object `{obj_key}` not found in {meta_path}")
        om = obj.get("meta", {})
        oid = str(om.get("oid", ""))
        category = str(om.get("class_name", ""))
        pam_name = self._resolve_pam_name(source_key, oid)
        has_ref = pam_name in self.object_records_by_name
        object_id = int(self.object_name_to_id[pam_name]) if has_ref else 0
        # mask_id follows the official Omni6DPose convention: the int prefix of the
        # object key (e.g. "2_toy_train_005" -> 2), matching the mask EXR ids.
        try:
            mask_id = int(str(obj_key).split("_", 1)[0])
        except ValueError:
            mask_id = int(obj.get("id", 0))

        gt_mask_full = (
            omni6dpose_read_object_mask(mask_path, mask_id)
            if mask_path.is_file()
            else None
        )

        processor = DemoScenePreprocessor(resolution=self.resolution, z_far=self.z_far)
        (
            scene_tensor,
            depth_tensor,
            mask_tensor,
            display_image,
            display_depth,
            gt_mask,
            intrinsic,
        ) = load_omni6dpose_scene_frame_inputs(
            processor,
            color_path,
            depth_path,
            meta["camera"]["intrinsics"],
            gt_mask_full,
            self.resolution,
            self.device,
            target_crop=target_crop,
        )

        # Ground truth: object canonical (PAM Aligned.obj) -> camera. R_align is identity.
        gt_rotation = quaternion_wxyz_to_matrix(obj["quaternion_wxyz"]).astype(np.float32)
        gt_translation_cam = np.asarray(obj["translation"], dtype=np.float32).reshape(3)
        gt_size = np.clip(
            np.asarray(om.get("bbox_side_len", [1.0, 1.0, 1.0]), dtype=np.float32).reshape(3), 1e-6, None
        )
        axis_length = max(float(np.linalg.norm(gt_size)) * 0.25, 1e-3)
        gt_bbox_obj = centered_axis_bbox_corners(gt_size)
        gt_image = draw_bbox_axes_overlay_on_image(
            display_image, intrinsic, gt_rotation, gt_translation_cam, gt_bbox_obj,
            axis_length, AXIS_COLORS, GT_BBOX_COLOR,
        )
        gt_mask_image = mask_overlay_image(display_image, gt_mask, color=(80, 255, 120)) if gt_mask is not None else None

        # Objects without a rendered reference cannot be fed to the reference-based
        # model — show GT-only visualization instead of erroring out.
        if not has_ref:
            point_cloud_glb = None
            if show_point_cloud_pose:
                point_cloud_glb = export_point_cloud_pose_glb(
                    f"{source_key}_{scene_name}", frame_name, display_image, display_depth, intrinsic,
                    gt_rotation_cam=gt_rotation, gt_translation_cam=gt_translation_cam, gt_bbox_obj=gt_bbox_obj,
                    axis_length=axis_length, point_cloud_stride=point_cloud_stride,
                )
            pose_lines = [
                "### Prediction skipped — no object reference",
                f"- dataset: `{source_key}`  scene: `{scene_name}`  frame: `{frame_name}`",
                f"- object: `{obj_key}` · oid `{oid}` · `{category}` (mask id {mask_id})",
                f"- this object's canonical mesh `{pam_name}` has **no rendered diverse24 reference** under "
                f"`{self.object_image_root}`, so the reference-based model cannot estimate its pose.",
                "- Only the ground-truth pose is visualized (orange). Pick an object labelled `→ ref ...` to run the model.",
                "",
                "### Ground Truth",
                f"- rotation matrix: `{np.round(gt_rotation, 6).tolist()}`",
                f"- camera-frame translation (m): `{np.round(gt_translation_cam, 6).tolist()}`",
                f"- bbox_side_len xyz (m): `{np.round(gt_size, 6).tolist()}`",
            ]
            return "\n".join(pose_lines), display_image, gt_image, point_cloud_glb, None, gt_mask_image

        object_dir = self.object_records_by_name[pam_name]["object_dir"]
        object_tensor, _ = load_omni6dpose_object_tensor(
            object_dir, self.object_views, self.resolution, self.device
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
            raise RuntimeError(f"Model output missing object_presence_logits: {sorted(outputs.keys())}")
        presence_logit = float(outputs["object_presence_logits"].reshape(-1)[0].detach().float().cpu())
        presence_prob = float(torch.sigmoid(torch.tensor(presence_logit)).item())
        pred_present = presence_prob >= 0.5

        pred_mask = None
        pred_mask_image = None
        if "object_mask" in outputs:
            pred_mask = outputs["object_mask"][0, 0].detach().float().cpu().numpy()
            pred_mask_image = mask_overlay_image(display_image, pred_mask, color=(255, 64, 64))

        if "object_pose" not in outputs or "object_translation" not in outputs:
            raise RuntimeError(f"Model output missing pose keys: {sorted(outputs.keys())}")
        pred_rot6d = outputs["object_pose"][0].detach().float().cpu().numpy()
        pred_translation_cam = outputs["object_translation"][0].detach().float().cpu().numpy().astype(np.float32)
        pred_rotation = rot6d_to_matrix(pred_rot6d).astype(np.float32)

        depth_mean_scale = self._compute_depth_mean_scale(
            display_depth,
            outputs.get("depth"),
            use_depth_input,
            z_far=self.z_far,
        )
        if use_depth_scale:
            pred_translation_cam = pred_translation_cam * np.float32(depth_mean_scale)

        pred_size = None
        if "object_size" in outputs:
            pred_size = outputs["object_size"].reshape(-1, 3)[0].detach().float().cpu().numpy().astype(np.float32)
        elif "object_size_log" in outputs:
            pred_size = np.exp(outputs["object_size_log"].reshape(-1, 3)[0].detach().float().cpu().numpy()).astype(np.float32)

        raw_pred_size = pred_size if pred_size is not None else gt_size
        size_for_pred_box, size_was_clipped = clamp_predicted_size_for_bbox(raw_pred_size, gt_size)
        visualized_pred_size = gt_size if pred_pose_gt_bbox else size_for_pred_box
        pred_bbox_obj = centered_axis_bbox_corners(visualized_pred_size)

        rot_error_deg, sym_count = symmetric_rotation_error_degrees(
            pred_rotation,
            gt_rotation,
            object_id,
            self.symmetry_info_path,
            self.symmetry_continuous_steps,
        )
        trans_error = translation_error(pred_translation_cam, gt_translation_cam)
        size_error = translation_error(raw_pred_size, gt_size)
        mask_score = mask_iou(pred_mask, gt_mask)
        raw_bbox_iou_details = bbox_iou_3d_details(
            raw_pred_size, pred_rotation, pred_translation_cam, gt_size, gt_rotation, gt_translation_cam
        )
        visualized_bbox_iou_details = bbox_iou_3d_details(
            visualized_pred_size, pred_rotation, pred_translation_cam, gt_size, gt_rotation, gt_translation_cam
        )

        pred_image = draw_bbox_axes_overlay_on_image(
            display_image, intrinsic, pred_rotation, pred_translation_cam, pred_bbox_obj,
            axis_length, AXIS_COLORS, PRED_BBOX_COLOR,
        )

        point_cloud_glb = None
        if show_point_cloud_pose:
            point_cloud_glb = export_point_cloud_pose_glb(
                f"{source_key}_{scene_name}",
                frame_name,
                display_image,
                display_depth,
                intrinsic,
                pred_rotation_cam=pred_rotation,
                pred_translation_cam=pred_translation_cam,
                pred_bbox_obj=pred_bbox_obj,
                gt_rotation_cam=gt_rotation,
                gt_translation_cam=gt_translation_cam,
                gt_bbox_obj=gt_bbox_obj,
                axis_length=axis_length,
                point_cloud_stride=point_cloud_stride,
            )

        pose_lines = [
            "### Prediction Summary",
            f"- checkpoint: `{self.checkpoint_path}`",
            f"- dataset: `{source_key}`",
            f"- scene: `{scene_name}`",
            f"- frame: `{frame_name}`",
            f"- target crop input: `{target_crop}`",
            f"- predicted pose with GT bbox size: `{pred_pose_gt_bbox}`",
            f"- object: `{obj_key}` · oid `{oid}` · `{category}` (mask id {mask_id})",
            f"- reference render: `{pam_name}` (object id `{object_id}`)"
            + (f" · **{self._ref_note(source_key, oid).lstrip(' ·')}** → pose is approximate"
               if self._is_category_approx(source_key, oid) else ""),
            f"- predicted presence probability: `{presence_prob:.6f}` (logit `{presence_logit:.6f}`)",
            f"- predicted present @0.5: `{bool(pred_present)}`",
            f"- use depth input: `{use_depth_input}`",
            f"- use depth scale: `{use_depth_scale}`",
            "",
            "### Predicted Pose (object canonical frame, R_align = I)",
            f"- depth_mean_scale (m): `{depth_mean_scale:.6f}` "
            f"({'GT depth mean' if use_depth_input else 'predicted depth mean'})",
            f"- applied depth scale to translation: `{use_depth_scale}`",
            f"- camera-frame translation (m): `{np.round(pred_translation_cam, 6).tolist()}`",
            f"- camera-frame rotation matrix: `{np.round(pred_rotation, 6).tolist()}`",
            f"- raw predicted size xyz (m): `{np.round(raw_pred_size, 6).tolist()}`",
            f"- visualized predicted bbox size xyz (m): `{np.round(visualized_pred_size, 6).tolist()}`"
            + (" from GT bbox" if pred_pose_gt_bbox else (" clipped for display" if size_was_clipped else "")),
            "",
            "### Ground Truth",
            f"- rotation matrix: `{np.round(gt_rotation, 6).tolist()}`",
            f"- camera-frame translation (m): `{np.round(gt_translation_cam, 6).tolist()}`",
            f"- bbox_side_len xyz (m): `{np.round(gt_size, 6).tolist()}`",
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
            f"- mask IoU @0.5: `{mask_score:.6f}`" if mask_score is not None else "- mask IoU @0.5: `N/A` (enable Target Crop / mask EXR)",
        ]
        return "\n".join(pose_lines), pred_image, gt_image, point_cloud_glb, pred_mask_image, gt_mask_image


def build_demo(app: DemoApp):
    default_dataset = app.default_dataset
    default_scenes = app.get_scene_choices(default_dataset)
    default_scene = default_scenes[0] if default_scenes else None
    default_frames = app.get_frame_choices(default_dataset, default_scene) if default_scene else []
    default_frame = default_frames[0] if default_frames else None
    default_object_choices = (
        app.object_choices_for_frame(default_dataset, default_scene, default_frame)
        if default_scene and default_frame
        else []
    )
    default_object = default_object_choices[0][1] if default_object_choices else None

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
            "# OmniVGGT 6D Pose Demo · Omni6DPose ROPE + SOPE (train / test)",
            f"- config: `{app.config_path}`",
            f"- Omni6DPose root: `{app.omni6dpose_root}`",
            f"- ROPE root: `{app.rope_root}`",
            f"- SOPE root: `{app.sope_root}`",
            f"- diverse24 object refs: `{app.object_image_root}`",
            f"- ROPE real-* ref categories: `{ {c: len(v) for c, v in app.real_refs_by_category.items()} }`",
            f"- ROPE exact synthetic-ref entries (oid_to_pam): `{len(app.rope_oid_to_pam)}`",
            f"- symmetry info: `{app.symmetry_info_path}`",
            f"- loaded checkpoint: `{app.checkpoint_path}`",
            f"- object views: `{app.object_views}`",
            f"- inference resolution: `{app.resolution}`",
            "",
            "Pred / GT 都以 PAM `Aligned.obj` 物體坐標系顯示 (`R_align = I`)。GT pose 直接由 meta JSON 的 ",
            "`quaternion_wxyz` + `translation` (object→camera) 取得，size 由 `bbox_side_len`。模型預測的 translation ",
            "為 depth-mean 正規化值，勾選 *Use Depth Scale* 會乘回 `depth.mean()` 還原 metric translation。",
            "ROPE 為真實評測集 (flat layout)；SOPE 為合成集，分 train / test split。",
            "**SOPE/test (novel source-class)**：source-class（來源+類別）在 train 沒出現過的 test 物件/場景 ",
            "（含同類別但新 mesh 來源，如 phocal-laundry_detergent；共 15 類 / 577 場景）。",
            "**SOPE/test (novel class)**：純語意 class 名稱在 train 完全沒出現過的（目前僅 guitar / 36 場景）。",
            "**ROPE 物件無自己的 PAM mesh/ref**，故借用同類別的「合成」diverse24 reference：",
            "(1) 少數有 `rope_oid_to_pam.json` 對到同一 instance 的合成 render（如 real-chess_001 → ",
            "omniobject3d-chess_001）；(2) 其餘取語意類別相同、instance id 最接近的合成 ref（多數屬此），",
            "標示為 `synthetic category-approx`，pose 為近似值。僅 camera / laptop 另有 real-* reference。",
        ]
    )

    with gr.Blocks(title="OmniVGGT Omni6DPose Demo", css=demo_css) as demo:
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
            dataset_dropdown = gr.Dropdown(
                choices=app.dataset_choices,
                value=default_dataset,
                label="Dataset",
            )
            scene_dropdown = gr.Dropdown(choices=default_scenes, value=default_scene, label="Scene")
            frame_dropdown = gr.Dropdown(choices=default_frames, value=default_frame, label="Frame")
            object_dropdown = gr.Dropdown(choices=default_object_choices, value=default_object, label="Object")
            use_depth_checkbox = gr.Checkbox(value=True, label="Use Depth Input")
            use_depth_scale_checkbox = gr.Checkbox(value=True, label="Use Depth Scale")
            target_crop_checkbox = gr.Checkbox(value=False, label="Target Crop Input")
            pred_pose_gt_bbox_checkbox = gr.Checkbox(value=False, label="Pred Pose + GT BBox")
            show_point_cloud_checkbox = gr.Checkbox(value=False, label="Show Point Cloud Pose")
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
                scene_input_image = compat_image(
                    label="Scene Input (RGB)",
                    height=360,
                    elem_id="scene_inputs",
                    interactive=False,
                )
                object_input_gallery = gr.Gallery(
                    label="Object Inputs",
                    columns=max(len(app.object_views), 1),
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

        def refresh_inputs(source_key, scene_name, frame_name, obj_key):
            # Scene RGB shows for any (source, scene, frame); the object gallery is
            # only populated when the selected object has a reference render.
            if not (source_key and scene_name and frame_name):
                return None, []
            return app.input_gallery(source_key, scene_name, frame_name, obj_key)

        def refresh_dataset_controls(
            source_key: str,
            preferred_scene: str | None = None,
            preferred_frame: str | None = None,
            preferred_object: str | None = None,
        ):
            scenes = app.get_scene_choices(source_key)
            scene_value = preferred_scene if preferred_scene in scenes else (scenes[0] if scenes else None)
            frames = app.get_frame_choices(source_key, scene_value) if scene_value else []
            frame_value = preferred_frame if preferred_frame in frames else (frames[0] if frames else None)
            object_choices = (
                app.object_choices_for_frame(source_key, scene_value, frame_value)
                if scene_value and frame_value
                else []
            )
            object_values = [value for _, value in object_choices]
            object_value = preferred_object if preferred_object in object_values else (object_values[0] if object_values else None)
            scene_image, object_gallery = refresh_inputs(source_key, scene_value, frame_value, object_value)
            return (
                gr.update(choices=scenes, value=scene_value),
                gr.update(choices=frames, value=frame_value),
                gr.update(choices=object_choices, value=object_value),
                scene_image,
                object_gallery,
            )

        def refresh_scene_controls(
            source_key: str,
            scene_name: str,
            preferred_frame: str | None = None,
            preferred_object: str | None = None,
        ):
            frames = app.get_frame_choices(source_key, scene_name) if scene_name else []
            frame_value = preferred_frame if preferred_frame in frames else (frames[0] if frames else None)
            object_choices = (
                app.object_choices_for_frame(source_key, scene_name, frame_value)
                if scene_name and frame_value
                else []
            )
            object_values = [value for _, value in object_choices]
            object_value = preferred_object if preferred_object in object_values else (object_values[0] if object_values else None)
            return gr.update(choices=frames, value=frame_value), gr.update(choices=object_choices, value=object_value)

        def refresh_frame_controls(
            source_key: str,
            scene_name: str,
            frame_name: str,
            preferred_object: str | None = None,
        ):
            if not (source_key and scene_name and frame_name):
                return gr.update()
            object_choices = app.object_choices_for_frame(source_key, scene_name, frame_name)
            object_values = [value for _, value in object_choices]
            object_value = preferred_object if preferred_object in object_values else (object_values[0] if object_values else None)
            return gr.update(choices=object_choices, value=object_value)

        def sync_checkpoint_path(selected_value: str):
            return selected_value

        dataset_dropdown.change(
            refresh_dataset_controls,
            inputs=[dataset_dropdown, scene_dropdown, frame_dropdown, object_dropdown],
            outputs=[scene_dropdown, frame_dropdown, object_dropdown, scene_input_image, object_input_gallery],
        )
        scene_dropdown.change(
            refresh_scene_controls,
            inputs=[dataset_dropdown, scene_dropdown, frame_dropdown, object_dropdown],
            outputs=[frame_dropdown, object_dropdown],
        )
        scene_dropdown.change(
            refresh_inputs,
            inputs=[dataset_dropdown, scene_dropdown, frame_dropdown, object_dropdown],
            outputs=[scene_input_image, object_input_gallery],
        )
        frame_dropdown.change(
            refresh_frame_controls,
            inputs=[dataset_dropdown, scene_dropdown, frame_dropdown, object_dropdown],
            outputs=object_dropdown,
        )
        frame_dropdown.change(
            refresh_inputs,
            inputs=[dataset_dropdown, scene_dropdown, frame_dropdown, object_dropdown],
            outputs=[scene_input_image, object_input_gallery],
        )
        object_dropdown.change(
            refresh_inputs,
            inputs=[dataset_dropdown, scene_dropdown, frame_dropdown, object_dropdown],
            outputs=[scene_input_image, object_input_gallery],
        )
        checkpoint_dropdown.change(sync_checkpoint_path, inputs=checkpoint_dropdown, outputs=checkpoint_textbox)
        load_model_button.click(
            app.load_checkpoint,
            inputs=checkpoint_textbox,
            outputs=[checkpoint_status, checkpoint_dropdown, checkpoint_textbox],
        )
        infer_button.click(
            app.run_inference,
            inputs=[
                dataset_dropdown,
                scene_dropdown,
                frame_dropdown,
                object_dropdown,
                use_depth_checkbox,
                use_depth_scale_checkbox,
                show_point_cloud_checkbox,
                point_cloud_stride_slider,
                target_crop_checkbox,
                pred_pose_gt_bbox_checkbox,
            ],
            outputs=[summary_markdown, pred_image, gt_image, point_cloud_model, pred_mask_image, gt_mask_image],
        )
        if default_scene is not None and default_frame is not None:
            demo.load(
                refresh_inputs,
                inputs=[dataset_dropdown, scene_dropdown, frame_dropdown, object_dropdown],
                outputs=[scene_input_image, object_input_gallery],
            )
    return demo


def main():
    parser = argparse.ArgumentParser(
        description="Gradio demo for OmniVGGT 6D pose inference on Omni6DPose ROPE/SOPE (train/test)."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--omni6dpose-root", type=Path, default=DEFAULT_OMNI6DPOSE_ROOT)
    parser.add_argument("--rope-root", type=Path, default=DEFAULT_ROPE_ROOT)
    parser.add_argument("--sope-root", type=Path, default=DEFAULT_SOPE_ROOT)
    parser.add_argument("--object-image-root", type=Path, default=DEFAULT_OBJECT_IMAGE_ROOT)
    parser.add_argument("--rope-oid-to-pam", type=Path, default=DEFAULT_ROPE_OID_TO_PAM)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    app = DemoApp(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        omni6dpose_root=args.omni6dpose_root,
        rope_root=args.rope_root,
        sope_root=args.sope_root,
        object_image_root=args.object_image_root,
        rope_oid_to_pam=args.rope_oid_to_pam,
    )
    demo = build_demo(app)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[
            str(app.omni6dpose_root),
            str(app.rope_root),
            str(app.sope_root),
            str(app.object_image_root),
        ],
    )


if __name__ == "__main__":
    main()
