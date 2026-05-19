import os
from pathlib import Path

# Must be set before `import gradio` so cache files land in a writable location.
PROJECT_ROOT = Path(__file__).resolve().parent
_LOCAL_TMP = PROJECT_ROOT / "tmp"
_LOCAL_TMP.mkdir(parents=True, exist_ok=True)
(_LOCAL_TMP / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", str(_LOCAL_TMP))
os.environ.setdefault("GRADIO_CACHE_DIR", str(_LOCAL_TMP))
os.environ.setdefault("TMPDIR", str(_LOCAL_TMP))
os.environ.setdefault("MPLCONFIGDIR", str(_LOCAL_TMP / "matplotlib"))
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# cd /mnt/train-data-4-hdd/yian/freepose/omni-object_clone
# python3 demo_gradio_6dpose_co3d.py --port 7860

import argparse
import gzip
import inspect
import json
import re
import runpy
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import cv2
import gradio as gr
import numpy as np
import torch
import trimesh
from PIL import Image
from safetensors.torch import load_file as load_safetensors_file

from omnivggt.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from omnivggt.models.omnivggt import OmniVGGT
from omnivggt.utils.image import ImgNorm


# ============================================================
# Constants
# ============================================================
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train_oo9d.py"
DEFAULT_PRETRAIN_MODEL = Path(
    "/mnt/train-data-4-hdd/yian/freepose/omni-object_clone/outputs/0515_LARGE/model.safetensors"
)
DEFAULT_CO3D_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose/co3d/data")
DEFAULT_OV9D_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d")
DEFAULT_OBJECT_IMAGE_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d_around_image")
DEFAULT_TRAIN_SPLIT_JSON = Path(
    "/mnt/train-data-4-hdd/yian/freepose/omni-object_clone/splits_ov9d_seen_unseen_scene/single/train.json"
)

# Anchor reference views used by the OO9D 0515 model.
FIXED_OBJECT_VIEWS = (1, 5, 10, 15)

# Alternative: pick the first few CO3D frames of the scene itself as the object
# reference. Requires the scene to have at least CO3D_OBJECT_SOURCE_MIN_FRAMES
# annotated frames; each one's mask is used to white out the background.
CO3D_OBJECT_FRAMES = (1, 2, 3, 4)
CO3D_OBJECT_SOURCE_MIN_FRAMES = 5

OBJECT_SOURCE_ANCHOR = "anchor"
OBJECT_SOURCE_CO3D_SCENE = "co3d_scene"

PRED_AXIS_COLORS = ((255, 64, 64), (0, 255, 255), (255, 215, 0))
PRED_BBOX_COLOR = (0, 255, 0)
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


def resolve_resolution(cfg: Dict) -> Tuple[int, int]:
    return tuple(int(v) for v in cfg.get("resolution", (518, 518)))


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


# ============================================================
# CO3D dataset access
# ============================================================
def read_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=64)
def co3d_load_frame_annotations(category_dir: str) -> Dict[Tuple[str, int], Dict]:
    """Return a dict keyed by (sequence_name, frame_number) -> annotation entry."""
    path = Path(category_dir) / "frame_annotations.jgz"
    with gzip.open(path, "rt") as handle:
        records = json.load(handle)
    return {(str(r["sequence_name"]), int(r["frame_number"])): r for r in records}


@lru_cache(maxsize=64)
def co3d_load_sequence_annotations(category_dir: str) -> Dict[str, Dict]:
    path = Path(category_dir) / "sequence_annotations.jgz"
    if not path.is_file():
        return {}
    with gzip.open(path, "rt") as handle:
        records = json.load(handle)
    return {str(r["sequence_name"]): r for r in records}


def co3d_list_categories(co3d_root: Path) -> List[str]:
    return sorted(
        p.name for p in co3d_root.iterdir()
        if p.is_dir() and not p.name.startswith("_") and (p / "frame_annotations.jgz").is_file()
    )


def co3d_list_scenes(co3d_root: Path, category: str) -> List[str]:
    category_dir = co3d_root / category
    if not category_dir.is_dir():
        return []
    return sorted(
        p.name for p in category_dir.iterdir()
        if p.is_dir() and (p / "images").is_dir()
    )


def co3d_list_frames(co3d_root: Path, category: str, scene_name: str) -> List[str]:
    images_dir = co3d_root / category / scene_name / "images"
    if not images_dir.is_dir():
        return []
    frames = []
    for path in sorted(images_dir.glob("frame*.jpg")):
        match = re.match(r"frame(\d+)$", path.stem)
        if match:
            frames.append(f"{int(match.group(1)):06d}")
    return frames


def co3d_ndc_isotropic_to_pixel_intrinsics(viewpoint: Dict, image_size_hw: Tuple[int, int]) -> np.ndarray:
    """Convert PyTorch3D ndc_isotropic camera params to an OpenCV-style pixel K matrix.

    Uses s = min(H, W) / 2 so both axes share the same NDC unit, and flips the
    PyTorch3D principal-point sign (x left, y up) to OpenCV (x right, y down).
    """
    height, width = int(image_size_hw[0]), int(image_size_hw[1])
    fl = viewpoint["focal_length"]
    pp = viewpoint["principal_point"]
    s = 0.5 * min(height, width)
    fx = float(fl[0]) * s
    fy = float(fl[1]) * s
    cx = -float(pp[0]) * s + 0.5 * width
    cy = -float(pp[1]) * s + 0.5 * height
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def co3d_load_depth_scene_units(depth_path: Path, scale_adjustment: float) -> np.ndarray:
    """Decode a CO3D 16-bit PNG depth (float16-packed) into a 2D float32 array.

    Returned values are in CO3D *scene units* (NOT metres). CO3D V2 reconstructions
    are metric-ambiguous: depth and viewpoint T share the same arbitrary scale.
    Callers must multiply by a per-scene factor before feeding the depth to a
    metric-trained model.
    """
    with Image.open(depth_path) as image:
        raw = np.array(image, dtype=np.uint16)
    depth = np.frombuffer(raw.tobytes(), dtype=np.float16).astype(np.float32).reshape(raw.shape)
    depth = depth * float(scale_adjustment)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0.0] = 0.0
    return depth.astype(np.float32)


def co3d_load_depth_validity(mask_path: Path | None, shape_hw: Tuple[int, int]) -> np.ndarray | None:
    if mask_path is None or not Path(mask_path).is_file():
        return None
    arr = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    if arr.shape != tuple(shape_hw):
        arr = cv2.resize(arr, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (arr > 0).astype(np.float32)


def co3d_load_foreground_mask(mask_path: Path | None, shape_hw: Tuple[int, int]) -> np.ndarray | None:
    if mask_path is None or not Path(mask_path).is_file():
        return None
    arr = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    if arr.shape != tuple(shape_hw):
        arr = cv2.resize(arr, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (arr > 0).astype(np.float32)


def robust_extent(points: np.ndarray, lower: float = 5.0, upper: float = 95.0) -> np.ndarray | None:
    points = np.asarray(points, dtype=np.float32)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 32:
        return None
    lo = np.percentile(points, lower, axis=0)
    hi = np.percentile(points, upper, axis=0)
    extent = hi - lo
    if not np.all(np.isfinite(extent)) or np.any(extent <= 1e-6):
        return None
    return extent.astype(np.float32)


def estimate_scale_from_sequence_pointcloud(
    co3d_root: Path,
    category: str,
    scene_name: str,
    anchor_size_m: np.ndarray,
) -> Tuple[float | None, str | None]:
    sequence_annotations = co3d_load_sequence_annotations(str(co3d_root / category))
    sequence_record = sequence_annotations.get(str(scene_name))
    point_cloud_record = sequence_record.get("point_cloud") if sequence_record else None
    point_cloud_rel = point_cloud_record.get("path") if point_cloud_record else None
    if not point_cloud_rel:
        return None, None
    point_cloud_path = co3d_root / point_cloud_rel
    if not point_cloud_path.is_file():
        return None, None
    try:
        cloud = trimesh.load(point_cloud_path, process=False)
        vertices = np.asarray(cloud.vertices, dtype=np.float32)
    except Exception as exc:
        return None, f"pointcloud scale unavailable ({exc})"
    extent_scene = robust_extent(vertices)
    if extent_scene is None:
        return None, "pointcloud scale unavailable (invalid pointcloud extent)"
    anchor_diag_m = float(np.linalg.norm(np.asarray(anchor_size_m, dtype=np.float32).reshape(3)))
    scene_diag = float(np.linalg.norm(extent_scene))
    if anchor_diag_m <= 1e-6 or scene_diag <= 1e-6:
        return None, "pointcloud scale unavailable (invalid diagonal)"
    scale = float(np.clip(anchor_diag_m / scene_diag, 1e-3, 10.0))
    reason = (
        f"pointcloud robust extent scene={np.round(extent_scene, 4).tolist()} "
        f"anchor_size_m={np.round(anchor_size_m, 4).tolist()}"
    )
    return scale, reason


def estimate_scale_from_frame_mask(
    co3d_root: Path,
    record: Dict,
    intrinsics: np.ndarray,
    image_size_hw: Tuple[int, int],
    anchor_size_m: np.ndarray,
) -> Tuple[float, str]:
    height, width = int(image_size_hw[0]), int(image_size_hw[1])
    mask_rel = record.get("mask", {}).get("path")
    mask = co3d_load_foreground_mask(co3d_root / mask_rel if mask_rel else None, (height, width))
    if mask is None or not np.any(mask > 0):
        return 1.0, "fallback=1.0 (missing foreground mask)"

    depth_scene = None
    if record.get("depth"):
        depth_path = co3d_root / record["depth"]["path"]
        if depth_path.is_file():
            depth_scene = co3d_load_depth_scene_units(
                depth_path,
                float(record["depth"].get("scale_adjustment", 1.0)),
            )
            validity = co3d_load_depth_validity(
                co3d_root / record["depth"].get("mask_path", ""),
                shape_hw=depth_scene.shape,
            )
            if validity is not None:
                depth_scene = depth_scene * validity

    z_scene = None
    z_source = "camera translation norm"
    if depth_scene is not None and depth_scene.shape[:2] == mask.shape[:2]:
        valid_depth = depth_scene[(mask > 0) & np.isfinite(depth_scene) & (depth_scene > 1e-6)]
        if valid_depth.size:
            z_scene = float(np.median(valid_depth))
            z_source = "foreground depth median"

    if z_scene is None:
        viewpoint = record.get("viewpoint", {})
        t = np.asarray(viewpoint.get("T", [0.0, 0.0, 1.0]), dtype=np.float32).reshape(3)
        z_scene = float(np.linalg.norm(t))
        if not np.isfinite(z_scene) or z_scene <= 1e-6:
            z_scene = abs(float(t[2])) if t.size >= 3 else 1.0

    ys, xs = np.nonzero(mask > 0)
    if len(xs) < 16:
        return 1.0, "fallback=1.0 (foreground mask too small)"
    bbox_w = max(float(xs.max() - xs.min() + 1), 1.0)
    bbox_h = max(float(ys.max() - ys.min() + 1), 1.0)
    fx = max(float(intrinsics[0, 0]), 1e-6)
    fy = max(float(intrinsics[1, 1]), 1e-6)
    extent_x_scene = bbox_w * z_scene / fx
    extent_y_scene = bbox_h * z_scene / fy
    visible_diag_scene = float(np.hypot(extent_x_scene, extent_y_scene))
    anchor_diag_m = float(np.linalg.norm(np.asarray(anchor_size_m, dtype=np.float32).reshape(3)[:2]))
    if visible_diag_scene <= 1e-6 or anchor_diag_m <= 1e-6:
        return 1.0, "fallback=1.0 (invalid mask projected size)"
    scale = float(np.clip(anchor_diag_m / visible_diag_scene, 1e-3, 10.0))
    reason = (
        f"mask projected extent scene=({extent_x_scene:.4f},{extent_y_scene:.4f}) "
        f"z_source={z_source} anchor_xy_diag_m={anchor_diag_m:.4f}"
    )
    return scale, reason


def resolve_co3d_scene_to_meter_scale(
    co3d_root: Path,
    category: str,
    scene_name: str,
    record: Dict,
    intrinsics: np.ndarray,
    image_size_hw: Tuple[int, int],
    anchor_size_m: np.ndarray,
) -> Tuple[float, str]:
    scale, reason = estimate_scale_from_sequence_pointcloud(co3d_root, category, scene_name, anchor_size_m)
    if scale is not None:
        return scale, f"sequence pointcloud: {reason}"
    fallback_scale, fallback_reason = estimate_scale_from_frame_mask(
        co3d_root,
        record,
        intrinsics,
        image_size_hw,
        anchor_size_m,
    )
    if reason:
        fallback_reason = f"{fallback_reason}; {reason}"
    return fallback_scale, f"frame mask fallback: {fallback_reason}"


# ============================================================
# OV9D anchor object reference (used as the model's object-query input)
# ============================================================
class OV9DAnchorIndex:
    """Resolves CO3D category -> OV9D anchor objects -> oo3d9dsingle render dirs."""

    def __init__(
        self,
        ov9d_root: Path,
        train_split_json: Path,
        object_views: Sequence[int],
        object_image_root: Path = DEFAULT_OBJECT_IMAGE_ROOT,
    ):
        self.ov9d_root = Path(ov9d_root)
        self.train_split_json = Path(train_split_json)
        self.object_image_root = Path(object_image_root)
        self.object_views = tuple(int(v) for v in object_views)

        self.class_list: List[str] = read_json(self.ov9d_root / "class_list.json")
        self.cid2oid: Dict[str, List[str]] = read_json(self.ov9d_root / "cid2oid.json")
        self.name2oid: Dict[str, int] = {
            str(k): int(v) for k, v in read_json(self.ov9d_root / "name2oid.json").items()
        }
        self.oid2name: Dict[int, str] = {v: k for k, v in self.name2oid.items()}
        self.models_info: Dict[str, Dict] = read_json(self.ov9d_root / "models_info.json")

        train_payload = read_json(self.train_split_json)
        self.anchor_oids: set = {int(x) for x in train_payload.get("anchor_object_ids", [])}
        self.anchor_oids.update(
            int(item["object_id"]) for item in train_payload.get("scenes", []) if "object_id" in item
        )

        # Index the OO9D around-image references by oid.
        self.single_dir_by_oid: Dict[int, Path] = self._index_oo3d9dsingle()

        # category name -> list of anchor oids that we can actually serve
        # (have an oo3d9dsingle dir with all required views).
        self.anchor_oids_by_category: Dict[str, List[int]] = {}
        for cid_str, oid_strs in self.cid2oid.items():
            cid_idx = int(cid_str) - 1
            if not (0 <= cid_idx < len(self.class_list)):
                continue
            category_name = self.class_list[cid_idx]
            category_anchors = sorted(
                int(o) for o in oid_strs
                if int(o) in self.anchor_oids and self._has_views_for_oid(int(o))
            )
            if category_anchors:
                self.anchor_oids_by_category[category_name] = category_anchors

    def _index_oo3d9dsingle(self) -> Dict[int, Path]:
        if self.object_image_root.is_dir():
            index = {}
            for object_dir in sorted(p for p in self.object_image_root.iterdir() if p.is_dir()):
                match = re.search(r"(\d+)$", object_dir.name)
                if match:
                    index.setdefault(int(match.group(1)), object_dir)
            return index

        single_root = self.ov9d_root / "oo3d9dsingle"
        index: Dict[int, Path] = {}
        if not single_root.is_dir():
            return index
        for scene_dir in sorted(p for p in single_root.iterdir() if p.is_dir()):
            parts = scene_dir.name.split("_")
            object_instance = "_".join(parts[:-1]) if len(parts) > 2 else scene_dir.name
            oid = self.name2oid.get(object_instance)
            if oid is None:
                continue
            index.setdefault(int(oid), scene_dir)
        return index

    def _has_views_for_oid(self, oid: int) -> bool:
        scene_dir = self.single_dir_by_oid.get(int(oid))
        if scene_dir is None:
            return False
        for view in self.object_views:
            rgb = scene_dir / "rgb" / f"{int(view):06d}.png"
            mask = scene_dir / "mask_visib" / f"{int(view):06d}_000000.png"
            if not rgb.is_file():
                return False
        return True

    def display_label(self, oid: int) -> str:
        return f"obj_{int(oid):06d} · {self.oid2name.get(int(oid), 'unknown')}"

    def size_m(self, oid: int) -> np.ndarray:
        info = self.models_info.get(str(int(oid)), {})
        return np.asarray(
            [info.get("size_x", 100.0), info.get("size_y", 100.0), info.get("size_z", 100.0)],
            dtype=np.float32,
        ) / 1000.0

    def axis_length_m(self, oid: int) -> float:
        return max(float(np.linalg.norm(self.size_m(oid))) * 0.25, 1e-3)


# ============================================================
# Scene preprocessing & tensor loaders
# ============================================================
class DemoScenePreprocessor(BaseStereoViewDataset):
    def __init__(self, resolution):
        super().__init__(dset="demo", resolution=resolution, transform=ImgNorm, seed=0)


def load_co3d_scene_inputs(
    co3d_root: Path,
    category: str,
    scene_name: str,
    image_id: int,
    resolution,
    device: torch.device,
    use_depth: bool,
    depth_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, np.ndarray, np.ndarray | None, np.ndarray]:
    """Returns (image_tensor, depth_tensor, mask_tensor, display_image, display_depth, intrinsic).

    ``depth_scale`` converts CO3D scene units into metres (the unit the model
    was trained on). It is resolved automatically before this function is called
    and is applied to both model input depth and visualisation point cloud.
    """
    annotations = co3d_load_frame_annotations(str(co3d_root / category))
    record = annotations.get((scene_name, int(image_id)))
    if record is None:
        raise FileNotFoundError(
            f"Missing frame_annotations entry for ({category}, {scene_name}, frame {image_id})"
        )

    image_path = co3d_root / record["image"]["path"]
    image = Image.open(image_path).convert("RGB")

    image_size_hw = tuple(int(v) for v in record["image"]["size"])  # (H, W)
    intrinsics = co3d_ndc_isotropic_to_pixel_intrinsics(record["viewpoint"], image_size_hw)

    if use_depth and "depth" in record:
        depth_path = co3d_root / record["depth"]["path"]
        depthmap = co3d_load_depth_scene_units(
            depth_path, float(record["depth"].get("scale_adjustment", 1.0)),
        )
        mask_rel = record["depth"].get("mask_path")
        validity = co3d_load_depth_validity(
            co3d_root / mask_rel if mask_rel else None,
            shape_hw=depthmap.shape,
        )
        if validity is not None:
            depthmap = depthmap * validity
        depthmap = depthmap * float(depth_scale)  # scene units → metres
    else:
        depthmap = np.zeros((image_size_hw[0], image_size_hw[1]), dtype=np.float32)

    processor = DemoScenePreprocessor(resolution=resolution)
    image, depthmap, intrinsics = processor._crop_resize_if_necessary(
        image=image,
        depthmap=depthmap,
        intrinsics=intrinsics.copy(),
        resolution=resolution,
        rng=np.random.default_rng(seed=0),
        info=str(image_path),
    )

    image_tensor = processor.transform(image).unsqueeze(0).unsqueeze(0).to(device)
    display_image = np.asarray(image.convert("RGB"))

    depth_tensor = None
    mask_tensor = None
    display_depth = None
    if use_depth:
        depth_arr = depthmap.astype(np.float32)
        display_depth = depth_arr
        depth_tensor = torch.from_numpy(np.ascontiguousarray(depth_arr))[None, None, :, :, None].to(device)
        mask_tensor = torch.from_numpy((depth_arr > 0).astype(np.float32))[None, None, :, :].to(device)

    return image_tensor, depth_tensor, mask_tensor, display_image, display_depth, intrinsics


def load_anchor_object_tensor(
    anchor_index: OV9DAnchorIndex,
    oid: int,
    resolution,
    device: torch.device,
) -> Tuple[torch.Tensor, List[Tuple[np.ndarray, str]]]:
    scene_dir = anchor_index.single_dir_by_oid.get(int(oid))
    if scene_dir is None:
        raise FileNotFoundError(f"No oo3d9dsingle dir for OV9D oid {oid}")
    processor = DemoScenePreprocessor(resolution=resolution)
    resampling = getattr(Image, "Resampling", Image)
    tensors: List[torch.Tensor] = []
    gallery: List[Tuple[np.ndarray, str]] = []
    for view in anchor_index.object_views:
        rgb_path = scene_dir / "rgb" / f"{int(view):06d}.png"
        mask_path = scene_dir / "mask_visib" / f"{int(view):06d}_000000.png"
        if not rgb_path.is_file():
            raise FileNotFoundError(f"Missing anchor object view: {rgb_path}")
        rgb_arr = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
        if mask_path.is_file():
            mask_arr = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
            white_bg = np.full_like(rgb_arr, 255)
            white_bg[mask_arr > 0] = rgb_arr[mask_arr > 0]
            image = Image.fromarray(white_bg, mode="RGB")
        else:
            image = Image.fromarray(rgb_arr, mode="RGB")
        image = image.resize(tuple(resolution), resampling.LANCZOS)
        tensors.append(processor.transform(image).to(device))
        gallery.append((np.asarray(image), f"Anchor view {int(view)}"))
    return torch.stack(tensors, dim=0).unsqueeze(0), gallery


def co3d_load_scene_object_views_tensor(
    co3d_root: Path,
    category: str,
    scene_name: str,
    frame_numbers: Sequence[int],
    resolution,
    device: torch.device,
) -> Tuple[torch.Tensor, List[Tuple[np.ndarray, str]]]:
    """Use the CO3D scene's own frames as the object reference, with backgrounds
    removed via the per-frame ``mask`` annotation.
    """
    annotations = co3d_load_frame_annotations(str(co3d_root / category))
    processor = DemoScenePreprocessor(resolution=resolution)
    resampling = getattr(Image, "Resampling", Image)
    tensors: List[torch.Tensor] = []
    gallery: List[Tuple[np.ndarray, str]] = []
    for fnum in frame_numbers:
        record = annotations.get((scene_name, int(fnum)))
        if record is None or "mask" not in record:
            raise FileNotFoundError(
                f"CO3D scene {category}/{scene_name} has no mask annotation for frame {fnum}"
            )
        rgb_arr = np.asarray(
            Image.open(co3d_root / record["image"]["path"]).convert("RGB"), dtype=np.uint8,
        )
        mask_arr = np.asarray(
            Image.open(co3d_root / record["mask"]["path"]).convert("L"), dtype=np.uint8,
        )
        if mask_arr.shape != rgb_arr.shape[:2]:
            mask_arr = cv2.resize(
                mask_arr, (rgb_arr.shape[1], rgb_arr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        white_bg = np.full_like(rgb_arr, 255)
        white_bg[mask_arr > 0] = rgb_arr[mask_arr > 0]
        image = Image.fromarray(white_bg, mode="RGB").resize(tuple(resolution), resampling.LANCZOS)
        tensors.append(processor.transform(image).to(device))
        gallery.append((np.asarray(image), f"CO3D frame {int(fnum)}"))
    return torch.stack(tensors, dim=0).unsqueeze(0), gallery


# ============================================================
# 2D visualization helpers
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


# ============================================================
# 3D point cloud + predicted pose export (.glb)
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


def export_pred_pose_pcd_glb(
    category: str,
    scene_name: str,
    frame_name: str,
    rgb_image: np.ndarray,
    depthmap: np.ndarray | None,
    intrinsic: np.ndarray,
    pred_rotation_cam: np.ndarray,
    pred_translation_cam: np.ndarray,
    pred_bbox_obj: np.ndarray,
    axis_length: float,
    point_cloud_stride: int = 2,
) -> str:
    stride = max(1, int(point_cloud_stride))
    max_points = 80000 if stride <= 2 else 50000

    scene_3d = trimesh.Scene()
    if depthmap is not None:
        points, colors = depth_to_camera_points(
            depthmap, intrinsic, rgb_image, stride=stride, max_points=max_points,
        )
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

    center = np.asarray(pred_translation_cam, dtype=np.float32)
    axis_pts = _axis_object_points(viewer_axis_length) @ pred_rotation_cam.T + pred_translation_cam[None, :]
    for idx, axis_color in enumerate(PRED_AXIS_COLORS):
        add_segment(center, axis_pts[idx + 1], axis_color)
    bbox_cam = np.asarray(pred_bbox_obj, dtype=np.float32) @ pred_rotation_cam.T + pred_translation_cam[None, :]
    for start_idx, end_idx in BBOX_EDGES:
        add_segment(bbox_cam[start_idx], bbox_cam[end_idx], PRED_BBOX_COLOR)

    safe_cat = re.sub(r"[^a-zA-Z0-9_]+", "_", category)[:60]
    safe_scene = re.sub(r"[^a-zA-Z0-9_]+", "_", scene_name)[:120]
    safe_frame = re.sub(r"[^a-zA-Z0-9_]+", "_", str(frame_name))[:120]
    out_dir = _LOCAL_TMP / "co3d_pred_pose_glb" / safe_cat / safe_scene
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe_frame}_stride{stride}.glb"
    scene_3d.export(out_path)
    return str(out_path)


# ============================================================
# DemoApp: CO3D scene + OV9D anchor reference
# ============================================================
class DemoApp:
    def __init__(
        self,
        config_path: Path,
        checkpoint_path: str | None,
        co3d_root: Path,
        ov9d_root: Path,
        train_split_json: Path,
    ):
        self.cfg = load_config(config_path)
        self.resolution = resolve_resolution(self.cfg)
        self.object_views = FIXED_OBJECT_VIEWS

        self.co3d_root = Path(co3d_root)
        self.anchor_index = OV9DAnchorIndex(
            ov9d_root=ov9d_root,
            train_split_json=train_split_json,
            object_views=self.object_views,
        )

        self.category_choices = self._intersect_categories()
        if not self.category_choices:
            raise RuntimeError(
                f"No CO3D categories under {self.co3d_root} have matching anchor objects "
                f"in {train_split_json}"
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = resolve_checkpoint_path(self.cfg, checkpoint_path)
        self.model = None
        self.available_checkpoints = self.discover_checkpoints()
        self.load_checkpoint(self.checkpoint_path)

        print(
            f"[demo] CO3D demo ready: categories={len(self.category_choices)} "
            f"anchor_categories={len(self.anchor_index.anchor_oids_by_category)} "
            f"object_views={self.object_views} resolution={self.resolution}"
        )

    # ----- Category / scene / frame / anchor choices -----
    def _intersect_categories(self) -> List[str]:
        co3d_cats = co3d_list_categories(self.co3d_root)
        with_anchors = set(self.anchor_index.anchor_oids_by_category.keys())
        return [c for c in co3d_cats if c in with_anchors]

    def list_scenes(self, category: str) -> List[str]:
        return co3d_list_scenes(self.co3d_root, category) if category else []

    def list_frames(self, category: str, scene_name: str) -> List[str]:
        return co3d_list_frames(self.co3d_root, category, scene_name) if category and scene_name else []

    def list_anchor_choices(self, category: str) -> List[Tuple[str, str]]:
        if not category:
            return []
        oids = self.anchor_index.anchor_oids_by_category.get(category, [])
        return [(self.anchor_index.display_label(oid), str(int(oid))) for oid in oids]

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

    # ----- Input gallery preview -----
    def input_gallery(
        self,
        category: str,
        scene_name: str,
        frame_name: str,
        anchor_oid_str: str,
        object_source: str = OBJECT_SOURCE_ANCHOR,
    ):
        if not (category and scene_name and frame_name and anchor_oid_str):
            return None, []
        scene_image = str(
            self.co3d_root / category / scene_name / "images" / f"frame{int(frame_name):06d}.jpg"
        )
        if object_source == OBJECT_SOURCE_CO3D_SCENE:
            scene_frames = co3d_list_frames(self.co3d_root, category, scene_name)
            if len(scene_frames) < CO3D_OBJECT_SOURCE_MIN_FRAMES:
                return scene_image, []
            _, gallery = co3d_load_scene_object_views_tensor(
                self.co3d_root, category, scene_name, CO3D_OBJECT_FRAMES,
                self.resolution, torch.device("cpu"),
            )
        else:
            _, gallery = load_anchor_object_tensor(
                self.anchor_index, int(anchor_oid_str), self.resolution, torch.device("cpu"),
            )
        return scene_image, gallery

    # ----- Inference + visualization (predicted bbox + axis only) -----
    def run_inference(
        self,
        category: str,
        scene_name: str,
        frame_name: str,
        anchor_oid_str: str,
        object_source: str,
        use_depth_input: bool,
        show_point_cloud: bool,
        point_cloud_stride: int,
    ):
        if not (category and scene_name and frame_name and anchor_oid_str):
            return "Please select a category, scene, frame, and anchor object.", None, None, None
        anchor_oid = int(anchor_oid_str)
        image_id = int(frame_name)
        use_depth_input = bool(use_depth_input)
        anchor_size = self.anchor_index.size_m(anchor_oid)

        if object_source == OBJECT_SOURCE_CO3D_SCENE:
            scene_frames = co3d_list_frames(self.co3d_root, category, scene_name)
            if len(scene_frames) < CO3D_OBJECT_SOURCE_MIN_FRAMES:
                return (
                    f"CO3D scene `{category}/{scene_name}` has only {len(scene_frames)} frame(s); "
                    f"the `co3d_scene` object source needs at least {CO3D_OBJECT_SOURCE_MIN_FRAMES}.",
                    None, None, None,
                )

        annotations = co3d_load_frame_annotations(str(self.co3d_root / category))
        record = annotations.get((scene_name, image_id))
        if record is None:
            raise FileNotFoundError(
                f"Missing frame_annotations entry for ({category}, {scene_name}, frame {image_id})"
            )
        image_size_hw = tuple(int(v) for v in record["image"]["size"])
        raw_intrinsic = co3d_ndc_isotropic_to_pixel_intrinsics(record["viewpoint"], image_size_hw)
        depth_scale, scale_reason = resolve_co3d_scene_to_meter_scale(
            self.co3d_root,
            category,
            scene_name,
            record,
            raw_intrinsic,
            image_size_hw,
            anchor_size,
        )

        # ---- Inputs ----
        (scene_tensor, depth_tensor, mask_tensor,
         display_image, display_depth, intrinsic) = load_co3d_scene_inputs(
            self.co3d_root, category, scene_name, image_id,
            self.resolution, self.device,
            use_depth=use_depth_input, depth_scale=depth_scale,
        )
        if object_source == OBJECT_SOURCE_CO3D_SCENE:
            object_tensor, _ = co3d_load_scene_object_views_tensor(
                self.co3d_root, category, scene_name, CO3D_OBJECT_FRAMES,
                self.resolution, self.device,
            )
            object_source_label = f"CO3D scene frames {list(CO3D_OBJECT_FRAMES)} (bg removed)"
        else:
            object_tensor, _ = load_anchor_object_tensor(
                self.anchor_index, anchor_oid, self.resolution, self.device,
            )
            object_source_label = f"OV9D anchor views {list(self.object_views)}"

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

        if "object_pose" not in outputs or "object_translation" not in outputs:
            raise RuntimeError(f"Model output missing pose keys: {sorted(outputs.keys())}")

        # ---- Decode predicted pose ----
        pred_rot6d_cam = outputs["object_pose"][0].detach().float().cpu().numpy()
        pred_translation_cam = outputs["object_translation"][0].detach().float().cpu().numpy().astype(np.float32)
        pred_rotation_cam = rot6d_to_matrix(pred_rot6d_cam).astype(np.float32)

        # ---- Decode predicted size; fall back to anchor models_info size ----
        pred_size = None
        if "object_size" in outputs:
            pred_size = outputs["object_size"].reshape(-1, 3)[0].detach().float().cpu().numpy().astype(np.float32)
        elif "object_size_log" in outputs:
            pred_size = np.exp(
                outputs["object_size_log"].reshape(-1, 3)[0].detach().float().cpu().numpy()
            ).astype(np.float32)
        pred_size_is_usable = (
            pred_size is not None
            and np.all(np.isfinite(pred_size))
            and np.all(pred_size > 0)
        )
        size_for_box = pred_size if pred_size_is_usable else anchor_size
        size_source = "model" if pred_size_is_usable else "anchor models_info"
        bbox_obj = centered_axis_bbox_corners(size_for_box)
        axis_length = self.anchor_index.axis_length_m(anchor_oid)

        # ---- Optional presence / pred mask outputs ----
        presence_prob = None
        presence_logit = None
        if "object_presence_logits" in outputs:
            presence_logit = float(outputs["object_presence_logits"].reshape(-1)[0].detach().float().cpu())
            presence_prob = float(torch.sigmoid(torch.tensor(presence_logit)).item())

        pred_mask_image = None
        if "object_mask" in outputs:
            pred_mask = outputs["object_mask"][0, 0].detach().float().cpu().numpy()
            pred_mask_image = mask_overlay_image(display_image, pred_mask, color=(255, 64, 64))

        # ---- 2D overlay ----
        pred_image = draw_bbox_axes_overlay_on_image(
            display_image, intrinsic, pred_rotation_cam, pred_translation_cam,
            bbox_obj, axis_length, PRED_AXIS_COLORS, PRED_BBOX_COLOR,
        )

        # ---- Optional 3D point cloud export ----
        point_cloud_glb = None
        if show_point_cloud:
            point_cloud_glb = export_pred_pose_pcd_glb(
                category, scene_name, f"{image_id:06d}",
                display_image,
                display_depth if use_depth_input else None,
                intrinsic,
                pred_rotation_cam, pred_translation_cam, bbox_obj, axis_length,
                point_cloud_stride=point_cloud_stride,
            )

        # ---- Markdown summary ----
        lines = [
            "### Prediction Summary",
            f"- checkpoint: `{self.checkpoint_path}`",
            f"- CO3D scene: `{category}/{scene_name}` frame `{image_id:06d}`",
            f"- anchor object (size source): `{self.anchor_index.display_label(anchor_oid)}`",
            f"- object image source: `{object_source_label}`",
            f"- use depth input: `{use_depth_input}`"
            + (f" · auto depth scale (scene units → m): `{depth_scale:.6f}`" if use_depth_input else ""),
            f"- scale source: `{scale_reason}`",
        ]
        if presence_prob is not None:
            lines.append(
                f"- predicted presence probability: `{presence_prob:.6f}` "
                f"(logit `{presence_logit:.6f}`, @0.5 → `{bool(presence_prob >= 0.5)}`)"
            )
        lines += [
            "",
            "### Predicted Pose (camera frame)",
            f"- translation (m): `{np.round(pred_translation_cam, 6).tolist()}`",
            f"- rotation matrix: `{np.round(pred_rotation_cam, 6).tolist()}`",
            f"- bbox size xyz (m): `{np.round(size_for_box, 6).tolist()}` (from {size_source})",
        ]

        return "\n".join(lines), pred_image, point_cloud_glb, pred_mask_image


# ============================================================
# Gradio UI
# ============================================================
def build_demo(app: DemoApp, image_focused_layout: bool = False):
    default_category = app.category_choices[0]
    default_scenes = app.list_scenes(default_category)
    default_scene = default_scenes[0] if default_scenes else None
    default_frames = app.list_frames(default_category, default_scene) if default_scene else []
    default_frame = default_frames[0] if default_frames else None
    default_anchors = app.list_anchor_choices(default_category)
    default_anchor = default_anchors[0][1] if default_anchors else None

    demo_css = """
    .gradio-container { max-width: 100% !important; }
    #scene_inputs, #pred_projection { background: #1c1c1c; }
    #scene_inputs img, #pred_projection img,
    #scene_inputs button img, #pred_projection button img {
        object-fit: contain !important;
        max-height: 100% !important;
        max-width: 100% !important;
        width: auto !important;
        height: auto !important;
    }
    #scene_inputs > div, #pred_projection > div { overflow: hidden !important; }
    #object_inputs img { object-fit: contain !important; background: #1c1c1c; }
    """
    if image_focused_layout:
        demo_css += """
        .gradio-container { padding: 8px 10px !important; }
        #demo_info { margin-bottom: 6px !important; }
        #demo_info p { margin: 0 !important; }
        #main_row { gap: 12px !important; }
        #inputs_col, #results_col { min-height: 80vh !important; }
        #scene_inputs, #pred_projection { min-height: 38vh !important; }
        #object_inputs { min-height: 22vh !important; }
        """

    info_markdown = "\n".join(
        [
            "# OmniVGGT 6D Pose Demo (CO3D, single-view)",
            f"- config: `{DEFAULT_CONFIG_PATH}`",
            f"- CO3D root: `{app.co3d_root}`",
            f"- OV9D anchor source: `{app.anchor_index.ov9d_root}`",
            f"- object references: `{app.anchor_index.object_image_root}`",
            f"- train split JSON: `{app.anchor_index.train_split_json}`",
            f"- anchor categories available: `{len(app.category_choices)}`",
            f"- default checkpoint: `{DEFAULT_PRETRAIN_MODEL}`",
            f"- loaded checkpoint: `{app.checkpoint_path}`",
            f"- anchor reference views (fixed): `{list(app.object_views)}`",
            f"- inference resolution: `{app.resolution}`",
            "",
            "選擇 `category` → `scene` → `frame` → `anchor object`,模型用 CO3D 的單張 RGB"
            " (可選 depth) 配上同類別 OV9D anchor 物件的固定 4 個 view 做 6D pose 預測,"
            "然後把預測的 bbox + axis 畫在 CO3D 影像上。若使用 depth,會自動把 CO3D scene-unit"
            " 對齊到 OV9D 訓練用的 meter 尺度。沒有 GT pose,所以不算 error。",
        ]
    )

    with gr.Blocks(title="OmniVGGT 6D Pose Demo (CO3D)", css=demo_css) as demo:
        if image_focused_layout:
            with gr.Accordion("Demo Info", open=False, elem_id="demo_info"):
                gr.Markdown(info_markdown)
        else:
            gr.Markdown(info_markdown, elem_id="demo_info")

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
            category_dropdown = gr.Dropdown(choices=app.category_choices, value=default_category, label="Category")
            scene_dropdown = gr.Dropdown(choices=default_scenes, value=default_scene, label="Scene")
            frame_dropdown = gr.Dropdown(choices=default_frames, value=default_frame, label="Frame")
            anchor_dropdown = gr.Dropdown(choices=default_anchors, value=default_anchor, label="Anchor Object")
            object_source_dropdown = gr.Dropdown(
                choices=[
                    (f"OV9D anchor (views {list(app.object_views)})", OBJECT_SOURCE_ANCHOR),
                    (
                        f"CO3D scene frames {list(CO3D_OBJECT_FRAMES)} "
                        f"(bg removed, needs ≥{CO3D_OBJECT_SOURCE_MIN_FRAMES} frames)",
                        OBJECT_SOURCE_CO3D_SCENE,
                    ),
                ],
                value=OBJECT_SOURCE_ANCHOR,
                label="Object Image Source",
            )
            use_depth_checkbox = gr.Checkbox(value=False, label="Use Depth Input")
            show_point_cloud_checkbox = gr.Checkbox(value=False, label="Show Point Cloud Pose")
            point_cloud_stride_slider = gr.Slider(
                minimum=1, maximum=8, value=2, step=1,
                label="Point Cloud Density", info="Smaller = denser RGB point cloud",
            )
            infer_button = gr.Button("Run Inference", variant="primary")

        big_height = "38vh" if image_focused_layout else 360
        thumb_height = "18vh" if image_focused_layout else 180

        with gr.Row(elem_id="main_row", equal_height=False):
            with gr.Column(scale=1, elem_id="inputs_col"):
                gr.Markdown("#### Inputs")
                scene_input_image = compat_image(
                    label="CO3D Scene (RGB)", height=big_height, elem_id="scene_inputs",
                    interactive=False,
                )
                object_input_gallery = gr.Gallery(
                    label=f"Anchor object views {list(app.object_views)}",
                    columns=max(len(app.object_views), 1), height=thumb_height,
                    elem_id="object_inputs", preview=False, object_fit="contain",
                    show_label=True, allow_preview=True,
                )
            with gr.Column(scale=1, elem_id="results_col"):
                gr.Markdown("#### Prediction (bbox + axis overlay)")
                pred_image = compat_image(
                    label="Predicted axes + bbox (camera frame)", height=big_height,
                    elem_id="pred_projection", interactive=False,
                    show_download_button=False, container=True,
                )
                point_cloud_model = gr.Model3D(
                    label="Depth Point Cloud + Predicted Pose",
                    height=520, zoom_speed=0.6, pan_speed=0.6,
                    display_mode="solid", clear_color=(0.0, 0.0, 0.0, 0.0),
                )
                pred_mask_image = compat_image(
                    label="Predicted object mask", height=thumb_height, interactive=False,
                    show_download_button=False, container=True,
                )

        summary_markdown = gr.Markdown()

        # ---- Event handlers ----
        def refresh_inputs(category, scene_name, frame_name, anchor_oid_str, object_source):
            return app.input_gallery(category, scene_name, frame_name, anchor_oid_str, object_source)

        def refresh_category_controls(category: str, object_source: str):
            scenes = app.list_scenes(category)
            scene_value = scenes[0] if scenes else None
            frames = app.list_frames(category, scene_value) if scene_value else []
            frame_value = frames[0] if frames else None
            anchors = app.list_anchor_choices(category)
            anchor_value = anchors[0][1] if anchors else None
            scene_image, gallery = refresh_inputs(category, scene_value, frame_value, anchor_value, object_source)
            return (
                gr.update(choices=scenes, value=scene_value),
                gr.update(choices=frames, value=frame_value),
                gr.update(choices=anchors, value=anchor_value),
                scene_image,
                gallery,
            )

        def refresh_scene_controls(category: str, scene_name: str):
            frames = app.list_frames(category, scene_name)
            frame_value = frames[0] if frames else None
            return gr.update(choices=frames, value=frame_value)

        def sync_checkpoint_path(selected_value: str):
            return selected_value

        category_dropdown.change(
            refresh_category_controls,
            inputs=[category_dropdown, object_source_dropdown],
            outputs=[scene_dropdown, frame_dropdown, anchor_dropdown,
                     scene_input_image, object_input_gallery],
        )
        scene_dropdown.change(
            refresh_scene_controls,
            inputs=[category_dropdown, scene_dropdown],
            outputs=frame_dropdown,
        )
        for trigger in (scene_dropdown, frame_dropdown, anchor_dropdown, object_source_dropdown):
            trigger.change(
                refresh_inputs,
                inputs=[category_dropdown, scene_dropdown, frame_dropdown,
                        anchor_dropdown, object_source_dropdown],
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
                category_dropdown,
                scene_dropdown,
                frame_dropdown,
                anchor_dropdown,
                object_source_dropdown,
                use_depth_checkbox,
                show_point_cloud_checkbox,
                point_cloud_stride_slider,
            ],
            outputs=[summary_markdown, pred_image, point_cloud_model, pred_mask_image],
        )

        if default_frame is not None and default_anchor is not None:
            demo.load(
                refresh_inputs,
                inputs=[category_dropdown, scene_dropdown, frame_dropdown,
                        anchor_dropdown, object_source_dropdown],
                outputs=[scene_input_image, object_input_gallery],
            )
    return demo


# ============================================================
# Entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Gradio demo for OmniVGGT 6D pose inference on CO3D")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--co3d-root", type=Path, default=DEFAULT_CO3D_ROOT)
    parser.add_argument("--ov9d-root", type=Path, default=DEFAULT_OV9D_ROOT)
    parser.add_argument("--train-split-json", type=Path, default=DEFAULT_TRAIN_SPLIT_JSON)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--image-focused-layout", action="store_true",
        help="Use a comparison-oriented layout that gives most of the page to images.",
    )
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    app = DemoApp(
        args.config,
        args.checkpoint,
        co3d_root=args.co3d_root,
        ov9d_root=args.ov9d_root,
        train_split_json=args.train_split_json,
    )
    demo = build_demo(app, image_focused_layout=args.image_focused_layout)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[str(app.co3d_root), str(app.anchor_index.ov9d_root)],
    )


if __name__ == "__main__":
    main()
