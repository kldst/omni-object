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
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import argparse
import inspect
import json
import pickle
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
from omnivggt.datasets.housecat6d.housecat6d_camera_pose import HouseCat6DCameraPose
from omnivggt.datasets.real275.real275_camera_pose import Real275CameraPose
from omnivggt.datasets.ycbv.ycbv_camera_pose import YCBVCameraPose
from omnivggt.loss import _load_symmetry_info
from omnivggt.models.omnivggt import OmniVGGT
from omnivggt.utils.image import ImgNorm


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train_oo9d_real275_ycbv_hc.py"
DEFAULT_YCBV_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose/datasets_real/ycbv")
DEFAULT_YCBV_TEST_ROOT = DEFAULT_YCBV_ROOT / "test"
DEFAULT_YCBV_OBJECT_IMAGE_ROOT = DEFAULT_YCBV_ROOT / "ycbv_aligned_object_refs"
DEFAULT_YCBV_MODELS_INFO = DEFAULT_YCBV_ROOT / "models" / "models_info.json"
DEFAULT_REAL275_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose/real275")
DEFAULT_REAL275_TEST_ROOT = DEFAULT_REAL275_ROOT / "real_test"
DEFAULT_REAL275_GT_ROOT = DEFAULT_REAL275_ROOT / "gts" / "real_test"
DEFAULT_REAL275_OBJ_MODELS_ROOT = DEFAULT_REAL275_ROOT / "obj_models" / "real_test"
DEFAULT_REAL275_OBJECT_IMAGE_ROOT = DEFAULT_REAL275_ROOT / "real275_aligned_object_refs"
DEFAULT_HOUSECAT6D_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose/housecat6d")
DEFAULT_HOUSECAT6D_OBJECT_IMAGE_ROOT = DEFAULT_HOUSECAT6D_ROOT / "housecat6d_aligned_object_refs"
DEFAULT_HOUSECAT6D_SCENE_NAMES: Tuple[str, ...] = (
    *(f"scene{idx:02d}" for idx in range(1, 35)),
    "test_scene1",
    "test_scene2",
    "test_scene3",
    "test_scene4",
    "test_scene5",
    "val_scene1",
    "val_scene2",
)
DEFAULT_ALIGN_JSON = PROJECT_ROOT / "dataset_align.json"
DEFAULT_PRETRAIN_MODEL = Path(
    "/mnt/train-data-4-hdd/yian/freepose/omni-object_clone/outputs/0525_REAL/model.safetensors"
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
YCBV_DATASET_LABEL = "YCBVCameraPose"
REAL275_DATASET_LABEL = "Real275CameraPose"
HOUSECAT6D_DATASET_LABEL = "HouseCat6DCameraPose"
REAL275_DATASET_KEY = "real275_test"
YCBV_DATASET_KEY = "ycbv_test"
HOUSECAT6D_DATASET_KEY = "housecat6d_test"
REAL275_CAMERA_K = np.asarray(
    [[591.0125, 0.0, 322.525], [0.0, 590.16775, 244.11084], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)
REAL275_DEPTH_SCALE = 1000.0


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
            default=cfg.get("fixed_object_view_ids", (1, 5, 10, 15)),
        )
    )
    resolution = tuple(int(v) for v in cfg.get("resolution", (518, 476)))
    return {
        "object_input_views": tuple(sorted(int(v) for v in object_input_views)),
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
    dataset_label: str = YCBV_DATASET_LABEL,
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


def ycbv_object_key(object_id: int) -> str:
    return f"obj_{int(object_id):06d}"


def ycbv_object_id_from_key(value) -> int:
    match = re.search(r"(\d+)$", str(value))
    if not match:
        raise ValueError(f"Could not parse YCB-V object id from: {value}")
    return int(match.group(1))


def ycbv_object_display_name(object_id: int, oid_to_category: Dict[int, str]) -> str:
    object_id = int(object_id)
    category = oid_to_category.get(object_id, "unknown")
    return f"{ycbv_object_key(object_id)} · {category}"


def ycbv_read_depth_m(depth_path: Path, camera_entry: Dict) -> np.ndarray:
    depth_raw = np.asarray(Image.open(depth_path), dtype=np.float32)
    depth_m = depth_raw * float(camera_entry.get("depth_scale", 1.0)) / 1000.0
    depth_m[~np.isfinite(depth_m)] = 0.0
    depth_m[depth_m < 0.0] = 0.0
    return depth_m.astype(np.float32)


def ycbv_read_binary_mask(mask_path: Path) -> np.ndarray:
    return (np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0).astype(np.float32)


def real275_read_depth_m(depth_path: Path) -> np.ndarray:
    depth_raw = np.asarray(Image.open(depth_path), dtype=np.float32)
    depth_m = depth_raw / float(REAL275_DEPTH_SCALE)
    depth_m[~np.isfinite(depth_m)] = 0.0
    depth_m[depth_m < 0.0] = 0.0
    return depth_m.astype(np.float32)


def real275_read_instance_mask(mask_path: Path, inst_id: int) -> np.ndarray | None:
    if not Path(mask_path).is_file():
        return None
    mask_arr = np.asarray(Image.open(mask_path), dtype=np.uint8)
    return (mask_arr == int(inst_id)).astype(np.float32)


def real275_read_meta(meta_path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if not Path(meta_path).is_file():
        return entries
    for line in Path(meta_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        try:
            inst_id = int(parts[0])
            class_id = int(parts[1])
        except ValueError:
            continue
        entries.append(
            {
                "inst_id": inst_id,
                "class_id": class_id,
                "model_name": parts[2],
            }
        )
    return entries


def real275_decompose_gt_rt(rt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    rt = np.asarray(rt, dtype=np.float64).reshape(4, 4)
    sr = rt[:3, :3]
    det = float(np.linalg.det(sr))
    scale = float(np.cbrt(abs(det))) if abs(det) > 1e-20 else 1.0
    if scale < 1e-12:
        scale = 1.0
    rotation = (sr / scale).astype(np.float32)
    if np.linalg.det(rotation) < 0:
        rotation = -rotation
        scale = -scale
    translation = rt[:3, 3].astype(np.float32)
    return rotation, translation, float(scale)


def real275_read_canonical_extent(obj_models_root: Path, model_name: str) -> np.ndarray | None:
    txt_path = Path(obj_models_root) / f"{model_name}.txt"
    if not txt_path.is_file():
        return None
    values = []
    for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line))
        except ValueError:
            return None
    if len(values) != 3:
        return None
    return np.asarray(values, dtype=np.float32)


def housecat6d_read_depth_m(depth_path: Path) -> np.ndarray:
    depth_raw = np.asarray(Image.open(depth_path), dtype=np.float32)
    depth_m = depth_raw / 1000.0
    depth_m[~np.isfinite(depth_m)] = 0.0
    depth_m[depth_m < 0.0] = 0.0
    return depth_m.astype(np.float32)


def housecat6d_read_intrinsics(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float32).reshape(3, 3)


def housecat6d_read_label(label_path: Path) -> Dict[str, Any]:
    with Path(label_path).open("rb") as handle:
        return pickle.load(handle)


def housecat6d_read_meta(meta_path: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    if not Path(meta_path).is_file():
        return records
    for line in Path(meta_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split(maxsplit=2)
        if len(parts) < 3:
            continue
        try:
            inst_id = int(parts[0])
            class_id = int(parts[1])
        except ValueError:
            continue
        records[parts[2]] = {
            "instance_id": inst_id,
            "class_id": class_id,
            "model_name": parts[2],
        }
    return records


def housecat6d_read_binary_mask(mask_path: Path) -> np.ndarray | None:
    if not Path(mask_path).is_file():
        return None
    return (np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0).astype(np.float32)


class DemoScenePreprocessor(BaseStereoViewDataset):
    def __init__(self, resolution):
        super().__init__(dset="ycbv_demo", resolution=resolution, transform=ImgNorm, seed=0)


def crop_resize_image_depth_mask(
    processor: DemoScenePreprocessor,
    image: Image.Image,
    depthmap: np.ndarray,
    object_mask: np.ndarray | None,
    intrinsics: np.ndarray,
    resolution,
    info: str,
    dataset_key: str = YCBV_DATASET_KEY,
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

    if dataset_key == REAL275_DATASET_KEY:
        crop_fn = Real275CameraPose._crop_resize_if_necessary_with_mask
    elif dataset_key == HOUSECAT6D_DATASET_KEY:
        crop_fn = HouseCat6DCameraPose._crop_resize_if_necessary_with_mask
    else:
        crop_fn = YCBVCameraPose._crop_resize_if_necessary_with_mask
    return crop_fn(
        processor,
        image=image,
        depthmap=depthmap,
        object_mask=object_mask,
        intrinsics=intrinsics.copy(),
        resolution=resolution,
        rng=np.random.default_rng(seed=0),
        info=info,
    )


def target_crop_resize_image_depth_mask(
    processor: DemoScenePreprocessor,
    image: Image.Image,
    depthmap: np.ndarray,
    object_mask: np.ndarray | None,
    intrinsics: np.ndarray,
    resolution,
    info: str,
    margin: float = 0.45,
    dataset_key: str = YCBV_DATASET_KEY,
):
    if object_mask is None or not np.any(object_mask > 0):
        return crop_resize_image_depth_mask(
            processor, image, depthmap, object_mask, intrinsics, resolution, info, dataset_key=dataset_key
        )

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


def load_ycbv_object_tensor(
    object_records_by_id: Dict[int, Dict[str, Any]],
    object_id: int,
    object_views: Sequence[int],
    resolution,
    device,
) -> Tuple[torch.Tensor, List[Tuple[np.ndarray, str]]]:
    object_id = int(object_id)
    if object_id not in object_records_by_id:
        raise FileNotFoundError(f"No YCB-V aligned reference renders for object id {object_id}")
    object_rec = object_records_by_id[object_id]
    object_dir: Path = object_rec["object_dir"]
    processor = DemoScenePreprocessor(resolution=resolution)
    resampling = getattr(Image, "Resampling", Image)
    tensors: List[torch.Tensor] = []
    gallery: List[Tuple[np.ndarray, str]] = []
    for image_id in object_views:
        image_path = object_dir / "rgb" / f"{int(image_id):06d}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing YCB-V object reference view: {image_path}")
        image = Image.open(image_path).convert("RGB")
        image = image.resize(tuple(resolution), resampling.LANCZOS)
        tensors.append(processor.transform(image).to(device))
        gallery.append((np.asarray(image), f"Object view {int(image_id):06d}"))
    return torch.stack(tensors, dim=0).unsqueeze(0), gallery


def load_ycbv_scene_frame_inputs(
    scene_dir: Path,
    image_id: int,
    object_index: int,
    resolution,
    device,
    target_crop: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    camera_entry = read_json(scene_dir / "scene_camera.json")[str(int(image_id))]
    image = Image.open(scene_dir / "rgb" / f"{int(image_id):06d}.png").convert("RGB")
    depthmap = ycbv_read_depth_m(scene_dir / "depth" / f"{int(image_id):06d}.png", camera_entry)
    intrinsics = np.asarray(camera_entry["cam_K"], dtype=np.float32).reshape(3, 3)

    object_mask = None
    mask_path = scene_dir / "mask_visib" / f"{int(image_id):06d}_{int(object_index):06d}.png"
    if mask_path.is_file():
        object_mask = ycbv_read_binary_mask(mask_path)

    processor = DemoScenePreprocessor(resolution=resolution)
    crop_kwargs = dict(dataset_key=YCBV_DATASET_KEY)
    crop_fn = target_crop_resize_image_depth_mask if target_crop else crop_resize_image_depth_mask
    image, depthmap, gt_mask, intrinsics = crop_fn(
        processor,
        image,
        depthmap,
        object_mask,
        intrinsics,
        resolution,
        info=str(scene_dir / "rgb" / f"{int(image_id):06d}.png"),
        **crop_kwargs,
    )
    image_tensor = processor.transform(image).unsqueeze(0).unsqueeze(0).to(device)
    depth_tensor = torch.from_numpy(np.ascontiguousarray(depthmap.astype(np.float32)))[None, None, :, :, None].to(device)
    mask_tensor = torch.from_numpy((depthmap > 0).astype(np.float32))[None, None, :, :].to(device)
    return image_tensor, depth_tensor, mask_tensor, np.asarray(image.convert("RGB")), depthmap, gt_mask, intrinsics


def load_real275_scene_frame_inputs(
    scene_dir: Path,
    frame_id: str,
    inst_id: int,
    resolution,
    device,
    target_crop: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    frame_token = f"{int(frame_id):04d}"
    image_path = scene_dir / f"{frame_token}_color.png"
    depth_path = scene_dir / f"{frame_token}_depth.png"
    mask_path = scene_dir / f"{frame_token}_mask.png"
    image = Image.open(image_path).convert("RGB")
    depthmap = real275_read_depth_m(depth_path)
    intrinsics = REAL275_CAMERA_K.copy()
    object_mask = real275_read_instance_mask(mask_path, inst_id)

    processor = DemoScenePreprocessor(resolution=resolution)
    crop_kwargs = dict(dataset_key=REAL275_DATASET_KEY)
    crop_fn = target_crop_resize_image_depth_mask if target_crop else crop_resize_image_depth_mask
    image, depthmap, gt_mask, intrinsics = crop_fn(
        processor,
        image,
        depthmap,
        object_mask,
        intrinsics,
        resolution,
        info=str(image_path),
        **crop_kwargs,
    )
    image_tensor = processor.transform(image).unsqueeze(0).unsqueeze(0).to(device)
    depth_tensor = torch.from_numpy(np.ascontiguousarray(depthmap.astype(np.float32)))[None, None, :, :, None].to(device)
    mask_tensor = torch.from_numpy((depthmap > 0).astype(np.float32))[None, None, :, :].to(device)
    return image_tensor, depth_tensor, mask_tensor, np.asarray(image.convert("RGB")), depthmap, gt_mask, intrinsics


def load_real275_object_tensor(
    object_records_by_name: Dict[str, Dict[str, Any]],
    object_name: str,
    object_views: Sequence[int],
    resolution,
    device,
) -> Tuple[torch.Tensor, List[Tuple[np.ndarray, str]]]:
    if object_name not in object_records_by_name:
        raise FileNotFoundError(f"No REAL275 aligned reference renders for {object_name}")
    object_rec = object_records_by_name[object_name]
    object_dir: Path = object_rec["object_dir"]
    processor = DemoScenePreprocessor(resolution=resolution)
    resampling = getattr(Image, "Resampling", Image)
    tensors: List[torch.Tensor] = []
    gallery: List[Tuple[np.ndarray, str]] = []
    for image_id in object_views:
        image_path = object_dir / "rgb" / f"{int(image_id):06d}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing REAL275 object reference view: {image_path}")
        image = Image.open(image_path).convert("RGB")
        image = image.resize(tuple(resolution), resampling.LANCZOS)
        tensors.append(processor.transform(image).to(device))
        gallery.append((np.asarray(image), f"Object view {int(image_id):06d}"))
    return torch.stack(tensors, dim=0).unsqueeze(0), gallery


def load_housecat6d_scene_frame_inputs(
    scene_dir: Path,
    image_id: int,
    object_name: str,
    resolution,
    device,
    target_crop: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    image_token = f"{int(image_id):06d}"
    rgb_path = scene_dir / "rgb" / f"{image_token}.png"
    depth_path = scene_dir / "depth" / f"{image_token}.png"
    mask_path = scene_dir / "instance" / f"{image_token}_{object_name}.png"
    image = Image.open(rgb_path).convert("RGB")
    depthmap = housecat6d_read_depth_m(depth_path)
    intrinsics = housecat6d_read_intrinsics(scene_dir / "intrinsics.txt")
    object_mask = housecat6d_read_binary_mask(mask_path)

    processor = DemoScenePreprocessor(resolution=resolution)
    crop_kwargs = dict(dataset_key=HOUSECAT6D_DATASET_KEY)
    crop_fn = target_crop_resize_image_depth_mask if target_crop else crop_resize_image_depth_mask
    image, depthmap, gt_mask, intrinsics = crop_fn(
        processor,
        image,
        depthmap,
        object_mask,
        intrinsics,
        resolution,
        info=str(rgb_path),
        **crop_kwargs,
    )
    image_tensor = processor.transform(image).unsqueeze(0).unsqueeze(0).to(device)
    depth_tensor = torch.from_numpy(np.ascontiguousarray(depthmap.astype(np.float32)))[None, None, :, :, None].to(device)
    mask_tensor = torch.from_numpy((depthmap > 0).astype(np.float32))[None, None, :, :].to(device)
    return image_tensor, depth_tensor, mask_tensor, np.asarray(image.convert("RGB")), depthmap, gt_mask, intrinsics


def load_housecat6d_object_tensor(
    object_records_by_name: Dict[str, Dict[str, Any]],
    object_name: str,
    object_views: Sequence[int],
    resolution,
    device,
) -> Tuple[torch.Tensor, List[Tuple[np.ndarray, str]]]:
    if object_name not in object_records_by_name:
        raise FileNotFoundError(f"No HouseCat6D aligned reference renders for {object_name}")
    object_rec = object_records_by_name[object_name]
    object_dir: Path = object_rec["object_dir"]
    processor = DemoScenePreprocessor(resolution=resolution)
    resampling = getattr(Image, "Resampling", Image)
    tensors: List[torch.Tensor] = []
    gallery: List[Tuple[np.ndarray, str]] = []
    for image_id in object_views:
        image_path = object_dir / "rgb" / f"{int(image_id):06d}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing HouseCat6D object reference view: {image_path}")
        image = Image.open(image_path).convert("RGB")
        image = image.resize(tuple(resolution), resampling.LANCZOS)
        tensors.append(processor.transform(image).to(device))
        gallery.append((np.asarray(image), f"Object view {int(image_id):06d}"))
    return torch.stack(tensors, dim=0).unsqueeze(0), gallery


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
    out_dir = _LOCAL_TMP / "ycbv_point_cloud_pose_glb" / safe_scene
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


class DemoApp:
    def __init__(
        self,
        config_path: Path,
        checkpoint_path: str | None,
        ycbv_root: Path,
        ycbv_test_root: Path | None = None,
        ycbv_object_image_root: Path | None = None,
        real275_root: Path | None = None,
        real275_test_root: Path | None = None,
        real275_gt_root: Path | None = None,
        real275_obj_models_root: Path | None = None,
        real275_object_image_root: Path | None = None,
        housecat6d_root: Path | None = None,
        housecat6d_object_image_root: Path | None = None,
        housecat6d_scene_names: Sequence[str] | None = None,
        align_json: Path | None = None,
    ):
        self.config_path = Path(config_path)
        self.cfg = load_config(self.config_path)
        self.runtime = resolve_runtime_settings(self.cfg)
        self.ycbv_root = Path(ycbv_root).expanduser()
        self.ycbv_test_root = Path(ycbv_test_root or (self.ycbv_root / "test")).expanduser()
        self.ycbv_object_image_root = Path(
            ycbv_object_image_root or (self.ycbv_root / "ycbv_aligned_object_refs")
        ).expanduser()
        self.real275_root = Path(real275_root or DEFAULT_REAL275_ROOT).expanduser()
        self.real275_test_root = Path(real275_test_root or (self.real275_root / "real_test")).expanduser()
        self.real275_gt_root = Path(real275_gt_root or (self.real275_root / "gts" / "real_test")).expanduser()
        self.real275_obj_models_root = Path(
            real275_obj_models_root or (self.real275_root / "obj_models" / "real_test")
        ).expanduser()
        self.real275_object_image_root = Path(
            real275_object_image_root or (self.real275_root / "real275_aligned_object_refs")
        ).expanduser()
        self.housecat6d_root = Path(housecat6d_root or DEFAULT_HOUSECAT6D_ROOT).expanduser()
        self.housecat6d_object_image_root = Path(
            housecat6d_object_image_root or (self.housecat6d_root / "housecat6d_aligned_object_refs")
        ).expanduser()
        self.housecat6d_scene_names: Tuple[str, ...] = tuple(
            str(name) for name in (housecat6d_scene_names or DEFAULT_HOUSECAT6D_SCENE_NAMES)
        )
        self.align_json_path = Path(align_json or DEFAULT_ALIGN_JSON).expanduser()
        self.models_info_path = self.ycbv_root / "models" / "models_info.json"

        self.object_views = tuple(int(v) for v in self.runtime["object_input_views"])
        self.resolution = tuple(int(v) for v in self.runtime["resolution"])
        self.symmetry_info_path = self._resolve_symmetry_info_path()

        align_root = read_json(self.align_json_path)["datasets"]
        ycbv_align = align_root["ycbv"]
        self.ycbv_obj_id_to_category = {int(k): str(v) for k, v in ycbv_align["obj_id_to_category"].items()}
        self.ycbv_global_r_align = np.asarray(ycbv_align["global_R_align"], dtype=np.float32).reshape(3, 3)
        self.ycbv_r_align_overrides = {
            int(k): np.asarray(v, dtype=np.float32).reshape(3, 3)
            for k, v in ycbv_align.get("per_object_overrides", {}).items()
        }
        self.ycbv_models_info = read_json(self.models_info_path)
        self.ycbv_object_records_by_id = self._build_ycbv_object_records_by_id()
        self.ycbv_scene_records = self._build_ycbv_scene_records()

        real275_align = align_root["real275"]
        self.real275_class_id_to_name = {int(k): str(v) for k, v in real275_align["class_id_to_name"].items()}
        self.real275_r_align_by_class_id = {
            int(k): np.asarray(v["R_align"], dtype=np.float32).reshape(3, 3)
            for k, v in real275_align["classes"].items()
        }
        self.real275_object_records_by_name = self._build_real275_object_records_by_name()
        self.real275_object_name_to_id = {
            name: idx + 1 for idx, name in enumerate(sorted(self.real275_object_records_by_name.keys()))
        }
        self.real275_class_to_refs = self._group_real275_refs_by_class()
        self.real275_scene_records = self._build_real275_scene_records()

        housecat6d_align = align_root.get("housecat6d", {})
        self.housecat6d_category_name_to_id = {
            str(k): int(v) for k, v in housecat6d_align.get("category_name_to_id", {}).items()
        }
        self.housecat6d_category_id_to_name = {
            v: k for k, v in self.housecat6d_category_name_to_id.items()
        }
        self.housecat6d_r_align_by_category = {
            str(name): np.asarray(item["R_align"], dtype=np.float32).reshape(3, 3)
            for name, item in housecat6d_align.get("classes", {}).items()
        }
        self.housecat6d_object_records_by_name = self._build_housecat6d_object_records_by_name()
        self.housecat6d_object_name_to_id = {
            name: idx + 1 for idx, name in enumerate(sorted(self.housecat6d_object_records_by_name.keys()))
        }
        self.housecat6d_scene_records = self._build_housecat6d_scene_records()

        if (
            not self.ycbv_scene_records
            and not self.real275_scene_records
            and not self.housecat6d_scene_records
        ):
            raise RuntimeError(
                "No YCB-V, REAL275, or HouseCat6D scenes were found. Check the configured paths."
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = resolve_checkpoint_path(self.cfg, checkpoint_path)
        self.model = None
        self.available_checkpoints = self.discover_checkpoints()
        self.load_checkpoint(self.checkpoint_path)

    def _resolve_symmetry_info_path(self) -> Path | None:
        value = self.cfg.get("object_srt_symmetry_info_path")
        if value:
            path = Path(str(value)).expanduser()
            if path.is_file():
                return path
        local = PROJECT_ROOT / "mixed_symmetry_info.json"
        return local if local.is_file() else None

    @property
    def dataset_choices(self) -> List[str]:
        choices: List[str] = []
        if self.ycbv_scene_records:
            choices.append(YCBV_DATASET_KEY)
        if self.real275_scene_records:
            choices.append(REAL275_DATASET_KEY)
        if self.housecat6d_scene_records:
            choices.append(HOUSECAT6D_DATASET_KEY)
        return choices or [YCBV_DATASET_KEY]

    @property
    def default_dataset(self) -> str:
        return self.dataset_choices[0]

    def _ycbv_r_align_for_object(self, object_id: int) -> np.ndarray:
        return self.ycbv_r_align_overrides.get(int(object_id), self.ycbv_global_r_align).astype(np.float32)

    def _real275_r_align_for_class(self, class_id: int) -> np.ndarray:
        return self.real275_r_align_by_class_id.get(int(class_id), np.eye(3, dtype=np.float32)).astype(np.float32)

    def _build_ycbv_object_records_by_id(self) -> Dict[int, Dict[str, Any]]:
        if not self.ycbv_object_image_root.is_dir():
            return {}
        records: Dict[int, Dict[str, Any]] = {}
        for object_dir in sorted(self.ycbv_object_image_root.glob("obj_*")):
            stem = object_dir.name.removeprefix("obj_")
            if not object_dir.is_dir() or not stem.isdigit():
                continue
            object_id = int(stem)
            rgb_dir = object_dir / "rgb"
            if not rgb_dir.is_dir():
                continue
            available_ids = sorted(int(path.stem) for path in rgb_dir.glob("*.png") if path.stem.isdigit())
            missing = [view_id for view_id in self.object_views if view_id not in available_ids]
            if missing:
                continue
            records[object_id] = {
                "object_id": object_id,
                "object_dir": object_dir,
                "image_ids": available_ids,
                "category": self.ycbv_obj_id_to_category.get(object_id, ""),
            }
        return records

    def _build_ycbv_scene_records(self) -> Dict[str, Dict[str, Any]]:
        records: Dict[str, Dict[str, Any]] = {}
        if not self.ycbv_test_root.is_dir():
            return records
        for scene_dir in sorted(self.ycbv_test_root.iterdir()):
            scene_gt_path = scene_dir / "scene_gt.json"
            if not scene_dir.is_dir() or not scene_gt_path.is_file():
                continue
            records[scene_dir.name] = {
                "scene_name": scene_dir.name,
                "scene_dir": scene_dir,
            }
        return records

    def _build_real275_object_records_by_name(self) -> Dict[str, Dict[str, Any]]:
        if not self.real275_object_image_root.is_dir():
            return {}
        records: Dict[str, Dict[str, Any]] = {}
        for object_dir in sorted(self.real275_object_image_root.iterdir()):
            if not object_dir.is_dir():
                continue
            rgb_dir = object_dir / "rgb"
            if not rgb_dir.is_dir():
                continue
            available_ids = sorted(int(path.stem) for path in rgb_dir.glob("*.png") if path.stem.isdigit())
            missing = [view_id for view_id in self.object_views if view_id not in available_ids]
            if missing:
                continue
            metadata_path = object_dir / "metadata.json"
            metadata = read_json(metadata_path) if metadata_path.is_file() else {}
            category = str(metadata.get("class_name", object_dir.name.split("_", 1)[0]))
            class_id = int(metadata.get("class_id", 0))
            records[object_dir.name] = {
                "object_name": object_dir.name,
                "object_dir": object_dir,
                "image_ids": available_ids,
                "category": category,
                "class_id": class_id,
                "metadata": metadata,
            }
        return records

    def _group_real275_refs_by_class(self) -> Dict[int, List[str]]:
        grouped: Dict[int, List[str]] = {}
        for object_name, rec in self.real275_object_records_by_name.items():
            class_id = int(rec.get("class_id", 0))
            grouped.setdefault(class_id, []).append(object_name)
        for class_id in grouped:
            grouped[class_id].sort()
        return grouped

    def _build_real275_scene_records(self) -> Dict[str, Dict[str, Any]]:
        records: Dict[str, Dict[str, Any]] = {}
        if not self.real275_test_root.is_dir():
            return records
        for scene_dir in sorted(self.real275_test_root.glob("scene_*")):
            if not scene_dir.is_dir():
                continue
            if not any(scene_dir.glob("*_color.png")):
                continue
            records[scene_dir.name] = {
                "scene_name": scene_dir.name,
                "scene_dir": scene_dir,
            }
        return records

    @staticmethod
    def _ycbv_scene_key(scene_name: str) -> str:
        return f"{YCBV_DATASET_KEY}::{scene_name}"

    @staticmethod
    def _real275_scene_key(scene_name: str) -> str:
        return f"{REAL275_DATASET_KEY}::{scene_name}"

    def scene_dir_for(self, dataset_key: str, scene_name: str) -> Path:
        if dataset_key == REAL275_DATASET_KEY:
            return self.real275_scene_records[scene_name]["scene_dir"]
        if dataset_key == HOUSECAT6D_DATASET_KEY:
            return self.housecat6d_scene_records[scene_name]["scene_dir"]
        return self.ycbv_scene_records[scene_name]["scene_dir"]

    def _ycbv_size_native(self, object_id: int) -> np.ndarray:
        info = self.ycbv_models_info[str(int(object_id))]
        return np.array([info["size_x"], info["size_y"], info["size_z"]], dtype=np.float32) / 1000.0

    def _ycbv_size_aligned(self, object_id: int) -> np.ndarray:
        size_native = self._ycbv_size_native(object_id)
        r_align = self._ycbv_r_align_for_object(object_id)
        size_aligned = np.abs(r_align) @ size_native
        return np.clip(size_aligned, 1e-6, None).astype(np.float32)

    def _ycbv_axis_length(self, object_id: int) -> float:
        return max(float(np.linalg.norm(self._ycbv_size_aligned(object_id))) * 0.25, 1e-3)

    def _real275_size_native(self, model_name: str) -> np.ndarray | None:
        return real275_read_canonical_extent(self.real275_obj_models_root, model_name)

    def _real275_size_aligned(self, model_name: str, class_id: int) -> np.ndarray | None:
        size_native = self._real275_size_native(model_name)
        if size_native is None:
            return None
        r_align = self._real275_r_align_for_class(class_id)
        size_aligned = np.abs(r_align) @ size_native
        return np.clip(size_aligned, 1e-6, None).astype(np.float32)

    def _real275_axis_length(self, model_name: str, class_id: int) -> float:
        size_aligned = self._real275_size_aligned(model_name, class_id)
        if size_aligned is None:
            return 0.05
        return max(float(np.linalg.norm(size_aligned)) * 0.25, 1e-3)

    def _build_housecat6d_object_records_by_name(self) -> Dict[str, Dict[str, Any]]:
        if not self.housecat6d_object_image_root.is_dir():
            return {}
        records: Dict[str, Dict[str, Any]] = {}
        for object_dir in sorted(self.housecat6d_object_image_root.iterdir()):
            if not object_dir.is_dir():
                continue
            rgb_dir = object_dir / "rgb"
            if not rgb_dir.is_dir():
                continue
            available_ids = sorted(int(path.stem) for path in rgb_dir.glob("*.png") if path.stem.isdigit())
            missing = [view_id for view_id in self.object_views if view_id not in available_ids]
            if missing:
                continue
            metadata_path = object_dir / "metadata.json"
            metadata = read_json(metadata_path) if metadata_path.is_file() else {}
            category = str(metadata.get("class_name", object_dir.name.split("-", 1)[0]))
            class_id = int(
                metadata.get("class_id", self.housecat6d_category_name_to_id.get(category, 0))
            )
            records[object_dir.name] = {
                "object_name": object_dir.name,
                "object_dir": object_dir,
                "image_ids": available_ids,
                "category": category,
                "class_id": class_id,
                "metadata": metadata,
            }
        return records

    def _build_housecat6d_scene_records(self) -> Dict[str, Dict[str, Any]]:
        records: Dict[str, Dict[str, Any]] = {}
        if not self.housecat6d_root.is_dir():
            return records
        for scene_name in self.housecat6d_scene_names:
            scene_dir = self.housecat6d_root / scene_name
            if not scene_dir.is_dir():
                continue
            if not (scene_dir / "labels").is_dir() or not (scene_dir / "rgb").is_dir():
                continue
            if not (scene_dir / "intrinsics.txt").is_file():
                continue
            records[scene_name] = {
                "scene_name": scene_name,
                "scene_dir": scene_dir,
            }
        return records

    def _housecat6d_r_align_for_category(self, category: str) -> np.ndarray:
        r = self.housecat6d_r_align_by_category.get(str(category))
        if r is None:
            return np.eye(3, dtype=np.float32)
        return r.astype(np.float32)

    def _housecat6d_size_aligned(self, native_size: np.ndarray, category: str) -> np.ndarray:
        size_native = np.asarray(native_size, dtype=np.float32).reshape(3)
        r_align = self._housecat6d_r_align_for_category(category)
        size_aligned = np.abs(r_align) @ size_native
        return np.clip(size_aligned, 1e-6, None).astype(np.float32)

    def _housecat6d_axis_length(self, size_aligned: np.ndarray) -> float:
        return max(float(np.linalg.norm(size_aligned)) * 0.25, 1e-3)

    def get_scene_choices(self, dataset_key: str) -> List[str]:
        if dataset_key == REAL275_DATASET_KEY:
            return sorted(self.real275_scene_records.keys())
        if dataset_key == HOUSECAT6D_DATASET_KEY:
            return sorted(self.housecat6d_scene_records.keys())
        return sorted(self.ycbv_scene_records.keys())

    def get_frame_choices(self, dataset_key: str, scene_name: str) -> List[str]:
        if not scene_name:
            return []
        if dataset_key == REAL275_DATASET_KEY:
            if scene_name not in self.real275_scene_records:
                return []
            scene_dir = self.real275_scene_records[scene_name]["scene_dir"]
            return [path.stem.split("_")[0] for path in sorted(scene_dir.glob("*_color.png"))]
        if dataset_key == HOUSECAT6D_DATASET_KEY:
            if scene_name not in self.housecat6d_scene_records:
                return []
            scene_dir = self.housecat6d_scene_records[scene_name]["scene_dir"]
            label_paths = sorted((scene_dir / "labels").glob("*_label.pkl"))
            return [f"{int(path.name.split('_', 1)[0]):06d}" for path in label_paths]
        if scene_name not in self.ycbv_scene_records:
            return []
        scene_dir = self.ycbv_scene_records[scene_name]["scene_dir"]
        scene_gt = read_json(scene_dir / "scene_gt.json")
        return [f"{int(x):06d}" for x in sorted(int(k) for k in scene_gt.keys())]

    def object_choices_for_frame(self, dataset_key: str, scene_name: str, frame_name: str):
        if not scene_name or not frame_name:
            return []
        if dataset_key == REAL275_DATASET_KEY:
            return self._real275_object_choices(scene_name, frame_name)
        if dataset_key == HOUSECAT6D_DATASET_KEY:
            return self._housecat6d_object_choices(scene_name, frame_name)
        return self._ycbv_object_choices(scene_name, frame_name)

    def _ycbv_object_choices(self, scene_name: str, frame_name: str):
        if scene_name not in self.ycbv_scene_records:
            return []
        scene_dir = self.ycbv_scene_records[scene_name]["scene_dir"]
        scene_gt = read_json(scene_dir / "scene_gt.json")
        entries = scene_gt.get(str(int(frame_name)), [])
        choices = []
        for object_index, gt in enumerate(entries):
            object_id = int(gt.get("obj_id", -1))
            if object_id not in self.ycbv_object_records_by_id:
                continue
            category = self.ycbv_obj_id_to_category.get(object_id, "unknown")
            label = f"[{category}] obj_{object_id:06d} (instance #{object_index})"
            value = f"ycbv::{object_index}::{object_id}"
            choices.append((label, value))
        return choices

    def _real275_object_choices(self, scene_name: str, frame_name: str):
        if scene_name not in self.real275_scene_records:
            return []
        scene_dir = self.real275_scene_records[scene_name]["scene_dir"]
        frame_token = f"{int(frame_name):04d}"
        entries = real275_read_meta(scene_dir / f"{frame_token}_meta.txt")
        choices = []
        for entry in entries:
            class_id = int(entry["class_id"])
            inst_id = int(entry["inst_id"])
            model_name = entry["model_name"]
            category = self.real275_class_id_to_name.get(class_id, "unknown")
            ref_candidates = self.real275_class_to_refs.get(class_id, [])
            if not ref_candidates:
                continue
            for ref_name in ref_candidates:
                label = (
                    f"[{category}] inst {inst_id} · {model_name} → ref {ref_name}"
                )
                value = f"real275::{inst_id}::{class_id}::{model_name}::{ref_name}"
                choices.append((label, value))
        return choices

    def _housecat6d_object_choices(self, scene_name: str, frame_name: str):
        if scene_name not in self.housecat6d_scene_records:
            return []
        scene_dir = self.housecat6d_scene_records[scene_name]["scene_dir"]
        label_path = scene_dir / "labels" / f"{int(frame_name):06d}_label.pkl"
        if not label_path.is_file():
            return []
        label = housecat6d_read_label(label_path)
        model_list = [str(name) for name in label.get("model_list", [])]
        class_ids = list(label.get("class_ids", []))
        choices = []
        for object_index, (model_name, class_id) in enumerate(zip(model_list, class_ids)):
            class_id = int(class_id)
            category = self.housecat6d_category_id_to_name.get(class_id, "unknown")
            if model_name not in self.housecat6d_object_records_by_name:
                continue
            label_text = f"[{category}] {model_name} (instance #{object_index})"
            value = f"housecat6d::{object_index}::{class_id}::{model_name}"
            choices.append((label_text, value))
        return choices

    @staticmethod
    def _parse_object_value(value: str):
        parts = str(value).split("::")
        if len(parts) < 2:
            raise ValueError(f"Could not parse object dropdown value: {value}")
        if parts[0] == "ycbv":
            if len(parts) != 3:
                raise ValueError(f"Bad YCB-V object value: {value}")
            return {
                "dataset": YCBV_DATASET_KEY,
                "object_index": int(parts[1]),
                "object_id": int(parts[2]),
            }
        if parts[0] == "real275":
            if len(parts) != 5:
                raise ValueError(f"Bad REAL275 object value: {value}")
            return {
                "dataset": REAL275_DATASET_KEY,
                "inst_id": int(parts[1]),
                "class_id": int(parts[2]),
                "model_name": parts[3],
                "ref_object_name": parts[4],
            }
        if parts[0] == "housecat6d":
            if len(parts) != 4:
                raise ValueError(f"Bad HouseCat6D object value: {value}")
            return {
                "dataset": HOUSECAT6D_DATASET_KEY,
                "object_index": int(parts[1]),
                "class_id": int(parts[2]),
                "model_name": parts[3],
            }
        # legacy YCB-V value: "{object_index}::{object_id}"
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return {
                "dataset": YCBV_DATASET_KEY,
                "object_index": int(parts[0]),
                "object_id": int(parts[1]),
            }
        raise ValueError(f"Unknown object dropdown value: {value}")

    @staticmethod
    def _compute_depth_mean_scale(
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

    def input_gallery(self, dataset_key: str, scene_name: str, frame_name: str, object_value: str):
        if not (dataset_key and scene_name and frame_name and object_value):
            return None, []
        parsed = self._parse_object_value(object_value)
        if parsed["dataset"] != dataset_key:
            return None, []
        scene_dir = self.scene_dir_for(dataset_key, scene_name)
        if dataset_key == REAL275_DATASET_KEY:
            scene_image = str(scene_dir / f"{int(frame_name):04d}_color.png")
            _, object_gallery = load_real275_object_tensor(
                self.real275_object_records_by_name,
                parsed["ref_object_name"],
                self.object_views,
                self.resolution,
                torch.device("cpu"),
            )
            return scene_image, object_gallery
        if dataset_key == HOUSECAT6D_DATASET_KEY:
            scene_image = str(scene_dir / "rgb" / f"{int(frame_name):06d}.png")
            _, object_gallery = load_housecat6d_object_tensor(
                self.housecat6d_object_records_by_name,
                parsed["model_name"],
                self.object_views,
                self.resolution,
                torch.device("cpu"),
            )
            return scene_image, object_gallery
        scene_image = str(scene_dir / "rgb" / f"{int(frame_name):06d}.png")
        _, object_gallery = load_ycbv_object_tensor(
            self.ycbv_object_records_by_id,
            parsed["object_id"],
            self.object_views,
            self.resolution,
            torch.device("cpu"),
        )
        return scene_image, object_gallery

    def run_inference(
        self,
        dataset_key: str,
        scene_name: str,
        frame_name: str,
        object_value: str,
        use_depth_input: bool,
        use_depth_scale: bool,
        show_point_cloud_pose: bool,
        point_cloud_stride: int,
        target_crop: bool,
        pred_pose_gt_bbox: bool,
    ):
        parsed = self._parse_object_value(object_value)
        if parsed["dataset"] != dataset_key:
            raise RuntimeError(
                f"Selected object dataset `{parsed['dataset']}` does not match active dataset `{dataset_key}`"
            )
        if dataset_key == REAL275_DATASET_KEY:
            return self._run_real275_inference(
                scene_name=scene_name,
                frame_name=frame_name,
                parsed=parsed,
                use_depth_input=bool(use_depth_input),
                use_depth_scale=bool(use_depth_scale),
                show_point_cloud_pose=bool(show_point_cloud_pose),
                point_cloud_stride=int(point_cloud_stride),
                target_crop=bool(target_crop),
                pred_pose_gt_bbox=bool(pred_pose_gt_bbox),
            )
        if dataset_key == HOUSECAT6D_DATASET_KEY:
            return self._run_housecat6d_inference(
                scene_name=scene_name,
                frame_name=frame_name,
                parsed=parsed,
                use_depth_input=bool(use_depth_input),
                use_depth_scale=bool(use_depth_scale),
                show_point_cloud_pose=bool(show_point_cloud_pose),
                point_cloud_stride=int(point_cloud_stride),
                target_crop=bool(target_crop),
                pred_pose_gt_bbox=bool(pred_pose_gt_bbox),
            )
        return self._run_ycbv_inference(
            scene_name=scene_name,
            frame_name=frame_name,
            parsed=parsed,
            use_depth_input=bool(use_depth_input),
            use_depth_scale=bool(use_depth_scale),
            show_point_cloud_pose=bool(show_point_cloud_pose),
            point_cloud_stride=int(point_cloud_stride),
            target_crop=bool(target_crop),
            pred_pose_gt_bbox=bool(pred_pose_gt_bbox),
        )

    def _run_ycbv_inference(
        self,
        *,
        scene_name: str,
        frame_name: str,
        parsed: Dict[str, Any],
        use_depth_input: bool,
        use_depth_scale: bool,
        show_point_cloud_pose: bool,
        point_cloud_stride: int,
        target_crop: bool,
        pred_pose_gt_bbox: bool,
    ):
        object_index = int(parsed["object_index"])
        object_id = int(parsed["object_id"])
        image_id = int(frame_name)
        scene_dir = self.scene_dir_for(YCBV_DATASET_KEY, scene_name)

        scene_tensor, depth_tensor, mask_tensor, display_image, display_depth, gt_mask, intrinsic = load_ycbv_scene_frame_inputs(
            scene_dir,
            image_id,
            object_index,
            self.resolution,
            self.device,
            target_crop=target_crop,
        )
        object_tensor, _ = load_ycbv_object_tensor(
            self.ycbv_object_records_by_id,
            object_id,
            self.object_views,
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
            raise RuntimeError(f"Model output missing object_presence_logits: {sorted(outputs.keys())}")
        presence_logit = float(outputs["object_presence_logits"].reshape(-1)[0].detach().float().cpu())
        presence_prob = float(torch.sigmoid(torch.tensor(presence_logit)).item())
        pred_present = presence_prob >= 0.5

        pred_mask = None
        pred_mask_image = None
        if "object_mask" in outputs:
            pred_mask = outputs["object_mask"][0, 0].detach().float().cpu().numpy()
            pred_mask_image = mask_overlay_image(display_image, pred_mask, color=(255, 64, 64))
        gt_mask_image = mask_overlay_image(display_image, gt_mask, color=(80, 255, 120)) if gt_mask is not None else None

        if "object_pose" not in outputs or "object_translation" not in outputs:
            raise RuntimeError(f"Model output missing pose keys: {sorted(outputs.keys())}")
        pred_rot6d = outputs["object_pose"][0].detach().float().cpu().numpy()
        pred_translation_cam = outputs["object_translation"][0].detach().float().cpu().numpy().astype(np.float32)
        pred_rotation_aligned = rot6d_to_matrix(pred_rot6d).astype(np.float32)

        depth_mean_scale = self._compute_depth_mean_scale(
            display_depth,
            outputs.get("depth"),
            use_depth_input,
        )
        if use_depth_scale:
            pred_translation_cam = pred_translation_cam * np.float32(depth_mean_scale)

        pred_size_aligned = None
        if "object_size" in outputs:
            pred_size_aligned = outputs["object_size"].reshape(-1, 3)[0].detach().float().cpu().numpy().astype(np.float32)
        elif "object_size_log" in outputs:
            pred_size_aligned = np.exp(outputs["object_size_log"].reshape(-1, 3)[0].detach().float().cpu().numpy()).astype(np.float32)

        scene_gt = read_json(scene_dir / "scene_gt.json")[str(image_id)]
        gt = scene_gt[object_index]
        gt_object_id = int(gt["obj_id"])
        if gt_object_id != int(object_id):
            raise RuntimeError(
                f"Scene GT obj_id {gt_object_id} does not match selected object {object_id} "
                f"at instance #{object_index}"
            )
        r_align = self._ycbv_r_align_for_object(object_id)
        gt_rotation_native = np.asarray(gt["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
        gt_translation_cam = np.asarray(gt["cam_t_m2c"], dtype=np.float32).reshape(3) / 1000.0
        gt_rotation_aligned = (gt_rotation_native @ r_align.T).astype(np.float32)
        gt_size_aligned = self._ycbv_size_aligned(object_id)
        gt_size_native = self._ycbv_size_native(object_id)

        axis_length = self._ycbv_axis_length(object_id)
        raw_pred_size = pred_size_aligned if pred_size_aligned is not None else gt_size_aligned
        size_for_pred_box, size_was_clipped = clamp_predicted_size_for_bbox(raw_pred_size, gt_size_aligned)
        visualized_pred_size = gt_size_aligned if pred_pose_gt_bbox else size_for_pred_box
        pred_bbox_obj = centered_axis_bbox_corners(visualized_pred_size)
        gt_bbox_obj = centered_axis_bbox_corners(gt_size_aligned)

        rot_error_deg, sym_count = symmetric_rotation_error_degrees(
            pred_rotation_aligned,
            gt_rotation_aligned,
            object_id,
            self.symmetry_info_path,
            int(self.cfg.get("object_srt_symmetry_continuous_steps", 72)),
        )
        trans_error = translation_error(pred_translation_cam, gt_translation_cam)
        size_error = translation_error(raw_pred_size, gt_size_aligned)
        mask_score = mask_iou(pred_mask, gt_mask)
        raw_bbox_iou_details = bbox_iou_3d_details(
            raw_pred_size,
            pred_rotation_aligned,
            pred_translation_cam,
            gt_size_aligned,
            gt_rotation_aligned,
            gt_translation_cam,
        )
        visualized_bbox_iou_details = bbox_iou_3d_details(
            visualized_pred_size,
            pred_rotation_aligned,
            pred_translation_cam,
            gt_size_aligned,
            gt_rotation_aligned,
            gt_translation_cam,
        )

        pred_image = draw_bbox_axes_overlay_on_image(
            display_image,
            intrinsic,
            pred_rotation_aligned,
            pred_translation_cam,
            pred_bbox_obj,
            axis_length,
            AXIS_COLORS,
            PRED_BBOX_COLOR,
        )
        gt_image = draw_bbox_axes_overlay_on_image(
            display_image,
            intrinsic,
            gt_rotation_aligned,
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
                pred_rotation_cam=pred_rotation_aligned,
                pred_translation_cam=pred_translation_cam,
                pred_bbox_obj=pred_bbox_obj,
                gt_rotation_cam=gt_rotation_aligned,
                gt_translation_cam=gt_translation_cam,
                gt_bbox_obj=gt_bbox_obj,
                axis_length=axis_length,
                point_cloud_stride=point_cloud_stride,
            )

        category = self.ycbv_obj_id_to_category.get(object_id, "unknown")
        pose_lines = [
            "### Prediction Summary",
            f"- checkpoint: `{self.checkpoint_path}`",
            f"- dataset: `YCB-V test`",
            f"- scene: `{scene_name}`",
            f"- frame: `{image_id:06d}`",
            f"- target crop input: `{target_crop}`",
            f"- predicted pose with GT bbox size: `{pred_pose_gt_bbox}`",
            f"- object: `{ycbv_object_key(object_id)} · {category}` (instance #{object_index})",
            f"- predicted presence probability: `{presence_prob:.6f}` (logit `{presence_logit:.6f}`)",
            f"- predicted present @0.5: `{bool(pred_present)}`",
            f"- use depth input: `{use_depth_input}`",
            f"- use depth scale: `{use_depth_scale}`",
            "",
            "### Predicted Pose (OV9D-aligned object frame)",
            f"- depth_mean_scale (m): `{depth_mean_scale:.6f}` "
            f"({'GT depth mean' if use_depth_input else 'predicted depth mean'})",
            f"- applied depth scale to translation: `{use_depth_scale}`",
            f"- camera-frame translation (m): `{np.round(pred_translation_cam, 6).tolist()}`",
            f"- camera-frame rotation matrix: `{np.round(pred_rotation_aligned, 6).tolist()}`",
            f"- raw predicted aligned size xyz (m): `{np.round(raw_pred_size, 6).tolist()}`",
            f"- visualized predicted bbox size xyz (m): `{np.round(visualized_pred_size, 6).tolist()}`"
            + (" from GT bbox" if pred_pose_gt_bbox else (" clipped for display" if size_was_clipped else "")),
            "",
            "### Ground Truth (aligned + native)",
            f"- aligned rotation matrix: `{np.round(gt_rotation_aligned, 6).tolist()}`",
            f"- native cam_R_m2c: `{np.round(gt_rotation_native, 6).tolist()}`",
            f"- camera-frame translation (m): `{np.round(gt_translation_cam, 6).tolist()}`",
            f"- aligned size xyz (m): `{np.round(gt_size_aligned, 6).tolist()}`",
            f"- native size xyz (m): `{np.round(gt_size_native, 6).tolist()}`",
            f"- R_align used: `{np.round(r_align, 6).tolist()}`",
            "",
            "### Errors",
            f"- translation L2 (m): `{trans_error['l2']:.6f}`",
            f"- translation abs xyz (m): `{np.round(trans_error['abs_xyz'], 6).tolist()}`",
            f"- raw predicted aligned size L2 (m): `{size_error['l2']:.6f}`",
            f"- raw predicted aligned size abs xyz (m): `{np.round(size_error['abs_xyz'], 6).tolist()}`",
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

    def _run_real275_inference(
        self,
        *,
        scene_name: str,
        frame_name: str,
        parsed: Dict[str, Any],
        use_depth_input: bool,
        use_depth_scale: bool,
        show_point_cloud_pose: bool,
        point_cloud_stride: int,
        target_crop: bool,
        pred_pose_gt_bbox: bool,
    ):
        inst_id = int(parsed["inst_id"])
        class_id = int(parsed["class_id"])
        model_name = str(parsed["model_name"])
        ref_object_name = str(parsed["ref_object_name"])
        frame_token = f"{int(frame_name):04d}"
        scene_dir = self.scene_dir_for(REAL275_DATASET_KEY, scene_name)

        scene_tensor, depth_tensor, mask_tensor, display_image, display_depth, gt_mask, intrinsic = load_real275_scene_frame_inputs(
            scene_dir,
            frame_token,
            inst_id,
            self.resolution,
            self.device,
            target_crop=target_crop,
        )
        object_tensor, _ = load_real275_object_tensor(
            self.real275_object_records_by_name,
            ref_object_name,
            self.object_views,
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
            raise RuntimeError(f"Model output missing object_presence_logits: {sorted(outputs.keys())}")
        presence_logit = float(outputs["object_presence_logits"].reshape(-1)[0].detach().float().cpu())
        presence_prob = float(torch.sigmoid(torch.tensor(presence_logit)).item())
        pred_present = presence_prob >= 0.5

        pred_mask = None
        pred_mask_image = None
        if "object_mask" in outputs:
            pred_mask = outputs["object_mask"][0, 0].detach().float().cpu().numpy()
            pred_mask_image = mask_overlay_image(display_image, pred_mask, color=(255, 64, 64))
        gt_mask_image = mask_overlay_image(display_image, gt_mask, color=(80, 255, 120)) if gt_mask is not None else None

        if "object_pose" not in outputs or "object_translation" not in outputs:
            raise RuntimeError(f"Model output missing pose keys: {sorted(outputs.keys())}")
        pred_rot6d = outputs["object_pose"][0].detach().float().cpu().numpy()
        pred_translation_cam = outputs["object_translation"][0].detach().float().cpu().numpy().astype(np.float32)
        pred_rotation_aligned = rot6d_to_matrix(pred_rot6d).astype(np.float32)

        depth_mean_scale = self._compute_depth_mean_scale(
            display_depth,
            outputs.get("depth"),
            use_depth_input,
        )
        if use_depth_scale:
            pred_translation_cam = pred_translation_cam * np.float32(depth_mean_scale)

        pred_size_aligned = None
        if "object_size" in outputs:
            pred_size_aligned = outputs["object_size"].reshape(-1, 3)[0].detach().float().cpu().numpy().astype(np.float32)
        elif "object_size_log" in outputs:
            pred_size_aligned = np.exp(outputs["object_size_log"].reshape(-1, 3)[0].detach().float().cpu().numpy()).astype(np.float32)

        gt_path = self.real275_gt_root / f"results_real_test_{scene_name}_{frame_token}.pkl"
        gt_rt = None
        gt_inst_index = None
        meta_entries = real275_read_meta(scene_dir / f"{frame_token}_meta.txt")
        try:
            meta_inst_index = next(idx for idx, entry in enumerate(meta_entries) if int(entry["inst_id"]) == inst_id)
        except StopIteration:
            meta_inst_index = None
        if gt_path.is_file():
            with gt_path.open("rb") as handle:
                gt_payload = pickle.load(handle)
            gt_rts = gt_payload.get("gt_RTs")
            if gt_rts is not None and meta_inst_index is not None and meta_inst_index < len(gt_rts):
                gt_rt = np.asarray(gt_rts[meta_inst_index], dtype=np.float64).reshape(4, 4)
                gt_inst_index = meta_inst_index

        gt_rotation_native = None
        gt_translation_cam = None
        gt_rotation_aligned = None
        gt_size_aligned = None
        gt_size_native = None
        gt_scale_from_rt = None
        r_align = self._real275_r_align_for_class(class_id)
        if gt_rt is not None:
            gt_rotation_native, gt_translation_cam, gt_scale_from_rt = real275_decompose_gt_rt(gt_rt)
            gt_rotation_aligned = (gt_rotation_native @ r_align.T).astype(np.float32)
        gt_size_native = self._real275_size_native(model_name)
        gt_size_aligned = self._real275_size_aligned(model_name, class_id)

        category = self.real275_class_id_to_name.get(class_id, "unknown")
        ref_size_native = None
        ref_size_aligned = None
        ref_object_id_for_symmetry = self.real275_object_name_to_id.get(ref_object_name)
        ref_metadata = self.real275_object_records_by_name[ref_object_name].get("metadata", {})
        ref_bounds = np.asarray(ref_metadata.get("mesh", {}).get("centered_bounds", []), dtype=np.float32)
        if ref_bounds.shape == (2, 3):
            ref_size_native = np.clip(ref_bounds[1] - ref_bounds[0], 1e-6, None).astype(np.float32)
            ref_size_aligned = np.abs(r_align) @ ref_size_native
            ref_size_aligned = np.clip(ref_size_aligned, 1e-6, None).astype(np.float32)
        size_fallback = gt_size_aligned if gt_size_aligned is not None else ref_size_aligned
        if size_fallback is None:
            size_fallback = np.array([0.1, 0.1, 0.1], dtype=np.float32)

        raw_pred_size = pred_size_aligned if pred_size_aligned is not None else size_fallback
        size_for_pred_box, size_was_clipped = clamp_predicted_size_for_bbox(raw_pred_size, size_fallback)
        visualized_pred_size = (
            gt_size_aligned
            if pred_pose_gt_bbox and gt_size_aligned is not None
            else size_for_pred_box
        )
        pred_bbox_obj = centered_axis_bbox_corners(visualized_pred_size)
        gt_bbox_obj = centered_axis_bbox_corners(gt_size_aligned) if gt_size_aligned is not None else None
        axis_length = (
            self._real275_axis_length(model_name, class_id)
            if gt_size_aligned is not None
            else max(float(np.linalg.norm(size_for_pred_box)) * 0.25, 1e-3)
        )

        rot_error_deg = None
        sym_count = 0
        trans_error_dict = None
        size_error_dict = None
        raw_bbox_iou_details = None
        visualized_bbox_iou_details = None
        if gt_rotation_aligned is not None and gt_translation_cam is not None:
            rot_error_deg, sym_count = symmetric_rotation_error_degrees(
                pred_rotation_aligned,
                gt_rotation_aligned,
                int(ref_object_id_for_symmetry) if ref_object_id_for_symmetry is not None else int(class_id),
                self.symmetry_info_path,
                int(self.cfg.get("object_srt_symmetry_continuous_steps", 72)),
                dataset_label=REAL275_DATASET_LABEL,
            )
            trans_error_dict = translation_error(pred_translation_cam, gt_translation_cam)
            if gt_size_aligned is not None:
                size_error_dict = translation_error(raw_pred_size, gt_size_aligned)
                raw_bbox_iou_details = bbox_iou_3d_details(
                    raw_pred_size,
                    pred_rotation_aligned,
                    pred_translation_cam,
                    gt_size_aligned,
                    gt_rotation_aligned,
                    gt_translation_cam,
                )
                visualized_bbox_iou_details = bbox_iou_3d_details(
                    visualized_pred_size,
                    pred_rotation_aligned,
                    pred_translation_cam,
                    gt_size_aligned,
                    gt_rotation_aligned,
                    gt_translation_cam,
                )
        mask_score = mask_iou(pred_mask, gt_mask)

        pred_image = draw_bbox_axes_overlay_on_image(
            display_image,
            intrinsic,
            pred_rotation_aligned,
            pred_translation_cam,
            pred_bbox_obj,
            axis_length,
            AXIS_COLORS,
            PRED_BBOX_COLOR,
        )
        gt_image = None
        if gt_rotation_aligned is not None and gt_translation_cam is not None and gt_bbox_obj is not None:
            gt_image = draw_bbox_axes_overlay_on_image(
                display_image,
                intrinsic,
                gt_rotation_aligned,
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
                frame_token,
                display_image,
                display_depth,
                intrinsic,
                pred_rotation_cam=pred_rotation_aligned,
                pred_translation_cam=pred_translation_cam,
                pred_bbox_obj=pred_bbox_obj,
                gt_rotation_cam=gt_rotation_aligned,
                gt_translation_cam=gt_translation_cam,
                gt_bbox_obj=gt_bbox_obj,
                axis_length=axis_length,
                point_cloud_stride=point_cloud_stride,
            )

        pose_lines = [
            "### Prediction Summary",
            f"- checkpoint: `{self.checkpoint_path}`",
            f"- dataset: `REAL275 real_test`",
            f"- scene: `{scene_name}`",
            f"- frame: `{frame_token}`",
            f"- target crop input: `{target_crop}`",
            f"- predicted pose with GT bbox size: `{bool(pred_pose_gt_bbox and gt_size_aligned is not None)}`",
            f"- REAL275 object: `inst {inst_id}` · `{category}` · `{model_name}`",
            f"- reference render: `{ref_object_name}` (symmetry id `{ref_object_id_for_symmetry}`)",
            f"- predicted presence probability: `{presence_prob:.6f}` (logit `{presence_logit:.6f}`)",
            f"- predicted present @0.5: `{bool(pred_present)}`",
            f"- use depth input: `{use_depth_input}`",
            f"- use depth scale: `{use_depth_scale}`",
            "",
            "### Predicted Pose (OV9D-aligned object frame)",
            f"- depth_mean_scale (m): `{depth_mean_scale:.6f}` "
            f"({'GT depth mean' if use_depth_input else 'predicted depth mean'})",
            f"- applied depth scale to translation: `{use_depth_scale}`",
            f"- camera-frame translation (m): `{np.round(pred_translation_cam, 6).tolist()}`",
            f"- camera-frame rotation matrix: `{np.round(pred_rotation_aligned, 6).tolist()}`",
            f"- raw predicted aligned size xyz (m): `{np.round(raw_pred_size, 6).tolist()}`",
            f"- visualized predicted bbox size xyz (m): `{np.round(visualized_pred_size, 6).tolist()}`"
            + (
                " from GT bbox"
                if pred_pose_gt_bbox and gt_size_aligned is not None
                else (" clipped for display" if size_was_clipped else "")
            ),
        ]
        if gt_rotation_aligned is not None and gt_translation_cam is not None:
            pose_lines += [
                "",
                "### Ground Truth (aligned + native)",
                f"- aligned rotation matrix: `{np.round(gt_rotation_aligned, 6).tolist()}`",
                f"- native rotation matrix: `{np.round(gt_rotation_native, 6).tolist()}`",
                f"- camera-frame translation (m): `{np.round(gt_translation_cam, 6).tolist()}`",
                f"- aligned size xyz (m): "
                f"`{np.round(gt_size_aligned, 6).tolist() if gt_size_aligned is not None else 'N/A'}`",
                f"- native size xyz (m): "
                f"`{np.round(gt_size_native, 6).tolist() if gt_size_native is not None else 'N/A'}`",
                f"- R_align used: `{np.round(r_align, 6).tolist()}`",
                f"- decomposed scale from gt_RT: `{gt_scale_from_rt:.6f}`",
                f"- gt instance index in pkl: `{gt_inst_index}`",
            ]
        else:
            pose_lines += [
                "",
                "### Ground Truth",
                f"- gt pkl found: `{gt_path.is_file()}`",
                "- pose comparison skipped (missing gt_RTs or instance mismatch)",
            ]
        pose_lines += [
            "",
            "### Errors",
            f"- translation L2 (m): `{trans_error_dict['l2']:.6f}`" if trans_error_dict is not None else "- translation L2 (m): `N/A`",
            (
                f"- translation abs xyz (m): `{np.round(trans_error_dict['abs_xyz'], 6).tolist()}`"
                if trans_error_dict is not None else "- translation abs xyz (m): `N/A`"
            ),
            f"- raw predicted size L2 (m): `{size_error_dict['l2']:.6f}`" if size_error_dict is not None else "- raw predicted size L2 (m): `N/A`",
            (
                f"- raw predicted size abs xyz (m): `{np.round(size_error_dict['abs_xyz'], 6).tolist()}`"
                if size_error_dict is not None else "- raw predicted size abs xyz (m): `N/A`"
            ),
            (
                f"- symmetric rotation error (deg): `{rot_error_deg:.6f}` using `{sym_count}` symmetry candidates"
                if rot_error_deg is not None else "- symmetric rotation error (deg): `N/A`"
            ),
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

    def _run_housecat6d_inference(
        self,
        *,
        scene_name: str,
        frame_name: str,
        parsed: Dict[str, Any],
        use_depth_input: bool,
        use_depth_scale: bool,
        show_point_cloud_pose: bool,
        point_cloud_stride: int,
        target_crop: bool,
        pred_pose_gt_bbox: bool,
    ):
        object_index = int(parsed["object_index"])
        class_id = int(parsed["class_id"])
        model_name = str(parsed["model_name"])
        image_id = int(frame_name)
        scene_dir = self.scene_dir_for(HOUSECAT6D_DATASET_KEY, scene_name)
        category = self.housecat6d_category_id_to_name.get(class_id, "unknown")

        scene_tensor, depth_tensor, mask_tensor, display_image, display_depth, gt_mask, intrinsic = load_housecat6d_scene_frame_inputs(
            scene_dir,
            image_id,
            model_name,
            self.resolution,
            self.device,
            target_crop=target_crop,
        )
        object_tensor, _ = load_housecat6d_object_tensor(
            self.housecat6d_object_records_by_name,
            model_name,
            self.object_views,
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
            raise RuntimeError(f"Model output missing object_presence_logits: {sorted(outputs.keys())}")
        presence_logit = float(outputs["object_presence_logits"].reshape(-1)[0].detach().float().cpu())
        presence_prob = float(torch.sigmoid(torch.tensor(presence_logit)).item())
        pred_present = presence_prob >= 0.5

        pred_mask = None
        pred_mask_image = None
        if "object_mask" in outputs:
            pred_mask = outputs["object_mask"][0, 0].detach().float().cpu().numpy()
            pred_mask_image = mask_overlay_image(display_image, pred_mask, color=(255, 64, 64))
        gt_mask_image = mask_overlay_image(display_image, gt_mask, color=(80, 255, 120)) if gt_mask is not None else None

        if "object_pose" not in outputs or "object_translation" not in outputs:
            raise RuntimeError(f"Model output missing pose keys: {sorted(outputs.keys())}")
        pred_rot6d = outputs["object_pose"][0].detach().float().cpu().numpy()
        pred_translation_cam = outputs["object_translation"][0].detach().float().cpu().numpy().astype(np.float32)
        pred_rotation_aligned = rot6d_to_matrix(pred_rot6d).astype(np.float32)

        depth_mean_scale = self._compute_depth_mean_scale(
            display_depth,
            outputs.get("depth"),
            use_depth_input,
        )
        if use_depth_scale:
            pred_translation_cam = pred_translation_cam * np.float32(depth_mean_scale)

        pred_size_aligned = None
        if "object_size" in outputs:
            pred_size_aligned = outputs["object_size"].reshape(-1, 3)[0].detach().float().cpu().numpy().astype(np.float32)
        elif "object_size_log" in outputs:
            pred_size_aligned = np.exp(outputs["object_size_log"].reshape(-1, 3)[0].detach().float().cpu().numpy()).astype(np.float32)

        label = housecat6d_read_label(scene_dir / "labels" / f"{image_id:06d}_label.pkl")
        label_model_list = [str(name) for name in label.get("model_list", [])]
        if object_index >= len(label_model_list) or label_model_list[object_index] != model_name:
            raise RuntimeError(
                f"HouseCat6D label model mismatch at instance #{object_index}: "
                f"expected `{model_name}`, label has `{label_model_list[object_index] if object_index < len(label_model_list) else 'N/A'}`"
            )
        r_align = self._housecat6d_r_align_for_category(category)
        gt_rotation_native = np.asarray(label["rotations"][object_index], dtype=np.float32).reshape(3, 3)
        gt_translation_cam = np.asarray(label["translations"][object_index], dtype=np.float32).reshape(3)
        gt_rotation_aligned = (gt_rotation_native @ r_align.T).astype(np.float32)
        gt_size_native = np.asarray(label["gt_scales"][object_index], dtype=np.float32).reshape(3)
        gt_size_aligned = self._housecat6d_size_aligned(gt_size_native, category)

        raw_pred_size = pred_size_aligned if pred_size_aligned is not None else gt_size_aligned
        size_for_pred_box, size_was_clipped = clamp_predicted_size_for_bbox(raw_pred_size, gt_size_aligned)
        visualized_pred_size = gt_size_aligned if pred_pose_gt_bbox else size_for_pred_box
        pred_bbox_obj = centered_axis_bbox_corners(visualized_pred_size)
        gt_bbox_obj = centered_axis_bbox_corners(gt_size_aligned)
        axis_length = self._housecat6d_axis_length(gt_size_aligned)

        symmetry_object_id = self.housecat6d_object_name_to_id.get(model_name, class_id)
        rot_error_deg, sym_count = symmetric_rotation_error_degrees(
            pred_rotation_aligned,
            gt_rotation_aligned,
            int(symmetry_object_id),
            self.symmetry_info_path,
            int(self.cfg.get("object_srt_symmetry_continuous_steps", 72)),
            dataset_label=HOUSECAT6D_DATASET_LABEL,
        )
        trans_error_dict = translation_error(pred_translation_cam, gt_translation_cam)
        size_error_dict = translation_error(raw_pred_size, gt_size_aligned)
        mask_score = mask_iou(pred_mask, gt_mask)
        raw_bbox_iou_details = bbox_iou_3d_details(
            raw_pred_size,
            pred_rotation_aligned,
            pred_translation_cam,
            gt_size_aligned,
            gt_rotation_aligned,
            gt_translation_cam,
        )
        visualized_bbox_iou_details = bbox_iou_3d_details(
            visualized_pred_size,
            pred_rotation_aligned,
            pred_translation_cam,
            gt_size_aligned,
            gt_rotation_aligned,
            gt_translation_cam,
        )

        pred_image = draw_bbox_axes_overlay_on_image(
            display_image,
            intrinsic,
            pred_rotation_aligned,
            pred_translation_cam,
            pred_bbox_obj,
            axis_length,
            AXIS_COLORS,
            PRED_BBOX_COLOR,
        )
        gt_image = draw_bbox_axes_overlay_on_image(
            display_image,
            intrinsic,
            gt_rotation_aligned,
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
                pred_rotation_cam=pred_rotation_aligned,
                pred_translation_cam=pred_translation_cam,
                pred_bbox_obj=pred_bbox_obj,
                gt_rotation_cam=gt_rotation_aligned,
                gt_translation_cam=gt_translation_cam,
                gt_bbox_obj=gt_bbox_obj,
                axis_length=axis_length,
                point_cloud_stride=point_cloud_stride,
            )

        pose_lines = [
            "### Prediction Summary",
            f"- checkpoint: `{self.checkpoint_path}`",
            f"- dataset: `HouseCat6D test/val`",
            f"- scene: `{scene_name}`",
            f"- frame: `{image_id:06d}`",
            f"- target crop input: `{target_crop}`",
            f"- predicted pose with GT bbox size: `{pred_pose_gt_bbox}`",
            f"- HouseCat6D object: `{model_name}` · `{category}` (instance #{object_index})",
            f"- symmetry id: `{symmetry_object_id}`",
            f"- predicted presence probability: `{presence_prob:.6f}` (logit `{presence_logit:.6f}`)",
            f"- predicted present @0.5: `{bool(pred_present)}`",
            f"- use depth input: `{use_depth_input}`",
            f"- use depth scale: `{use_depth_scale}`",
            "",
            "### Predicted Pose (OV9D-aligned object frame)",
            f"- depth_mean_scale (m): `{depth_mean_scale:.6f}` "
            f"({'GT depth mean' if use_depth_input else 'predicted depth mean'})",
            f"- applied depth scale to translation: `{use_depth_scale}`",
            f"- camera-frame translation (m): `{np.round(pred_translation_cam, 6).tolist()}`",
            f"- camera-frame rotation matrix: `{np.round(pred_rotation_aligned, 6).tolist()}`",
            f"- raw predicted aligned size xyz (m): `{np.round(raw_pred_size, 6).tolist()}`",
            f"- visualized predicted bbox size xyz (m): `{np.round(visualized_pred_size, 6).tolist()}`"
            + (" from GT bbox" if pred_pose_gt_bbox else (" clipped for display" if size_was_clipped else "")),
            "",
            "### Ground Truth (aligned + native)",
            f"- aligned rotation matrix: `{np.round(gt_rotation_aligned, 6).tolist()}`",
            f"- native cam_R_m2c: `{np.round(gt_rotation_native, 6).tolist()}`",
            f"- camera-frame translation (m): `{np.round(gt_translation_cam, 6).tolist()}`",
            f"- aligned size xyz (m): `{np.round(gt_size_aligned, 6).tolist()}`",
            f"- native size xyz (m): `{np.round(gt_size_native, 6).tolist()}`",
            f"- R_align used: `{np.round(r_align, 6).tolist()}`",
            "",
            "### Errors",
            f"- translation L2 (m): `{trans_error_dict['l2']:.6f}`",
            f"- translation abs xyz (m): `{np.round(trans_error_dict['abs_xyz'], 6).tolist()}`",
            f"- raw predicted aligned size L2 (m): `{size_error_dict['l2']:.6f}`",
            f"- raw predicted aligned size abs xyz (m): `{np.round(size_error_dict['abs_xyz'], 6).tolist()}`",
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
            "# OmniVGGT 6D Pose Demo · YCB-V test + REAL275 real_test + HouseCat6D test/val",
            f"- config: `{app.config_path}`",
            f"- YCB-V root: `{app.ycbv_root}`",
            f"- YCB-V test split: `{app.ycbv_test_root}`",
            f"- YCB-V aligned object refs: `{app.ycbv_object_image_root}`",
            f"- REAL275 root: `{app.real275_root}`",
            f"- REAL275 real_test split: `{app.real275_test_root}`",
            f"- REAL275 GT pkls: `{app.real275_gt_root}`",
            f"- REAL275 aligned object refs: `{app.real275_object_image_root}`",
            f"- REAL275 obj models (real_test): `{app.real275_obj_models_root}`",
            f"- HouseCat6D root: `{app.housecat6d_root}`",
            f"- HouseCat6D scenes: `{list(app.housecat6d_scene_names)}`",
            f"- HouseCat6D aligned object refs: `{app.housecat6d_object_image_root}`",
            f"- align json: `{app.align_json_path}`",
            f"- loaded checkpoint: `{app.checkpoint_path}`",
            f"- object views: `{app.object_views}`",
            f"- inference resolution: `{app.resolution}`",
            "",
            "Pred / GT 全部以 OV9D-aligned 物體坐標系顯示。YCB-V GT 由 `cam_R_m2c @ R_align^T`；",
            "REAL275 real_test GT 由 `gts/real_test/*.pkl` 的 `gt_RTs` 解出 `R = sR/scale` 後再乘 `R_align^T`；",
            "HouseCat6D GT 直接從 `labels/*_label.pkl` 的 `rotations / translations / gt_scales` 取出後乘 `R_align^T`。",
            "Object reference 必須是 `real275_aligned_object_refs/` 已存在的物件；real_test 真實物體 (e.g. `mug_daniel_norm`) 不在 ref pool，",
            "因此 dropdown 會列出同類別的所有 train-time aligned refs 供選擇。HouseCat6D 則直接以 label 中的 `model_name` 對應到 ref。",
        ]
    )

    with gr.Blocks(title="OmniVGGT YCB-V + REAL275 Demo", css=demo_css) as demo:
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

        def refresh_inputs(dataset_key, scene_name, frame_name, object_value):
            if not dataset_key or not scene_name or not frame_name or not object_value:
                return None, []
            return app.input_gallery(dataset_key, scene_name, frame_name, object_value)

        def refresh_dataset_controls(
            dataset_key: str,
            preferred_scene: str | None = None,
            preferred_frame: str | None = None,
            preferred_object: str | None = None,
        ):
            scenes = app.get_scene_choices(dataset_key)
            scene_value = preferred_scene if preferred_scene in scenes else (scenes[0] if scenes else None)
            frames = app.get_frame_choices(dataset_key, scene_value) if scene_value else []
            frame_value = preferred_frame if preferred_frame in frames else (frames[0] if frames else None)
            object_choices = (
                app.object_choices_for_frame(dataset_key, scene_value, frame_value)
                if scene_value and frame_value
                else []
            )
            object_values = [value for _, value in object_choices]
            object_value = preferred_object if preferred_object in object_values else (object_values[0] if object_values else None)
            scene_image, object_gallery = refresh_inputs(dataset_key, scene_value, frame_value, object_value)
            return (
                gr.update(choices=scenes, value=scene_value),
                gr.update(choices=frames, value=frame_value),
                gr.update(choices=object_choices, value=object_value),
                scene_image,
                object_gallery,
            )

        def refresh_scene_controls(
            dataset_key: str,
            scene_name: str,
            preferred_frame: str | None = None,
            preferred_object: str | None = None,
        ):
            frames = app.get_frame_choices(dataset_key, scene_name) if scene_name else []
            frame_value = preferred_frame if preferred_frame in frames else (frames[0] if frames else None)
            object_choices = (
                app.object_choices_for_frame(dataset_key, scene_name, frame_value)
                if scene_name and frame_value
                else []
            )
            object_values = [value for _, value in object_choices]
            object_value = preferred_object if preferred_object in object_values else (object_values[0] if object_values else None)
            return gr.update(choices=frames, value=frame_value), gr.update(choices=object_choices, value=object_value)

        def refresh_frame_controls(
            dataset_key: str,
            scene_name: str,
            frame_name: str,
            preferred_object: str | None = None,
        ):
            if not (dataset_key and scene_name and frame_name):
                return gr.update()
            object_choices = app.object_choices_for_frame(dataset_key, scene_name, frame_name)
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
        if default_object is not None and default_frame is not None:
            demo.load(
                refresh_inputs,
                inputs=[dataset_dropdown, scene_dropdown, frame_dropdown, object_dropdown],
                outputs=[scene_input_image, object_input_gallery],
            )
    return demo


def main():
    parser = argparse.ArgumentParser(
        description="Gradio demo for OmniVGGT 6D pose inference on YCB-V test and REAL275 real_test data"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--ycbv-root", type=Path, default=DEFAULT_YCBV_ROOT)
    parser.add_argument("--ycbv-test-root", type=Path, default=DEFAULT_YCBV_TEST_ROOT)
    parser.add_argument("--ycbv-object-image-root", type=Path, default=DEFAULT_YCBV_OBJECT_IMAGE_ROOT)
    parser.add_argument("--real275-root", type=Path, default=DEFAULT_REAL275_ROOT)
    parser.add_argument("--real275-test-root", type=Path, default=DEFAULT_REAL275_TEST_ROOT)
    parser.add_argument("--real275-gt-root", type=Path, default=DEFAULT_REAL275_GT_ROOT)
    parser.add_argument("--real275-obj-models-root", type=Path, default=DEFAULT_REAL275_OBJ_MODELS_ROOT)
    parser.add_argument("--real275-object-image-root", type=Path, default=DEFAULT_REAL275_OBJECT_IMAGE_ROOT)
    parser.add_argument("--housecat6d-root", type=Path, default=DEFAULT_HOUSECAT6D_ROOT)
    parser.add_argument(
        "--housecat6d-object-image-root",
        type=Path,
        default=DEFAULT_HOUSECAT6D_OBJECT_IMAGE_ROOT,
    )
    parser.add_argument(
        "--housecat6d-scenes",
        type=str,
        default=",".join(DEFAULT_HOUSECAT6D_SCENE_NAMES),
        help="Comma-separated list of HouseCat6D scene directory names under --housecat6d-root.",
    )
    parser.add_argument("--align-json", type=Path, default=DEFAULT_ALIGN_JSON)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    housecat6d_scene_names = [name.strip() for name in str(args.housecat6d_scenes).split(",") if name.strip()]
    app = DemoApp(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        ycbv_root=args.ycbv_root,
        ycbv_test_root=args.ycbv_test_root,
        ycbv_object_image_root=args.ycbv_object_image_root,
        real275_root=args.real275_root,
        real275_test_root=args.real275_test_root,
        real275_gt_root=args.real275_gt_root,
        real275_obj_models_root=args.real275_obj_models_root,
        real275_object_image_root=args.real275_object_image_root,
        housecat6d_root=args.housecat6d_root,
        housecat6d_object_image_root=args.housecat6d_object_image_root,
        housecat6d_scene_names=housecat6d_scene_names,
        align_json=args.align_json,
    )
    demo = build_demo(app)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[
            str(app.ycbv_root),
            str(app.ycbv_test_root),
            str(app.ycbv_object_image_root),
            str(app.real275_root),
            str(app.real275_test_root),
            str(app.real275_object_image_root),
            str(app.housecat6d_root),
            str(app.housecat6d_object_image_root),
        ],
    )


if __name__ == "__main__":
    main()
