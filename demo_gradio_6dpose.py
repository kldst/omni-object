import argparse
import os
import re
import runpy
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as load_safetensors_file

from omnivggt.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from omnivggt.datasets.utils.misc import threshold_depth_map
from omnivggt.models.omnivggt import OmniVGGT
from omnivggt.utils.image import ImgNorm

# python3 demo_gradio_6dpose.py --port 7860
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train.py"
OBJ_ROOT = PROJECT_ROOT.parent / "obj"
DEFAULT_PRETRAIN_MODEL = Path(
    "/mnt/train-data-4-hdd/yian/6dpose_obj/OmniVGGT-official/outputs/"
    "0405_omnivggt_single_image_pose_5sameobject/checkpoint-40-10000/model.safetensors"
)
PREDICTION_ROTATION_FIX = np.diag([1.0, 1.0, -1.0]).astype(np.float32)


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


def resolve_dataset_settings(cfg: Dict) -> Dict:
    dataset_expr = str(cfg.get("train_dataset", ""))
    dataset_root = Path(parse_dataset_ctor_arg(dataset_expr, "dataset_location", default="")).expanduser()
    object_root = Path(
        parse_dataset_ctor_arg(dataset_expr, "OBJECT_INPUT_ROOT", default=str(dataset_root / "object_space_rgb"))
    ).expanduser()
    selected_views = tuple(parse_dataset_ctor_arg(dataset_expr, "selected_views", default=(1,)))
    object_input_views = tuple(parse_dataset_ctor_arg(dataset_expr, "object_input_views", default=(1, 3, 4)))
    resolution = tuple(cfg.get("resolution", (518, 518)))
    exp_name = str(cfg.get("exp_name", ""))
    return {
        "dataset_root": dataset_root,
        "object_root": object_root,
        "selected_views": selected_views,
        "object_input_views": object_input_views,
        "resolution": resolution,
        "exp_name": exp_name,
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

    model_url = Path(str(cfg.get("model_url", ""))).expanduser()
    if model_url.is_file():
        return model_url
    raise FileNotFoundError("Unable to locate a local checkpoint from config or outputs directory.")


def build_model_from_config(cfg: Dict, checkpoint_path: Path, device: torch.device) -> OmniVGGT:
    model = OmniVGGT(
        enable_camera=cfg.get("enable_camera", True),
        enable_point=cfg.get("enable_point", True),
        enable_depth=cfg.get("enable_depth", True),
        enable_object_srt=cfg.get("enable_object_srt", False),
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


def decode_name(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def dataset_scale(name_or_path: str) -> float:
    value = str(name_or_path).lower()
    if "ycbv" in value:
        return 0.002
    if "handal" in value:
        return 0.0015
    if "hope" in value:
        return 0.003
    if "rupac" in value:
        return 0.002
    return 1.0


def image_filename(view_idx: int, suffix: str = ".jpg") -> str:
    return "Main_Camera.jpg" if int(view_idx) == 0 else f"Main_Camera_({int(view_idx)}){suffix}"


def object_image_filename(view_idx: int) -> str:
    return "Main_Camera_rgb.png" if int(view_idx) == 0 else f"Main_Camera_({int(view_idx)})_rgb.png"


def image_path_for_view(dataset_root: Path, run_name: str, view_idx: int) -> Path:
    path = dataset_root / "out_image" / run_name / image_filename(view_idx)
    if not path.is_file():
        raise FileNotFoundError(f"Missing scene image: {path}")
    return path


def camera_param_path(dataset_root: Path, run_name: str, view_idx: int) -> Path:
    filename = "camera_Main_Camera.npz" if int(view_idx) == 0 else f"camera_Main_Camera_({int(view_idx)}).npz"
    path = dataset_root / "out_cam_param" / run_name / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing camera param file: {path}")
    return path


def load_camera_params(dataset_root: Path, run_name: str, view_idx: int) -> Tuple[np.ndarray, np.ndarray, int, int]:
    data = np.load(camera_param_path(dataset_root, run_name, view_idx))
    intrinsic = np.asarray(data["intrinsics.K_flat9"], dtype=np.float32).reshape(3, 3)
    extrinsic = np.asarray(data["extrinsics.opencv.worldToCamera16"], dtype=np.float32).reshape(4, 4)[:3, :4]
    width = int(np.asarray(data["image.width"]).reshape(-1)[0])
    height = int(np.asarray(data["image.height"]).reshape(-1)[0])
    return intrinsic, extrinsic, width, height


def list_runs(dataset_root: Path) -> List[str]:
    out_image_root = dataset_root / "out_image"
    if not out_image_root.is_dir():
        return []
    return sorted(path.name for path in out_image_root.iterdir() if path.is_dir() and path.name.startswith("run_"))


def list_scene_views_for_run(dataset_root: Path, run_name: str) -> List[int]:
    run_dir = dataset_root / "out_image" / run_name
    if not run_dir.is_dir():
        return []
    views = []
    for path in run_dir.iterdir():
        if not path.is_file():
            continue
        name = path.name
        if name == "Main_Camera.jpg":
            views.append(0)
            continue
        match = re.match(r"Main_Camera_\((\d+)\)\.jpg$", name)
        if match:
            views.append(int(match.group(1)))
    return sorted(set(views))


def load_pose_lookup(dataset_root: Path, run_name: str) -> Dict[str, Dict[str, np.ndarray]]:
    pose_path = dataset_root / "out_pose" / f"{run_name}.npz"
    data = np.load(pose_path, allow_pickle=False)
    names = [decode_name(name) for name in data["names"]]
    positions = data["positions"].astype(np.float32)
    quats = data["rot_quat_wxyz"].astype(np.float32)
    result = {}
    for idx, name in enumerate(names):
        result[name] = {
            "translation": positions[idx],
            "quat_wxyz": quats[idx],
        }
    return result


def list_objects_for_run(dataset_root: Path, run_name: str) -> List[str]:
    return sorted(load_pose_lookup(dataset_root, run_name).keys())


def object_image_path(object_root: Path, object_name: str, view_idx: int) -> Path:
    path = object_root / object_name / object_image_filename(view_idx)
    if not path.is_file():
        raise FileNotFoundError(f"Missing object image: {path}")
    return path


def object_ply_path(object_name: str) -> Path:
    dataset_name, obj_id = object_name.split("_obj_")
    path = OBJ_ROOT / dataset_name / f"obj_{obj_id}.ply"
    if not path.is_file():
        raise FileNotFoundError(f"Missing object point cloud: {path}")
    return path


def load_ply_xyz(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        if handle.readline().decode("ascii", errors="ignore").strip() != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        fmt = None
        vertex_count = None
        properties = []
        in_vertex = False
        while True:
            line = handle.readline().decode("ascii", errors="ignore")
            if not line:
                raise ValueError(f"Unexpected EOF in header: {path}")
            line = line.strip()
            if line.startswith("format "):
                fmt = line.split()[1]
            elif line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
                in_vertex = True
            elif line.startswith("element "):
                in_vertex = False
            elif line.startswith("property ") and in_vertex:
                parts = line.split()
                properties.append((parts[1], parts[2]))
            elif line == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"PLY missing vertex count: {path}")
        xyz = []
        if fmt == "ascii":
            for _ in range(vertex_count):
                row = handle.readline().decode("ascii", errors="ignore").strip().split()
                xyz.append([float(row[0]), float(row[1]), float(row[2])])
            return np.asarray(xyz, dtype=np.float32)
        if fmt != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format {fmt}: {path}")
        type_map = {
            "char": "b",
            "uchar": "B",
            "int8": "b",
            "uint8": "B",
            "short": "h",
            "ushort": "H",
            "int16": "h",
            "uint16": "H",
            "int": "i",
            "uint": "I",
            "int32": "i",
            "uint32": "I",
            "float": "f",
            "float32": "f",
            "double": "d",
            "float64": "d",
        }
        fmt_str = "<" + "".join(type_map[dtype] for dtype, _ in properties)
        row_size = struct.calcsize(fmt_str)
        for _ in range(vertex_count):
            row = struct.unpack(fmt_str, handle.read(row_size))
            xyz.append([row[0], row[1], row[2]])
        return np.asarray(xyz, dtype=np.float32)


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float64)[:, :2].reshape(-1)


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


def rotation_error_degrees(pred_rot6d: np.ndarray, gt_quat_wxyz: np.ndarray) -> float:
    pred_rot = rot6d_to_matrix(pred_rot6d)
    gt_rot = quat_wxyz_to_matrix(gt_quat_wxyz)
    rel = pred_rot @ gt_rot.T
    cos_theta = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def translation_error(pred_t: np.ndarray, gt_t: np.ndarray) -> Dict[str, List[float] | float]:
    diff = np.asarray(pred_t, dtype=np.float64) - np.asarray(gt_t, dtype=np.float64)
    return {
        "l2": float(np.linalg.norm(diff)),
        "abs_xyz": np.abs(diff).tolist(),
        "signed_xyz": diff.tolist(),
    }


def load_rgb_as_tensor(image_path: Path, resolution: Sequence[int], device: torch.device) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    width, height = int(resolution[0]), int(resolution[1])
    image = image.resize((width, height), getattr(Image, "Resampling", Image).LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
    return tensor.to(device)


def depth_path_for_view(dataset_root: Path, run_name: str, view_idx: int) -> Path:
    filename = "Main_Camera_depth.png" if int(view_idx) == 0 else f"Main_Camera_({int(view_idx)})_depth.png"
    path = dataset_root / "out_depth" / run_name / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing scene depth: {path}")
    return path


class DemoScenePreprocessor(BaseStereoViewDataset):
    def __init__(self, resolution):
        super().__init__(dset="demo", resolution=resolution, transform=ImgNorm, seed=0)


def load_scene_inputs(dataset_root: Path, run_name: str, scene_view: int, resolution, device):
    image_path = image_path_for_view(dataset_root, run_name, scene_view)
    depth_path = depth_path_for_view(dataset_root, run_name, scene_view)
    image = Image.open(image_path).convert("RGB")
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Failed to read depth image: {depth_path}")
    if depth_raw.dtype != np.uint16:
        raise ValueError(f"Expected uint16 R16 depth image, got {depth_raw.dtype} for {depth_path}")

    depthmap = depth_raw.view(np.float16).astype(np.float32)
    depthmap[~np.isfinite(depthmap)] = 0.0
    depthmap[depthmap < 0] = 0.0
    depthmap = threshold_depth_map(depthmap, max_percentile=99, min_percentile=-1)

    intrinsics, _, _, _ = load_camera_params(dataset_root, run_name, scene_view)
    processor = DemoScenePreprocessor(resolution=resolution)
    rng = np.random.default_rng(seed=0)
    image, depthmap, _ = processor._crop_resize_if_necessary(
        image=image,
        depthmap=depthmap,
        intrinsics=intrinsics,
        resolution=resolution,
        rng=rng,
        info=str(image_path),
    )

    image_tensor = processor.transform(image).unsqueeze(0).unsqueeze(0).to(device)
    depth_tensor = torch.from_numpy(np.ascontiguousarray(depthmap.astype(np.float32)))[None, None, :, :, None].to(device)
    mask_tensor = torch.from_numpy((depthmap > 0).astype(np.float32))[None, None, :, :].to(device)
    return image_tensor, depth_tensor, mask_tensor


def load_object_tensor(object_root: Path, object_name: str, object_views: Sequence[int], resolution, device) -> torch.Tensor:
    images = [load_rgb_as_tensor(object_image_path(object_root, object_name, view_idx), resolution, device) for view_idx in object_views]
    return torch.stack(images, dim=0).unsqueeze(0)


def compute_bbox_corners(points_obj: np.ndarray) -> np.ndarray:
    pmin = points_obj.min(axis=0)
    pmax = points_obj.max(axis=0)
    return np.asarray(
        [
            [pmin[0], pmin[1], pmin[2]],
            [pmax[0], pmin[1], pmin[2]],
            [pmax[0], pmax[1], pmin[2]],
            [pmin[0], pmax[1], pmin[2]],
            [pmin[0], pmin[1], pmax[2]],
            [pmax[0], pmin[1], pmax[2]],
            [pmax[0], pmax[1], pmax[2]],
            [pmin[0], pmax[1], pmax[2]],
        ],
        dtype=np.float32,
    )


def project_points_with_mask(points_world: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray):
    points_cam = points_world @ extrinsic[:, :3].T + extrinsic[:, 3][None, :]
    z = points_cam[:, 2]
    valid = z > 1e-6
    uv = np.full((points_world.shape[0], 2), np.nan, dtype=np.float32)
    if np.any(valid):
        uvw = points_cam[valid] @ intrinsic.T
        uv[valid] = (uvw[:, :2] / uvw[:, 2:3]).astype(np.float32)
    return uv, valid


def draw_projected_bbox(
    image_path: Path,
    uv: np.ndarray,
    valid: np.ndarray,
    width: int,
    height: int,
    center_uv: np.ndarray,
    center_valid: bool,
    axis_uv: np.ndarray,
    axis_valid: np.ndarray,
) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[1] != width or image.shape[0] != height:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    overlay = image.copy()
    inside = valid.copy()
    inside &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    for start_idx, end_idx in edges:
        if inside[start_idx] and inside[end_idx]:
            p1 = tuple(np.round(uv[start_idx]).astype(np.int32))
            p2 = tuple(np.round(uv[end_idx]).astype(np.int32))
            cv2.line(overlay, p1, p2, (0, 255, 0), 2)

    if center_valid:
        center = tuple(np.round(center_uv).astype(np.int32))
        axis_colors = [(255, 64, 64), (0, 255, 255), (255, 215, 0)]
        for idx, color in enumerate(axis_colors):
            if axis_valid[idx]:
                end = tuple(np.round(axis_uv[idx]).astype(np.int32))
                cv2.arrowedLine(overlay, center, end, color, 3, tipLength=0.18)
    return overlay


def build_bbox_projection_gallery(
    dataset_root: Path,
    run_name: str,
    object_name: str,
    translation_world: np.ndarray,
    rotation_world: np.ndarray,
    preview_views: Iterable[int],
) -> List[Tuple[np.ndarray, str]]:
    points_obj = load_ply_xyz(object_ply_path(object_name)) * float(dataset_scale(object_name))
    bbox_obj = compute_bbox_corners(points_obj)
    bbox_world = bbox_obj @ rotation_world.T + translation_world[None, :]

    center_obj = np.zeros((1, 3), dtype=np.float32)
    axis_len = max(np.linalg.norm(bbox_obj.max(axis=0) - bbox_obj.min(axis=0)) * 0.25, 1e-3)
    axis_obj = np.asarray([[axis_len, 0.0, 0.0], [0.0, axis_len, 0.0], [0.0, 0.0, axis_len]], dtype=np.float32)
    center_world = center_obj @ rotation_world.T + translation_world[None, :]
    axis_world = axis_obj @ rotation_world.T + translation_world[None, :]

    gallery = []
    for view_idx in preview_views:
        image_path = image_path_for_view(dataset_root, run_name, view_idx)
        intrinsic, extrinsic, width, height = load_camera_params(dataset_root, run_name, view_idx)
        uv, valid = project_points_with_mask(bbox_world, extrinsic, intrinsic)
        center_uv, center_valid = project_points_with_mask(center_world, extrinsic, intrinsic)
        axis_uv, axis_valid = project_points_with_mask(axis_world, extrinsic, intrinsic)
        overlay = draw_projected_bbox(
            image_path,
            uv,
            valid,
            width,
            height,
            center_uv[0],
            bool(center_valid[0]),
            axis_uv,
            np.asarray(axis_valid, dtype=bool),
        )
        gallery.append((overlay, f"View {view_idx}"))
    return gallery


def camera_to_world_pose(dataset_root: Path, run_name: str, view_idx: int) -> np.ndarray:
    data = np.load(camera_param_path(dataset_root, run_name, view_idx))
    return np.asarray(data["extrinsics.opencv.cameraToWorld16"], dtype=np.float32).reshape(4, 4)


def format_pose_markdown(
    *,
    run_name: str,
    object_name: str,
    reference_view: int,
    checkpoint_path: Path,
    pred_translation_cam: np.ndarray,
    pred_rot6d_cam: np.ndarray,
    pred_translation_world: np.ndarray,
    pred_rotation_world: np.ndarray,
    gt_translation_cam: np.ndarray,
    gt_rotation_cam: np.ndarray,
    gt_translation_world: np.ndarray,
    gt_quat_wxyz: np.ndarray,
) -> str:
    rot_error_deg = rotation_error_degrees(matrix_to_rot6d(pred_rotation_world), gt_quat_wxyz)
    trans_error = translation_error(pred_translation_cam, gt_translation_cam)
    gt_rot_world = quat_wxyz_to_matrix(gt_quat_wxyz)
    return "\n".join(
        [
            f"### Prediction Summary",
            f"- checkpoint: `{checkpoint_path}`",
            f"- run: `{run_name}`",
            f"- object: `{object_name}`",
            f"- reference scene view: `{reference_view}`",
            "",
            f"### Predicted Pose",
            f"- camera-frame translation: `{np.round(pred_translation_cam, 6).tolist()}`",
            f"- camera-frame rot6d: `{np.round(pred_rot6d_cam, 6).tolist()}`",
            f"- world translation: `{np.round(pred_translation_world, 6).tolist()}`",
            f"- world rotation matrix: `{np.round(pred_rotation_world, 6).tolist()}`",
            "",
            f"### Ground Truth",
            f"- camera-frame translation: `{np.round(gt_translation_cam, 6).tolist()}`",
            f"- camera-frame rotation matrix: `{np.round(gt_rotation_cam, 6).tolist()}`",
            f"- world translation: `{np.round(gt_translation_world, 6).tolist()}`",
            f"- world rotation matrix: `{np.round(gt_rot_world, 6).tolist()}`",
            "",
            f"### Errors",
            f"- camera-frame translation L2: `{trans_error['l2']:.6f}`",
            f"- camera-frame translation abs xyz: `{np.round(trans_error['abs_xyz'], 6).tolist()}`",
            f"- camera-frame translation signed xyz: `{np.round(trans_error['signed_xyz'], 6).tolist()}`",
            f"- rotation error (deg): `{rot_error_deg:.6f}`",
        ]
    )


class DemoApp:
    def __init__(self, config_path: Path, checkpoint_path: str | None, use_gt_pose_for_prediction: bool = False):
        self.cfg = load_config(config_path)
        self.dataset_settings = resolve_dataset_settings(self.cfg)
        self.dataset_root = Path(self.dataset_settings["dataset_root"])
        self.object_root = Path(self.dataset_settings["object_root"])
        self.scene_views = tuple(int(v) for v in self.dataset_settings["selected_views"])
        self.object_views = tuple(int(v) for v in self.dataset_settings["object_input_views"])
        self.resolution = tuple(int(v) for v in self.dataset_settings["resolution"])
        self.use_gt_pose_for_prediction = bool(use_gt_pose_for_prediction)
        self.run_choices = list_runs(self.dataset_root)
        if not self.run_choices:
            raise RuntimeError(f"No runs found under {self.dataset_root / 'out_image'}")
        self.preview_views = self.scene_views
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = resolve_checkpoint_path(self.cfg, checkpoint_path)
        self.model = None
        self.available_checkpoints = self.discover_checkpoints()
        self.load_checkpoint(self.checkpoint_path)

    def discover_checkpoints(self) -> List[str]:
        candidates = set()
        if DEFAULT_PRETRAIN_MODEL.is_file():
            candidates.add(str(DEFAULT_PRETRAIN_MODEL))
        latest_ckpt = latest_checkpoint_for_experiment(str(self.cfg.get("exp_name", "")))
        if latest_ckpt is not None:
            candidates.add(str(latest_ckpt))
        model_url = Path(str(self.cfg.get("model_url", ""))).expanduser()
        if model_url.is_file():
            candidates.add(str(model_url))
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

    def get_object_choices(self, run_name: str) -> List[str]:
        return list_objects_for_run(self.dataset_root, run_name)

    def get_scene_view_choices(self, run_name: str) -> List[int]:
        return list_scene_views_for_run(self.dataset_root, run_name)

    def input_gallery(self, run_name: str, object_name: str, scene_view: int):
        scene_gallery = [
            (str(image_path_for_view(self.dataset_root, run_name, int(scene_view))), f"Scene view {int(scene_view)}")
        ]
        object_gallery = [
            (str(object_image_path(self.object_root, object_name, view_idx)), f"Object view {view_idx}")
            for view_idx in self.object_views
        ]
        return scene_gallery, object_gallery

    def run_inference(self, run_name: str, object_name: str, scene_view: int, use_depth_input: bool):
        pose_lookup = load_pose_lookup(self.dataset_root, run_name)
        if object_name not in pose_lookup:
            raise ValueError(f"Object {object_name} is not present in {run_name}")
        scene_view = int(scene_view)
        use_depth_input = bool(use_depth_input)

        scene_tensor, depth_tensor, mask_tensor = load_scene_inputs(
            self.dataset_root,
            run_name,
            scene_view,
            self.resolution,
            self.device,
        )
        object_tensor = load_object_tensor(self.object_root, object_name, self.object_views, self.resolution, self.device)

        with torch.inference_mode():
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
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
            raise RuntimeError(f"Model output does not contain object pose keys: {sorted(outputs.keys())}")

        pred_rot6d_cam = outputs["object_pose"][0].detach().float().cpu().numpy()
        pred_translation_cam = outputs["object_translation"][0].detach().float().cpu().numpy()
        pred_rotation_cam = rot6d_to_matrix(pred_rot6d_cam).astype(np.float32)
        pred_rotation_cam = pred_rotation_cam @ PREDICTION_ROTATION_FIX
        pred_rot6d_cam = matrix_to_rot6d(pred_rotation_cam).astype(np.float32)

        reference_view = int(scene_view)
        cam_to_world = camera_to_world_pose(self.dataset_root, run_name, reference_view)
        pred_rotation_world = cam_to_world[:3, :3] @ pred_rotation_cam
        pred_translation_world = cam_to_world[:3, :3] @ pred_translation_cam + cam_to_world[:3, 3]

        gt_translation_world = pose_lookup[object_name]["translation"]
        gt_quat_wxyz = pose_lookup[object_name]["quat_wxyz"]
        gt_rotation_world = quat_wxyz_to_matrix(gt_quat_wxyz).astype(np.float32)
        world_to_cam = load_camera_params(self.dataset_root, run_name, reference_view)[1]
        gt_rotation_cam = (world_to_cam[:, :3] @ gt_rotation_world).astype(np.float32)
        gt_translation_cam = (
            world_to_cam[:, :3] @ gt_translation_world.astype(np.float32) + world_to_cam[:, 3]
        ).astype(np.float32)
        if self.use_gt_pose_for_prediction:
            pred_rotation_world = gt_rotation_world
            pred_translation_world = gt_translation_world.astype(np.float32)
            pred_rotation_cam = cam_to_world[:3, :3].T @ pred_rotation_world
            pred_translation_cam = cam_to_world[:3, :3].T @ (pred_translation_world - cam_to_world[:3, 3])

        pred_gallery = build_bbox_projection_gallery(
            self.dataset_root,
            run_name,
            object_name,
            pred_translation_world.astype(np.float32),
            pred_rotation_world.astype(np.float32),
            [scene_view],
        )
        gt_gallery = build_bbox_projection_gallery(
            self.dataset_root,
            run_name,
            object_name,
            gt_translation_world.astype(np.float32),
            quat_wxyz_to_matrix(gt_quat_wxyz).astype(np.float32),
            [scene_view],
        )
        summary = format_pose_markdown(
            run_name=run_name,
            object_name=object_name,
            reference_view=reference_view,
            checkpoint_path=self.checkpoint_path,
            pred_translation_cam=pred_translation_cam,
            pred_rot6d_cam=pred_rot6d_cam,
            pred_translation_world=pred_translation_world,
            pred_rotation_world=pred_rotation_world,
            gt_translation_cam=gt_translation_cam,
            gt_rotation_cam=gt_rotation_cam,
            gt_translation_world=gt_translation_world,
            gt_quat_wxyz=gt_quat_wxyz,
        )
        summary = summary + f"\n- use depth input: `{use_depth_input}`"
        return summary, pred_gallery, gt_gallery


def build_demo(app: DemoApp, image_focused_layout: bool = False):
    default_run = app.run_choices[0]
    default_objects = app.get_object_choices(default_run)
    default_object = default_objects[0] if default_objects else None
    default_scene_views = app.get_scene_view_choices(default_run)
    default_scene_view = default_scene_views[0] if default_scene_views else (app.scene_views[0] if app.scene_views else 1)
    demo_css = """
    .gradio-container { max-width: 100% !important; }
    """
    if image_focused_layout:
        demo_css += """
        .gradio-container {
            padding-top: 8px !important;
            padding-left: 10px !important;
            padding-right: 10px !important;
            padding-bottom: 8px !important;
        }
        #demo_info { margin-bottom: 6px !important; }
        #demo_info p { margin: 0 !important; }
        #input_row, #compare_row { gap: 8px !important; }
        #scene_inputs, #object_inputs { min-height: 34vh !important; }
        #pred_projection, #gt_projection { min-height: 56vh !important; }
        """

    with gr.Blocks(title="OmniVGGT 6D Pose Demo", css=demo_css) as demo:
        info_markdown = "\n".join(
            [
                "# OmniVGGT 6D Pose Demo",
                f"- config: `{DEFAULT_CONFIG_PATH}`",
                f"- dataset: `{app.dataset_root}`",
                f"- object root: `{app.object_root}`",
                f"- default_pretrain_model: `{DEFAULT_PRETRAIN_MODEL}`",
                f"- checkpoint: `{app.checkpoint_path}`",
                f"- default scene views from config: `{app.scene_views}`",
                f"- object views: `{app.object_views}`",
                f"- inference resolution: `{app.resolution}`",
                f"- use_gt_pose_for_prediction: `{app.use_gt_pose_for_prediction}`",
                "",
                "選擇 `run` 和 `object` 後，會用 config 對應的 scene view 與 object reference views 跑 pose 預測，"
                "並只顯示相對於 input scene view 的 prediction / GT overlay。",
            ]
        )
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
            run_dropdown = gr.Dropdown(choices=app.run_choices, value=default_run, label="Run")
            object_dropdown = gr.Dropdown(choices=default_objects, value=default_object, label="Object")
            scene_view_dropdown = gr.Dropdown(choices=default_scene_views, value=default_scene_view, label="Input Scene View")
            use_depth_checkbox = gr.Checkbox(value=True, label="Use Depth Input")
            infer_button = gr.Button("Run Inference", variant="primary")

        input_height = "34vh" if image_focused_layout else 260
        compare_height = "56vh" if image_focused_layout else 420

        with gr.Row(elem_id="input_row"):
            scene_input_gallery = gr.Gallery(
                label="Scene Inputs",
                columns=1,
                height=input_height,
                elem_id="scene_inputs",
                preview=True,
            )
            object_input_gallery = gr.Gallery(
                label="Object Inputs",
                columns=len(app.object_views),
                height=input_height,
                elem_id="object_inputs",
                preview=True,
            )

        summary_markdown = gr.Markdown()

        with gr.Row(elem_id="compare_row"):
            pred_gallery = gr.Gallery(
                label="Predicted Pose Projection",
                columns=1,
                height=compare_height,
                elem_id="pred_projection",
                preview=True,
            )
            gt_gallery = gr.Gallery(
                label="Ground Truth Projection",
                columns=1,
                height=compare_height,
                elem_id="gt_projection",
                preview=True,
            )

        def refresh_run_controls(run_name: str):
            objects = app.get_object_choices(run_name)
            object_value = objects[0] if objects else None
            scene_views = app.get_scene_view_choices(run_name)
            scene_view_value = scene_views[0] if scene_views else None
            return (
                gr.update(choices=objects, value=object_value),
                gr.update(choices=scene_views, value=scene_view_value),
            )

        def refresh_inputs(run_name: str, object_name: str, scene_view):
            if not run_name or not object_name or scene_view is None:
                return [], []
            return app.input_gallery(run_name, object_name, int(scene_view))

        def sync_checkpoint_path(selected_value: str):
            return selected_value

        run_dropdown.change(refresh_run_controls, inputs=run_dropdown, outputs=[object_dropdown, scene_view_dropdown])
        run_dropdown.change(refresh_inputs, inputs=[run_dropdown, object_dropdown, scene_view_dropdown], outputs=[scene_input_gallery, object_input_gallery])
        object_dropdown.change(refresh_inputs, inputs=[run_dropdown, object_dropdown, scene_view_dropdown], outputs=[scene_input_gallery, object_input_gallery])
        scene_view_dropdown.change(refresh_inputs, inputs=[run_dropdown, object_dropdown, scene_view_dropdown], outputs=[scene_input_gallery, object_input_gallery])
        checkpoint_dropdown.change(sync_checkpoint_path, inputs=checkpoint_dropdown, outputs=checkpoint_textbox)
        load_model_button.click(
            app.load_checkpoint,
            inputs=checkpoint_textbox,
            outputs=[checkpoint_status, checkpoint_dropdown, checkpoint_textbox],
        )
        infer_button.click(
            app.run_inference,
            inputs=[run_dropdown, object_dropdown, scene_view_dropdown, use_depth_checkbox],
            outputs=[summary_markdown, pred_gallery, gt_gallery],
        )

        if default_object is not None:
            demo.load(
                refresh_inputs,
                inputs=[run_dropdown, object_dropdown, scene_view_dropdown],
                outputs=[scene_input_gallery, object_input_gallery],
            )
    return demo


def main():
    parser = argparse.ArgumentParser(description="Gradio demo for OmniVGGT 6D pose inference")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--image-focused-layout",
        action="store_true",
        help="Use a comparison-oriented layout that gives most of the page to images.",
    )
    parser.add_argument(
        "--use-gt-pose-for-prediction",
        action="store_true",
        help="Replace the predicted pose with ground-truth pose before drawing the prediction overlay.",
    )
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    app = DemoApp(
        args.config,
        args.checkpoint,
        use_gt_pose_for_prediction=args.use_gt_pose_for_prediction,
    )
    demo = build_demo(app, image_focused_layout=args.image_focused_layout)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, allowed_paths=[str(app.dataset_root), str(app.object_root), str(OBJ_ROOT)])


if __name__ == "__main__":
    main()
