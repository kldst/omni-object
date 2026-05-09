import os
from pathlib import Path

# Must be set before `import gradio` so Gradio's vibe_edit_history / cache
# files land in a writable location instead of /tmp/gradio.
PROJECT_ROOT = Path(__file__).resolve().parent
_LOCAL_TMP = PROJECT_ROOT / "tmp"
_LOCAL_TMP.mkdir(parents=True, exist_ok=True)
(_LOCAL_TMP / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", str(_LOCAL_TMP))
os.environ.setdefault("GRADIO_CACHE_DIR", str(_LOCAL_TMP))
os.environ.setdefault("TMPDIR", str(_LOCAL_TMP))
os.environ.setdefault("MPLCONFIGDIR", str(_LOCAL_TMP / "matplotlib"))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# cd /mnt/train-data-4-hdd/yian/freepose/omni-object
# python3 demo_gradio_6dpose.py \
#   --port 7860 \
#   --train-dataset-root /mnt/train-data-4-hdd/yian/freepose/dataset/0421_randon_4000scene_30pose

import argparse
import inspect
import re
import runpy
import warnings
from typing import Dict, List, Sequence, Tuple

import cv2
import gradio as gr
import numpy as np
import torch
import trimesh
from PIL import Image
from safetensors.torch import load_file as load_safetensors_file

from omnivggt.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from omnivggt.datasets.utils.misc import threshold_depth_map
from omnivggt.models.omnivggt import OmniVGGT
from omnivggt.utils.image import ImgNorm

# python3 demo_gradio_6dpose.py --port 7860
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train.py"
DEFAULT_DATASET_ROOT = Path(
    "/mnt/train-data-4-hdd/yian/freepose/dataset/0504_4000scene_30pose_test"
)
DEFAULT_OBJECT_RENDER_ROOT = Path(
    "/mnt/train-data-4-hdd/yian/freepose/object_space_renders_all"
)
DEFAULT_OBJ_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose/obj")
DEFAULT_PRETRAIN_MODEL = Path(
    "/mnt/train-data-4-hdd/yian/freepose/omni-object/outputs/"
    "0422_omnivggt_4000multiscene/model.safetensors"
)
PREDICTION_ROTATION_FIX = np.diag([1.0, 1.0, -1.0]).astype(np.float32)

PRED_AXIS_COLORS = ((255, 64, 64), (0, 255, 255), (255, 215, 0))
GT_AXIS_COLORS = ((255, 80, 200), (180, 255, 180), (200, 0, 255))
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
    value_str = match.group(1).strip()
    try:
        return eval(value_str, {"__builtins__": {}}, {})
    except Exception:
        return default


def resolve_runtime_settings(cfg: Dict) -> Dict:
    dataset_expr = str(cfg.get("train_dataset", ""))
    object_input_views = tuple(
        parse_dataset_ctor_arg(dataset_expr, "object_input_views", default=(1, 5, 10, 15))
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


def google_object_core_name(name: str) -> str | None:
    name = str(name)
    prefixes = ("google_", "Assets_obj_google_")
    suffix = "_meshes_model"
    for prefix in prefixes:
        if name.startswith(prefix) and name.endswith(suffix):
            return name[len(prefix):-len(suffix)]
    return None


def candidate_object_dir_names(object_name: str) -> List[str]:
    object_name = str(object_name)
    candidates = [object_name]
    if object_name.startswith("freepose_obj_"):
        remainder = object_name[len("freepose_obj_"):]
        candidates.append(remainder.replace("_obj_", "__obj_"))
    core = google_object_core_name(object_name)
    if core is not None:
        candidates.append(f"google__{core}__meshes")
    return list(dict.fromkeys(candidates))


def resolve_object_render_dir(object_render_root: Path, object_name: str) -> Path:
    for candidate in candidate_object_dir_names(object_name):
        path = object_render_root / candidate
        if path.is_dir():
            return path
    raise FileNotFoundError(f"Unable to resolve object render directory for {object_name}")


def object_image_filename(view_idx: int) -> str:
    return "Main_Camera_rgb.png" if int(view_idx) == 0 else f"Main_Camera_({int(view_idx)})_rgb.png"


def object_image_paths(
    object_render_root: Path,
    object_name: str,
    object_views: Sequence[int],
) -> List[Path]:
    object_dir = resolve_object_render_dir(object_render_root, object_name)
    paths = [object_dir / object_image_filename(v) for v in object_views]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing object render images for {object_name}: {missing}")
    return paths


def scale_for_object_name(name: str) -> float:
    if "ycbv" in name:
        return 0.002
    if "handal" in name:
        return 0.0015
    if "hope" in name:
        return 0.003
    if "rupac" in name:
        return 0.002
    if "google" in name:
        return 0.2
    return 1.0


def mesh_path_for_object_name(name: str, obj_root: Path) -> Path:
    search_roots = [Path(obj_root)]
    if obj_root.name == "google":
        search_roots.append(obj_root.parent)
    elif obj_root.name in {"obj", "data"}:
        sibling = obj_root.parent / ("data" if obj_root.name == "obj" else "obj")
        if sibling not in search_roots:
            search_roots.append(sibling)

    folder = google_object_core_name(name)
    if folder is not None:
        for root in search_roots:
            candidates = [
                root / "google" / folder / "meshes" / "model.obj",
                root / folder / "meshes" / "model.obj" if root.name == "google" else None,
            ]
            for path in candidates:
                if path is not None and path.is_file():
                    return path

    prefix_to_dir = {
        "freepose_obj_ycbv_": "ycbv",
        "freepose_obj_handal_": "handal",
        "freepose_obj_hope_": "hope",
        "freepose_obj_rupac_": "rupac",
    }
    for prefix, directory in prefix_to_dir.items():
        if name.startswith(prefix):
            obj_name = name[len(prefix):]
            for root in search_roots:
                for suffix in (".obj", ".ply"):
                    path = root / directory / f"{obj_name}{suffix}"
                    if path.is_file():
                        return path

    raise FileNotFoundError(f"Could not resolve mesh for object name: {name}")


def load_mesh_vertices(mesh_path: Path) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        mesh = trimesh.load(mesh_path, process=False, maintain_order=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"Failed to load vertices from {mesh_path}")
    return vertices


def axis_length_for_object(object_name: str, obj_root: Path) -> float:
    try:
        vertices = load_mesh_vertices(mesh_path_for_object_name(object_name, obj_root))
        vertices = vertices * float(scale_for_object_name(object_name))
        extent = vertices.max(axis=0) - vertices.min(axis=0)
        return max(float(np.linalg.norm(extent)) * 0.25, 1e-3)
    except Exception as exc:
        print(f"[demo] axis_length_for_object fallback for {object_name}: {exc}")
        return 0.1


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


def list_scenes(dataset_root: Path) -> List[str]:
    if not dataset_root.is_dir():
        return []
    return sorted(p.name for p in dataset_root.iterdir() if p.is_dir() and p.name.startswith("scene_"))


def list_frames_for_scene(dataset_root: Path, scene_name: str) -> List[str]:
    image_root = dataset_root / scene_name / "out_image"
    cam_root = dataset_root / scene_name / "out_cam_param"
    if not image_root.is_dir():
        return []
    frames = []
    for path in sorted(image_root.iterdir()):
        if not path.is_dir() or not path.name.startswith("frame_"):
            continue
        if not (path / "camera.jpg").is_file():
            continue
        if not (cam_root / path.name / "camera_camera.npz").is_file():
            continue
        frames.append(path.name)
    return frames


def image_path_for_frame(dataset_root: Path, scene_name: str, frame_name: str) -> Path:
    path = dataset_root / scene_name / "out_image" / frame_name / "camera.jpg"
    if not path.is_file():
        raise FileNotFoundError(f"Missing scene image: {path}")
    return path


def depth_path_for_frame(dataset_root: Path, scene_name: str, frame_name: str) -> Path:
    path = dataset_root / scene_name / "out_depth" / frame_name / "camera_depth.png"
    if not path.is_file():
        raise FileNotFoundError(f"Missing scene depth: {path}")
    return path


def cam_param_path_for_frame(dataset_root: Path, scene_name: str, frame_name: str) -> Path:
    path = dataset_root / scene_name / "out_cam_param" / frame_name / "camera_camera.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing camera param file: {path}")
    return path


def load_camera_params(dataset_root: Path, scene_name: str, frame_name: str) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, int, int
]:
    data = np.load(cam_param_path_for_frame(dataset_root, scene_name, frame_name))
    intrinsic = np.asarray(data["intrinsics.K_flat9"], dtype=np.float32).reshape(3, 3)
    world_to_camera = np.asarray(data["extrinsics.opencv.worldToCamera16"], dtype=np.float32).reshape(4, 4)
    camera_to_world = np.asarray(data["extrinsics.opencv.cameraToWorld16"], dtype=np.float32).reshape(4, 4)
    width = int(np.asarray(data["image.width"]).reshape(-1)[0])
    height = int(np.asarray(data["image.height"]).reshape(-1)[0])
    return intrinsic, world_to_camera, camera_to_world, width, height


def load_scene_pose_lookup(dataset_root: Path, scene_name: str) -> Dict[str, Dict[str, np.ndarray]]:
    pose_path = dataset_root / scene_name / "out_pose" / "poses.npz"
    data = np.load(pose_path, allow_pickle=False)
    names = [decode_name(name) for name in data["names"]]
    positions = data["positions"].astype(np.float32)
    quats = data["rot_quat_wxyz"].astype(np.float32)
    return {
        name: {"translation_world": positions[i], "quat_wxyz": quats[i]}
        for i, name in enumerate(names)
    }


def list_objects_for_scene(dataset_root: Path, scene_name: str) -> List[str]:
    try:
        return sorted(load_scene_pose_lookup(dataset_root, scene_name).keys())
    except FileNotFoundError:
        return []


class DemoScenePreprocessor(BaseStereoViewDataset):
    def __init__(self, resolution):
        super().__init__(dset="demo", resolution=resolution, transform=ImgNorm, seed=0)


def load_rgb_as_tensor(image_path: Path, resolution: Sequence[int], device: torch.device) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    width, height = int(resolution[0]), int(resolution[1])
    image = image.resize((width, height), getattr(Image, "Resampling", Image).LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
    return tensor.to(device)


def load_object_tensor(
    object_render_root: Path,
    object_name: str,
    object_views: Sequence[int],
    resolution,
    device,
) -> torch.Tensor:
    images = [
        load_rgb_as_tensor(p, resolution, device)
        for p in object_image_paths(object_render_root, object_name, object_views)
    ]
    return torch.stack(images, dim=0).unsqueeze(0)


def load_scene_frame_inputs(
    dataset_root: Path,
    scene_name: str,
    frame_name: str,
    resolution,
    device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_path = image_path_for_frame(dataset_root, scene_name, frame_name)
    depth_path = depth_path_for_frame(dataset_root, scene_name, frame_name)

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

    intrinsics = (
        np.load(cam_param_path_for_frame(dataset_root, scene_name, frame_name))["intrinsics.K_flat9"]
        .astype(np.float32)
        .reshape(3, 3)
    )
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


def load_depth_map_for_visualization(dataset_root: Path, scene_name: str, frame_name: str) -> np.ndarray:
    depth_path = depth_path_for_frame(dataset_root, scene_name, frame_name)
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Failed to read depth image: {depth_path}")
    if depth_raw.dtype != np.uint16:
        raise ValueError(f"Expected uint16 R16 depth image, got {depth_raw.dtype} for {depth_path}")
    depthmap = depth_raw.view(np.float16).astype(np.float32)
    depthmap[~np.isfinite(depthmap)] = 0.0
    depthmap[depthmap < 0] = 0.0
    return threshold_depth_map(depthmap, max_percentile=99, min_percentile=-1)


def project_camera_points(points_cam: np.ndarray, intrinsic: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    z = points_cam[:, 2]
    valid = z > 1e-6
    uv = np.full((points_cam.shape[0], 2), np.nan, dtype=np.float32)
    if np.any(valid):
        uvw = points_cam[valid] @ intrinsic.T
        uv[valid] = (uvw[:, :2] / uvw[:, 2:3]).astype(np.float32)
    return uv, valid


def draw_axes_overlay(
    image_path: Path,
    intrinsic: np.ndarray,
    rotation_cam: np.ndarray,
    translation_cam: np.ndarray,
    axis_length: float,
    width: int,
    height: int,
    axis_colors: Sequence[Tuple[int, int, int]],
) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[1] != width or image.shape[0] != height:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    pts_obj = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, axis_length],
        ],
        dtype=np.float32,
    )
    pts_cam = pts_obj @ rotation_cam.T + translation_cam[None, :]
    uv, valid = project_camera_points(pts_cam, intrinsic)

    overlay = image.copy()
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


def build_axes_gallery(
    dataset_root: Path,
    scene_name: str,
    frame_name: str,
    rotation_cam: np.ndarray,
    translation_cam: np.ndarray,
    axis_length: float,
    label: str,
    axis_colors: Sequence[Tuple[int, int, int]],
) -> List[Tuple[np.ndarray, str]]:
    intrinsic, _, _, width, height = load_camera_params(dataset_root, scene_name, frame_name)
    image_path = image_path_for_frame(dataset_root, scene_name, frame_name)
    overlay = draw_axes_overlay(
        image_path, intrinsic, rotation_cam, translation_cam, axis_length, width, height, axis_colors,
    )
    return [(overlay, f"{label} ({frame_name})")]


def pose_axis_points_camera_frame(
    rotation_cam: np.ndarray,
    translation_cam: np.ndarray,
    axis_length: float,
) -> np.ndarray:
    pts_obj = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, axis_length],
        ],
        dtype=np.float32,
    )
    return pts_obj @ rotation_cam.T + translation_cam[None, :]


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
    dataset_root: Path,
    scene_name: str,
    frame_name: str,
    pred_rotation_cam: np.ndarray,
    pred_translation_cam: np.ndarray,
    gt_rotation_cam: np.ndarray,
    gt_translation_cam: np.ndarray,
    axis_length: float,
    point_cloud_stride: int = 2,
) -> str:
    intrinsic, _, _, _, _ = load_camera_params(dataset_root, scene_name, frame_name)
    depthmap = load_depth_map_for_visualization(dataset_root, scene_name, frame_name)
    rgb_image = np.asarray(Image.open(image_path_for_frame(dataset_root, scene_name, frame_name)).convert("RGB"))
    stride = max(1, int(point_cloud_stride))
    max_points = 80000 if stride <= 2 else 50000
    points, colors = depth_to_camera_points(depthmap, intrinsic, rgb_image, stride=stride, max_points=max_points)
    safe_scene = re.sub(r"[^a-zA-Z0-9_]+", "_", scene_name)[:120]
    safe_frame = re.sub(r"[^a-zA-Z0-9_]+", "_", frame_name)[:120]
    out_dir = _LOCAL_TMP / "point_cloud_pose_glb" / safe_scene
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe_frame}_stride{stride}.glb"

    scene_3d = trimesh.Scene()
    if len(points):
        point_cloud = trimesh.PointCloud(vertices=points, colors=colors if colors is not None else None)
        scene_3d.add_geometry(point_cloud)

    viewer_axis_length = float(axis_length) * 1.8
    axis_radius = max(viewer_axis_length * 0.08, 2.5e-4)
    center_radius = axis_radius * 2.4
    tip_radius = axis_radius * 1.5

    def add_pose_axes(rotation_cam, translation_cam, axis_colors):
        pts = pose_axis_points_camera_frame(rotation_cam, translation_cam, viewer_axis_length)
        center = pts[0]
        center_mesh = trimesh.creation.icosphere(radius=center_radius, subdivisions=1)
        center_mesh.apply_translation(center)
        center_mesh.visual.face_colors = np.tile(np.array([[255, 255, 255, 255]], dtype=np.uint8), (len(center_mesh.faces), 1))
        scene_3d.add_geometry(center_mesh)
        for idx, color in enumerate(axis_colors):
            axis_mesh = trimesh.creation.cylinder(
                radius=axis_radius,
                segment=np.stack([center, pts[idx + 1]], axis=0),
            )
            rgba = np.array([[color[0], color[1], color[2], 255]], dtype=np.uint8)
            axis_mesh.visual.face_colors = np.tile(rgba, (len(axis_mesh.faces), 1))
            scene_3d.add_geometry(axis_mesh)
            tip_mesh = trimesh.creation.icosphere(radius=tip_radius, subdivisions=1)
            tip_mesh.apply_translation(pts[idx + 1])
            tip_mesh.visual.face_colors = np.tile(rgba, (len(tip_mesh.faces), 1))
            scene_3d.add_geometry(tip_mesh)

    add_pose_axes(pred_rotation_cam, pred_translation_cam, PRED_AXIS_COLORS)
    add_pose_axes(gt_rotation_cam, gt_translation_cam, GT_AXIS_COLORS)
    scene_3d.export(out_path)
    return str(out_path)


def format_pose_markdown(
    *,
    scene_name: str,
    frame_name: str,
    object_name: str,
    checkpoint_path: Path,
    pred_translation_cam: np.ndarray,
    pred_rotation_cam: np.ndarray,
    pred_translation_world: np.ndarray,
    pred_rotation_world: np.ndarray,
    gt_translation_cam: np.ndarray,
    gt_rotation_cam: np.ndarray,
    gt_translation_world: np.ndarray,
    gt_rotation_world: np.ndarray,
    use_depth_input: bool,
) -> str:
    rot_error_deg = rotation_error_degrees(pred_rotation_cam, gt_rotation_cam)
    trans_error = translation_error(pred_translation_cam, gt_translation_cam)
    return "\n".join(
        [
            "### Prediction Summary",
            f"- checkpoint: `{checkpoint_path}`",
            f"- scene: `{scene_name}`",
            f"- frame: `{frame_name}`",
            f"- object: `{object_name}`",
            f"- use depth input: `{use_depth_input}`",
            "",
            "### Predicted Pose",
            f"- camera-frame translation: `{np.round(pred_translation_cam, 6).tolist()}`",
            f"- camera-frame rotation matrix: `{np.round(pred_rotation_cam, 6).tolist()}`",
            f"- world translation: `{np.round(pred_translation_world, 6).tolist()}`",
            f"- world rotation matrix: `{np.round(pred_rotation_world, 6).tolist()}`",
            "",
            "### Ground Truth (camera frame from out_cam_param)",
            f"- camera-frame translation: `{np.round(gt_translation_cam, 6).tolist()}`",
            f"- camera-frame rotation matrix: `{np.round(gt_rotation_cam, 6).tolist()}`",
            f"- world translation: `{np.round(gt_translation_world, 6).tolist()}`",
            f"- world rotation matrix: `{np.round(gt_rotation_world, 6).tolist()}`",
            "",
            "### Errors (camera frame, pred vs GT)",
            f"- translation L2: `{trans_error['l2']:.6f}`",
            f"- translation abs xyz: `{np.round(trans_error['abs_xyz'], 6).tolist()}`",
            f"- translation signed xyz: `{np.round(trans_error['signed_xyz'], 6).tolist()}`",
            f"- rotation error (deg): `{rot_error_deg:.6f}`",
        ]
    )


SEEN_MARKER = "🔴 "


class DemoApp:
    def __init__(
        self,
        config_path: Path,
        checkpoint_path: str | None,
        dataset_root: Path,
        object_render_root: Path,
        obj_root: Path,
        train_dataset_root: Path | None = None,
        use_gt_pose_for_prediction: bool = False,
    ):
        self.cfg = load_config(config_path)
        self.runtime = resolve_runtime_settings(self.cfg)
        self.dataset_root = Path(dataset_root)
        self.object_render_root = Path(object_render_root)
        self.obj_root = Path(obj_root)
        self.train_dataset_root = Path(train_dataset_root) if train_dataset_root else None
        self.object_views = tuple(int(v) for v in self.runtime["object_input_views"])
        self.resolution = tuple(int(v) for v in self.runtime["resolution"])
        self.use_gt_pose_for_prediction = bool(use_gt_pose_for_prediction)

        self.scene_choices = list_scenes(self.dataset_root)
        if not self.scene_choices:
            raise RuntimeError(f"No scenes found under {self.dataset_root}")

        self.seen_objects = self._scan_train_objects()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = resolve_checkpoint_path(self.cfg, checkpoint_path)
        self.model = None
        self.available_checkpoints = self.discover_checkpoints()
        self.load_checkpoint(self.checkpoint_path)

    def _scan_train_objects(self) -> set:
        if self.train_dataset_root is None:
            return set()
        if not self.train_dataset_root.is_dir():
            print(f"[demo] train_dataset_root not found: {self.train_dataset_root}; annotation disabled.")
            return set()
        objs: set = set()
        for scene in self.train_dataset_root.iterdir():
            if not scene.is_dir() or not scene.name.startswith("scene_"):
                continue
            try:
                objs |= set(load_scene_pose_lookup(self.train_dataset_root, scene.name).keys())
            except FileNotFoundError:
                continue
        print(f"[demo] loaded {len(objs)} train objects from {self.train_dataset_root}")
        return objs

    def decorate_object_choices(self, names) -> list:
        if not self.seen_objects:
            return [(n, n) for n in names]
        return [
            ((SEEN_MARKER + n) if n in self.seen_objects else n, n)
            for n in names
        ]

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

    def get_frame_choices(self, scene_name: str) -> List[str]:
        return list_frames_for_scene(self.dataset_root, scene_name) if scene_name else []

    def get_object_choices(self, scene_name: str) -> List[str]:
        return list_objects_for_scene(self.dataset_root, scene_name) if scene_name else []

    def input_gallery(self, scene_name: str, frame_name: str, object_name: str):
        scene_image = str(image_path_for_frame(self.dataset_root, scene_name, frame_name))
        object_gallery = []
        try:
            paths = object_image_paths(self.object_render_root, object_name, self.object_views)
            for view, path in zip(self.object_views, paths):
                object_gallery.append((str(path), f"Object view {view}"))
        except FileNotFoundError as exc:
            print(f"[demo] {exc}")
        return scene_image, object_gallery

    def run_inference(
        self,
        scene_name: str,
        frame_name: str,
        object_name: str,
        use_depth_input: bool,
        show_point_cloud_pose: bool,
        point_cloud_stride: int,
    ):
        pose_lookup = load_scene_pose_lookup(self.dataset_root, scene_name)
        if object_name not in pose_lookup:
            raise ValueError(f"Object {object_name} not found in {scene_name}/out_pose/poses.npz")
        use_depth_input = bool(use_depth_input)

        scene_tensor, depth_tensor, mask_tensor = load_scene_frame_inputs(
            self.dataset_root, scene_name, frame_name, self.resolution, self.device,
        )
        object_tensor = load_object_tensor(
            self.object_render_root, object_name, self.object_views, self.resolution, self.device,
        )

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
        pred_translation_cam = outputs["object_translation"][0].detach().float().cpu().numpy().astype(np.float32)
        pred_rotation_cam = (rot6d_to_matrix(pred_rot6d_cam) @ PREDICTION_ROTATION_FIX).astype(np.float32)

        intrinsic, world_to_cam, cam_to_world, width, height = load_camera_params(
            self.dataset_root, scene_name, frame_name,
        )
        pred_rotation_world = (cam_to_world[:3, :3] @ pred_rotation_cam).astype(np.float32)
        pred_translation_world = (cam_to_world[:3, :3] @ pred_translation_cam + cam_to_world[:3, 3]).astype(np.float32)

        gt_translation_world = pose_lookup[object_name]["translation_world"].astype(np.float32)
        gt_rotation_world = quat_wxyz_to_matrix(pose_lookup[object_name]["quat_wxyz"]).astype(np.float32)
        gt_rotation_cam = (world_to_cam[:3, :3] @ gt_rotation_world).astype(np.float32)
        gt_translation_cam = (world_to_cam[:3, :3] @ gt_translation_world + world_to_cam[:3, 3]).astype(np.float32)

        if self.use_gt_pose_for_prediction:
            pred_rotation_world = gt_rotation_world
            pred_translation_world = gt_translation_world
            pred_rotation_cam = gt_rotation_cam
            pred_translation_cam = gt_translation_cam

        axis_length = axis_length_for_object(object_name, self.obj_root)

        pred_gallery = build_axes_gallery(
            self.dataset_root, scene_name, frame_name,
            pred_rotation_cam, pred_translation_cam, axis_length,
            "Predicted axes", PRED_AXIS_COLORS,
        )
        gt_gallery = build_axes_gallery(
            self.dataset_root, scene_name, frame_name,
            gt_rotation_cam, gt_translation_cam, axis_length,
            "GT axes", GT_AXIS_COLORS,
        )
        pred_image = pred_gallery[0][0]
        gt_image = gt_gallery[0][0]

        summary = format_pose_markdown(
            scene_name=scene_name,
            frame_name=frame_name,
            object_name=object_name,
            checkpoint_path=self.checkpoint_path,
            pred_translation_cam=pred_translation_cam,
            pred_rotation_cam=pred_rotation_cam,
            pred_translation_world=pred_translation_world,
            pred_rotation_world=pred_rotation_world,
            gt_translation_cam=gt_translation_cam,
            gt_rotation_cam=gt_rotation_cam,
            gt_translation_world=gt_translation_world,
            gt_rotation_world=gt_rotation_world,
            use_depth_input=use_depth_input,
        )
        point_cloud_glb = None
        if bool(show_point_cloud_pose):
            point_cloud_glb = export_point_cloud_pose_glb(
                self.dataset_root,
                scene_name,
                frame_name,
                pred_rotation_cam,
                pred_translation_cam,
                gt_rotation_cam,
                gt_translation_cam,
                axis_length,
                point_cloud_stride=point_cloud_stride,
            )
        return summary, pred_image, gt_image, point_cloud_glb


def build_demo(app: DemoApp, image_focused_layout: bool = False):
    default_scene = app.scene_choices[0]
    default_frames = app.get_frame_choices(default_scene)
    default_frame = default_frames[0] if default_frames else None
    default_objects = app.get_object_choices(default_scene)
    default_object_choices = app.decorate_object_choices(default_objects)
    default_object = default_objects[0] if default_objects else None

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

    with gr.Blocks(title="OmniVGGT 6D Pose Demo (scene/frame)", css=demo_css) as demo:
        info_markdown = "\n".join(
            [
                "# OmniVGGT 6D Pose Demo (single-view, scene/frame layout)",
                f"- config: `{DEFAULT_CONFIG_PATH}`",
                f"- dataset: `{app.dataset_root}`",
                f"- object render root: `{app.object_render_root}`",
                f"- obj (mesh) root: `{app.obj_root}`",
                f"- default checkpoint: `{DEFAULT_PRETRAIN_MODEL}`",
                f"- loaded checkpoint: `{app.checkpoint_path}`",
                f"- object views: `{app.object_views}`",
                f"- inference resolution: `{app.resolution}`",
                f"- use_gt_pose_for_prediction: `{app.use_gt_pose_for_prediction}`",
                f"- train_dataset_root: `{app.train_dataset_root}` "
                + (
                    f"(annotated {len(app.seen_objects)} seen objects with 🔴)"
                    if app.seen_objects
                    else "(annotation disabled)"
                ),
                "",
                "選擇 `scene` → `frame` → `object`,只用單一 frame 的 RGB+Depth 做 pose 預測;"
                "GT 是 `out_pose/poses.npz`(world)經 `out_cam_param/<frame>` 轉到 camera frame,"
                "兩者比較 translation L2 與 rotation 角度誤差;畫面只畫物體中心的 X(紅)/Y(青)/Z(黃) 軸。"
                "勾選點雲選項後，會再把原始深度投影成相機座標系點雲，並畫出 Pred / GT pose 軸。"
                + (
                    " 物件名稱前方的 🔴 表示該物件曾出現在 `--train-dataset-root` 指定的 dataset 中。"
                    if app.seen_objects
                    else ""
                ),
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
            scene_dropdown = gr.Dropdown(choices=app.scene_choices, value=default_scene, label="Scene")
            frame_dropdown = gr.Dropdown(choices=default_frames, value=default_frame, label="Frame")
            object_dropdown = gr.Dropdown(choices=default_object_choices, value=default_object, label="Object")
            use_depth_checkbox = gr.Checkbox(value=True, label="Use Depth Input")
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

        big_height    = "38vh" if image_focused_layout else 360
        thumb_height  = "18vh" if image_focused_layout else 180

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

        summary_markdown = gr.Markdown()

        def refresh_scene_controls(scene_name: str):
            frames = app.get_frame_choices(scene_name)
            objects = app.get_object_choices(scene_name)
            frame_value = frames[0] if frames else None
            object_value = objects[0] if objects else None
            return (
                gr.update(choices=frames, value=frame_value),
                gr.update(choices=app.decorate_object_choices(objects), value=object_value),
            )

        def refresh_inputs(scene_name, frame_name, object_name):
            if not scene_name or not frame_name or not object_name:
                return None, []
            return app.input_gallery(scene_name, frame_name, object_name)

        def sync_checkpoint_path(selected_value: str):
            return selected_value

        scene_dropdown.change(
            refresh_scene_controls, inputs=scene_dropdown, outputs=[frame_dropdown, object_dropdown],
        )
        scene_dropdown.change(
            refresh_inputs,
            inputs=[scene_dropdown, frame_dropdown, object_dropdown],
            outputs=[scene_input_image, object_input_gallery],
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
            outputs=[summary_markdown, pred_image, gt_image, point_cloud_model],
        )

        if default_object is not None and default_frame is not None:
            demo.load(
                refresh_inputs,
                inputs=[scene_dropdown, frame_dropdown, object_dropdown],
                outputs=[scene_input_image, object_input_gallery],
            )
    return demo


def main():
    parser = argparse.ArgumentParser(description="Gradio demo for OmniVGGT 6D pose inference (scene/frame layout)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--object-render-root", type=Path, default=DEFAULT_OBJECT_RENDER_ROOT)
    parser.add_argument("--obj-root", type=Path, default=DEFAULT_OBJ_ROOT)
    parser.add_argument(
        "--train-dataset-root", type=Path, default=None,
        help="Optional dataset whose object names should be marked as 'seen' (🔴) in the Object dropdown."
        " e.g. --train-dataset-root /mnt/train-data-4-hdd/yian/freepose/dataset/0421_randon_4000scene_30pose",
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
        object_render_root=args.object_render_root,
        obj_root=args.obj_root,
        train_dataset_root=args.train_dataset_root,
        use_gt_pose_for_prediction=args.use_gt_pose_for_prediction,
    )
    demo = build_demo(app, image_focused_layout=args.image_focused_layout)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[
            str(app.dataset_root),
            str(app.object_render_root),
            str(app.obj_root),
        ],
    )


if __name__ == "__main__":
    main()
