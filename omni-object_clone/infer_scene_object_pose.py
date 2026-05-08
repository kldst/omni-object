import argparse
import json
import re
import runpy
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw
from safetensors.torch import load_file as load_safetensors_file
from tqdm import tqdm

from omnivggt.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from omnivggt.datasets.utils.misc import threshold_depth_map
from omnivggt.models.omnivggt import OmniVGGT
from omnivggt.utils.image import ImgNorm


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train.py"
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "0420" / "model.safetensors"
DEFAULT_DATASET_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose/0420_trajectory_40scene_500frame")
DEFAULT_OBJECT_RENDER_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose/object_space_renders_all")
DEFAULT_OBJ_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose/data")
DEFAULT_OUTPUT = Path("/mnt/train-data-4-hdd/yian/freepose/omni-object/infer")
PREDICTION_ROTATION_FIX = np.diag([1.0, 1.0, -1.0]).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run OmniVGGT object pose inference for a single object across all frames "
            "of one trajectory scene, then save per-frame overlay images."
        )
    )
    parser.add_argument("--scene", required=True, help="Scene directory name, e.g. scene_0000.")
    parser.add_argument(
        "--object-name",
        default=None,
        help="Exact object name stored in pose npz files. If omitted, process every object found in the scene.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--object-render-root", type=Path, default=DEFAULT_OBJECT_RENDER_ROOT)
    parser.add_argument(
        "--obj-root",
        type=Path,
        default=DEFAULT_OBJ_ROOT,
        help="Base root for object meshes. The resolver will try both data/google/... and obj/google/... style layouts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to save overlays and pose summaries. Default: omni-object/outputs/scene_pose_inference/<scene>/<object>.",
    )
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda, cuda:0, cpu.")
    parser.add_argument("--frame-start", type=int, default=None, help="Inclusive frame index filter.")
    parser.add_argument("--frame-end", type=int, default=None, help="Inclusive frame index filter.")
    parser.add_argument(
        "--object-views",
        type=int,
        nargs="+",
        default=None,
        help="Object render view ids. Default comes from config train_dataset.",
    )
    parser.add_argument("--use-depth-input", action="store_true", default=True)
    parser.add_argument("--no-depth-input", dest="use_depth_input", action="store_false")
    parser.add_argument("--draw-gt", action="store_true", help="Also draw GT bbox/axes when pose exists.")
    parser.add_argument(
        "--scale-mode",
        choices=("auto", "none"),
        default="auto",
        help="Apply dataset-specific mesh scaling before projection.",
    )
    parser.add_argument(
        "--bbox-from",
        choices=("mesh", "aabb"),
        default="aabb",
        help="Currently kept for clarity; projected overlay is computed from the scaled mesh axis-aligned bbox.",
    )
    parser.add_argument("--video-fps", type=int, default=12, help="FPS used when exporting overlay video.")
    parser.add_argument("--no-video", dest="export_video", action="store_false", help="Skip MP4 export.")
    parser.set_defaults(export_video=True)
    return parser.parse_args()


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
    resolution = tuple(int(v) for v in cfg.get("resolution", (518, 518)))
    object_input_views = tuple(parse_dataset_ctor_arg(dataset_expr, "object_input_views", default=(1, 5, 10, 15)))
    return {
        "resolution": resolution,
        "object_input_views": tuple(int(v) for v in object_input_views),
    }


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
        print(f"[infer_scene_object_pose] Missing keys: {missing}")
    if unexpected:
        print(f"[infer_scene_object_pose] Unexpected keys: {unexpected}")
    model.eval().to(device)
    return model


def image_filename(view_idx: int) -> str:
    return "Main_Camera_rgb.png" if int(view_idx) == 0 else f"Main_Camera_({int(view_idx)})_rgb.png"


def candidate_object_dir_names(object_name: str) -> List[str]:
    object_name = str(object_name)
    candidates = [object_name]
    if object_name.startswith("freepose_obj_"):
        remainder = object_name[len("freepose_obj_") :]
        candidates.append(remainder.replace("_obj_", "__obj_"))
    if object_name.startswith("google_") and object_name.endswith("_meshes_model"):
        core = object_name[len("google_") : -len("_meshes_model")]
        candidates.append(f"google__{core}__meshes")
    return list(dict.fromkeys(candidates))


def resolve_object_render_dir(object_render_root: Path, object_name: str) -> Path:
    for candidate in candidate_object_dir_names(object_name):
        path = object_render_root / candidate
        if path.is_dir():
            return path
    raise FileNotFoundError(f"Unable to resolve object render directory for {object_name}")


def object_image_paths(
    object_render_root: Path,
    object_name: str,
    object_views: Sequence[int],
) -> List[Path]:
    object_dir = resolve_object_render_dir(object_render_root, object_name)
    paths = [object_dir / image_filename(view_idx) for view_idx in object_views]
    missing = [str(path) for path in paths if not path.is_file()]
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
    raise KeyError(f"Unknown object scale for {name}")


def mesh_path_for_object_name(name: str, obj_root: Path) -> Path:
    search_roots = [Path(obj_root)]
    if obj_root.name == "google":
        search_roots.append(obj_root.parent)
    elif obj_root.name in {"obj", "data"}:
        sibling = obj_root.parent / ("data" if obj_root.name == "obj" else "obj")
        if sibling not in search_roots:
            search_roots.append(sibling)

    if name.startswith("google_") and name.endswith("_meshes_model"):
        folder = name[len("google_") : -len("_meshes_model")]
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
            obj_name = name[len(prefix) :]
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
        dtype=np.float32,
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
    return np.stack([x, y, z], axis=1).astype(np.float32)


def decode_name(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def load_pose_lookup(pose_path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    data = np.load(pose_path, allow_pickle=False)
    names = [decode_name(name) for name in data["names"]]
    positions = data["positions"].astype(np.float32)
    quats = data["rot_quat_wxyz"].astype(np.float32)
    result = {}
    for idx, name in enumerate(names):
        result[name] = {
            "translation_world": positions[idx],
            "quat_wxyz": quats[idx],
        }
    return result


def load_camera_params(camera_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    data = np.load(camera_path)
    intrinsic = np.asarray(data["intrinsics.K_flat9"], dtype=np.float32).reshape(3, 3)
    world_to_camera = np.asarray(data["extrinsics.opencv.worldToCamera16"], dtype=np.float32).reshape(4, 4)
    camera_to_world = np.asarray(data["extrinsics.opencv.cameraToWorld16"], dtype=np.float32).reshape(4, 4)
    width = int(np.asarray(data["image.width"]).reshape(-1)[0])
    height = int(np.asarray(data["image.height"]).reshape(-1)[0])
    return intrinsic, world_to_camera, camera_to_world, width, height


def project_camera_points(points_cam: np.ndarray, intrinsic: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    z = points_cam[:, 2]
    valid = z > 1e-6
    uv = np.full((points_cam.shape[0], 2), np.nan, dtype=np.float32)
    if np.any(valid):
        projected = points_cam[valid] @ intrinsic.T
        uv[valid] = (projected[:, :2] / projected[:, 2:3]).astype(np.float32)
    return uv, valid


def draw_bbox_and_axes(
    image_path: Path,
    intrinsic: np.ndarray,
    bbox_points_cam: np.ndarray,
    center_cam: np.ndarray,
    axis_points_cam: np.ndarray,
    output_path: Path,
    label: str,
    bbox_color: Tuple[int, int, int],
    axis_colors: Sequence[Tuple[int, int, int]],
    width: int,
    height: int,
    gt_bbox_points_cam: np.ndarray | None = None,
    gt_center_cam: np.ndarray | None = None,
    gt_axis_points_cam: np.ndarray | None = None,
) -> None:
    image = Image.open(image_path).convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), getattr(Image, "Resampling", Image).BILINEAR)
    draw = ImageDraw.Draw(image)

    def _draw_single(
        draw_obj: ImageDraw.ImageDraw,
        bbox_cam: np.ndarray,
        center_cam_one: np.ndarray,
        axis_cam: np.ndarray,
        bbox_color_one: Tuple[int, int, int],
        axis_colors_one: Sequence[Tuple[int, int, int]],
        label_text: str | None,
    ) -> None:
        uv, valid = project_camera_points(bbox_cam, intrinsic)
        center_uv, center_valid = project_camera_points(center_cam_one[None], intrinsic)
        axis_uv, axis_valid = project_camera_points(axis_cam, intrinsic)

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        inside = valid.copy()
        inside &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        for start_idx, end_idx in edges:
            if inside[start_idx] and inside[end_idx]:
                p1 = tuple(np.round(uv[start_idx]).astype(np.int32))
                p2 = tuple(np.round(uv[end_idx]).astype(np.int32))
                draw_obj.line([p1, p2], fill=bbox_color_one, width=2)

        if bool(center_valid[0]):
            center_xy = tuple(np.round(center_uv[0]).astype(np.int32))
            radius = 4
            draw_obj.ellipse(
                (center_xy[0] - radius, center_xy[1] - radius, center_xy[0] + radius, center_xy[1] + radius),
                fill=bbox_color_one,
            )
            for axis_idx, axis_color in enumerate(axis_colors_one):
                if axis_valid[axis_idx]:
                    end_xy = tuple(np.round(axis_uv[axis_idx]).astype(np.int32))
                    draw_obj.line([center_xy, end_xy], fill=axis_color, width=2)
                    direction = np.asarray(end_xy, dtype=np.float32) - np.asarray(center_xy, dtype=np.float32)
                    norm = float(np.linalg.norm(direction))
                    if norm > 1e-6:
                        direction = direction / norm
                        left = np.array([-direction[1], direction[0]], dtype=np.float32)
                        tip = np.asarray(end_xy, dtype=np.float32)
                        back = tip - direction * 8.0
                        p_left = tuple(np.round(back + left * 4.0).astype(np.int32))
                        p_right = tuple(np.round(back - left * 4.0).astype(np.int32))
                        draw_obj.polygon([tuple(end_xy), p_left, p_right], fill=axis_color)
            if label_text:
                draw_obj.text(
                    (center_xy[0] + 6, max(18, center_xy[1] - 8)),
                    label_text,
                    fill=bbox_color_one,
                )

    _draw_single(draw, bbox_points_cam, center_cam, axis_points_cam, bbox_color, axis_colors, label)
    if gt_bbox_points_cam is not None and gt_center_cam is not None and gt_axis_points_cam is not None:
        _draw_single(
            draw,
            gt_bbox_points_cam,
            gt_center_cam,
            gt_axis_points_cam,
            (255, 0, 255),
            ((255, 128, 255), (255, 80, 200), (200, 0, 255)),
            f"{label} [GT]",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


class SceneFramePreprocessor(BaseStereoViewDataset):
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
    resolution: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    images = [load_rgb_as_tensor(path, resolution, device) for path in object_image_paths(object_render_root, object_name, object_views)]
    return torch.stack(images, dim=0).unsqueeze(0)


def load_scene_frame_inputs(
    scene_dir: Path,
    frame_name: str,
    resolution: Sequence[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_path = scene_dir / "out_image" / frame_name / "camera.jpg"
    depth_path = scene_dir / "out_depth" / frame_name / "camera_depth.png"
    camera_path = scene_dir / "out_cam_param" / frame_name / "camera_camera.npz"

    image = Image.open(image_path).convert("RGB")
    depth_raw = np.array(Image.open(depth_path))
    if depth_raw.dtype != np.uint16:
        raise ValueError(f"Expected uint16 R16 depth image, got {depth_raw.dtype} for {depth_path}")

    depthmap = depth_raw.view(np.float16).astype(np.float32)
    depthmap[~np.isfinite(depthmap)] = 0.0
    depthmap[depthmap < 0] = 0.0
    depthmap = threshold_depth_map(depthmap, max_percentile=99, min_percentile=-1)

    intrinsics = np.load(camera_path)["intrinsics.K_flat9"].astype(np.float32).reshape(3, 3)
    processor = SceneFramePreprocessor(resolution=resolution)
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


def list_frame_names(scene_dir: Path) -> List[str]:
    image_root = scene_dir / "out_image"
    frame_names = []
    if not image_root.is_dir():
        return frame_names
    for path in sorted(image_root.iterdir()):
        if not path.is_dir() or not path.name.startswith("frame_"):
            continue
        if not (path / "camera.jpg").is_file():
            continue
        if not (scene_dir / "out_cam_param" / path.name / "camera_camera.npz").is_file():
            continue
        frame_names.append(path.name)
    return frame_names


def list_scene_object_names(scene_dir: Path, frame_names: Sequence[str]) -> List[str]:
    object_names = set()
    for frame_name in frame_names:
        pose_path = scene_dir / "out_pose" / f"{frame_name}.npz"
        if not pose_path.is_file():
            continue
        pose_lookup = load_pose_lookup(pose_path)
        object_names.update(pose_lookup.keys())
    return sorted(object_names)


def filter_frame_names(frame_names: Sequence[str], frame_start: int | None, frame_end: int | None) -> List[str]:
    filtered = []
    for frame_name in frame_names:
        match = re.search(r"(\d+)$", frame_name)
        frame_idx = int(match.group(1)) if match else None
        if frame_start is not None and frame_idx is not None and frame_idx < frame_start:
            continue
        if frame_end is not None and frame_idx is not None and frame_idx > frame_end:
            continue
        filtered.append(frame_name)
    return filtered


def make_output_dir(base_output_dir: Path | None, scene_name: str, object_name: str, multi_object: bool) -> Path:
    safe_object = re.sub(r"[^A-Za-z0-9_.-]+", "_", object_name)
    if base_output_dir is not None:
        return base_output_dir / safe_object if multi_object else base_output_dir
    return PROJECT_ROOT / "outputs" / "scene_pose_inference" / scene_name / safe_object


def to_serializable(value):
    if isinstance(value, dict):
        return {str(key): to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    return value


def export_overlay_video(overlay_dir: Path, video_path: Path, fps: int) -> bool:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        print(f"[infer_scene_object_pose] ffmpeg not found; skipping video export for {overlay_dir}")
        return False

    cmd = [
        ffmpeg_path,
        "-y",
        "-framerate",
        str(int(fps)),
        "-i",
        str(overlay_dir / "frame_%04d.jpg"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"[infer_scene_object_pose] ffmpeg failed for {overlay_dir}:\n{result.stderr}")
        return False
    return True


def run_inference_for_object(
    *,
    model: OmniVGGT,
    scene_dir: Path,
    scene_name: str,
    object_name: str,
    object_render_root: Path,
    obj_root: Path,
    frame_names: Sequence[str],
    resolution: Sequence[int],
    object_views: Sequence[int],
    device: torch.device,
    output_dir: Path,
    use_depth_input: bool,
    draw_gt: bool,
    scale_mode: str,
    export_video: bool,
    video_fps: int,
    args,
) -> Dict[str, object]:
    object_tensor = load_object_tensor(object_render_root, object_name, object_views, resolution, device)

    mesh_vertices = load_mesh_vertices(mesh_path_for_object_name(object_name, obj_root))
    mesh_scale = 1.0 if scale_mode == "none" else float(scale_for_object_name(object_name))
    mesh_vertices = mesh_vertices * mesh_scale
    bbox_obj = compute_bbox_corners(mesh_vertices)
    bbox_extent = bbox_obj.max(axis=0) - bbox_obj.min(axis=0)
    axis_length = max(float(np.linalg.norm(bbox_extent)) * 0.25, 1e-3)
    center_obj = np.zeros((3,), dtype=np.float32)
    axis_obj = np.asarray(
        [[axis_length, 0.0, 0.0], [0.0, axis_length, 0.0], [0.0, 0.0, axis_length]],
        dtype=np.float32,
    )

    overlay_dir = output_dir / "overlays"
    summary_path = output_dir / "predictions.json"
    gt_summary_path = output_dir / "gt_poses.json"
    metadata_path = output_dir / "run_metadata.json"
    video_path = output_dir / "overlays.mp4"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    gt_records = []
    for frame_name in tqdm(frame_names, desc=f"{scene_name}:{object_name}"):
        image_path = scene_dir / "out_image" / frame_name / "camera.jpg"
        pose_path = scene_dir / "out_pose" / f"{frame_name}.npz"
        camera_path = scene_dir / "out_cam_param" / frame_name / "camera_camera.npz"

        scene_tensor, depth_tensor, mask_tensor = load_scene_frame_inputs(scene_dir, frame_name, resolution, device)
        intrinsic, world_to_camera, camera_to_world, width, height = load_camera_params(camera_path)

        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
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
            raise RuntimeError(f"Model output does not contain object pose keys: {sorted(outputs.keys())}")

        pred_rot6d_cam = outputs["object_pose"][0].detach().float().cpu().numpy()
        pred_translation_cam = outputs["object_translation"][0].detach().float().cpu().numpy().astype(np.float32)
        pred_rotation_cam = rot6d_to_matrix(pred_rot6d_cam) @ PREDICTION_ROTATION_FIX
        pred_rotation_world = camera_to_world[:3, :3] @ pred_rotation_cam
        pred_translation_world = camera_to_world[:3, :3] @ pred_translation_cam + camera_to_world[:3, 3]

        pred_bbox_cam = bbox_obj @ pred_rotation_cam.T + pred_translation_cam[None, :]
        pred_center_cam = center_obj @ pred_rotation_cam.T + pred_translation_cam
        pred_axis_cam = axis_obj @ pred_rotation_cam.T + pred_translation_cam[None, :]

        gt_record = None
        gt_bbox_cam = None
        gt_center_cam = None
        gt_axis_cam = None
        if pose_path.is_file():
            pose_lookup = load_pose_lookup(pose_path)
            gt_pose = pose_lookup.get(object_name)
            if gt_pose is not None:
                gt_translation_world = gt_pose["translation_world"].astype(np.float32)
                gt_rotation_world = quat_wxyz_to_matrix(gt_pose["quat_wxyz"]).astype(np.float32)
                gt_rotation_cam = world_to_camera[:3, :3] @ gt_rotation_world
                gt_translation_cam = world_to_camera[:3, :3] @ gt_translation_world + world_to_camera[:3, 3]
                gt_bbox_cam = bbox_obj @ gt_rotation_cam.T + gt_translation_cam[None, :]
                gt_center_cam = center_obj @ gt_rotation_cam.T + gt_translation_cam
                gt_axis_cam = axis_obj @ gt_rotation_cam.T + gt_translation_cam[None, :]
                gt_record = {
                    "translation_world": gt_translation_world,
                    "rotation_world": gt_rotation_world,
                    "translation_cam": gt_translation_cam,
                    "rotation_cam": gt_rotation_cam,
                    "quat_wxyz": gt_pose["quat_wxyz"],
                }

        overlay_path = overlay_dir / f"{frame_name}.jpg"
        draw_bbox_and_axes(
            image_path=image_path,
            intrinsic=intrinsic,
            bbox_points_cam=pred_bbox_cam,
            center_cam=pred_center_cam,
            axis_points_cam=pred_axis_cam,
            output_path=overlay_path,
            label=object_name,
            bbox_color=(0, 255, 0),
            axis_colors=((255, 64, 64), (0, 255, 255), (255, 215, 0)),
            width=width,
            height=height,
            gt_bbox_points_cam=gt_bbox_cam if draw_gt else None,
            gt_center_cam=gt_center_cam if draw_gt else None,
            gt_axis_points_cam=gt_axis_cam if draw_gt else None,
        )

        record = {
            "frame_name": frame_name,
            "image_path": str(image_path),
            "overlay_path": str(overlay_path),
            "pred_translation_cam": pred_translation_cam,
            "pred_rotation_cam": pred_rotation_cam,
            "pred_translation_world": pred_translation_world,
            "pred_rotation_world": pred_rotation_world,
        }
        if gt_record is not None:
            record["gt"] = gt_record
            gt_records.append(
                {
                    "frame_name": frame_name,
                    "image_path": str(image_path),
                    "translation_world": gt_record["translation_world"],
                    "rotation_world": gt_record["rotation_world"],
                    "translation_cam": gt_record["translation_cam"],
                    "rotation_cam": gt_record["rotation_cam"],
                    "quat_wxyz": gt_record["quat_wxyz"],
                }
            )
        records.append({key: to_serializable(value) for key, value in record.items()})

    video_written = export_overlay_video(overlay_dir, video_path, video_fps) if export_video else False
    metadata = {
        "scene": scene_name,
        "object_name": object_name,
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "object_render_root": str(args.object_render_root),
        "obj_root": str(args.obj_root),
        "output_dir": str(output_dir),
        "device": str(device),
        "frame_count": len(records),
        "object_views": list(object_views),
        "resolution": list(resolution),
        "use_depth_input": bool(use_depth_input),
        "mesh_scale_applied": mesh_scale,
        "draw_gt": bool(draw_gt),
        "video_path": str(video_path) if video_written else None,
        "video_fps": int(video_fps),
    }

    summary_path.write_text(json.dumps(records, indent=2))
    gt_summary_path.write_text(json.dumps(to_serializable(gt_records), indent=2))
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"Saved {len(records)} overlay frames to {overlay_dir}")
    print(f"Saved pose summary to {summary_path}")
    print(f"Saved GT pose summary to {gt_summary_path}")
    if video_written:
        print(f"Saved overlay video to {video_path}")
    print(f"Saved run metadata to {metadata_path}")
    return metadata


def main() -> int:
    args = parse_args()

    if not args.config.is_file():
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    scene_dir = args.dataset_root / args.scene
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Scene not found: {scene_dir}")

    cfg = load_config(args.config)
    runtime_settings = resolve_runtime_settings(cfg)
    resolution = runtime_settings["resolution"]
    object_views = tuple(args.object_views or runtime_settings["object_input_views"])

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model_from_config(cfg, args.checkpoint, device)

    frame_names = filter_frame_names(list_frame_names(scene_dir), args.frame_start, args.frame_end)
    if not frame_names:
        raise RuntimeError(f"No frames found for scene {args.scene}")

    object_names = [args.object_name] if args.object_name else list_scene_object_names(scene_dir, frame_names)
    if not object_names:
        raise RuntimeError(f"No objects found in scene {args.scene}")

    multi_object = len(object_names) > 1
    all_metadata = []
    for object_name in object_names:
        output_dir = make_output_dir(args.output_dir, args.scene, object_name, multi_object)
        all_metadata.append(
            run_inference_for_object(
                model=model,
                scene_dir=scene_dir,
                scene_name=args.scene,
                object_name=object_name,
                object_render_root=args.object_render_root,
                obj_root=args.obj_root,
                frame_names=frame_names,
                resolution=resolution,
                object_views=object_views,
                device=device,
                output_dir=output_dir,
                use_depth_input=args.use_depth_input,
                draw_gt=args.draw_gt,
                scale_mode=args.scale_mode,
                export_video=args.export_video,
                video_fps=args.video_fps,
                args=args,
            )
        )

    if multi_object:
        manifest_path = args.output_dir / "scene_manifest.json"
        manifest = {
            "scene": args.scene,
            "object_names": object_names,
            "frame_count": len(frame_names),
            "runs": all_metadata,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(to_serializable(manifest), indent=2))
        print(f"Saved scene manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
