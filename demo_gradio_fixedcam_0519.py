import argparse
import glob
import os
import re
import struct
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import gradio as gr
import numpy as np
import torch
import trimesh
from PIL import Image

from demo_gradio_6dpose_0519 import (
    PROJECT_ROOT,
    _LOCAL_TMP,
    DEFAULT_CONFIG_PATH,
    DEFAULT_PRETRAIN_MODEL,
    build_model_from_config,
    compat_image,
    depth_to_camera_points,
    draw_bbox_axes_overlay_on_image,
    load_config,
    mask_overlay_image,
    rot6d_to_matrix,
    DemoScenePreprocessor,
    crop_resize_image_depth_mask,
    PRED_AXIS_COLORS,
    GT_AXIS_COLORS,
    PRED_BBOX_COLOR,
    GT_BBOX_COLOR,
    BBOX_EDGES,
)


DEFAULT_FIXEDCAM_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose/0407_fixedCam_diffpose_15k_google")
DEFAULT_OBJ_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose/obj")
DEFAULT_OBJECT_IMAGE_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose/object_space_renders_all")
DEFAULT_EXTRA_OBJ_ROOTS = (
    Path("/mnt/train-data-4-hdd/yian/freepose/obj_test"),
    Path("/mnt/train-data-4-hdd/yian/freepose/obj_test_v2"),
)
DEFAULT_SCENE_VIEW = 0
DEFAULT_OBJECT_VIEWS = (1, 5, 10, 15)
AXIS_COLORS = PRED_AXIS_COLORS


OBJECT_RENDER_DIR_OVERRIDES = {
    "Assets_obj_google_0ac212e883884e7abe72a03aebecede0_0ac212e883884e7abe72a03aebecede0_glb": "test",
    "Assets_obj_google_513b3435fbe34a0f82440c9c51ef86db_513b3435fbe34a0f82440c9c51ef86db_glb": "513b3435fbe34a0f82440c9c51ef86db",
    "Assets_obj_google_776c684daac8458b98817df99b33456b_776c684daac8458b98817df99b33456b_glb": "776c684daac8458b98817df99b33456b",
    "Assets_obj_google_19429f27ad8a49cc97c99f2d783ad459_19429f27ad8a49cc97c99f2d783ad459_glb": "19429f27ad8a49cc97c99f2d783ad459",
    "Assets_obj_google_14b98b39474e49b49f0421b086a749aa_14b98b39474e49b49f0421b086a749aa_glb": "14b98b39474e49b49f0421b086a749aa",
    "Assets_obj_google_Krill_Oil_red_capsule_meshes_model": "google__Krill_Oil_red_capsule__meshes",
    "google_Krill_Oil_red_capsule_meshes": "google__Krill_Oil_red_capsule__meshes",
    "google_Krill_Oil_capsule_meshes": "google__Krill_Oil_red_capsule__meshes",
}


def view_filename(stem: str, view_idx: int, suffix: str) -> str:
    return f"{stem}{suffix}" if int(view_idx) == 0 else f"{stem}_({int(view_idx)}){suffix}"


def image_path_for_view(root: Path, run_name: str, view_idx: int) -> Path:
    return root / "out_image" / run_name / view_filename("Main_Camera", view_idx, ".jpg")


def depth_path_for_view(root: Path, run_name: str, view_idx: int) -> Path:
    return root / "out_depth" / run_name / view_filename("Main_Camera", view_idx, "_depth.png")


def camera_path_for_view(root: Path, run_name: str, view_idx: int) -> Path:
    return root / "out_cam_param" / run_name / view_filename("camera_Main_Camera", view_idx, ".npz")


def decode_name(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def canonicalize_object_name(object_name: str) -> str:
    if object_name.startswith("Assets_obj_google_") and object_name.endswith("_meshes_model"):
        return f"google_{object_name[len('Assets_obj_google_'):-len('_meshes_model')]}_meshes"
    if object_name.startswith("Assets_obj_google_") and object_name.endswith("_meshes"):
        return f"google_{object_name[len('Assets_obj_google_'):-len('_meshes')]}_meshes"
    if object_name.startswith("google_") and object_name.endswith("_meshes_model"):
        return f"google_{object_name[len('google_'):-len('_meshes_model')]}_meshes"
    return object_name


def load_pose_npz(root: Path, run_name: str):
    data = np.load(root / "out_pose" / f"{run_name}.npz", allow_pickle=False)
    return (
        [decode_name(x) for x in data["names"]],
        np.asarray(data["positions"], dtype=np.float32),
        np.asarray(data["rot_quat_wxyz"], dtype=np.float32),
    )


def find_object_index(names: Sequence[str], object_name: str):
    target = canonicalize_object_name(object_name)
    for idx, candidate in enumerate(names):
        if canonicalize_object_name(candidate) == target:
            return idx
    return None


def object_name_to_object_id(object_name: str) -> str:
    if object_name.startswith("google_") and object_name.endswith("_meshes_model"):
        return f"google/{object_name[len('google_'):-len('_meshes_model')]}/meshes"
    if object_name.startswith("google_") and object_name.endswith("_meshes"):
        return f"google/{object_name[len('google_'):-len('_meshes')]}/meshes"
    if object_name.startswith("Assets_obj_google_") and object_name.endswith("_meshes_model"):
        return f"google/{object_name[len('Assets_obj_google_'):-len('_meshes_model')]}/meshes"
    if object_name.startswith("Assets_obj_google_") and object_name.endswith("_meshes"):
        return f"google/{object_name[len('Assets_obj_google_'):-len('_meshes')]}/meshes"
    for prefix, dataset_name in {
        "freepose_obj_ycbv_": "ycbv",
        "freepose_obj_handal_": "handal",
        "freepose_obj_hope_": "hope",
        "freepose_obj_rupac_": "rupac",
    }.items():
        if object_name.startswith(prefix):
            return f"{dataset_name}/{object_name[len(prefix):]}"
    for prefix, dataset_name in {
        "ycbv_obj_": "ycbv",
        "handal_obj_": "handal",
        "hope_obj_": "hope",
        "rupac_obj_": "rupac",
    }.items():
        if object_name.startswith(prefix):
            return f"{dataset_name}/{object_name[len(dataset_name) + 1:]}"
    raise KeyError(f"Unsupported object name: {object_name}")


def object_id_to_render_dirname(object_id: str) -> str:
    return object_id.replace("/", "__")


def object_render_dirname(object_name: str) -> str:
    return OBJECT_RENDER_DIR_OVERRIDES.get(object_name) or object_id_to_render_dirname(object_name_to_object_id(object_name))


def object_render_dir(object_image_root: Path, object_name: str) -> Path:
    return object_image_root / object_render_dirname(object_name)


def object_image_path(object_image_root: Path, object_name: str, view_idx: int) -> Path:
    filename = "Main_Camera_rgb.png" if int(view_idx) == 0 else f"Main_Camera_({int(view_idx)})_rgb.png"
    path = object_render_dir(object_image_root, object_name) / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing object render image: {path}")
    return path


def available_object_views(object_image_root: Path, object_name: str) -> List[int]:
    views = []
    for path in sorted(glob.glob(str(object_render_dir(object_image_root, object_name) / "*_rgb.png"))):
        name = Path(path).name
        if name == "Main_Camera_rgb.png":
            views.append(0)
        else:
            match = re.match(r"Main_Camera_\((\d+)\)_rgb\.png$", name)
            if match:
                views.append(int(match.group(1)))
    return sorted(set(views))


def dataset_scale(name_or_path: str) -> float:
    s = str(name_or_path).lower()
    if "ycbv" in s:
        return 0.002
    if "handal" in s:
        return 0.0015
    if "hope" in s:
        return 0.003
    if "rupac" in s:
        return 0.002
    if "google" in s:
        return 0.2
    return 1.0


def object_mesh_path(obj_root: Path, extra_roots: Sequence[Path], object_name: str) -> Path:
    object_id = object_name_to_object_id(object_name)
    parts = object_id.split("/")
    roots = [Path(obj_root), *[Path(p) for p in extra_roots]]
    candidates = []
    if parts[0] == "google":
        candidates = [root / parts[0] / parts[1] / parts[2] / "model.obj" for root in roots]
    else:
        for root in roots:
            candidates.extend([root / parts[0] / f"obj_{parts[1]}.ply", root / parts[0] / f"obj_{parts[1]}.obj"])
    for path in candidates:
        if path.is_file():
            return path
    if parts[0] == "google":
        target_norm = re.sub(r"[^a-z0-9]+", "", parts[1].lower())
        for root in roots:
            google_root = root / "google"
            if not google_root.is_dir():
                continue
            for obj_dir in google_root.iterdir():
                if re.sub(r"[^a-z0-9]+", "", obj_dir.name.lower()) != target_norm:
                    continue
                matches = []
                for pattern in ("meshes/model.obj", "meshes/model.ply", "meshes/model.glb", "*.obj", "*.ply", "*.glb"):
                    matches.extend(obj_dir.glob(pattern))
                if matches:
                    return sorted(matches)[0]
    raise FileNotFoundError(f"Missing mesh for {object_name}")


def load_ply_xyz(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        if f.readline().decode("ascii", errors="ignore").strip() != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        fmt = None
        vertex_count = None
        properties = []
        in_vertex = False
        while True:
            line = f.readline().decode("ascii", errors="ignore")
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
        if fmt == "ascii":
            return np.asarray([[float(x) for x in f.readline().decode("ascii", errors="ignore").split()[:3]] for _ in range(vertex_count)], dtype=np.float32)
        if fmt != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format {fmt}: {path}")
        type_map = {
            "char": "b", "uchar": "B", "int8": "b", "uint8": "B",
            "short": "h", "ushort": "H", "int16": "h", "uint16": "H",
            "int": "i", "uint": "I", "int32": "i", "uint32": "I",
            "float": "f", "float32": "f", "double": "d", "float64": "d",
        }
        fmt_str = "<" + "".join(type_map[t] for t, _ in properties)
        row_size = struct.calcsize(fmt_str)
        rows = [struct.unpack(fmt_str, f.read(row_size))[:3] for _ in range(vertex_count)]
    return np.asarray(rows, dtype=np.float32)


def load_mesh_vertices(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".ply":
        return load_ply_xyz(path)
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No mesh geometry in {path}")
        loaded = trimesh.util.concatenate(meshes)
    return np.asarray(loaded.vertices, dtype=np.float32)


def bbox_corners_from_vertices(vertices: np.ndarray) -> np.ndarray:
    pmin = np.asarray(vertices, dtype=np.float32).min(axis=0)
    pmax = np.asarray(vertices, dtype=np.float32).max(axis=0)
    return np.asarray(
        [
            [pmin[0], pmin[1], pmin[2]], [pmax[0], pmin[1], pmin[2]],
            [pmin[0], pmax[1], pmin[2]], [pmax[0], pmax[1], pmin[2]],
            [pmin[0], pmin[1], pmax[2]], [pmax[0], pmin[1], pmax[2]],
            [pmin[0], pmax[1], pmax[2]], [pmax[0], pmax[1], pmax[2]],
        ],
        dtype=np.float32,
    )


def read_encoded_depth(depth_path: Path) -> np.ndarray:
    raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Failed to read depth image: {depth_path}")
    depth = raw.view(np.float16).astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0] = 0.0
    return depth


def load_camera(camera_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(camera_path, allow_pickle=False)
    extrinsic = np.asarray(data["extrinsics.opencv.worldToCamera16"], dtype=np.float32).reshape(4, 4)[:3]
    intrinsic = np.asarray(data["intrinsics.K_flat9"], dtype=np.float32).reshape(3, 3)
    return extrinsic, intrinsic


def quat_wxyz_to_matrix(quat) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def rotation_error_degrees(pred_rot: np.ndarray, gt_rot: np.ndarray) -> float:
    rel = np.asarray(pred_rot, dtype=np.float64) @ np.asarray(gt_rot, dtype=np.float64).T
    cos_theta = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def translation_error(pred_t: np.ndarray, gt_t: np.ndarray) -> Dict:
    diff = np.asarray(pred_t, dtype=np.float64) - np.asarray(gt_t, dtype=np.float64)
    return {"l2": float(np.linalg.norm(diff)), "abs_xyz": np.abs(diff).tolist()}


def load_scene_frame(root: Path, run_name: str, view_idx: int, resolution, device):
    image = Image.open(image_path_for_view(root, run_name, view_idx)).convert("RGB")
    depthmap = read_encoded_depth(depth_path_for_view(root, run_name, view_idx))
    _, intrinsic = load_camera(camera_path_for_view(root, run_name, view_idx))
    processor = DemoScenePreprocessor(resolution=resolution)
    image, depthmap, _, intrinsic = crop_resize_image_depth_mask(
        processor, image, depthmap, None, intrinsic, resolution,
        info=str(image_path_for_view(root, run_name, view_idx)),
    )
    image_tensor = processor.transform(image).unsqueeze(0).unsqueeze(0).to(device)
    depth_tensor = torch.from_numpy(np.ascontiguousarray(depthmap.astype(np.float32)))[None, None, :, :, None].to(device)
    mask_tensor = torch.from_numpy((depthmap > 0).astype(np.float32))[None, None, :, :].to(device)
    return image_tensor, depth_tensor, mask_tensor, np.asarray(image), depthmap, intrinsic


def load_object_tensor(object_image_root: Path, object_name: str, object_views: Sequence[int], resolution, device):
    processor = DemoScenePreprocessor(resolution=resolution)
    tensors = []
    gallery = []
    resampling = getattr(Image, "Resampling", Image)
    for view_idx in object_views:
        path = object_image_path(object_image_root, object_name, int(view_idx))
        image = Image.open(path).convert("RGB").resize(tuple(resolution), resampling.LANCZOS)
        tensors.append(processor.transform(image).to(device))
        gallery.append((np.asarray(image), f"Object view {int(view_idx)}"))
    return torch.stack(tensors, dim=0).unsqueeze(0), gallery


def export_fixedcam_point_cloud_pose_glb(
    run_name: str,
    view_idx: int,
    rgb_image: np.ndarray,
    depthmap: np.ndarray,
    intrinsic: np.ndarray,
    pred_rotation: np.ndarray,
    pred_translation: np.ndarray,
    pred_bbox_obj: np.ndarray,
    gt_rotation: np.ndarray | None,
    gt_translation: np.ndarray | None,
    gt_bbox_obj: np.ndarray | None,
    axis_length: float,
    stride: int,
) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", run_name)
    out_dir = _LOCAL_TMP / "fixedcam_pose_glb" / safe
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"view_{int(view_idx):06d}.glb"
    points, colors = depth_to_camera_points(depthmap, intrinsic, rgb_image, stride=max(1, int(stride)), max_points=80000)
    scene = trimesh.Scene()
    if len(points):
        scene.add_geometry(trimesh.PointCloud(vertices=points, colors=colors))

    def add_segment(start, end, color):
        if float(np.linalg.norm(np.asarray(end) - np.asarray(start))) < 1e-8:
            return
        mesh = trimesh.creation.cylinder(radius=max(axis_length * 0.015, 2e-4), segment=np.stack([start, end], axis=0))
        rgba = np.array([[color[0], color[1], color[2], 255]], dtype=np.uint8)
        mesh.visual.face_colors = np.tile(rgba, (len(mesh.faces), 1))
        scene.add_geometry(mesh)

    def add_pose(rotation, translation, bbox_obj, bbox_color, axis_colors):
        center = np.asarray(translation, dtype=np.float32)
        axes = np.asarray([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]], dtype=np.float32)
        axes_cam = axes @ rotation.T + center[None, :]
        for i, color in enumerate(axis_colors):
            add_segment(center, axes_cam[i], color)
        bbox_cam = np.asarray(bbox_obj, dtype=np.float32) @ rotation.T + center[None, :]
        for i, j in BBOX_EDGES:
            add_segment(bbox_cam[i], bbox_cam[j], bbox_color)

    add_pose(pred_rotation, pred_translation, pred_bbox_obj, PRED_BBOX_COLOR, PRED_AXIS_COLORS)
    if gt_rotation is not None and gt_translation is not None and gt_bbox_obj is not None:
        add_pose(gt_rotation, gt_translation, gt_bbox_obj, GT_BBOX_COLOR, GT_AXIS_COLORS)
    scene.export(out_path)
    return str(out_path)


class FixedCamDemoApp:
    def __init__(
        self,
        config_path: Path,
        checkpoint_path: str | None,
        data_root: Path,
        obj_root: Path,
        object_image_root: Path,
        extra_obj_roots: Sequence[Path] = DEFAULT_EXTRA_OBJ_ROOTS,
        depth_median_min: float | None = None,
        depth_median_max: float | None = None,
        depth_filter_view: int = DEFAULT_SCENE_VIEW,
    ):
        self.config_path = Path(config_path)
        self.cfg = load_config(self.config_path)
        self.resolution = tuple(int(v) for v in self.cfg.get("resolution", (518, 476)))
        self.data_root = Path(data_root)
        self.obj_root = Path(obj_root)
        self.object_image_root = Path(object_image_root)
        self.extra_obj_roots = tuple(Path(p) for p in extra_obj_roots)
        self.object_views = DEFAULT_OBJECT_VIEWS
        self.depth_median_min = depth_median_min
        self.depth_median_max = depth_median_max
        self.depth_filter_view = int(depth_filter_view)
        self._depth_median_cache: Dict[Tuple[str, int], float] = {}
        all_runs = sorted(p.name for p in (self.data_root / "out_image").iterdir() if p.is_dir() and p.name.startswith("run_"))
        self.runs = [run for run in all_runs if self.run_passes_depth_filter(run)]
        if not self.runs:
            raise RuntimeError(
                f"No runs found under {self.data_root / 'out_image'} matching depth filter "
                f"median_min={self.depth_median_min} median_max={self.depth_median_max} "
                f"view={self.depth_filter_view}"
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_PRETRAIN_MODEL)
        self.model = build_model_from_config(self.cfg, self.checkpoint_path, self.device)

    @property
    def has_depth_filter(self) -> bool:
        return self.depth_median_min is not None or self.depth_median_max is not None

    def depth_median_for_frame(self, run_name: str, view_idx: int) -> float:
        key = (str(run_name), int(view_idx))
        if key not in self._depth_median_cache:
            depth = read_encoded_depth(depth_path_for_view(self.data_root, run_name, int(view_idx)))
            valid = depth[np.isfinite(depth) & (depth > 0)]
            self._depth_median_cache[key] = float(np.median(valid)) if valid.size else float("nan")
        return self._depth_median_cache[key]

    def run_passes_depth_filter(self, run_name: str) -> bool:
        if not self.has_depth_filter:
            return True
        try:
            median = self.depth_median_for_frame(run_name, self.depth_filter_view)
        except Exception:
            return False
        if not np.isfinite(median):
            return False
        if self.depth_median_min is not None and median < float(self.depth_median_min):
            return False
        if self.depth_median_max is not None and median > float(self.depth_median_max):
            return False
        return True

    def frame_choices(self, run_name: str) -> List[str]:
        paths = sorted((self.data_root / "out_image" / run_name).glob("Main_Camera*.jpg"))
        choices = []
        for path in paths:
            if path.name == "Main_Camera.jpg":
                choices.append("000000")
            else:
                match = re.match(r"Main_Camera_\((\d+)\)\.jpg$", path.name)
                if match:
                    choices.append(f"{int(match.group(1)):06d}")
        return sorted(choices)

    def object_choices(self, run_name: str) -> List[str]:
        names, _, _ = load_pose_npz(self.data_root, run_name)
        merged = []
        for name in names:
            canonical = canonicalize_object_name(name)
            if canonical in merged:
                continue
            try:
                if all(v in available_object_views(self.object_image_root, canonical) for v in self.object_views):
                    object_mesh_path(self.obj_root, self.extra_obj_roots, canonical)
                    merged.append(canonical)
            except Exception:
                continue
        return merged

    def input_gallery(self, run_name: str, frame_name: str, object_name: str):
        if not (run_name and frame_name and object_name):
            return None, []
        _, gallery = load_object_tensor(
            self.object_image_root, object_name, self.object_views, self.resolution, torch.device("cpu")
        )
        return str(image_path_for_view(self.data_root, run_name, int(frame_name))), gallery

    def object_bbox(self, object_name: str, scale_multiplier: float) -> np.ndarray:
        mesh_path = object_mesh_path(self.obj_root, self.extra_obj_roots, object_name)
        vertices = load_mesh_vertices(mesh_path)
        vertices = vertices * float(dataset_scale(object_name)) * float(scale_multiplier)
        return bbox_corners_from_vertices(vertices)

    def gt_pose_camera(self, run_name: str, frame_idx: int, object_name: str):
        names, positions, quats = load_pose_npz(self.data_root, run_name)
        idx = find_object_index(names, object_name)
        if idx is None:
            return None, None
        r_obj = quat_wxyz_to_matrix(quats[idx])
        t_obj = positions[idx].astype(np.float32)
        extrinsic, _ = load_camera(camera_path_for_view(self.data_root, run_name, frame_idx))
        r_cam = extrinsic[:, :3] @ r_obj
        t_cam = extrinsic[:, :3] @ t_obj + extrinsic[:, 3]
        return r_cam.astype(np.float32), t_cam.astype(np.float32)

    def run_inference(
        self,
        run_name: str,
        frame_name: str,
        object_name: str,
        use_depth_input: bool,
        show_point_cloud: bool,
        point_cloud_stride: int,
        object_scale_multiplier: float,
    ):
        frame_idx = int(frame_name)
        scene_tensor, depth_tensor, mask_tensor, display_image, display_depth, intrinsic = load_scene_frame(
            self.data_root, run_name, frame_idx, self.resolution, self.device
        )
        object_tensor, _ = load_object_tensor(
            self.object_image_root, object_name, self.object_views, self.resolution, self.device
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
            raise RuntimeError(f"Model output missing pose keys: {sorted(outputs.keys())}")
        pred_rotation = rot6d_to_matrix(outputs["object_pose"][0].detach().float().cpu().numpy()).astype(np.float32)
        pred_translation = outputs["object_translation"][0].detach().float().cpu().numpy().astype(np.float32)
        bbox_obj = self.object_bbox(object_name, object_scale_multiplier)
        diag = float(np.linalg.norm(bbox_obj.max(axis=0) - bbox_obj.min(axis=0)))
        axis_length = max(diag * 0.25, 1e-3)

        gt_rotation, gt_translation = self.gt_pose_camera(run_name, frame_idx, object_name)
        pred_image = draw_bbox_axes_overlay_on_image(
            display_image, intrinsic, pred_rotation, pred_translation, bbox_obj,
            axis_length, PRED_AXIS_COLORS, PRED_BBOX_COLOR,
        )
        gt_image = None
        if gt_rotation is not None:
            gt_image = draw_bbox_axes_overlay_on_image(
                display_image, intrinsic, gt_rotation, gt_translation, bbox_obj,
                axis_length, GT_AXIS_COLORS, GT_BBOX_COLOR,
            )

        pred_mask_image = None
        if "object_mask" in outputs:
            pred_mask = outputs["object_mask"][0, 0].detach().float().cpu().numpy()
            pred_mask_image = mask_overlay_image(display_image, pred_mask, color=(255, 64, 64))

        point_cloud_glb = None
        if show_point_cloud:
            point_cloud_glb = export_fixedcam_point_cloud_pose_glb(
                run_name, frame_idx, display_image, display_depth, intrinsic,
                pred_rotation, pred_translation, bbox_obj,
                gt_rotation, gt_translation, bbox_obj if gt_rotation is not None else None,
                axis_length, point_cloud_stride,
            )

        lines = [
            "### Prediction Summary",
            f"- checkpoint: `{self.checkpoint_path}`",
            f"- run/frame: `{run_name}` / `{frame_idx:06d}`",
            f"- object: `{object_name}`",
            f"- object views: `{list(self.object_views)}`",
            f"- use depth input: `{bool(use_depth_input)}`",
            f"- frame depth median (m): `{self.depth_median_for_frame(run_name, frame_idx):.4f}`",
            f"- object bbox scale multiplier: `{float(object_scale_multiplier):.4f}`",
            "",
            "### Predicted Pose (camera frame)",
            f"- translation (m): `{np.round(pred_translation, 6).tolist()}`",
            f"- rotation matrix: `{np.round(pred_rotation, 6).tolist()}`",
        ]
        if "object_presence_logits" in outputs:
            logit = float(outputs["object_presence_logits"].reshape(-1)[0].detach().float().cpu())
            prob = float(torch.sigmoid(torch.tensor(logit)).item())
            lines.insert(6, f"- predicted presence probability: `{prob:.6f}` (logit `{logit:.6f}`)")
        if gt_rotation is not None:
            terr = translation_error(pred_translation, gt_translation)
            lines += [
                "",
                "### Ground Truth Pose (camera frame)",
                f"- translation (m): `{np.round(gt_translation, 6).tolist()}`",
                f"- rotation matrix: `{np.round(gt_rotation, 6).tolist()}`",
                "",
                "### Errors",
                f"- translation L2 (m): `{terr['l2']:.6f}`",
                f"- translation abs xyz (m): `{np.round(terr['abs_xyz'], 6).tolist()}`",
                f"- rotation error (deg): `{rotation_error_degrees(pred_rotation, gt_rotation):.6f}`",
            ]
        return "\n".join(lines), pred_image, gt_image, point_cloud_glb, pred_mask_image


def build_demo(app: FixedCamDemoApp, image_focused_layout: bool = False):
    default_run = app.runs[0]
    default_frames = app.frame_choices(default_run)
    default_frame = f"{DEFAULT_SCENE_VIEW:06d}" if f"{DEFAULT_SCENE_VIEW:06d}" in default_frames else default_frames[0]
    default_objects = app.object_choices(default_run)
    default_object = default_objects[0] if default_objects else None
    css = ".gradio-container { max-width: 100% !important; } #scene_inputs img,#pred_projection img,#gt_projection img{object-fit:contain!important;}"
    info = "\n".join(
        [
            "# OmniVGGT 6D Pose Demo (FixedCam Google)",
            f"- data root: `{app.data_root}`",
            f"- obj root: `{app.obj_root}`",
            f"- object renders: `{app.object_image_root}`",
            f"- checkpoint: `{app.checkpoint_path}`",
            f"- resolution: `{app.resolution}`",
            f"- depth filter: `median_min={app.depth_median_min}, median_max={app.depth_median_max}, view={app.depth_filter_view}`",
            "",
            "選擇 `run`、單張 scene frame 和其中一個物體；object image 使用固定 1/5/10/15 視角。BBox 由 `/obj` mesh 頂點算出，並提供 scale multiplier 微調。",
        ]
    )
    with gr.Blocks(title="OmniVGGT FixedCam Demo", css=css) as demo:
        gr.Markdown(info)
        with gr.Row():
            run_dropdown = gr.Dropdown(choices=app.runs, value=default_run, label="Run")
            frame_dropdown = gr.Dropdown(choices=default_frames, value=default_frame, label="Frame")
            object_dropdown = gr.Dropdown(choices=default_objects, value=default_object, label="Object")
            use_depth = gr.Checkbox(value=True, label="Use Depth Input")
            show_pcd = gr.Checkbox(value=False, label="Show Point Cloud Pose")
            stride = gr.Slider(1, 8, value=2, step=1, label="Point Cloud Density")
            scale_mult = gr.Slider(0.05, 5.0, value=1.0, step=0.05, label="Object BBox Scale")
            run_button = gr.Button("Run Inference", variant="primary")
        big_height = "38vh" if image_focused_layout else 360
        with gr.Row():
            with gr.Column():
                scene_image = compat_image(label="Scene Input", height=big_height, elem_id="scene_inputs", interactive=False)
                object_gallery = gr.Gallery(label="Object Inputs", columns=4, height=180, object_fit="contain")
            with gr.Column():
                pred_image = compat_image(label="Predicted Pose", height=big_height, elem_id="pred_projection", interactive=False)
                gt_image = compat_image(label="GT Pose", height=big_height, elem_id="gt_projection", interactive=False)
                model3d = gr.Model3D(label="Depth Point Cloud + Pose", height=480)
                pred_mask = compat_image(label="Predicted Mask", height=180, interactive=False)
        summary = gr.Markdown()

        def refresh_run(run_name):
            frames = app.frame_choices(run_name)
            frame_value = f"{DEFAULT_SCENE_VIEW:06d}" if f"{DEFAULT_SCENE_VIEW:06d}" in frames else (frames[0] if frames else None)
            objects = app.object_choices(run_name)
            object_value = objects[0] if objects else None
            scene, gallery = app.input_gallery(run_name, frame_value, object_value)
            return gr.update(choices=frames, value=frame_value), gr.update(choices=objects, value=object_value), scene, gallery

        def refresh_inputs(run_name, frame_name, object_name):
            return app.input_gallery(run_name, frame_name, object_name)

        run_dropdown.change(refresh_run, inputs=run_dropdown, outputs=[frame_dropdown, object_dropdown, scene_image, object_gallery])
        frame_dropdown.change(refresh_inputs, inputs=[run_dropdown, frame_dropdown, object_dropdown], outputs=[scene_image, object_gallery])
        object_dropdown.change(refresh_inputs, inputs=[run_dropdown, frame_dropdown, object_dropdown], outputs=[scene_image, object_gallery])
        run_button.click(
            app.run_inference,
            inputs=[run_dropdown, frame_dropdown, object_dropdown, use_depth, show_pcd, stride, scale_mult],
            outputs=[summary, pred_image, gt_image, model3d, pred_mask],
        )
        if default_object is not None:
            demo.load(refresh_inputs, inputs=[run_dropdown, frame_dropdown, object_dropdown], outputs=[scene_image, object_gallery])
    return demo


def main():
    parser = argparse.ArgumentParser(description="Gradio demo for OmniVGGT inference on fixed-camera FreePose data")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_PRETRAIN_MODEL))
    parser.add_argument("--fixedcam-root", type=Path, default=DEFAULT_FIXEDCAM_ROOT)
    parser.add_argument("--obj-root", type=Path, default=DEFAULT_OBJ_ROOT)
    parser.add_argument("--object-image-root", type=Path, default=DEFAULT_OBJECT_IMAGE_ROOT)
    parser.add_argument("--fixedcam-depth-median-min", type=float, default=None)
    parser.add_argument("--fixedcam-depth-median-max", type=float, default=None)
    parser.add_argument(
        "--fixedcam-depth-filter-view",
        type=int,
        default=DEFAULT_SCENE_VIEW,
        help="Camera view index used to compute each run's depth median filter.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--image-focused-layout", action="store_true")
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    app = FixedCamDemoApp(
        args.config,
        args.checkpoint,
        args.fixedcam_root,
        args.obj_root,
        args.object_image_root,
        depth_median_min=args.fixedcam_depth_median_min,
        depth_median_max=args.fixedcam_depth_median_max,
        depth_filter_view=args.fixedcam_depth_filter_view,
    )
    demo = build_demo(app, image_focused_layout=args.image_focused_layout)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[str(app.data_root), str(app.obj_root), str(app.object_image_root), *[str(p) for p in app.extra_obj_roots]],
    )


if __name__ == "__main__":
    main()
