"""Render GIFs of HouseCat6D / YCB-V / REAL275 objects across all frames in a scene.

For each scene supplied via ``--scene-paths`` (or the default housecat6d list),
this iterates every unique object instance, runs the OmniVGGT model loaded from
``--checkpoint``, and draws on the cropped scene image:

* predicted 3-axis arrows on the predicted pose
* GT 3D bbox in red (GT size + GT pose)
* "predicted" 3D bbox in green (GT size + predicted pose) — isolates pose
  error from size prediction.

Dataset format is auto-detected from each scene path (looks for
``housecat6d`` / ``ycbv`` / ``real275`` substring).

Translation predictions are scaled by the per-frame depth mean (GT depth when
``--use-depth-input`` is on, otherwise the predicted depth mean), matching the
gradio demo's ``use_depth_scale=True`` path.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
_LOCAL_TMP = PROJECT_ROOT / "tmp"
_LOCAL_TMP.mkdir(parents=True, exist_ok=True)
(_LOCAL_TMP / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(_LOCAL_TMP))
os.environ.setdefault("MPLCONFIGDIR", str(_LOCAL_TMP / "matplotlib"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import argparse
import pickle
import time
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

_BENCHMARK_REQUEST: Dict[str, Any] | None = None

from demo_gradio_6dpose_real import (
    AXIS_COLORS,
    BBOX_EDGES,
    DEFAULT_ALIGN_JSON,
    DEFAULT_CONFIG_PATH,
    DEFAULT_HOUSECAT6D_OBJECT_IMAGE_ROOT,
    DEFAULT_HOUSECAT6D_ROOT,
    DEFAULT_PRETRAIN_MODEL,
    DEFAULT_REAL275_GT_ROOT,
    DEFAULT_REAL275_OBJ_MODELS_ROOT,
    DEFAULT_REAL275_OBJECT_IMAGE_ROOT,
    DEFAULT_REAL275_TEST_ROOT,
    DEFAULT_YCBV_MODELS_INFO,
    DEFAULT_YCBV_OBJECT_IMAGE_ROOT,
    DEFAULT_YCBV_TEST_ROOT,
    build_model_from_config,
    centered_axis_bbox_corners,
    draw_axes_overlay_on_image,
    housecat6d_read_label,
    load_config,
    load_housecat6d_object_tensor,
    load_housecat6d_scene_frame_inputs,
    load_real275_object_tensor,
    load_real275_scene_frame_inputs,
    load_ycbv_object_tensor,
    load_ycbv_scene_frame_inputs,
    project_camera_points,
    read_json,
    real275_decompose_gt_rt,
    real275_read_canonical_extent,
    real275_read_meta,
    resolve_runtime_settings,
    rot6d_to_matrix,
    ycbv_object_key,
)

PRED_BBOX_COLOR = (0, 255, 0)
GT_BBOX_COLOR = (255, 0, 0)

DEFAULT_HOUSECAT6D_SCENES: Tuple[str, ...] = (
    "test_scene1",
    "test_scene2",
    "test_scene3",
    "test_scene4",
    "test_scene5",
    "val_scene1",
    "val_scene2",
    "scene01",
    "scene02",
)
DEFAULT_YCBV_SCENES: Tuple[str, ...] = ("000049", "000050")
DEFAULT_REAL275_SCENES: Tuple[str, ...] = ("scene_1", "scene_2")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "gif_outputs" / "0525_REAL_multi"


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _draw_bbox_lines(
    overlay: np.ndarray,
    intrinsic: np.ndarray,
    rotation_cam: np.ndarray,
    translation_cam: np.ndarray,
    bbox_obj: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    bbox_cam = (
        np.asarray(bbox_obj, dtype=np.float32) @ np.asarray(rotation_cam, dtype=np.float32).T
        + np.asarray(translation_cam, dtype=np.float32)[None, :]
    )
    uv, valid = project_camera_points(bbox_cam, intrinsic)
    height, width = overlay.shape[:2]
    rect = (0, 0, int(width), int(height))
    for start_idx, end_idx in BBOX_EDGES:
        if not (bool(valid[start_idx]) and bool(valid[end_idx])):
            continue
        p1 = tuple(np.round(uv[start_idx]).astype(np.int32))
        p2 = tuple(np.round(uv[end_idx]).astype(np.int32))
        ok, cp1, cp2 = cv2.clipLine(rect, p1, p2)
        if ok:
            cv2.line(overlay, cp1, cp2, color, int(thickness), lineType=cv2.LINE_AA)
    return overlay


def draw_pred_axes_pred_gt_bbox(
    image_rgb: np.ndarray,
    intrinsic: np.ndarray,
    pred_rotation_cam: np.ndarray,
    pred_translation_cam: np.ndarray,
    pred_bbox_obj: np.ndarray,
    gt_rotation_cam: np.ndarray,
    gt_translation_cam: np.ndarray,
    gt_bbox_obj: np.ndarray,
    axis_length: float,
) -> np.ndarray:
    overlay = draw_axes_overlay_on_image(
        image_rgb,
        intrinsic,
        pred_rotation_cam,
        pred_translation_cam,
        axis_length,
        AXIS_COLORS,
    )
    overlay = _draw_bbox_lines(
        overlay, intrinsic, gt_rotation_cam, gt_translation_cam, gt_bbox_obj, GT_BBOX_COLOR,
    )
    overlay = _draw_bbox_lines(
        overlay, intrinsic, pred_rotation_cam, pred_translation_cam, pred_bbox_obj, PRED_BBOX_COLOR,
    )
    return overlay


def compute_depth_mean_scale(
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


# ---------------------------------------------------------------------------
# Align + size helpers (HouseCat6D / YCBV / Real275)
# ---------------------------------------------------------------------------


def _size_aligned(native_size: np.ndarray, r_align: np.ndarray) -> np.ndarray:
    size_native = np.asarray(native_size, dtype=np.float32).reshape(3)
    size_aligned = np.abs(r_align) @ size_native
    return np.clip(size_aligned, 1e-6, None).astype(np.float32)


def load_align(align_json_path: Path) -> Dict[str, Any]:
    root = read_json(align_json_path)["datasets"]

    hc = root.get("housecat6d", {})
    hc_name_to_id = {str(k): int(v) for k, v in hc.get("category_name_to_id", {}).items()}
    hc_id_to_name = {v: k for k, v in hc_name_to_id.items()}
    hc_r_align = {
        str(name): np.asarray(item["R_align"], dtype=np.float32).reshape(3, 3)
        for name, item in hc.get("classes", {}).items()
    }

    ycbv = root["ycbv"]
    ycbv_global = np.asarray(ycbv["global_R_align"], dtype=np.float32).reshape(3, 3)
    ycbv_overrides = {
        int(k): np.asarray(v, dtype=np.float32).reshape(3, 3)
        for k, v in ycbv.get("per_object_overrides", {}).items()
    }
    ycbv_id_to_category = {int(k): str(v) for k, v in ycbv["obj_id_to_category"].items()}

    r275 = root.get("real275", {})
    r275_class_id_to_name = {int(k): str(v) for k, v in r275.get("class_id_to_name", {}).items()}
    r275_r_align = {
        int(k): np.asarray(v["R_align"], dtype=np.float32).reshape(3, 3)
        for k, v in r275.get("classes", {}).items()
    }

    return {
        "housecat6d": {
            "category_id_to_name": hc_id_to_name,
            "r_align_by_category": hc_r_align,
        },
        "ycbv": {
            "global_r_align": ycbv_global,
            "r_align_overrides": ycbv_overrides,
            "obj_id_to_category": ycbv_id_to_category,
        },
        "real275": {
            "class_id_to_name": r275_class_id_to_name,
            "r_align_by_class_id": r275_r_align,
        },
    }


def _hc_r_align(align: Dict[str, Any], category: str) -> np.ndarray:
    r = align["housecat6d"]["r_align_by_category"].get(str(category))
    if r is None:
        return np.eye(3, dtype=np.float32)
    return r.astype(np.float32)


def _ycbv_r_align(align: Dict[str, Any], object_id: int) -> np.ndarray:
    overrides = align["ycbv"]["r_align_overrides"]
    if int(object_id) in overrides:
        return overrides[int(object_id)].astype(np.float32)
    return align["ycbv"]["global_r_align"].astype(np.float32)


def _r275_r_align(align: Dict[str, Any], class_id: int) -> np.ndarray:
    r = align["real275"]["r_align_by_class_id"].get(int(class_id))
    if r is None:
        return np.eye(3, dtype=np.float32)
    return r.astype(np.float32)


def ycbv_size_native_m(models_info: Dict[str, Any], object_id: int) -> np.ndarray:
    info = models_info[str(int(object_id))]
    return np.array([info["size_x"], info["size_y"], info["size_z"]], dtype=np.float32) / 1000.0


# ---------------------------------------------------------------------------
# Object reference record builders
# ---------------------------------------------------------------------------


def build_housecat6d_object_records(
    object_image_root: Path, object_views: Sequence[int]
) -> Dict[str, Dict[str, Any]]:
    if not object_image_root.is_dir():
        return {}
    records: Dict[str, Dict[str, Any]] = {}
    for object_dir in sorted(object_image_root.iterdir()):
        if not object_dir.is_dir():
            continue
        rgb_dir = object_dir / "rgb"
        if not rgb_dir.is_dir():
            continue
        available = sorted(int(p.stem) for p in rgb_dir.glob("*.png") if p.stem.isdigit())
        if any(v not in available for v in object_views):
            continue
        records[object_dir.name] = {
            "object_name": object_dir.name,
            "object_dir": object_dir,
            "image_ids": available,
        }
    return records


def build_ycbv_object_records(
    object_image_root: Path, object_views: Sequence[int]
) -> Dict[int, Dict[str, Any]]:
    records: Dict[int, Dict[str, Any]] = {}
    if not object_image_root.is_dir():
        return records
    for object_dir in sorted(object_image_root.glob("obj_*")):
        stem = object_dir.name.removeprefix("obj_")
        if not object_dir.is_dir() or not stem.isdigit():
            continue
        rgb_dir = object_dir / "rgb"
        if not rgb_dir.is_dir():
            continue
        available = sorted(int(p.stem) for p in rgb_dir.glob("*.png") if p.stem.isdigit())
        if any(v not in available for v in object_views):
            continue
        records[int(stem)] = {
            "object_id": int(stem),
            "object_dir": object_dir,
            "image_ids": available,
        }
    return records


def build_real275_object_records(
    object_image_root: Path, object_views: Sequence[int]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[int, List[str]]]:
    records: Dict[str, Dict[str, Any]] = {}
    if not object_image_root.is_dir():
        return records, {}
    for object_dir in sorted(object_image_root.iterdir()):
        if not object_dir.is_dir():
            continue
        rgb_dir = object_dir / "rgb"
        if not rgb_dir.is_dir():
            continue
        available = sorted(int(p.stem) for p in rgb_dir.glob("*.png") if p.stem.isdigit())
        if any(v not in available for v in object_views):
            continue
        metadata_path = object_dir / "metadata.json"
        metadata = read_json(metadata_path) if metadata_path.is_file() else {}
        class_id = int(metadata.get("class_id", 0))
        records[object_dir.name] = {
            "object_name": object_dir.name,
            "object_dir": object_dir,
            "image_ids": available,
            "class_id": class_id,
        }
    by_class: Dict[int, List[str]] = {}
    for name, rec in records.items():
        by_class.setdefault(int(rec.get("class_id", 0)), []).append(name)
    for cid in by_class:
        by_class[cid].sort()
    return records, by_class


# ---------------------------------------------------------------------------
# Scene enumeration
# ---------------------------------------------------------------------------


def enumerate_housecat6d_frames(scene_dir: Path) -> List[int]:
    label_dir = scene_dir / "labels"
    if not label_dir.is_dir():
        return []
    return sorted(int(p.name.split("_", 1)[0]) for p in label_dir.glob("*_label.pkl"))


def enumerate_housecat6d_objects(
    scene_dir: Path, frame_ids: Sequence[int]
) -> List[Tuple[str, int]]:
    seen: Dict[str, int] = {}
    for frame_id in frame_ids:
        label_path = scene_dir / "labels" / f"{int(frame_id):06d}_label.pkl"
        if not label_path.is_file():
            continue
        label = housecat6d_read_label(label_path)
        for name, cid in zip(label.get("model_list", []), label.get("class_ids", [])):
            seen.setdefault(str(name), int(cid))
    return sorted(seen.items())


def enumerate_ycbv_frames(scene_dir: Path) -> List[int]:
    sgt = scene_dir / "scene_gt.json"
    if not sgt.is_file():
        return []
    return sorted(int(k) for k in read_json(sgt).keys())


def enumerate_ycbv_objects(scene_dir: Path) -> List[int]:
    sgt = scene_dir / "scene_gt.json"
    if not sgt.is_file():
        return []
    objs: set = set()
    for entries in read_json(sgt).values():
        for e in entries:
            objs.add(int(e["obj_id"]))
    return sorted(objs)


def enumerate_real275_frames(scene_dir: Path) -> List[int]:
    return sorted(int(p.name.split("_", 1)[0]) for p in scene_dir.glob("*_color.png"))


def enumerate_real275_objects(
    scene_dir: Path, frame_ids: Sequence[int]
) -> List[Tuple[int, int, str]]:
    """Return sorted [(inst_id, class_id, model_name)] for unique inst_id."""
    seen: Dict[int, Tuple[int, str]] = {}
    for frame_id in frame_ids:
        meta_path = scene_dir / f"{int(frame_id):04d}_meta.txt"
        for entry in real275_read_meta(meta_path):
            seen.setdefault(int(entry["inst_id"]), (int(entry["class_id"]), str(entry["model_name"])))
    return sorted((inst_id, cid, name) for inst_id, (cid, name) in seen.items())


# ---------------------------------------------------------------------------
# Model inference + pose extraction
# ---------------------------------------------------------------------------


def _run_inference(
    model, scene_tensor, object_tensor, depth_tensor, mask_tensor, use_depth_input, device,
    gpu_timing_ms: List[float] | None = None,
):
    use_event = gpu_timing_ms is not None and device.type == "cuda"
    start_evt = end_evt = None
    if use_event:
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            if use_event:
                start_evt.record()
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
            if use_event:
                end_evt.record()
    if use_event:
        torch.cuda.synchronize()
        gpu_timing_ms.append(start_evt.elapsed_time(end_evt))
    return outputs


def _benchmark_inference(
    *,
    model,
    scene_tensor,
    object_tensor,
    depth_tensor,
    mask_tensor,
    display_depth,
    use_depth_input: bool,
    use_depth_scale: bool,
    device: torch.device,
    warmup: int,
    runs: int,
    label: str,
) -> Dict[str, float]:
    print(f"[benchmark] {label}: warmup x{warmup}, timed x{runs}")
    for _ in range(warmup):
        outputs = _run_inference(
            model, scene_tensor, object_tensor, depth_tensor, mask_tensor, use_depth_input, device,
        )
        _extract_pred_pose(outputs, display_depth, use_depth_input, use_depth_scale)
    if device.type == "cuda":
        torch.cuda.synchronize()

    gpu_ms: List[float] = []
    pose_ms: List[float] = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = _run_inference(
            model, scene_tensor, object_tensor, depth_tensor, mask_tensor, use_depth_input, device,
            gpu_timing_ms=gpu_ms if device.type == "cuda" else None,
        )
        _extract_pred_pose(outputs, display_depth, use_depth_input, use_depth_scale)
        if device.type == "cuda":
            torch.cuda.synchronize()
        pose_ms.append((time.perf_counter() - t0) * 1000.0)

    pose_arr = np.asarray(pose_ms, dtype=np.float64)
    print(f"\n[benchmark results, ms over {runs} runs]")
    if gpu_ms:
        gpu_arr = np.asarray(gpu_ms, dtype=np.float64)
        print(
            f"  model.inference() GPU only      : "
            f"mean={gpu_arr.mean():7.2f}  median={np.median(gpu_arr):7.2f}  "
            f"min={gpu_arr.min():7.2f}  max={gpu_arr.max():7.2f}  std={gpu_arr.std():6.2f}"
        )
    print(
        f"  fwd + extract pose (until pose) : "
        f"mean={pose_arr.mean():7.2f}  median={np.median(pose_arr):7.2f}  "
        f"min={pose_arr.min():7.2f}  max={pose_arr.max():7.2f}  std={pose_arr.std():6.2f}"
    )
    fps = 1000.0 / pose_arr.mean() if pose_arr.mean() > 0 else float("nan")
    print(f"  => single-view throughput        : {fps:.2f} FPS (based on mean pose-output time)")
    return {"gpu_ms_mean": float(np.mean(gpu_ms)) if gpu_ms else float("nan"),
            "pose_ms_mean": float(pose_arr.mean())}


def _extract_pred_pose(outputs, display_depth, use_depth_input, use_depth_scale):
    if "object_pose" not in outputs or "object_translation" not in outputs:
        return None
    pred_rot6d = outputs["object_pose"][0].detach().float().cpu().numpy()
    pred_rotation = rot6d_to_matrix(pred_rot6d).astype(np.float32)
    pred_translation = outputs["object_translation"][0].detach().float().cpu().numpy().astype(np.float32)
    if use_depth_scale:
        depth_mean = compute_depth_mean_scale(display_depth, outputs.get("depth"), use_depth_input)
        pred_translation = pred_translation * np.float32(depth_mean)
    return pred_rotation, pred_translation


def _label_frame(frame_image: np.ndarray, text: str) -> None:
    cv2.putText(frame_image, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame_image, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _save_gif(frames: List[np.ndarray], output_path: Path, fps: float) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = 1000.0 / max(float(fps), 1e-3)
    imageio.mimsave(str(output_path), frames, format="GIF", duration=duration_ms, loop=0)
    return output_path


# ---------------------------------------------------------------------------
# HouseCat6D GIF renderer
# ---------------------------------------------------------------------------


def render_housecat6d_gif(
    *,
    model,
    scene_name: str,
    scene_dir: Path,
    frame_ids: Sequence[int],
    model_name: str,
    class_id: int,
    object_records: Dict[str, Dict[str, Any]],
    object_views: Sequence[int],
    align: Dict[str, Any],
    resolution: Tuple[int, int],
    device: torch.device,
    output_path: Path,
    use_depth_input: bool,
    use_depth_scale: bool,
    fps: float,
    max_frames: int | None,
    save_frames: bool = False,
) -> Path | None:
    if model_name not in object_records:
        print(f"[skip] housecat6d/{scene_name}/{model_name}: no aligned object refs")
        return None

    object_tensor, _ = load_housecat6d_object_tensor(
        object_records, model_name, object_views, resolution, device
    )
    category = align["housecat6d"]["category_id_to_name"].get(int(class_id), "unknown")
    r_align = _hc_r_align(align, category)

    iter_frames = list(frame_ids[: int(max_frames)] if max_frames is not None else frame_ids)
    frames: List[np.ndarray] = []
    frames_dir = output_path.with_name(output_path.stem + "_frames") if save_frames else None
    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)
    for frame_id in iter_frames:
        label_path = scene_dir / "labels" / f"{int(frame_id):06d}_label.pkl"
        if not label_path.is_file():
            continue
        label = housecat6d_read_label(label_path)
        label_model_list = [str(n) for n in label.get("model_list", [])]
        try:
            model_index = label_model_list.index(model_name)
        except ValueError:
            continue

        try:
            (
                scene_tensor,
                depth_tensor,
                mask_tensor,
                display_image,
                display_depth,
                _gt_mask,
                intrinsic,
            ) = load_housecat6d_scene_frame_inputs(
                scene_dir, int(frame_id), model_name, resolution, device, target_crop=False,
            )
        except FileNotFoundError as exc:
            print(f"[skip] housecat6d/{scene_name}/{model_name} frame {int(frame_id):06d}: {exc}")
            continue

        if _BENCHMARK_REQUEST is not None:
            _benchmark_inference(
                model=model,
                scene_tensor=scene_tensor,
                object_tensor=object_tensor,
                depth_tensor=depth_tensor,
                mask_tensor=mask_tensor,
                display_depth=display_depth,
                use_depth_input=use_depth_input,
                use_depth_scale=use_depth_scale,
                device=device,
                warmup=int(_BENCHMARK_REQUEST["warmup"]),
                runs=int(_BENCHMARK_REQUEST["runs"]),
                label=f"housecat6d/{scene_name}/{model_name} frame {int(frame_id):06d}",
            )
            return None

        outputs = _run_inference(model, scene_tensor, object_tensor, depth_tensor, mask_tensor, use_depth_input, device)
        pred = _extract_pred_pose(outputs, display_depth, use_depth_input, use_depth_scale)
        if pred is None:
            print(f"[skip] housecat6d/{scene_name}/{model_name} frame {int(frame_id):06d}: missing pose outputs")
            continue
        pred_rotation_cam, pred_translation_cam = pred

        gt_rotation_native = np.asarray(label["rotations"][model_index], dtype=np.float32).reshape(3, 3)
        gt_translation_cam = np.asarray(label["translations"][model_index], dtype=np.float32).reshape(3)
        gt_rotation_aligned = (gt_rotation_native @ r_align.T).astype(np.float32)
        gt_size_native = np.asarray(label["gt_scales"][model_index], dtype=np.float32).reshape(3)
        gt_size_aligned = _size_aligned(gt_size_native, r_align)

        bbox_obj = centered_axis_bbox_corners(gt_size_aligned)
        axis_length = max(float(np.linalg.norm(gt_size_aligned)) * 0.25, 1e-3)

        frame_image = draw_pred_axes_pred_gt_bbox(
            display_image, intrinsic,
            pred_rotation_cam, pred_translation_cam, bbox_obj,
            gt_rotation_aligned, gt_translation_cam, bbox_obj,
            axis_length,
        )
        _label_frame(frame_image, f"housecat6d {scene_name} {model_name} frame {int(frame_id):06d}")
        if frames_dir is not None:
            frame_jpg = frames_dir / f"{int(frame_id):06d}.jpg"
            cv2.imwrite(str(frame_jpg), cv2.cvtColor(frame_image, cv2.COLOR_RGB2BGR),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        frames.append(frame_image)

    if not frames:
        print(f"[skip] housecat6d/{scene_name}/{model_name}: no frames rendered")
        return None
    out = _save_gif(frames, output_path, fps)
    print(f"[ok] housecat6d/{scene_name}/{model_name}: {len(frames)} frames -> {out}")
    return out


# ---------------------------------------------------------------------------
# YCB-V GIF renderer
# ---------------------------------------------------------------------------


def render_ycbv_gif(
    *,
    model,
    scene_name: str,
    scene_dir: Path,
    frame_ids: Sequence[int],
    object_id: int,
    object_records: Dict[int, Dict[str, Any]],
    object_views: Sequence[int],
    align: Dict[str, Any],
    models_info: Dict[str, Any],
    resolution: Tuple[int, int],
    device: torch.device,
    output_path: Path,
    use_depth_input: bool,
    use_depth_scale: bool,
    fps: float,
    max_frames: int | None,
) -> Path | None:
    object_id = int(object_id)
    if object_id not in object_records:
        print(f"[skip] ycbv/{scene_name}/{ycbv_object_key(object_id)}: no aligned object refs")
        return None
    object_tensor, _ = load_ycbv_object_tensor(
        object_records, object_id, object_views, resolution, device
    )

    r_align = _ycbv_r_align(align, object_id)
    gt_size_native = ycbv_size_native_m(models_info, object_id)
    gt_size_aligned = _size_aligned(gt_size_native, r_align)
    bbox_obj = centered_axis_bbox_corners(gt_size_aligned)
    axis_length = max(float(np.linalg.norm(gt_size_aligned)) * 0.25, 1e-3)

    scene_gt = read_json(scene_dir / "scene_gt.json")

    iter_frames = list(frame_ids[: int(max_frames)] if max_frames is not None else frame_ids)
    frames: List[np.ndarray] = []
    for frame_id in iter_frames:
        entries = scene_gt.get(str(int(frame_id)))
        if entries is None:
            continue
        object_index = next(
            (idx for idx, e in enumerate(entries) if int(e["obj_id"]) == object_id), None
        )
        if object_index is None:
            continue
        gt_entry = entries[object_index]

        try:
            (
                scene_tensor,
                depth_tensor,
                mask_tensor,
                display_image,
                display_depth,
                _gt_mask,
                intrinsic,
            ) = load_ycbv_scene_frame_inputs(
                scene_dir, int(frame_id), int(object_index), resolution, device, target_crop=False,
            )
        except FileNotFoundError as exc:
            print(f"[skip] ycbv/{scene_name}/{ycbv_object_key(object_id)} frame {int(frame_id):06d}: {exc}")
            continue

        if _BENCHMARK_REQUEST is not None:
            _benchmark_inference(
                model=model,
                scene_tensor=scene_tensor,
                object_tensor=object_tensor,
                depth_tensor=depth_tensor,
                mask_tensor=mask_tensor,
                display_depth=display_depth,
                use_depth_input=use_depth_input,
                use_depth_scale=use_depth_scale,
                device=device,
                warmup=int(_BENCHMARK_REQUEST["warmup"]),
                runs=int(_BENCHMARK_REQUEST["runs"]),
                label=f"ycbv/{scene_name}/{ycbv_object_key(object_id)} frame {int(frame_id):06d}",
            )
            return None

        outputs = _run_inference(model, scene_tensor, object_tensor, depth_tensor, mask_tensor, use_depth_input, device)
        pred = _extract_pred_pose(outputs, display_depth, use_depth_input, use_depth_scale)
        if pred is None:
            continue
        pred_rotation_cam, pred_translation_cam = pred

        gt_rotation_native = np.asarray(gt_entry["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
        gt_translation_cam = np.asarray(gt_entry["cam_t_m2c"], dtype=np.float32).reshape(3) / 1000.0
        gt_rotation_aligned = (gt_rotation_native @ r_align.T).astype(np.float32)

        frame_image = draw_pred_axes_pred_gt_bbox(
            display_image, intrinsic,
            pred_rotation_cam, pred_translation_cam, bbox_obj,
            gt_rotation_aligned, gt_translation_cam, bbox_obj,
            axis_length,
        )
        _label_frame(
            frame_image,
            f"ycbv {scene_name} {ycbv_object_key(object_id)} frame {int(frame_id):06d}",
        )
        frames.append(frame_image)

    if not frames:
        print(f"[skip] ycbv/{scene_name}/{ycbv_object_key(object_id)}: no frames rendered")
        return None
    out = _save_gif(frames, output_path, fps)
    print(f"[ok] ycbv/{scene_name}/{ycbv_object_key(object_id)}: {len(frames)} frames -> {out}")
    return out


# ---------------------------------------------------------------------------
# Real275 GIF renderer
# ---------------------------------------------------------------------------


def render_real275_gif(
    *,
    model,
    scene_name: str,
    scene_dir: Path,
    frame_ids: Sequence[int],
    inst_id: int,
    class_id: int,
    model_name: str,
    object_records: Dict[str, Dict[str, Any]],
    object_refs_by_class: Dict[int, List[str]],
    object_views: Sequence[int],
    align: Dict[str, Any],
    obj_models_root: Path,
    gt_root: Path,
    resolution: Tuple[int, int],
    device: torch.device,
    output_path: Path,
    use_depth_input: bool,
    use_depth_scale: bool,
    fps: float,
    max_frames: int | None,
) -> Path | None:
    inst_id = int(inst_id)
    class_id = int(class_id)
    model_name = str(model_name)

    ref_name = model_name if model_name in object_records else None
    if ref_name is None:
        candidates = object_refs_by_class.get(class_id, [])
        ref_name = candidates[0] if candidates else None
    if ref_name is None:
        print(f"[skip] real275/{scene_name}/{model_name}: no aligned object refs for class {class_id}")
        return None
    object_tensor, _ = load_real275_object_tensor(
        object_records, ref_name, object_views, resolution, device
    )

    r_align = _r275_r_align(align, class_id)
    gt_size_native = real275_read_canonical_extent(obj_models_root, model_name)
    if gt_size_native is None:
        print(f"[skip] real275/{scene_name}/{model_name}: no canonical extent")
        return None
    gt_size_aligned = _size_aligned(gt_size_native, r_align)
    bbox_obj = centered_axis_bbox_corners(gt_size_aligned)
    axis_length = max(float(np.linalg.norm(gt_size_aligned)) * 0.25, 1e-3)

    iter_frames = list(frame_ids[: int(max_frames)] if max_frames is not None else frame_ids)
    frames: List[np.ndarray] = []
    for frame_id in iter_frames:
        frame_token = f"{int(frame_id):04d}"
        meta_entries = real275_read_meta(scene_dir / f"{frame_token}_meta.txt")
        meta_index = next(
            (idx for idx, e in enumerate(meta_entries) if int(e["inst_id"]) == inst_id), None
        )
        if meta_index is None:
            continue

        gt_path = gt_root / f"results_real_test_{scene_name}_{frame_token}.pkl"
        if not gt_path.is_file():
            continue
        with gt_path.open("rb") as fh:
            gt_payload = pickle.load(fh)
        gt_rts = gt_payload.get("gt_RTs")
        if gt_rts is None or meta_index >= len(gt_rts):
            continue
        gt_rt = np.asarray(gt_rts[meta_index], dtype=np.float64).reshape(4, 4)
        gt_rotation_native, gt_translation_cam, _ = real275_decompose_gt_rt(gt_rt)
        gt_rotation_aligned = (gt_rotation_native @ r_align.T).astype(np.float32)

        try:
            (
                scene_tensor,
                depth_tensor,
                mask_tensor,
                display_image,
                display_depth,
                _gt_mask,
                intrinsic,
            ) = load_real275_scene_frame_inputs(
                scene_dir, frame_token, inst_id, resolution, device, target_crop=False,
            )
        except FileNotFoundError as exc:
            print(f"[skip] real275/{scene_name}/{model_name} frame {frame_token}: {exc}")
            continue

        if _BENCHMARK_REQUEST is not None:
            _benchmark_inference(
                model=model,
                scene_tensor=scene_tensor,
                object_tensor=object_tensor,
                depth_tensor=depth_tensor,
                mask_tensor=mask_tensor,
                display_depth=display_depth,
                use_depth_input=use_depth_input,
                use_depth_scale=use_depth_scale,
                device=device,
                warmup=int(_BENCHMARK_REQUEST["warmup"]),
                runs=int(_BENCHMARK_REQUEST["runs"]),
                label=f"real275/{scene_name}/inst{inst_id:02d}/{model_name} frame {frame_token}",
            )
            return None

        outputs = _run_inference(model, scene_tensor, object_tensor, depth_tensor, mask_tensor, use_depth_input, device)
        pred = _extract_pred_pose(outputs, display_depth, use_depth_input, use_depth_scale)
        if pred is None:
            continue
        pred_rotation_cam, pred_translation_cam = pred

        frame_image = draw_pred_axes_pred_gt_bbox(
            display_image, intrinsic,
            pred_rotation_cam, pred_translation_cam, bbox_obj,
            gt_rotation_aligned, gt_translation_cam, bbox_obj,
            axis_length,
        )
        _label_frame(
            frame_image,
            f"real275 {scene_name} inst{inst_id:02d}/{model_name} frame {frame_token}",
        )
        frames.append(frame_image)

    if not frames:
        print(f"[skip] real275/{scene_name}/{model_name}: no frames rendered")
        return None
    out = _save_gif(frames, output_path, fps)
    print(f"[ok] real275/{scene_name}/inst{inst_id:02d}/{model_name}: {len(frames)} frames -> {out}")
    return out


# ---------------------------------------------------------------------------
# Dataset dispatch + work-item enumeration
# ---------------------------------------------------------------------------


def detect_dataset(scene_path: Path) -> str:
    s = str(scene_path).lower()
    if "housecat6d" in s:
        return "housecat6d"
    if "ycbv" in s:
        return "ycbv"
    if "real275" in s:
        return "real275"
    raise ValueError(f"Cannot detect dataset from path: {scene_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_PRETRAIN_MODEL)
    parser.add_argument(
        "--scene-paths",
        nargs="+",
        type=Path,
        default=None,
        help=(
            "Absolute scene directory paths. Dataset is auto-detected from the path. "
            "If not given, falls back to the built-in housecat6d test/val/scene01-02 list "
            "plus ycbv 000049/000050 and real275 scene_1/scene_2."
        ),
    )
    parser.add_argument("--housecat6d-root", type=Path, default=DEFAULT_HOUSECAT6D_ROOT)
    parser.add_argument("--housecat6d-object-image-root", type=Path, default=DEFAULT_HOUSECAT6D_OBJECT_IMAGE_ROOT)
    parser.add_argument("--ycbv-test-root", type=Path, default=DEFAULT_YCBV_TEST_ROOT)
    parser.add_argument("--ycbv-object-image-root", type=Path, default=DEFAULT_YCBV_OBJECT_IMAGE_ROOT)
    parser.add_argument("--ycbv-models-info", type=Path, default=DEFAULT_YCBV_MODELS_INFO)
    parser.add_argument("--real275-test-root", type=Path, default=DEFAULT_REAL275_TEST_ROOT)
    parser.add_argument("--real275-object-image-root", type=Path, default=DEFAULT_REAL275_OBJECT_IMAGE_ROOT)
    parser.add_argument("--real275-obj-models-root", type=Path, default=DEFAULT_REAL275_OBJ_MODELS_ROOT)
    parser.add_argument("--real275-gt-root", type=Path, default=DEFAULT_REAL275_GT_ROOT)
    parser.add_argument("--align-json", type=Path, default=DEFAULT_ALIGN_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--object", type=str, default=None, help="Restrict to a single object name/id.")
    parser.add_argument("--object-prefixes", nargs="+", default=None,
                        help="Restrict to housecat6d objects whose model_name starts with any of these "
                             "(e.g. 'box- tube- remote- cutlery-' to focus on hard categories).")
    parser.add_argument("--view-ids", nargs="+", type=int, default=None,
                        help="Override fixed_object_view_ids (e.g. 0 5 8 9). If not set, "
                             "uses the config's fixed_object_view_ids.")
    parser.add_argument("--save-frames", action="store_true",
                        help="Also dump each rendered frame as a JPG next to the GIF "
                             "(under <gif_stem>_frames/<frame_id:06d}.jpg).")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--no-depth-input", action="store_true")
    parser.add_argument("--no-depth-scale", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--benchmark", action="store_true",
                        help="Measure single-view inference time on the first valid frame, then exit.")
    parser.add_argument("--benchmark-warmup", type=int, default=5)
    parser.add_argument("--benchmark-runs", type=int, default=50)
    args = parser.parse_args()
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise SystemExit(
            f"Bad sharding: shard_index={args.shard_index}, num_shards={args.num_shards}"
        )

    cfg = load_config(Path(args.config))
    runtime = resolve_runtime_settings(cfg)
    resolution = tuple(int(v) for v in runtime["resolution"])
    if getattr(args, "view_ids", None):
        object_views = tuple(int(v) for v in args.view_ids)
        print(f"[demo_gif] view-ids override from CLI: {object_views}")
    else:
        object_views = tuple(int(v) for v in runtime["object_input_views"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    print(f"[info] device={device}, checkpoint={checkpoint_path}")
    model = build_model_from_config(cfg, checkpoint_path, device)

    align = load_align(Path(args.align_json).expanduser())

    # Build object record dicts lazily — only if the corresponding dataset is needed.
    hc_object_records: Dict[str, Dict[str, Any]] | None = None
    ycbv_object_records: Dict[int, Dict[str, Any]] | None = None
    ycbv_models_info: Dict[str, Any] | None = None
    r275_object_records: Dict[str, Dict[str, Any]] | None = None
    r275_object_refs_by_class: Dict[int, List[str]] | None = None

    # Assemble scene paths.
    if args.scene_paths:
        scene_paths = [Path(p).expanduser() for p in args.scene_paths]
    else:
        scene_paths = []
        hc_root = Path(args.housecat6d_root).expanduser()
        for name in DEFAULT_HOUSECAT6D_SCENES:
            scene_paths.append(hc_root / name)
        ycbv_root = Path(args.ycbv_test_root).expanduser()
        for name in DEFAULT_YCBV_SCENES:
            scene_paths.append(ycbv_root / name)
        r275_root = Path(args.real275_test_root).expanduser()
        for name in DEFAULT_REAL275_SCENES:
            scene_paths.append(r275_root / name)

    # Build work items.
    work_items: List[Tuple[str, Path, Path, Dict[str, Any]]] = []
    for scene_path in scene_paths:
        if not scene_path.is_dir():
            print(f"[skip] scene dir missing: {scene_path}")
            continue
        dataset_key = detect_dataset(scene_path)
        scene_name = scene_path.name

        if dataset_key == "housecat6d":
            if hc_object_records is None:
                hc_object_records = build_housecat6d_object_records(
                    Path(args.housecat6d_object_image_root).expanduser(), object_views
                )
                if not hc_object_records:
                    raise SystemExit(
                        f"No HouseCat6D aligned object refs found under {args.housecat6d_object_image_root}"
                    )
            frame_ids = enumerate_housecat6d_frames(scene_path)
            if not frame_ids:
                print(f"[skip] housecat6d/{scene_name}: no label frames")
                continue
            objects = enumerate_housecat6d_objects(scene_path, frame_ids)
            if args.object:
                objects = [item for item in objects if item[0] == args.object]
            if args.object_prefixes:
                prefixes = tuple(args.object_prefixes)
                objects = [item for item in objects if str(item[0]).startswith(prefixes)]
            if not objects:
                print(f"[skip] housecat6d/{scene_name}: no objects parsed")
                continue
            for model_name, class_id in objects:
                output_path = Path(args.output_dir) / "housecat6d" / scene_name / f"{model_name}.gif"
                work_items.append((
                    dataset_key, scene_path, output_path,
                    {
                        "scene_name": scene_name,
                        "frame_ids": frame_ids,
                        "model_name": model_name,
                        "class_id": int(class_id),
                    },
                ))

        elif dataset_key == "ycbv":
            if ycbv_object_records is None:
                ycbv_object_records = build_ycbv_object_records(
                    Path(args.ycbv_object_image_root).expanduser(), object_views
                )
                if not ycbv_object_records:
                    raise SystemExit(
                        f"No YCB-V aligned object refs found under {args.ycbv_object_image_root}"
                    )
                ycbv_models_info = read_json(Path(args.ycbv_models_info).expanduser())
            frame_ids = enumerate_ycbv_frames(scene_path)
            if not frame_ids:
                print(f"[skip] ycbv/{scene_name}: no scene_gt frames")
                continue
            objects = enumerate_ycbv_objects(scene_path)
            if args.object:
                try:
                    target_id = int(args.object)
                    objects = [o for o in objects if o == target_id]
                except ValueError:
                    objects = []
            if not objects:
                print(f"[skip] ycbv/{scene_name}: no objects parsed")
                continue
            for object_id in objects:
                output_path = Path(args.output_dir) / "ycbv" / scene_name / f"{ycbv_object_key(object_id)}.gif"
                work_items.append((
                    dataset_key, scene_path, output_path,
                    {
                        "scene_name": scene_name,
                        "frame_ids": frame_ids,
                        "object_id": int(object_id),
                    },
                ))

        elif dataset_key == "real275":
            if r275_object_records is None:
                r275_object_records, r275_object_refs_by_class = build_real275_object_records(
                    Path(args.real275_object_image_root).expanduser(), object_views
                )
                if not r275_object_records:
                    raise SystemExit(
                        f"No REAL275 aligned object refs found under {args.real275_object_image_root}"
                    )
            frame_ids = enumerate_real275_frames(scene_path)
            if not frame_ids:
                print(f"[skip] real275/{scene_name}: no color frames")
                continue
            objects = enumerate_real275_objects(scene_path, frame_ids)
            if args.object:
                objects = [o for o in objects if o[2] == args.object or str(o[0]) == args.object]
            if not objects:
                print(f"[skip] real275/{scene_name}: no instances parsed")
                continue
            for inst_id, class_id, model_name in objects:
                output_path = Path(args.output_dir) / "real275" / scene_name / f"inst{inst_id:02d}_{model_name}.gif"
                work_items.append((
                    dataset_key, scene_path, output_path,
                    {
                        "scene_name": scene_name,
                        "frame_ids": frame_ids,
                        "inst_id": int(inst_id),
                        "class_id": int(class_id),
                        "model_name": model_name,
                    },
                ))

    sharded = [item for idx, item in enumerate(work_items) if idx % args.num_shards == args.shard_index]
    print(
        f"[info] shard {args.shard_index}/{args.num_shards}: "
        f"{len(sharded)} of {len(work_items)} (dataset, scene, object) item(s)"
    )

    use_depth_input = not bool(args.no_depth_input)
    use_depth_scale = not bool(args.no_depth_scale)

    if args.benchmark:
        global _BENCHMARK_REQUEST
        _BENCHMARK_REQUEST = {"warmup": int(args.benchmark_warmup), "runs": int(args.benchmark_runs)}

    for dataset_key, scene_dir, output_path, payload in sharded:
        if args.skip_existing and output_path.is_file():
            print(f"[skip] {dataset_key}/{payload['scene_name']}: already exists at {output_path}")
            continue

        if dataset_key == "housecat6d":
            render_housecat6d_gif(
                model=model,
                scene_name=payload["scene_name"],
                scene_dir=scene_dir,
                frame_ids=payload["frame_ids"],
                model_name=payload["model_name"],
                class_id=int(payload["class_id"]),
                object_records=hc_object_records,
                object_views=object_views,
                align=align,
                resolution=resolution,
                device=device,
                output_path=output_path,
                use_depth_input=use_depth_input,
                use_depth_scale=use_depth_scale,
                fps=args.fps,
                max_frames=args.max_frames,
                save_frames=bool(args.save_frames),
            )
        elif dataset_key == "ycbv":
            render_ycbv_gif(
                model=model,
                scene_name=payload["scene_name"],
                scene_dir=scene_dir,
                frame_ids=payload["frame_ids"],
                object_id=int(payload["object_id"]),
                object_records=ycbv_object_records,
                object_views=object_views,
                align=align,
                models_info=ycbv_models_info,
                resolution=resolution,
                device=device,
                output_path=output_path,
                use_depth_input=use_depth_input,
                use_depth_scale=use_depth_scale,
                fps=args.fps,
                max_frames=args.max_frames,
            )
        elif dataset_key == "real275":
            render_real275_gif(
                model=model,
                scene_name=payload["scene_name"],
                scene_dir=scene_dir,
                frame_ids=payload["frame_ids"],
                inst_id=int(payload["inst_id"]),
                class_id=int(payload["class_id"]),
                model_name=payload["model_name"],
                object_records=r275_object_records,
                object_refs_by_class=r275_object_refs_by_class,
                object_views=object_views,
                align=align,
                obj_models_root=Path(args.real275_obj_models_root).expanduser(),
                gt_root=Path(args.real275_gt_root).expanduser(),
                resolution=resolution,
                device=device,
                output_path=output_path,
                use_depth_input=use_depth_input,
                use_depth_scale=use_depth_scale,
                fps=args.fps,
                max_frames=args.max_frames,
            )

        if _BENCHMARK_REQUEST is not None:
            break


if __name__ == "__main__":
    main()
