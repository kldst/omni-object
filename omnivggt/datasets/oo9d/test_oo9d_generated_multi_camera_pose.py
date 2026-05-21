#!/usr/bin/env python3
import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.INTER_NEAREST = 0
    cv2_stub.INTER_LINEAR = 1
    cv2_stub.INTER_CUBIC = 2

    def _resize(image: Any, dsize: tuple[int, int], fx: float = 0.0, fy: float = 0.0, interpolation: int = 0) -> Any:
        resampling = getattr(Image, "Resampling", Image)
        mode = resampling.NEAREST if interpolation == cv2_stub.INTER_NEAREST else resampling.BICUBIC
        return np.asarray(Image.fromarray(np.asarray(image)).resize(tuple(dsize), mode))

    cv2_stub.resize = _resize
    sys.modules["cv2"] = cv2_stub

DEFAULT_DATA_ROOT = "/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d"
DEFAULT_GENERATED_MULTI_ROOT = (
    "/mnt/train-data-4-hdd/yian/freepose/ov9d/render_script/ov9d_2000_scenes_3modes_4views_v2"
)
DEFAULT_SINGLE_SPLIT_JSON = (
    "/mnt/train-data-4-hdd/yian/freepose/omni-object_clone/"
    "splits_ov9d_unseen_category_generalization/single/train.json"
)
DEFAULT_SINGLE_ROOT = "/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d/oo3d9dsingle"
DEFAULT_OBJECT_IMAGE_ROOT = "/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d_around_image"
DEFAULT_OBJECT_VIEW_IDS = (1, 5, 10, 15)
BBOX_EDGES = (
    (0, 1),
    (1, 3),
    (3, 2),
    (2, 0),
    (4, 5),
    (5, 7),
    (7, 6),
    (6, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def assert_file(path: str | Path, label: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise AssertionError(f"{label} does not exist: {path}")
    return path


def assert_under(path: str | Path, root: str | Path, label: str) -> None:
    path = Path(path).resolve()
    root = Path(root).resolve()
    if root not in path.parents and path != root:
        raise AssertionError(f"{label} is not under {root}: {path}")


def sample_image_to_pil(sample: dict[str, Any]) -> Image.Image:
    tensor = sample["images"][0].detach().cpu().float().clamp(0.0, 1.0)
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def project_points(points_cam: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = points_cam[:, 2]
    valid = np.isfinite(points_cam).all(axis=1) & (z > 1e-6)
    pixels = np.full((points_cam.shape[0], 2), np.nan, dtype=np.float32)
    projected = (intrinsic @ points_cam[valid].T).T
    pixels[valid] = projected[:, :2] / projected[:, 2:3]
    return pixels, valid


def object_bbox_corners(object_size: np.ndarray) -> np.ndarray:
    sx, sy, sz = (np.asarray(object_size, dtype=np.float32) / 2.0).tolist()
    return np.asarray(
        [
            [-sx, -sy, -sz],
            [sx, -sy, -sz],
            [-sx, sy, -sz],
            [sx, sy, -sz],
            [-sx, -sy, sz],
            [sx, -sy, sz],
            [-sx, sy, sz],
            [sx, sy, sz],
        ],
        dtype=np.float32,
    )


def draw_line_if_visible(
    draw: ImageDraw.ImageDraw,
    pixels: np.ndarray,
    valid: np.ndarray,
    i: int,
    j: int,
    color: tuple[int, int, int],
    width: int,
) -> None:
    if valid[i] and valid[j] and np.isfinite(pixels[[i, j]]).all():
        draw.line([tuple(pixels[i]), tuple(pixels[j])], fill=color, width=width)


def save_pose_overlay(sample: dict[str, Any], save_dir: Path, idx: int, axis_length_scale: float) -> Path:
    image = sample_image_to_pil(sample)
    draw = ImageDraw.Draw(image)
    intrinsic = np.asarray(sample["intrinsic"][0], dtype=np.float32)
    rotation = np.asarray(sample["object_rotation"], dtype=np.float32).reshape(3, 3)
    translation = np.asarray(sample.get("object_translation_metric", sample["object_translation"]), dtype=np.float32).reshape(3)
    object_size = np.asarray(sample["object_size"], dtype=np.float32).reshape(3)

    corners_obj = object_bbox_corners(object_size)
    corners_cam = (rotation @ corners_obj.T).T + translation[None]
    corner_pixels, corner_valid = project_points(corners_cam, intrinsic)
    for i, j in BBOX_EDGES:
        draw_line_if_visible(draw, corner_pixels, corner_valid, i, j, (255, 220, 0), 2)

    axis_length = max(float(np.max(object_size)) * axis_length_scale, 1e-4)
    axes_obj = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, axis_length],
        ],
        dtype=np.float32,
    )
    axes_cam = (rotation @ axes_obj.T).T + translation[None]
    axis_pixels, axis_valid = project_points(axes_cam, intrinsic)
    draw_line_if_visible(draw, axis_pixels, axis_valid, 0, 1, (255, 0, 0), 4)
    draw_line_if_visible(draw, axis_pixels, axis_valid, 0, 2, (0, 220, 0), 4)
    draw_line_if_visible(draw, axis_pixels, axis_valid, 0, 3, (0, 90, 255), 4)

    if axis_valid[0]:
        x, y = axis_pixels[0]
        r = 4
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255), outline=(0, 0, 0))

    save_dir.mkdir(parents=True, exist_ok=True)
    scene_name = str(sample["scene_name"]).replace("/", "_")
    source = str(sample["scene_source"])
    object_id = int(sample["object_id"])
    image_id = int(sample["ids"][0])
    overlay_path = save_dir / f"{idx:06d}_{source}_{scene_name}_obj_{object_id:06d}_view_{image_id:06d}_pose_overlay.png"
    image.save(overlay_path)

    metadata_path = overlay_path.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "idx": idx,
                "scene_source": source,
                "scene_name": sample["scene_name"],
                "image_id": image_id,
                "object_id": object_id,
                "scene_rgb_path": sample["scene_rgb_path"],
                "intrinsic": intrinsic.tolist(),
                "object_rotation": rotation.tolist(),
                "object_translation": translation.tolist(),
                "object_size": object_size.tolist(),
                "bbox_corner_pixels": corner_pixels.tolist(),
                "bbox_corner_valid": corner_valid.tolist(),
                "axis_pixels": axis_pixels.tolist(),
                "axis_valid": axis_valid.tolist(),
            },
            handle,
            indent=2,
        )
    return overlay_path


def verify_sample(dataset: Any, idx: int, args: argparse.Namespace) -> None:
    sample = dataset[idx]
    record = dataset.records[idx % len(dataset.records)]
    object_id = int(sample["object_id"])
    image_id = int(sample["ids"][0])

    scene_dir = Path(record["scene_dir"])
    scene_gt_path = assert_file(scene_dir / "scene_gt.json", "scene_gt_path")
    assert_file(sample["scene_rgb_path"], "scene_rgb_path")
    assert_file(sample["scene_depth_path"], "scene_depth_path")
    assert_file(sample["scene_camera_path"], "scene_camera_path")
    expected_scene_root = dataset.single_root if sample["scene_source"] == "single" else dataset.generated_multi_root
    assert_under(sample["scene_rgb_path"], expected_scene_root, "scene_rgb_path")
    assert_under(sample["scene_depth_path"], expected_scene_root, "scene_depth_path")

    scene_gt = load_json(scene_gt_path)
    present_ids = [int(gt.get("obj_id", -1)) for gt in scene_gt[str(image_id)]]
    if object_id not in present_ids:
        raise AssertionError(
            f"object_id={object_id} not present in {scene_gt_path} frame={image_id}; present_ids={present_ids}"
        )

    object_rgb_paths = list(sample["object_rgb_paths"])
    object_cam_indices = [int(x) for x in sample["object_cam_indices"].tolist()]
    if len(object_rgb_paths) != len(object_cam_indices):
        raise AssertionError("object_rgb_paths and object_cam_indices length mismatch")
    for view_id, path in zip(object_cam_indices, object_rgb_paths):
        expected = dataset.object_image_root / f"obj_{object_id:06d}" / "rgb" / f"{view_id:06d}.png"
        if Path(path) != expected:
            raise AssertionError(f"unexpected object ref path for view={view_id}: got={path}, expected={expected}")
        assert_file(path, f"object_rgb_path[{view_id}]")

    print("-" * 120)
    print(f"idx: {idx}")
    print(f"scene_source: {sample['scene_source']}")
    print(f"scene_name: {sample['scene_name']}")
    print(f"scene_dir: {scene_dir}")
    print(f"image_id: {image_id}")
    print(f"object_id: {object_id}")
    print(f"present_ids_in_scene_gt: {present_ids}")
    print(f"scene_rgb_path: {sample['scene_rgb_path']}")
    print(f"scene_depth_path: {sample['scene_depth_path']}")
    print(f"scene_camera_path: {sample['scene_camera_path']}")
    print(f"scene_mask_path: {sample['scene_mask_path'] or '<missing; zero mask used>'}")
    print(f"object_reference_scene_name: {sample['object_reference_scene_name']}")
    print(f"object_cam_indices: {object_cam_indices}")
    print("object_rgb_paths:")
    for path in object_rgb_paths:
        print(f"  {path}")
    print(f"images_shape: {tuple(sample['images'].shape)}")
    print(f"object_images_shape: {tuple(sample['object_images'].shape)}")
    if "depth_mean_scale" in sample:
        metric_translation = np.asarray(sample["object_translation_metric"], dtype=np.float32)
        normalized_translation = np.asarray(sample["object_translation_normalized"], dtype=np.float32)
        restored_translation = normalized_translation * np.asarray(sample["depth_mean_scale"], dtype=np.float32)
        print(f"depth_mean_scale: {float(sample['depth_mean_scale']):.6f}")
        print(f"object_translation_metric: {metric_translation.tolist()}")
        print(f"object_translation_normalized: {normalized_translation.tolist()}")
        print(f"object_translation_restored: {restored_translation.tolist()}")
        print(f"object_translation_restore_abs_err_max: {float(np.max(np.abs(restored_translation - metric_translation))):.9f}")
    if not args.no_save_overlays:
        overlay_path = save_pose_overlay(sample, args.save_dir, idx, args.axis_length_scale)
        print(f"pose_overlay_path: {overlay_path}")


def verify_paths_without_torch(args: argparse.Namespace) -> None:
    generated_multi_root = Path(args.generated_multi_root)
    single_root = Path(args.single_root)
    object_image_root = Path(args.object_image_root)
    allowed_object_ids = None
    if not args.no_single_filter:
        payload = load_json(Path(args.single_split_json))
        allowed_object_ids = {int(item["object_id"]) for item in payload.get("scenes", [])}

    print("WARNING: torch is unavailable in this Python environment; running metadata-only path checks.")
    print(f"generated_multi_root: {generated_multi_root}")
    print(f"single_root: {single_root}")
    print(f"single_split_json: {args.single_split_json}")
    print(f"object_image_root: {object_image_root}")

    checked_counts = {"generated_multi": 0}
    if not args.no_single_targets:
        checked_counts["single"] = 0
    scene_dirs = [(scene_dir, "generated_multi") for scene_dir in sorted(generated_multi_root.glob("scene_*"))]
    if not args.no_single_targets:
        payload = load_json(Path(args.single_split_json))
        scene_dirs.extend((single_root / str(item["scene_name"]), "single") for item in payload.get("scenes", []))

    for scene_dir, scene_source in scene_dirs:
        scene_gt_path = scene_dir / "scene_gt.json"
        scene_camera_path = scene_dir / "scene_camera.json"
        if not scene_gt_path.is_file() or not scene_camera_path.is_file():
            continue
        scene_gt = load_json(scene_gt_path)
        for image_id_str, gts in sorted(scene_gt.items(), key=lambda item: int(item[0])):
            image_id = int(image_id_str)
            if checked_counts.get(scene_source, 0) >= args.samples:
                continue
            for gt in gts:
                object_id = int(gt.get("obj_id", -1))
                if object_id < 0:
                    continue
                if allowed_object_ids is not None and object_id not in allowed_object_ids:
                    continue
                scene_rgb_path = scene_dir / "rgb" / f"{image_id:06d}.png"
                scene_depth_path = scene_dir / "depth" / f"{image_id:06d}.png"
                object_paths = [
                    object_image_root / f"obj_{object_id:06d}" / "rgb" / f"{view_id:06d}.png"
                    for view_id in DEFAULT_OBJECT_VIEW_IDS
                ]
                assert_file(scene_rgb_path, "scene_rgb_path")
                assert_file(scene_depth_path, "scene_depth_path")
                assert_file(scene_camera_path, "scene_camera_path")
                for path in object_paths:
                    assert_file(path, "object_rgb_path")

                print("-" * 120)
                print(f"scene_source: {scene_source}")
                print(f"scene_name: {scene_dir.name}")
                print(f"image_id: {image_id}")
                print(f"object_id: {object_id}")
                print(f"scene_rgb_path: {scene_rgb_path}")
                print(f"scene_depth_path: {scene_depth_path}")
                print(f"scene_camera_path: {scene_camera_path}")
                print("object_rgb_paths:")
                for path in object_paths:
                    print(f"  {path}")
                checked_counts[scene_source] += 1
                if all(count >= args.samples for count in checked_counts.values()):
                    print(f"metadata_only_checked: {checked_counts}")
                    return
    raise RuntimeError("No generated multi samples with valid around-image references were found.")


def select_indices_per_source(dataset: Any, samples_per_source: int) -> list[int]:
    wanted_sources = ["generated_multi", "single"]
    counts = {source: 0 for source in wanted_sources}
    selected = []
    for idx, record in enumerate(dataset.records):
        source = record.get("scene_source", "")
        if source not in counts or counts[source] >= samples_per_source:
            continue
        selected.append(idx)
        counts[source] += 1
        if all(count >= samples_per_source for count in counts.values()):
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify OO9D generated multi dataset paths.")
    parser.add_argument("--dataset-location", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--generated-multi-root", default=DEFAULT_GENERATED_MULTI_ROOT)
    parser.add_argument("--single-root", default=DEFAULT_SINGLE_ROOT)
    parser.add_argument("--single-split-json", default=DEFAULT_SINGLE_SPLIT_JSON)
    parser.add_argument("--object-image-root", default=DEFAULT_OBJECT_IMAGE_ROOT)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--resolution-width", type=int, default=None)
    parser.add_argument("--resolution-height", type=int, default=None)
    parser.add_argument("--samples", type=int, default=2, help="Samples to verify per source.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=Path, default=Path(__file__).resolve().parent / "debug_pose_overlays")
    parser.add_argument("--axis-length-scale", type=float, default=0.75)
    parser.add_argument("--no-save-overlays", action="store_true")
    parser.add_argument("--no-single-filter", action="store_true")
    parser.add_argument("--no-single-targets", action="store_true")
    parser.add_argument("--no-verify-files", action="store_true")
    args = parser.parse_args()

    try:
        from omnivggt.datasets import OO9DGeneratedMultiCameraPose
    except ModuleNotFoundError:
        verify_paths_without_torch(args)
        return

    resolution_width = int(args.resolution_width or args.resolution)
    resolution_height = int(args.resolution_height or args.resolution)

    dataset = OO9DGeneratedMultiCameraPose(
        dataset_location=args.dataset_location,
        generated_multi_root=args.generated_multi_root,
        single_root=args.single_root,
        single_split_json=args.single_split_json,
        object_image_root=args.object_image_root,
        filter_single_train_objects=not args.no_single_filter,
        include_single_targets=not args.no_single_targets,
        verify_files=not args.no_verify_files,
        resolution=(resolution_width, resolution_height),
        seed=args.seed,
    )
    print(f"dataset_length: {len(dataset)}")
    print(f"generated_multi_root: {dataset.generated_multi_root}")
    print(f"single_root: {dataset.single_root}")
    print(f"single_split_json: {dataset.single_split_json}")
    print(f"object_image_root: {dataset.object_image_root}")
    print(f"resolution: {(resolution_width, resolution_height)}")
    if not args.no_save_overlays:
        print(f"pose_overlay_dir: {args.save_dir}")
    print(f"reference_object_count: {len(dataset.single_records_by_object_id)}")

    indices = select_indices_per_source(dataset, args.samples)
    print(f"selected_indices: {indices}")
    for idx in indices:
        verify_sample(dataset, idx, args)


if __name__ == "__main__":
    main()
