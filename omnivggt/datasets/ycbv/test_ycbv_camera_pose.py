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


DEFAULT_DATA_ROOT = "/mnt/train-data-4-hdd/yian/freepose/datasets_real/ycbv"
DEFAULT_OBJECT_IMAGE_ROOT = "/mnt/train-data-4-hdd/yian/freepose/datasets_real/ycbv/ycbv_aligned_object_refs"
DEFAULT_ALIGN_JSON = "/mnt/train-data-4-hdd/yian/freepose/omni-object_clone/dataset_align.json"
DEFAULT_SAVE_DIR = "/mnt/train-data-4-hdd/yian/freepose/omni-object_clone/omnivggt/datasets/ycbv/verification"
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


def tensor_image_to_pil(tensor) -> Image.Image:
    tensor = tensor.detach().cpu().float().clamp(0.0, 1.0)
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


def draw_line_if_visible(draw, pixels, valid, i, j, color, width) -> None:
    if valid[i] and valid[j] and np.isfinite(pixels[[i, j]]).all():
        draw.line([tuple(pixels[i]), tuple(pixels[j])], fill=color, width=width)


def draw_axis_label_if_visible(draw, pixels, valid, axis_index, label, color) -> None:
    if not valid[axis_index] or not np.isfinite(pixels[axis_index]).all():
        return
    x, y = pixels[axis_index]
    x += 6
    y -= 16
    draw.text((x - 1, y - 1), label, fill=(255, 255, 255))
    draw.text((x + 1, y - 1), label, fill=(255, 255, 255))
    draw.text((x - 1, y + 1), label, fill=(255, 255, 255))
    draw.text((x + 1, y + 1), label, fill=(255, 255, 255))
    draw.text((x, y), label, fill=color)


def save_pose_overlay(sample: dict[str, Any], save_dir: Path, idx: int, axis_length_scale: float) -> Path:
    image = tensor_image_to_pil(sample["images"][0])
    draw = ImageDraw.Draw(image)
    intrinsic = np.asarray(sample["intrinsic"][0], dtype=np.float32)
    rotation = np.asarray(sample["object_rotation"], dtype=np.float32).reshape(3, 3)
    translation = np.asarray(sample["object_translation_metric"], dtype=np.float32).reshape(3)
    object_size = np.asarray(sample["object_size"], dtype=np.float32).reshape(3)

    corners_obj = object_bbox_corners(object_size)
    corners_cam = (rotation @ corners_obj.T).T + translation[None]
    corner_pixels, corner_valid = project_points(corners_cam, intrinsic)
    for i, j in BBOX_EDGES:
        draw_line_if_visible(draw, corner_pixels, corner_valid, i, j, (255, 220, 0), 2)

    axis_length = max(float(np.max(object_size)) * axis_length_scale, 1e-4)
    axes_obj = np.asarray(
        [[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]],
        dtype=np.float32,
    )
    axes_cam = (rotation @ axes_obj.T).T + translation[None]
    axis_pixels, axis_valid = project_points(axes_cam, intrinsic)
    draw_line_if_visible(draw, axis_pixels, axis_valid, 0, 1, (255, 0, 0), 4)
    draw_line_if_visible(draw, axis_pixels, axis_valid, 0, 2, (0, 220, 0), 4)
    draw_line_if_visible(draw, axis_pixels, axis_valid, 0, 3, (0, 90, 255), 4)
    draw_axis_label_if_visible(draw, axis_pixels, axis_valid, 1, "X", (255, 0, 0))
    draw_axis_label_if_visible(draw, axis_pixels, axis_valid, 2, "Y", (0, 160, 0))
    draw_axis_label_if_visible(draw, axis_pixels, axis_valid, 3, "Z", (0, 90, 255))
    if axis_valid[0]:
        x, y = axis_pixels[0]
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 255, 255), outline=(0, 0, 0))

    ref_images = [
        tensor_image_to_pil(sample["object_images"][i]).resize((128, 128))
        for i in range(sample["object_images"].shape[0])
    ]
    canvas = Image.new("RGB", (image.width, image.height + 128), (255, 255, 255))
    canvas.paste(image, (0, 0))
    for i, ref in enumerate(ref_images):
        canvas.paste(ref, (i * 128, image.height))

    save_dir.mkdir(parents=True, exist_ok=True)
    object_name = str(sample["object_name"])
    scene_source = str(sample["scene_source"])
    scene_name = str(sample["scene_name"])
    image_id = int(sample["ids"][0])
    overlay_path = save_dir / f"{idx:06d}_{scene_source}_{scene_name}_{object_name}_{image_id:06d}_axes.png"
    canvas.save(overlay_path)

    with overlay_path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "idx": idx,
                "scene_source": scene_source,
                "scene_name": scene_name,
                "image_id": image_id,
                "object_name": object_name,
                "object_id": int(sample["object_id"]),
                "category": sample["category"],
                "scene_rgb_path": sample["scene_rgb_path"],
                "scene_gt_path": sample["scene_gt_path"],
                "scene_mask_path": sample["scene_mask_path"],
                "object_rgb_paths": list(sample["object_rgb_paths"]),
                "intrinsic": intrinsic.tolist(),
                "object_rotation_aligned_to_cam": rotation.tolist(),
                "object_rotation_native_to_cam": np.asarray(sample["object_rotation_native"]).tolist(),
                "R_align_ycbv_to_ov9d": np.asarray(sample["R_align_ycbv_to_ov9d"]).tolist(),
                "object_translation_metric": translation.tolist(),
                "object_size_aligned": object_size.tolist(),
                "object_size_native": np.asarray(sample["object_size_native"]).tolist(),
                "bbox_corner_pixels": corner_pixels.tolist(),
                "bbox_corner_valid": corner_valid.tolist(),
                "axis_pixels": axis_pixels.tolist(),
                "axis_valid": axis_valid.tolist(),
            },
            handle,
            indent=2,
        )
    return overlay_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify YCB-V dataset samples and draw OV9D-aligned axes.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-root", default=None)
    parser.add_argument("--object-image-root", default=DEFAULT_OBJECT_IMAGE_ROOT)
    parser.add_argument("--align-json", default=DEFAULT_ALIGN_JSON)
    parser.add_argument("--save-dir", type=Path, default=Path(DEFAULT_SAVE_DIR))
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, nargs=2, default=(518, 518), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--dset", default="test")
    parser.add_argument("--only-scene-name", default="")
    parser.add_argument("--only-object-id", type=int, default=None)
    parser.add_argument("--only-category", default="")
    parser.add_argument("--axis-length-scale", type=float, default=0.65)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--no-save-overlays", action="store_true")
    return parser


def main() -> None:
    from omnivggt.datasets.ycbv import YCBVCameraPose

    args = build_argparser().parse_args()
    dataset = YCBVCameraPose(
        dataset_location=args.dataset_root,
        split_root=args.split_root,
        object_image_root=args.object_image_root,
        align_json=args.align_json,
        dset=args.dset,
        resolution=tuple(args.resolution),
        seed=args.seed,
        only_scene_name=args.only_scene_name,
        only_object_id=args.only_object_id,
        only_category=args.only_category,
        max_records=args.max_records,
    )

    print(f"dataset_root: {dataset.dataset_location}")
    print(f"split_root: {dataset.split_root}")
    print(f"object_image_root: {dataset.object_image_root}")
    print(f"align_json: {dataset.align_json}")
    print(f"num_records: {len(dataset)}")
    print(f"num_objects: {len(dataset.object_records_by_id)}")
    print(f"fixed_object_view_ids: {dataset.fixed_object_view_ids}")

    end_idx = min(args.start_idx + args.num_samples, len(dataset))
    for idx in range(args.start_idx, end_idx):
        sample = dataset[idx]
        print("-" * 120)
        print(f"idx: {idx}")
        print(f"scene_source: {sample['scene_source']}")
        print(f"scene_name: {sample['scene_name']}")
        print(f"image_id: {int(sample['ids'][0])}")
        print(f"object_name: {sample['object_name']}")
        print(f"object_id/category: {int(sample['object_id'])}/{sample['category']}")
        print(f"scene_rgb_path: {sample['scene_rgb_path']}")
        print(f"scene_depth_path: {sample['scene_depth_path']}")
        print(f"scene_gt_path: {sample['scene_gt_path']}")
        print(f"scene_mask_path: {sample['scene_mask_path'] or '<missing; zero mask used>'}")
        print(f"object_cam_indices: {[int(x) for x in sample['object_cam_indices'].tolist()]}")
        print("object_rgb_paths:")
        for path in sample["object_rgb_paths"]:
            print(f"  {path}")
        print(
            "shapes:",
            f"images={tuple(sample['images'].shape)}",
            f"object_images={tuple(sample['object_images'].shape)}",
            f"depth={tuple(sample['depth'].shape)}",
            f"object_masks={tuple(sample['object_masks'].shape)}",
        )
        print(f"object_translation_metric: {np.asarray(sample['object_translation_metric']).tolist()}")
        print(f"object_translation_normalized: {np.asarray(sample['object_translation_normalized']).tolist()}")
        print(f"object_size_aligned: {np.asarray(sample['object_size']).tolist()}")
        print(f"object_size_native: {np.asarray(sample['object_size_native']).tolist()}")
        print(f"R_align_ycbv_to_ov9d: {np.asarray(sample['R_align_ycbv_to_ov9d']).tolist()}")
        if not args.no_save_overlays:
            overlay_path = save_pose_overlay(sample, args.save_dir, idx, args.axis_length_scale)
            print(f"pose_overlay_path: {overlay_path}")


if __name__ == "__main__":
    main()
