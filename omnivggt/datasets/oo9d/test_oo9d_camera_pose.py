#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

from omnivggt.datasets import OO9DCameraPose  # noqa: E402


def exists_text(path: str) -> str:
    return f"exists={Path(path).is_file()}"


def shape_text(value: Any) -> str:
    return str(tuple(value.shape)) if hasattr(value, "shape") else str(type(value))


def load_json(path: str) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_scene_object(sample: dict[str, Any]) -> str:
    scene_gt_path = Path(sample["scene_mask_path"]).parents[1] / "scene_gt.json"
    scene_gt = load_json(str(scene_gt_path))
    image_id = int(sample["ids"][0])
    object_id = int(sample["object_id"])
    present_ids = [int(gt.get("obj_id", -1)) for gt in scene_gt[str(image_id)]]
    return f"scene_gt_has_object={object_id in present_ids} present_ids={present_ids}"


def verify_object_refs(sample: dict[str, Any]) -> str:
    object_id = int(sample["object_id"])
    checks = []
    if sample["object_rgb_paths"]:
        scene_gt_path = Path(sample["object_rgb_paths"][0]).parents[1] / "scene_gt.json"
        if scene_gt_path.is_file():
            scene_gt = load_json(str(scene_gt_path))
            for view_id in sample["object_cam_indices"].tolist():
                ref_id = int(scene_gt.get(str(view_id), [{}])[0].get("obj_id", -1))
                checks.append(f"{view_id:06d}:{ref_id == object_id}")
    return "around_scene_gt_obj_id=" + ",".join(checks)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().float().clamp(0.0, 1.0)
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape={tuple(tensor.shape)}")
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def mask_to_pil(mask: Any) -> Image.Image:
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[0]
    return Image.fromarray((array.astype(bool) * 255).astype(np.uint8), mode="L")


def depth_to_pil(depth: Any) -> Image.Image:
    array = np.asarray(depth, dtype=np.float32)
    if array.ndim == 4:
        array = array[0, :, :, 0]
    elif array.ndim == 3:
        array = array[:, :, 0]
    valid = np.isfinite(array) & (array > 0)
    if not valid.any():
        return Image.fromarray(np.zeros(array.shape, dtype=np.uint8), mode="L")
    lo, hi = np.percentile(array[valid], [1, 99])
    if hi <= lo:
        lo = float(array[valid].min())
        hi = float(array[valid].max())
    vis = np.clip((array - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    vis[~valid] = 0.0
    return Image.fromarray((vis * 255.0).round().astype(np.uint8), mode="L")


def save_sample_visuals(sample: dict[str, Any], sample_label: str, save_dir: Path) -> None:
    sample_dir = save_dir / sample_label
    sample_dir.mkdir(parents=True, exist_ok=True)

    tensor_to_pil(sample["images"][0]).save(sample_dir / "scene_rgb_resized.png")
    depth_to_pil(sample["depth"]).save(sample_dir / "scene_depth_resized_vis.png")
    np.save(sample_dir / "scene_depth_resized_m.npy", np.asarray(sample["depth"], dtype=np.float32))
    mask_to_pil(sample["valid_mask"]).save(sample_dir / "scene_valid_mask.png")
    mask_to_pil(sample["object_masks"]).save(sample_dir / "scene_object_mask.png")

    object_dir = sample_dir / "object_refs"
    object_dir.mkdir(exist_ok=True)
    for view_id, tensor in zip(sample["object_cam_indices"].tolist(), sample["object_images"]):
        tensor_to_pil(tensor).save(object_dir / f"object_rgb_{view_id:06d}_resized.png")

    metadata = {
        "scene_source": sample.get("scene_source", ""),
        "seq_name": sample["seq_name"],
        "scene_name": sample["scene_name"],
        "image_id": int(sample["ids"][0]),
        "object_id": int(sample["object_id"]),
        "object_name": sample["object_name"],
        "scene_rgb_path": sample["scene_rgb_path"],
        "scene_depth_path": sample["scene_depth_path"],
        "scene_mask_path": sample["scene_mask_path"],
        "scene_camera_path": sample["scene_camera_path"],
        "object_cam_indices": sample["object_cam_indices"].tolist(),
        "object_rgb_paths": sample["object_rgb_paths"],
        "scene_images_shape": tuple(sample["images"].shape),
        "scene_depth_shape": tuple(np.asarray(sample["depth"]).shape),
        "object_images_shape": tuple(sample["object_images"].shape),
    }
    with (sample_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"saved_visuals_dir: {sample_dir}")


def print_sample(dataset: OO9DCameraPose, idx: int, save_dir: Path | None = None, sample_label: str | None = None) -> None:
    sample = dataset[idx]
    rec = dataset.records[idx % len(dataset.records)]
    print("-" * 120)
    print(f"idx: {idx}")
    print(f"scene_source: {sample.get('scene_source', '')}")
    print(f"seq_name: {sample['seq_name']}")
    print(f"scene_name: {sample['scene_name']}")
    print(f"scene_dir: {rec['scene_dir']}")
    print(f"image_id: {int(sample['ids'][0])}")
    print(f"object_id: {int(sample['object_id'])}")
    print(f"object_name: {sample['object_name']}")
    print(f"has_object: {bool(sample['has_object'])}")
    print(f"scene_rgb_path: {sample['scene_rgb_path']}  {exists_text(sample['scene_rgb_path'])}")
    print(f"scene_depth_path: {sample['scene_depth_path']}  {exists_text(sample['scene_depth_path'])}")
    print(f"scene_mask_path: {sample['scene_mask_path']}  {exists_text(sample['scene_mask_path'])}")
    print(f"scene_camera_path: {sample['scene_camera_path']}  {exists_text(sample['scene_camera_path'])}")
    print(f"scene_gt_path: {Path(sample['scene_mask_path']).parents[1] / 'scene_gt.json'}")
    print(f"scene_check: {verify_scene_object(sample)}")
    print(f"object_reference_scene_name: {sample['object_reference_scene_name']}")
    print(f"object_cam_indices: {sample['object_cam_indices'].tolist()}")
    print("object_rgb_paths:")
    for view_id, path in zip(sample["object_cam_indices"].tolist(), sample["object_rgb_paths"]):
        print(f"  {view_id:06d}: {path}  {exists_text(path)}")
    print(f"object_ref_check: {verify_object_refs(sample)}")
    print(f"scene_images_shape: {shape_text(sample['images'])}")
    print(f"scene_depth_shape: {shape_text(sample['depth'])}")
    print(f"scene_valid_mask_shape: {shape_text(sample['valid_mask'])}")
    print(f"scene_object_masks_shape: {shape_text(sample['object_masks'])}")
    print(f"object_images_shape: {shape_text(sample['object_images'])}")
    print(f"object_true_shape: {sample['object_true_shape']}")
    print(f"object_rotation_shape: {shape_text(sample['object_rotation'])}")
    print(f"object_translation: {sample['object_translation']}")
    print(f"object_size: {sample['object_size']}")
    if save_dir is not None:
        save_sample_visuals(sample, sample_label or f"sample_{idx:04d}", save_dir)


def select_indices(dataset: OO9DCameraPose, samples_per_source: int, sources: set[str]) -> list[int]:
    counts = {source: 0 for source in sources}
    selected = []
    for idx, rec in enumerate(dataset.records):
        source = rec.get("scene_source", "")
        if source not in sources or counts[source] >= samples_per_source:
            continue
        selected.append(idx)
        counts[source] += 1
        if all(count >= samples_per_source for count in counts.values()):
            break
    return selected


def build_dataset(
    args: argparse.Namespace,
    dset: str,
    max_records: int | None = None,
    only_scene_name: str = "",
    only_object_id: int | None = None,
) -> OO9DCameraPose:
    return OO9DCameraPose(
        dataset_location=args.dataset_location,
        dset=dset,
        split_root=args.split_root,
        multi_split_json=args.multi_split_json if dset == "train" else args.val_split_json,
        single_split_json=args.single_split_json,
        object_image_root=args.object_image_root,
        resolution=(args.resolution_width, args.resolution_height),
        max_records=max_records,
        only_scene_name=only_scene_name,
        only_object_id=only_object_id,
        verify_files=not args.no_verify_files,
        seed=args.seed,
    )


def has_fixed_object_refs(args: argparse.Namespace, object_id: int) -> bool:
    object_dir = Path(args.object_image_root) / f"obj_{object_id:06d}" / "rgb"
    return all((object_dir / f"{view_id:06d}.png").is_file() for view_id in OO9DCameraPose.DEFAULT_OBJECT_VIEW_IDS)


def find_quick_candidates(args: argparse.Namespace, source: str, count: int) -> list[tuple[str, int]]:
    candidates = []
    if source == "multi":
        split_json = Path(args.multi_split_json) if args.multi_split_json else Path(args.split_root) / "multi" / "train.json"
        payload = load_json(str(split_json))
        for item in payload.get("scenes", []):
            for object_id in item.get("eligible_object_ids", item.get("object_ids", [])):
                object_id = int(object_id)
                if has_fixed_object_refs(args, object_id):
                    candidates.append((str(item["scene_name"]), object_id))
                    break
            if len(candidates) >= count:
                break
    elif source == "single":
        split_json = Path(args.single_split_json) if args.single_split_json else Path(args.split_root) / "single" / "train.json"
        payload = load_json(str(split_json))
        for item in payload.get("scenes", []):
            object_id = int(item["object_id"])
            if has_fixed_object_refs(args, object_id):
                candidates.append((str(item["scene_name"]), object_id))
            if len(candidates) >= count:
                break
    else:
        raise ValueError(f"Unknown source: {source}")
    return candidates


def print_quick_train_samples(args: argparse.Namespace, sources: set[str]) -> None:
    print("=" * 120)
    print("TRAIN DATASET QUICK SAMPLES")
    print(f"multi_split_json: {args.multi_split_json or Path(args.split_root) / 'multi' / 'train.json'}")
    print(f"single_split_json: {args.single_split_json or Path(args.split_root) / 'single' / 'train.json'}")
    print(f"sources: {sorted(sources)}")
    for source in ["multi", "single"]:
        if source not in sources:
            continue
        candidates = find_quick_candidates(args, source, args.samples_per_source)
        print(f"{source}_candidates: {candidates}")
        for scene_name, object_id in candidates:
            dataset = build_dataset(
                args,
                "train",
                max_records=1,
                only_scene_name=scene_name,
                only_object_id=object_id,
            )
            if not dataset.records:
                print(f"WARNING: no records for source={source} scene={scene_name} object_id={object_id}")
                continue
            label = f"train_{source}_{scene_name}_obj_{object_id:06d}"
            print_sample(dataset, 0, args.save_dir if args.save_visuals else None, label)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print full OO9D dataloader sample paths.")
    parser.add_argument("--dataset-location", default=OO9DCameraPose.DEFAULT_DATA_ROOT)
    parser.add_argument("--split-root", default=OO9DCameraPose.DEFAULT_SPLIT_ROOT)
    parser.add_argument("--multi-split-json", default=None)
    parser.add_argument("--single-split-json", default=None)
    parser.add_argument("--val-split-json", default=None)
    parser.add_argument("--object-image-root", default=OO9DCameraPose.DEFAULT_OBJECT_IMAGE_ROOT)
    parser.add_argument("--resolution", type=int, default=None, help="Square resolution shorthand. Overrides width/height.")
    parser.add_argument("--resolution-width", type=int, default=518)
    parser.add_argument("--resolution-height", type=int, default=518)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-source", type=int, default=2)
    parser.add_argument("--include-val", action="store_true")
    parser.add_argument("--save-visuals", action="store_true")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "debug_visuals",
        help="Directory for resized RGB/depth/mask visualizations.",
    )
    parser.add_argument("--full-scan", action="store_true")
    parser.add_argument("--only-multi", action="store_true")
    parser.add_argument("--only-single", action="store_true")
    parser.add_argument("--no-verify-files", action="store_true")
    args = parser.parse_args()

    if args.only_multi and args.only_single:
        raise ValueError("--only-multi and --only-single cannot both be set")
    if args.resolution is not None:
        args.resolution_width = int(args.resolution)
        args.resolution_height = int(args.resolution)

    sources = {"multi", "single"}
    if args.only_multi:
        sources = {"multi"}
    if args.only_single:
        sources = {"single"}

    if args.full_scan:
        train_dataset = build_dataset(args, "train")
        print("=" * 120)
        print("TRAIN DATASET")
        print(f"multi_split_json: {train_dataset.split_json}")
        print(f"single_split_json: {train_dataset.single_split_json}")
        print(f"dataset_length: {len(train_dataset)}")
        print(f"object_reference_records: {len(train_dataset.single_records_by_object_id)}")
        train_indices = select_indices(train_dataset, args.samples_per_source, sources)
        print(f"selected_indices: {train_indices}")
        for idx in train_indices:
            label = f"train_full_{idx:06d}_{train_dataset.records[idx].get('scene_source', 'scene')}"
            print_sample(train_dataset, idx, args.save_dir if args.save_visuals else None, label)
    else:
        print_quick_train_samples(args, sources)

    if args.include_val:
        val_dataset = build_dataset(args, "val", max_records=args.samples_per_source)
        print("=" * 120)
        print("VAL DATASET")
        print(f"val_split_json: {val_dataset.split_json}")
        print(f"dataset_length: {len(val_dataset)}")
        print(f"object_reference_records: {len(val_dataset.single_records_by_object_id)}")
        for idx in range(min(args.samples_per_source, len(val_dataset))):
            label = f"val_{idx:04d}_{val_dataset.records[idx].get('scene_source', 'scene')}"
            print_sample(val_dataset, idx, args.save_dir if args.save_visuals else None, label)


if __name__ == "__main__":
    main()
