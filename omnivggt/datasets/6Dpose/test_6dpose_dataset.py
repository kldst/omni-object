import argparse
import sys
from pathlib import Path


def build_argparser():
    parser = argparse.ArgumentParser(description="Inspect SixDPose dataset samples and print resolved file paths.")
    parser.add_argument(
        "--dataset-root",
        default="/mnt/train-data-4-hdd/yian/6dpose_obj/0405_fixedCam_diffpose_1k",
        help="Root directory containing out_image/out_depth/out_cam_param/out_pose.",
    )
    parser.add_argument(
        "--object-root",
        default=None,
        help="Optional object_space_rgb root. If omitted, dataset fallback logic is used.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of dataset samples to inspect.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic sampling seed passed into the dataset.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=(518, 518),
        metavar=("WIDTH", "HEIGHT"),
        help="Resolution passed to the dataset.",
    )
    parser.add_argument(
        "--dset",
        default="train",
        help="Dataset split label. Use 'test' or 'val' for deterministic non-random view selection.",
    )
    parser.add_argument(
        "--only-run-name",
        default="",
        help="Optional run name filter, e.g. run_0001.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero status if any resolved path is missing.",
    )
    return parser


def assert_exists(path_str):
    path = Path(path_str)
    exists = path.is_file()
    status = "OK" if exists else "MISSING"
    print(f"  [{status}] {path}")
    return exists


def main():
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))

    from omnivggt.datasets import SixDPose

    args = build_argparser().parse_args()

    dataset = SixDPose(
        dataset_location=args.dataset_root,
        OBJECT_INPUT_ROOT=args.object_root,
        dset=args.dset,
        resolution=tuple(args.resolution),
        seed=args.seed,
        only_run_name=args.only_run_name,
    )

    print(f"dataset_root: {args.dataset_root}")
    print(f"object_root: {dataset.object_root}")
    print(f"num_records: {len(dataset)}")
    print()

    missing_paths = []
    max_samples = min(args.num_samples, len(dataset))
    for sample_idx in range(max_samples):
        record = dataset.records[sample_idx]
        sample = dataset[sample_idx]

        print(f"[sample {sample_idx}] run={record['run_name']} object={record['object_name']}")
        print(f"scene_view_indices: {sample['camera_indices'].tolist()}")
        print(f"object_view_indices: {sample['object_cam_indices'].tolist()}")
        print("object_paths:")
        for cam_idx in sample["object_cam_indices"].tolist():
            path = dataset._resolve_object_image_path(record["object_name"], cam_idx)
            if not assert_exists(path):
                missing_paths.append(path)
        print("scene_paths:")
        for cam_idx in sample["camera_indices"].tolist():
            path = dataset._resolve_scene_image_path(record["run_name"], cam_idx)
            if not assert_exists(path):
                missing_paths.append(path)
        print("depth_paths:")
        for cam_idx in sample["camera_indices"].tolist():
            path = dataset._resolve_depth_path(record["run_name"], cam_idx)
            if not assert_exists(path):
                missing_paths.append(path)
        print("camera_npz_paths:")
        for cam_idx in sample["camera_indices"].tolist():
            path = dataset._resolve_camera_path(record["run_name"], cam_idx)
            if not assert_exists(path):
                missing_paths.append(path)
        print(
            "shapes:",
            f"images={tuple(sample['images'].shape)}",
            f"object_images={tuple(sample['object_images'].shape)}",
            f"depth={tuple(sample['depth'].shape)}",
            f"world_points={tuple(sample['world_points'].shape)}",
        )
        valid_mask = sample["valid_mask"]
        valid_ratio = float(valid_mask.mean())
        filtered_ratio = 1.0 - valid_ratio
        valid_pixels = int(valid_mask.sum())
        total_pixels = int(valid_mask.size)
        depth = sample["depth"][..., 0]
        depth_min = float(depth.min())
        depth_mean = float(depth.mean())
        depth_max = float(depth.max())
        if valid_pixels > 0:
            valid_depth = depth[valid_mask]
            valid_depth_min = float(valid_depth.min())
            valid_depth_mean = float(valid_depth.mean())
            valid_depth_max = float(valid_depth.max())
        else:
            valid_depth_min = float("nan")
            valid_depth_mean = float("nan")
            valid_depth_max = float("nan")
        print(
            "mask_stats:",
            f"valid_ratio={valid_ratio:.4%}",
            f"filtered_ratio={filtered_ratio:.4%}",
            f"valid_pixels={valid_pixels}",
            f"total_pixels={total_pixels}",
        )
        print(
            "depth_stats_all:",
            f"min={depth_min:.6f}",
            f"mean={depth_mean:.6f}",
            f"max={depth_max:.6f}",
        )
        print(
            "depth_stats_valid:",
            f"min={valid_depth_min:.6f}",
            f"mean={valid_depth_mean:.6f}",
            f"max={valid_depth_max:.6f}",
        )
        print(
            "pose:",
            f"object_rotation={tuple(sample['object_rotation'].shape)}",
            f"object_translation={tuple(sample['object_translation'].shape)}",
        )
        print()

    print(f"checked_samples: {max_samples}")
    print(f"missing_path_count: {len(missing_paths)}")
    if missing_paths:
        print("missing_paths:")
        for path in missing_paths:
            print(f"  {path}")

    if args.strict and missing_paths:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
