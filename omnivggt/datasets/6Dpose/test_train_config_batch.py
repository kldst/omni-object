import argparse
from collections import Counter
import math
import sys
from pathlib import Path

# MPLCONFIGDIR=/tmp/mpl /mnt/train-data-5-hdd/yian/anacond/envs/vggt/bin/python \
# /mnt/train-data-4-hdd/yian/6dpose_obj/OmniVGGT-official/omnivggt/datasets/6Dpose/test_train_config_batch.py


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Inspect the first training batch resolved from configs/train.py."
    )
    parser.add_argument(
        "--config",
        default="/mnt/train-data-4-hdd/yian/6dpose_obj/OmniVGGT-official/configs/train.py",
        help="Path to the training config.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=0,
        help="Epoch index used to seed the dataset/sampler before fetching the first batch.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Override dataloader workers for debugging.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=8,
        help="Maximum number of batch samples to print.",
    )
    return parser


def _to_list(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return [value]


def _infer_batch_size(batch):
    for key in ("run_name", "object_name", "images", "object_images", "object_rotation"):
        if key not in batch:
            continue
        value = batch[key]
        if hasattr(value, "shape") and len(value.shape) > 0:
            return int(value.shape[0])
        items = _to_list(value)
        if items:
            return len(items)
    return 0


def _unwrap_dataset(dataset):
    current = dataset
    wrappers = []
    while hasattr(current, "dataset"):
        wrappers.append(type(current).__name__)
        current = current.dataset
    return current, wrappers


def main():
    args = build_argparser().parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))

    from omnivggt.utils.configs import read_config
    from omnivggt.datasets import get_data_loader

    cfg = read_config(args.config)
    train_batch_images = int(cfg.get("train_batch_images", 1))
    gradient_accumulation_steps = int(cfg.get("gradient_accumulation_steps", 1))
    num_train_epochs = int(cfg.get("num_train_epochs", 1))

    loader = get_data_loader(
        cfg.train_dataset,
        batch_size=train_batch_images,
        num_workers=args.num_workers,
        pin_mem=True,
        shuffle=True,
        drop_last=True,
    )

    if hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(args.epoch)
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(args.epoch)

    base_dataset, wrappers = _unwrap_dataset(loader.dataset)
    batch = next(iter(loader))
    base_records = getattr(base_dataset, "records", None)
    run_counter = Counter(record["run_name"] for record in base_records) if base_records is not None else Counter()

    dataset_total_samples = len(loader.dataset)
    sampler_total_samples = len(loader.sampler) if hasattr(loader, "sampler") else None
    dataloader_batches_per_epoch = len(loader)
    optimizer_steps_per_epoch = dataloader_batches_per_epoch // max(1, gradient_accumulation_steps)
    total_optimizer_steps = optimizer_steps_per_epoch * num_train_epochs
    actual_batch_samples = _infer_batch_size(batch)

    print(f"config: {args.config}")
    print(f"dataset wrappers: {wrappers if wrappers else ['None']}")
    print(f"base dataset type: {type(base_dataset).__name__}")
    print(f"train_dataset expr: {cfg.train_dataset}")
    if base_records is not None:
        print(f"base SixDPose raw samples: {len(base_records)}")
        print(f"base SixDPose unique runs: {len(run_counter)}")
        if run_counter:
            run_object_counts = list(run_counter.values())
            print(
                "objects per run stats: "
                f"min={min(run_object_counts)} "
                f"max={max(run_object_counts)} "
                f"mean={sum(run_object_counts) / len(run_object_counts):.2f}"
            )
    print(f"num_train_epochs: {num_train_epochs}")
    print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")
    print(f"configured train_batch_images: {train_batch_images}")
    print(f"dataset total samples per epoch: {dataset_total_samples}")
    if sampler_total_samples is not None:
        print(f"sampler samples per epoch: {sampler_total_samples}")
    print(f"dataloader batches per epoch: {dataloader_batches_per_epoch}")
    print(f"optimizer steps per epoch: {optimizer_steps_per_epoch}")
    print(f"total optimizer steps: {total_optimizer_steps}")
    print(f"actual first-batch sample count: {actual_batch_samples}")
    print()

    if run_counter:
        print("first 10 runs object counts:")
        for run_name, count in sorted(run_counter.items())[:10]:
            print(f"  {run_name}: {count}")
        print()

    shapes_to_print = [
        "images",
        "extrinsic",
        "intrinsic",
        "depth",
        "valid_mask",
        "object_images",
        "object_rotation",
        "object_translation",
    ]
    print("batch tensor shapes:")
    for key in shapes_to_print:
        if key in batch and hasattr(batch[key], "shape"):
            print(f"  {key}: {tuple(batch[key].shape)}")
    print()

    run_names = [str(x) for x in _to_list(batch.get("run_name", []))]
    object_names = [str(x) for x in _to_list(batch.get("object_name", []))]
    camera_indices = _to_list(batch.get("camera_indices", []))
    object_cam_indices = _to_list(batch.get("object_cam_indices", []))
    ref_cam_indices = _to_list(batch.get("object_pose_reference_camera_index", []))

    samples_to_print = min(args.max_samples, actual_batch_samples, len(run_names), len(object_names))
    print(f"first batch resolved paths (showing {samples_to_print} samples):")
    for sample_idx in range(samples_to_print):
        run_name = run_names[sample_idx]
        object_name = object_names[sample_idx]
        scene_views = _to_list(camera_indices[sample_idx]) if sample_idx < len(camera_indices) else []
        object_views = _to_list(object_cam_indices[sample_idx]) if sample_idx < len(object_cam_indices) else []
        ref_view = ref_cam_indices[sample_idx] if sample_idx < len(ref_cam_indices) else None

        print(f"[sample {sample_idx}] run={run_name} object={object_name} ref_camera={ref_view}")
        print("  input scene RGB:")
        for cam_idx in scene_views:
            print(f"    {base_dataset._resolve_scene_image_path(run_name, int(cam_idx))}")
        print("  input scene depth:")
        for cam_idx in scene_views:
            print(f"    {base_dataset._resolve_depth_path(run_name, int(cam_idx))}")
        print("  input camera npz:")
        for cam_idx in scene_views:
            print(f"    {base_dataset._resolve_camera_path(run_name, int(cam_idx))}")
        print("  input object RGB:")
        for cam_idx in object_views:
            print(f"    {base_dataset._resolve_object_image_path(object_name, int(cam_idx))}")
        print("  gt pose source:")
        print(f"    {base_dataset.pose_root}/{run_name}.npz")
        print()


if __name__ == "__main__":
    main()
