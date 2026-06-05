#!/usr/bin/env python3
import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
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


DEFAULT_DATA_ROOT = "/mnt/train-data-4-hdd/yian/freepose/housecat6d"
DEFAULT_OBJECT_IMAGE_ROOT = "/mnt/train-data-4-hdd/yian/freepose/housecat6d/housecat6d_aligned_object_refs"
DEFAULT_ALIGN_JSON = "/mnt/train-data-4-hdd/yian/freepose/omni-object_clone/dataset_align.json"
DEFAULT_SAVE_DIR = (
    "/mnt/train-data-4-hdd/yian/freepose/omni-object_clone/omnivggt/datasets/"
    "housecat6d/verification"
)
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


def sample_image_to_pil(sample: dict[str, Any]) -> Image.Image:
    tensor = sample["images"][0].detach().cpu().float().clamp(0.0, 1.0)
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def object_ref_to_pil(sample: dict[str, Any], index: int) -> Image.Image:
    tensor = sample["object_images"][index].detach().cpu().float().clamp(0.0, 1.0)
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


def draw_axis_label_if_visible(
    draw: ImageDraw.ImageDraw,
    pixels: np.ndarray,
    valid: np.ndarray,
    axis_index: int,
    label: str,
    color: tuple[int, int, int],
) -> None:
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
    image = sample_image_to_pil(sample)
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
    draw_axis_label_if_visible(draw, axis_pixels, axis_valid, 1, "X", (255, 0, 0))
    draw_axis_label_if_visible(draw, axis_pixels, axis_valid, 2, "Y", (0, 160, 0))
    draw_axis_label_if_visible(draw, axis_pixels, axis_valid, 3, "Z", (0, 90, 255))

    if axis_valid[0]:
        x, y = axis_pixels[0]
        r = 4
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255), outline=(0, 0, 0))

    save_dir.mkdir(parents=True, exist_ok=True)
    scene_name = str(sample["scene_name"]).replace("/", "_")
    object_name = str(sample["object_name"]).replace("/", "_")
    image_id = int(sample["ids"][0])
    overlay_path = save_dir / f"{idx:06d}_{scene_name}_{object_name}_{image_id:06d}_axes.png"

    ref_images = [object_ref_to_pil(sample, i).resize((128, 128)) for i in range(sample["object_images"].shape[0])]
    canvas = Image.new("RGB", (image.width, image.height + 128), (255, 255, 255))
    canvas.paste(image, (0, 0))
    for i, ref in enumerate(ref_images):
        canvas.paste(ref, (i * 128, image.height))
    canvas.save(overlay_path)

    metadata_path = overlay_path.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "idx": idx,
                "scene_name": sample["scene_name"],
                "image_id": image_id,
                "object_name": sample["object_name"],
                "object_id": int(sample["object_id"]),
                "class_id": int(sample["class_id"]),
                "category": sample["category"],
                "scene_rgb_path": sample["scene_rgb_path"],
                "scene_label_path": sample["scene_label_path"],
                "scene_mask_path": sample["scene_mask_path"],
                "object_rgb_paths": list(sample["object_rgb_paths"]),
                "intrinsic": intrinsic.tolist(),
                "object_rotation_aligned_to_cam": rotation.tolist(),
                "object_rotation_native_to_cam": np.asarray(sample["object_rotation_native"]).tolist(),
                "R_align_housecat6d_to_ov9d": np.asarray(sample["R_align_housecat6d_to_ov9d"]).tolist(),
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


def _relative_angle_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    rel = rot_a.T @ rot_b
    cos = (np.trace(rel) - 1.0) / 2.0
    cos = float(np.clip(cos, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def run_relative_pose_test(args) -> None:
    """Verify the paired sampler and the relative-pose loss numerically.

    Checks:
      1. rot6d <-> matrix round trip.
      2. Paired sampler yields adjacent (same object, different frame) pairs.
      3. GT-as-prediction -> loss_rel_rot ~= 0; perturbed -> loss > 0.
      4. Right-multiplying a symmetric object's prediction by a symmetry -> loss ~= 0.
    """
    import torch

    from omnivggt.datasets.housecat6d import HouseCat6DCameraPose
    from omnivggt.datasets.base.batched_sampler import PairedObjectBatchSampler
    from omnivggt.loss import (
        _rot6d_to_matrix,
        _rotation_matrix_to_rot6d,
        _load_symmetry_info,
        compute_object_relative_pose_loss,
    )

    print("=" * 90)
    print("[1] rot6d <-> matrix round trip")
    torch.manual_seed(0)
    q = torch.randn(5, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    # build random rotations via Gram-Schmidt on random 3x3
    rand = torch.randn(5, 3, 3)
    u, _, v = torch.linalg.svd(rand)
    R = u @ v
    det = torch.linalg.det(R)
    R[det < 0, :, 2] *= -1
    rt = _rot6d_to_matrix(_rotation_matrix_to_rot6d(R))
    err = (rt - R).abs().max().item()
    print(f"    max |rot6d_to_matrix(rot6d(R)) - R| = {err:.3e}  ->  {'OK' if err < 1e-5 else 'FAIL'}")

    print("=" * 90)
    print("[2] paired sampler adjacency")
    dataset = HouseCat6DCameraPose(
        dataset_location=args.dataset_root,
        object_image_root=args.object_image_root,
        align_json=args.align_json,
        dset=args.dset,
        resolution=tuple(args.resolution),
        seed=args.seed,
        only_scene_name=args.only_scene_name,
        scene_glob=args.scene_glob,
        object_presence_prob=1.0,  # avoid absent-object injection during the test
        relative_pose_pairing=True,
        pair_min_frame_gap=args.pair_min_frame_gap,
        max_records=args.max_records,
    )
    groups, group_image_ids = dataset.build_pair_groups()
    print(f"    eligible (scene,object) groups (>=2 frames): {len(groups)}")
    sampler = PairedObjectBatchSampler(
        dataset, batch_size=args.batch_size, pool_size=1, groups=groups,
        group_image_ids=group_image_ids, min_frame_gap=args.pair_min_frame_gap,
    )
    sampler.set_epoch(0)
    flat = list(sampler)[: args.batch_size]
    rec_indices = [int(idx) for idx, _ in flat]
    bad = 0
    for k in range(0, len(rec_indices), 2):
        ri, rj = rec_indices[k], rec_indices[k + 1]
        rec_i, rec_j = dataset.records[ri], dataset.records[rj]
        same_obj = (rec_i["scene_name"], rec_i["object_name"]) == (rec_j["scene_name"], rec_j["object_name"])
        diff_frame = rec_i["image_id"] != rec_j["image_id"]
        if not (same_obj and diff_frame):
            bad += 1
        if k < 6:
            print(f"    pair {k//2}: {rec_i['scene_name']}/{rec_i['object_name']} "
                  f"frames=({rec_i['image_id']},{rec_j['image_id']}) "
                  f"same_obj={same_obj} diff_frame={diff_frame}")
    print(f"    invalid pairs in first batch: {bad}/{args.batch_size // 2}  ->  {'OK' if bad == 0 else 'FAIL'}")

    print("=" * 90)
    print("[3] relative-pose loss with GT-as-prediction and perturbation")
    # Build a small paired batch (2 pairs = 4 samples) from real data.
    sample_indices = rec_indices[:4]
    samples = [dataset[i] for i in sample_indices]
    gt_rot = torch.stack([torch.from_numpy(np.asarray(s["object_rotation"], dtype=np.float32)) for s in samples])
    object_id = torch.tensor([int(s["object_id"]) for s in samples], dtype=torch.long)
    has_object = torch.tensor([bool(s["has_object"]) for s in samples], dtype=torch.bool)
    dataset_labels = [str(s["dataset"]) for s in samples]
    scene_names = [str(s["scene_name"]) for s in samples]
    gt_trans_norm = torch.stack([torch.from_numpy(np.asarray(s["object_translation"], dtype=np.float32)) for s in samples])
    gt_trans_metric = torch.stack([torch.from_numpy(np.asarray(s["object_translation_metric"], dtype=np.float32)) for s in samples])
    gt_scale = torch.tensor([float(np.asarray(s["object_translation_scale"])) for s in samples], dtype=torch.float32)

    batch = {
        "object_rotation": gt_rot,
        "object_id": object_id,
        "has_object": has_object,
        "dataset": dataset_labels,
        "scene_name": scene_names,
        "object_translation": gt_trans_norm,
        "object_translation_metric": gt_trans_metric,
        "object_translation_scale": gt_scale,
    }
    print(f"    pair0 GT relative angle (cam motion): {_relative_angle_deg(gt_rot[0].numpy(), gt_rot[1].numpy()):.2f} deg")
    print(f"    pair1 GT relative angle (cam motion): {_relative_angle_deg(gt_rot[2].numpy(), gt_rot[3].numpy()):.2f} deg")

    sym_path = args.symmetry_info_path
    common = dict(weight_rot=1.0, weight_trans=1.0, loss_type="l1",
                  symmetry_info_path=sym_path, symmetry_continuous_steps=72)

    # GT as prediction -> loss ~ 0
    pred_gt = {"object_pose": _rotation_matrix_to_rot6d(gt_rot).clone(),
               "object_translation": gt_trans_norm.clone()}
    out_gt = compute_object_relative_pose_loss(pred_gt, batch, **common)
    # The residual ~1.4e-3 rad (0.08 deg) is the arccos clamp floor (sqrt(2*eps)),
    # not a real error -- treat anything < 0.5 deg as zero.
    gt_deg = np.degrees(out_gt['loss_rel_rot'].item())
    print(f"    GT-pred: loss_rel_rot={gt_deg:.3f} deg "
          f"loss_rel_trans={out_gt['loss_rel_trans'].item():.3e}  "
          f"->  {'OK' if gt_deg < 0.5 else 'FAIL'}")

    # Perturb only view 0 by a 20-deg rotation about z -> relative rotation breaks
    ang = np.radians(20.0)
    Rz = torch.tensor([[np.cos(ang), -np.sin(ang), 0.0],
                       [np.sin(ang), np.cos(ang), 0.0],
                       [0.0, 0.0, 1.0]], dtype=torch.float32)
    perturbed = gt_rot.clone()
    perturbed[0] = Rz @ perturbed[0]
    pred_bad = {"object_pose": _rotation_matrix_to_rot6d(perturbed).clone(),
                "object_translation": gt_trans_norm.clone()}
    out_bad = compute_object_relative_pose_loss(pred_bad, batch, **common)
    print(f"    perturbed-pred(+20deg on view0): loss_rel_rot={np.degrees(out_bad['loss_rel_rot'].item()):.2f} deg  "
          f"->  {'OK' if out_bad['loss_rel_rot'].item() > 0.1 else 'FAIL'}")

    print("=" * 90)
    print("[4] symmetry invariance of the relative-pose loss")
    sym_info = _load_symmetry_info(str(sym_path), 72) if sym_path else {}
    # find a pair whose object is symmetric (more than identity)
    sym_pair = None
    for k in range(0, 4, 2):
        key = f"{dataset_labels[k]}:{int(object_id[k])}"
        sym = sym_info.get(key)
        if sym is not None and sym.shape[0] > 1:
            sym_pair = (k, sym)
            break
    if sym_pair is None:
        print("    no symmetric object in the sampled pairs; skipping (not a failure).")
    else:
        k, sym = sym_pair
        S = sym[1]  # a non-identity symmetry rotation
        pred_sym_rot = gt_rot.clone()
        # right-multiply both views by (possibly different) symmetries
        pred_sym_rot[k] = gt_rot[k] @ S
        pred_sym_rot[k + 1] = gt_rot[k + 1] @ sym[min(2, sym.shape[0] - 1)]
        pred_sym = {"object_pose": _rotation_matrix_to_rot6d(pred_sym_rot).clone(),
                    "object_translation": gt_trans_norm.clone()}
        # isolate this pair by zeroing has_object on the other pair
        batch_one = dict(batch)
        ho = has_object.clone()
        for m in range(4):
            if m not in (k, k + 1):
                ho[m] = False
        batch_one["has_object"] = ho
        out_sym = compute_object_relative_pose_loss(pred_sym, batch_one, **common)
        print(f"    object_id={int(object_id[k])} sym candidates={sym.shape[0]}")
        print(f"    sym-perturbed pred (both views shifted by a symmetry): "
              f"loss_rel_rot={np.degrees(out_sym['loss_rel_rot'].item()):.3f} deg  "
              f"->  {'OK (symmetry absorbed)' if out_sym['loss_rel_rot'].item() < 1e-2 else 'FAIL'}")
    print("=" * 90)
    print("relative-pose test done.")


def _decollate_item(batch: dict[str, Any], i: int, batch_size: int) -> dict[str, Any]:
    """Pull sample ``i`` out of a collated batch, restoring the per-sample dict
    layout that ``HouseCat6DCameraPose.__getitem__`` produces.

    Handles default_collate quirks: str fields become a length-B list (index ``i``),
    while nested lists like ``object_rgb_paths`` become a transposed list of
    per-view tuples (gather ``[view[i] for view in value]``).
    """
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value[i]
        elif isinstance(value, (list, tuple)):
            if len(value) > 0 and isinstance(value[0], (list, tuple)):
                out[key] = [col[i] for col in value]          # transposed nested list
            elif len(value) == batch_size:
                out[key] = value[i]                            # per-sample list (e.g. str)
            else:
                out[key] = value
        else:
            out[key] = value
    return out


def run_dump_batch(args) -> None:
    """Pull ONE batch through the real DataLoader (same sampler + collate as
    training) and dump, per item: the scene image with GT pose overlaid, the
    object reference images, and a JSON of all paths + pose values, so the
    object<->scene<->pose<->path correspondence can be eyeballed."""
    import torch
    from torch.utils.data import DataLoader

    from omnivggt.datasets.housecat6d import HouseCat6DCameraPose
    from omnivggt.datasets import _intersection_collate

    dataset = HouseCat6DCameraPose(
        dataset_location=args.dataset_root,
        object_image_root=args.object_image_root,
        align_json=args.align_json,
        dset=args.dset,
        resolution=tuple(args.resolution),
        seed=args.seed,
        only_scene_name=args.only_scene_name,
        only_object_name=args.only_object_name,
        only_category=args.only_category,
        scene_glob=args.scene_glob,
        object_presence_prob=1.0,
        relative_pose_pairing=args.relative_pose_pairing,
        pair_min_frame_gap=args.pair_min_frame_gap,
        max_records=args.max_records,
    )

    batch_size = int(args.batch_size)
    sampler = dataset.make_sampler(batch_size, shuffle=True, world_size=1, rank=0, drop_last=True)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(0)
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=_intersection_collate,
        drop_last=True,
    )
    batch = next(iter(loader))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"pairing={args.relative_pose_pairing}  batch_size={batch_size}")
    print(f"collated batch keys: {sorted(batch.keys())}")
    print(f"images shape: {tuple(batch['images'].shape)}  object_images shape: {tuple(batch['object_images'].shape)}")

    n_dump = min(int(args.dump_num), batch_size)
    summary = []
    for i in range(n_dump):
        sample = _decollate_item(batch, i, batch_size)
        overlay_path = save_pose_overlay(sample, save_dir, i, args.axis_length_scale)
        pair_idx = i // 2 if args.relative_pose_pairing else None
        rec = {
            "batch_index": i,
            "pair_index": pair_idx,
            "scene_name": str(sample["scene_name"]),
            "image_id": int(sample["ids"][0]),
            "object_name": str(sample["object_name"]),
            "object_id": int(sample["object_id"]),
            "has_object": bool(sample["has_object"]),
            "scene_rgb_path": str(sample["scene_rgb_path"]),
            "object_rgb_paths": [str(p) for p in sample["object_rgb_paths"]],
            "object_translation_metric": np.asarray(sample["object_translation_metric"]).reshape(-1).tolist(),
            "overlay_path": str(overlay_path),
        }
        summary.append(rec)
        tag = f"pair{pair_idx}:" if pair_idx is not None else ""
        print(f"[{i}] {tag} {rec['scene_name']}/{rec['object_name']} frame={rec['image_id']} "
              f"has_object={rec['has_object']} -> {overlay_path.name}")

    if args.relative_pose_pairing:
        print("-" * 60)
        print("pair check (consecutive items should share scene+object, differ in frame):")
        for p in range(0, n_dump - 1, 2):
            a, b = summary[p], summary[p + 1]
            ok = (a["scene_name"], a["object_name"]) == (b["scene_name"], b["object_name"]) and a["image_id"] != b["image_id"]
            print(f"  pair {p//2}: {a['object_name']} frames=({a['image_id']},{b['image_id']})  {'OK' if ok else 'MISMATCH'}")

    summary_path = save_dir / "batch_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nsaved {n_dump} overlays + {summary_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify HouseCat6D dataset samples and draw aligned axes.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--object-image-root", default=DEFAULT_OBJECT_IMAGE_ROOT)
    parser.add_argument("--align-json", default=DEFAULT_ALIGN_JSON)
    parser.add_argument("--save-dir", type=Path, default=Path(DEFAULT_SAVE_DIR))
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, nargs=2, default=(518, 518), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--dset", default="test")
    parser.add_argument("--only-scene-name", default="")
    parser.add_argument("--only-object-name", default="")
    parser.add_argument("--only-category", default="")
    parser.add_argument("--axis-length-scale", type=float, default=0.65)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--no-save-overlays", action="store_true")
    parser.add_argument("--test-relative-pose", action="store_true",
                        help="Run the paired-sampler + relative-pose loss verification instead of overlays.")
    parser.add_argument("--scene-glob", default="scene*")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pair-min-frame-gap", type=int, default=20)
    parser.add_argument(
        "--symmetry-info-path",
        default="/mnt/train-data-4-hdd/yian/freepose/omni-object_clone/mixed_symmetry_info.json",
    )
    parser.add_argument("--dump-batch", action="store_true",
                        help="Pull one batch through the real DataLoader (sampler+collate) and dump "
                             "scene+object images, paths and GT pose overlay per item.")
    parser.add_argument("--dump-num", type=int, default=8, help="How many batch items to dump.")
    parser.add_argument("--relative-pose-pairing", action="store_true",
                        help="Use the paired sampler so consecutive items are same-object pairs.")
    return parser


def main() -> None:
    from omnivggt.datasets.housecat6d import HouseCat6DCameraPose

    args = build_argparser().parse_args()

    if args.test_relative_pose:
        run_relative_pose_test(args)
        return

    if args.dump_batch:
        run_dump_batch(args)
        return

    dataset = HouseCat6DCameraPose(
        dataset_location=args.dataset_root,
        object_image_root=args.object_image_root,
        align_json=args.align_json,
        dset=args.dset,
        resolution=tuple(args.resolution),
        seed=args.seed,
        only_scene_name=args.only_scene_name,
        only_object_name=args.only_object_name,
        only_category=args.only_category,
        max_records=args.max_records,
    )

    print(f"dataset_root: {dataset.dataset_location}")
    print(f"object_image_root: {dataset.object_image_root}")
    print(f"align_json: {dataset.align_json}")
    print(f"num_records: {len(dataset)}")
    print(f"num_objects: {len(dataset.object_records_by_name)}")
    print(f"fixed_object_view_ids: {dataset.fixed_object_view_ids}")

    end_idx = min(args.start_idx + args.num_samples, len(dataset))
    for idx in range(args.start_idx, end_idx):
        sample = dataset[idx]
        print("-" * 120)
        print(f"idx: {idx}")
        print(f"scene_name: {sample['scene_name']}")
        print(f"image_id: {int(sample['ids'][0])}")
        print(f"object_name: {sample['object_name']}")
        print(f"object_id: {int(sample['object_id'])}")
        print(f"class_id/category: {int(sample['class_id'])}/{sample['category']}")
        print(f"scene_rgb_path: {sample['scene_rgb_path']}")
        print(f"scene_depth_path: {sample['scene_depth_path']}")
        print(f"scene_label_path: {sample['scene_label_path']}")
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
        print(f"R_align_housecat6d_to_ov9d: {np.asarray(sample['R_align_housecat6d_to_ov9d']).tolist()}")
        if not args.no_save_overlays:
            overlay_path = save_pose_overlay(sample, args.save_dir, idx, args.axis_length_scale)
            print(f"pose_overlay_path: {overlay_path}")


if __name__ == "__main__":
    main()
