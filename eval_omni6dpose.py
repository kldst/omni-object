"""Evaluate a checkpoint on Omni6DPose (ROPE/SOPE) object 6D pose.

Pipeline (mirrors ``eval_housecat_official.py`` but for Omni6DPose):
  1. Build ``Omni6DPoseCameraPose`` over the requested scenes. Each sample is a
     (scene, frame, object-with-reference) triplet. Object reference views come
     from PAM meshes rendered by ``render_omni6dpose_object_refs_bpy.py``.
  2. Run ``model.inference`` with scene RGB + depth + object reference views and
     read ``object_pose`` (rot6d), ``object_translation`` (depth-normalized),
     ``object_size``.
  3. Because PAM ``Aligned.obj`` is already the GT canonical frame (R_align = I),
     the predicted rotation is compared directly to the GT object->camera
     rotation. Translation is rescaled by the per-sample depth mean.
  4. Report standard 6D pose metrics: mean/median rotation (deg) and translation
     (cm) error, and accuracy at {5deg2cm, 5deg5cm, 10deg2cm, 10deg5cm,
     10deg10cm}, overall and per class. Raw predictions + GT are saved to a pkl
     so the official cutoop evaluator (IoU / VUS-AUC) can be run later.

Example:
  python eval_omni6dpose.py \
      --checkpoint outputs/0531_REFER/lr_1e5_1000/model.safetensors \
      --object-image-root outputs/omni6dpose_refs/diverse24 \
      --view-ids 0 5 8 19 \
      --output-dir outputs/eval_omni6dpose_0531_REFER \
      --gpus 0
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from eval_real_multi import (
    build_model,
    rot6d_to_matrix,
    rotation_error_degrees,
    translation_error_cm,
)
from omnivggt.datasets.omni6dpose.omni6dpose_camera_pose import Omni6DPoseCameraPose
from omnivggt.datasets.utils.transforms import ImgNorm


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "0531_REFER" / "lr_1e5_1000" / "model.safetensors"
DEFAULT_DATA_ROOT = Path(
    "/mnt/train-data-4-hdd/yian/freepose/Omni6dpose/Omni6DPoseAPI/data/Omni6DPose/ROPE"
)
DEFAULT_OBJECT_IMAGE_ROOT = PROJECT_ROOT / "outputs" / "omni6dpose_refs" / "diverse24"
DEFAULT_OID_TO_PAM = PROJECT_ROOT / "outputs" / "omni6dpose_refs" / "rope_oid_to_pam.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eval_omni6dpose"
DEFAULT_RESOLUTION = (518, 476)
DEFAULT_VIEW_IDS = (0, 5, 8, 19)

# (rotation_deg, translation_cm) accuracy cells.
ACC_CELLS = [(5, 2), (5, 5), (10, 2), (10, 5), (10, 10)]


def build_dataset(args, scenes: Optional[List[str]]) -> Omni6DPoseCameraPose:
    view_ids = tuple(int(v) for v in (args.view_ids or DEFAULT_VIEW_IDS))
    ds = Omni6DPoseCameraPose(
        dataset_location=str(args.data_root),
        dset="test",
        object_image_root=str(args.object_image_root),
        oid_to_pam_json=str(args.oid_to_pam),
        scenes=scenes,
        fixed_object_view_ids=view_ids,
        num_object_views=len(view_ids),
        strict_fixed_object_view_ids=True,
        expand_records_by_object=True,
        normalize_object_translation_by_depth_mean=True,
        verify_files=True,
        z_far=int(args.z_far),
        resolution=DEFAULT_RESOLUTION,
        transform=ImgNorm,
        seed=42,
    )
    stride = int(getattr(args, "frame_stride", 1) or 1)
    if stride > 1:
        before = len(ds.records)
        ds.records = [r for r in ds.records if int(r.get("image_id", 0)) % stride == 0]
        ds.scenes = ds.records
        print(f"[frame-stride={stride}] {before} -> {len(ds.records)} records")
    return ds


def _to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def _batch_item(value, i: int):
    if torch.is_tensor(value):
        return value[i].detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value[i]
    if isinstance(value, (list, tuple)):
        return value[i]
    return value


def run_inference(args, dataset, model, device) -> List[Dict[str, Any]]:
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=False,
    )
    results: List[Dict[str, Any]] = []
    pbar = tqdm(loader, desc="infer omni6dpose", dynamic_ncols=True)
    seen = 0
    for batch in pbar:
        batch_dev = _to_device(batch, device)
        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=(not args.no_amp) and device.type == "cuda",
            ):
                outputs = model.inference(
                    images=batch_dev["images"],
                    object_images=batch_dev["object_images"],
                    extrinsics=None,
                    intrinsics=None,
                    depth=batch_dev["depth"],
                    mask=batch_dev["valid_mask"],
                    camera_gt_index=[],
                    depth_gt_index=[0],
                )
        if "object_pose" not in outputs or "object_translation" not in outputs:
            raise RuntimeError(f"Model output missing pose keys: {sorted(outputs.keys())}")
        pred_pose = outputs["object_pose"].detach().float().cpu().numpy()
        pred_trans = outputs["object_translation"].detach().float().cpu().numpy()
        if "object_size" in outputs:
            pred_size_batch = outputs["object_size"].detach().float().cpu().numpy()
        elif "object_size_log" in outputs:
            pred_size_batch = np.exp(outputs["object_size_log"].detach().float().cpu().numpy())
        else:
            pred_size_batch = None

        bs = int(pred_pose.shape[0])
        for i in range(bs):
            depth_mean = float(np.asarray(_batch_item(batch["depth_mean_scale"], i)).reshape(-1)[0])
            pred_R = rot6d_to_matrix(pred_pose[i])  # R_align = I -> already object->cam
            pred_t = np.asarray(pred_trans[i], dtype=np.float64).reshape(3) * depth_mean
            gt_R = np.asarray(_batch_item(batch["object_rotation"], i), dtype=np.float64).reshape(3, 3)
            gt_t = np.asarray(_batch_item(batch["object_translation_metric"], i), dtype=np.float64).reshape(3)
            gt_size = np.asarray(_batch_item(batch["object_size"], i), dtype=np.float64).reshape(3)
            pred_size = (
                np.asarray(pred_size_batch[i], dtype=np.float64).reshape(3)
                if pred_size_batch is not None else None
            )
            results.append(
                {
                    "scene_name": str(_batch_item(batch["scene_name"], i)),
                    "oid": str(_batch_item(batch["oid"], i)),
                    "object_name": str(_batch_item(batch["object_name"], i)),
                    "category": str(_batch_item(batch["category"], i)),
                    "class_id": int(np.asarray(_batch_item(batch["class_id"], i)).reshape(-1)[0]),
                    "image_id": int(np.asarray(_batch_item(batch["ids"], i)).reshape(-1)[0]),
                    "rot_err_deg": rotation_error_degrees(pred_R, gt_R),
                    "trans_err_cm": translation_error_cm(pred_t, gt_t),
                    "size_err_cm": (float(np.linalg.norm(pred_size - gt_size) * 100.0)
                                    if pred_size is not None else None),
                    "pred_R": pred_R, "pred_t": pred_t, "pred_size": pred_size,
                    "gt_R": gt_R, "gt_t": gt_t, "gt_size": gt_size,
                    "depth_mean": depth_mean,
                }
            )
            seen += 1
        pbar.set_postfix(samples=seen)
        if args.limit is not None and seen >= args.limit:
            break
    pbar.close()
    return results


def _accuracy(records: List[Dict[str, Any]]) -> Dict[str, float]:
    n = len(records)
    if n == 0:
        return {}
    rot = np.array([r["rot_err_deg"] for r in records], dtype=np.float64)
    tr = np.array([r["trans_err_cm"] for r in records], dtype=np.float64)
    out = {
        "count": n,
        "rot_mean_deg": float(rot.mean()),
        "rot_median_deg": float(np.median(rot)),
        "trans_mean_cm": float(tr.mean()),
        "trans_median_cm": float(np.median(tr)),
    }
    for deg, cm in ACC_CELLS:
        out[f"acc_{deg}deg_{cm}cm"] = float(np.mean((rot <= deg) & (tr <= cm)))
    out["acc_rot_5deg"] = float(np.mean(rot <= 5))
    out["acc_rot_10deg"] = float(np.mean(rot <= 10))
    out["acc_trans_2cm"] = float(np.mean(tr <= 2))
    out["acc_trans_5cm"] = float(np.mean(tr <= 5))
    return out


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    overall = _accuracy(results)
    by_class: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        by_class[r["category"] or str(r["class_id"])].append(r)
    per_class = {cat: _accuracy(recs) for cat, recs in sorted(by_class.items())}
    return {"overall": overall, "per_class": per_class}


def print_summary(summary: Dict[str, Any]) -> None:
    o = summary["overall"]
    if not o:
        print("[eval] no samples.")
        return
    print("\n================ Omni6DPose pose eval ================")
    print(f"samples={o['count']}")
    print(f"rotation    mean={o['rot_mean_deg']:.2f}deg  median={o['rot_median_deg']:.2f}deg  "
          f"(<=5: {o['acc_rot_5deg']*100:.1f}%  <=10: {o['acc_rot_10deg']*100:.1f}%)")
    print(f"translation mean={o['trans_mean_cm']:.2f}cm   median={o['trans_median_cm']:.2f}cm  "
          f"(<=2: {o['acc_trans_2cm']*100:.1f}%  <=5: {o['acc_trans_5cm']*100:.1f}%)")
    for deg, cm in ACC_CELLS:
        print(f"  acc {deg}deg/{cm}cm : {o[f'acc_{deg}deg_{cm}cm']*100:.2f}%")
    print("------------------ per class --------------------------")
    print(f"{'class':24s} {'n':>5s} {'rotMed':>7s} {'trMed':>7s} {'5/5':>6s} {'10/5':>6s}")
    for cat, m in summary["per_class"].items():
        print(f"{cat[:24]:24s} {m['count']:5d} {m['rot_median_deg']:7.2f} {m['trans_median_cm']:7.2f} "
              f"{m['acc_5deg_5cm']*100:5.1f} {m['acc_10deg_5cm']*100:5.1f}")
    print("======================================================\n")


def save_outputs(results: List[Dict[str, Any]], summary: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.pkl").open("wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as h:
        json.dump(summary, h, indent=2)
    print(f"[eval] saved predictions.pkl + metrics.json to {out_dir}")


def discover_scenes(data_root: Path) -> List[str]:
    return sorted(d.name for d in data_root.iterdir() if d.is_dir() and len(d.name) == 6 and d.name.isdigit())


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--object-image-root", type=Path, default=DEFAULT_OBJECT_IMAGE_ROOT)
    p.add_argument("--oid-to-pam", type=Path, default=DEFAULT_OID_TO_PAM)
    p.add_argument("--view-ids", nargs="+", type=int, default=list(DEFAULT_VIEW_IDS))
    p.add_argument("--scenes", nargs="+", default=None, help="Scene ids (e.g. 000000 000001). Default: all.")
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--z-far", type=float, default=20.0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--gpus", type=str, default=None, help="Single GPU id to pin (e.g. 0).")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus).split(",")[0]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device} checkpoint={args.checkpoint}")
    scenes = args.scenes if args.scenes else discover_scenes(args.data_root)
    print(f"[eval] scenes={scenes}")

    model = build_model(args.checkpoint, device)
    dataset = build_dataset(args, scenes)
    print(f"[eval] dataset records={len(dataset.records)} objects={len(dataset.object_records_by_name)}")

    t0 = time.time()
    results = run_inference(args, dataset, model, device)
    summary = summarize(results)
    summary["meta"] = {
        "checkpoint": str(args.checkpoint),
        "scenes": scenes,
        "view_ids": list(args.view_ids),
        "frame_stride": args.frame_stride,
        "num_samples": len(results),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    print_summary(summary)
    save_outputs(results, summary, args.output_dir)


if __name__ == "__main__":
    main()
