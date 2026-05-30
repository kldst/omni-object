"""REAL275 evaluation using the NOCS symmetry handling.

Mirrors ``NOCS_CVPR2019/utils.py:compute_RT_degree_cm_symmetry`` semantics:
  - bottle / bowl / can         -> y-axis continuous symmetry
  - mug with handle_visibility=0 -> y-axis continuous symmetry
  - mug with handle_visibility=1 -> plain rotation error
  - camera / laptop             -> plain rotation error

REAL275 GT pickles do not ship ``handle_visibility``; following NOCS detect_eval.py
we default it to 1 (i.e. mug is treated as non-symmetric). Pass --mug-handle-invisible
to flip the default.

Reports per-class accuracy at 5/2cm, 5/5cm, 10/2cm, 10/5cm plus the class-mean.

Reuses the dataset, model builder and worker loop from ``eval_real_multi.py``.

Multi-GPU example:
    python eval_real275_nocs.py --gpus 0,1,2,3 \
        --checkpoint outputs/0521/10000/model.safetensors \
        --output-dir outputs/eval_real275_nocs_0521
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from eval_real_multi import (
    DEFAULT_ALIGN_JSON,
    DEFAULT_CHECKPOINT,
    DEFAULT_REAL275_ROOT,
    PROJECT_ROOT,
    Real275RealTestCameraPose,
    _batch_item,
    _to_device,
    build_model,
    rot6d_to_matrix,
)


REAL275_CLASS_ID_TO_NAME = {
    1: "bottle",
    2: "bowl",
    3: "camera",
    4: "can",
    5: "laptop",
    6: "mug",
}
NOCS_SYMMETRIC_CLASSES = {"bottle", "bowl", "can"}
ORDERED_CLASS_NAMES = ("bottle", "bowl", "camera", "can", "laptop", "mug")

THRESHOLDS = (
    ("5deg_2cm", 5.0, 2.0),
    ("5deg_5cm", 5.0, 5.0),
    ("10deg_2cm", 10.0, 2.0),
    ("10deg_5cm", 10.0, 5.0),
)


# ============================================================================
# NOCS-style symmetric rotation error.
# ============================================================================
def nocs_rotation_error_degrees(
    pred_R: np.ndarray,
    gt_R: np.ndarray,
    class_name: str,
    handle_visibility: int,
) -> float:
    """Mirror NOCS ``compute_RT_degree_cm_symmetry`` for the rotation term.

    handle_visibility: 1 = handle visible (treat mug as non-symmetric),
                       0 = handle invisible (mug uses y-axis continuous symmetry).
    """
    pred_R = np.asarray(pred_R, dtype=np.float64).reshape(3, 3)
    gt_R = np.asarray(gt_R, dtype=np.float64).reshape(3, 3)
    use_y_symmetry = (
        class_name in NOCS_SYMMETRIC_CLASSES
        or (class_name == "mug" and int(handle_visibility) == 0)
    )
    if use_y_symmetry:
        y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        y1 = pred_R @ y
        y2 = gt_R @ y
        denom = float(np.linalg.norm(y1) * np.linalg.norm(y2))
        if denom < 1e-12:
            return float("nan")
        cos = float(np.dot(y1, y2) / denom)
    else:
        rel = pred_R @ gt_R.T
        cos = float((np.trace(rel) - 1.0) * 0.5)
    cos = max(-1.0, min(1.0, cos))
    return float(np.degrees(np.arccos(cos)))


def translation_error_cm(pred_t_metric: np.ndarray, gt_t_metric: np.ndarray) -> float:
    diff = np.asarray(pred_t_metric, dtype=np.float64) - np.asarray(gt_t_metric, dtype=np.float64)
    return float(np.linalg.norm(diff) * 100.0)


# ============================================================================
# Worker.
# ============================================================================
def build_dataset(args) -> Real275RealTestCameraPose:
    from eval_real_multi import DEFAULT_OBJECT_VIEWS, DEFAULT_RESOLUTION
    from omnivggt.datasets.utils.transforms import ImgNorm

    return Real275RealTestCameraPose(
        dataset_location=str(args.real275_root),
        split_root=str(args.real275_root / "real_test"),
        gt_root=str(args.real275_root / "gts" / "real_test"),
        object_image_root=str(args.real275_root / "real275_aligned_object_refs"),
        object_image_root_instance=str(args.real275_instance_refs) if args.real275_instance_refs else None,
        align_json=str(args.align_json),
        only_scene_name=str(args.real275_only_scene or ""),
        num_object_views=4,
        fixed_object_view_ids=DEFAULT_OBJECT_VIEWS,
        normalize_object_translation_by_depth_mean=True,
        verify_files=True,
        z_far=20,
        resolution=DEFAULT_RESOLUTION,
        transform=ImgNorm,
        seed=42,
    )


def evaluate_real275(
    dataset: Real275RealTestCameraPose,
    model,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
    shard_index: int,
    num_shards: int,
    limit: Optional[int],
    mug_handle_visibility: int,
) -> List[Dict[str, Any]]:
    indices = list(range(shard_index, len(dataset), num_shards))
    if limit is not None:
        indices = indices[:limit]
    if not indices:
        return []
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=False,
    )
    samples: List[Dict[str, Any]] = []
    pbar = tqdm(loader, desc=f"shard {shard_index}/{num_shards} real275", dynamic_ncols=True)
    for batch in pbar:
        batch_dev = _to_device(batch, device)
        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"
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
        pred_pose = outputs["object_pose"].detach().float().cpu().numpy()
        pred_translation = outputs["object_translation"].detach().float().cpu().numpy()
        bs = int(pred_pose.shape[0])
        for i in range(bs):
            if not bool(np.asarray(_batch_item(batch["has_object"], i)).reshape(-1)[0]):
                continue
            class_id = int(np.asarray(_batch_item(batch["class_id"], i)).reshape(-1)[0])
            class_name = REAL275_CLASS_ID_TO_NAME.get(class_id, str(class_id))
            gt_rot = np.asarray(_batch_item(batch["object_rotation"], i), dtype=np.float64).reshape(3, 3)
            gt_t_metric = np.asarray(
                _batch_item(batch["object_translation_metric"], i), dtype=np.float64
            ).reshape(3)
            depth_mean = float(np.asarray(_batch_item(batch["depth_mean_scale"], i)).reshape(-1)[0])
            pred_rot = rot6d_to_matrix(pred_pose[i])
            pred_t_norm = np.asarray(pred_translation[i], dtype=np.float64).reshape(3)
            pred_t_metric = pred_t_norm * depth_mean
            rot_err = nocs_rotation_error_degrees(
                pred_rot, gt_rot, class_name, mug_handle_visibility
            )
            trans_err = translation_error_cm(pred_t_metric, gt_t_metric)
            samples.append(
                {
                    "scene_name": str(_batch_item(batch["scene_name"], i)),
                    "image_id": int(np.asarray(_batch_item(batch["ids"], i)).reshape(-1)[0]),
                    "inst_id": int(np.asarray(_batch_item(batch["inst_id"], i)).reshape(-1)[0]),
                    "class_id": class_id,
                    "class_name": class_name,
                    "object_name": str(_batch_item(batch["object_name"], i)),
                    "handle_visibility": int(mug_handle_visibility) if class_name == "mug" else 1,
                    "rotation_error_deg": float(rot_err),
                    "translation_error_cm": float(trans_err),
                    "pred_translation_metric": pred_t_metric.tolist(),
                    "gt_translation_metric": gt_t_metric.tolist(),
                    "depth_mean_scale_m": float(depth_mean),
                }
            )
    return samples


# ============================================================================
# Reports.
# ============================================================================
def summarize(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {"num_samples": 0, "per_class": {}, "class_mean": {}, "overall": {}}

    rot = np.asarray([s["rotation_error_deg"] for s in samples], dtype=np.float64)
    trans = np.asarray([s["translation_error_cm"] for s in samples], dtype=np.float64)
    finite_mask = np.isfinite(rot) & np.isfinite(trans)
    classes = np.asarray([s["class_name"] for s in samples], dtype=object)

    per_class: Dict[str, Dict[str, Any]] = {}
    for cname in ORDERED_CLASS_NAMES:
        mask = (classes == cname) & finite_mask
        n = int(mask.sum())
        entry: Dict[str, Any] = {"n": n}
        if n == 0:
            for tname, _, _ in THRESHOLDS:
                entry[tname] = None
            entry["rot_mean"] = None
            entry["trans_mean_cm"] = None
        else:
            for tname, rt, tt in THRESHOLDS:
                entry[tname] = 100.0 * float(np.mean((rot[mask] <= rt) & (trans[mask] <= tt)))
            entry["rot_mean"] = float(rot[mask].mean())
            entry["rot_median"] = float(np.median(rot[mask]))
            entry["trans_mean_cm"] = float(trans[mask].mean())
            entry["trans_median_cm"] = float(np.median(trans[mask]))
        per_class[cname] = entry

    class_mean: Dict[str, Optional[float]] = {}
    for tname, _, _ in THRESHOLDS:
        vals = [per_class[c][tname] for c in ORDERED_CLASS_NAMES if per_class[c][tname] is not None]
        class_mean[tname] = float(np.mean(vals)) if vals else None

    overall: Dict[str, Any] = {"n": int(finite_mask.sum())}
    if overall["n"] > 0:
        for tname, rt, tt in THRESHOLDS:
            overall[tname] = 100.0 * float(np.mean((rot[finite_mask] <= rt) & (trans[finite_mask] <= tt)))
        overall["rot_mean"] = float(rot[finite_mask].mean())
        overall["trans_mean_cm"] = float(trans[finite_mask].mean())

    return {
        "num_samples": len(samples),
        "num_finite": int(finite_mask.sum()),
        "per_class": per_class,
        "class_mean": class_mean,
        "overall": overall,
    }


def print_report(summary: Dict[str, Any]) -> None:
    headers = ["class", "n", *[name for name, _, _ in THRESHOLDS], "rot_mean", "trans_cm_mean"]
    rows: List[List[str]] = []
    for cname in ORDERED_CLASS_NAMES:
        entry = summary["per_class"].get(cname, {"n": 0})
        cells = [cname, str(entry.get("n", 0))]
        for name, _, _ in THRESHOLDS:
            v = entry.get(name)
            cells.append(f"{v:.2f}" if v is not None else "N/A")
        rm = entry.get("rot_mean")
        tm = entry.get("trans_mean_cm")
        cells.append(f"{rm:.2f}" if rm is not None else "N/A")
        cells.append(f"{tm:.2f}" if tm is not None else "N/A")
        rows.append(cells)

    mean_cells = ["class_mean", ""]
    for name, _, _ in THRESHOLDS:
        v = summary["class_mean"].get(name)
        mean_cells.append(f"{v:.2f}" if v is not None else "N/A")
    mean_cells.extend(["", ""])
    rows.append(mean_cells)

    overall = summary.get("overall", {})
    overall_cells = ["overall(no_avg)", str(overall.get("n", 0))]
    for name, _, _ in THRESHOLDS:
        v = overall.get(name)
        overall_cells.append(f"{v:.2f}" if v is not None else "N/A")
    rm_o = overall.get("rot_mean")
    tm_o = overall.get("trans_mean_cm")
    overall_cells.append(f"{rm_o:.2f}" if rm_o is not None else "N/A")
    overall_cells.append(f"{tm_o:.2f}" if tm_o is not None else "N/A")
    rows.append(overall_cells)

    widths = [max(len(c) for c in col) for col in zip(headers, *rows)]
    fmt = " | ".join("{:<" + str(w) + "}" for w in widths)
    print(fmt.format(*headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt.format(*row))


# ============================================================================
# Orchestration.
# ============================================================================
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--align-json", type=Path, default=DEFAULT_ALIGN_JSON)
    parser.add_argument("--real275-root", type=Path, default=DEFAULT_REAL275_ROOT)
    parser.add_argument(
        "--real275-instance-refs",
        type=Path,
        default=DEFAULT_REAL275_ROOT / "real275_aligned_object_refs_test",
    )
    parser.add_argument("--real275-only-scene", type=str, default="")
    parser.add_argument(
        "--mug-handle-invisible",
        action="store_true",
        help="Treat every mug sample as handle_visibility=0 (y-axis symmetric). "
             "Default follows NOCS detect_eval.py fallback (handle_visibility=1).",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "eval_real275_nocs",
    )
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args(argv)


def run_worker(args: argparse.Namespace, shard_index: int, num_shards: int) -> None:
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[worker] shard={shard_index}/{num_shards} device={device}")
    model = build_model(args.checkpoint, device)
    dataset = build_dataset(args)
    samples = evaluate_real275(
        dataset=dataset,
        model=model,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        amp=not args.no_amp,
        shard_index=shard_index,
        num_shards=num_shards,
        limit=args.limit,
        mug_handle_visibility=0 if args.mug_handle_invisible else 1,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = args.output_dir / f"samples_real275_shard_{shard_index:02d}.jsonl"
    with shard_path.open("w", encoding="utf-8") as h:
        for s in samples:
            h.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[worker] samples={len(samples)} wrote {shard_path}")


def orchestrate(args: argparse.Namespace, gpu_ids: List[str]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    num_shards = len(gpu_ids)
    print(f"[orchestrator] launching {num_shards} shards on GPUs {gpu_ids}")
    base_cmd: List[str] = [
        sys.executable, "-u", str(Path(__file__).resolve()),
        "--checkpoint", str(args.checkpoint),
        "--align-json", str(args.align_json),
        "--real275-root", str(args.real275_root),
        "--real275-instance-refs", str(args.real275_instance_refs),
        "--real275-only-scene", str(args.real275_only_scene or ""),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--num-shards", str(num_shards),
        "--output-dir", str(args.output_dir),
    ]
    if args.mug_handle_invisible:
        base_cmd.append("--mug-handle-invisible")
    if args.no_amp:
        base_cmd.append("--no-amp")
    if args.limit is not None:
        base_cmd.extend(["--limit", str(args.limit)])

    procs: List[Tuple[int, subprocess.Popen]] = []
    for shard_index, gpu_id in enumerate(gpu_ids):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        log_path = args.output_dir / f"shard_{shard_index:02d}.log"
        cmd = base_cmd + ["--shard-index", str(shard_index)]
        log_fh = open(log_path, "w", encoding="utf-8")
        print(f"[orchestrator] shard {shard_index} -> GPU {gpu_id}  log={log_path}")
        procs.append((shard_index, subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)))

    start = time.time()
    while True:
        time.sleep(15)
        alive = [s for s, p in procs if p.poll() is None]
        for shard_index, p in procs:
            log_path = args.output_dir / f"shard_{shard_index:02d}.log"
            try:
                with log_path.open("rb") as h:
                    h.seek(0, os.SEEK_END)
                    size = h.tell()
                    h.seek(max(0, size - 600), os.SEEK_SET)
                    tail = h.read().decode("utf-8", errors="replace")
                last = tail.strip().splitlines()[-1] if tail.strip() else ""
                print(f"  shard {shard_index}: alive={p.poll() is None}  tail={last[-180:]}")
            except FileNotFoundError:
                print(f"  shard {shard_index}: log missing")
        print(f"[orchestrator] t={time.time() - start:6.0f}s  alive={len(alive)}/{len(procs)}")
        if not alive:
            break

    return_codes = [p.wait() for _, p in procs]
    if any(rc != 0 for rc in return_codes):
        print(f"[orchestrator] return codes: {return_codes}")

    merged: List[Dict[str, Any]] = []
    for shard_index, _ in procs:
        shard_path = args.output_dir / f"samples_real275_shard_{shard_index:02d}.jsonl"
        if not shard_path.is_file():
            continue
        with shard_path.open("r", encoding="utf-8") as h:
            for line in h:
                line = line.strip()
                if line:
                    merged.append(json.loads(line))
    merged_path = args.output_dir / "samples_real275.jsonl"
    with merged_path.open("w", encoding="utf-8") as h:
        for s in merged:
            h.write(json.dumps(s, ensure_ascii=False) + "\n")

    summary = summarize(merged)
    summary_meta = {
        "checkpoint": str(args.checkpoint),
        "symmetry_rules": "NOCS_CVPR2019 compute_RT_degree_cm_symmetry",
        "mug_handle_visibility_default": 0 if args.mug_handle_invisible else 1,
        "thresholds": [{"name": n, "rot_deg": r, "trans_cm": t} for n, r, t in THRESHOLDS],
        "ordered_classes": list(ORDERED_CLASS_NAMES),
        "summary": summary,
    }
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as h:
        json.dump(summary_meta, h, indent=2)
    print(f"[orchestrator] wrote {summary_path}  merged={len(merged)} samples")
    print_report(summary)


def main(argv=None) -> None:
    args = parse_args(argv)
    is_worker = args.shard_index is not None
    if not is_worker and args.gpus is not None and "," in args.gpus:
        gpu_ids = [x.strip() for x in args.gpus.split(",") if x.strip()]
        if len(gpu_ids) > 1:
            return orchestrate(args, gpu_ids)
    if not is_worker and args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    shard_index = 0 if args.shard_index is None else int(args.shard_index)
    num_shards = int(args.num_shards) if args.num_shards else 1
    run_worker(args, shard_index=shard_index, num_shards=num_shards)

    if not is_worker:
        shard_path = args.output_dir / f"samples_real275_shard_{shard_index:02d}.jsonl"
        samples: List[Dict[str, Any]] = []
        if shard_path.is_file():
            with shard_path.open("r", encoding="utf-8") as h:
                for line in h:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
        with (args.output_dir / "samples_real275.jsonl").open("w", encoding="utf-8") as h:
            for s in samples:
                h.write(json.dumps(s, ensure_ascii=False) + "\n")
        summary = summarize(samples)
        summary_meta = {
            "checkpoint": str(args.checkpoint),
            "symmetry_rules": "NOCS_CVPR2019 compute_RT_degree_cm_symmetry",
            "mug_handle_visibility_default": 0 if args.mug_handle_invisible else 1,
            "thresholds": [{"name": n, "rot_deg": r, "trans_cm": t} for n, r, t in THRESHOLDS],
            "ordered_classes": list(ORDERED_CLASS_NAMES),
            "summary": summary,
        }
        summary_path = args.output_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as h:
            json.dump(summary_meta, h, indent=2)
        print(f"[eval] wrote {summary_path}  samples={len(samples)}")
        print_report(summary)


if __name__ == "__main__":
    main()
