"""Evaluate a checkpoint on HouseCat6D test_scene{1..5} via the official eval.

Pipeline:
  1. Run model on every (scene, image, object) sample that has reference views.
  2. Convert each prediction from the OV9D-aligned frame back to the HouseCat6D
     native frame using R_align stored in ``dataset_align.json``.
  3. For each frame, dump a per-image pkl whose schema matches what VI-Net's
     ``test_func`` writes:
         gt_class_ids, gt_RTs, gt_scales, gt_bboxes,
         pred_class_ids, pred_RTs, pred_scales, pred_bboxes, pred_scores
     GT comes from the original housecat6d label pkl (covers every object in
     the frame, not just those with refs) so mAP recall is computed against the
     full ground-truth set.
  4. Call ``HouseCat6D/VI-Net/utils/evaluation_utils.evaluate_housecat`` which
     prints 3D IoU mAP at {25, 50, 75}% and pose mAP at the four cells the user
     cares about: 5deg/2cm, 5deg/5cm, 10deg/2cm, 10deg/5cm.

The official metric uses class-name-based symmetry handling
(``compute_RT_degree_cm_symmetry``): bottle / can / glass are treated as y-axis
symmetric (compares y-axis vectors only). No mixed_symmetry_info.json is read.

Multi-GPU: ``--gpus 0,1,2,3`` splits the five scenes across GPUs as subprocesses.
After all shards finish, the orchestrator calls evaluate_housecat once.

Examples:
  python eval_housecat_official.py \\
      --checkpoint outputs/0521/10000/model.safetensors \\
      --output-dir outputs/eval_housecat_official_0521 \\
      --gpus 0,1,2,3

  # Re-run the evaluator only (no inference):
  python eval_housecat_official.py \\
      --output-dir outputs/eval_housecat_official_0521 --eval-only
"""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
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
    DEFAULT_OBJECT_VIEWS,
    DEFAULT_RESOLUTION,
    HouseCat6DTestSceneCameraPose,
    build_model,
    rot6d_to_matrix,
)
from omnivggt.datasets.utils.transforms import ImgNorm


PROJECT_ROOT = Path(__file__).resolve().parent
HOUSECAT_VINET = Path("/mnt/train-data-4-hdd/yian/freepose/HouseCat6D/VI-Net")

DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "0521" / "10000" / "model.safetensors"
DEFAULT_HOUSECAT_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose/housecat6d")
DEFAULT_ALIGN_JSON = PROJECT_ROOT / "dataset_align.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eval_housecat_official_0521"
TEST_SCENES = ("test_scene1", "test_scene2", "test_scene3", "test_scene4", "test_scene5")


# ============================================================================
# Inference helpers.
# ============================================================================
def build_dataset(scene: str, args) -> HouseCat6DTestSceneCameraPose:
    object_image_root = (
        Path(args.object_image_root) if getattr(args, "object_image_root", None)
        else args.housecat_root / "housecat6d_aligned_object_refs"
    )
    view_ids = tuple(int(v) for v in getattr(args, "view_ids", None) or DEFAULT_OBJECT_VIEWS)
    return HouseCat6DTestSceneCameraPose(
        dataset_location=str(args.housecat_root),
        dset="test",
        object_image_root=str(object_image_root),
        align_json=str(args.align_json),
        only_scene_name=scene,
        expand_records_by_object=True,
        num_object_views=len(view_ids),
        fixed_object_view_ids=view_ids,
        strict_fixed_object_view_ids=True,
        normalize_object_translation_by_depth_mean=True,
        verify_files=True,
        object_presence_prob=1.0,
        z_far=20,
        resolution=DEFAULT_RESOLUTION,
        transform=ImgNorm,
        seed=42,
    )


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


def run_scene_inference(
    scene: str,
    dataset: HouseCat6DTestSceneCameraPose,
    model,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
    limit: Optional[int],
) -> Dict[int, List[Dict]]:
    """Returns: image_id -> list of per-object pred dicts (one per dataset sample)."""
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=False,
    )
    by_image: Dict[int, List[Dict]] = defaultdict(list)
    pbar = tqdm(loader, desc=f"infer {scene}", dynamic_ncols=True)
    seen = 0
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
        if "object_pose" not in outputs or "object_translation" not in outputs:
            raise RuntimeError(f"Model output missing pose keys: {sorted(outputs.keys())}")
        pred_pose = outputs["object_pose"].detach().float().cpu().numpy()
        pred_trans = outputs["object_translation"].detach().float().cpu().numpy()
        if "object_size" in outputs:
            pred_size_aligned_batch = outputs["object_size"].detach().float().cpu().numpy()
        elif "object_size_log" in outputs:
            pred_size_aligned_batch = np.exp(outputs["object_size_log"].detach().float().cpu().numpy())
        else:
            pred_size_aligned_batch = None

        bs = int(pred_pose.shape[0])
        for i in range(bs):
            image_id = int(np.asarray(_batch_item(batch["ids"], i)).reshape(-1)[0])
            class_id = int(np.asarray(_batch_item(batch["class_id"], i)).reshape(-1)[0])
            object_name = str(_batch_item(batch["object_name"], i))
            bbox = np.asarray(_batch_item(batch["bbox_xyxy"], i), dtype=np.float64).reshape(4)
            r_align = np.asarray(
                _batch_item(batch["R_align_housecat6d_to_ov9d"], i), dtype=np.float64
            ).reshape(3, 3)
            depth_mean = float(np.asarray(_batch_item(batch["depth_mean_scale"], i)).reshape(-1)[0])

            # Predicted R in OV9D-aligned frame -> back to HouseCat6D native frame.
            r_aligned_pred = rot6d_to_matrix(pred_pose[i])
            r_native_pred = r_aligned_pred @ r_align
            t_pred_metric = np.asarray(pred_trans[i], dtype=np.float64).reshape(3) * depth_mean

            if pred_size_aligned_batch is not None:
                size_aligned_pred = np.asarray(pred_size_aligned_batch[i], dtype=np.float64).reshape(3)
                size_native_pred = np.abs(r_align).T @ size_aligned_pred
                size_native_pred = np.clip(size_native_pred, 1e-6, None)
            else:
                size_native_pred = None  # fallback handled at write time

            by_image[image_id].append(
                {
                    "class_id": class_id,
                    "object_name": object_name,
                    "bbox_xyxy": bbox,
                    "pred_R_native": r_native_pred,
                    "pred_t_metric": t_pred_metric,
                    "pred_size_native": size_native_pred,
                }
            )
            seen += 1
        pbar.set_postfix(samples=seen, frames=len(by_image))
        if limit is not None and seen >= limit:
            break
    pbar.close()
    return by_image


# ============================================================================
# Per-frame pkl writing.
# ============================================================================
def load_full_gt_for_frame(scene_dir: Path, image_id: int) -> Dict[str, Any]:
    """Read the original housecat6d label pkl for the frame. Returns full GT lists
    (covers every object, including those without OV9D refs)."""
    label_path = scene_dir / "labels" / f"{image_id:06d}_label.pkl"
    with label_path.open("rb") as h:
        label = pickle.load(h)
    n = len(label["class_ids"])
    class_ids = np.asarray(label["class_ids"], dtype=np.int32)
    bboxes = np.asarray(label["bboxes"], dtype=np.float64).reshape(n, 4)
    gt_RTs = np.tile(np.eye(4, dtype=np.float64)[None], (n, 1, 1))
    gt_scales = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        gt_RTs[i, :3, :3] = np.asarray(label["rotations"][i], dtype=np.float64).reshape(3, 3)
        gt_RTs[i, :3, 3] = np.asarray(label["translations"][i], dtype=np.float64).reshape(3)
        gt_scales[i] = np.asarray(label["gt_scales"][i], dtype=np.float64).reshape(3)
    return {
        "gt_class_ids": class_ids,
        "gt_RTs": gt_RTs,
        "gt_scales": gt_scales,
        "gt_bboxes": bboxes,
        "model_list": [str(x) for x in label["model_list"]],
    }


def write_scene_pkls(
    scene: str,
    by_image: Dict[int, List[Dict]],
    housecat_root: Path,
    out_dir: Path,
) -> int:
    """Combine full GT (from original labels) with collected preds. Returns frame count."""
    scene_dir_src = housecat_root / scene
    scene_dir_dst = out_dir / scene
    scene_dir_dst.mkdir(parents=True, exist_ok=True)

    written = 0
    for image_id, entries in by_image.items():
        if not entries:
            continue
        gt_block = load_full_gt_for_frame(scene_dir_src, image_id)
        k = len(entries)
        pred_class_ids = np.array([e["class_id"] for e in entries], dtype=np.int32)
        pred_RTs = np.tile(np.eye(4, dtype=np.float64)[None], (k, 1, 1))
        pred_scales = np.zeros((k, 3), dtype=np.float64)
        pred_bboxes = np.zeros((k, 4), dtype=np.float64)
        for i, e in enumerate(entries):
            pred_RTs[i, :3, :3] = e["pred_R_native"]
            pred_RTs[i, :3, 3] = e["pred_t_metric"]
            if e["pred_size_native"] is not None:
                pred_scales[i] = e["pred_size_native"]
            else:
                # Fallback: match GT size if model didn't predict it. Looks the
                # GT by object_name when possible (instance level), else by class.
                gt_idx = next(
                    (j for j, m in enumerate(gt_block["model_list"]) if m == e["object_name"]),
                    None,
                )
                if gt_idx is None:
                    gt_idx = next(
                        (j for j in range(len(gt_block["gt_class_ids"])) if int(gt_block["gt_class_ids"][j]) == int(e["class_id"])),
                        0,
                    )
                pred_scales[i] = gt_block["gt_scales"][gt_idx]
            pred_bboxes[i] = e["bbox_xyxy"]

        result = {
            "gt_class_ids": gt_block["gt_class_ids"],
            "gt_RTs": gt_block["gt_RTs"],
            "gt_scales": gt_block["gt_scales"],
            "gt_bboxes": gt_block["gt_bboxes"],
            "pred_class_ids": pred_class_ids,
            "pred_RTs": pred_RTs,
            "pred_scales": pred_scales,
            "pred_bboxes": pred_bboxes,
            "pred_scores": np.ones(k, dtype=np.float32),
        }
        with (scene_dir_dst / f"{image_id:06d}.pkl").open("wb") as h:
            pickle.dump(result, h, protocol=pickle.HIGHEST_PROTOCOL)
        written += 1
    return written


# ============================================================================
# Official evaluator entry.
# ============================================================================
def run_evaluation(out_dir: Path) -> None:
    sys.path.insert(0, str(HOUSECAT_VINET))
    sys.path.insert(0, str(HOUSECAT_VINET / "utils"))
    sys.path.insert(0, str(HOUSECAT_VINET / "lib"))
    from utils.evaluation_utils import evaluate_housecat  # type: ignore

    import logging
    logger = logging.getLogger("housecat_eval")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[housecat_eval] %(message)s"))
        logger.addHandler(h)
    fh = logging.FileHandler(str(out_dir / "eval.log"), mode="w")
    fh.setFormatter(logging.Formatter("[housecat_eval] %(message)s"))
    logger.addHandler(fh)
    print(f"[eval] running evaluate_housecat on {out_dir}")
    evaluate_housecat(str(out_dir), logger=logger)
    logger.removeHandler(fh)
    fh.close()


# ============================================================================
# Orchestrator / worker entry points.
# ============================================================================
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--housecat-root", type=Path, default=DEFAULT_HOUSECAT_ROOT)
    p.add_argument("--align-json", type=Path, default=DEFAULT_ALIGN_JSON)
    p.add_argument("--object-image-root", type=Path, default=None,
                   help="Override reference-view root (e.g. .../housecat6d_aligned_object_refs_diverse24)")
    p.add_argument("--view-ids", nargs="+", type=int, default=None,
                   help="Override fixed_object_view_ids (e.g. 0 5 8 19 for diverse24)")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--scenes", nargs="+", default=list(TEST_SCENES))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="Smoke-test cap on samples per scene.")
    p.add_argument("--gpus", type=str, default=None,
                   help="Comma-separated GPU ids. If multiple, fan out as one subprocess per GPU.")
    p.add_argument("--shard-scenes", type=str, default=None,
                   help="Worker mode: comma-separated subset of scenes to run.")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip inference and only call evaluate_housecat on existing pkls in --output-dir.")
    return p.parse_args(argv)


def orchestrate(args: argparse.Namespace, gpu_ids: List[str]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Round-robin distribute scenes across GPUs.
    scenes_per_gpu: List[List[str]] = [[] for _ in gpu_ids]
    for i, scene in enumerate(args.scenes):
        scenes_per_gpu[i % len(gpu_ids)].append(scene)

    procs: List[subprocess.Popen] = []
    log_paths: List[Path] = []
    for j, (gpu_id, scenes) in enumerate(zip(gpu_ids, scenes_per_gpu)):
        if not scenes:
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = [
            sys.executable, "-u", str(Path(__file__).resolve()),
            "--checkpoint", str(args.checkpoint),
            "--housecat-root", str(args.housecat_root),
            "--align-json", str(args.align_json),
            "--output-dir", str(args.output_dir),
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--shard-scenes", ",".join(scenes),
        ]
        if args.no_amp:
            cmd.append("--no-amp")
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        if args.object_image_root is not None:
            cmd.extend(["--object-image-root", str(args.object_image_root)])
        if args.view_ids is not None:
            cmd.extend(["--view-ids", *(str(v) for v in args.view_ids)])
        log_path = args.output_dir / f"shard_{j:02d}.log"
        log_paths.append(log_path)
        log_fh = open(log_path, "w", encoding="utf-8")
        print(f"[orchestrator] shard {j} GPU={gpu_id} scenes={scenes} log={log_path}")
        procs.append(subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT))

    start = time.time()
    while True:
        time.sleep(20)
        alive = [p for p in procs if p.poll() is None]
        statuses = []
        for j, (p, lp) in enumerate(zip(procs, log_paths)):
            try:
                with lp.open("rb") as h:
                    h.seek(0, os.SEEK_END)
                    sz = h.tell()
                    h.seek(max(0, sz - 600), os.SEEK_SET)
                    tail = h.read().decode("utf-8", errors="replace")
                last = tail.strip().splitlines()[-1] if tail.strip() else ""
                statuses.append(f"  shard {j}: alive={p.poll() is None}  tail={last[-160:]}")
            except FileNotFoundError:
                statuses.append(f"  shard {j}: log missing")
        print(f"[orchestrator] t={time.time() - start:6.0f}s  alive={len(alive)}/{len(procs)}")
        for line in statuses:
            print(line)
        if not alive:
            break
    rc = [p.wait() for p in procs]
    if any(r != 0 for r in rc):
        print(f"[orchestrator] non-zero return codes: {rc}")
    run_evaluation(args.output_dir)


def run_worker(args: argparse.Namespace) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[worker] device={device}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args.checkpoint, device)
    scenes = (
        [s.strip() for s in args.shard_scenes.split(",") if s.strip()]
        if args.shard_scenes
        else list(args.scenes)
    )
    for scene in scenes:
        try:
            dataset = build_dataset(scene, args)
        except Exception as exc:  # noqa: BLE001
            print(f"[worker] {scene} dataset build failed: {exc!r}")
            continue
        by_image = run_scene_inference(
            scene=scene,
            dataset=dataset,
            model=model,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            amp=not args.no_amp,
            limit=args.limit,
        )
        written = write_scene_pkls(scene, by_image, args.housecat_root, args.output_dir)
        print(f"[worker] {scene}: ran {sum(len(v) for v in by_image.values())} samples, "
              f"wrote {written} per-frame pkls into {args.output_dir / scene}")


def main(argv=None) -> None:
    args = parse_args(argv)

    if args.eval_only:
        run_evaluation(args.output_dir)
        return

    if args.gpus and "," in args.gpus and args.shard_scenes is None:
        gpu_ids = [x.strip() for x in args.gpus.split(",") if x.strip()]
        if len(gpu_ids) > 1:
            orchestrate(args, gpu_ids)
            return
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
    elif args.gpus and args.shard_scenes is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)

    run_worker(args)
    # Single-process path also runs the evaluator at the end.
    if args.shard_scenes is None:
        run_evaluation(args.output_dir)


if __name__ == "__main__":
    main()
