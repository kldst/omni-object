"""Inference-only eval for the 0511 model on splits_multi_4_3000/test2.json.

Differences vs eval_ov9d_unseen.py:
  - Uses configs/train_ov9d_camera_pose.py settings (resolution 518x518,
    fixed_object_view_ids (10, 20, 30, 40)).
  - Object reference views come from <ov9d_root>/oo3d9dsingle/<scene>/rgb,
    masked with mask_visib/<view>_000000.png and a white background — matching
    how the 0511 checkpoint was trained.
  - Only evaluates the multi-scene split (test2.json).

    python eval_test2_multi.py --batch-size 16 --gpus 0,1,2,3
"""

import argparse
import json
import math
import os
import re
import runpy
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file as load_safetensors_file
from tqdm import tqdm

# Reuse all helpers from the previous eval script so we keep behaviour aligned.
from eval_ov9d_unseen import (
    EvalScenePreprocessor,
    InferenceWrap,
    batched_forward,
    build_model,
    bucket_by_distance,
    aggregate,
    decode_sample,
    crop_resize_image_depth_mask,
    format_metric_block,
    iter_multi_targets,
    load_config,
    load_scene_frame_inputs_cpu,
    ov9d_object_id_from_key,
    ov9d_read_binary_mask,
    ov9d_read_depth_m,
    read_json,
    report,
    resolve_local_path,
    resolve_scene_dir,
    rot6d_to_matrix,
    rotation_error_degrees,
    shard_iter,
    symmetric_rot6d_l1,
    symmetric_rotation_error_degrees,
    _self_test_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train_ov9d_camera_pose.py"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "0511" / "model.safetensors"
DEFAULT_DATASET_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d")
DEFAULT_TEST_JSON = PROJECT_ROOT / "splits_multi_4_3000" / "test2.json"
DEFAULT_OBJECT_VIEWS = (10, 20, 30, 40)
DEFAULT_RESOLUTION = (518, 518)


# ============================================================
# Object reference index built from oo3d9dsingle/ + name2oid.json
# (NOT from ov9d_around_image)
# ============================================================
def build_single_scene_object_index(
    single_root: Path,
    name_to_oid_path: Path,
    required_views: Sequence[int],
) -> Dict[int, Path]:
    """Map object_id -> path to a single-scene folder under oo3d9dsingle/.

    Mirrors OV9DCameraPose._build_single_records_by_object_id, but only keeps
    one scene per object_id (the first that has the required fixed views and
    matching masks). The returned dir is used to load RGB+mask reference views.
    """
    if not single_root.is_dir():
        raise FileNotFoundError(f"oo3d9dsingle root not found: {single_root}")
    if not name_to_oid_path.is_file():
        raise FileNotFoundError(f"name2oid.json not found: {name_to_oid_path}")
    name_to_oid = {str(k): int(v) for k, v in read_json(name_to_oid_path).items()}

    index: Dict[int, Path] = {}
    for scene_dir in sorted(p for p in single_root.iterdir() if p.is_dir()):
        name_parts = scene_dir.name.split("_")
        # The object instance is everything except the trailing scene hash.
        object_instance = "_".join(name_parts[:-1]) if len(name_parts) > 2 else scene_dir.name
        object_id = name_to_oid.get(object_instance)
        if object_id is None:
            continue
        if int(object_id) in index:
            continue  # keep the first matching single scene (sorted order)
        rgb_dir = scene_dir / "rgb"
        mask_dir = scene_dir / "mask_visib"
        if not rgb_dir.is_dir() or not mask_dir.is_dir():
            continue
        ok = True
        for view_id in required_views:
            if not (rgb_dir / f"{int(view_id):06d}.png").is_file():
                ok = False
                break
            if not (mask_dir / f"{int(view_id):06d}_000000.png").is_file():
                ok = False
                break
        if not ok:
            continue
        index[int(object_id)] = scene_dir
    return index


# ============================================================
# Object reference loading: oo3d9dsingle RGB × mask, white background
# ============================================================
def load_object_tensor_cpu_singlescene(
    object_ref_dir: Path,
    object_views: Sequence[int],
    resolution,
) -> torch.Tensor:
    processor = EvalScenePreprocessor(resolution=resolution)
    resampling = getattr(Image, "Resampling", Image)
    tensors: List[torch.Tensor] = []
    for view_id in object_views:
        image_path = object_ref_dir / "rgb" / f"{int(view_id):06d}.png"
        mask_path = object_ref_dir / "mask_visib" / f"{int(view_id):06d}_000000.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing object reference view: {image_path}")
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing object reference mask: {mask_path}")
        rgb_arr = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        mask_arr = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        white_bg = np.full_like(rgb_arr, 255)
        white_bg[mask_arr > 0] = rgb_arr[mask_arr > 0]
        image = Image.fromarray(white_bg, mode="RGB").resize(tuple(resolution), resampling.LANCZOS)
        tensors.append(processor.transform(image))
    return torch.stack(tensors, dim=0)  # (K, 3, H, W)


# ============================================================
# prepare_sample variant that uses the single-scene object loader
# ============================================================
def prepare_sample(
    scene_dir: Path,
    image_id: int,
    object_id: int,
    object_ref_dir: Path,
    object_views: Sequence[int],
    resolution,
    *,
    scene_gt_cache: Dict[Path, Dict],
    object_tensor_cache: Dict[int, torch.Tensor],
) -> Dict | None:
    if scene_dir in scene_gt_cache:
        scene_gt = scene_gt_cache[scene_dir]
    else:
        scene_gt = read_json(scene_dir / "scene_gt.json")
        scene_gt_cache[scene_dir] = scene_gt
    gts = scene_gt.get(str(image_id), [])
    object_index = next(
        (idx for idx, gt in enumerate(gts) if int(gt.get("obj_id", -1)) == int(object_id)),
        None,
    )
    if object_index is None:
        return None

    image_tensor, depth_tensor, mask_tensor = load_scene_frame_inputs_cpu(
        scene_dir, image_id, object_index, resolution,
    )
    if int(object_id) in object_tensor_cache:
        object_tensor = object_tensor_cache[int(object_id)]
    else:
        object_tensor = load_object_tensor_cpu_singlescene(object_ref_dir, object_views, resolution)
        object_tensor_cache[int(object_id)] = object_tensor

    gt = gts[object_index]
    gt_rotation_cam = np.asarray(gt["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
    gt_translation_cam = np.asarray(gt["cam_t_m2c"], dtype=np.float32).reshape(3) / 1000.0
    return {
        "image": image_tensor,
        "depth": depth_tensor,
        "mask": mask_tensor,
        "object": object_tensor,
        "object_id": int(object_id),
        "gt_rotation_cam": gt_rotation_cam,
        "gt_translation_cam": gt_translation_cam,
    }


# ============================================================
# Orchestrator: fan out shards across GPUs as subprocesses
# ============================================================
def orchestrate_shards(args: argparse.Namespace, gpu_ids: List[str]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    num_shards = len(gpu_ids)
    print(f"[orchestrator] launching {num_shards} workers, one per GPU: {gpu_ids}")
    print(f"[orchestrator] batch_size={args.batch_size}  output_dir={args.output_dir}")

    base_cmd = [
        sys.executable, "-u", str(Path(__file__).resolve()),
        "--config", str(args.config),
        "--checkpoint", str(args.checkpoint),
        "--dataset-root", str(args.dataset_root),
        "--test-json", str(args.test_json),
        "--output-dir", str(args.output_dir),
        "--batch-size", str(args.batch_size),
        "--num-shards", str(num_shards),
        "--skip-self-test",
    ]
    if args.no_depth:
        base_cmd.append("--no-depth")
    if args.limit is not None:
        base_cmd.extend(["--limit", str(args.limit)])

    procs: List[Tuple[int, subprocess.Popen]] = []
    for shard_index, gpu_id in enumerate(gpu_ids):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        log_path = args.output_dir / f"shard_{shard_index:02d}.log"
        cmd = base_cmd + ["--shard-index", str(shard_index)]
        log_fh = open(log_path, "w", encoding="utf-8")
        print(f"[orchestrator] shard {shard_index} → CUDA_VISIBLE_DEVICES={gpu_id}  log={log_path}")
        procs.append((shard_index, subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)))

    start = time.time()
    while True:
        time.sleep(15)
        alive = [shard for shard, p in procs if p.poll() is None]
        statuses = []
        for shard_index, p in procs:
            log_path = args.output_dir / f"shard_{shard_index:02d}.log"
            try:
                with log_path.open("r", encoding="utf-8") as handle:
                    handle.seek(0, os.SEEK_END)
                    size = handle.tell()
                    handle.seek(max(0, size - 600), os.SEEK_SET)
                    tail = handle.read()
                last_line = tail.strip().splitlines()[-1] if tail.strip() else ""
                statuses.append(f"  shard {shard_index}: alive={p.poll() is None}  last={last_line[-200:]}")
            except FileNotFoundError:
                statuses.append(f"  shard {shard_index}: log missing")
        elapsed = time.time() - start
        print(f"[orchestrator] t={elapsed:7.0f}s  alive={len(alive)}/{len(procs)}")
        for line in statuses:
            print(line)
        if not alive:
            break

    return_codes = [p.wait() for _, p in procs]
    if any(rc != 0 for rc in return_codes):
        print(f"[orchestrator] workers returned: {return_codes} (non-zero indicates failure)")

    all_samples: List[Dict] = []
    for shard_index, _ in procs:
        shard_path = args.output_dir / f"samples_shard_{shard_index:02d}.jsonl"
        if not shard_path.is_file():
            print(f"[orchestrator] missing shard file: {shard_path}")
            continue
        with shard_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec.pop("split", None)
                all_samples.append(rec)
    print(f"[orchestrator] merged samples: total={len(all_samples)}")

    test2_report = report("test2", all_samples)

    summary = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "test_json": str(args.test_json),
        "use_depth": not args.no_depth,
        "shards": [
            {"shard_index": i, "gpu": gpu_ids[i], "return_code": return_codes[i]}
            for i in range(len(procs))
        ],
        "test2": test2_report,
    }
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\n[orchestrator] wrote summary to {summary_path}")

    samples_path = args.output_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for s in all_samples:
            handle.write(json.dumps({"split": "test2", **s}) + "\n")
    print(f"[orchestrator] wrote merged per-sample records to {samples_path}")


# ============================================================
# Entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Inference-only eval for 0511 on test2.json")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--test-json", type=Path, default=DEFAULT_TEST_JSON)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "eval_0511_test2")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-depth", action="store_true",
                        help="Drop depth input (default uses depth, matching training).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Per-shard sample cap (smoke test).")
    parser.add_argument("--gpu", type=str, default=None,
                        help="CUDA_VISIBLE_DEVICES override.")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU ids to fan out across (e.g. '0,1,2,3').")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--self-test-only", action="store_true")
    parser.add_argument("--skip-self-test", action="store_true")
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    if not args.skip_self_test:
        _self_test_metrics()
    if args.self_test_only:
        return

    is_worker = args.shard_index is not None
    if not is_worker and args.gpus is not None and "," in str(args.gpus):
        gpu_ids = [x.strip() for x in str(args.gpus).split(",") if x.strip() != ""]
        if len(gpu_ids) > 1:
            return orchestrate_shards(args, gpu_ids)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.device:
        primary_device = torch.device(args.device)
    elif torch.cuda.is_available():
        primary_device = torch.device("cuda:0")
    else:
        primary_device = torch.device("cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if primary_device.type == "cuda":
        if args.gpus is not None:
            device_ids = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
        else:
            device_ids = list(range(torch.cuda.device_count()))
        if not device_ids:
            device_ids = [0]
        primary_device = torch.device(f"cuda:{device_ids[0]}")
    else:
        device_ids = []

    cfg = load_config(args.config)
    resolution = tuple(int(v) for v in cfg.get("resolution", DEFAULT_RESOLUTION))
    object_views = tuple(int(v) for v in cfg.get("fixed_object_view_ids", DEFAULT_OBJECT_VIEWS))
    sym_path = resolve_local_path(cfg.get("object_srt_symmetry_info_path", ""))
    sym_path_str = str(sym_path) if sym_path is not None else ""
    sym_steps = int(cfg.get("object_srt_symmetry_continuous_steps", 72))

    print(f"[eval] config={args.config}")
    print(f"[eval] checkpoint={args.checkpoint}")
    print(f"[eval] dataset_root={args.dataset_root}")
    print(f"[eval] resolution={resolution}  object_views={object_views}")
    print(f"[eval] symmetry_info={sym_path_str} steps={sym_steps}")
    print(f"[eval] use_depth={not args.no_depth}")
    print(f"[eval] device(primary)={primary_device} data_parallel_ids={device_ids} batch_size={args.batch_size}")

    model = build_model(cfg, args.checkpoint, primary_device)
    inference_module = InferenceWrap(model, use_depth=not args.no_depth).to(primary_device).eval()

    # Object reference index built from oo3d9dsingle/ (NOT ov9d_around_image).
    single_root = args.dataset_root / "oo3d9dsingle"
    name2oid_path = args.dataset_root / "name2oid.json"
    object_reference_index = build_single_scene_object_index(single_root, name2oid_path, object_views)
    print(f"[eval] single-scene object refs (with all required views): {len(object_reference_index)}")

    # test2 eligible_object_ids 已經是同類別未見過的物體,直接用,不再 filter against train.json
    train_object_ids: set = set()

    test_payload = read_json(args.test_json)
    print(f"[eval] test2 scenes: {len(test_payload.get('scenes', []))}")

    eligible_total = 0
    kept_total = 0
    kept_objects = set()
    missing_ref_objects = set()
    for sc in test_payload.get("scenes", []):
        for oid in sc.get("eligible_object_ids", []) or []:
            oid = int(oid)
            eligible_total += 1
            if oid in object_reference_index:
                kept_total += 1
                kept_objects.add(oid)
            else:
                missing_ref_objects.add(oid)
    print(
        f"[eval] eligible occurrences: total={eligible_total} kept={kept_total} "
        f"kept_unique_objects={len(kept_objects)} missing_reference={len(missing_ref_objects)}"
    )

    shard_index = int(args.shard_index) if args.shard_index is not None else 0
    num_shards = int(args.num_shards) if args.num_shards else 1
    if num_shards > 1:
        print(f"[eval] worker shard: index={shard_index} / total={num_shards}")

    targets_iter = shard_iter(
        iter_multi_targets(test_payload, args.dataset_root, train_object_ids, object_reference_index),
        shard_index, num_shards,
    )

    samples: List[Dict] = []
    skipped = 0
    scene_gt_cache: Dict[Path, Dict] = {}
    object_tensor_cache: Dict[int, torch.Tensor] = {}

    def flush(batch: List[Dict]):
        if not batch:
            return
        preds = batched_forward(inference_module, batch, primary_device, use_depth=not args.no_depth)
        for meta, pred in zip(batch, preds):
            decoded = decode_sample(
                pred_rot6d_cam=pred["pred_rot6d_cam"],
                pred_translation_cam=pred["pred_translation_cam"],
                presence_logit=pred["presence_logit"],
                object_id=meta["object_id"],
                gt_rotation_cam=meta["gt_rotation_cam"],
                gt_translation_cam=meta["gt_translation_cam"],
                symmetry_info_path=sym_path_str,
                symmetry_continuous_steps=sym_steps,
            )
            decoded.update({"scene_name": meta["scene_name"], "frame_id": meta["frame_id"]})
            samples.append(decoded)

    batch: List[Dict] = []
    pbar = tqdm(targets_iter, desc="test2", unit="sample", smoothing=0.02)
    prev_scene_dir: Path | None = None
    for (scene_name, scene_dir, frame_id, object_id, ref_dir) in pbar:
        if prev_scene_dir is not None and prev_scene_dir != scene_dir:
            scene_gt_cache.pop(prev_scene_dir, None)
        prev_scene_dir = scene_dir
        try:
            prepared = prepare_sample(
                scene_dir=scene_dir,
                image_id=frame_id,
                object_id=object_id,
                object_ref_dir=ref_dir,
                object_views=object_views,
                resolution=resolution,
                scene_gt_cache=scene_gt_cache,
                object_tensor_cache=object_tensor_cache,
            )
        except FileNotFoundError as exc:
            pbar.write(f"  [skip] {scene_name} frame={frame_id} obj={object_id}: {exc}")
            skipped += 1
            continue
        if prepared is None:
            continue
        prepared.update({"scene_name": scene_name, "frame_id": frame_id})
        batch.append(prepared)

        if len(batch) >= args.batch_size:
            flush(batch)
            batch = []
            if args.limit is not None and len(samples) >= args.limit:
                break

        if len(object_tensor_cache) > 64:
            for k in list(object_tensor_cache.keys())[:32]:
                object_tensor_cache.pop(k, None)

    if batch and (args.limit is None or len(samples) < args.limit):
        flush(batch)

    if args.limit is not None and len(samples) > args.limit:
        samples = samples[: args.limit]

    print(f"[eval] test2: collected {len(samples)} samples, skipped {skipped}")

    if is_worker:
        shard_path = args.output_dir / f"samples_shard_{shard_index:02d}.jsonl"
        with shard_path.open("w", encoding="utf-8") as handle:
            for s in samples:
                handle.write(json.dumps({"split": "test2", **s}) + "\n")
        print(f"[eval] shard {shard_index} wrote {shard_path}")
        return

    test_report = report("test2", samples)
    summary = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "test_json": str(args.test_json),
        "use_depth": not args.no_depth,
        "resolution": list(resolution),
        "object_views": list(object_views),
        "symmetry_info_path": sym_path_str,
        "filter": {"eligible_total": eligible_total, "kept": kept_total},
        "test2": test_report,
    }
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[eval] wrote summary to {summary_path}")

    samples_path = args.output_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for s in samples:
            handle.write(json.dumps({"split": "test2", **s}) + "\n")
    print(f"[eval] wrote per-sample records to {samples_path}")


if __name__ == "__main__":
    main()
