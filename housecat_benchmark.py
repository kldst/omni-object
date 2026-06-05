"""In-process HouseCat6D official benchmark, callable from the training loop.

Reuses the inference + pkl-writing helpers from ``eval_housecat_official.py`` but,
unlike that script's ``run_evaluation`` (which only prints), ``evaluate_and_collect``
captures the mAP arrays returned by VI-Net's ``compute_independent_mAP`` and returns
the headline numbers as a flat dict, so they can be logged to wandb/tensorboard
during validation.

Used by ``train_omnivggt.run_validation`` when ``validation_mode == "benchmark"``.
"""

from __future__ import annotations

import glob
import logging
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from eval_housecat_official import (
    HOUSECAT_VINET,
    build_dataset,
    run_scene_inference,
    write_scene_pkls,
)

# Official HouseCat6D synset names (must match VI-Net's evaluate_housecat).
SYNSET_NAMES = [
    "BG", "box", "bottle", "can", "cup", "remote",
    "teapot", "cutlery", "glass", "shoe", "tube",
]

DEGREE_THRESHOLDS = [5, 10]
SHIFT_THRESHOLDS = [2, 5, 10]
IOU_3D_THRESHOLDS = [0.10, 0.25, 0.50, 0.75]


def evaluate_and_collect(
    out_dir: Path,
    scenes: Sequence[str],
    degree_thresholds: Sequence[int] = DEGREE_THRESHOLDS,
    shift_thresholds: Sequence[int] = SHIFT_THRESHOLDS,
    iou_3d_thresholds: Sequence[float] = IOU_3D_THRESHOLDS,
) -> Dict[str, float]:
    """Run VI-Net's compute_independent_mAP over the written pkls and return the
    overall (mean-over-classes) headline metrics as a flat dict."""
    # Prefer the vendored copy that ships inside this repo (housecat_eval_utils.py),
    # so the benchmark works on a deployment where only omni-object_clone is pushed
    # (the external HouseCat6D/VI-Net checkout is absent there). Fall back to VI-Net's
    # utils.evaluation_utils when running from the full local tree.
    try:
        from housecat_eval_utils import compute_independent_mAP  # type: ignore
    except ImportError:
        for p in (str(HOUSECAT_VINET), str(HOUSECAT_VINET / "utils"), str(HOUSECAT_VINET / "lib")):
            if p not in sys.path:
                sys.path.insert(0, p)
        from utils.evaluation_utils import compute_independent_mAP  # type: ignore

    pkl_list: List[str] = []
    for scene in scenes:
        pkl_list.extend(glob.glob(os.path.join(str(out_dir), scene, "*.pkl")))
    pkl_list = sorted(pkl_list)
    if not pkl_list:
        return {"num_frames": 0}

    final_results = []
    for pkl_path in pkl_list:
        with open(pkl_path, "rb") as handle:
            result = pickle.load(handle)
        result["gt_handle_visibility"] = np.ones_like(result["gt_class_ids"])
        final_results.append(result)

    # compute_independent_mAP calls logger.warning/info unconditionally, so a
    # real logger is required (passing None raises AttributeError).
    bench_logger = logging.getLogger("housecat_benchmark.mAP")
    if not bench_logger.handlers:
        bench_logger.addHandler(logging.NullHandler())
    iou_3d_aps, pose_aps = compute_independent_mAP(
        final_results,
        SYNSET_NAMES,
        degree_thresholds=list(degree_thresholds),
        shift_thresholds=list(shift_thresholds),
        iou_3d_thresholds=list(iou_3d_thresholds),
        logger=bench_logger,
    )

    # Index lists exactly as compute_independent_mAP builds them internally.
    degree_list = list(degree_thresholds) + [360]
    shift_list = list(shift_thresholds) + [100]
    iou_list = list(iou_3d_thresholds)
    overall = -1  # last row = mean over foreground classes

    def iou(th):
        return float(iou_3d_aps[overall, iou_list.index(th)] * 100.0)

    def pose(deg, cm):
        return float(pose_aps[overall, degree_list.index(deg), shift_list.index(cm)] * 100.0)

    return {
        "num_frames": len(pkl_list),
        "iou_25": iou(0.25),
        "iou_50": iou(0.50),
        "iou_75": iou(0.75),
        "pose_5deg_2cm": pose(5, 2),
        "pose_5deg_5cm": pose(5, 5),
        "pose_10deg_2cm": pose(10, 2),
        "pose_10deg_5cm": pose(10, 5),
    }


def run_benchmark_inference(
    model,
    device: torch.device,
    *,
    housecat_root: str,
    align_json: str,
    object_image_root: str,
    view_ids: Sequence[int],
    scenes: Sequence[str],
    output_dir: Path,
    frame_stride: int = 1,
    batch_size: int = 8,
    num_workers: int = 4,
    amp: bool = True,
    limit: Optional[int] = None,
    object_ref_color_jitter: bool = False,
    shard_id: int = 0,
    num_shards: int = 1,
) -> int:
    """Run inference on the given scenes and write per-frame pkls into
    ``output_dir/<scene>/``. Does NOT evaluate. Returns total frames written.

    Frame-level sharding (load balanced): every rank is given the SAME ``scenes``
    list, but processes only the interleaved frame slice ``records[shard_id::num_shards]``
    of each scene. This spreads ~equal frame counts across ranks regardless of how
    big each scene is (whole-scene sharding left the rank with 2 scenes running ~2x
    longer). Every frame is still covered exactly once across ranks, and per-frame
    pkls never collide because frame ids are disjoint between shards.
    ``model`` must be the unwrapped module on ``device`` in eval mode.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = SimpleNamespace(
        housecat_root=Path(housecat_root),
        align_json=Path(align_json),
        object_image_root=str(object_image_root),
        view_ids=list(int(v) for v in view_ids),
        frame_stride=int(frame_stride),
        object_ref_color_jitter=bool(object_ref_color_jitter),
    )

    total_frames = 0
    for scene in scenes:
        dataset = build_dataset(scene, args)
        if int(num_shards) > 1 and hasattr(dataset, "records"):
            # CRITICAL: records are per-OBJECT here (expand_records_by_object=True),
            # so we must shard by FRAME (image_id), NOT by record. write_scene_pkls
            # writes one <image_id>.pkl per frame to a shared dir; if two shards each
            # hold some objects of the same frame, their pkls overwrite each other and
            # all-but-one shard's predictions for that frame are lost -> recall (and
            # hence IoU/pose mAP) collapses. Sharding whole frames keeps every frame's
            # objects together in exactly one shard, written exactly once.
            frame_ids = sorted({int(r.get("image_id", 0)) for r in dataset.records})
            my_frames = set(frame_ids[int(shard_id)::int(num_shards)])
            dataset.records = [r for r in dataset.records if int(r.get("image_id", 0)) in my_frames]
            dataset.scenes = dataset.records
            if len(dataset.records) == 0:
                continue  # this rank drew no frames from this scene
        by_image = run_scene_inference(
            scene=scene,
            dataset=dataset,
            model=model,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            amp=amp,
            limit=limit,
        )
        total_frames += write_scene_pkls(scene, by_image, Path(housecat_root), output_dir)
    return total_frames


def run_housecat_benchmark(
    model,
    device: torch.device,
    *,
    housecat_root: str,
    align_json: str,
    object_image_root: str,
    view_ids: Sequence[int],
    scenes: Sequence[str],
    output_dir: Path,
    frame_stride: int = 1,
    batch_size: int = 8,
    num_workers: int = 4,
    amp: bool = True,
    limit: Optional[int] = None,
) -> Dict[str, float]:
    """Single-process convenience: run inference on ALL scenes then evaluate.

    For multi-GPU in-training use, call ``run_benchmark_inference`` per rank with a
    scene shard + a barrier, then ``evaluate_and_collect`` on rank 0 instead.
    """
    run_benchmark_inference(
        model, device,
        housecat_root=housecat_root, align_json=align_json,
        object_image_root=object_image_root, view_ids=view_ids,
        scenes=scenes, output_dir=output_dir, frame_stride=frame_stride,
        batch_size=batch_size, num_workers=num_workers, amp=amp, limit=limit,
    )
    return evaluate_and_collect(Path(output_dir), scenes)
