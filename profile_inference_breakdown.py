"""Per-component breakdown of OmniVGGT single-view inference time.

Monkey-patches the model's main inference sub-steps with cuda.Event timers and
runs warmup + N timed iterations. Reports mean/median ms per component.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
_LOCAL_TMP = PROJECT_ROOT / "tmp"
_LOCAL_TMP.mkdir(parents=True, exist_ok=True)
(_LOCAL_TMP / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(_LOCAL_TMP))
os.environ.setdefault("MPLCONFIGDIR", str(_LOCAL_TMP / "matplotlib"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import argparse
import functools
import time
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
import torch._dynamo

from demo_gradio_6dpose_real import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_HOUSECAT6D_ROOT,
    DEFAULT_HOUSECAT6D_OBJECT_IMAGE_ROOT,
    DEFAULT_PRETRAIN_MODEL,
    build_model_from_config,
    housecat6d_read_label,
    load_config,
    load_housecat6d_object_tensor,
    load_housecat6d_scene_frame_inputs,
    resolve_runtime_settings,
)
from demo_gif_real import (
    build_housecat6d_object_records,
    enumerate_housecat6d_frames,
    enumerate_housecat6d_objects,
)


class CudaTimer:
    def __init__(self):
        self.totals: Dict[str, List[float]] = defaultdict(list)

    def wrap(self, obj, attr, label):
        orig = getattr(obj, attr)

        @functools.wraps(orig)
        def wrapped(*a, **kw):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = orig(*a, **kw)
            end.record()
            torch.cuda.synchronize()
            self.totals[label].append(start.elapsed_time(end))
            return out

        setattr(obj, attr, wrapped)

    def reset(self):
        self.totals.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_PRETRAIN_MODEL)
    parser.add_argument("--scene", type=Path,
                        default=DEFAULT_HOUSECAT6D_ROOT / "test_scene1")
    parser.add_argument("--object-image-root", type=Path,
                        default=DEFAULT_HOUSECAT6D_OBJECT_IMAGE_ROOT)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--cache-object-encoder", action="store_true",
                        help="Run object encoder once, then patch it to return cached output.")
    parser.add_argument("--skip-mask-head", action="store_true",
                        help="Set object_mask_head=None to skip its forward (unused for pose).")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the scene aggregator (long first-call compile time).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    runtime = resolve_runtime_settings(cfg)
    resolution = tuple(int(v) for v in runtime["resolution"])
    object_views = tuple(int(v) for v in runtime["object_input_views"])

    device = torch.device("cuda")
    print(f"[info] device={device}, ckpt={args.checkpoint}")
    model = build_model_from_config(cfg, args.checkpoint, device)

    hc_records = build_housecat6d_object_records(args.object_image_root, object_views)
    if not hc_records:
        raise SystemExit("No housecat6d aligned object refs")

    frame_ids = enumerate_housecat6d_frames(args.scene)
    objects = enumerate_housecat6d_objects(args.scene, frame_ids)
    if not objects:
        raise SystemExit("No housecat6d objects in scene")
    model_name = next((n for n, _ in objects if n in hc_records), None)
    if model_name is None:
        raise SystemExit("No object overlap between scene and aligned refs")

    object_tensor, _ = load_housecat6d_object_tensor(
        hc_records, model_name, object_views, resolution, device,
    )

    # find first valid frame
    scene_tensor = depth_tensor = mask_tensor = display_depth = None
    chosen_frame = None
    for fid in frame_ids:
        if not (args.scene / "labels" / f"{int(fid):06d}_label.pkl").is_file():
            continue
        label = housecat6d_read_label(args.scene / "labels" / f"{int(fid):06d}_label.pkl")
        if model_name not in [str(n) for n in label.get("model_list", [])]:
            continue
        try:
            t = load_housecat6d_scene_frame_inputs(
                args.scene, int(fid), model_name, resolution, device, target_crop=False,
            )
        except FileNotFoundError:
            continue
        scene_tensor, depth_tensor, mask_tensor, _di, display_depth, _gm, _intr = t
        chosen_frame = fid
        break
    if scene_tensor is None:
        raise SystemExit("No valid frame")

    print(f"[info] using housecat6d/{args.scene.name}/{model_name} frame {int(chosen_frame):06d}")
    print(f"[info] scene tensor: {tuple(scene_tensor.shape)}  object tensor: {tuple(object_tensor.shape)}")

    if args.skip_mask_head:
        model.object_mask_head = None
        print("[info] object_mask_head disabled")

    if args.compile:
        print("[info] torch.compile each transformer block (individually)")
        torch._dynamo.config.suppress_errors = True
        agg = model.aggregator
        for i in range(len(agg.frame_blocks)):
            agg.frame_blocks[i] = torch.compile(agg.frame_blocks[i], mode="default", dynamic=False)
        for i in range(len(agg.global_blocks)):
            agg.global_blocks[i] = torch.compile(agg.global_blocks[i], mode="default", dynamic=False)

    if args.cache_object_encoder:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            _cached = model._encode_object_prototypes(object_tensor)
        torch.cuda.synchronize()

        def _cached_encoder(_object_images):
            return _cached

        model._encode_object_prototypes = _cached_encoder
        print("[info] object encoder cached — will return precomputed prototypes")

    timer = CudaTimer()
    # wrap top-level components
    timer.wrap(model, "_encode_object_prototypes", "object_encoder (aggregator on object_images)")
    timer.wrap(model, "_apply_progressive_object_prototype_cross_attention", "object_cross_attn (per-call, x4 layers)")
    timer.wrap(model.aggregator, "inference", "scene_aggregator")
    if model.object_mask_head is not None:
        timer.wrap(model.object_mask_head, "forward", "object_mask_head")
    if model.object_srt_head is not None:
        timer.wrap(model.object_srt_head, "forward", "object_srt_head (object_pose)")

    def run_once():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return model.inference(
                images=scene_tensor,
                object_images=object_tensor,
                extrinsics=None,
                intrinsics=None,
                depth=depth_tensor,
                mask=mask_tensor,
                camera_gt_index=[],
                depth_gt_index=[0],
            )

    print(f"[info] warmup x{args.warmup}")
    for _ in range(args.warmup):
        run_once()
    torch.cuda.synchronize()
    timer.reset()

    print(f"[info] timed x{args.runs}")
    total_ms: List[float] = []
    for _ in range(args.runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = run_once()
        # also force the same .cpu() the demo does (for fair comparison)
        outputs["object_pose"][0].detach().float().cpu().numpy()
        outputs["object_translation"][0].detach().float().cpu().numpy()
        torch.cuda.synchronize()
        total_ms.append((time.perf_counter() - t0) * 1000.0)

    total_arr = np.asarray(total_ms)
    print(f"\n[total per-iter, ms over {args.runs}] "
          f"mean={total_arr.mean():.2f}  median={np.median(total_arr):.2f}  "
          f"min={total_arr.min():.2f}  max={total_arr.max():.2f}")

    print("\n[per-component breakdown, ms per inference iteration]")
    print(f"{'component':<55} {'mean':>8} {'median':>8} {'%total':>8} {'calls/iter':>10}")
    rows = []
    for label, vals in timer.totals.items():
        arr = np.asarray(vals)
        calls_per_iter = len(vals) / args.runs
        # mean ms per iteration = sum / runs
        per_iter_ms = arr.sum() / args.runs
        rows.append((label, per_iter_ms, np.median(arr) * calls_per_iter, arr.mean(), calls_per_iter))
    rows.sort(key=lambda r: -r[1])
    for label, per_iter_ms, per_iter_median, per_call_mean, calls in rows:
        pct = per_iter_ms / total_arr.mean() * 100.0
        print(f"  {label:<53} {per_iter_ms:8.2f} {per_iter_median:8.2f} {pct:7.1f}% {calls:10.1f}")

    accounted = sum(r[1] for r in rows)
    other = total_arr.mean() - accounted
    print(f"  {'<other / unaccounted>':<53} {other:8.2f} {'':>8} {other/total_arr.mean()*100:7.1f}%")


if __name__ == "__main__":
    main()
