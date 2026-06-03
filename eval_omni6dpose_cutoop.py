"""Evaluate a checkpoint on Omni6DPose SOPE with the OFFICIAL cutoop metrics.

Companion to ``eval_omni6dpose.py`` (which reports simple deg/cm accuracy on ROPE).
This script runs the same inference pipeline but feeds predictions+GT into
``cutoop.eval_utils.DetectMatch`` (Omni6DPoseAPI) to report the official benchmark
metrics: 3D IoU acc/mAP/AUC, pose acc (deg-cm) / AUC / VUS, mean rotation &
translation error, per class and class-mean.

Inference convention (identical to demo_gradio_6dpose_omni6dpose.py):
  - object_pose (symmetric_rot6d) -> 3x3 rotation (R_align = I -> object->camera)
  - object_translation is depth-mean normalized -> multiply by GT depth.mean()
  - object_size (or exp(object_size_log)) -> bbox side lengths

Symmetry: the canonical per-oid source Meta/obj_meta.json is a broken HTML download
on this machine, so we fall back to a class-level symmetry map built from the VALID
Meta/real_obj_meta.json. Override with --sym-meta <valid obj_meta.json>, or disable
with --no-symmetry.

Example:
  python eval_omni6dpose_cutoop.py --split test --patches 00 01 02 04 05 --gpus 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "0603_omni6dpose" / "6000" / "model.safetensors"
DEFAULT_OMNI6DPOSE = Path("/mnt/train-data-4-hdd/yian/freepose/Omni6dpose/Omni6DPoseAPI/data/Omni6DPose")
OMNI6DPOSE_API = Path("/mnt/train-data-4-hdd/yian/freepose/Omni6dpose/Omni6DPoseAPI")
DEFAULT_RESOLUTION = (518, 476)
DEFAULT_VIEW_IDS = (0, 5, 8, 19)
IOU_THRESHOLDS = [0.25, 0.50, 0.75]
POSE_THRESHOLDS = [(5, 2), (5, 5), (10, 2), (10, 5)]


def _batch_item(value, i: int):
    if torch.is_tensor(value):
        return value[i].detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value[i]
    if isinstance(value, (list, tuple)):
        return value[i]
    return value


def affine(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    A = np.eye(4, dtype=np.float64)
    A[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    A[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return A


def build_symmetry_resolver(meta_root: Path, sym_meta: Path | None, no_symmetry: bool):
    """Return (resolver(oid, class_name) -> SymLabel, description)."""
    from common.rotation import SymLabel
    from common.obj_meta import ObjectMetaData

    no_sym = SymLabel(any=False, x="none", y="none", z="none")
    if no_symmetry:
        return (lambda oid, cn: no_sym), "disabled (all 'none')"

    candidate = Path(sym_meta) if sym_meta else (meta_root / "obj_meta.json")
    if candidate.is_file():
        try:
            inst = ObjectMetaData.load_json(str(candidate)).instance_dict
            if inst:
                return (lambda oid, cn: inst[oid].tag.symmetry if oid in inst else no_sym,
                        f"per-oid from {candidate.name}")
        except Exception as exc:
            print(f"[eval] {candidate.name} unusable ({exc!r}); trying real_obj_meta class map.")

    real_meta = meta_root / "real_obj_meta.json"
    if real_meta.is_file():
        try:
            inst = ObjectMetaData.load_json(str(real_meta)).instance_dict
            class_to_sym: Dict[str, Any] = {}
            for info in inst.values():
                cn = getattr(info, "class_name", "") or ""
                if cn and (cn not in class_to_sym or class_to_sym[cn] == no_sym):
                    class_to_sym[cn] = info.tag.symmetry
            return (lambda oid, cn: class_to_sym.get(cn, no_sym),
                    f"class-level from real_obj_meta.json ({len(class_to_sym)} classes)")
        except Exception as exc:
            print(f"[eval] real_obj_meta.json unusable ({exc!r}); using no symmetry.")

    return (lambda oid, cn: no_sym), "unavailable (all 'none')"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--omni6dpose-root", type=Path, default=DEFAULT_OMNI6DPOSE)
    ap.add_argument("--split", default="test")
    ap.add_argument("--patches", nargs="+", default=["00", "01", "02", "04", "05"])
    ap.add_argument("--scene", default="", help="restrict to one scene, e.g. '00/test/ikea/0096'")
    ap.add_argument("--view-ids", nargs="+", type=int, default=list(DEFAULT_VIEW_IDS))
    ap.add_argument("--use-depth-input", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use-depth-scale", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--frame-stride", type=int, default=1, help="evaluate every Nth frame id")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--z-far", type=float, default=20.0)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--sym-meta", type=Path, default=None)
    ap.add_argument("--no-symmetry", action="store_true")
    ap.add_argument("--gpus", type=str, default=None)
    ap.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "eval_omni6dpose_cutoop")
    # multi-GPU sharding: each process handles records[shard_id::num_shards] and
    # dumps raw predictions; a final --merge pass concatenates them for the metrics.
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--save-predictions", type=Path, default=None,
                    help="dump raw (gt/pred) arrays to this pkl instead of computing metrics")
    ap.add_argument("--resume", action="store_true",
                    help="with --save-predictions: resume from the partial pkl if present")
    ap.add_argument("--checkpoint-every", type=int, default=100,
                    help="save partial predictions every N batches (for crash-safe resume)")
    ap.add_argument("--merge", type=Path, default=None,
                    help="directory of shard .pkl files -> concat and compute metrics, then exit")
    ap.add_argument("--max-trans-cm", type=float, default=None,
                    help="merge: drop objects whose translation error exceeds this (cm) before metrics")
    args = ap.parse_args(argv)

    # ---- merge mode: combine shard predictions and compute metrics ----
    if args.merge is not None:
        import pickle
        if str(OMNI6DPOSE_API) not in sys.path:
            sys.path.insert(0, str(OMNI6DPOSE_API))
        from common.eval_utils import DetectMatch
        shards = sorted(Path(args.merge).glob("shard_*.pkl"))
        if not shards:
            raise SystemExit(f"no shard_*.pkl under {args.merge}")
        acc = {k: [] for k in ("gt_aff", "gt_sz", "gt_sym", "gt_cls", "pr_aff", "pr_sz")}
        meta0 = None
        for sp in shards:
            with open(sp, "rb") as fh:
                d = pickle.load(fh)
            for k in acc:
                acc[k].extend(d[k])
            meta0 = meta0 or d.get("meta", {})
            print(f"[merge] {sp.name}: +{len(d['gt_aff'])}")
        gt_aff = np.stack(acc["gt_aff"]); pr_aff = np.stack(acc["pr_aff"])
        gt_sz = np.stack(acc["gt_sz"]); pr_sz = np.stack(acc["pr_sz"])
        gt_sym = np.array(acc["gt_sym"], dtype=object); gt_cls = np.array(acc["gt_cls"])
        total = len(gt_aff)
        out_dir = Path(args.merge)
        if args.max_trans_cm is not None:
            trans_cm = np.linalg.norm(gt_aff[:, :3, 3] - pr_aff[:, :3, 3], axis=1) * 100.0
            keep = trans_cm <= float(args.max_trans_cm)
            print(f"[merge] translation-error filter <= {args.max_trans_cm}cm: "
                  f"kept {int(keep.sum())}/{total} ({100*keep.mean():.1f}%), dropped {int((~keep).sum())}")
            gt_aff, pr_aff, gt_sz, pr_sz, gt_sym, gt_cls = (
                gt_aff[keep], pr_aff[keep], gt_sz[keep], pr_sz[keep], gt_sym[keep], gt_cls[keep])
            (meta0 := meta0 or {})["trans_filter_cm"] = args.max_trans_cm
            (meta0 := meta0 or {})["kept_fraction"] = float(keep.mean())
            out_dir = Path(args.merge) / f"filtered_trans{int(args.max_trans_cm)}cm"
            out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[merge] {len(gt_aff)} predictions; computing metrics ...")
        dm = DetectMatch(
            gt_affine=gt_aff, gt_size=gt_sz, gt_sym_labels=gt_sym, gt_class_labels=gt_cls,
            pred_affine=pr_aff, pred_size=pr_sz,
        ).calibrate_rotation()
        _report_metrics(dm.metrics(iou_thresholds=IOU_THRESHOLDS, pose_thresholds=POSE_THRESHOLDS),
                        out_dir, meta0 or {}, len(gt_aff))
        return

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus).split(",")[0]
    if str(OMNI6DPOSE_API) not in sys.path:
        sys.path.insert(0, str(OMNI6DPOSE_API))

    import torch  # noqa: re-import after CUDA_VISIBLE_DEVICES
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from eval_real_multi import build_model, rot6d_to_matrix
    from omnivggt.datasets.omni6dpose.omni6dpose_camera_pose import Omni6DPoseCameraPose
    from omnivggt.datasets.utils.transforms import ImgNorm
    from common.eval_utils import DetectMatch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sope_root = str(args.omni6dpose_root / "SOPE")
    object_image_root = str(args.omni6dpose_root / "omni6dpose_ref" / "diverse24")
    meta_root = args.omni6dpose_root / "Meta"
    available = sorted(p.name for p in (args.omni6dpose_root / "SOPE").iterdir() if p.is_dir())
    patches = [p for p in args.patches if p in available]
    view_ids = tuple(int(v) for v in args.view_ids)
    print(f"[eval] device={device} split={args.split} patches={patches} depth_in/scale={args.use_depth_input}/{args.use_depth_scale}")

    print("[eval] building model ...")
    model = build_model(args.checkpoint, device)

    print("[eval] building dataset ...")
    dataset = Omni6DPoseCameraPose(
        dataset_location=sope_root, dset="test", layout="sope",
        patches=patches, split=args.split,
        object_image_root=object_image_root, oid_to_pam_json=None,
        fixed_object_view_ids=view_ids, num_object_views=len(view_ids),
        strict_fixed_object_view_ids=True, expand_records_by_object=True,
        normalize_object_translation_by_depth_mean=True, verify_files=True,
        only_scene_name=args.scene,
        z_far=int(args.z_far), resolution=DEFAULT_RESOLUTION, transform=ImgNorm, seed=42,
    )
    if args.frame_stride > 1:
        before = len(dataset.records)
        dataset.records = [r for r in dataset.records if int(r.get("image_id", 0)) % args.frame_stride == 0]
        dataset.scenes = dataset.records
        print(f"[eval] frame-stride={args.frame_stride}: {before} -> {len(dataset.records)} records")
    if args.num_shards > 1:
        before = len(dataset.records)
        dataset.records = dataset.records[args.shard_id::args.num_shards]
        dataset.scenes = dataset.records
        print(f"[eval] shard {args.shard_id}/{args.num_shards}: {before} -> {len(dataset.records)} records")
    print(f"[eval] records={len(dataset.records)} objects={len(dataset.object_records_by_name)}")

    resolve_sym, sym_desc = build_symmetry_resolver(meta_root, args.sym_meta, args.no_symmetry)
    print(f"[eval] symmetry source: {sym_desc}")

    run_meta = {
        "checkpoint": str(args.checkpoint), "split": args.split, "patches": patches,
        "symmetry": sym_desc, "use_depth_input": args.use_depth_input,
        "use_depth_scale": args.use_depth_scale,
    }

    import pickle

    def _dump(path, lists):
        tmp = Path(str(path) + ".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump({**lists, "meta": run_meta}, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)  # atomic, so a kill mid-write never corrupts the pkl

    gt_aff, gt_sz, gt_sym, gt_cls, pr_aff, pr_sz = [], [], [], [], [], []
    # Resume: reload partial predictions and skip the records already done (records
    # are processed in order, so the first len(done) of this shard's slice are done).
    if args.resume and args.save_predictions is not None and Path(args.save_predictions).is_file():
        with open(args.save_predictions, "rb") as fh:
            d = pickle.load(fh)
        gt_aff, gt_sz, gt_sym = d["gt_aff"], d["gt_sz"], d["gt_sym"]
        gt_cls, pr_aff, pr_sz = d["gt_cls"], d["pr_aff"], d["pr_sz"]
        done = len(gt_aff)
        dataset.records = dataset.records[done:]
        dataset.scenes = dataset.records
        print(f"[eval] resume: {done} already done, {len(dataset.records)} remaining")

    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False,
                        num_workers=int(args.num_workers), pin_memory=device.type == "cuda", drop_last=False)

    def _lists():
        return {"gt_aff": gt_aff, "gt_sz": gt_sz, "gt_sym": gt_sym,
                "gt_cls": gt_cls, "pr_aff": pr_aff, "pr_sz": pr_sz}

    seen, t0, nb = 0, time.time(), 0
    if args.save_predictions is not None:
        Path(args.save_predictions).parent.mkdir(parents=True, exist_ok=True)
    pbar = tqdm(loader, desc="infer SOPE", dynamic_ncols=True)
    for batch in pbar:
        images = batch["images"].to(device, non_blocking=True)
        object_images = batch["object_images"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True) if args.use_depth_input else None
        mask = batch["valid_mask"].to(device, non_blocking=True) if args.use_depth_input else None
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=(not args.no_amp) and device.type == "cuda"):
                out = model.inference(
                    images=images, object_images=object_images, extrinsics=None, intrinsics=None,
                    depth=depth, mask=mask, camera_gt_index=[],
                    depth_gt_index=[0] if args.use_depth_input else [],
                )
        pred_pose = out["object_pose"].detach().float().cpu().numpy()
        pred_trans = out["object_translation"].detach().float().cpu().numpy()
        if "object_size" in out:
            pred_size_b = out["object_size"].detach().float().cpu().numpy().reshape(len(pred_pose), 3)
        else:
            pred_size_b = np.exp(out["object_size_log"].detach().float().cpu().numpy()).reshape(len(pred_pose), 3)

        for i in range(int(pred_pose.shape[0])):
            scale = float(np.asarray(_batch_item(batch["depth_mean_scale"], i)).reshape(-1)[0])
            pred_R = rot6d_to_matrix(pred_pose[i])
            pred_t = np.asarray(pred_trans[i], dtype=np.float64).reshape(3) * (scale if args.use_depth_scale else 1.0)
            gt_R = np.asarray(_batch_item(batch["object_rotation"], i), dtype=np.float64).reshape(3, 3)
            gt_t = np.asarray(_batch_item(batch["object_translation_metric"], i), dtype=np.float64).reshape(3)
            gt_size = np.asarray(_batch_item(batch["object_size"], i), dtype=np.float64).reshape(3)
            oid = str(_batch_item(batch["oid"], i))
            cname = str(_batch_item(batch["category"], i))

            gt_aff.append(affine(gt_R, gt_t)); pr_aff.append(affine(pred_R, pred_t))
            gt_sz.append(gt_size); pr_sz.append(np.asarray(pred_size_b[i], dtype=np.float64).reshape(3))
            gt_cls.append(int(np.asarray(_batch_item(batch["class_id"], i)).reshape(-1)[0]))
            gt_sym.append(resolve_sym(oid, cname))
            seen += 1
        nb += 1
        pbar.set_postfix(samples=len(gt_aff))
        if args.save_predictions is not None and nb % max(1, args.checkpoint_every) == 0:
            _dump(args.save_predictions, _lists())
        if args.limit is not None and seen >= args.limit:
            break
    pbar.close()

    print(f"[eval] collected {len(gt_aff)} predictions in {time.time()-t0:.1f}s.")

    # Shard mode: dump raw arrays for a later --merge pass instead of metrics.
    if args.save_predictions is not None:
        _dump(args.save_predictions, _lists())
        # mark done so the orchestrator/merge knows this shard completed fully
        Path(str(args.save_predictions) + ".done").write_text(str(len(gt_aff)))
        print(f"[eval] shard predictions ({len(gt_aff)}) -> {args.save_predictions}")
        return

    print("[eval] computing cutoop metrics ...")
    dm = DetectMatch(
        gt_affine=np.stack(gt_aff), gt_size=np.stack(gt_sz),
        gt_sym_labels=np.array(gt_sym, dtype=object), gt_class_labels=np.array(gt_cls),
        pred_affine=np.stack(pr_aff), pred_size=np.stack(pr_sz),
    ).calibrate_rotation()
    _report_metrics(dm.metrics(iou_thresholds=IOU_THRESHOLDS, pose_thresholds=POSE_THRESHOLDS),
                    args.output_dir, run_meta, len(gt_aff))


def _report_metrics(m, out_dir: Path, meta: dict, n: int) -> None:
    cm = m.class_means
    split = meta.get("split", "?")
    print(f"\n================ SOPE/{split} cutoop metrics (class-mean) ================")
    print(f"records={n}  classes={len(m.class_metrics)}  symmetry={meta.get('symmetry')}")
    print(f"patches={meta.get('patches')}  depth_in/scale={meta.get('use_depth_input')}/{meta.get('use_depth_scale')}")
    print(f"mean rotation error : {cm.deg_mean:.3f} deg")
    print(f"mean translation err: {cm.sht_mean:.3f} cm")
    print(f"3D IoU mean (mIoU)  : {float(np.mean(cm.iou_mean)):.4f}")
    for t, a in zip(IOU_THRESHOLDS, cm.iou_acc):
        print(f"  IoU acc @ {t:<4}     : {a:.4f}  ({a*100:.1f}%)")
    for (d, sht), a in zip(POSE_THRESHOLDS, cm.pose_acc):
        print(f"  pose acc {d}deg/{sht}cm   : {a:.4f}  ({a*100:.1f}%)")
    print(f"  rot AUC (0-5deg)    : {cm.deg_auc.auc:.4f}")
    print(f"  trans AUC (0-10cm)  : {cm.sht_auc.auc:.4f}")
    print(f"  IoU AUC             : {[round(x.auc, 4) for x in cm.iou_auc]}")
    print(f"  pose VUS            : {[round(x.auc, 4) for x in cm.pose_auc]}")
    print("==============================================================================\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    m.dump_json(str(out_dir / "cutoop_metrics.json"))
    summary = {
        **meta, "n_records": n,
        "deg_mean": float(cm.deg_mean), "sht_mean_cm": float(cm.sht_mean),
        "iou_mean": float(np.mean(cm.iou_mean)),
        "iou_acc": {str(t): float(a) for t, a in zip(IOU_THRESHOLDS, cm.iou_acc)},
        "pose_acc": {f"{d}deg{sht}cm": float(a) for (d, sht), a in zip(POSE_THRESHOLDS, cm.pose_acc)},
        "rot_auc_0_5deg": float(cm.deg_auc.auc), "trans_auc_0_10cm": float(cm.sht_auc.auc),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[eval] per-class -> {out_dir/'cutoop_metrics.json'}")
    print(f"[eval] summary  -> {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
