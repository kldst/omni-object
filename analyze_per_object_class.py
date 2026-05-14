"""Per-category + per-train-frequency breakdown of rotation error on val set.

Loads the trained baseline model (no ablation), runs on val set, then:
  1. groups samples by category (parsed from name2oid.json prefix)
  2. computes train-set frequency for each object_id from train.json
  3. reports rotation error as a function of category AND of train frequency

Run:
  python analyze_per_object_class.py \\
      --ckpt outputs/0511/12000/model.safetensors --max-batches 500 --batch-size 8
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from accelerate import PartialState

from omnivggt.utils.configs import read_config
from train_utils import build_dataset, load_model
from ablate_cross_attn_layers import (
    _geodesic_deg,
    _rot6d_to_matrix,
    build_inputs,
    load_symmetry_info,
    move_to_device,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_ov9d_camera_pose.py")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=500,
                   help="None = full val set")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--out-csv", default="per_object_analysis.csv",
                   help="Write raw per-sample rows here")
    return p.parse_args()


def per_sample_errors(predictions, batch, sym_info):
    """Yield per-sample dicts. Filters by has_object mask."""
    if "object_pose" not in predictions or "object_rotation" not in batch:
        return []
    has_object = batch.get("has_object")
    object_ids = batch.get("object_id")
    pred_pose = predictions["object_pose"].detach()
    gt_rot = batch["object_rotation"].detach()
    pred_trans = predictions.get("object_translation").detach() \
        if predictions.get("object_translation") is not None else None
    gt_trans = batch.get("object_translation")
    pred_size_log = predictions.get("object_size_log").detach() \
        if predictions.get("object_size_log") is not None else None
    gt_size_log = batch.get("object_size_log")

    if has_object is not None:
        mask = has_object.bool().detach()
        if mask.sum() == 0:
            return []
        pred_pose = pred_pose[mask]
        gt_rot = gt_rot[mask]
        if object_ids is not None:
            object_ids = object_ids[mask]
        if pred_trans is not None and gt_trans is not None:
            pred_trans = pred_trans[mask]
            gt_trans = gt_trans[mask]
        if pred_size_log is not None and gt_size_log is not None:
            pred_size_log = pred_size_log[mask]
            gt_size_log = gt_size_log[mask]

    pred_rot = _rot6d_to_matrix(pred_pose.float())
    gt_rot_f = gt_rot.float()
    rot_naive = _geodesic_deg(pred_rot, gt_rot_f)

    rot_sym = rot_naive.clone()
    oids = object_ids.detach().cpu().reshape(-1).tolist() if object_ids is not None else []
    for i, oid in enumerate(oids):
        sym_rots = sym_info.get(int(oid))
        if sym_rots is None:
            continue
        sym_rots = sym_rots.to(pred_rot.device)
        gt_cands = torch.matmul(gt_rot_f[i].unsqueeze(0), sym_rots)
        pred_i = pred_rot[i].unsqueeze(0).expand_as(gt_cands)
        rot_sym[i] = _geodesic_deg(pred_i, gt_cands).min()

    trans_cm = None
    if pred_trans is not None and gt_trans is not None:
        trans_cm = torch.norm(pred_trans.float() - gt_trans.float(), dim=-1) * 100.0
    size_rel = None
    if pred_size_log is not None and gt_size_log is not None:
        pred_size = torch.exp(pred_size_log.float())
        gt_size = torch.exp(gt_size_log.float())
        size_rel = (torch.abs(pred_size - gt_size) / gt_size.clamp(min=1e-6)).mean(dim=-1)

    rows = []
    for i in range(len(oids)):
        rows.append(dict(
            object_id=int(oids[i]),
            rot_sym=float(rot_sym[i].cpu()),
            rot_naive=float(rot_naive[i].cpu()),
            trans_cm=float(trans_cm[i].cpu()) if trans_cm is not None else float("nan"),
            size_rel=float(size_rel[i].cpu()) if size_rel is not None else float("nan"),
        ))
    return rows


def build_oid_to_name(name2oid_path):
    n2o = json.load(open(name2oid_path))
    return {int(v): k for k, v in n2o.items()}


def name_to_category(name):
    """`mug_001` -> `mug`, `coffee_mug_017` -> `coffee_mug`."""
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return name


def build_train_frequency(train_json_path):
    """Returns {object_id: train_sample_count}.

    Approximation: for each train scene, an object listed in `eligible_object_ids`
    contributes `object_view_counts[str(oid)]` records (each (scene, oid, image_id)
    is one training sample as per OV9DCameraPose._build_records).
    """
    payload = json.load(open(train_json_path))
    freq = defaultdict(int)
    for scene in payload.get("scenes", []):
        ovc = scene.get("object_view_counts", {})
        for oid in scene.get("eligible_object_ids", scene.get("object_ids", [])):
            views = int(ovc.get(str(oid), 0))
            freq[int(oid)] += views
    return dict(freq)


def summarize(values):
    if not values:
        return dict(n=0, mean=float("nan"), median=float("nan"), p25=float("nan"), p75=float("nan"))
    xs = sorted(values)
    n = len(xs)
    return dict(
        n=n,
        mean=sum(xs) / n,
        median=xs[n // 2],
        p25=xs[max(0, int(0.25 * n))],
        p75=xs[min(n - 1, int(0.75 * n))],
    )


def main():
    args = parse_args()
    PartialState()
    print(f"[load] config = {args.config}")
    cfg = read_config(args.config)
    cfg.model_url = args.ckpt
    cfg.model_load_strict = False
    cfg.num_workers = 0
    cfg.val_batch_images = args.batch_size

    device = torch.device(args.device)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    model, _ = load_model(cfg, device)
    model = model.to(device).eval()

    sym_info_path = cfg.get("object_srt_symmetry_info_path", "")
    sym_steps = int(cfg.get("object_srt_symmetry_continuous_steps", 72))
    sym_info = load_symmetry_info(sym_info_path, sym_steps) if sym_info_path else {}
    print(f"[setup] symmetry info: {len(sym_info)} objects")

    ov9d_root = Path(cfg.get("ov9d_root", "/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d"))
    oid_to_name = build_oid_to_name(ov9d_root / "name2oid.json")
    print(f"[setup] oid_to_name entries: {len(oid_to_name)}")

    train_json = Path("/mnt/train-data-4-hdd/yian/freepose/omni-object_clone/splits_multi_4_3000/train.json")
    train_freq = build_train_frequency(train_json)
    print(f"[setup] train objects with >=1 sample: {sum(1 for v in train_freq.values() if v > 0)}, "
          f"max freq = {max(train_freq.values()) if train_freq else 0}")

    val_loader = build_dataset(
        dataset=cfg.val_dataset,
        batch_size=args.batch_size,
        num_workers=cfg.get("num_workers", 0),
        test=True,
    )

    rows = []
    autocast_kwargs = dict(device_type="cuda", dtype=dtype,
                           enabled=(device.type == "cuda" and dtype != torch.float32))
    print(f"[run] batches={args.max_batches} batch_size={args.batch_size}")
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if args.max_batches is not None and i >= args.max_batches:
                break
            batch = move_to_device(batch, device)
            inputs = build_inputs(batch)
            with torch.amp.autocast(**autocast_kwargs):
                preds = model(**inputs)
            new_rows = per_sample_errors(preds, batch, sym_info)
            for r in new_rows:
                r["name"] = oid_to_name.get(r["object_id"], f"unk_{r['object_id']}")
                r["category"] = name_to_category(r["name"])
                r["train_freq"] = train_freq.get(r["object_id"], 0)
            rows.extend(new_rows)
            if (i + 1) % 50 == 0:
                print(f"  batch {i+1} | n={len(rows)}")

    print(f"[done] total samples = {len(rows)}")

    # Write CSV
    out_csv = Path(args.out_csv)
    with open(out_csv, "w") as f:
        f.write("object_id,name,category,train_freq,rot_sym,rot_naive,trans_cm,size_rel\n")
        for r in rows:
            f.write(f"{r['object_id']},{r['name']},{r['category']},{r['train_freq']},"
                    f"{r['rot_sym']:.4f},{r['rot_naive']:.4f},"
                    f"{r['trans_cm']:.4f},{r['size_rel']:.4f}\n")
    print(f"[csv] wrote {out_csv}")

    # === Per-category breakdown ===
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["rot_sym"])
    cat_stats = [(cat, summarize(vs)) for cat, vs in by_cat.items()]
    cat_stats.sort(key=lambda x: x[1]["median"])

    print()
    print("=" * 92)
    print("Per-category rot_sym (sorted by median ascending)")
    print("-" * 92)
    print(f"{'category':>22} | {'n':>5} | {'median':>8} | {'mean':>8} | {'p25':>8} | {'p75':>8}")
    print("-" * 92)
    for cat, s in cat_stats:
        if s["n"] < 3:
            continue
        print(f"{cat:>22} | {s['n']:>5} | {s['median']:>7.2f}° | {s['mean']:>7.2f}° | "
              f"{s['p25']:>7.2f}° | {s['p75']:>7.2f}°")
    print("=" * 92)

    # === Per-train-frequency bucket ===
    BUCKETS = [(0, 0, "unseen (0)"),
               (1, 5, "1-5"),
               (6, 20, "6-20"),
               (21, 50, "21-50"),
               (51, 100, "51-100"),
               (101, 10**9, "101+")]
    print()
    print("=" * 80)
    print("Per-train-frequency bucket: does more training help?")
    print("-" * 80)
    print(f"{'bucket':>20} | {'n':>6} | {'unique_objs':>11} | {'median':>8} | {'mean':>8}")
    print("-" * 80)
    for lo, hi, label in BUCKETS:
        vals = [r["rot_sym"] for r in rows if lo <= r["train_freq"] <= hi]
        uniq = len({r["object_id"] for r in rows if lo <= r["train_freq"] <= hi})
        s = summarize(vals)
        if s["n"] == 0:
            continue
        print(f"{label:>20} | {s['n']:>6} | {uniq:>11} | {s['median']:>7.2f}° | {s['mean']:>7.2f}°")
    print("=" * 80)

    # === Correlation rot_sym vs log(train_freq) ===
    pairs = [(r["train_freq"], r["rot_sym"]) for r in rows]
    if pairs:
        import math
        # Pearson on log(freq+1)
        xs = [math.log(p[0] + 1) for p in pairs]
        ys = [p[1] for p in pairs]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / n
        sx = (sum((x - mx) ** 2 for x in xs) / n) ** 0.5
        sy = (sum((y - my) ** 2 for y in ys) / n) ** 0.5
        r_pearson = cov / (sx * sy) if sx > 0 and sy > 0 else float("nan")
        print()
        print(f"Pearson correlation: log(train_freq + 1) vs rot_sym = {r_pearson:+.4f}")
        print("  (negative => more training -> lower error, as expected)")
        print(f"  unique val objects: {len({r['object_id'] for r in rows})}")
        n_unseen = sum(1 for r in rows if r["train_freq"] == 0)
        print(f"  unseen-in-train val samples: {n_unseen} / {len(rows)} "
              f"({100.0 * n_unseen / len(rows):.1f}%)")


if __name__ == "__main__":
    main()
