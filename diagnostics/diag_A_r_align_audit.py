"""DIAGNOSTIC A: R_align consistency audit across all 4 training datasets.

For each (dataset, category), gather all training instances' GT R_native_to_cam,
apply the canonical R_align, and check whether the R_aligned rotations form a
coherent canonical frame (good) or are scattered (bad).

Pipeline:
  1. Per dataset, scan all training label files in parallel using multiprocessing.
  2. For each (dataset, instance) pair, collect per-frame R_aligned values.
  3. Compute the "representative" R_aligned per instance (best orthogonal match
     to the per-instance mean).
  4. For each category, compute pairwise rotation distance between instances'
     representatives -- HIGH variance ⇒ R_align inconsistent or training labels
     conflict ⇒ model cannot learn a canonical pose.

Output: outputs/diag_A_r_align_audit/{summary.json, per_dataset_table.txt}.
"""
from __future__ import annotations
import json
import multiprocessing as mp
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

FREEPOSE = Path('/mnt/train-data-4-hdd/yian/freepose')
ALIGN_JSON = FREEPOSE / 'omni-object_clone/dataset_align.json'
OUT_DIR = FREEPOSE / 'omni-object_clone/outputs/diag_A_r_align_audit'
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(ALIGN_JSON) as fh:
    ALIGN = json.load(fh)


# --------------- helpers ---------------
def rot_dist_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    R1 = R1 / np.cbrt(np.linalg.det(R1))
    R2 = R2 / np.cbrt(np.linalg.det(R2))
    R = R1 @ R2.T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def representative(R_list: List[np.ndarray]) -> np.ndarray | None:
    if not R_list:
        return None
    R_arr = np.stack(R_list)
    M = R_arr.mean(axis=0)
    U, _, Vt = np.linalg.svd(M)
    R_mean = U @ Vt
    if np.linalg.det(R_mean) < 0:
        U[:, -1] *= -1
        R_mean = U @ Vt
    best, bd = 0, 1e9
    for i, R in enumerate(R_arr):
        d = rot_dist_deg(R, R_mean)
        if d < bd:
            bd, best = d, i
    return R_arr[best]


# --------------- HouseCat6D scanner ---------------
def _hc_load_label(args):
    label_path, hc_id_to_cat, hc_r_align_by_cat = args
    out = []
    try:
        with open(label_path, 'rb') as h:
            lab = pickle.load(h)
    except Exception as e:
        return out
    for i, name in enumerate(lab.get('model_list', [])):
        name = str(name)
        cls_id = int(lab['class_ids'][i])
        cat = hc_id_to_cat.get(cls_id)
        if cat is None:
            continue
        R_native = np.asarray(lab['rotations'][i], dtype=np.float64).reshape(3, 3)
        R_align = hc_r_align_by_cat.get(cat, np.eye(3))
        R_aligned = R_native @ R_align.T
        out.append((name, cat, R_aligned))
    return out


def scan_housecat6d(split: str, n_workers: int = 16):
    """split = 'train' (scene01..34) or 'test' (test_scene*)."""
    hc_align = ALIGN['datasets']['housecat6d']
    hc_id_to_cat = {int(v): k for k, v in hc_align['category_name_to_id'].items()}
    hc_r_align_by_cat = {k: np.asarray(v['R_align'], dtype=np.float64).reshape(3, 3)
                         for k, v in hc_align['classes'].items()}
    hc_root = FREEPOSE / 'housecat6d'
    if split == 'train':
        scene_dirs = [d for d in sorted(hc_root.glob('scene*'))
                      if d.is_dir() and not d.name.startswith(('test_', 'val_'))]
    else:
        scene_dirs = sorted(hc_root.glob('test_scene*'))
    label_paths: List[Path] = []
    for d in scene_dirs:
        ldir = d / 'labels'
        if not ldir.is_dir():
            continue
        label_paths.extend(sorted(ldir.glob('*_label.pkl')))
    # subsample every 10 frames to keep it fast — R_aligned varies slowly over a scene anyway
    label_paths = label_paths[::10]
    with mp.Pool(n_workers) as pool:
        chunks = pool.map(
            _hc_load_label,
            [(p, hc_id_to_cat, hc_r_align_by_cat) for p in label_paths],
            chunksize=8,
        )
    per_inst = defaultdict(list)
    for chunk in chunks:
        for name, cat, R in chunk:
            per_inst[(name, cat)].append(R)
    return per_inst


# --------------- YCBV scanner ---------------
def _ycbv_load_scene_gt(args):
    gt_path, info_path, ycbv_r_align, obj_id_to_class = args
    out = []
    try:
        with open(gt_path) as h:
            gt = json.load(h)
    except Exception:
        return out
    for frame_id, entries in gt.items():
        for entry in entries:
            obj_id = int(entry['obj_id'])
            if obj_id not in ycbv_r_align:
                continue
            R = np.asarray(entry['cam_R_m2c'], dtype=np.float64).reshape(3, 3)
            R_align = ycbv_r_align[obj_id]
            R_aligned = R @ R_align.T
            out.append((obj_id_to_class.get(obj_id, f'obj_{obj_id:06d}'), obj_id, R_aligned))
    return out


def scan_ycbv(n_workers: int = 16):
    ycbv_root = FREEPOSE / 'datasets_real/ycbv'
    if not ycbv_root.exists():
        return {}
    ycbv_align = ALIGN['datasets'].get('ycbv', {})
    classes = ycbv_align.get('classes', {})
    ycbv_r_align = {int(k): np.asarray(v['R_align'], dtype=np.float64).reshape(3, 3)
                    for k, v in classes.items()}
    obj_id_to_class = {int(k): str(v.get('name', f'obj{k}')) for k, v in classes.items()}
    train_dir = ycbv_root / 'train_real'
    if not train_dir.is_dir():
        return {}
    gt_paths = sorted(train_dir.glob('*/scene_gt.json'))
    with mp.Pool(n_workers) as pool:
        chunks = pool.map(
            _ycbv_load_scene_gt,
            [(p, None, ycbv_r_align, obj_id_to_class) for p in gt_paths],
            chunksize=4,
        )
    per_inst = defaultdict(list)
    for chunk in chunks:
        for name, obj_id, R in chunk:
            per_inst[(name, str(obj_id))].append(R)
    return per_inst


# --------------- REAL275 scanner ---------------
def _real275_load_gt(args):
    pkl_path, r_align_by_class = args
    out = []
    try:
        with open(pkl_path, 'rb') as h:
            d = pickle.load(h)
    except Exception:
        return out
    if 'gt_RTs' not in d or 'gt_class_ids' not in d:
        return out
    rts = np.asarray(d['gt_RTs'], dtype=np.float64)
    cls_ids = np.asarray(d['gt_class_ids'], dtype=np.int32)
    names = d.get('model_list', [f'inst_{i}' for i in range(len(cls_ids))])
    for i in range(len(cls_ids)):
        cid = int(cls_ids[i])
        if cid not in r_align_by_class:
            continue
        rotmat = rts[i, :3, :3]
        # remove uniform scale
        cn = np.linalg.norm(rotmat[:, 0])
        if cn < 1e-8:
            continue
        R_native = rotmat / cn
        R_align = r_align_by_class[cid]
        R_aligned = R_native @ R_align.T
        out.append((str(names[i]), cid, R_aligned))
    return out


def scan_real275(n_workers: int = 16):
    r275_root = FREEPOSE / 'real275'
    if not r275_root.exists():
        return {}
    r275_align = ALIGN['datasets'].get('real275', {})
    classes = r275_align.get('classes', {})
    r_align_by_class = {int(k): np.asarray(v['R_align'], dtype=np.float64).reshape(3, 3)
                        for k, v in classes.items()}
    class_id_to_name = r275_align.get('class_id_to_name', {})
    gt_dir = r275_root / 'gts' / 'real_train_umeyama'
    if not gt_dir.is_dir():
        return {}
    gt_paths = sorted(gt_dir.glob('*.pkl'))[::5]  # subsample
    with mp.Pool(n_workers) as pool:
        chunks = pool.map(_real275_load_gt,
                          [(p, r_align_by_class) for p in gt_paths],
                          chunksize=8)
    per_inst = defaultdict(list)
    for chunk in chunks:
        for name, cid, R in chunk:
            cat_name = class_id_to_name.get(str(cid), f'cls{cid}')
            per_inst[(name, cat_name)].append(R)
    return per_inst


# --------------- OO9D scanner ---------------
def _oo9d_load_split(args):
    json_path, oo9d_root = args
    out = []
    try:
        with open(json_path) as h:
            split = json.load(h)
    except Exception:
        return out
    items = split if isinstance(split, list) else split.get('items', [])
    for ent in items:
        category = ent.get('category', 'unknown')
        obj_name = ent.get('object_name', ent.get('name', ''))
        R = ent.get('R') or ent.get('rotation')
        if R is None:
            continue
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        out.append((obj_name, category, R))
    return out


def scan_oo9d_quick():
    """OO9D split JSON is huge; skip in-depth scan for the audit -- just check
    that it exists and report record count. Detailed analysis would require
    rendering each object; defer to part C."""
    split_root = FREEPOSE / 'omni-object_clone/splits_ov9d_unseen_category_generalization'
    if not split_root.exists():
        return {}, 0
    train_json = split_root / 'single' / 'train.json'
    if not train_json.is_file():
        return {}, 0
    with open(train_json) as h:
        split = json.load(h)
    items = split if isinstance(split, list) else split.get('items', list(split.values())[0] if split else [])
    return {}, len(items)


# --------------- main ---------------
def analyze_dataset(name: str, per_inst: Dict[Tuple[str, Any], List[np.ndarray]]):
    per_cat = defaultdict(list)
    for (inst, cat), R_list in per_inst.items():
        rep = representative(R_list)
        if rep is not None:
            per_cat[cat].append((inst, rep))
    rows = []
    for cat in sorted(per_cat.keys(), key=str):
        reps = [r for _, r in per_cat[cat]]
        if len(reps) < 2:
            rows.append({'category': str(cat), 'n_instances': len(reps),
                         'pairwise_med_deg': None, 'pairwise_max_deg': None})
            continue
        dists = [rot_dist_deg(reps[i], reps[j])
                 for i in range(len(reps)) for j in range(i+1, len(reps))]
        rows.append({
            'category': str(cat),
            'n_instances': len(reps),
            'pairwise_med_deg': float(np.median(dists)),
            'pairwise_max_deg': float(np.max(dists)),
            'pairwise_p90_deg': float(np.percentile(dists, 90)),
        })
    return rows


def main():
    print("Scanning HouseCat6D train (parallel)...")
    hc_train = scan_housecat6d('train')
    print(f"  HouseCat6D train: {len(hc_train)} (instance, category) tuples")
    print("Scanning HouseCat6D test (parallel)...")
    hc_test = scan_housecat6d('test')
    print(f"  HouseCat6D test: {len(hc_test)} tuples")
    print("Scanning YCBV train (parallel)...")
    ycbv_train = scan_ycbv()
    print(f"  YCBV train: {len(ycbv_train)} tuples")
    print("Scanning REAL275 train (parallel)...")
    r275_train = scan_real275()
    print(f"  REAL275 train: {len(r275_train)} tuples")
    print("Probing OO9D (split JSON)...")
    _, oo9d_count = scan_oo9d_quick()
    print(f"  OO9D split records: {oo9d_count}")

    out = {}
    for name, data in [('housecat6d_train', hc_train), ('housecat6d_test', hc_test),
                       ('ycbv_train', ycbv_train), ('real275_train', r275_train)]:
        rows = analyze_dataset(name, data)
        out[name] = rows

    # Pretty print
    print("\n" + "="*88)
    print("R_align consistency: pairwise rotation distance between instances "
          "within each category")
    print("(LOW = R_align makes canonical orientations agree across instances; "
          "HIGH = chaotic)")
    print("="*88)
    for name, rows in out.items():
        print(f"\n--- {name} ---")
        print(f"{'category':16s} {'N':>4s} {'pair_med':>10s} {'pair_p90':>10s} {'pair_max':>10s}")
        for r in sorted(rows, key=lambda x: -(x['pairwise_med_deg'] or 0)):
            if r['pairwise_med_deg'] is None:
                print(f"{r['category']:16s} {r['n_instances']:>4d}   (single instance)")
            else:
                print(f"{r['category']:16s} {r['n_instances']:>4d} "
                      f"{r['pairwise_med_deg']:>9.1f}° "
                      f"{r['pairwise_p90_deg']:>9.1f}° "
                      f"{r['pairwise_max_deg']:>9.1f}°")

    json_path = OUT_DIR / 'summary.json'
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {json_path}")


if __name__ == '__main__':
    main()
