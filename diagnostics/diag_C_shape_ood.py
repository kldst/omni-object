"""DIAGNOSTIC C: Train/test shape & appearance OOD quantification.

Per housecat6d category, characterize:
  (1) Size distribution: gt_scales (x,y,z) and derived aspect ratios.
  (2) Reference-view "texture / contrast" proxy: mean luminance variance over
      the 4 views the model consumes.

For each category compare TRAIN vs TEST distributions. High shift / non-overlap
= OOD; low texture variance = "featureless" -> hard to learn orientation from
appearance.

Output: outputs/diag_C_shape_ood/{summary.json, per_category.txt, plots/}.
"""
from __future__ import annotations
import json
import multiprocessing as mp
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

FREEPOSE = Path('/mnt/train-data-4-hdd/yian/freepose')
HC = FREEPOSE / 'housecat6d'
REF = HC / 'housecat6d_aligned_object_refs'
OUT = FREEPOSE / 'omni-object_clone/outputs/diag_C_shape_ood'
OUT.mkdir(parents=True, exist_ok=True)
VIEWS = (1, 5, 10, 15)

# --------------- helpers ---------------
def _load_instance_size(args):
    """Return (model_name, category, gt_scales[3]) for first frame seen.

    The size is the same across frames so first occurrence is enough.
    """
    label_path = args
    try:
        with open(label_path, 'rb') as h:
            lab = pickle.load(h)
    except Exception:
        return []
    out = []
    for i, name in enumerate(lab.get('model_list', [])):
        name = str(name)
        cls = int(lab['class_ids'][i])
        size = np.asarray(lab['gt_scales'][i], dtype=np.float64).reshape(3)
        out.append((name, cls, size))
    return out


def collect_sizes(scenes: List[Path]) -> Dict[Tuple[str, int], np.ndarray]:
    paths = []
    for d in scenes:
        ldir = d / 'labels'
        if not ldir.is_dir():
            continue
        # only need 1 label per scene -- size is constant per instance
        paths.extend(sorted(ldir.glob('*_label.pkl'))[:1])
    with mp.Pool(8) as pool:
        chunks = pool.map(_load_instance_size, paths)
    seen = {}
    for chunk in chunks:
        for name, cls, size in chunk:
            seen.setdefault((name, cls), size)
    return seen


def _view_stats(args):
    """Return (instance, std_lum, mean_lum) averaged over the 4 ref views."""
    instance = args
    rgb_dir = REF / instance / 'rgb'
    if not rgb_dir.is_dir():
        return None
    stds, means = [], []
    for v in VIEWS:
        p = rgb_dir / f'{int(v):06d}.png'
        if not p.is_file():
            return None
        im = np.asarray(Image.open(p).convert('L'), dtype=np.float32)
        # crop to foreground: pixels not nearly-white (background)
        fg = im[im < 240]
        if fg.size == 0:
            stds.append(0.0); means.append(255.0); continue
        stds.append(float(fg.std()))
        means.append(float(fg.mean()))
    return instance, float(np.mean(stds)), float(np.mean(means))


def main():
    hc_align = json.load(open(FREEPOSE / 'omni-object_clone/dataset_align.json'))['datasets']['housecat6d']
    id_to_cat = {int(v): k for k, v in hc_align['category_name_to_id'].items()}

    print("Collecting train sizes...")
    train_scenes = [d for d in sorted(HC.glob('scene*'))
                    if d.is_dir() and not d.name.startswith(('test_', 'val_'))]
    train_sizes = collect_sizes(train_scenes)
    print(f"  {len(train_sizes)} train instances")

    print("Collecting test sizes...")
    test_scenes = sorted(HC.glob('test_scene*'))
    test_sizes = collect_sizes(test_scenes)
    print(f"  {len(test_sizes)} test instances")

    # Group by category
    per_cat = defaultdict(lambda: {'train': [], 'test': []})
    for (name, cls), size in train_sizes.items():
        cat = id_to_cat.get(cls)
        if cat: per_cat[cat]['train'].append((name, size))
    for (name, cls), size in test_sizes.items():
        cat = id_to_cat.get(cls)
        if cat: per_cat[cat]['test'].append((name, size))

    # Texture stats per instance
    print("Computing reference-view texture stats...")
    all_insts = set()
    for cat, d in per_cat.items():
        for n, _ in d['train'] + d['test']:
            all_insts.add(n)
    with mp.Pool(12) as pool:
        results = [r for r in pool.map(_view_stats, sorted(all_insts)) if r is not None]
    tex_by_inst = {r[0]: (r[1], r[2]) for r in results}

    summary = {}
    rows = []
    for cat in sorted(per_cat.keys()):
        d = per_cat[cat]
        tr_sizes = np.stack([s for _, s in d['train']]) if d['train'] else np.zeros((0,3))
        te_sizes = np.stack([s for _, s in d['test']]) if d['test'] else np.zeros((0,3))
        def stats(arr):
            if arr.size == 0: return {}
            sorted_arr = np.sort(arr, axis=1)
            min_d, mid_d, max_d = sorted_arr.T
            aspect_max_min = max_d / np.clip(min_d, 1e-6, None)
            aspect_max_mid = max_d / np.clip(mid_d, 1e-6, None)
            return {
                'N': int(arr.shape[0]),
                'diag_cm': [float(np.linalg.norm(s) * 100) for s in arr],
                'diag_median_cm': float(np.median(np.linalg.norm(arr, axis=1)) * 100),
                'aspect_max_over_min_median': float(np.median(aspect_max_min)),
                'aspect_max_over_min_p90': float(np.percentile(aspect_max_min, 90)),
                'aspect_max_over_mid_median': float(np.median(aspect_max_mid)),
            }
        st_tr = stats(tr_sizes); st_te = stats(te_sizes)
        tex_tr = [tex_by_inst[n][0] for n, _ in d['train'] if n in tex_by_inst]
        tex_te = [tex_by_inst[n][0] for n, _ in d['test'] if n in tex_by_inst]
        mean_tr = [tex_by_inst[n][1] for n, _ in d['train'] if n in tex_by_inst]
        mean_te = [tex_by_inst[n][1] for n, _ in d['test'] if n in tex_by_inst]
        rows.append({
            'category': cat,
            'train_N': st_tr.get('N', 0), 'test_N': st_te.get('N', 0),
            'train_diag_median_cm': st_tr.get('diag_median_cm', None),
            'test_diag_median_cm':  st_te.get('diag_median_cm', None),
            'train_aspect_max_min_median': st_tr.get('aspect_max_over_min_median', None),
            'test_aspect_max_min_median':  st_te.get('aspect_max_over_min_median', None),
            'train_ref_texture_std_median': float(np.median(tex_tr)) if tex_tr else None,
            'test_ref_texture_std_median':  float(np.median(tex_te)) if tex_te else None,
            'train_ref_luminance_mean': float(np.mean(mean_tr)) if mean_tr else None,
            'test_ref_luminance_mean':  float(np.mean(mean_te)) if mean_te else None,
        })
        summary[cat] = {'train': st_tr, 'test': st_te,
                        'train_tex_std': tex_tr, 'test_tex_std': tex_te}

    # Print table
    print("\n" + "="*120)
    print(f"{'cat':10s} {'Ntr':>3s}/{'Nte':>3s}  "
          f"{'diag_med_cm (tr/te)':>22s}  "
          f"{'aspect_max/min (tr/te)':>24s}  "
          f"{'tex_std (tr/te)':>20s}  "
          f"{'lum (tr/te)':>14s}")
    print("="*120)
    for r in sorted(rows, key=lambda x: x['category']):
        ar_tr = f"{r['train_aspect_max_min_median']:.2f}" if r['train_aspect_max_min_median'] else "-"
        ar_te = f"{r['test_aspect_max_min_median']:.2f}" if r['test_aspect_max_min_median'] else "-"
        di_tr = f"{r['train_diag_median_cm']:.1f}" if r['train_diag_median_cm'] else "-"
        di_te = f"{r['test_diag_median_cm']:.1f}" if r['test_diag_median_cm'] else "-"
        tx_tr = f"{r['train_ref_texture_std_median']:.1f}" if r['train_ref_texture_std_median'] else "-"
        tx_te = f"{r['test_ref_texture_std_median']:.1f}" if r['test_ref_texture_std_median'] else "-"
        lm_tr = f"{r['train_ref_luminance_mean']:.0f}" if r['train_ref_luminance_mean'] else "-"
        lm_te = f"{r['test_ref_luminance_mean']:.0f}" if r['test_ref_luminance_mean'] else "-"
        print(f"{r['category']:10s} {r['train_N']:>3d}/{r['test_N']:>3d}  "
              f"{di_tr:>10s} / {di_te:>10s}  "
              f"{ar_tr:>10s} / {ar_te:>10s}  "
              f"{tx_tr:>9s} / {tx_te:>9s}  "
              f"{lm_tr:>6s} / {lm_te:>6s}")

    # Save
    out_json = OUT / 'summary.json'
    with open(out_json, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved {out_json}")

    # Per-category note text
    note = []
    for r in rows:
        cat = r['category']
        notes = []
        # aspect ratio shift
        if r['train_aspect_max_min_median'] and r['test_aspect_max_min_median']:
            shift = r['test_aspect_max_min_median'] / r['train_aspect_max_min_median']
            if shift > 1.5 or shift < 1/1.5:
                notes.append(f"aspect ratio shift ×{shift:.2f}")
        # texture diff
        if r['train_ref_texture_std_median'] and r['test_ref_texture_std_median']:
            t_tr, t_te = r['train_ref_texture_std_median'], r['test_ref_texture_std_median']
            if abs(t_tr - t_te) / max(t_tr, t_te) > 0.3:
                notes.append(f"texture-std shift {t_tr:.1f} -> {t_te:.1f}")
        # absolute texture
        if r['train_ref_texture_std_median'] and r['train_ref_texture_std_median'] < 25:
            notes.append("TRAIN ref is featureless (std<25)")
        if r['test_ref_texture_std_median'] and r['test_ref_texture_std_median'] < 25:
            notes.append("TEST ref is featureless (std<25)")
        note.append(f"{cat}: {' ; '.join(notes) if notes else 'no major shift'}")
    print("\n".join(note))


if __name__ == '__main__':
    main()
