"""Compare per-object camera-frame translation distribution between HouseCat6D and
YCBV (train_real). Helps explain why a model trained on one transfers poorly to
the other.

Loads:
  HouseCat6D:  label["translations"]   (already in meters)
  YCBV:        scene_gt.json cam_t_m2c (in mm -> divide by 1000)

Outputs:
  outputs/diag_translation_dist/{
      hc_translations.npy, ycbv_translations.npy,
      summary.json,                  # mean, std, p5..p95
      histogram_xyz.png,             # per-axis histograms
      histogram_norm.png,            # ||t|| histogram
      scatter_xy_xz.png,             # 2D scatter
  }
"""
from __future__ import annotations
import json
import multiprocessing as mp
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FREEPOSE = Path('/mnt/train-data-4-hdd/yian/freepose')
HC = FREEPOSE / 'housecat6d'
YCBV_TRAIN = FREEPOSE / 'datasets_real/ycbv/train_real'
OUT = FREEPOSE / 'omni-object_clone/outputs/diag_translation_dist'
OUT.mkdir(parents=True, exist_ok=True)


# ---------- HouseCat6D ----------
def _hc_load_translations(label_path):
    try:
        with open(label_path, 'rb') as h:
            lab = pickle.load(h)
    except Exception:
        return np.zeros((0, 4))
    n = len(lab.get('translations', []))
    if n == 0:
        return np.zeros((0, 4))
    out = np.zeros((n, 4), dtype=np.float64)  # x, y, z, class_id
    for i in range(n):
        t = np.asarray(lab['translations'][i], dtype=np.float64).reshape(3)
        cid = int(lab['class_ids'][i])
        out[i] = (t[0], t[1], t[2], cid)
    return out


def collect_hc():
    paths = []
    for d in sorted(HC.glob('*')):
        if not d.is_dir(): continue
        lp = d / 'labels'
        if not lp.is_dir(): continue
        # Subsample: every 5th label to keep it fast
        paths.extend(sorted(lp.glob('*_label.pkl'))[::5])
    print(f"HouseCat6D label files (subsampled 1/5): {len(paths)}")
    with mp.Pool(16) as pool:
        results = pool.map(_hc_load_translations, paths)
    arr = np.concatenate([r for r in results if r.size > 0], axis=0)
    print(f"HouseCat6D translations: N={len(arr)}")
    return arr


# ---------- YCBV ----------
def _ycbv_load_translations(gt_path):
    try:
        with open(gt_path) as h:
            d = json.load(h)
    except Exception:
        return np.zeros((0, 4))
    rows = []
    for image_id_str, entries in d.items():
        for ent in entries:
            t_mm = np.asarray(ent['cam_t_m2c'], dtype=np.float64).reshape(3)
            t_m = t_mm / 1000.0   # convert mm -> m
            obj_id = int(ent['obj_id'])
            rows.append((t_m[0], t_m[1], t_m[2], obj_id))
    return np.asarray(rows) if rows else np.zeros((0, 4))


def collect_ycbv():
    paths = sorted(YCBV_TRAIN.glob('*/scene_gt.json'))
    print(f"YCBV train_real scene_gt files: {len(paths)}")
    with mp.Pool(16) as pool:
        results = pool.map(_ycbv_load_translations, paths)
    arr = np.concatenate([r for r in results if r.size > 0], axis=0)
    # Subsample to comparable size
    if len(arr) > 200000:
        idx = np.linspace(0, len(arr) - 1, 200000, dtype=int)
        arr = arr[idx]
    print(f"YCBV translations: N={len(arr)}")
    return arr


# ---------- Stats ----------
def stats(arr_xyz):
    """arr_xyz: (N, 3) array of translations in metres."""
    t = arr_xyz[:, :3]
    norms = np.linalg.norm(t, axis=1)
    return {
        'N': int(len(t)),
        'x': {'mean': float(t[:, 0].mean()), 'std': float(t[:, 0].std()),
              'p5': float(np.percentile(t[:, 0], 5)), 'p50': float(np.median(t[:, 0])),
              'p95': float(np.percentile(t[:, 0], 95)), 'min': float(t[:, 0].min()),
              'max': float(t[:, 0].max())},
        'y': {'mean': float(t[:, 1].mean()), 'std': float(t[:, 1].std()),
              'p5': float(np.percentile(t[:, 1], 5)), 'p50': float(np.median(t[:, 1])),
              'p95': float(np.percentile(t[:, 1], 95)), 'min': float(t[:, 1].min()),
              'max': float(t[:, 1].max())},
        'z': {'mean': float(t[:, 2].mean()), 'std': float(t[:, 2].std()),
              'p5': float(np.percentile(t[:, 2], 5)), 'p50': float(np.median(t[:, 2])),
              'p95': float(np.percentile(t[:, 2], 95)), 'min': float(t[:, 2].min()),
              'max': float(t[:, 2].max())},
        'norm': {'mean': float(norms.mean()), 'std': float(norms.std()),
                 'p5': float(np.percentile(norms, 5)), 'p50': float(np.median(norms)),
                 'p95': float(np.percentile(norms, 95)), 'min': float(norms.min()),
                 'max': float(norms.max())},
    }


def main():
    print("=== Collecting HouseCat6D translations ===")
    hc = collect_hc()
    np.save(OUT / 'hc_translations.npy', hc)

    print("\n=== Collecting YCBV translations ===")
    ycbv = collect_ycbv()
    np.save(OUT / 'ycbv_translations.npy', ycbv)

    print("\n=== Stats ===")
    hc_stat = stats(hc[:, :3])
    yc_stat = stats(ycbv[:, :3])
    summary = {'housecat6d': hc_stat, 'ycbv_train_real': yc_stat}
    with (OUT / 'summary.json').open('w') as f:
        json.dump(summary, f, indent=2)

    print(f"{'metric':12s}  {'housecat6d':>18s}  {'ycbv_train_real':>18s}")
    print("-" * 56)
    for k in ['x', 'y', 'z', 'norm']:
        for stat in ['p5', 'p50', 'p95', 'mean', 'std']:
            print(f"{k}_{stat:6s}  {hc_stat[k][stat]:>18.4f}  {yc_stat[k][stat]:>18.4f}")
        print('-'*56)

    # Plot histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, axis in enumerate(['x', 'y', 'z']):
        ax = axes[i]
        ax.hist(hc[:, i], bins=80, alpha=0.55, label=f'HouseCat6D (N={len(hc)})', color='C0', density=True)
        ax.hist(ycbv[:, i], bins=80, alpha=0.55, label=f'YCBV train_real (N={len(ycbv)})', color='C1', density=True)
        ax.set_title(f'Camera-frame translation {axis} (meters)')
        ax.set_xlabel(f'{axis} (m)')
        ax.set_ylabel('density')
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / 'histogram_xyz.png', dpi=120)
    plt.close()
    print(f"\nSaved histogram_xyz.png")

    # Norm histogram
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    hc_norm = np.linalg.norm(hc[:, :3], axis=1)
    yc_norm = np.linalg.norm(ycbv[:, :3], axis=1)
    ax.hist(hc_norm, bins=80, alpha=0.55, label=f'HouseCat6D', color='C0', density=True)
    ax.hist(yc_norm, bins=80, alpha=0.55, label=f'YCBV train_real', color='C1', density=True)
    ax.set_title('Distance |t| from camera to object (meters)')
    ax.set_xlabel('|t| (m)')
    ax.set_ylabel('density')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / 'histogram_norm.png', dpi=120)
    plt.close()
    print("Saved histogram_norm.png")

    # XY and XZ scatter (subsample to 5000 each for plotting)
    def sub(arr, n=5000):
        if len(arr) <= n: return arr
        idx = np.random.RandomState(42).choice(len(arr), n, replace=False)
        return arr[idx]
    hc_s = sub(hc); yc_s = sub(ycbv)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].scatter(hc_s[:, 0], hc_s[:, 1], s=3, alpha=0.4, label='HouseCat6D', color='C0')
    axes[0].scatter(yc_s[:, 0], yc_s[:, 1], s=3, alpha=0.4, label='YCBV', color='C1')
    axes[0].set_xlabel('x (m)'); axes[0].set_ylabel('y (m)')
    axes[0].set_title('Translation XY (camera frame)')
    axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_aspect('equal')

    axes[1].scatter(hc_s[:, 0], hc_s[:, 2], s=3, alpha=0.4, label='HouseCat6D', color='C0')
    axes[1].scatter(yc_s[:, 0], yc_s[:, 2], s=3, alpha=0.4, label='YCBV', color='C1')
    axes[1].set_xlabel('x (m)'); axes[1].set_ylabel('z = depth (m)')
    axes[1].set_title('Translation XZ (X vs depth)')
    axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].set_aspect('equal')
    plt.tight_layout()
    plt.savefig(OUT / 'scatter_xy_xz.png', dpi=120)
    plt.close()
    print("Saved scatter_xy_xz.png")


if __name__ == '__main__':
    main()
