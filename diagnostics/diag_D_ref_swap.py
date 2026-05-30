"""DIAGNOSTIC D: Reference-swap experiment.

For each failing test instance, run model inference with DIFFERENT reference
view instances (e.g., the original, then swapped with 2 train instances and
1 other test instance of the same class). Measure how much R / t / IoU
changes with the ref swap.

If R changes wildly with ref swap -> model is locked onto reference appearance
("memorize-and-match"). If R is stable -> model uses scene cues independently
of the reference (so the reference views aren't the bottleneck).

Output: outputs/diag_D_ref_swap/per_case.json + console table.
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

PROJECT = Path('/mnt/train-data-4-hdd/yian/freepose/omni-object_clone')
sys.path.insert(0, str(PROJECT))
from eval_real_multi import (
    DEFAULT_OBJECT_VIEWS, DEFAULT_RESOLUTION,
    HouseCat6DTestSceneCameraPose,
    build_model, rot6d_to_matrix,
)
from omnivggt.datasets.utils.transforms import ImgNorm

FREEPOSE = Path('/mnt/train-data-4-hdd/yian/freepose')
HC_ROOT = FREEPOSE / 'housecat6d'
REF_ROOT = HC_ROOT / 'housecat6d_aligned_object_refs'
CHECKPOINT = PROJECT / 'outputs/0521/14000/model.safetensors'
OUT_DIR = PROJECT / 'outputs/diag_D_ref_swap'
OUT_DIR.mkdir(parents=True, exist_ok=True)
ALIGN_JSON = PROJECT / 'dataset_align.json'

# Cases: (test_instance_name, test_scene, other_instances_to_swap_in)
CASES = [
    ('box-colgate', 'test_scene1',
     ['box-iglo', 'box-toffifee', 'box-koala', 'box-crunchy_muesli']),
    ('tube-baby_nivea_comfort', 'test_scene1',
     ['tube-kiwi', 'tube-shoe_polish', 'tube-baby_nivea_spf', 'tube-bb_cream']),
    ('remote-toy', 'test_scene4',
     ['remote-jaxster', 'remote-comfee', 'remote-aircon_chunghop', 'remote-seki_care']),
    ('cutlery-spoon_2_new', 'test_scene1',
     ['cutlery-fork_1_new', 'cutlery-knife_1_new', 'cutlery-spoon_1_new', 'cutlery-knife_2_new']),
    # Control: bottle which works well -- expect small variance under ref swap
    ('bottle-avene_skincare', 'test_scene1',
     ['bottle-soupline', 'bottle-eres_inox', 'bottle-evian_red']),
]


def rot_err_deg(R1, R2):
    R1 = R1/np.cbrt(np.linalg.det(R1)); R2 = R2/np.cbrt(np.linalg.det(R2))
    R = R1 @ R2.T
    return float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))))


def load_ref_tensor(instance: str, views=DEFAULT_OBJECT_VIEWS, resolution=DEFAULT_RESOLUTION):
    rgb_dir = REF_ROOT / instance / 'rgb'
    if not rgb_dir.is_dir():
        raise FileNotFoundError(rgb_dir)
    tensors, shapes = [], []
    for v in views:
        p = rgb_dir / f'{int(v):06d}.png'
        im = Image.open(p).convert('RGB')
        shapes.append(np.array(im.size[::-1], dtype=np.int32))
        im_r = im.resize(tuple(resolution), Image.LANCZOS)
        tensors.append(ImgNorm(im_r))
    return torch.stack(tensors), np.stack(shapes)


def run_one(model, device, sample, ref_tensor, ref_shape):
    """Run model on a single dataset sample with a CUSTOM reference tensor.
    Returns dict(R_aligned_pred, t_pred_metric, size_aligned_pred)."""
    scene_t = sample['images'].unsqueeze(0).to(device, non_blocking=True)
    obj_t = ref_tensor.unsqueeze(0).to(device, non_blocking=True)
    depth_t = torch.as_tensor(sample['depth']).unsqueeze(0).to(device, non_blocking=True)
    mask_t = torch.as_tensor(sample['valid_mask']).unsqueeze(0).to(device, non_blocking=True)
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type=='cuda'):
            out = model.inference(
                images=scene_t, object_images=obj_t,
                extrinsics=None, intrinsics=None,
                depth=depth_t, mask=mask_t,
                camera_gt_index=[], depth_gt_index=[0],
            )
    pose = out['object_pose'].detach().float().cpu().numpy()[0]
    trans = out['object_translation'].detach().float().cpu().numpy()[0]
    if 'object_size' in out:
        size = out['object_size'].detach().float().cpu().numpy()[0]
    elif 'object_size_log' in out:
        size = np.exp(out['object_size_log'].detach().float().cpu().numpy()[0])
    else:
        size = None
    depth_mean = float(np.asarray(sample['depth_mean_scale']).reshape(-1)[0])
    R_aligned = rot6d_to_matrix(pose)
    t_metric = trans.astype(np.float64) * depth_mean
    return R_aligned, t_metric, size


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Loading model on {device}...')
    model = build_model(CHECKPOINT, device)

    results = []
    for test_inst, scene, swap_refs in CASES:
        print(f'\n=== CASE: test={test_inst}  scene={scene} ===')
        # build dataset and locate sample idx for the target instance
        ds = HouseCat6DTestSceneCameraPose(
            dataset_location=str(HC_ROOT), dset='test',
            object_image_root=str(REF_ROOT), align_json=str(ALIGN_JSON),
            only_scene_name=scene, only_object_name=test_inst,
            expand_records_by_object=True, num_object_views=4,
            fixed_object_view_ids=DEFAULT_OBJECT_VIEWS,
            strict_fixed_object_view_ids=True,
            normalize_object_translation_by_depth_mean=True,
            verify_files=True, object_presence_prob=1.0,
            z_far=20, resolution=DEFAULT_RESOLUTION, transform=ImgNorm, seed=42,
        )
        if len(ds) == 0:
            print(f'  no records for {test_inst} in {scene}')
            continue

        # Sample 5 frames evenly across the scene to avoid one-frame artefacts
        N = len(ds)
        sample_idxs = list(np.linspace(0, N-1, 5, dtype=int))
        # all candidate refs: original + swaps
        refs = [test_inst] + swap_refs
        # Load all ref tensors once
        ref_data = {}
        for r in refs:
            try:
                ref_data[r] = load_ref_tensor(r)
            except Exception as e:
                print(f'  cannot load ref {r}: {e}');
        case_results = {'test_instance': test_inst, 'scene': scene, 'frames': []}
        for fi, sidx in enumerate(sample_idxs):
            sample = ds[int(sidx)]
            gt_R = np.asarray(sample['object_rotation'], dtype=np.float64).reshape(3,3)
            gt_t = np.asarray(sample['object_translation_metric'], dtype=np.float64).reshape(3)
            r_align = np.asarray(sample['R_align_housecat6d_to_ov9d'], dtype=np.float64).reshape(3,3)
            image_id = int(np.asarray(sample['ids']).reshape(-1)[0])
            frame_rec = {'frame_idx': int(sidx), 'image_id': image_id, 'preds': {}}
            preds_R = {}
            for r_name, (rt, rs) in ref_data.items():
                R_pred_aligned, t_pred, size_pred = run_one(model, device, sample, rt, rs)
                preds_R[r_name] = R_pred_aligned
                trans_err = float(np.linalg.norm(t_pred - gt_t) * 100)
                rot_err = rot_err_deg(R_pred_aligned, gt_R)
                frame_rec['preds'][r_name] = {
                    'rotation_error_deg_vs_gt': rot_err,
                    'translation_error_cm_vs_gt': trans_err,
                }
            # Pairwise R difference between original and swapped refs
            origin_R = preds_R[test_inst]
            pairwise_swap_drift = {r: rot_err_deg(preds_R[r], origin_R)
                                   for r in preds_R if r != test_inst}
            frame_rec['drift_vs_original_ref_deg'] = pairwise_swap_drift
            case_results['frames'].append(frame_rec)
            print(f'  frame {sidx}: ' +
                  ' | '.join(f"{r}={d:.1f}°" for r,d in pairwise_swap_drift.items()))
        results.append(case_results)

    out_path = OUT_DIR / 'per_case.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved {out_path}')

    # Summary table
    print('\n' + '='*100)
    print(f"{'instance':32s} {'mean drift (deg)':>20s} {'max drift (deg)':>20s} {'rot err with own ref':>22s}")
    print('='*100)
    for case in results:
        all_drifts = []
        own_errs = []
        for fr in case['frames']:
            all_drifts.extend(fr['drift_vs_original_ref_deg'].values())
            own_errs.append(fr['preds'][case['test_instance']]['rotation_error_deg_vs_gt'])
        if not all_drifts: continue
        print(f"{case['test_instance']:32s} {np.mean(all_drifts):>19.1f}° {np.max(all_drifts):>19.1f}° {np.mean(own_errs):>21.1f}°")


if __name__ == '__main__':
    main()
