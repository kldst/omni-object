#!/usr/bin/env python3
"""Build a symmetry-info JSON for Omni6DPose objects in the loss's expected format.

Symmetry is treated as a CLASS property: we derive a per-class symmetry from
ROPE's real_obj_meta.json (per-axis majority over that class's instances), then
assign it to every reference object by class name. This sidesteps the corrupted
SOPE obj_meta.json while still giving symmetric objects (bottle/bowl/can/ball/...)
the correct equivalence set during the rotation loss.

Output format (consumed by omnivggt/loss.py:_load_symmetry_info), keyed by
"<dataset_label>:<object_id>" where object_id matches the dataset's
object_name_to_id = {name: idx+1 for idx, name in enumerate(sorted(ref names))}:

  {
    "Omni6DPoseCameraPose:<id>": {
        "symmetries_continuous": [{"axis": [0,1,0]}],   # 'any'
        "symmetries_discrete":   [[4x4 flattened], ...]  # 'half'/'quarter'
    }, ...
  }
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DATASET_LABEL = "Omni6DPoseCameraPose"
AXIS_VEC = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}


def axis_angle_4x4(axis, angle_deg):
    ax = np.asarray(axis, dtype=np.float64)
    ax = ax / max(np.linalg.norm(ax), 1e-9)
    a = np.radians(angle_deg)
    x, y, z = ax
    c, s, C = np.cos(a), np.sin(a), 1 - np.cos(a)
    R = np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])
    M = np.eye(4)
    M[:3, :3] = R
    return M.reshape(-1).tolist()


def class_symmetry_from_real(real_meta_path: Path):
    real = json.loads(real_meta_path.read_text(encoding="utf-8"))["instance_dict"]
    votes = defaultdict(lambda: {"x": Counter(), "y": Counter(), "z": Counter()})
    for v in real.values():
        c = v["class_name"]
        sym = v.get("tag", {}).get("symmetry", {})
        for ax in "xyz":
            votes[c][ax][sym.get(ax, "none")] += 1
    return {c: {ax: (votes[c][ax].most_common(1)[0][0] if votes[c][ax] else "none") for ax in "xyz"}
            for c in votes}


def sym_to_loss_entry(per_axis: dict):
    """Convert {'x':none/half/quarter/any, ...} -> loss entry dict."""
    cont, disc = [], []
    for ax, val in per_axis.items():
        if val == "any":
            cont.append({"axis": AXIS_VEC[ax]})
        elif val == "half":
            disc.append(axis_angle_4x4(AXIS_VEC[ax], 180.0))
        elif val == "quarter":
            for ang in (90.0, 180.0, 270.0):
                disc.append(axis_angle_4x4(AXIS_VEC[ax], ang))
        # 'none' -> nothing
    entry = {}
    if cont:
        entry["symmetries_continuous"] = cont
    if disc:
        entry["symmetries_discrete"] = disc
    return entry


def class_of(ref_name: str) -> str:
    ci = ref_name.split("-", 1)[1] if "-" in ref_name else ref_name
    return ci.rsplit("_", 1)[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real-obj-meta", type=Path,
                   default="/mnt/train-data-4-hdd/yian/freepose/Omni6dpose/Omni6DPoseAPI/data/Omni6DPose/Meta/real_obj_meta.json")
    p.add_argument("--object-image-root", type=Path,
                   default="/mnt/train-data-4-hdd/yian/freepose/Omni6dpose/Omni6DPoseAPI/data/Omni6DPose/omni6dpose_ref/diverse24")
    p.add_argument("--out", type=Path,
                   default="/mnt/train-data-4-hdd/yian/freepose/omni-object_clone/omni6dpose_symmetry_info.json")
    args = p.parse_args()

    cls_sym = class_symmetry_from_real(args.real_obj_meta)
    print(f"[class table] {len(cls_sym)} classes from real_obj_meta")

    # object_name_to_id MUST match the dataset loader: sorted ref dir names, idx+1.
    ref_names = sorted(d.name for d in args.object_image_root.iterdir()
                       if d.is_dir() and (d / "rgb").is_dir())
    name_to_id = {name: i + 1 for i, name in enumerate(ref_names)}
    print(f"[refs] {len(ref_names)} reference objects -> object_id 1..{len(ref_names)}")

    out = {}
    n_sym = 0
    no_class = []
    for name, oid in name_to_id.items():
        cls = class_of(name)
        per_axis = cls_sym.get(cls)
        if per_axis is None:
            no_class.append(cls)
            continue
        entry = sym_to_loss_entry(per_axis)
        if entry:  # only store objects that actually have symmetry
            out[f"{DATASET_LABEL}:{oid}"] = entry
            n_sym += 1

    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[done] {n_sym}/{len(ref_names)} objects have symmetry -> {args.out}")
    if no_class:
        from collections import Counter as C
        print(f"[note] {len(no_class)} objects had no class match (default no-symmetry): "
              f"{sorted(set(no_class))[:10]}")
    # sanity: show a few
    for k in list(out.keys())[:3]:
        print("  sample", k, "->", out[k])


if __name__ == "__main__":
    main()
