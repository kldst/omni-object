"""Draw bbox + 3-axes on RAW YCBV BOP images using scene_gt.json directly.

No dataset class involved — pure raw BOP files:
  - rgb/<image_id>.png       (640x480)
  - scene_gt.json            (cam_R_m2c, cam_t_m2c per object in mm)
  - scene_camera.json        (cam_K per image)
  - models/models_info.json  (size_x/y/z in mm)

Usage:
    python verify_ycbv_raw.py \
        --ycbv-root /mnt/train-data-4-hdd/yian/freepose/datasets_real/ycbv \
        --split train_real --num-samples 32 \
        --output-dir verify_out/ycbv_train_raw
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from verify_dataset_pose import draw_pose_on_image, draw_caption  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ycbv-root",
                   default="/mnt/train-data-4-hdd/yian/freepose/datasets_real/ycbv")
    p.add_argument("--split", default="train_real",
                   choices=["train_real", "train_pbr", "test"])
    p.add_argument("--num-samples", type=int, default=32)
    p.add_argument("--output-dir", default="verify_out/ycbv_train_raw")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only-obj-id", type=int, default=None,
                   help="Restrict to one object id (1..21)")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.ycbv_root)
    split_root = root / args.split
    models_info = json.loads((root / "models" / "models_info.json").read_text())
    align = json.loads((THIS_DIR / "dataset_align.json").read_text())
    obj_id_to_category = {int(k): str(v) for k, v in
                          align["datasets"]["ycbv"]["obj_id_to_category"].items()}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a flat list of (scene_name, image_id, obj_index, obj_id).
    samples = []
    for scene_dir in sorted(split_root.iterdir()):
        gt_path = scene_dir / "scene_gt.json"
        cam_path = scene_dir / "scene_camera.json"
        if not gt_path.is_file() or not cam_path.is_file():
            continue
        scene_gt = json.loads(gt_path.read_text())
        for image_id_str, gts in scene_gt.items():
            for obj_index, gt in enumerate(gts):
                oid = int(gt["obj_id"])
                if args.only_obj_id is not None and oid != args.only_obj_id:
                    continue
                samples.append((scene_dir.name, image_id_str, obj_index, oid))
    print(f"[verify-raw] total (scene,img,obj) tuples = {len(samples)}", flush=True)
    if not samples:
        return

    rng = random.Random(args.seed)
    rng.shuffle(samples)
    chosen = samples[: args.num_samples]

    n_saved = 0
    for scene_name, image_id_str, obj_index, obj_id in chosen:
        scene_dir = split_root / scene_name
        scene_gt = json.loads((scene_dir / "scene_gt.json").read_text())
        scene_cam = json.loads((scene_dir / "scene_camera.json").read_text())

        gt = scene_gt[image_id_str][obj_index]
        cam = scene_cam[image_id_str]
        image_id = int(image_id_str)
        K = np.asarray(cam["cam_K"], dtype=np.float64).reshape(3, 3)
        R = np.asarray(gt["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        t_m = np.asarray(gt["cam_t_m2c"], dtype=np.float64) / 1000.0

        info = models_info[str(int(obj_id))]
        size_m = np.array([info["size_x"], info["size_y"], info["size_z"]],
                          dtype=np.float64) / 1000.0

        rgb_path = scene_dir / "rgb" / f"{image_id:06d}.png"
        img = Image.open(rgb_path).convert("RGB")
        rendered = draw_pose_on_image(img, K, R, t_m, size_m,
                                      draw_bbox=True, upscale=2)

        caption = [
            f"YCBV-raw {args.split} scene={scene_name} img={image_id:06d}",
            f"obj_id={obj_id} cat={obj_id_to_category.get(obj_id, '')}",
            f"size_mm=({info['size_x']:.1f},{info['size_y']:.1f},{info['size_z']:.1f})",
            f"t_metric=({t_m[0]:+.2f},{t_m[1]:+.2f},{t_m[2]:+.2f}) m",
        ]
        rendered = draw_caption(rendered, caption)

        out_name = f"{n_saved:04d}_{scene_name}_img{image_id:06d}_obj{obj_id:03d}.png"
        rendered.save(out_dir / out_name)
        n_saved += 1
        print(f"[ok] {out_dir / out_name}", flush=True)

    print(f"[done] saved={n_saved} dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()
