#!/usr/bin/env python3
"""Project OV9D camera-frame GT bbox for dataset sanity checks."""

import argparse
import sys
from pathlib import Path
from importlib import import_module

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OV9DCameraPose = import_module("omnivggt.datasets.6Dpose.ov9d_camera_pose").OV9DCameraPose


EDGES = (
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def bbox_corners_m(info):
    min_xyz = np.array([info["min_x"], info["min_y"], info["min_z"]], dtype=np.float32) / 1000.0
    size = np.array([info["size_x"], info["size_y"], info["size_z"]], dtype=np.float32) / 1000.0
    max_xyz = min_xyz + size
    xs = [min_xyz[0], max_xyz[0]]
    ys = [min_xyz[1], max_xyz[1]]
    zs = [min_xyz[2], max_xyz[2]]
    return np.array([[x, y, z] for z in zs for y in ys for x in xs], dtype=np.float32)


def project(points, rotation, translation, intrinsic):
    pts_cam = points @ rotation.T + translation.reshape(1, 3)
    valid = pts_cam[:, 2] > 1e-6
    uvw = pts_cam @ intrinsic.T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-6, None)
    return uv, valid


def draw_bbox(image, uv, valid):
    out = Image.fromarray(image)
    draw = ImageDraw.Draw(out)
    for a, b in EDGES:
        if not (valid[a] and valid[b]):
            continue
        p0 = tuple(np.round(uv[a]).astype(int).tolist())
        p1 = tuple(np.round(uv[b]).astype(int).tolist())
        draw.line((p0, p1), fill=(0, 255, 0), width=2)
    return np.asarray(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", default="outputs/ov9d_camera_pose_sanity.png")
    parser.add_argument("--split-json", default=None)
    parser.add_argument("--data-root", default="/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d")
    args = parser.parse_args()

    dataset = OV9DCameraPose(
        dataset_location=args.data_root,
        split_json=args.split_json,
        dset="test1",
        num_object_views=4,
        verify_files=True,
        z_far=20,
        resolution=(518, 518),
        seed=42,
    )
    sample = dataset[args.index]
    image = sample["images"][0].permute(1, 2, 0).numpy()
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

    object_id = int(sample["object_id"])
    corners = bbox_corners_m(dataset.models_info[str(object_id)])
    uv, valid = project(corners, sample["object_rotation"], sample["object_translation"], sample["intrinsic"][0])
    out = draw_bbox(image, uv, valid)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(output)
    print(f"Wrote {output}")
    print(f"sample={sample['seq_name']} object_id={object_id}")
    print(f"translation_m={sample['object_translation'].tolist()}")
    print(f"size_m={sample['object_size'].tolist()}")


if __name__ == "__main__":
    main()
