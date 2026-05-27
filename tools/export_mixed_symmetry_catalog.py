#!/usr/bin/env python3
"""Export a browsable catalog for mixed_symmetry_info.json entries."""

import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FREEPOSE_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose")
DEFAULT_SYMMETRY_INFO = PROJECT_ROOT / "mixed_symmetry_info.json"
DEFAULT_ALIGN_JSON = PROJECT_ROOT / "dataset_align.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mixed_symmetry_catalog"
AXIS_COLORS = {
    "x": (235, 40, 40),
    "y": (30, 180, 70),
    "z": (45, 105, 235),
}


def load_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def first_existing_image(root: Path, preferred_ids: Iterable[int] = (1, 5, 10, 15)) -> Optional[Path]:
    rgb_dir = root / "rgb"
    for image_id in preferred_ids:
        path = rgb_dir / f"{int(image_id):06d}.png"
        if path.is_file():
            return path
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        found = sorted(rgb_dir.glob(pattern))
        if found:
            return found[0]
    return None


def category_from_instance_name(name: str) -> str:
    return re.sub(r"_\d+$", "", str(name))


def build_oo9d_lookup(freepose_root: Path) -> Dict[int, Dict[str, Any]]:
    object_root = freepose_root / "ov9d" / "ov9d_around_image"
    name2oid = load_json(freepose_root / "ov9d" / "ov9d" / "name2oid.json", {})
    lookup: Dict[int, Dict[str, Any]] = {}
    for object_instance, object_id in name2oid.items():
        object_id = int(object_id)
        object_dir = object_root / f"obj_{object_id:06d}"
        metadata = load_json(object_dir / "metadata.json", {})
        lookup[object_id] = {
            "dataset": "OO9DSingleCameraPose",
            "object_id": object_id,
            "object_name": str(object_instance),
            "category": category_from_instance_name(str(object_instance)),
            "object_dir": object_dir,
            "metadata": metadata,
            "image_path": first_existing_image(object_dir),
        }
    return lookup


def build_sorted_ref_lookup(
    dataset_label: str,
    object_root: Path,
    default_category_split: str,
) -> Dict[int, Dict[str, Any]]:
    lookup: Dict[int, Dict[str, Any]] = {}
    dirs = sorted(path for path in object_root.iterdir() if path.is_dir()) if object_root.is_dir() else []
    for object_id, object_dir in enumerate(dirs, start=1):
        metadata = load_json(object_dir / "metadata.json", {})
        lookup[object_id] = {
            "dataset": dataset_label,
            "object_id": object_id,
            "object_name": object_dir.name,
            "category": str(metadata.get("class_name", object_dir.name.split(default_category_split, 1)[0])),
            "object_dir": object_dir,
            "metadata": metadata,
            "image_path": first_existing_image(object_dir),
        }
    return lookup


def build_ycbv_lookup(freepose_root: Path, align_json: Path) -> Dict[int, Dict[str, Any]]:
    object_root = freepose_root / "datasets_real" / "ycbv" / "ycbv_aligned_object_refs"
    align = load_json(align_json, {})
    categories = (
        align.get("datasets", {})
        .get("ycbv", {})
        .get("obj_id_to_category", {})
    )
    lookup: Dict[int, Dict[str, Any]] = {}
    dirs = sorted(path for path in object_root.glob("obj_*") if path.is_dir()) if object_root.is_dir() else []
    for object_dir in dirs:
        try:
            object_id = int(object_dir.name.removeprefix("obj_"))
        except ValueError:
            continue
        lookup[object_id] = {
            "dataset": "YCBVCameraPose",
            "object_id": object_id,
            "object_name": object_dir.name,
            "category": str(categories.get(str(object_id), "")),
            "object_dir": object_dir,
            "metadata": {},
            "image_path": first_existing_image(object_dir),
        }
    return lookup


def build_lookups(freepose_root: Path, align_json: Path) -> Dict[str, Dict[int, Dict[str, Any]]]:
    return {
        "OO9DSingleCameraPose": build_oo9d_lookup(freepose_root),
        "Real275CameraPose": build_sorted_ref_lookup(
            "Real275CameraPose",
            freepose_root / "real275" / "real275_aligned_object_refs",
            "_",
        ),
        "HouseCat6DCameraPose": build_sorted_ref_lookup(
            "HouseCat6DCameraPose",
            freepose_root / "housecat6d" / "housecat6d_aligned_object_refs",
            "-",
        ),
        "YCBVCameraPose": build_ycbv_lookup(freepose_root, align_json),
    }


def copy_or_link_image(source: Optional[Path], destination: Path, symlink: bool) -> str:
    if source is None or not source.is_file():
        return ""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if symlink:
        destination.symlink_to(source)
    else:
        shutil.copy2(source, destination)
    return destination.name


def symmetry_summary(info: Dict[str, Any]) -> Dict[str, Any]:
    discrete = info.get("symmetries_discrete") or []
    continuous = info.get("symmetries_continuous") or []
    axes = [item.get("axis") for item in continuous if isinstance(item, dict)]
    return {
        "num_discrete": len(discrete),
        "num_continuous": len(continuous),
        "continuous_axes": axes,
    }


def axis_angle_rotation(axis: Iterable[float], angle: float) -> np.ndarray:
    axis_arr = np.asarray(list(axis), dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis_arr))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float32)
    x, y, z = axis_arr / norm
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    one_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float32,
    )


def parse_discrete_rotation(value: Any) -> Optional[np.ndarray]:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape == (3, 3):
        return arr
    if arr.shape == (4, 4):
        return arr[:3, :3]
    flat = arr.reshape(-1)
    if flat.size == 9:
        return flat.reshape(3, 3)
    if flat.size == 16:
        return flat.reshape(4, 4)[:3, :3]
    return None


def symmetry_rotations(info: Dict[str, Any], continuous_steps: int) -> list[tuple[str, np.ndarray]]:
    rotations: list[tuple[str, np.ndarray]] = [("identity", np.eye(3, dtype=np.float32))]

    for idx, item in enumerate(info.get("symmetries_discrete") or []):
        rot = parse_discrete_rotation(item)
        if rot is not None:
            rotations.append((f"discrete {idx}", rot.astype(np.float32)))

    for axis_idx, item in enumerate(info.get("symmetries_continuous") or []):
        if not isinstance(item, dict):
            continue
        axis = item.get("axis", [0, 1, 0])
        steps = max(1, int(continuous_steps))
        for step in range(steps):
            angle = 2.0 * math.pi * float(step) / float(steps)
            degrees = int(round(math.degrees(angle))) % 360
            label = f"axis {axis_idx} {degrees}deg"
            rotations.append((label, axis_angle_rotation(axis, angle)))

    return rotations


def project_axis_point(point: np.ndarray, center: tuple[float, float], scale: float) -> tuple[float, float]:
    x, y, z = [float(v) for v in point]
    u = center[0] + scale * (0.92 * x + 0.42 * z)
    v = center[1] + scale * (-0.92 * y + 0.28 * z)
    return u, v


def draw_rotation_panel(label: str, rotation: np.ndarray, size: tuple[int, int]) -> Image.Image:
    panel = Image.new("RGB", size, (248, 248, 245))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    w, h = size
    center = (w * 0.5, h * 0.58)
    scale = min(w, h) * 0.32
    origin = project_axis_point(np.zeros(3, dtype=np.float32), center, scale)
    axes = {
        "x": rotation @ np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        "y": rotation @ np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        "z": rotation @ np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    }
    draw.rectangle((0, 0, w - 1, h - 1), outline=(210, 210, 205))
    draw.text((8, 8), label[:32], fill=(20, 20, 20), font=font)
    for axis_name, point in axes.items():
        end = project_axis_point(point, center, scale)
        color = AXIS_COLORS[axis_name]
        draw.line([origin, end], fill=color, width=4)
        draw.ellipse((end[0] - 4, end[1] - 4, end[0] + 4, end[1] + 4), fill=color)
        draw.text((end[0] + 5, end[1] - 6), axis_name.upper(), fill=color, font=font)
    draw.ellipse((origin[0] - 4, origin[1] - 4, origin[0] + 4, origin[1] + 4), fill=(20, 20, 20))
    return panel


def draw_preview_panel(image_path: Optional[Path], metadata: Dict[str, Any], size: tuple[int, int]) -> Image.Image:
    panel = Image.new("RGB", size, (245, 245, 242))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    w, h = size
    if image_path is not None and image_path.is_file():
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((w - 28, h - 74), Image.Resampling.LANCZOS)
        x = (w - image.width) // 2
        y = 40
        panel.paste(image, (x, y))
    draw.rectangle((0, 0, w - 1, h - 1), outline=(190, 190, 185))
    lines = [
        str(metadata.get("symmetry_key", "")),
        f"{metadata.get('category', '')} / {metadata.get('object_name', '')}",
    ]
    y = 8
    for line in lines:
        draw.text((8, y), line[:36], fill=(20, 20, 20), font=font)
        y += 14
    return panel


def write_symmetry_preview(
    image_path: Optional[Path],
    metadata: Dict[str, Any],
    sym_info: Dict[str, Any],
    output_path: Path,
    continuous_steps: int,
    panel_size: tuple[int, int] = (220, 180),
    columns: int = 4,
) -> str:
    rotations = symmetry_rotations(sym_info, continuous_steps)
    panels = [draw_preview_panel(image_path, metadata, panel_size)]
    if len(rotations) == 1 and not (sym_info.get("symmetries_discrete") or sym_info.get("symmetries_continuous")):
        panels.append(draw_rotation_panel("no symmetry", np.eye(3, dtype=np.float32), panel_size))
    else:
        panels.extend(draw_rotation_panel(label, rot, panel_size) for label, rot in rotations)

    pad = 8
    columns = max(1, int(columns))
    rows = int(math.ceil(len(panels) / columns))
    sheet = Image.new(
        "RGB",
        (columns * panel_size[0] + (columns - 1) * pad, rows * panel_size[1] + (rows - 1) * pad),
        (35, 35, 35),
    )
    for idx, panel in enumerate(panels):
        x = (idx % columns) * (panel_size[0] + pad)
        y = (idx // columns) * (panel_size[1] + pad)
        sheet.paste(panel, (x, y))
    sheet.save(output_path)
    return output_path.name


def export_catalog(args: argparse.Namespace) -> None:
    symmetry_info = load_json(args.symmetry_info, {})
    lookups = build_lookups(args.freepose_root, args.align_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    items = sorted(symmetry_info.items(), key=lambda item: item[0])
    if args.limit is not None:
        items = items[: int(args.limit)]

    for key, sym_info in items:
        if ":" not in key:
            dataset_label, object_id_text = "", key
        else:
            dataset_label, object_id_text = key.split(":", 1)
        try:
            object_id = int(object_id_text)
        except ValueError:
            object_id = -1

        object_info = lookups.get(dataset_label, {}).get(object_id, {})
        object_name = str(object_info.get("object_name", ""))
        category = str(object_info.get("category", ""))
        object_dir = object_info.get("object_dir")
        image_path = object_info.get("image_path")
        summary = symmetry_summary(sym_info)

        folder_name = f"{safe_name(dataset_label)}__{object_id:06d}__{safe_name(category)}__{safe_name(object_name)}"
        entry_dir = args.output_dir / folder_name
        entry_dir.mkdir(parents=True, exist_ok=True)
        preview_name = copy_or_link_image(image_path, entry_dir / "preview.png", args.symlink)

        metadata = {
            "symmetry_key": key,
            "dataset": dataset_label,
            "object_id": object_id,
            "object_name": object_name,
            "category": category,
            "object_dir": str(object_dir) if object_dir else "",
            "preview_source_path": str(image_path) if image_path else "",
            "preview_file": preview_name,
            **summary,
            "symmetry_info": sym_info,
        }
        symmetry_preview_name = write_symmetry_preview(
            image_path,
            metadata,
            sym_info,
            entry_dir / "symmetry_rotations.png",
            args.symmetry_preview_steps,
            columns=args.symmetry_preview_columns,
        )
        metadata["symmetry_preview_file"] = symmetry_preview_name
        with (entry_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

        rows.append({
            "symmetry_key": key,
            "dataset": dataset_label,
            "object_id": object_id,
            "category": category,
            "object_name": object_name,
            "entry_dir": str(entry_dir),
            "preview_file": preview_name,
            "symmetry_preview_file": symmetry_preview_name,
            "preview_source_path": str(image_path) if image_path else "",
            "num_discrete": summary["num_discrete"],
            "num_continuous": summary["num_continuous"],
            "continuous_axes": json.dumps(summary["continuous_axes"]),
        })

    with (args.output_dir / "index.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    with (args.output_dir / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print(f"Exported {len(rows)} entries to {args.output_dir}")
    print(f"Index JSON: {args.output_dir / 'index.json'}")
    print(f"Index CSV:  {args.output_dir / 'index.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symmetry-info", type=Path, default=DEFAULT_SYMMETRY_INFO)
    parser.add_argument("--freepose-root", type=Path, default=DEFAULT_FREEPOSE_ROOT)
    parser.add_argument("--align-json", type=Path, default=DEFAULT_ALIGN_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Export only the first N entries for a quick check.")
    parser.add_argument("--symlink", action="store_true", help="Symlink preview images instead of copying them.")
    parser.add_argument("--symmetry-preview-steps", type=int, default=8, help="Number of sampled rotations for each continuous symmetry axis.")
    parser.add_argument("--symmetry-preview-columns", type=int, default=4, help="Number of columns in each symmetry_rotations.png sheet.")
    args = parser.parse_args()
    export_catalog(args)


if __name__ == "__main__":
    main()
