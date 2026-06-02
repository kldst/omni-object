"""Omni6DPose (ROPE / SOPE) camera-pose dataset with PAM-aligned object refs.

Omni6DPose stores per-frame ground truth in ``{scene}/{frame:06d}_meta.json``:
each object carries a ``quaternion_wxyz`` + ``translation`` that maps the object
canonical frame to camera space (``p_cam = R @ p_obj + t``), where the object
canonical frame is exactly the PAM mesh ``object_meshes/<NAME>/Aligned.obj``.

Because the PAM reference views are rendered from that same ``Aligned.obj`` frame
(see ``render_omni6dpose_object_refs_bpy.py``), no per-class ``R_align`` is needed:
``R_align`` is the identity and the model's predicted rotation lands directly in
the GT frame. This mirrors the HouseCat6D loader (which *does* realign) so the
downstream eval code can stay almost identical.

Object identity -> reference folder is resolved through ``oid`` (e.g.
``real-chess_001`` / ``omniobject3d-waffle_001``). For ROPE real objects whose
scanned mesh is absent from PAM, the same ``<class>_<instance>`` synthetic mesh
is used (resolved upstream into ``oid_to_pam.json``); objects with no resolvable
reference are skipped, exactly like HouseCat6D's ``skipped_no_ref``.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch
from PIL import Image

from omnivggt.datasets.base.base_stereo_view_dataset import (
    BaseStereoViewDataset,
    is_good_type,
    transpose_to_landscape,
    view_name,
)
from omnivggt.datasets.base.batched_sampler import BatchedRandomSampler
from omnivggt.utils.geometry import depthmap_to_absolute_camera_coordinates


logger = logging.getLogger(__name__)


def quaternion_wxyz_to_matrix(quat_wxyz: Sequence[float]) -> np.ndarray:
    """Convert a scale-first (wxyz) quaternion to a 3x3 rotation matrix."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


class Omni6DPoseCameraPose(BaseStereoViewDataset):
    """Omni6DPose ROPE/SOPE loader producing OV9D-aligned object-pose samples."""

    DEFAULT_DATA_ROOT = "/mnt/train-data-4-hdd/yian/freepose/Omni6dpose/Omni6DPoseAPI/data/Omni6DPose/ROPE"
    DEFAULT_OBJECT_IMAGE_ROOT = (
        "/mnt/train-data-4-hdd/yian/freepose/omni-object_clone/outputs/omni6dpose_refs/diverse24"
    )
    DEFAULT_OID_TO_PAM = (
        "/mnt/train-data-4-hdd/yian/freepose/omni-object_clone/outputs/omni6dpose_refs/rope_oid_to_pam.json"
    )
    DEFAULT_OBJECT_VIEW_IDS = (0, 5, 8, 19)

    def __init__(
        self,
        dataset_location: str = DEFAULT_DATA_ROOT,
        dset: str = "test",
        object_image_root: Optional[str] = None,
        oid_to_pam_json: Optional[str] = None,
        fixed_object_view_ids: Optional[Sequence[int]] = DEFAULT_OBJECT_VIEW_IDS,
        num_object_views: int = 4,
        strict_fixed_object_view_ids: bool = True,
        verify_files: bool = True,
        max_records: Optional[int] = None,
        only_scene_name: str = "",
        only_object_name: str = "",
        scene_glob: str = "[0-9]" * 6,
        scenes: Optional[Sequence[str]] = None,
        layout: str = "flat",
        patches: Optional[Sequence[str]] = None,
        split: str = "train",
        expand_records_by_object: bool = True,
        object_presence_prob: float = 1.0,
        normalize_object_translation_by_depth_mean: bool = True,
        depth_mean_eps: float = 1e-6,
        *args,
        **kwargs,
    ):
        super().__init__(dset=dset, *args, **kwargs)
        self.dataset_label = "Omni6DPoseCameraPose"
        self.dataset_location = Path(dataset_location)
        self.object_image_root = Path(object_image_root or self.DEFAULT_OBJECT_IMAGE_ROOT)
        self.fixed_object_view_ids = tuple(int(x) for x in fixed_object_view_ids) if fixed_object_view_ids else None
        self.num_object_views = int(num_object_views)
        self.strict_fixed_object_view_ids = bool(strict_fixed_object_view_ids)
        self.verify_files = bool(verify_files)
        self.max_records = int(max_records) if max_records is not None else None
        self.only_scene_name = str(only_scene_name).strip()
        self.only_object_name = str(only_object_name).strip()
        self.scene_glob = str(scene_glob)
        self.scenes_filter = set(str(s) for s in scenes) if scenes else None
        # layout='flat': scene dirs directly under dataset_location matching scene_glob (ROPE).
        # layout='sope': nested dataset_location/<patch>/<split>/<source>/<scene>/ (SOPE).
        self.layout = str(layout)
        self.patches = [str(p) for p in patches] if patches is not None else None
        self.split = str(split)
        self.expand_records_by_object = bool(expand_records_by_object)
        self.object_presence_prob = float(object_presence_prob)
        self.normalize_object_translation_by_depth_mean = bool(normalize_object_translation_by_depth_mean)
        self.depth_mean_eps = float(depth_mean_eps)

        # oid -> reference folder name. SOPE: oid IS the ref dir name (identity).
        # ROPE: pass oid_to_pam_json mapping real-* -> the proxy mesh used for refs.
        self.oid_to_pam: Dict[str, str] = {}
        if oid_to_pam_json is not None:
            mapping = self._load_json(Path(oid_to_pam_json))
            self.oid_to_pam = dict(mapping.get("oid_to_pam", mapping))

        self.object_records_by_name = self._build_object_records_by_name()
        self.object_name_to_id = {
            name: idx + 1 for idx, name in enumerate(sorted(self.object_records_by_name.keys()))
        }
        self.object_id_to_name = {idx: name for name, idx in self.object_name_to_id.items()}
        self.records = self._build_records()
        self.scenes = self.records
        if not self.records:
            raise RuntimeError(
                f"No Omni6DPose samples found. root={self.dataset_location} "
                f"object_image_root={self.object_image_root}"
            )
        logger.info(
            "Omni6DPoseCameraPose initialized: records=%d objects=%d object_views=%s",
            len(self.records),
            len(self.object_records_by_name),
            self.fixed_object_view_ids,
        )

    # ------------------------------------------------------------------ io utils
    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _read_depth_m(depth_path: Path) -> np.ndarray:
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Failed to read depth EXR: {depth_path}")
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth = depth.astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        depth[depth < 0.0] = 0.0
        return depth

    # ----------------------------------------------------------- object refs
    def _build_object_records_by_name(self) -> Dict[str, Dict[str, Any]]:
        if not self.object_image_root.is_dir():
            raise FileNotFoundError(f"Omni6DPose object image root not found: {self.object_image_root}")

        image_ids = [int(x) for x in (self.fixed_object_view_ids or self.DEFAULT_OBJECT_VIEW_IDS)]
        records: Dict[str, Dict[str, Any]] = {}
        missing_fixed = []
        for object_dir in sorted(self.object_image_root.iterdir()):
            if not object_dir.is_dir():
                continue
            rgb_dir = object_dir / "rgb"
            if not rgb_dir.is_dir():
                continue
            available_ids = sorted(int(p.stem) for p in rgb_dir.glob("*.png") if p.stem.isdigit())
            if not available_ids:
                continue
            if self.fixed_object_view_ids is not None and self.strict_fixed_object_view_ids:
                missing = [i for i in image_ids if i not in available_ids]
                if missing:
                    missing_fixed.append((object_dir.name, missing))
                    continue
                selected_ids = image_ids
            else:
                selected_ids = image_ids if self.fixed_object_view_ids is not None else available_ids[: self.num_object_views]
            metadata_path = object_dir / "metadata.json"
            metadata = self._load_json(metadata_path) if metadata_path.is_file() else {}
            records[object_dir.name] = {
                "object_name": object_dir.name,
                "object_dir": object_dir,
                "image_ids": [int(x) for x in selected_ids],
                "metadata": metadata,
            }
        if missing_fixed:
            logger.warning(
                "Skipped %d Omni6DPose object refs missing fixed views. First: %s",
                len(missing_fixed),
                missing_fixed[:5],
            )
        return records

    def _iter_meta_files(self):
        """Yield (scene_name, meta_path). scene_name is unique per scene dir."""
        if self.layout == "sope":
            patches = self.patches if self.patches is not None else \
                sorted(p.name for p in self.dataset_location.iterdir() if p.is_dir())
            for patch in patches:
                base = self.dataset_location / patch / self.split
                if not base.is_dir():
                    continue
                # <patch>/<split>/<source>/<scene>/<frame>_meta.json
                for meta_path in sorted(base.glob("*/*/*_meta.json")):
                    scene_dir = meta_path.parent
                    scene_name = str(scene_dir.relative_to(self.dataset_location))
                    if self.only_scene_name and scene_name != self.only_scene_name:
                        continue
                    if self.scenes_filter is not None and scene_name not in self.scenes_filter:
                        continue
                    yield scene_name, scene_dir, meta_path
        else:
            for scene_dir in sorted(self.dataset_location.glob(self.scene_glob)):
                if not scene_dir.is_dir():
                    continue
                if self.only_scene_name and scene_dir.name != self.only_scene_name:
                    continue
                if self.scenes_filter is not None and scene_dir.name not in self.scenes_filter:
                    continue
                for meta_path in sorted(scene_dir.glob("*_meta.json")):
                    yield scene_dir.name, scene_dir, meta_path

    def _build_records(self) -> List[Dict[str, Any]]:
        if not self.dataset_location.is_dir():
            raise FileNotFoundError(f"Omni6DPose root not found: {self.dataset_location}")

        records: List[Dict[str, Any]] = []
        skipped_no_ref = 0
        skipped_missing_file = 0
        skipped_invalid = 0

        for scene_name, scene_dir, meta_path in self._iter_meta_files():
                # frame paths are siblings of the meta file (handles 4- or 6-digit ids)
                stem = meta_path.name[: -len("_meta.json")]
                frame_id = int(stem)
                color_path = meta_path.with_name(f"{stem}_color.png")
                depth_path = meta_path.with_name(f"{stem}_depth.exr")
                if self.verify_files and (not color_path.is_file() or not depth_path.is_file()):
                    skipped_missing_file += 1
                    continue
                meta = self._load_json(meta_path)
                intr = meta["camera"]["intrinsics"]
                objects = meta.get("objects", {})
                obj_items = list(objects.items())
                if not self.expand_records_by_object:
                    obj_items = obj_items[:1]
                for obj_key, obj in obj_items:
                    if not obj.get("is_valid", True):
                        skipped_invalid += 1
                        continue
                    om = obj["meta"]
                    if om.get("is_background", False):
                        continue
                    oid = str(om["oid"])
                    # identity by default (SOPE: oid == ref dir name); mapping for ROPE.
                    pam_name = self.oid_to_pam.get(oid, oid)
                    if pam_name not in self.object_records_by_name:
                        skipped_no_ref += 1
                        continue
                    if self.only_object_name and pam_name != self.only_object_name:
                        continue
                    records.append(
                        {
                            "scene_name": scene_name,
                            "run_name": scene_name,
                            "scene_dir": scene_dir,
                            "image_id": frame_id,
                            "image_ids": [frame_id],
                            "color_path": color_path,
                            "depth_path": depth_path,
                            "meta_path": meta_path,
                            "intrinsics": dict(intr),
                            "obj_key": obj_key,
                            "oid": oid,
                            "object_name": pam_name,
                            "object_id": self.object_name_to_id[pam_name],
                            "class_id": int(om.get("class_label", 0)),
                            "category": str(om.get("class_name", "")),
                            "quaternion_wxyz": [float(v) for v in obj["quaternion_wxyz"]],
                            "translation": [float(v) for v in obj["translation"]],
                            "bbox_side_len": [float(v) for v in om.get("bbox_side_len", [1.0, 1.0, 1.0])],
                            "scene_source": "omni6dpose",
                        }
                    )
                    if self.max_records is not None and len(records) >= self.max_records:
                        logger.info("Omni6DPose records capped at max_records=%d", self.max_records)
                        return records

        logger.info(
            "Omni6DPose records built: total=%d skipped_no_ref=%d skipped_missing_file=%d skipped_invalid=%d",
            len(records),
            skipped_no_ref,
            skipped_missing_file,
            skipped_invalid,
        )
        return records

    # ------------------------------------------------------------- scene view
    def _scaled_intrinsic(self, intr: Dict[str, Any], width: int, height: int) -> np.ndarray:
        scale_x = width / float(intr["width"])
        scale_y = height / float(intr["height"])
        return np.array(
            [
                [float(intr["fx"]) * scale_x, 0.0, float(intr["cx"]) * scale_x],
                [0.0, float(intr["fy"]) * scale_y, float(intr["cy"]) * scale_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def _load_scene_view(self, rec: Dict[str, Any], resolution, rng):
        image = Image.open(rec["color_path"]).convert("RGB")
        depthmap = self._read_depth_m(rec["depth_path"])
        width, height = image.size
        if depthmap.shape[:2] != (height, width):
            depthmap = cv2.resize(depthmap, (width, height), interpolation=cv2.INTER_NEAREST)
        intrinsic = self._scaled_intrinsic(rec["intrinsics"], width, height)

        image, depthmap, intrinsic = self._crop_resize_if_necessary(
            image, depthmap, intrinsic, resolution, rng, info=str(rec["color_path"])
        )
        extrinsic = np.concatenate([np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)], axis=1)
        _, point_mask = depthmap_to_absolute_camera_coordinates(depthmap, intrinsic, extrinsic, z_far=self.z_far)
        return {
            "img": image,
            "depthmap": depthmap.astype(np.float32),
            "camera_pose": extrinsic.astype(np.float32),
            "camera_intrinsics": intrinsic.astype(np.float32),
            "point_mask": point_mask,
            "label": rec["scene_name"],
            "instance": rec["color_path"].name,
            "dataset": self.dataset_label,
            "image_path": str(rec["color_path"]),
            "depth_path": str(rec["depth_path"]),
            "camera_path": str(rec["meta_path"]),
        }

    # ----------------------------------------------------------- object views
    @staticmethod
    def _resize_image(image: Image.Image, resolution) -> Image.Image:
        width, height = resolution
        resampling = getattr(Image, "Resampling", Image)
        return image.resize((width, height), resampling.LANCZOS)

    def _sample_object_view_ids(self, available_ids: List[int], rng, object_name: str = "") -> List[int]:
        available_ids = [int(x) for x in available_ids]
        if self.fixed_object_view_ids is not None:
            missing = [int(x) for x in self.fixed_object_view_ids if int(x) not in available_ids]
            if missing and self.strict_fixed_object_view_ids:
                raise RuntimeError(
                    f"Missing fixed object views for {object_name}: missing={missing} available={available_ids}"
                )
            return [int(x) for x in self.fixed_object_view_ids if int(x) in available_ids]
        count = min(self.num_object_views, len(available_ids))
        return available_ids[:count]

    def _sample_absent_object_name(self, rng, positive_object_name: str) -> Optional[str]:
        """Pick a reference object different from the present one (negative example)."""
        candidates = [n for n in self.object_records_by_name if n != str(positive_object_name)]
        if self.only_object_name:
            candidates = [n for n in candidates if n == self.only_object_name]
        if not candidates:
            return None
        return str(rng.choice(np.asarray(sorted(candidates), dtype=object)))

    @staticmethod
    def _size_from_ref_metadata(metadata: Dict[str, Any]) -> np.ndarray:
        bounds = np.asarray(metadata.get("mesh", {}).get("centered_bounds", []), dtype=np.float32)
        if bounds.shape == (2, 3):
            return np.clip(bounds[1] - bounds[0], 1e-6, None).astype(np.float32)
        return np.ones(3, dtype=np.float32)

    def _load_object_images(self, object_name: str, resolution, rng) -> Dict[str, Any]:
        object_rec = self.object_records_by_name[object_name]
        image_ids = self._sample_object_view_ids(object_rec["image_ids"], rng, object_name=object_name)
        tensors, true_shapes, image_paths = [], [], []
        for image_id in image_ids:
            image_path = object_rec["object_dir"] / "rgb" / f"{image_id:06d}.png"
            image = Image.open(image_path).convert("RGB")
            true_shapes.append(np.array(image.size[::-1], dtype=np.int32))
            tensors.append(self.transform(self._resize_image(image, resolution)))
            image_paths.append(str(image_path))
        return {
            "object_images": torch.stack(tensors),
            "object_true_shape": np.stack(true_shapes),
            "object_cam_indices": np.asarray(image_ids, dtype=np.int64),
            "object_reference_scene_name": object_name,
            "object_rgb_paths": image_paths,
            "object_mask_paths": [""] * len(image_paths),
        }

    # ----------------------------------------------------------------- item
    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx, ar_idx, *_ = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0
        if self.seed:
            rng = np.random.default_rng(seed=self.seed + int(idx))
        else:
            rng = np.random.default_rng(seed=torch.initial_seed())

        resolution = self._resolutions[ar_idx]
        rec = self.records[int(idx) % len(self.records)]

        # Object-presence sampling: with prob (1 - object_presence_prob) replace the
        # target with an absent reference object (negative example for the presence head).
        target_object_name = str(rec["object_name"])
        target_oid = str(rec["oid"])
        target_object_id = int(rec["object_id"])
        target_class_id = int(rec["class_id"])
        target_category = str(rec["category"])
        has_object = True
        if (self.dset not in {"test", "val", "validation"} and self.object_presence_prob < 1.0
                and float(rng.random()) > self.object_presence_prob):
            absent = self._sample_absent_object_name(rng, rec["object_name"])
            if absent is not None:
                target_object_name = absent
                target_oid = absent
                target_object_id = int(self.object_name_to_id[absent])
                target_category = str(absent.split("-", 1)[-1].rsplit("_", 1)[0])
                target_class_id = 0
                has_object = False

        view = self._load_scene_view(rec, resolution, rng)
        view["object_mask"] = np.zeros_like(view["point_mask"], dtype=np.bool_)
        view["idx"] = (idx, ar_idx, 0)
        view["z_far"] = self.z_far
        view["true_shape"] = np.int32(view["img"].size[::-1])
        view["img"] = self.transform(view["img"])
        for key, value in view.items():
            res, err_msg = is_good_type(key, value)
            assert res, f"{err_msg} with {key}={value} for view {view_name(view)}"
        transpose_to_landscape(view)
        view["rng"] = int.from_bytes(rng.bytes(4), "big")

        # Pose: object canonical (= PAM Aligned.obj) -> camera. R_align is identity.
        if has_object:
            object_rotation = quaternion_wxyz_to_matrix(rec["quaternion_wxyz"])
            object_translation_metric = np.asarray(rec["translation"], dtype=np.float32).reshape(3)
            object_size = np.clip(np.asarray(rec["bbox_side_len"], dtype=np.float32).reshape(3), 1e-6, None)
        else:
            object_rotation = np.eye(3, dtype=np.float32)
            object_translation_metric = np.zeros(3, dtype=np.float32)
            object_size = self._size_from_ref_metadata(
                self.object_records_by_name[target_object_name].get("metadata", {})
            )
        object_size_log = np.log(object_size).astype(np.float32)
        r_align = np.eye(3, dtype=np.float32)

        depth = view["depthmap"]
        valid_depth = depth[np.asarray(view["point_mask"], dtype=np.bool_)]
        if valid_depth.size == 0:
            depth_mean_scale = np.float32(1.0)
        else:
            depth_mean_scale = np.float32(max(float(valid_depth.mean()), self.depth_mean_eps))
        object_translation_normalized = (object_translation_metric / depth_mean_scale).astype(np.float32)
        object_translation = (
            object_translation_normalized if self.normalize_object_translation_by_depth_mean
            else object_translation_metric
        )

        image_id = int(rec["image_id"])
        object_id = int(target_object_id)
        result = {
            "images": torch.stack([view["img"]]),
            "depth": np.stack([view["depthmap"][:, :, None]]),
            "extrinsic": np.stack([view["camera_pose"][:3]]),
            "intrinsic": np.stack([view["camera_intrinsics"]]),
            "true_shape": np.stack([view["true_shape"]]),
            "valid_mask": np.stack([view["point_mask"]]),
            "object_masks": np.stack([view["object_mask"]]),
            "label": [view["label"]],
            "instance": [view["instance"]],
            "dataset": self.dataset_label,
            "ids": np.array([image_id], dtype=np.int64),
            "camera_indices": np.array([image_id], dtype=np.int64),
            "seq_name": f"omni6dpose/{rec['scene_name']}/{target_oid}/{image_id:06d}",
            "scene_name": rec["scene_name"],
            "scene_source": rec["scene_source"],
            "run_name": rec["run_name"],
            "object_name": target_object_name,
            "oid": target_oid,
            "obj_key": rec["obj_key"],
            "object_id": np.array(object_id, dtype=np.int64),
            "class_id": np.array(int(target_class_id), dtype=np.int64),
            "category": target_category,
            "has_object": np.array(has_object, dtype=np.bool_),
            "object_rotation": object_rotation.astype(np.float32),
            "object_translation": object_translation.astype(np.float32),
            "object_translation_metric": object_translation_metric.astype(np.float32),
            "object_translation_normalized": object_translation_normalized.astype(np.float32),
            "depth_mean_scale": np.asarray(depth_mean_scale, dtype=np.float32),
            "normalization_scale": np.asarray(depth_mean_scale, dtype=np.float32),
            "object_translation_scale": np.asarray(depth_mean_scale, dtype=np.float32),
            "object_size": object_size,
            "object_size_log": object_size_log,
            "object_srt": np.concatenate([object_rotation.reshape(-1), object_translation, object_size_log]).astype(np.float32),
            "object_rotation_native": object_rotation.astype(np.float32),
            "object_size_native": object_size,
            "R_align_omni6dpose_to_ov9d": r_align,
            "scene_rgb_path": view["image_path"],
            "scene_depth_path": view["depth_path"],
            "scene_camera_path": view["camera_path"],
            "scene_meta_path": str(rec["meta_path"]),
            "bbox_xyxy": np.zeros(4, dtype=np.float32),
        }
        result.update(self._load_object_images(target_object_name, resolution, rng))
        return result

    def make_sampler(self, batch_size, shuffle=True, world_size=1, rank=0, drop_last=True):
        return BatchedRandomSampler(
            self,
            batch_size=batch_size,
            pool_size=len(self._resolutions),
            world_size=world_size,
            rank=rank,
            drop_last=drop_last,
        )
