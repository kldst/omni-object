"""Evaluate ``outputs/0521/10000/model.safetensors`` on the real test splits.

Per dataset, computes the fraction of samples that meet each of the four
rotation/translation thresholds (5°/2 cm, 5°/5 cm, 10°/2 cm, 10°/5 cm).

Predicted translations come out of the network normalized by depth.mean(); they
are multiplied by ``depth_mean_scale`` (in metres) and converted to centimetres
before being compared to the metric GT translations.

Rotation error uses :data:`mixed_symmetry_info.json` so that symmetric objects
score the minimum error across their stored rotations. Continuous rotation
symmetries are sampled at ``--continuous-steps`` (default 360, i.e. 1°).

Splits evaluated (one row each):
  - ycbv_test           : datasets_real/ycbv/test
  - housecat_test_scene1..5 : housecat6d/test_scene{1..5}
  - real275_real_test   : real275/real_test
  - oo9d_unseen         : oo9d single test_unseen_category_unseen_object split

Multi-GPU: pass ``--gpus 0,1,2,3`` to fan out as one subprocess per GPU; each
worker processes idx % num_shards == shard_index and writes its own jsonl. The
orchestrator merges shard outputs and prints per-split metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file as load_safetensors_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from omnivggt.datasets.base.base_stereo_view_dataset import (
    BaseStereoViewDataset,
    is_good_type,
    transpose_to_landscape,
    view_name,
)
from omnivggt.datasets.housecat6d.housecat6d_camera_pose import HouseCat6DCameraPose
from omnivggt.datasets.oo9d.oo9d_single_camera_pose import OO9DSingleCameraPose
from omnivggt.datasets.real275.real275_camera_pose import Real275CameraPose
from omnivggt.datasets.utils.transforms import ImgNorm
from omnivggt.datasets.ycbv.ycbv_camera_pose import YCBVCameraPose
from omnivggt.loss import _load_symmetry_info
from omnivggt.models.omnivggt import OmniVGGT
from omnivggt.utils.geometry import depthmap_to_absolute_camera_coordinates


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "0521" / "10000" / "model.safetensors"
DEFAULT_SYMMETRY_INFO = PROJECT_ROOT / "mixed_symmetry_info.json"
DEFAULT_ALIGN_JSON = PROJECT_ROOT / "dataset_align.json"

DEFAULT_YCBV_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose/datasets_real/ycbv")
DEFAULT_HOUSECAT_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose/housecat6d")
DEFAULT_REAL275_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose/real275")
DEFAULT_OO9D_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d")
DEFAULT_OO9D_OBJECT_REFS = Path("/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d_around_image")
DEFAULT_OO9D_SPLIT_JSON = (
    PROJECT_ROOT
    / "splits_ov9d_unseen_category_generalization"
    / "single"
    / "test_unseen_category_unseen_object.json"
)

DEFAULT_OBJECT_VIEWS = (1, 5, 10, 15)
DEFAULT_RESOLUTION = (518, 476)
DEFAULT_CONTINUOUS_STEPS = 360
THRESHOLDS = (
    ("5deg_2cm", 5.0, 2.0),
    ("5deg_5cm", 5.0, 5.0),
    ("10deg_2cm", 10.0, 2.0),
    ("10deg_5cm", 10.0, 5.0),
)
SPLITS_ORDER = (
    "ycbv_test",
    "housecat_test_scene1",
    "housecat_test_scene2",
    "housecat_test_scene3",
    "housecat_test_scene4",
    "housecat_test_scene5",
    "real275_real_test",
    "oo9d_unseen",
)
DATASET_LABEL_BY_SPLIT = {
    "ycbv_test": "YCBVCameraPose",
    "housecat_test_scene1": "HouseCat6DCameraPose",
    "housecat_test_scene2": "HouseCat6DCameraPose",
    "housecat_test_scene3": "HouseCat6DCameraPose",
    "housecat_test_scene4": "HouseCat6DCameraPose",
    "housecat_test_scene5": "HouseCat6DCameraPose",
    "real275_real_test": "Real275CameraPose",
    "oo9d_unseen": "OO9DSingleCameraPose",
}


# ============================================================================
# Dataset subclasses needed to handle test-split file naming differences.
# ============================================================================
class HouseCat6DTestSceneCameraPose(HouseCat6DCameraPose):
    """Like ``HouseCat6DCameraPose`` but globs ``test_scene*`` instead of ``scene*``.

    Use ``only_scene_name='test_scene3'`` etc. to evaluate one scene at a time.
    """

    SCENE_GLOB = "test_scene*"

    def _build_records(self):
        import logging
        logger = logging.getLogger("HouseCat6DTestSceneCameraPose")
        if not self.dataset_location.is_dir():
            raise FileNotFoundError(f"HouseCat6D root not found: {self.dataset_location}")

        records: List[Dict[str, Any]] = []
        skipped_no_ref = 0
        skipped_missing_file = 0

        for scene_dir in sorted(self.dataset_location.glob(self.SCENE_GLOB)):
            if not scene_dir.is_dir() or not (scene_dir / "meta.txt").is_file():
                continue
            scene_name = scene_dir.name
            if self.only_scene_name and scene_name != self.only_scene_name:
                continue
            intrinsics_path = scene_dir / "intrinsics.txt"
            if self.verify_files and not intrinsics_path.is_file():
                skipped_missing_file += 1
                continue
            scene_meta = self._load_scene_meta(scene_dir)

            for label_path in sorted((scene_dir / "labels").glob("*_label.pkl")):
                image_id = self._frame_id_from_label_path(label_path)
                rgb_path = scene_dir / "rgb" / f"{image_id:06d}.png"
                depth_path = scene_dir / "depth" / f"{image_id:06d}.png"
                if self.verify_files and (not rgb_path.is_file() or not depth_path.is_file()):
                    skipped_missing_file += 1
                    continue
                label = self._load_label(label_path)
                model_list = [str(x) for x in label["model_list"]]
                object_indices = list(range(len(model_list)))
                if not self.expand_records_by_object:
                    object_indices = object_indices[:1]

                for object_index in object_indices:
                    object_name = model_list[object_index]
                    if self.only_object_name and object_name != self.only_object_name:
                        continue
                    meta = scene_meta.get(object_name)
                    class_id = int(label["class_ids"][object_index])
                    category = str(
                        meta["category"] if meta else self.category_id_to_name.get(class_id, object_name.split("-", 1)[0])
                    )
                    if self.only_category and category != self.only_category:
                        continue
                    if object_name not in self.object_records_by_name:
                        skipped_no_ref += 1
                        continue
                    mask_path = scene_dir / "instance" / f"{image_id:06d}_{object_name}.png"
                    if self.verify_files and not mask_path.is_file():
                        mask_path = None
                    records.append(
                        {
                            "scene_name": scene_name,
                            "run_name": scene_name,
                            "scene_dir": scene_dir,
                            "image_id": image_id,
                            "image_ids": [image_id],
                            "label_path": label_path,
                            "intrinsics_path": intrinsics_path,
                            "rgb_path": rgb_path,
                            "depth_path": depth_path,
                            "mask_path": mask_path,
                            "object_index": object_index,
                            "object_name": object_name,
                            "object_id": self.object_name_to_id[object_name],
                            "instance_id": meta.get("instance_id") if meta else None,
                            "class_id": class_id,
                            "category": category,
                            "scene_source": "housecat6d",
                        }
                    )
                    if self.max_records is not None and len(records) >= self.max_records:
                        return records

        logger.info(
            "HouseCat6D-test records built: total=%d skipped_no_ref=%d skipped_missing_file=%d",
            len(records),
            skipped_no_ref,
            skipped_missing_file,
        )
        return records


# ============================================================================
# REAL275 real_test loader. GT pkls only carry gt_RTs/image_path, so meta.txt
# is parsed for class/instance/name. Rotation column-norm gives the NOCS scale.
# ============================================================================
class Real275RealTestCameraPose(BaseStereoViewDataset):
    """REAL275 real_test evaluation dataset.

    The official real_test GT pickles contain only ``gt_RTs`` (5x4x4) and
    ``image_path``. Per-frame meta.txt is parsed to recover the instance id,
    class id and model name for each entry. The model name in real_test is not
    present in ``real275_aligned_object_refs`` (those are real_train objects);
    a representative same-category training object is used for reference views.
    """

    K_REAL = Real275CameraPose.K_REAL

    def __init__(
        self,
        dataset_location: str,
        split_root: str,
        gt_root: str,
        object_image_root: str,
        align_json: str,
        object_image_root_instance: Optional[str] = None,
        scene_glob: str = "scene_*",
        fixed_object_view_ids: Sequence[int] = DEFAULT_OBJECT_VIEWS,
        num_object_views: int = 4,
        verify_files: bool = True,
        max_records: Optional[int] = None,
        only_scene_name: str = "",
        normalize_object_translation_by_depth_mean: bool = True,
        depth_mean_eps: float = 1e-6,
        z_far: float = 20,
        resolution=DEFAULT_RESOLUTION,
        transform=ImgNorm,
        seed: int = 42,
    ):
        super().__init__(dset="test", resolution=resolution, transform=transform, seed=seed, z_far=z_far)
        self.dataset_label = "Real275CameraPose"
        self.dataset_location = Path(dataset_location)
        self.split_root = Path(split_root)
        self.gt_root = Path(gt_root)
        self.object_image_root = Path(object_image_root)
        self.object_image_root_instance = (
            Path(object_image_root_instance) if object_image_root_instance else None
        )
        self.align_json = Path(align_json)
        self.scene_glob = str(scene_glob)
        self.fixed_object_view_ids = tuple(int(v) for v in fixed_object_view_ids)
        self.num_object_views = int(num_object_views)
        self.verify_files = bool(verify_files)
        self.max_records = int(max_records) if max_records is not None else None
        self.only_scene_name = str(only_scene_name).strip()
        self.normalize_object_translation_by_depth_mean = bool(normalize_object_translation_by_depth_mean)
        self.depth_mean_eps = float(depth_mean_eps)

        align = self._load_json(self.align_json)["datasets"]["real275"]
        self.class_id_to_name = {int(k): str(v) for k, v in align["class_id_to_name"].items()}
        self.r_align_by_class_id = {
            int(class_id): np.asarray(item["R_align"], dtype=np.float32).reshape(3, 3)
            for class_id, item in align["classes"].items()
        }

        self.object_refs_by_class_id = self._build_object_refs_by_class_id()
        self.instance_refs_by_name = self._build_instance_refs_by_name()
        self.records = self._build_records()
        self.scenes = self.records
        if not self.records:
            raise RuntimeError(
                f"No REAL275 real_test samples found. split_root={self.split_root} gt_root={self.gt_root}"
            )

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _load_json(path):
        with Path(path).open("r", encoding="utf-8") as h:
            return json.load(h)

    @staticmethod
    def _load_pickle(path):
        with Path(path).open("rb") as h:
            return pickle.load(h)

    @staticmethod
    def _resize_image(image, resolution):
        resampling = getattr(Image, "Resampling", Image)
        return image.resize(tuple(resolution), resampling.LANCZOS)

    @staticmethod
    def _read_instance_mask(mask_path: Path, inst_id: int) -> np.ndarray:
        raw = np.asarray(Image.open(mask_path), dtype=np.uint8)
        if raw.ndim == 3:
            raw = raw[..., 0]
        return (raw == int(inst_id)).astype(np.float32)

    def _build_instance_refs_by_name(self) -> Dict[str, Path]:
        """Index ``object_image_root_instance`` by model_name. Each folder must
        contain the required ``rgb/<view>.png`` views."""
        out: Dict[str, Path] = {}
        if self.object_image_root_instance is None or not self.object_image_root_instance.is_dir():
            return out
        target_views = list(self.fixed_object_view_ids)
        for obj_dir in sorted(self.object_image_root_instance.iterdir()):
            if not obj_dir.is_dir():
                continue
            rgb_dir = obj_dir / "rgb"
            if not rgb_dir.is_dir():
                continue
            available = {int(p.stem) for p in rgb_dir.glob("*.png") if p.stem.isdigit()}
            if any(v not in available for v in target_views):
                continue
            out[obj_dir.name] = obj_dir
        return out

    def _build_object_refs_by_class_id(self) -> Dict[int, Path]:
        """For each class id, pick the first training-object folder whose
        ``metadata.json`` matches the class name."""
        target_views = list(self.fixed_object_view_ids)
        out: Dict[int, Path] = {}
        if not self.object_image_root.is_dir():
            raise FileNotFoundError(f"REAL275 object image root missing: {self.object_image_root}")
        name_to_class = {name: cid for cid, name in self.class_id_to_name.items()}
        for obj_dir in sorted(self.object_image_root.iterdir()):
            if not obj_dir.is_dir():
                continue
            meta_path = obj_dir / "metadata.json"
            class_id = None
            if meta_path.is_file():
                meta = self._load_json(meta_path)
                class_name = str(meta.get("class_name", ""))
                if class_name in name_to_class:
                    class_id = int(name_to_class[class_name])
            if class_id is None:
                class_name_guess = obj_dir.name.split("_", 1)[0]
                class_id = int(name_to_class.get(class_name_guess, 0))
            if class_id <= 0:
                continue
            rgb_dir = obj_dir / "rgb"
            if not rgb_dir.is_dir():
                continue
            available = {int(p.stem) for p in rgb_dir.glob("*.png") if p.stem.isdigit()}
            if any(v not in available for v in target_views):
                continue
            out.setdefault(class_id, obj_dir)
        return out

    @staticmethod
    def _parse_meta_txt(path: Path) -> Dict[int, Dict[str, Any]]:
        out: Dict[int, Dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as h:
            for line in h:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=2)
                if len(parts) < 3:
                    continue
                inst_id, class_id, model_name = parts
                out[int(inst_id)] = {
                    "inst_id": int(inst_id),
                    "class_id": int(class_id),
                    "model_name": model_name.strip(),
                }
        return out

    def _build_records(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        skipped_no_ref = 0
        skipped_missing = 0
        if not self.split_root.is_dir():
            raise FileNotFoundError(f"REAL275 split root missing: {self.split_root}")
        if not self.gt_root.is_dir():
            raise FileNotFoundError(f"REAL275 GT root missing: {self.gt_root}")

        for scene_dir in sorted(self.split_root.glob(self.scene_glob)):
            if not scene_dir.is_dir():
                continue
            scene_name = scene_dir.name
            if self.only_scene_name and scene_name != self.only_scene_name:
                continue
            for rgb_path in sorted(scene_dir.glob("*_color.png")):
                frame_id = rgb_path.name.split("_", 1)[0]
                depth_path = scene_dir / f"{frame_id}_depth.png"
                mask_path = scene_dir / f"{frame_id}_mask.png"
                meta_path = scene_dir / f"{frame_id}_meta.txt"
                gt_path = self.gt_root / f"results_real_test_{scene_name}_{frame_id}.pkl"
                if self.verify_files and (
                    not depth_path.is_file()
                    or not mask_path.is_file()
                    or not meta_path.is_file()
                    or not gt_path.is_file()
                ):
                    skipped_missing += 1
                    continue
                meta_entries = self._parse_meta_txt(meta_path)
                gt = self._load_pickle(gt_path)
                gt_rts = np.asarray(gt["gt_RTs"], dtype=np.float64)
                for inst_id, entry in meta_entries.items():
                    rt_index = int(inst_id) - 1
                    if rt_index < 0 or rt_index >= len(gt_rts):
                        continue
                    class_id = entry["class_id"]
                    has_instance_ref = entry["model_name"] in self.instance_refs_by_name
                    has_class_ref = class_id in self.object_refs_by_class_id
                    if not (has_instance_ref or has_class_ref):
                        skipped_no_ref += 1
                        continue
                    records.append(
                        {
                            "scene_name": scene_name,
                            "scene_dir": scene_dir,
                            "frame_id": frame_id,
                            "image_id": int(frame_id),
                            "rgb_path": rgb_path,
                            "depth_path": depth_path,
                            "mask_path": mask_path,
                            "meta_path": meta_path,
                            "gt_path": gt_path,
                            "rt_index": rt_index,
                            "inst_id": inst_id,
                            "class_id": class_id,
                            "category": self.class_id_to_name.get(class_id, ""),
                            "model_name": entry["model_name"],
                        }
                    )
                    if self.max_records is not None and len(records) >= self.max_records:
                        return records
        print(
            f"[Real275RealTest] records={len(records)} skipped_no_ref={skipped_no_ref} "
            f"skipped_missing={skipped_missing}"
        )
        return records

    @staticmethod
    def _read_depth_m(depth_path: Path, depth_scale: float = 1000.0) -> np.ndarray:
        depth_raw = np.asarray(Image.open(depth_path), dtype=np.float32)
        depth_m = depth_raw / float(depth_scale)
        depth_m[~np.isfinite(depth_m)] = 0.0
        depth_m[depth_m < 0.0] = 0.0
        return depth_m

    def _crop_and_load(self, rec: Dict[str, Any], resolution, rng):
        image = Image.open(rec["rgb_path"]).convert("RGB")
        depthmap = self._read_depth_m(rec["depth_path"], 1000.0)
        intrinsic = self.K_REAL.copy()
        object_mask = self._read_instance_mask(rec["mask_path"], int(rec["inst_id"]))
        # Reuse Real275CameraPose's mask-aware crop helper bound to ``self``.
        image, depthmap, object_mask, intrinsic = Real275CameraPose._crop_resize_if_necessary_with_mask(
            self,
            image,
            depthmap,
            object_mask,
            intrinsic,
            resolution,
            rng,
            info=str(rec["rgb_path"]),
        )
        extrinsic = np.concatenate([np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)], axis=1)
        _, point_mask = depthmap_to_absolute_camera_coordinates(
            depthmap, intrinsic, extrinsic, z_far=self.z_far
        )
        return {
            "img": image,
            "depthmap": depthmap.astype(np.float32),
            "camera_pose": extrinsic.astype(np.float32),
            "camera_intrinsics": intrinsic.astype(np.float32),
            "point_mask": point_mask,
            "object_mask": object_mask.astype(np.bool_),
            "label": f"real275/{rec['scene_name']}",
            "instance": rec["rgb_path"].name,
            "dataset": self.dataset_label,
            "image_path": str(rec["rgb_path"]),
            "depth_path": str(rec["depth_path"]),
            "camera_path": str(rec["gt_path"]),
        }

    def _load_object_images(self, model_name: str, class_id: int, resolution) -> Dict[str, Any]:
        # Prefer instance-specific refs (rendered from obj_models/real_test/<model_name>.obj).
        obj_dir = self.instance_refs_by_name.get(str(model_name))
        ref_kind = "instance"
        if obj_dir is None:
            obj_dir = self.object_refs_by_class_id.get(int(class_id))
            ref_kind = "class_fallback"
        if obj_dir is None:
            raise RuntimeError(
                f"No reference views available for model={model_name} class_id={class_id}"
            )
        tensors: List[torch.Tensor] = []
        true_shapes: List[np.ndarray] = []
        image_paths: List[str] = []
        for vid in self.fixed_object_view_ids:
            img_path = obj_dir / "rgb" / f"{int(vid):06d}.png"
            image = Image.open(img_path).convert("RGB")
            true_shapes.append(np.array(image.size[::-1], dtype=np.int32))
            tensors.append(self.transform(self._resize_image(image, resolution)))
            image_paths.append(str(img_path))
        return {
            "object_images": torch.stack(tensors),
            "object_true_shape": np.stack(true_shapes),
            "object_cam_indices": np.asarray(list(self.fixed_object_view_ids), dtype=np.int64),
            "object_reference_scene_name": obj_dir.name,
            "object_reference_kind": ref_kind,
            "object_rgb_paths": image_paths,
            "object_mask_paths": [""] * len(image_paths),
        }

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx, ar_idx, *_ = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0
        rng = np.random.default_rng(seed=(self.seed or 42) + int(idx))
        resolution = self._resolutions[ar_idx]
        rec = self.records[int(idx) % len(self.records)]

        view = self._crop_and_load(rec, resolution, rng)
        view["idx"] = (idx, ar_idx, 0)
        view["z_far"] = self.z_far
        view["true_shape"] = np.int32(view["img"].size[::-1])
        view["img"] = self.transform(view["img"])
        for key, value in view.items():
            res, err_msg = is_good_type(key, value)
            assert res, f"{err_msg} with {key}={value} for view {view_name(view)}"
        transpose_to_landscape(view)

        # Decompose gt_RTs[idx] = [s*R | t]
        gt = self._load_pickle(rec["gt_path"])
        rt = np.asarray(gt["gt_RTs"][int(rec["rt_index"])], dtype=np.float64)
        rotation_with_scale = rt[:3, :3]
        column_norm = float(np.linalg.norm(rotation_with_scale[:, 0]))
        if column_norm < 1e-8:
            r_native_to_cam = np.eye(3, dtype=np.float32)
            t_cam_metric = np.zeros(3, dtype=np.float32)
            size_native = np.ones(3, dtype=np.float32)
        else:
            r_native_to_cam = (rotation_with_scale / column_norm).astype(np.float32)
            t_cam_metric = rt[:3, 3].astype(np.float32)
            size_native = np.full(3, column_norm, dtype=np.float32)

        r_align = self.r_align_by_class_id.get(int(rec["class_id"]), np.eye(3, dtype=np.float32))
        r_aligned_to_cam = (r_native_to_cam @ r_align.T).astype(np.float32)
        size_aligned = np.clip(np.abs(r_align) @ size_native, 1e-6, None).astype(np.float32)
        size_log = np.log(size_aligned).astype(np.float32)

        depth = view["depthmap"]
        valid_depth = depth[np.asarray(view["point_mask"], dtype=np.bool_)]
        if valid_depth.size == 0:
            depth_mean_scale = np.float32(1.0)
        else:
            depth_mean_scale = np.float32(max(float(valid_depth.mean()), self.depth_mean_eps))
        t_normalized = (t_cam_metric / depth_mean_scale).astype(np.float32)
        t_used = t_normalized if self.normalize_object_translation_by_depth_mean else t_cam_metric

        image_id = int(rec["image_id"])
        # Object id within the symmetry catalog (Real275CameraPose:<int>):
        # we don't have a stable object id mapping for real_test, so fall back
        # to class id which the symmetry catalog also keys by name lookup.
        # When the symmetry catalog has no matching entry we still get the
        # plain rotation error.
        object_id = int(rec["class_id"])
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
            "seq_name": f"real275/{rec['scene_name']}/{rec['model_name']}/{image_id:04d}",
            "scene_name": rec["scene_name"],
            "scene_source": "real275_real_test",
            "run_name": rec["scene_name"],
            "object_name": rec["model_name"],
            "object_id": np.array(object_id, dtype=np.int64),
            "inst_id": np.array(int(rec["inst_id"]), dtype=np.int64),
            "class_id": np.array(int(rec["class_id"]), dtype=np.int64),
            "category": rec["category"],
            "has_object": np.array(True, dtype=np.bool_),
            "object_rotation": r_aligned_to_cam,
            "object_translation": t_used,
            "object_translation_metric": t_cam_metric,
            "object_translation_normalized": t_normalized,
            "depth_mean_scale": np.asarray(depth_mean_scale, dtype=np.float32),
            "normalization_scale": np.asarray(depth_mean_scale, dtype=np.float32),
            "object_translation_scale": np.asarray(depth_mean_scale, dtype=np.float32),
            "object_size": size_aligned,
            "object_size_log": size_log,
            "object_srt": np.concatenate(
                [r_aligned_to_cam.reshape(-1), t_used, size_log]
            ).astype(np.float32),
            "scene_rgb_path": view["image_path"],
            "scene_depth_path": view["depth_path"],
            "scene_camera_path": view["camera_path"],
            "scene_gt_path": str(rec["gt_path"]),
            "scene_meta_path": str(rec["meta_path"]),
            "scene_mask_path": str(rec["mask_path"]),
        }
        result.update(self._load_object_images(rec["model_name"], int(rec["class_id"]), resolution))
        return result


# ============================================================================
# Dataset constructors keyed by split name.
# ============================================================================
def build_dataset(split: str, args) -> Dataset:
    common = dict(
        num_object_views=4,
        fixed_object_view_ids=DEFAULT_OBJECT_VIEWS,
        strict_fixed_object_view_ids=True,
        normalize_object_translation_by_depth_mean=True,
        verify_files=True,
        object_presence_prob=1.0,
        z_far=20,
        resolution=DEFAULT_RESOLUTION,
        transform=ImgNorm,
        seed=42,
    )
    if split == "ycbv_test":
        return YCBVCameraPose(
            dataset_location=str(args.ycbv_root),
            dset="test",
            split_root=str(args.ycbv_root / "test"),
            object_image_root=str(args.ycbv_root / "ycbv_aligned_object_refs"),
            align_json=str(args.align_json),
            expand_records_by_object=True,
            **common,
        )
    if split.startswith("housecat_test_scene"):
        scene_name = split.removeprefix("housecat_")
        return HouseCat6DTestSceneCameraPose(
            dataset_location=str(args.housecat_root),
            dset="test",
            object_image_root=str(args.housecat_root / "housecat6d_aligned_object_refs"),
            align_json=str(args.align_json),
            only_scene_name=scene_name,
            expand_records_by_object=True,
            **common,
        )
    if split == "real275_real_test":
        return Real275RealTestCameraPose(
            dataset_location=str(args.real275_root),
            split_root=str(args.real275_root / "real_test"),
            gt_root=str(args.real275_root / "gts" / "real_test"),
            object_image_root=str(args.real275_root / "real275_aligned_object_refs"),
            object_image_root_instance=str(args.real275_instance_refs) if args.real275_instance_refs else None,
            align_json=str(args.align_json),
            only_scene_name=str(getattr(args, "real275_only_scene", "") or ""),
            num_object_views=common["num_object_views"],
            fixed_object_view_ids=common["fixed_object_view_ids"],
            normalize_object_translation_by_depth_mean=common["normalize_object_translation_by_depth_mean"],
            verify_files=common["verify_files"],
            z_far=common["z_far"],
            resolution=common["resolution"],
            transform=common["transform"],
            seed=common["seed"],
        )
    if split == "oo9d_unseen":
        return OO9DSingleCameraPose(
            dataset_location=str(args.oo9d_root),
            dset="test",
            split_root=None,
            single_split_json=str(args.oo9d_split_json),
            object_image_root=str(args.oo9d_object_refs),
            expand_records_by_view=True,
            **common,
        )
    raise ValueError(f"Unknown split: {split}")


# ============================================================================
# Model.
# ============================================================================
def build_model(checkpoint_path: Path, device: torch.device) -> OmniVGGT:
    model = OmniVGGT(
        enable_camera=False,
        enable_point=False,
        enable_depth=False,
        enable_object_mask=True,
        enable_object_srt=True,
        always_use_depth_gt=True,
        patch_embed_pretrained_path=None,
        load_patch_embed_from_hub=False,
        cam_drop_prob=1.0,
        depth_drop_prob=0.0,
        object_pose_context_pool="flatten",
        object_pose_use_global_scene_object_concat=False,
        object_pose_transformer_depth=6,
        object_pose_transformer_heads=8,
        object_pose_transformer_mlp_dim=1024,
        object_pose_transformer_dim_head=64,
        object_pose_transformer_dropout=0.0,
        object_pose_transformer_emb_dropout=0.0,
        object_pose_transformer_norm="layer",
        object_pose_transformer_dim=1024,
        object_pose_ief_iters=1,
        object_pose_init_params_path=None,
        enable_multi_layer_object_prototype_cross_attn=True,
        object_prototype_layer_indices=(4, 11, 17, 23),
        object_prototype_num_tokens=32,
        object_prototype_object_encoder_no_grad=True,
        object_cross_attn_heads=16,
    )
    state = load_safetensors_file(str(checkpoint_path), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[eval] missing keys ({len(missing)}); first 8: {missing[:8]}")
    if unexpected:
        print(f"[eval] unexpected keys ({len(unexpected)}); first 8: {unexpected[:8]}")
    return model.eval().to(device)


# ============================================================================
# Metric helpers.
# ============================================================================
def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float64).reshape(3, 2)
    x_raw = rot6d[:, 0]
    y_raw = rot6d[:, 1]
    x = x_raw / max(np.linalg.norm(x_raw), 1e-12)
    z = np.cross(x, y_raw)
    z = z / max(np.linalg.norm(z), 1e-12)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1).astype(np.float64)


def rotation_error_degrees(pred_rot: np.ndarray, gt_rot: np.ndarray) -> float:
    rel = np.asarray(pred_rot, dtype=np.float64).T @ np.asarray(gt_rot, dtype=np.float64)
    cos_theta = np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def symmetric_rotation_error_degrees(
    pred_rot: np.ndarray,
    gt_rot: np.ndarray,
    dataset_label: str,
    object_id: int,
    symmetry_info: Dict,
) -> float:
    sym_rots = None
    if dataset_label:
        sym_rots = symmetry_info.get(f"{dataset_label}:{int(object_id)}")
    if sym_rots is None:
        sym_rots = symmetry_info.get(int(object_id))
    if sym_rots is None:
        return rotation_error_degrees(pred_rot, gt_rot)
    sym_np = sym_rots.detach().cpu().numpy().astype(np.float64)
    gt = np.asarray(gt_rot, dtype=np.float64)
    return float(min(rotation_error_degrees(pred_rot, gt @ sym) for sym in sym_np))


def translation_error_cm(pred_t_metric: np.ndarray, gt_t_metric: np.ndarray) -> float:
    diff = np.asarray(pred_t_metric, dtype=np.float64) - np.asarray(gt_t_metric, dtype=np.float64)
    return float(np.linalg.norm(diff) * 100.0)


# ============================================================================
# Visualization helpers.
# ============================================================================
BBOX_EDGES = (
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
AXIS_COLORS = ((255, 64, 64), (64, 255, 64), (64, 128, 255))  # X red / Y green / Z blue


def _bbox_corners_object(size_xyz: np.ndarray) -> np.ndarray:
    half = np.asarray(size_xyz, dtype=np.float64).reshape(3) * 0.5
    xs = [-half[0], half[0]]
    ys = [-half[1], half[1]]
    zs = [-half[2], half[2]]
    return np.asarray([[x, y, z] for z in zs for y in ys for x in xs], dtype=np.float64)


def _project_points(points_cam: np.ndarray, intrinsic: np.ndarray):
    intrinsic = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    points = np.asarray(points_cam, dtype=np.float64)
    z = points[:, 2]
    valid = z > 1e-6
    uv = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    if np.any(valid):
        proj = points[valid] @ intrinsic.T
        uv[valid] = proj[:, :2] / proj[:, 2:3]
    return uv, valid


def _draw_box(draw: ImageDraw.ImageDraw, uv: np.ndarray, valid: np.ndarray,
              color: tuple, width: int = 2):
    for a, b in BBOX_EDGES:
        if not (valid[a] and valid[b]):
            continue
        p0 = (float(uv[a, 0]), float(uv[a, 1]))
        p1 = (float(uv[b, 0]), float(uv[b, 1]))
        draw.line([p0, p1], fill=color, width=width)


def _draw_axes(draw: ImageDraw.ImageDraw, R: np.ndarray, t: np.ndarray,
               intrinsic: np.ndarray, length: float, width: int = 3):
    origin = np.asarray(t, dtype=np.float64).reshape(3)
    pts = np.stack([origin] + [origin + R[:, i] * length for i in range(3)], axis=0)
    uv, valid = _project_points(pts, intrinsic)
    if not valid[0]:
        return
    o = (float(uv[0, 0]), float(uv[0, 1]))
    for i, color in enumerate(AXIS_COLORS):
        if not valid[i + 1]:
            continue
        e = (float(uv[i + 1, 0]), float(uv[i + 1, 1]))
        draw.line([o, e], fill=color, width=width)


def _tensor_to_uint8_chw(t: torch.Tensor) -> np.ndarray:
    """Inverse of ImgNorm (== ToTensor): map (C, H, W) float [0,1] to uint8 (H, W, C)."""
    arr = t.detach().float().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    arr = np.transpose(arr, (1, 2, 0))
    return np.uint8(np.round(arr * 255.0))


def dump_real275_visualization(args, sample_index: int, sample: Dict, out_dir: Path):
    """Save scene + GT pose overlay + object reference views for one REAL275 sample."""
    scene_chw = sample["images"][0]  # (3, H, W) tensor
    scene_arr = _tensor_to_uint8_chw(scene_chw)
    scene_img = Image.fromarray(scene_arr, mode="RGB")

    intrinsic = np.asarray(sample["intrinsic"][0], dtype=np.float64).reshape(3, 3)
    gt_R = np.asarray(sample["object_rotation"], dtype=np.float64).reshape(3, 3)
    gt_t = np.asarray(sample["object_translation_metric"], dtype=np.float64).reshape(3)
    size = np.asarray(sample["object_size"], dtype=np.float64).reshape(3)

    corners_obj = _bbox_corners_object(size)
    corners_cam = corners_obj @ gt_R.T + gt_t[None, :]
    uv, valid = _project_points(corners_cam, intrinsic)

    overlay = scene_img.copy()
    draw = ImageDraw.Draw(overlay)
    _draw_box(draw, uv, valid, color=(255, 160, 0), width=3)
    _draw_axes(draw, gt_R, gt_t, intrinsic, length=float(size.max() * 0.5), width=4)
    label_lines = [
        f"scene={sample['scene_name']} frame={int(sample['ids'][0])} "
        f"model={sample['object_name']} cls={int(sample['class_id'])}",
        f"size={np.round(size, 3).tolist()}",
        f"t_m={np.round(gt_t, 3).tolist()}",
    ]
    draw.rectangle((4, 4, 4 + 460, 4 + 18 * len(label_lines) + 8), fill=(0, 0, 0))
    for i, line in enumerate(label_lines):
        draw.text((10, 10 + i * 18), line, fill=(255, 255, 255))

    prefix = out_dir / f"{sample_index:04d}_{sample['scene_name']}_f{int(sample['ids'][0]):04d}_inst{int(sample['inst_id'])}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    scene_img.save(prefix.with_name(prefix.name + "_scene_input.jpg"), quality=92)
    overlay.save(prefix.with_name(prefix.name + "_scene_gt_pose.jpg"), quality=92)

    object_tensor = sample["object_images"]  # (K, 3, H, W)
    for k in range(object_tensor.shape[0]):
        obj_arr = _tensor_to_uint8_chw(object_tensor[k])
        Image.fromarray(obj_arr, mode="RGB").save(
            prefix.with_name(prefix.name + f"_obj_ref_{int(sample['object_cam_indices'][k]):06d}.jpg"),
            quality=92,
        )

    meta = {
        "scene_name": str(sample["scene_name"]),
        "frame_id": int(sample["ids"][0]),
        "inst_id": int(sample["inst_id"]),
        "class_id": int(sample["class_id"]),
        "category": str(sample["category"]),
        "model_name": str(sample["object_name"]),
        "object_id_used_as_symmetry_key": int(np.asarray(sample["object_id"]).reshape(-1)[0]),
        "intrinsic": intrinsic.tolist(),
        "gt_rotation_aligned": gt_R.tolist(),
        "gt_translation_metric_m": gt_t.tolist(),
        "object_size_aligned_m": size.tolist(),
        "depth_mean_scale_m": float(np.asarray(sample["depth_mean_scale"]).reshape(-1)[0]),
        "scene_rgb_path": str(sample["scene_rgb_path"]),
        "scene_mask_path": str(sample["scene_mask_path"]),
        "scene_gt_path": str(sample["scene_gt_path"]),
        "scene_meta_path": str(sample["scene_meta_path"]),
        "object_reference_dir": str(sample["object_reference_scene_name"]),
        "object_reference_kind": str(sample.get("object_reference_kind", "")),
        "object_rgb_paths": list(sample["object_rgb_paths"]),
    }
    with prefix.with_name(prefix.name + "_meta.json").open("w", encoding="utf-8") as h:
        json.dump(meta, h, indent=2)


def dump_real275_visualizations(args) -> None:
    """Dump the first ``--vis-real275-max`` REAL275 real_test samples to ``--vis-real275-dir``."""
    vis_dir: Path = args.vis_real275_dir
    vis_dir.mkdir(parents=True, exist_ok=True)
    print(f"[vis] dumping REAL275 real_test samples to {vis_dir}")
    dataset = build_dataset("real275_real_test", args)
    n = min(int(args.vis_real275_max), len(dataset))
    for idx in tqdm(range(n), desc="vis real275"):
        sample = dataset[idx]
        dump_real275_visualization(args, idx, sample, vis_dir)
    print(f"[vis] wrote {n} samples to {vis_dir}")


def dump_real275_pred_visualization(
    args,
    sample_index: int,
    sample: Dict,
    pred_rot_aligned: np.ndarray,
    pred_t_metric: np.ndarray,
    pred_size: Optional[np.ndarray],
    rot_err_deg: float,
    trans_err_cm: float,
    out_dir: Path,
):
    """Save GT + Pred bbox/axes overlay for one REAL275 sample."""
    scene_chw = sample["images"][0]
    scene_arr = _tensor_to_uint8_chw(scene_chw)
    scene_img = Image.fromarray(scene_arr, mode="RGB")

    intrinsic = np.asarray(sample["intrinsic"][0], dtype=np.float64).reshape(3, 3)
    gt_R = np.asarray(sample["object_rotation"], dtype=np.float64).reshape(3, 3)
    gt_t = np.asarray(sample["object_translation_metric"], dtype=np.float64).reshape(3)
    gt_size = np.asarray(sample["object_size"], dtype=np.float64).reshape(3)
    pred_R = np.asarray(pred_rot_aligned, dtype=np.float64).reshape(3, 3)
    pred_t = np.asarray(pred_t_metric, dtype=np.float64).reshape(3)
    if pred_size is None or not np.all(np.isfinite(pred_size)):
        pred_size_use = gt_size.copy()
    else:
        pred_size_use = np.clip(np.asarray(pred_size, dtype=np.float64).reshape(3), 1e-4, None)

    gt_corners_cam = _bbox_corners_object(gt_size) @ gt_R.T + gt_t[None, :]
    gt_uv, gt_valid = _project_points(gt_corners_cam, intrinsic)
    pred_corners_cam = _bbox_corners_object(pred_size_use) @ pred_R.T + pred_t[None, :]
    pred_uv, pred_valid = _project_points(pred_corners_cam, intrinsic)

    overlay = scene_img.copy()
    draw = ImageDraw.Draw(overlay)
    _draw_box(draw, gt_uv, gt_valid, color=(255, 160, 0), width=3)        # GT = orange
    _draw_box(draw, pred_uv, pred_valid, color=(0, 220, 80), width=3)     # Pred = green
    _draw_axes(draw, gt_R, gt_t, intrinsic, length=float(gt_size.max() * 0.5), width=4)
    _draw_axes(draw, pred_R, pred_t, intrinsic, length=float(pred_size_use.max() * 0.5), width=4)
    label_lines = [
        f"GT(orange) vs Pred(green)  model={sample['object_name']}  cls={int(sample['class_id'])}",
        f"scene={sample['scene_name']} frame={int(sample['ids'][0])} inst={int(sample['inst_id'])}",
        f"rot_err={rot_err_deg:.2f}deg   trans_err={trans_err_cm:.2f}cm",
        f"gt_t={np.round(gt_t, 3).tolist()}  pred_t={np.round(pred_t, 3).tolist()}",
    ]
    draw.rectangle((4, 4, 4 + 500, 4 + 18 * len(label_lines) + 8), fill=(0, 0, 0))
    for i, line in enumerate(label_lines):
        draw.text((10, 10 + i * 18), line, fill=(255, 255, 255))

    safe_obj = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sample["object_name"]))[:60]
    prefix = out_dir / (
        f"{sample_index:05d}_{sample['scene_name']}_f{int(sample['ids'][0]):04d}"
        f"_inst{int(sample['inst_id'])}_{safe_obj}"
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    scene_img.save(prefix.with_name(prefix.name + "_scene_input.jpg"), quality=92)
    overlay.save(prefix.with_name(prefix.name + "_scene_gt_pred.jpg"), quality=92)

    object_tensor = sample["object_images"]
    for k in range(object_tensor.shape[0]):
        obj_arr = _tensor_to_uint8_chw(object_tensor[k])
        Image.fromarray(obj_arr, mode="RGB").save(
            prefix.with_name(prefix.name + f"_obj_ref_{int(sample['object_cam_indices'][k]):06d}.jpg"),
            quality=92,
        )

    meta = {
        "scene_name": str(sample["scene_name"]),
        "frame_id": int(sample["ids"][0]),
        "inst_id": int(sample["inst_id"]),
        "class_id": int(sample["class_id"]),
        "category": str(sample["category"]),
        "model_name": str(sample["object_name"]),
        "object_reference_kind": str(sample.get("object_reference_kind", "")),
        "object_reference_dir": str(sample["object_reference_scene_name"]),
        "intrinsic": intrinsic.tolist(),
        "gt_rotation_aligned": gt_R.tolist(),
        "gt_translation_metric_m": gt_t.tolist(),
        "gt_object_size_m": gt_size.tolist(),
        "pred_rotation_aligned": pred_R.tolist(),
        "pred_translation_metric_m": pred_t.tolist(),
        "pred_object_size_m": pred_size_use.tolist(),
        "rotation_error_deg": float(rot_err_deg),
        "translation_error_cm": float(trans_err_cm),
        "depth_mean_scale_m": float(np.asarray(sample["depth_mean_scale"]).reshape(-1)[0]),
        "scene_rgb_path": str(sample["scene_rgb_path"]),
        "scene_gt_path": str(sample["scene_gt_path"]),
    }
    with prefix.with_name(prefix.name + "_meta.json").open("w", encoding="utf-8") as h:
        json.dump(meta, h, indent=2)


def dump_real275_pred_visualizations(args) -> None:
    """Run model on REAL275 real_test and dump GT-vs-Pred bbox overlays.

    Filters: ``--vis-real275-only-objects`` (comma-separated model names) and
    ``--real275-only-scene``. Caps via ``--vis-real275-pred-max``.
    """
    vis_dir: Path = args.vis_real275_pred_dir
    vis_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print(f"[vis-pred] device={device} dumping to {vis_dir}")
    symmetry_info = _load_symmetry_info(str(args.symmetry_info), int(args.continuous_steps))
    model = build_model(args.checkpoint, device)
    dataset = build_dataset("real275_real_test", args)

    only_objects = set()
    if args.vis_real275_only_objects:
        only_objects = {s.strip() for s in str(args.vis_real275_only_objects).split(",") if s.strip()}
    cap = int(args.vis_real275_pred_max) if args.vis_real275_pred_max is not None else None

    written = 0
    pbar = tqdm(range(len(dataset)), desc="vis-pred real275", dynamic_ncols=True)
    for idx in pbar:
        if cap is not None and written >= cap:
            break
        # Peek at the record to filter without paying the full __getitem__ cost.
        rec = dataset.records[idx]
        if only_objects and str(rec["model_name"]) not in only_objects:
            continue
        sample = dataset[idx]

        scene_t = sample["images"].unsqueeze(0).to(device, non_blocking=True)
        obj_t = sample["object_images"].unsqueeze(0).to(device, non_blocking=True)
        depth_t = torch.as_tensor(sample["depth"]).unsqueeze(0).to(device, non_blocking=True)
        mask_t = torch.as_tensor(sample["valid_mask"]).unsqueeze(0).to(device, non_blocking=True)
        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=(not args.no_amp) and device.type == "cuda"
            ):
                outputs = model.inference(
                    images=scene_t,
                    object_images=obj_t,
                    extrinsics=None,
                    intrinsics=None,
                    depth=depth_t,
                    mask=mask_t,
                    camera_gt_index=[],
                    depth_gt_index=[0],
                )

        pred_pose = outputs["object_pose"].detach().float().cpu().numpy()[0]
        pred_trans_norm = outputs["object_translation"].detach().float().cpu().numpy()[0]
        if "object_size" in outputs:
            pred_size = outputs["object_size"].detach().float().cpu().numpy()[0]
        elif "object_size_log" in outputs:
            pred_size = np.exp(outputs["object_size_log"].detach().float().cpu().numpy()[0])
        else:
            pred_size = None
        depth_mean = float(np.asarray(sample["depth_mean_scale"]).reshape(-1)[0])
        pred_R = rot6d_to_matrix(pred_pose)
        pred_t_metric = np.asarray(pred_trans_norm, dtype=np.float64) * depth_mean

        gt_R = np.asarray(sample["object_rotation"], dtype=np.float64).reshape(3, 3)
        gt_t_metric = np.asarray(sample["object_translation_metric"], dtype=np.float64).reshape(3)
        object_id = int(np.asarray(sample["object_id"]).reshape(-1)[0])
        rot_err = symmetric_rotation_error_degrees(
            pred_R, gt_R, "Real275CameraPose", object_id, symmetry_info
        )
        trans_err = translation_error_cm(pred_t_metric, gt_t_metric)

        dump_real275_pred_visualization(
            args, idx, sample, pred_R, pred_t_metric, pred_size, rot_err, trans_err, vis_dir
        )
        written += 1
        pbar.set_postfix(written=written, last_rot=f"{rot_err:.1f}", last_trans=f"{trans_err:.1f}")
    pbar.close()
    print(f"[vis-pred] wrote {written} samples to {vis_dir}")


# ============================================================================
# Worker entry: evaluate one shard of one split.
# ============================================================================
def _to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def _batch_item(value, i: int):
    if torch.is_tensor(value):
        return value[i].detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value[i]
    if isinstance(value, (list, tuple)):
        return value[i]
    return value


def evaluate_split(
    split: str,
    dataset: Dataset,
    model: OmniVGGT,
    device: torch.device,
    symmetry_info: Dict,
    batch_size: int,
    num_workers: int,
    amp: bool,
    shard_index: int,
    num_shards: int,
    limit: Optional[int],
) -> List[Dict]:
    indices = list(range(shard_index, len(dataset), num_shards))
    if limit is not None:
        indices = indices[:limit]
    if not indices:
        return []
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=False,
    )
    dataset_label = DATASET_LABEL_BY_SPLIT.get(split, "")
    samples: List[Dict] = []
    pbar = tqdm(loader, desc=f"shard {shard_index}/{num_shards} {split}", dynamic_ncols=True)
    for batch in pbar:
        batch_dev = _to_device(batch, device)
        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"
            ):
                outputs = model.inference(
                    images=batch_dev["images"],
                    object_images=batch_dev["object_images"],
                    extrinsics=None,
                    intrinsics=None,
                    depth=batch_dev["depth"],
                    mask=batch_dev["valid_mask"],
                    camera_gt_index=[],
                    depth_gt_index=[0],
                )
        if "object_pose" not in outputs or "object_translation" not in outputs:
            raise RuntimeError(f"Model output missing pose keys: {sorted(outputs.keys())}")
        pred_pose = outputs["object_pose"].detach().float().cpu().numpy()
        pred_translation = outputs["object_translation"].detach().float().cpu().numpy()

        bs = int(pred_pose.shape[0])
        for i in range(bs):
            has_object = bool(np.asarray(_batch_item(batch["has_object"], i)).reshape(-1)[0])
            if not has_object:
                continue
            object_id = int(np.asarray(_batch_item(batch["object_id"], i)).reshape(-1)[0])
            gt_rot = np.asarray(_batch_item(batch["object_rotation"], i), dtype=np.float64).reshape(3, 3)
            gt_t_metric = np.asarray(
                _batch_item(batch["object_translation_metric"], i), dtype=np.float64
            ).reshape(3)
            depth_mean = float(np.asarray(_batch_item(batch["depth_mean_scale"], i)).reshape(-1)[0])

            pred_rot = rot6d_to_matrix(pred_pose[i])
            pred_t_norm = np.asarray(pred_translation[i], dtype=np.float64).reshape(3)
            pred_t_metric = pred_t_norm * depth_mean

            rot_err = symmetric_rotation_error_degrees(
                pred_rot, gt_rot, dataset_label, object_id, symmetry_info
            )
            trans_err = translation_error_cm(pred_t_metric, gt_t_metric)
            samples.append(
                {
                    "split": split,
                    "scene_name": str(_batch_item(batch["scene_name"], i)),
                    "image_id": int(np.asarray(_batch_item(batch["ids"], i)).reshape(-1)[0]),
                    "object_id": object_id,
                    "object_name": str(_batch_item(batch["object_name"], i)),
                    "category": str(_batch_item(batch.get("category", [""] * bs), i)) if "category" in batch else "",
                    "rotation_error_deg": float(rot_err),
                    "translation_error_cm": float(trans_err),
                    "pred_translation_metric": pred_t_metric.tolist(),
                    "gt_translation_metric": gt_t_metric.tolist(),
                    "depth_mean_scale_m": float(depth_mean),
                }
            )
    return samples


# ============================================================================
# Reports.
# ============================================================================
def summarize(split: str, samples: List[Dict]) -> Dict[str, Any]:
    n = len(samples)
    out: Dict[str, Any] = {"split": split, "num_samples": n}
    if n == 0:
        for name, _, _ in THRESHOLDS:
            out[name] = None
        out["rotation_error_deg_mean"] = None
        out["translation_error_cm_mean"] = None
        return out
    rot = np.asarray([s["rotation_error_deg"] for s in samples], dtype=np.float64)
    trans = np.asarray([s["translation_error_cm"] for s in samples], dtype=np.float64)
    for name, rot_t, trans_t in THRESHOLDS:
        out[name] = 100.0 * float(np.mean((rot <= rot_t) & (trans <= trans_t)))
    out["rotation_error_deg_mean"] = float(rot.mean())
    out["rotation_error_deg_median"] = float(np.median(rot))
    out["translation_error_cm_mean"] = float(trans.mean())
    out["translation_error_cm_median"] = float(np.median(trans))
    return out


def print_table(rows: List[Dict[str, Any]]):
    headers = ["split", "n", *(name for name, _, _ in THRESHOLDS), "rot_mean", "trans_cm_mean"]
    table_rows = []
    for r in rows:
        table_rows.append(
            [
                r["split"],
                str(r["num_samples"]),
                *[f"{r[name]:.2f}" if r[name] is not None else "N/A" for name, _, _ in THRESHOLDS],
                f"{r['rotation_error_deg_mean']:.2f}" if r.get("rotation_error_deg_mean") is not None else "N/A",
                f"{r['translation_error_cm_mean']:.2f}" if r.get("translation_error_cm_mean") is not None else "N/A",
            ]
        )
    widths = [max(len(c) for c in col) for col in zip(headers, *table_rows)]
    fmt = " | ".join("{:<" + str(w) + "}" for w in widths)
    print(fmt.format(*headers))
    print("-+-".join("-" * w for w in widths))
    for row in table_rows:
        print(fmt.format(*row))


# ============================================================================
# Orchestrator.
# ============================================================================
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--symmetry-info", type=Path, default=DEFAULT_SYMMETRY_INFO)
    parser.add_argument("--continuous-steps", type=int, default=DEFAULT_CONTINUOUS_STEPS,
                        help="Sampling step for continuous rotation symmetries (larger = finer).")
    parser.add_argument("--align-json", type=Path, default=DEFAULT_ALIGN_JSON)
    parser.add_argument("--ycbv-root", type=Path, default=DEFAULT_YCBV_ROOT)
    parser.add_argument("--housecat-root", type=Path, default=DEFAULT_HOUSECAT_ROOT)
    parser.add_argument("--real275-root", type=Path, default=DEFAULT_REAL275_ROOT)
    parser.add_argument("--real275-instance-refs", type=Path,
                        default=DEFAULT_REAL275_ROOT / "real275_aligned_object_refs_test",
                        help="Directory of per-test-instance reference views (rendered from "
                             "obj_models/real_test/*.obj). Used preferentially; falls back to "
                             "same-class train refs when missing.")
    parser.add_argument("--real275-only-scene", type=str, default="",
                        help="If set (e.g. 'scene_1'), only evaluate that REAL275 real_test scene.")
    parser.add_argument("--oo9d-root", type=Path, default=DEFAULT_OO9D_ROOT)
    parser.add_argument("--oo9d-object-refs", type=Path, default=DEFAULT_OO9D_OBJECT_REFS)
    parser.add_argument("--oo9d-split-json", type=Path, default=DEFAULT_OO9D_SPLIT_JSON)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS_ORDER))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Optional per-shard cap for smoke tests.")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "eval_real_multi_0521")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated CUDA device ids. If multiple are passed, "
                             "fan out as subprocesses (one shard per GPU).")
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--vis-real275-dir", type=Path, default=None,
                        help="If set, dump REAL275 real_test sample visualizations "
                             "(cropped scene, object reference views, GT 3D bbox + axes overlay, meta.json) "
                             "into this directory.")
    parser.add_argument("--vis-real275-max", type=int, default=30,
                        help="How many REAL275 samples to dump when --vis-real275-dir is set.")
    parser.add_argument("--vis-real275-pred-dir", type=Path, default=None,
                        help="If set, run inference and dump GT-vs-Pred bbox overlays on REAL275 real_test "
                             "samples into this directory. Honors --real275-only-scene and "
                             "--vis-real275-only-objects filters.")
    parser.add_argument("--vis-real275-pred-max", type=int, default=30,
                        help="Cap on how many REAL275 samples to dump for --vis-real275-pred-dir.")
    parser.add_argument("--vis-real275-only-objects", type=str, default="",
                        help="Comma-separated model_names; if set, only those objects are dumped "
                             "by --vis-real275-pred-dir.")
    parser.add_argument("--vis-only", action="store_true",
                        help="If set with --vis-real275-dir or --vis-real275-pred-dir, skip evaluation.")
    return parser.parse_args(argv)


def run_worker(args: argparse.Namespace, gpu_id: Optional[str], shard_index: int, num_shards: int) -> None:
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"[worker] shard={shard_index}/{num_shards} device={device} ({gpu_id})")

    symmetry_info = _load_symmetry_info(str(args.symmetry_info), int(args.continuous_steps))
    print(f"[worker] symmetry entries: {len(symmetry_info)} continuous_steps={args.continuous_steps}")

    model = build_model(args.checkpoint, device)
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        try:
            dataset = build_dataset(split, args)
        except Exception as exc:  # noqa: BLE001 - we want to log and skip
            print(f"[worker] split={split} dataset build failed: {exc!r}")
            continue
        samples = evaluate_split(
            split=split,
            dataset=dataset,
            model=model,
            device=device,
            symmetry_info=symmetry_info,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            amp=not args.no_amp,
            shard_index=shard_index,
            num_shards=num_shards,
            limit=args.limit,
        )
        shard_path = out_dir / f"samples_{split}_shard_{shard_index:02d}.jsonl"
        with shard_path.open("w", encoding="utf-8") as h:
            for s in samples:
                h.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"[worker] split={split} samples={len(samples)} wrote {shard_path}")


def orchestrate(args: argparse.Namespace, gpu_ids: List[str]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    num_shards = len(gpu_ids)
    print(f"[orchestrator] launching {num_shards} shards on GPUs {gpu_ids}")
    base_cmd = [
        sys.executable, "-u", str(Path(__file__).resolve()),
        "--checkpoint", str(args.checkpoint),
        "--symmetry-info", str(args.symmetry_info),
        "--continuous-steps", str(args.continuous_steps),
        "--align-json", str(args.align_json),
        "--ycbv-root", str(args.ycbv_root),
        "--housecat-root", str(args.housecat_root),
        "--real275-root", str(args.real275_root),
        "--real275-instance-refs", str(args.real275_instance_refs),
        "--real275-only-scene", str(args.real275_only_scene or ""),
        "--oo9d-root", str(args.oo9d_root),
        "--oo9d-object-refs", str(args.oo9d_object_refs),
        "--oo9d-split-json", str(args.oo9d_split_json),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--num-shards", str(num_shards),
        "--output-dir", str(args.output_dir),
    ]
    if args.no_amp:
        base_cmd.append("--no-amp")
    if args.limit is not None:
        base_cmd.extend(["--limit", str(args.limit)])
    base_cmd.extend(["--splits", *args.splits])

    procs: List[Tuple[int, subprocess.Popen]] = []
    for shard_index, gpu_id in enumerate(gpu_ids):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        log_path = args.output_dir / f"shard_{shard_index:02d}.log"
        cmd = base_cmd + ["--shard-index", str(shard_index)]
        log_fh = open(log_path, "w", encoding="utf-8")
        print(f"[orchestrator] shard {shard_index} -> GPU {gpu_id}  log={log_path}")
        procs.append((shard_index, subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)))

    start = time.time()
    while True:
        time.sleep(15)
        alive = [s for s, p in procs if p.poll() is None]
        statuses = []
        for shard_index, p in procs:
            log_path = args.output_dir / f"shard_{shard_index:02d}.log"
            try:
                with log_path.open("rb") as h:
                    h.seek(0, os.SEEK_END)
                    size = h.tell()
                    h.seek(max(0, size - 600), os.SEEK_SET)
                    tail = h.read().decode("utf-8", errors="replace")
                last = tail.strip().splitlines()[-1] if tail.strip() else ""
                statuses.append(f"  shard {shard_index}: alive={p.poll() is None}  tail={last[-180:]}")
            except FileNotFoundError:
                statuses.append(f"  shard {shard_index}: log missing")
        print(f"[orchestrator] t={time.time() - start:6.0f}s  alive={len(alive)}/{len(procs)}")
        for line in statuses:
            print(line)
        if not alive:
            break

    return_codes = [p.wait() for _, p in procs]
    if any(rc != 0 for rc in return_codes):
        print(f"[orchestrator] return codes: {return_codes}")

    rows: List[Dict[str, Any]] = []
    for split in args.splits:
        merged: List[Dict] = []
        for shard_index, _ in procs:
            shard_path = args.output_dir / f"samples_{split}_shard_{shard_index:02d}.jsonl"
            if not shard_path.is_file():
                continue
            with shard_path.open("r", encoding="utf-8") as h:
                for line in h:
                    line = line.strip()
                    if line:
                        merged.append(json.loads(line))
        with (args.output_dir / f"samples_{split}.jsonl").open("w", encoding="utf-8") as h:
            for s in merged:
                h.write(json.dumps(s, ensure_ascii=False) + "\n")
        rows.append(summarize(split, merged))

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as h:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "symmetry_info": str(args.symmetry_info),
                "continuous_steps": int(args.continuous_steps),
                "thresholds": [{"name": n, "rot_deg": r, "trans_cm": t} for n, r, t in THRESHOLDS],
                "splits": rows,
            },
            h,
            indent=2,
        )
    print(f"[orchestrator] wrote {summary_path}")
    print_table(rows)


def main(argv=None):
    args = parse_args(argv)
    is_worker = args.shard_index is not None
    if not is_worker and args.vis_real275_dir is not None:
        dump_real275_visualizations(args)
        if args.vis_only and args.vis_real275_pred_dir is None:
            return
    if not is_worker and args.vis_real275_pred_dir is not None:
        dump_real275_pred_visualizations(args)
        if args.vis_only:
            return
    if not is_worker and args.gpus is not None and "," in args.gpus:
        gpu_ids = [x.strip() for x in args.gpus.split(",") if x.strip()]
        if len(gpu_ids) > 1:
            return orchestrate(args, gpu_ids)
    if not is_worker and args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    shard_index = 0 if args.shard_index is None else int(args.shard_index)
    num_shards = int(args.num_shards) if args.num_shards else 1
    # Single-process path (or single worker invoked by orchestrator).
    run_worker(args, gpu_id=None, shard_index=shard_index, num_shards=num_shards)

    if not is_worker:
        # Single-shard summary table for the single-GPU case.
        rows = []
        for split in args.splits:
            shard_path = args.output_dir / f"samples_{split}_shard_{shard_index:02d}.jsonl"
            samples: List[Dict] = []
            if shard_path.is_file():
                with shard_path.open("r", encoding="utf-8") as h:
                    for line in h:
                        line = line.strip()
                        if line:
                            samples.append(json.loads(line))
            with (args.output_dir / f"samples_{split}.jsonl").open("w", encoding="utf-8") as h:
                for s in samples:
                    h.write(json.dumps(s, ensure_ascii=False) + "\n")
            rows.append(summarize(split, samples))
        summary_path = args.output_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as h:
            json.dump(
                {
                    "checkpoint": str(args.checkpoint),
                    "symmetry_info": str(args.symmetry_info),
                    "continuous_steps": int(args.continuous_steps),
                    "thresholds": [{"name": n, "rot_deg": r, "trans_cm": t} for n, r, t in THRESHOLDS],
                    "splits": rows,
                },
                h,
                indent=2,
            )
        print(f"[eval] wrote {summary_path}")
        print_table(rows)


if __name__ == "__main__":
    main()
