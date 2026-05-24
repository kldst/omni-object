import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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
import omnivggt.datasets.utils.cropping as cropping
from omnivggt.utils.geometry import depthmap_to_absolute_camera_coordinates


logger = logging.getLogger(__name__)


class Real275CameraPose(BaseStereoViewDataset):
    """REAL275 real_train dataset with Umeyama poses and OV9D-aligned object coordinates.

    REAL275 real_train provides RGB/depth/mask/meta, but not the same official GT
    pkl files as real_test in this workspace. This dataset consumes the
    ``real275/gts/real_train_umeyama`` files generated from coord/depth/mask.

    ``dataset_align.json`` stores ``R_align`` such that
    ``p_ov9d = R_align @ p_real275_native``. The object rotation consumed by the
    model is therefore ``R_native_to_cam @ R_align.T``.
    """

    DEFAULT_DATA_ROOT = "/mnt/train-data-4-hdd/yian/freepose/real275"
    DEFAULT_SPLIT_ROOT = "/mnt/train-data-4-hdd/yian/freepose/real275/real_train"
    DEFAULT_GT_ROOT = "/mnt/train-data-4-hdd/yian/freepose/real275/gts/real_train_umeyama"
    DEFAULT_OBJECT_IMAGE_ROOT = "/mnt/train-data-4-hdd/yian/freepose/real275/real275_aligned_object_refs"
    DEFAULT_ALIGN_JSON = "/mnt/train-data-4-hdd/yian/freepose/omni-object_clone/dataset_align.json"
    DEFAULT_OBJECT_VIEW_IDS = (1, 5, 10, 15)
    K_REAL = np.array(
        [[591.0125, 0.0, 322.525], [0.0, 590.16775, 244.11084], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    def __init__(
        self,
        dataset_location: str = DEFAULT_DATA_ROOT,
        dset: str = "train",
        split_root: Optional[str] = None,
        gt_root: Optional[str] = None,
        object_image_root: Optional[str] = None,
        align_json: Optional[str] = None,
        fixed_object_view_ids: Optional[Sequence[int]] = DEFAULT_OBJECT_VIEW_IDS,
        num_object_views: int = 4,
        strict_fixed_object_view_ids: bool = True,
        verify_files: bool = True,
        max_records: Optional[int] = None,
        only_scene_name: str = "",
        only_object_name: str = "",
        only_category: str = "",
        expand_records_by_object: bool = True,
        object_presence_prob: float = 1.0,
        normalize_object_translation_by_depth_mean: bool = True,
        depth_mean_eps: float = 1e-6,
        *args,
        **kwargs,
    ):
        super().__init__(dset=dset, *args, **kwargs)
        self.dataset_label = "Real275CameraPose"
        self.dataset_location = Path(dataset_location)
        self.split_root = Path(split_root or self.DEFAULT_SPLIT_ROOT)
        self.gt_root = Path(gt_root or self.DEFAULT_GT_ROOT)
        self.object_image_root = Path(object_image_root or self.DEFAULT_OBJECT_IMAGE_ROOT)
        self.align_json = Path(align_json or self.DEFAULT_ALIGN_JSON)
        self.fixed_object_view_ids = tuple(int(x) for x in fixed_object_view_ids) if fixed_object_view_ids else None
        self.num_object_views = int(num_object_views)
        self.strict_fixed_object_view_ids = bool(strict_fixed_object_view_ids)
        self.verify_files = bool(verify_files)
        self.max_records = int(max_records) if max_records is not None else None
        self.only_scene_name = str(only_scene_name).strip()
        self.only_object_name = str(only_object_name).strip()
        self.only_category = str(only_category).strip()
        self.expand_records_by_object = bool(expand_records_by_object)
        self.object_presence_prob = float(object_presence_prob)
        if not 0.0 <= self.object_presence_prob <= 1.0:
            raise ValueError(f"object_presence_prob must be in [0, 1], got {self.object_presence_prob}")
        self.normalize_object_translation_by_depth_mean = bool(normalize_object_translation_by_depth_mean)
        self.depth_mean_eps = float(depth_mean_eps)

        self.align_data = self._load_json(self.align_json)
        real_align = self.align_data["datasets"]["real275"]
        self.class_id_to_name = {int(k): str(v) for k, v in real_align["class_id_to_name"].items()}
        self.category_name_to_id = {v: k for k, v in self.class_id_to_name.items()}
        self.r_align_by_class_id = {
            int(class_id): np.asarray(item["R_align"], dtype=np.float32).reshape(3, 3)
            for class_id, item in real_align["classes"].items()
        }

        self.object_records_by_name = self._build_object_records_by_name()
        self.object_name_to_id = {
            name: idx + 1 for idx, name in enumerate(sorted(self.object_records_by_name.keys()))
        }
        self.object_id_to_name = {idx: name for name, idx in self.object_name_to_id.items()}
        self.records = self._build_records()
        self.scenes = self.records
        if not self.records:
            raise RuntimeError(
                f"No REAL275 samples found. split_root={self.split_root} gt_root={self.gt_root} "
                f"object_image_root={self.object_image_root}"
            )
        logger.info(
            "Real275CameraPose initialized: records=%d objects=%d object_views=%s",
            len(self.records),
            len(self.object_records_by_name),
            self.fixed_object_view_ids,
        )

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _load_pickle(path: Path) -> Dict[str, Any]:
        with path.open("rb") as handle:
            return pickle.load(handle)

    @staticmethod
    def _read_instance_mask(mask_path: Path, inst_id: int) -> np.ndarray:
        mask = np.asarray(Image.open(mask_path), dtype=np.uint8)
        return (mask == int(inst_id)).astype(np.float32)

    @staticmethod
    def _resize_image(image: Image.Image, resolution) -> Image.Image:
        width, height = resolution
        resampling = getattr(Image, "Resampling", Image)
        return image.resize((width, height), resampling.LANCZOS)

    def _build_object_records_by_name(self) -> Dict[str, Dict[str, Any]]:
        if not self.object_image_root.is_dir():
            raise FileNotFoundError(f"REAL275 object image root not found: {self.object_image_root}")
        image_ids = [int(x) for x in (self.fixed_object_view_ids or self.DEFAULT_OBJECT_VIEW_IDS)]
        records: Dict[str, Dict[str, Any]] = {}
        missing_fixed = []
        for object_dir in sorted(self.object_image_root.iterdir()):
            if not object_dir.is_dir():
                continue
            rgb_dir = object_dir / "rgb"
            if not rgb_dir.is_dir():
                continue
            available_ids = sorted(int(path.stem) for path in rgb_dir.glob("*.png") if path.stem.isdigit())
            if not available_ids:
                continue
            if self.fixed_object_view_ids is not None and self.strict_fixed_object_view_ids:
                missing = [image_id for image_id in image_ids if image_id not in available_ids]
                if missing:
                    missing_fixed.append((object_dir.name, missing))
                    continue
                selected_ids = image_ids
            else:
                selected_ids = image_ids if self.fixed_object_view_ids is not None else available_ids[: self.num_object_views]
            metadata_path = object_dir / "metadata.json"
            metadata = self._load_json(metadata_path) if metadata_path.is_file() else {}
            category = str(metadata.get("class_name", object_dir.name.split("_", 1)[0]))
            class_id = int(metadata.get("class_id", self.category_name_to_id.get(category, 0)))
            records[object_dir.name] = {
                "object_name": object_dir.name,
                "object_dir": object_dir,
                "image_ids": [int(x) for x in selected_ids],
                "category": category,
                "class_id": class_id,
                "metadata": metadata,
            }
        if missing_fixed:
            logger.warning(
                "Skipped %d REAL275 object refs missing fixed views. First examples: %s",
                len(missing_fixed),
                missing_fixed[:5],
            )
        return records

    def _gt_path_for(self, scene_name: str, frame_id: str) -> Path:
        return self.gt_root / f"results_real_train_{scene_name}_{frame_id}.pkl"

    def _build_records(self) -> List[Dict[str, Any]]:
        if not self.split_root.is_dir():
            raise FileNotFoundError(f"REAL275 split root not found: {self.split_root}")
        if not self.gt_root.is_dir():
            raise FileNotFoundError(f"REAL275 GT root not found: {self.gt_root}")

        records: List[Dict[str, Any]] = []
        skipped_no_ref = 0
        skipped_missing_file = 0
        for scene_dir in sorted(self.split_root.glob("scene_*")):
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
                gt_path = self._gt_path_for(scene_name, frame_id)
                if self.verify_files and (
                    not depth_path.is_file()
                    or not mask_path.is_file()
                    or not meta_path.is_file()
                    or not gt_path.is_file()
                ):
                    skipped_missing_file += 1
                    continue
                gt = self._load_pickle(gt_path)
                object_indices = list(range(len(gt["model_names"])))
                if not self.expand_records_by_object:
                    object_indices = object_indices[:1]
                for object_index in object_indices:
                    object_name = str(gt["model_names"][object_index])
                    class_id = int(gt["class_ids"][object_index])
                    inst_id = int(gt["inst_ids"][object_index])
                    category = self.class_id_to_name.get(class_id, "")
                    if self.only_object_name and object_name != self.only_object_name:
                        continue
                    if self.only_category and category != self.only_category:
                        continue
                    if object_name not in self.object_records_by_name:
                        skipped_no_ref += 1
                        continue
                    records.append(
                        {
                            "scene_name": scene_name,
                            "run_name": scene_name,
                            "scene_dir": scene_dir,
                            "frame_id": frame_id,
                            "image_id": int(frame_id),
                            "image_ids": [int(frame_id)],
                            "rgb_path": rgb_path,
                            "depth_path": depth_path,
                            "mask_path": mask_path,
                            "meta_path": meta_path,
                            "gt_path": gt_path,
                            "object_index": int(object_index),
                            "object_name": object_name,
                            "object_id": self.object_name_to_id[object_name],
                            "inst_id": inst_id,
                            "class_id": class_id,
                            "category": category,
                            "scene_source": "real275",
                        }
                    )
                    if self.max_records is not None and len(records) >= self.max_records:
                        return records

        logger.info(
            "REAL275 records built: total=%d skipped_no_ref=%d skipped_missing_file=%d",
            len(records),
            skipped_no_ref,
            skipped_missing_file,
        )
        return records

    def _read_depth_m(self, depth_path: Path, depth_scale: float = 1000.0) -> np.ndarray:
        depth_raw = np.asarray(Image.open(depth_path), dtype=np.float32)
        depth_m = depth_raw / float(depth_scale)
        depth_m[~np.isfinite(depth_m)] = 0.0
        depth_m[depth_m < 0.0] = 0.0
        return depth_m.astype(np.float32)

    def _crop_resize_if_necessary_with_mask(
        self,
        image,
        depthmap,
        object_mask,
        intrinsics,
        resolution,
        rng=None,
        info=None,
    ):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        if object_mask.shape[:2] != depthmap.shape[:2]:
            raise ValueError(
                f"Object mask shape mismatch for {info}: mask={object_mask.shape[:2]} depth={depthmap.shape[:2]}"
            )
        width, height = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, width - cx)
        min_margin_y = min(cy, height - cy)
        assert min_margin_x > width / 5, f"Bad principal point in view={info}"
        assert min_margin_y > height / 5, f"Bad principal point in view={info}"
        left, top = cx - min_margin_x, cy - min_margin_y
        right, bottom = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (left, top, right, bottom)
        image, depthmap, intrinsics = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)
        object_mask = object_mask[top:bottom, left:right]

        target_resolution = np.array(resolution)
        if self.aug_focal:
            crop_scale = self.aug_focal + (1.0 - self.aug_focal) * np.random.beta(0.5, 0.5)
            image, depthmap, intrinsics = cropping.center_crop_image_depthmap(image, depthmap, intrinsics, crop_scale)
        if self.aug_crop > 1:
            target_resolution += rng.integers(0, self.aug_crop)

        input_resolution = np.array(image.size)
        output_resolution = np.floor(input_resolution * (max(target_resolution / image.size) + 1e-8)).astype(int)
        image, depthmap, intrinsics = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)
        resampling = getattr(Image, "Resampling", Image)
        object_mask = np.asarray(
            Image.fromarray((object_mask > 0).astype(np.uint8) * 255).resize(
                tuple(output_resolution),
                resampling.NEAREST,
            ),
            dtype=np.uint8,
        ) > 0

        intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        left, top, right, bottom = crop_bbox
        image, depthmap, intrinsics2 = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)
        object_mask = object_mask[top:bottom, left:right]
        return image, depthmap, object_mask.astype(np.bool_), intrinsics2

    def _load_scene_view(self, rec: Dict[str, Any], gt: Dict[str, Any], resolution, rng):
        image = Image.open(rec["rgb_path"]).convert("RGB")
        depthmap = self._read_depth_m(rec["depth_path"], gt.get("depth_scale", 1000.0))
        intrinsic = np.asarray(gt.get("camera_K", self.K_REAL), dtype=np.float32).reshape(3, 3)
        if rec.get("mask_path") is None:
            image, depthmap, intrinsic = self._crop_resize_if_necessary(
                image, depthmap, intrinsic, resolution, rng, info=str(rec["rgb_path"])
            )
            object_mask = np.zeros(depthmap.shape[:2], dtype=np.bool_)
        else:
            object_mask = self._read_instance_mask(rec["mask_path"], int(rec["inst_id"]))
            image, depthmap, object_mask, intrinsic = self._crop_resize_if_necessary_with_mask(
                image,
                depthmap,
                object_mask,
                intrinsic,
                resolution,
                rng,
                info=str(rec["rgb_path"]),
            )

        extrinsic = np.concatenate([np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)], axis=1)
        _, point_mask = depthmap_to_absolute_camera_coordinates(depthmap, intrinsic, extrinsic, z_far=self.z_far)
        view = {
            "img": image,
            "depthmap": depthmap.astype(np.float32),
            "camera_pose": extrinsic.astype(np.float32),
            "camera_intrinsics": intrinsic.astype(np.float32),
            "point_mask": point_mask,
            "object_mask": object_mask,
            "label": f"real275/{rec['scene_name']}",
            "instance": rec["rgb_path"].name,
            "dataset": self.dataset_label,
            "image_path": str(rec["rgb_path"]),
            "depth_path": str(rec["depth_path"]),
            "camera_path": str(rec["gt_path"]),
        }
        return view

    @staticmethod
    def _object_size_from_metadata(metadata: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        bounds = np.asarray(metadata.get("mesh", {}).get("centered_bounds", []), dtype=np.float32)
        if bounds.shape == (2, 3):
            size = np.clip(bounds[1] - bounds[0], 1e-6, None)
        else:
            size = np.ones(3, dtype=np.float32)
        return size.astype(np.float32), np.log(size).astype(np.float32)

    def _sample_absent_object_name(self, gt: Dict[str, Any], positive_object_name: str, rng) -> Optional[str]:
        present_names = {str(name) for name in gt.get("model_names", [])}
        candidates = [
            name
            for name, rec in self.object_records_by_name.items()
            if name not in present_names and name != str(positive_object_name)
        ]
        if self.only_object_name:
            candidates = [name for name in candidates if name == self.only_object_name]
        if self.only_category:
            candidates = [name for name in candidates if self.object_records_by_name[name].get("category", "") == self.only_category]
        if not candidates:
            return None
        return str(rng.choice(np.asarray(sorted(candidates), dtype=object)))

    def _sample_object_view_ids(self, available_ids: List[int], rng, object_name: str = "") -> List[int]:
        available_ids = [int(x) for x in available_ids]
        if self.fixed_object_view_ids is not None:
            if self.strict_fixed_object_view_ids:
                missing_ids = [int(x) for x in self.fixed_object_view_ids if int(x) not in available_ids]
                if missing_ids:
                    raise RuntimeError(f"Missing fixed object views for {object_name}: {missing_ids}")
                return list(self.fixed_object_view_ids)
            fixed_ids = [int(x) for x in self.fixed_object_view_ids if int(x) in available_ids]
            fallback_ids = [x for x in available_ids if x not in fixed_ids]
            needed = max(0, self.num_object_views - len(fixed_ids))
            extra_ids = []
            if needed:
                if len(fallback_ids) < needed:
                    raise RuntimeError(f"Insufficient object views for {object_name}: available={available_ids}")
                extra_ids = (
                    rng.choice(np.asarray(fallback_ids), size=needed, replace=False).astype(int).tolist()
                    if self.dset not in {"test", "val", "validation"}
                    else fallback_ids[:needed]
                )
            return fixed_ids + extra_ids
        count = min(self.num_object_views, len(available_ids))
        if self.dset not in {"test", "val", "validation"}:
            return rng.choice(np.asarray(available_ids), size=count, replace=False).astype(int).tolist()
        return available_ids[:count]

    def _load_object_images(self, object_name: str, resolution, rng) -> Dict[str, Any]:
        object_rec = self.object_records_by_name[object_name]
        image_ids = self._sample_object_view_ids(object_rec["image_ids"], rng, object_name=object_name)
        tensors = []
        true_shapes = []
        image_paths = []
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

    def _aligned_pose_and_size(self, gt: Dict[str, Any], object_index: int, class_id: int):
        r_native_to_cam = np.asarray(gt["rotations"][object_index], dtype=np.float32).reshape(3, 3)
        t_cam = np.asarray(gt["translations"][object_index], dtype=np.float32).reshape(3)
        size_native = np.asarray(gt["extents_m"][object_index], dtype=np.float32).reshape(3)
        r_align = self.r_align_by_class_id.get(int(class_id))
        if r_align is None:
            logger.warning("No REAL275 R_align for class_id=%s; using identity", class_id)
            r_align = np.eye(3, dtype=np.float32)
        r_aligned_to_cam = r_native_to_cam @ r_align.T
        size_aligned = np.abs(r_align) @ size_native
        size_aligned = np.clip(size_aligned, 1e-6, None)
        return (
            r_aligned_to_cam.astype(np.float32),
            t_cam.astype(np.float32),
            size_aligned.astype(np.float32),
            np.log(size_aligned).astype(np.float32),
            r_align.astype(np.float32),
            r_native_to_cam.astype(np.float32),
            size_native.astype(np.float32),
        )

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
        gt = self._load_pickle(rec["gt_path"])
        target_object_name = str(rec["object_name"])
        target_object_id = int(rec["object_id"])
        target_category = str(rec["category"])
        target_class_id = int(rec["class_id"])
        has_object = True
        view_rec = rec
        if self.dset not in {"test", "val", "validation"} and self.object_presence_prob < 1.0 and float(rng.random()) > self.object_presence_prob:
            absent_object_name = self._sample_absent_object_name(gt, target_object_name, rng)
            if absent_object_name is not None:
                absent_rec = self.object_records_by_name[absent_object_name]
                target_object_name = absent_object_name
                target_object_id = int(self.object_name_to_id[absent_object_name])
                target_category = str(absent_rec.get("category", ""))
                target_class_id = int(absent_rec.get("class_id", 0))
                has_object = False
                view_rec = dict(rec, mask_path=None)

        view = self._load_scene_view(view_rec, gt, resolution, rng)
        view["idx"] = (idx, ar_idx, 0)
        view["z_far"] = self.z_far
        view["true_shape"] = np.int32(view["img"].size[::-1])
        view["img"] = self.transform(view["img"])
        for key, value in view.items():
            res, err_msg = is_good_type(key, value)
            assert res, f"{err_msg} with {key}={value} for view {view_name(view)}"
        transpose_to_landscape(view)
        view["rng"] = int.from_bytes(rng.bytes(4), "big")

        object_index = int(rec["object_index"])
        if has_object:
            object_rotation, object_translation_metric, object_size, object_size_log, r_align, r_native, size_native = (
                self._aligned_pose_and_size(gt, object_index, int(rec["class_id"]))
            )
        else:
            object_rotation = np.eye(3, dtype=np.float32)
            object_translation_metric = np.zeros(3, dtype=np.float32)
            object_size, object_size_log = self._object_size_from_metadata(
                self.object_records_by_name[target_object_name].get("metadata", {})
            )
            r_align = np.eye(3, dtype=np.float32)
            r_native = np.eye(3, dtype=np.float32)
            size_native = object_size.copy()
        depth = view["depthmap"]
        valid_depth = depth[np.asarray(view["point_mask"], dtype=np.bool_)]
        if valid_depth.size == 0:
            depth_mean_scale = np.float32(1.0)
        else:
            depth_mean_scale = np.float32(max(float(valid_depth.mean()), self.depth_mean_eps))
        object_translation_normalized = (object_translation_metric / depth_mean_scale).astype(np.float32)
        object_translation = object_translation_normalized if self.normalize_object_translation_by_depth_mean else object_translation_metric

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
            "seq_name": f"real275/{rec['scene_name']}/{target_object_name}/{image_id:04d}",
            "scene_name": rec["scene_name"],
            "scene_source": rec["scene_source"],
            "run_name": rec["run_name"],
            "object_name": target_object_name,
            "object_id": np.array(object_id, dtype=np.int64),
            "inst_id": np.array(int(rec["inst_id"]), dtype=np.int64),
            "class_id": np.array(target_class_id, dtype=np.int64),
            "category": target_category,
            "has_object": np.array(has_object, dtype=np.bool_),
            "object_rotation": object_rotation,
            "object_translation": object_translation.astype(np.float32),
            "object_translation_metric": object_translation_metric,
            "object_translation_normalized": object_translation_normalized,
            "depth_mean_scale": np.asarray(depth_mean_scale, dtype=np.float32),
            "normalization_scale": np.asarray(depth_mean_scale, dtype=np.float32),
            "object_translation_scale": np.asarray(depth_mean_scale, dtype=np.float32),
            "object_size": object_size,
            "object_size_log": object_size_log,
            "object_srt": np.concatenate([object_rotation.reshape(-1), object_translation, object_size_log]).astype(np.float32),
            "object_rotation_native": r_native,
            "object_size_native": size_native,
            "R_align_real275_to_ov9d": r_align,
            "scene_rgb_path": view["image_path"],
            "scene_depth_path": view["depth_path"],
            "scene_camera_path": view["camera_path"],
            "scene_gt_path": str(rec["gt_path"]),
            "scene_label_path": str(rec["gt_path"]),
            "scene_meta_path": str(rec["meta_path"]),
            "scene_mask_path": str(rec["mask_path"]) if has_object else "",
            "rmse_m": np.asarray(gt["rmse_m"][object_index], dtype=np.float32),
            "n_points": np.asarray(gt["n_points"][object_index], dtype=np.int32),
            "inliers": np.asarray(gt["inliers"][object_index], dtype=np.int32),
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
