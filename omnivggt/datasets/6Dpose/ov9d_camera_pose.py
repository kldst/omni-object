import json
import os
import os.path as osp
import random
import logging
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
from omnivggt.utils.geometry import closed_form_inverse_se3, depthmap_to_absolute_camera_coordinates


logger = logging.getLogger(__name__)


class OV9DCameraPose(BaseStereoViewDataset):
    """OV9D camera-frame object pose dataset.

    Each sample is one scene camera view plus same-object reference renders.
    Object pose supervision stays in the input camera coordinate system:
    ``cam_R_m2c`` and ``cam_t_m2c / 1000``.
    """

    DEFAULT_DATA_ROOT = "/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d"

    def __init__(
        self,
        dataset_location: str = DEFAULT_DATA_ROOT,
        dset: str = "train",
        split_json: Optional[str] = None,
        multi_root: Optional[str] = None,
        single_root: Optional[str] = None,
        models_info_path: Optional[str] = None,
        name_to_oid_path: Optional[str] = None,
        num_object_views: int = 4,
        fixed_object_view_ids: Optional[Sequence[int]] = None,
        strict_fixed_object_view_ids: bool = True,
        verify_files: bool = True,
        max_records: Optional[int] = None,
        only_scene_name: str = "",
        only_object_id: Optional[int] = None,
        object_presence_prob: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dataset_label = "OV9DCameraPose"
        self.dataset_location = Path(dataset_location)
        self.dset = str(dset)
        self.training = self.dset.lower() not in {"test", "val", "validation", "test1", "test2", "test3"}
        self.multi_root = Path(multi_root) if multi_root else self.dataset_location / "oo3d9dmulti"
        self.single_root = Path(single_root) if single_root else self.dataset_location / "oo3d9dsingle"
        self.split_json = Path(split_json) if split_json else self._default_split_json(self.dset)
        self.models_info_path = Path(models_info_path) if models_info_path else self.dataset_location / "models_info.json"
        self.name_to_oid_path = Path(name_to_oid_path) if name_to_oid_path else self.dataset_location / "name2oid.json"
        self.num_object_views = int(num_object_views)
        self.fixed_object_view_ids = tuple(int(x) for x in fixed_object_view_ids) if fixed_object_view_ids else None
        self.strict_fixed_object_view_ids = bool(strict_fixed_object_view_ids)
        self.verify_files = bool(verify_files)
        self.max_records = int(max_records) if max_records is not None else None
        self.only_scene_name = str(only_scene_name).strip()
        self.only_object_id = int(only_object_id) if only_object_id is not None else None
        self.object_presence_prob = float(object_presence_prob)
        if not 0.0 <= self.object_presence_prob <= 1.0:
            raise ValueError(f"object_presence_prob must be in [0, 1], got {self.object_presence_prob}")

        self.models_info = self._load_json(self.models_info_path)
        self.name_to_oid = self._load_json(self.name_to_oid_path)
        self.single_records_by_object_id = self._build_single_records_by_object_id()
        self.records = self._build_records()
        self.scenes = self.records
        if not self.records:
            raise RuntimeError(
                "No OV9D camera-frame samples found. "
                f"split_json={self.split_json}, multi_root={self.multi_root}, single_root={self.single_root}"
            )
        logger.info(
            "OV9DCameraPose initialized: dset=%s records=%d object_presence_prob=%.3f "
            "fixed_object_view_ids=%s strict_fixed_object_view_ids=%s",
            self.dset,
            len(self.records),
            self.object_presence_prob,
            self.fixed_object_view_ids,
            self.strict_fixed_object_view_ids,
        )

    def _default_split_json(self, dset: str) -> Path:
        split = "test1" if dset in {"val", "validation", "test"} else str(dset)
        candidates = [
            self.dataset_location / "splits_multi_0504_500" / f"{split}.json",
            self.dataset_location / "splits_multi_0430_5000" / f"{split}.json",
            self.dataset_location / "splits_multi_0431_300" / f"{split}.json",
            self.dataset_location / "splits" / f"{split}.json",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return candidates[0]

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _as_matrix(values: Any, shape: tuple[int, int]) -> np.ndarray:
        return np.asarray(values, dtype=np.float32).reshape(shape)

    @staticmethod
    def _object_index_for_id(gts: List[Dict[str, Any]], object_id: int) -> Optional[int]:
        for idx, gt in enumerate(gts):
            if int(gt.get("obj_id", -1)) == int(object_id):
                return idx
        return None

    @classmethod
    def _image_ids_with_object(cls, scene_gt: Dict[str, Any], object_id: int) -> List[int]:
        image_ids = []
        for image_id_str, gts in scene_gt.items():
            if cls._object_index_for_id(gts, object_id) is not None:
                image_ids.append(int(image_id_str))
        return sorted(image_ids)

    def _verify_scene_files(self, scene_dir: Path) -> bool:
        required = [scene_dir / "scene_gt.json", scene_dir / "scene_camera.json", scene_dir / "rgb", scene_dir / "depth"]
        return all(path.exists() for path in required)

    def _build_single_records_by_object_id(self) -> Dict[int, List[Dict[str, Any]]]:
        records: Dict[int, List[Dict[str, Any]]] = {}
        if not self.single_root.is_dir():
            return records

        for scene_dir in sorted(path for path in self.single_root.iterdir() if path.is_dir()):
            name_parts = scene_dir.name.split("_")
            object_instance = "_".join(name_parts[:-1]) if len(name_parts) > 2 else scene_dir.name
            object_id = self.name_to_oid.get(object_instance)
            if object_id is None:
                continue
            image_ids = []
            for rgb_path in sorted((scene_dir / "rgb").glob("*.png")):
                image_id = int(rgb_path.stem)
                mask_path = scene_dir / "mask_visib" / f"{image_id:06d}_000000.png"
                if not self.verify_files or mask_path.is_file():
                    image_ids.append(image_id)
            if self.fixed_object_view_ids is not None and self.strict_fixed_object_view_ids:
                missing_ids = [image_id for image_id in self.fixed_object_view_ids if image_id not in image_ids]
                if missing_ids:
                    logger.warning(
                        "Skipping object reference scene %s because fixed_object_view_ids are missing: %s",
                        scene_dir.name,
                        missing_ids,
                    )
                    continue
            if len(image_ids) < self.num_object_views:
                continue
            records.setdefault(int(object_id), []).append(
                {
                    "scene_dir": scene_dir,
                    "scene_name": scene_dir.name,
                    "image_ids": image_ids,
                    "object_instance": object_instance,
                }
            )
        return records

    def _build_records(self) -> List[Dict[str, Any]]:
        if not self.split_json.is_file():
            raise FileNotFoundError(f"Split JSON not found: {self.split_json}")
        payload = self._load_json(self.split_json)
        records: List[Dict[str, Any]] = []
        for item in payload.get("scenes", []):
            scene_name = str(item["scene_name"])
            if self.only_scene_name and scene_name != self.only_scene_name:
                continue
            scene_dir = self.multi_root / scene_name
            if self.verify_files and not self._verify_scene_files(scene_dir):
                continue
            scene_gt = self._load_json(scene_dir / "scene_gt.json")
            object_ids = [int(x) for x in item.get("eligible_object_ids", item.get("object_ids", []))]
            if self.only_object_id is not None:
                object_ids = [object_id for object_id in object_ids if object_id == self.only_object_id]
            for object_id in object_ids:
                if object_id not in self.single_records_by_object_id:
                    continue
                image_ids = self._image_ids_with_object(scene_gt, object_id)
                if not image_ids:
                    continue
                for image_id in image_ids:
                    records.append(
                        {
                            "scene_name": scene_name,
                            "run_name": scene_name,
                            "scene_dir": scene_dir,
                            "object_id": object_id,
                            "image_ids": [image_id],
                        }
                    )
                    if self.max_records is not None and len(records) >= self.max_records:
                        return records
        return records

    def _read_depth_m(self, depth_path: Path, camera_entry: Dict[str, Any]) -> np.ndarray:
        depth_raw = np.asarray(Image.open(depth_path), dtype=np.float32)
        depth_scale = float(camera_entry.get("depth_scale", 1.0))
        depth_m = depth_raw * depth_scale / 1000.0
        depth_m[~np.isfinite(depth_m)] = 0.0
        depth_m[depth_m < 0.0] = 0.0
        return depth_m.astype(np.float32)

    @staticmethod
    def _read_binary_mask(mask_path: Path) -> np.ndarray:
        return (np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0).astype(np.float32)

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
            input_resolution = np.array(image.size)
            output_resolution = np.floor(input_resolution * crop_scale).astype(int)
            margins = input_resolution - output_resolution
            offset = margins / 2
            left, top = offset.astype(int)
            right = left + output_resolution[0]
            bottom = top + output_resolution[1]
            crop_bbox = (left, top, right, bottom)
            image, depthmap, intrinsics = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)
            object_mask = object_mask[top:bottom, left:right]

        if self.aug_crop > 1:
            target_resolution += rng.integers(0, self.aug_crop)

        input_resolution = np.array(image.size)
        scale_final = max(target_resolution / image.size) + 1e-8
        output_resolution = np.floor(input_resolution * scale_final).astype(int)
        image, depthmap, intrinsics = cropping.rescale_image_depthmap(
            image,
            depthmap,
            intrinsics,
            target_resolution,
        )
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

    def _load_scene_view(
        self,
        scene_dir: Path,
        image_id: int,
        camera_entry: Dict[str, Any],
        resolution,
        rng,
        object_mask_path: Optional[Path] = None,
    ):
        image_path = scene_dir / "rgb" / f"{image_id:06d}.png"
        depth_path = scene_dir / "depth" / f"{image_id:06d}.png"
        mask_dir = scene_dir / "mask_visib"
        image = Image.open(image_path).convert("RGB")
        depthmap = self._read_depth_m(depth_path, camera_entry)
        object_mask = self._read_binary_mask(object_mask_path) if object_mask_path is not None else None
        intrinsic = self._as_matrix(camera_entry["cam_K"], (3, 3))
        r_w2c = self._as_matrix(camera_entry["cam_R_w2c"], (3, 3))
        t_w2c = np.asarray(camera_entry["cam_t_w2c"], dtype=np.float32).reshape(3) / 1000.0
        extrinsic = np.concatenate([r_w2c, t_w2c[:, None]], axis=1).astype(np.float32)
        c2w = closed_form_inverse_se3(
            np.concatenate([extrinsic, np.array([[0, 0, 0, 1]], dtype=np.float32)], axis=0)[None]
        )[0]

        if object_mask is None:
            image, depthmap, intrinsic = self._crop_resize_if_necessary(
                image, depthmap, intrinsic, resolution, rng, info=str(image_path)
            )
        else:
            image, depthmap, object_mask, intrinsic = self._crop_resize_if_necessary_with_mask(
                image,
                depthmap,
                object_mask,
                intrinsic,
                resolution,
                rng,
                info=str(image_path),
            )
        _, point_mask = depthmap_to_absolute_camera_coordinates(depthmap, intrinsic, c2w, z_far=self.z_far)
        view = {
            "img": image,
            "depthmap": depthmap.astype(np.float32),
            "camera_pose": extrinsic,
            "camera_intrinsics": intrinsic.astype(np.float32),
            "point_mask": point_mask,
            "label": scene_dir.name,
            "instance": image_path.name,
            "image_path": str(image_path),
            "depth_path": str(depth_path),
            "camera_path": str(scene_dir / "scene_camera.json"),
            "mask_dir": str(mask_dir),
        }
        if object_mask is not None:
            view["object_mask"] = object_mask
        return view

    @staticmethod
    def _resize_image(image: Image.Image, resolution) -> Image.Image:
        width, height = resolution
        resampling = getattr(Image, "Resampling", Image)
        return image.resize((width, height), resampling.LANCZOS)

    @staticmethod
    def _apply_mask_white_background(rgb: Image.Image, mask_path: Path) -> Image.Image:
        rgb_arr = np.asarray(rgb.convert("RGB"), dtype=np.uint8)
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        out = np.full_like(rgb_arr, 255)
        out[mask > 0] = rgb_arr[mask > 0]
        return Image.fromarray(out, mode="RGB")

    def _sample_object_ids(self, available_ids: List[int], rng, object_name: str = "") -> List[int]:
        available_ids = [int(x) for x in available_ids]
        if self.fixed_object_view_ids is not None:
            if self.strict_fixed_object_view_ids:
                missing_ids = [int(x) for x in self.fixed_object_view_ids if int(x) not in available_ids]
                if missing_ids:
                    raise RuntimeError(
                        f"Missing fixed_object_view_ids for {object_name or 'object'}: "
                        f"missing={missing_ids}, available={available_ids}"
                    )
                return list(self.fixed_object_view_ids)
            fixed_ids = [int(x) for x in self.fixed_object_view_ids if int(x) in available_ids]
            missing_ids = [int(x) for x in self.fixed_object_view_ids if int(x) not in available_ids]
            fallback_ids = [image_id for image_id in available_ids if image_id not in fixed_ids]
            needed = max(0, self.num_object_views - len(fixed_ids))
            if needed > 0:
                if len(fallback_ids) < needed:
                    raise RuntimeError(
                        f"Object reference views are insufficient for {object_name or 'object'}: "
                        f"fixed={self.fixed_object_view_ids}, available={available_ids}, "
                        f"need={self.num_object_views}"
                    )
                if self.training:
                    extra_ids = rng.choice(np.asarray(fallback_ids), size=needed, replace=False).tolist()
                    extra_ids = [int(x) for x in extra_ids]
                else:
                    extra_ids = fallback_ids[:needed]
            else:
                extra_ids = []
            selected_ids = fixed_ids + extra_ids
            if missing_ids:
                logger.warning(
                    "Missing fixed_object_view_ids for %s: missing=%s available_count=%d fallback=%s selected=%s",
                    object_name or "object",
                    missing_ids,
                    len(available_ids),
                    extra_ids,
                    selected_ids,
                )
            return selected_ids
        count = min(self.num_object_views, len(available_ids))
        if self.training:
            return [int(x) for x in rng.choice(np.asarray(available_ids), size=count, replace=False).tolist()]
        return [int(x) for x in available_ids[:count]]

    def _load_object_images(self, object_id: int, resolution, rng) -> Dict[str, Any]:
        candidates = self.single_records_by_object_id[int(object_id)]
        single_rec = candidates[int(rng.integers(len(candidates)))] if self.training else candidates[0]
        image_ids = self._sample_object_ids(single_rec["image_ids"], rng, object_name=single_rec["scene_name"])
        tensors = []
        true_shapes = []
        image_paths = []
        mask_paths = []
        for image_id in image_ids:
            image_path = single_rec["scene_dir"] / "rgb" / f"{image_id:06d}.png"
            mask_path = single_rec["scene_dir"] / "mask_visib" / f"{image_id:06d}_000000.png"
            image = self._apply_mask_white_background(Image.open(image_path), mask_path)
            true_shapes.append(np.array(image.size[::-1], dtype=np.int32))
            tensors.append(self.transform(self._resize_image(image, resolution)))
            image_paths.append(str(image_path))
            mask_paths.append(str(mask_path))
        return {
            "object_images": torch.stack(tensors),
            "object_true_shape": np.stack(true_shapes),
            "object_cam_indices": np.asarray(image_ids, dtype=np.int64),
            "object_reference_scene_name": single_rec["scene_name"],
            "object_rgb_paths": image_paths,
            "object_mask_paths": mask_paths,
        }

    def _object_size_targets(self, object_id: int) -> tuple[np.ndarray, np.ndarray]:
        info = self.models_info[str(int(object_id))]
        size_m = np.array([info["size_x"], info["size_y"], info["size_z"]], dtype=np.float32) / 1000.0
        size_m = np.clip(size_m, 1e-6, None)
        return size_m.astype(np.float32), np.log(size_m).astype(np.float32)

    def _sample_absent_object_id(self, scene_gt: Dict[str, Any], image_id: int, positive_object_id: int, rng) -> Optional[int]:
        present_ids = {int(gt.get("obj_id", -1)) for gt in scene_gt[str(image_id)]}
        candidates = [
            object_id
            for object_id in self.single_records_by_object_id.keys()
            if object_id not in present_ids and object_id != int(positive_object_id)
        ]
        if self.only_object_id is not None:
            candidates = [object_id for object_id in candidates if object_id == self.only_object_id]
        if not candidates:
            return None
        return int(rng.choice(np.asarray(sorted(candidates), dtype=np.int64)))

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
        scene_gt = self._load_json(rec["scene_dir"] / "scene_gt.json")
        scene_camera = self._load_json(rec["scene_dir"] / "scene_camera.json")
        image_id = int(rng.choice(rec["image_ids"])) if self.training else int(rec["image_ids"][0])
        target_object_id = int(rec["object_id"])
        has_object = True
        if self.training and self.object_presence_prob < 1.0 and float(rng.random()) > self.object_presence_prob:
            absent_object_id = self._sample_absent_object_id(scene_gt, image_id, target_object_id, rng)
            if absent_object_id is not None:
                target_object_id = absent_object_id
                has_object = False

        object_index = self._object_index_for_id(scene_gt[str(image_id)], target_object_id)
        if has_object:
            if object_index is None:
                raise KeyError(f"Object {target_object_id} not found in {rec['scene_name']} frame {image_id}")
            scene_mask_path = rec["scene_dir"] / "mask_visib" / f"{image_id:06d}_{object_index:06d}.png"
        else:
            scene_mask_path = None

        view = self._load_scene_view(
            rec["scene_dir"],
            image_id,
            scene_camera[str(image_id)],
            resolution,
            rng,
            object_mask_path=scene_mask_path,
        )
        if "object_mask" not in view:
            view["object_mask"] = np.zeros_like(view["point_mask"], dtype=np.bool_)
        view["idx"] = (idx, ar_idx, 0)
        view["dataset"] = self.dataset_label
        view["z_far"] = self.z_far
        view["true_shape"] = np.int32(view["img"].size[::-1])
        view["img"] = self.transform(view["img"])
        for key, value in view.items():
            res, err_msg = is_good_type(key, value)
            assert res, f"{err_msg} with {key}={value} for view {view_name(view)}"
        transpose_to_landscape(view)
        view["rng"] = int.from_bytes(rng.bytes(4), "big")

        if has_object:
            gt = scene_gt[str(image_id)][object_index]
            object_rotation = self._as_matrix(gt["cam_R_m2c"], (3, 3))
            object_translation = np.asarray(gt["cam_t_m2c"], dtype=np.float32).reshape(3) / 1000.0
        else:
            object_rotation = np.eye(3, dtype=np.float32)
            object_translation = np.zeros(3, dtype=np.float32)
        object_size, object_size_log = self._object_size_targets(target_object_id)

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
            "seq_name": f"ov9d_camera_pose/{rec['scene_name']}/obj_{target_object_id:06d}/{image_id:06d}",
            "scene_name": rec["scene_name"],
            "run_name": rec["scene_name"],
            "object_name": f"obj_{target_object_id:06d}",
            "object_id": np.array(target_object_id, dtype=np.int64),
            "has_object": np.array(has_object, dtype=np.bool_),
            "object_rotation": object_rotation.astype(np.float32),
            "object_translation": object_translation.astype(np.float32),
            "object_size": object_size,
            "object_size_log": object_size_log,
            "object_srt": np.concatenate([object_rotation.reshape(-1), object_translation, object_size_log]).astype(np.float32),
            "scene_rgb_path": view["image_path"],
            "scene_depth_path": view["depth_path"],
            "scene_camera_path": view["camera_path"],
            "scene_mask_path": str(scene_mask_path) if scene_mask_path is not None else "",
        }
        result.update(self._load_object_images(target_object_id, resolution, rng))
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
