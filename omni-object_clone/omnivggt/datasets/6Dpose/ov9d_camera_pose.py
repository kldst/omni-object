import json
import os
import os.path as osp
import random
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
from omnivggt.utils.geometry import closed_form_inverse_se3, depthmap_to_absolute_camera_coordinates


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
        verify_files: bool = True,
        max_records: Optional[int] = None,
        only_scene_name: str = "",
        only_object_id: Optional[int] = None,
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
        self.verify_files = bool(verify_files)
        self.max_records = int(max_records) if max_records is not None else None
        self.only_scene_name = str(only_scene_name).strip()
        self.only_object_id = int(only_object_id) if only_object_id is not None else None

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

    def _load_scene_view(self, scene_dir: Path, image_id: int, camera_entry: Dict[str, Any], resolution, rng):
        image_path = scene_dir / "rgb" / f"{image_id:06d}.png"
        depth_path = scene_dir / "depth" / f"{image_id:06d}.png"
        mask_dir = scene_dir / "mask_visib"
        image = Image.open(image_path).convert("RGB")
        depthmap = self._read_depth_m(depth_path, camera_entry)
        intrinsic = self._as_matrix(camera_entry["cam_K"], (3, 3))
        r_w2c = self._as_matrix(camera_entry["cam_R_w2c"], (3, 3))
        t_w2c = np.asarray(camera_entry["cam_t_w2c"], dtype=np.float32).reshape(3) / 1000.0
        extrinsic = np.concatenate([r_w2c, t_w2c[:, None]], axis=1).astype(np.float32)
        c2w = closed_form_inverse_se3(
            np.concatenate([extrinsic, np.array([[0, 0, 0, 1]], dtype=np.float32)], axis=0)[None]
        )[0]

        image, depthmap, intrinsic = self._crop_resize_if_necessary(
            image, depthmap, intrinsic, resolution, rng, info=str(image_path)
        )
        _, point_mask = depthmap_to_absolute_camera_coordinates(depthmap, intrinsic, c2w, z_far=self.z_far)
        return {
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

    def _sample_object_ids(self, available_ids: List[int], rng) -> List[int]:
        if self.fixed_object_view_ids is not None:
            return list(self.fixed_object_view_ids)
        count = min(self.num_object_views, len(available_ids))
        if self.training:
            return [int(x) for x in rng.choice(np.asarray(available_ids), size=count, replace=False).tolist()]
        return [int(x) for x in available_ids[:count]]

    def _load_object_images(self, object_id: int, resolution, rng) -> Dict[str, Any]:
        candidates = self.single_records_by_object_id[int(object_id)]
        single_rec = candidates[int(rng.integers(len(candidates)))] if self.training else candidates[0]
        image_ids = self._sample_object_ids(single_rec["image_ids"], rng)
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
        object_index = self._object_index_for_id(scene_gt[str(image_id)], rec["object_id"])
        if object_index is None:
            raise KeyError(f"Object {rec['object_id']} not found in {rec['scene_name']} frame {image_id}")
        scene_mask_path = rec["scene_dir"] / "mask_visib" / f"{image_id:06d}_{object_index:06d}.png"

        view = self._load_scene_view(rec["scene_dir"], image_id, scene_camera[str(image_id)], resolution, rng)
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

        gt = scene_gt[str(image_id)][object_index]
        object_rotation = self._as_matrix(gt["cam_R_m2c"], (3, 3))
        object_translation = np.asarray(gt["cam_t_m2c"], dtype=np.float32).reshape(3) / 1000.0
        object_size, object_size_log = self._object_size_targets(rec["object_id"])

        result = {
            "images": torch.stack([view["img"]]),
            "depth": np.stack([view["depthmap"][:, :, None]]),
            "extrinsic": np.stack([view["camera_pose"][:3]]),
            "intrinsic": np.stack([view["camera_intrinsics"]]),
            "true_shape": np.stack([view["true_shape"]]),
            "valid_mask": np.stack([view["point_mask"]]),
            "label": [view["label"]],
            "instance": [view["instance"]],
            "dataset": self.dataset_label,
            "ids": np.array([image_id], dtype=np.int64),
            "camera_indices": np.array([image_id], dtype=np.int64),
            "seq_name": f"ov9d_camera_pose/{rec['scene_name']}/obj_{rec['object_id']:06d}/{image_id:06d}",
            "scene_name": rec["scene_name"],
            "run_name": rec["scene_name"],
            "object_name": f"obj_{rec['object_id']:06d}",
            "object_id": np.array(rec["object_id"], dtype=np.int64),
            "has_object": np.array(True, dtype=np.bool_),
            "object_rotation": object_rotation.astype(np.float32),
            "object_translation": object_translation.astype(np.float32),
            "object_size": object_size,
            "object_size_log": object_size_log,
            "object_srt": np.concatenate([object_rotation.reshape(-1), object_translation, object_size_log]).astype(np.float32),
            "scene_rgb_path": view["image_path"],
            "scene_depth_path": view["depth_path"],
            "scene_camera_path": view["camera_path"],
            "scene_mask_path": str(scene_mask_path),
        }
        result.update(self._load_object_images(rec["object_id"], resolution, rng))
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
