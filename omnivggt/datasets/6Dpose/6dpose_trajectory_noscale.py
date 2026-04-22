# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# 6D pose dataset with scene RGB + scene depth + object reference images
# --------------------------------------------------------
import os
import os.path as osp
import re
import json
from typing import Dict, List, Optional, Sequence

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
from omnivggt.datasets.utils.misc import threshold_depth_map
from omnivggt.utils.geometry import closed_form_inverse_se3


class SixDPose(BaseStereoViewDataset):
    DEFAULT_OBJECT_VIEWS = (1, 5, 10, 15)

    def __init__(
        self,
        dataset_location="/mnt/train-data-4-hdd/yian/freepose/0421_randon_4000scene_30pose",
        dset="train",
        OBJECT_INPUT_ROOT: Optional[str] = None,
        OBJECT_IMAGE_ROOT: Optional[str] = None,
        selected_views: Optional[Sequence[int]] = None,
        scene_num_views: int = 1,
        object_input_views: Optional[Sequence[int]] = None,
        depth_scale: float = 1000.0,
        use_opencv_camera: bool = True,
        verify_files: bool = True,
        only_scene_name: str = "",
        only_scene_names: Optional[List[str]] = None,
        only_scene_start: Optional[str] = None,
        only_scene_end: Optional[str] = None,
        only_frame_name: str = "",
        quick: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.dataset_label = "SixDPose"
        self.dset = dset
        self.dataset_location = dataset_location
        self.training = str(dset).lower() not in {"test", "val", "validation"}
        self.object_root = self._resolve_object_root(dataset_location, OBJECT_INPUT_ROOT or OBJECT_IMAGE_ROOT)
        self.scene_root = dataset_location
        self.depth_root = dataset_location
        self.cam_root = dataset_location
        self.pose_root = dataset_location

        self.depth_scale = float(depth_scale)
        self.use_opencv_camera = bool(use_opencv_camera)
        self.verify_files = bool(verify_files)
        self.quick = bool(quick)
        self.scene_view_pool = (0,)
        self.scene_num_views = 1
        self.object_view_pool = tuple(int(v) for v in (object_input_views or self.DEFAULT_OBJECT_VIEWS))
        self.object_dir_lookup = self._build_object_dir_lookup(self.object_root)
        self.only_scene_name = (only_scene_name or "").strip()
        self.only_scene_names = [x.strip() for x in (only_scene_names or []) if str(x).strip()]
        if self.only_scene_name:
            self.only_scene_names = [self.only_scene_name]
        self.only_scene_start_idx = (
            self._parse_scene_index(only_scene_start) if only_scene_start is not None else None
        )
        self.only_scene_end_idx = (
            self._parse_scene_index(only_scene_end) if only_scene_end is not None else None
        )
        self.only_frame_name = (only_frame_name or "").strip()

        self.records = self._build_records()
        self.scenes = self.records

        if not self.records:
            raise RuntimeError(
                "No valid 6D pose samples found. "
                f"scene_root={self.scene_root}, depth_root={self.depth_root}, "
                f"cam_root={self.cam_root}, pose_root={self.pose_root}, object_root={self.object_root}"
            )

    def _resolve_object_root(self, dataset_location: str, object_root: Optional[str]) -> str:
        candidates = [
            object_root,
            osp.join(dataset_location, "object_space_rgb"),
            "/mnt/train-data-4-hdd/yian/6dpose_obj/0316_fixedCam_1k/object_space_rgb",
            "/mnt/train-data-4-hdd/yian/6dpose_obj/0315_fixedCam_1k/object_space_rgb",
        ]
        for candidate in candidates:
            if candidate and osp.isdir(candidate):
                return candidate
        raise FileNotFoundError(
            "Unable to locate object image root. "
            "Please provide OBJECT_INPUT_ROOT/OBJECT_IMAGE_ROOT explicitly."
        )

    @staticmethod
    def _normalize_object_key(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(name).lower())

    @classmethod
    def _candidate_object_dir_names(cls, object_name: str) -> List[str]:
        object_name = str(object_name)
        candidates = [object_name]
        if object_name.startswith("freepose_obj_"):
            remainder = object_name[len("freepose_obj_") :]
            candidates.append(remainder.replace("_obj_", "__obj_"))
        if object_name.startswith("google_") and object_name.endswith("_meshes_model"):
            core = object_name[len("google_") : -len("_meshes_model")]
            candidates.append(f"google__{core}__meshes")
        return list(dict.fromkeys(candidates))

    @classmethod
    def _build_object_dir_lookup(cls, object_root: str) -> Dict[str, str]:
        lookup = {}
        for entry in os.listdir(object_root):
            entry_path = osp.join(object_root, entry)
            if not osp.isdir(entry_path):
                continue
            lookup.setdefault(cls._normalize_object_key(entry), entry)
        return lookup

    def _resolve_object_dir_name(self, object_name: str) -> Optional[str]:
        for candidate in self._candidate_object_dir_names(object_name):
            candidate_path = osp.join(self.object_root, candidate)
            if osp.isdir(candidate_path):
                return candidate

        for candidate in self._candidate_object_dir_names(object_name):
            normalized = self._normalize_object_key(candidate)
            matched = self.object_dir_lookup.get(normalized)
            if matched:
                return matched
        return None

    @staticmethod
    def _parse_scene_index(scene_name: Optional[str]) -> Optional[int]:
        if scene_name is None:
            return None
        value = str(scene_name).strip()
        if not value:
            return None
        if value.isdigit():
            return int(value)
        if value.startswith("scene_") and value[6:].isdigit():
            return int(value[6:])
        raise ValueError(f"Invalid scene spec '{scene_name}'. Expected e.g. 'scene_0000' or '0'.")

    @staticmethod
    def _quat_wxyz_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
        w, x, y, z = [float(v) for v in quat_wxyz]
        n = np.sqrt(w * w + x * x + y * y + z * z)
        if n < 1e-8:
            return np.eye(3, dtype=np.float32)
        w, x, y, z = w / n, x / n, y / n, z / n
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float32,
        )

    def _load_pose_lookup(self, pose_path: str) -> Dict[str, Dict[str, np.ndarray]]:
        pose_data = np.load(pose_path, allow_pickle=False)
        names = [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in pose_data["names"]]
        positions = pose_data["positions"].astype(np.float32)
        quats = pose_data["rot_quat_wxyz"].astype(np.float32)

        pose_lookup = {}
        for idx, object_name in enumerate(names):
            pose_lookup[object_name] = {
                "object_rotation": self._quat_wxyz_to_rotmat(quats[idx]),
                "object_translation": positions[idx].astype(np.float32),
            }
        return pose_lookup

    def _load_scene_pose_ranges(self, scene_name: str) -> List[Dict[str, object]]:
        pose_json_path = osp.join(self.dataset_location, scene_name, "out_pose", "poses.json")
        if not osp.isfile(pose_json_path):
            return []

        with open(pose_json_path, "r", encoding="utf-8") as f:
            pose_json = json.load(f)

        pose_entries = pose_json.get("poses", [])
        pose_ranges = []
        for entry in pose_entries:
            object_name = str(entry["name"])
            pose_ranges.append(
                {
                    "name": object_name,
                    "frame_start": int(entry["frame_start"]),
                    "frame_end": int(entry["frame_end"]),
                    "pose": {
                        "object_rotation": self._quat_wxyz_to_rotmat(np.asarray(entry["rotation_quat_wxyz"], dtype=np.float32)),
                        "object_translation": np.asarray(entry["position_xyz"], dtype=np.float32),
                    },
                }
            )
        return pose_ranges

    @staticmethod
    def _frame_name_to_index(frame_name: str) -> int:
        match = re.match(r"^frame_(\d+)$", str(frame_name))
        if not match:
            raise ValueError(f"Invalid frame name '{frame_name}'. Expected e.g. frame_0000.")
        return int(match.group(1))

    def _filter_sequence_names(self, sequence_names: List[str]) -> List[str]:
        if self.quick:
            sequence_names = sequence_names[: min(len(sequence_names), 16)]
        if self.only_scene_names:
            normalized_scene_names = set()
            for sequence_name in self.only_scene_names:
                normalized_scene_names.add(sequence_name)
                sequence_idx = self._parse_scene_index(sequence_name)
                if sequence_idx is not None:
                    normalized_scene_names.add(f"scene_{sequence_idx:04d}")
            sequence_names = [sequence_name for sequence_name in sequence_names if sequence_name in normalized_scene_names]
        if self.only_scene_start_idx is not None or self.only_scene_end_idx is not None:
            filtered = []
            for sequence_name in sequence_names:
                sequence_idx = self._parse_scene_index(sequence_name)
                if self.only_scene_start_idx is not None and sequence_idx < self.only_scene_start_idx:
                    continue
                if self.only_scene_end_idx is not None and sequence_idx > self.only_scene_end_idx:
                    continue
                filtered.append(sequence_name)
            sequence_names = filtered
        return sequence_names

    def _list_scene_names(self) -> List[str]:
        if not osp.isdir(self.dataset_location):
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_location}")
        scene_names = sorted([d for d in os.listdir(self.dataset_location) if d.startswith("scene_")])
        return self._filter_sequence_names(scene_names)

    @staticmethod
    def _resolve_scene_frame_image_path(scene_name: str, frame_name: str) -> str:
        return osp.join(scene_name, "out_image", frame_name, "camera.jpg")

    @staticmethod
    def _resolve_scene_frame_depth_path(scene_name: str, frame_name: str) -> str:
        return osp.join(scene_name, "out_depth", frame_name, "camera_depth.png")

    @staticmethod
    def _resolve_scene_frame_camera_path(scene_name: str, frame_name: str) -> str:
        return osp.join(scene_name, "out_cam_param", frame_name, "camera_camera.npz")

    @staticmethod
    def _resolve_scene_frame_pose_path(scene_name: str, frame_name: str) -> str:
        return osp.join(scene_name, "out_pose", f"{frame_name}.npz")

    def _resolve_object_image_path(self, object_name: str, cam_idx: int) -> str:
        object_dir = self._resolve_object_dir_name(object_name)
        if object_dir is None:
            return osp.join(self.object_root, object_name, f"Main_Camera_({cam_idx})_rgb.png")
        return osp.join(self.object_root, object_dir, f"Main_Camera_({cam_idx})_rgb.png")

    def _verify_object_views(self, object_name: str) -> bool:
        return self._resolve_object_dir_name(object_name) is not None and all(
            osp.isfile(self._resolve_object_image_path(object_name, cam_idx)) for cam_idx in self.object_view_pool
        )

    def _list_scene_frame_names(self, scene_name: str) -> List[str]:
        image_root = osp.join(self.dataset_location, scene_name, "out_image")
        depth_root = osp.join(self.dataset_location, scene_name, "out_depth")
        camera_root = osp.join(self.dataset_location, scene_name, "out_cam_param")
        if not (osp.isdir(image_root) and osp.isdir(depth_root) and osp.isdir(camera_root)):
            return []

        frame_names = []
        for file_name in os.listdir(image_root):
            frame_dir = osp.join(image_root, file_name)
            if not file_name.startswith("frame_") or not osp.isdir(frame_dir):
                continue
            if self.only_frame_name and file_name != self.only_frame_name:
                continue
            frame_name = file_name
            image_path = osp.join(self.dataset_location, self._resolve_scene_frame_image_path(scene_name, frame_name))
            depth_path = osp.join(self.dataset_location, self._resolve_scene_frame_depth_path(scene_name, frame_name))
            camera_path = osp.join(self.dataset_location, self._resolve_scene_frame_camera_path(scene_name, frame_name))
            if not (osp.isfile(image_path) and osp.isfile(depth_path) and osp.isfile(camera_path)):
                continue
            frame_names.append(frame_name)
        return sorted(frame_names)

    def _build_records(self) -> List[Dict]:
        return self._build_scene_records()

    def _build_scene_records(self) -> List[Dict]:
        records = []
        for scene_name in self._list_scene_names():
            frame_names = self._list_scene_frame_names(scene_name)
            if not frame_names:
                continue

            scene_pose_ranges = self._load_scene_pose_ranges(scene_name)
            for frame_name in frame_names:
                pose_lookup = {}
                pose_path = osp.join(self.dataset_location, self._resolve_scene_frame_pose_path(scene_name, frame_name))
                if osp.isfile(pose_path):
                    pose_lookup = self._load_pose_lookup(pose_path)
                elif scene_pose_ranges:
                    frame_idx = self._frame_name_to_index(frame_name)
                    pose_lookup = {
                        entry["name"]: entry["pose"]
                        for entry in scene_pose_ranges
                        if int(entry["frame_start"]) <= frame_idx <= int(entry["frame_end"])
                    }

                if not pose_lookup:
                    continue

                for object_name, pose in pose_lookup.items():
                    object_dir_name = self._resolve_object_dir_name(object_name)
                    if object_dir_name is None:
                        continue
                    if self.verify_files and not self._verify_object_views(object_name):
                        continue
                    records.append(
                        {
                            "scene_name": scene_name,
                            "frame_name": frame_name,
                            "run_name": scene_name,
                            "object_name": object_name,
                            "object_dir_name": object_dir_name,
                            "available_scene_views": [0],
                            "object_cam_indices": list(self.object_view_pool),
                            "pose": pose,
                        }
                    )
        return records

    def _load_camera_npz(self, camera_path: str):
        data = np.load(camera_path)
        intrinsics = data["intrinsics.K_flat9"].astype(np.float32).reshape(3, 3)
        extrinsics_key = (
            "extrinsics.opencv.cameraToWorld16" if self.use_opencv_camera else "extrinsics.unity.cameraToWorld16"
        )
        camera_pose = data[extrinsics_key].astype(np.float32).reshape(4, 4)
        return intrinsics, camera_pose

    def _load_scene_frame_view(self, scene_name: str, frame_name: str, resolution, rng) -> Dict:
        image_path = osp.join(self.dataset_location, self._resolve_scene_frame_image_path(scene_name, frame_name))
        depth_path = osp.join(self.dataset_location, self._resolve_scene_frame_depth_path(scene_name, frame_name))
        camera_path = osp.join(self.dataset_location, self._resolve_scene_frame_camera_path(scene_name, frame_name))

        image = Image.open(image_path).convert("RGB")
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise FileNotFoundError(f"Failed to read depth image: {depth_path}")
        if depth_raw.dtype != np.uint16:
            raise ValueError(f"Expected uint16 R16 depth image, got {depth_raw.dtype} for {depth_path}")

        depthmap = depth_raw.view(np.float16).astype(np.float32)
        depthmap[~np.isfinite(depthmap)] = 0.0
        depthmap[depthmap < 0] = 0.0

        intrinsics, camera_pose = self._load_camera_npz(camera_path)
        depthmap = threshold_depth_map(depthmap, max_percentile=99, min_percentile=-1)
        image, depthmap, intrinsics = self._crop_resize_if_necessary(image, depthmap, intrinsics, resolution, rng, info=image_path)

        return {
            "img": image,
            "depthmap": depthmap,
            "camera_pose": camera_pose,
            "camera_intrinsics": intrinsics,
            "dataset": self.dataset_label,
            "label": scene_name,
            "instance": f"{frame_name}/{osp.basename(image_path)}",
        }

    @staticmethod
    def _resize_object_image(image: Image.Image, resolution) -> Image.Image:
        width, height = resolution
        resampling = getattr(Image, "Resampling", Image)
        return image.resize((width, height), resampling.LANCZOS)

    def _load_object_images(self, object_name: str, resolution) -> Dict[str, object]:
        object_tensors = []
        object_sizes = []
        for cam_idx in self.object_view_pool:
            image_path = self._resolve_object_image_path(object_name, cam_idx)
            image = Image.open(image_path).convert("RGB")
            object_sizes.append(np.array(image.size[::-1], dtype=np.int32))
            image = self._resize_object_image(image, resolution)
            object_tensors.append(self.transform(image))
        return {
            "object_images": torch.stack(object_tensors),
            "object_true_shape": np.stack(object_sizes),
            "object_cam_indices": np.array(self.object_view_pool, dtype=np.int64),
        }

    @staticmethod
    def _make_pose_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = rotation.astype(np.float32)
        transform[:3, 3] = translation.astype(np.float32)
        return transform

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx, ar_idx, *num_args = idx
            num_views = num_args[0] if num_args else self.scene_num_views
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0
            num_views = self.scene_num_views

        if self.seed:
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, "_rng"):
            self._rng = np.random.default_rng(seed=torch.initial_seed())

        resolution = self._resolutions[ar_idx]
        record = self.records[idx]
        scene_view_indices = [0]
        views = [
            self._load_scene_frame_view(record["scene_name"], record["frame_name"], resolution, self._rng)
        ]
        assert len(views) == num_views

        object_pose_world = self._make_pose_matrix(
            record["pose"]["object_rotation"],
            record["pose"]["object_translation"],
        )
        reference_camera_pose_c2w = views[0]["camera_pose"].copy()
        reference_camera_pose_w2c = closed_form_inverse_se3(reference_camera_pose_c2w[None])[0]
        object_pose_camera = reference_camera_pose_w2c @ object_pose_world
        object_rotation_camera = object_pose_camera[:3, :3].astype(np.float32)
        object_translation_camera = object_pose_camera[:3, 3].astype(np.float32)

        for view_idx, view in enumerate(views):
            assert "pts3d" not in view and "valid_mask" not in view, (
                f"pts3d/valid_mask should not be present in view {view_name(view)}"
            )
            assert "camera_intrinsics" in view
            assert np.isfinite(view["depthmap"]).all(), f"NaN in depthmap for view {view_name(view)}"

            view["idx"] = (idx, ar_idx, view_idx)
            view["z_far"] = self.z_far
            view["true_shape"] = np.int32(view["img"].size[::-1])
            view["img"] = self.transform(view["img"])

            if "camera_pose" not in view:
                view["camera_pose"] = np.full((4, 4), np.nan, dtype=np.float32)
            else:
                assert np.isfinite(view["camera_pose"]).all(), f"NaN in camera pose for view {view_name(view)}"

            for key, value in view.items():
                res, err_msg = is_good_type(key, value)
                assert res, f"{err_msg} with {key}={value} for view {view_name(view)}"

            view["camera_pose"] = closed_form_inverse_se3(view["camera_pose"][None])[0]
            point_mask = view["depthmap"] > 0.0
            if self.z_far > 0:
                point_mask = point_mask & (view["depthmap"] < self.z_far)
            view["point_mask"] = point_mask.astype(np.bool_)

        for view in views:
            transpose_to_landscape(view)
            view["rng"] = int.from_bytes(self._rng.bytes(4), "big")

        field_config = {
            "img": ("images", torch.stack),
            "depthmap": ("depth", lambda x: np.stack([d[:, :, np.newaxis] for d in x])),
            "camera_pose": ("extrinsic", lambda x: np.stack([p[:3] for p in x])),
            "camera_intrinsics": ("intrinsic", np.stack),
            "true_shape": ("true_shape", np.array),
            "point_mask": ("valid_mask", np.stack),
            "label": ("label", lambda x: x),
            "instance": ("instance", lambda x: x),
        }

        result = {}
        for field_key, (output_key, stack_func) in field_config.items():
            result[output_key] = stack_func([view[field_key] for view in views])

        result.update(self._load_object_images(record["object_name"], resolution))
        result["dataset"] = self.dataset_label
        result["camera_indices"] = np.array(scene_view_indices, dtype=np.int64)
        result["object_name"] = record["object_name"]
        result["run_name"] = record["run_name"]
        result["scene_name"] = record.get("scene_name", record["run_name"])
        result["frame_name"] = record.get("frame_name", "")
        result["has_object"] = np.array(True, dtype=np.bool_)
        result["object_rotation"] = object_rotation_camera
        result["object_translation"] = object_translation_camera
        result["object_rotation_world"] = record["pose"]["object_rotation"].astype(np.float32)
        result["object_translation_world"] = record["pose"]["object_translation"].astype(np.float32)
        result["object_pose_reference_camera_index"] = np.int64(scene_view_indices[0])
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
