import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from .base_oo9d_camera_pose import OO9DCameraPoseBase

logger = logging.getLogger(__name__)


class OO9DSingleCameraPose(OO9DCameraPoseBase):
    """OO9D camera-pose dataset that reads only oo3d9dsingle scenes.

    Scene images and GT poses come from ``ov9d/oo3d9dsingle``. Object reference
    images come from ``ov9d_around_image/obj_XXXXXX/rgb`` using fixed view ids.
    No generated multi-object scenes are scanned or sampled.
    """

    DEFAULT_DATA_ROOT = "/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d"
    DEFAULT_SPLIT_ROOT = (
        "/mnt/train-data-4-hdd/yian/freepose/omni-object_clone/"
        "splits_ov9d_unseen_category_generalization/single"
    )
    DEFAULT_OBJECT_IMAGE_ROOT = "/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d_around_image"
    DEFAULT_OBJECT_VIEW_IDS = (1, 5, 10, 15)

    def __init__(
        self,
        dataset_location: str = DEFAULT_DATA_ROOT,
        dset: str = "train",
        split_root: Optional[str] = None,
        single_split_json: Optional[str] = None,
        split_json: Optional[str] = None,
        single_root: Optional[str] = None,
        object_image_root: Optional[str] = None,
        fixed_object_view_ids: Optional[Sequence[int]] = DEFAULT_OBJECT_VIEW_IDS,
        num_object_views: int = 4,
        strict_fixed_object_view_ids: bool = True,
        expand_records_by_view: bool = True,
        normalize_object_translation_by_depth_mean: bool = True,
        depth_mean_eps: float = 1e-6,
        *args,
        **kwargs,
    ):
        self.split_root = Path(split_root) if split_root else Path(self.DEFAULT_SPLIT_ROOT)
        self.single_split_json = Path(single_split_json or split_json) if (single_split_json or split_json) else self._default_split_json(dset)
        self.object_image_root = Path(object_image_root or self.DEFAULT_OBJECT_IMAGE_ROOT)
        self.expand_records_by_view = bool(expand_records_by_view)
        self.normalize_object_translation_by_depth_mean = bool(normalize_object_translation_by_depth_mean)
        self.depth_mean_eps = float(depth_mean_eps)

        super().__init__(
            dataset_location=dataset_location,
            dset=dset,
            split_json=str(self.single_split_json),
            single_root=single_root,
            fixed_object_view_ids=fixed_object_view_ids,
            num_object_views=num_object_views,
            strict_fixed_object_view_ids=strict_fixed_object_view_ids,
            *args,
            **kwargs,
        )
        self.dataset_label = "OO9DSingleCameraPose"

    def _default_split_json(self, dset: str) -> Path:
        split = str(dset).lower()
        if split == "train":
            return self.split_root / "train.json"
        if split in {"val", "validation", "test", "test1", "unseen"}:
            return self.split_root / "test_unseen_category_unseen_object.json"
        if split in {"test_same_category", "same_category"}:
            return self.split_root / "test_same_category_unseen_object.json"
        return self.split_root / f"{dset}.json"

    def _build_single_records_by_object_id(self) -> Dict[int, List[Dict[str, Any]]]:
        if not self.single_split_json.is_file():
            raise FileNotFoundError(f"OO9D single split JSON not found: {self.single_split_json}")

        payload = self._load_json(self.single_split_json)
        records: Dict[int, List[Dict[str, Any]]] = {}
        skipped_missing_refs = []
        skipped_bad_gt = []
        image_ids = [int(x) for x in (self.fixed_object_view_ids or self.DEFAULT_OBJECT_VIEW_IDS)]

        for item in payload.get("scenes", []):
            object_id = int(item["object_id"])
            object_dir = self.object_image_root / f"obj_{object_id:06d}"
            rgb_dir = object_dir / "rgb"
            image_paths = [rgb_dir / f"{image_id:06d}.png" for image_id in image_ids]

            if self.verify_files:
                missing = [image_id for image_id, path in zip(image_ids, image_paths) if not path.is_file()]
                if missing:
                    skipped_missing_refs.append((object_id, missing))
                    continue
                scene_gt_path = object_dir / "scene_gt.json"
                if scene_gt_path.is_file():
                    scene_gt = self._load_json(scene_gt_path)
                    mismatched = [
                        image_id
                        for image_id in image_ids
                        if int(scene_gt.get(str(image_id), [{}])[0].get("obj_id", -1)) != object_id
                    ]
                    if mismatched:
                        skipped_bad_gt.append((object_id, mismatched))
                        continue

            records.setdefault(object_id, []).append(
                {
                    "scene_dir": object_dir,
                    "scene_name": object_dir.name,
                    "image_ids": image_ids,
                    "object_instance": item.get("object_instance", f"obj_{object_id:06d}"),
                    "category": item.get("category", ""),
                }
            )

        if skipped_missing_refs:
            logger.warning(
                "Skipped %d OO9D single object refs with missing fixed views. First examples: %s",
                len(skipped_missing_refs),
                skipped_missing_refs[:5],
            )
        if skipped_bad_gt:
            logger.warning(
                "Skipped %d OO9D single object refs with mismatched around-image scene_gt obj_id. First examples: %s",
                len(skipped_bad_gt),
                skipped_bad_gt[:5],
            )
        return records

    def _build_records(self) -> List[Dict[str, Any]]:
        if not self.single_split_json.is_file():
            raise FileNotFoundError(f"OO9D single split JSON not found: {self.single_split_json}")

        payload = self._load_json(self.single_split_json)
        records: List[Dict[str, Any]] = []
        skipped_missing_ref = 0
        skipped_missing_scene = 0
        skipped_no_visible_frame = 0

        for item in payload.get("scenes", []):
            scene_name = str(item["scene_name"])
            if self.only_scene_name and scene_name != self.only_scene_name:
                continue
            object_id = int(item["object_id"])
            if self.only_object_id is not None and object_id != self.only_object_id:
                continue
            if object_id not in self.single_records_by_object_id:
                skipped_missing_ref += 1
                continue

            scene_dir = self.single_root / scene_name
            if self.verify_files and not self._verify_scene_files(scene_dir):
                skipped_missing_scene += 1
                continue
            scene_gt = self._load_json(scene_dir / "scene_gt.json")
            image_ids = self._image_ids_with_object(scene_gt, object_id)
            if not image_ids:
                skipped_no_visible_frame += 1
                continue

            record_image_id_groups = [[image_id] for image_id in image_ids] if self.expand_records_by_view else [image_ids]
            for record_image_ids in record_image_id_groups:
                records.append(
                    {
                        "scene_name": scene_name,
                        "run_name": scene_name,
                        "scene_dir": scene_dir,
                        "object_id": object_id,
                        "image_ids": record_image_ids,
                        "scene_source": "single",
                    }
                )
                if self.max_records is not None and len(records) >= self.max_records:
                    return records

        logger.info(
            "OO9D single records built: total=%d skipped_missing_ref=%d "
            "skipped_missing_scene=%d skipped_no_visible_frame=%d",
            len(records),
            skipped_missing_ref,
            skipped_missing_scene,
            skipped_no_visible_frame,
        )
        return records

    def _load_object_images(self, object_id: int, resolution, rng) -> Dict[str, Any]:
        object_id = int(object_id)
        candidates = self.single_records_by_object_id[object_id]
        object_rec = candidates[int(rng.integers(len(candidates)))] if self.training else candidates[0]
        image_ids = self._sample_object_ids(object_rec["image_ids"], rng, object_name=object_rec["scene_name"])

        tensors = []
        true_shapes = []
        image_paths = []
        for image_id in image_ids:
            image_path = object_rec["scene_dir"] / "rgb" / f"{image_id:06d}.png"
            image = Image.open(image_path).convert("RGB")
            true_shapes.append(np.array(image.size[::-1], dtype=np.int32))
            tensors.append(self.transform(self._resize_image(image, resolution)))
            image_paths.append(str(image_path))

        return {
            "object_images": torch.stack(tensors),
            "object_true_shape": np.stack(true_shapes),
            "object_cam_indices": np.asarray(image_ids, dtype=np.int64),
            "object_reference_scene_name": object_rec["scene_name"],
            "object_rgb_paths": image_paths,
            "object_mask_paths": [""] * len(image_paths),
        }

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        depth = np.asarray(sample["depth"], dtype=np.float32)
        valid_mask = np.asarray(sample["valid_mask"], dtype=np.bool_)
        valid_depth = depth[..., 0][valid_mask]
        if valid_depth.size == 0:
            depth_mean_scale = np.float32(1.0)
        else:
            depth_mean_scale = np.float32(max(float(valid_depth.mean()), self.depth_mean_eps))

        object_translation_metric = np.asarray(sample["object_translation"], dtype=np.float32).reshape(3)
        object_translation_normalized = (object_translation_metric / depth_mean_scale).astype(np.float32)

        sample["depth_mean_scale"] = np.asarray(depth_mean_scale, dtype=np.float32)
        sample["normalization_scale"] = np.asarray(depth_mean_scale, dtype=np.float32)
        sample["object_translation_scale"] = np.asarray(depth_mean_scale, dtype=np.float32)
        sample["object_translation_metric"] = object_translation_metric.astype(np.float32)
        sample["object_translation_normalized"] = object_translation_normalized

        if self.normalize_object_translation_by_depth_mean:
            sample["object_translation"] = object_translation_normalized
            sample["object_srt"] = np.concatenate(
                [
                    np.asarray(sample["object_rotation"], dtype=np.float32).reshape(-1),
                    object_translation_normalized,
                    np.asarray(sample["object_size_log"], dtype=np.float32).reshape(-1),
                ]
            ).astype(np.float32)

        return sample
