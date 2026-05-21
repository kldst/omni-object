import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import torch
from PIL import Image

from .base_oo9d_camera_pose import OO9DCameraPoseBase

logger = logging.getLogger(__name__)


class OO9DGeneratedMultiCameraPose(OO9DCameraPoseBase):
    """OO9D camera-pose dataloader for generated multi-object scenes.

    Scene targets are read from a generated ``scene_XXXX`` root. Object
    reference images are read directly from ``ov9d_around_image/obj_XXXXXX``.
    The single train split is used only as an optional object-id allowlist, so
    generated scenes do not leak objects outside the requested training split.
    """

    DEFAULT_DATA_ROOT = "/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d"
    DEFAULT_GENERATED_MULTI_ROOT = (
        "/mnt/train-data-4-hdd/yian/freepose/ov9d/render_script/ov9d_2000_scenes_3modes_4views_v2"
    )
    DEFAULT_SINGLE_SPLIT_JSON = (
        "/mnt/train-data-4-hdd/yian/freepose/omni-object_clone/"
        "splits_ov9d_unseen_category_generalization/single/train.json"
    )
    DEFAULT_SINGLE_ROOT = "/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d/oo3d9dsingle"
    DEFAULT_OBJECT_IMAGE_ROOT = "/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d_around_image"
    DEFAULT_OBJECT_VIEW_IDS = (1, 5, 10, 15)

    def __init__(
        self,
        dataset_location: str = DEFAULT_DATA_ROOT,
        dset: str = "train",
        generated_multi_root: Optional[str] = None,
        multi_root: Optional[str] = None,
        single_root: Optional[str] = None,
        single_split_json: Optional[str] = None,
        object_image_root: Optional[str] = None,
        fixed_object_view_ids: Optional[Sequence[int]] = DEFAULT_OBJECT_VIEW_IDS,
        num_object_views: int = 4,
        strict_fixed_object_view_ids: bool = True,
        filter_single_train_objects: bool = True,
        include_single_targets: bool = True,
        expand_records_by_view: bool = True,
        normalize_object_translation_by_depth_mean: bool = True,
        depth_mean_eps: float = 1e-6,
        *args,
        **kwargs,
    ):
        self.generated_multi_root = Path(generated_multi_root or multi_root or self.DEFAULT_GENERATED_MULTI_ROOT)
        self.single_root = Path(single_root or self.DEFAULT_SINGLE_ROOT)
        self.single_split_json = Path(single_split_json or self.DEFAULT_SINGLE_SPLIT_JSON)
        self.object_image_root = Path(object_image_root or self.DEFAULT_OBJECT_IMAGE_ROOT)
        self.filter_single_train_objects = bool(filter_single_train_objects)
        self.include_single_targets = bool(include_single_targets)
        self.expand_records_by_view = bool(expand_records_by_view)
        self.normalize_object_translation_by_depth_mean = bool(normalize_object_translation_by_depth_mean)
        self.depth_mean_eps = float(depth_mean_eps)

        super().__init__(
            dataset_location=dataset_location,
            dset=dset,
            split_json=str(self.generated_multi_root),
            multi_root=str(self.generated_multi_root),
            single_root=str(self.single_root),
            fixed_object_view_ids=fixed_object_view_ids,
            num_object_views=num_object_views,
            strict_fixed_object_view_ids=strict_fixed_object_view_ids,
            *args,
            **kwargs,
        )
        self.dataset_label = "OO9DGeneratedMultiCameraPose"

    def _default_split_json(self, dset: str) -> Path:
        return self.generated_multi_root

    def _allowed_object_ids_from_single_split(self) -> Optional[Set[int]]:
        if not self.filter_single_train_objects:
            return None
        if not self.single_split_json.is_file():
            raise FileNotFoundError(f"Single train split JSON not found: {self.single_split_json}")
        payload = self._load_json(self.single_split_json)
        return {int(item["object_id"]) for item in payload.get("scenes", [])}

    def _build_single_records_by_object_id(self) -> Dict[int, List[Dict[str, Any]]]:
        allowed_object_ids = self._allowed_object_ids_from_single_split()
        object_ids = sorted(allowed_object_ids) if allowed_object_ids is not None else sorted(
            int(path.name.removeprefix("obj_"))
            for path in self.object_image_root.glob("obj_*")
            if path.is_dir() and path.name.removeprefix("obj_").isdigit()
        )

        records: Dict[int, List[Dict[str, Any]]] = {}
        skipped_missing_refs = []
        skipped_bad_gt = []
        image_ids = [int(x) for x in (self.fixed_object_view_ids or self.DEFAULT_OBJECT_VIEW_IDS)]

        for object_id in object_ids:
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
                    "object_instance": f"obj_{object_id:06d}",
                    "category": "",
                }
            )

        if skipped_missing_refs:
            logger.warning(
                "Skipped %d generated-multi objects with missing around reference images. First examples: %s",
                len(skipped_missing_refs),
                skipped_missing_refs[:5],
            )
        if skipped_bad_gt:
            logger.warning(
                "Skipped %d generated-multi objects with mismatched around scene_gt obj_id. First examples: %s",
                len(skipped_bad_gt),
                skipped_bad_gt[:5],
            )
        return records

    def _build_records(self) -> List[Dict[str, Any]]:
        if not self.generated_multi_root.is_dir():
            raise FileNotFoundError(f"Generated multi root not found: {self.generated_multi_root}")

        records: List[Dict[str, Any]] = []
        source_counts = {"generated_multi": 0, "single": 0}
        skipped_missing_refs = 0

        def add_scene_object_record(scene_name: str, scene_dir: Path, object_id: int, scene_source: str) -> bool:
            nonlocal skipped_missing_refs
            if object_id not in self.single_records_by_object_id:
                skipped_missing_refs += 1
                return False
            if self.verify_files and not self._verify_scene_files(scene_dir):
                return False
            scene_gt = self._load_json(scene_dir / "scene_gt.json")
            image_ids = self._image_ids_with_object(scene_gt, object_id)
            if not image_ids:
                return False
            record_image_id_groups = [[image_id] for image_id in image_ids] if self.expand_records_by_view else [image_ids]
            for record_image_ids in record_image_id_groups:
                records.append(
                    {
                        "scene_name": scene_name,
                        "run_name": scene_name,
                        "scene_dir": scene_dir,
                        "object_id": object_id,
                        "image_ids": record_image_ids,
                        "scene_source": scene_source,
                    }
                )
                source_counts[scene_source] += 1
                if self.max_records is not None and len(records) >= self.max_records:
                    return True
            return False

        for scene_dir in sorted(self.generated_multi_root.glob("scene_*")):
            if not scene_dir.is_dir():
                continue
            scene_name = scene_dir.name
            if self.only_scene_name and scene_name != self.only_scene_name:
                continue
            if self.verify_files and not self._verify_scene_files(scene_dir):
                continue

            scene_gt = self._load_json(scene_dir / "scene_gt.json")
            object_ids = sorted(
                {
                    int(gt.get("obj_id", -1))
                    for gts in scene_gt.values()
                    for gt in gts
                    if int(gt.get("obj_id", -1)) >= 0
                }
            )
            if self.only_object_id is not None:
                object_ids = [object_id for object_id in object_ids if object_id == self.only_object_id]

            for object_id in object_ids:
                if add_scene_object_record(scene_name, scene_dir, object_id, "generated_multi"):
                    return records

        if self.include_single_targets:
            single_payload = self._load_json(self.single_split_json)
            for item in single_payload.get("scenes", []):
                scene_name = str(item["scene_name"])
                if self.only_scene_name and scene_name != self.only_scene_name:
                    continue
                object_id = int(item["object_id"])
                if self.only_object_id is not None and object_id != self.only_object_id:
                    continue
                scene_dir = self.single_root / scene_name
                if add_scene_object_record(scene_name, scene_dir, object_id, "single"):
                    return records

        logger.info(
            "OO9D generated multi records built: generated_multi=%d single=%d total=%d "
            "skipped_missing_or_filtered_refs=%d",
            source_counts["generated_multi"],
            source_counts["single"],
            len(records),
            skipped_missing_refs,
        )
        return records

    def _load_object_images(self, object_id: int, resolution, rng) -> Dict[str, Any]:
        object_id = int(object_id)
        object_rec = self.single_records_by_object_id[object_id][0]
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
