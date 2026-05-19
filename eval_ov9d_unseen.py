"""Inference-only eval for OV9D test splits (single + multi).

Reports rotation error (with object symmetry), translation error (cm), and the
training-style pose/translation L1 loss, bucketed into near/mid/far based on
the GT camera-frame distance.

    python eval_ov9d_unseen.py \\
        --checkpoint /path/to/model.safetensors \\
        --output-dir outputs/eval_unseen
"""

import argparse
import json
import math
import os
import re
import runpy
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file as load_safetensors_file
from tqdm import tqdm

from omnivggt.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from omnivggt.loss import (
    _load_symmetry_info,
    _rotation_matrix_to_rot6d,
    _symmetric_rot6d_loss,
    _vector_loss,
)
from omnivggt.models.omnivggt import OmniVGGT
from omnivggt.utils.image import ImgNorm

# ============================================================
# Defaults (mirror demo_gradio_6dpose_0519_validation.py paths)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train_oo9d.py"
DEFAULT_DATASET_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d")
DEFAULT_OBJECT_IMAGE_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d_around_image")
DEFAULT_CHECKPOINT = Path(
    "/mnt/train-data-4-hdd/yian/freepose/omni-object_clone/outputs/0515_LARGE/model.safetensors"
)
DEFAULT_SINGLE_TEST_JSON = (
    PROJECT_ROOT / "splits_ov9d_seen_unseen_scene" / "single" / "test_same_category_unseen_object.json"
)
DEFAULT_MULTI_TEST_JSON = (
    PROJECT_ROOT / "splits_ov9d_seen_unseen_scene" / "multi" / "test_same_category_unseen_object_unseen_scene.json"
)
DEFAULT_SINGLE_TRAIN_JSON = PROJECT_ROOT / "splits_ov9d_seen_unseen_scene" / "single" / "train.json"
DEFAULT_OBJECT_VIEWS = (1, 5, 10, 15)
DEFAULT_RESOLUTION = (518, 476)


# ============================================================
# JSON helpers + path remapping (config writes /dataset/* paths)
# ============================================================
def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_local_path(value, *, default=None) -> Path | None:
    if value in (None, ""):
        return Path(default) if default is not None else None
    path = Path(str(value)).expanduser()
    if path.exists():
        return path
    text = str(path)
    replacements = {
        "/dataset/ov9d_around_image": str(DEFAULT_OBJECT_IMAGE_ROOT),
        "/dataset/ov9d": str(DEFAULT_DATASET_ROOT),
        "/dataset": str(DEFAULT_DATASET_ROOT),
        "/omni-object_clone": str(PROJECT_ROOT),
    }
    for prefix, replacement in replacements.items():
        if text == prefix or text.startswith(prefix + "/"):
            candidate = Path(replacement + text[len(prefix):])
            if candidate.exists():
                return candidate
    return path


def load_config(config_path: Path) -> Dict:
    cfg = runpy.run_path(str(config_path))
    return {key: value for key, value in cfg.items() if not key.startswith("__")}


# ============================================================
# Pose math
# ============================================================
def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float64).reshape(3, 2)
    x_raw, y_raw = rot6d[:, 0], rot6d[:, 1]
    x = x_raw / max(np.linalg.norm(x_raw), 1e-12)
    y = y_raw - np.dot(x, y_raw) * x
    if np.linalg.norm(y) < 1e-12:
        fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        y = fallback - np.dot(x, fallback) * x
    y = y / max(np.linalg.norm(y), 1e-12)
    z = np.cross(x, y)
    z = z / max(np.linalg.norm(z), 1e-12)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def rotation_error_degrees(R1: np.ndarray, R2: np.ndarray) -> float:
    rel = np.asarray(R1, dtype=np.float64) @ np.asarray(R2, dtype=np.float64).T
    cos_theta = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def symmetric_rotation_error_degrees(
    pred_rot: np.ndarray,
    gt_rot: np.ndarray,
    object_id: int,
    symmetry_info_path: str,
    continuous_steps: int,
) -> float:
    if not symmetry_info_path:
        return rotation_error_degrees(pred_rot, gt_rot)
    symmetry_info = _load_symmetry_info(str(symmetry_info_path), int(continuous_steps))
    sym_rots = symmetry_info.get(int(object_id))
    if sym_rots is None:
        return rotation_error_degrees(pred_rot, gt_rot)
    candidates = sym_rots.detach().cpu().numpy().astype(np.float64)
    gt = np.asarray(gt_rot, dtype=np.float64)
    return float(min(rotation_error_degrees(pred_rot, gt @ sym) for sym in candidates))


def symmetric_rot6d_l1(
    pred_rot6d: np.ndarray,
    gt_rot: np.ndarray,
    object_id: int,
    symmetry_info_path: str,
    continuous_steps: int,
) -> float:
    """L1 distance between predicted rot6d and the nearest symmetric GT rot6d."""
    pred = np.asarray(pred_rot6d, dtype=np.float64).reshape(-1)
    gt = np.asarray(gt_rot, dtype=np.float64).reshape(3, 3)
    gt_rot6d = gt[:, :2].reshape(-1)

    sym_rots = None
    if symmetry_info_path:
        symmetry_info = _load_symmetry_info(str(symmetry_info_path), int(continuous_steps))
        sym_rots = symmetry_info.get(int(object_id))
    if sym_rots is None:
        return float(np.abs(pred - gt_rot6d).mean())

    candidates = sym_rots.detach().cpu().numpy().astype(np.float64)  # (S, 3, 3)
    best = math.inf
    for sym in candidates:
        cand = (gt @ sym)[:, :2].reshape(-1)
        loss = float(np.abs(pred - cand).mean())
        if loss < best:
            best = loss
    return best


# ============================================================
# Self-test: verify metric math against omnivggt.loss reference impl
# ============================================================
def _self_test_metrics():
    rng = np.random.default_rng(0)

    # 1) rotation_error_degrees: identity -> 0
    R = np.eye(3)
    assert abs(rotation_error_degrees(R, R)) < 1e-6, "identity rot error should be 0"

    # 2) rotation_error_degrees: 90 deg about z
    Rz90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    err = rotation_error_degrees(R, Rz90)
    assert abs(err - 90.0) < 1e-4, f"expected 90 deg, got {err}"

    # 3) rotation_error_degrees: 30 deg about y
    theta = math.radians(30)
    Ry30 = np.array([
        [math.cos(theta), 0.0, math.sin(theta)],
        [0.0, 1.0, 0.0],
        [-math.sin(theta), 0.0, math.cos(theta)],
    ])
    err = rotation_error_degrees(R, Ry30)
    assert abs(err - 30.0) < 1e-4, f"expected 30 deg, got {err}"

    # 4) rot6d round-trip: matrix -> rot6d -> matrix
    a = rng.normal(size=3)
    axis = a / np.linalg.norm(a)
    angle = float(rng.uniform(0.1, 2.0))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R_rand = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
    rot6d = R_rand[:, :2].reshape(-1)
    R_back = rot6d_to_matrix(rot6d)
    assert np.allclose(R_rand, R_back, atol=1e-6), "rot6d round-trip mismatch"

    # 5) symmetric_rot6d_l1 (no symmetry) == direct rot6d L1
    pred_rot6d = rng.normal(size=6)
    gt_R = R_rand
    gt_rot6d = gt_R[:, :2].reshape(-1)
    direct = float(np.abs(pred_rot6d - gt_rot6d).mean())
    ours = symmetric_rot6d_l1(pred_rot6d, gt_R, object_id=999999, symmetry_info_path="", continuous_steps=72)
    assert abs(direct - ours) < 1e-9, f"sym L1 no-sym mismatch: direct={direct}, ours={ours}"

    # 6) Cross-check our symmetric_rot6d_l1 vs omnivggt.loss._symmetric_rot6d_loss on a real OV9D model
    sym_path = resolve_local_path("/dataset/ov9d/models_info.json")
    if sym_path is not None and Path(sym_path).is_file():
        # pick a couple of object ids (some are symmetric — e.g. bottles/cans tend to be)
        sym_info = _load_symmetry_info(str(sym_path), 72)
        sample_ids = list(sym_info.keys())[:3]
        if sample_ids:
            for oid in sample_ids:
                gt_t = torch.tensor(R_rand, dtype=torch.float32).unsqueeze(0)
                pred_t = torch.tensor(pred_rot6d, dtype=torch.float32).unsqueeze(0)
                obj_t = torch.tensor([int(oid)], dtype=torch.long)
                ref_loss = float(_symmetric_rot6d_loss(
                    pred_t, gt_t, obj_t, "l1", str(sym_path), 72,
                ).item())
                ours = symmetric_rot6d_l1(pred_rot6d, R_rand, int(oid), str(sym_path), 72)
                assert abs(ref_loss - ours) < 1e-5, (
                    f"symmetric rot6d L1 mismatch (obj_id={oid}): ref={ref_loss} ours={ours}"
                )
        print(f"[self_test] symmetric_rot6d_l1 matches reference on obj_ids={sample_ids[:3]}")

    # 7) translation L1: matches _vector_loss
    pred_t_np = rng.normal(size=3).astype(np.float32)
    gt_t_np = rng.normal(size=3).astype(np.float32)
    ours_tl = float(np.abs(pred_t_np.astype(np.float64) - gt_t_np.astype(np.float64)).mean())
    ref_tl = float(_vector_loss(torch.tensor(pred_t_np), torch.tensor(gt_t_np), "l1").item())
    assert abs(ours_tl - ref_tl) < 1e-6, f"translation L1 mismatch: ours={ours_tl} ref={ref_tl}"

    # 8) translation_err_cm: norm in cm
    diff_m = pred_t_np.astype(np.float64) - gt_t_np.astype(np.float64)
    expected_cm = float(np.linalg.norm(diff_m) * 100.0)
    assert expected_cm > 0
    print("[self_test] all metric checks passed")


# ============================================================
# OV9D readers / preprocessing
# ============================================================
def ov9d_object_key(object_id: int) -> str:
    return f"obj_{int(object_id):06d}"


def ov9d_object_id_from_key(value) -> int:
    match = re.search(r"(\d+)$", str(value))
    if not match:
        raise ValueError(f"Could not parse OV9D object id from: {value}")
    return int(match.group(1))


def ov9d_read_depth_m(depth_path: Path, camera_entry: Dict) -> np.ndarray:
    depth_raw = np.asarray(Image.open(depth_path), dtype=np.float32)
    depth_m = depth_raw * float(camera_entry.get("depth_scale", 1.0)) / 1000.0
    depth_m[~np.isfinite(depth_m)] = 0.0
    depth_m[depth_m < 0.0] = 0.0
    return depth_m.astype(np.float32)


def ov9d_read_binary_mask(mask_path: Path) -> np.ndarray:
    return (np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0).astype(np.float32)


class EvalScenePreprocessor(BaseStereoViewDataset):
    def __init__(self, resolution):
        super().__init__(dset="eval", resolution=resolution, transform=ImgNorm, seed=0)


def crop_resize_image_depth_mask(
    processor: EvalScenePreprocessor,
    image: Image.Image,
    depthmap: np.ndarray,
    object_mask: np.ndarray | None,
    intrinsics: np.ndarray,
    resolution,
    info: str,
):
    if object_mask is None:
        image, depthmap, intrinsics = processor._crop_resize_if_necessary(
            image=image,
            depthmap=depthmap,
            intrinsics=intrinsics.copy(),
            resolution=resolution,
            rng=np.random.default_rng(seed=0),
            info=info,
        )
        return image, depthmap, None, intrinsics

    from importlib import import_module
    ov9d_cls = import_module("omnivggt.datasets.6Dpose.ov9d_camera_pose").OV9DCameraPose
    return ov9d_cls._crop_resize_if_necessary_with_mask(
        processor,
        image=image,
        depthmap=depthmap,
        object_mask=object_mask,
        intrinsics=intrinsics.copy(),
        resolution=resolution,
        rng=np.random.default_rng(seed=0),
        info=info,
    )


def load_scene_frame_inputs_cpu(
    scene_dir: Path,
    image_id: int,
    object_index: int,
    resolution,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (image_tensor, depth_tensor, mask_tensor) as CPU tensors of shape:
        image: (1, 3, H, W)
        depth: (1, H, W, 1)
        mask:  (1, H, W)
    """
    camera_entry = read_json(scene_dir / "scene_camera.json")[str(image_id)]
    image = Image.open(scene_dir / "rgb" / f"{image_id:06d}.png").convert("RGB")
    depthmap = ov9d_read_depth_m(scene_dir / "depth" / f"{image_id:06d}.png", camera_entry)
    intrinsics = np.asarray(camera_entry["cam_K"], dtype=np.float32).reshape(3, 3)
    mask_path = scene_dir / "mask_visib" / f"{image_id:06d}_{object_index:06d}.png"
    object_mask = ov9d_read_binary_mask(mask_path)

    processor = EvalScenePreprocessor(resolution=resolution)
    image, depthmap, _gt_mask, _intrinsics = crop_resize_image_depth_mask(
        processor, image, depthmap, object_mask, intrinsics, resolution,
        info=str(scene_dir / "rgb" / f"{image_id:06d}.png"),
    )
    image_tensor = processor.transform(image).unsqueeze(0)  # (1, 3, H, W)
    depth_tensor = torch.from_numpy(np.ascontiguousarray(depthmap.astype(np.float32)))[None, :, :, None]  # (1, H, W, 1)
    mask_tensor = torch.from_numpy((depthmap > 0).astype(np.float32))[None, :, :]  # (1, H, W)
    return image_tensor, depth_tensor, mask_tensor


def load_object_tensor_cpu(
    object_ref_dir: Path,
    object_views: Sequence[int],
    resolution,
) -> torch.Tensor:
    """Return object reference tensor of shape (K, 3, H, W) on CPU."""
    processor = EvalScenePreprocessor(resolution=resolution)
    resampling = getattr(Image, "Resampling", Image)
    tensors: List[torch.Tensor] = []
    for image_id in object_views:
        image_path = object_ref_dir / "rgb" / f"{int(image_id):06d}.png"
        mask_path = object_ref_dir / "mask_visib" / f"{int(image_id):06d}_000000.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing object reference view: {image_path}")
        rgb_arr = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        if mask_path.is_file():
            mask_arr = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
            white_bg = np.full_like(rgb_arr, 255)
            white_bg[mask_arr > 0] = rgb_arr[mask_arr > 0]
            image = Image.fromarray(white_bg, mode="RGB")
        else:
            image = Image.fromarray(rgb_arr, mode="RGB")
        image = image.resize(tuple(resolution), resampling.LANCZOS)
        tensors.append(processor.transform(image))
    return torch.stack(tensors, dim=0)  # (K, 3, H, W)


# ============================================================
# DataParallel-friendly wrapper around model.inference
# ============================================================
class InferenceWrap(nn.Module):
    """Expose ``model.inference`` via ``forward`` so we can wrap it in DataParallel."""

    def __init__(self, model: OmniVGGT, use_depth: bool):
        super().__init__()
        self.model = model
        self.use_depth = bool(use_depth)
        # Output keys we actually need downstream — gathering tensors only.
        self.output_keys = ("object_pose", "object_translation", "object_presence_logits")

    def forward(
        self,
        images: torch.Tensor,
        object_images: torch.Tensor,
        depth: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ):
        outputs = self.model.inference(
            images=images,
            object_images=object_images,
            extrinsics=None,
            intrinsics=None,
            depth=depth if self.use_depth else None,
            mask=mask if self.use_depth else None,
            camera_gt_index=[],
            depth_gt_index=[0] if self.use_depth else [],
        )
        return {k: outputs[k] for k in self.output_keys if k in outputs}


# ============================================================
# Model build
# ============================================================
def build_model(cfg: Dict, checkpoint_path: Path, device: torch.device) -> OmniVGGT:
    model = OmniVGGT(
        enable_camera=cfg.get("enable_camera", True),
        enable_point=cfg.get("enable_point", True),
        enable_depth=cfg.get("enable_depth", True),
        enable_object_mask=cfg.get("enable_object_mask", False),
        enable_object_srt=cfg.get("enable_object_srt", False),
        enable_object_size=cfg.get("enable_object_size", True),
        always_use_depth_gt=cfg.get("always_use_depth_gt", False),
        patch_embed_pretrained_path=cfg.get("patch_embed_pretrained_path", None),
        load_patch_embed_from_hub=cfg.get("load_patch_embed_from_hub", True),
        cam_drop_prob=cfg.get("cam_drop_prob", 0.1),
        depth_drop_prob=cfg.get("depth_drop_prob", 0.1),
        object_pose_context_pool=cfg.get("object_pose_context_pool", "flatten"),
        object_pose_use_global_scene_object_concat=cfg.get("object_pose_use_global_scene_object_concat", False),
        object_pose_transformer_depth=cfg.get("object_pose_transformer_depth", 6),
        object_pose_transformer_heads=cfg.get("object_pose_transformer_heads", 8),
        object_pose_transformer_mlp_dim=cfg.get("object_pose_transformer_mlp_dim", 1024),
        object_pose_transformer_dim_head=cfg.get("object_pose_transformer_dim_head", 64),
        object_pose_transformer_dropout=cfg.get("object_pose_transformer_dropout", 0.0),
        object_pose_transformer_emb_dropout=cfg.get("object_pose_transformer_emb_dropout", 0.0),
        object_pose_transformer_norm=cfg.get("object_pose_transformer_norm", "layer"),
        object_pose_transformer_dim=cfg.get("object_pose_transformer_dim", 1024),
        object_pose_ief_iters=cfg.get("object_pose_ief_iters", 1),
        object_pose_init_params_path=cfg.get("object_pose_init_params_path", None),
        enable_multi_layer_object_prototype_cross_attn=cfg.get("enable_multi_layer_object_prototype_cross_attn", False),
        object_prototype_layer_indices=cfg.get("object_prototype_layer_indices", (4, 11, 17, 23)),
        object_prototype_num_tokens=cfg.get("object_prototype_num_tokens", 4),
        object_prototype_object_encoder_no_grad=cfg.get("object_prototype_object_encoder_no_grad", False),
        object_cross_attn_heads=cfg.get("object_cross_attn_heads", 16),
    )
    if checkpoint_path.suffix == ".safetensors":
        state_dict = load_safetensors_file(str(checkpoint_path), device="cpu")
    else:
        raw = torch.load(checkpoint_path, map_location="cpu")
        state_dict = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    missing, unexpected = model.load_state_dict(state_dict, strict=bool(cfg.get("model_load_strict", False)))
    if missing:
        print(f"[eval] Missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"[eval] Unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
    model.eval().to(device)
    return model


# ============================================================
# Split readers + train-object filter
# ============================================================
def collect_object_ids_from_split(path: Path) -> set:
    payload = read_json(path)
    ids: set = set()
    for scene in payload.get("scenes", []):
        if "object_id" in scene:
            ids.add(int(scene["object_id"]))
        for key in ("object_ids", "eligible_object_ids", "anchor_object_ids"):
            for v in scene.get(key, []) or []:
                ids.add(int(v))
    for v in payload.get("anchor_object_ids", []) or []:
        ids.add(int(v))
    return ids


def build_object_reference_index(
    object_image_root: Path,
    required_views: Sequence[int],
) -> Dict[int, Path]:
    index: Dict[int, Path] = {}
    if not object_image_root.is_dir():
        return index
    for d in sorted(p for p in object_image_root.iterdir() if p.is_dir()):
        try:
            oid = ov9d_object_id_from_key(d.name)
        except ValueError:
            continue
        image_ids = {int(p.stem) for p in (d / "rgb").glob("*.png")}
        if all(v in image_ids for v in required_views):
            index[oid] = d
    return index


def shard_iter(iterable, shard_index: int, num_shards: int):
    """Yield every Nth element starting at shard_index (deterministic interleaved partition)."""
    if num_shards <= 1:
        yield from iterable
        return
    for i, item in enumerate(iterable):
        if i % num_shards == shard_index:
            yield item


def resolve_scene_dir(dataset_root: Path, scene_entry: Dict) -> Path:
    rel = scene_entry.get("relative_path")
    if rel:
        cand = dataset_root / str(rel)
        if cand.is_dir():
            return cand
    name = str(scene_entry["scene_name"])
    for sub in ("oo3d9dmulti", "oo3d9dsingle"):
        cand = dataset_root / sub / name
        if cand.is_dir():
            return cand
    return dataset_root / name


# ============================================================
# Sample preparation (CPU tensors) + per-sample metric decode
# ============================================================
def prepare_sample(
    scene_dir: Path,
    image_id: int,
    object_id: int,
    object_ref_dir: Path,
    object_views: Sequence[int],
    resolution,
    *,
    scene_gt_cache: Dict[Path, Dict],
    object_tensor_cache: Dict[int, torch.Tensor],
) -> Dict | None:
    """Return a dict with CPU tensors + GT info, or None if the object is not in the frame."""
    if scene_dir in scene_gt_cache:
        scene_gt = scene_gt_cache[scene_dir]
    else:
        scene_gt = read_json(scene_dir / "scene_gt.json")
        scene_gt_cache[scene_dir] = scene_gt
    gts = scene_gt.get(str(image_id), [])
    object_index = next(
        (idx for idx, gt in enumerate(gts) if int(gt.get("obj_id", -1)) == int(object_id)),
        None,
    )
    if object_index is None:
        return None

    image_tensor, depth_tensor, mask_tensor = load_scene_frame_inputs_cpu(
        scene_dir, image_id, object_index, resolution,
    )
    if int(object_id) in object_tensor_cache:
        object_tensor = object_tensor_cache[int(object_id)]
    else:
        object_tensor = load_object_tensor_cpu(object_ref_dir, object_views, resolution)
        object_tensor_cache[int(object_id)] = object_tensor

    gt = gts[object_index]
    gt_rotation_cam = np.asarray(gt["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
    gt_translation_cam = np.asarray(gt["cam_t_m2c"], dtype=np.float32).reshape(3) / 1000.0  # m

    return {
        "image": image_tensor,  # (1, 3, H, W)
        "depth": depth_tensor,  # (1, H, W, 1)
        "mask": mask_tensor,    # (1, H, W)
        "object": object_tensor,  # (K, 3, H, W)
        "object_id": int(object_id),
        "gt_rotation_cam": gt_rotation_cam,
        "gt_translation_cam": gt_translation_cam,
    }


def decode_sample(
    pred_rot6d_cam: np.ndarray,
    pred_translation_cam: np.ndarray,
    presence_logit: float | None,
    object_id: int,
    gt_rotation_cam: np.ndarray,
    gt_translation_cam: np.ndarray,
    symmetry_info_path: str,
    symmetry_continuous_steps: int,
) -> Dict:
    pred_rotation_cam = rot6d_to_matrix(pred_rot6d_cam).astype(np.float32)
    rot_err_deg = symmetric_rotation_error_degrees(
        pred_rotation_cam, gt_rotation_cam, object_id, symmetry_info_path, symmetry_continuous_steps,
    )
    trans_diff_m = pred_translation_cam.astype(np.float64) - gt_translation_cam.astype(np.float64)
    trans_err_cm = float(np.linalg.norm(trans_diff_m) * 100.0)
    pose_loss = symmetric_rot6d_l1(
        pred_rot6d_cam, gt_rotation_cam, object_id, symmetry_info_path, symmetry_continuous_steps,
    )
    translation_loss = float(np.abs(trans_diff_m).mean())
    gt_distance_m = float(np.linalg.norm(gt_translation_cam.astype(np.float64)))
    gt_z_m = float(gt_translation_cam[2])
    presence_prob = (
        float(torch.sigmoid(torch.tensor(presence_logit)).item()) if presence_logit is not None else None
    )
    return {
        "object_id": int(object_id),
        "rot_err_deg": float(rot_err_deg),
        "trans_err_cm": float(trans_err_cm),
        "pose_loss": float(pose_loss),
        "translation_loss": float(translation_loss),
        "srt_loss": float(pose_loss + translation_loss),
        "gt_distance_m": gt_distance_m,
        "gt_z_m": gt_z_m,
        "presence_logit": float(presence_logit) if presence_logit is not None else None,
        "presence_prob": presence_prob,
    }


def batched_forward(
    inference_module: nn.Module,
    samples: List[Dict],
    primary_device: torch.device,
    use_depth: bool,
) -> List[Dict]:
    """Stack CPU tensors of `samples`, run a single batched forward, return per-sample outputs (CPU numpy)."""
    images = torch.stack([s["image"] for s in samples], dim=0).to(primary_device, non_blocking=True)  # (B,1,3,H,W)
    objects = torch.stack([s["object"] for s in samples], dim=0).to(primary_device, non_blocking=True)  # (B,K,3,H,W)
    if use_depth:
        depth = torch.stack([s["depth"] for s in samples], dim=0).to(primary_device, non_blocking=True)  # (B,1,H,W,1)
        mask = torch.stack([s["mask"] for s in samples], dim=0).to(primary_device, non_blocking=True)  # (B,1,H,W)
    else:
        depth = None
        mask = None

    with torch.inference_mode():
        with torch.autocast(device_type=primary_device.type, dtype=torch.bfloat16, enabled=primary_device.type == "cuda"):
            outputs = inference_module(images=images, object_images=objects, depth=depth, mask=mask)

    pose = outputs["object_pose"].detach().float().cpu().numpy()  # (B, 6)
    trans = outputs["object_translation"].detach().float().cpu().numpy().astype(np.float32)  # (B, 3)
    presence = (
        outputs["object_presence_logits"].detach().float().cpu().numpy().reshape(-1)
        if "object_presence_logits" in outputs else None
    )
    decoded = []
    for i in range(len(samples)):
        decoded.append({
            "pred_rot6d_cam": pose[i],
            "pred_translation_cam": trans[i],
            "presence_logit": float(presence[i]) if presence is not None else None,
        })
    return decoded


# ============================================================
# Aggregation
# ============================================================
def aggregate(samples: List[Dict]) -> Dict:
    if not samples:
        return {"count": 0}
    rot = np.array([s["rot_err_deg"] for s in samples], dtype=np.float64)
    trans = np.array([s["trans_err_cm"] for s in samples], dtype=np.float64)
    pose_loss = np.array([s["pose_loss"] for s in samples], dtype=np.float64)
    trans_loss = np.array([s["translation_loss"] for s in samples], dtype=np.float64)
    srt_loss = np.array([s["srt_loss"] for s in samples], dtype=np.float64)
    distance = np.array([s["gt_distance_m"] for s in samples], dtype=np.float64)
    presence_prob = np.array(
        [s["presence_prob"] for s in samples if s["presence_prob"] is not None], dtype=np.float64
    )

    def acc(mask):
        return float(np.mean(mask)) if mask.size > 0 else float("nan")

    def pct_dict(arr: np.ndarray) -> Dict[str, float]:
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.median(arr)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(arr.max()),
        }

    return {
        "count": int(len(samples)),
        "gt_distance_m": {
            "min": float(distance.min()),
            "p33": float(np.percentile(distance, 33)),
            "median": float(np.median(distance)),
            "p66": float(np.percentile(distance, 66)),
            "max": float(distance.max()),
            "mean": float(distance.mean()),
        },
        "rot_err_deg": pct_dict(rot),
        "trans_err_cm": pct_dict(trans),
        "pose_loss_l1": float(pose_loss.mean()),
        "translation_loss_l1": float(trans_loss.mean()),
        "srt_loss_total": float(srt_loss.mean()),
        "accuracy_rot_deg": {
            "<2": acc(rot < 2.0),
            "<5": acc(rot < 5.0),
            "<10": acc(rot < 10.0),
            "<15": acc(rot < 15.0),
            "<20": acc(rot < 20.0),
            "<30": acc(rot < 30.0),
        },
        "accuracy_trans_cm": {
            "<1": acc(trans < 1.0),
            "<2": acc(trans < 2.0),
            "<5": acc(trans < 5.0),
            "<10": acc(trans < 10.0),
            "<20": acc(trans < 20.0),
        },
        "accuracy_joint": {
            "rot<5deg_and_trans<2cm": acc((rot < 5.0) & (trans < 2.0)),
            "rot<10deg_and_trans<5cm": acc((rot < 10.0) & (trans < 5.0)),
            "rot<15deg_and_trans<10cm": acc((rot < 15.0) & (trans < 10.0)),
            "rot<30deg_and_trans<20cm": acc((rot < 30.0) & (trans < 20.0)),
        },
        "presence_prob_mean": float(presence_prob.mean()) if presence_prob.size > 0 else None,
        "presence_recall_at_0.5": (
            float(np.mean(presence_prob >= 0.5)) if presence_prob.size > 0 else None
        ),
    }


def bucket_by_distance(samples: List[Dict]) -> Tuple[Dict[str, List[Dict]], Dict[str, float]]:
    distances = np.array([s["gt_distance_m"] for s in samples], dtype=np.float64)
    p33, p66 = (float(x) for x in np.percentile(distances, [33.333, 66.667]))
    near, mid, far = [], [], []
    for s in samples:
        d = s["gt_distance_m"]
        if d < p33:
            near.append(s)
        elif d < p66:
            mid.append(s)
        else:
            far.append(s)
    return {"near": near, "mid": mid, "far": far}, {"p33": p33, "p66": p66}


# ============================================================
# Iteration over single + multi splits
# ============================================================
def iter_single_targets(
    split_payload: Dict,
    dataset_root: Path,
    train_object_ids: set,
    object_reference_index: Dict[int, Path],
):
    """Yield (scene_name, scene_dir, frame_id, object_id, object_ref_dir) for every present-object frame."""
    for scene in split_payload.get("scenes", []):
        object_id = int(scene["object_id"])
        if object_id in train_object_ids:
            continue  # already seen during training
        if object_id not in object_reference_index:
            continue  # no reference view set we can use
        scene_dir = resolve_scene_dir(dataset_root, scene)
        if not scene_dir.is_dir():
            continue
        scene_gt = read_json(scene_dir / "scene_gt.json")
        ref_dir = object_reference_index[object_id]
        for frame_key in sorted(scene_gt.keys(), key=lambda x: int(x)):
            frame_id = int(frame_key)
            yield scene["scene_name"], scene_dir, frame_id, object_id, ref_dir


def iter_multi_targets(
    split_payload: Dict,
    dataset_root: Path,
    train_object_ids: set,
    object_reference_index: Dict[int, Path],
):
    for scene in split_payload.get("scenes", []):
        eligible = [int(x) for x in scene.get("eligible_object_ids", []) or []]
        # Per user request: only evaluate eligible ids that are NOT in single train.json.
        targets = [oid for oid in eligible if oid not in train_object_ids and oid in object_reference_index]
        if not targets:
            continue
        scene_dir = resolve_scene_dir(dataset_root, scene)
        if not scene_dir.is_dir():
            continue
        scene_gt = read_json(scene_dir / "scene_gt.json")
        for frame_key in sorted(scene_gt.keys(), key=lambda x: int(x)):
            frame_id = int(frame_key)
            present_ids = {int(gt.get("obj_id", -1)) for gt in scene_gt[frame_key]}
            for object_id in targets:
                if object_id not in present_ids:
                    continue
                yield scene["scene_name"], scene_dir, frame_id, object_id, object_reference_index[object_id]


# ============================================================
# Pretty printing
# ============================================================
def format_metric_block(name: str, agg: Dict) -> str:
    if agg["count"] == 0:
        return f"[{name}] no samples"
    lines = [f"[{name}] count={agg['count']}"]
    r = agg["rot_err_deg"]
    lines.append(
        f"  rot_err_deg   mean={r['mean']:.3f}  median={r['median']:.3f}  std={r['std']:.3f}"
    )
    lines.append(
        f"                p25={r['p25']:.3f}  p75={r['p75']:.3f}  p90={r['p90']:.3f}  "
        f"p95={r['p95']:.3f}  max={r['max']:.3f}"
    )
    t = agg["trans_err_cm"]
    lines.append(
        f"  trans_err_cm  mean={t['mean']:.3f}  median={t['median']:.3f}  std={t['std']:.3f}"
    )
    lines.append(
        f"                p25={t['p25']:.3f}  p75={t['p75']:.3f}  p90={t['p90']:.3f}  "
        f"p95={t['p95']:.3f}  max={t['max']:.3f}"
    )
    lines.append(
        f"  losses        pose_l1={agg['pose_loss_l1']:.4f} "
        f"trans_l1={agg['translation_loss_l1']:.4f} total={agg['srt_loss_total']:.4f}"
    )
    ar = agg["accuracy_rot_deg"]
    lines.append(
        f"  acc rot       <2={ar['<2']*100:.2f}% <5={ar['<5']*100:.2f}% <10={ar['<10']*100:.2f}% "
        f"<15={ar['<15']*100:.2f}% <20={ar['<20']*100:.2f}% <30={ar['<30']*100:.2f}%"
    )
    at = agg["accuracy_trans_cm"]
    lines.append(
        f"  acc trans     <1cm={at['<1']*100:.2f}% <2cm={at['<2']*100:.2f}% <5cm={at['<5']*100:.2f}% "
        f"<10cm={at['<10']*100:.2f}% <20cm={at['<20']*100:.2f}%"
    )
    aj = agg["accuracy_joint"]
    lines.append(
        f"  acc joint     5deg&2cm={aj['rot<5deg_and_trans<2cm']*100:.2f}% "
        f"10deg&5cm={aj['rot<10deg_and_trans<5cm']*100:.2f}% "
        f"15deg&10cm={aj['rot<15deg_and_trans<10cm']*100:.2f}% "
        f"30deg&20cm={aj['rot<30deg_and_trans<20cm']*100:.2f}%"
    )
    if agg.get("presence_recall_at_0.5") is not None:
        lines.append(
            f"  presence      mean_prob={agg['presence_prob_mean']:.3f} "
            f"recall@0.5={agg['presence_recall_at_0.5']*100:.2f}%"
        )
    dist = agg["gt_distance_m"]
    lines.append(
        f"  gt_distance   min={dist['min']:.3f}m p33={dist['p33']:.3f}m "
        f"median={dist['median']:.3f}m p66={dist['p66']:.3f}m max={dist['max']:.3f}m"
    )
    return "\n".join(lines)


def report(name: str, samples: List[Dict]) -> Dict:
    print()
    print("=" * 70)
    print(f"== Split: {name}")
    print("=" * 70)
    if not samples:
        print(f"[{name}] no evaluated samples")
        return {"split": name, "overall": {"count": 0}, "buckets": {}}
    overall = aggregate(samples)
    print(format_metric_block(f"{name}/overall", overall))
    buckets, thresholds = bucket_by_distance(samples)
    bucket_aggs = {}
    print(
        f"  distance buckets  p33={thresholds['p33']:.3f}m p66={thresholds['p66']:.3f}m "
        f"(near<{thresholds['p33']:.3f} <= mid < {thresholds['p66']:.3f} <= far)"
    )
    for bucket_name in ("near", "mid", "far"):
        bucket_agg = aggregate(buckets[bucket_name])
        bucket_aggs[bucket_name] = bucket_agg
        print(format_metric_block(f"{name}/{bucket_name}", bucket_agg))
    return {
        "split": name,
        "overall": overall,
        "bucket_thresholds_m": thresholds,
        "buckets": bucket_aggs,
    }


# ============================================================
# Orchestrator: fan out shards across GPUs as subprocesses
# ============================================================
def orchestrate_shards(args: argparse.Namespace, gpu_ids: List[str]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    num_shards = len(gpu_ids)
    print(f"[orchestrator] launching {num_shards} workers, one per GPU: {gpu_ids}")
    print(f"[orchestrator] batch_size={args.batch_size}  output_dir={args.output_dir}")

    base_cmd = [
        sys.executable, "-u", str(Path(__file__).resolve()),
        "--config", str(args.config),
        "--checkpoint", str(args.checkpoint),
        "--dataset-root", str(args.dataset_root),
        "--object-image-root", str(args.object_image_root),
        "--single-test-json", str(args.single_test_json),
        "--multi-test-json", str(args.multi_test_json),
        "--single-train-json", str(args.single_train_json),
        "--output-dir", str(args.output_dir),
        "--batch-size", str(args.batch_size),
        "--num-shards", str(num_shards),
        "--skip-self-test",  # parent already ran self-tests
    ]
    if args.no_depth:
        base_cmd.append("--no-depth")
    if args.limit is not None:
        base_cmd.extend(["--limit", str(args.limit)])

    procs: List[Tuple[int, subprocess.Popen]] = []
    for shard_index, gpu_id in enumerate(gpu_ids):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        log_path = args.output_dir / f"shard_{shard_index:02d}.log"
        cmd = base_cmd + ["--shard-index", str(shard_index)]
        log_fh = open(log_path, "w", encoding="utf-8")
        print(f"[orchestrator] shard {shard_index} → CUDA_VISIBLE_DEVICES={gpu_id}  log={log_path}")
        p = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        procs.append((shard_index, p))

    # ---- Live progress: tail bytes from each shard's log ----
    start = time.time()
    while True:
        time.sleep(15)
        alive = [shard for shard, p in procs if p.poll() is None]
        statuses = []
        for shard_index, p in procs:
            log_path = args.output_dir / f"shard_{shard_index:02d}.log"
            try:
                with log_path.open("r", encoding="utf-8") as handle:
                    handle.seek(0, os.SEEK_END)
                    size = handle.tell()
                    tail_bytes = 600
                    handle.seek(max(0, size - tail_bytes), os.SEEK_SET)
                    tail = handle.read()
                last_line = tail.strip().splitlines()[-1] if tail.strip() else ""
                statuses.append(f"  shard {shard_index}: alive={p.poll() is None}  last={last_line[-200:]}")
            except FileNotFoundError:
                statuses.append(f"  shard {shard_index}: log missing")
        elapsed = time.time() - start
        print(f"[orchestrator] t={elapsed:7.0f}s  alive={len(alive)}/{len(procs)}")
        for line in statuses:
            print(line)
        if not alive:
            break

    return_codes = [p.wait() for _, p in procs]
    if any(rc != 0 for rc in return_codes):
        print(f"[orchestrator] workers returned: {return_codes} (non-zero indicates failure)")

    # ---- Merge per-shard samples and aggregate ----
    all_single: List[Dict] = []
    all_multi: List[Dict] = []
    for shard_index, _ in procs:
        shard_path = args.output_dir / f"samples_shard_{shard_index:02d}.jsonl"
        if not shard_path.is_file():
            print(f"[orchestrator] missing shard file: {shard_path}")
            continue
        with shard_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                tag = rec.pop("split", None)
                if tag == "single":
                    all_single.append(rec)
                elif tag == "multi":
                    all_multi.append(rec)
    print(f"[orchestrator] merged samples: single={len(all_single)}  multi={len(all_multi)}")

    single_report = report("single", all_single)
    multi_report = report("multi", all_multi)
    combined_report = report("combined", all_single + all_multi)

    summary = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "use_depth": not args.no_depth,
        "shards": [
            {"shard_index": i, "gpu": gpu_ids[i], "return_code": return_codes[i]}
            for i in range(len(procs))
        ],
        "single": single_report,
        "multi": multi_report,
        "combined": combined_report,
    }
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\n[orchestrator] wrote summary to {summary_path}")

    samples_path = args.output_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for tag, items in (("single", all_single), ("multi", all_multi)):
            for s in items:
                handle.write(json.dumps({"split": tag, **s}) + "\n")
    print(f"[orchestrator] wrote merged per-sample records to {samples_path}")


# ============================================================
# Entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Inference-only eval on OV9D test splits")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--object-image-root", type=Path, default=DEFAULT_OBJECT_IMAGE_ROOT)
    parser.add_argument("--single-test-json", type=Path, default=DEFAULT_SINGLE_TEST_JSON)
    parser.add_argument("--multi-test-json", type=Path, default=DEFAULT_MULTI_TEST_JSON)
    parser.add_argument("--single-train-json", type=Path, default=DEFAULT_SINGLE_TRAIN_JSON)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "eval_unseen")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-depth", action="store_true",
                        help="Drop depth input (default uses depth, matching training).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on samples per split (for smoke tests).")
    parser.add_argument("--gpu", type=str, default=None,
                        help="Optional CUDA_VISIBLE_DEVICES override (e.g. '0' or '0,1,2,3').")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU ids to use for DataParallel after CUDA_VISIBLE_DEVICES "
                             "mapping (e.g. '0,1,2,3'). Defaults to all visible CUDA devices.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Inference batch size (per global step; split across GPUs by DataParallel).")
    parser.add_argument("--self-test-only", action="store_true",
                        help="Run metric self-tests and exit (no model load, no inference).")
    parser.add_argument("--skip-self-test", action="store_true",
                        help="Skip the metric self-test that runs before inference.")
    parser.add_argument("--shard-index", type=int, default=None,
                        help="Worker mode: 0-based shard index (used internally when sharding across GPUs).")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Worker mode: total number of shards. Workers process samples whose global index "
                             "modulo num_shards equals shard_index.")
    args = parser.parse_args()

    if not args.skip_self_test:
        _self_test_metrics()
    if args.self_test_only:
        return

    # ---- Orchestrator: fan out to subprocesses, one per GPU ----
    is_worker = args.shard_index is not None
    if not is_worker and args.gpus is not None and "," in str(args.gpus):
        gpu_ids = [x.strip() for x in str(args.gpus).split(",") if x.strip() != ""]
        if len(gpu_ids) > 1:
            return orchestrate_shards(args, gpu_ids)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.device:
        primary_device = torch.device(args.device)
    elif torch.cuda.is_available():
        primary_device = torch.device("cuda:0")
    else:
        primary_device = torch.device("cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve device list for DataParallel.
    if primary_device.type == "cuda":
        if args.gpus is not None:
            device_ids = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
        else:
            device_ids = list(range(torch.cuda.device_count()))
        if not device_ids:
            device_ids = [0]
        primary_device = torch.device(f"cuda:{device_ids[0]}")
    else:
        device_ids = []

    cfg = load_config(args.config)
    resolution = tuple(int(v) for v in cfg.get("resolution", DEFAULT_RESOLUTION))
    object_views = tuple(int(v) for v in cfg.get("fixed_object_view_ids", DEFAULT_OBJECT_VIEWS))
    symmetry_info_path = resolve_local_path(cfg.get("object_srt_symmetry_info_path", ""))
    symmetry_info_path_str = str(symmetry_info_path) if symmetry_info_path is not None else ""
    symmetry_continuous_steps = int(cfg.get("object_srt_symmetry_continuous_steps", 72))

    print(f"[eval] config={args.config}")
    print(f"[eval] checkpoint={args.checkpoint}")
    print(f"[eval] dataset_root={args.dataset_root}")
    print(f"[eval] object_image_root={args.object_image_root}")
    print(f"[eval] resolution={resolution}  object_views={object_views}")
    print(f"[eval] symmetry_info={symmetry_info_path_str}  steps={symmetry_continuous_steps}")
    print(f"[eval] use_depth={not args.no_depth}")
    print(f"[eval] device(primary)={primary_device}  data_parallel_ids={device_ids}  batch_size={args.batch_size}")

    model = build_model(cfg, args.checkpoint, primary_device)
    inference_module = InferenceWrap(model, use_depth=not args.no_depth).to(primary_device).eval()
    if len(device_ids) > 1:
        inference_module = nn.DataParallel(inference_module, device_ids=device_ids)

    object_reference_index = build_object_reference_index(args.object_image_root, object_views)
    print(f"[eval] object_reference_index: {len(object_reference_index)} objects with all required views")

    train_object_ids = collect_object_ids_from_split(args.single_train_json)
    print(f"[eval] single/train.json contains {len(train_object_ids)} object_ids (treated as seen)")

    single_payload = read_json(args.single_test_json)
    multi_payload = read_json(args.multi_test_json)

    # ----- Quick stats on filtering -----
    multi_eligible_total = 0
    multi_eligible_kept = 0
    multi_filtered_objects = set()
    multi_kept_objects = set()
    for sc in multi_payload.get("scenes", []):
        for oid in sc.get("eligible_object_ids", []) or []:
            oid = int(oid)
            multi_eligible_total += 1
            if oid in train_object_ids:
                multi_filtered_objects.add(oid)
            elif oid not in object_reference_index:
                multi_filtered_objects.add(oid)
            else:
                multi_eligible_kept += 1
                multi_kept_objects.add(oid)
    print(
        f"[eval] multi eligible occurrences: total={multi_eligible_total} "
        f"kept={multi_eligible_kept}  filtered_objects={len(multi_filtered_objects)} "
        f"kept_unique_objects={len(multi_kept_objects)}"
    )

    single_total = 0
    single_kept = 0
    single_filtered_objects = set()
    single_kept_objects = set()
    for sc in single_payload.get("scenes", []):
        oid = int(sc["object_id"])
        single_total += 1
        if oid in train_object_ids or oid not in object_reference_index:
            single_filtered_objects.add(oid)
        else:
            single_kept += 1
            single_kept_objects.add(oid)
    print(
        f"[eval] single scenes: total={single_total} kept={single_kept} "
        f"filtered_unique_objects={len(single_filtered_objects)} "
        f"kept_unique_objects={len(single_kept_objects)}"
    )

    use_depth = not args.no_depth

    def run_split(name: str, targets_iter, payload: Dict):
        scenes = payload.get("scenes", []) if payload else []
        print(f"\n[eval] running {name} over {len(scenes)} scenes")
        samples: List[Dict] = []
        skipped = 0
        scene_gt_cache: Dict[Path, Dict] = {}
        object_tensor_cache: Dict[int, torch.Tensor] = {}

        def flush(batch: List[Dict]):
            if not batch:
                return
            preds = batched_forward(
                inference_module, batch, primary_device, use_depth=use_depth,
            )
            for meta, pred in zip(batch, preds):
                decoded = decode_sample(
                    pred_rot6d_cam=pred["pred_rot6d_cam"],
                    pred_translation_cam=pred["pred_translation_cam"],
                    presence_logit=pred["presence_logit"],
                    object_id=meta["object_id"],
                    gt_rotation_cam=meta["gt_rotation_cam"],
                    gt_translation_cam=meta["gt_translation_cam"],
                    symmetry_info_path=symmetry_info_path_str,
                    symmetry_continuous_steps=symmetry_continuous_steps,
                )
                decoded.update({"scene_name": meta["scene_name"], "frame_id": meta["frame_id"]})
                samples.append(decoded)

        batch: List[Dict] = []
        pbar = tqdm(targets_iter, desc=name, unit="sample", smoothing=0.02)
        prev_scene_dir: Path | None = None
        for (scene_name, scene_dir, frame_id, object_id, ref_dir) in pbar:
            if prev_scene_dir is not None and prev_scene_dir != scene_dir:
                scene_gt_cache.pop(prev_scene_dir, None)
            prev_scene_dir = scene_dir
            try:
                prepared = prepare_sample(
                    scene_dir=scene_dir,
                    image_id=frame_id,
                    object_id=object_id,
                    object_ref_dir=ref_dir,
                    object_views=object_views,
                    resolution=resolution,
                    scene_gt_cache=scene_gt_cache,
                    object_tensor_cache=object_tensor_cache,
                )
            except FileNotFoundError as exc:
                pbar.write(f"  [skip] {scene_name} frame={frame_id} obj={object_id}: {exc}")
                skipped += 1
                continue
            if prepared is None:
                continue
            prepared.update({"scene_name": scene_name, "frame_id": frame_id})
            batch.append(prepared)

            if len(batch) >= args.batch_size:
                flush(batch)
                batch = []
                pbar.update(0)
                if args.limit is not None and len(samples) >= args.limit:
                    break

            if len(object_tensor_cache) > 64:
                for k in list(object_tensor_cache.keys())[:32]:
                    object_tensor_cache.pop(k, None)

        if batch and (args.limit is None or len(samples) < args.limit):
            flush(batch)

        if args.limit is not None and len(samples) > args.limit:
            samples = samples[: args.limit]
        print(f"[eval] {name}: collected {len(samples)} samples, skipped {skipped}")
        return samples

    shard_index = int(args.shard_index) if args.shard_index is not None else 0
    num_shards = int(args.num_shards) if args.num_shards else 1
    if num_shards > 1:
        print(f"[eval] worker shard: index={shard_index} / total={num_shards}")

    single_iter = shard_iter(
        iter_single_targets(single_payload, args.dataset_root, train_object_ids, object_reference_index),
        shard_index, num_shards,
    )
    multi_iter = shard_iter(
        iter_multi_targets(multi_payload, args.dataset_root, train_object_ids, object_reference_index),
        shard_index, num_shards,
    )

    single_samples = run_split("single", single_iter, single_payload)
    multi_samples = run_split("multi", multi_iter, multi_payload)

    # ---- Worker mode: write per-shard samples and exit (no aggregate) ----
    if is_worker:
        shard_path = args.output_dir / f"samples_shard_{shard_index:02d}.jsonl"
        with shard_path.open("w", encoding="utf-8") as handle:
            for tag, items in (("single", single_samples), ("multi", multi_samples)):
                for s in items:
                    handle.write(json.dumps({"split": tag, **s}) + "\n")
        print(f"[eval] shard {shard_index} wrote {shard_path}")
        return

    single_report = report("single", single_samples)
    multi_report = report("multi", multi_samples)
    combined_report = report("combined", single_samples + multi_samples)

    summary = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "use_depth": use_depth,
        "resolution": list(resolution),
        "object_views": list(object_views),
        "symmetry_info_path": symmetry_info_path_str,
        "train_object_ids_count": len(train_object_ids),
        "filter": {
            "single": {"total": single_total, "kept": single_kept},
            "multi": {"total": multi_eligible_total, "kept": multi_eligible_kept},
        },
        "single": single_report,
        "multi": multi_report,
        "combined": combined_report,
    }
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\n[eval] wrote summary to {summary_path}")

    samples_path = args.output_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for tag, items in (("single", single_samples), ("multi", multi_samples)):
            for s in items:
                handle.write(json.dumps({"split": tag, **s}) + "\n")
    print(f"[eval] wrote per-sample records to {samples_path}")


if __name__ == "__main__":
    main()
