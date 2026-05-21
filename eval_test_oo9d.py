"""Evaluate model.safetensors on a single rendered scene under test_oo9d.

Reuses the inference path from demo_gradio_6dpose_0519.py but runs over every
frame in the rendered scene and prints rotation / translation losses.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
_LOCAL_TMP = PROJECT_ROOT / "tmp"
_LOCAL_TMP.mkdir(parents=True, exist_ok=True)
(_LOCAL_TMP / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(_LOCAL_TMP))
os.environ.setdefault("MPLCONFIGDIR", str(_LOCAL_TMP / "matplotlib"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import argparse
import json
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as load_safetensors_file

import runpy

from omnivggt.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from omnivggt.loss import _load_symmetry_info
from omnivggt.models.omnivggt import OmniVGGT
from omnivggt.utils.image import ImgNorm


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train_oo9d.py"
DEFAULT_CHECKPOINT = Path(
    "/mnt/train-data-4-hdd/yian/freepose/omni-object_clone/outputs/0515_LARGE/model.safetensors"
)
DEFAULT_TEST_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose/ov9d/test_oo9d")
DEFAULT_OBJECT_NAME = "obj_000002"


def load_config(config_path: Path) -> Dict:
    cfg = runpy.run_path(str(config_path))
    return {k: v for k, v in cfg.items() if not k.startswith("__")}


def parse_dataset_ctor_arg(dataset_expr: str, arg_name: str, default=None):
    pattern = rf"{re.escape(arg_name)}\s*=\s*([^,()]+|\([^)]*\))"
    match = re.search(pattern, dataset_expr)
    if not match:
        return default
    try:
        return eval(match.group(1).strip(), {"__builtins__": {}}, {})
    except Exception:
        return default


def resolve_runtime_settings(cfg: Dict) -> Dict:
    dataset_expr = str(cfg.get("val_dataset", cfg.get("train_dataset", "")))
    object_input_views = tuple(
        parse_dataset_ctor_arg(
            dataset_expr,
            "object_input_views",
            default=parse_dataset_ctor_arg(dataset_expr, "fixed_object_view_ids", default=(1, 5, 10, 15)),
        )
    )
    resolution = tuple(int(v) for v in cfg.get("resolution", (518, 518)))
    return {
        "object_input_views": tuple(int(v) for v in object_input_views),
        "resolution": resolution,
    }


def build_model(cfg: Dict, ckpt: Path, device: torch.device) -> OmniVGGT:
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
    state = load_safetensors_file(str(ckpt), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=bool(cfg.get("model_load_strict", False)))
    if missing:
        print(f"[eval] Missing keys ({len(missing)}): showing first 8: {missing[:8]}")
    if unexpected:
        print(f"[eval] Unexpected keys ({len(unexpected)}): showing first 8: {unexpected[:8]}")
    model.eval().to(device)
    return model


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float64).reshape(3, 2)
    x_raw = rot6d[:, 0]
    y_raw = rot6d[:, 1]
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


def rotation_error_degrees(pred_rot: np.ndarray, gt_rot: np.ndarray) -> float:
    rel = np.asarray(pred_rot, dtype=np.float64) @ np.asarray(gt_rot, dtype=np.float64).T
    cos_theta = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def symmetric_rotation_error_degrees(
    pred_rot: np.ndarray,
    gt_rot: np.ndarray,
    object_id: int,
    symmetry_info_path: Path | None,
    continuous_steps: int,
) -> Tuple[float, int]:
    if symmetry_info_path is None or not symmetry_info_path.is_file():
        return rotation_error_degrees(pred_rot, gt_rot), 1
    symmetry_info = _load_symmetry_info(str(symmetry_info_path), int(continuous_steps))
    sym_rots = symmetry_info.get(int(object_id))
    if sym_rots is None:
        return rotation_error_degrees(pred_rot, gt_rot), 1
    candidates = sym_rots.detach().cpu().numpy().astype(np.float64)
    errors = [rotation_error_degrees(pred_rot, np.asarray(gt_rot, dtype=np.float64) @ s) for s in candidates]
    return float(min(errors)), len(errors)


def read_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as h:
        return json.load(h)


def read_depth_m(depth_path: Path, camera_entry: Dict) -> np.ndarray:
    raw = np.asarray(Image.open(depth_path), dtype=np.float32)
    depth_m = raw * float(camera_entry.get("depth_scale", 1.0)) / 1000.0
    depth_m[~np.isfinite(depth_m)] = 0.0
    depth_m[depth_m < 0.0] = 0.0
    return depth_m.astype(np.float32)


def read_binary_mask(mask_path: Path) -> np.ndarray:
    return (np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0).astype(np.float32)


class DemoScenePreprocessor(BaseStereoViewDataset):
    def __init__(self, resolution):
        super().__init__(dset="ov9d_demo", resolution=resolution, transform=ImgNorm, seed=0)


def crop_resize(processor, image, depthmap, object_mask, intrinsics, resolution, info):
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


def load_scene_frame(scene_dir: Path, image_id: int, object_id: int, resolution, device):
    cam_entry = read_json(scene_dir / "scene_camera.json")[str(image_id)]
    gts = read_json(scene_dir / "scene_gt.json")[str(image_id)]
    object_index = next(
        (i for i, gt in enumerate(gts) if int(gt.get("obj_id", -1)) == int(object_id)),
        None,
    )

    image = Image.open(scene_dir / "rgb" / f"{image_id:06d}.png").convert("RGB")
    depthmap = read_depth_m(scene_dir / "depth" / f"{image_id:06d}.png", cam_entry)
    intrinsics = np.asarray(cam_entry["cam_K"], dtype=np.float32).reshape(3, 3)
    object_mask = None
    if object_index is not None:
        mp = scene_dir / "mask_visib" / f"{image_id:06d}_{object_index:06d}.png"
        if mp.is_file():
            object_mask = read_binary_mask(mp)

    processor = DemoScenePreprocessor(resolution=resolution)
    image, depthmap, gt_mask, intrinsics = crop_resize(
        processor,
        image,
        depthmap,
        object_mask,
        intrinsics,
        resolution,
        info=str(scene_dir / "rgb" / f"{image_id:06d}.png"),
    )
    image_tensor = processor.transform(image).unsqueeze(0).unsqueeze(0).to(device)
    depth_tensor = torch.from_numpy(np.ascontiguousarray(depthmap.astype(np.float32)))[None, None, :, :, None].to(device)
    mask_tensor = torch.from_numpy((depthmap > 0).astype(np.float32))[None, None, :, :].to(device)
    return image_tensor, depth_tensor, mask_tensor, intrinsics, gts[object_index] if object_index is not None else None


def load_object_tensor(object_dir: Path, views: Sequence[int], resolution, device):
    processor = DemoScenePreprocessor(resolution=resolution)
    resampling = getattr(Image, "Resampling", Image)
    tensors = []
    for vid in views:
        img_path = object_dir / "rgb" / f"{int(vid):06d}.png"
        mask_path = object_dir / "mask_visib" / f"{int(vid):06d}_000000.png"
        if not img_path.is_file():
            raise FileNotFoundError(f"Missing object view: {img_path}")
        rgb = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        if mask_path.is_file():
            mk = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
            white_bg = np.full_like(rgb, 255)
            white_bg[mk > 0] = rgb[mk > 0]
            image = Image.fromarray(white_bg, mode="RGB")
        else:
            image = Image.fromarray(rgb, mode="RGB")
        image = image.resize(tuple(resolution), resampling.LANCZOS)
        tensors.append(processor.transform(image).to(device))
    return torch.stack(tensors, dim=0).unsqueeze(0)


def resolve_local_path(value):
    if not value:
        return None
    p = Path(str(value))
    if p.exists():
        return p
    replacements = {
        "/dataset/ov9d": "/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d",
        "/dataset": "/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d",
    }
    text = str(p)
    for prefix, replacement in replacements.items():
        if text == prefix or text.startswith(prefix + "/"):
            cand = Path(replacement + text[len(prefix):])
            if cand.exists():
                return cand
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT,
                        help="Directory containing <obj_name>/ and _object_images/<obj_name>/.")
    parser.add_argument("--obj-name", type=str, default=DEFAULT_OBJECT_NAME)
    parser.add_argument("--object-id", type=int, default=None,
                        help="Override object id; defaults to int from obj_name.")
    parser.add_argument("--use-depth-input", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    runtime = resolve_runtime_settings(cfg)
    object_views = runtime["object_input_views"]
    resolution = runtime["resolution"]

    sym_path = resolve_local_path(cfg.get("object_srt_symmetry_info_path"))
    sym_steps = int(cfg.get("object_srt_symmetry_continuous_steps", 72))

    scene_dir = args.test_root / args.obj_name
    object_dir = args.test_root / "_object_images" / args.obj_name
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Scene dir missing: {scene_dir}")
    if not object_dir.is_dir():
        raise FileNotFoundError(f"Object dir missing: {object_dir}")

    object_id = args.object_id if args.object_id is not None else int(re.search(r"(\d+)$", args.obj_name).group(1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device: {device}")
    print(f"[eval] resolution: {resolution}  object_views: {object_views}")
    print(f"[eval] checkpoint: {args.checkpoint}")
    print(f"[eval] scene_dir: {scene_dir}")
    print(f"[eval] object_dir: {object_dir}")
    print(f"[eval] symmetry_info: {sym_path}")

    model = build_model(cfg, args.checkpoint, device)
    object_tensor = load_object_tensor(object_dir, object_views, resolution, device)

    frame_ids = sorted(int(k) for k in read_json(scene_dir / "scene_gt.json").keys())

    per_frame_rows: List[Dict] = []
    rot_errs: List[float] = []
    trans_errs: List[float] = []
    trans_xyz_errs: List[np.ndarray] = []

    for fid in frame_ids:
        image_tensor, depth_tensor, mask_tensor, intrinsics, gt_entry = load_scene_frame(
            scene_dir, fid, object_id, resolution, device,
        )
        if gt_entry is None:
            print(f"[eval] frame {fid:06d}: no GT for object {object_id} -- skip")
            continue
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = model.inference(
                    images=image_tensor,
                    object_images=object_tensor,
                    extrinsics=None,
                    intrinsics=None,
                    depth=depth_tensor if args.use_depth_input else None,
                    mask=mask_tensor if args.use_depth_input else None,
                    camera_gt_index=[],
                    depth_gt_index=[0] if args.use_depth_input else [],
                )
        if "object_pose" not in outputs or "object_translation" not in outputs:
            raise RuntimeError(f"Model outputs missing pose keys: {sorted(outputs.keys())}")

        pred_rot6d = outputs["object_pose"][0].detach().float().cpu().numpy()
        pred_rot = rot6d_to_matrix(pred_rot6d).astype(np.float32)
        pred_t = outputs["object_translation"][0].detach().float().cpu().numpy().astype(np.float32)

        gt_rot = np.asarray(gt_entry["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
        gt_t = np.asarray(gt_entry["cam_t_m2c"], dtype=np.float32).reshape(3) / 1000.0

        rot_err, sym_count = symmetric_rotation_error_degrees(pred_rot, gt_rot, object_id, sym_path, sym_steps)
        plain_rot_err = rotation_error_degrees(pred_rot, gt_rot)
        diff = pred_t - gt_t
        trans_err = float(np.linalg.norm(diff))

        per_frame_rows.append({
            "frame": fid,
            "rot_err_sym_deg": float(rot_err),
            "rot_err_plain_deg": float(plain_rot_err),
            "trans_err_m": float(trans_err),
            "trans_abs_xyz_m": np.abs(diff).tolist(),
            "sym_count": int(sym_count),
        })
        rot_errs.append(float(rot_err))
        trans_errs.append(float(trans_err))
        trans_xyz_errs.append(np.abs(diff))

        presence = None
        if "object_presence_logits" in outputs:
            logit = float(outputs["object_presence_logits"].reshape(-1)[0].detach().float().cpu())
            presence = float(torch.sigmoid(torch.tensor(logit)).item())
        print(
            f"[eval] frame {fid:06d}: rot(sym)={rot_err:7.4f}° rot(plain)={plain_rot_err:7.4f}° "
            f"trans={trans_err:.5f}m abs_xyz={np.round(np.abs(diff), 5).tolist()} "
            + (f"presence={presence:.4f}" if presence is not None else "")
        )

    if not rot_errs:
        print("[eval] No frames evaluated.")
        return

    rot_arr = np.asarray(rot_errs, dtype=np.float64)
    trans_arr = np.asarray(trans_errs, dtype=np.float64)
    xyz_arr = np.stack(trans_xyz_errs, axis=0).astype(np.float64)

    summary = {
        "frames_evaluated": len(rot_errs),
        "rot_err_sym_deg": {
            "mean": float(rot_arr.mean()),
            "median": float(np.median(rot_arr)),
            "min": float(rot_arr.min()),
            "max": float(rot_arr.max()),
        },
        "trans_err_m": {
            "mean": float(trans_arr.mean()),
            "median": float(np.median(trans_arr)),
            "min": float(trans_arr.min()),
            "max": float(trans_arr.max()),
        },
        "trans_abs_xyz_m_mean": xyz_arr.mean(axis=0).tolist(),
    }
    print("\n[eval] === Summary ===")
    print(json.dumps(summary, indent=2))

    out_json = args.test_root / args.obj_name / "eval_summary.json"
    out_json.write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "use_depth_input": bool(args.use_depth_input),
        "object_id": int(object_id),
        "obj_name": args.obj_name,
        "object_views": list(object_views),
        "resolution": list(resolution),
        "per_frame": per_frame_rows,
        "summary": summary,
    }, indent=2))
    print(f"[eval] Wrote {out_json}")


if __name__ == "__main__":
    main()
