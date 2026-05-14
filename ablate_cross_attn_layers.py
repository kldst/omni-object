"""Ablate cross-attention layers on a trained OmniVGGT and compare pose error.

For each configuration we disable a subset of cross-attn blocks by replacing
their forward with `lambda q, c: q` (identity on the query side). All other
weights remain unchanged. Then we run inference over the val set and aggregate
rotation / translation / size errors.

Configurations:
  baseline      : all four layers active (4, 11, 17, 23)
  disable_L     : only layer L disabled (one of 4, 11, 17, 23)
  only_23       : only L23 active (single-cross sanity check)
  only_4        : only L4 active
  disable_11_17 : both L11 and L17 disabled (checks redundancy)

Run:
  python ablate_cross_attn_layers.py \
      --ckpt outputs/0511/12000/model.safetensors \
      --max-batches 64 --batch-size 8
"""

import argparse
import contextlib
import json
import math
from types import MethodType

import torch
from accelerate import PartialState

from omnivggt.utils.configs import read_config
from omnivggt.datasets.utils.misc import merge_dicts
from omnivggt.utils.normalization import normalize_camera_extrinsics_and_points_batch
from train_utils import build_dataset, load_model


def _axis_angle_to_matrix(axis, angle):
    axis = torch.tensor(axis, dtype=torch.float32)
    axis = axis / axis.norm().clamp(min=1e-8)
    x, y, z = axis.tolist()
    c, s = math.cos(angle), math.sin(angle)
    one_c = 1.0 - c
    return torch.tensor([
        [c + x*x*one_c, x*y*one_c - z*s, x*z*one_c + y*s],
        [y*x*one_c + z*s, c + y*y*one_c, y*z*one_c - x*s],
        [z*x*one_c - y*s, z*y*one_c + x*s, c + z*z*one_c],
    ], dtype=torch.float32)


def load_symmetry_info(path, continuous_steps=72):
    """Same convention as omnivggt/loss.py: returns {object_id: (K, 3, 3) tensor}."""
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        models_info = json.load(f)
    out = {}
    for oid, info in models_info.items():
        rots = [torch.eye(3, dtype=torch.float32)]
        for transform in info.get("symmetries_discrete", []) or []:
            mat = torch.tensor(transform, dtype=torch.float32).reshape(4, 4)
            rots.append(mat[:3, :3])
        for transform in info.get("symmetries_continuous", []) or []:
            axis = transform.get("axis", [0, 0, 1])
            for step in range(1, int(continuous_steps)):
                ang = 2.0 * math.pi * float(step) / float(continuous_steps)
                rots.append(_axis_angle_to_matrix(axis, ang))
        out[int(oid)] = torch.stack(rots, dim=0)
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_ov9d_camera_pose.py")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=64,
                   help="Cap number of batches per configuration (None = full val set)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    return p.parse_args()


def _identity_forward(self, query_tokens, context_tokens):
    return query_tokens


@contextlib.contextmanager
def disabled_layers(model, layers_to_disable):
    """Temporarily replace forward of selected cross-attn blocks with identity."""
    blocks = model.object_token_cross_attn_blocks
    saved = {}
    for layer_idx in layers_to_disable:
        key = str(layer_idx)
        if key not in blocks:
            raise KeyError(f"layer {layer_idx} not in cross-attn blocks: {list(blocks.keys())}")
        saved[key] = blocks[key].forward
        blocks[key].forward = MethodType(_identity_forward, blocks[key])
    try:
        yield
    finally:
        for key, fn in saved.items():
            blocks[key].forward = fn


def build_inputs(batch):
    if isinstance(batch, list):
        batch = merge_dicts(batch)
    input_extrinsics = batch["extrinsic"].clone()
    input_depths = batch["depth"].clone()
    input_mask = batch["valid_mask"].clone()
    if "world_points" in batch:
        new_extrinsics, _, new_world_points, new_depths = normalize_camera_extrinsics_and_points_batch(
            extrinsics=batch["extrinsic"],
            cam_points=None,
            world_points=batch["world_points"],
            depths=batch["depth"],
            point_masks=batch["valid_mask"],
        )
        batch["extrinsic"] = new_extrinsics
        batch["world_points"] = new_world_points
        batch["depth"] = new_depths
    inputs = {
        "images": batch["images"],
        "extrinsics": input_extrinsics,
        "intrinsics": batch["intrinsic"],
        "depth": input_depths,
        "mask": input_mask,
    }
    if "object_images" in batch:
        inputs["object_images"] = batch["object_images"]
    return inputs


def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    return obj


def _rot6d_to_matrix(rot6d):
    rot6d = rot6d.reshape(-1, 3, 2)
    x_raw = rot6d[:, :, 0]
    y_raw = rot6d[:, :, 1]
    x = torch.nn.functional.normalize(x_raw, dim=-1)
    z = torch.nn.functional.normalize(torch.cross(x, y_raw, dim=-1), dim=-1)
    y = torch.cross(z, x, dim=-1)
    return torch.stack((x, y, z), dim=-1)


def _geodesic_deg(R_pred, R_gt):
    """R_pred, R_gt: (..., 3, 3) -> angle in degrees, shape (...)."""
    rel = torch.matmul(R_pred.transpose(-1, -2), R_gt)
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos_t = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos_t))


def compute_pose_errors(predictions, batch, sym_info):
    """Returns per-sample tensors: rot_naive, rot_sym, trans_cm, size_rel.
    Symmetric rotation error takes min over symmetry-equivalent GT rotations
    (matches the training loss in omnivggt/loss.py:_symmetric_rot6d_loss).
    Masked by has_object."""
    if "object_pose" not in predictions or "object_rotation" not in batch:
        return None, None, None, None

    pred_pose = predictions["object_pose"].detach()
    gt_rot = batch["object_rotation"].detach()
    object_ids = batch.get("object_id")
    has_object = batch.get("has_object", None)
    pred_trans = predictions.get("object_translation", None)
    gt_trans = batch.get("object_translation", None)
    pred_size_log = predictions.get("object_size_log", None)
    gt_size_log = batch.get("object_size_log", None)

    if has_object is not None:
        mask = has_object.bool().detach()
        if mask.sum() == 0:
            return None, None, None, None
        pred_pose = pred_pose[mask]
        gt_rot = gt_rot[mask]
        if object_ids is not None:
            object_ids = object_ids[mask]
        if pred_trans is not None and gt_trans is not None:
            pred_trans = pred_trans.detach()[mask]
            gt_trans = gt_trans.detach()[mask]
        if pred_size_log is not None and gt_size_log is not None:
            pred_size_log = pred_size_log.detach()[mask]
            gt_size_log = gt_size_log.detach()[mask]

    pred_rot = _rot6d_to_matrix(pred_pose.float())
    gt_rot_f = gt_rot.float()

    rot_naive = _geodesic_deg(pred_rot, gt_rot_f)

    rot_sym = torch.empty_like(rot_naive)
    if object_ids is None or not sym_info:
        rot_sym = rot_naive.clone()
    else:
        oids = object_ids.detach().cpu().reshape(-1).tolist()
        for i, oid in enumerate(oids):
            sym_rots = sym_info.get(int(oid))
            if sym_rots is None:
                rot_sym[i] = rot_naive[i]
                continue
            sym_rots = sym_rots.to(pred_rot.device)
            # gt candidates = gt @ sym  (matches loss.py convention)
            gt_cands = torch.matmul(gt_rot_f[i].unsqueeze(0), sym_rots)  # (K,3,3)
            pred_i = pred_rot[i].unsqueeze(0).expand_as(gt_cands)
            angles = _geodesic_deg(pred_i, gt_cands)
            rot_sym[i] = angles.min()

    trans_err_cm = None
    if pred_trans is not None and gt_trans is not None:
        trans_err_cm = torch.norm(pred_trans.float() - gt_trans.float(), dim=-1) * 100.0

    size_rel_err = None
    if pred_size_log is not None and gt_size_log is not None:
        pred_size = torch.exp(pred_size_log.float())
        gt_size = torch.exp(gt_size_log.float())
        size_rel_err = (torch.abs(pred_size - gt_size) / gt_size.clamp(min=1e-6)).mean(dim=-1)

    return (rot_naive.cpu(),
            rot_sym.cpu(),
            trans_err_cm.cpu() if trans_err_cm is not None else None,
            size_rel_err.cpu() if size_rel_err is not None else None)


def run_eval(model, val_loader, device, dtype, max_batches, sym_info):
    rot_naive_all, rot_sym_all, trans_all, size_all = [], [], [], []
    autocast_kwargs = dict(device_type="cuda", dtype=dtype,
                           enabled=(device.type == "cuda" and dtype != torch.float32))
    n_samples = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_batches is not None and i >= max_batches:
                break
            batch = move_to_device(batch, device)
            inputs = build_inputs(batch)
            with torch.amp.autocast(**autocast_kwargs):
                preds = model(**inputs)
            rot_naive, rot_sym, trans, size = compute_pose_errors(preds, batch, sym_info)
            if rot_naive is not None:
                rot_naive_all.append(rot_naive)
                rot_sym_all.append(rot_sym)
                n_samples += rot_naive.numel()
            if trans is not None:
                trans_all.append(trans)
            if size is not None:
                size_all.append(size)
    def _stat(xs):
        return torch.cat(xs) if xs else None
    rn, rs, tc, sc = _stat(rot_naive_all), _stat(rot_sym_all), _stat(trans_all), _stat(size_all)
    return dict(
        n=n_samples,
        rot_naive_mean=rn.mean().item() if rn is not None else float("nan"),
        rot_naive_med=rn.median().item() if rn is not None else float("nan"),
        rot_sym_mean=rs.mean().item() if rs is not None else float("nan"),
        rot_sym_med=rs.median().item() if rs is not None else float("nan"),
        trans_err_cm_mean=tc.mean().item() if tc is not None else float("nan"),
        trans_err_cm_med=tc.median().item() if tc is not None else float("nan"),
        size_rel_err_mean=sc.mean().item() if sc is not None else float("nan"),
    )


def main():
    args = parse_args()
    PartialState()
    print(f"[load] config = {args.config}")
    cfg = read_config(args.config)
    cfg.model_url = args.ckpt
    cfg.model_load_strict = False
    cfg.num_workers = 0
    cfg.val_batch_images = args.batch_size

    device = torch.device(args.device)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    print(f"[load] ckpt = {args.ckpt}")
    model, _ = load_model(cfg, device)
    model = model.to(device).eval()

    if not getattr(model, "enable_multi_layer_object_prototype_cross_attn", False):
        raise RuntimeError("Multi-layer cross-attn not enabled in this model")
    layer_indices = sorted(int(k) for k in model.object_token_cross_attn_blocks.keys())
    print(f"[setup] cross-attn layers = {layer_indices}")

    configurations = [
        ("baseline", []),
        ("disable_L4", [4]),
        ("disable_L11", [11]),
        ("disable_L17", [17]),
        ("disable_L23", [23]),
        ("only_L23", [4, 11, 17]),
        ("only_L4", [11, 17, 23]),
        ("disable_L11_L17", [11, 17]),
    ]
    # Filter configs to ones whose disable-set is a subset of available layers
    configurations = [(n, d) for n, d in configurations if set(d).issubset(set(layer_indices))]

    print(f"[setup] batch_size={args.batch_size}, max_batches={args.max_batches}, dtype={args.dtype}")

    sym_info_path = cfg.get("object_srt_symmetry_info_path", "")
    sym_steps = int(cfg.get("object_srt_symmetry_continuous_steps", 72))
    sym_info = load_symmetry_info(sym_info_path, sym_steps) if sym_info_path else {}
    print(f"[setup] symmetry info: path={sym_info_path}, objects={len(sym_info)}, "
          f"continuous_steps={sym_steps}")

    results = {}
    for name, disable in configurations:
        print(f"\n--- config: {name}  (disabled = {disable})")
        val_loader = build_dataset(
            dataset=cfg.val_dataset,
            batch_size=args.batch_size,
            num_workers=cfg.get("num_workers", 0),
            test=True,
        )
        with disabled_layers(model, disable):
            metrics = run_eval(model, val_loader, device, dtype, args.max_batches, sym_info)
        results[name] = metrics
        print(f"  n={metrics['n']}  "
              f"rot_sym={metrics['rot_sym_mean']:.3f}° (med {metrics['rot_sym_med']:.3f}°)  "
              f"rot_naive={metrics['rot_naive_mean']:.3f}°  "
              f"trans={metrics['trans_err_cm_mean']:.3f}cm  "
              f"size_rel={metrics['size_rel_err_mean']:.4f}")

    # Final comparison table
    base = results["baseline"]
    print()
    print("=" * 125)
    print(f"{'config':>17} | {'n':>5} | "
          f"{'rot_sym':>14} (Δ) | {'rot_naive':>14} (Δ) | "
          f"{'trans_cm':>13} (Δ) | {'size_rel':>10}")
    print("-" * 125)
    for name, _ in configurations:
        m = results[name]
        d_sym = m["rot_sym_mean"] - base["rot_sym_mean"]
        d_naive = m["rot_naive_mean"] - base["rot_naive_mean"]
        d_trans = m["trans_err_cm_mean"] - base["trans_err_cm_mean"]
        print(f"{name:>17} | {m['n']:>5} | "
              f"{m['rot_sym_mean']:>8.3f}° ({d_sym:+7.3f}) | "
              f"{m['rot_naive_mean']:>8.3f}° ({d_naive:+7.3f}) | "
              f"{m['trans_err_cm_mean']:>8.3f}cm ({d_trans:+6.3f}) | "
              f"{m['size_rel_err_mean']:>10.4f}")
    print("=" * 125)
    print()
    print("Reading:")
    print("  * rot_sym  = symmetry-aware (min over symmetric-equivalent GT rotations)")
    print("  * rot_naive = raw geodesic angle (inflated for rotationally-symmetric objects)")
    print("  * use rot_sym as the primary metric, rot_naive for context")


if __name__ == "__main__":
    main()
