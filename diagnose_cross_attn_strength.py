"""Measure cross-attention injection strength on a trained OmniVGGT.

Run from omni-object_clone/:
    python diagnose_cross_attn_strength.py \
        --config configs/train_ov9d_camera_pose.py \
        --ckpt outputs/0511/12000/model.safetensors \
        --num-batches 8

For each ObjectTokenCrossAttentionBlock at layers (4, 11, 17, 23) it reports:
  * rel_delta  = ||fused - scene|| / ||scene||
  * cos        = cosine(fused, scene) per token (mean)
  * scene_norm = ||scene|| baseline
A small rel_delta at L4/L11 vs large at L23 supports the "frozen layers wash
out early injection" hypothesis.
"""

import argparse
import statistics
from collections import defaultdict
from types import MethodType

import torch
from accelerate import PartialState

from omnivggt.utils.configs import read_config
from omnivggt.datasets.utils.misc import merge_dicts
from omnivggt.utils.normalization import normalize_camera_extrinsics_and_points_batch
from train_utils import build_dataset, load_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_ov9d_camera_pose.py")
    p.add_argument("--ckpt", required=True, help="Path to trained safetensors / pt")
    p.add_argument("--num-batches", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override val_batch_images (default: from config)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    return p.parse_args()


def make_hooked_forward(block, layer_idx, stats):
    """Wrap ObjectTokenCrossAttentionBlock.forward to record stats."""
    orig_forward = block.forward

    def hooked_forward(self, query_tokens, context_tokens):
        scene_before = query_tokens.detach().float()
        fused = orig_forward(query_tokens, context_tokens)
        scene_after = fused.detach().float()

        delta = scene_after - scene_before
        # Per-token norms: shape (B, N)
        scene_norm = scene_before.norm(dim=-1)
        delta_norm = delta.norm(dim=-1)
        # Avoid div-by-zero
        rel_delta = (delta_norm / scene_norm.clamp_min(1e-8))

        # Cosine similarity per token
        dot = (scene_before * scene_after).sum(dim=-1)
        cos = dot / (scene_before.norm(dim=-1).clamp_min(1e-8) * scene_after.norm(dim=-1).clamp_min(1e-8))

        stats[layer_idx]["rel_delta"].extend(rel_delta.flatten().cpu().tolist())
        stats[layer_idx]["cos"].extend(cos.flatten().cpu().tolist())
        stats[layer_idx]["scene_norm"].extend(scene_norm.flatten().cpu().tolist())
        stats[layer_idx]["delta_norm"].extend(delta_norm.flatten().cpu().tolist())
        return fused

    block.forward = MethodType(hooked_forward, block)


def build_inputs(batch):
    """Mirror train_omnivggt._prepare_batch_and_compute_loss input construction."""
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


def summarize(values):
    if not values:
        return dict(mean=float("nan"), median=float("nan"), p05=float("nan"),
                    p95=float("nan"), std=float("nan"), n=0)
    values = sorted(values)
    n = len(values)
    return dict(
        mean=sum(values) / n,
        median=values[n // 2],
        p05=values[int(0.05 * n)],
        p95=values[int(0.95 * n)],
        std=statistics.pstdev(values) if n > 1 else 0.0,
        n=n,
    )


def main():
    args = parse_args()
    PartialState()  # required by accelerate logger used inside load_model

    print(f"[load] config = {args.config}")
    cfg = read_config(args.config)
    cfg.model_url = args.ckpt
    cfg.model_load_strict = False
    cfg.num_workers = 0
    if args.batch_size is not None:
        cfg.val_batch_images = args.batch_size

    device = torch.device(args.device)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    print(f"[load] ckpt = {args.ckpt}")
    model, _ = load_model(cfg, device)
    model = model.to(device).eval()

    if not getattr(model, "enable_multi_layer_object_prototype_cross_attn", False):
        raise RuntimeError(
            "Model has no multi-layer object cross-attn enabled — nothing to diagnose."
        )
    blocks = model.object_token_cross_attn_blocks
    if blocks is None:
        raise RuntimeError("object_token_cross_attn_blocks is None")
    layer_indices = sorted(int(k) for k in blocks.keys())
    print(f"[setup] cross-attn layers = {layer_indices}")

    stats = defaultdict(lambda: defaultdict(list))
    for layer_idx in layer_indices:
        make_hooked_forward(blocks[str(layer_idx)], layer_idx, stats)

    print(f"[data] building val loader (batch_size={cfg.get('val_batch_images')})")
    val_loader = build_dataset(
        dataset=cfg.val_dataset,
        batch_size=cfg.get("val_batch_images", cfg.get("train_batch_images", 24)),
        num_workers=cfg.get("num_workers", 0),
        test=True,
    )

    print(f"[run] {args.num_batches} batches, dtype={args.dtype}")
    autocast_kwargs = dict(device_type="cuda", dtype=dtype, enabled=(device.type == "cuda" and dtype != torch.float32))
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= args.num_batches:
                break
            batch = move_to_device(batch, device)
            inputs = build_inputs(batch)
            with torch.amp.autocast(**autocast_kwargs):
                _ = model(**inputs)
            print(f"  batch {i+1}/{args.num_batches} done")

    # Report
    print()
    print("=" * 92)
    print(f"{'layer':>6} | {'rel_delta (mean / med / p95)':>34} | {'cos(before,after) (mean)':>26} | {'scene_norm (mean)':>20}")
    print("-" * 92)
    for layer_idx in layer_indices:
        rd = summarize(stats[layer_idx]["rel_delta"])
        co = summarize(stats[layer_idx]["cos"])
        sn = summarize(stats[layer_idx]["scene_norm"])
        print(f"{layer_idx:>6} | "
              f"{rd['mean']:>10.4f} / {rd['median']:>8.4f} / {rd['p95']:>8.4f} | "
              f"{co['mean']:>24.4f} | "
              f"{sn['mean']:>18.4f}")
    print("=" * 92)
    print()
    print("Interpretation:")
    print("  * rel_delta near 0  => the cross-attn at this layer barely modifies scene tokens")
    print("                        (either learned to do nothing, or the injected signal is weak)")
    print("  * cos near 1.0      => scene direction unchanged (modulation is tiny)")
    print("  * If L4 << L23      => 'multi-cross degenerates toward single-cross' hypothesis")
    print("                        is consistent with measurements")
    print("  * If all comparable => multi-cross is doing real work at every depth")


if __name__ == "__main__":
    main()
