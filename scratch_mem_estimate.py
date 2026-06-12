"""
Estimate GPU memory for a given config WITHOUT running on GPU (builds on CPU,
so it can't OOM). Reports the static training-state footprint that AdamW
allocates at optimizer.step() -- which is what overflows in train_hc_diverse24.

Usage:
    python scratch_mem_estimate.py --config configs/train_hc_diverse24.py
"""
import argparse
import torch
from mmengine.config import Config

from train_utils import load_model, build_optimizer


def fmt(nbytes):
    return f"{nbytes / (1024**3):8.3f} GiB ({nbytes / (1024**2):9.1f} MiB)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_hc_diverse24.py")
    args = ap.parse_args()

    # load_model uses accelerate's logger, which needs the state initialized.
    from accelerate import PartialState
    PartialState()

    cfg = Config.fromfile(args.config)
    cpu = torch.device("cpu")

    # Build exactly like training, but on CPU.
    model, weight_dtype = load_model(cfg, cpu)
    # build_optimizer finalizes requires_grad (patch_embed / head freezes) and the
    # real param groups -- so the trainable set below matches training exactly.
    _ = build_optimizer(model, cfg)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    # accelerate mixed_precision="bf16" keeps fp32 MASTER weights; autocast casts to
    # bf16 only inside ops. So param/grad/optimizer storage is fp32 (4 bytes).
    BPP = 4  # bytes per fp32 element

    w_all = total * BPP                 # all weights live in fp32
    grads = trainable * BPP             # one grad buffer per trainable param
    adam = trainable * BPP * 2          # exp_avg + exp_avg_sq
    # _multi_tensor_adamw does torch._foreach_sqrt(exp_avg_sq) -> a transient full
    # copy of exp_avg_sq at step time (this is the exact line that OOM'd).
    step_transient = trainable * BPP

    static = w_all + grads + adam
    static_peak = static + step_transient

    free, total_mem = (torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0))

    print(f"\n=== {args.config} ===")
    print(f"model_requires_grad : {cfg.get('model_requires_grad', True)}")
    print(f"optimizer           : {cfg.get('optimizer_type', 'adamw')}  (state = 2x params)")
    print(f"mixed_precision     : {cfg.get('mixed_precision', 'no')}  -> fp32 master weights/grads/state\n")

    print(f"params total        : {total/1e6:9.2f} M")
    print(f"params trainable    : {trainable/1e6:9.2f} M  ({100*trainable/total:.1f}%)")
    print(f"params frozen       : {frozen/1e6:9.2f} M\n")

    print("--- STATIC training state (batch-size independent) ---")
    print(f"weights (fp32, all) : {fmt(w_all)}")
    print(f"gradients (fp32)    : {fmt(grads)}")
    print(f"AdamW state (2x)    : {fmt(adam)}")
    print(f"  subtotal (static) : {fmt(static)}")
    print(f"+ step() transient  : {fmt(step_transient)}   <-- _foreach_sqrt copy (the OOM line)")
    print(f"  STATIC PEAK       : {fmt(static_peak)}\n")

    print("--- NOTE on activations (the batch-size-dependent part) ---")
    print("In this run backward COMPLETED before the OOM (loss printed, then OOM at")
    print("optimizer.step). So activations for batch=2 already fit; the static state")
    print("above is what pushes it over. Gradient checkpointing keeps activations low.")
    print("Total need ~= STATIC PEAK + activation peak + ~0.5-1.5 GiB CUDA/cuDNN ctx.\n")

    if total_mem:
        print(f"this GPU total      : {fmt(total_mem)}")
        print(f"this GPU free now   : {fmt(free)}")
        if static_peak > total_mem:
            print(">>> STATIC PEAK alone EXCEEDS this GPU -- OOM is expected here.")


if __name__ == "__main__":
    main()
