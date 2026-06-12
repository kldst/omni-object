"""
Standalone reproduction for the CheckpointError seen when model_requires_grad=True.

Does NOT modify any project code. Run:
    python scratch_ckpt_repro.py
"""
import torch
from torch.utils.checkpoint import checkpoint

from omnivggt.layers.block import Block
from omnivggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter

torch.manual_seed(0)
dev = "cuda"
dtype = torch.bfloat16

DIM = 1024
HEADS = 16
# odd-ish seq length similar to the real run (note: real error shows 1263)
H, W = 39, 31          # grid -> P = 1209 patches
B = 2

rope = RotaryPositionEmbedding2D(frequency=100).to(dev)
posget = PositionGetter()


def build_block():
    blk = Block(dim=DIM, num_heads=HEADS, qk_norm=True, rope=rope).to(dev).to(dtype)
    return blk


def run(requires_grad_block: bool, requires_grad_input: bool, tag: str, autocast=False):
    # autocast=True mirrors the real run: fp32 weights + torch.autocast(bf16).
    blk = Block(dim=DIM, num_heads=HEADS, qk_norm=True, rope=rope).to(dev)
    if not autocast:
        blk = blk.to(dtype)
    blk.train()
    blk.requires_grad_(requires_grad_block)

    P = H * W
    in_dtype = torch.float32 if autocast else dtype
    x = torch.randn(B, P, DIM, device=dev, dtype=in_dtype, requires_grad=requires_grad_input)
    pos = posget(B, H, W, dev)

    def fn(inp, p):
        return blk(inp, pos=p)

    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if autocast else torch.autocast("cuda", enabled=False)
    try:
        with ctx:
            out = checkpoint(fn, x, pos, use_reentrant=False)
        if out.requires_grad:
            out.float().sum().backward()
            print(f"[{tag}] OK  (block_grad={requires_grad_block}, input_grad={requires_grad_input}) "
                  f"-> backward ran, no CheckpointError")
        else:
            print(f"[{tag}] SKIP (block_grad={requires_grad_block}, input_grad={requires_grad_input}) "
                  f"-> output does NOT require grad, checkpoint never recomputes (no backward)")
    except Exception as e:
        msg = str(e).splitlines()[0]
        print(f"[{tag}] ERROR (block_grad={requires_grad_block}, input_grad={requires_grad_input}) "
              f"-> {type(e).__name__}: {msg}")


if __name__ == "__main__":
    print("torch", torch.__version__)
    # mirrors model_requires_grad=False (aggregator frozen, patch_embed frozen):
    run(requires_grad_block=False, requires_grad_input=False, tag="frozen (model_requires_grad=False)")
    # mirrors model_requires_grad=True:
    run(requires_grad_block=True, requires_grad_input=False, tag="trainable (model_requires_grad=True)")
    # extra: input also requires grad
    run(requires_grad_block=True, requires_grad_input=True, tag="trainable + input.requires_grad")
    print("--- with autocast(bf16) + fp32 weights (matches accelerate mixed_precision=bf16) ---")
    run(requires_grad_block=False, requires_grad_input=False, tag="AUTOCAST frozen", autocast=True)
    run(requires_grad_block=True, requires_grad_input=False, tag="AUTOCAST trainable", autocast=True)
