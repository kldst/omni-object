from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from omnivggt.models.omnivggt import OmniVGGT
from omnivggt.utils.pose_enc import pose_encoding_to_extri_intri
from visual_util import load_images_and_cameras

device = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_NAME = "OmniVGGT.safetensors"
MODEL_REPO_ID = "Livioni/OmniVGGT"


def ensure_checkpoint() -> Path:
    checkpoint_path = CHECKPOINT_DIR / CHECKPOINT_NAME
    if checkpoint_path.exists():
        return checkpoint_path

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint not found at {checkpoint_path}. Downloading from {MODEL_REPO_ID}...")
    downloaded_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=CHECKPOINT_NAME,
        local_dir=CHECKPOINT_DIR,
    )
    return Path(downloaded_path)

# Load the model
model = OmniVGGT().to(device)
checkpoint_path = ensure_checkpoint()
state_dict = load_file(str(checkpoint_path))
model.load_state_dict(state_dict, strict=True)
model.eval()

# Load and preprocess images
images, extrinsics, intrinsics, depthmaps, masks, depth_indices, camera_indices = \
    load_images_and_cameras(
        image_folder=str(PROJECT_ROOT / "example/office/images"),
        camera_folder=None,  # Optional
        depth_folder=None,   # Optional
        target_size=518
    )

# Prepare inputs
inputs = {
    'images': images.to(device),
    'extrinsics': extrinsics.to(device),
    'intrinsics': intrinsics.to(device),
    'depth': depthmaps.to(device),
    'mask': masks.to(device),
    'depth_gt_index': depth_indices,
    'camera_gt_index': camera_indices
}

# Run inference
with torch.no_grad():
    predictions = model.inference(**inputs)
