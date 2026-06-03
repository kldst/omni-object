# ======================================================
# Omni6DPose SOPE camera-frame object pose training
# with diverse24 reference views (top + front + back + bottom)
# ======================================================
#
# accelerate launch --num_processes=1 train_omnivggt.py --config configs/train_omnipose.py
#
# Setup:
#   1. Train / val use ONLY Omni6DPoseCameraPose on SOPE (synthetic) scenes.
#   2. SOPE patches 00..30 (03 absent) train split; nested <patch>/train/<source>/<scene>/.
#   3. Object reference views: data/Omni6DPose/omni6dpose_ref/diverse24/<oid>/rgb/
#      rendered by render_omni6dpose_object_refs_bpy.py (PAM Aligned.obj, R_align = identity).
#   4. fixed_object_view_ids = (0, 5, 8, 19)  ->  top + front + back + bottom.
#   5. Model predicts depth-mean-normalized translation
#      (normalize_object_translation_by_depth_mean=True): t_pred ~= t_metric / depth.mean().

output_dir = "outputs"
exp_name = "omni6dpose_sope_diverse24_0602_ft"
logging_dir = "logs"

wandb = True
tensorboard = False
report_to = "tensorboard"
num_save_log = 1
num_save_visual = 100000
checkpointing_steps = 1000  # smaller dataset -> save more often

# Model (local warm-start from the 0531_REFER checkpoint; set to None to train from scratch)
# model_url = "/mnt/train-data-4-hdd/yian/freepose/omni-object_clone/outputs/0531_REFER/lr_1e5_1000/model.safetensors"
model_url = "/omni-object_clone_real/outputs/omni6dpose_sope_diverse24_0602/checkpoint-2-7000/model.safetensors"  # remote
model_load_strict = False
model_requires_grad = False
patch_embed_freeze = True
load_patch_embed_from_hub = False
patch_embed_pretrained_path = None
enable_point = False
enable_depth = False
enable_camera = False
enable_object_mask = True
object_mask_head_freeze = False
enable_object_presence = True
enable_object_srt = True
object_srt_head_freeze = False
enable_multi_layer_object_prototype_cross_attn = True
object_cross_attn_freeze = False
object_prototype_poolers_freeze = False
object_prototype_layer_indices = (4, 11, 17, 23)
object_prototype_num_tokens = 32
object_prototype_object_encoder_no_grad = True
object_cross_attn_heads = 16
object_pose_context_pool = "flatten"
object_pose_use_global_scene_object_concat = False
object_pose_transformer_depth = 6
object_pose_transformer_heads = 8
object_pose_transformer_mlp_dim = 1024
object_pose_transformer_dim_head = 64
object_pose_transformer_dropout = 0.0
object_pose_transformer_emb_dropout = 0.0
object_pose_transformer_norm = "layer"
object_pose_transformer_dim = 1024
object_pose_ief_iters = 1
object_pose_init_params_path = None

# Training
mixed_precision = "bf16"
seed = 42
debug = False
num_train_epochs = 100
gradient_accumulation_steps = 1
max_grad_norm = 1.0
debug_print_object_paths = False
debug_print_object_paths_steps = 10
debug_print_object_paths_max_samples = 10
debug_print_object_batch = False
debug_print_object_batch_steps = 10
debug_print_object_batch_max_samples = 10
debug_print_object_batch_depth_stats = False
object_presence_prob = 0.95
cam_drop_prob = 1.0
depth_drop_prob = 0.0
always_use_depth_gt = True
save_each_epoch = False
val_only = False
resume_model_path = None

# Optimizer
optimizer_type = "adamw"
adam_beta1 = 0.9
adam_beta2 = 0.95
adam_epsilon = 1e-8
adam_weight_decay = 0.05
lr = 1e-4
lr_patch_embed = 1e-4
lr_camera_head = 1e-4
lr_depth_head = 1e-4
lr_point_head = 1e-4
lr_object_mask_head = 1e-4
lr_object_srt_head = 1e-4
lr_object_cross_attn = 1e-4
lr_object_prototype_poolers = 1e-4
lr_scheduler_type = "cosine_with_warmup"
warmup_steps = 0
eta_min_factor = 1e-4

# Loss
camera_loss_weight = 0.0
camera_loss_type = "l1"
depth_loss_weight = 0.0
depth_gradient_loss_fn = "grad"
depth_valid_range = 0.98
point_loss_weight = 0.0
point_gradient_loss_fn = "normal"
point_valid_range = 0.98
object_mask_loss_weight = 1.0
object_mask_bce_weight = 1.0
object_mask_dice_weight = 1.0
object_mask_pos_weight = 1.0
object_presence_loss_weight = 1.0
object_srt_loss_weight = 1.0
object_srt_loss_type = "l1"
object_srt_pose_rep = "symmetric_rot6d"
object_srt_weight_pose = 1.0
object_srt_weight_translation = 1.0
object_srt_weight_size = 1.0
# Class-level symmetry table built from ROPE real_obj_meta (build_omni6dpose_symmetry.py),
# keyed Omni6DPoseCameraPose:<object_id>. Symmetric objects (bottle/bowl/can/ball/...) get
# their rotation equivalence set so the pose loss isn't penalized for equivalent rotations.
# Remote: change this path to wherever omni6dpose_symmetry_info.json lives.
# object_srt_symmetry_info_path = "/mnt/train-data-4-hdd/yian/freepose/omni-object_clone/omni6dpose_symmetry_info.json"
object_srt_symmetry_info_path = "/omni-object_clone_real/omni6dpose_symmetry_info.json"  # remote
object_srt_symmetry_continuous_steps = 72

# Dataset
train_batch_images = 60
val_batch_images = 60
val_epoch_freq = 5      # eval more often since training data is smaller
num_workers = 0
resolution = (518, 476)
# diverse24 view selection: 0 = near top-down, 5 = front (az 0°/el +20°),
# 8 = back (az 180°/el +20°), 19 = near bottom-up.  See render_aligned_object_refs_bpy.py
# DIVERSE_24_SCHEDULE for the full geometry.
fixed_object_view_ids = (0, 5, 8, 19)
strict_fixed_object_view_ids = True
val_max_records_per_dataset = 1000

#* ---- Omni6DPose SOPE/ROPE paths (local) ----
# omni_root = f"{freepose_root}/omni-object_clone"
# omni6dpose_root = f"{freepose_root}/Omni6dpose/Omni6DPoseAPI/data/Omni6DPose"
# sope_root = f"{omni6dpose_root}/SOPE"
# rope_root = f"{omni6dpose_root}/ROPE"
# diverse24 PAM-mesh object references (top/front/back/bottom), views 0/5/8/19
# object_image_root = f"{omni6dpose_root}/omni6dpose_ref/diverse24"

#* ---- Omni6DPose SOPE paths (remote) ----
omni_root = "/omni-object_clone_real"        # omni-object_clone 在遠端的位置
omni6dpose_root = "/dataset/omni6dpose"      # 兩個 tar 解壓到這裡
sope_root = omni6dpose_root
object_image_root = f"{omni6dpose_root}/diverse24"

# SOPE patches to train on: 00..05, excluding 03 (absent on disk).
sope_train_patches = [f"{i:02d}" for i in range(6) if i != 3]
# Validation: SOPE *test* split of patch 00 (held out from training frames).
# Only objects that have rendered references are kept, so val may be a subset.
sope_val_patches = ["00"]
sope_val_split = "test"

train_dataset = (
    "Omni6DPoseCameraPose("
    f"dataset_location='{sope_root}', "
    "dset='train', "
    "layout='sope', "
    f"patches={sope_train_patches}, "
    "split='train', "
    f"object_image_root='{object_image_root}', "
    "oid_to_pam_json=None, "          # SOPE: oid IS the reference folder name
    "num_object_views=4, "
    f"fixed_object_view_ids={fixed_object_view_ids}, "
    f"strict_fixed_object_view_ids={strict_fixed_object_view_ids}, "
    "normalize_object_translation_by_depth_mean=True, "  # predict t / depth.mean()
    "expand_records_by_object=True, "
    "verify_files=True, "
    f"object_presence_prob={object_presence_prob}, "
    "z_far=20, "
    f"resolution={resolution}, "
    "transform=ColorJitter, "
    "seed=42)"
)

val_dataset = (
    "Omni6DPoseCameraPose("
    f"dataset_location='{sope_root}', "
    "dset='val', "
    "layout='sope', "
    f"patches={sope_val_patches}, "
    f"split='{sope_val_split}', "
    f"object_image_root='{object_image_root}', "
    "oid_to_pam_json=None, "
    "num_object_views=4, "
    f"fixed_object_view_ids={fixed_object_view_ids}, "
    f"strict_fixed_object_view_ids={strict_fixed_object_view_ids}, "
    "normalize_object_translation_by_depth_mean=True, "
    "expand_records_by_object=True, "
    "verify_files=True, "
    f"max_records={val_max_records_per_dataset}, "
    "z_far=20, "
    f"resolution={resolution}, "
    "seed=42)"
)
