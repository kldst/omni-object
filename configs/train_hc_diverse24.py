# ======================================================
# HouseCat6D-only camera-frame object pose training
# with diverse24 reference views (top + front + back + bottom)
# warm-started from outputs/0521/14000/model.safetensors
# ======================================================
#
# accelerate launch --num_processes=1 train_omnivggt.py --config configs/train_hc_diverse24.py
#
# Key differences vs train_oo9d_real275_ycbv_hc.py:
#   1. Train / val use ONLY HouseCat6DCameraPose -- no OO9D / REAL275 / YCBV.
#   2. housecat6d_object_image_root points to housecat6d_aligned_object_refs_diverse24.
#   3. fixed_object_view_ids = (0, 5, 8, 19)  →  top + front + back + bottom.
#   4. model_url warm-starts from the best 0521 checkpoint (step 14000).
#   5. Smaller dataset → faster epoch, so checkpointing_steps reduced to 1000.

output_dir = "outputs"
exp_name = "hc_only_diverse24_warmstart_14k_0530_regular_cache"
logging_dir = "logs"

wandb = True
tensorboard = False
report_to = "tensorboard"
num_save_log = 1
num_save_visual = 100000
checkpointing_steps = 688  # smaller dataset -> save more often

# Model
# model_url = "/mnt/train-data-4-hdd/yian/freepose/omni-object_clone/outputs/0521/14000/model.safetensors"
model_url = "/omni-object_clone_real/outputs/oo9d_real275_ycbv_hc_camera_pose_size_mask_presence_0528/checkpoint-0-4000/model.safetensors"
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
# Object-encoder token cache.
#   Training: OFF (saves GPU memory; train refs use ColorJitter -> non-deterministic,
#             not cache-safe anyway -- see object_ref_color_jitter=True in train_dataset).
#   Benchmark: ON via benchmark_object_encode_cache below (eval has no backprop
#             activations; same object recurs across many frames -> big speedup).
object_encode_cache = False
object_encode_cache_max = 256
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
lr = 1e-5
lr_patch_embed = 1e-5
lr_camera_head = 1e-5
lr_depth_head = 1e-5
lr_point_head = 1e-5
lr_object_mask_head = 1e-5
lr_object_srt_head = 1e-5
lr_object_cross_attn = 1e-5
lr_object_prototype_poolers = 1e-5
lr_scheduler_type = "cosine_with_warmup"
warmup_steps = 0
eta_min_factor = 1e-5

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
# object_srt_symmetry_info_path = "/mnt/train-data-4-hdd/yian/freepose/omni-object_clone/mixed_symmetry_info.json"
object_srt_symmetry_info_path = "/omni-object_clone_real/mixed_symmetry_info.json"
object_srt_symmetry_continuous_steps = 72
# SMOC-Net style cross-view relative-pose regularization (requires paired sampling
# in the dataset, see relative_pose_pairing=True below). Symmetry-aware; reuses the
# object_srt symmetry table. Keep the weight small -- this is a regularizer.
relative_pose_loss_weight = 0.1
relative_pose_weight_rot = 1.0
relative_pose_weight_trans = 0.0   # translation consistency off by default
relative_pose_loss_type = "l1"

# Dataset
train_batch_images = 60
# val_batch_images = 60
val_epoch_freq = 1      # eval more often since training data is smaller
num_workers = 0
resolution = (518, 476)
# diverse24 view selection: 0 = near top-down, 5 = front (az 0°/el +20°),
# 8 = back (az 180°/el +20°), 19 = near bottom-up.  See render_aligned_object_refs_bpy.py
# DIVERSE_24_SCHEDULE for the full geometry.
fixed_object_view_ids = (0, 5, 8, 19)
strict_fixed_object_view_ids = True
val_max_records_per_dataset = 1000

# Validation mode: "loss" = current loss/rot_err val over val_dataset;
# "benchmark" = run the official HouseCat6D mAP benchmark in-process and log to wandb.
validation_mode = "benchmark"
benchmark_scenes = ["test_scene1", "test_scene2", "test_scene3", "test_scene4", "test_scene5"]
benchmark_frame_stride = 1       # subsample frames per scene to bound eval time
benchmark_batch_size = 30
benchmark_limit = None           # cap samples/scene for smoke tests (None = full)
# Benchmark-only object-encoder cache (independent of training's object_encode_cache):
# eval has no backprop, same object recurs across frames -> big speedup. Toggled on
# only during the benchmark, then cleared.
benchmark_object_encode_cache = True
# Use ColorJitter on benchmark object refs too. NOTE: with the cache ON, this freezes
# ONE random jitter per object id (cached on first occurrence, reused for all its
# frames) -- not per-frame augmentation. Set False for clean/deterministic refs.
benchmark_object_ref_color_jitter = True

# freepose_root = "/mnt/train-data-4-hdd/yian/freepose"
# omni_root = f"{freepose_root}/omni-object_clone"
# align_json = f"{omni_root}/dataset_align.json"

# housecat6d_root = f"{freepose_root}/housecat6d"
# diverse24 refs (top/front/back/bottom mix) -- generated by
# render_aligned_object_refs_bpy.py --view-strategy diverse24
# housecat6d_object_image_root = f"{housecat6d_root}/housecat6d_aligned_object_refs_diverse24"

#* 緯創
freepose_root = "/dataset"
omni_root = f"/omni-object_clone_real"
align_json = f"{omni_root}/dataset_align.json"
housecat6d_root = f"{freepose_root}/housecat6d"
housecat6d_object_image_root = f"{housecat6d_root}/housecat6d_aligned_object_refs_diverse24"

train_dataset = (
    "HouseCat6DCameraPose("
    f"dataset_location='{housecat6d_root}', "
    "dset='train', "
    f"object_image_root='{housecat6d_object_image_root}', "
    f"align_json='{align_json}', "
    "num_object_views=4, "
    f"fixed_object_view_ids={fixed_object_view_ids}, "
    f"strict_fixed_object_view_ids={strict_fixed_object_view_ids}, "
    "normalize_object_translation_by_depth_mean=True, "
    "expand_records_by_object=True, "
    "verify_files=True, "
    f"object_presence_prob={object_presence_prob}, "
    "z_far=20, "
    f"resolution={resolution}, "
    "transform=ColorJitter, "
    # Object refs use ColorJitter during training (restored). NOTE: this makes refs
    # non-deterministic, so training must keep object_encode_cache=False (it is).
    "object_ref_color_jitter=True, "
    "scene_glob='scene*', "  # train scenes: scene01..scene34
    # Pair consecutive batch items (2k, 2k+1) as two views of the same static
    # object instance, for the SMOC-Net relative-pose loss. Requires even
    # train_batch_images. pair_min_frame_gap avoids near-identical views.
    "relative_pose_pairing=True, "
    "pair_min_frame_gap=20, "
    "seed=42)"
)

val_dataset = (
    "HouseCat6DCameraPose("
    f"dataset_location='{housecat6d_root}', "
    "dset='val', "
    f"object_image_root='{housecat6d_object_image_root}', "
    f"align_json='{align_json}', "
    "num_object_views=4, "
    f"fixed_object_view_ids={fixed_object_view_ids}, "
    f"strict_fixed_object_view_ids={strict_fixed_object_view_ids}, "
    "normalize_object_translation_by_depth_mean=True, "
    "expand_records_by_object=True, "
    "verify_files=True, "
    f"max_records={val_max_records_per_dataset}, "
    "z_far=20, "
    f"resolution={resolution}, "
    # Remote-friendly: use scene34 (last train scene) as stand-in for val since
    # val_scene1/2 may not be uploaded to the cluster. Switch back to
    # 'val_scene*' once those dirs exist locally.
    "scene_glob='scene34', "
    "seed=42)"
)
