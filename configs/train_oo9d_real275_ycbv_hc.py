# ======================================================
# OO9D + REAL275 + YCB-V + HouseCat6D camera-frame object pose training
# ======================================================
#
# accelerate launch --num_processes=1 train_omnivggt.py --config configs/train_oo9d_real275_ycbv_hc.py


output_dir = "outputs"
exp_name = "oo9d_real275_ycbv_hc_camera_pose_size_mask_presence_0524"
logging_dir = "logs"

wandb = True
tensorboard = False
report_to = "tensorboard"
num_save_log = 1
num_save_visual = 100000
checkpointing_steps = 2000

# Model
# model_url = "/all_data/model.safetensors"
model_url = "/omni-object_clone_0520/outputs/oo9d_camera_pose_mask_presence_0521_norm_translate/checkpoint-12-12000"
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
# Mixed table uses dataset-aware keys such as "YCBVCameraPose:13", so object ids
# from OO9D / REAL275 / YCB-V / HouseCat6D cannot collide.
# object_srt_symmetry_info_path = "/mnt/train-data-4-hdd/yian/freepose/omni-object_clone/mixed_symmetry_info.json"
object_srt_symmetry_info_path = "/omni-object_clone_real/mixed_symmetry_info.json"
object_srt_symmetry_continuous_steps = 72

# Dataset
train_batch_images = 1
val_batch_images = 1
val_epoch_freq = 10
num_workers = 0
resolution = (518, 476)
fixed_object_view_ids = (1, 5, 10, 15)
strict_fixed_object_view_ids = True
val_max_records_per_dataset = 1000

# freepose_root = "/mnt/train-data-4-hdd/yian/freepose"
# omni_root = f"{freepose_root}/omni-object_clone"
# align_json = f"{omni_root}/dataset_align.json"

# oo9d_root = f"{freepose_root}/ov9d/ov9d"
# oo9d_single_root = f"{oo9d_root}/oo3d9dsingle"
# oo9d_split_root = f"{omni_root}/splits_ov9d_unseen_category_generalization"
# oo9d_object_image_root = f"{freepose_root}/ov9d/ov9d_around_image"

# real275_root = f"{freepose_root}/real275"
# real275_split_root = f"{real275_root}/real_train"
# real275_gt_root = f"{real275_root}/gts/real_train_umeyama"
# real275_object_image_root = f"{real275_root}/real275_aligned_object_refs"

# ycbv_root = f"{freepose_root}/datasets_real/ycbv"
# ycbv_train_split_root = f"{ycbv_root}/train_real"
# ycbv_val_split_root = f"{ycbv_root}/test"
# ycbv_object_image_root = f"{ycbv_root}/ycbv_aligned_object_refs"

# housecat6d_root = f"{freepose_root}/housecat6d"
# housecat6d_object_image_root = f"{housecat6d_root}/housecat6d_aligned_object_refs"

#* 緯創
freepose_root = "/dataset"
omni_root = f"/omni-object_clone_real"
align_json = f"{omni_root}/dataset_align.json"

oo9d_root = f"{freepose_root}/ov9d"
oo9d_single_root = f"{oo9d_root}/oo3d9dsingle"
oo9d_split_root = f"{omni_root}/splits_ov9d_unseen_category_generalization"
oo9d_object_image_root = f"{freepose_root}/ov9d_around_image"

real275_root = f"{freepose_root}/real275"
real275_split_root = f"{real275_root}/real_train"
real275_gt_root = f"{real275_root}/gts/real_train_umeyama"
real275_object_image_root = f"{real275_root}/real275_aligned_object_refs"

ycbv_root = f"{freepose_root}/ycbv"
ycbv_train_split_root = f"{ycbv_root}/train_real"
ycbv_val_split_root = f"{ycbv_root}/test"
ycbv_object_image_root = f"{ycbv_root}/ycbv_aligned_object_refs"

housecat6d_root = f"{freepose_root}/housecat6d"
housecat6d_object_image_root = f"{housecat6d_root}/housecat6d_aligned_object_refs"

train_dataset = (
    "torch.utils.data.ConcatDataset(("
    "OO9DSingleCameraPose("
    f"dataset_location='{oo9d_root}', "
    "dset='train', "
    f"single_root='{oo9d_single_root}', "
    f"single_split_json='{oo9d_split_root}/single/train.json', "
    f"object_image_root='{oo9d_object_image_root}', "
    "num_object_views=4, "
    f"fixed_object_view_ids={fixed_object_view_ids}, "
    f"strict_fixed_object_view_ids={strict_fixed_object_view_ids}, "
    "normalize_object_translation_by_depth_mean=True, "
    "expand_records_by_view=True, "
    "verify_files=True, "
    f"object_presence_prob={object_presence_prob}, "
    "z_far=20, "
    f"resolution={resolution}, "
    "transform=ColorJitter, "
    "seed=42), "
    "Real275CameraPose("
    f"dataset_location='{real275_root}', "
    "dset='train', "
    f"split_root='{real275_split_root}', "
    f"gt_root='{real275_gt_root}', "
    f"object_image_root='{real275_object_image_root}', "
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
    "seed=42), "
    "YCBVCameraPose("
    f"dataset_location='{ycbv_root}', "
    "dset='train_real', "
    f"split_root='{ycbv_train_split_root}', "
    f"object_image_root='{ycbv_object_image_root}', "
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
    "seed=42), "
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
    "seed=42)"
    "))"
)

val_dataset = (
    "torch.utils.data.ConcatDataset(("
    "OO9DSingleCameraPose("
    f"dataset_location='{oo9d_root}', "
    "dset='val', "
    f"single_root='{oo9d_single_root}', "
    f"single_split_json='{oo9d_split_root}/single/test_unseen_category_unseen_object.json', "
    f"object_image_root='{oo9d_object_image_root}', "
    "num_object_views=4, "
    f"fixed_object_view_ids={fixed_object_view_ids}, "
    f"strict_fixed_object_view_ids={strict_fixed_object_view_ids}, "
    "normalize_object_translation_by_depth_mean=True, "
    "expand_records_by_view=True, "
    "verify_files=True, "
    "object_presence_prob=0.5, "
    f"max_records={val_max_records_per_dataset}, "
    "z_far=20, "
    f"resolution={resolution}, "
    "seed=42), "
    "Real275CameraPose("
    f"dataset_location='{real275_root}', "
    "dset='val', "
    f"split_root='{real275_split_root}', "
    f"gt_root='{real275_gt_root}', "
    f"object_image_root='{real275_object_image_root}', "
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
    "seed=42), "
    "YCBVCameraPose("
    f"dataset_location='{ycbv_root}', "
    "dset='test', "
    f"split_root='{ycbv_val_split_root}', "
    f"object_image_root='{ycbv_object_image_root}', "
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
    "seed=42), "
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
    "seed=42)"
    "))"
)
