# ======================================================
# OV9D camera-frame object pose training
# ======================================================
#
# accelerate launch --num_processes=1 train_omnivggt.py --config configs/train_ov9d_camera_pose.py

output_dir = "outputs"
exp_name = "ov9d_camera_pose_size_mask_presence"
logging_dir = "logs"

wandb = True
tensorboard = False
report_to = "tensorboard"
num_save_log = 1
num_save_visual = 100000
checkpointing_steps = 6000

# Model
model_url = "/all_data/model.safetensors"
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
object_presence_prob = 0.3
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
# object_srt_symmetry_info_path = "/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d/models_info.json"
object_srt_symmetry_info_path = "/dataset/ov9d/models_info.json"
object_srt_symmetry_continuous_steps = 72

# Dataset
train_batch_images = 60
val_batch_images = 60
val_epoch_freq = 10
num_workers = 0
resolution = (518, 518)
# ov9d_root = "/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d"
ov9d_root = "/dataset/ov9d"
fixed_object_view_ids = (10, 20, 30, 40)
strict_fixed_object_view_ids = True

train_dataset = (
    "OV9DCameraPose("
    f"dataset_location='{ov9d_root}', "
    "dset='train', "
    f"split_json='/omni-object_clone/splits_multi_4_3000/train.json', "
    # f"split_json='/mnt/train-data-4-hdd/yian/freepose/omni-object_clone/splits_multi_4_3000/train.json', "
    "num_object_views=4, "
    f"fixed_object_view_ids={fixed_object_view_ids}, "
    f"strict_fixed_object_view_ids={strict_fixed_object_view_ids}, "
    "verify_files=True, "
    f"object_presence_prob={object_presence_prob}, "
    "z_far=20, "
    "resolution=(518, 518), "
    "transform=ColorJitter, "
    "seed=42)"
)

val_dataset = (
    "OV9DCameraPose("
    f"dataset_location='{ov9d_root}', "
    "dset='test1', "
    f"split_json='/omni-object_clone/splits_multi_4_3000/test1.json', "
    # f"split_json='/mnt/train-data-4-hdd/yian/freepose/omni-object_clone/splits_multi_4_3000/test1.json', "
    "num_object_views=4, "
    f"fixed_object_view_ids={fixed_object_view_ids}, "
    f"strict_fixed_object_view_ids={strict_fixed_object_view_ids}, "
    "verify_files=True, "
    "object_presence_prob=0.3, "
    "z_far=20, "
    "resolution=(518, 518), "
    "seed=42)"
)
